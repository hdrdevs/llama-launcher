const { app, BrowserWindow, ipcMain, dialog, shell } = require('electron');
const { spawn, exec } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');
const https = require('https');
const zlib = require('zlib');
const { DownloadManager } = require('./download-manager');

let mainWindow = null;
let serverProc = null;
let serverErrBuf = '';
let closeConfirmed = false;
let closePending = false;

/* ---------------- System stats monitor ---------------- */

let prevCpuTimes = null;
let gpuTotal = null; // VRAM total (bytes), sumado sobre los adapters
let gpuUsed = null; // VRAM en uso (bytes), sumado sobre los adapters
let gpuUsages = {}; // adapter (LUID) -> dedicated usage en bytes
let gpuProc = null;
let gpuBuf = '';

function readCpuPct() {
  const cpus = os.cpus();
  let idle = 0;
  let total = 0;
  for (const c of cpus) {
    idle += c.times.idle;
    total += c.times.user + c.times.nice + c.times.sys + c.times.idle + c.times.irq;
  }
  if (prevCpuTimes) {
    const dIdle = idle - prevCpuTimes.idle;
    const dTotal = total - prevCpuTimes.total;
    if (dTotal > 0) {
      const pct = (1 - dIdle / dTotal) * 100;
      prevCpuTimes = { idle, total };
      return Math.max(0, Math.min(100, pct));
    }
  }
  prevCpuTimes = { idle, total };
  return 0;
}

function detectGpuTotal() {
  // La capacidad de VRAM se lee del registro de drivers (REG_QWORD, soporta >4 GB).
  // Win32_VideoController.AdapterRAM es UInt32 y se trunca en ~4 GB, así que queda
  // solo como respaldo para drivers que no publiquen qwMemorySize.
  const script =
    "$cls = 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}'; " +
    '$total = 0; ' +
    "Get-ChildItem $cls -ErrorAction SilentlyContinue | ForEach-Object { " +
    "$key = $_.PSChildName; " +
    "if ($key -eq 'Configuration' -or $key -eq 'Properties') { return }; " +
    'try { ' +
    "$p = Get-ItemProperty $_.PSPath; " +
    "$v = $p.'HardwareInformation.qwMemorySize'; " +
    'if ($null -ne $v -and $v -gt 0) { $total += $v } ' +
    '} catch {} ' +
    '}; ' +
    'if ($total -le 0) { ' +
    'Get-CimInstance Win32_VideoController | ForEach-Object { ' +
    '$v = [int64]$_.AdapterRAM; ' +
    'if ($v -gt $total) { $total = $v } ' +
    '} ' +
    '}; ' +
    "'GPUTOTAL ' + $total";
  const p = spawn('powershell', ['-NoProfile', '-Command', script], { windowsHide: true });
  let out = Buffer.alloc(0);
  p.stdout.on('data', (d) => {
    out = Buffer.concat([out, d]);
  });
  p.on('close', () => {
    const text = out.includes(0) ? out.toString('utf16le') : out.toString('utf8');
    const m = text.match(/GPUTOTAL\s+(\d+)/);
    const n = m ? Number(m[1]) : 0;
    if (n > 0) gpuTotal = n;
  });
  p.on('error', () => {});
}

function startGpuMonitor() {
  const script =
    "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage' -Continuous -SampleInterval 1 | ForEach-Object { foreach ($s in $_.CounterSamples) { $p = [string]$s.Path; $li = $p.LastIndexOf('\\'); $before = $p.Substring(0, $li); $open = $before.IndexOf('('); $close = $before.LastIndexOf(')'); if ($open -ge 0 -and $close -gt $open) { $name = $before.Substring($open + 1, $close - $open - 1) } else { $name = $before }; 'GPU|' + $name + '|' + [int64]$s.CookedValue } }";
  try {
    gpuProc = spawn('powershell', ['-NoProfile', '-Command', script], { windowsHide: true });
  } catch (e) {
    return;
  }
  gpuProc.stdout.on('data', (d) => {
    gpuBuf += d.toString('utf8');
    let i;
    while ((i = gpuBuf.indexOf('\n')) >= 0) {
      const line = gpuBuf.slice(0, i).trim();
      gpuBuf = gpuBuf.slice(i + 1);
      if (!line.startsWith('GPU|')) continue;
      const parts = line.split('|');
      if (parts.length < 3) continue;
      const value = Number(parts[2]);
      if (!Number.isFinite(value) || value <= 0) continue;
      gpuUsages[parts[1]] = value;
      gpuUsed = Object.keys(gpuUsages).reduce((acc, k) => acc + gpuUsages[k], 0);
    }
  });
  gpuProc.on('error', () => {});
}

function startStatsMonitor() {
  detectGpuTotal();
  startGpuMonitor();
  setInterval(() => {
    const ram = { used: os.totalmem() - os.freemem(), total: os.totalmem() };
    send('stats:update', {
      cpu: readCpuPct(),
      ram,
      gpu: gpuTotal ? { used: Math.min(gpuUsed || 0, gpuTotal), total: gpuTotal } : null,
    });
  }, 1000);
}

/* ---------------- GGUF model inspector ---------------- */

// bytes per gguf_type (uint8,int8,uint16,int16,uint32,int32,float32,bool,string,array,uint64,int64,float64)
const GGUF_TYPE_SIZES = [1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8];

// [ggml_type] -> [bytes per block, elements per block]
const GGML_BLOCK_INFO = {
  0: [4, 1], // F32
  1: [2, 1], // F16
  2: [18, 32], // Q4_0
  3: [20, 32], // Q4_1
  6: [22, 32], // Q5_0
  7: [24, 32], // Q5_1
  8: [34, 32], // Q8_0
  9: [36, 32], // Q8_1
  10: [84, 256], // Q2_K
  11: [110, 256], // Q3_K
  12: [144, 256], // Q4_K
  13: [176, 256], // Q5_K
  14: [210, 256], // Q6_K
  15: [292, 256], // Q8_K
  16: [66, 256], // IQ2_XXS
  17: [74, 256], // IQ2_XS
  18: [98, 256], // IQ3_XXS
  19: [50, 256], // IQ1_S
  20: [18, 32], // IQ4_NL
  21: [110, 256], // IQ3_S
  22: [82, 256], // IQ2_S
  23: [136, 256], // IQ4_XS
  24: [1, 1], // I8
  25: [2, 1], // I16
  26: [4, 1], // I32
  27: [8, 1], // I64
  28: [8, 1], // F64
  29: [56, 256], // IQ1_M
  30: [2, 1], // BF16
  34: [54, 256], // TQ1_0
  35: [66, 256], // TQ2_0
  39: [17, 32], // MXFP4
  40: [36, 64], // NVFP4
  41: [18, 128], // Q1_0
  42: [18, 64], // Q2_0
};

function inspectGgufModel(filePath) {
  let fd = null;
  try {
    fd = fs.openSync(filePath, 'r');
  } catch (e) {
    return null;
  }
  let buf = Buffer.alloc(0);
  let fileOff = 0;
  const read = (n) => {
    while (buf.length < n) {
      const chunk = Buffer.alloc(1 << 20);
      const nread = fs.readSync(fd, chunk, 0, chunk.length, fileOff);
      if (nread <= 0) throw new Error('EOF');
      buf = Buffer.concat([buf, chunk.subarray(0, nread)]);
      fileOff += nread;
    }
    const out = buf.subarray(0, n);
    buf = buf.subarray(n);
    return out;
  };
  const str = () => {
    const len = Number(read(8).readBigUInt64LE(0));
    if (len > 1 << 20) throw new Error('string too long');
    return read(len).toString('utf8');
  };
  try {
    if (read(4).toString('utf8') !== 'GGUF') return null;
    const version = read(4).readUInt32LE(0);
    if (version < 2 || version > 3) return null;
    const nTensors = Number(read(8).readBigUInt64LE(0));
    const nKv = Number(read(8).readBigUInt64LE(0));

    const skipValue = (type, count) => {
      if (type === 8) {
        for (let i = 0; i < count; i++) str();
        return;
      }
      const sz = GGUF_TYPE_SIZES[type];
      if (sz === undefined || sz === 0) throw new Error('bad gguf type ' + type);
      read(sz * count);
    };

    let totalLayers = 0;
    let nEmbds = 0;
    let nHeads = 0;
    let nHeadsKv = 0;
    let keyLen = 0;
    let valLen = 0;
    let contextLength = 0;
    let isMoe = false;
    let nExperts = 0;
    for (let i = 0; i < nKv; i++) {
      const key = str();
      const type = read(4).readInt32LE(0);
      if (type === 9) {
        const elemType = read(4).readInt32LE(0);
        const count = Number(read(8).readBigUInt64LE(0));
        skipValue(elemType, count);
      } else if (type === 4) {
        const val = read(4).readUInt32LE(0);
        if (/\.block_count$/.test(key)) totalLayers = val;
        else if (/\.embedding_length$/.test(key)) nEmbds = val;
        else if (/\.attention\.head_count$/.test(key)) nHeads = val;
        else if (/\.attention\.head_count_kv$/.test(key)) nHeadsKv = val;
        else if (/\.attention\.key_length$/.test(key)) keyLen = val;
        else if (/\.attention\.value_length$/.test(key)) valLen = val;
        else if (/\.context_length$/.test(key)) contextLength = val;
        else if (/\.expert_count$/.test(key)) nExperts = val;
      } else if (type === 10) {
        const val = Number(read(8).readBigUInt64LE(0));
        if (/\.context_length$/.test(key)) contextLength = val;
      } else {
        skipValue(type, 1);
      }
    }

    const layerBytesMap = new Map();
    let weightBytes = 0;
    for (let i = 0; i < nTensors; i++) {
      const name = str();
      const nDims = read(4).readUInt32LE(0);
      let nelems = 1;
      for (let j = 0; j < Math.min(nDims, 4); j++) {
        nelems *= Number(read(8).readBigUInt64LE(0));
      }
      const gtype = read(4).readInt32LE(0);
      read(8); // offset
      const info = GGML_BLOCK_INFO[gtype];
      if (!info) continue;
      const bytes = info[0] * (nelems / info[1]);
      weightBytes += bytes;
      const m = /^blk\.(\d+)\./.exec(name);
      if (m) {
        const idx = Number(m[1]);
        layerBytesMap.set(idx, (layerBytesMap.get(idx) || 0) + bytes);
      }
      if (/exps|expn|ffn_down_exp/.test(name)) isMoe = true;
    }

    if (nExperts > 0) isMoe = true;

    if (totalLayers <= 0) totalLayers = layerBytesMap.size;
    let blkBytes = 0;
    for (const b of layerBytesMap.values()) blkBytes += b;
    const layerBytes =
      layerBytesMap.size > 0 ? blkBytes / layerBytesMap.size : totalLayers > 0 ? weightBytes / totalLayers : 0;

    const nHeadKv = nHeadsKv || nHeads;
    const headDim = keyLen || valLen || (nHeads > 0 ? Math.round(nEmbds / nHeads) : 0);

    return {
      totalLayers,
      layerBytes: Math.round(layerBytes),
      weightBytes: Math.round(weightBytes),
      tensorCount: nTensors,
      fileBytes: fs.fstatSync(fd).size,
      headDim: headDim > 0 ? headDim : null,
      nHeadKv: nHeadKv > 0 ? nHeadKv : null,
      contextLength: contextLength > 0 ? contextLength : null,
      isMoe,
      nExperts: nExperts > 0 ? nExperts : null,
    };
  } catch (e) {
    return null;
  } finally {
    try {
      fs.closeSync(fd);
    } catch (e) {}
  }
}

const dataDir = path.join(app.getPath('userData'), 'llama-launcher');
const profilesFile = path.join(dataDir, 'profiles.json');
const settingsFile = path.join(dataDir, 'settings.json');

let state = { profiles: [], settings: {} };
let dlMgr = null;

function initDownloadManager() {
  dlMgr = new DownloadManager({
    dataDir,
    onProgress: (record, speed) => {
      send('dl:progress', {
        id: record.id,
        url: record.url,
        dest: record.dest,
        totalBytes: record.totalBytes,
        receivedBytes: record.receivedBytes,
        status: record.status,
        speed,
        meta: record.meta,
      });
    },
    onComplete: (record) => {
      send('dl:complete', {
        id: record.id,
        url: record.url,
        dest: record.dest,
        totalBytes: record.totalBytes,
        meta: record.meta,
      });
      dlNotifyDone(record);
    },
    onError: (record) => {
      send('dl:error', {
        id: record.id,
        url: record.url,
        dest: record.dest,
        error: record.error,
        meta: record.meta,
      });
      dlNotifyDone(record);
    },
    onPause: (record) => {
      send('dl:paused', {
        id: record.id,
        url: record.url,
        dest: record.dest,
        receivedBytes: record.receivedBytes,
        totalBytes: record.totalBytes,
        meta: record.meta,
      });
    },
    onResume: (record) => {
      send('dl:resumed', {
        id: record.id,
        url: record.url,
        dest: record.dest,
        meta: record.meta,
      });
    },
    onCancelled: (record) => {
      send('dl:cancelled', {
        id: record.id,
        url: record.url,
        meta: record.meta,
      });
      dlNotifyDone(record);
    },
  });
}

function ensureStore() {
  try {
    fs.mkdirSync(dataDir, { recursive: true });
  } catch (e) {}
  try {
    state.profiles = JSON.parse(fs.readFileSync(profilesFile, 'utf8'));
  } catch (e) {
    state.profiles = [];
  }
  if (!Array.isArray(state.profiles)) state.profiles = [];
  try {
    state.settings = JSON.parse(fs.readFileSync(settingsFile, 'utf8'));
  } catch (e) {
    state.settings = {};
  }
}

function persistProfiles() {
  try {
    fs.writeFileSync(profilesFile, JSON.stringify(state.profiles, null, 2));
  } catch (e) {}
}

function persistSettings() {
  try {
    fs.writeFileSync(settingsFile, JSON.stringify(state.settings, null, 2));
  } catch (e) {}
}

function send(channel, payload) {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send(channel, payload);
  }
}

function uid() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
}

function findServerExeIn(folder) {
  const candidates = [
    path.join(folder, 'llama-server.exe'),
    path.join(folder, 'build', 'bin', 'Release', 'llama-server.exe'),
    path.join(folder, 'build', 'bin', 'Debug', 'llama-server.exe'),
  ];
  for (const c of candidates) {
    try {
      if (fs.existsSync(c)) return c;
    } catch (e) {}
  }
  return null;
}

function findServer() {
  const cwd = process.cwd();
  const candidates = [];
  candidates.push(path.join(cwd, 'llama-server.exe'));
  try {
    const entries = fs.readdirSync(cwd, { withFileTypes: true });
    for (const e of entries) {
      if (e.isDirectory()) {
        candidates.push(path.join(cwd, e.name, 'llama-server.exe'));
        candidates.push(path.join(cwd, e.name, 'build', 'bin', 'Release', 'llama-server.exe'));
      }
    }
  } catch (e) {}
  for (const c of candidates) {
    try {
      if (fs.existsSync(c)) return c;
    } catch (e) {}
  }
  return '';
}

function parseArgError(text) {
  if (!text) return null;
  const lines = String(text).split(/\r?\n/);
  for (const raw of lines) {
    const line = raw.trim();
    if (!line) continue;
    let m = line.match(/error while handling argument "(--[a-z0-9][a-z0-9-]*)":\s*(?:error:\s*)?([^\n]+)/i);
    if (m) return { flag: m[1], detail: m[2].trim(), line };
    m = line.match(/error:\s*(--[a-z0-9][a-z0-9-]*)\b[^\n]*/i);
    if (m) return { flag: m[1], detail: line, line };
    m = line.match(/(?:unknown|unrecognized|invalid|unexpected|missing)\s+(?:option|argument)\s*[:=]?\s*["']?(--[a-z0-9][a-z0-9-]*)["']?/i);
    if (m) return { flag: m[1], detail: line, line };
    m = line.match(/(--[a-z0-9][a-z0-9-]*)\s*[:=]\s*(?:invalid|out of range)/i);
    if (m) return { flag: m[1], detail: line, line };
    m = line.match(/error:\s*(?:invalid|out of range|expected|must be)\s+[^\n]*?\s+(--[a-z0-9][a-z0-9-]*)\b/i);
    if (m) return { flag: m[1], detail: line, line };
  }
  return null;
}

function scanDir(dir, depth, out) {
  if (!dir || depth > 5) return out;
  let entries;
  try {
    entries = fs.readdirSync(dir, { withFileTypes: true });
  } catch (e) {
    return out;
  }
  for (const e of entries) {
    if (e.name.startsWith('.')) continue;
    const full = path.join(dir, e.name);
    if (e.isDirectory()) scanDir(full, depth + 1, out);
    else if (e.name.toLowerCase().endsWith('.gguf')) out.push(full);
  }
  return out;
}

/* ---------------- llama.cpp release downloader ---------------- */

let releasesCache = null;
let releasesCacheAt = 0;
let dlState = null; // { active, cancelled, work }

const LLAMA_DL_EVENT = 'llama:event';

function sendEvent(type, payload) {
  send(LLAMA_DL_EVENT, Object.assign({ type }, payload || {}));
}

function githubJson(url) {
  return new Promise((resolve, reject) => {
    const req = https.get(
      url,
      { headers: { 'User-Agent': 'llama-launcher', Accept: 'application/vnd.github+json' } },
      (res) => {
        const code = res.statusCode || 0;
        if (code >= 300 && code < 400 && res.headers.location) {
          res.resume();
          return githubJson(res.headers.location).then(resolve, reject);
        }
        if (code !== 200) {
          res.resume();
          return reject(new Error('El servidor respondió HTTP ' + code + '. Probablemente se alcanzó el límite de la API de GitHub; reintentá en unos minutos.'));
        }
        let body = '';
        res.setEncoding('utf8');
        res.on('data', (d) => {
          body += d;
        });
        res.on('end', () => {
          try {
            resolve(JSON.parse(body));
          } catch (e) {
            reject(new Error('Respuesta inválida del servidor de GitHub.'));
          }
        });
        res.on('error', reject);
      }
    );
    req.on('error', reject);
  });
}

function releaseWinAssets(release, arch) {
  const out = { cpu: null, vulkan: null, cuda: [] };
  const reCpu = new RegExp('^llama-b\\d+-bin-win-cpu-' + arch + '\\.zip$', 'i');
  const reVulkan = new RegExp('^llama-b\\d+-bin-win-vulkan-' + arch + '\\.zip$', 'i');
  const reCuda = new RegExp('^llama-b\\d+-bin-win-cuda-([\\d.]+)-' + arch + '\\.zip$', 'i');
  const reCudaLegacy = new RegExp('^llama-b\\d+-bin-win-(cuda|cublas)-' + arch + '\\.zip$', 'i');
  const reCudart = new RegExp('^cudart-llama-bin-win-cuda-([\\d.]+)-' + arch + '\\.zip$', 'i');
  const reCudartLegacy = new RegExp('^cudart-llama-bin-win-(cuda|cublas)-' + arch + '\\.zip$', 'i');
  // La API de GitHub devuelve los assets en orden alfabético: los zips "cudart-…"
  // aparecen antes que los "llama-b…", así que el cudart se recopila en una primera
  // pasada y se adjunta a su build CUDA en una segunda, sin depender del orden.
  const cudarts = [];
  for (const a of release.assets || []) {
    if (reCpu.test(a.name)) out.cpu = a;
    else if (reVulkan.test(a.name)) out.vulkan = a;
    else if (reCuda.test(a.name)) {
      out.cuda.push({ version: reCuda.exec(a.name)[1], asset: a, cudart: null });
    } else if (reCudaLegacy.test(a.name)) {
      out.cuda.push({ version: '', asset: a, cudart: null });
    } else if (reCudart.test(a.name)) {
      cudarts.push({ version: reCudart.exec(a.name)[1], asset: a });
    } else if (reCudartLegacy.test(a.name)) {
      cudarts.push({ version: '', asset: a });
    }
  }
  for (const c of cudarts) {
    const cu = out.cuda.find((x) => x.version === c.version);
    if (cu) cu.cudart = c.asset;
  }
  return out;
}

function releaseToView(r, win) {
  const base = { tag: r.tag_name, published: r.published_at, name: r.name || '' };
  const pick = (a) => (a ? { name: a.name, size: a.size, url: a.browser_download_url } : null);
  base.cpu = pick(win.cpu);
  base.vulkan = pick(win.vulkan);
  base.cuda = win.cuda.map((c) => ({
    version: c.version,
    name: c.asset.name,
    size: c.asset.size,
    url: c.asset.browser_download_url,
    cudart: pick(c.cudart),
  }));
  return base;
}

async function listLlamaReleases() {
  if (releasesCache && Date.now() - releasesCacheAt < 10 * 60 * 1000) return releasesCache;
  const data = await githubJson('https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=100');
  const arch = process.arch === 'arm64' ? 'arm64' : 'x64';
  const releases = [];
  for (const r of data || []) {
    if (r.prerelease) continue;
    const win = releaseWinAssets(r, arch);
    if (!win.cpu && !win.vulkan && win.cuda.length === 0) continue;
    releases.push(releaseToView(r, win));
  }
  releasesCache = releases;
  releasesCacheAt = Date.now();
  return releases;
}

function downloadFile(url, dest, onProgress, cancelFlag) {
  return new Promise((resolve, reject) => {
    const req = https.get(url, { headers: { 'User-Agent': 'llama-launcher' } }, (res) => {
      const code = res.statusCode || 0;
      if (code >= 300 && code < 400 && res.headers.location) {
        res.resume();
        let next = res.headers.location;
        try {
          next = new URL(next, url).href;
        } catch (e) {}
        return downloadFile(next, dest, onProgress, cancelFlag).then(resolve, reject);
      }
      if (code !== 200) {
        res.resume();
        return reject(new Error('HTTP ' + code));
      }
      const total = Number(res.headers['content-length']) || 0;
      const ws = fs.createWriteStream(dest);
      let received = 0;
      let last = 0;
      res.on('data', (chunk) => {
        if (cancelFlag && cancelFlag.cancelled) {
          ws.destroy();
          res.destroy();
          reject(new Error('Cancelled'));
          return;
        }
        received += chunk.length;
        const t = Date.now();
        if (t - last >= 120 || received >= total) {
          last = t;
          try {
            onProgress && onProgress({ received, total });
          } catch (e) {}
        }
      });
      res.pipe(ws);
      ws.on('error', reject);
      res.on('error', reject);
      ws.on('finish', () => {
        try {
          ws.close(() => resolve({ bytes: received, total }));
        } catch (e) {
          resolve({ bytes: received, total });
        }
      });
    });
    req.on('error', reject);
  });
}

function parseZipCentral(zipPath) {
  const fd = fs.openSync(zipPath, 'r');
  try {
    const stat = fs.fstatSync(fd);
    const tailLen = Math.min(stat.size, 65557);
    const tail = Buffer.alloc(tailLen);
    fs.readSync(fd, tail, 0, tailLen, stat.size - tailLen);
    const eocd = tail.lastIndexOf(Buffer.from([0x50, 0x4b, 0x05, 0x06]));
    if (eocd < 0) throw new Error('El archivo no es un ZIP válido.');
    let count = tail.readUInt16LE(eocd + 10);
    let cdSize = tail.readUInt32LE(eocd + 12);
    let cdOffset = tail.readUInt32LE(eocd + 16);
    if (count === 0xffff || cdSize === 0xffffffff || cdOffset === 0xffffffff) {
      const locIdx = eocd - 20;
      if (locIdx >= 0 && tail.readUInt32LE(locIdx) === 0x07064b50) {
        const z64Off = Number(tail.readBigUInt64LE(locIdx + 8));
        const z64 = Buffer.alloc(56);
        fs.readSync(fd, z64, 0, 56, z64Off);
        if (z64.readUInt32LE(0) === 0x06064b50) {
          count = Number(z64.readBigUInt64LE(32));
          cdSize = Number(z64.readBigUInt64LE(40));
          cdOffset = Number(z64.readBigUInt64LE(48));
        }
      }
    }
    const cd = Buffer.alloc(cdSize);
    fs.readSync(fd, cd, 0, cdSize, cdOffset);
    const entries = [];
    let pos = 0;
    for (let i = 0; i < count; i++) {
      if (cd.readUInt32LE(pos) !== 0x02014b50) throw new Error('Directorio central del ZIP corrupto.');
      let method = cd.readUInt16LE(pos + 10);
      let compSize = cd.readUInt32LE(pos + 20);
      let uncompSize = cd.readUInt32LE(pos + 24);
      const nameLen = cd.readUInt16LE(pos + 28);
      const extraLen = cd.readUInt16LE(pos + 30);
      const commentLen = cd.readUInt16LE(pos + 32);
      let localOff = cd.readUInt32LE(pos + 42);
      const name = cd.toString('utf8', pos + 46, pos + 46 + nameLen);
      if (compSize === 0xffffffff || uncompSize === 0xffffffff || localOff === 0xffffffff) {
        const extra = cd.subarray(pos + 46 + nameLen, pos + 46 + nameLen + extraLen);
        let ep = 0;
        while (ep + 4 <= extra.length) {
          const id = extra.readUInt16LE(ep);
          const sz = extra.readUInt16LE(ep + 2);
          if (id === 0x0001) {
            let q = ep + 4;
            if (uncompSize === 0xffffffff) {
              uncompSize = Number(extra.readBigUInt64LE(q));
              q += 8;
            }
            if (compSize === 0xffffffff) {
              compSize = Number(extra.readBigUInt64LE(q));
              q += 8;
            }
            if (localOff === 0xffffffff) {
              localOff = Number(extra.readBigUInt64LE(q));
              q += 8;
            }
            break;
          }
          ep += 4 + sz;
        }
      }
      entries.push({ method, compSize, uncompSize, name, localOff });
      pos += 46 + nameLen + extraLen + commentLen;
    }
    return { entries };
  } finally {
    fs.closeSync(fd);
  }
}

function sanitizeZipName(name) {
  const n = String(name).replace(/\\/g, '/');
  const parts = n.split('/').filter(Boolean);
  if (parts.some((p) => p === '..' || p.indexOf(':') >= 0)) throw new Error('Ruta insegura dentro del ZIP.');
  return parts.join(path.sep);
}

function streamEntry(src, dst) {
  return new Promise((resolve, reject) => {
    let bytes = 0;
    src.on('data', (d) => {
      bytes += d.length;
    });
    src.on('error', reject);
    dst.on('error', reject);
    dst.on('finish', () => resolve(bytes));
    src.pipe(dst);
  });
}

function readEntryData(fd, localOff, method, compSize, outPath) {
  const lh = Buffer.alloc(30);
  fs.readSync(fd, lh, 0, 30, localOff);
  const nameLen = lh.readUInt16LE(26);
  const extraLen = lh.readUInt16LE(28);
  const dataStart = localOff + 30 + nameLen + extraLen;
  const src = fs.createReadStream(null, { fd, start: dataStart, end: dataStart + compSize - 1, autoClose: false });
  const dst = fs.createWriteStream(outPath);
  if (method === 8) return streamEntry(src.pipe(zlib.createInflateRaw()), dst);
  if (method === 0) return streamEntry(src, dst);
  src.destroy();
  dst.destroy();
  return Promise.reject(new Error('Método de compresión ZIP no soportado (' + method + ').'));
}

async function extractZip(zipPath, destDir, onProgress, cancelFlag) {
  const { entries } = parseZipCentral(zipPath);
  const total = entries.reduce((a, e) => a + e.uncompSize, 0);
  let written = 0;
  const fd = fs.openSync(zipPath, 'r');
  try {
    for (const e of entries) {
      if (cancelFlag && cancelFlag.cancelled) throw new Error('Cancelled');
      const safe = sanitizeZipName(e.name);
      const outPath = path.join(destDir, safe);
      if (e.name.endsWith('/')) {
        fs.mkdirSync(outPath, { recursive: true });
        continue;
      }
      fs.mkdirSync(path.dirname(outPath), { recursive: true });
      written += await readEntryData(fd, e.localOff, e.method, e.compSize, outPath);
      try {
        onProgress && onProgress({ written, total, file: safe });
      } catch (err) {}
    }
  } finally {
    fs.closeSync(fd);
  }
  return { files: entries.filter((e) => !e.name.endsWith('/')).length };
}

function findServerExeRecursive(dir, depth) {
  if (depth > 6) return null;
  const direct = path.join(dir, 'llama-server.exe');
  try {
    if (fs.existsSync(direct)) return direct;
  } catch (e) {}
  let entries;
  try {
    entries = fs.readdirSync(dir, { withFileTypes: true });
  } catch (e) {
    return null;
  }
  for (const en of entries) {
    if (!en.isDirectory() || en.name.startsWith('.')) continue;
    const found = findServerExeRecursive(path.join(dir, en.name), depth + 1);
    if (found) return found;
  }
  return null;
}

function moveDir(src, dest) {
  try {
    fs.renameSync(src, dest);
    return;
  } catch (e) {}
  fs.mkdirSync(dest, { recursive: true });
  const stack = [src];
  while (stack.length) {
    const cur = stack.pop();
    let items;
    try {
      items = fs.readdirSync(cur, { withFileTypes: true });
    } catch (e) {
      continue;
    }
    for (const en of items) {
      const s = path.join(cur, en.name);
      const d = path.join(dest, path.relative(src, s));
      if (en.isDirectory()) {
        fs.mkdirSync(d, { recursive: true });
        stack.push(s);
      } else {
        fs.copyFileSync(s, d);
      }
    }
  }
  fs.rmSync(src, { recursive: true, force: true });
}

function buildTestZip(zipPath, files) {
  const chunks = [];
  const central = [];
  let offset = 0;
  for (const f of files) {
    const name = Buffer.from(f.name, 'utf8');
    let comp = f.data;
    let method = 0;
    if (f.method === 8) {
      comp = zlib.deflateRawSync(f.data);
      method = 8;
    }
    const crc = crc32(f.data);
    const lh = Buffer.alloc(30);
    lh.writeUInt32LE(0x04034b50, 0);
    lh.writeUInt16LE(20, 4);
    lh.writeUInt16LE(method, 8);
    lh.writeUInt32LE(crc, 14);
    lh.writeUInt32LE(comp.length, 18);
    lh.writeUInt32LE(f.data.length, 22);
    lh.writeUInt16LE(name.length, 26);
    const local = Buffer.concat([lh, name, comp]);
    const cd = Buffer.alloc(46);
    cd.writeUInt32LE(0x02014b50, 0);
    cd.writeUInt16LE(20, 4);
    cd.writeUInt16LE(20, 6);
    cd.writeUInt16LE(method, 10);
    cd.writeUInt32LE(crc, 16);
    cd.writeUInt32LE(comp.length, 20);
    cd.writeUInt32LE(f.data.length, 24);
    cd.writeUInt16LE(name.length, 28);
    cd.writeUInt32LE(offset, 42);
    central.push({ cd, name });
    chunks.push(local);
    offset += local.length;
  }
  const cdStart = offset;
  const cdBuf = Buffer.concat(central.map((c) => Buffer.concat([c.cd, c.name])));
  const eocd = Buffer.alloc(22);
  eocd.writeUInt32LE(0x06054b50, 0);
  eocd.writeUInt16LE(files.length, 8);
  eocd.writeUInt16LE(files.length, 10);
  eocd.writeUInt32LE(cdBuf.length, 12);
  eocd.writeUInt32LE(cdStart, 16);
  fs.writeFileSync(zipPath, Buffer.concat(chunks.concat([cdBuf, eocd])));
}

let crcTable = null;
function crc32(buf) {
  if (!crcTable) {
    crcTable = new Int32Array(256);
    for (let n = 0; n < 256; n++) {
      let c = n;
      for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
      crcTable[n] = c;
    }
  }
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i++) c = crcTable[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function llamaDownloadDir() {
  return path.join(app.getPath('userData'), 'llama');
}

const EXACT_PROTECTED = () =>
  [
    process.env.WINDIR,
    process.env.SystemRoot,
    process.env.ProgramFiles,
    process.env['ProgramFiles(x86)'],
    process.env.ProgramData,
    process.env.APPDATA,
    process.env.LOCALAPPDATA,
    process.env.USERPROFILE,
    app.getPath('home'),
    app.getPath('userData'),
    app.getPath('temp'),
  ]
    .filter(Boolean)
    .map((p) => path.resolve(p));

function deleteInstallDir(install) {
  const dir = path.resolve(install.path || '');
  if (!dir || path.parse(dir).root === dir) {
    throw new Error('No se puede eliminar una unidad de disco completa.');
  }
  if (!fs.existsSync(dir)) return 'missing';
  const stat = fs.statSync(dir);
  if (!stat.isDirectory()) throw new Error('La ruta no es una carpeta.');
  for (const root of EXACT_PROTECTED()) {
    if (dir === root) {
      throw new Error('No se puede eliminar: la carpeta está protegida por el sistema.');
    }
  }
  if (install.modelsDir && dir === path.resolve(install.modelsDir)) {
    throw new Error('No se puede eliminar la carpeta de modelos.');
  }
  if (serverProc && install.exePath) {
    const exe = path.resolve(install.exePath);
    if (exe === dir || exe.startsWith(dir + path.sep)) {
      throw new Error('Esa versión está en uso por el servidor en ejecución. Detenelo antes.');
    }
  }
  if (!findServerExeRecursive(dir, 0)) {
    throw new Error('La carpeta no contiene llama-server.exe; no se elimina por seguridad.');
  }
  fs.rmSync(dir, { recursive: true, force: true });
  return 'deleted';
}

function deleteModelsFolder(dir) {
  const target = path.resolve(dir || '');
  if (!target || path.parse(target).root === target) {
    throw new Error('No se puede eliminar una unidad de disco completa.');
  }
  if (!fs.existsSync(target)) return 'missing';
  const stat = fs.statSync(target);
  if (!stat.isDirectory()) throw new Error('La ruta no es una carpeta.');
  const allowed = [state.settings.modelsDir]
    .filter(Boolean)
    .map((d) => path.resolve(d));
  if (allowed.some((root) => target === root)) {
    throw new Error('No se puede eliminar la carpeta de modelos.');
  }
  const inside = allowed.some((root) => (target + path.sep).toLowerCase().startsWith((root + path.sep).toLowerCase()));
  if (!inside) {
    throw new Error('La carpeta no está dentro de las carpetas de modelos.');
  }
  if (serverProc) {
    throw new Error('Detené el servidor antes de eliminar modelos.');
  }
  fs.rmSync(target, { recursive: true, force: true });
  return 'deleted';
}

async function runLlamaDownload(opts) {
  const { tag, backend, cudaVersion } = opts || {};
  const fail = (msg) => {
    sendEvent('error', { message: msg });
    return { ok: false, error: msg };
  };
  if (dlState && dlState.active) return fail('Ya hay una descarga en curso.');
  if (!tag || !backend) return fail('Faltan datos de la descarga.');
  let releases;
  try {
    releases = await listLlamaReleases();
  } catch (e) {
    return fail(e && e.message ? e.message : 'No se pudo consultar las versiones de GitHub.');
  }
  const rel = releases.find((r) => r.tag === tag);
  if (!rel) return fail('No se encontró la versión ' + tag + '.');
  let assets = [];
  let label = '';
  if (backend === 'cpu') {
    if (!rel.cpu) return fail('Esta versión no tiene build de CPU para este sistema.');
    assets = [rel.cpu];
    label = 'CPU';
  } else if (backend === 'vulkan') {
    if (!rel.vulkan) return fail('Esta versión no tiene build de Vulkan para este sistema.');
    assets = [rel.vulkan];
    label = 'Vulkan';
  } else if (backend === 'cuda') {
    let cu = rel.cuda.find((c) => c.version === String(cudaVersion));
    if (!cu) cu = rel.cuda[rel.cuda.length - 1];
    if (!cu) return fail('Esta versión no tiene build de CUDA para este sistema.');
    assets = [cu].concat(cu.cudart ? [cu.cudart] : []);
    label = 'CUDA' + (cu.version ? ' ' + cu.version : '');
  } else {
    return fail('Backend inválido.');
  }

  const work = path.join(app.getPath('temp'), 'llama-dl-' + uid());
  const extractDir = path.join(work, 'extract');
  try {
    fs.mkdirSync(extractDir, { recursive: true });
  } catch (e) {
    return fail('No se pudo crear la carpeta temporal de descarga.');
  }
  dlState = { active: true, cancelled: false, work };
  try {
    for (const a of assets) {
      if (dlState.cancelled) throw new Error('Cancelled');
      const dest = path.join(work, path.basename(a.name));
      sendEvent('progress', { phase: 'download', label: 'Descargando ' + a.name, file: a.name, size: a.size });
      await downloadFile(
        a.url,
        dest,
        (p) => {
          sendEvent('progress', {
            phase: 'download',
            label: 'Descargando ' + a.name,
            file: a.name,
            size: a.size,
            received: p.received,
            total: p.total || a.size,
          });
        },
        dlState
      );
    }
    for (const a of assets) {
      if (dlState.cancelled) throw new Error('Cancelled');
      const zipPath = path.join(work, path.basename(a.name));
      sendEvent('progress', { phase: 'extract', label: 'Extrayendo ' + a.name, file: a.name });
      await extractZip(
        zipPath,
        extractDir,
        (p) => {
          sendEvent('progress', { phase: 'extract', label: 'Extrayendo ' + a.name, file: p.file, written: p.written, total: p.total });
        },
        dlState
      );
    }
    if (dlState.cancelled) throw new Error('Cancelled');
    const exe = findServerExeRecursive(extractDir, 0);
    if (!exe) throw new Error('No se encontró llama-server.exe en el paquete descargado.');
    const finalDir = path.join(llamaDownloadDir(), 'v' + tag);
    sendEvent('progress', { phase: 'move', label: 'Instalando en ' + finalDir });
    if (fs.existsSync(finalDir)) fs.rmSync(finalDir, { recursive: true, force: true });
    fs.mkdirSync(path.dirname(finalDir), { recursive: true });
    moveDir(extractDir, finalDir);
    const exeFinal = path.join(finalDir, path.relative(extractDir, exe));
    const install = { id: uid(), name: 'llama-' + tag + ' (' + label + ')', path: finalDir, exePath: exeFinal };
    if (!Array.isArray(state.settings.installations)) state.settings.installations = [];
    state.settings.installations.push(install);
    persistSettings();
    sendEvent('done', { install });
    return { ok: true, install };
  } catch (e) {
    const cancelled = dlState.cancelled;
    const msg = e && e.message ? e.message : String(e);
    if (cancelled || msg === 'Cancelled') {
      sendEvent('cancelled', { message: 'Descarga cancelada.' });
      return { ok: false, error: 'Cancelled' };
    }
    sendEvent('error', { message: msg });
    return { ok: false, error: msg };
  } finally {
    dlState = null;
    try {
      fs.rmSync(work, { recursive: true, force: true });
    } catch (e) {}
  }
}

/* ---------------- HuggingFace model downloader ---------------- */

let mdlState = null; // { active, cancelled }

const HF_DL_EVENT = 'model:event';

function sendModelEvent(type, payload) {
  send(HF_DL_EVENT, Object.assign({ type }, payload || {}));
}

function hfJson(url) {
  return new Promise((resolve, reject) => {
    const req = https.get(url, { headers: { 'User-Agent': 'llama-launcher' } }, (res) => {
      const code = res.statusCode || 0;
      if (code >= 300 && code < 400 && res.headers.location) {
        res.resume();
        return hfJson(res.headers.location).then(resolve, reject);
      }
      if (code !== 200) {
        res.resume();
        return reject(new Error('HuggingFace respondió HTTP ' + code + '.'));
      }
      let body = '';
      res.setEncoding('utf8');
      res.on('data', (d) => {
        body += d;
      });
      res.on('end', () => {
        try {
          resolve(JSON.parse(body));
        } catch (e) {
          reject(new Error('Respuesta inválida de HuggingFace.'));
        }
      });
      res.on('error', reject);
    });
    req.on('error', reject);
  });
}

function encodeHfPath(p) {
  return String(p)
    .split('/')
    .map((s) => encodeURIComponent(s))
    .join('/');
}

function hfNum(v) {
  return v === null || v === undefined || Number.isNaN(Number(v)) ? null : Number(v);
}

function hfGetText(url) {
  return new Promise((resolve, reject) => {
    const req = https.get(url, { headers: { 'User-Agent': 'llama-launcher' } }, (res) => {
      const code = res.statusCode || 0;
      if (code >= 300 && code < 400 && res.headers.location) {
        res.resume();
        return hfGetText(new URL(res.headers.location, url).href).then(resolve, reject);
      }
      if (code !== 200) {
        res.resume();
        return reject(new Error('HuggingFace respondió HTTP ' + code + '.'));
      }
      let body = '';
      res.setEncoding('utf8');
      res.on('data', (d) => {
        body += d;
      });
      res.on('end', () => resolve(body));
      res.on('error', reject);
    });
    req.on('error', reject);
  });
}

async function hfConfigJson(repo) {
  try {
    const raw = await hfGetText('https://huggingface.co/' + encodeHfPath(repo) + '/resolve/main/config.json');
    return JSON.parse(raw);
  } catch (e) {
    return null;
  }
}

async function hfGenerationConfig(repo) {
  try {
    const raw = await hfGetText('https://huggingface.co/' + encodeHfPath(repo) + '/resolve/main/generation_config.json');
    return JSON.parse(raw);
  } catch (e) {
    return null;
  }
}

function parseHfFrontMatterParams(text) {
  const out = {};
  const lines = String(text || '').split(/\r?\n/);
  if (!lines.length || lines[0].trim() !== '---') return out;
  const fm = [];
  let i = 1;
  for (; i < lines.length; i++) {
    if (lines[i].trim() === '---') break;
    fm.push(lines[i]);
  }
  let inParams = false;
  for (const line of fm) {
    if (!inParams) {
      if (/^parameters:\s*$/.test(line)) {
        inParams = true;
        continue;
      }
      continue;
    }
    if (!/^\s/.test(line)) break;
    const m = line.match(/^\s+([A-Za-z_][\w-]*)\s*:\s*(.*)$/);
    if (!m) continue;
    let v = String(m[2]).trim();
    if (!v || v.startsWith('#')) continue;
    v = v.replace(/^['"]|['"]$/g, '');
    out[m[1]] = v;
  }
  return out;
}

function parseHfReadmeSampling(text) {
  const out = {};
  const fm = parseHfFrontMatterParams(text);
  const keys = ['temperature', 'top_p', 'top_k', 'min_p', 'presence_penalty', 'repetition_penalty'];
  const lines = String(text || '').split(/\r?\n/);
  for (const line of lines) {
    for (const k of keys) {
      const m = line.match(new RegExp('\\b' + k.replace(/_/g, '[_]') + '\\s*=\\s*([0-9]*\\.?[0-9]+)'));
      if (m) out[k] = m[1];
    }
  }
  for (const k of keys) {
    if (fm[k] !== undefined) out[k] = fm[k];
  }
  return out;
}

async function hfReadmeParams(repo) {
  try {
    const raw = await hfGetText('https://huggingface.co/' + encodeHfPath(repo) + '/raw/main/README.md');
    return parseHfReadmeSampling(raw);
  } catch (e) {
    return {};
  }
}

function hfPickGenParams(...sources) {
  const res = {};
  const pick = (name, key) => {
    for (const src of sources) {
      if (src && typeof src === 'object' && src[name] != null && res[key] == null) res[key] = hfNum(src[name]);
    }
  };
  pick('temperature', 'temp');
  pick('top_p', 'topP');
  pick('top_k', 'topK');
  pick('min_p', 'minP');
  pick('presence_penalty', 'presencePenalty');
  pick('repetition_penalty', 'repeatPenalty');
  return res;
}

async function hfModelConfig(repo) {
  const clean = String(repo || '')
    .trim()
    .replace(/^https?:\/\/(?:www\.)?huggingface\.co\//i, '')
    .replace(/\/+$/, '');
  if (!clean) throw new Error('Indicá el repositorio en Hugging Face (ej. ggml-org/Qwen3-8B-GGUF).');
  const data = await hfJson('https://huggingface.co/api/models/' + encodeHfPath(clean));
  const tags = Array.isArray(data.tags) ? data.tags : [];
  let cfg = await hfConfigJson(clean);
  let sourceRepo = clean;
  const cfgHasCtx = (c) =>
    c && typeof c === 'object' && (c.max_position_embeddings != null || (c.text_config && c.text_config.max_position_embeddings != null));
  if (!cfgHasCtx(cfg)) {
    let base = null;
    const bm = data.cardData && data.cardData.base_model;
    if (typeof bm === 'string') base = bm;
    else if (Array.isArray(bm) && bm.length) base = bm[0];
    if (!base) {
      const tag = tags.find((t) => /^base_model:[^:]+$/.test(String(t)));
      if (tag) base = String(tag).replace(/^base_model:/, '');
    }
    if (base && base !== clean) {
      const bcfg = await hfConfigJson(base);
      if (cfgHasCtx(bcfg)) {
        cfg = bcfg;
        sourceRepo = base;
      }
    }
  }
  const c = cfg && typeof cfg === 'object' ? cfg : {};
  const textCfg = c.text_config && typeof c.text_config === 'object' ? c.text_config : c;
  const visionCfg = c.vision_config && typeof c.vision_config === 'object' ? c.vision_config : null;
  let ctxSize = hfNum(c.sliding_window);
  if (ctxSize === null) ctxSize = hfNum(textCfg.max_position_embeddings) || hfNum(c.max_position_embeddings);
  if (ctxSize === null) ctxSize = hfNum(visionCfg && visionCfg.max_position_embeddings);
  const rope = c.rope_scaling && typeof c.rope_scaling === 'object' ? c.rope_scaling : null;
  const modelType =
    c.model_type ||
    (Array.isArray(c.architectures) ? c.architectures[0] : '') ||
    (Array.isArray(data.architectures) ? data.architectures[0] : '') ||
    '';
  const nExperts = hfNum(c.num_experts) || hfNum(c.n_routed_experts) || hfNum(c.num_local_experts) || hfNum(textCfg.num_experts) || null;
  const isMoe = !!(
    nExperts ||
    /moe|mixtral|deepseek|qwen.*moe|olmoe/i.test(String(modelType || '')) ||
    (Array.isArray(c.architectures) && c.architectures.some((a) => /moe|mixtral/i.test(String(a)))) ||
    tags.some((t) => /moe|mixture-of-experts/i.test(String(t)))
  );
  const genCfgRepo = await hfGenerationConfig(clean);
  const genCfgBase = sourceRepo !== clean ? await hfGenerationConfig(sourceRepo) : null;
  const readmeRepo = await hfReadmeParams(clean);
  const readmeBase = sourceRepo !== clean ? await hfReadmeParams(sourceRepo) : null;
  const gen = hfPickGenParams(genCfgBase, genCfgRepo, readmeBase, readmeRepo);
  return {
    repo: clean,
    source: sourceRepo,
    modelType: String(modelType),
    isMoe,
    nExperts: nExperts && nExperts > 0 ? nExperts : null,
    ctxSize: ctxSize && ctxSize > 0 && ctxSize < 20000000 ? Math.round(ctxSize) : null,
    isVision: !!(
      visionCfg ||
      /(vision|image|_vl)/i.test(String(modelType || '')) ||
      tags.some((t) => /image-text|vision/i.test(String(t)))
    ),
    ropeType: rope ? String(rope.type || '') : '',
    ropeFactor: rope && hfNum(rope.factor) !== null ? hfNum(rope.factor) : null,
    temp: gen.temp != null ? gen.temp : null,
    topP: gen.topP != null ? gen.topP : null,
    topK: gen.topK != null ? gen.topK : null,
    minP: gen.minP != null ? gen.minP : null,
    presencePenalty: gen.presencePenalty != null ? gen.presencePenalty : null,
    repeatPenalty: gen.repeatPenalty != null ? gen.repeatPenalty : null,
  };
}

const HF_CATEGORY_TAGS = {
  llm: ['text-generation'],
  vision: ['image-text-to-text', 'image-to-text'],
  t2i: ['text-to-image'],
  audio: ['text-to-audio', 'audio-to-text', 'audio-to-audio', 'automatic-speech-recognition', 'text-to-speech', 'text-to-music', 'audio-text-to-text'],
  embed: ['feature-extraction', 'sentence-similarity'],
};

function parseModelParams(id) {
  const clean = String(id || '')
    .split('/')
    .pop();
  const m = clean.match(/(\d+(?:\.\d+)?)-?B/);
  return m ? Number(m[1]) : null;
}

async function searchHfModels({ query, category } = {}) {
  const tags = HF_CATEGORY_TAGS[category] || [];
  const q = String(query || '').trim();
  const fetchFor = (tag) => {
    let url = 'https://huggingface.co/api/models?sort=downloads&direction=-1&limit=100&full=false';
    if (q) url += '&search=' + encodeURIComponent(q);
    if (tag) url += '&filter=' + encodeURIComponent(tag);
    return hfJson(url);
  };
  const lists = tags.length ? await Promise.all(tags.map(fetchFor)) : [await fetchFor(null)];
  const seen = new Set();
  const models = [];
  for (const list of lists) {
    for (const m of list || []) {
      if (!m || !m.id || seen.has(m.id)) continue;
      seen.add(m.id);
      models.push({
        id: m.id,
        downloads: m.downloads || 0,
        likes: m.likes || 0,
        task: m.pipeline_tag || '',
        params: parseModelParams(m.id),
      });
    }
  }
  models.sort((a, b) => (b.downloads || 0) - (a.downloads || 0));
  return models;
}

async function listHfRepoFiles(repo) {
  const url = 'https://huggingface.co/api/models/' + encodeHfPath(repo) + '/tree/main?recursive=false&limit=1000';
  const data = await hfJson(url);
  const files = [];
  for (const f of data || []) {
    if (!f || f.type === 'directory') continue;
    files.push({ name: f.path, size: f.size || 0 });
  }
  return files;
}

function classifyHfFiles(files) {
  const main = [];
  const vision = [];
  const mtp = [];
  for (const f of files) {
    const name = f.name || '';
    if (!/\.gguf$/i.test(name)) continue;
    if (/mmproj|vision-?proj|clip|image/i.test(name)) {
      vision.push(f);
      continue;
    }
    if (/mtp|draft/i.test(name)) {
      mtp.push(f);
      continue;
    }
    main.push(f);
  }
  main.sort((a, b) => (a.size || 0) - (b.size || 0));
  const quantOf = (name) => {
    const base = String(name).replace(/\.gguf$/i, '');
    const seg = base.split('-').pop();
    return seg || base;
  };
  return {
    main: main.map((f) => Object.assign({}, f, { quant: quantOf(f.name) })),
    vision,
    mtp,
  };
}

function hfFileUrl(repo, file) {
  return 'https://huggingface.co/' + encodeHfPath(repo) + '/resolve/main/' + encodeHfPath(file) + '?download=true';
}

const dlWaiters = new Map();

function dlWaitFor(dlId) {
  return new Promise((resolve) => {
    dlWaiters.set(dlId, resolve);
  });
}

function dlNotifyDone(record) {
  const resolve = dlWaiters.get(record.id);
  if (!resolve) return;
  dlWaiters.delete(record.id);
  if (record.status === 'completed') resolve({ ok: true });
  else if (record.status === 'error') resolve({ ok: false, error: record.error });
  else resolve({ ok: false, error: record.status });
}

async function runModelDownload(opts) {
  const { repo, folder, files } = opts || {};
  const fail = (msg) => {
    sendModelEvent('error', { message: msg });
    return { ok: false, error: msg };
  };
  if (mdlState && mdlState.active) return fail('Ya hay una descarga de modelo en curso.');
  const modelsDir = state.settings.modelsDir;
  if (!modelsDir) return fail('Falta la carpeta de modelos (Ajustes → General).');
  if (!repo || !Array.isArray(files) || !files.length) return fail('No hay archivos para descargar.');
  const safeFolder = String(folder || 'modelo')
    .replace(/[\\/:*?"<>|]+/g, '-')
    .trim();
  const destDir = path.join(modelsDir, safeFolder);
  mdlState = { active: true, cancelled: false, dlIds: [] };
  try {
    fs.mkdirSync(destDir, { recursive: true });
    const names = files.map((f) => (typeof f === 'string' ? f : f.name));
    const dlIds = [];
    for (let i = 0; i < names.length; i++) {
      const name = names[i];
      const dest = path.join(destDir, String(name).replace(/[\\/]/g, path.sep));
      fs.mkdirSync(path.dirname(dest), { recursive: true });
      const dlId = 'hf-' + repo.replace(/[/:]/g, '-') + '-' + name.replace(/[\\/]/g, '-');
      dlIds.push(dlId);
      dlMgr.register({
        id: dlId,
        url: hfFileUrl(repo, name),
        dest,
        meta: { repo, file: name, index: i, count: names.length, folder: safeFolder },
      });
    }
    mdlState.dlIds = dlIds;

    const promises = names.map((name, i) => {
      const dlId = dlIds[i];
      sendModelEvent('progress', { phase: 'prepare', label: 'Descargando ' + name, file: name, index: i, count: names.length });
      const waiter = dlWaitFor(dlId);
      const result = dlMgr.start({
        id: dlId,
        onProgress: (record, speed) => {
          sendModelEvent('progress', {
            phase: 'download',
            label: 'Descargando ' + name,
            file: name,
            index: i,
            count: names.length,
            received: record.receivedBytes,
            total: record.totalBytes,
            speed,
            dlId: record.id,
          });
        },
      });
      if (!result.ok) return { ok: false, error: result.error };
      return waiter;
    });

    const results = await Promise.all(promises);
    if (mdlState.cancelled) throw new Error('Cancelled');
    const failed = results.find((r) => !r.ok);
    if (failed) throw new Error(failed.error || 'Download failed');

    sendModelEvent('done', { folder: safeFolder, dir: destDir, repo });
    return { ok: true, dir: destDir };
  } catch (e) {
    const cancelled = mdlState.cancelled;
    const msg = e && e.message ? e.message : String(e);
    if (cancelled || msg === 'Cancelled') {
      sendModelEvent('cancelled', { message: 'Descarga cancelada.' });
      return { ok: false, error: 'Cancelled' };
    }
    sendModelEvent('error', { message: msg });
    return { ok: false, error: msg };
  } finally {
    mdlState = null;
  }
}

async function runSmoke(win) {
  const wc = win.webContents;
  const origProfiles = JSON.stringify(state.profiles);
  const origSettings = JSON.stringify(state.settings);
  const fakeFolder = path.join(app.getPath('temp'), 'llama-launcher-smoke-' + Date.now());
  try {
  const zipDir = path.join(fakeFolder, 'ziptest');
  fs.mkdirSync(zipDir, { recursive: true });
  const zipFile = path.join(zipDir, 'test.zip');
  buildTestZip(zipFile, [
    { name: 'bin/llama-server.exe', data: Buffer.from('MZfake'), method: 8 },
    { name: 'readme.txt', data: Buffer.from('hola smoke', 'utf8'), method: 0 },
  ]);
  const zipTest = await extractZip(zipFile, path.join(zipDir, 'out'), () => {});
  const exeFound = fs.existsSync(path.join(zipDir, 'out', 'bin', 'llama-server.exe'));
  const txtOk = fs.readFileSync(path.join(zipDir, 'out', 'readme.txt'), 'utf8') === 'hola smoke';
  let ready = false;
  for (let i = 0; i < 40 && !ready; i++) {
    await new Promise((r) => setTimeout(r, 250));
    try {
      ready = (await wc.executeJavaScript(`document.querySelectorAll('#profileList .profile-item').length`)) > 0;
    } catch (e) {}
  }
  const results = await wc.executeJavaScript(`(() => ({
    api: typeof window.api,
    profiles: document.querySelectorAll('#profileList .profile-item').length,
    samplingFields: document.querySelectorAll('#panel-sampling .field').length,
    renderingFields: document.querySelectorAll('#panel-rendering .field').length,
    modalHidden: document.getElementById('modal').className,
    logEmpty: document.getElementById('log').childNodes.length,
    name: document.getElementById('nameInput').value,
    installSelect: !!document.getElementById('installSelect'),
    installList: !!document.getElementById('installList'),
    addInstallBtn: !!document.getElementById('addInstallBtn'),
    installModal: !!document.getElementById('installModal'),
    installSelectOptions: document.getElementById('installSelect').options.length,
    dashboardVisible: !document.getElementById('dashboardView').classList.contains('hidden'),
    editorHidden: document.getElementById('editorView').classList.contains('hidden'),
    dashCards: document.querySelectorAll('#dashGrid .dash-card').length,
    dashEmptyHidden: document.getElementById('dashEmpty').classList.contains('hidden'),
    consoleCollapsedDefault: document.getElementById('statsBar').classList.contains('collapsed'),
    consoleToggle: !!document.getElementById('bottomCollapseBtn'),
    bottomTabs: Array.from(document.querySelectorAll('.stats-head .tab')).map((t) => t.textContent),
    titlebar: !!document.getElementById('titlebar'),
    winControls: typeof window.api.windowControls,
    winButtons: document.querySelectorAll('#titlebar .win-btn').length,
    gearBtn: !!document.getElementById('gearBtn'),
    gearInTitlebar: !!document.querySelector('.titlebar .win-btn[title="Ajustes"]'),
    topbarSettingsGone: !document.getElementById('dashSettingsBtn') && !document.getElementById('settingsBtn'),
    mtpPresets: document.querySelectorAll('.mtp-presets .preset-btn').length,
    mtpPresetLabels: Array.from(document.querySelectorAll('.mtp-presets .preset-btn')).map((b) => b.textContent),
    kvDescRendered: Array.from(document.querySelectorAll('#panel-rendering .card select option')).some((o) => o.textContent.includes('—')),
    kvCacheOptions: Array.from(document.querySelectorAll('#panel-rendering .card select option')).map((o) => o.textContent),
    kvPresets: document.querySelectorAll('.kv-preset-row .preset-btn').length,
    kvPresetLabels: Array.from(document.querySelectorAll('.kv-preset-row .preset-btn')).map((b) => b.textContent),
    kvPresetDesc: !!document.querySelector('.kv-presets .preset-desc'),
    ctxCardHasKv: (() => {
      const card = [...document.querySelectorAll('#panel-rendering .card')].find((c) => {
        const l = c.querySelector('.field-head label');
        return l && l.textContent.startsWith('Contexto');
      });
      return (
        !!card &&
        !!card.querySelector('select[data-key="cacheTypeK"]') &&
        !!card.querySelector('select[data-key="cacheTypeV"]') &&
        !!card.querySelector('.kv-preset-row')
      );
    })(),
    gpuLayersSlider: (() => {
      const card = [...document.querySelectorAll('#panel-rendering .card')].find(
        (c) => c.querySelector('.field-head label') && c.querySelector('.field-head label').textContent.startsWith('Capas en GPU')
      );
      const wrap = card ? card.querySelector('.field[data-field-wrap="gpuLayers"]') : null;
      return !!wrap && !!wrap.querySelector('input[type="range"]') && !wrap.querySelector('input[type="number"]');
    })(),
    gpuCardGrouped: (() => {
      const card = [...document.querySelectorAll('#panel-rendering .card')].find(
        (c) => c.querySelector('.field-head label') && c.querySelector('.field-head label').textContent.startsWith('Capas en GPU')
      );
      return (
        !!card &&
        ['parallel', 'imageMinTokens', 'fit', 'reasoning'].every((k) => !!card.querySelector('.field[data-field-wrap="' + k + '"]'))
      );
    })(),
    dockTest: (() => {
      const sb = document.getElementById('statsBar');
      const tab = (n) => document.getElementById('bottomTab' + n[0].toUpperCase() + n.slice(1));
      const panel = (n) => document.getElementById('bottom' + n[0].toUpperCase() + n.slice(1));
      tab('consola').click();
      const onConsola = panel('consola').classList.contains('active') && !panel('recursos').classList.contains('active') && tab('consola').classList.contains('active');
      tab('recursos').click();
      const onRecursos = panel('recursos').classList.contains('active') && !panel('consola').classList.contains('active') && tab('recursos').classList.contains('active');
      sb.classList.remove('collapsed');
      document.getElementById('bottomCollapseBtn').click();
      const collapsed = sb.classList.contains('collapsed');
      document.getElementById('bottomCollapseBtn').click();
      const expanded = !sb.classList.contains('collapsed');
      return { onConsola, onRecursos, collapsed, expanded };
    })(),
    cmdTest: (() => {
      addProfile();
      const p = profiles[profiles.length - 1];
      p.modelPath = 'C:\\Users\\test\\My Models\\model.gguf';
      p.host = '127.0.0.1';
      if (settings.installations && settings.installations[0]) p.installId = settings.installations[0].id;
      refresh();
      document.querySelector('.tabs .tab[data-tab="command"]').click();
      const area = document.getElementById('cmdTextarea');
      const before = area.value;
      const hasModel = before.includes('--model');
      const pathQuoted = /^"[^"]+llama-server\.exe"/.test(before.trim());
      const hostBefore = (before.match(/--host (\\S+)/) || [])[1] || null;
      let error = null;
      try {
        area.value = before.replace(/--host (\\S+)/, '--host 10.20.30.40');
        document.getElementById('cmdApplyBtn').click();
      } catch (e) { error = String(e); }
      const after = document.getElementById('cmdTextarea').value;
      const hostAfter = (after.match(/--host (\\S+)/) || [])[1] || null;
      const hostFieldSynced = document.getElementById('hostInput').value === '10.20.30.40';
      const statusText = document.getElementById('cmdStatus').textContent;
      const statusOk = statusText.includes('Aplicado');
      const reverted = (() => { area.value = before; document.getElementById('cmdApplyBtn').click(); return ((document.getElementById('cmdTextarea').value.match(/--host (\\S+)/) || [])[1] || null) === hostBefore; })();
      profiles = profiles.filter((x) => x.id !== p.id);
      selectProfile(null);
      refresh();
      return { hasModel, pathQuoted, hostBefore, hostAfter, hostFieldSynced, statusOk, error, reverted };
    })(),
  }))()`);

  const spawnTest = await wc.executeJavaScript(`
    window.api.startServer({ exe: 'C:\\\\Windows\\\\System32\\\\cmd.exe', args: ['/c', 'echo LLAMA_SMOKE_OK'] })
  `);
  await new Promise((r) => setTimeout(r, 2000));
  const status = await wc.executeJavaScript(`window.api.getServerStatus()`);
  const log = await wc.executeJavaScript(`document.getElementById('log').textContent`);
  await wc.executeJavaScript(`window.api.stopServer()`);

  try {
    fs.mkdirSync(fakeFolder, { recursive: true });
    fs.copyFileSync(process.env.ComSpec, path.join(fakeFolder, 'llama-server.exe'));
  } catch (e) {}
  const ggufFile = path.join(fakeFolder, 'smoke.gguf');
  const u64 = (v) => {
    const x = Buffer.alloc(8);
    x.writeBigUInt64LE(BigInt(v));
    return x;
  };
  const u32 = (v) => {
    const x = Buffer.alloc(4);
    x.writeUInt32LE(v);
    return x;
  };
  const gstr = (s) => {
    const x = Buffer.from(s, 'utf8');
    return Buffer.concat([u64(x.length), x]);
  };
  fs.writeFileSync(
    ggufFile,
    Buffer.concat([
      Buffer.from('GGUF'),
      u32(3), // version
      u64(2), // n_tensors
      u64(7), // n_kv
      gstr('general.alignment'), u32(4), u32(32),
      gstr('llama.block_count'), u32(4), u32(33),
      gstr('llama.embedding_length'), u32(4), u32(2048),
      gstr('llama.attention.head_count'), u32(4), u32(32),
      gstr('llama.attention.head_count_kv'), u32(4), u32(8),
      gstr('llama.attention.key_length'), u32(4), u32(128),
      gstr('llama.attention.value_length'), u32(4), u32(128),
      gstr('blk.0.attn_q.weight'), u32(2), u64(32), u64(32), u32(1), u64(0),
      gstr('tok_embeddings.weight'), u32(2), u64(32), u64(32), u32(1), u64(2048),
    ])
  );
  const ggufMain = inspectGgufModel(ggufFile);
  const verified = findServerExeIn(fakeFolder);
  state.settings = {
    installations: [{ id: 'smoke1', name: 'Smoke v1', path: fakeFolder, exePath: verified }],
    modelsDir: fakeFolder,
  };
  persistSettings();

  await wc.reload();
  let ready2 = false;
  for (let i = 0; i < 40 && !ready2; i++) {
    await new Promise((r) => setTimeout(r, 250));
    try {
      ready2 = (await wc.executeJavaScript(`document.querySelectorAll('#profileList .profile-item').length`)) > 0;
    } catch (e) {}
  }
  if (!ready2) {
    await wc.executeJavaScript(`addProfile(); refresh(); true`);
    await new Promise((r) => setTimeout(r, 300));
  }
  const installTest = await wc.executeJavaScript(`({
    selectOptions: document.getElementById('installSelect').options.length,
    summary: document.getElementById('installSummary').textContent,
  })`);

  const navTest = await wc.executeJavaScript(`(async () => {
    const emptyHidden = document.getElementById('dashEmpty').classList.contains('hidden');
    document.querySelector('#profileList .profile-item').click();
    await new Promise(r => setTimeout(r, 200));
    const inEditor = !document.getElementById('editorView').classList.contains('hidden')
      && document.getElementById('dashboardView').classList.contains('hidden');
    const wizardClosed = document.getElementById('wizardModal').classList.contains('hidden');
    const openedName = document.getElementById('nameInput').value;
    const cardsInEditor = document.querySelectorAll('#dashGrid .dash-card').length;
    document.getElementById('backBtn').click();
    await new Promise(r => setTimeout(r, 200));
    const backDashboard = !document.getElementById('dashboardView').classList.contains('hidden')
      && document.getElementById('editorView').classList.contains('hidden');
    const cardsOnDash = document.querySelectorAll('#dashGrid .dash-card').length;
    const dashNavActive = document.getElementById('dashNavBtn').classList.contains('active');
    return { emptyHidden, inEditor, wizardClosed, openedName, cardsInEditor, backDashboard, cardsOnDash, dashNavActive };
  })()`);

  const dashSelTest = await wc.executeJavaScript(`(async () => {
    const btn = document.getElementById('dashSelBtn');
    const bar = document.getElementById('dashSelBar');
    btn.click();
    await new Promise(r => setTimeout(r, 150));
    const barVisible = !bar.classList.contains('hidden');
    const selecting = !!document.querySelector('.dash-card.selecting');
    const nBefore = document.querySelectorAll('#dashGrid .dash-card').length;
    document.querySelector('#dashGrid .dash-card').click();
    await new Promise(r => setTimeout(r, 150));
    const oneSelected = !!document.querySelector('.dash-card.selected');
    const countText = document.getElementById('dashSelCount').textContent;
    document.getElementById('dashSelAllBtn').click();
    await new Promise(r => setTimeout(r, 150));
    const allSelected = document.querySelectorAll('.dash-card.selected').length === nBefore;
    const allBtnLabel = document.getElementById('dashSelAllBtn').textContent;
    document.getElementById('dashSelNoneBtn').click();
    await new Promise(r => setTimeout(r, 150));
    const noneSelected = document.querySelectorAll('.dash-card.selected').length === 0;
    document.getElementById('dashSelCancelBtn').click();
    await new Promise(r => setTimeout(r, 150));
    const barHidden = bar.classList.contains('hidden');
    const noSelecting = !document.querySelector('.dash-card.selecting');
    return { barVisible, selecting, oneSelected, countText, allSelected, allBtnLabel, noneSelected, barHidden, noSelecting };
  })()`);

  await new Promise((r) => setTimeout(r, 2500));
  const statsTest = await wc.executeJavaScript(`({
    cpu: document.getElementById('statCpu').textContent,
    ram: document.getElementById('statRam').textContent,
    gpu: document.getElementById('statGpu').textContent,
    canvasW: document.getElementById('statsCanvas').width,
    canvasH: document.getElementById('statsCanvas').height,
    barVisible: !document.getElementById('statsBar').classList.contains('hidden'),
  })`);

  const mtpTest = await wc.executeJavaScript(`(async () => {
    const cardByLabel = (label) => [...document.querySelectorAll('#panel-rendering .card')]
      .find((c) => c.querySelector('.field-head label') && c.querySelector('.field-head label').textContent === label);
    const inputOf = (label) => cardByLabel(label).querySelector('input');
    document.querySelector('#profileList .profile-item').click();
    await new Promise((r) => setTimeout(r, 200));
    const mtpCheck = cardByLabel('MTP (Multi-Token Prediction)').querySelector('input[type="checkbox"]');
    mtpCheck.click();
    await new Promise((r) => setTimeout(r, 200));
    const presetBtn = [...document.querySelectorAll('.mtp-presets .preset-btn')].find((b) => b.textContent === 'Máx. velocidad');
    presetBtn.click();
    await new Promise((r) => setTimeout(r, 200));
    const nmaxVal = inputOf('Draft: tokens a predecir').value;
    const nminVal = inputOf('Draft: tokens mínimos').value;
    const psplitVal = cardByLabel('Draft: probabilidad de split').querySelector('input[type="number"]').value;
    const activeLabel = (document.querySelector('.mtp-presets .preset-btn.active') || {}).textContent || '';
    const nmaxInput = inputOf('Draft: tokens a predecir');
    nmaxInput.value = 7;
    nmaxInput.dispatchEvent(new Event('input'));
    await new Promise((r) => setTimeout(r, 200));
    const hasCustom = !!document.querySelector('.preset-custom');
    document.getElementById('backBtn').click();
    return { nmaxVal, nminVal, psplitVal, activeLabel, hasCustom };
  })()`);

  state.settings = JSON.parse(origSettings);
  persistSettings();
  if (gpuProc) {
    try { gpuProc.kill(); } catch (e) {}
  }

  const wizardTest = await wc.executeJavaScript(`(async () => {
    const sleep = (ms) => new Promise(r => setTimeout(r, ms));
    document.getElementById('dashNewBtn').click();
    await sleep(150);
    const choiceOpened = !document.getElementById('confirmModal').classList.contains('hidden');
    document.getElementById('confirmOkBtn').click();
    await sleep(150);
    const opened = !document.getElementById('wizardModal').classList.contains('hidden');
    const stepCount = document.querySelectorAll('#wizardSteps .wizard-step-label').length;
    const installOptions = document.getElementById('wizInstall').options.length;
    const installVal = document.getElementById('wizInstall').value;
    document.getElementById('wizardNextBtn').click();
    await sleep(100);
    const onModel = !document.querySelector('#wizardModal .wizard-step[data-step="model"]').classList.contains('hidden');
    const modelOptions = document.getElementById('wizModel').options.length;
    const modelSel = document.getElementById('wizModel');
    modelSel.value = ${JSON.stringify(ggufFile)};
    modelSel.dispatchEvent(new Event('change'));
    await sleep(700);
    const modelInfoText = document.getElementById('wizModelInfo').textContent;
    const ctxPresets = document.querySelectorAll('#wizCtxPresets .preset-btn').length;
    const ctxDefault = document.getElementById('wizCtx').value;
    const ctxBtn = [...document.querySelectorAll('#wizCtxPresets .preset-btn')].find((b) => b.textContent === '16K');
    if (ctxBtn) { ctxBtn.click(); await sleep(50); }
    const ctxAfter = document.getElementById('wizCtx').value;
    document.getElementById('wizardNextBtn').click();
    await sleep(100);
    const onVision = !document.querySelector('#wizardModal .wizard-step[data-step="vision"]').classList.contains('hidden');
    document.getElementById('wizardNextBtn').click();
    await sleep(100);
    const onMtp = !document.querySelector('#wizardModal .wizard-step[data-step="mtp"]').classList.contains('hidden');
    document.getElementById('wizardNextBtn').click();
    await sleep(100);
    const vramOptions = document.querySelectorAll('#wizVramOptions .vram-option').length;
    const vramInfo = document.getElementById('wizVramInfo').textContent;
    const qualityOpt = [...document.querySelectorAll('#wizVramOptions .vram-option')].find((o) => o.textContent.includes('Máxima calidad'));
    qualityOpt.click();
    await sleep(100);
    document.getElementById('wizardNextBtn').click();
    await sleep(100);
    const onNet = !document.querySelector('#wizardModal .wizard-step[data-step="net"]').classList.contains('hidden');
    const hostVal = document.getElementById('wizHost').value;
    const portVal = document.getElementById('wizPort').value;
    document.getElementById('wizPort').value = 9090;
    document.getElementById('wizPort').dispatchEvent(new Event('input'));
    document.getElementById('wizardNextBtn').click();
    await sleep(150);
    const onDone = !document.querySelector('#wizardModal .wizard-step[data-step="done"]').classList.contains('hidden');
    const summary = document.getElementById('wizSummary').textContent;
    const cmd = document.querySelector('.summary-cmd').textContent;
    const finalBtnsVisible = !document.getElementById('wizardFinalBtns').classList.contains('hidden');
    const nameInput = document.getElementById('wizName');
    nameInput.value = 'Smoke Wizard';
    nameInput.dispatchEvent(new Event('input'));
    document.getElementById('wizardFinishBtn').click();
    await sleep(300);
    const closed = document.getElementById('wizardModal').classList.contains('hidden');
    const profileCount = document.querySelectorAll('#profileList .profile-item').length;
    const inDashboard = !document.getElementById('dashboardView').classList.contains('hidden');
    const lastItemText = Array.from(document.querySelectorAll('#profileList .profile-item')).pop().textContent;
    return { choiceOpened, opened, stepCount, installOptions, installVal, onModel, modelOptions, modelInfoText, ctxPresets, ctxDefault, ctxAfter, onVision, onMtp, vramOptions, vramInfo, onNet, hostVal, portVal, onDone, summary, cmd, finalBtnsVisible, closed, profileCount, inDashboard, lastItemText };
  })()`);

  const ggufTest = await wc.executeJavaScript(`(async () => {
    document.querySelector('#profileList .profile-item').click();
    await new Promise(r => setTimeout(r, 300));
    const sel = document.getElementById('modelSelect');
    sel.value = ${JSON.stringify(ggufFile)};
    sel.dispatchEvent(new Event('change'));
    await new Promise(r => setTimeout(r, 900));
    const hint = document.querySelector('#panel-rendering .card[data-field="gpuLayers"] .field-hint');
    const card = hint ? hint.closest('.card') : null;
    const wrap = card ? card.querySelector('.field[data-field-wrap="gpuLayers"]') : null;
    const range = wrap ? wrap.querySelector('input[type="range"]') : null;
    const before = hint ? hint.textContent : '';
    if (range) {
      range.value = 15;
      range.dispatchEvent(new Event('input'));
      await new Promise(r => setTimeout(r, 100));
    }
    const after = hint ? hint.textContent : '';
    const ctxCard = [...document.querySelectorAll('#panel-rendering .card')].find((c) => c.dataset.field === 'ctxSize');
    const ctxHint = ctxCard ? ctxCard.querySelector('.field-hint') : null;
    const ctxBefore = ctxHint ? ctxHint.textContent : '';
    const ctxInput = ctxCard ? ctxCard.querySelector('input[type="number"]') : null;
    if (ctxInput) {
      ctxInput.value = 65536;
      ctxInput.dispatchEvent(new Event('input'));
      await new Promise(r => setTimeout(r, 100));
    }
    const ctxAfter = ctxHint ? ctxHint.textContent : '';
    const layersAfterCtx = hint ? hint.textContent : '';
    if (range) {
      range.value = 10;
      range.dispatchEvent(new Event('input'));
      await new Promise(r => setTimeout(r, 100));
    }
    const ctxAfterLayers = ctxHint ? ctxHint.textContent : '';
    const kvPresetBtn = ctxCard
      ? [...ctxCard.querySelectorAll('.kv-preset-row .preset-btn')].find((b) => b.textContent === 'Mínima VRAM')
      : null;
    if (kvPresetBtn) {
      kvPresetBtn.click();
      await new Promise(r => setTimeout(r, 150));
    }
    const freshCtx = [...document.querySelectorAll('#panel-rendering .card')].find((c) => c.dataset.field === 'ctxSize');
    const kSel = freshCtx ? freshCtx.querySelector('select[data-key="cacheTypeK"]') : null;
    const vSel = freshCtx ? freshCtx.querySelector('select[data-key="cacheTypeV"]') : null;
    const kvDescText = freshCtx && freshCtx.querySelector('.preset-desc') ? freshCtx.querySelector('.preset-desc').textContent : '';
    return {
      hintExists: !!hint,
      hintVisible: hint ? hint.style.display !== 'none' : false,
      hasNumberInput: wrap ? !!wrap.querySelector('input[type="number"]') : null,
      sliderMax: range ? range.max : null,
      liveUpdate: !!before && !!after && before !== after && after.includes('15'),
      ctxHintExists: !!ctxHint,
      ctxHintVisible: ctxHint ? ctxHint.style.display !== 'none' : false,
      ctxHintLiveUpdate: !!ctxBefore && !!ctxAfter && ctxBefore !== ctxAfter && ctxAfter.includes('KV cache'),
      layersHintSeesCtx: layersAfterCtx.includes('KV cache'),
      ctxHintSeesLayers: !!ctxAfterLayers && ctxAfterLayers.includes('capas en GPU'),
      kvPresetApplies: !!kSel && kSel.value === 'q4_0' && !!vSel && vSel.value === 'q4_0' && kvDescText.includes('72%'),
      kvPresetHighlighted: freshCtx ? !!freshCtx.querySelector('.kv-preset-row .preset-btn.active') : false,
    };
  })()`);

  await wc.executeJavaScript(`openHf()`);
  await new Promise((r) => setTimeout(r, 300));
  const hfRect = await wc.executeJavaScript(`(() => {
    const r = document.getElementById('hfQuery').getBoundingClientRect();
    const el = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
    return { x: r.left + r.width / 2, y: r.top + r.height / 2, topEl: el ? (el.id || el.tagName) : 'null' };
  })()`);
  wc.sendInputEvent({ type: 'mouseDown', x: hfRect.x, y: hfRect.y });
  wc.sendInputEvent({ type: 'mouseUp', x: hfRect.x, y: hfRect.y });
  await new Promise((r) => setTimeout(r, 200));
  wc.sendInputEvent({ type: 'keyDown', keyCode: 'Q' });
  wc.sendInputEvent({ type: 'char', keyCode: 'Q' });
  wc.sendInputEvent({ type: 'keyUp', keyCode: 'Q' });
  await new Promise((r) => setTimeout(r, 200));
  const hfTyped = await wc.executeJavaScript(`(() => {
    const el = document.elementFromPoint(${hfRect.x}, ${hfRect.y});
    return {
      active: document.activeElement && document.activeElement.id,
      topEl: el ? (el.id || el.tagName) : 'null',
      typed: document.getElementById('hfQuery').value,
    };
  })()`);
  await wc.executeJavaScript(`closeHf()`);
  console.log('SMOKE_HFTYPING ' + JSON.stringify(hfTyped));

  console.log('SMOKE_RESULT ' + JSON.stringify(results));
  console.log('SMOKE_SPAWN ' + JSON.stringify({ spawnTest, status, stdoutForwarded: log.includes('LLAMA_SMOKE_OK') }));
  console.log('SMOKE_INSTALL ' + JSON.stringify({ verified, installTest }));
  console.log('SMOKE_NAV ' + JSON.stringify(navTest));
  console.log('SMOKE_SEL ' + JSON.stringify(dashSelTest));
  console.log('SMOKE_STATS ' + JSON.stringify(statsTest));
  console.log('SMOKE_MTP ' + JSON.stringify(mtpTest));
  console.log('SMOKE_WIZARD ' + JSON.stringify(wizardTest));
  console.log('SMOKE_GGUF ' + JSON.stringify({ main: ggufMain, renderer: ggufTest }));

  const delFolder = path.join(fakeFolder, 'del-test');  const delExe = path.join(delFolder, 'llama-server.exe');
  fs.mkdirSync(delFolder, { recursive: true });
  fs.copyFileSync(process.env.ComSpec, delExe);
  const delInstall = { id: 'del-test', name: 'Del Test', path: delFolder, exePath: delExe };
  state.settings.installations.push(delInstall);
  const delOk = await wc.executeJavaScript(`window.api.deleteInstall('del-test')`);
  const delFolderGone = !fs.existsSync(delFolder);
  state.settings.installations.push({ id: 'del-pro', name: 'Protegida', path: process.env.WINDIR, exePath: '' });
  const delPro = await wc.executeJavaScript(`window.api.deleteInstall('del-pro')`);
  console.log('SMOKE_DELETE ' + JSON.stringify({ delOk, delFolderGone, delPro }));
  state.settings.installations = state.settings.installations.filter((x) => x.id !== 'del-test' && x.id !== 'del-pro');

  const modFolder = path.join(fakeFolder, 'mod-test');
  fs.mkdirSync(modFolder, { recursive: true });
  fs.writeFileSync(path.join(modFolder, 'm.gguf'), 'fake');
  state.settings.modelsDir = fakeFolder;
  const modDel = await wc.executeJavaScript(`window.api.deleteModelsFolder(${JSON.stringify(modFolder)})`);
  const modGone = !fs.existsSync(modFolder);
  const modRoot = await wc.executeJavaScript(`window.api.deleteModelsFolder(${JSON.stringify(fakeFolder)})`);
  const rootKept = fs.existsSync(fakeFolder);
  console.log('SMOKE_MODFOLDER ' + JSON.stringify({ modDel, modGone, modRoot, rootKept }));
  state.settings = JSON.parse(origSettings);
  persistSettings();

  const problemTest = await wc.executeJavaScript(`(async () => {
    const origLen = profiles.length;
    const origSettingsLen = settings.installations.length;
    settings.installations.push({ id: 'gone', name: 'Version Borrada', path: 'Z:\\\\no\\\\ver', exePath: 'Z:\\\\no\\\\ver\\\\llama-server.exe' });
    profiles.push(Object.assign(defaultProfile(), {
      id: 'pb-test', name: 'Con Problemas', installId: 'gone',
      modelPath: 'Z:\\\\no\\\\existe.gguf', mmprojPath: 'Z:\\\\no\\\\vision.gguf', mtp: true, specDraftModel: 'Z:\\\\no\\\\mtp.gguf',
    }));
    await refreshFileStates();
    const card = [...document.querySelectorAll('#dashGrid .dash-card')].find((c) => c.querySelector('.dash-card-name').textContent === 'Con Problemas');
    const probs = profileProblems(profiles[profiles.length - 1]);
    const res = {
      card: !!card,
      hasAlert: card ? !!card.querySelector('.dash-badge.alert') : false,
      isProblem: card ? card.classList.contains('problem') : false,
      footWarn: card ? !!card.querySelector('.dash-open.warn-text') : false,
      nProblems: probs.length,
      firstProblem: probs[0] || '',
    };
    profiles.length = origLen;
    settings.installations.length = origSettingsLen;
    await refreshFileStates();
    return res;
  })()`);
  console.log('SMOKE_PROBLEMS ' + JSON.stringify(problemTest));

  console.log('SMOKE_ZIP ' + JSON.stringify({ files: zipTest.files, exeFound, txtOk }));
  } catch (e) {
  console.log('SMOKE_ERROR ' + (e && e.message ? e.message : String(e)));
  } finally {
  await new Promise((r) => setTimeout(r, 500));
  state.profiles = JSON.parse(origProfiles);
  persistProfiles();
  state.settings = JSON.parse(origSettings);
  persistSettings();
  if (gpuProc) {
    try { gpuProc.kill(); } catch (e) {}
  }
  try { fs.rmSync(fakeFolder, { recursive: true, force: true }); } catch (e) {}
  app.exit(0);
  }
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 840,
    minWidth: 980,
    minHeight: 640,
    frame: false,
    backgroundColor: '#0e1116',
    title: 'Llama Launcher',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      spellcheck: false,
    },
  });
  mainWindow.setMenuBarVisibility(false);
  mainWindow.on('maximize', () => send('window:maximized', true));
  mainWindow.on('unmaximize', () => send('window:maximized', false));
  mainWindow.loadFile(path.join(__dirname, 'renderer', 'index.html'));
  if (process.argv.includes('--smoke')) {
    mainWindow.webContents.on('console-message', (event) => {
      console.log('[RENDERER][' + event.level + '] ' + event.message);
    });
    mainWindow.webContents.once('did-finish-load', () => runSmoke(mainWindow));
  }

  mainWindow.on('close', (e) => {
    if (closeConfirmed) return;
    if (!dlMgr) return;
    const active = dlMgr.list().filter((d) => d.status === 'active' || d.status === 'pending');
    if (active.length === 0) return;
    if (closePending) { e.preventDefault(); return; }
    e.preventDefault();
    closePending = true;
    send('app:confirmClose', { count: active.length });
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

function registerIpc() {
  ipcMain.handle('dialog:selectFile', async (_e, opts) => {
    const res = await dialog.showOpenDialog(mainWindow, {
      title: opts.title || 'Seleccionar archivo',
      filters: opts.filters || [],
      properties: ['openFile'],
    });
    if (res.canceled || !res.filePaths.length) return null;
    return res.filePaths[0];
  });

  ipcMain.handle('dialog:selectDirectory', async (_e, opts) => {
    const res = await dialog.showOpenDialog(mainWindow, {
      title: opts.title || 'Seleccionar carpeta',
      properties: ['openDirectory'],
    });
    if (res.canceled || !res.filePaths.length) return null;
    return res.filePaths[0];
  });

  ipcMain.handle('models:scan', (_e, dir) => scanDir(dir, 0, []));

  ipcMain.handle('folder:serverExe', (_e, folder) => (folder ? findServerExeIn(folder) : null));

  ipcMain.handle('fs:exists', (_e, paths) => {
    const out = {};
    if (Array.isArray(paths)) {
      for (const p of paths) {
        if (typeof p !== 'string' || !p) continue;
        try {
          out[p] = fs.existsSync(p);
        } catch (e) {
          out[p] = false;
        }
      }
    }
    return out;
  });

  ipcMain.handle('shell:openExternal', (_e, url) => {
    if (typeof url === 'string' && url.match(/^https?:\/\//)) {
      shell.openExternal(url);
      return true;
    }
    return false;
  });


  ipcMain.handle('shell:openPath', async (_e, dir) => {
    if (typeof dir === 'string' && dir && fs.existsSync(dir)) {
      await shell.openPath(dir);
      return { ok: true };
    }
    return { ok: false, error: 'La carpeta no existe' };
  });

  ipcMain.handle('model:inspect', (_e, filePath) => {
    const info = typeof filePath === 'string' && filePath ? inspectGgufModel(filePath) : null;
    if (!info) return { ok: false };
    let layersFit = null;
    const gpuTotalBytes = gpuTotal || null;
    if (gpuTotal && info.layerBytes > 0) {
      layersFit = Math.max(0, Math.floor(gpuTotal / info.layerBytes));
    }
    return { ok: true, ...info, layersFit, gpuTotalBytes };
  });

  ipcMain.handle('profiles:load', () => state.profiles);

  ipcMain.handle('profiles:save', (_e, profiles) => {
    state.profiles = profiles;
    persistProfiles();
    return true;
  });

  ipcMain.handle('settings:load', () => {
    const s = state.settings;
    if (!s.modelsDir) s.modelsDir = path.join(app.getPath('userData'), 'models');
    if (!s.hfRepoMap || typeof s.hfRepoMap !== 'object') s.hfRepoMap = {};
    if (!Array.isArray(s.installations)) s.installations = [];
    if (s.serverPath && s.installations.length === 0) {
      const exe = s.serverPath;
      s.installations.push({
        id: uid(),
        name: 'llama.cpp',
        path: path.dirname(exe),
        exePath: exe,
      });
      delete s.serverPath;
    }
    if (s.installations.length === 0) {
      const exe = findServer();
      if (exe) {
        s.installations.push({ id: uid(), name: 'llama.cpp', path: path.dirname(exe), exePath: exe });
      }
    }
    return s;
  });

  ipcMain.handle('settings:findServer', () => findServer());

  ipcMain.handle('settings:save', (_e, settings) => {
    state.settings = settings;
    persistSettings();
    return true;
  });

  ipcMain.handle('server:start', (_e, cmd) => {
    if (serverProc) {
      return { ok: false, error: 'Ya hay un servidor en ejecución. Detenelo primero.' };
    }
    const exe = cmd.exe || '';
    if (!exe) {
      return { ok: false, error: 'No se definió la ruta de llama-server.exe.' };
    }
    if (!fs.existsSync(exe)) {
      return { ok: false, error: 'No se encontró el ejecutable: ' + exe };
    }
    const cwd = path.dirname(exe);
    try {
      serverErrBuf = '';
      serverProc = spawn(exe, cmd.args || [], { cwd, windowsHide: false });
    } catch (err) {
      return { ok: false, error: err.message };
    }
    serverProc.stdout.on('data', (d) => send('server:log', { stream: 'out', text: d.toString() }));
    serverProc.stderr.on('data', (d) => {
      const text = d.toString();
      serverErrBuf = (serverErrBuf + text).slice(-131072);
      send('server:log', { stream: 'err', text });
    });
    serverProc.on('error', (err) => {
      send('server:log', { stream: 'err', text: 'Error al iniciar el proceso: ' + err.message + '\n' });
    });
    serverProc.on('close', (code) => {
      const argError = code === 0 ? null : parseArgError(serverErrBuf);
      send('server:exit', { code, argError });
      serverProc = null;
    });
    send('server:state', { running: true, pid: serverProc.pid });
    return { ok: true, pid: serverProc.pid };
  });

  ipcMain.handle('server:stop', () => {
    if (!serverProc) return { ok: false, error: 'No hay servidor en ejecución.' };
    const pid = serverProc.pid;
    try {
      exec(`taskkill /PID ${pid} /T /F`, { windowsHide: true }, () => {});
    } catch (e) {}
    serverProc = null;
    send('server:state', { running: false });
    return { ok: true };
  });

  ipcMain.handle('server:status', () => ({
    running: !!serverProc,
    pid: serverProc ? serverProc.pid : null,
  }));

  ipcMain.handle('llama:listReleases', async () => {
    try {
      return { ok: true, releases: await listLlamaReleases() };
    } catch (e) {
      return { ok: false, error: e && e.message ? e.message : 'No se pudieron listar las versiones.' };
    }
  });

  ipcMain.handle('llama:start', (_e, opts) => runLlamaDownload(opts));

  ipcMain.handle('llama:cancel', () => {
    if (dlState && dlState.active) dlState.cancelled = true;
    return true;
  });

  ipcMain.handle('llama:downloadDir', () => llamaDownloadDir());

  ipcMain.handle('llama:delete', (_e, id) => {
    const installs = state.settings.installations || [];
    const install = installs.find((x) => x.id === id);
    if (!install) return { ok: false, error: 'No se encontró la versión registrada.' };
    try {
      const result = deleteInstallDir({ ...install, modelsDir: state.settings.modelsDir });
      return { ok: true, result };
    } catch (e) {
      return { ok: false, error: e && e.message ? e.message : 'No se pudo eliminar la carpeta.' };
    }
  });

  ipcMain.handle('models:deleteFolder', (_e, dir) => {
    try {
      return { ok: true, result: deleteModelsFolder(dir) };
    } catch (e) {
      return { ok: false, error: e && e.message ? e.message : 'No se pudo eliminar la carpeta.' };
    }
  });

  ipcMain.handle('hf:search', async (_e, opts) => {
    try {
      return { ok: true, models: await searchHfModels(opts) };
    } catch (e) {
      return { ok: false, error: e && e.message ? e.message : 'No se pudieron buscar modelos.' };
    }
  });

  ipcMain.handle('hf:files', async (_e, repo) => {
    try {
      const files = await listHfRepoFiles(repo);
      return { ok: true, repo, files, classified: classifyHfFiles(files) };
    } catch (e) {
      return { ok: false, error: e && e.message ? e.message : 'No se pudieron listar los archivos del modelo.' };
    }
  });

  ipcMain.handle('hf:start', (_e, opts) => runModelDownload(opts));

  ipcMain.handle('hf:cancel', () => {
    if (mdlState && mdlState.active) {
      mdlState.cancelled = true;
      if (mdlState.dlIds) {
        for (const id of mdlState.dlIds) {
          const rec = dlMgr.get(id);
          if (rec && rec.status !== 'completed') {
            dlMgr.cancel(id);
          }
        }
      }
    }
    return true;
  });

  ipcMain.handle('hf:modelConfig', async (_e, repo) => {
    try {
      return { ok: true, info: await hfModelConfig(repo) };
    } catch (e) {
      return { ok: false, error: (e && e.message) || 'No se pudo consultar el modelo en Hugging Face.' };
    }
  });


  ipcMain.handle('dl:start', (_e, opts) => {
    if (!dlMgr) return { ok: false, error: 'Download manager not initialized.' };
    return dlMgr.start(opts);
  });

  ipcMain.handle('dl:pause', (_e, id) => {
    if (!dlMgr) return { ok: false, error: 'Download manager not initialized.' };
    return dlMgr.pause(id);
  });

  ipcMain.handle('dl:resume', (_e, id) => {
    if (!dlMgr) return { ok: false, error: 'Download manager not initialized.' };
    return dlMgr.resume(id);
  });

  ipcMain.handle('dl:cancel', (_e, id) => {
    if (!dlMgr) return { ok: false, error: 'Download manager not initialized.' };
    return dlMgr.cancel(id);
  });

  ipcMain.handle('dl:remove', (_e, id) => {
    if (!dlMgr) return { ok: false, error: 'Download manager not initialized.' };
    return dlMgr.remove(id);
  });

  ipcMain.handle('dl:retry', (_e, id) => {
    if (!dlMgr) return { ok: false, error: 'Download manager not initialized.' };
    console.log('[DL] retry called, id=' + id + ', state has id=' + (dlMgr.get(id) ? 'yes status=' + dlMgr.get(id).status : 'NO'));
    return dlMgr.retry(id);
  });

  ipcMain.handle('dl:list', () => {
    if (!dlMgr) return [];
    return dlMgr.list();
  });

  ipcMain.handle('dl:listAll', () => {
    if (!dlMgr) return [];
    return dlMgr.listAll();
  });

  ipcMain.handle('dl:clearCompleted', () => {
    if (!dlMgr) return 0;
    return dlMgr.clearCompleted();
  });

  ipcMain.on('app:closeResponse', (_e, confirmed) => {
    closePending = false;
    if (!confirmed) return;
    closeConfirmed = true;
    if (dlMgr) {
      const items = dlMgr.list();
      for (const d of items) {
        if (d.status === 'active' || d.status === 'pending') {
          dlMgr.pause(d.id);
        }
      }
    }
    if (mainWindow && !mainWindow.isDestroyed()) mainWindow.close();
  });

  ipcMain.on('window:minimize', () => {
    if (mainWindow && !mainWindow.isDestroyed()) mainWindow.minimize();
  });
  ipcMain.on('window:maximizeToggle', () => {
    if (!mainWindow || mainWindow.isDestroyed()) return;
    if (mainWindow.isMaximized()) mainWindow.unmaximize();
    else mainWindow.maximize();
  });
  ipcMain.on('window:close', () => {
    if (mainWindow && !mainWindow.isDestroyed()) mainWindow.close();
  });
}

app.whenReady().then(() => {
  ensureStore();
  initDownloadManager();
  registerIpc();
  createWindow();
  startStatsMonitor();
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('quit', () => {
  if (dlMgr) {
    try { dlMgr.shutdown(); } catch (e) {}
  }
  if (serverProc) {
    try {
      exec(`taskkill /PID ${serverProc.pid} /T /F`, { windowsHide: true }, () => {});
    } catch (e) {}
  }
  if (gpuProc) {
    try {
      gpuProc.kill();
    } catch (e) {}
  }
});
