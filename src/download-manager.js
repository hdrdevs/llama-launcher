const { app } = require('electron');
const fs = require('fs');
const https = require('https');
const http = require('http');
const path = require('path');

const STATE_FILE = 'downloads.json';
const SAVE_INTERVAL_MS = 2000;

function uid() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
}

function getStatePath(dataDir) {
  return path.join(dataDir, STATE_FILE);
}

class DownloadManager {
  constructor(opts) {
    this.dataDir = opts.dataDir || path.join(app.getPath('userData'), 'llama-launcher');
    this.onProgress = opts.onProgress || (() => {});
    this.onComplete = opts.onComplete || (() => {});
    this.onError = opts.onError || (() => {});
    this.onPause = opts.onPause || (() => {});
    this.onResume = opts.onResume || (() => {});
    this.onCancelled = opts.onCancelled || (() => {});

    this._active = new Map();
    this._state = {};
    this._saveTimer = null;
    this._load();
  }

  _load() {
    try {
      const raw = fs.readFileSync(getStatePath(this.dataDir), 'utf8');
      const arr = JSON.parse(raw);
      if (Array.isArray(arr)) {
        for (const s of arr) {
          if (s && s.id) {
            if (s.status === 'active') s.status = 'error';
            this._state[s.id] = s;
          }
        }
      }
    } catch (e) {}
  }

  _persist() {
    try {
      fs.mkdirSync(this.dataDir, { recursive: true });
    } catch (e) {}
    const arr = Object.values(this._state);
    try {
      fs.writeFileSync(getStatePath(this.dataDir), JSON.stringify(arr, null, 2));
    } catch (e) {}
  }

  _scheduleSave() {
    if (this._saveTimer) return;
    this._saveTimer = setTimeout(() => {
      this._saveTimer = null;
      this._persist();
    }, SAVE_INTERVAL_MS);
  }

  list() {
    return Object.values(this._state).filter((s) => s.status !== 'completed');
  }

  listAll() {
    return Object.values(this._state);
  }

  get(id) {
    return this._state[id] || null;
  }

  register(opts) {
    const { id, url, dest, headers, meta } = opts;
    const dlId = id || uid();
    if (this._state[dlId]) return { ok: true, id: dlId };
    const record = {
      id: dlId,
      url,
      dest,
      totalBytes: 0,
      receivedBytes: 0,
      status: 'pending',
      etag: null,
      lastModified: null,
      createdAt: Date.now(),
      updatedAt: Date.now(),
      error: null,
      headers: headers || {},
      meta: meta || {},
    };
    this._state[dlId] = record;
    this._persist();
    this.onProgress(record, 0);
    return { ok: true, id: dlId };
  }

  start(opts) {
    const { id, url, dest, headers, meta, onProgress, onComplete, onError } = opts;
    const dlId = id || uid();

    if (this._active.has(dlId)) {
      return { ok: false, error: 'Download already active.' };
    }

    const existing = this._state[dlId];

    if (existing && existing.status === 'pending') {
      existing.status = 'active';
      existing.updatedAt = Date.now();
      if (onProgress) existing._cbProgress = onProgress;
      if (onComplete) existing._cbComplete = onComplete;
      if (onError) existing._cbError = onError;
      this._persist();
      this._beginDownload(existing);
      return { ok: true, id: dlId };
    }

    let receivedBytes = 0;

    if (existing && existing.status === 'paused' && existing.dest === dest) {
      try {
        const stat = fs.statSync(dest);
        receivedBytes = stat.size;
      } catch (e) {
        receivedBytes = 0;
      }
    }

    const record = {
      id: dlId,
      url,
      dest,
      totalBytes: (existing && existing.totalBytes) || 0,
      receivedBytes,
      status: 'active',
      etag: (existing && existing.etag) || null,
      lastModified: (existing && existing.lastModified) || null,
      createdAt: (existing && existing.createdAt) || Date.now(),
      updatedAt: Date.now(),
      error: null,
      headers: headers || {},
      meta: meta || (existing && existing.meta) || {},
      _cbProgress: onProgress || null,
      _cbComplete: onComplete || null,
      _cbError: onError || null,
    };
    this._state[dlId] = record;
    this._persist();

    this._beginDownload(record);
    return { ok: true, id: dlId };
  }

  _beginDownload(record) {
    const ctrl = { request: null, destroyed: false };
    this._active.set(record.id, ctrl);

    const doRequest = (resumeOffset) => {
      const parsed = new URL(record.url);
      const isHttps = parsed.protocol === 'https:';
      const transport = isHttps ? https : http;

      const reqHeaders = Object.assign({}, record.headers || {});
      if (resumeOffset > 0) {
        reqHeaders['Range'] = 'bytes=' + resumeOffset + '-';
      }

      const req = transport.get(
        record.url,
        { headers: reqHeaders },
        (res) => {
          const code = res.statusCode || 0;

          if (code >= 300 && code < 400 && res.headers.location) {
            res.resume();
            record.url = new URL(res.headers.location, record.url).href;
            this._scheduleSave();
            doRequest(resumeOffset);
            return;
          }

          const acceptRange = res.headers['accept-ranges'];
          const contentLength = Number(res.headers['content-length']) || 0;

          if (resumeOffset > 0 && code === 200) {
            resumeOffset = 0;
            record.receivedBytes = 0;
            try {
              fs.truncateSync(record.dest, 0);
            } catch (e) {}
          } else if (resumeOffset > 0 && code === 416) {
            res.resume();
            record.status = 'completed';
            record.updatedAt = Date.now();
            this._active.delete(record.id);
            this._persist();
            this.onComplete(record);
            if (record._cbComplete) record._cbComplete(record);
            return;
          } else if (code !== 200 && code !== 206) {
            res.resume();
            record.status = 'error';
            record.error = 'HTTP ' + code;
            record.updatedAt = Date.now();
            this._active.delete(record.id);
            this._persist();
            this.onError(record);
            if (record._cbError) record._cbError(record);
            return;
          }

          if (code === 206 || (resumeOffset === 0 && code === 200)) {
            const newTotal = resumeOffset + contentLength;
            if (newTotal > 0) record.totalBytes = newTotal;
          } else if (contentLength > 0) {
            record.totalBytes = contentLength;
          }

          if (res.headers.etag) record.etag = res.headers.etag;
          if (res.headers['last-modified']) record.lastModified = res.headers['last-modified'];

          const flags = resumeOffset > 0 ? 'r+' : 'w';
          let fd;
          try {
            fd = fs.openSync(record.dest, flags);
          } catch (e) {
            res.resume();
            record.status = 'error';
            record.error = 'Cannot open file: ' + e.message;
            record.updatedAt = Date.now();
            this._active.delete(record.id);
            this._persist();
            this.onError(record);
            if (record._cbError) record._cbError(record);
            return;
          }

          let writeOffset = resumeOffset;
          let lastReport = Date.now();
          let lastBytes = resumeOffset;

          res.on('data', (chunk) => {
            if (ctrl.destroyed) {
              res.destroy();
              try { fs.closeSync(fd); } catch (e) {}
              return;
            }
            try {
              fs.writeSync(fd, chunk, 0, chunk.length, writeOffset);
              writeOffset += chunk.length;
              record.receivedBytes = writeOffset;
              record.updatedAt = Date.now();

              const now = Date.now();
              if (now - lastReport >= 300 || writeOffset >= record.totalBytes) {
                lastReport = now;
                const speed = writeOffset - lastBytes;
                lastBytes = writeOffset;
                this.onProgress(record, speed);
                if (record._cbProgress) record._cbProgress(record, speed);
              }
            } catch (e) {
              ctrl.destroyed = true;
              res.destroy();
              try { fs.closeSync(fd); } catch (err) {}
              record.status = 'error';
              record.error = 'Write error: ' + e.message;
              record.updatedAt = Date.now();
              this._active.delete(record.id);
              this._persist();
              this.onError(record);
            if (record._cbError) record._cbError(record);
            }
          });

          res.on('error', (err) => {
            if (ctrl.destroyed) return;
            try { fs.closeSync(fd); } catch (e) {}
            record.status = 'error';
            record.error = err.message || 'Network error';
            record.updatedAt = Date.now();
            this._active.delete(record.id);
            this._persist();
            this.onError(record);
            if (record._cbError) record._cbError(record);
          });

          res.on('end', () => {
            try { fs.closeSync(fd); } catch (e) {}
            if (ctrl.destroyed) return;
            record.status = 'completed';
            record.receivedBytes = writeOffset;
            record.updatedAt = Date.now();
            this._active.delete(record.id);
            this._persist();
            this.onComplete(record);
            if (record._cbComplete) record._cbComplete(record);
          });
        }
      );

      req.on('error', (err) => {
        if (ctrl.destroyed) return;
        record.status = 'error';
        record.error = err.message || 'Connection failed';
        record.updatedAt = Date.now();
        this._active.delete(record.id);
        this._persist();
        this.onError(record);
        if (record._cbError) record._cbError(record);
      });

      ctrl.request = req;
    };

    doRequest(record.receivedBytes || 0);
  }

  pause(id) {
    const record = this._state[id];
    if (!record || record.status !== 'active') {
      return { ok: false, error: 'Download is not active.' };
    }
    const ctrl = this._active.get(id);
    if (ctrl) {
      ctrl.destroyed = true;
      if (ctrl.request) {
        try { ctrl.request.destroy(); } catch (e) {}
      }
      this._active.delete(id);
    }

    if (record.dest) {
      try {
        const stat = fs.statSync(record.dest);
        record.receivedBytes = stat.size;
      } catch (e) {}
    }

    record.status = 'paused';
    record.updatedAt = Date.now();
    this._persist();
    this.onPause(record);
    return { ok: true };
  }

  resume(id) {
    const record = this._state[id];
    if (!record || record.status !== 'paused') {
      return { ok: false, error: 'Download is not paused.' };
    }

    if (record.dest) {
      try {
        const stat = fs.statSync(record.dest);
        record.receivedBytes = stat.size;
      } catch (e) {}
    }

    record.status = 'active';
    record.error = null;
    record.updatedAt = Date.now();
    this._persist();
    this._beginDownload(record);
    this.onResume(record);
    return { ok: true };
  }

  cancel(id) {
    const record = this._state[id];
    if (!record) return { ok: false, error: 'Download not found.' };

    const ctrl = this._active.get(id);
    if (ctrl) {
      ctrl.destroyed = true;
      if (ctrl.request) {
        try { ctrl.request.destroy(); } catch (e) {}
      }
      this._active.delete(id);
    }

    if (record.dest) {
      try { fs.rmSync(record.dest, { force: true }); } catch (e) {}
    }

    delete this._state[id];
    this._persist();
    this.onCancelled(record);
    return { ok: true };
  }

  remove(id) {
    const record = this._state[id];
    if (!record) return { ok: false, error: 'Download not found.' };
    if (record.status === 'active') return { ok: false, error: 'Cannot remove an active download.' };
    delete this._state[id];
    this._persist();
    this.onCancelled(record);
    return { ok: true };
  }

  retry(id) {
    const record = this._state[id];
    if (!record) return { ok: false, error: 'Download not found: ' + id };
    if (record.status !== 'error') return { ok: false, error: 'Can only retry failed downloads. Status: ' + record.status };

    if (record.dest) {
      try {
        const stat = fs.statSync(record.dest);
        record.receivedBytes = stat.size;
      } catch (e) {
        record.receivedBytes = 0;
      }
    }

    record.status = 'active';
    record.error = null;
    record.updatedAt = Date.now();
    this._persist();
    this._beginDownload(record);
    return { ok: true };
  }

  clearCompleted() {
    const ids = Object.keys(this._state).filter((id) => this._state[id].status === 'completed');
    for (const id of ids) delete this._state[id];
    this._persist();
    return ids.length;
  }

  shutdown() {
    for (const [id, ctrl] of this._active) {
      ctrl.destroyed = true;
      if (ctrl.request) {
        try { ctrl.request.destroy(); } catch (e) {}
      }
    }
    this._active.clear();
    if (this._saveTimer) {
      clearTimeout(this._saveTimer);
      this._saveTimer = null;
    }
    this._persist();
  }
}

module.exports = { DownloadManager };
