const $ = (sel) => document.querySelector(sel);

const FLAGS = {
  gpuLayers: '--gpu-layers',
  fit: '--fit',
  imageMinTokens: '--image-min-tokens',
  reasoning: '--reasoning',
  ctxSize: '--ctx-size',
  cacheTypeK: '--cache-type-k',
  cacheTypeV: '--cache-type-v',
  temp: '--temp',
  topP: '--top-p',
  topK: '--top-k',
  minP: '--min-p',
  presencePenalty: '--presence-penalty',
  repeatPenalty: '--repeat-penalty',
  parallel: '--parallel',
  host: '--host',
  port: '--port',
  alias: '--alias',
  cacheIdleSlots: '--cache-idle-slots',
  mtp: '--spec-type draft-mtp',
  specDraftModel: '--spec-draft-model',
  specDraftNMax: '--spec-draft-n-max',
  specDraftNMin: '--spec-draft-n-min',
  specDraftPSplit: '--spec-draft-p-split',
  flashAttn: '--flash-attn',
  threads: '--threads',
  threadsBatch: '--threads-batch',
  loadMode: '--load-mode',
  kvOffload: '--kv-offload',
  opOffload: '--op-offload',
  cpuMoe: '--cpu-moe',
  nCpuMoe: '--n-cpu-moe',
  splitMode: '--split-mode',
  mainGpu: '--main-gpu',
  tensorSplit: '--tensor-split',
  device: '--device',
  ropeScaling: '--rope-scaling',
  ropeScale: '--rope-scale',
  ropeFreqBase: '--rope-freq-base',
  ropeFreqScale: '--rope-freq-scale',
  yarnOrigCtx: '--yarn-orig-ctx',
  yarnExtFactor: '--yarn-ext-factor',
  yarnAttnFactor: '--yarn-attn-factor',
  yarnBetaSlow: '--yarn-beta-slow',
  yarnBetaFast: '--yarn-beta-fast',
  seed: '--seed',
  samplers: '--samplers',
  ignoreEos: '--ignore-eos',
  topNSigma: '--top-nsigma',
  typicalP: '--typical',
  xtcProbability: '--xtc-probability',
  xtcThreshold: '--xtc-threshold',
  frequencyPenalty: '--frequency-penalty',
  repeatLastN: '--repeat-last-n',
  dryMultiplier: '--dry-multiplier',
  dryBase: '--dry-base',
  dryAllowedLength: '--dry-allowed-length',
  dryPenaltyLastN: '--dry-penalty-last-n',
  dynatempRange: '--dynatemp-range',
  dynatempExp: '--dynatemp-exp',
  mirostat: '--mirostat',
  mirostatLr: '--mirostat-lr',
  mirostatEnt: '--mirostat-ent',
  adaptiveTarget: '--adaptive-target',
  adaptiveDecay: '--adaptive-decay',
  logitBias: '--logit-bias',
  grammar: '--grammar',
  jsonSchema: '--json-schema',
  apiKey: '--api-key',
  reversePrompt: '--reverse-prompt',
  timeout: '--timeout',
  threadsHttp: '--threads-http',
  apiPrefix: '--api-prefix',
  corsOrigins: '--cors-origins',
  reusePort: '--reuse-port',
  noHost: '--no-host',
  metrics: '--metrics',
  props: '--props',
  slots: '--slots',
  contBatching: '--cont-batching',
  cachePrompt: '--cache-prompt',
  cacheReuse: '--cache-reuse',
  embedding: '--embedding',
  pooling: '--pooling',
};

const SILENT_VALUES = {
  fit: 'off',
  gpuLayers: 0,
  ctxSize: 2048,
  cacheTypeK: 'f16',
  cacheTypeV: 'f16',
  temp: 0.8,
  topK: 40,
  minP: 0,
  presencePenalty: 0,
  repeatPenalty: 1,
  parallel: 1,
  seed: -1,
  topNSigma: 0,
  typicalP: 1,
  xtcProbability: 0,
  xtcThreshold: 0.1,
  frequencyPenalty: 0,
  repeatLastN: 64,
  dynatempRange: 0,
  dynatempExp: 1,
  mirostat: 0,
  mirostatLr: 0.1,
  mirostatEnt: 5,
  adaptiveTarget: 0,
  adaptiveDecay: 8,
  flashAttn: 'auto',
  splitMode: 'layer',
};

const KV_CACHE_DESCRIPTIONS = {
  f32: 'máxima precisión · +VRAM',
  bf16: 'alta precisión',
  f16: 'estándar (default)',
  q8_0: '8-bit · ~50% menos VRAM',
  q5_1: '5-bit · más calidad',
  q5_0: '5-bit · menos VRAM',
  q4_1: '4-bit · bajo VRAM',
  q4_0: '4-bit · mínimo VRAM',
  iq4_nl: '4-bit sin lista · mínimo VRAM',
};

const LOAD_MODE_DESCRIPTIONS = {
  '': 'No pasar (default: mmap)',
  mmap: 'memory-map (default de llama.cpp)',
  mlock: 'mantener el modelo en RAM',
  'mmap+mlock': 'mmap + RAM fija (mejor para latencia)',
  dio: 'DirectIO si está disponible',
  none: 'sin modo especial',
};

const ROPE_SCALING_DESCRIPTIONS = {
  '': 'Default (según lo que pida el modelo)',
  linear: 'escalado lineal de contexto',
  yarn: 'YaRN (usá junto a los parámetros yarn-*)',
  none: 'sin escalado',
};

const SPLIT_MODE_DESCRIPTIONS = {
  '': 'Default (layer)',
  layer: 'divide capas y KV entre GPUs (default)',
  row: 'divide pesos por filas (paralelo)',
  tensor: 'divide tensores',
  none: 'una sola GPU',
};

const POOLING_DESCRIPTIONS = {
  '': 'Default del modelo',
  none: 'sin pooling',
  mean: 'promedio de tokens',
  cls: 'token CLS',
  last: 'último token',
  rank: 'ranking',
};

const MIROSTAT_DESCRIPTIONS = {
  '0': 'Desactivado',
  '1': 'Mirostat',
  '2': 'Mirostat 2.0',
};

const KV_PRESETS = [
  { label: 'Mínima VRAM', values: { cacheTypeK: 'q4_0', cacheTypeV: 'q4_0' }, desc: 'Q4_0 en K y V: ~72% menos VRAM que f16. Menor precisión; ideal para modelos grandes o contexto muy largo.' },
  { label: 'Moderada VRAM', values: { cacheTypeK: 'q8_0', cacheTypeV: 'q8_0' }, desc: 'Q8_0 en K y V: ~47% menos VRAM que f16 con pérdida mínima de calidad. Buen equilibrio entre memoria y precisión.' },
  { label: 'Default', values: { cacheTypeK: 'f16', cacheTypeV: 'f16' }, desc: 'F16 en K y V: el estándar de llama.cpp, sin pérdida de calidad. Consume el doble de VRAM que la opción 8-bit.' },
  { label: 'Máxima VRAM', values: { cacheTypeK: 'f32', cacheTypeV: 'f32' }, desc: 'F32 en K y V: máxima precisión (4 bytes por elemento), el doble de VRAM que el default. Úsalo solo si te sobra VRAM.' },
];

const SAMPLING_FIELDS = [
  { key: 'temp', label: 'Temperature', type: 'slider', min: 0, max: 2, step: 0.05 },
  { key: 'seed', label: 'Semilla (seed)', type: 'number', min: -1, max: 2147483647, flag: true },
  { key: 'topK', label: 'Top-K', type: 'number', min: 0, max: 1000 },
  { key: 'topP', label: 'Top-P', type: 'slider', min: 0, max: 1, step: 0.01 },
  { key: 'minP', label: 'Min-P', type: 'slider', min: 0, max: 1, step: 0.01 },
  { key: 'topNSigma', label: 'Top-N-Sigma', type: 'slider', min: 0, max: 3, step: 0.05, flag: true },
  { key: 'typicalP', label: 'Typical-P', type: 'slider', min: 0.5, max: 1, step: 0.01, flag: true },
  { key: 'xtcProbability', label: 'XTC: probabilidad', type: 'slider', min: 0, max: 1, step: 0.01, flag: true },
  { key: 'xtcThreshold', label: 'XTC: umbral', type: 'slider', min: 0, max: 1, step: 0.01, flag: true },
  { key: 'presencePenalty', label: 'Presence penalty', type: 'slider', min: -2, max: 2, step: 0.05 },
  { key: 'frequencyPenalty', label: 'Frequency penalty', type: 'slider', min: -2, max: 2, step: 0.05, flag: true },
  { key: 'repeatPenalty', label: 'Repeat penalty', type: 'slider', min: 1, max: 2, step: 0.05 },
  { key: 'repeatLastN', label: 'Repeat: últimas N tokens', type: 'number', min: 0, max: 100000, flag: true },
  { key: 'dryMultiplier', label: 'DRY: multiplicador', type: 'slider', min: 0, max: 2, step: 0.05, flag: true },
  { key: 'dryBase', label: 'DRY: base', type: 'slider', min: 1, max: 3, step: 0.05, flag: true },
  { key: 'dryAllowedLength', label: 'DRY: longitud permitida', type: 'number', min: 1, max: 100, flag: true },
  { key: 'dryPenaltyLastN', label: 'DRY: últimas N tokens', type: 'number', min: -1, max: 100000, flag: true },
  { key: 'dynatempRange', label: 'Dynamic temperature: rango', type: 'slider', min: 0, max: 2, step: 0.05, flag: true },
  { key: 'dynatempExp', label: 'Dynamic temperature: exp', type: 'slider', min: 0, max: 2, step: 0.05, flag: true },
  { key: 'mirostat', label: 'Mirostat', type: 'select', options: ['0', '1', '2'], descriptions: MIROSTAT_DESCRIPTIONS, flag: true },
  { key: 'mirostatLr', label: 'Mirostat: tasa de aprendizaje', type: 'slider', min: 0.001, max: 1, step: 0.001, flag: true },
  { key: 'mirostatEnt', label: 'Mirostat: entropía objetivo', type: 'slider', min: 1, max: 10, step: 0.1, flag: true },
  { key: 'adaptiveTarget', label: 'Adaptive-P: target', type: 'slider', min: 0, max: 1, step: 0.01, flag: true },
  { key: 'adaptiveDecay', label: 'Adaptive-P: decay', type: 'number', min: 1, max: 100, flag: true },
  { key: 'ignoreEos', label: 'Ignorar EOS (no detenerse)', type: 'toggle', flag: true },
  { key: 'samplers', label: 'Orden de samplers', type: 'text', placeholder: 'penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature', flag: true },
  { key: 'logitBias', label: 'Logit bias (token:peso)', type: 'text', placeholder: '15043:1.5;15166:-0.5', flag: true },
  { key: 'grammar', label: 'Gramática GBNF', type: 'text', placeholder: 'root ::= ...', flag: true },
  { key: 'jsonSchema', label: 'JSON Schema', type: 'text', placeholder: '{"type":"object","properties":{...}}', flag: true },
];

const RENDERING_FIELDS = [
  { key: 'ctxSize', label: 'Contexto (ctx-size)', type: 'number', min: 64, max: 1000000, wide: true, hint: true, presets: [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288], group: 'ctx' },
  { key: 'cacheTypeK', label: 'KV cache K (cuantizar contexto)', type: 'select', options: ['f16', 'q8_0', 'q4_0', 'q4_1', 'q5_0', 'q5_1', 'iq4_nl', 'bf16', 'f32'], descriptions: KV_CACHE_DESCRIPTIONS, group: 'ctx' },
  { key: 'cacheTypeV', label: 'KV cache V (cuantizar contexto)', type: 'select', options: ['f16', 'q8_0', 'q4_0', 'q4_1', 'q5_0', 'q5_1', 'iq4_nl', 'bf16', 'f32'], descriptions: KV_CACHE_DESCRIPTIONS, group: 'ctx' },
  { key: 'gpuLayers', label: 'Capas en GPU (gpu-layers)', type: 'slider', min: 0, max: 999, step: 1, hint: true, noNumber: true, group: 'gpu' },
  { key: 'parallel', label: 'Slots paralelos', type: 'number', min: 1, max: 64, group: 'gpu' },
  { key: 'imageMinTokens', label: 'Tokens mínimos de imagen', type: 'number', min: 0, max: 100000, group: 'gpu' },
  { key: 'fit', label: 'Modo fit', type: 'select', options: ['off', 'on'], group: 'gpu' },
  { key: 'reasoning', label: 'Reasoning', type: 'select', options: ['off', 'on', 'auto'], group: 'gpu' },
  { key: 'flashAttn', label: 'Flash Attention', type: 'select', options: ['auto', 'on', 'off'], flag: true, group: 'gpu' },
  { key: 'threads', label: 'Hilos CPU (threads)', type: 'number', min: 1, max: 1024, flag: true, group: 'gpu' },
  { key: 'threadsBatch', label: 'Hilos batch (threads-batch)', type: 'number', min: 1, max: 1024, flag: true, group: 'gpu' },
  { key: 'loadMode', label: 'Modo de carga (load-mode)', type: 'select', options: ['', 'mmap', 'mlock', 'mmap+mlock', 'dio', 'none'], descriptions: LOAD_MODE_DESCRIPTIONS, flag: true, group: 'gpu' },
  { key: 'splitMode', label: 'División multi-GPU (split-mode)', type: 'select', options: ['', 'layer', 'row', 'tensor', 'none'], descriptions: SPLIT_MODE_DESCRIPTIONS, flag: true, group: 'multigpu' },
  { key: 'mainGpu', label: 'GPU principal (main-gpu)', type: 'number', min: 0, max: 16, flag: true, group: 'multigpu' },
  { key: 'tensorSplit', label: 'División de tensores (tensor-split)', type: 'text', placeholder: '0,1,0', flag: true, group: 'multigpu' },
  { key: 'device', label: 'Dispositivos (device)', type: 'text', placeholder: 'CUDA0,CUDA1', flag: true, group: 'multigpu' },
  { key: 'kvOffload', label: 'KV cache en GPU (kv-offload)', type: 'toggle', flag: true, group: 'multigpu' },
  { key: 'opOffload', label: 'Operaciones en GPU (op-offload)', type: 'toggle', flag: true, group: 'multigpu' },
  { key: 'cpuMoe', label: 'MoE en CPU (cpu-moe)', type: 'toggle', flag: true, group: 'multigpu' },
  { key: 'nCpuMoe', label: 'n-cpu-moe (capas MoE en CPU)', type: 'number', min: 0, max: 100000, flag: true, group: 'multigpu', showIfMoe: true },
  { key: 'ropeScaling', label: 'Escalado RoPE (rope-scaling)', type: 'select', options: ['', 'linear', 'yarn', 'none'], descriptions: ROPE_SCALING_DESCRIPTIONS, flag: true, group: 'longctx' },
  { key: 'ropeScale', label: 'Factor de escala (rope-scale)', type: 'number', min: 1, max: 100, flag: true, group: 'longctx' },
  { key: 'ropeFreqBase', label: 'Frecuencia base (rope-freq-base)', type: 'number', min: 1, max: 10000000, flag: true, group: 'longctx' },
  { key: 'ropeFreqScale', label: 'Escala de frecuencia (rope-freq-scale)', type: 'number', min: 0, max: 10, flag: true, group: 'longctx' },
  { key: 'yarnOrigCtx', label: 'YaRN: contexto original', type: 'number', min: 0, max: 1000000, flag: true, group: 'longctx' },
  { key: 'yarnExtFactor', label: 'YaRN: factor de extrapolación', type: 'number', min: 0, max: 100, flag: true, group: 'longctx' },
  { key: 'yarnAttnFactor', label: 'YaRN: factor de atención', type: 'number', min: 0, max: 10, flag: true, group: 'longctx' },
  { key: 'yarnBetaSlow', label: 'YaRN: beta slow', type: 'number', min: 0, max: 100, flag: true, group: 'longctx' },
  { key: 'yarnBetaFast', label: 'YaRN: beta fast', type: 'number', min: 0, max: 1000, flag: true, group: 'longctx' },
  { key: 'cacheIdleSlots', label: 'Cache idle slots', type: 'toggle' },
  { key: 'mtp', label: 'MTP (Multi-Token Prediction)', type: 'toggle', presetKeys: ['specDraftNMax', 'specDraftNMin', 'specDraftPSplit'], presets: [
    { label: 'Estándar', values: { specDraftNMax: 3, specDraftNMin: 0, specDraftPSplit: 0.1 } },
    { label: 'Equilibrado', values: { specDraftNMax: 5, specDraftNMin: 0, specDraftPSplit: 0.15 } },
    { label: 'Máx. velocidad', values: { specDraftNMax: 8, specDraftNMin: 0, specDraftPSplit: 0.3 } },
    { label: 'Conservador', values: { specDraftNMax: 2, specDraftNMin: 1, specDraftPSplit: 0.05 } },
  ] },
  { key: 'specDraftModel', label: 'Archivo MTP separado (modelo draft)', type: 'file', dependsOn: 'mtp', placeholder: 'Ruta al archivo .gguf con la cabeza MTP' },
  { key: 'specDraftNMax', label: 'Draft: tokens a predecir', type: 'number', min: 1, max: 64, dependsOn: 'mtp' },
  { key: 'specDraftNMin', label: 'Draft: tokens mínimos', type: 'number', min: 0, max: 64, dependsOn: 'mtp' },
  { key: 'specDraftPSplit', label: 'Draft: probabilidad de split', type: 'slider', min: 0, max: 1, step: 0.01, dependsOn: 'mtp' },
];

const LONGCTX_KEYS = ['ropeScaling', 'ropeScale', 'ropeFreqBase', 'ropeFreqScale', 'yarnOrigCtx', 'yarnExtFactor', 'yarnAttnFactor', 'yarnBetaSlow', 'yarnBetaFast'];

const LONGCTX_PRESETS = [
  { label: 'Nativo', values: { ropeScaling: '', ropeScale: null, ropeFreqBase: null, ropeFreqScale: null, yarnOrigCtx: null, yarnExtFactor: null, yarnAttnFactor: null, yarnBetaSlow: null, yarnBetaFast: null }, desc: 'Sin extensión de contexto: usa lo que trae el modelo (default de llama.cpp).' },
  { label: 'Lineal 2x', values: { ropeScaling: 'linear', ropeScale: 2, ropeFreqBase: null, ropeFreqScale: null, yarnOrigCtx: null, yarnExtFactor: null, yarnAttnFactor: null, yarnBetaSlow: null, yarnBetaFast: null }, desc: 'Escalado lineal x2: duplica el contexto con degradación moderada en tokens lejanos.' },
  { label: 'YaRN 2x', values: { ropeScaling: 'yarn', ropeScale: 2, ropeFreqBase: null, ropeFreqScale: null, yarnOrigCtx: 4096, yarnExtFactor: 2, yarnAttnFactor: 1, yarnBetaSlow: 1, yarnBetaFast: 32 }, desc: 'YaRN x2 sobre 4096 nativo: muy buena calidad hasta 8K. Ajustá "yarn-orig-ctx" al contexto real de tu modelo.' },
  { label: 'YaRN 4x', values: { ropeScaling: 'yarn', ropeScale: 4, ropeFreqBase: null, ropeFreqScale: null, yarnOrigCtx: 4096, yarnExtFactor: 4, yarnAttnFactor: 1, yarnBetaSlow: 1, yarnBetaFast: 32 }, desc: 'YaRN x4 sobre 4096: extiende a ~16K con buena calidad.' },
  { label: 'YaRN 8x', values: { ropeScaling: 'yarn', ropeScale: 8, ropeFreqBase: null, ropeFreqScale: null, yarnOrigCtx: 4096, yarnExtFactor: 8, yarnAttnFactor: 1, yarnBetaSlow: 1, yarnBetaFast: 32 }, desc: 'YaRN x8 sobre 4096: hasta ~32K. La calidad decae en el extremo lejano.' },
];

const SERVER_FIELDS = [
  { key: 'apiKey', label: 'API key (autenticación)', type: 'text', placeholder: 'sk-...', flag: true },
  { key: 'reversePrompt', label: 'Prompt de corte (reverse-prompt)', type: 'text', placeholder: 'USER:', flag: true },
  { key: 'timeout', label: 'Timeout del servidor (seg)', type: 'number', min: 0, max: 86400, flag: true },
  { key: 'threadsHttp', label: 'Hilos HTTP (threads-http)', type: 'number', min: -1, max: 1024, flag: true },
  { key: 'apiPrefix', label: 'Prefijo de la API', type: 'text', placeholder: '/api/v1', flag: true },
  { key: 'corsOrigins', label: 'Orígenes CORS', type: 'text', placeholder: '*', flag: true },
  { key: 'reusePort', label: 'Reutilizar puerto (reuse-port)', type: 'toggle', flag: true },
  { key: 'noHost', label: 'No enlazar host (no-host)', type: 'toggle', flag: true },
  { key: 'metrics', label: 'Endpoint Prometheus (metrics)', type: 'toggle', flag: true },
  { key: 'props', label: 'Cambiar props por API (props)', type: 'toggle', flag: true },
  { key: 'slots', label: 'Endpoint de slots (slots)', type: 'toggle', flag: true },
  { key: 'contBatching', label: 'Batching continuo (cont-batching)', type: 'toggle', flag: true },
  { key: 'cachePrompt', label: 'Cache de prompt (cache-prompt)', type: 'toggle', flag: true },
  { key: 'cacheReuse', label: 'Mín. chunk para cache-reuse', type: 'number', min: 0, max: 100000, flag: true },
  { key: 'embedding', label: 'Modo solo embeddings (embedding)', type: 'toggle', flag: true },
  { key: 'pooling', label: 'Pooling (embeddings)', type: 'select', options: ['', 'none', 'mean', 'cls', 'last', 'rank'], descriptions: POOLING_DESCRIPTIONS, flag: true },
];

const ALL_FIELDS = SAMPLING_FIELDS.concat(RENDERING_FIELDS, SERVER_FIELDS);

const FIELD_BY_KEY = {};
ALL_FIELDS.forEach((f) => (FIELD_BY_KEY[f.key] = f));

const ARG_HELP = {};
Object.keys(FLAGS).forEach((key) => {
  const first = String(FLAGS[key]).split(' ')[0];
  if (first) ARG_HELP[first] = key;
});

const ARG_FIX_HINTS = {
  '--dry-penalty-last-n': { min: 0, fix: 64 },
};

let profiles = [];
let currentId = null;
let mtpCardEl = null;
const mtpDraftKeys = ['specDraftNMax', 'specDraftNMin', 'specDraftPSplit'];
let settings = {};
let modelsCache = [];
let running = false;
let runningProfileId = null;
let view = 'dashboard';
let saveTimer = null;
let dashSelMode = false;
let dashSelected = new Set();
let boardConfig = null;

function showDashboard() {
  view = 'dashboard';
  $('#dashboardView').classList.remove('hidden');
  $('#boardView').classList.add('hidden');
  $('#editorView').classList.add('hidden');
  $('#downloadsView').classList.add('hidden');
  $('#dashTopbar').classList.remove('hidden');
  $('#dashNavBtn').classList.add('active');
  $('#dlQueueNavBtn').classList.remove('active');
  renderDashboard();
}

function showEditor() {
  if (dashSelMode) exitDashSelection();
  view = 'editor';
  $('#dashboardView').classList.add('hidden');
  $('#boardView').classList.add('hidden');
  $('#editorView').classList.remove('hidden');
  $('#downloadsView').classList.add('hidden');
  $('#dashTopbar').classList.add('hidden');
  $('#dashNavBtn').classList.remove('active');
  $('#dlQueueNavBtn').classList.remove('active');
}

function showBoard() {
  if (dashSelMode) exitDashSelection();
  view = 'board';
  $('#dashboardView').classList.add('hidden');
  $('#boardView').classList.remove('hidden');
  $('#editorView').classList.add('hidden');
  $('#downloadsView').classList.add('hidden');
  $('#dashTopbar').classList.add('hidden');
  $('#dashNavBtn').classList.remove('active');
  $('#dlQueueNavBtn').classList.remove('active');
  setBoardTab('objects');
  renderBoard();
}

function setBoardTab(name) {
  document.querySelectorAll('.board-tab').forEach((b) => b.classList.toggle('active', b.dataset.boardTab === name));
  document.querySelectorAll('.board-tab-panel').forEach((p) => p.classList.toggle('active', p.dataset.boardPanel === name));
}

function openCommand(id) {
  selectProfile(id);
  showEditor();
}

function uid() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  }[c]));
}

function defaultProfile() {
  return {
    id: uid(),
    name: 'Nueva instancia',
    installId: null,
    modelPath: '',
    mmprojPath: '',
    gpuLayers: 26,
    gpuLayersAll: false,
    fit: 'off',
    imageMinTokens: 256,
    cacheIdleSlots: true,
    reasoning: 'on',
    ctxSize: 2048,
    cacheTypeK: 'f16',
    cacheTypeV: 'f16',
    temp: 0.6,
    topP: 0.95,
    topK: 20,
    minP: 0,
    presencePenalty: 0,
    repeatPenalty: 1,
    parallel: 1,
    host: '127.0.0.1',
    port: 8080,
    alias: '',
    mtp: false,
    specDraftModel: '',
    specDraftNMax: 3,
    specDraftNMin: 0,
    specDraftPSplit: 0.1,
    flashAttn: 'auto',
    threads: null,
    threadsBatch: null,
    loadMode: '',
    splitMode: '',
    mainGpu: null,
    tensorSplit: '',
    device: '',
    kvOffload: true,
    opOffload: true,
    cpuMoe: false,
    nCpuMoe: null,
    ropeScaling: '',
    ropeScale: null,
    ropeFreqBase: null,
    ropeFreqScale: null,
    yarnOrigCtx: null,
    yarnExtFactor: null,
    yarnAttnFactor: null,
    yarnBetaSlow: null,
    yarnBetaFast: null,
    seed: -1,
    samplers: '',
    ignoreEos: false,
    topNSigma: 0,
    typicalP: 1,
    xtcProbability: 0,
    xtcThreshold: 0.1,
    frequencyPenalty: 0,
    repeatLastN: 64,
    dryMultiplier: 0,
    dryBase: 1.75,
    dryAllowedLength: 2,
    dryPenaltyLastN: null,
    dynatempRange: 0,
    dynatempExp: 1,
    mirostat: 0,
    mirostatLr: 0.1,
    mirostatEnt: 5,
    adaptiveTarget: 0,
    adaptiveDecay: 8,
    logitBias: '',
    grammar: '',
    jsonSchema: '',
    apiKey: '',
    reversePrompt: '',
    timeout: null,
    threadsHttp: null,
    apiPrefix: '',
    corsOrigins: '',
    reusePort: false,
    noHost: false,
    metrics: false,
    props: false,
    slots: true,
    contBatching: true,
    cachePrompt: true,
    cacheReuse: null,
    embedding: false,
    pooling: '',
    configured: [],
  };
}

function markConfigured(p) {
  if (!p || !Array.isArray(p.configured)) return;
  const keys = Array.from(arguments).slice(1);
  const set = new Set(p.configured);
  keys.forEach((k) => set.add(k));
  p.configured = Array.from(set);
}

function unmarkConfigured(p) {
  if (!p || !Array.isArray(p.configured)) return;
  const keys = new Set(Array.from(arguments).slice(1));
  p.configured = p.configured.filter((k) => !keys.has(k));
}

function current() {
  return profiles.find((p) => p.id === currentId) || null;
}

function installationById(id) {
  return (settings.installations || []).find((i) => i.id === id) || null;
}

function effectiveInstall(p) {
  const prof = p || current();
  const sel = prof && prof.installId ? installationById(prof.installId) : null;
  return sel || (settings.installations && settings.installations[0]) || null;
}

function installBuildNum(install) {
  const s = String((install && (install.path || install.name)) || '');
  const m = s.match(/vb?(\d{4,})/i);
  return m ? parseInt(m[1], 10) : null;
}

function isNewDialect(install) {
  const n = installBuildNum(install);
  return n !== null && n >= 10361;
}

function installLabel(i) {
  if (!i) return '';
  const base = i.path ? i.path.split(/[\\/]/).filter(Boolean).pop() : '';
  return base ? i.name + ' — ' + base : i.name;
}

function num(v) {
  return v === '' || v === null || v === undefined || Number.isNaN(Number(v)) ? null : Number(v);
}

function buildArgs(p) {
  const legacy = !Array.isArray(p.configured);
  const cfg = (k) => legacy || p.configured.includes(k);
  const silent = (k, v) => SILENT_VALUES[k] !== undefined && String(v) === String(SILENT_VALUES[k]);
  const install = effectiveInstall(p);
  const newD = isNewDialect(install);
  const args = [];
  const pushStr = (flag, val, key) => {
    if (val !== '' && val !== null && val !== undefined && cfg(key) && !silent(key, val)) args.push(flag, String(val));
  };
  const pushNum = (flag, val, key) => {
    const n = num(val);
    if (n !== null && cfg(key) && !silent(key, n)) args.push(flag, String(n));
  };
  const pushOn = (flag, val, key) => {
    if (val && cfg(key)) args.push(flag);
  };
  const pushOff = (flag, val, key) => {
    if (val === false && cfg(key)) args.push(flag);
  };

  pushStr('--model', p.modelPath, 'modelPath');
  pushStr('--mmproj', p.mmprojPath, 'mmprojPath');
  if (p.gpuLayersAll && newD) {
    args.push(FLAGS.gpuLayers, 'all');
  } else {
    pushNum(FLAGS.gpuLayers, p.gpuLayers, 'gpuLayers');
  }
  if (newD) {
    args.push(FLAGS.fit, 'off');
  } else {
    pushStr(FLAGS.fit, p.fit, 'fit');
  }
  pushNum(FLAGS.imageMinTokens, p.imageMinTokens, 'imageMinTokens');
  pushOn(FLAGS.cacheIdleSlots, p.cacheIdleSlots, 'cacheIdleSlots');
  pushStr(FLAGS.reasoning, p.reasoning, 'reasoning');
  pushNum(FLAGS.ctxSize, p.ctxSize, 'ctxSize');
  pushStr(FLAGS.cacheTypeK, p.cacheTypeK, 'cacheTypeK');
  pushStr(FLAGS.cacheTypeV, p.cacheTypeV, 'cacheTypeV');
  if (newD && p.mtp && cfg('mtp') && cfg('cacheTypeK')) args.push('--cache-type-k-draft', String(p.cacheTypeK));
  if (newD && p.mtp && cfg('mtp') && cfg('cacheTypeV')) args.push('--cache-type-v-draft', String(p.cacheTypeV));
  pushNum(FLAGS.temp, p.temp, 'temp');
  pushNum(FLAGS.topP, p.topP, 'topP');
  pushNum(FLAGS.topK, p.topK, 'topK');
  pushNum(FLAGS.minP, p.minP, 'minP');
  pushNum(FLAGS.presencePenalty, p.presencePenalty, 'presencePenalty');
  pushNum(FLAGS.repeatPenalty, p.repeatPenalty, 'repeatPenalty');
  pushNum(FLAGS.parallel, p.parallel, 'parallel');
  pushStr(FLAGS.host, p.host, 'host');
  pushNum(FLAGS.port, p.port, 'port');
  pushStr(FLAGS.alias, p.alias, 'alias');
  if (p.mtp && cfg('mtp')) {
    if (newD) {
      pushStr('--model-draft', p.specDraftModel, 'specDraftModel');
      args.push('--spec-type', 'draft-mtp');
      pushNum('--spec-draft-n-max', p.specDraftNMax, 'specDraftNMax');
    } else {
      args.push('--spec-type', 'draft-mtp');
      pushStr(FLAGS.specDraftModel, p.specDraftModel, 'specDraftModel');
      pushNum(FLAGS.specDraftNMax, p.specDraftNMax, 'specDraftNMax');
      pushNum(FLAGS.specDraftNMin, p.specDraftNMin, 'specDraftNMin');
      pushNum(FLAGS.specDraftPSplit, p.specDraftPSplit, 'specDraftPSplit');
    }
  }
  pushStr(FLAGS.flashAttn, p.flashAttn, 'flashAttn');
  pushNum(FLAGS.threads, p.threads, 'threads');
  pushNum(FLAGS.threadsBatch, p.threadsBatch, 'threadsBatch');
  pushStr(FLAGS.loadMode, p.loadMode, 'loadMode');
  pushOff('--no-kv-offload', p.kvOffload, 'kvOffload');
  pushOff('--no-op-offload', p.opOffload, 'opOffload');
  pushOn(FLAGS.cpuMoe, p.cpuMoe, 'cpuMoe');
  if (num(p.nCpuMoe) > 0 && cfg('nCpuMoe')) pushNum(FLAGS.nCpuMoe, p.nCpuMoe, 'nCpuMoe');
  pushStr(FLAGS.splitMode, p.splitMode, 'splitMode');
  pushNum(FLAGS.mainGpu, p.mainGpu, 'mainGpu');
  pushStr(FLAGS.tensorSplit, p.tensorSplit, 'tensorSplit');
  pushStr(FLAGS.device, p.device, 'device');
  pushStr(FLAGS.ropeScaling, p.ropeScaling, 'ropeScaling');
  pushNum(FLAGS.ropeScale, p.ropeScale, 'ropeScale');
  pushNum(FLAGS.ropeFreqBase, p.ropeFreqBase, 'ropeFreqBase');
  pushNum(FLAGS.ropeFreqScale, p.ropeFreqScale, 'ropeFreqScale');
  pushNum(FLAGS.yarnOrigCtx, p.yarnOrigCtx, 'yarnOrigCtx');
  pushNum(FLAGS.yarnExtFactor, p.yarnExtFactor, 'yarnExtFactor');
  pushNum(FLAGS.yarnAttnFactor, p.yarnAttnFactor, 'yarnAttnFactor');
  pushNum(FLAGS.yarnBetaSlow, p.yarnBetaSlow, 'yarnBetaSlow');
  pushNum(FLAGS.yarnBetaFast, p.yarnBetaFast, 'yarnBetaFast');
  pushNum(FLAGS.seed, p.seed, 'seed');
  pushStr(FLAGS.samplers, p.samplers, 'samplers');
  pushOn(FLAGS.ignoreEos, p.ignoreEos, 'ignoreEos');
  pushNum(FLAGS.topNSigma, p.topNSigma, 'topNSigma');
  pushNum(FLAGS.typicalP, p.typicalP, 'typicalP');
  pushNum(FLAGS.xtcProbability, p.xtcProbability, 'xtcProbability');
  pushNum(FLAGS.xtcThreshold, p.xtcThreshold, 'xtcThreshold');
  pushNum(FLAGS.frequencyPenalty, p.frequencyPenalty, 'frequencyPenalty');
  pushNum(FLAGS.repeatLastN, p.repeatLastN, 'repeatLastN');
  if (num(p.dryMultiplier) > 0 && cfg('dryMultiplier')) {
    pushNum(FLAGS.dryMultiplier, p.dryMultiplier, 'dryMultiplier');
    pushNum(FLAGS.dryBase, p.dryBase, 'dryBase');
    pushNum(FLAGS.dryAllowedLength, p.dryAllowedLength, 'dryAllowedLength');
    pushNum(FLAGS.dryPenaltyLastN, p.dryPenaltyLastN, 'dryPenaltyLastN');
  }
  pushNum(FLAGS.dynatempRange, p.dynatempRange, 'dynatempRange');
  pushNum(FLAGS.dynatempExp, p.dynatempExp, 'dynatempExp');
  pushStr(FLAGS.mirostat, p.mirostat, 'mirostat');
  pushNum(FLAGS.mirostatLr, p.mirostatLr, 'mirostatLr');
  pushNum(FLAGS.mirostatEnt, p.mirostatEnt, 'mirostatEnt');
  pushNum(FLAGS.adaptiveTarget, p.adaptiveTarget, 'adaptiveTarget');
  pushNum(FLAGS.adaptiveDecay, p.adaptiveDecay, 'adaptiveDecay');
  pushStr(FLAGS.logitBias, p.logitBias, 'logitBias');
  pushStr(FLAGS.grammar, p.grammar, 'grammar');
  pushStr(FLAGS.jsonSchema, p.jsonSchema, 'jsonSchema');
  pushStr(FLAGS.apiKey, p.apiKey, 'apiKey');
  pushStr(FLAGS.reversePrompt, p.reversePrompt, 'reversePrompt');
  pushNum(FLAGS.timeout, p.timeout, 'timeout');
  pushNum(FLAGS.threadsHttp, p.threadsHttp, 'threadsHttp');
  pushStr(FLAGS.apiPrefix, p.apiPrefix, 'apiPrefix');
  pushStr(FLAGS.corsOrigins, p.corsOrigins, 'corsOrigins');
  pushOn(FLAGS.reusePort, p.reusePort, 'reusePort');
  pushOn(FLAGS.noHost, p.noHost, 'noHost');
  pushOn(FLAGS.metrics, p.metrics, 'metrics');
  pushOn(FLAGS.props, p.props, 'props');
  pushOff('--no-slots', p.slots, 'slots');
  pushOff('--no-cont-batching', p.contBatching, 'contBatching');
  pushOff('--no-cache-prompt', p.cachePrompt, 'cachePrompt');
  pushNum(FLAGS.cacheReuse, p.cacheReuse, 'cacheReuse');
  pushOn(FLAGS.embedding, p.embedding, 'embedding');
  pushStr(FLAGS.pooling, p.pooling, 'pooling');
  return args;
}

/* ---------------- Editor de instancia (tab "Instancia") ---------------- */

const CMD_FLAG_TO_KEY = {};
Object.keys(FLAGS).forEach((key) => {
  const flag = String(FLAGS[key]).split(' ')[0];
  if (flag) CMD_FLAG_TO_KEY[flag] = key;
});

const CMD_NO_VALUE_TRUE = new Set([
  '--cache-idle-slots',
  '--cpu-moe',
  '--ignore-eos',
  '--reuse-port',
  '--no-host',
  '--metrics',
  '--props',
  '--embedding',
  '--slots',
  '--cont-batching',
  '--cache-prompt',
  '--kv-offload',
  '--op-offload',
]);

const CMD_SPECIAL = {
  '--model': { key: 'modelPath', kind: 'value' },
  '--mmproj': { key: 'mmprojPath', kind: 'value' },
  '--spec-type': { key: 'mtp', kind: 'mtp' },
  '--model-draft': { key: 'specDraftModel', kind: 'value' },
  '--no-kv-offload': { key: 'kvOffload', kind: 'false' },
  '--no-op-offload': { key: 'opOffload', kind: 'false' },
  '--no-slots': { key: 'slots', kind: 'false' },
  '--no-cont-batching': { key: 'contBatching', kind: 'false' },
  '--no-cache-prompt': { key: 'cachePrompt', kind: 'false' },
};

function tokenizeCommand(text) {
  const tokens = [];
  const re = /"([^"]*)"|'([^']*)'|(\S+)/g;
  let m;
  while ((m = re.exec(String(text || '')))) {
    tokens.push(m[1] !== undefined ? m[1] : m[2] !== undefined ? m[2] : m[3]);
  }
  return tokens;
}

function applyCommandToProfile(text) {
  const tokens = tokenizeCommand(text);
  const values = {};
  const errors = [];
  let i = 0;
  if (tokens.length && tokens[0].toLowerCase().includes('llama-server')) i = 1;
  while (i < tokens.length) {
    const raw = tokens[i];
    i++;
    let flag = raw;
    let inline = null;
    const eq = raw.indexOf('=');
    if (eq > 0) {
      flag = raw.slice(0, eq);
      inline = raw.slice(eq + 1);
    }
    if (!flag.startsWith('--')) {
      errors.push('Token inesperado: “' + raw + '”');
      continue;
    }
    if (CMD_NO_VALUE_TRUE.has(flag)) {
      const k = CMD_FLAG_TO_KEY[flag];
      if (k) values[k] = true;
      else errors.push('Flag desconocido: ' + flag);
      continue;
    }
    if (CMD_SPECIAL[flag]) {
      const spec = CMD_SPECIAL[flag];
      if (spec.kind === 'false') {
        values[spec.key] = false;
        continue;
      }
      let val = inline;
      if (val === null) {
        val = tokens[i];
        i++;
      }
      if (val === undefined) {
        errors.push(flag + ' requiere un valor');
        continue;
      }
      if (spec.kind === 'mtp') {
        values.mtp = true;
        if (val !== 'draft-mtp') errors.push(flag + ' solo admite "draft-mtp"');
        continue;
      }
      values[spec.key] = String(val);
      continue;
    }
    const key = CMD_FLAG_TO_KEY[flag];
    if (!key) {
      errors.push('Flag desconocido: ' + flag);
      continue;
    }
    const field = FIELD_BY_KEY[key];
    let val = inline;
    if (val === null) {
      val = tokens[i];
      i++;
    }
    if (val === undefined) {
      errors.push(flag + ' requiere un valor');
      continue;
    }
    if (field && (field.type === 'number' || field.type === 'slider')) {
      const n = num(val);
      if (n === null) errors.push('Valor inválido para ' + flag + ': “' + val + '”');
      else values[key] = n;
    } else {
      values[key] = String(val);
    }
  }
  return { values, errors };
}

function buildCommandLine(p) {
  const install = effectiveInstall(p);
  const exe = install && install.exePath ? install.exePath : 'llama-server.exe';
  const quote = (a) => {
    const s = String(a);
    if (!/[\s]/.test(s)) return s;
    return s.includes('"') ? "'" + s + "'" : '"' + s + '"';
  };
  const parts = [exe].concat(buildArgs(p)).map(quote);
  return parts.map((a) => (/^--/.test(a) ? '\n' + a : a)).join(' ');
}

function renderCommandPanel() {
  const area = $('#cmdTextarea');
  const status = $('#cmdStatus');
  if (!area) return;
  const p = current();
  area.value = p ? buildCommandLine(p) : '';
  if (status) {
    status.textContent = '';
    status.classList.remove('err');
  }
}

function applyCommandEdit() {
  const p = current();
  if (!p) return;
  const src = $('#cmdTextarea').value;
  const res = applyCommandToProfile(src);
  const status = $('#cmdStatus');
  if (res.errors.length) {
    status.textContent = res.errors.join('\n');
    status.classList.add('err');
    return;
  }
  const def = defaultProfile();
  ['modelPath', 'mmprojPath'].forEach((k) => (p[k] = def[k]));
  Object.keys(FLAGS).forEach((k) => (p[k] = def[k]));
  Object.assign(p, res.values);
  p.configured = Object.keys(res.values);
  const first = tokenizeCommand(src)[0];
  if (first && first.toLowerCase().endsWith('.exe')) {
    const match = (settings.installations || []).find((inst) => String(inst.exePath || '').toLowerCase() === first.toLowerCase());
    if (match) p.installId = match.id;
  }
  scheduleSave();
  renderCommandPanel();
  renderSchemaPanels();
  applyFieldVisibility();
  syncGeneral();
  refresh();
  status.textContent = t('cmd_status');
  status.classList.remove('err');
  toast(t('toast_instance_applied'), 'ok');
}

function scheduleSave() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(async () => {
    try {
      await window.api.saveProfiles(profiles);
    } catch (e) {}
    scheduleFileCheck();
  }, 250);
}

function toast(msg, kind) {
  const el = $('#toast');
  el.textContent = msg;
  el.className = 'toast' + (kind ? ' ' + kind : '');
  clearTimeout(toast._t);
  toast._t = setTimeout(() => el.classList.add('hidden'), 2600);
}

function confirmDialog(opts) {
  opts = opts || {};
  return new Promise((resolve) => {
    const modal = $('#confirmModal');
    const titleEl = $('#confirmTitle');
    const msgEl = $('#confirmMsg');
    const okBtn = $('#confirmOkBtn');
    const cancelBtn = $('#confirmCancelBtn');
    const closeBtn = $('#confirmClose');
    titleEl.textContent = opts.title || 'Confirmar';
    msgEl.textContent = opts.message || '';
    okBtn.textContent = opts.okLabel || 'Confirmar';
    okBtn.className = 'btn ' + (opts.danger ? 'danger' : 'btn-accent');
    cancelBtn.textContent = opts.cancelLabel || 'Cancelar';
    modal.classList.remove('hidden');
    okBtn.focus();
    const done = (val) => {
      modal.classList.add('hidden');
      okBtn.removeEventListener('click', onOk);
      cancelBtn.removeEventListener('click', onCancel);
      closeBtn.removeEventListener('click', onCancel);
      document.removeEventListener('keydown', onKey);
      modal.removeEventListener('mousedown', onBackdrop);
      resolve(val);
    };
    const onOk = () => done(true);
    const onCancel = () => done(false);
    const onKey = (e) => {
      if (e.key === 'Escape') {
        e.stopPropagation();
        onCancel();
      } else if (e.key === 'Enter') {
        e.stopPropagation();
        onOk();
      }
    };
    const onBackdrop = (e) => {
      if (e.target === modal) onCancel();
    };
    okBtn.addEventListener('click', onOk);
    cancelBtn.addEventListener('click', onCancel);
    closeBtn.addEventListener('click', onCancel);
    document.addEventListener('keydown', onKey);
    modal.addEventListener('mousedown', onBackdrop);
  });
}

function setRunning(value) {
  running = value;
  const badge = $('#statusBadge');
  const btn = $('#runBtn');
  badge.textContent = value ? t('status_running') : t('status_stopped');
  badge.className = 'badge ' + (value ? 'running' : 'stopped');
  btn.textContent = value ? t('btn_stop') : t('btn_start');
  btn.className = 'btn run-btn ' + (value ? 'running' : 'stopped');
  const boardBtn = $('#boardRunBtn');
  if (boardBtn) {
    boardBtn.textContent = value ? t('btn_stop') : t('btn_start');
    boardBtn.className = 'btn run-btn ' + (value ? 'running' : 'stopped');
  }
  const dashBadge = $('#dashStatusBadge');
  dashBadge.textContent = value ? t('status_running') : t('status_stopped');
  dashBadge.className = 'badge ' + (value ? 'running' : 'stopped');
  if (value) resetSpeed();
  renderSidebar();
  renderDashboard();
}

/* ---------------- General panel ---------------- */

function buildGeneralPanel() {
  const panel = $('#panel-general');

  const installCard = document.createElement('div');
  installCard.className = 'card';
  installCard.style.flex = '1 1 100%';
  installCard.style.minWidth = '100%';
  installCard.innerHTML = `
    <div class="card-head">
      <span>Versión de llama.cpp</span>
      <button id="manageInstallsBtn" class="btn small">Gestionar versiones</button>
    </div>
    <div class="field" style="margin-bottom:8px">
      <div class="field-head"><label>Instalación a usar en esta instancia</label></div>
      <select id="installSelect" class="text-input"></select>
    </div>
    <div id="installSummary" class="hint"></div>
  `;
  panel.appendChild(installCard);

  const modelCard = document.createElement('div');
  modelCard.className = 'card';
  modelCard.innerHTML = `
    <div class="card-head"><span>Modelo y red</span></div>
    <div class="field">
      <div class="field-head"><label>Modelo principal</label><code class="flag-chip">--model</code></div>
      <div class="pickrow">
        <select id="modelSelect" class="text-input"><option value="">— Elegir modelo —</option></select>
        <button id="browseModelBtn" class="btn small">Examinar</button>
        <button id="clearModelBtn" class="btn small ghost">Quitar</button>
      </div>
      <div id="modelPathDisplay" class="path-display empty">Ningún modelo seleccionado</div>
    </div>
    <div class="check-field">
      <input type="checkbox" id="useVision" />
      <span>Usar modelo de visión (multimodal)</span>
      <code class="flag-chip">--mmproj</code>
    </div>
    <div id="visionRow" class="field" style="display:none">
      <div class="pickrow">
        <select id="visionSelect" class="text-input"><option value="">— Elegir archivo —</option></select>
        <button id="browseVisionBtn" class="btn small">Examinar</button>
      </div>
      <div id="visionPathDisplay" class="path-display empty">Ningún archivo de visión seleccionado</div>
    </div>
  `;
  panel.appendChild(modelCard);

  const netCard = document.createElement('div');
  netCard.className = 'card';
  netCard.innerHTML = `
    <div class="card-head"><span>Red</span></div>
    <div class="field">
      <div class="field-head"><label>Host</label><code class="flag-chip">--host</code></div>
      <input id="hostInput" type="text" class="text-input" placeholder="127.0.0.1" />
    </div>
    <div class="field">
      <div class="field-head"><label>Puerto</label><code class="flag-chip">--port</code></div>
      <input id="portInput" type="number" class="text-input" placeholder="8080" min="0" max="65535" />
    </div>
    <div class="field">
      <div class="field-head"><label>Alias</label><code class="flag-chip">--alias</code></div>
      <input id="aliasInput" type="text" class="text-input" placeholder="mi-modelo" />
    </div>
  `;
  panel.appendChild(netCard);

  $('#browseModelBtn').addEventListener('click', async () => {
    const f = await window.api.selectModelFile();
    if (f) {
      const p = current();
      markConfigured(p, 'modelPath');
      p.modelPath = f;
      syncGeneral();
      refresh();
    }
  });
  $('#clearModelBtn').addEventListener('click', () => {
    markConfigured(current(), 'modelPath');
    current().modelPath = '';
    syncGeneral();
    refresh();
  });
  $('#modelSelect').addEventListener('change', (e) => {
    if (e.target.value) {
      markConfigured(current(), 'modelPath');
      current().modelPath = e.target.value;
      syncGeneral();
      refresh();
    }
  });
  $('#browseVisionBtn').addEventListener('click', async () => {
    const f = await window.api.selectVisionFile();
    if (f) {
      markConfigured(current(), 'mmprojPath');
      current().mmprojPath = f;
      syncGeneral();
      refresh();
    }
  });
  $('#visionSelect').addEventListener('change', (e) => {
    if (e.target.value) {
      markConfigured(current(), 'mmprojPath');
      current().mmprojPath = e.target.value;
      syncGeneral();
      refresh();
    }
  });
  $('#useVision').addEventListener('change', (e) => {
    if (e.target.checked && !current().mmprojPath) {
      window.api.selectVisionFile().then((f) => {
        if (f) {
          markConfigured(current(), 'mmprojPath');
          current().mmprojPath = f;
        } else {
          e.target.checked = false;
        }
        syncGeneral();
        refresh();
      });
      return;
    }
    if (!e.target.checked) current().mmprojPath = '';
    syncGeneral();
    refresh();
  });

  const bindText = (sel, key) => {
    $(sel).addEventListener('input', (e) => {
      const p = current();
      markConfigured(p, key);
      p[key] = e.target.value;
      refresh();
    });
  };
  const bindNumber = (sel, key) => {
    $(sel).addEventListener('input', (e) => {
      const p = current();
      markConfigured(p, key);
      p[key] = e.target.value === '' ? null : Number(e.target.value);
      refresh();
    });
  };
  bindText('#hostInput', 'host');
  bindNumber('#portInput', 'port');
  bindText('#aliasInput', 'alias');

  $('#installSelect').addEventListener('change', (e) => {
    current().installId = e.target.value || null;
    refresh();
  });
  $('#manageInstallsBtn').addEventListener('click', openSettings);
}

function populateInstallSelect() {
  const sel = $('#installSelect');
  const installs = settings.installations || [];
  sel.innerHTML =
    '<option value="">— Usar la primera versión —</option>' +
    installs
      .map((i) => `<option value="${escapeHtml(i.id)}">${escapeHtml(installLabel(i))}</option>`)
      .join('');
}

function syncGeneral() {
  const p = current();
  if (!p) return;
  const hasVision = !!p.mmprojPath;
  $('#useVision').checked = hasVision;
  $('#visionRow').style.display = hasVision ? '' : 'none';
  $('#modelPathDisplay').textContent = p.modelPath || 'Ningún modelo seleccionado';
  $('#modelPathDisplay').classList.toggle('empty', !p.modelPath);
  $('#visionPathDisplay').textContent = p.mmprojPath || 'Ningún archivo de visión seleccionado';
  $('#visionPathDisplay').classList.toggle('empty', !p.mmprojPath);
  $('#modelSelect').value = p.modelPath;
  $('#visionSelect').value = p.mmprojPath;
  $('#hostInput').value = p.host || '';
  $('#portInput').value = p.port ?? '';
  $('#aliasInput').value = p.alias || '';

  populateInstallSelect();
  $('#installSelect').value = p.installId || '';
  const eff = effectiveInstall();
  const summary = $('#installSummary');
  if (eff) {
    summary.innerHTML =
      'Usando <strong>' + escapeHtml(installLabel(eff)) + '</strong> · ' +
      '<span class="' + (eff.exePath ? 'text-ok' : 'text-err') + '">' +
      (eff.exePath ? escapeHtml(eff.exePath) : 'no se encontró llama-server.exe en la carpeta') +
      '</span>';
  } else {
    summary.textContent = 'No hay versiones registradas. Agregá una en Configuración.';
  }

  const modelPath = p.modelPath || '';
  if (modelPath !== lastModelPath) {
    lastModelPath = modelPath;
    refreshModelInfo();
  }
}

/* ---------------- Schema-driven panels ---------------- */

function buildControl(field, getValue, onValue) {
  const wrap = document.createElement('div');
  wrap.className = 'field';
  wrap.dataset.fieldWrap = field.key;

  if (field.type === 'slider') {
    wrap.innerHTML = `
      <div class="field-head">
        <label>${escapeHtml(field.label)}</label>
        <span class="value-badge"></span>
      </div>
      <div class="slider-row">
        <input type="range" min="${field.min}" max="${field.max}" step="${field.step}" />
        ${field.noNumber ? '' : `<input type="number" class="slider-num" min="${field.min}" max="${field.max}" step="${field.step}" />`}
      </div>
    `;
    const range = wrap.querySelector('input[type="range"]');
    const number = wrap.querySelector('input[type="number"]');
    const badge = wrap.querySelector('.value-badge');
    const updateBadge = (v) => {
      badge.textContent = Number(v.toFixed(4));
      if (field.hint) runFieldHints();
    };
    const apply = () => {
      const v = getValue();
      const vn = num(v) ?? 0;
      range.value = vn;
      if (number) number.value = vn;
      updateBadge(vn);
    };
    range.addEventListener('input', () => {
      onValue(Number(range.value));
      updateBadge(Number(range.value));
    });
    if (number) {
      number.addEventListener('input', () => {
        const n = num(number.value);
        if (n !== null) {
          onValue(n);
          range.value = n;
          updateBadge(n);
        }
      });
    }
    apply();
  } else if (field.type === 'number') {
    wrap.innerHTML = `
      <div class="field-head">
        <label>${escapeHtml(field.label)}</label>
      </div>
      <input type="number" class="text-input${field.wide ? ' num-wide' : ''}" min="${field.min}" max="${field.max}" />
    `;
    const input = wrap.querySelector('input');
    input.value = getValue() ?? '';
    const highlightPreset = (n) => {
      if (!field.presets) return;
      wrap.querySelectorAll('.preset-btn').forEach((b) => b.classList.toggle('active', num(b.dataset.val) === n));
    };
    input.addEventListener('input', () => {
      const n = num(input.value);
      onValue(n === null ? null : n);
      highlightPreset(n);
      if (field.hint) runFieldHints();
    });
    if (field.presets) {
      const row = document.createElement('div');
      row.className = 'preset-row';
      field.presets.forEach((v) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'preset-btn';
        btn.dataset.val = v;
        btn.textContent = v >= 1024 ? String(v / 1024) + 'K' : String(v);
        btn.addEventListener('click', () => {
          input.value = v;
          onValue(v);
          highlightPreset(v);
          if (field.hint) runFieldHints();
        });
        row.appendChild(btn);
      });
      wrap.appendChild(row);
    }
    highlightPreset(getValue());
  } else if (field.type === 'select') {
    wrap.innerHTML = `
      <div class="field-head">
        <label>${escapeHtml(field.label)}</label>
      </div>
      <select data-key="${field.key}">${field.options
        .map((o) => {
          const label = field.descriptions && field.descriptions[o] ? (o ? o + ' — ' + field.descriptions[o] : field.descriptions[o]) : o;
          return `<option value="${escapeHtml(o)}">${escapeHtml(label)}</option>`;
        })
        .join('')}</select>
    `;
    const sel = wrap.querySelector('select');
    sel.value = getValue() ?? '';
    sel.addEventListener('change', () => {
      onValue(sel.value);
      if (field.hint || field.key === 'cacheTypeK' || field.key === 'cacheTypeV') runFieldHints();
    });
  } else if (field.type === 'text') {
    wrap.innerHTML = `
      <div class="field-head">
        <label>${escapeHtml(field.label)}</label>
      </div>
      <input type="text" class="text-input" placeholder="${escapeHtml(field.placeholder || '')}" />
    `;
    const input = wrap.querySelector('input');
    input.value = getValue() ?? '';
    input.addEventListener('input', () => onValue(input.value));
  } else if (field.type === 'file') {
    wrap.innerHTML = `
      <div class="field-head">
        <label>${escapeHtml(field.label)}</label>
      </div>
      <div class="pickrow">
        <input type="text" class="text-input" placeholder="${escapeHtml(field.placeholder || '')}" />
        <button type="button" class="btn small">Examinar</button>
        <button type="button" class="btn small ghost">Quitar</button>
      </div>
    `;
    const input = wrap.querySelector('input');
    input.value = getValue() ?? '';
    input.addEventListener('input', () => onValue(input.value));
    const btns = wrap.querySelectorAll('button');
    btns[0].addEventListener('click', async () => {
      const f = await window.api.selectModelFile();
      if (f) {
        input.value = f;
        onValue(f);
      }
    });
    btns[1].addEventListener('click', () => {
      input.value = '';
      onValue('');
    });
  } else if (field.type === 'toggle') {
    wrap.innerHTML = `
      <div class="field-head">
        <label>${escapeHtml(field.label)}</label>
        <code class="flag-chip">${escapeHtml(FLAGS[field.key] || '')}</code>
      </div>
      <div class="check-field" style="margin-bottom:0">
        <input type="checkbox" />
        <span>Activo</span>
      </div>
    `;
    const cb = wrap.querySelector('input');
    cb.checked = !!getValue();
    cb.addEventListener('change', () => onValue(cb.checked));
    if (field.presets) {
      const row = document.createElement('div');
      row.className = 'preset-row mtp-presets';
      const p = current();
      const active = p ? field.presets.find((pr) => field.presetKeys.every((k) => p[k] === pr.values[k])) : null;
      field.presets.forEach((pr) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'preset-btn' + (active === pr ? ' active' : '');
        btn.textContent = pr.label;
        btn.title = field.presetKeys.map((k) => FLAGS[k] + ' = ' + pr.values[k]).join('  ·  ');
        btn.addEventListener('click', () => {
          const prof = current();
          if (!prof) return;
          markConfigured(prof, field.key, ...field.presetKeys);
          field.presetKeys.forEach((k) => {
            prof[k] = pr.values[k];
          });
          renderSchemaPanels();
          refresh();
          applyFieldVisibility();
        });
        row.appendChild(btn);
      });
      if (p && !active) {
        const custom = document.createElement('span');
        custom.className = 'preset-custom';
        custom.textContent = 'Custom';
        custom.title = 'Valores personalizados';
        row.appendChild(custom);
      }
      wrap.appendChild(row);
    }
  }

  if (field.flag && field.type !== 'toggle') {
    const label = wrap.querySelector('.field-head label');
    if (label) label.insertAdjacentHTML('beforeend', ' <code class="flag-chip">' + escapeHtml(FLAGS[field.key] || '') + '</code>');
  }
  return wrap;
}

/* ---------------- Sidebar ---------------- */

function renderSidebar() {
  const list = $('#profileList');
  list.innerHTML = '';
  profiles.forEach((p) => {
    const item = document.createElement('button');
    item.className = 'profile-item' + (p.id === currentId ? ' active' : '');
    const model = p.modelPath ? p.modelPath.split(/[\\/]/).pop() : t('card_no_model');
    item.innerHTML = `
      <span class="item-dot"></span>
      <span style="flex:1;min-width:0">
        <div class="item-name">${escapeHtml(p.name || t('card_no_name'))}</div>
        <div class="item-model">${escapeHtml(model)}</div>
      </span>
    `;
    item.addEventListener('click', () => openCommand(p.id));
    list.appendChild(item);
  });
}

/* ---------------- Validación de instancias ---------------- */

let fileStates = {};
let fileCheckTimer = null;

function profileProblems(p) {
  const problems = [];
  const install = effectiveInstall(p);
  if (p.installId && !installationById(p.installId)) {
    problems.push('La versión de llama.cpp configurada ya no existe');
  } else if (!install) {
    problems.push('No hay versión de llama.cpp configurada');
  } else if (install.exePath && fileStates[install.exePath] === false) {
    problems.push('La versión de llama.cpp no está en disco');
  }
  if (!p.modelPath) {
    problems.push('No hay modelo configurado');
  } else if (fileStates[p.modelPath] === false) {
    problems.push('El modelo no está en disco');
  }
  if (p.mmprojPath && fileStates[p.mmprojPath] === false) {
    problems.push('El archivo de visión no está en disco');
  }
  if (p.mtp && p.specDraftModel && fileStates[p.specDraftModel] === false) {
    problems.push('El modelo MTP no está en disco');
  }
  return problems;
}

function profileHasProblems(p) {
  return profileProblems(p).length > 0;
}

async function refreshFileStates() {
  const paths = new Set();
  profiles.forEach((p) => {
    const install = effectiveInstall(p);
    if (install && install.exePath) paths.add(install.exePath);
    if (p.modelPath) paths.add(p.modelPath);
    if (p.mmprojPath) paths.add(p.mmprojPath);
    if (p.mtp && p.specDraftModel) paths.add(p.specDraftModel);
  });
  const list = [...paths];
  if (!list.length) {
    fileStates = {};
    renderDashboard();
    return;
  }
  let res = {};
  try {
    res = (await window.api.checkFiles(list)) || {};
  } catch (e) {}
  fileStates = res;
  renderDashboard();
}

function scheduleFileCheck() {
  clearTimeout(fileCheckTimer);
  fileCheckTimer = setTimeout(refreshFileStates, 350);
}

function renderDashboard() {
  const grid = $('#dashGrid');
  grid.innerHTML = '';
  profiles.forEach((p) => {
    const model = p.modelPath ? p.modelPath.split(/[\\/]/).pop() : '';
    const install = installationById(p.installId) || (settings.installations && settings.installations[0]);
    const isRunning = p.id === runningProfileId;
    const selected = dashSelected.has(p.id);
    const problems = profileProblems(p);
    const card = document.createElement('div');
    card.className = 'dash-card' + (dashSelMode ? ' selecting' : '') + (selected ? ' selected' : '') + (problems.length ? ' problem' : '');
    card.innerHTML = `
      ${dashSelMode ? '<label class="dash-check"><input type="checkbox"' + (selected ? ' checked' : '') + ' /></label>' : ''}
      <div class="dash-card-top">
        <div class="dash-card-name">${escapeHtml(p.name || t('card_no_name'))}</div>
        <div class="dash-card-badges">
          ${isRunning ? '<span class="dash-badge running">' + t('status_running') + '</span>' : ''}
          ${install ? '<span class="dash-badge version">' + escapeHtml(install.name) + '</span>' : ''}
          ${problems.length ? '<span class="dash-badge alert" title="' + escapeHtml(problems.join(String.fromCharCode(10))) + '">⚠ Hay problemas</span>' : ''}
        </div>
      </div>
      <div class="dash-card-model${model ? '' : ' warn'}">${model ? escapeHtml(model) : t('card_no_model')}</div>
      <div class="dash-card-meta">
        ${p.ctxSize ? '<span class="dash-meta-item">ctx ' + p.ctxSize + '</span>' : ''}
        ${p.gpuLayers != null ? '<span class="dash-meta-item">gpu ' + p.gpuLayers + '</span>' : ''}
        ${p.port != null ? '<span class="dash-meta-item">:' + p.port + '</span>' : ''}
        ${p.mmprojPath ? '<span class="dash-meta-item">visión</span>' : ''}
        ${p.mtp ? '<span class="dash-meta-item">mtp</span>' : ''}
      </div>
      <div class="dash-card-foot">
        ${problems.length
          ? '<span class="dash-open warn-text" title="' + escapeHtml(problems.join(String.fromCharCode(10))) + '">⚠ ' + escapeHtml(problems[0]) + '</span>'
          : '<div class="dash-card-actions">' +
            '<button class="dash-action-btn play" data-action="play" title="' + (isRunning ? t('btn_stop') : t('btn_start')) + '">' + (isRunning ? '⏹' : '▶') + '</button>' +
            (isRunning ? '<button class="dash-action-btn web" data-action="web" title="Abrir en navegador">🌐</button>' : '') +
            '<button class="dash-action-btn" data-action="edit" title="' + t('btn_edit') + '">✎</button>' +
            '</div>'}
      </div>
    `;
    card.addEventListener('click', (e) => {
      if (dashSelMode) {
        if (e.target.closest('.dash-check')) return;
        toggleDashSelected(p.id);
      }
    });
    const playBtn = card.querySelector('[data-action="play"]');
    if (playBtn) {
      playBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        if (isRunning) {
          window.api.stopServer();
          runningProfileId = null;
          setRunning(false);
          toast(t('toast_server_stopped'));
        } else {
          currentId = p.id;
          selectProfile(p.id);
          toggleServer();
        }
      });
    }
    const editBtn = card.querySelector('[data-action="edit"]');
    if (editBtn) {
      editBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        openCommand(p.id);
      });
    }
    const webBtn = card.querySelector('[data-action="web"]');
    if (webBtn) {
      webBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        const host = p.host || '127.0.0.1';
        const port = p.port || 8080;
        window.api.openExternal('http://' + host + ':' + port);
      });
    }
    const cb = card.querySelector('.dash-check input');
    if (cb) {
      cb.addEventListener('change', () => {
        if (cb.checked) dashSelected.add(p.id);
        else dashSelected.delete(p.id);
        updateDashSelBar();
        renderDashboard();
      });
    }
    grid.appendChild(card);
  });
  $('#dashEmpty').classList.toggle('hidden', profiles.length > 0);
}

function enterDashSelection() {
  dashSelMode = true;
  $('#dashSelBtn').classList.add('active');
  $('#dashSelBar').classList.remove('hidden');
  updateDashSelBar();
  renderDashboard();
}

function exitDashSelection() {
  dashSelMode = false;
  dashSelected.clear();
  $('#dashSelBtn').classList.remove('active');
  $('#dashSelBar').classList.add('hidden');
  renderDashboard();
}

function toggleDashSelected(id) {
  if (dashSelected.has(id)) dashSelected.delete(id);
  else dashSelected.add(id);
  updateDashSelBar();
  renderDashboard();
}

function updateDashSelBar() {
  $('#dashSelCount').textContent = dashSelected.size === 1 ? '1 seleccionado' : dashSelected.size + ' seleccionados';
  $('#dashSelAllBtn').textContent = dashSelected.size === profiles.length ? 'Ninguno' : 'Todo';
}

async function deleteSelected() {
  const n = dashSelected.size;
  if (!n) return;
  const ok = await confirmDialog({
    title: t('toast_confirm_delete'),
    message: n === 1 ? t('toast_confirm_delete_one') : t('toast_confirm_delete_n', { n }),
    okLabel: t('confirm_delete'),
    danger: true,
  });
  if (!ok) return;
  profiles = profiles.filter((x) => !dashSelected.has(x.id));
  if (dashSelected.has(currentId)) currentId = null;
  exitDashSelection();
  scheduleSave();
  toast(n === 1 ? t('toast_instance_deleted') : t('toast_instances_deleted', { n }));
  showDashboard();
  refresh();
}

function refresh() {
  renderSidebar();
  renderDashboard();
  syncGeneral();
  scheduleSave();
  applyLanguage();
}

function syncDetail() {
  syncGeneral();
  // re-run schema controls value sync by re-rendering schema panels
  renderSchemaPanels();
}

function buildSchemaControl(field) {
  return buildControl(
    field,
    () => (current() ? current()[field.key] : null),
    (v) => {
      if (current()) {
        const prof = current();
        markConfigured(prof, field.key);
        if (field.key === 'mtp' && v === true) markConfigured(prof, 'specDraftNMax', 'specDraftNMin', 'specDraftPSplit');
        prof[field.key] = v;
        refresh();
        applyFieldVisibility();
        if (mtpDraftKeys.includes(field.key)) renderMtpPresetCard();
        if (field.key === 'cacheTypeK' || field.key === 'cacheTypeV') renderSchemaPanels();
      }
    }
  );
}

function renderMtpPresetCard() {
  if (!mtpCardEl) return;
  const field = RENDERING_FIELDS.find((f) => f.key === 'mtp');
  mtpCardEl.innerHTML = '';
  mtpCardEl.appendChild(buildSchemaControl(field));
}

function renderSchemaPanels() {
  const sampling = $('#panel-sampling');
  const rendering = $('#panel-rendering');
  const server = $('#panel-server');
  sampling.innerHTML = '';
  rendering.innerHTML = '';
  if (server) server.innerHTML = '';
  SAMPLING_FIELDS.forEach((field) => {
    const card = document.createElement('div');
    card.className = 'card';
    card.appendChild(buildSchemaControl(field));
    sampling.appendChild(card);
  });
  if (server) {
    SERVER_FIELDS.forEach((field) => {
      const card = document.createElement('div');
      card.className = 'card';
      card.appendChild(buildSchemaControl(field));
      server.appendChild(card);
    });
  }
  const groups = new Map();
  RENDERING_FIELDS.forEach((field) => {
    const g = field.group || field.key;
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g).push(field);
  });
  groups.forEach((fields, g) => {
    const card = document.createElement('div');
    card.className = 'card';
    if (fields.length === 1) {
      const field = fields[0];
      card.dataset.field = field.key;
      if (field.dependsOn) card.dataset.depends = field.dependsOn;
      if (field.showIfMoe) card.dataset.dependsMoe = '1';
      if (field.key === 'mtp') mtpCardEl = card;
      card.appendChild(buildSchemaControl(field));
    } else {
      fields.forEach((field) => card.appendChild(buildSchemaControl(field)));
      if (g === 'ctx') appendKvPresetRow(card);
      if (g === 'longctx') appendLongCtxPresetRow(card);
      const hintField = fields.find((f) => f.hint);
      card.dataset.field = hintField ? hintField.key : fields[0].key;
    }
    rendering.appendChild(card);
  });
  applyFieldVisibility();
  attachFieldHints();
}

function appendKvPresetRow(card) {
  const p = current();
  const row = document.createElement('div');
  row.className = 'kv-presets';
  row.innerHTML = `
    <div class="field-head"><label>Preset de KV cache</label></div>
    <div class="preset-row kv-preset-row"></div>
    <p class="preset-desc"></p>
  `;
  const btns = row.querySelector('.kv-preset-row');
  const desc = row.querySelector('.preset-desc');
  KV_PRESETS.forEach((preset) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'preset-btn';
    btn.dataset.preset = preset.label;
    btn.textContent = preset.label;
    if (p && p.cacheTypeK === preset.values.cacheTypeK && p.cacheTypeV === preset.values.cacheTypeV) {
      btn.classList.add('active');
      desc.textContent = preset.desc;
    }
    btn.addEventListener('click', () => {
      const prof = current();
      if (!prof) return;
      markConfigured(prof, 'cacheTypeK', 'cacheTypeV');
      prof.cacheTypeK = preset.values.cacheTypeK;
      prof.cacheTypeV = preset.values.cacheTypeV;
      renderSchemaPanels();
      scheduleSave();
      runFieldHints();
    });
    btns.appendChild(btn);
  });
  card.appendChild(row);
}

function appendLongCtxPresetRow(card) {
  const p = current();
  const row = document.createElement('div');
  row.className = 'kv-presets longctx-presets';
  row.innerHTML = `
    <div class="field-head"><label>Presets de contexto largo</label></div>
    <div class="preset-row kv-preset-row"></div>
    <p class="preset-desc"></p>
  `;
  const btns = row.querySelector('.kv-preset-row');
  const desc = row.querySelector('.preset-desc');
  LONGCTX_PRESETS.forEach((preset) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'preset-btn';
    btn.textContent = preset.label;
    if (p && LONGCTX_KEYS.every((k) => p[k] === preset.values[k])) {
      btn.classList.add('active');
      desc.textContent = preset.desc;
    }
    btn.addEventListener('click', () => {
      const prof = current();
      if (!prof) return;
      markConfigured(prof, ...LONGCTX_KEYS);
      LONGCTX_KEYS.forEach((k) => {
        prof[k] = preset.values[k];
      });
      renderSchemaPanels();
      scheduleSave();
    });
    btns.appendChild(btn);
  });
  card.appendChild(row);
}

function applyFieldVisibility() {
  document.querySelectorAll('[data-depends]').forEach((card) => {
    const show = current() ? !!current()[card.dataset.depends] : false;
    card.classList.toggle('hidden', !show);
  });
  document.querySelectorAll('[data-depends-moe]').forEach((card) => {
    const show = currentModelInfo && currentModelInfo.isMoe;
    card.classList.toggle('hidden', !show);
  });
}

/* ---------------- Field hints (VRAM / capas) ---------------- */

let lastModelPath = '';
let currentModelInfo = null;
let gpuLayersHint = null;
let ctxSizeHint = null;
let gpuUsedBytes = 0;
let gpuTotalBytes = 0;

const KV_ELEM_BYTES = {
  f32: 4, bf16: 2, f16: 2, q8_0: 34 / 32, q5_1: 24 / 32, q5_0: 22 / 32, q4_1: 20 / 32, q4_0: 18 / 32, iq4_nl: 18 / 32,
};

function fmtGb(bytes) {
  if (bytes >= 1073741824) {
    const gb = bytes / 1073741824;
    return (gb >= 100 ? Math.round(gb) : Math.round(gb * 10) / 10) + ' GB';
  }
  if (bytes >= 1048576) return Math.round(bytes / 1048576) + ' MB';
  return Math.round(bytes / 1024) + ' KB';
}

async function refreshModelInfo() {
  const p = current();
  const modelPath = p ? p.modelPath : '';
  if (!modelPath) {
    currentModelInfo = null;
    runFieldHints();
    return;
  }
  let info = null;
  try {
    const res = await window.api.inspectModel(modelPath);
    if (res && res.ok) info = res;
  } catch (e) {}
  currentModelInfo = info;
  if (info && info.isMoe && p && p.nCpuMoe == null) {
    p.nCpuMoe = 0;
    const experts = info.nExperts ? ' (' + info.nExperts + ' expertos)' : '';
    toast('Modelo MoE detectado' + experts + '. Configurá n-cpu-moe en el tab Rendimiento.', 'ok');
  }
  runFieldHints();
}

function runFieldHints() {
  Object.keys(FIELD_HINTS).forEach((k) => {
    if (FIELD_HINTS[k]) FIELD_HINTS[k]();
  });
}

function updateGpuLayersHint() {
  if (!gpuLayersHint || !gpuLayersHint.isConnected) return;
  const p = current();
  const info = currentModelInfo;
  if (!info) {
    gpuLayersHint.style.display = 'none';
    return;
  }
  const v = p && p.gpuLayers != null ? p.gpuLayers : 0;
  const est = estimateVram(p, info);
  const avail = gpuTotalBytes > 0 ? Math.max(0, gpuTotalBytes - gpuUsedBytes) : (info.gpuTotalBytes || 0);
  const fit =
    avail > 0 && info.layerBytes > 0 ? Math.max(0, Math.floor((avail - est.kvBytes) / info.layerBytes)) : info.layersFit;
  let html = 'Capas a cargar en GPU: <b>' + v + '</b> de ' + info.totalLayers;
  html += '<br>El modelo tiene <b>' + info.totalLayers + '</b> capas · todo en GPU ≈ <b>' + fmtGb(info.layerBytes * info.totalLayers) + '</b>';
  if (fit != null) {
    if (fit >= info.totalLayers) {
      html += ' · caben <b>todas</b> en ' + fmtGb(avail || info.gpuTotalBytes) + ' de VRAM';
    } else {
      html += ' · caben ~<b>' + fit + '</b> de ' + info.totalLayers + ' capas en ' + fmtGb(avail || info.gpuTotalBytes) + ' de VRAM';
    }
  }
  if (est.kvBytes > 0) html += '<br>Contexto ' + est.nCtx + ' → KV cache ≈ <b>' + fmtGb(est.kvBytes) + '</b>';
  if (v > 0) {
    if (v > info.totalLayers) {
      html += '<br><span class="text-err">La cantidad configurada (' + v + ') supera el total de capas del modelo (' + info.totalLayers + ').</span>';
    } else if (fit != null && v > fit) {
      html += '<br><span class="text-err">No cabe en la VRAM: capas + contexto ≈ <b>' + fmtGb(est.total) + '</b> y hay ' + fmtGb(avail) + ' disponible.</span>';
    } else {
      html += '<br><span class="text-ok">Cabe en la VRAM disponible.</span>';
    }
  }
  gpuLayersHint.innerHTML = html;
  gpuLayersHint.style.display = 'block';

  const card = gpuLayersHint.closest('.card');
  if (card && info) {
    const wrap = card.querySelector('.field[data-field-wrap="gpuLayers"]');
    const range = wrap ? wrap.querySelector('input[type="range"]') : null;
    const number = wrap ? wrap.querySelector('input[type="number"]') : null;
    if (range) {
      const max = Math.max(info.totalLayers, p && p.gpuLayers != null ? p.gpuLayers : 0);
      range.max = max;
      if (number) number.max = max;
    }
  }
}

function kvElemBytes(p) {
  const k = p && p.cacheTypeK ? KV_ELEM_BYTES[p.cacheTypeK] : undefined;
  const v = p && p.cacheTypeV ? KV_ELEM_BYTES[p.cacheTypeV] : undefined;
  return { k: k || 2, v: v || 2 };
}

function estimateVram(p, info) {
  if (!info) return null;
  const gpuLayers = p && p.gpuLayers != null ? p.gpuLayers : 0;
  const weightsBytes = gpuLayers * (info.layerBytes || 0);
  const { k, v } = kvElemBytes(p);
  const nCtx = p && p.ctxSize != null ? p.ctxSize : 0;
  const kvBytes =
    info.headDim != null && nCtx > 0 ? info.totalLayers * nCtx * info.nHeadKv * info.headDim * (k + v) : 0;
  return { weightsBytes, kvBytes, total: weightsBytes + kvBytes, gpuLayers, nCtx };
}

function updateCtxHint() {
  if (!ctxSizeHint || !ctxSizeHint.isConnected) return;
  const p = current();
  const info = currentModelInfo;
  const nCtx = p && p.ctxSize != null ? p.ctxSize : 0;
  if (!info || info.headDim == null || !nCtx) {
    ctxSizeHint.style.display = 'none';
    return;
  }
  const est = estimateVram(p, info);
  const available = gpuTotalBytes > 0 ? Math.max(0, gpuTotalBytes - gpuUsedBytes) : gpuTotalBytes;
  let html = 'KV cache ≈ <b>' + fmtGb(est.kvBytes) + '</b>';
  if (est.weightsBytes > 0) html += ' + capas en GPU ≈ ' + fmtGb(est.weightsBytes);
  html += ' = ≈ <b>' + fmtGb(est.total) + '</b>';
  if (available > 0) {
    html += est.total <= available
      ? '<br><span class="text-ok">Cabe en la VRAM (' + fmtGb(available) + ' disponible).</span>'
      : '<br><span class="text-err">No cabe en la VRAM (' + fmtGb(available) + ' disponible, faltan ≈ ' + fmtGb(est.total - available) + ').</span>';
  }
  ctxSizeHint.innerHTML = html;
  ctxSizeHint.style.display = 'block';
}

const FIELD_HINTS = {
  gpuLayers: updateGpuLayersHint,
  ctxSize: updateCtxHint,
};

function attachFieldHints() {
  gpuLayersHint = null;
  ctxSizeHint = null;
  document.querySelectorAll('#panel-rendering .card[data-field]').forEach((card) => {
    const key = card.dataset.field;
    const field = RENDERING_FIELDS.find((f) => f.key === key);
    if (!field || !field.hint || !FIELD_HINTS[key]) return;
    const hint = document.createElement('div');
    hint.className = 'field-hint';
    card.appendChild(hint);
    if (key === 'gpuLayers') {
      gpuLayersHint = hint;
      const wrap = card.querySelector('.field[data-field-wrap="gpuLayers"]');
      if (wrap && wrap.nextSibling) card.insertBefore(hint, wrap.nextSibling);
    } else if (key === 'ctxSize') {
      ctxSizeHint = hint;
    }
    FIELD_HINTS[key]();
  });
}

/* ---------------- Selection / CRUD ---------------- */

function selectProfile(id) {
  currentId = id;
  renderSidebar();
  renderDashboard();
  syncDetail();
  $('#nameInput').value = current() ? current().name : '';
}

function addProfile() {
  const p = defaultProfile();
  profiles.push(p);
  selectProfile(p.id);
  showEditor();
  $('#nameInput').focus();
}

function duplicateProfile() {
  const p = current();
  if (!p) return;
  const copy = JSON.parse(JSON.stringify(p));
  copy.id = uid();
  copy.name = p.name + ' (copia)';
  profiles.push(copy);
  selectProfile(copy.id);
  scheduleSave();
  toast(t('toast_instance_duplicated'), 'ok');
}

async function deleteProfile() {
  const p = current();
  if (!p) return;
  const ok = await confirmDialog({
    title: t('toast_confirm_delete_name', { name: p.name }),
    message: t('toast_confirm_delete_name', { name: p.name }),
    okLabel: t('confirm_delete'),
    danger: true,
  });
  if (!ok) return;
  profiles = profiles.filter((x) => x.id !== p.id);
  currentId = null;
  scheduleSave();
  toast(t('toast_instance_deleted'));
  showDashboard();
  refresh();
}

/* ---------------- Models ---------------- */

function relativeTo(p, dir) {
  if (!dir) return p;
  const d = dir.replace(/[\\/]+$/, '');
  if (p.startsWith(d + '\\') || p.startsWith(d + '/')) return p.slice(d.length + 1);
  return p;
}

function modelKind(filePath) {
  const name = String(filePath || '').split(/[\\/]/).pop().toLowerCase();
  if (/mmproj|vision-?proj|clip|image/i.test(name)) return 'vision';
  if (/mtp|draft/i.test(name)) return 'mtp';
  if (/embed/i.test(name)) return 'embed';
  return 'main';
}

function modelLabel(p) {
  const d = settings.modelsDir;
  if (d) {
    const rel = relativeTo(p, d);
    if (rel !== p) return rel;
  }
  return p;
}

function populateModelSelects() {
  const modelSel = $('#modelSelect');
  const visionSel = $('#visionSelect');
  const mains = modelsCache.filter((m) => modelKind(m) === 'main');
  const visions = modelsCache.filter((m) => modelKind(m) === 'vision');
  modelSel.innerHTML = '<option value="">— Elegir modelo —</option>' + mains
    .map((m) => `<option value="${escapeHtml(m)}">${escapeHtml(modelLabel(m))}</option>`)
    .join('');
  visionSel.innerHTML = '<option value="">— Elegir archivo —</option>' + visions
    .map((m) => `<option value="${escapeHtml(m)}">${escapeHtml(modelLabel(m))}</option>`)
    .join('');
}

async function scanAllModels() {
  const dirs = [];
  if (settings.modelsDir) dirs.push(settings.modelsDir);
  const lists = await Promise.all(dirs.map((d) => window.api.scanModels(d)));
  const seen = new Set();
  const out = [];
  lists.forEach((list) => {
    (list || []).forEach((m) => {
      if (!seen.has(m)) {
        seen.add(m);
        out.push(m);
      }
    });
  });
  return out;
}

async function scanModels() {
  if (!settings.modelsDir) {
    toast(t('toast_config_models_dir_first'), 'err');
    openSettings();
    return;
  }
  modelsCache = await scanAllModels();
  populateModelSelects();
  toast(modelsCache.length ? `Se encontraron ${modelsCache.length} modelos` : 'No se encontraron .gguf en esas carpetas', modelsCache.length ? 'ok' : 'err');
}

/* ---------------- Instance board ---------------- */

const BOARD_CONTEXT_KEYS = ['ctxSize', 'cacheTypeK', 'cacheTypeV'];
const BOARD_GPU_KEYS = ['gpuLayers', 'gpuLayersAll', 'mainGpu', 'tensorSplit', 'device'];
const BOARD_MTP_KEYS = ['mtp', 'specDraftModel', 'specDraftNMax', 'specDraftNMin', 'specDraftPSplit'];

function boardHasAnyConfigured(p, keys) {
  return !!(p && Array.isArray(p.configured) && keys.some((k) => p.configured.includes(k)));
}

function boardOptionHtml(items, emptyLabel) {
  return '<option value="">' + escapeHtml(emptyLabel) + '</option>' + items
    .map((m) => `<option value="${escapeHtml(m)}">${escapeHtml(modelLabel(m))}</option>`)
    .join('');
}

function boardIcon(type) {
  const icons = {
    model: '<svg viewBox="0 0 24 24"><path d="M12 3 3 7.5l9 4.5 9-4.5L12 3Z"/><path d="M3 12l9 4.5 9-4.5"/><path d="M3 16.5 12 21l9-4.5"/></svg>',
    vision: '<svg viewBox="0 0 24 24"><path d="M2.5 12s3.5-6 9.5-6 9.5 6 9.5 6-3.5 6-9.5 6-9.5-6-9.5-6Z"/><circle cx="12" cy="12" r="3"/></svg>',
    mtp: '<svg viewBox="0 0 24 24"><path d="m4 5 8 7-8 7V5Z"/><path d="m12 5 8 7-8 7V5Z"/></svg>',
    context: '<svg viewBox="0 0 24 24"><rect x="4" y="3" width="16" height="18" rx="2"/><path d="M8 8h8M8 12h8M8 16h5"/></svg>',
    gpu: '<svg viewBox="0 0 24 24"><rect x="7" y="7" width="10" height="10" rx="2"/><path d="M9 1v4M15 1v4M9 19v4M15 19v4M1 9h4M1 15h4M19 9h4M19 15h4"/></svg>',
  };
  return icons[type] || icons.model;
}

function boardHydrateIcons(root) {
  (root || document).querySelectorAll('[data-icon]').forEach((el) => {
    el.innerHTML = boardIcon(el.dataset.icon);
  });
}

function boardRelatedModelFile(kind) {
  const p = current();
  if (!p || !p.modelPath) return '';
  const dir = String(p.modelPath).replace(/[\\/][^\\/]+$/, '').toLowerCase();
  const candidates = modelsCache.filter((m) => modelKind(m) === kind && String(m).replace(/[\\/][^\\/]+$/, '').toLowerCase() === dir);
  return candidates.length === 1 ? candidates[0] : '';
}

function openBoardForNewProfile() {
  const p = defaultProfile();
  const installs = settings.installations || [];
  if (installs.length) p.installId = installs[0].id;
  profiles.push(p);
  selectProfile(p.id);
  showBoard();
  $('#boardNameInput').focus();
  scheduleSave();
}

function renderBoardPalette() {
  const p = current();
  const setComponentVisible = (type, visible) => {
    const el = document.querySelector(`.board-palette-card[data-board-type="${type}"]`);
    if (el) el.style.display = visible ? '' : 'none';
  };
  setComponentVisible('model', !(p && p.modelPath));
  setComponentVisible('vision', !(p && p.mmprojPath));
  setComponentVisible('mtp', !(p && p.mtp));
  setComponentVisible('context', !(p && boardHasAnyConfigured(p, BOARD_CONTEXT_KEYS)));
  setComponentVisible('gpu', !(p && boardHasAnyConfigured(p, BOARD_GPU_KEYS)));
  const visibleComponents = ['model', 'vision', 'mtp', 'context', 'gpu'].some((type) => {
    const el = document.querySelector(`.board-palette-card[data-board-type="${type}"]`);
    return el && el.style.display !== 'none';
  });
  $('#boardComponentsEmpty').classList.toggle('hidden', visibleComponents);
  boardHydrateIcons($('#boardView'));
}

function boardBlock(type, title, detail, configured) {
  return `
    <div class="board-block ${configured ? '' : 'muted'}" data-board-block="${escapeHtml(type)}" draggable="${configured ? 'true' : 'false'}">
      <div class="board-block-main">
        <div class="board-block-icon">${boardIcon(type)}</div>
        <div class="board-block-copy">
          <span class="board-block-kicker">${configured ? 'Asignado' : 'Sugerido'}</span>
          <b>${escapeHtml(title)}</b>
          <span>${escapeHtml(detail)}</span>
        </div>
      </div>
      <div class="board-block-actions">
        <button class="btn small ghost" data-board-edit="${escapeHtml(type)}">Editar</button>
      </div>
    </div>`;
}

function renderBoard() {
  renderBoardPalette();
  const p = current();
  if (!p) return;
  const nameInput = $('#boardNameInput');
  if (nameInput) nameInput.value = p.name || '';
  const blocks = [];
  if (p.modelPath) blocks.push(boardBlock('model', 'Modelo', modelLabel(p.modelPath), true));
  if (p.mmprojPath) blocks.push(boardBlock('vision', 'Visión', modelLabel(p.mmprojPath), true));
  if (p.mtp) blocks.push(boardBlock('mtp', 'MTP', p.specDraftModel ? modelLabel(p.specDraftModel) : 'MTP incluido en el modelo', true));
  if (boardHasAnyConfigured(p, BOARD_CONTEXT_KEYS)) blocks.push(boardBlock('context', 'Contexto', `${p.ctxSize || 0} tokens · KV ${String(p.cacheTypeK || 'f16').toUpperCase()}/${String(p.cacheTypeV || 'f16').toUpperCase()}`, true));
  if (boardHasAnyConfigured(p, BOARD_GPU_KEYS)) {
    const gpuDetail = p.gpuLayersAll ? 'Todas las capas en GPU disponible' : `${p.gpuLayers || 0} capas`;
    blocks.push(boardBlock('gpu', 'GPU', gpuDetail + (p.mainGpu !== null && p.mainGpu !== undefined ? ` · principal ${p.mainGpu}` : ''), true));
  }
  $('#boardAssigned').innerHTML = blocks.join('');
  boardHydrateIcons($('#boardView'));
  $('#boardEmpty').style.display = blocks.length ? 'none' : '';
  const problems = profileProblems(p);
  const status = $('#boardStatus');
  status.textContent = problems.length ? 'Incompleta' : 'Lista';
  status.className = 'board-status ' + (problems.length ? 'warn' : 'ok');
  $('#boardValidation').textContent = problems.length ? problems[0] : 'Lista para ejecutar.';
  $('#boardCommand').textContent = buildCommandLine(p);
  $('#boardRunBtn').disabled = problems.length > 0 && !running;
  $('#boardRunBtn').textContent = running ? t('btn_stop') : t('btn_start');
  $('#boardRunBtn').className = 'btn run-btn ' + (running ? 'running' : 'stopped');
}

function boardCloseConfig() {
  boardConfig = null;
  $('#boardConfigModal').classList.add('hidden');
}

function boardApplyAndRefresh(applyFn) {
  const p = current();
  if (!p) return;
  applyFn(p);
  scheduleSave();
  syncDetail();
  renderBoard();
}

async function boardModelInfoForContext(p) {
  if (!p || !p.modelPath) return null;
  if (currentModelInfo && lastModelPath === p.modelPath) return currentModelInfo;
  let info = null;
  try {
    const res = await window.api.inspectModel(p.modelPath);
    if (res && res.ok) info = res;
  } catch (e) {}
  currentModelInfo = info;
  lastModelPath = p.modelPath;
  return info;
}

function boardUpdateContextInfo(info) {
  const ctxInput = $('#boardCfgCtx');
  const activeKv = document.querySelector('#boardCfgKvCards .board-kv-card.active');
  const out = $('#boardCfgCtxInfo');
  if (!ctxInput || !out) return;
  const ctx = num(ctxInput.value) || 0;
  const kvIdx = activeKv ? Number(activeKv.dataset.kvIndex) : 0;
  const preset = KV_PRESETS[kvIdx] || KV_PRESETS[0];
  if (!info || !info.headDim || !ctx) {
    out.innerHTML = 'Preset de contexto: <b>' + ctx + '</b> tokens.';
    return;
  }
  const p = Object.assign(defaultProfile(), current() || {}, {
    ctxSize: ctx,
    cacheTypeK: preset.values.cacheTypeK,
    cacheTypeV: preset.values.cacheTypeV,
  });
  const est = estimateVram(p, info);
  const rec = info.contextLength ? ' · recomendado por el modelo: <b>' + info.contextLength + '</b>' : '';
  out.innerHTML = 'Preset de contexto: <b>' + ctx + '</b> tokens' + rec + ' · KV cache ≈ <b>' + fmtGb(est.kvBytes) + '</b> (con KV ' +
    p.cacheTypeK.toUpperCase() + '/' + p.cacheTypeV.toUpperCase() + ').';
}

async function boardOpenConfig(type, payload) {
  const p = current();
  if (!p) return;
  const visions = modelsCache.filter((m) => modelKind(m) === 'vision');
  const mtps = modelsCache.filter((m) => modelKind(m) === 'mtp');
  const installs = settings.installations || [];
  if (type === 'vision') {
    const related = boardRelatedModelFile('vision');
    if (related) {
      boardApplyAndRefresh((prof) => {
        prof.mmprojPath = related;
        markConfigured(prof, 'mmprojPath');
      });
      toast('Visión agregada: ' + modelLabel(related), 'ok');
      return;
    }
  }
  boardConfig = { type, payload: payload || {} };
  $('#boardConfigTitle').textContent = 'Configurar ' + ({ model: 'modelo', vision: 'visión', mtp: 'MTP', context: 'contexto', gpu: 'GPU' }[type] || 'objeto');
  const body = $('#boardConfigBody');
  if (type === 'model') {
    const mains = modelsCache.filter((m) => modelKind(m) === 'main');
    const modelPath = payload && payload.path ? payload.path : p.modelPath;
    body.innerHTML = `
      <div class="field"><div class="field-head"><label>Versión de llama.cpp</label></div><select id="boardCfgInstall" class="text-input">${'<option value="">— Elegir versión —</option>' + installs.map((i) => `<option value="${escapeHtml(i.id)}">${escapeHtml(installLabel(i))}</option>`).join('')}</select></div>
      <div class="field"><div class="field-head"><label>Modelo descargado</label></div><select id="boardCfgModel" class="text-input">${boardOptionHtml(mains, '— Elegir modelo GGUF —')}</select></div>`;
    $('#boardCfgInstall').value = p.installId || (installs[0] && installs[0].id) || '';
    $('#boardCfgModel').value = modelPath || '';
  } else if (type === 'vision') {
    const related = boardRelatedModelFile('vision');
    if (related) {
      body.innerHTML = `<div class="field"><div class="field-head"><label>Archivo MMProj detectado</label></div><div class="board-picked-file">${escapeHtml(modelLabel(related))}</div><input id="boardCfgVisionFile" type="hidden" value="${escapeHtml(related)}" /></div>`;
    } else {
      body.innerHTML = `<div class="field"><div class="field-head"><label>Archivo MMProj</label></div><select id="boardCfgVisionFile" class="text-input">${boardOptionHtml(visions, '— Elegir archivo MMProj —')}</select></div>`;
      $('#boardCfgVisionFile').value = p.mmprojPath || '';
    }
  } else if (type === 'mtp') {
    const related = boardRelatedModelFile('mtp');
    body.innerHTML = `
      ${related ? `<div class="field"><div class="field-head"><label>Archivo MTP detectado</label></div><div class="board-picked-file">${escapeHtml(modelLabel(related))}</div><input id="boardCfgMtpFile" type="hidden" value="${escapeHtml(related)}" /></div>` : `<label class="check-field"><input id="boardCfgMtpBuiltin" type="checkbox" ${p.specDraftModel ? '' : 'checked'} /><span>Usar el MTP incluido en el modelo</span></label><div class="field"><div class="field-head"><label>Archivo MTP separado</label></div><select id="boardCfgMtpFile" class="text-input">${boardOptionHtml(mtps, '— Elegir archivo MTP —')}</select></div>`}
      <div class="field"><div class="field-head"><label>Tokens a predecir</label></div><input id="boardCfgMtpNMax" type="number" class="text-input" min="1" max="32" value="${escapeHtml(p.specDraftNMax || 3)}" /></div>
      <div class="field"><div class="field-head"><label>Tokens mínimos</label></div><input id="boardCfgMtpNMin" type="number" class="text-input" min="0" max="32" value="${escapeHtml(p.specDraftNMin || 0)}" /></div>
      <div class="field"><div class="field-head"><label>Probabilidad de split</label></div><input id="boardCfgMtpPSplit" type="number" class="text-input" min="0" max="1" step="0.05" value="${escapeHtml(p.specDraftPSplit || 0.1)}" /></div>`;
    if (!related) $('#boardCfgMtpFile').value = p.specDraftModel || '';
  } else if (type === 'context') {
    const info = await boardModelInfoForContext(p);
    const ctxValue = !boardHasAnyConfigured(p, BOARD_CONTEXT_KEYS) && info && info.contextLength ? info.contextLength : (p.ctxSize || 4096);
    body.innerHTML = `
      <div class="field"><div class="field-head"><label>Contexto (ctx-size)</label></div>
        <div class="preset-row board-ctx-presets">${WIZARD_CTX_PRESETS.map((v) => `<button class="preset-btn" type="button" data-board-ctx="${v}">${v >= 1024 ? v / 1024 + 'K' : v}</button>`).join('')}</div>
        <div class="pickrow"><input id="boardCfgCtx" type="number" class="text-input" min="64" max="1000000" value="${escapeHtml(ctxValue)}" /><span class="hint">tokens</span></div>
        <p id="boardCfgCtxInfo" class="hint"></p>
      </div>
      <div class="field"><div class="field-head"><label>Cuantización del contexto (KV cache)</label></div>
        <div id="boardCfgKvCards" class="board-kv-grid">${KV_PRESETS.map((x, i) => `<button type="button" class="board-kv-card" data-kv-index="${i}"><b>${escapeHtml(x.label)}</b><span>${escapeHtml(x.values.cacheTypeK.toUpperCase() + '/' + x.values.cacheTypeV.toUpperCase())}</span><small>${escapeHtml(x.desc)}</small></button>`).join('')}</div>
      </div>`;
    const kvIdx = Math.max(0, KV_PRESETS.findIndex((x) => x.values.cacheTypeK === p.cacheTypeK && x.values.cacheTypeV === p.cacheTypeV));
    const selectKv = (idx) => {
      body.querySelectorAll('.board-kv-card').forEach((b) => b.classList.toggle('active', Number(b.dataset.kvIndex) === idx));
      boardUpdateContextInfo(info);
    };
    selectKv(kvIdx);
    body.querySelectorAll('[data-board-ctx]').forEach((b) => b.addEventListener('click', () => {
      $('#boardCfgCtx').value = b.dataset.boardCtx;
      body.querySelectorAll('[data-board-ctx]').forEach((x) => x.classList.toggle('active', x === b));
      boardUpdateContextInfo(info);
    }));
    body.querySelectorAll('.board-kv-card').forEach((b) => b.addEventListener('click', () => selectKv(Number(b.dataset.kvIndex))));
    $('#boardCfgCtx').addEventListener('input', () => boardUpdateContextInfo(info));
    body.querySelectorAll('[data-board-ctx]').forEach((b) => b.classList.toggle('active', Number(b.dataset.boardCtx) === Number(ctxValue)));
    boardUpdateContextInfo(info);
  } else if (type === 'gpu') {
    body.innerHTML = `
      <label class="check-field"><input id="boardCfgGpuAll" type="checkbox" ${p.gpuLayersAll ? 'checked' : ''} /><span>Cargar todas las capas posibles en GPU</span></label>
      <div class="field"><div class="field-head"><label>Capas en GPU</label></div><input id="boardCfgGpuLayers" type="number" class="text-input" min="0" max="999" value="${escapeHtml(p.gpuLayers || 0)}" /></div>
      <div class="field"><div class="field-head"><label>GPU principal</label></div><input id="boardCfgMainGpu" type="number" class="text-input" min="0" max="16" placeholder="Automática" value="${p.mainGpu === null || p.mainGpu === undefined ? '' : escapeHtml(p.mainGpu)}" /></div>
      <div class="field"><div class="field-head"><label>Tensor split</label></div><input id="boardCfgTensorSplit" class="text-input" placeholder="Ej: 1,1" value="${escapeHtml(p.tensorSplit || '')}" /></div>
      <div class="field"><div class="field-head"><label>Device</label></div><input id="boardCfgDevice" class="text-input" placeholder="Ej: CUDA0" value="${escapeHtml(p.device || '')}" /></div>`;
  }
  $('#boardConfigModal').classList.remove('hidden');
}

function boardApplyConfig() {
  if (!boardConfig) return;
  const type = boardConfig.type;
  const payload = boardConfig.payload || {};
  if (type === 'model' && !(payload.path || $('#boardCfgModel').value)) {
    toast('Elegí un modelo descargado', 'err');
    return;
  }
  boardApplyAndRefresh((p) => {
    if (type === 'model') {
      const modelPath = payload.path || $('#boardCfgModel').value || p.modelPath;
      p.installId = $('#boardCfgInstall').value || p.installId;
      p.modelPath = modelPath;
      if ((!p.name || p.name === 'Nueva instancia') && modelPath) p.name = modelPath.split(/[\\/]/).pop().replace(/\.gguf$/i, '');
      markConfigured(p, 'modelPath');
    } else if (type === 'vision') {
      p.mmprojPath = $('#boardCfgVisionFile').value || '';
      if (p.mmprojPath) markConfigured(p, 'mmprojPath');
    } else if (type === 'mtp') {
      p.mtp = true;
      const builtin = $('#boardCfgMtpBuiltin');
      p.specDraftModel = builtin && builtin.checked ? '' : ($('#boardCfgMtpFile').value || '');
      p.specDraftNMax = num($('#boardCfgMtpNMax').value) || 3;
      p.specDraftNMin = num($('#boardCfgMtpNMin').value) || 0;
      p.specDraftPSplit = num($('#boardCfgMtpPSplit').value);
      if (p.specDraftPSplit === null) p.specDraftPSplit = 0.1;
      markConfigured(p, ...BOARD_MTP_KEYS);
    } else if (type === 'context') {
      const ctx = num($('#boardCfgCtx').value);
      p.ctxSize = ctx !== null && ctx >= 64 ? ctx : 4096;
      const activeKv = document.querySelector('#boardCfgKvCards .board-kv-card.active');
      const preset = KV_PRESETS[activeKv ? Number(activeKv.dataset.kvIndex) : 0] || KV_PRESETS[0];
      p.cacheTypeK = preset.values.cacheTypeK;
      p.cacheTypeV = preset.values.cacheTypeV;
      markConfigured(p, ...BOARD_CONTEXT_KEYS);
    } else if (type === 'gpu') {
      p.gpuLayersAll = $('#boardCfgGpuAll').checked;
      p.gpuLayers = num($('#boardCfgGpuLayers').value) || 0;
      p.mainGpu = num($('#boardCfgMainGpu').value);
      p.tensorSplit = $('#boardCfgTensorSplit').value.trim();
      p.device = $('#boardCfgDevice').value.trim();
      markConfigured(p, ...BOARD_GPU_KEYS);
    }
  });
  boardCloseConfig();
}

function boardRemoveBlock(type) {
  boardApplyAndRefresh((p) => {
    if (type === 'model') {
      p.modelPath = '';
      unmarkConfigured(p, 'modelPath');
    } else if (type === 'vision') {
      p.mmprojPath = '';
      unmarkConfigured(p, 'mmprojPath');
    } else if (type === 'mtp') {
      p.mtp = false;
      p.specDraftModel = '';
      unmarkConfigured(p, ...BOARD_MTP_KEYS);
    } else if (type === 'context') {
      unmarkConfigured(p, ...BOARD_CONTEXT_KEYS);
    } else if (type === 'gpu') {
      unmarkConfigured(p, ...BOARD_GPU_KEYS);
    }
  });
}

/* ---------------- Settings ---------------- */

let editingInstallId = null;

function renderInstallList() {
  const list = $('#installList');
  const installs = settings.installations || [];
  if (!installs.length) {
    list.innerHTML = '<div class="install-empty">Todavía no hay versiones registradas.</div>';
    return;
  }
  list.innerHTML = '';
  installs.forEach((i) => {
    const item = document.createElement('div');
    item.className = 'install-item';
    item.innerHTML = `
      <div class="install-info">
        <div class="install-name">${escapeHtml(i.name || 'Sin nombre')}</div>
        <div class="install-path" title="${escapeHtml(i.path || '')}">${escapeHtml(i.path || '')}</div>
      </div>
      ${i.exePath ? '<span class="install-ok">OK</span>' : '<span class="text-err">sin llama-server.exe</span>'}
      <div class="install-actions">
        <button class="btn small" data-action="edit">Editar</button>
        <button class="btn small ghost danger" data-action="delete">Eliminar</button>
      </div>
    `;
    item.querySelector('[data-action="edit"]').addEventListener('click', () => openInstallEditor(i.id));
    item.querySelector('[data-action="delete"]').addEventListener('click', () => deleteInstall(i.id));
    list.appendChild(item);
  });
}

function openInstallEditor(id) {
  editingInstallId = id || null;
  const i = id ? installationById(id) : null;
  $('#installModalTitle').textContent = i ? 'Editar versión' : 'Nueva versión';
  $('#installName').value = i ? i.name : '';
  $('#installPath').value = i ? i.path : '';
  $('#installPathHint').textContent = '';
  $('#installModal').classList.remove('hidden');
  $('#installName').focus();
}

function closeInstallEditor() {
  $('#installModal').classList.add('hidden');
  editingInstallId = null;
}

async function deleteInstall(id) {
  const i = installationById(id);
  if (!i) return;
  const msg =
    'También se borrará la carpeta del disco:\n' + i.path;
  const ok = await confirmDialog({
    title: t('toast_confirm_delete_version', { name: i.name }),
    message: t('toast_confirm_delete_version', { name: i.name }) + '\n\n' + i.path,
    okLabel: t('confirm_delete'),
    danger: true,
  });
  if (!ok) return;
  const res = await window.api.deleteInstall(id);
  if (!res || !res.ok) {
    toast((res && res.error) || t('toast_version_delete_failed'), 'err');
    return;
  }
  settings.installations = (settings.installations || []).filter((x) => x.id !== id);
  try {
    await window.api.saveSettings(settings);
  } catch (e) {}
  renderInstallList();
  refresh();
  scheduleFileCheck();
    toast(t('toast_version_deleted'), 'ok');
}

function setSettingsTab(tab) {
  document.querySelectorAll('#modal .settings-tab').forEach((b) => b.classList.toggle('active', b.dataset.settingsTab === tab));
  document.querySelectorAll('#modal .settings-pane').forEach((p) => p.classList.toggle('hidden', p.dataset.pane !== tab));
  if (tab === 'modelos') renderDownloadedModels();
}

function renderDownloadedModels() {
  const box = $('#downloadedModels');
  if (!box) return;
  const byDir = {};
  modelsCache.forEach((m) => {
    const dir = m.split(/[\\/]/).slice(0, -1).join('/');
    (byDir[dir] = byDir[dir] || []).push(m);
  });
  const dirs = Object.keys(byDir).sort();
  if (!dirs.length) {
    box.innerHTML = '<div class="install-empty">Todavía no hay modelos en la carpeta de modelos.</div>';
    return;
  }
  box.innerHTML = '';
  dirs.forEach((dir) => {
    const files = byDir[dir].sort();
    const base = dir.split(/[\\/]/).filter(Boolean).pop() || dir;
    const repoUrl = files.map((f) => settings.hfRepoMap && settings.hfRepoMap[f]).filter(Boolean)[0] || '';
    const item = document.createElement('div');
    item.className = 'install-item';
    item.innerHTML =
      '<div class="install-info">' +
      '<div class="install-name">' + escapeHtml(base) + '</div>' +
      '<div class="install-path" title="' + escapeHtml(dir) + '">' + files.length + ' archivo(s) · ' + escapeHtml(dir) + '</div>' +
      (repoUrl
        ? '<div class="install-path" title="' + escapeHtml(repoUrl) + '">' + escapeHtml(repoUrl.replace(/^https?:\/\//i, '')) + '</div>'
        : '') +
      '</div>' +
      '<div class="install-actions"><button class="btn small" data-action="use">Usar</button>' +
      '<button class="btn small danger" data-action="delete">Eliminar</button></div>';
    item.querySelector('[data-action="use"]').addEventListener('click', () => useDownloadedModel(dir, files));
    item.querySelector('[data-action="delete"]').addEventListener('click', () => deleteModelFolder(dir));
    box.appendChild(item);
  });
}

async function deleteModelFolder(dir) {
  const ok = await confirmDialog({
    title: t('toast_confirm_delete_model'),
    message: t('toast_confirm_delete_model') + '\n\n' + dir,
    okLabel: t('confirm_delete'),
    danger: true,
  });
  if (!ok) return;
  const res = await window.api.deleteModelsFolder(dir);
  if (!res || !res.ok) {
    toast((res && res.error) || t('toast_model_delete_failed'), 'err');
    return;
  }
  const dp = String(dir).replace(/\\/g, '/').toLowerCase().replace(/\/+$/, '');
  modelsCache = modelsCache.filter((m) => !String(m).replace(/\\/g, '/').toLowerCase().startsWith(dp + '/'));
  if (settings.hfRepoMap && typeof settings.hfRepoMap === 'object') {
    Object.keys(settings.hfRepoMap).forEach((k) => {
      if (String(k).replace(/\\/g, '/').toLowerCase().startsWith(dp + '/')) delete settings.hfRepoMap[k];
    });
    try { await window.api.saveSettings(settings); } catch (e) {}
  }
  renderDownloadedModels();
  populateModelSelects();
  refresh();
  scheduleFileCheck();
  toast(res.result === 'missing' ? 'La carpeta ya no existía' : 'Modelo eliminado', 'ok');
}

function pickMainLocal(paths) {
  const main = (paths || []).filter(
    (p) => !/mmproj|vision-?proj|clip|image|mtp|draft|embed/i.test(p.split(/[\\/]/).pop())
  );
  if (!main.length) return null;
  main.sort((a, b) => {
    const rank = (s) => {
      const n = s.toLowerCase();
      if (n.includes('q4_k_m')) return 0;
      if (n.includes('q4_0')) return 1;
      if (n.includes('q5')) return 2;
      if (n.includes('q6')) return 3;
      if (n.includes('q8')) return 4;
      if (n.includes('bf16')) return 5;
      if (n.includes('f16')) return 6;
      return 7;
    };
    return rank(a) - rank(b) || a.localeCompare(b);
  });
  return main[0];
}

function useDownloadedModel(dir, files) {
  const p = current();
  if (!p) {
    toast(t('toast_select_instance'), 'err');
    return;
  }
  const main = pickMainLocal(files);
  if (!main) {
    toast('No hay un archivo .gguf principal en esa carpeta', 'err');
    return;
  }
  const vision = files.find((f) => /mmproj|vision-?proj/i.test(f.split(/[\\/]/).pop())) || '';
  const mtpFile = files.find((f) => /mtp|draft/i.test(f.split(/[\\/]/).pop())) || '';
  p.modelPath = main;
  markConfigured(p, 'modelPath');
  p.mmprojPath = vision;
  if (vision) markConfigured(p, 'mmprojPath');
  if (mtpFile) {
    p.mtp = true;
    p.specDraftModel = mtpFile;
    markConfigured(p, 'mtp', 'specDraftModel', 'specDraftNMax', 'specDraftNMin', 'specDraftPSplit');
  } else {
    p.mtp = false;
    p.specDraftModel = '';
  }
  scheduleSave();
  refresh();
  closeSettings();
  showEditor();
    toast(t('toast_model_applied', { name: p.name }), 'ok');
}

function openSettings(tab) {
  const target = typeof tab === 'string' ? tab : 'general';
  $('#modelsDir').value = settings.modelsDir || '';
  $('#langSelect').value = settings.lang || 'es';
  renderInstallList();
  renderThemeGrid();
  setSettingsTab(target);
  renderDownloadedModels();
  $('#modal').classList.remove('hidden');
}

function closeSettings() {
  $('#modal').classList.add('hidden');
}

/* ---------------- Themes ---------------- */

const THEMES = [
  { id: 'noche', name: 'Noche', tag: 'oscuro', dark: true, colors: ['#0e1116', '#161b23', '#7c5cff'] },
  { id: 'dracula', name: 'Dracula', tag: 'oscuro', dark: true, colors: ['#282a36', '#343746', '#bd93f9'] },
  { id: 'nord', name: 'Nord', tag: 'oscuro', dark: true, colors: ['#2e3440', '#3b4252', '#88c0d0'] },
  { id: 'monokai', name: 'Monokai', tag: 'oscuro', dark: true, colors: ['#272822', '#3e3d32', '#a6e22e'] },
  { id: 'gruvbox', name: 'Gruvbox', tag: 'oscuro', dark: true, colors: ['#282828', '#3c3836', '#d65d0e'] },
  { id: 'onedark', name: 'One Dark', tag: 'oscuro', dark: true, colors: ['#282c34', '#333842', '#61afef'] },
  { id: 'solarized', name: 'Solarized', tag: 'claro', dark: false, colors: ['#fdf6e3', '#eee8d5', '#268bd2'] },
  { id: 'github', name: 'GitHub', tag: 'claro', dark: false, colors: ['#f6f8fa', '#ffffff', '#0969da'] },
  { id: 'gruvbox-light', name: 'Gruvbox Light', tag: 'claro', dark: false, colors: ['#fbf1c7', '#f9f1cc', '#cc241d'] },
  { id: 'rosepine', name: 'Rosé Pine', tag: 'claro', dark: false, colors: ['#faf4ed', '#fffaf3', '#907aa9'] },
  { id: 'catppuccin', name: 'Catppuccin', tag: 'claro', dark: false, colors: ['#eff1f5', '#ccd0da', '#8839ef'] },
];

function applyTheme(id) {
  const t = THEMES.find((x) => x.id === id);
  document.documentElement.setAttribute('data-theme', t ? t.id : 'noche');
}

function renderThemeGrid() {
  const box = $('#themeGrid');
  if (!box) return;
  const current = settings.theme || 'noche';
  box.innerHTML = '';
  THEMES.forEach((t) => {
    const card = document.createElement('div');
    card.className = 'theme-card' + (t.id === current ? ' active' : '');
    card.dataset.themeId = t.id;
    card.innerHTML =
      '<div class="theme-card-preview">' +
      t.colors.map((c) => '<div class="theme-card-color" style="background:' + c + '"></div>').join('') +
      '</div>' +
      '<div class="theme-card-name">' + t.name + '</div>' +
      '<div class="theme-card-tag">' + t.tag + '</div>';
    card.addEventListener('click', () => {
      box.querySelectorAll('.theme-card').forEach((c) => c.classList.remove('active'));
      card.classList.add('active');
      settings.theme = t.id;
      applyTheme(t.id);
    });
    box.appendChild(card);
  });
}

/* ---------------- Server run ---------------- */

async function toggleServer() {
  if (running) {
    await window.api.stopServer();
    runningProfileId = null;
    setRunning(false);
    toast(t('toast_server_stopped'));
    return;
  }
  const p = current();
  if (!p) {
    toast(t('toast_no_instance'), 'err');
    return;
  }
  if (!p.modelPath) {
    toast(t('toast_no_model'), 'err');
    return;
  }
  const install = effectiveInstall();
  if (!install || !install.exePath) {
    toast(t('toast_no_install'), 'err');
    openSettings();
    return;
  }
  const problems = profileProblems(p);
  if (problems.length) {
    toast(t('toast_no_instance') + ': ' + problems[0], 'err');
    return;
  }
  appendLog('\n→ Iniciando llama-server...\n', 'sys');
  expandBottomDock();
  const res = await window.api.startServer({ exe: install.exePath, args: buildArgs(p) });
  if (!res.ok) {
    toast(res.error || t('toast_no_instance'), 'err');
    appendLog('\n[error] ' + res.error + '\n', 'err');
    return;
  }
  runningProfileId = p.id;
  setRunning(true);
  toast(t('toast_server_started'), 'ok');
}

function appendLog(text, kind) {
  updateSpeed(text);
  const el = $('#log');
  const div = document.createElement('div');
  div.className = 'l-' + (kind || 'out');
  div.textContent = text;
  el.appendChild(div);
  while (el.childNodes.length > 3000) el.removeChild(el.firstChild);
  el.scrollTop = el.scrollHeight;
}

let speedBuffer = '';

function updateSpeed(text) {
  if (!text) return;
  speedBuffer = (speedBuffer + String(text)).slice(-4000);
  let m = speedBuffer.match(/tg_3s\s*=\s*([\d.]+)\s*t\/s/);
  if (!m) m = speedBuffer.match(/tg\s*=\s*([\d.]+)\s*t\/s/);
  if (!m) m = speedBuffer.match(/([\d.]+)\s+tokens? per second/);
  if (!m) return;
  const tps = parseFloat(m[1]);
  if (!Number.isFinite(tps)) return;
  const badge = $('#speedBadge');
  if (!badge) return;
  badge.textContent = tps.toFixed(1) + ' tok/s';
  badge.classList.add('live');
}

function resetSpeed() {
  speedBuffer = '';
  const badge = $('#speedBadge');
  if (badge) {
    badge.textContent = '— tok/s';
    badge.classList.remove('live');
  }
}

function showBottomTab(name) {
  document.querySelectorAll('.stats-head .tab').forEach((t) => t.classList.toggle('active', t.dataset.bottom === name));
  $('#bottomRecursos').classList.toggle('active', name === 'recursos');
  $('#bottomConsola').classList.toggle('active', name === 'consola');
}

function expandBottomDock() {
  $('#statsBar').classList.remove('collapsed');
}

/* ---------------- Detección de argumentos inválidos ---------------- */

function fieldPanelOf(key) {
  if (SAMPLING_FIELDS.some((f) => f.key === key)) return 'sampling';
  if (SERVER_FIELDS.some((f) => f.key === key)) return 'server';
  return 'rendering';
}

function argFixInfo(err) {
  const flag = (err && err.flag) || '';
  const key = ARG_HELP[flag] || null;
  const field = key ? FIELD_BY_KEY[key] : null;
  const hint = ARG_FIX_HINTS[flag] || {};
  const p = current();
  const val = p && key ? p[key] : null;
  const min = hint.min !== undefined ? hint.min : field ? field.min : undefined;
  const max = hint.max !== undefined ? hint.max : field ? field.max : undefined;
  let fixVal = null;
  if (hint.fix !== undefined) fixVal = hint.fix;
  else if (field && field.type === 'select') {
    const opts = (field.options || []).filter((o) => o !== '');
    fixVal = opts.length ? opts[0] : '';
  } else {
    const n = num(val);
    if (n !== null && min !== undefined && n < min) fixVal = min;
    else if (n !== null && max !== undefined && n > max) fixVal = max;
  }
  const panel = key ? fieldPanelOf(key) : null;
  return {
    flag,
    key,
    field,
    val,
    fixVal,
    panel,
    fieldLabel: field ? field.label : null,
    tabLabel: panel === 'sampling' ? 'Muestreo' : panel === 'server' ? 'Servidor' : 'Rendimiento',
  };
}

function renderArgError(err) {
  const box = $('#argErrorBox');
  const body = $('#argErrorBody');
  const info = argFixInfo(err);
  const flag = info.flag || '--desconocido';
  const detail = (err && err.detail) || 'Parámetro rechazado por llama-server.';
  let html = '<p class="arg-err-line"><code>' + escapeHtml(err && err.line ? err.line : flag) + '</code></p>';
  if (info.fieldLabel) {
    html +=
      '<p>El parámetro <code>' + escapeHtml(flag) + '</code> (campo <strong>' + escapeHtml(info.fieldLabel) +
      '</strong>, pestaña ' + escapeHtml(info.tabLabel) + ') fue rechazado por llama-server: <em>' +
      escapeHtml(detail) + '</em>.</p>';
  } else {
    html +=
      '<p>El parámetro <code>' + escapeHtml(flag) + '</code> fue rechazado por llama-server: <em>' +
      escapeHtml(detail) + '</em>. Revisá los valores en la instancia o la consola completa.</p>';
  }
  body.innerHTML = html;
  const canFix = info.fixVal !== null && info.key && current();
  const fixBtn = $('#argErrorFixBtn');
  const fieldBtn = $('#argErrorFieldBtn');
  fixBtn.classList.toggle('hidden', !canFix);
  fieldBtn.classList.toggle('hidden', !info.key);
  fixBtn.dataset.flag = flag;
  fixBtn.dataset.key = info.key || '';
  fixBtn.dataset.val = info.fixVal === null ? '' : String(info.fixVal);
  fixBtn.dataset.old = info.val === null || info.val === undefined ? '' : String(info.val);
  fieldBtn.dataset.flag = flag;
  fieldBtn.dataset.key = info.key || '';
  box.classList.remove('hidden');
  expandBottomDock();
  showBottomTab('consola');
}

function focusFieldCard(key) {
  const wrap = document.querySelector('[data-field-wrap="' + key + '"]');
  if (!wrap) return;
  const card = wrap.closest('.card');
  if (card) {
    card.classList.remove('flash');
    void card.offsetWidth;
    card.classList.add('flash');
    setTimeout(() => card.classList.remove('flash'), 1800);
    card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
}

/* ---------------- Wizard (asistente de nueva instancia) ---------------- */

const WIZARD_STEPS = ['install', 'model', 'vision', 'mtp', 'vram', 'net', 'done'];
const WIZARD_STEP_LABELS = {
  install: 'Instalación',
  model: 'Modelo',
  vision: 'Visión',
  mtp: 'MTP',
  vram: 'VRAM',
  net: 'Servidor',
  done: 'Resumen',
};

const WIZARD_CTX_PRESETS = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288];

const VRAM_PRESETS = [
  {
    id: 'min',
    label: 'Mínima VRAM',
    desc: 'Consume la menor VRAM posible: KV cache en 4-bit y flash attention activado para liberar memoria. Todo se intenta cargar en GPU; si no entra, llama.cpp reparte el resto.',
    values: { gpuLayers: 'all', fit: 'on', cacheTypeK: 'q4_0', cacheTypeV: 'q4_0', flashAttn: 'on', splitMode: 'layer' },
  },
  {
    id: 'normal',
    label: 'Balanceado',
    desc: 'Equilibrio entre calidad y VRAM: KV cache 8-bit y flash attention automático. El modelo se reparte en capas entre las GPUs si no entra completo.',
    values: { gpuLayers: 'all', fit: 'on', cacheTypeK: 'q8_0', cacheTypeV: 'q8_0', flashAttn: 'auto', splitMode: 'layer' },
  },
  {
    id: 'quality',
    label: 'Máxima calidad',
    desc: 'Prioriza velocidad y calidad: KV cache f16 y división por filas (row) para aprovechar varias GPUs en paralelo. Si no entra, llama.cpp distribuye lo que quede.',
    values: { gpuLayers: 'all', fit: 'on', cacheTypeK: 'f16', cacheTypeV: 'f16', flashAttn: 'auto', splitMode: 'row' },
  },
];

let wizard = {
  open: false,
  step: 0,
  installId: null,
  modelPath: '',
  mmprojPath: '',
  mtp: false,
  mtpMode: 'builtin',
  mtpFile: '',
  vramPreset: 'normal',
  ctxSize: 4096,
  temp: null,
  topP: null,
  topK: null,
  minP: null,
  presencePenalty: null,
  repeatPenalty: null,
  host: '127.0.0.1',
  port: 8080,
  name: '',
  modelInfo: null,
  hfRepo: '',
  hfApplied: false,
};

function wizardInstall() {
  const sel = wizard.installId ? installationById(wizard.installId) : null;
  return sel || (settings.installations && settings.installations[0]) || null;
}

function wizardBuildProfile() {
  const p = defaultProfile();
  const preset = VRAM_PRESETS.find((x) => x.id === wizard.vramPreset) || VRAM_PRESETS[1];
  Object.assign(p, preset.values);
  const cfg = ['modelPath', 'ctxSize', 'host', 'port', 'cacheIdleSlots', 'reasoning', 'imageMinTokens'];
  Object.keys(preset.values).forEach((k) => cfg.push(k));
  if (preset.values.gpuLayers === 'all') {
    p.gpuLayersAll = true;
    p.gpuLayers = wizard.modelInfo && wizard.modelInfo.totalLayers ? wizard.modelInfo.totalLayers : 999;
    cfg.push('gpuLayersAll');
  }
  p.installId = wizard.installId;
  p.modelPath = wizard.modelPath;
  p.mmprojPath = wizard.mmprojPath;
  if (p.mmprojPath) cfg.push('mmprojPath');
  const ctx = num(wizard.ctxSize);
  p.ctxSize = ctx !== null && ctx >= 64 ? ctx : 4096;
  p.host = wizard.host || '127.0.0.1';
  const port = num(wizard.port);
  p.port = port !== null ? Math.max(0, Math.min(65535, port)) : 8080;
  ['temp', 'topP', 'topK', 'minP', 'presencePenalty', 'repeatPenalty'].forEach((k) => {
    if (wizard[k] !== null && wizard[k] !== undefined) {
      p[k] = wizard[k];
      cfg.push(k);
    }
  });
  p.mtp = wizard.mtp;
  if (wizard.mtp) {
    cfg.push('mtp', 'specDraftNMax', 'specDraftNMin', 'specDraftPSplit');
    p.specDraftModel = wizard.mtpMode === 'file' ? wizard.mtpFile : '';
    if (wizard.mtpMode === 'file') cfg.push('specDraftModel');
  }
  if (wizard.modelInfo && wizard.modelInfo.isMoe) {
    p.nCpuMoe = 0;
    cfg.push('nCpuMoe');
  }
  const base = wizard.modelPath ? wizard.modelPath.split(/[\\/]/).pop().replace(/\.gguf$/i, '') : '';
  p.name = (wizard.name || base || 'Nueva instancia').trim();
  markConfigured(p, ...cfg);
  return p;
}

function wizardCanProceed() {
  const name = WIZARD_STEPS[wizard.step];
  if (name === 'install') return !!(wizard.installId && installationById(wizard.installId));
  if (name === 'model') return !!wizard.modelPath;
  if (name === 'vision') return true;
  if (name === 'mtp') {
    if (!wizard.mtp) return true;
    return wizard.mtpMode === 'builtin' || !!wizard.mtpFile;
  }
  if (name === 'vram') return true;
  if (name === 'net') {
    const port = num(wizard.port);
    return !!wizard.host && port !== null && port >= 0 && port <= 65535;
  }
  return true;
}

function wizardUpdateNav() {
  $('#wizardNextBtn').disabled = !wizardCanProceed();
}

function wizardRenderSteps() {
  const bar = $('#wizardSteps');
  bar.innerHTML = '';
  WIZARD_STEPS.forEach((s, i) => {
    const el = document.createElement('span');
    el.className = 'wizard-step-label' + (i === wizard.step ? ' active' : i < wizard.step ? ' done' : '');
    el.textContent = (i + 1) + '. ' + WIZARD_STEP_LABELS[s];
    bar.appendChild(el);
  });
}

function wizardAddOption(sel, val, label) {
  const opt = document.createElement('option');
  opt.value = val;
  opt.textContent = label || val.split(/[\\/]/).pop();
  sel.appendChild(opt);
  sel.value = val;
}

function wizardPopulate() {
  const installs = settings.installations || [];
  const installSel = $('#wizInstall');
  installSel.innerHTML =
    '<option value="">— Elegir una versión —</option>' +
    installs.map((i) => `<option value="${escapeHtml(i.id)}">${escapeHtml(installLabel(i))}</option>`).join('');
  if (installs.length && !installationById(wizard.installId)) wizard.installId = installs[0].id;
  installSel.value = wizard.installId || '';
  const mains = modelsCache.filter((m) => modelKind(m) === 'main');
  const visions = modelsCache.filter((m) => modelKind(m) === 'vision');
  const mtps = modelsCache.filter((m) => modelKind(m) === 'mtp');
  $('#wizModel').innerHTML = '<option value="">— Elegir modelo —</option>' + mains
    .map((m) => `<option value="${escapeHtml(m)}">${escapeHtml(modelLabel(m))}</option>`)
    .join('');
  $('#wizVisionFile').innerHTML = '<option value="">— Elegir archivo —</option>' + visions
    .map((m) => `<option value="${escapeHtml(m)}">${escapeHtml(modelLabel(m))}</option>`)
    .join('');
  $('#wizMtpFile').innerHTML = '<option value="">— Elegir archivo MTP —</option>' + mtps
    .map((m) => `<option value="${escapeHtml(m)}">${escapeHtml(modelLabel(m))}</option>`)
    .join('');
  $('#wizModel').value = wizard.modelPath;
  $('#wizVisionFile').value = wizard.mmprojPath;
  $('#wizMtpFile').value = wizard.mtpFile;
  const repoInput = $('#wizHfRepo');
  if (repoInput) {
    const mapped = wizard.modelPath && settings.hfRepoMap ? (settings.hfRepoMap[wizard.modelPath] || '') : '';
    const mappedId = mapped ? mapped.replace(/^https?:\/\/huggingface\.co\//i, '').replace(/\/+$/, '') : '';
    const pref = wizard.hfRepo || mappedId || hf.lastRepo || '';
    if (!repoInput.value || repoInput.value !== pref) repoInput.value = pref;
  }
  wizardSyncVisionUi();
  wizardSyncMtpUi();
  wizardUpdateNav();
}

function wizardSyncVisionUi() {
  const cb = $('#wizVision');
  if (!cb) return;
  const hasVision = !!wizard.mmprojPath;
  cb.checked = hasVision;
  $('#wizVisionRow').style.display = hasVision ? '' : 'none';
  $('#wizVisionFile').value = wizard.mmprojPath || '';
}

function wizardSyncMtpUi() {
  const cb = $('#wizMtp');
  if (!cb) return;
  cb.checked = wizard.mtp;
  $('#wizMtpRow').style.display = wizard.mtp ? '' : 'none';
  document.querySelectorAll('input[name="wizMtpMode"]').forEach((r) => {
    r.checked = r.value === wizard.mtpMode;
  });
  $('#wizMtpFileRow').style.display = wizard.mtp && wizard.mtpMode === 'file' ? '' : 'none';
  $('#wizMtpFile').value = wizard.mtpFile || '';
  wizardUpdateNav();
}

function wizardAutoFillFromDownload() {
  if (!hf.lastDir || !wizard.modelPath) return;
  const mp = String(wizard.modelPath).replace(/\\/g, '/').toLowerCase();
  const dir = String(hf.lastDir).replace(/\\/g, '/').toLowerCase();
  if (!mp.startsWith(dir + '/')) return;
  if (hf.lastVisionName && !wizard.mmprojPath) {
    wizard.mmprojPath = hf.lastDir + '\\' + hf.lastVisionName;
  }
  if (hf.lastMtpName && !wizard.mtpFile) {
    wizard.mtp = true;
    wizard.mtpMode = 'file';
    wizard.mtpFile = hf.lastDir + '\\' + hf.lastMtpName;
  }
  wizardSyncVisionUi();
  wizardSyncMtpUi();
}

async function wizardRefreshModelInfo() {
  const infoEl = $('#wizModelInfo');
  if (!wizard.modelPath) {
    wizard.modelInfo = null;
    infoEl.innerHTML = 'Ningún modelo seleccionado.';
    wizardVramInfo();
    wizardCtxHint();
    return;
  }
  infoEl.innerHTML = 'Analizando modelo…';
  let info = null;
  try {
    const res = await window.api.inspectModel(wizard.modelPath);
    if (res && res.ok) info = res;
  } catch (e) {}
  wizard.modelInfo = info;
  if (info) {
    infoEl.innerHTML =
      '<span class="text-ok">' + escapeHtml(wizard.modelPath.split(/[\\/]/).pop()) + '</span> · ' +
      fmtGb(info.fileBytes) + ' · <b>' + info.totalLayers + '</b> capas' +
      (info.layerBytes ? ' · ~' + fmtGb(info.layerBytes) + '/capa' : '') +
      (info.isMoe ? ' · <b>MoE</b>' + (info.nExperts ? ' (' + info.nExperts + ' expertos)' : '') : '');
    if (info.isMoe) {
      toast('Modelo MoE detectado' + (info.nExperts ? ' (' + info.nExperts + ' expertos)' : '') + '. Se configurará n-cpu-moe automáticamente.', 'ok');
    }
  } else {
    infoEl.innerHTML = '<span class="text-err">No se pudo analizar el modelo (puede no ser GGUF o estar dañado).</span>';
  }
  wizardVramInfo();
  wizardCtxHint();
}

function wizardRenderCtxPresets() {
  const row = $('#wizCtxPresets');
  const input = $('#wizCtx');
  if (!row) return;
  row.innerHTML = '';
  WIZARD_CTX_PRESETS.forEach((v) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'preset-btn' + (Number(wizard.ctxSize) === v ? ' active' : '');
    btn.dataset.ctx = v;
    btn.textContent = v >= 1024 ? String(v / 1024) + 'K' : String(v);
    btn.addEventListener('click', () => {
      wizard.ctxSize = v;
      input.value = v;
      row.querySelectorAll('.preset-btn').forEach((b) => b.classList.toggle('active', b === btn));
      wizardCtxHint();
    });
    row.appendChild(btn);
  });
  if (input) input.value = wizard.ctxSize;
  wizardCtxHint();
}

function wizardCtxHint() {
  const el = $('#wizCtxHint');
  if (!el) return;
  const nCtx = num(wizard.ctxSize);
  const info = wizard.modelInfo;
  if (!info || !info.headDim || !nCtx) {
    el.innerHTML = 'Preset de contexto: <b>' + (nCtx || 0) + '</b> tokens.';
    return;
  }
  const p = wizardBuildProfile();
  const est = estimateVram(p, info);
  el.innerHTML =
    'Preset de contexto: <b>' + nCtx + '</b> tokens · KV cache ≈ <b>' + fmtGb(est.kvBytes) + '</b> (con KV ' +
    p.cacheTypeK.toUpperCase() + '/' + p.cacheTypeV.toUpperCase() + ').';
}

async function wizardApplyHfRecommended(repoOverride) {
  const input = $('#wizHfRepo');
  const info = $('#wizHfRecInfo');
  const repo = (repoOverride || input.value || '').trim();
  if (repoOverride && input && input.value !== repoOverride) input.value = repoOverride;
  if (!repo) {
    info.innerHTML = '<span class="text-err">Escribí el repositorio de Hugging Face primero.</span>';
    toast(t('toast_hf_repo_empty'), 'err');
    return;
  }
  info.innerHTML = 'Consultando <b>' + escapeHtml(repo) + '</b>…';
  let res = null;
  try {
    res = await window.api.getHfModelConfig(repo);
  } catch (e) {
    res = { ok: false, error: (e && e.message) || 'Error de red.' };
  }
  if (!res || !res.ok) {
    info.innerHTML = '<span class="text-err">' + escapeHtml((res && res.error) || 'No se pudo consultar el modelo.') + '</span>';
    toast('No se pudo consultar el modelo', 'err');
    return;
  }
  const m = res.info || {};
  wizard.hfRepo = repo;
  wizard.hfApplied = true;
  const applied = [];
  if (m.ctxSize) {
    wizard.ctxSize = m.ctxSize;
    applied.push('contexto ' + Number(m.ctxSize).toLocaleString('es-AR'));
  }
  [['temp', 'temperatura'], ['topP', 'top_p'], ['topK', 'top_k'], ['minP', 'min_p'], ['presencePenalty', 'presence_penalty'], ['repeatPenalty', 'repetition_penalty']].forEach(([key, label]) => {
    if (m[key] === null || m[key] === undefined) return;
    wizard[key] = m[key];
    applied.push(label + ' ' + Number(m[key]));
  });
  const ctxInput = $('#wizCtx');
  if (ctxInput) ctxInput.value = wizard.ctxSize ?? '';
  wizardRenderCtxPresets();
  wizardCtxHint();
  let html = '<span class="text-ok">Aplicado a la instancia:</span> ' + escapeHtml(applied.join(', ') || 'sin cambios detectados') + '.';
  html += '<br>Modelo: <b>' + escapeHtml(m.modelType || '—') + '</b>';
  if (m.isVision) html += ' · <span class="text-ok">multimodal</span>';
  if (m.ropeType) html += ' · rope: <b>' + escapeHtml(m.ropeType) + '</b>' + (m.ropeFactor ? ' x' + m.ropeFactor : '');
  html += '<br><span class="hint">Fuente: https://huggingface.co/' + escapeHtml(repo) + '</span>';
  info.innerHTML = html;
  toast('Parámetros recomendados aplicados', 'ok');
}

async function wizardApplyHfFromMap() {
  const repo = wizard.modelPath && settings.hfRepoMap ? (settings.hfRepoMap[wizard.modelPath] || '') : '';
  if (!repo) return;
  const repoId = repo.replace(/^https?:\/\/huggingface\.co\//i, '').replace(/\/+$/, '');
  if (wizard.hfRepo === repoId && wizard.hfApplied) return;
  await wizardApplyHfRecommended(repoId);
}

function wizardRenderVramOptions() {
  const box = $('#wizVramOptions');
  box.innerHTML = '';
  VRAM_PRESETS.forEach((pr) => {
    const label = document.createElement('label');
    label.className = 'vram-option' + (pr.id === wizard.vramPreset ? ' active' : '');
    label.innerHTML = `
      <input type="radio" name="wizVram" value="${pr.id}" ${pr.id === wizard.vramPreset ? 'checked' : ''} />
      <span class="vram-option-body">
        <span class="vram-option-title">${escapeHtml(pr.label)}</span>
        <span class="vram-option-desc">${escapeHtml(pr.desc)}</span>
      </span>
    `;
    label.addEventListener('click', () => {
      wizard.vramPreset = pr.id;
      box.querySelectorAll('.vram-option').forEach((o) => o.classList.toggle('active', o === label));
      wizardVramInfo();
    });
    box.appendChild(label);
  });
  wizardVramInfo();
}

function wizardVramInfo() {
  const el = $('#wizVramInfo');
  const pr = VRAM_PRESETS.find((x) => x.id === wizard.vramPreset);
  const info = wizard.modelInfo;
  if (!pr) {
    el.textContent = '';
    return;
  }
  let html = pr.desc;
  if (info) {
    const p = wizardBuildProfile();
    const est = estimateVram(p, info);
    if (est) {
      html +=
        '<br>Modelo: <b>' + info.totalLayers + '</b> capas · todo en GPU ≈ <b>' + fmtGb(info.layerBytes * info.totalLayers) + '</b>' +
        '<br>Contexto <b>' + est.nCtx + '</b> · KV <b>' + p.cacheTypeK.toUpperCase() + '/' + p.cacheTypeV.toUpperCase() + '</b> ≈ <b>' + fmtGb(est.kvBytes) + '</b>' +
        '<br>Flash attention: <b>' + p.flashAttn + '</b> · división: <b>' + (p.splitMode || 'layer') + '</b> · total ≈ <b>' + fmtGb(est.total) + '</b>';
    }
  } else {
    html += '<br>Seleccioná un modelo en el paso anterior para ver la estimación de VRAM.';
  }
  el.innerHTML = html;
}

function wizardRenderSummary() {
  const p = wizardBuildProfile();
  const install = installationById(p.installId);
  const cmd = buildCommandLine(p);
  const preset = VRAM_PRESETS.find((x) => x.id === wizard.vramPreset) || VRAM_PRESETS[1];
  const rows = [
    ['Nombre', '<input id="wizName" type="text" class="text-input" value="' + escapeHtml(p.name) + '" />'],
    ['Versión', install ? escapeHtml(installLabel(install)) : '—'],
    ['Modelo', p.modelPath ? escapeHtml(p.modelPath) : '—'],
    ['Contexto', p.ctxSize ? String(p.ctxSize) + ' tokens' : '—'],
    ['Visión', p.mmprojPath ? escapeHtml(p.mmprojPath.split(/[\\/]/).pop()) : 'No'],
    ['MTP', p.mtp ? (p.specDraftModel ? 'Archivo: ' + escapeHtml(p.specDraftModel) : 'Incluido en el modelo') : 'No'],
    ['VRAM', preset.label + ' · KV ' + String(p.cacheTypeK).toUpperCase() + '/' + String(p.cacheTypeV).toUpperCase()],
    ['Servidor', escapeHtml(p.host) + ':' + p.port],
  ];
  $('#wizSummary').innerHTML =
    '<div class="summary-rows">' +
    rows.map(([l, v]) => '<div class="summary-row"><span class="summary-label">' + l + '</span><span class="summary-value">' + v + '</span></div>').join('') +
    '</div>' +
    '<details class="summary-cmd-wrap">' +
    '<summary class="summary-cmd-toggle">Ver instancia completa</summary>' +
    '<pre class="summary-cmd">' + escapeHtml(cmd) + '</pre>' +
    '</details>';
  const nameInput = $('#wizName');
  if (nameInput) nameInput.addEventListener('input', () => { wizard.name = nameInput.value; });
}

function wizardShowStep(i) {
  wizard.step = Math.max(0, Math.min(WIZARD_STEPS.length - 1, i));
  const name = WIZARD_STEPS[wizard.step];
  document.querySelectorAll('#wizardModal .wizard-step').forEach((s) => {
    s.classList.toggle('hidden', s.dataset.step !== name);
  });
  $('#wizardBackBtn').style.visibility = wizard.step === 0 ? 'hidden' : 'visible';
  const isLast = wizard.step === WIZARD_STEPS.length - 1;
  $('#wizardNextBtn').classList.toggle('hidden', isLast);
  $('#wizardFinalBtns').classList.toggle('hidden', !isLast);
  if (isLast) {
    wizardRenderSummary();
  } else if (name === 'model') {
    wizardRenderCtxPresets();
    wizardRefreshModelInfo();
  } else if (name === 'vision') {
    wizardSyncVisionUi();
  } else if (name === 'mtp') {
    wizardSyncMtpUi();
  } else if (name === 'vram') {
    wizardRenderVramOptions();
  } else if (name === 'net') {
    $('#wizHost').value = wizard.host;
    $('#wizPort').value = wizard.port ?? '';
  }
  wizardRenderSteps();
  wizardUpdateNav();
}

function openWizard() {
  wizard = {
    open: true,
    step: 0,
    installId: null,
    modelPath: '',
    mmprojPath: '',
    mtp: false,
    mtpMode: 'builtin',
    mtpFile: '',
    vramPreset: 'normal',
    ctxSize: 4096,
    temp: null,
    topP: null,
    topK: null,
    minP: null,
    presencePenalty: null,
    repeatPenalty: null,
    host: '127.0.0.1',
    port: 8080,
    name: '',
    modelInfo: null,
    hfRepo: '',
    hfApplied: false,
  };
  const installs = settings.installations || [];
  if (installs.length) wizard.installId = installs[0].id;
  wizardPopulate();
  $('#wizardModal').classList.remove('hidden');
  wizardShowStep(0);
}

function closeWizard() {
  wizard.open = false;
  $('#wizardModal').classList.add('hidden');
}

async function wizardFinish(run) {
  const p = wizardBuildProfile();
  const install = installationById(p.installId);
  if (!p.modelPath) {
    toast('Elegí un modelo antes de finalizar', 'err');
    return;
  }
  if (!install || !install.exePath) {
    toast('Configurá una versión de llama.cpp primero', 'err');
    return;
  }
  profiles.push(p);
  selectProfile(p.id);
  closeWizard();
  let started = false;
  if (run) {
    const problems = profileProblems(p);
    if (problems.length) {
      toast('No se puede ejecutar: ' + problems[0], 'err');
      showDashboard();
      refresh();
      return;
    }
    const res = await window.api.startServer({ exe: install.exePath, args: buildArgs(p) });
    if (res.ok) {
      runningProfileId = p.id;
      setRunning(true);
      started = true;
    } else {
      toast(res.error || 'No se pudo iniciar', 'err');
    }
  }
  scheduleSave();
  showDashboard();
  refresh();
    toast(t('toast_instance_saved_started'), 'ok');
}

function wizardDetails() {
  const p = wizardBuildProfile();
  if (!p.modelPath) {
    toast('Elegí un modelo antes de continuar', 'err');
    return;
  }
  profiles.push(p);
  selectProfile(p.id);
  closeWizard();
  scheduleSave();
  showEditor();
  $('#nameInput').focus();
}

/* ---------------- Descargador de llama.cpp ---------------- */

const DL_STEPS = ['version', 'backend', 'progress'];
const DL_LABELS = { version: 'Versión', backend: 'Backend', progress: 'Descarga' };

const dl = {
  open: false,
  step: 0,
  releases: [],
  tag: null,
  backend: null,
  cudaVersion: null,
  running: false,
  doneInstall: null,
  _lastT: null,
  _lastB: 0,
};

function fmtBytes(b) {
  const n = Number(b) || 0;
  if (n < 1024) return n + ' B';
  if (n < 1024 * 1024) return (n / 1024).toFixed(1) + ' KB';
  if (n < 1024 * 1024 * 1024) return (n / 1024 / 1024).toFixed(1) + ' MB';
  return (n / 1024 / 1024 / 1024).toFixed(2) + ' GB';
}

function dlRenderSteps() {
  const bar = $('#dlSteps');
  if (!bar) return;
  bar.innerHTML = '';
  DL_STEPS.forEach((s, i) => {
    const el = document.createElement('span');
    el.className = 'wizard-step-label' + (i === dl.step ? ' active' : i < dl.step ? ' done' : '');
    el.textContent = (i + 1) + '. ' + DL_LABELS[s];
    bar.appendChild(el);
  });
}

function dlShowStep(i) {
  dl.step = Math.max(0, Math.min(DL_STEPS.length - 1, i));
  const name = DL_STEPS[dl.step];
  document.querySelectorAll('#dlModal .wizard-step').forEach((s) => {
    s.classList.toggle('hidden', s.dataset.step !== name);
  });
  $('#dlBackBtn').style.visibility = dl.step === 1 ? 'visible' : 'hidden';
  $('#dlNextBtn').style.display = dl.step === 2 ? 'none' : '';
  $('#dlCancelBtn').classList.add('hidden');
  $('#dlDoneBtns').classList.add('hidden');
  if (dl.step === 0) {
    dlUpdateNav();
  } else if (dl.step === 1) {
    dlRenderBackends();
    dlUpdateNav();
  } else {
    $('#dlProgressFill').classList.remove('ok');
    $('#dlProgressFill').style.width = '0%';
    $('#dlProgressPct').textContent = '0%';
    $('#dlProgressDetail').textContent = '';
    $('#dlResult').innerHTML = '';
  }
  dlRenderSteps();
}

function dlUpdateNav() {
  if (dl.step === 0) {
    $('#dlNextBtn').disabled = !dl.tag;
  } else if (dl.step === 1) {
    $('#dlNextBtn').disabled = !dl.backend || (dl.backend === 'cuda' && !dl.cudaVersion);
  }
}

async function openDownloader() {
  dl.open = true;
  dl.step = 0;
  dl.releases = [];
  dl.tag = null;
  dl.backend = null;
  dl.cudaVersion = null;
  dl.running = false;
  dl.doneInstall = null;
  $('#dlVersionErr').classList.add('hidden');
  $('#dlVersionErr').textContent = '';
  $('#dlVersionLoading').style.display = '';
  $('#dlVersionList').innerHTML = '';
  $('#dlVersionEmpty').style.display = 'none';
  $('#dlModal').classList.remove('hidden');
  dlShowStep(0);
  try {
    const targetDir = await window.api.getLlamaDownloadDir();
    $('#dlTargetDir').textContent = targetDir;
  } catch (e) {}
  await loadReleases();
}

function closeDownloader() {
  dl.open = false;
  $('#dlModal').classList.add('hidden');
}

async function loadReleases() {
  $('#dlVersionLoading').style.display = '';
  $('#dlVersionErr').classList.add('hidden');
  const res = await window.api.listLlamaReleases();
  $('#dlVersionLoading').style.display = 'none';
  if (!res || !res.ok) {
    $('#dlVersionErr').textContent = res && res.error ? res.error : 'No se pudieron cargar las versiones.';
    $('#dlVersionErr').classList.remove('hidden');
    return;
  }
  dl.releases = res.releases || [];
  dlRenderList();
}

function dlRenderList() {
  const box = $('#dlVersionList');
  box.innerHTML = '';
  if (!dl.releases.length) {
    $('#dlVersionEmpty').style.display = '';
    return;
  }
  dl.releases.forEach((r) => {
    const item = document.createElement('div');
    item.className = 'dl-item' + (r.tag === dl.tag ? ' active' : '');
    const date = r.published ? new Date(r.published).toLocaleDateString() : '';
    const badges = [];
    if (r.cpu) badges.push('<span class="dl-badge b-cpu">CPU</span>');
    if (r.vulkan) badges.push('<span class="dl-badge b-vulkan">Vulkan</span>');
    if (r.cuda.length) badges.push('<span class="dl-badge b-cuda">CUDA</span>');
    item.innerHTML =
      '<div class="dl-item-main"><div class="dl-item-tag">' + escapeHtml(r.tag) + '</div>' +
      '<div class="dl-item-date">' + escapeHtml(date) + '</div></div>' +
      '<div class="dl-badges">' + badges.join('') + '</div>';
    item.addEventListener('click', () => {
      dl.tag = r.tag;
      dl.backend = null;
      dl.cudaVersion = null;
      box.querySelectorAll('.dl-item').forEach((x) => x.classList.toggle('active', x === item));
      dlUpdateNav();
    });
    box.appendChild(item);
  });
}

function dlRenderBackends() {
  const rel = dl.releases.find((r) => r.tag === dl.tag);
  const box = $('#dlBackends');
  box.innerHTML = '';
  if (!rel) return;
  const defs = [
    { id: 'cpu', label: 'Solo CPU', desc: 'Compilación para CPU. No requiere GPU ni controladores adicionales.', avail: !!rel.cpu },
    { id: 'vulkan', label: 'Vulkan', desc: 'Aceleración con GPUs compatibles con Vulkan (AMD, Intel, NVIDIA).', avail: !!rel.vulkan },
    {
      id: 'cuda',
      label: 'NVIDIA (CUDA)',
      desc: 'Aceleración con GPUs NVIDIA. Se descargan también las DLL del runtime de CUDA y se instalan junto al ejecutable.',
      avail: rel.cuda.length > 0,
    },
  ];
  defs.forEach((d) => {
    const label = document.createElement('label');
    label.className = 'vram-option' + (dl.backend === d.id ? ' active' : '') + (d.avail ? '' : ' dl-disabled');
    label.innerHTML =
      '<input type="radio" name="dlBackend" value="' + d.id + '" ' + (d.avail ? '' : 'disabled') + ' ' +
      (dl.backend === d.id ? 'checked' : '') + ' />' +
      '<span class="vram-option-body"><span class="vram-option-title">' + escapeHtml(d.label) + '</span>' +
      '<span class="vram-option-desc">' + escapeHtml(d.desc) + '</span></span>';
    if (d.avail) {
      label.addEventListener('click', () => {
        dl.backend = d.id;
        box.querySelectorAll('.vram-option').forEach((o) => o.classList.toggle('active', o === label));
        dlRenderCuda();
        dlUpdateNav();
      });
    }
    box.appendChild(label);
  });
  dlRenderCuda();
}

function dlRenderCuda() {
  const row = $('#dlCudaRow');
  const sel = $('#dlCudaVersion');
  if (dl.backend !== 'cuda') {
    row.style.display = 'none';
    return;
  }
  const rel = dl.releases.find((r) => r.tag === dl.tag);
  const versions = rel ? rel.cuda.map((c) => c.version || 'Legacy') : [];
  sel.innerHTML = versions.map((v) => '<option value="' + escapeHtml(v) + '">CUDA ' + escapeHtml(v) + '</option>').join('');
  if (versions.length === 1) {
    row.style.display = 'none';
    dl.cudaVersion = versions[0];
    return;
  }
  if (versions.length && !versions.includes(dl.cudaVersion)) dl.cudaVersion = versions[0];
  sel.value = dl.cudaVersion;
  row.style.display = '';
  sel.onchange = () => {
    dl.cudaVersion = sel.value;
    dlUpdateNav();
  };
}

function dlSetRunningUI(running, ok) {
  if (running) {
    $('#dlCancelBtn').classList.remove('hidden');
    $('#dlCancelBtn').textContent = 'Cancelar';
    $('#dlDoneBtns').classList.add('hidden');
  } else if (ok) {
    $('#dlCancelBtn').classList.add('hidden');
    $('#dlDoneBtns').classList.remove('hidden');
  } else {
    $('#dlCancelBtn').classList.remove('hidden');
    $('#dlCancelBtn').textContent = 'Cerrar';
    $('#dlDoneBtns').classList.add('hidden');
  }
}

async function dlStartDownload() {
  dl._lastT = null;
  dl._lastB = 0;
  $('#dlProgressFill').classList.remove('ok');
  $('#dlProgressFill').style.width = '0%';
  $('#dlProgressPct').textContent = '0%';
  $('#dlProgressDetail').textContent = '';
  $('#dlResult').innerHTML = '';
  $('#dlProgressTitle').textContent = 'Descargando…';
  dl.running = true;
  dlSetRunningUI(true);
  window.api.startLlamaDownload({ tag: dl.tag, backend: dl.backend, cudaVersion: dl.cudaVersion }).catch(() => {});
}

function dlHandleProgress(p) {
  const fill = $('#dlProgressFill');
  const pctEl = $('#dlProgressPct');
  const detail = $('#dlProgressDetail');
  $('#dlProgressTitle').textContent = p.label || 'Descargando…';
  if (p.phase === 'download') {
    const received = p.received || 0;
    const total = p.total || 0;
    const pct = total ? Math.min(100, (received / total) * 100) : 0;
    fill.style.width = pct + '%';
    pctEl.textContent = pct.toFixed(0) + '%';
    let speed = 0;
    const now = Date.now();
    if (dl._lastT) {
      const dt = (now - dl._lastT) / 1000;
      const db = received - dl._lastB;
      if (dt > 0 && db >= 0) speed = db / dt;
    }
    dl._lastT = now;
    dl._lastB = received;
    detail.textContent = fmtBytes(received) + ' / ' + fmtBytes(total) + (speed ? ' · ' + fmtBytes(speed) + '/s' : '');
  } else if (p.phase === 'extract') {
    const written = p.written || 0;
    const total = p.total || 0;
    const pct = total ? Math.min(100, (written / total) * 100) : 0;
    fill.style.width = pct + '%';
    pctEl.textContent = pct.toFixed(0) + '%';
    detail.textContent = p.file ? escapeHtml(p.file) : '';
  } else if (p.phase === 'move') {
    fill.style.width = '100%';
    pctEl.textContent = '…';
    detail.textContent = p.label || '';
  }
}

function dlHandleDone(install) {
  dl.running = false;
  dl.doneInstall = install;
  $('#dlProgressTitle').textContent = 'Instalado correctamente';
  $('#dlProgressDetail').textContent = '';
  $('#dlProgressFill').style.width = '100%';
  $('#dlProgressFill').classList.add('ok');
  $('#dlProgressPct').textContent = '100%';
  $('#dlResult').innerHTML =
    '<p class="hint" style="margin-bottom:2px"><span class="text-ok">Versión instalada:</span> <b>' + escapeHtml(install.name) + '</b></p>' +
    '<p class="hint">' + escapeHtml(install.path) + '</p>';
  dlSetRunningUI(false, true);
}

function dlHandleError(msg) {
  dl.running = false;
  $('#dlProgressTitle').textContent = 'No se pudo completar';
  $('#dlProgressDetail').textContent = '';
  $('#dlProgressFill').style.width = '0%';
  $('#dlProgressPct').textContent = '';
  $('#dlResult').innerHTML = '<p class="hint text-err">' + escapeHtml(msg) + '</p>';
  dlSetRunningUI(false, false);
}

async function dlRefreshSettings() {
  try {
    settings = await window.api.loadSettings();
  } catch (e) {}
  refresh();
}

async function dlFinish() {
  await dlRefreshSettings();
  closeDownloader();
  if (wizard.open) wizardPopulate();
  toast(t('toast_version_installed'), 'ok');
}

async function dlUse() {
  await dlRefreshSettings();
  closeDownloader();
  wizard.installId = dl.doneInstall ? dl.doneInstall.id : null;
  openWizard();
  if (wizard.installId && $('#wizInstall')) $('#wizInstall').value = wizard.installId;
  wizardUpdateNav();
}

/* ---------------- Descargador de modelos de HuggingFace ---------------- */

const HF_STEPS = ['search', 'files', 'progress'];
const HF_LABELS = { search: 'Buscar', files: 'Archivos', progress: 'Descarga' };
const HF_CATEGORIES = [
  { id: '', label: 'Todos' },
  { id: 'llm', label: 'LLM (texto)' },
  { id: 'vision', label: 'Imagen ↔ Texto (visión)' },
  { id: 't2i', label: 'Texto → Imagen' },
  { id: 'audio', label: 'Audio' },
  { id: 'embed', label: 'Embeddings' },
];
const HF_KNOWN_TASKS = [
  'text-generation',
  'image-text-to-text',
  'image-to-text',
  'text-to-image',
  'text-to-audio',
  'audio-to-text',
  'audio-to-audio',
  'automatic-speech-recognition',
  'text-to-speech',
  'text-to-music',
  'audio-text-to-text',
  'feature-extraction',
  'sentence-similarity',
];

const hf = {
  open: false,
  step: 0,
  models: [],
  repo: null,
  files: null,
  quant: null,
  visionFile: null,
  mtpFile: null,
  running: false,
  lastRepo: '',
  lastDir: null,
  lastVisionName: null,
  lastMtpName: null,
  lastFiles: [],
  rangeMin: 0,
  rangeMax: 100,
  _lastT: null,
  _lastB: 0,
};

function hfNumFiltered() {
  const min = hf.rangeMin;
  const max = hf.rangeMax;
  const filterParams = min > 0 || max < 100;
  const ggufOnly = $('#hfGgufOnly').checked;
  const cat = $('#hfCategory').value;
  return hf.models.filter((m) => {
    if (ggufOnly && !/-GGUF$/i.test(m.id)) return false;
    if (cat === 'other' && HF_KNOWN_TASKS.includes(m.task)) return false;
    if (!filterParams) return true;
    if (m.params == null) return false;
    return m.params >= min && m.params <= max;
  });
}

const HF_RANGE_SNAPS = [0, 0.5, 1, 1.5, 2, 3, 4, 7, 8, 13, 14, 27, 30, 32, 34, 70, 72, 100];
const HF_RANGE_TICKS = [
  { v: 0.5, label: '0.5' },
  { v: 1, label: '1' },
  { v: 3, label: '3' },
  { v: 7, label: '7' },
  { v: 13, label: '13' },
  { v: 30, label: '30' },
  { v: 70, label: '70' },
  { v: 100, label: '100+' },
];

function hfSnapToTick(v) {
  let best = HF_RANGE_SNAPS[0];
  let bestDist = Math.abs(v - best);
  for (let i = 1; i < HF_RANGE_SNAPS.length; i++) {
    const d = Math.abs(v - HF_RANGE_SNAPS[i]);
    if (d < bestDist) {
      best = HF_RANGE_SNAPS[i];
      bestDist = d;
    }
  }
  return best;
}

function hfRangePercent(v) {
  return Math.max(0, Math.min(100, (v / 100) * 100));
}

function hfUpdateRange() {
  const fill = $('#hfFill');
  const hMin = $('#hfHandleMin');
  const hMax = $('#hfHandleMax');
  const minP = hfRangePercent(hf.rangeMin);
  const maxP = hfRangePercent(hf.rangeMax);
  fill.style.left = minP + '%';
  fill.style.width = (maxP - minP) + '%';
  hMin.style.left = minP + '%';
  hMax.style.left = maxP + '%';
  $('#hfMinVal').textContent = hf.rangeMin === 0 ? '0B' : hf.rangeMin + 'B';
  $('#hfMaxVal').textContent = hf.rangeMax >= 100 ? '100B+' : hf.rangeMax + 'B';
  hfRenderResults();
}

function initHfRange() {
  const track = $('#hfTrack');
  const hMin = $('#hfHandleMin');
  const hMax = $('#hfHandleMax');
  const ticks = $('#hfTicks');
  if (!track || !hMin || !hMax) return;

  HF_RANGE_TICKS.forEach((t) => {
    const el = document.createElement('div');
    el.className = 'hf-range-tick';
    el.style.left = hfRangePercent(t.v) + '%';
    el.innerHTML = '<div class="hf-range-tick-mark"></div><div class="hf-range-tick-label">' + t.label + '</div>';
    ticks.appendChild(el);
  });

  function pxToVal(clientX) {
    const rect = track.getBoundingClientRect();
    const pct = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    return pct * 100;
  }

  let dragging = null;

  function onPointerDown(e) {
    if (e.target === hMin) dragging = 'min';
    else if (e.target === hMax) dragging = 'max';
    else {
      const v = pxToVal(e.clientX);
      const distMin = Math.abs(v - hf.rangeMin);
      const distMax = Math.abs(v - hf.rangeMax);
      dragging = distMin <= distMax ? 'min' : 'max';
    }
    e.target.setPointerCapture(e.pointerId);
    e.preventDefault();
  }

  function onPointerMove(e) {
    if (!dragging) return;
    const raw = pxToVal(e.clientX);
    const snapped = hfSnapToTick(raw);
    if (dragging === 'min') {
      hf.rangeMin = Math.min(snapped, hf.rangeMax);
    } else {
      hf.rangeMax = Math.max(snapped, hf.rangeMin);
    }
    hfUpdateRange();
  }

  function onPointerUp() {
    dragging = null;
  }

  track.addEventListener('pointerdown', onPointerDown);
  track.addEventListener('pointermove', onPointerMove);
  track.addEventListener('pointerup', onPointerUp);
  track.addEventListener('pointercancel', onPointerUp);
  hMin.addEventListener('pointerdown', onPointerDown);
  hMax.addEventListener('pointerdown', onPointerDown);

  hfUpdateRange();
}

function hfRenderSteps() {
  const bar = $('#hfSteps');
  if (!bar) return;
  bar.innerHTML = '';
  HF_STEPS.forEach((s, i) => {
    const el = document.createElement('span');
    el.className = 'wizard-step-label' + (i === hf.step ? ' active' : i < hf.step ? ' done' : '');
    el.textContent = (i + 1) + '. ' + HF_LABELS[s];
    bar.appendChild(el);
  });
}

function hfShowStep(i) {
  hf.step = Math.max(0, Math.min(HF_STEPS.length - 1, i));
  const name = HF_STEPS[hf.step];
  document.querySelectorAll('#hfModal .wizard-step').forEach((s) => {
    s.classList.toggle('hidden', s.dataset.step !== name);
  });
  $('#hfBackBtn').style.visibility = hf.step === 1 ? 'visible' : 'hidden';
  $('#hfNextBtn').style.display = hf.step === 1 ? '' : 'none';
  $('#hfCancelBtn').classList.add('hidden');
  $('#hfDoneBtns').classList.add('hidden');
  $('#hfNextBtn').disabled = false;
  if (hf.step === 1) hfUpdateNav();
  hfRenderSteps();
}

function hfUpdateNav() {
  if (hf.step === 1) {
    $('#hfNextBtn').disabled = !hf.quant;
    $('#hfFilesHint').textContent = 'Vas a descargar ' + hfFilesToDownload().length + ' archivo(s).';
  }
}

function fmtDownloads(n) {
  const v = Number(n) || 0;
  if (v >= 1000000) return (v / 1000000).toFixed(1).replace(/\.0$/, '') + 'M';
  if (v >= 1000) return (v / 1000).toFixed(1).replace(/\.0$/, '') + 'k';
  return String(v);
}

function hfRenderResults() {
  const box = $('#hfResults');
  if (!box) return;
  box.innerHTML = '';
  const list = hfNumFiltered();
  $('#hfEmpty').style.display = list.length ? 'none' : '';
  list.forEach((m) => {
    const item = document.createElement('div');
    item.className = 'dl-item';
    const badges = [];
    if (m.params != null) badges.push('<span class="dl-badge b-cpu">' + m.params + 'B</span>');
    if (m.task) badges.push('<span class="dl-badge b-vulkan">' + escapeHtml(m.task) + '</span>');
    if (/-GGUF$/i.test(m.id)) badges.push('<span class="dl-badge b-cuda">GGUF</span>');
    item.innerHTML =
      '<div class="dl-item-main"><div class="dl-item-tag">' + escapeHtml(m.id) + '</div>' +
      '<div class="dl-item-date">' + fmtDownloads(m.downloads) + ' descargas · ♥ ' + m.likes + '</div></div>' +
      '<div class="dl-badges">' + badges.join('') + '</div>';
    item.addEventListener('click', () => hfSelectRepo(m));
    box.appendChild(item);
  });
}

async function hfSearch() {
  if (hf.running) return;
  const query = $('#hfQuery').value.trim();
  const category = $('#hfCategory').value;
  $('#hfErr').classList.add('hidden');
  $('#hfErr').textContent = '';
  $('#hfLoading').style.display = '';
  $('#hfResults').innerHTML = '';
  const res = await window.api.searchHfModels({ query, category });
  $('#hfLoading').style.display = 'none';
  if (!res || !res.ok) {
    $('#hfErr').textContent = (res && res.error) || 'No se pudieron buscar modelos.';
    $('#hfErr').classList.remove('hidden');
    hf.models = [];
    hfRenderResults();
    return;
  }
  hf.models = res.models || [];
  hfRenderResults();
}

async function hfSelectRepo(m) {
  if (hf.running) return;
  $('#hfErr').classList.add('hidden');
  $('#hfNextBtn').disabled = true;
  $('#hfNextBtn').textContent = 'Cargando…';
  const res = await window.api.getHfFiles(m.id);
  $('#hfNextBtn').textContent = 'Descargar';
  if (!res || !res.ok) {
    $('#hfErr').textContent = (res && res.error) || 'No se pudieron listar los archivos.';
    $('#hfErr').classList.remove('hidden');
    return;
  }
  hf.repo = m.id;
  hf.files = res.classified || { main: [], vision: [], mtp: [] };
  hf.quant = null;
  hf.visionFile = null;
  hf.mtpFile = null;
  hfRenderFiles();
  hfShowStep(1);
}

function hfFilesToDownload() {
  const out = [];
  if (hf.quant) out.push(hf.quant);
  if ($('#hfVisionOn') && $('#hfVisionOn').checked && hf.visionFile) out.push(hf.visionFile);
  if ($('#hfMtpOn') && $('#hfMtpOn').checked && hf.mtpFile) out.push(hf.mtpFile);
  return out;
}

function hfQuantHue(t) {
  const x = Math.max(0, Math.min(1, t));
  if (x < 0.5) return Math.round(140 - (140 - 45) * (x / 0.5));
  return Math.round(45 - 45 * ((x - 0.5) / 0.5));
}

function hfRenderFiles() {
  const box = $('#hfQuantList');
  box.innerHTML = '';
  const main = hf.files.main || [];
  $('#hfNoQuant').textContent = main.length ? '' : 'No se encontraron cuantizaciones .gguf en este repositorio.';
  $('#hfNoQuant').style.display = main.length ? 'none' : '';
  const sizes = main.map((f) => f.size || 0).filter(Boolean);
  const minS = sizes.length ? Math.min.apply(null, sizes) : 0;
  const maxS = sizes.length ? Math.max.apply(null, sizes) : 0;
  main.forEach((f) => {
    const label = document.createElement('label');
    label.className = 'vram-option';
    label.title = f.name;
    const t = maxS > minS ? Math.max(0, Math.min(1, ((f.size || 0) - minS) / (maxS - minS))) : 0.5;
    label.style.setProperty('--qc-h', hfQuantHue(t));
    label.innerHTML =
      '<input type="radio" name="hfQuant" value="" />' +
      '<span class="hf-quant-chip"><span class="hf-quant-name">' + escapeHtml(f.quant) + '</span>' +
      '<span class="hf-quant-size">' + (f.size ? fmtBytes(f.size) : '') + '</span></span>';
    label.addEventListener('click', () => {
      hf.quant = f.name;
      box.querySelectorAll('.vram-option').forEach((o) => o.classList.toggle('active', o === label));
      hfUpdateNav();
    });
    box.appendChild(label);
  });

  const vision = hf.files.vision || [];
  const mtp = hf.files.mtp || [];
  $('#hfVisionBox').classList.toggle('hidden', !vision.length);
  $('#hfMtpBox').classList.toggle('hidden', !mtp.length);
  $('#hfVisionOn').checked = true;
  $('#hfMtpOn').checked = true;
  const vs = $('#hfVisionFile');
  vs.innerHTML = vision.map((f) => '<option value="' + escapeHtml(f.name) + '">' + escapeHtml(f.name) + ' · ' + fmtBytes(f.size) + '</option>').join('');
  hf.visionFile = vision.length ? vision[0].name : null;
  const ms = $('#hfMtpFile');
  ms.innerHTML = mtp.map((f) => '<option value="' + escapeHtml(f.name) + '">' + escapeHtml(f.name) + ' · ' + fmtBytes(f.size) + '</option>').join('');
  hf.mtpFile = mtp.length ? mtp[0].name : null;

  const safeFolder = hf.repo.split('/').pop().replace(/[\\/:*?"<>|]+/g, '-');
  $('#hfDest').textContent = (settings.modelsDir ? settings.modelsDir + '\\' : '') + safeFolder;
  hfUpdateNav();
}

function hfSetRunningUI(running, ok) {
  if (running) {
    $('#hfCancelBtn').classList.remove('hidden');
    $('#hfCancelBtn').textContent = 'Cancelar';
    $('#hfDoneBtns').classList.add('hidden');
  } else if (ok) {
    $('#hfCancelBtn').classList.add('hidden');
    $('#hfDoneBtns').classList.remove('hidden');
  } else {
    $('#hfCancelBtn').classList.remove('hidden');
    $('#hfCancelBtn').textContent = 'Cerrar';
    $('#hfDoneBtns').classList.add('hidden');
  }
}

function hfStartDownload() {
  const files = hfFilesToDownload();
  if (!files.length) {
    toast('Elegí al menos la cuantización', 'err');
    return;
  }
  hf._lastT = null;
  hf._lastB = 0;
  hf.running = true;
  hf.lastRepo = hf.repo;
  hf.lastFiles = files;
  hf.lastDir = (settings.modelsDir ? settings.modelsDir.replace(/[\\/]+$/, '') : '') +
    '\\' + hf.repo.split('/').pop().replace(/[\\/:*?"<>|]+/g, '-');
  const safeFolder = hf.repo.split('/').pop().replace(/[\\/:*?"<>|]+/g, '-');
  window.api.startModelDownload({ repo: hf.repo, folder: safeFolder, files }).catch(() => {});
  closeHf();
  showDownloads();
}

function hfHandleProgress(p) {
  const fill = $('#hfProgressFill');
  const pctEl = $('#hfProgressPct');
  const detail = $('#hfProgressDetail');
  $('#hfProgressTitle').textContent = p.label || 'Descargando…';
  if (p.phase === 'download') {
    const received = p.received || 0;
    const total = p.total || 0;
    const pct = total ? Math.min(100, (received / total) * 100) : 0;
    fill.style.width = pct + '%';
    pctEl.textContent = pct.toFixed(0) + '%';
    let speed = 0;
    const now = Date.now();
    if (hf._lastT) {
      const dt = (now - hf._lastT) / 1000;
      const db = received - hf._lastB;
      if (dt > 0 && db >= 0) speed = db / dt;
    }
    hf._lastT = now;
    hf._lastB = received;
    const multi = p.count > 1 ? ' [' + ((p.index || 0) + 1) + '/' + p.count + ']' : '';
    detail.textContent = fmtBytes(received) + ' / ' + fmtBytes(total) + multi + (speed ? ' · ' + fmtBytes(speed) + '/s' : '');
  } else if (p.phase === 'prepare') {
    fill.style.width = '0%';
    pctEl.textContent = '0%';
    detail.textContent = p.file || '';
  }
}

function hfHandleDone(d) {
  hf.running = false;
  const lastFiles = (hf.lastFiles || []).map(String);
  hf.lastVisionName = lastFiles.find((f) => /mmproj|vision-?proj|clip|image/i.test(f)) || null;
  hf.lastMtpName = lastFiles.find((f) => /mtp|draft/i.test(f)) || null;
  if (hf.lastDir && hf.repo && lastFiles.length) {
    if (!settings.hfRepoMap || typeof settings.hfRepoMap !== 'object') settings.hfRepoMap = {};
    const readmeUrl = 'https://huggingface.co/' + hf.repo;
    lastFiles.forEach((f) => {
      settings.hfRepoMap[hf.lastDir + '\\' + f] = readmeUrl;
    });
    window.api.saveSettings(settings).catch(() => {});
  }
  scanAllModels().then((cache) => {
    modelsCache = cache;
    populateModelSelects();
    renderDownloadedModels();
    refresh();
    scheduleFileCheck();
    if (wizard.open) wizardPopulate();
  });
  if (hf.open) {
    $('#hfProgressTitle').textContent = 'Descargado correctamente';
    $('#hfProgressDetail').textContent = '';
    $('#hfProgressFill').style.width = '100%';
    $('#hfProgressFill').classList.add('ok');
    $('#hfProgressPct').textContent = '100%';
    $('#hfResult').innerHTML = '<p class="hint"><span class="text-ok">Modelo guardado en:</span> <b>' + escapeHtml(d.dir || '') + '</b></p>';
    hfSetRunningUI(false, true);
  }
  toast(t('toast_model_downloaded'), 'ok');
}

function hfHandleError(msg) {
  hf.running = false;
  if (hf.open) {
    $('#hfProgressTitle').textContent = 'No se pudo completar';
    $('#hfProgressDetail').textContent = '';
    $('#hfProgressFill').style.width = '0%';
    $('#hfProgressPct').textContent = '';
    $('#hfResult').innerHTML = '<p class="hint text-err">' + escapeHtml(msg) + '</p>';
    hfSetRunningUI(false, false);
  }
}

function hfFinish() {
  closeHf();
}

function openHf() {
  hf.open = true;
  hf.step = 0;
  hf.models = [];
  hf.repo = null;
  hf.files = null;
  hf.quant = null;
  hf.running = false;
  hf._lastT = null;
  hf._lastB = 0;
  $('#hfCategory').innerHTML = HF_CATEGORIES.map((c) => '<option value="' + c.id + '">' + c.label + '</option>').join('');
  $('#hfErr').classList.add('hidden');
  $('#hfErr').textContent = '';
  $('#hfLoading').style.display = 'none';
  $('#hfResults').innerHTML = '';
  $('#hfEmpty').style.display = 'none';
  $('#hfQuery').value = '';
  $('#hfCategory').value = '';
  hf.rangeMin = 0;
  hf.rangeMax = 100;
  hfUpdateRange();
  $('#hfGgufOnly').checked = true;
  $('#hfModal').classList.remove('hidden');
  hfShowStep(0);
  $('#hfQuery').focus();
}

function closeHf() {
  hf.open = false;
  $('#hfModal').classList.add('hidden');
}

/* ---------------- Boot ---------------- */

async function init() {
  buildGeneralPanel();
  renderSchemaPanels();
  [profiles, settings] = await Promise.all([window.api.loadProfiles(), window.api.loadSettings()]);
  applyTheme(settings.theme || 'noche');
  setLang(settings.lang || 'es');
  applyLanguage();
  profiles = profiles.map((p) => {
    const merged = Object.assign(defaultProfile(), p);
    if (!Array.isArray(p.configured)) delete merged.configured;
    return merged;
  });
  profiles.forEach((p) => {
    if (num(p.dryMultiplier) <= 0 && p.dryPenaltyLastN === -1) p.dryPenaltyLastN = null;
  });
  modelsCache = await scanAllModels();
  populateModelSelects();

  const st = await window.api.getServerStatus();
  setRunning(st.running);

  if (profiles.length) selectProfile(profiles[0].id);
  showDashboard();
  scheduleFileCheck();

  window.api.onServerLog(({ stream, text }) => appendLog(text, stream === 'err' ? 'err' : 'out'));
  window.api.onServerExit(({ code, argError }) => {
    runningProfileId = null;
    setRunning(false);
    appendLog(`\n[Proceso terminado — código de salida ${code}]\n`, 'sys');
    if (code !== 0 && argError) {
      renderArgError(argError);
      toast(t('toast_server_arg_error'), 'err');
    } else {
      toast(t('toast_server_finished'), code === 0 ? 'ok' : 'err');
    }
  });
  window.api.onServerState((s) => {
    if (!s.running) runningProfileId = null;
    setRunning(s.running);
  });

  window.api.onLlamaEvent((ev) => {
    if (ev.type === 'progress') dlHandleProgress(ev);
    else if (ev.type === 'done') dlHandleDone(ev.install);
    else if (ev.type === 'cancelled') dlHandleError(ev.message || 'Descarga cancelada.');
    else if (ev.type === 'error') dlHandleError(ev.message || 'Error en la descarga.');
  });

  window.api.onModelEvent((ev) => {
    if (ev.type === 'progress') hfHandleProgress(ev);
    else if (ev.type === 'done') hfHandleDone(ev);
    else if (ev.type === 'cancelled') hfHandleError(ev.message || 'Descarga cancelada.');
    else if (ev.type === 'error') hfHandleError(ev.message || 'Error en la descarga.');
  });

  /* --- Downloads view (event listeners) --- */
  window.api.onDlProgress((d) => {
    dlq.items[d.id] = Object.assign(dlq.items[d.id] || {}, d, { status: d.status || 'active' });
    dlqUpdateBadge();
    if (view === 'downloads') dlqRender();
  });
  window.api.onDlComplete((d) => {
    dlq.items[d.id] = Object.assign(dlq.items[d.id] || {}, d, { status: 'completed' });
    dlqUpdateBadge();
    if (view === 'downloads') dlqRender();
  });
  window.api.onDlError((d) => {
    dlq.items[d.id] = Object.assign(dlq.items[d.id] || {}, d, { status: 'error' });
    dlqUpdateBadge();
    if (view === 'downloads') dlqRender();
  });
  window.api.onDlPaused((d) => {
    dlq.items[d.id] = Object.assign(dlq.items[d.id] || {}, d, { status: 'paused' });
    dlqUpdateBadge();
    if (view === 'downloads') dlqRender();
  });
  window.api.onDlResumed((d) => {
    dlq.items[d.id] = Object.assign(dlq.items[d.id] || {}, d, { status: 'active' });
    dlqUpdateBadge();
    if (view === 'downloads') dlqRender();
  });
  window.api.onDlCancelled((d) => {
    delete dlq.items[d.id];
    dlqUpdateBadge();
    if (view === 'downloads') dlqRender();
  });

  $('#dlSections').addEventListener('click', async (e) => {
    const btn = e.target.closest('[data-dlq]');
    if (!btn) return;
    const action = btn.dataset.dlq;
    const id = btn.dataset.id;
    if (action === 'pause') window.api.dlPause(id);
    else if (action === 'resume') window.api.dlResume(id);
    else if (action === 'cancel') {
      const d = dlq.items[id];
      const name = (d && d.meta && d.meta.file) ? d.meta.file.split('/').pop() : id;
      const ok = await confirmDialog({
        title: 'Cancelar descarga',
        message: '¿Cancelar la descarga de "' + name + '"? Se eliminará el archivo parcial.',
        okLabel: 'Cancelar descarga',
        danger: true,
      });
      if (ok) window.api.dlCancel(id);
    }
    else if (action === 'retry') {
      window.api.dlRetry(id).then((res) => {
        if (!res || !res.ok) toast((res && res.error) || 'No se pudo reintentar', 'err');
      });
    } else if (action === 'open') {
      const d = dlq.items[id];
      if (d && d.dest) {
        const dir = d.dest.replace(/[/\\][^/\\]+$/, '');
        window.api.openPath(dir);
      }
    } else if (action === 'remove') {
      const d = dlq.items[id];
      const name = (d && d.meta && d.meta.file) ? d.meta.file.split('/').pop() : id;
      const ok = await confirmDialog({
        title: 'Quitar de la lista',
        message: '¿Quitar "' + name + '" de la lista? El archivo descargado se conserva.',
        okLabel: 'Quitar',
      });
      if (ok) window.api.dlRemove(id);
    }
  });

  $('#dlClearDoneBtn').addEventListener('click', () => {
    window.api.dlClearCompleted().then((n) => {
      if (n > 0) toast(n + ' descarga(s) limpiada(s)', 'ok');
      dlqRender();
    });
  });
  $('#dlNewBtn').addEventListener('click', openHf);
  $('#dlQueueNavBtn').addEventListener('click', showDownloads);

  window.api.onConfirmClose(async (d) => {
    const ok = await confirmDialog({
      title: 'Descargas en curso',
      message: 'Hay ' + (d.count || '') + ' descarga(s) activa(s). Si cerrás, se pausarán y podrás reanudarlas después. ¿Cerrar la aplicación?',
      okLabel: 'Cerrar y pausar',
      cancelLabel: 'Seguir descargando',
      danger: true,
    });
    window.api.sendCloseResponse(ok);
  });

  window.api.onStats(onStats);
  window.addEventListener('resize', drawStatsChart);
  window.addEventListener('focus', () => scheduleFileCheck());
  drawStatsChart();

  $('#winMinBtn').addEventListener('click', () => window.api.windowControls.minimize());
  $('#winMaxBtn').addEventListener('click', () => window.api.windowControls.maximizeToggle());
  $('#winCloseBtn').addEventListener('click', () => window.api.windowControls.close());
  window.api.windowControls.onMaximized((max) => {
    $('#winMaxBtn').textContent = max ? '❐' : '▢';
    $('#winMaxBtn').title = max ? 'Restaurar' : 'Maximizar';
  });
}

/* ---------------- System stats chart ---------------- */

const MAX_POINTS = 60;
const statsSeries = {
  cpu: { color: '#7c5cff', data: [] },
  ram: { color: '#34d399', data: [] },
  gpu: { color: '#fbbf24', data: [] },
};

function pushStat(arr, v) {
  arr.push(v === null || v === undefined ? null : Number(v));
  if (arr.length > MAX_POINTS) arr.shift();
}

function fmtGigabytes(b) {
  return (b / 1024 / 1024 / 1024).toFixed(1);
}

function onStats(d) {
  pushStat(statsSeries.cpu.data, d.cpu);
  pushStat(statsSeries.ram.data, d.ram && d.ram.total ? (d.ram.used / d.ram.total) * 100 : null);
  pushStat(statsSeries.gpu.data, d.gpu && d.gpu.total ? (d.gpu.used / d.gpu.total) * 100 : null);

  gpuUsedBytes = d.gpu && d.gpu.used ? d.gpu.used : 0;
  gpuTotalBytes = d.gpu && d.gpu.total ? d.gpu.total : 0;
  if (ctxSizeHint && ctxSizeHint.isConnected) updateCtxHint();

  const cpuEl = $('#statCpu');
  const ramEl = $('#statRam');
  const gpuEl = $('#statGpu');
  if (cpuEl) cpuEl.textContent = Math.round(d.cpu) + '%';
  if (ramEl) ramEl.textContent = d.ram ? fmtGigabytes(d.ram.used) + ' / ' + fmtGigabytes(d.ram.total) + ' GB' : 'N/A';
  if (gpuEl) gpuEl.textContent = d.gpu ? fmtGigabytes(d.gpu.used) + ' / ' + fmtGigabytes(d.gpu.total) + ' GB' : 'N/A';
  drawStatsChart();
}

function drawStatsChart() {
  const canvas = $('#statsCanvas');
  if (!canvas) return;
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(50, Math.floor(canvas.clientWidth));
  const h = Math.max(40, Math.floor(canvas.clientHeight));
  const pw = Math.floor(w * dpr);
  const ph = Math.floor(h * dpr);
  if (canvas.width !== pw || canvas.height !== ph) {
    canvas.width = pw;
    canvas.height = ph;
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  ctx.strokeStyle = 'rgba(139,148,163,0.14)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = Math.round((h * i) / 4) + 0.5;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  drawSeriesLine(ctx, statsSeries.cpu, w, h);
  drawSeriesLine(ctx, statsSeries.ram, w, h);
  drawSeriesLine(ctx, statsSeries.gpu, w, h);
}

function drawSeriesLine(ctx, series, w, h) {
  const data = series.data;
  if (data.length < 2) return;
  const stepX = w / (MAX_POINTS - 1);
  ctx.strokeStyle = series.color;
  ctx.lineWidth = 2;
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.beginPath();
  let pen = false;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (v === null || v === undefined) {
      pen = false;
      continue;
    }
    const x = w - (MAX_POINTS - 1 - i) * stepX;
    const y = h - 3 - (Math.max(0, Math.min(100, v)) / 100) * (h - 6);
    if (!pen) {
      ctx.moveTo(x, y);
      pen = true;
    } else {
      ctx.lineTo(x, y);
    }
  }
  ctx.stroke();
}

/* ---------------- Downloads view ---------------- */

const dlq = { items: {} };

function showDownloads() {
  view = 'downloads';
  $('#dashboardView').classList.add('hidden');
  $('#boardView').classList.add('hidden');
  $('#editorView').classList.add('hidden');
  $('#downloadsView').classList.remove('hidden');
  $('#dashTopbar').classList.add('hidden');
  $('#dashNavBtn').classList.remove('active');
  $('#dlQueueNavBtn').classList.add('active');
  window.api.dlListAll().then((items) => {
    dlq.items = {};
    for (const d of items) dlq.items[d.id] = d;
    dlqRender();
  });
  dlqRender();
}

function dlqUpdateBadge() {
  const active = Object.values(dlq.items).filter((d) => d.status === 'active' || d.status === 'paused' || d.status === 'pending');
  $('#dlQueueNavBtn').textContent = active.length > 0 ? 'Descargas (' + active.length + ')' : 'Descargas';
}

function dlqFmtBytes(b) {
  if (!b || b <= 0) return '0 B';
  if (b < 1024) return b + ' B';
  if (b < 1048576) return (b / 1024).toFixed(1) + ' KB';
  if (b < 1073741824) return (b / 1048576).toFixed(1) + ' MB';
  return (b / 1073741824).toFixed(1) + ' GB';
}

function dlqFmtSpeed(bps) {
  if (!bps || bps <= 0) return '';
  return ' · ' + dlqFmtBytes(bps) + '/s';
}

function dlqRender() {
  const items = Object.values(dlq.items);
  const active = items.filter((d) => d.status === 'active' || d.status === 'pending');
  const paused = items.filter((d) => d.status === 'paused');
  const errors = items.filter((d) => d.status === 'error');
  const done = items.filter((d) => d.status === 'completed');

  const hasAny = items.length > 0;
  $('#dlEmptyState').classList.toggle('hidden', hasAny);
  $('#dlActiveSection').classList.toggle('hidden', active.length === 0);
  $('#dlPausedSection').classList.toggle('hidden', paused.length === 0);
  $('#dlErrorSection').classList.toggle('hidden', errors.length === 0);
  $('#dlDoneSection').classList.toggle('hidden', done.length === 0);

  $('#dlActiveCount').textContent = active.length;
  $('#dlPausedCount').textContent = paused.length;
  $('#dlErrorCount').textContent = errors.length;
  $('#dlDoneCount').textContent = done.length;

  $('#dlActiveList').innerHTML = active.map(dlqRenderItem).join('');
  $('#dlPausedList').innerHTML = paused.map(dlqRenderItem).join('');
  $('#dlErrorList').innerHTML = errors.map(dlqRenderItem).join('');
  $('#dlDoneList').innerHTML = done.map(dlqRenderItem).join('');
}

function dlqRenderItem(d) {
  const pct = d.totalBytes ? Math.min(100, (d.receivedBytes / d.totalBytes) * 100) : 0;
  const name = (d.meta && d.meta.file) ? d.meta.file.split('/').pop() : (d.dest || d.url || '').split(/[/\\]/).pop() || d.id;
  const statusClass = d.status || 'active';
  const statusLabel = { active: 'Descargando', paused: 'Pausado', error: 'Error', completed: 'Completado', pending: 'En cola' }[statusClass] || statusClass;
  const idAttr = ' data-id="' + escapeHtml(d.id) + '"';

  let actions = '';
  if (d.status === 'active') {
    actions = '<button class="dlq-action-btn" data-dlq="pause" title="Pausar"' + idAttr + '>⏸</button>';
    actions += '<button class="dlq-action-btn danger" data-dlq="cancel" title="Cancelar"' + idAttr + '>✕</button>';
  } else if (d.status === 'pending') {
    actions = '<button class="dlq-action-btn danger" data-dlq="cancel" title="Cancelar"' + idAttr + '>✕</button>';
  } else if (d.status === 'paused') {
    actions = '<button class="dlq-action-btn" data-dlq="resume" title="Reanudar"' + idAttr + '>▶</button>';
    actions += '<button class="dlq-action-btn danger" data-dlq="cancel" title="Cancelar"' + idAttr + '>✕</button>';
  } else if (d.status === 'error') {
    actions = '<button class="dlq-action-btn" data-dlq="retry" title="Reintentar"' + idAttr + '>↻</button>';
    actions += '<button class="dlq-action-btn danger" data-dlq="cancel" title="Eliminar"' + idAttr + '>✕</button>';
  } else if (d.status === 'completed') {
    actions = '<button class="dlq-action-btn" data-dlq="open" title="Abrir carpeta"' + idAttr + '>📂</button>';
    actions += '<button class="dlq-action-btn danger" data-dlq="remove" title="Quitar de la lista"' + idAttr + '>✕</button>';
  }

  const iconMap = { active: '⬇', paused: '⏸', error: '⚠', completed: '✓', pending: '⏳' };
  const icon = iconMap[statusClass] || '⬇';

  let meta = '';
  if (d.status === 'completed') {
    meta = dlqFmtBytes(d.totalBytes || d.receivedBytes);
  } else if (d.status === 'pending') {
    meta = d.meta && d.meta.count > 1 ? 'Archivo ' + ((d.meta.index || 0) + 1) + ' de ' + d.meta.count : 'En espera';
  } else {
    meta = dlqFmtBytes(d.receivedBytes || 0) + ' / ' + dlqFmtBytes(d.totalBytes || 0);
    if (d.status === 'active' && d.speed) meta += dlqFmtSpeed(d.speed);
  }

  const errorHtml = d.error ? '<div class="dl-item-error">' + escapeHtml(d.error) + '</div>' : '';

  return '<div class="dl-item">' +
    '<div class="dl-item-icon ' + statusClass + '">' + icon + '</div>' +
    '<div class="dl-item-body">' +
      '<div class="dl-item-top">' +
        '<span class="dl-item-name" title="' + escapeHtml(name) + '">' + escapeHtml(name) + '</span>' +
        '<span class="dl-item-status ' + statusClass + '">' + statusLabel + '</span>' +
      '</div>' +
      '<div class="dl-item-progress"><div class="dl-item-progress-fill ' + statusClass + '" style="width:' + pct + '%"></div></div>' +
      '<div class="dl-item-meta">' +
        '<span>' + meta + '</span>' +
        '<div class="dl-item-actions">' + actions + '</div>' +
      '</div>' +
      errorHtml +
    '</div>' +
  '</div>';
}

/* ---------------- Events ---------------- */

document.addEventListener('DOMContentLoaded', () => {
  init();

  const createNewCommand = () => {
    if (dashSelMode) exitDashSelection();
    openBoardForNewProfile();
  };

  $('#dashNavBtn').addEventListener('click', () => {
    if (dashSelMode) exitDashSelection();
    showDashboard();
  });
  $('#backBtn').addEventListener('click', showDashboard);
  $('#dashSelBtn').addEventListener('click', () => (dashSelMode ? exitDashSelection() : enterDashSelection()));
  $('#dashSelAllBtn').addEventListener('click', () => {
    if (dashSelected.size === profiles.length) dashSelected.clear();
    else profiles.forEach((p) => dashSelected.add(p.id));
    updateDashSelBar();
    renderDashboard();
  });
  $('#dashSelNoneBtn').addEventListener('click', () => {
    dashSelected.clear();
    updateDashSelBar();
    renderDashboard();
  });
  $('#dashSelDeleteBtn').addEventListener('click', deleteSelected);
  $('#dashSelCancelBtn').addEventListener('click', exitDashSelection);
  $('#newBtn').addEventListener('click', createNewCommand);
  $('#dashNewBtn').addEventListener('click', () => {
    if (dashSelMode) exitDashSelection();
    createNewCommand();
  });
  $('#dashEmptyNewBtn').addEventListener('click', () => {
    if (dashSelMode) exitDashSelection();
    createNewCommand();
  });
  $('#boardBackBtn').addEventListener('click', showDashboard);
  $('#boardDetailsBtn').addEventListener('click', showEditor);
  $('#boardWizardBtn').addEventListener('click', openWizard);
  $('#editorBoardBtn').addEventListener('click', showBoard);
  document.querySelectorAll('.board-tab').forEach((b) => b.addEventListener('click', () => setBoardTab(b.dataset.boardTab)));
  $('#boardNameInput').addEventListener('input', (e) => {
    if (current()) {
      current().name = e.target.value;
      renderSidebar();
      renderDashboard();
      scheduleSave();
    }
  });
  $('#boardSaveBtn').addEventListener('click', () => {
    scheduleSave();
    renderSidebar();
    renderDashboard();
    toast(t('toast_settings_saved'), 'ok');
  });
  $('#boardRunBtn').addEventListener('click', toggleServer);
  $('#boardRescanBtn').addEventListener('click', async () => {
    modelsCache = await scanAllModels();
    populateModelSelects();
    renderBoard();
    toast('Modelos re-escaneados (' + modelsCache.length + ')', 'ok');
  });
  $('#boardView').addEventListener('dragstart', (e) => {
    if (e.target.closest('button')) {
      e.preventDefault();
      return;
    }
    const block = e.target.closest('[data-board-block]');
    if (block) {
      e.dataTransfer.effectAllowed = 'move';
      e.dataTransfer.setData('application/json', JSON.stringify({ type: block.dataset.boardBlock, assigned: true }));
      return;
    }
    const item = e.target.closest('[data-board-type]');
    if (!item) return;
    e.dataTransfer.effectAllowed = 'copy';
    e.dataTransfer.setData('application/json', JSON.stringify({ type: item.dataset.boardType, path: item.dataset.path || '' }));
  });
  $('#boardDropzone').addEventListener('dragover', (e) => {
    e.preventDefault();
    $('#boardDropzone').classList.add('drag-over');
  });
  $('#boardDropzone').addEventListener('dragleave', (e) => {
    if (!$('#boardDropzone').contains(e.relatedTarget)) $('#boardDropzone').classList.remove('drag-over');
  });
  $('#boardDropzone').addEventListener('drop', (e) => {
    e.preventDefault();
    $('#boardDropzone').classList.remove('drag-over');
    let data = null;
    try { data = JSON.parse(e.dataTransfer.getData('application/json') || '{}'); } catch (err) {}
    if (data && data.assigned) return;
    if (data && data.type) boardOpenConfig(data.type, data);
  });
  $('#boardAssigned').addEventListener('click', (e) => {
    const edit = e.target.closest('[data-board-edit]');
    if (edit) boardOpenConfig(edit.dataset.boardEdit, {});
  });
  $('.board-palette').addEventListener('dragover', (e) => {
    e.preventDefault();
    $('.board-palette').classList.add('drag-over');
  });
  $('.board-palette').addEventListener('dragleave', (e) => {
    if (!$('.board-palette').contains(e.relatedTarget)) $('.board-palette').classList.remove('drag-over');
  });
  $('.board-palette').addEventListener('drop', (e) => {
    e.preventDefault();
    $('.board-palette').classList.remove('drag-over');
    let data = null;
    try { data = JSON.parse(e.dataTransfer.getData('application/json') || '{}'); } catch (err) {}
    if (data && data.assigned && data.type) boardRemoveBlock(data.type);
  });
  $('#boardConfigClose').addEventListener('click', boardCloseConfig);
  $('#boardConfigCancel').addEventListener('click', boardCloseConfig);
  $('#boardConfigApply').addEventListener('click', boardApplyConfig);
  $('#boardConfigModal').addEventListener('click', (e) => {
    if (e.target === $('#boardConfigModal')) boardCloseConfig();
  });
  $('#gearBtn').addEventListener('click', openSettings);
  $('#dupBtn').addEventListener('click', duplicateProfile);
  $('#delBtn').addEventListener('click', deleteProfile);
  $('#nameInput').addEventListener('input', (e) => {
    if (current()) {
      current().name = e.target.value;
      renderSidebar();
      scheduleSave();
    }
  });

  function setTab(name) {
    document.querySelectorAll('.tabs .tab').forEach((b) => b.classList.toggle('active', b.dataset.tab === name));
    document.querySelectorAll('.panel').forEach((p) => p.classList.remove('active'));
    const panel = $('#panel-' + name);
    if (panel) panel.classList.add('active');
    if (name === 'command') renderCommandPanel();
  }

  document.querySelectorAll('.tabs .tab').forEach((btn) => {
    btn.addEventListener('click', () => setTab(btn.dataset.tab));
  });

  $('#cmdApplyBtn').addEventListener('click', applyCommandEdit);
  $('#cmdResetBtn').addEventListener('click', renderCommandPanel);

  $('#runBtn').addEventListener('click', toggleServer);

  $('#argErrorClose').addEventListener('click', () => $('#argErrorBox').classList.add('hidden'));
  $('#argErrorFixBtn').addEventListener('click', () => {
    const btn = $('#argErrorFixBtn');
    const p = current();
    const key = btn.dataset.key;
    if (!key || !p) return;
    const raw = btn.dataset.val;
    const n = num(raw);
    const val = n !== null ? n : raw;
    const old = btn.dataset.old;
    markConfigured(p, key);
    p[key] = val;
    scheduleSave();
    renderSchemaPanels();
    applyFieldVisibility();
    refresh();
    appendLog('\n→ Corregido automáticamente: ' + btn.dataset.flag + ' = ' + String(val) + ' (era ' + old + '). Volvé a iniciar.\n', 'sys');
    $('#argErrorBox').classList.add('hidden');
    toast('Parámetro corregido', 'ok');
  });
  $('#argErrorFieldBtn').addEventListener('click', () => {
    const btn = $('#argErrorFieldBtn');
    const key = btn.dataset.key;
    if (!key) return;
    const info = argFixInfo({ flag: btn.dataset.flag });
    if (info.panel) setTab(info.panel);
    focusFieldCard(key);
  });

  $('#bottomCollapseBtn').addEventListener('click', () => {
    $('#statsBar').classList.toggle('collapsed');
  });
  $('#bottomTabRecursos').addEventListener('click', () => showBottomTab('recursos'));
  $('#bottomTabConsola').addEventListener('click', () => showBottomTab('consola'));

  /* --- Wizard --- */
  $('#wizardClose').addEventListener('click', closeWizard);
  $('#wizardModal').addEventListener('click', (e) => {
    if (e.target === $('#wizardModal')) closeWizard();
  });
  $('#wizardBackBtn').addEventListener('click', () => wizardShowStep(wizard.step - 1));
  $('#wizardNextBtn').addEventListener('click', () => {
    if (wizardCanProceed()) wizardShowStep(wizard.step + 1);
  });
  $('#wizManageInstalls').addEventListener('click', () => {
    openSettings();
    wizardPopulate();
  });
  $('#wizInstall').addEventListener('change', (e) => {
    wizard.installId = e.target.value || null;
    wizardUpdateNav();
  });
  $('#wizModel').addEventListener('change', async (e) => {
    wizard.modelPath = e.target.value || '';
    wizardRefreshModelInfo();
    wizardAutoFillFromDownload();
    await wizardApplyHfFromMap();
    wizardUpdateNav();
  });
  $('#wizBrowseModel').addEventListener('click', async () => {
    const f = await window.api.selectModelFile();
    if (f) {
      wizard.modelPath = f;
      wizardAddOption($('#wizModel'), f);
      wizardRefreshModelInfo();
      wizardAutoFillFromDownload();
      await wizardApplyHfFromMap();
      wizardUpdateNav();
    }
  });
  $('#wizCtx').addEventListener('input', (e) => {
    wizard.ctxSize = e.target.value;
    const row = $('#wizCtxPresets');
    const n = num(e.target.value);
    if (row) row.querySelectorAll('.preset-btn').forEach((b) => b.classList.toggle('active', Number(b.dataset.ctx) === n));
    wizardCtxHint();
  });
  $('#wizHfRecommendBtn').addEventListener('click', wizardApplyHfRecommended);
  $('#wizHfRepo').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      wizardApplyHfRecommended();
    }
  });
  $('#wizVision').addEventListener('change', (e) => {
    $('#wizVisionRow').style.display = e.target.checked ? '' : 'none';
  });
  $('#wizVisionFile').addEventListener('change', (e) => {
    wizard.mmprojPath = e.target.value || '';
  });
  $('#wizBrowseVision').addEventListener('click', async () => {
    const f = await window.api.selectVisionFile();
    if (f) {
      wizard.mmprojPath = f;
      wizardAddOption($('#wizVisionFile'), f);
    }
  });
  $('#wizMtp').addEventListener('change', (e) => {
    wizard.mtp = e.target.checked;
    $('#wizMtpRow').style.display = e.target.checked ? '' : 'none';
    wizardUpdateNav();
  });
  document.querySelectorAll('input[name="wizMtpMode"]').forEach((r) => {
    r.addEventListener('change', (e) => {
      wizard.mtpMode = e.target.value;
      $('#wizMtpFileRow').style.display = e.target.value === 'file' ? '' : 'none';
      wizardUpdateNav();
    });
  });
  $('#wizMtpFile').addEventListener('change', (e) => {
    wizard.mtpFile = e.target.value || '';
    wizardUpdateNav();
  });
  $('#wizBrowseMtp').addEventListener('click', async () => {
    const f = await window.api.selectModelFile();
    if (f) {
      wizard.mtpFile = f;
      wizardAddOption($('#wizMtpFile'), f);
      wizardUpdateNav();
    }
  });
  $('#wizHost').addEventListener('input', (e) => {
    wizard.host = e.target.value;
    wizardUpdateNav();
  });
  $('#wizPort').addEventListener('input', (e) => {
    wizard.port = e.target.value;
    wizardUpdateNav();
  });
  $('#wizardFinishBtn').addEventListener('click', () => wizardFinish(false));
  $('#wizardRunBtn').addEventListener('click', () => wizardFinish(true));
  $('#wizardDetailsBtn').addEventListener('click', wizardDetails);

  /* --- Descargador de llama.cpp --- */
  const dlTryClose = () => {
    if (dl.running) {
        toast(t('toast_dl_cancel'), 'err');
      return;
    }
    closeDownloader();
  };
  $('#dlClose').addEventListener('click', dlTryClose);
  $('#dlModal').addEventListener('click', (e) => {
    if (e.target === $('#dlModal')) dlTryClose();
  });
  $('#dlBackBtn').addEventListener('click', () => dlShowStep(dl.step - 1));
  $('#dlNextBtn').addEventListener('click', () => {
    if (!dl.tag || (dl.step === 1 && !dl.backend)) return;
    if (dl.step === 0) {
      dlShowStep(1);
    } else if (dl.step === 1) {
      dlShowStep(2);
      dlStartDownload();
    }
  });
  $('#dlCancelBtn').addEventListener('click', () => {
    if (dl.running) {
      window.api.cancelLlamaDownload();
      toast('Cancelando descarga…');
    } else {
      closeDownloader();
    }
  });
  $('#dlFinishBtn').addEventListener('click', dlFinish);
  $('#dlUseBtn').addEventListener('click', dlUse);
  $('#downloadInstallBtn').addEventListener('click', () => {
    closeSettings();
    openDownloader();
  });
  $('#wizDownloadInstall').addEventListener('click', openDownloader);

  $('#modalClose').addEventListener('click', closeSettings);
  $('#modal').addEventListener('click', (e) => {
    if (e.target === $('#modal')) closeSettings();
  });
  $('#installModal').addEventListener('click', (e) => {
    if (e.target === $('#installModal')) closeInstallEditor();
  });

  $('#addInstallBtn').addEventListener('click', () => openInstallEditor(null));
  $('#installModalClose').addEventListener('click', closeInstallEditor);
  $('#installBrowseBtn').addEventListener('click', async () => {
    const d = await window.api.selectDirectory();
    if (!d) return;
    $('#installPath').value = d;
    const exe = await window.api.verifyServerFolder(d);
    $('#installPathHint').textContent = exe
      ? 'Encontrado: ' + exe
      : 'No se encontró llama-server.exe en esta carpeta.';
    $('#installPathHint').className = 'hint ' + (exe ? 'text-ok' : 'text-err');
  });
  $('#installSaveBtn').addEventListener('click', async () => {
    const name = $('#installName').value.trim();
    const folder = $('#installPath').value.trim();
    if (!name) {
      toast('Poné un nombre a la versión', 'err');
      return;
    }
    if (!folder) {
      toast('Elegí la carpeta de llama.cpp', 'err');
      return;
    }
    const exe = await window.api.verifyServerFolder(folder);
    if (!exe) {
      toast('Esa carpeta no contiene llama-server.exe', 'err');
      return;
    }
    if (!Array.isArray(settings.installations)) settings.installations = [];
    if (editingInstallId) {
      const i = installationById(editingInstallId);
      if (i) {
        i.name = name;
        i.path = folder;
        i.exePath = exe;
      }
    } else {
      settings.installations.push({ id: uid(), name, path: folder, exePath: exe });
    }
    closeInstallEditor();
    renderInstallList();
    refresh();
    scheduleFileCheck();
    toast(t('toast_settings_saved'), 'ok');
  });

  $('#browseDirBtn').addEventListener('click', async () => {
    const d = await window.api.selectDirectory();
    if (d) $('#modelsDir').value = d;
  });
  $('#saveSettingsBtn').addEventListener('click', async () => {
    settings.modelsDir = $('#modelsDir').value.trim();
    settings.theme = settings.theme || 'noche';
    settings.lang = $('#langSelect').value || 'es';
    setLang(settings.lang);
    applyLanguage();
    await window.api.saveSettings(settings);
    closeSettings();
    refresh();
    scheduleFileCheck();
    scanModels();
    if (wizard.open) wizardPopulate();
    toast(t('toast_settings_saved'), 'ok');
  });

  /* --- Settings tabs --- */
  document.querySelectorAll('.settings-tab').forEach((b) => {
    b.addEventListener('click', () => setSettingsTab(b.dataset.settingsTab));
  });
  $('#scanModelsBtn').addEventListener('click', async () => {
    modelsCache = await scanAllModels();
    populateModelSelects();
    renderDownloadedModels();
    toast('Modelos re-escaneados (' + modelsCache.length + ')', 'ok');
  });
  $('#openModelsDirBtn').addEventListener('click', async () => {
    const dir = (settings.modelsDir || '').trim();
    if (!dir) {
      toast(t('toast_config_models_dir_first'), 'err');
      openSettings();
      return;
    }
    const res = await window.api.openPath(dir);
    if (!res || !res.ok) toast((res && res.error) || 'No se pudo abrir la carpeta', 'err');
  });

  /* --- Descargador de HuggingFace --- */
  const hfTryClose = () => {
    if (hf.running) {
      closeHf();
      showDownloads();
      toast('La descarga continúa en el gestor', 'ok');
      return;
    }
    closeHf();
  };
  $('#hfClose').addEventListener('click', hfTryClose);
  $('#hfModal').addEventListener('click', (e) => {
    if (e.target === $('#hfModal')) hfTryClose();
  });
  $('#hfOpenBtn').addEventListener('click', openHf);
  $('#wizHfBtn').addEventListener('click', () => {
    closeWizard();
    openHf();
  });
  $('#hfSearchBtn').addEventListener('click', hfSearch);
  $('#hfQuery').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') hfSearch();
  });
  $('#hfCategory').addEventListener('change', hfSearch);
  initHfRange();
  $('#hfGgufOnly').addEventListener('change', hfRenderResults);
  $('#hfBackBtn').addEventListener('click', () => hfShowStep(hf.step - 1));
  $('#hfNextBtn').addEventListener('click', () => {
    if (hf.step === 1 && hf.quant) {
      hfShowStep(2);
      hfStartDownload();
    }
  });
  $('#hfCancelBtn').addEventListener('click', () => {
    if (hf.running) {
      window.api.cancelModelDownload();
      toast('Cancelando descarga…');
    } else {
      closeHf();
    }
  });
  $('#hfFinishBtn').addEventListener('click', hfFinish);
  $('#hfVisionOn').addEventListener('change', hfUpdateNav);
  $('#hfMtpOn').addEventListener('change', hfUpdateNav);
  $('#hfVisionFile').addEventListener('change', (e) => {
    hf.visionFile = e.target.value;
    hfUpdateNav();
  });
  $('#hfMtpFile').addEventListener('change', (e) => {
    hf.mtpFile = e.target.value;
    hfUpdateNav();
  });
});
