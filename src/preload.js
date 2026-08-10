const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('api', {
  selectModelFile: () =>
    ipcRenderer.invoke('dialog:selectFile', {
      title: 'Seleccionar modelo GGUF',
      filters: [{ name: 'Modelos GGUF', extensions: ['gguf'] }],
    }),
  selectVisionFile: () =>
    ipcRenderer.invoke('dialog:selectFile', {
      title: 'Seleccionar archivo de visión (mmproj)',
      filters: [{ name: 'Modelos GGUF', extensions: ['gguf'] }],
    }),
  selectServerExe: () =>
    ipcRenderer.invoke('dialog:selectFile', {
      title: 'Seleccionar llama-server.exe',
      filters: [{ name: 'Ejecutables', extensions: ['exe'] }],
    }),
  selectDirectory: () =>
    ipcRenderer.invoke('dialog:selectDirectory', {
      title: 'Seleccionar carpeta',
    }),
  scanModels: (dir) => ipcRenderer.invoke('models:scan', dir),
  deleteModelsFolder: (dir) => ipcRenderer.invoke('models:deleteFolder', dir),
  inspectModel: (filePath) => ipcRenderer.invoke('model:inspect', filePath),
  verifyServerFolder: (dir) => ipcRenderer.invoke('folder:serverExe', dir),
  checkFiles: (paths) => ipcRenderer.invoke('fs:exists', paths),
  openExternal: (url) => ipcRenderer.invoke('shell:openExternal', url),
  loadProfiles: () => ipcRenderer.invoke('profiles:load'),
  saveProfiles: (profiles) => ipcRenderer.invoke('profiles:save', profiles),
  loadSettings: () => ipcRenderer.invoke('settings:load'),
  saveSettings: (settings) => ipcRenderer.invoke('settings:save', settings),
  detectServer: () => ipcRenderer.invoke('settings:findServer'),
  startServer: (cmd) => ipcRenderer.invoke('server:start', cmd),
  stopServer: () => ipcRenderer.invoke('server:stop'),
  getServerStatus: () => ipcRenderer.invoke('server:status'),
  listLlamaReleases: () => ipcRenderer.invoke('llama:listReleases'),
  startLlamaDownload: (opts) => ipcRenderer.invoke('llama:start', opts),
  cancelLlamaDownload: () => ipcRenderer.invoke('llama:cancel'),
  getLlamaDownloadDir: () => ipcRenderer.invoke('llama:downloadDir'),
  deleteInstall: (id) => ipcRenderer.invoke('llama:delete', id),
  onLlamaEvent: (cb) => ipcRenderer.on('llama:event', (_e, d) => cb(d)),
  searchHfModels: (opts) => ipcRenderer.invoke('hf:search', opts),
  getHfFiles: (repo) => ipcRenderer.invoke('hf:files', repo),
  startModelDownload: (opts) => ipcRenderer.invoke('hf:start', opts),
  cancelModelDownload: () => ipcRenderer.invoke('hf:cancel'),
  getHfModelConfig: (repo) => ipcRenderer.invoke('hf:modelConfig', repo),
  onModelEvent: (cb) => ipcRenderer.on('model:event', (_e, d) => cb(d)),
  onServerLog: (cb) => ipcRenderer.on('server:log', (_e, d) => cb(d)),
  onServerExit: (cb) => ipcRenderer.on('server:exit', (_e, d) => cb(d)),
  onServerState: (cb) => ipcRenderer.on('server:state', (_e, d) => cb(d)),
  onStats: (cb) => ipcRenderer.on('stats:update', (_e, d) => cb(d)),
  windowControls: {
    minimize: () => ipcRenderer.send('window:minimize'),
    maximizeToggle: () => ipcRenderer.send('window:maximizeToggle'),
    close: () => ipcRenderer.send('window:close'),
    onMaximized: (cb) => ipcRenderer.on('window:maximized', (_e, v) => cb(v)),
  },
});
