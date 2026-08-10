const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('bimaStorage', {
  abortFile: (token) => ipcRenderer.invoke('bima-storage:abort', token),
  appendFile: (token, chunk) => ipcRenderer.invoke('bima-storage:append', token, chunk),
  beginFile: (relativePath, expectedBytes) => ipcRenderer.invoke('bima-storage:begin', relativePath, expectedBytes),
  chooseFolder: () => ipcRenderer.invoke('bima-storage:choose-folder'),
  finishFile: (token) => ipcRenderer.invoke('bima-storage:finish', token),
  getFolder: () => ipcRenderer.invoke('bima-storage:get-folder'),
});
