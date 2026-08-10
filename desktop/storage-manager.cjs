const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');

const CONFIG_VERSION = 1;

function resolveArchivePath(root, relativePath) {
  if (typeof relativePath !== 'string' || !relativePath.trim() || path.isAbsolute(relativePath)) {
    throw new Error('Archive path must be a non-empty relative path');
  }
  const normalized = relativePath.replaceAll('\\', '/');
  const segments = normalized.split('/');
  if (segments.some((segment) => !segment || segment === '.' || segment === '..' || segment.includes('\0'))) {
    throw new Error('Archive path contains an unsafe segment');
  }
  const destination = path.resolve(root, ...segments);
  const rootWithSeparator = `${path.resolve(root)}${path.sep}`;
  if (!destination.startsWith(rootWithSeparator)) throw new Error('Archive path escapes the selected folder');
  return destination;
}

function createStorageManager(configPath) {
  const writes = new Map();

  function readConfig() {
    try {
      const parsed = JSON.parse(fs.readFileSync(configPath, 'utf8'));
      if (parsed?.version === CONFIG_VERSION && typeof parsed.directory === 'string' && path.isAbsolute(parsed.directory)) {
        return parsed;
      }
    } catch { /* no saved folder yet */ }
    return { version: CONFIG_VERSION, directory: '' };
  }

  function getInfo() {
    const { directory } = readConfig();
    return { configured: !!directory, directory, available: !!directory && fs.existsSync(directory) };
  }

  function setDirectory(directory) {
    if (typeof directory !== 'string' || !path.isAbsolute(directory)) throw new Error('A full folder path is required');
    fs.mkdirSync(directory, { recursive: true });
    fs.accessSync(directory, fs.constants.R_OK | fs.constants.W_OK);
    fs.mkdirSync(path.dirname(configPath), { recursive: true });
    const temp = `${configPath}.${crypto.randomUUID()}.tmp`;
    fs.writeFileSync(temp, JSON.stringify({ version: CONFIG_VERSION, directory }, null, 2));
    fs.renameSync(temp, configPath);
    return getInfo();
  }

  function begin(relativePath, expectedBytes = 0) {
    const info = getInfo();
    if (!info.configured) throw new Error('Choose a BIMA data folder before exporting');
    if (!info.available) throw new Error('The selected BIMA data folder is unavailable');
    const destination = resolveArchivePath(info.directory, relativePath);
    fs.mkdirSync(path.dirname(destination), { recursive: true });
    const token = crypto.randomUUID();
    const temp = `${destination}.${token}.partial`;
    const handle = fs.openSync(temp, 'wx');
    writes.set(token, {
      destination,
      expectedBytes: Number.isSafeInteger(expectedBytes) && expectedBytes >= 0 ? expectedBytes : 0,
      handle,
      hash: crypto.createHash('sha256'),
      relativePath: relativePath.replaceAll('\\', '/'),
      temp,
      written: 0,
    });
    return token;
  }

  function append(token, chunk) {
    const pending = writes.get(token);
    if (!pending) throw new Error('Archive write is not active');
    const bytes = Buffer.from(chunk);
    fs.writeSync(pending.handle, bytes);
    pending.hash.update(bytes);
    pending.written += bytes.length;
    if (pending.expectedBytes && pending.written > pending.expectedBytes) {
      abort(token);
      throw new Error('Archive write exceeded its declared size');
    }
    return pending.written;
  }

  function finish(token) {
    const pending = writes.get(token);
    if (!pending) throw new Error('Archive write is not active');
    try {
      if (pending.expectedBytes && pending.written !== pending.expectedBytes) {
        throw new Error(`Archive write was truncated (${pending.written}/${pending.expectedBytes} bytes)`);
      }
      fs.fsyncSync(pending.handle);
      fs.closeSync(pending.handle);
      pending.handle = null;
      fs.renameSync(pending.temp, pending.destination);
      writes.delete(token);
      return {
        bytes: pending.written,
        relativePath: pending.relativePath,
        sha256: pending.hash.digest('hex'),
      };
    } catch (error) {
      abort(token);
      throw error;
    }
  }

  function abort(token) {
    const pending = writes.get(token);
    if (!pending) return false;
    writes.delete(token);
    if (pending.handle !== null) {
      try { fs.closeSync(pending.handle); } catch { /* already closed */ }
    }
    try { fs.rmSync(pending.temp, { force: true }); } catch { /* best effort */ }
    return true;
  }

  function abortAll() {
    for (const token of [...writes.keys()]) abort(token);
  }

  return { abort, abortAll, append, begin, finish, getInfo, setDirectory };
}

module.exports = { createStorageManager, resolveArchivePath };
