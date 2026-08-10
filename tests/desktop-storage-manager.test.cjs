const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');
const { createStorageManager, resolveArchivePath } = require('../desktop/storage-manager.cjs');

test('desktop archive streams an integrity-checked file into the selected folder', () => {
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'bima-storage-'));
  try {
    const archive = path.join(temp, 'chosen-data');
    const manager = createStorageManager(path.join(temp, 'profile', 'storage.json'));
    assert.deepEqual(manager.getInfo(), { configured: false, directory: '', available: false });
    assert.equal(manager.setDirectory(archive).available, true);

    const bytes = Buffer.from('BIMA-data');
    const token = manager.begin('participants/sub-0001/sessions/ses-test/sensor/data.csv', bytes.length);
    manager.append(token, bytes.subarray(0, 4));
    manager.append(token, bytes.subarray(4));
    const result = manager.finish(token);
    assert.equal(result.bytes, bytes.length);
    assert.equal(result.sha256, crypto.createHash('sha256').update(bytes).digest('hex'));
    assert.equal(fs.readFileSync(path.join(archive, ...result.relativePath.split('/')), 'utf8'), 'BIMA-data');
    assert.equal(manager.getInfo().directory, archive);

    const changed = path.join(temp, 'changed-data');
    assert.equal(manager.setDirectory(changed).directory, changed);
    assert.equal(manager.getInfo().directory, changed);
  } finally {
    fs.rmSync(temp, { recursive: true, force: true });
  }
});

test('desktop archive rejects traversal and removes truncated partial writes', () => {
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'bima-storage-'));
  try {
    const archive = path.join(temp, 'archive');
    const manager = createStorageManager(path.join(temp, 'profile', 'storage.json'));
    manager.setDirectory(archive);
    assert.throws(() => resolveArchivePath(archive, '../escape.csv'), /unsafe|escapes/);
    assert.throws(() => resolveArchivePath(archive, 'C:\\escape.csv'), /relative/);

    const token = manager.begin('sessions/test.csv', 5);
    manager.append(token, Buffer.from('four'));
    assert.throws(() => manager.finish(token), /truncated/);
    assert.equal(fs.existsSync(path.join(archive, 'sessions', 'test.csv')), false);
    assert.deepEqual(fs.readdirSync(path.join(archive, 'sessions')), []);
  } finally {
    fs.rmSync(temp, { recursive: true, force: true });
  }
});

test('desktop archive can safely replace a regenerated manifest', () => {
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'bima-storage-'));
  try {
    const archive = path.join(temp, 'archive');
    const manager = createStorageManager(path.join(temp, 'profile', 'storage.json'));
    manager.setDirectory(archive);
    for (const value of ['recording', 'complete']) {
      const bytes = Buffer.from(value);
      const token = manager.begin('sessions/ses-1/manifest.json', bytes.length);
      manager.append(token, bytes);
      manager.finish(token);
    }
    assert.equal(fs.readFileSync(path.join(archive, 'sessions', 'ses-1', 'manifest.json'), 'utf8'), 'complete');
  } finally {
    fs.rmSync(temp, { recursive: true, force: true });
  }
});
