const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawn } = require('node:child_process');

const root = path.resolve(__dirname, '..');
const electron = path.join(root, 'node_modules', 'electron', 'dist', 'electron.exe');
const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'bima-renderer-recovery-'));
const logPath = path.join(tempRoot, 'recovery.log');

const child = spawn(electron, [path.join(root, 'desktop')], {
  cwd: root,
  windowsHide: true,
  env: {
    ...process.env,
    BIMA_PORT: '4920',
    BIMA_SAM31_PORT: '4931',
    BIMA_DISABLE_SAM: '1',
    BIMA_CRASH_PROBE: '1',
    BIMA_USER_DATA_DIR: path.join(tempRoot, 'profile'),
    BIMA_DEBUG_LOG: logPath,
  },
  stdio: 'ignore',
});

const timeout = setTimeout(() => {
  spawn('taskkill', ['/pid', String(child.pid), '/T', '/F'], { windowsHide: true });
}, 45_000);

child.once('exit', (code) => {
  clearTimeout(timeout);
  const log = fs.existsSync(logPath) ? fs.readFileSync(logPath, 'utf8') : '';
  assert.match(log, /crash-probe-triggered/);
  assert.match(log, /renderer-loss reason=crashed attempt=1 recover=true exhausted=false/);
  assert.match(log, /renderer-recovering reason=crashed attempt=1/);
  assert.match(log, /crash-probe-recovered/);
  assert.equal(code, 0);
  console.log('Electron renderer crash was recovered in a real desktop process.');
});
