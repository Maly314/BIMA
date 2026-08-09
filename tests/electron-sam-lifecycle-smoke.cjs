const assert = require('node:assert/strict');
const { execFileSync, spawn, spawnSync } = require('node:child_process');
const fs = require('node:fs');
const net = require('node:net');
const os = require('node:os');
const path = require('node:path');

const root = path.resolve(__dirname, '..');
const electron = path.join(root, 'node_modules', 'electron', 'dist', 'electron.exe');
const fakeWorker = path.join(root, 'tests', 'fixtures', 'fake-sam31-worker.py');

async function freePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const { port } = server.address();
      server.close(() => resolve(port));
    });
  });
}

async function waitFor(url, timeoutMs = 30_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url);
      if (response.ok) return response;
    } catch { /* starting or stopped */ }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error(`timed out waiting for ${url}`);
}

async function waitClosed(port, timeoutMs = 20_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const closed = await new Promise((resolve) => {
      const socket = net.connect(port, '127.0.0.1');
      socket.once('connect', () => { socket.destroy(); resolve(false); });
      socket.once('error', () => resolve(true));
    });
    if (closed) return;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error(`port ${port} remained open after Electron quit`);
}

function launch({ appPort, samPort, profile, debugLog, tempRoot, quitMarker, sleepMarker }) {
  return spawn(electron, [path.join(root, 'desktop')], {
    cwd: root,
    windowsHide: true,
    env: {
      ...process.env,
      BIMA_PORT: String(appPort),
      BIMA_SAM31_PORT: String(samPort),
      BIMA_USER_DATA_DIR: profile,
      BIMA_DEBUG_LOG: debugLog,
      BIMA_HEADLESS_PROBE: '1',
      BIMA_QUIT_PROBE_FILE: quitMarker,
      BIMA_SAM31_TEMP_ROOT: tempRoot,
      BIMA_SAM31_CHUNK_WORKER: fakeWorker,
      BIMA_SAM31_VIDEO_ENCODER: 'libx264',
      BIMA_FAKE_WORKER_SLEEP_ONCE: sleepMarker,
      BIMA_FAKE_WORKER_SLEEP_SECONDS: '20',
    },
    stdio: 'ignore',
  });
}

async function waitExit(child, timeoutMs = 30_000) {
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      spawnSync('taskkill', ['/pid', String(child.pid), '/T', '/F'], { windowsHide: true });
      reject(new Error('Electron did not exit after the quit marker'));
    }, timeoutMs);
    child.once('exit', (code) => { clearTimeout(timeout); resolve(code); });
  });
}

function noTempWorkers(tempRoot) {
  const escaped = tempRoot.replaceAll("'", "''");
  return execFileSync('powershell', ['-NoProfile', '-Command', `Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^(ffmpeg|python)' -and $_.CommandLine -like '*${escaped}*' } | Select-Object -ExpandProperty ProcessId`], { encoding: 'utf8', windowsHide: true }).trim();
}

(async () => {
  const appPort = await freePort();
  const samPort = await freePort();
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'bima-electron-sam-lifecycle-'));
  const profile = path.join(temp, 'profile');
  const serviceTemp = path.join(temp, 'sam-temp');
  const debugLog = path.join(temp, 'desktop.log');
  const quitMarker = path.join(temp, 'quit-1');
  const sleepMarker = path.join(temp, 'worker-slept');
  const input = path.join(temp, 'input.webm');
  execFileSync('ffmpeg', ['-y', '-v', 'error', '-f', 'lavfi', '-i', 'color=c=green:s=320x180:r=30:d=0.266667', '-frames:v', '8', '-c:v', 'libvpx-vp9', input], { windowsHide: true });

  const first = launch({ appPort, samPort, profile, debugLog, tempRoot: serviceTemp, quitMarker, sleepMarker });
  await waitFor(`http://127.0.0.1:${appPort}/`);
  await waitFor(`http://127.0.0.1:${samPort}/health`);
  const startedResponse = await fetch(`http://127.0.0.1:${samPort}/process-video-full`, {
    method: 'POST', headers: { 'content-type': 'video/webm' }, body: fs.readFileSync(input),
  });
  const started = await startedResponse.json();
  assert.equal(startedResponse.status, 202);
  const deadline = Date.now() + 30_000;
  let active = false;
  while (Date.now() < deadline) {
    const status = await (await fetch(`http://127.0.0.1:${samPort}/result/${started.jobId}/status`)).json();
    if (String(status.phase).startsWith('tracking-chunk-')) { active = true; break; }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  assert.equal(active, true, 'SAM worker never became active');
  fs.writeFileSync(quitMarker, 'quit');
  assert.equal(await waitExit(first), 0);
  await Promise.all([waitClosed(appPort), waitClosed(samPort)]);
  assert.equal(noTempWorkers(serviceTemp), '');
  assert.match(fs.readFileSync(debugLog, 'utf8'), /quit-probe-triggered/);

  const quitMarker2 = path.join(temp, 'quit-2');
  const second = launch({ appPort, samPort, profile, debugLog, tempRoot: serviceTemp, quitMarker: quitMarker2, sleepMarker });
  await waitFor(`http://127.0.0.1:${appPort}/`);
  await waitFor(`http://127.0.0.1:${samPort}/health`);
  assert.deepEqual(fs.readdirSync(serviceTemp), [], 'service restart did not clear abandoned temporary patient video');
  fs.writeFileSync(quitMarker2, 'quit');
  assert.equal(await waitExit(second), 0);
  await Promise.all([waitClosed(appPort), waitClosed(samPort)]);
  console.log('Electron closed during active SAM work, removed every child process, and relaunched cleanly.');
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
