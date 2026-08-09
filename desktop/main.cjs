/* ============================================================================
   desktop/main.cjs — Electron shell for the BIMA movement-capture app.

   The app itself is the Next.js (vinext) project one folder up. This shell:
     1. starts its production server on a private port (building first if the
        dist/ output is missing),
     2. opens a native window on it — hidden titlebar with Windows' own
        caption buttons overlaid, so minimise/maximise/close are the real OS
        controls and the in-page bar is the drag region,
     3. grants camera + Web Serial (the sensor board) to this app only,
     4. kills the server again when the window closes.

   Launch:  electron.exe <this file>   (the Desktop shortcut does exactly that)
========================================================================== */

const path = require('node:path');
const fs = require('node:fs');
const http = require('node:http');
const { spawn, spawnSync } = require('node:child_process');
const { app, BrowserWindow, session, shell } = require('electron');
const { applyCaptureRuntimeSwitches, parseListeningPids } = require('./capture-runtime.cjs');
const { isAllowedPermission, selectSerialPort } = require('./device-policy.cjs');
const { wireRendererRecovery } = require('./renderer-recovery.cjs');

const ROOT = path.join(__dirname, '..');
const PORT = Number(process.env.BIMA_PORT || 4820);
const SAM31_PORT = Number(process.env.BIMA_SAM31_PORT || 4831);
const APP_URL = `http://127.0.0.1:${PORT}`;
const BG = '#e4ebee';                       // matches the gradient scene
const ICON_PATH = path.join(ROOT, 'public', 'bima-desktop.ico');
const BRAND_LOGO_DATA = fs.existsSync(path.join(ROOT, 'public', 'bima-logo.png'))
  ? `data:image/png;base64,${fs.readFileSync(path.join(ROOT, 'public', 'bima-logo.png')).toString('base64')}`
  : '';

app.setName('BIMA');
if (process.platform === 'win32') app.setAppUserModelId('org.bima.capture');
if (process.env.BIMA_USER_DATA_DIR) app.setPath('userData', process.env.BIMA_USER_DATA_DIR);

/* The capture window must not inherit Chromium's software-only fallback.
   This machine has a discrete RTX GPU, and MediaPipe's worker delegate needs
   WebGL plus accelerated video decode to keep hand inference realtime. The
   flags are set before app ready so the GPU process sees them from launch. */
applyCaptureRuntimeSwitches(app.commandLine);
const STARTUP_DEBUG = process.env.BIMA_DEBUG_LOG || path.join(__dirname, 'startup-debug.log');
fs.writeFileSync(STARTUP_DEBUG, `loaded ${new Date().toISOString()}\n`);
process.on('uncaughtException', (error) => fs.appendFileSync(STARTUP_DEBUG, `uncaught ${error.stack || error}\n`));
process.on('unhandledRejection', (error) => fs.appendFileSync(STARTUP_DEBUG, `rejection ${error?.stack || error}\n`));

let win = null;
let server = null;
let sam31Server = null;

/* Only one instance — a second launch focuses the existing window. */
const gotSingleInstanceLock = app.requestSingleInstanceLock();
fs.appendFileSync(STARTUP_DEBUG, `single-instance ${gotSingleInstanceLock}\n`);
if (!gotSingleInstanceLock) app.quit();
app.on('second-instance', () => {
  if (win) { if (win.isMinimized()) win.restore(); win.focus(); }
});

/* ---- server management ------------------------------------------------- */

const cmd = (line) =>
  spawn(process.env.ComSpec || 'cmd.exe', ['/d', '/s', '/c', line], {
    cwd: ROOT, stdio: 'ignore', windowsHide: true,
  });

function ping() {
  return new Promise((resolve) => {
    const req = http.get(APP_URL, (res) => { res.resume(); resolve(true); });
    req.on('error', () => resolve(false));
    req.setTimeout(1500, () => { req.destroy(); resolve(false); });
  });
}

async function waitForServer(timeoutMs) {
  const t0 = Date.now();
  while (Date.now() - t0 < timeoutMs) {
    if (await ping()) return true;
    await new Promise((r) => setTimeout(r, 400));
  }
  return false;
}

function killServer() {
  if (server && server.pid && !server.killed) {
    // The npm launcher spawns children; take the whole tree down.
    spawnSync('taskkill', ['/pid', String(server.pid), '/T', '/F'], { windowsHide: true });
  }
  server = null;
  if (sam31Server && sam31Server.pid && !sam31Server.killed) {
    spawnSync('taskkill', ['/pid', String(sam31Server.pid), '/T', '/F'], { windowsHide: true });
  }
  sam31Server = null;
}

function startSam31Server() {
  if (process.env.BIMA_DISABLE_SAM === '1') {
    fs.appendFileSync(STARTUP_DEBUG, 'sam31 service disabled for recovery probe\n');
    return;
  }
  killPortHolder(SAM31_PORT);
  const python = path.join(ROOT, '.sam31-venv', 'Scripts', 'python.exe');
  const service = path.join(__dirname, 'sam31_service.py');
  if (!fs.existsSync(python) || !fs.existsSync(service)) {
    fs.appendFileSync(STARTUP_DEBUG, 'sam31 service unavailable: runtime is not installed\n');
    return;
  }
  sam31Server = spawn(python, [service], {
    cwd: ROOT,
    windowsHide: true,
    env: {
      ...process.env,
      BIMA_SAM31_PORT: String(SAM31_PORT),
      BIMA_APP_ORIGIN: APP_URL,
      BIMA_SAM31_CHUNK_FRAMES: '4',
      BIMA_SAM31_GPU_MEMORY_FRACTION: '0.75',
      BIMA_SAM31_WEIGHT_DTYPE: 'backbones-bfloat16',
      CUDA_MODULE_LOADING: 'LAZY',
      PYTORCH_CUDA_ALLOC_CONF: 'expandable_segments:True',
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  const logSam = (chunk) => fs.appendFileSync(STARTUP_DEBUG, String(chunk));
  sam31Server.stdout.on('data', logSam);
  sam31Server.stderr.on('data', logSam);
  sam31Server.on('exit', (code) => {
    fs.appendFileSync(STARTUP_DEBUG, `sam31 service exited code=${code}\n`);
    sam31Server = null;
  });
}

/* Kill whatever process is listening on our port. A leftover server from an
   earlier launch serves whatever build existed when IT started — reusing it
   means silently running stale code after every update. Always start fresh. */
function killPortHolder(port) {
  const out = spawnSync('netstat', ['-ano', '-p', 'tcp'], { windowsHide: true, encoding: 'utf8' });
  const pids = parseListeningPids(out.stdout, port);
  for (const pid of pids) {
    fs.appendFileSync(STARTUP_DEBUG, `killing stale server pid ${pid} on port ${port}\n`);
    spawnSync('taskkill', ['/pid', pid, '/T', '/F'], { windowsHide: true });
  }
  return pids.length > 0;
}

/* ---- boot splash (shown while the server starts / builds) --------------- */

const splash = (message) => 'data:text/html;charset=utf-8,' + encodeURIComponent(`
  <body style="margin:0;display:grid;place-items:center;height:100vh;
    background:linear-gradient(155deg,#e9f0f2,#dde9ec 45%,#e7e4ec);
    font-family:'Segoe UI',sans-serif;color:#48525b">
  <div style="text-align:center">
    ${BRAND_LOGO_DATA ? `<img src="${BRAND_LOGO_DATA}" alt="BIMA — Biometric Infant Motor Assessment" style="width:210px;height:auto;margin:0 auto 20px;display:block">` : ''}
    <div style="width:34px;height:34px;margin:0 auto 16px;border-radius:50%;
      border:3px solid rgba(8,120,134,.2);border-top-color:#087886;
      animation:s 1s linear infinite"></div>
    <div style="font-size:15px">${message}</div>
  </div>
  <style>@keyframes s{to{transform:rotate(360deg)}}</style></body>`);

/* ---- window ------------------------------------------------------------- */

function createWindow() {
  let crashProbeTriggered = false;
  let stabilityProbeStarted = false;
  win = new BrowserWindow({
    width: 1560,
    height: 1020,
    minWidth: 1180,
    minHeight: 800,
    show: false,
    backgroundColor: BG,
    icon: ICON_PATH,
    autoHideMenuBar: true,
    title: 'BIMA',
    /* Native Windows caption buttons overlaid on our own bar: real
       minimise/maximise/close with no IPC plumbing in the web app. */
    titleBarStyle: 'hidden',
    titleBarOverlay: { color: '#edf2f3', symbolColor: '#3d424a', height: 45 },
    /* backgroundThrottling:false — rAF and timers keep full rate even if the
       window is minimised or covered, so a running capture never stalls. */
    webPreferences: { contextIsolation: true, nodeIntegration: false, sandbox: true, backgroundThrottling: false },
  });

  win.once('ready-to-show', () => {
    if (process.env.BIMA_HEADLESS_PROBE !== '1') { win.maximize(); win.show(); }
  });
  win.on('closed', () => { win = null; });
  wireRendererRecovery(win, {
    log: (message) => fs.appendFileSync(STARTUP_DEBUG, `${new Date().toISOString()} ${message}\n`),
    recover: async ({ reason, attempt }) => {
      if (!win || win.isDestroyed()) return;
      fs.appendFileSync(STARTUP_DEBUG, `${new Date().toISOString()} renderer-recovering reason=${reason} attempt=${attempt}\n`);
      await win.loadURL(splash('Recovering the capture windowâ€¦'));
      await new Promise((resolve) => setTimeout(resolve, 750));
      if (win && !win.isDestroyed()) await win.loadURL(APP_URL);
    },
  });
  // Preserve the lightweight, once-per-second tracking telemetry outside the
  // renderer so a real run can be audited after the window closes. This does
  // not log frames, landmarks, or camera pixels.
  win.webContents.on('console-message', (_event, _level, message) => {
    if (typeof message === 'string' && (message.includes('[tracking] pipeline') || message.includes('[camera] negotiation'))) {
      fs.appendFileSync(STARTUP_DEBUG, `${new Date().toISOString()} ${message}\n`);
      try {
        const metrics = app.getAppMetrics().map((metric) => ({
          pid: metric.pid,
          type: metric.type,
          cpu: Number(metric.cpu?.percentCPUUsage ?? 0),
          workingSetMb: Number(((metric.memory?.workingSetSize ?? 0) / 1024).toFixed(1)),
        }));
        fs.appendFileSync(STARTUP_DEBUG, `${new Date().toISOString()} [runtime] ${JSON.stringify(metrics)}\n`);
      } catch (error) {
        fs.appendFileSync(STARTUP_DEBUG, `${new Date().toISOString()} [runtime-error] ${error?.stack || error}\n`);
      }
    }
  });

  // Keep outbound links in the real browser, never in the app window.
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https?:$/.test(new URL(url).protocol) && !url.startsWith(APP_URL)) {
      shell.openExternal(url);
      return { action: 'deny' };
    }
    return { action: 'deny' };
  });

  // Desktop-only fit: full-bleed sheet, and room for the caption buttons.
  // !important throughout — injected sheets sit before the page's own in the
  // cascade, so at equal specificity the page wins and nothing applies.
  win.webContents.on('did-finish-load', () => {
    if (!win || !win.webContents.getURL().startsWith(APP_URL)) return;
    win.webContents.insertCSS(`
      .page-shell { padding: 0 !important; align-items: stretch !important; }
      .app-window { width: 100% !important; min-height: 100vh !important;
                    border: 0 !important; border-radius: 0 !important;
                    box-shadow: inset 0 1px 0 rgba(255,255,255,.75) !important; }
      .window-bar { padding-right: 170px !important; }
    `);
    if (process.env.BIMA_CRASH_PROBE === '1') {
      if (!crashProbeTriggered) {
        crashProbeTriggered = true;
        fs.appendFileSync(STARTUP_DEBUG, 'crash-probe-triggered\n');
        setTimeout(() => {
          if (win && !win.isDestroyed()) win.webContents.forcefullyCrashRenderer();
        }, 250);
      } else {
        fs.appendFileSync(STARTUP_DEBUG, 'crash-probe-recovered\n');
        setTimeout(() => app.quit(), 250);
      }
    }
    const stabilitySeconds = Number(process.env.BIMA_STABILITY_PROBE_SECONDS || 0);
    if (stabilitySeconds > 0 && !stabilityProbeStarted) {
      stabilityProbeStarted = true;
      let ticks = 0;
      let errors = 0;
      const interval = setInterval(async () => {
        if (!win || win.isDestroyed()) return;
        try {
          await win.webContents.executeJavaScript('document.body && document.body.getBoundingClientRect().width > 0');
          ticks += 1;
        } catch {
          errors += 1;
        }
      }, 1000);
      setTimeout(() => {
        clearInterval(interval);
        fs.appendFileSync(STARTUP_DEBUG, `stability-probe-complete ticks=${ticks} errors=${errors}\n`);
        app.quit();
      }, stabilitySeconds * 1000);
    }
  });
}

/* ---- permissions: camera + the sensor board's serial port --------------- */

function wireDevices(ses) {
  ses.setPermissionRequestHandler((wc, permission, callback) => {
    callback(isAllowedPermission(permission, wc?.getURL(), APP_URL));
  });
  // Chromium consults the CHECK handler (not the request handler) before
  // getUserMedia in an Electron window. Without it the default denies and the
  // page sees NotAllowedError — "Camera unavailable or permission denied" —
  // without ever showing a prompt.
  ses.setPermissionCheckHandler((wc, permission, requestingOrigin) => {
    const origin = requestingOrigin || wc?.getURL();
    return isAllowedPermission(permission, origin, APP_URL);
  });
  ses.setDevicePermissionHandler((details) => details.deviceType === 'serial' && isAllowedPermission('serial', details.origin, APP_URL));
  // Windows also exposes legacy COM ports (the motherboard's COM1 shows up as
  // an ACPI device with no USB vendor id). Select the Teensy USB Serial
  // identity explicitly instead of taking whichever port happens to be first.
  //
  ses.on('select-serial-port', (event, ports, wc, callback) => {
    event.preventDefault();
    const describe = (port) =>
      `${port.portName || port.portId} vid=${port.vendorId} pid=${port.productId} "${port.displayName || ''}"`;
    console.log('[serial] ports offered:', ports.map(describe).join(' | ') || '(none)');

    const chosen = selectSerialPort(ports);

    console.log('[serial] selected:', chosen ? describe(chosen) : 'NONE');
    callback(chosen ? chosen.portId : '');
  });
}

/* ---- boot --------------------------------------------------------------- */

app.whenReady().then(async () => {
  fs.appendFileSync(STARTUP_DEBUG, 'ready\n');
  // Keep the runtime GPU decision visible in the local startup log. A
  // renderer can report a healthy camera while MediaPipe is actually running
  // on a software WebGL path, which is the difference between single-digit
  // and real-time hand tracking on this app.
  try {
    fs.appendFileSync(STARTUP_DEBUG, `gpu-features-before-renderer ${JSON.stringify(app.getGPUFeatureStatus())}\n`);
    const gpuInfo = await app.getGPUInfo('complete');
    fs.appendFileSync(STARTUP_DEBUG, `gpu-info ${JSON.stringify(gpuInfo)}\n`);
  } catch (error) {
    fs.appendFileSync(STARTUP_DEBUG, `gpu-features-error ${error?.stack || error}\n`);
  }
  wireDevices(session.defaultSession);
  createWindow();
  startSam31Server();

  // Never reuse a leftover server — it serves the build that existed when it
  // started, which silently runs stale code after every update.
  killPortHolder(PORT);
  if (!fs.existsSync(path.join(ROOT, 'dist'))) {
    win.loadURL(splash('First run — building the app…'));
    await new Promise((resolve) => cmd('npm run build').on('exit', resolve));
  } else {
    win.loadURL(splash('Starting…'));
  }
  server = cmd(`npm run start -- -p ${PORT} -H 127.0.0.1`);
  server.on('exit', () => { server = null; });

  if (await waitForServer(90_000)) {
    win.loadURL(APP_URL);
    // Feature status can be provisional before a renderer has created a GL
    // context. Sample it again after the app is loaded, including the actual
    // renderer's WebGL vendor/renderer, so the startup diagnosis reflects the
    // path MediaPipe will use rather than only the initial GPU process state.
    setTimeout(async () => {
      if (!win || win.isDestroyed()) return;
      try {
        const rendererGpu = await win.webContents.executeJavaScript(`(() => {
          const canvas = document.createElement('canvas');
          const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
          if (!gl) return { available: false };
          const ext = gl.getExtension('WEBGL_debug_renderer_info');
          return { available: true, vendor: ext ? gl.getParameter(ext.UNMASKED_VENDOR_WEBGL) : '', renderer: ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : '' };
        })()`);
        fs.appendFileSync(STARTUP_DEBUG, `gpu-features-after-load ${JSON.stringify(app.getGPUFeatureStatus())}\n`);
        fs.appendFileSync(STARTUP_DEBUG, `renderer-webgl ${JSON.stringify(rendererGpu)}\n`);
      } catch (error) {
        fs.appendFileSync(STARTUP_DEBUG, `renderer-gpu-error ${error?.stack || error}\n`);
      }
    }, 2000);
  } else {
    win.loadURL(splash('The app server did not start. Close this window and try again.'));
  }

  if (process.env.BIMA_PROBE) {
    win.webContents.on('did-finish-load', async () => {
      try {
        await win.webContents.insertCSS('.window-bar { padding-right: 160px; }');
        const out = await win.webContents.executeJavaScript(
          `(async () => {
            const aw = document.querySelector('.app-window');
            const cs = getComputedStyle(aw);
            const r = { url: location.href };
            r.viaProp = cs.getPropertyValue('backdrop-filter');
            r.viaWebkit = cs.getPropertyValue('-webkit-backdrop-filter');
            r.varBlur = getComputedStyle(document.documentElement).getPropertyValue('--blur');
            aw.style.backdropFilter = 'blur(26px) saturate(1.5)';
            r.inline = getComputedStyle(aw).getPropertyValue('backdrop-filter');
            aw.style.backdropFilter = '';
            r.padAfterInsert = getComputedStyle(document.querySelector('.window-bar')).paddingRight;
            // find the actual declarations Blink kept for .app-window
            r.rules = [];
            for (const s of document.styleSheets) {
              try {
                for (const rule of s.cssRules) {
                  if (rule.selectorText === '.app-window') {
                    r.rules.push({
                      bf: rule.style.getPropertyValue('backdrop-filter'),
                      wbf: rule.style.getPropertyValue('-webkit-backdrop-filter'),
                      len: rule.style.length,
                    });
                  }
                }
              } catch {}
            }
            return JSON.stringify(r);
          })()`);
        process.stderr.write('PROBE ' + out + '\n');
      } catch (e) { process.stderr.write('PROBE-ERR ' + e.message + '\n'); }
    });
  }
});

app.on('child-process-gone', (_event, details) => {
  fs.appendFileSync(STARTUP_DEBUG, `${new Date().toISOString()} child-process-gone ${JSON.stringify(details)}\n`);
});

app.on('window-all-closed', () => { killServer(); app.quit(); });
app.on('will-quit', killServer);
