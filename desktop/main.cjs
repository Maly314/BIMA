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

const ROOT = path.join(__dirname, '..');
const PORT = 4820;
const APP_URL = `http://127.0.0.1:${PORT}`;
const BG = '#e4ebee';                       // matches the gradient scene
const ICON_PATH = path.join(ROOT, 'public', 'bima-desktop.ico');
const BRAND_LOGO_DATA = fs.existsSync(path.join(ROOT, 'public', 'bima-logo.png'))
  ? `data:image/png;base64,${fs.readFileSync(path.join(ROOT, 'public', 'bima-logo.png')).toString('base64')}`
  : '';

app.setName('BIMA');
if (process.platform === 'win32') app.setAppUserModelId('org.bima.capture');

/* Chromium's native window occlusion tracker misfires on Windows for
   custom-titlebar windows like this one, marking a plainly visible window as
   occluded — which throttles the renderer to 1 fps and freezes the pose
   tracking loop (measured directly in this shell). This app is a capture
   instrument; it must never be throttled. */
app.commandLine.appendSwitch('disable-features', 'CalculateNativeWinOcclusion');
app.commandLine.appendSwitch('disable-renderer-backgrounding');
const STARTUP_DEBUG = path.join(__dirname, 'startup-debug.log');
fs.writeFileSync(STARTUP_DEBUG, `loaded ${new Date().toISOString()}\n`);
process.on('uncaughtException', (error) => fs.appendFileSync(STARTUP_DEBUG, `uncaught ${error.stack || error}\n`));
process.on('unhandledRejection', (error) => fs.appendFileSync(STARTUP_DEBUG, `rejection ${error?.stack || error}\n`));

let win = null;
let server = null;

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
}

/* Kill whatever process is listening on our port. A leftover server from an
   earlier launch serves whatever build existed when IT started — reusing it
   means silently running stale code after every update. Always start fresh. */
function killPortHolder(port) {
  const out = spawnSync('netstat', ['-ano', '-p', 'tcp'], { windowsHide: true, encoding: 'utf8' });
  const pids = new Set();
  for (const line of String(out.stdout || '').split('\n')) {
    const m = line.match(/TCP\s+\S+:(\d+)\s+\S+\s+LISTENING\s+(\d+)/);
    if (m && Number(m[1]) === port && Number(m[2]) > 0) pids.add(m[2]);
  }
  for (const pid of pids) {
    fs.appendFileSync(STARTUP_DEBUG, `killing stale server pid ${pid} on port ${port}\n`);
    spawnSync('taskkill', ['/pid', pid, '/T', '/F'], { windowsHide: true });
  }
  return pids.size > 0;
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

  win.once('ready-to-show', () => { win.maximize(); win.show(); });
  win.on('closed', () => { win = null; });

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
  });
}

/* ---- permissions: camera + the sensor board's serial port --------------- */

function wireDevices(ses) {
  const ALLOWED = ['media', 'serial', 'camera', 'microphone'];
  const isApp = (url) => typeof url === 'string' && url.startsWith(APP_URL);

  ses.setPermissionRequestHandler((wc, permission, callback) => {
    callback(isApp(wc?.getURL()) && ALLOWED.includes(permission));
  });
  // Chromium consults the CHECK handler (not the request handler) before
  // getUserMedia in an Electron window. Without it the default denies and the
  // page sees NotAllowedError — "Camera unavailable or permission denied" —
  // without ever showing a prompt.
  ses.setPermissionCheckHandler((wc, permission, requestingOrigin) => {
    const origin = requestingOrigin || wc?.getURL();
    return isApp(origin) && ALLOWED.includes(permission);
  });
  ses.setDevicePermissionHandler((details) => details.deviceType === 'serial');
  // Windows also exposes legacy COM ports (the motherboard's COM1 shows up as
  // an ACPI device with no USB vendor id). Select the Teensy USB Serial
  // identity explicitly instead of taking whichever port happens to be first.
  //
  // Electron reports vendorId/productId as DECIMAL strings — the Teensy is
  // "5824"/"1155", not "16c0"/"0483". Match either representation so this
  // survives a change in Electron's formatting.
  const idMatches = (value, hex) => {
    const raw = String(value ?? '').trim().toLowerCase().replace(/^0x/, '');
    if (!raw) return false;
    const wanted = parseInt(hex, 16);
    return parseInt(raw, 10) === wanted || parseInt(raw, 16) === wanted;
  };

  ses.on('select-serial-port', (event, ports, wc, callback) => {
    event.preventDefault();
    const describe = (port) =>
      `${port.portName || port.portId} vid=${port.vendorId} pid=${port.productId} "${port.displayName || ''}"`;
    console.log('[serial] ports offered:', ports.map(describe).join(' | ') || '(none)');

    const teensy = ports.find(
      (port) => idMatches(port.vendorId, '16C0') && idMatches(port.productId, '0483')
    );
    // Fall back to any USB serial device. Legacy/ACPI ports carry no vendorId,
    // so this cannot accidentally select COM1.
    const chosen = teensy || ports.find((port) => String(port.vendorId || '').trim() !== '');

    console.log('[serial] selected:', chosen ? describe(chosen) : 'NONE');
    callback(chosen ? chosen.portId : '');
  });
}

/* ---- boot --------------------------------------------------------------- */

app.whenReady().then(async () => {
  fs.appendFileSync(STARTUP_DEBUG, 'ready\n');
  wireDevices(session.defaultSession);
  createWindow();

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

app.on('window-all-closed', () => { killServer(); app.quit(); });
app.on('will-quit', killServer);
