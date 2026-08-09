const RECOVERABLE_RENDERER_REASONS = new Set([
  'abnormal-exit',
  'crashed',
  'integrity-failure',
  'killed',
  'launch-failed',
  'oom',
]);

function createRendererRecoveryGate({ maxAttempts = 4, windowMs = 60_000, now = Date.now } = {}) {
  let attempts = [];
  return {
    record(details = {}) {
      const timestamp = now();
      attempts = attempts.filter((value) => timestamp - value < windowMs);
      const reason = String(details.reason || 'unknown');
      if (!RECOVERABLE_RENDERER_REASONS.has(reason)) return { recover: false, reason, attempt: attempts.length };
      if (attempts.length >= maxAttempts) return { recover: false, reason, attempt: attempts.length, exhausted: true };
      attempts.push(timestamp);
      return { recover: true, reason, attempt: attempts.length };
    },
  };
}

function wireRendererRecovery(win, {
  log = () => {},
  recover,
  schedule = setTimeout,
  cancel = clearTimeout,
  crashDelayMs = 250,
  unresponsiveDelayMs = 8_000,
  gate = createRendererRecoveryGate(),
} = {}) {
  if (!win?.webContents || typeof recover !== 'function') throw new Error('Renderer recovery requires a BrowserWindow and recover callback');
  let pending = null;
  const queue = (details, delayMs) => {
    const decision = gate.record(details);
    log(`renderer-loss reason=${decision.reason} attempt=${decision.attempt} recover=${decision.recover} exhausted=${Boolean(decision.exhausted)}`);
    if (!decision.recover || pending !== null) return decision;
    pending = schedule(async () => {
      pending = null;
      if (win.isDestroyed?.()) return;
      await recover(decision);
    }, delayMs);
    return decision;
  };

  win.webContents.on('render-process-gone', (_event, details) => queue(details, crashDelayMs));
  win.on('unresponsive', () => queue({ reason: 'abnormal-exit' }, unresponsiveDelayMs));
  win.on('responsive', () => {
    if (pending !== null) cancel(pending);
    pending = null;
  });
  return { queue };
}

module.exports = { RECOVERABLE_RENDERER_REASONS, createRendererRecoveryGate, wireRendererRecovery };
