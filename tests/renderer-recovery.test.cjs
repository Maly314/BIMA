const assert = require('node:assert/strict');
const { EventEmitter } = require('node:events');
const test = require('node:test');

const { createRendererRecoveryGate, wireRendererRecovery } = require('../desktop/renderer-recovery.cjs');

test('renderer recovery accepts OOM/crashes but ignores a clean exit', () => {
  let now = 1000;
  const gate = createRendererRecoveryGate({ now: () => now, maxAttempts: 2, windowMs: 100 });
  assert.equal(gate.record({ reason: 'clean-exit' }).recover, false);
  assert.deepEqual(gate.record({ reason: 'oom' }), { recover: true, reason: 'oom', attempt: 1 });
  now += 10;
  assert.deepEqual(gate.record({ reason: 'crashed' }), { recover: true, reason: 'crashed', attempt: 2 });
  now += 10;
  assert.equal(gate.record({ reason: 'killed' }).exhausted, true);
  now += 101;
  assert.equal(gate.record({ reason: 'oom' }).recover, true);
});

test('renderer loss schedules one reload and responsive cancellation prevents a false reload', async () => {
  const win = new EventEmitter();
  win.webContents = new EventEmitter();
  win.isDestroyed = () => false;
  const callbacks = new Map();
  let nextId = 0;
  const recovered = [];
  wireRendererRecovery(win, {
    recover: async (decision) => recovered.push(decision.reason),
    schedule: (callback) => { const id = ++nextId; callbacks.set(id, callback); return id; },
    cancel: (id) => callbacks.delete(id),
  });

  win.emit('unresponsive');
  win.emit('responsive');
  assert.equal(callbacks.size, 0);
  win.webContents.emit('render-process-gone', {}, { reason: 'oom', exitCode: 5 });
  win.webContents.emit('render-process-gone', {}, { reason: 'crashed', exitCode: 6 });
  assert.equal(callbacks.size, 1);
  await [...callbacks.values()][0]();
  assert.deepEqual(recovered, ['oom']);
});

test('a failed renderer reload is logged and retried within the bounded gate', async () => {
  const win = new EventEmitter();
  win.webContents = new EventEmitter();
  win.isDestroyed = () => false;
  const callbacks = [];
  const logs = [];
  let attempts = 0;
  wireRendererRecovery(win, {
    recover: async () => {
      attempts += 1;
      if (attempts === 1) throw new Error('app server unavailable');
    },
    log: (message) => logs.push(message),
    schedule: (callback) => { callbacks.push(callback); return callbacks.length; },
    crashDelayMs: 0,
    retryDelayMs: 0,
  });

  win.webContents.emit('render-process-gone', {}, { reason: 'oom' });
  await callbacks.shift()();
  assert.match(logs.join("\n"), /renderer-recovery-failed.*app server unavailable/);
  assert.equal(callbacks.length, 1);
  await callbacks.shift()();
  assert.equal(attempts, 2);
});
