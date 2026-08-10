import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import test from 'node:test';
import { archiveCaptureSession, archiveRecording } from '../app/desktop-storage.ts';

function storageHarness() {
  const pending = new Map();
  const files = new Map();
  let next = 0;
  return {
    files,
    bridge: {
      async getFolder() { return { configured: true, directory: 'D:\\BIMA data', available: true }; },
      async chooseFolder() { return this.getFolder(); },
      async beginFile(relativePath, expectedBytes) {
        const token = `token-${++next}`;
        pending.set(token, { relativePath, expectedBytes, chunks: [] });
        return token;
      },
      async appendFile(token, chunk) { pending.get(token).chunks.push(Buffer.from(chunk)); return chunk.byteLength; },
      async finishFile(token) {
        const write = pending.get(token);
        const bytes = Buffer.concat(write.chunks);
        assert.equal(bytes.length, write.expectedBytes);
        files.set(write.relativePath, bytes);
        pending.delete(token);
        return { bytes: bytes.length, relativePath: write.relativePath, sha256: crypto.createHash('sha256').update(bytes).digest('hex') };
      },
      async abortFile(token) { return pending.delete(token); },
    },
  };
}

test('recording archive writes raw, analysis, dictionary, and integrity files', async () => {
  const harness = storageHarness();
  globalThis.window = { bimaStorage: harness.bridge };
  try {
    const result = await archiveRecording({
      id: 'recording-1', patientNumber: 7, kind: 'sensor', captureSessionId: 'session-1',
      filename: 'patient-7-sensors.csv', blob: new Blob(['wide']),
      sidecarFilename: 'patient-7-sensors-long.csv', sidecarBlob: new Blob(['long']), date: 1,
    });
    assert.equal(result.archived, true);
    const names = [...harness.files.keys()];
    assert.ok(names.some((name) => name.endsWith('/sensor/patient-7-sensors.csv')));
    assert.ok(names.some((name) => name.endsWith('/sensor/patient-7-sensors-long.csv')));
    assert.ok(names.some((name) => name.endsWith('/sensor/sensor_data_dictionary.csv')));
    const integrityName = names.find((name) => name.endsWith('/sensor/recording-1-integrity.json'));
    const integrity = JSON.parse(harness.files.get(integrityName).toString('utf8'));
    assert.equal(integrity.files.length, 3);
    assert.match(integrity.files[0].sha256, /^[a-f0-9]{64}$/);
  } finally {
    delete globalThis.window;
  }
});

test('session archive writes participant metadata, manifest, and Excel-ready summary', async () => {
  const harness = storageHarness();
  globalThis.window = { bimaStorage: harness.bridge };
  try {
    const result = await archiveCaptureSession({
      id: 'session-1', schemaVersion: 3, patientNumber: 7, suspected: true,
      ageYears: 0, ageMonths: 0, ageDays: 0, dateOfBirth: '2026-07-01',
      gestationalAgeWeeks: 34, gestationalAgeDays: 0, gestationalAgeAtBirthDays: 238,
      chronologicalAgeDays: 40, prematurityCorrectionDays: 42, correctedAgeDays: -2,
      postmenstrualAgeDays: 278, expectedDueDate: '2026-08-12', preterm: true,
      useCorrectedAge: true, studyDate: '2026-08-10', weightKg: 2.4,
      studyId: '@study', note: '=operator note', startedAtEpochMs: 1000,
      stoppedAtEpochMs: 2000, durationMs: 1000,
      clock: { type: 'performance-time-origin', unit: 'milliseconds', zero: 'capture-start' },
      status: 'complete', assets: [],
    });
    assert.equal(result.archived, true);
    assert.ok(harness.files.has('BIMA_ARCHIVE_README.txt'));
    assert.ok(harness.files.has('participants/sub-0007/participant.json'));
    assert.ok(harness.files.has('participants/sub-0007/sessions/ses-session-1/manifest.json'));
    const summary = harness.files.get('participants/sub-0007/sessions/ses-session-1/session_summary.csv').toString('utf8');
    assert.ok(summary.startsWith('\uFEFFid,schemaVersion,patientNumber'));
    assert.match(summary, /'@study/);
    assert.match(summary, /'=operator note/);
  } finally {
    delete globalThis.window;
  }
});
