import "fake-indexeddb/auto";

import assert from "node:assert/strict";
import test from "node:test";

import {
  addCaptureAsset,
  addCaptureSession,
  addRecording,
  deleteRecording,
  getCalibration,
  getCaptureSession,
  getLastPatient,
  listRecordings,
  setCalibration,
  setLastPatient,
  updateCaptureSession,
} from "../app/recordings.ts";

class MemoryStorage {
  #values = new Map();

  getItem(key) {
    return this.#values.get(key) ?? null;
  }

  setItem(key, value) {
    this.#values.set(key, String(value));
  }
}

globalThis.localStorage = new MemoryStorage();

const makeSession = (id) => ({
  id,
  schemaVersion: 3,
  patientNumber: 7,
  suspected: false,
  ageYears: 0,
  ageMonths: 2,
  ageDays: 1,
  dateOfBirth: "2026-05-01",
  gestationalAgeWeeks: 32,
  gestationalAgeDays: 0,
  gestationalAgeAtBirthDays: 224,
  chronologicalAgeDays: 64,
  prematurityCorrectionDays: 56,
  correctedAgeDays: 8,
  postmenstrualAgeDays: 288,
  expectedDueDate: "2026-06-26",
  preterm: true,
  useCorrectedAge: true,
  studyDate: "2026-07-04",
  weightKg: 3.2,
  studyId: "BIMA-0007",
  note: "baseline",
  startedAtEpochMs: 1_000,
  clock: { type: "performance-time-origin", unit: "milliseconds", zero: "capture-start" },
  status: "recording",
  assets: [],
});

const makeRecording = (id, date, captureSessionId) => ({
  id,
  patientNumber: 7,
  suspected: false,
  ageYears: 0,
  ageMonths: 2,
  ageDays: 1,
  kind: "sensor",
  date,
  blob: new Blob([id], { type: "text/csv" }),
  filename: `${id}.csv`,
  size: id.length,
  captureSessionId,
  rawBlob: new Blob([`raw-${id}`], { type: "video/webm" }),
  rawFilename: `${id}-raw.webm`,
  sidecarBlob: new Blob([`sidecar-${id}`], { type: "application/json" }),
  sidecarFilename: `${id}.json`,
});

test("recording persistence preserves session transitions and asset integrity", async () => {
  const session = makeSession("session-main");
  await addCaptureSession(session);
  assert.deepEqual(await getCaptureSession(session.id), session);

  await updateCaptureSession(session.id, { status: "complete", note: "finished" });
  await updateCaptureSession(session.id, { status: "processing" });
  assert.equal((await getCaptureSession(session.id)).status, "complete");

  await addCaptureAsset(session.id, {
    recordingId: "sensor-old",
    kind: "sensor",
    filename: "sensor-old.csv",
    sampleCount: 100,
    streamStartOffsetMs: 4,
    size: 400,
  });
  await addCaptureAsset(session.id, {
    recordingId: "pose-main",
    kind: "pose",
    filename: "pose.webm",
    sampleCount: 30,
    streamStartOffsetMs: 10,
    size: 800,
  });
  await addCaptureAsset(session.id, {
    recordingId: "sensor-new",
    kind: "sensor",
    filename: "sensor-new.csv",
    sampleCount: 120,
    streamStartOffsetMs: 3,
    size: 450,
  });
  const withAssets = await getCaptureSession(session.id);
  assert.equal(withAssets.status, "complete");
  assert.deepEqual(withAssets.assets.map((asset) => asset.recordingId).sort(), ["pose-main", "sensor-new"]);

  await assert.rejects(
    addCaptureAsset("missing-session", {
      recordingId: "orphan",
      kind: "sensor",
      filename: "orphan.csv",
      sampleCount: 0,
      streamStartOffsetMs: 0,
      size: 0,
    }),
    /was not found/,
  );
});

test("recordings stay newest-first and deletion repairs the owning session", async () => {
  const session = makeSession("session-delete");
  session.assets = [
    { recordingId: "recording-delete", kind: "sensor", filename: "delete.csv", sampleCount: 10, streamStartOffsetMs: 0, size: 10 },
    { recordingId: "pose-keep", kind: "pose", filename: "keep.webm", sampleCount: 10, streamStartOffsetMs: 0, size: 10 },
  ];
  session.status = "complete";
  await addCaptureSession(session);
  await addRecording(makeRecording("recording-old", 1_000));
  await addRecording(makeRecording("recording-delete", 2_000, session.id));

  const recordings = await listRecordings();
  assert.deepEqual(recordings.map((recording) => recording.id), ["recording-delete", "recording-old"]);
  assert.equal(await recordings[0].rawBlob.text(), "raw-recording-delete");
  assert.equal(await recordings[0].sidecarBlob.text(), "sidecar-recording-delete");
  await deleteRecording("recording-delete");
  assert.deepEqual((await listRecordings()).map((recording) => recording.id), ["recording-old"]);
  const repaired = await getCaptureSession(session.id);
  assert.equal(repaired.status, "partial");
  assert.deepEqual(repaired.assets.map((asset) => asset.recordingId), ["pose-keep"]);
});

test("deleting a session's last recording removes the empty session", async () => {
  const session = makeSession("session-empty");
  session.assets = [
    { recordingId: "only-recording", kind: "sensor", filename: "only.csv", sampleCount: 1, streamStartOffsetMs: 0, size: 1 },
  ];
  await addCaptureSession(session);
  await addRecording(makeRecording("only-recording", 3_000, session.id));
  await deleteRecording("only-recording");
  assert.equal(await getCaptureSession(session.id), undefined);
});

test("patient sequence and calibration settings round-trip without decreasing", () => {
  assert.equal(getLastPatient(), 7);
  setLastPatient(8);
  setLastPatient(3);
  assert.equal(getLastPatient(), 8);

  const calibration = {
    "1": { version: 2, gravity: [0, 0, 1], gyroBias: [0.1, 0.2, 0.3], sampleCount: 300, date: 123 },
  };
  setCalibration(calibration);
  assert.deepEqual(getCalibration(), calibration);
});
