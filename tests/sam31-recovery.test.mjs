import assert from "node:assert/strict";
import test from "node:test";

import { markInterruptedSam31Recordings, recoverSam31Recording } from "../app/sam31-recovery.ts";

const processingRecording = () => ({
  id: "pose-1",
  patientNumber: 4,
  suspected: true,
  ageYears: 0,
  ageMonths: 1,
  ageDays: 2,
  kind: "pose",
  date: 1000,
  blob: new Blob(["untouched-camera"], { type: "video/webm" }),
  filename: "patient-4-sam31-raw.webm",
  size: 16,
  annotationStatus: "processing",
  samJobId: "job-recover",
  captureSessionId: "capture-1",
  studyId: "BIMA-0004",
  note: "test",
  sync: { schemaVersion: 3, clock: "performance-time-origin", startedAtEpochMs: 1000, streamStartOffsetMs: 4, sampleCount: 0 },
});

test("renderer restart reconnects to SAM and preserves both annotated and raw video", async () => {
  const saved = [];
  const assets = [];
  const completed = await recoverSam31Recording(processingRecording(), new AbortController().signal, {
    resume: async (jobId) => ({
      jobId,
      frames: [{ frameIndex: 0, sourceVideoTimeMs: 0, segments: [], source: "sam31-native-propagation" }],
      processingMs: 20,
      annotatedBlob: new Blob(["annotated"], { type: "video/mp4" }),
    }),
    saveRecording: async (recording) => saved.push(recording),
    saveAsset: async (sessionId, asset) => assets.push({ sessionId, asset }),
  });

  assert.equal(completed.annotationStatus, "complete");
  assert.equal(completed.filename, "patient-4-sam31-tracked.mp4");
  assert.equal(await completed.blob.text(), "annotated");
  assert.equal(await completed.rawBlob.text(), "untouched-camera");
  assert.match(await completed.sidecarBlob.text(), /recoveredAfterRendererRestart/);
  assert.equal(saved.length, 1);
  assert.equal(assets[0].sessionId, "capture-1");
  assert.equal(assets[0].asset.sampleCount, 1);
});

test("failed recovery keeps the raw recording and records the error", async () => {
  const saved = [];
  const failed = await recoverSam31Recording({ ...processingRecording(), samJobId: "job-failed" }, new AbortController().signal, {
    resume: async () => { throw new Error("isolated worker OOM"); },
    saveRecording: async (recording) => saved.push(recording),
    saveAsset: async () => { throw new Error("asset must not be added"); },
  });

  assert.equal(failed.annotationStatus, "failed");
  assert.equal(failed.processingError, "isolated worker OOM");
  assert.equal(await failed.blob.text(), "untouched-camera");
  assert.equal(saved.length, 1);
});

test("capture-manifest failure never overwrites a completed annotated recovery", async () => {
  const saved = [];
  const completed = await recoverSam31Recording(processingRecording(), new AbortController().signal, {
    resume: async (jobId) => ({
      jobId,
      frames: [{ frameIndex: 0, sourceVideoTimeMs: 0, segments: [], source: "sam31-native-propagation" }],
      processingMs: 25,
      annotatedBlob: new Blob(["annotated-safe"], { type: "video/mp4" }),
    }),
    saveRecording: async (recording) => saved.push(recording),
    saveAsset: async () => { throw new Error("manifest transaction failed"); },
  });

  assert.equal(completed.annotationStatus, "complete");
  assert.equal(await completed.blob.text(), "annotated-safe");
  assert.equal(await completed.rawBlob.text(), "untouched-camera");
  assert.equal(saved.length, 1);
  assert.equal(saved[0].annotationStatus, "complete");
});

test("startup converts an unrecoverable processing record into raw-only failure", async () => {
  const saved = [];
  const interrupted = { ...processingRecording(), samJobId: undefined };
  const [reconciled] = await markInterruptedSam31Recordings([interrupted], async (recording) => saved.push(recording));

  assert.equal(reconciled.annotationStatus, "failed");
  assert.match(reconciled.processingError, /raw video was preserved/i);
  assert.equal(await reconciled.blob.text(), "untouched-camera");
  assert.equal(saved.length, 1);
});
