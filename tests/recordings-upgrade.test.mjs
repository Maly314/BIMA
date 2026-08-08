import "fake-indexeddb/auto";

import assert from "node:assert/strict";
import test from "node:test";

const openVersionOne = () => new Promise((resolve, reject) => {
  const request = indexedDB.open("movement-capture", 1);
  request.onupgradeneeded = () => request.result.createObjectStore("recordings", { keyPath: "id" });
  request.onsuccess = () => resolve(request.result);
  request.onerror = () => reject(request.error);
});

const transactionComplete = (transaction) => new Promise((resolve, reject) => {
  transaction.oncomplete = () => resolve();
  transaction.onerror = () => reject(transaction.error);
});

test("opening schema v2 preserves v1 recordings and adds capture sessions", async () => {
  const oldDatabase = await openVersionOne();
  const oldTransaction = oldDatabase.transaction("recordings", "readwrite");
  oldTransaction.objectStore("recordings").put({
    id: "legacy-recording",
    patientNumber: 1,
    suspected: false,
    ageYears: 0,
    ageMonths: 1,
    ageDays: 0,
    kind: "sensor",
    date: 100,
    blob: new Blob(["legacy"]),
    filename: "legacy.csv",
    size: 6,
  });
  await transactionComplete(oldTransaction);
  oldDatabase.close();

  const { addCaptureSession, getCaptureSession, listRecordings } = await import("../app/recordings.ts");
  const session = {
    id: "upgraded-session",
    schemaVersion: 3,
    patientNumber: 1,
    suspected: false,
    ageYears: 0,
    ageMonths: 1,
    ageDays: 0,
    dateOfBirth: "2026-01-01",
    gestationalAgeWeeks: 40,
    gestationalAgeDays: 0,
    gestationalAgeAtBirthDays: 280,
    chronologicalAgeDays: 30,
    prematurityCorrectionDays: 0,
    correctedAgeDays: 30,
    postmenstrualAgeDays: 310,
    expectedDueDate: "2026-01-01",
    preterm: false,
    useCorrectedAge: false,
    studyDate: "2026-01-31",
    weightKg: 4,
    studyId: "upgrade",
    note: "",
    startedAtEpochMs: 100,
    clock: { type: "performance-time-origin", unit: "milliseconds", zero: "capture-start" },
    status: "recording",
    assets: [],
  };
  await addCaptureSession(session);

  assert.deepEqual((await listRecordings()).map((recording) => recording.id), ["legacy-recording"]);
  assert.deepEqual(await getCaptureSession(session.id), session);
});
