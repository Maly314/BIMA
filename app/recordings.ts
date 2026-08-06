// Client-side recording library backed by IndexedDB (blobs can be large video).
// All functions are browser-only and are called from client components/effects.

export type Recording = {
  id: string;
  patientNumber: number;
  suspected: boolean;
  ageYears: number;
  ageMonths: number;
  ageDays: number;
  dateOfBirth?: string;
  gestationalAgeWeeks?: number;
  gestationalAgeDays?: number;
  gestationalAgeAtBirthDays?: number;
  chronologicalAgeDays?: number;
  prematurityCorrectionDays?: number;
  correctedAgeDays?: number;
  postmenstrualAgeDays?: number;
  expectedDueDate?: string;
  preterm?: boolean;
  useCorrectedAge?: boolean;
  studyDate?: string;
  weightKg?: number;
  kind: "sensor" | "video" | "pose";
  date: number;          // epoch ms
  blob: Blob;
  filename: string;
  size: number;
  thumbnail?: string;    // data URL
  sidecarBlob?: Blob;    // timestamped pose/hand landmarks for video records
  sidecarFilename?: string;
  captureSessionId?: string;
  studyId?: string;
  note?: string;
  sync?: {
    schemaVersion: number;
    clock: "performance-time-origin";
    startedAtEpochMs: number;
    streamStartOffsetMs: number;
    sampleCount: number;
  };
};

export type CaptureAsset = {
  recordingId: string;
  kind: "sensor" | "pose";
  filename: string;
  sidecarFilename?: string;
  sampleCount: number;
  streamStartOffsetMs: number;
  size: number;
  metadata?: Record<string, unknown>;
};

export type CaptureSessionRecord = {
  id: string;
  schemaVersion: number;
  patientNumber: number;
  suspected: boolean;
  ageYears: number;
  ageMonths: number;
  ageDays: number;
  dateOfBirth: string;
  gestationalAgeWeeks: number;
  gestationalAgeDays: number;
  gestationalAgeAtBirthDays: number;
  chronologicalAgeDays: number;
  prematurityCorrectionDays: number;
  correctedAgeDays: number;
  postmenstrualAgeDays: number;
  expectedDueDate: string;
  preterm: boolean;
  useCorrectedAge: boolean;
  studyDate: string;
  weightKg: number;
  studyId: string;
  note: string;
  startedAtEpochMs: number;
  stoppedAtEpochMs?: number;
  durationMs?: number;
  clock: {
    type: "performance-time-origin";
    unit: "milliseconds";
    zero: "capture-start";
  };
  status: "recording" | "processing" | "complete" | "partial";
  assets: CaptureAsset[];
};

const DB_NAME = "movement-capture";
const STORE = "recordings";
const SESSION_STORE = "capture-sessions";
const LAST_PATIENT_KEY = "movement-capture:lastPatient";

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 2);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE)) db.createObjectStore(STORE, { keyPath: "id" });
      if (!db.objectStoreNames.contains(SESSION_STORE)) db.createObjectStore(SESSION_STORE, { keyPath: "id" });
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

export async function addCaptureSession(session: CaptureSessionRecord): Promise<void> {
  if (typeof indexedDB === "undefined") return;
  const db = await openDb();
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(SESSION_STORE, "readwrite");
    tx.objectStore(SESSION_STORE).put(session);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

export async function updateCaptureSession(id: string, update: Partial<CaptureSessionRecord>): Promise<void> {
  if (typeof indexedDB === "undefined") return;
  const db = await openDb();
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(SESSION_STORE, "readwrite");
    const store = tx.objectStore(SESSION_STORE);
    const request = store.get(id);
    request.onsuccess = () => {
      const current = request.result as CaptureSessionRecord | undefined;
      if (!current) return;
      const next = { ...current, ...update, id };
      if (current.status === "complete" && update.status === "processing") next.status = "complete";
      store.put(next);
    };
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

export async function addCaptureAsset(id: string, asset: CaptureAsset): Promise<void> {
  if (typeof indexedDB === "undefined") return;
  const db = await openDb();
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(SESSION_STORE, "readwrite");
    const store = tx.objectStore(SESSION_STORE);
    const request = store.get(id);
    request.onsuccess = () => {
      const current = request.result as CaptureSessionRecord | undefined;
      if (!current) { tx.abort(); return; }
      const assets = [...current.assets.filter((item) => item.kind !== asset.kind), asset];
      const complete = assets.some((item) => item.kind === "sensor") && assets.some((item) => item.kind === "pose");
      store.put({ ...current, assets, status: complete ? "complete" : current.status });
    };
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error ?? new Error(`Capture session ${id} was not found`));
  });
}

export async function getCaptureSession(id: string): Promise<CaptureSessionRecord | undefined> {
  if (typeof indexedDB === "undefined") return undefined;
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const request = db.transaction(SESSION_STORE, "readonly").objectStore(SESSION_STORE).get(id);
    request.onsuccess = () => resolve(request.result as CaptureSessionRecord | undefined);
    request.onerror = () => reject(request.error);
  });
}

export async function listRecordings(): Promise<Recording[]> {
  if (typeof indexedDB === "undefined") return [];
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const request = db.transaction(STORE, "readonly").objectStore(STORE).getAll();
    request.onsuccess = () => resolve((request.result as Recording[]).sort((a, b) => b.date - a.date));
    request.onerror = () => reject(request.error);
  });
}

export async function addRecording(recording: Recording): Promise<void> {
  if (typeof indexedDB === "undefined") return;
  const db = await openDb();
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(STORE, "readwrite");
    tx.objectStore(STORE).put(recording);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
  setLastPatient(recording.patientNumber);
}

export async function deleteRecording(id: string): Promise<void> {
  if (typeof indexedDB === "undefined") return;
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction([STORE, SESSION_STORE], "readwrite");
    const recordings = tx.objectStore(STORE);
    const sessions = tx.objectStore(SESSION_STORE);
    const request = recordings.get(id);
    request.onsuccess = () => {
      const recording = request.result as Recording | undefined;
      recordings.delete(id);
      if (!recording?.captureSessionId) return;
      const sessionRequest = sessions.get(recording.captureSessionId);
      sessionRequest.onsuccess = () => {
        const session = sessionRequest.result as CaptureSessionRecord | undefined;
        if (!session) return;
        const assets = session.assets.filter((asset) => asset.recordingId !== id);
        if (assets.length) sessions.put({ ...session, assets, status: "partial" });
        else sessions.delete(session.id);
      };
    };
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

export function getLastPatient(): number {
  if (typeof localStorage === "undefined") return 0;
  const raw = localStorage.getItem(LAST_PATIENT_KEY);
  const value = raw ? parseInt(raw, 10) : 0;
  return Number.isFinite(value) ? value : 0;
}

export function setLastPatient(n: number): void {
  if (typeof localStorage === "undefined") return;
  localStorage.setItem(LAST_PATIENT_KEY, String(Math.max(n, getLastPatient())));
}

// Per-IMU calibration, keyed by the physical IMU index printed by the Teensy.
// The optional fields keep old records parseable for a clean migration, but the
// acquisition UI requires a version 2 record before it enables recording.
export type Calibration = {
  version?: number;
  gravity: number[];
  gyroBias: number[];
  accelNoise?: number;
  gyroNoise?: number;
  sampleCount?: number;
  date: number;
};

const CAL_KEY = "movement-capture:calibration";

export function getCalibration(): Record<string, Calibration> {
  if (typeof localStorage === "undefined") return {};
  try { return JSON.parse(localStorage.getItem(CAL_KEY) || "{}"); } catch { return {}; }
}

export function setCalibration(cal: Record<string, Calibration>): void {
  if (typeof localStorage === "undefined") return;
  localStorage.setItem(CAL_KEY, JSON.stringify(cal));
}

export function newId(): string {
  if (typeof crypto !== "undefined" && crypto.randomUUID) return crypto.randomUUID();
  return `rec-${Date.now()}-${Math.floor(performance.now() * 1000)}`;
}
