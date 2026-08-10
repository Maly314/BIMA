import type { CaptureSessionRecord, Recording } from "./recording-types.ts";
import { buildSensorDataDictionaryCsv, csvCell } from "./sensor-export.ts";

export type StorageInfo = { configured: boolean; directory: string; available: boolean };
export type ArchiveFileResult = { bytes: number; relativePath: string; sha256: string };
export type RecordingSaveResult = { archived: boolean; directory?: string; error?: string };

type StorageBridge = {
  abortFile(token: string): Promise<boolean>;
  appendFile(token: string, chunk: Uint8Array): Promise<number>;
  beginFile(relativePath: string, expectedBytes: number): Promise<string>;
  chooseFolder(): Promise<StorageInfo>;
  finishFile(token: string): Promise<ArchiveFileResult>;
  getFolder(): Promise<StorageInfo>;
};

declare global { interface Window { bimaStorage?: StorageBridge } }

const textBlob = (text: string, type: string) => new Blob([text], { type });
const safeSegment = (value: string) => value.normalize("NFKD").replace(/[^a-zA-Z0-9._-]+/g, "-").replace(/^-+|-+$/g, "") || "unknown";

export async function getStorageInfo(): Promise<StorageInfo & { desktop: boolean }> {
  if (typeof window === "undefined" || !window.bimaStorage) return { configured: false, directory: "", available: false, desktop: false };
  return { ...(await window.bimaStorage.getFolder()), desktop: true };
}

export async function chooseStorageFolder(): Promise<StorageInfo> {
  if (!window.bimaStorage) throw new Error("Folder selection is available in the BIMA desktop app");
  return window.bimaStorage.chooseFolder();
}

export async function writeArchiveBlob(relativePath: string, blob: Blob): Promise<ArchiveFileResult> {
  const bridge = window.bimaStorage;
  if (!bridge) throw new Error("Disk archive is available in the BIMA desktop app");
  const token = await bridge.beginFile(relativePath, blob.size);
  try {
    const chunkBytes = 4 * 1024 * 1024;
    for (let offset = 0; offset < blob.size; offset += chunkBytes) {
      const bytes = new Uint8Array(await blob.slice(offset, Math.min(blob.size, offset + chunkBytes)).arrayBuffer());
      await bridge.appendFile(token, bytes);
    }
    return await bridge.finishFile(token);
  } catch (error) {
    await bridge.abortFile(token).catch(() => false);
    throw error;
  }
}

function recordingRoot(recording: Recording): string {
  const patient = `sub-${String(recording.patientNumber).padStart(4, "0")}`;
  const session = recording.captureSessionId ? `ses-${safeSegment(recording.captureSessionId)}` : `solo-${safeSegment(recording.id)}`;
  return `participants/${patient}/sessions/${session}`;
}

export async function archiveRecording(recording: Recording): Promise<RecordingSaveResult> {
  const info = await getStorageInfo();
  if (!info.desktop || !info.configured) return { archived: false };
  if (!info.available) return { archived: false, directory: info.directory, error: "Selected data folder is unavailable" };
  const stream = recording.kind === "sensor" ? "sensor" : "video";
  const root = `${recordingRoot(recording)}/${stream}`;
  try {
    const files: ArchiveFileResult[] = [];
    files.push(await writeArchiveBlob(`${root}/${safeSegment(recording.filename)}`, recording.blob));
    if (recording.rawBlob && recording.rawFilename) files.push(await writeArchiveBlob(`${root}/${safeSegment(recording.rawFilename)}`, recording.rawBlob));
    if (recording.sidecarBlob && recording.sidecarFilename) files.push(await writeArchiveBlob(`${root}/${safeSegment(recording.sidecarFilename)}`, recording.sidecarBlob));
    if (recording.kind === "sensor") {
      files.push(await writeArchiveBlob(`${root}/sensor_data_dictionary.csv`, textBlob(buildSensorDataDictionaryCsv(), "text/csv;charset=utf-8")));
    }
    const metadata = {
      schemaVersion: 1,
      recordingId: recording.id,
      captureSessionId: recording.captureSessionId ?? null,
      kind: recording.kind,
      annotationStatus: recording.annotationStatus ?? null,
      createdAtEpochMs: recording.date,
      files,
    };
    await writeArchiveBlob(`${root}/${safeSegment(recording.id)}-integrity.json`, textBlob(JSON.stringify(metadata, null, 2), "application/json"));
    return { archived: true, directory: info.directory };
  } catch (error) {
    return { archived: false, directory: info.directory, error: error instanceof Error ? error.message : String(error) };
  }
}

function sessionSummaryCsv(session: CaptureSessionRecord): string {
  const columns: Array<keyof CaptureSessionRecord | "asset_count"> = [
    "id", "schemaVersion", "patientNumber", "studyId", "studyDate", "suspected",
    "dateOfBirth", "gestationalAgeAtBirthDays", "chronologicalAgeDays", "correctedAgeDays",
    "postmenstrualAgeDays", "weightKg", "startedAtEpochMs", "stoppedAtEpochMs", "durationMs",
    "status", "asset_count", "note",
  ];
  const row: Record<string, unknown> = { ...session, asset_count: session.assets.length };
  return "\uFEFF" + columns.map(csvCell).join(",") + "\r\n" + columns.map((column) => csvCell(row[column])).join(",");
}

export async function archiveCaptureSession(session: CaptureSessionRecord): Promise<RecordingSaveResult> {
  const info = await getStorageInfo();
  if (!info.desktop || !info.configured) return { archived: false };
  const patient = `sub-${String(session.patientNumber).padStart(4, "0")}`;
  const root = `participants/${patient}/sessions/ses-${safeSegment(session.id)}`;
  try {
    const participant = {
      schemaVersion: 1,
      participantId: patient,
      patientNumber: session.patientNumber,
      dateOfBirth: session.dateOfBirth,
      gestationalAgeAtBirthDays: session.gestationalAgeAtBirthDays,
    };
    const readme = [
      "BIMA local research archive",
      "",
      "Each participant has one folder. Each synchronized capture has one session folder.",
      "sensor/*-sensors.csv is the immutable wide acquisition-cycle table.",
      "sensor/*-sensors-long.csv is the Excel/analysis table with one row per sensor sample.",
      "sensor/sensor_data_dictionary.csv defines units and quality fields.",
      "video/* contains the pose/SAM video, raw recovery video when retained, and frame-level tracking data.",
      "Every stream folder includes an integrity JSON with SHA-256 hashes.",
      "manifest.json and session_summary.csv describe synchronization and clinical-age metadata.",
      "",
      "Do not edit raw capture files. Make analysis copies and preserve this archive as the source record.",
    ].join("\r\n");
    await writeArchiveBlob("BIMA_ARCHIVE_README.txt", textBlob(readme, "text/plain;charset=utf-8"));
    await writeArchiveBlob(`participants/${patient}/participant.json`, textBlob(JSON.stringify(participant, null, 2), "application/json"));
    await writeArchiveBlob(`${root}/manifest.json`, textBlob(JSON.stringify(session, null, 2), "application/json"));
    await writeArchiveBlob(`${root}/session_summary.csv`, textBlob(sessionSummaryCsv(session), "text/csv;charset=utf-8"));
    return { archived: true, directory: info.directory };
  } catch (error) {
    return { archived: false, directory: info.directory, error: error instanceof Error ? error.message : String(error) };
  }
}
