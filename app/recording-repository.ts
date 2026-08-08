import { CAPTURE_SESSION_STORE, openRecordingDatabase, RECORDING_STORE } from "./recording-database.ts";
import { setLastPatient } from "./local-preferences.ts";
import type { CaptureSessionRecord, Recording } from "./recording-types.ts";

export async function listRecordings(): Promise<Recording[]> {
  if (typeof indexedDB === "undefined") return [];
  const database = await openRecordingDatabase();
  return new Promise((resolve, reject) => {
    const request = database.transaction(RECORDING_STORE, "readonly").objectStore(RECORDING_STORE).getAll();
    request.onsuccess = () => resolve((request.result as Recording[]).sort((a, b) => b.date - a.date));
    request.onerror = () => reject(request.error);
  });
}

export async function addRecording(recording: Recording): Promise<void> {
  if (typeof indexedDB === "undefined") return;
  const database = await openRecordingDatabase();
  await new Promise<void>((resolve, reject) => {
    const transaction = database.transaction(RECORDING_STORE, "readwrite");
    transaction.objectStore(RECORDING_STORE).put(recording);
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error);
  });
  setLastPatient(recording.patientNumber);
}

export async function deleteRecording(id: string): Promise<void> {
  if (typeof indexedDB === "undefined") return;
  const database = await openRecordingDatabase();
  return new Promise((resolve, reject) => {
    const transaction = database.transaction([RECORDING_STORE, CAPTURE_SESSION_STORE], "readwrite");
    const recordings = transaction.objectStore(RECORDING_STORE);
    const sessions = transaction.objectStore(CAPTURE_SESSION_STORE);
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
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error);
  });
}
