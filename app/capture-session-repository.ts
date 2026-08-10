import { CAPTURE_SESSION_STORE, openRecordingDatabase } from "./recording-database.ts";
import type { CaptureAsset, CaptureSessionRecord } from "./recording-types.ts";
import { archiveCaptureSession } from "./desktop-storage.ts";

async function requireArchiveSuccess(session: CaptureSessionRecord): Promise<void> {
  const result = await archiveCaptureSession(session);
  if (result.error) throw new Error(result.error);
}

export async function addCaptureSession(session: CaptureSessionRecord): Promise<void> {
  if (typeof indexedDB === "undefined") return;
  const database = await openRecordingDatabase();
  await new Promise<void>((resolve, reject) => {
    const transaction = database.transaction(CAPTURE_SESSION_STORE, "readwrite");
    transaction.objectStore(CAPTURE_SESSION_STORE).put(session);
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error);
  });
  await requireArchiveSuccess(session);
}

export async function updateCaptureSession(id: string, update: Partial<CaptureSessionRecord>): Promise<void> {
  if (typeof indexedDB === "undefined") return;
  const database = await openRecordingDatabase();
  let persisted: CaptureSessionRecord | undefined;
  await new Promise<void>((resolve, reject) => {
    const transaction = database.transaction(CAPTURE_SESSION_STORE, "readwrite");
    const store = transaction.objectStore(CAPTURE_SESSION_STORE);
    const request = store.get(id);
    request.onsuccess = () => {
      const current = request.result as CaptureSessionRecord | undefined;
      if (!current) return;
      const next = { ...current, ...update, id };
      if (current.status === "complete" && update.status === "processing") next.status = "complete";
      persisted = next;
      store.put(next);
    };
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error);
  });
  if (persisted) await requireArchiveSuccess(persisted);
}

export async function addCaptureAsset(id: string, asset: CaptureAsset): Promise<void> {
  if (typeof indexedDB === "undefined") return;
  const database = await openRecordingDatabase();
  let persisted: CaptureSessionRecord | undefined;
  await new Promise<void>((resolve, reject) => {
    const transaction = database.transaction(CAPTURE_SESSION_STORE, "readwrite");
    const store = transaction.objectStore(CAPTURE_SESSION_STORE);
    const request = store.get(id);
    request.onsuccess = () => {
      const current = request.result as CaptureSessionRecord | undefined;
      if (!current) {
        transaction.abort();
        return;
      }
      const assets = [...current.assets.filter((item) => item.kind !== asset.kind), asset];
      const complete = assets.some((item) => item.kind === "sensor") && assets.some(
        (item) => item.kind === "pose" && item.metadata?.annotationStatus !== "failed",
      );
      const terminalFailure = asset.kind === "pose" && asset.metadata?.annotationStatus === "failed";
      persisted = { ...current, assets, status: complete ? "complete" : terminalFailure ? "partial" : current.status };
      store.put(persisted);
    };
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error);
    transaction.onabort = () => reject(transaction.error ?? new Error(`Capture session ${id} was not found`));
  });
  if (persisted) await requireArchiveSuccess(persisted);
}

export async function getCaptureSession(id: string): Promise<CaptureSessionRecord | undefined> {
  if (typeof indexedDB === "undefined") return undefined;
  const database = await openRecordingDatabase();
  return new Promise((resolve, reject) => {
    const request = database.transaction(CAPTURE_SESSION_STORE, "readonly").objectStore(CAPTURE_SESSION_STORE).get(id);
    request.onsuccess = () => resolve(request.result as CaptureSessionRecord | undefined);
    request.onerror = () => reject(request.error);
  });
}
