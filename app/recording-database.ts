export const RECORDING_STORE = "recordings";
export const CAPTURE_SESSION_STORE = "capture-sessions";

const DATABASE_NAME = "movement-capture";
const DATABASE_VERSION = 2;

export function openRecordingDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DATABASE_NAME, DATABASE_VERSION);
    request.onupgradeneeded = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(RECORDING_STORE)) {
        database.createObjectStore(RECORDING_STORE, { keyPath: "id" });
      }
      if (!database.objectStoreNames.contains(CAPTURE_SESSION_STORE)) {
        database.createObjectStore(CAPTURE_SESSION_STORE, { keyPath: "id" });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}
