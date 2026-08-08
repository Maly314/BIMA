import type { Calibration } from "./recording-types.ts";

const LAST_PATIENT_KEY = "movement-capture:lastPatient";
const CALIBRATION_KEY = "movement-capture:calibration";

export function getLastPatient(): number {
  if (typeof localStorage === "undefined") return 0;
  const raw = localStorage.getItem(LAST_PATIENT_KEY);
  const value = raw ? parseInt(raw, 10) : 0;
  return Number.isFinite(value) ? value : 0;
}

export function setLastPatient(patientNumber: number): void {
  if (typeof localStorage === "undefined") return;
  localStorage.setItem(LAST_PATIENT_KEY, String(Math.max(patientNumber, getLastPatient())));
}

export function getCalibration(): Record<string, Calibration> {
  if (typeof localStorage === "undefined") return {};
  try {
    return JSON.parse(localStorage.getItem(CALIBRATION_KEY) || "{}");
  } catch {
    return {};
  }
}

export function setCalibration(calibration: Record<string, Calibration>): void {
  if (typeof localStorage === "undefined") return;
  localStorage.setItem(CALIBRATION_KEY, JSON.stringify(calibration));
}

export function newId(): string {
  if (typeof crypto !== "undefined" && crypto.randomUUID) return crypto.randomUUID();
  return `rec-${Date.now()}-${Math.floor(performance.now() * 1000)}`;
}
