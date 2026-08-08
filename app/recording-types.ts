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
  date: number;
  blob: Blob;
  filename: string;
  size: number;
  rawBlob?: Blob;
  rawFilename?: string;
  annotationStatus?: "complete" | "failed";
  processingError?: string;
  thumbnail?: string;
  sidecarBlob?: Blob;
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

export type Calibration = {
  version?: number;
  gravity: number[];
  gyroBias: number[];
  accelNoise?: number;
  gyroNoise?: number;
  sampleCount?: number;
  date: number;
};
