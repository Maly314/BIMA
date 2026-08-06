export const CAPTURE_SCHEMA_VERSION = 3;

export type CaptureRun = {
  id: string;
  active: boolean;
  startedAtEpochMs: number;
  startedAtPerfMs: number;
  stoppedAtEpochMs?: number;
  stoppedAtPerfMs?: number;
  studyId: string;
  note: string;
};

export function captureElapsedMs(run: CaptureRun, perfNow = performance.now()): number {
  return Math.max(0, Math.round(perfNow - run.startedAtPerfMs));
}

export function captureEpochMs(run: CaptureRun, perfNow = performance.now()): number {
  return run.startedAtEpochMs + captureElapsedMs(run, perfNow);
}

export function stopCaptureRun(run: CaptureRun, epochNow = Date.now(), perfNow = performance.now()): CaptureRun {
  return {
    ...run,
    active: false,
    stoppedAtEpochMs: epochNow,
    stoppedAtPerfMs: perfNow,
  };
}

export function captureDurationMs(run: CaptureRun): number {
  return Math.max(0, Math.round((run.stoppedAtPerfMs ?? performance.now()) - run.startedAtPerfMs));
}

