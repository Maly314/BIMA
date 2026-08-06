import type { Calibration } from "./recordings";

export const STANDARD_GRAVITY = 9.80665;
export const CALIBRATION_SETTLE_MS = 5000;
export const CALIBRATION_CAPTURE_MS = 5000;
export const MIN_CALIBRATION_SAMPLES = 30;

export type ImuSample = {
  ax: number;
  ay: number;
  az: number;
  gx: number;
  gy: number;
  gz: number;
};

export type CalibrationResult =
  | { ok: true; calibration: Calibration }
  | { ok: false; reason: string };

// Teensy diagnostic serial format (native USB):
// IMU1 A: ax, ay, az m/s^2 | G: gx, gy, gz rad/s
export const DATA_RE = /^IMU(\d)\s+A:\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\s*m\/s\^2\s*\|\s*G:\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)/;
export const INVALID_RE = /^IMU(\d)\s+INVALID/;

// Closing line of each record: the Teensy's own clock, sample counter, and the
// per-IMU read offsets.
//   T: <microseconds since boot>  N: <sequence>  O: <o1,o2,o3,o4>
// The four IMUs sit on two buses and are read in sequence, so they are NOT
// simultaneous — IMU 4 trails IMU 1 by most of a read cycle. IMU n's true
// sample time is deviceUs + offsetsUs[n]. O is optional so a board running
// older firmware still parses.
export const DEVICE_CLOCK_RE = /^T:\s*(\d+)\s+N:\s*(\d+)(?:\s+O:\s*([\d,]+))?\s*$/;

export type DeviceClockStamp = { deviceUs: number; seq: number; offsetsUs: number[] };

export function parseDeviceClockLine(line: string): DeviceClockStamp | null {
  const match = line.trim().match(DEVICE_CLOCK_RE);
  if (!match) return null;
  const deviceUs = Number(match[1]);
  const seq = Number(match[2]);
  if (!Number.isFinite(deviceUs) || !Number.isFinite(seq)) return null;
  const offsetsUs = match[3]
    ? match[3].split(",").map(Number).filter(Number.isFinite)
    : [];
  return { deviceUs, seq, offsetsUs };
}

export function parseInvalidImuLine(line: string): number | null {
  const match = line.trim().match(INVALID_RE);
  if (!match) return null;
  const imu = Number.parseInt(match[1], 10) - 1;
  return imu >= 0 && imu <= 3 ? imu : null;
}

// Least-squares fit of host time against the device clock, accumulated
// incrementally so a long capture costs no memory.
//
// This is the whole reason the device stamps exist: without it the host's
// arrival time silently absorbs USB scheduling jitter and the alignment error
// is not merely large, it is unmeasurable. With it, residualRmsMs is a number
// that can go in a methods section.
export type ClockFitAccumulator = { n: number; sx: number; sy: number; sxx: number; sxy: number; syy: number };
export type ClockFit = { slope: number; interceptMs: number; residualRmsMs: number; samples: number };

export function newClockFit(): ClockFitAccumulator {
  return { n: 0, sx: 0, sy: 0, sxx: 0, sxy: 0, syy: 0 };
}

export function addClockPoint(fit: ClockFitAccumulator, deviceMs: number, hostMs: number): void {
  fit.n += 1;
  fit.sx += deviceMs;
  fit.sy += hostMs;
  fit.sxx += deviceMs * deviceMs;
  fit.sxy += deviceMs * hostMs;
  fit.syy += hostMs * hostMs;
}

export function solveClockFit(fit: ClockFitAccumulator): ClockFit | null {
  if (fit.n < 3) return null;
  const denominator = fit.n * fit.sxx - fit.sx * fit.sx;
  if (!denominator) return null;
  const slope = (fit.n * fit.sxy - fit.sx * fit.sy) / denominator;
  const interceptMs = (fit.sy - slope * fit.sx) / fit.n;
  // Sum of squared residuals expanded so it needs only the accumulated moments.
  const ss = fit.syy
    - 2 * slope * fit.sxy
    - 2 * interceptMs * fit.sy
    + slope * slope * fit.sxx
    + 2 * slope * interceptMs * fit.sx
    + fit.n * interceptMs * interceptMs;
  return {
    slope: Number(slope.toFixed(9)),
    interceptMs: Number(interceptMs.toFixed(4)),
    residualRmsMs: Number(Math.sqrt(Math.max(0, ss) / fit.n).toFixed(4)),
    samples: fit.n,
  };
}

export function parseImuLine(line: string): { imu: number; sample: ImuSample } | null {
  const match = line.trim().match(DATA_RE);
  if (!match) return null;
  const imu = Number.parseInt(match[1], 10) - 1;
  if (imu < 0 || imu > 3) return null;
  const values = match.slice(2, 8).map(Number);
  if (!values.every(Number.isFinite)) return null;
  return {
    imu,
    sample: {
      ax: values[0], ay: values[1], az: values[2],
      gx: values[3], gy: values[4], gz: values[5],
    },
  };
}

function quantile(values: number[], q: number): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const position = (sorted.length - 1) * q;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return sorted[lower];
  const mix = position - lower;
  return sorted[lower] * (1 - mix) + sorted[upper] * mix;
}

const median = (values: number[]) => quantile(values, 0.5);

export function calibrationIsUsable(calibration: Calibration | undefined): calibration is Calibration {
  if (!calibration) return false;
  const values = [...(calibration.gravity || []), ...(calibration.gyroBias || [])];
  const gravityMagnitude = Math.hypot(...(calibration.gravity || []));
  return calibration.version === 2
    && values.length === 6
    && values.every(Number.isFinite)
    && Number.isFinite(calibration.accelNoise)
    && Number.isFinite(calibration.gyroNoise)
    && (calibration.sampleCount ?? 0) >= MIN_CALIBRATION_SAMPLES
    && gravityMagnitude > 7
    && gravityMagnitude < 12;
}

export function buildCalibration(samples: ImuSample[], date = Date.now()): CalibrationResult {
  if (samples.length < MIN_CALIBRATION_SAMPLES) {
    return { ok: false, reason: `only ${samples.length} stable samples received` };
  }

  // Medians prevent a desk bump or one malformed serial sample from shifting zero.
  const gravity = [
    median(samples.map((sample) => sample.ax)),
    median(samples.map((sample) => sample.ay)),
    median(samples.map((sample) => sample.az)),
  ];
  const gyroBias = [
    median(samples.map((sample) => sample.gx)),
    median(samples.map((sample) => sample.gy)),
    median(samples.map((sample) => sample.gz)),
  ];

  const gravityMagnitude = Math.hypot(...gravity);
  if (Math.abs(gravityMagnitude - STANDARD_GRAVITY) > 1.4) {
    return { ok: false, reason: `gravity check failed (${gravityMagnitude.toFixed(2)} m/s²)` };
  }

  const accelDeviation = samples.map((sample) =>
    Math.hypot(sample.ax - gravity[0], sample.ay - gravity[1], sample.az - gravity[2]),
  );
  const gyroDeviation = samples.map((sample) =>
    Math.hypot(sample.gx - gyroBias[0], sample.gy - gyroBias[1], sample.gz - gyroBias[2]),
  );
  const accelP90 = quantile(accelDeviation, 0.9);
  const gyroP90 = quantile(gyroDeviation, 0.9);

  // These thresholds are deliberately above normal ICM-20948 rest noise but low
  // enough to reject a calibration in which the board was handled or bumped.
  if (accelP90 > 0.35) {
    return { ok: false, reason: `sensor moved (acceleration spread ${accelP90.toFixed(2)} m/s²)` };
  }
  if (gyroP90 > 0.065) {
    return { ok: false, reason: `sensor rotated (gyro spread ${gyroP90.toFixed(3)} rad/s)` };
  }

  const magnitudeNoise = quantile(
    samples.map((sample) => Math.abs(Math.hypot(sample.ax, sample.ay, sample.az) - gravityMagnitude)),
    0.9,
  );

  return {
    ok: true,
    calibration: {
      version: 2,
      gravity,
      gyroBias,
      accelNoise: Math.max(0.008, magnitudeNoise),
      gyroNoise: Math.max(0.003, gyroP90),
      sampleCount: samples.length,
      date,
    },
  };
}

export function correctedGyro(sample: ImuSample, calibration?: Calibration): [number, number, number] {
  const bias = calibrationIsUsable(calibration) ? calibration.gyroBias : [0, 0, 0];
  return [sample.gx - bias[0], sample.gy - bias[1], sample.gz - bias[2]];
}

// Magnitude-based acceleration is independent of which way a sensor is facing.
// That prevents simply rotating an ankle/wrist in gravity from looking like a
// translation, while retaining the high-frequency acceleration used for jitter.
export function calibratedMovement(sample: ImuSample, calibration?: Calibration): number {
  const expectedGravity = calibrationIsUsable(calibration)
    ? Math.hypot(...calibration.gravity)
    : STANDARD_GRAVITY;
  const noiseFloor = calibration?.accelNoise ?? 0.025;
  const residual = Math.abs(Math.hypot(sample.ax, sample.ay, sample.az) - expectedGravity);
  return Math.max(0, residual - noiseFloor * 2.5);
}
