import assert from "node:assert/strict";
import test from "node:test";

import {
  addClockPoint,
  buildCalibration,
  calibratedMovement,
  calibrationIsUsable,
  newClockFit,
  parseDeviceClockLine,
  parseInvalidImuLine,
  solveClockFit,
} from "../app/sensor-calibration.ts";

test("the device clock line carries the Teensy clock, sequence and per-IMU offsets", () => {
  // Verbatim from the flashed firmware.
  const stamp = parseDeviceClockLine("T: 6044866 N: 2902 O: 0,436,871,1307");
  assert.deepEqual(stamp, { deviceUs: 6044866, seq: 2902, offsetsUs: [0, 436, 871, 1307] });

  // The four IMUs are read in sequence on two buses, so IMU 4 genuinely trails
  // IMU 1. A record must not claim they share one instant.
  assert.equal(stamp.offsetsUs[3] - stamp.offsetsUs[0], 1307);

  // Older firmware without the offsets still parses.
  assert.deepEqual(parseDeviceClockLine("T: 12 N: 3"), { deviceUs: 12, seq: 3, offsetsUs: [] });

  assert.equal(parseDeviceClockLine("IMU1 A: 1, 2, 3 m/s^2 | G: 0, 0, 0 rad/s"), null);
  assert.equal(parseDeviceClockLine("STATS hz:480.0"), null);
  assert.equal(parseDeviceClockLine(""), null);
});

test("a failed IMU read is reported so the row records a gap, not a stale repeat", () => {
  assert.equal(parseInvalidImuLine("IMU3 INVALID"), 2);
  assert.equal(parseInvalidImuLine("IMU1 INVALID"), 0);
  assert.equal(parseInvalidImuLine("IMU1 A: 1, 2, 3 m/s^2 | G: 0, 0, 0 rad/s"), null);
});

test("the host/device clock fit recovers the true rate and reports a usable residual", () => {
  // A clean host clock tracking the device exactly must fit slope 1 with no
  // residual — this is the number that would go in a methods section.
  const clean = newClockFit();
  for (let i = 0; i < 500; i++) addClockPoint(clean, i * 2.083, i * 2.083 + 40);
  const exact = solveClockFit(clean);
  assert.ok(Math.abs(exact.slope - 1) < 1e-9, `slope ${exact.slope}`);
  assert.ok(Math.abs(exact.interceptMs - 40) < 1e-6);
  assert.ok(exact.residualRmsMs < 1e-6, `residual ${exact.residualRmsMs}`);

  // Injecting USB-style arrival jitter must leave the slope intact but surface
  // the jitter in the residual. That separation is the whole point: without the
  // device clock the jitter would be invisible rather than merely present.
  const jittered = newClockFit();
  for (let i = 0; i < 500; i++) {
    const wobble = ((i * 7919) % 13) - 6; // deterministic, +/-6 ms
    addClockPoint(jittered, i * 2.083, i * 2.083 + 40 + wobble);
  }
  const noisy = solveClockFit(jittered);
  assert.ok(Math.abs(noisy.slope - 1) < 5e-3, `slope ${noisy.slope}`);
  assert.ok(noisy.residualRmsMs > 2, `residual ${noisy.residualRmsMs}`);
  assert.equal(noisy.samples, 500);

  // Too few points to fit is null rather than a fabricated number.
  assert.equal(solveClockFit(newClockFit()), null);
});

function stableSamples(count = 50) {
  return Array.from({ length: count }, (_, index) => ({
    ax: Math.sin(index * 0.7) * 0.012,
    ay: Math.cos(index * 0.4) * 0.01,
    az: 9.80665 + Math.sin(index * 0.25) * 0.014,
    gx: 0.018 + Math.sin(index * 0.3) * 0.002,
    gy: -0.011 + Math.cos(index * 0.27) * 0.002,
    gz: 0.007 + Math.sin(index * 0.19) * 0.0015,
  }));
}

test("stable capture produces a usable per-IMU calibration", () => {
  const result = buildCalibration(stableSamples(), 1234);
  assert.equal(result.ok, true);
  if (!result.ok) return;
  assert.equal(result.calibration.version, 2);
  assert.equal(result.calibration.sampleCount, 50);
  assert.equal(result.calibration.date, 1234);
  assert.equal(calibrationIsUsable(result.calibration), true);
  assert.ok(Math.abs(result.calibration.gyroBias[0] - 0.018) < 0.003);
});

test("one isolated desk bump does not shift the robust zero", () => {
  const samples = stableSamples();
  samples[8] = { ax: 2.4, ay: -1.2, az: 10.4, gx: 0.5, gy: -0.3, gz: 0.2 };
  const result = buildCalibration(samples);
  assert.equal(result.ok, true);
  if (!result.ok) return;
  assert.ok(Math.abs(result.calibration.gravity[2] - 9.80665) < 0.03);
});

test("continued handling during calibration is rejected", () => {
  const samples = stableSamples().map((sample, index) => ({
    ...sample,
    ax: Math.sin(index * 0.9) * 0.8,
    gx: sample.gx + Math.cos(index * 0.55) * 0.11,
  }));
  const result = buildCalibration(samples);
  assert.equal(result.ok, false);
  if (result.ok) return;
  assert.match(result.reason, /moved|rotated/);
});

test("orientation alone does not create acceleration movement", () => {
  const result = buildCalibration(stableSamples());
  assert.equal(result.ok, true);
  if (!result.ok) return;
  const xFacing = { ax: 9.80665, ay: 0, az: 0, gx: 0, gy: 0, gz: 0 };
  const zFacing = { ax: 0, ay: 0, az: 9.80665, gx: 0, gy: 0, gz: 0 };
  assert.ok(calibratedMovement(xFacing, result.calibration) < 0.02);
  assert.ok(calibratedMovement(zFacing, result.calibration) < 0.02);
});
