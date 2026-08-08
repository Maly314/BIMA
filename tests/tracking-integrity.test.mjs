import assert from "node:assert/strict";
import test from "node:test";
import { assessTrackingIntegrity } from "../app/tracking-integrity.ts";

const point = (x = 0.5, y = 0.5, z = 0) => ({ x, y, z });

test("raw tracking sidecar reports a clean sequential capture", () => {
  const result = assessTrackingIntegrity([
    { frameIndex: 0, sessionTimeMs: 0, hands: [[point()]] },
    { frameIndex: 1, sessionTimeMs: 33.3, hands: [] },
    { frameIndex: 2, sessionTimeMs: 66.6, hands: [[point(0.6, 0.4), point(0.61, 0.41)]] },
  ]);
  assert.deepEqual(result, {
    rawLandmarks: true,
    frameCount: 3,
    frameIndexSequential: true,
    timestampsMonotonic: true,
    coordinatesFinite: true,
    coordinatesNormalized: true,
    framesWithHands: 2,
    handPresenceRate: 0.6667,
    pointCount: 3,
  });
});

test("raw tracking sidecar flags gaps, time reversal, and invalid coordinates", () => {
  const result = assessTrackingIntegrity([
    { frameIndex: 0, sessionTimeMs: 10, hands: [[point(0.2, 0.3)]] },
    { frameIndex: 2, sessionTimeMs: 5, hands: [[point(1.2, 0.3, Number.NaN)]] },
  ]);
  assert.equal(result.frameIndexSequential, false);
  assert.equal(result.timestampsMonotonic, false);
  assert.equal(result.coordinatesFinite, false);
  assert.equal(result.coordinatesNormalized, false);
  assert.equal(result.handPresenceRate, 1);
});
