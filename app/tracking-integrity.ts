export type IntegrityPoint = { x: number; y: number; z: number; visibility?: number };
export type IntegrityFrame = {
  frameIndex: number;
  sessionTimeMs: number;
  hands: IntegrityPoint[][];
};

/**
 * Validate the raw landmark sidecar without looking at display prediction.
 * This is deliberately pure so a saved recording can be checked in tests or
 * by an importer without starting React, MediaPipe, or the camera.
 */
export function assessTrackingIntegrity(frames: IntegrityFrame[]) {
  let frameIndexSequential = true;
  let timestampsMonotonic = true;
  let coordinatesFinite = true;
  let coordinatesNormalized = true;
  let framesWithHands = 0;
  let pointCount = 0;

  let previousTime = -Infinity;
  frames.forEach((frame, index) => {
    if (frame.frameIndex !== index) frameIndexSequential = false;
    if (!Number.isFinite(frame.sessionTimeMs) || frame.sessionTimeMs < previousTime) timestampsMonotonic = false;
    previousTime = frame.sessionTimeMs;
    if (frame.hands.length) framesWithHands += 1;
    frame.hands.forEach((hand) => hand.forEach((point) => {
      pointCount += 1;
      if (![point.x, point.y, point.z].every(Number.isFinite)) coordinatesFinite = false;
      if (point.x < 0 || point.x > 1 || point.y < 0 || point.y > 1) coordinatesNormalized = false;
    }));
  });

  return {
    rawLandmarks: true,
    frameCount: frames.length,
    frameIndexSequential,
    timestampsMonotonic,
    coordinatesFinite,
    coordinatesNormalized,
    framesWithHands,
    handPresenceRate: frames.length ? Number((framesWithHands / frames.length).toFixed(4)) : 0,
    pointCount,
  };
}
