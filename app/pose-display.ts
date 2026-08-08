export type TrackedPoint = { x: number; y: number; z: number; visibility?: number };

type PointVelocity = { x: number; y: number; z: number };

export type DisplayHandHistory = {
  raw: TrackedPoint[][];
  displayed: TrackedPoint[][];
  velocity: PointVelocity[][];
  capturedAt: number;
  lastSeenAt: number;
};

const DISPLAY_HOLD_MS = 110;
const MAX_PREDICTION_MS = 42;
const MAX_DISPLAY_LEAD = 0.018;
const REACQUIRE_DISTANCE = 0.12;
// Keep enough damping to remove estimator shimmer without making small,
// deliberate motion feel disconnected from the operator's hand.
const MIN_BLEND = 0.28;
const MAX_BLEND = 0.94;

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));
const wristDistance = (a: TrackedPoint[], b: TrackedPoint[]) => {
  const aw = a[0], bw = b[0];
  return aw && bw ? Math.hypot(aw.x - bw.x, aw.y - bw.y) : Number.POSITIVE_INFINITY;
};

// MediaPipe may reverse its two result slots when hands cross. Preserve the
// display identity by choosing the assignment with the shortest wrist travel.
// This changes only array order; no measured coordinate is altered.
function alignHands(history: DisplayHandHistory | null, raw: TrackedPoint[][]): TrackedPoint[][] {
  if (!history || raw.length !== 2 || history.raw.length !== 2) return raw;
  const straight = wristDistance(history.raw[0], raw[0]) + wristDistance(history.raw[1], raw[1]);
  const crossed = wristDistance(history.raw[0], raw[1]) + wristDistance(history.raw[1], raw[0]);
  return crossed + 0.015 < straight ? [raw[1], raw[0]] : raw;
}

function emptyHistory(capturedAt: number): DisplayHandHistory {
  return { raw: [], displayed: [], velocity: [], capturedAt, lastSeenAt: capturedAt };
}

// Stabilize only the operator overlay. The raw MediaPipe landmarks passed in
// here are never mutated and the recorder stores those raw values separately.
// Slow sub-pixel estimator jitter is heavily damped; deliberate motion raises
// the blend automatically so the overlay follows without a long trailing lag.
export function predictHandsForDisplay(
  history: DisplayHandHistory | null,
  rawInput: TrackedPoint[][],
  capturedAt: number,
  renderedAt: number,
): { hands: TrackedPoint[][]; history: DisplayHandHistory } {
  if (!rawInput.length) {
    if (history?.displayed.length && capturedAt - history.lastSeenAt <= DISPLAY_HOLD_MS) {
      return { hands: extrapolateHandsForDisplay(history, renderedAt), history };
    }
    const reset = emptyHistory(capturedAt);
    return { hands: [], history: reset };
  }

  const raw = alignHands(history, rawInput);
  const compatible = !!history
    && history.raw.length === raw.length
    && history.raw.every((hand, index) => hand.length === raw[index]?.length);
  if (!compatible || !history) {
    const displayed = raw.map((hand) => hand.map((point) => ({ ...point })));
    const velocity = raw.map((hand) => hand.map(() => ({ x: 0, y: 0, z: 0 })));
    return { hands: displayed, history: { raw, displayed, velocity, capturedAt, lastSeenAt: capturedAt } };
  }

  const frameDeltaMs = clamp(capturedAt - history.capturedAt, 8, 100);
  const predictionMs = clamp(renderedAt - capturedAt, 0, Math.min(MAX_PREDICTION_MS, frameDeltaMs * 1.1));
  const velocity: PointVelocity[][] = [];
  const displayed: TrackedPoint[][] = [];

  raw.forEach((hand, handIndex) => {
    const previousHand = history.raw[handIndex];
    const previousDisplayHand = history.displayed[handIndex];
    const handJump = wristDistance(previousHand, hand) > REACQUIRE_DISTANCE;
    const handVelocity: PointVelocity[] = [];
    const displayedHand: TrackedPoint[] = [];

    hand.forEach((point, pointIndex) => {
      const previousRaw = previousHand[pointIndex];
      const previousDisplay = previousDisplayHand[pointIndex];
      const previousVelocity = history.velocity[handIndex][pointIndex];
      const rawVelocity = {
        x: (point.x - previousRaw.x) / frameDeltaMs,
        y: (point.y - previousRaw.y) / frameDeltaMs,
        z: (point.z - previousRaw.z) / frameDeltaMs,
      };
      const rawSpeed = Math.hypot(rawVelocity.x, rawVelocity.y) * 1000;
      // Damp derivative noise strongly at rest and follow fast real motion.
      const velocityBlend = clamp(0.12 + rawSpeed * 0.45, 0.12, 0.46);
      const filteredVelocity = handJump ? { x: 0, y: 0, z: 0 } : {
        x: previousVelocity.x + (rawVelocity.x - previousVelocity.x) * velocityBlend,
        y: previousVelocity.y + (rawVelocity.y - previousVelocity.y) * velocityBlend,
        z: previousVelocity.z + (rawVelocity.z - previousVelocity.z) * velocityBlend,
      };
      handVelocity.push(filteredVelocity);

      if (handJump) {
        displayedHand.push({ ...point });
        return;
      }

      const speed = Math.hypot(filteredVelocity.x, filteredVelocity.y) * 1000;
      const blend = clamp(MIN_BLEND + speed * 1.65, MIN_BLEND, MAX_BLEND);
      const leadGain = clamp(speed * 0.8, 0, 0.35);
      const leadX = clamp(filteredVelocity.x * predictionMs * leadGain, -MAX_DISPLAY_LEAD, MAX_DISPLAY_LEAD);
      const leadY = clamp(filteredVelocity.y * predictionMs * leadGain, -MAX_DISPLAY_LEAD, MAX_DISPLAY_LEAD);
      displayedHand.push({
        ...point,
        x: clamp(previousDisplay.x + (point.x - previousDisplay.x) * blend + leadX, -0.05, 1.05),
        y: clamp(previousDisplay.y + (point.y - previousDisplay.y) * blend + leadY, -0.05, 1.05),
        z: previousDisplay.z + (point.z - previousDisplay.z) * Math.max(0.2, blend * 0.8),
      });
    });

    velocity.push(handVelocity);
    displayed.push(displayedHand);
  });

  return { hands: displayed, history: { raw, displayed, velocity, capturedAt, lastSeenAt: capturedAt } };
}

// Paint at display refresh rate between model results, but use a filtered
// velocity and a short horizon. At rest the lead becomes effectively zero;
// after 42 ms the overlay stops rather than inventing movement.
export function extrapolateHandsForDisplay(history: DisplayHandHistory | null, renderedAt: number): TrackedPoint[][] {
  if (!history) return [];
  const deltaMs = clamp(renderedAt - history.capturedAt, 0, MAX_PREDICTION_MS);
  return history.displayed.map((hand, handIndex) => hand.map((point, pointIndex) => {
    const velocity = history.velocity[handIndex][pointIndex];
    const speed = Math.hypot(velocity.x, velocity.y) * 1000;
    const leadGain = clamp(speed * 0.8, 0, 0.35);
    return {
      ...point,
      x: clamp(point.x + velocity.x * deltaMs * leadGain, -0.05, 1.05),
      y: clamp(point.y + velocity.y * deltaMs * leadGain, -0.05, 1.05),
      z: point.z + velocity.z * deltaMs * leadGain,
    };
  }));
}
