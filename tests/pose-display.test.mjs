import assert from "node:assert/strict";
import test from "node:test";
import { extrapolateHandsForDisplay, predictHandsForDisplay } from "../app/pose-display.ts";

const point = (x, y, z = 0) => ({ x, y, z });

test("display prediction is bounded and does not mutate measured landmarks", () => {
  const firstRaw = [[point(0.40, 0.50)]];
  const first = predictHandsForDisplay(null, firstRaw, 100, 150);
  const secondRaw = [[point(0.50, 0.54)]];
  const before = structuredClone(secondRaw);
  const second = predictHandsForDisplay(first.history, secondRaw, 133, 200);

  assert.deepEqual(secondRaw, before);
  assert.ok(second.hands[0][0].x > secondRaw[0][0].x);
  assert.ok(second.hands[0][0].x <= secondRaw[0][0].x + 0.035);
  assert.ok(second.hands[0][0].y <= secondRaw[0][0].y + 0.035);
});

test("a changed hand layout resets history instead of mixing subjects", () => {
  const first = predictHandsForDisplay(null, [[point(0.2, 0.2)]], 100, 100);
  const twoHands = [[point(0.8, 0.8)], [point(0.1, 0.1)]];
  const next = predictHandsForDisplay(first.history, twoHands, 133, 180);
  assert.deepEqual(next.hands, twoHands);
});

test("between model results the overlay advances without changing the stored result", () => {
  const first = predictHandsForDisplay(null, [[point(0.2, 0.2)]], 100, 100);
  const second = predictHandsForDisplay(first.history, [[point(0.3, 0.2)]], 133, 133);
  const projected = extrapolateHandsForDisplay(second.history, 166);
  assert.ok(projected[0][0].x > second.hands[0][0].x);
  assert.deepEqual(second.history.raw[0][0], point(0.3, 0.2));
});

test("stationary landmark noise is attenuated without altering raw input", () => {
  let state = predictHandsForDisplay(null, [[point(0.5, 0.5)]], 100, 100);
  const raw = [[point(0.501, 0.499)]];
  const before = structuredClone(raw);
  state = predictHandsForDisplay(state.history, raw, 133, 133);

  const rawTravel = Math.hypot(0.001, -0.001);
  const displayTravel = Math.hypot(
    state.hands[0][0].x - 0.5,
    state.hands[0][0].y - 0.5,
  );
  assert.ok(displayTravel < rawTravel * 0.5);
  assert.deepEqual(raw, before);
  assert.deepEqual(state.history.raw, raw);
});

test("deliberate fast movement remains responsive", () => {
  const first = predictHandsForDisplay(null, [[point(0.2, 0.2)]], 100, 100);
  const next = predictHandsForDisplay(first.history, [[point(0.3, 0.2)]], 133, 133);

  assert.ok(next.hands[0][0].x > 0.28);
  assert.ok(next.hands[0][0].x <= 0.318);
});

test("a brief missed detection holds the overlay but a sustained miss clears it", () => {
  const seen = predictHandsForDisplay(null, [[point(0.4, 0.4)]], 100, 100);
  const briefMiss = predictHandsForDisplay(seen.history, [], 170, 170);
  const longMiss = predictHandsForDisplay(briefMiss.history, [], 230, 230);

  assert.equal(briefMiss.hands.length, 1);
  assert.deepEqual(briefMiss.history.raw, seen.history.raw);
  assert.deepEqual(longMiss.hands, []);
});

test("two-hand result order is associated by wrist travel", () => {
  const first = predictHandsForDisplay(
    null,
    [[point(0.2, 0.4)], [point(0.8, 0.4)]],
    100,
    100,
  );
  const reversedDetectorOrder = [[point(0.79, 0.4)], [point(0.21, 0.4)]];
  const next = predictHandsForDisplay(first.history, reversedDetectorOrder, 133, 133);

  assert.equal(next.history.raw[0][0].x, 0.21);
  assert.equal(next.history.raw[1][0].x, 0.79);
});

test("display jitter RMS is materially below raw jitter RMS", () => {
  let history = null;
  const rawXs = [];
  const displayXs = [];
  for (let frame = 0; frame < 120; frame += 1) {
    const noise = ((frame * 37) % 11 - 5) * 0.00035;
    const rawX = 0.5 + noise;
    rawXs.push(rawX);
    const next = predictHandsForDisplay(history, [[point(rawX, 0.5 - noise)]], 100 + frame * 33, 100 + frame * 33);
    displayXs.push(next.hands[0][0].x);
    history = next.history;
  }
  const rms = (values) => Math.sqrt(values.reduce((sum, value) => sum + (value - 0.5) ** 2, 0) / values.length);

  assert.ok(rms(displayXs.slice(10)) < rms(rawXs.slice(10)) * 0.5);
});
