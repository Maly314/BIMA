/// <reference lib="webworker" />
// Hand-tracking inference worker.
//
// Inference lives here — NOT on the page's main thread — for two measured
// reasons:
//   1. MediaPipe's GPU delegate decays under the page's compositing load
//      (~12 ms/frame rotting to ~90 ms within seconds, worse the larger the
//      window). In a worker it has its own WebGL context, isolated from the
//      compositor.
//   2. A synchronous detectForVideo on the main thread blocks React, canvas
//      drawing and the serial parser for its whole duration. Here it blocks
//      nobody.
//
// The page taps the camera track with MediaStreamTrackProcessor and transfers
// each VideoFrame here (zero-copy). After inference the SAME frame is
// transferred back with its landmarks, so the page draws the image and its
// own skeleton together — the skeleton sits exactly on the hand, which no
// amount of speed can achieve if the skeleton is painted onto a live layer
// showing a newer frame. The page drops frames instead of queueing while this
// worker is busy, so backlog cannot build up.
//
// Self-healing: if sustained cost degrades, the detector is rebuilt; if a
// rebuild does not cure it, the worker demotes GPU → CPU rather than
// silently limping.

import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

const WASM_PATH = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm";
const HAND_MODEL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

type Tier = { label: string; delegate: "GPU" | "CPU" };
const TIERS: Tier[] = [
  { label: "gpu", delegate: "GPU" },
  { label: "cpu", delegate: "CPU" },
];

// Healthy hand inference is well under a 30 fps frame budget on any modern
// machine; sustained cost past DEGRADED_MS means the delegate has rotted.
const DEGRADED_MS = 55;
const DEGRADED_HOLD_MS = 2000;
const RECOVER_COOLDOWN_MS = 10000;

let hand: HandLandmarker | null = null;
let tier = 0;
let emaMs = 0;
let degradedSince = 0;
let lastRecoverAt = 0;
let lastRecoverWasRebuild = false;
let recovering = false;

async function createDetector(): Promise<void> {
  const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
  const next = await HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: HAND_MODEL, delegate: TIERS[tier].delegate },
    runningMode: "VIDEO", numHands: 2,
    minHandDetectionConfidence: 0.45, minHandPresenceConfidence: 0.45, minTrackingConfidence: 0.45,
  });
  const old = hand;
  hand = next;
  old?.close();
}

function maybeRecover(now: number): void {
  if (emaMs <= DEGRADED_MS) { degradedSince = 0; return; }
  if (!degradedSince) { degradedSince = now; return; }
  if (now - degradedSince < DEGRADED_HOLD_MS) return;
  if (recovering || now - lastRecoverAt < RECOVER_COOLDOWN_MS) return;

  recovering = true;
  lastRecoverAt = now;
  // A rebuild that did not cure the decay means this tier cannot hold
  // real-time here — demote instead of thrashing.
  if (lastRecoverWasRebuild && tier < TIERS.length - 1) {
    tier += 1;
    lastRecoverWasRebuild = false;
  } else {
    lastRecoverWasRebuild = true;
  }
  createDetector()
    .then(() => {
      emaMs = 0;
      degradedSince = 0;
      postMessage({ type: "recovered", tier: TIERS[tier].label });
    })
    .catch((error) => postMessage({ type: "recover-failed", error: String(error) }))
    .finally(() => { recovering = false; });
}

// tasks-vision accepts VideoFrame directly in current Chromium; if this build
// ever rejects it, fall back to an in-worker ImageBitmap conversion once and
// keep going.
let needsBitmap = false;

async function detectFrame(frame: VideoFrame, ts: number) {
  let input: VideoFrame | ImageBitmap = frame;
  if (needsBitmap) input = await createImageBitmap(frame);
  try {
    return hand!.detectForVideo(input as unknown as ImageBitmap, ts).landmarks;
  } catch (error) {
    if (needsBitmap) throw error;
    needsBitmap = true;
    input = await createImageBitmap(frame);
    return hand!.detectForVideo(input as unknown as ImageBitmap, ts).landmarks;
  } finally {
    if (input !== frame) (input as ImageBitmap).close();
  }
}

self.onmessage = async (event: MessageEvent) => {
  const message = event.data;

  if (message.type === "init") {
    try {
      await createDetector();
      postMessage({ type: "ready", tier: TIERS[tier].label });
    } catch (error) {
      let initialised = false;
      while (!initialised && tier < TIERS.length - 1) {
        tier += 1;
        try { await createDetector(); initialised = true; } catch { /* next tier */ }
      }
      if (initialised) postMessage({ type: "ready", tier: TIERS[tier].label });
      else postMessage({ type: "init-failed", error: String(error) });
    }
    return;
  }

  if (message.type === "frame") {
    const frame: VideoFrame = message.frame;
    // ts comes from the page's clock, so sidecar rows land on the same
    // timebase as the CaptureRun and the sensor stream.
    const ts: number = message.ts;
    if (!hand || recovering) {
      frame.close();
      postMessage({ type: "result", ts, videoTimeMs: 0, hands: [], inferMs: 0, skipped: true, tier: TIERS[tier].label });
      return;
    }
    const videoTimeMs = frame.timestamp != null ? Math.round(frame.timestamp / 1000) : 0;
    const t0 = performance.now();
    let hands: unknown[][] = [];
    let bitmap: ImageBitmap | null = null;
    try {
      hands = await detectFrame(frame, ts);
      // The image goes back as an ImageBitmap so the page can hand it to a
      // bitmaprenderer canvas — a compositor-direct path. Rasterising the
      // frame through a 2D canvas on the page was measured poisoning the GPU
      // delegate when the window was maximised (49→190 ms/frame); this path
      // does not.
      bitmap = await createImageBitmap(frame);
    } catch (error) {
      frame.close();
      postMessage({ type: "result", ts, videoTimeMs, hands: [], inferMs: 0, skipped: true, tier: TIERS[tier].label });
      postMessage({ type: "recover-failed", error: String(error) });
      return;
    }
    frame.close();

    const spent = performance.now() - t0;
    emaMs = emaMs ? emaMs * 0.9 + spent * 0.1 : spent;

    // Image and the landmarks measured on it travel together, so the page can
    // never draw them out of step.
    postMessage({
      type: "result",
      ts,
      videoTimeMs,
      hands,
      inferMs: Math.round(emaMs * 10) / 10,
      skipped: false,
      tier: TIERS[tier].label,
      bitmap,
    }, [bitmap]);
    maybeRecover(performance.now());
  }
};
