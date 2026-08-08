/// <reference lib="webworker" />
// Hand-tracking inference worker. The camera preview is rendered independently
// by the page, so this worker only returns landmarks. Keeping pixels out of the
// return path removes a full ImageBitmap allocation and compositor transfer on
// every tracked frame. VideoFrame keeps the camera texture on the handoff
// path; only the detector input is bounded inside the worker.

import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

const WASM_PATH = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm";
const HAND_MODEL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
// Preserve the 512x288 spatial input used by the validated adult-hand run.
// Camera acquisition can operate at 60 Hz independently; inference reports
// its real measured rate and drops stale frames rather than duplicating data.
const INFERENCE_MAX_WIDTH = 512;
const WARMUP_FRAMES = 12;
const DIRECT_SLOW_MS = 45;
const DIRECT_SLOW_RESULTS = 2;

type Tier = { label: string; delegate: "GPU" | "CPU" };
const TIERS: Tier[] = [
  { label: "gpu", delegate: "GPU" },
  { label: "cpu", delegate: "CPU" },
];

let hand: HandLandmarker | null = null;
let tier = 0;
let emaMs = 0;
// A bounded 512 px input is intentional. The hand model resizes internally,
// but handing it a full 1280x720 camera frame can make the detector jump to
// 100+ ms when a hand is present on some C922/Chromium paths. Keeping the
// camera preview at 1280x720 while inference sees 512x288 preserves framing
// and landmark coordinates while keeping the work in the realtime budget.
let useScaledInput = true;
let directSlowResults = 0;
let inferenceCanvas: OffscreenCanvas | null = null;
let inferenceContext: OffscreenCanvasRenderingContext2D | null = null;

async function createDetector(): Promise<void> {
  const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
  const next = await HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: HAND_MODEL, delegate: TIERS[tier].delegate },
    runningMode: "VIDEO",
    numHands: 2,
    // Slightly lower tracking confidence helps preserve a hand through fast
    // finger movement while detection/presence remain conservative enough to
    // avoid turning background edges into hands.
    minHandDetectionConfidence: 0.42,
    minHandPresenceConfidence: 0.42,
    minTrackingConfidence: 0.35,
  });
  const old = hand;
  hand = next;

  // MediaPipe's GPU delegate lazily compiles its graph on the first real
  // frames. If we expose the camera as "Connected" immediately, operators
  // see the exact 5–7 fps / 100+ ms warm-up stall in the live overlay. Run a
  // short blank-frame warm-up first so the first visible result is already on
  // the steady-state path. It does not create recording data.
  try {
    const width = INFERENCE_MAX_WIDTH;
    const height = Math.max(1, Math.round(INFERENCE_MAX_WIDTH * 9 / 16));
    const canvas = new OffscreenCanvas(width, height);
    const context = canvas.getContext("2d", { alpha: false }) as OffscreenCanvasRenderingContext2D | null;
    context?.fillRect(0, 0, width, height);
    for (let i = 0; i < WARMUP_FRAMES; i += 1) {
      next.detectForVideo(canvas as unknown as ImageBitmap, i + 1);
    }
  } catch {
    // A browser that rejects OffscreenCanvas still gets normal VideoFrame
    // inference; do not fail camera startup just because warm-up is unavailable.
  }
  old?.close();
}

function scaledInput(frame: VideoFrame): OffscreenCanvas | VideoFrame {
  if (frame.displayWidth <= INFERENCE_MAX_WIDTH) return frame;
  const width = INFERENCE_MAX_WIDTH;
  const height = Math.max(1, Math.round(width * frame.displayHeight / frame.displayWidth));
  if (!inferenceCanvas || inferenceCanvas.width !== width || inferenceCanvas.height !== height) {
    inferenceCanvas = new OffscreenCanvas(width, height);
    inferenceContext = inferenceCanvas.getContext("2d", {
      alpha: false,
      desynchronized: true,
    }) as OffscreenCanvasRenderingContext2D | null;
  }
  if (!inferenceContext || !inferenceCanvas) return frame;
  inferenceContext.drawImage(frame, 0, 0, width, height);
  return inferenceCanvas;
}

function detectFrame(frame: VideoFrame, ts: number): { hands: unknown[][]; mode: "direct" | "scaled" } {
  // Use the bounded worker canvas first. MediaPipe's model input is much
  // smaller than the camera frame, so this avoids paying the full-resolution
  // texture upload cost on every hand-present frame.
  if (!useScaledInput) {
    try {
      return { hands: hand!.detectForVideo(frame as unknown as ImageBitmap, ts).landmarks, mode: "direct" };
    } catch {
      // A few Chromium builds do not accept VideoFrame in a worker. Fall back
      // once to the explicitly supported canvas source and keep that choice
      // stable for the rest of this worker lifetime.
      useScaledInput = true;
    }
  }

  const input = scaledInput(frame);
  try {
    return { hands: hand!.detectForVideo(input as unknown as ImageBitmap, ts).landmarks, mode: "scaled" };
  } catch (error) {
    // If the canvas path is rejected, retry direct once so a transient canvas
    // failure cannot strand tracking. The worker will return to the bounded
    // canvas as soon as the direct path reports a slow sample.
    if (input === frame) throw error;
    useScaledInput = false;
    return { hands: hand!.detectForVideo(frame as unknown as ImageBitmap, ts).landmarks, mode: "direct" };
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

  if (message.type !== "frame") return;
  const frame: VideoFrame = message.frame;
  const ts: number = message.ts;
  if (!hand) {
    frame.close();
    postMessage({ type: "result", ts, videoTimeMs: 0, hands: [], inferMs: 0, skipped: true, tier: TIERS[tier].label });
    return;
  }

  const videoTimeMs = frame.timestamp != null ? Math.round(frame.timestamp / 1000) : 0;
  const startedAt = performance.now();
  try {
    const detected = detectFrame(frame, ts);
    const hands = detected.hands;
    const spent = performance.now() - startedAt;
    emaMs = emaMs ? emaMs * 0.88 + spent * 0.12 : spent;
    // If a runtime rejects the canvas and forces direct input, switch back to
    // the bounded canvas as soon as the direct path is measurably slow. This
    // is a performance decision, not a lower-confidence or synthetic-data
    // fallback; raw landmarks keep the same schema either way.
    if (detected.mode === "direct" && emaMs > DIRECT_SLOW_MS) directSlowResults += 1;
    else if (detected.mode === "direct") directSlowResults = 0;
    if (detected.mode === "direct" && directSlowResults >= DIRECT_SLOW_RESULTS) {
      useScaledInput = true;
      directSlowResults = 0;
      emaMs = 0;
      postMessage({ type: "input-mode", mode: "scaled", reason: "direct-slow" });
    }
    postMessage({
      type: "result",
      ts,
      videoTimeMs,
      hands,
      inferMs: Math.round(emaMs * 10) / 10,
      inferSampleMs: Math.round(spent * 10) / 10,
      skipped: false,
      tier: TIERS[tier].label,
      inputMode: detected.mode,
    });
  } catch (error) {
    postMessage({ type: "result", ts, videoTimeMs, hands: [], inferMs: 0, skipped: true, tier: TIERS[tier].label });
    postMessage({ type: "recover-failed", error: String(error) });
  } finally {
    frame.close();
  }
};
