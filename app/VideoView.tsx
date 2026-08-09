"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { addCaptureAsset, addRecording, newId } from "./recordings";
import { CAPTURE_SCHEMA_VERSION, captureDurationMs, captureElapsedMs, captureEpochMs, stopCaptureRun, type CaptureRun } from "./capture-sync";
import { ageWeeks, clinicalAgeMetadata } from "./session-domain";
import { extrapolateHandsForDisplay, predictHandsForDisplay, type DisplayHandHistory, type TrackedPoint } from "./pose-display";
import { assessTrackingIntegrity } from "./tracking-integrity";
import { acknowledgeSam31Job, processSam31Video } from "./sam31-client";
import { CAMERA_DSP_BRIGHTNESS, cameraMediaConstraints, cameraStartErrorMessage } from "./camera-config";
import type { VideoViewProps } from "./capture-view-types";

// Model URLs, tier logic, and all inference live in app/pose-worker.ts.

// Chromium API not yet in the TS dom lib: taps a MediaStreamTrack as a
// ReadableStream of VideoFrames.
declare class MediaStreamTrackProcessor {
  constructor(init: { track: MediaStreamTrack; maxBufferSize?: number });
  readable: ReadableStream<VideoFrame>;
}

const HAND_LINKS = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];

function drawTrackedHands(ctx: CanvasRenderingContext2D, hands: TrackedPoint[][], width: number, height: number) {
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.lineWidth = Math.max(2, width / 420);
  ctx.strokeStyle = "#ffd166";
  ctx.fillStyle = "#ffffff";
  for (const landmarks of hands) {
    ctx.beginPath();
    HAND_LINKS.forEach(([a, b]) => {
      const p = landmarks[a], q = landmarks[b];
      ctx.moveTo(p.x * width, p.y * height);
      ctx.lineTo(q.x * width, q.y * height);
    });
    ctx.stroke();
    ctx.beginPath();
    landmarks.forEach((point, index) => {
      const radius = index === 0 ? 5 : 3;
      const x = point.x * width, y = point.y * height;
      ctx.moveTo(x + radius, y);
      ctx.arc(x, y, radius, 0, Math.PI * 2);
    });
    ctx.fill();
    ctx.stroke();
  }
}

// Raw measurements remain visible as small dots while the connected skeleton
// is display-stabilized. This makes subtle measured motion inspectable without
// turning detector shimmer into a violently twitching pose. It is a visual
// distinction only: recording already stores the raw coordinates directly.
function drawRawHandMeasurements(ctx: CanvasRenderingContext2D, hands: TrackedPoint[][], width: number, height: number) {
  ctx.fillStyle = "#24d2c1";
  ctx.globalAlpha = 0.9;
  for (const landmarks of hands) {
    ctx.beginPath();
    for (const point of landmarks) {
      const x = point.x * width, y = point.y * height;
      ctx.moveTo(x + 2, y);
      ctx.arc(x, y, 2, 0, Math.PI * 2);
    }
    ctx.fill();
  }
  ctx.globalAlpha = 1;
}

function paintBinaryRle(image: ImageData, runs: number[], color: [number, number, number, number]) {
  let index = 0;
  let foreground = false;
  for (const run of runs) {
    const end = Math.min(image.width * image.height, index + Math.max(0, run));
    if (foreground) {
      for (; index < end; index += 1) image.data.set(color, index * 4);
    } else index = end;
    foreground = !foreground;
  }
}
type Sam31Instance = {
  id: number;
  confidence?: number;
  bbox: [number, number, number, number];
  centroid: [number, number];
  maskWidth: number;
  maskHeight: number;
  rle: number[];
};
type PoseTrackingFrame = {
  frameIndex: number;
  sessionTimeMs: number;
  epochMs: number;
  sourceVideoTimeMs: number;
  // Hands are measured fresh on every recorded frame — never carried forward.
  hands: TrackedPoint[][];
};
type Sam31TrackingFrame = {
  frameIndex: number;
  sessionTimeMs: number;
  epochMs: number;
  sourceVideoTimeMs: number;
  segments: Sam31Instance[];
  source?: "sam31" | "optical-flow" | "sam31-native-propagation";
};
type TrackingFrame = PoseTrackingFrame | Sam31TrackingFrame;
type InferenceMode = "pose" | "sam31";
const PREVIEW_WIDTH = 300; // backing-store width of the sensor-page pose preview
// The recorded skeleton does not need camera resolution. Encoding VP9 at 720p
// in real time was the single largest cost in the capture loop; the landmarks
// are stored at full precision in the sidecar regardless, so the video is a
// visual aid and 640 wide is ample.
const POSE_CANVAS_WIDTH = 640;
const POSE_VIDEO_BITRATE = 1_800_000;
// The C922's native MJPEG mode is 1280x720 at 30 Hz. Requesting the earlier
// non-native 960x540 mode made Chromium choose its 1024x576 YUYV path, which
// is capped at 15 Hz. Inference still downsamples to 640 wide in the worker.
// The live C922 path on this machine delivers roughly 30 unique frames/second.
// Ask for that proven rate and let continuous exposure brighten the room rather
// than holding an underexposed 60 Hz shutter that the device cannot sustain.
// Modest UVC digital-brightness lift for the 60 Hz shutter. It does not alter
// exposure time or frame cadence; controlled front lighting remains the source
// of real signal and lower-noise landmark detail.
export default function VideoView({ session, captureRun, onReadyChange, onSaved, posePreviewRef, registerCameraControl }: VideoViewProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const poseCanvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const videoTrackRef = useRef<CanvasCaptureMediaStreamTrack | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const trackingFramesRef = useRef<TrackingFrame[]>([]);
  const frameIndexRef = useRef(0);
  const framingCandidateRef = useRef<{ value: "waiting" | "partial" | "ready"; since: number }>({ value: "waiting", since: 0 });
  const recordingRef = useRef(false);
  const captureRef = useRef<CaptureRun | null>(null);
  const recorderStartedOffsetRef = useRef(0);
  const [cameraState, setCameraState] = useState<"off" | "starting" | "ready" | "error">("off");
  // Mirrors cameraState for the imperative enable path, which must not depend on
  // a re-render having landed before the parent awaits it.
  const cameraStateRef = useRef<"off" | "starting" | "ready" | "error">("off");
  const [status, setStatus] = useState("Camera is off");
  const [hands, setHands] = useState(0);
  const [framing, setFraming] = useState<"waiting" | "partial" | "ready">("waiting");
  const [recording, setRecording] = useState(false);
  const [videoOnlyRun, setVideoOnlyRun] = useState<CaptureRun | null>(null);
  const [videoProcessing, setVideoProcessing] = useState(false);
  const [elapsed, setElapsed] = useState("00:00");
  const [downloadUrl, setDownloadUrl] = useState("");
  const [rawDownloadUrl, setRawDownloadUrl] = useState("");
  const [trackingDownloadUrl, setTrackingDownloadUrl] = useState("");
  const [trackingDownloadFilename, setTrackingDownloadFilename] = useState("");
  const [downloadBaseName, setDownloadBaseName] = useState("movement-pose");
  const [downloadFilename, setDownloadFilename] = useState("movement-pose.webm");
  const [trackingFps, setTrackingFps] = useState(0);
  const [cameraFps, setCameraFps] = useState(0);
  const [trackingLatencyMs, setTrackingLatencyMs] = useState(0);
  const [inputMode, setInputMode] = useState<"direct" | "scaled" | "unknown">("direct");
  const [inferenceMode, setInferenceMode] = useState<InferenceMode>("pose");
  const [modelState, setModelState] = useState<"idle" | "loading" | "running" | "error">("idle");
  const [modelError, setModelError] = useState("");

  // Cached contexts and the rate instrumentation.
  const overlayCtxRef = useRef<CanvasRenderingContext2D | null>(null);
  const poseCtxRef = useRef<CanvasRenderingContext2D | null>(null);
  const previewCtxRef = useRef<CanvasRenderingContext2D | null>(null);
  const trackedFrameCountRef = useRef(0);
  const skippedFrameCountRef = useRef(0);
  const inferSamplesRef = useRef<number[]>([]);
  const fpsWindowStartRef = useRef(0);
  const inputFrameCountRef = useRef(0);
  const inputDropCountRef = useRef(0);
  const inputWindowStartRef = useRef(0);
  const backdropRef = useRef<HTMLCanvasElement | null>(null);
  const displayHistoryRef = useRef<DisplayHandHistory | null>(null);
  const latestRawHandsRef = useRef<TrackedPoint[][]>([]);
  const handsCountRef = useRef(0);
  const framingRef = useRef<"waiting" | "partial" | "ready">("waiting");
  const [inferenceMs, setInferenceMs] = useState(0);
  // All inference happens in app/pose-worker.ts — its own thread, its own GPU
  // context, self-healing, tier-demoting. The camera track itself is
  // transferred to the worker, which reads frames directly: the main thread
  // does NO per-frame capture work at all — it only draws results as they
  // arrive. This also makes tracking immune to every main-thread scheduling
  // hazard found along the way (rVFC's occluded-window 1 fps mode, rAF
  // starvation, compositor backpressure).
  const workerRef = useRef<Worker | null>(null);
  const frameReaderRef = useRef<ReadableStreamDefaultReader<VideoFrame> | null>(null);
  const processorTrackRef = useRef<MediaStreamTrack | null>(null);
  const inFlightRef = useRef(0);
  // The detector is synchronous and serial inside one worker. Keep one
  // replaceable latest-frame slot instead of a FIFO queue: the worker can be
  // busy, but it must never process a frame that is older than the newest one
  // waiting at the camera boundary.
  const pendingFrameRef = useRef<{ frame: VideoFrame; ts: number } | null>(null);
  const flushPendingRef = useRef<(() => void) | null>(null);
  const tierRef = useRef("gpu");
  const lastResultAtRef = useRef(0);
  const restartingRef = useRef(false);
  const cameraStartRef = useRef<Promise<boolean> | null>(null);
  const [tierLabel, setTierLabel] = useState("gpu");
  const inferenceModeRef = useRef<InferenceMode>("pose");
  const captureModeRef = useRef<InferenceMode>("pose");
  const samLoopTokenRef = useRef(0);
  const samRequestAbortRef = useRef<AbortController | null>(null);
  const samInputCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const samMaskCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const samMotionRef = useRef<{ centroid: [number, number]; at: number; velocity: [number, number] }[]>([]);
  const captureSourceRef = useRef<"sync" | "video">("sync");

  const stopInferencePipeline = useCallback(() => {
    samLoopTokenRef.current += 1;
    samRequestAbortRef.current?.abort();
    samRequestAbortRef.current = null;
    frameReaderRef.current?.cancel().catch(() => {});
    frameReaderRef.current = null;
    if (pendingFrameRef.current) {
      pendingFrameRef.current.frame.close();
      pendingFrameRef.current = null;
    }
    flushPendingRef.current = null;
    inFlightRef.current = 0;
    processorTrackRef.current?.stop();
    processorTrackRef.current = null;
    workerRef.current?.terminate();
    workerRef.current = null;
    displayHistoryRef.current = null;
    latestRawHandsRef.current = [];
    samMotionRef.current = [];
  }, []);

  const drawOverlay = useCallback((displayHands: TrackedPoint[][], rawHands = latestRawHandsRef.current) => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video || canvas.offsetParent === null) return;
    const frameWidth = video.videoWidth || 960;
    const frameHeight = video.videoHeight || 540;
    const overlayWidth = Math.min(960, frameWidth);
    const overlayHeight = Math.max(1, Math.round((overlayWidth * frameHeight) / frameWidth));
    if (canvas.width !== overlayWidth || canvas.height !== overlayHeight) {
      canvas.width = overlayWidth;
      canvas.height = overlayHeight;
    }
    if (!overlayCtxRef.current) {
      overlayCtxRef.current = canvas.getContext("2d", { alpha: true, desynchronized: true });
    }
    const ctx = overlayCtxRef.current;
    if (!ctx) return;
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    drawTrackedHands(ctx, displayHands, canvas.width, canvas.height);
    drawRawHandMeasurements(ctx, rawHands, canvas.width, canvas.height);
    ctx.restore();
  }, []);

  const drawSam31Result = useCallback((instances: Sam31Instance[], sourceAt = performance.now()) => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video || canvas.offsetParent === null) return;
    const frameWidth = video.videoWidth || 960;
    const frameHeight = video.videoHeight || 540;
    const overlayWidth = Math.min(960, frameWidth);
    const overlayHeight = Math.max(1, Math.round(overlayWidth * frameHeight / frameWidth));
    if (canvas.width !== overlayWidth || canvas.height !== overlayHeight) {
      canvas.width = overlayWidth;
      canvas.height = overlayHeight;
      overlayCtxRef.current = null;
    }
    if (!overlayCtxRef.current) overlayCtxRef.current = canvas.getContext("2d", { alpha: true, desynchronized: true });
    const ctx = overlayCtxRef.current;
    if (!ctx) return;
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    const renderAt = performance.now();
    instances.forEach((instance, index) => {
      const previous = samMotionRef.current[index];
      const sampleDelta = previous ? Math.max(16, sourceAt - previous.at) : 0;
      const observedVelocity: [number, number] = previous && sampleDelta
        ? [(instance.centroid[0] - previous.centroid[0]) / sampleDelta, (instance.centroid[1] - previous.centroid[1]) / sampleDelta]
        : [0, 0];
      const velocity: [number, number] = previous
        ? [observedVelocity[0] * 0.65 + previous.velocity[0] * 0.35, observedVelocity[1] * 0.65 + previous.velocity[1] * 0.35]
        : observedVelocity;
      // The segmentation is computed on a frame from the past. A bounded
      // constant-velocity projection keeps the overlay on the live hand while
      // preserving the raw SAM mask in the downloaded sidecar.
      const ageMs = Math.max(0, renderAt - sourceAt);
      const shiftX = Math.max(-0.14, Math.min(0.14, velocity[0] * ageMs));
      const shiftY = Math.max(-0.14, Math.min(0.14, velocity[1] * ageMs));
      const shiftedBox: [number, number, number, number] = [
        Math.max(0, Math.min(1, instance.bbox[0] + shiftX)),
        Math.max(0, Math.min(1, instance.bbox[1] + shiftY)),
        Math.max(0, Math.min(1, instance.bbox[2] + shiftX)),
        Math.max(0, Math.min(1, instance.bbox[3] + shiftY)),
      ];
      const shiftedCentroid: [number, number] = [
        Math.max(0, Math.min(1, instance.centroid[0] + shiftX)),
        Math.max(0, Math.min(1, instance.centroid[1] + shiftY)),
      ];
      samMotionRef.current[index] = { centroid: instance.centroid, at: sourceAt, velocity };
      ctx.save();
      ctx.translate(shiftX * canvas.width, shiftY * canvas.height);
      let maskCanvas = samMaskCanvasRef.current;
      if (!maskCanvas) {
        maskCanvas = document.createElement("canvas");
        samMaskCanvasRef.current = maskCanvas;
      }
      maskCanvas.width = instance.maskWidth;
      maskCanvas.height = instance.maskHeight;
      const maskCtx = maskCanvas.getContext("2d");
      if (!maskCtx) return;
      const pixels = maskCtx.createImageData(instance.maskWidth, instance.maskHeight);
      paintBinaryRle(pixels, instance.rle, index ? [255, 209, 102, 86] : [36, 210, 193, 86]);
      maskCtx.putImageData(pixels, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(maskCanvas, 0, 0, canvas.width, canvas.height);
      const [x0, y0, x1, y1] = shiftedBox;
      ctx.strokeStyle = index ? "#ffd166" : "#24d2c1";
      ctx.lineWidth = 2;
      ctx.strokeRect(x0 * canvas.width, y0 * canvas.height, (x1 - x0) * canvas.width, (y1 - y0) * canvas.height);
      ctx.beginPath();
      ctx.arc(shiftedCentroid[0] * canvas.width, shiftedCentroid[1] * canvas.height, 4, 0, Math.PI * 2);
      ctx.fillStyle = ctx.strokeStyle;
      ctx.fill();
      ctx.restore();
    });
    ctx.restore();

    const poseCanvas = poseCanvasRef.current;
    if (!poseCanvas || (!recordingRef.current && !posePreviewRef.current?.offsetParent)) return;
    poseCanvas.width = canvas.width;
    poseCanvas.height = canvas.height;
    if (!poseCtxRef.current) poseCtxRef.current = poseCanvas.getContext("2d", { alpha: false });
    poseCtxRef.current!.fillStyle = "#10171b";
    poseCtxRef.current!.fillRect(0, 0, poseCanvas.width, poseCanvas.height);
    poseCtxRef.current!.drawImage(canvas, 0, 0);
    const preview = posePreviewRef.current;
    if (preview?.offsetParent) {
      const height = Math.max(1, Math.round(PREVIEW_WIDTH * poseCanvas.height / poseCanvas.width));
      preview.width = PREVIEW_WIDTH;
      preview.height = height;
      if (!previewCtxRef.current) previewCtxRef.current = preview.getContext("2d");
      previewCtxRef.current?.drawImage(poseCanvas, 0, 0, preview.width, preview.height);
    }
  }, [posePreviewRef]);

  // The browser compositor paints the camera at camera rate. This callback
  // only paints the lightweight overlay and optional skeleton recording.
  const drawResult = useCallback((displayHands: TrackedPoint[][], recordedHands: TrackedPoint[][]) => {
    const poseCanvas = poseCanvasRef.current;
    const video = videoRef.current;
    if (!poseCanvas) return;
    const frameWidth = video?.videoWidth || 960;
    const frameHeight = video?.videoHeight || 540;
    drawOverlay(displayHands, recordedHands);

    // The live view only exists on the Video tab. While the operator is on
    // the Sensor tab the whole pane is display:none, so painting it every
    // tick is pure waste — and that is exactly when a capture is usually
    // running.
    const poseWidth = Math.min(POSE_CANVAS_WIDTH, frameWidth);
    const poseHeight = Math.max(1, Math.round((poseWidth * frameHeight) / frameWidth));
    if (poseCanvas.width !== poseWidth || poseCanvas.height !== poseHeight) {
      poseCanvas.width = poseWidth; poseCanvas.height = poseHeight;
    }
    // Contexts are cached: getContext is not free at 30–60 calls a second.
    if (!poseCtxRef.current) poseCtxRef.current = poseCanvas.getContext("2d", { alpha: false });
    const poseCtx = poseCtxRef.current;
    if (!poseCtx) return;

    // This second canvas is the only canvas sent to MediaRecorder. It contains
    // landmarks on a neutral background and never receives webcam pixels.
    // The recording canvas and its picture-in-picture copy only need painting
    // when something consumes them. On the Video tab, with no capture running,
    // nobody does — and that work was competing with inference for the frame.
    const preview = posePreviewRef.current;
    const previewVisible = !!preview && preview.offsetParent !== null;
    if (!recordingRef.current && !previewVisible) return;

    // The backdrop and its grid never change, so they are rasterised once and
    // blitted afterwards rather than re-stroked on every frame.
    let backdrop = backdropRef.current;
    if (!backdrop || backdrop.width !== poseCanvas.width || backdrop.height !== poseCanvas.height) {
      backdrop = document.createElement("canvas");
      backdrop.width = poseCanvas.width; backdrop.height = poseCanvas.height;
      const backdropCtx = backdrop.getContext("2d");
      if (backdropCtx) {
        backdropCtx.fillStyle = "#10171b";
        backdropCtx.fillRect(0, 0, backdrop.width, backdrop.height);
        // One path for the whole grid rather than ~25 separate stroke calls.
        backdropCtx.strokeStyle = "rgba(255,255,255,.06)";
        backdropCtx.lineWidth = 1;
        backdropCtx.beginPath();
        for (let x = 0; x <= backdrop.width; x += Math.max(40, backdrop.width / 16)) { backdropCtx.moveTo(x, 0); backdropCtx.lineTo(x, backdrop.height); }
        for (let y = 0; y <= backdrop.height; y += Math.max(40, backdrop.height / 9)) { backdropCtx.moveTo(0, y); backdropCtx.lineTo(backdrop.width, y); }
        backdropCtx.stroke();
      }
      backdropRef.current = backdrop;
    }

    poseCtx.save();
    poseCtx.setTransform(1, 0, 0, 1, 0, 0);
    poseCtx.drawImage(backdrop, 0, 0);
    poseCtx.translate(poseCanvas.width, 0); poseCtx.scale(-1, 1);
    drawTrackedHands(poseCtx, recordedHands, poseCanvas.width, poseCanvas.height);
    poseCtx.restore();

    // Mirror the skeleton into the sensor page's picture-in-picture. This is a
    // copy of the pose canvas, never the webcam, so the preview shows exactly
    // what gets recorded.
    if (preview && previewVisible && poseCanvas.width) {
      const targetHeight = Math.max(1, Math.round((PREVIEW_WIDTH * poseCanvas.height) / poseCanvas.width));
      if (preview.width !== PREVIEW_WIDTH || preview.height !== targetHeight) {
        preview.width = PREVIEW_WIDTH;
        preview.height = targetHeight;
      }
      if (!previewCtxRef.current) previewCtxRef.current = preview.getContext("2d");
      previewCtxRef.current?.drawImage(poseCanvas, 0, 0, preview.width, preview.height);
    }
  }, [drawOverlay, posePreviewRef]);

  // Handles every message from the inference worker: draws, updates the
  // status panel, and appends to the recording sidecar.
  const handleWorkerMessage = useCallback((message: {
    type: string; tier?: string; error?: string; skipped?: boolean;
    ts?: number; videoTimeMs?: number; inferMs?: number; inferSampleMs?: number; inputMode?: string;
    hands?: TrackedPoint[][];
  }) => {
    if (message.type === "recover-failed") {
      console.warn(`[tracking] worker-error ${message.error ?? "unknown"}`);
      return;
    }
    if (message.type !== "result") return;
    lastResultAtRef.current = performance.now();
    inFlightRef.current = Math.max(0, inFlightRef.current - 1);
    // Hand the newest waiting frame to the serial worker immediately. This is
    // deliberately before the skipped-result return so an inference error
    // cannot strand the reader with a permanently full handoff slot.
    flushPendingRef.current?.();
    if (message.skipped) { skippedFrameCountRef.current += 1; return; }

    const now = performance.now();
    const rawHands = message.hands ?? [];
    tierRef.current = message.tier ?? tierRef.current;
    const capturedAt = message.ts ?? now;
    const endToEndMs = Math.max(0, now - capturedAt);
    if (typeof message.inferSampleMs === "number" && Number.isFinite(message.inferSampleMs)) inferSamplesRef.current.push(message.inferSampleMs);

    // Rolling effective frame rate, for the readout beside the camera state.
    trackedFrameCountRef.current += 1;
    if (now - fpsWindowStartRef.current >= 1000) {
      setTrackingFps(Math.round((trackedFrameCountRef.current * 1000) / (now - fpsWindowStartRef.current)));
      setInferenceMs(message.inferMs ?? 0);
      setTierLabel(tierRef.current);
      setTrackingLatencyMs(Math.round(endToEndMs));
      setInputMode(message.inputMode === "scaled" ? "scaled" : message.inputMode === "direct" ? "direct" : "unknown");
      const inputWindowMs = now - inputWindowStartRef.current;
      const inferSamples = inferSamplesRef.current.slice().sort((a, b) => a - b);
      const p95Index = Math.min(inferSamples.length - 1, Math.floor(inferSamples.length * 0.95));
      const p95InferMs = inferSamples.length ? inferSamples[p95Index] : message.inferMs ?? 0;
      const maxInferMs = inferSamples.length ? inferSamples[inferSamples.length - 1] : message.inferMs ?? 0;
      console.info(`[tracking] pipeline ${JSON.stringify({
        inputFps: Math.round(inputFrameCountRef.current * 1000 / inputWindowMs),
        resultFps: Math.round(trackedFrameCountRef.current * 1000 / inputWindowMs),
        skipped: skippedFrameCountRef.current,
        dropped: inputDropCountRef.current,
        inferMs: message.inferMs ?? 0,
        p95InferMs,
        maxInferMs,
        tier: tierRef.current,
        inputMode: message.inputMode ?? "unknown",
      })}`);
      trackedFrameCountRef.current = 0;
      skippedFrameCountRef.current = 0;
      fpsWindowStartRef.current = now;
      inputFrameCountRef.current = 0;
      inputDropCountRef.current = 0;
      inputWindowStartRef.current = now;
      inferSamplesRef.current = [];
    }

    // The analysis path never receives display smoothing.
    // Raw measurements drive framing, recording, and the cyan measurement
    // dots. Only the connected yellow operator skeleton is display-stabilized.
    const inFrame = rawHands.filter((landmarks) => landmarks.some((point) => point.x > .02 && point.x < .98 && point.y > .02 && point.y < .98)).length;
    const nextFraming = !rawHands.length ? "waiting" : inFrame === rawHands.length ? "ready" : "partial";
    if (framingCandidateRef.current.value !== nextFraming) framingCandidateRef.current = { value: nextFraming, since: now };
    else if (now - framingCandidateRef.current.since >= 350 && framingRef.current !== nextFraming) {
      framingRef.current = nextFraming;
      setFraming(nextFraming);
    }
    if (handsCountRef.current !== rawHands.length) {
      handsCountRef.current = rawHands.length;
      setHands(rawHands.length);
    }
    const displayRaw = [...rawHands].sort((a, b) => (a[0]?.x ?? 0) - (b[0]?.x ?? 0));
    latestRawHandsRef.current = rawHands;
    const predicted = predictHandsForDisplay(displayHistoryRef.current, displayRaw, capturedAt, now);
    displayHistoryRef.current = predicted.history;
    drawResult(predicted.hands, rawHands);

    const run = captureRef.current;
    if (recordingRef.current && run) {
      // Rounding by arithmetic rather than toFixed: this runs 42 landmarks x
      // 4 fields per recorded frame, and the string round-trip was showing up.
      const compact = (point: TrackedPoint): TrackedPoint => ({
        x: Math.round(point.x * 1e5) / 1e5, y: Math.round(point.y * 1e5) / 1e5, z: Math.round(point.z * 1e5) / 1e5,
        ...(point.visibility == null ? {} : { visibility: Math.round(point.visibility * 1e4) / 1e4 }),
      });
      // Timestamps are from frame CAPTURE, not result arrival — the sidecar
      // stays aligned to the camera even if inference cost varies.
      trackingFramesRef.current.push({
        frameIndex: frameIndexRef.current++,
        sessionTimeMs: captureElapsedMs(run, capturedAt),
        epochMs: captureEpochMs(run, capturedAt),
        sourceVideoTimeMs: message.videoTimeMs ?? 0,
        hands: rawHands.map((landmarks) => landmarks.map(compact)),
      });
      // captureStream(0) records only explicitly requested frames. This makes
      // the sidecar frame index and the pose-video frame order share one clock.
      videoTrackRef.current?.requestFrame();
    }
  }, [drawResult]);
  const handleWorkerMessageRef = useRef(handleWorkerMessage);
  handleWorkerMessageRef.current = handleWorkerMessage;

  // The camera/model result cadence is hardware-dependent. Paint the latest
  // bounded prediction on every display tick so the visible skeleton stays
  // fluid without pretending that extra raw measurements were captured.
  useEffect(() => {
    let frame = 0;
    const paint = (now: number) => {
      const history = displayHistoryRef.current;
      if (history && inferenceModeRef.current === "pose") drawOverlay(extrapolateHandsForDisplay(history, now));
      frame = window.requestAnimationFrame(paint);
    };
    frame = window.requestAnimationFrame(paint);
    return () => window.cancelAnimationFrame(frame);
  }, [drawOverlay]);

  // Decode cadence is independent from inference cadence. Showing both makes
  // a camera/driver cap distinguishable from a slow model without adding a
  // per-frame React update or another paint loop.
  useEffect(() => {
    if (cameraState !== "ready") { setCameraFps(0); return; }
    const video = videoRef.current;
    if (!video?.getVideoPlaybackQuality) return;
    let lastFrames = video.getVideoPlaybackQuality().totalVideoFrames;
    let lastAt = performance.now();
    const id = window.setInterval(() => {
      const now = performance.now();
      const frames = video.getVideoPlaybackQuality().totalVideoFrames;
      const elapsedMs = now - lastAt;
      if (elapsedMs > 0) setCameraFps(Math.round((frames - lastFrames) * 1000 / elapsedMs));
      lastFrames = frames;
      lastAt = now;
    }, 1000);
    return () => window.clearInterval(id);
  }, [cameraState]);

  // Spin up (or replace) the inference worker, wait for its models to come
  // online, then start feeding it camera frames. MediaStreamTrackProcessor
  // taps a clone of the camera track and each VideoFrame is TRANSFERRED to
  // the worker — zero copies, no per-frame canvas work. When the worker is
  // busy, frames are closed instead of queued, so it always sees the latest
  // camera image and backlog cannot build up. (The track itself would be the
  // cleaner transfer, but MediaStreamTrack is not transferable without a
  // Chromium feature flag; VideoFrame is.)
  const startPoseWorker = async (): Promise<void> => {
    const track = streamRef.current?.getVideoTracks()[0];
    if (!track) throw new Error("camera track not available");
    stopInferencePipeline();
    // A CLASSIC worker, deliberately: MediaPipe's wasm bootstrap calls
    // importScripts, which module workers forbid ("ModuleFactory not set").
    // The bundler compiles this file and its imports into a single IIFE.
    const worker = new Worker(new URL("./pose-worker.ts", import.meta.url));
    workerRef.current = worker;
    worker.onmessage = (event) => handleWorkerMessageRef.current(event.data);
    await new Promise<void>((resolve, reject) => {
      const onInit = (event: MessageEvent) => {
        if (event.data?.type === "ready") {
          tierRef.current = event.data.tier; setTierLabel(event.data.tier);
          worker.removeEventListener("message", onInit); resolve();
        } else if (event.data?.type === "init-failed") {
          worker.removeEventListener("message", onInit); reject(new Error(event.data.error));
        }
      };
      worker.addEventListener("message", onInit);
      worker.postMessage({ type: "init" });
    });

    const processorTrack = track.clone();
    processorTrackRef.current = processorTrack;
    // Keep a one-frame pull buffer. The reader remains active while inference
    // is busy, replacing a single pending frame so the worker always receives
    // the newest available camera image rather than building a stale queue.
    const processor = new MediaStreamTrackProcessor({ track: processorTrack, maxBufferSize: 1 });
    const reader = processor.readable.getReader();
    frameReaderRef.current = reader;
    inFlightRef.current = 0;
    lastResultAtRef.current = performance.now();
    inputFrameCountRef.current = 0;
    inputDropCountRef.current = 0;
    skippedFrameCountRef.current = 0;
    inferSamplesRef.current = [];
    inputWindowStartRef.current = performance.now();
    const send = (next: { frame: VideoFrame; ts: number }) => {
      if (workerRef.current !== worker) { next.frame.close(); return; }
      inFlightRef.current = 1;
      worker.postMessage({ type: "frame", frame: next.frame, ts: next.ts }, [next.frame]);
    };
    // Called after every inference. At most one latest frame is retained, and
    // an arriving frame replaces (and closes) the old pending one instead of
    // extending a stale FIFO queue.
    flushPendingRef.current = () => {
      if (inFlightRef.current !== 0) return;
      const pending = pendingFrameRef.current;
      if (!pending) return;
      pendingFrameRef.current = null;
      send(pending);
    };
    (async () => {
      for (;;) {
        const { value: frame, done } = await reader.read();
        if (done) break;
        if (!frame) continue;
        inputFrameCountRef.current += 1;
        // A replaced worker means this loop is stale — stop it.
        if (workerRef.current !== worker) { frame.close(); reader.cancel().catch(() => {}); processorTrack.stop(); break; }
        const next = { frame, ts: performance.now() };
        if (inFlightRef.current === 0 && !pendingFrameRef.current) send(next);
        else {
          // Keep the newest camera image, not the oldest queued image. The
          // replaced frame is explicitly closed so it cannot leak GPU memory.
          if (pendingFrameRef.current) {
            pendingFrameRef.current.frame.close();
            inputDropCountRef.current += 1;
          }
          pendingFrameRef.current = next;
        }
      }
    })().catch(() => {}).finally(() => {
      if (pendingFrameRef.current) {
        pendingFrameRef.current.frame.close();
        pendingFrameRef.current = null;
      }
      flushPendingRef.current = null;
      if (processorTrackRef.current === processorTrack) {
        processorTrack.stop();
        processorTrackRef.current = null;
      }
    });
  };
  const startPoseWorkerRef = useRef(startPoseWorker);
  startPoseWorkerRef.current = startPoseWorker;

  const startSam31 = async (): Promise<void> => {
    const video = videoRef.current;
    if (!video) throw new Error("camera video is not available");
    stopInferencePipeline();
    setModelState("loading");
    setModelError("");
    setStatus("Loading experimental SAM 3.1…");
    const loadController = new AbortController();
    samRequestAbortRef.current = loadController;
    const loadResponse = await fetch("http://127.0.0.1:4831/load", { method: "POST", signal: loadController.signal });
    const loadResult = await loadResponse.json().catch(() => ({})) as { error?: string };
    if (!loadResponse.ok) throw new Error(loadResult.error || "SAM 3.1 service could not load the model");
    if (inferenceModeRef.current !== "sam31") return;

    setModelState("running");
    setStatus("Camera connected · experimental SAM 3.1 active");
    const token = ++samLoopTokenRef.current;
    let resultCount = 0;
    let windowStarted = performance.now();
    const loop = async () => {
      if (token !== samLoopTokenRef.current || inferenceModeRef.current !== "sam31" || cameraStateRef.current !== "ready") return;
      let input = samInputCanvasRef.current;
      if (!input) {
        input = document.createElement("canvas");
        samInputCanvasRef.current = input;
      }
      input.width = 640;
      input.height = Math.max(1, Math.round(640 * (video.videoHeight || 720) / (video.videoWidth || 1280)));
      input.getContext("2d", { alpha: false })?.drawImage(video, 0, 0, input.width, input.height);
      const jpeg = await new Promise<Blob | null>((resolve) => input!.toBlob(resolve, "image/jpeg", 0.76));
      if (!jpeg || token !== samLoopTokenRef.current) return;
      const capturedAt = performance.now();
      const controller = new AbortController();
      samRequestAbortRef.current = controller;
      try {
        const response = await fetch("http://127.0.0.1:4831/infer", {
          method: "POST",
          headers: { "Content-Type": "image/jpeg" },
          body: jpeg,
          signal: controller.signal,
        });
        const result = await response.json() as { instances?: Sam31Instance[]; inferenceMs?: number; error?: string };
        if (!response.ok) throw new Error(result.error || "SAM 3.1 inference failed");
        if (token !== samLoopTokenRef.current) return;
        const instances = result.instances ?? [];
        drawSam31Result(instances, capturedAt);
        setHands(instances.length);
        setFraming(instances.length ? "ready" : "waiting");
        setInferenceMs(Math.round(result.inferenceMs ?? 0));
        setTrackingLatencyMs(Math.round(performance.now() - capturedAt));
        resultCount += 1;
        const now = performance.now();
        if (now - windowStarted >= 1000) {
          setTrackingFps(Math.round(resultCount * 1000 / (now - windowStarted)));
          resultCount = 0;
          windowStarted = now;
        }
        const run = captureRef.current;
        if (recordingRef.current && run) {
          trackingFramesRef.current.push({
            frameIndex: frameIndexRef.current++,
            sessionTimeMs: captureElapsedMs(run, capturedAt),
            epochMs: captureEpochMs(run, capturedAt),
            sourceVideoTimeMs: video.currentTime * 1000,
            segments: instances,
          });
          videoTrackRef.current?.requestFrame();
        }
      } catch (error) {
        if (controller.signal.aborted || token !== samLoopTokenRef.current) return;
        const message = error instanceof Error ? error.message : "SAM 3.1 inference failed";
        setModelState("error");
        setModelError(message);
        setStatus("SAM 3.1 unavailable");
        return;
      }
      window.setTimeout(loop, 0);
    };
    void loop();
  };
  const startSam31Ref = useRef(startSam31);
  startSam31Ref.current = startSam31;

  const processSamRecordedVideo = async (blob: Blob, run: CaptureRun, onJobStarted?: (jobId: string) => void | Promise<void>): Promise<{ frames: Sam31TrackingFrame[]; annotatedBlob: Blob; processingMs: number; samKeyframes: number }> => {
    const controller = new AbortController();
    samRequestAbortRef.current = controller;
    setStatus("SAM 3.1 full propagation · uploading video");
    try {
      const result = await processSam31Video<Sam31Instance>(blob, {
        signal: controller.signal,
        onJobStarted,
        onProgress: (job) => {
          setStatus(`SAM 3.1 full propagation · ${job.phase ?? "processing"} · ${Math.round(job.progress ?? 0)}%${job.processedFrames ? ` · ${job.processedFrames}/${job.frameCount ?? "?"} frames` : ""}`);
        },
        onPhase: (phase) => {
          setStatus(phase === "retrieving-video"
            ? "SAM 3.1 native tracking complete · retrieving annotated video"
            : "SAM 3.1 native tracking complete · retrieving tracking metadata");
        },
      });
      const frames = result.frames.map((frame) => ({
        ...frame,
        sessionTimeMs: frame.sourceVideoTimeMs,
        epochMs: run.startedAtEpochMs + frame.sourceVideoTimeMs,
      }));
      return { frames, annotatedBlob: result.annotatedBlob, processingMs: result.processingMs, samKeyframes: frames.length };
    } finally {
      if (samRequestAbortRef.current === controller) samRequestAbortRef.current = null;
    }
  };

  const startVideoOnlyRecording = async () => {
    if (inferenceModeRef.current !== "sam31" || recordingRef.current || captureRun || videoProcessing) return;
    if (cameraStateRef.current !== "ready") {
      const opened = await enableCamera();
      if (!opened) return;
    }
    const run: CaptureRun = {
      id: newId(),
      active: true,
      startedAtEpochMs: Date.now(),
      startedAtPerfMs: performance.now(),
      studyId: `Patient ${session.patientNumber} · video-only`,
      note: "SAM 3.1 video-only capture",
    };
    setVideoOnlyRun(run);
  };

  const stopVideoOnlyRecording = () => {
    if (!videoOnlyRun?.active) return;
    setVideoOnlyRun(stopCaptureRun(videoOnlyRun));
  };

  const startSelectedInference = async (mode: InferenceMode) => {
    if (mode === "sam31") await startSam31Ref.current();
    else {
      setModelState("running");
      setModelError("");
      await startPoseWorkerRef.current();
    }
  };
  const selectInferenceMode = async (mode: InferenceMode) => {
    if (mode === inferenceModeRef.current || recordingRef.current || videoProcessing) return;
    stopInferencePipeline();
    inferenceModeRef.current = mode;
    setInferenceMode(mode);
    setHands(0);
    setFraming("waiting");
    setTrackingFps(0);
    setInferenceMs(0);
    setTrackingLatencyMs(0);
    const ctx = canvasRef.current?.getContext("2d");
    if (ctx && canvasRef.current) ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    if (cameraStateRef.current !== "ready") {
      setModelState("idle");
      setStatus("Camera is off");
      return;
    }
    setModelState("loading");
    setStatus(mode === "sam31" ? "Switching to experimental SAM 3.1…" : "Switching to hand pose…");
    try {
      await startSelectedInference(mode);
      if (mode === "pose") setStatus("Camera connected · hand pose active");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Inference could not start";
      setModelState("error");
      setModelError(message);
      setStatus(mode === "sam31" ? "SAM 3.1 unavailable" : "Hand pose unavailable");
    }
  };

  // Watchdog: results should arrive continuously. Three silent seconds means the
  // worker died or wedged — replace it wholesale. The camera stream lives on
  // the main thread, so a fresh clone of the track restarts frame delivery.
  useEffect(() => {
    if (cameraState !== "ready" || inferenceMode !== "pose") return;
    const id = window.setInterval(() => {
      if (performance.now() - lastResultAtRef.current > 3000 && !restartingRef.current) {
        restartingRef.current = true;
        console.warn("[tracking] worker went silent — replacing it");
        startPoseWorkerRef.current()
          .catch((error) => console.error("[tracking] worker restart failed", error))
          .finally(() => { restartingRef.current = false; lastResultAtRef.current = performance.now(); });
      }
    }, 1000);
    return () => window.clearInterval(id);
  }, [cameraState, inferenceMode]);

  const enableCamera = async (): Promise<boolean> => {
    if (cameraStateRef.current === "ready") return true;
    if (cameraStateRef.current === "starting") return cameraStartRef.current ?? false;
    cameraStateRef.current = "starting";
    const pending = (async () => {
    setCameraState("starting"); setStatus("Starting camera and movement tracking…");
    try {
      // `exact` mode requests make Chromium surface a generic NotReadableError
      // for several Windows driver negotiation failures. That was incorrectly
      // shown as "another application" even when Windows reported no active
      // camera client. Ask for the desired profile while allowing the camera
      // driver to return its closest native 720p/30 mode.
      const stream = await navigator.mediaDevices.getUserMedia(cameraMediaConstraints());
      streamRef.current = stream;
      const cameraTrack = stream.getVideoTracks()[0];
      const capabilities = cameraTrack?.getCapabilities?.() as MediaTrackCapabilities & {
        exposureMode?: string[];
        brightness?: { min: number; max: number; step: number };
      };
      // The C922 retains the prior hardware exposure mode across streams. Make
      // the bright 30 fps profile explicit; merely omitting the old manual
      // constraint leaves its short, underexposed shutter active.
      if (cameraTrack && capabilities?.exposureMode?.includes("continuous")) {
        await cameraTrack.applyConstraints({
          advanced: [{ exposureMode: "continuous" } as MediaTrackConstraintSet],
        });
      }
      if (cameraTrack && capabilities?.brightness) {
        const range = capabilities.brightness;
        const target = Math.min(range.max, Math.max(range.min, CAMERA_DSP_BRIGHTNESS));
        const brightness = range.min + Math.round((target - range.min) / range.step) * range.step;
        try {
          await cameraTrack.applyConstraints({ advanced: [{ brightness } as MediaTrackConstraintSet] });
        } catch (error) {
          // The camera stays usable if a driver exposes brightness capability
          // metadata but rejects the live control.
          console.warn("[camera] brightness control unavailable", error);
        }
      }
      console.info(`[camera] negotiation ${JSON.stringify({
        label: cameraTrack?.label,
        settings: cameraTrack?.getSettings?.(),
        capabilities,
      })}`);
      if (videoRef.current) { videoRef.current.srcObject = stream; await videoRef.current.play(); }
      setCameraState("ready"); cameraStateRef.current = "ready";
      try {
        await startSelectedInference(inferenceModeRef.current);
        setStatus(inferenceModeRef.current === "sam31" ? "Camera connected · experimental SAM 3.1 active" : "Camera connected · hand pose active");
      } catch (inferenceError) {
        const message = inferenceError instanceof Error ? inferenceError.message : "Inference could not start";
        setModelState("error");
        setModelError(message);
        if (inferenceModeRef.current === "sam31") {
          setStatus("Camera connected · SAM 3.1 unavailable");
          return true;
        }
        throw inferenceError;
      }
      return true;
    } catch (error) {
      console.error(error);
      setCameraState("error"); cameraStateRef.current = "error";
      setStatus(cameraStartErrorMessage(error));
      stopInferencePipeline();
      streamRef.current?.getTracks().forEach((track) => track.stop()); streamRef.current=null;
      return false;
    }
    })();
    cameraStartRef.current = pending;
    try { return await pending; }
    finally { if (cameraStartRef.current === pending) cameraStartRef.current = null; }
  };

  // The parent calls this to switch the camera on when a synchronized capture
  // starts from the sensor page. Registered once; the ref keeps it current.
  const enableCameraRef = useRef(enableCamera);
  enableCameraRef.current = enableCamera;
  useEffect(() => { registerCameraControl(() => enableCameraRef.current()); }, [registerCameraControl]);

  useEffect(() => {
    const activeRun = captureRun ?? videoOnlyRun;
    if (!activeRun) return;
    if (activeRun.active && !recordingRef.current) {
      const poseCanvas = poseCanvasRef.current;
      if (!poseCanvas || cameraState !== "ready") return;
      if (downloadUrl) URL.revokeObjectURL(downloadUrl);
      if (rawDownloadUrl) URL.revokeObjectURL(rawDownloadUrl);
      if (trackingDownloadUrl) URL.revokeObjectURL(trackingDownloadUrl);
      setDownloadUrl(""); setRawDownloadUrl(""); setTrackingDownloadUrl(""); chunksRef.current=[]; trackingFramesRef.current=[];
      frameIndexRef.current = 0;
      captureRef.current = activeRun;
      captureSourceRef.current = captureRun ? "sync" : "video";
      captureModeRef.current = inferenceModeRef.current;
      const savedMode = captureModeRef.current;
      if (savedMode === "sam31") {
        // SAM is analyzed after capture. Stop its live request loop so the
        // camera and MediaRecorder never wait on segmentation.
        samLoopTokenRef.current += 1;
        samRequestAbortRef.current?.abort();
        samRequestAbortRef.current = null;
      }
      // Use a cloned camera track for SAM recording so the recorder starts on
      // the Record press without taking ownership of the preview stream.
      const stream = savedMode === "sam31" ? streamRef.current?.clone() : poseCanvas.captureStream(0);
      if (!stream) return;
      videoTrackRef.current = savedMode === "sam31" ? null : stream.getVideoTracks()[0] as CanvasCaptureMediaStreamTrack;
      const mimeType = MediaRecorder.isTypeSupported("video/webm;codecs=vp9") ? "video/webm;codecs=vp9" : "video/webm";
      const recorder = new MediaRecorder(stream,{mimeType, videoBitsPerSecond: POSE_VIDEO_BITRATE}); recorderRef.current=recorder;
      recorderStartedOffsetRef.current = captureElapsedMs(activeRun, performance.now());
      recorder.ondataavailable=(event)=>{ if(event.data.size) chunksRef.current.push(event.data); };
      recorder.onstop=async()=>{
        const run = captureRef.current ?? activeRun;
        let frames = trackingFramesRef.current;
        const durationMs = captureDurationMs(run);
        const stoppedAt = run.stoppedAtEpochMs ?? Date.now();
        const blob=new Blob(chunksRef.current,{type:mimeType});
        if (savedMode === "sam31") {
          // Recording has ended, so release camera textures before the full
          // SAM model claims VRAM. Keeping this stream alive was unnecessary
          // and contributed directly to Electron's renderer being evicted.
          stream.getTracks().forEach((track)=>track.stop());
          videoTrackRef.current = null;
          streamRef.current?.getTracks().forEach((track)=>track.stop());
          streamRef.current = null;
          if (videoRef.current) videoRef.current.srcObject = null;
          cameraStateRef.current = "off";
          setCameraState("off");
        }
        const captureSource = captureSourceRef.current;
        const baseName=`patient-${session.patientNumber}-${session.suspected?"susp":"non"}-wk${ageWeeks(session)}-${run.id.slice(0,8)}`;
        const recordingId = newId();
        const rawSamFilename = `${baseName}-sam31-raw.webm`;
        const provisionalSamRecording = savedMode === "sam31" ? {
          id:recordingId, patientNumber:session.patientNumber, suspected:session.suspected, ageYears:session.ageYears, ageMonths:session.ageMonths, ageDays:session.ageDays,
          ...clinicalAgeMetadata(session), studyDate:session.studyDate, weightKg:session.weightKg, studyId:run.studyId, note:run.note, kind:"pose" as const,
          date:stoppedAt, blob, filename:rawSamFilename, size:blob.size, annotationStatus:"processing" as const,
          thumbnail:poseCanvasRef.current?.toDataURL("image/png"), captureSessionId:captureSource === "sync" ? run.id : undefined,
          sync:{ schemaVersion:CAPTURE_SCHEMA_VERSION, clock:"performance-time-origin" as const, startedAtEpochMs:run.startedAtEpochMs, streamStartOffsetMs:recorderStartedOffsetRef.current, sampleCount:0 },
        } : null;
        let savedBlob = blob;
        let rawBlob: Blob | undefined;
        let samProcessingMs = 0;
        let samKeyframes = 0;
        let processingError = "";
        let durableSamJobId = "";
        if (savedMode === "sam31") {
          setVideoProcessing(true);
          setStatus("SAM 3.1 post-processing · starting");
          try {
            if (!provisionalSamRecording) throw new Error("Raw SAM recording metadata was not created");
            // Commit the untouched recording before inference. A renderer or
            // GPU-process loss can no longer erase the captured source video.
            await addRecording(provisionalSamRecording);
            const processed = await processSamRecordedVideo(blob, run, async (samJobId) => {
              durableSamJobId = samJobId;
              await addRecording({ ...provisionalSamRecording, samJobId, samPipelineVersion:"sam31-native-v12" });
            });
            frames = processed.frames;
            savedBlob = processed.annotatedBlob;
            rawBlob = blob;
            samProcessingMs = processed.processingMs;
            samKeyframes = processed.samKeyframes;
            setStatus(`Annotated video saved · ${frames.length} tracked frames`);
          } catch (error) {
            if (error instanceof DOMException && error.name === "AbortError" && durableSamJobId) { setVideoProcessing(false); return; }
            processingError = error instanceof Error ? error.message : "SAM 3.1 post-processing failed";
            if (error instanceof DOMException && error.name === "AbortError") processingError = "SAM processing was interrupted before a recoverable job was created. The raw video was preserved.";
            setModelError(processingError);
            setStatus("Video saved · SAM post-processing failed");
          }
        }
        const integrity = savedMode === "pose"
          ? assessTrackingIntegrity(frames.filter((frame): frame is PoseTrackingFrame => "hands" in frame))
          : {
              rawLandmarks: false,
              frameCount: frames.length,
              frameIndexSequential: frames.every((frame, index) => frame.frameIndex === index),
              timestampsMonotonic: frames.every((frame, index) => index === 0 || frame.sessionTimeMs >= frames[index - 1].sessionTimeMs),
              framesWithSegments: frames.filter((frame) => "segments" in frame && frame.segments.length > 0).length,
            };
        const sidecar = new Blob([JSON.stringify({
          schemaVersion: CAPTURE_SCHEMA_VERSION,
          capture: { sessionId: run.id, patientNumber: session.patientNumber, studyId: run.studyId, note: run.note, studyDate: session.studyDate, ...clinicalAgeMetadata(session), ageBasis: session.useCorrectedAge ? "corrected" : "chronological", weightKg: session.weightKg, suspected: session.suspected, startedAtEpochMs: run.startedAtEpochMs, startedAt: new Date(run.startedAtEpochMs).toISOString(), durationMs },
          synchronization: { clock: "performance-time-origin", unit: "milliseconds", zero: "capture-start", recorderStartedOffsetMs: recorderStartedOffsetRef.current, frameOrderMatchesPoseVideo: true },
          tracking: savedMode === "pose"
            ? { observedFrameRateHz: durationMs ? Number((frames.length * 1000 / durationMs).toFixed(3)) : 0, coordinateSpace: "normalized-camera", handLandmarksPerHand: 21, handsMeasuredEveryFrame: true, rawCameraStored: false, visualization: "hands-only", inferenceBackend: tierRef.current, integrity }
            : { observedFrameRateHz: durationMs ? Number((frames.length * 1000 / durationMs).toFixed(3)) : 0, coordinateSpace: "normalized-camera", output: "native-hand-mask-rle-bbox-centroid", maskResolution: [640, 360], rawCameraStored: true, visualization: "native-mask-overlay", inferenceBackend: "Meta SAM 3.1 native propagate_in_video", prompt: "human hand on frame 0", nativeTrackedFrames: samKeyframes, processingMs: samProcessingMs, postProcessed: true, processingError: processingError || undefined, integrity },
          frames,
        })], { type: "application/json" });
        const filename = `${baseName}-${savedMode === "sam31" && !processingError ? "sam31-tracked.mp4" : savedMode === "sam31" ? "sam31-raw.webm" : "pose.webm"}`;
        const rawFilename = savedMode === "sam31" && rawBlob ? `${baseName}-sam31-raw.webm` : undefined;
        const sidecarFilename = `${baseName}-${savedMode === "sam31" ? "segments" : "landmarks"}.json`;
        const annotationFailed = savedMode === "sam31" && Boolean(processingError);
        setDownloadBaseName(baseName); setDownloadFilename(filename); setDownloadUrl(annotationFailed ? "" : URL.createObjectURL(savedBlob)); setRawDownloadUrl(annotationFailed ? URL.createObjectURL(blob) : rawBlob ? URL.createObjectURL(rawBlob) : ""); setTrackingDownloadUrl(URL.createObjectURL(sidecar)); setTrackingDownloadFilename(sidecarFilename);
        stream.getTracks().forEach((track)=>track.stop());
        videoTrackRef.current = null;
        addRecording({ id:recordingId, patientNumber:session.patientNumber, suspected:session.suspected, ageYears:session.ageYears, ageMonths:session.ageMonths, ageDays:session.ageDays, ...clinicalAgeMetadata(session), studyDate:session.studyDate, weightKg:session.weightKg, studyId:run.studyId, note:run.note, kind:"pose", date:stoppedAt, blob:savedBlob, filename, size:savedBlob.size + (rawBlob?.size ?? 0), rawBlob, rawFilename, annotationStatus:savedMode === "sam31" ? (processingError ? "failed" : "complete") : undefined, processingError:processingError || undefined, thumbnail:poseCanvasRef.current?.toDataURL("image/png"), sidecarBlob:sidecar, sidecarFilename, captureSessionId:captureSource === "sync" ? run.id : undefined, sync:{ schemaVersion:CAPTURE_SCHEMA_VERSION, clock:"performance-time-origin", startedAtEpochMs:run.startedAtEpochMs, streamStartOffsetMs:recorderStartedOffsetRef.current, sampleCount:frames.length } })
          .then(() => captureSource === "sync" ? addCaptureAsset(run.id, { recordingId, kind:"pose", filename, sidecarFilename, sampleCount:frames.length, streamStartOffsetMs:recorderStartedOffsetRef.current, size:savedBlob.size + (rawBlob?.size ?? 0) + sidecar.size, metadata:savedMode === "sam31" ? { annotationStatus:annotationFailed ? "failed" : "complete", processingError:processingError || undefined } : undefined }) : undefined)
          .then(() => durableSamJobId && !annotationFailed ? acknowledgeSam31Job(durableSamJobId).catch(() => {}) : undefined)
          .then(() => { setVideoProcessing(false); if (savedMode === "sam31") void enableCameraRef.current(); if (captureSource === "sync") onSaved("pose", !annotationFailed); })
          .catch(() => { setVideoProcessing(false); if (savedMode === "sam31") void enableCameraRef.current(); if (captureSource === "sync") onSaved("pose", false); });
        captureRef.current = null;
        if (captureSource === "video") setVideoOnlyRun(null);
      };
      recordingRef.current=true;
      recorder.start(1000);
      setStatus(savedMode === "sam31" ? "Recording raw video · SAM runs after Stop" : "Recording video and sensors");
      setRecording(true); setElapsed("00:00");
      return;
    }
    if (!activeRun.active && recordingRef.current) {
      recordingRef.current=false;
      captureRef.current = activeRun;
      if (recorderRef.current?.state === "recording") recorderRef.current.stop();
      setRecording(false);
    }
  }, [cameraState, captureRun, downloadUrl, onSaved, rawDownloadUrl, session, trackingDownloadUrl, videoOnlyRun]);

  useEffect(() => {
    if (!recording) return;
    const timer = window.setInterval(() => { const total=Math.floor(captureElapsedMs(captureRef.current!, performance.now())/1000); setElapsed(`${String(Math.floor(total/60)).padStart(2,"0")}:${String(total%60).padStart(2,"0")}`); },250);
    return () => window.clearInterval(timer);
  }, [recording]);

  useEffect(() => () => {
    recordingRef.current = false;
    if (recorderRef.current?.state === "recording") recorderRef.current.stop();
    stopInferencePipeline();
    streamRef.current?.getTracks().forEach((track) => track.stop());
  }, [stopInferencePipeline]);

  useEffect(() => { onReadyChange(cameraState === "ready"); }, [cameraState, onReadyChange]);
  useEffect(() => () => onReadyChange(false), [onReadyChange]);

  return (
    <section className="video-view" aria-label="Video movement tracking">
      <div className="video-stage">
        <video ref={videoRef} muted playsInline aria-hidden="true" />
        <canvas ref={canvasRef} />
        <canvas ref={poseCanvasRef} className="pose-recording-canvas" aria-hidden="true" />
        {cameraState !== "ready" && <div className="camera-empty"><div className="camera-outline" /><h2>Video movement capture</h2><p>Choose an inference mode, then enable the camera.</p><button className="enable-camera" type="button" onClick={enableCamera} disabled={cameraState==="starting"}>{cameraState==="starting" ? "Starting…" : "Enable camera"}</button></div>}
        {cameraState === "ready" && <div className="tracking-badge"><span className={`live-dot ${modelState === "error" ? "error" : ""}`} />{modelState === "loading" ? "Loading model" : modelState === "error" ? "Inference unavailable" : framing === "ready" ? "Hands in frame" : framing === "partial" ? "Keep hands fully in view" : "Looking for hands"}</div>}
        {cameraState === "ready" && inferenceMode === "pose" && <div className="overlay-key"><span><i className="hand-key" />Stable pose</span><span><i className="raw-key" />Raw landmarks</span></div>}
        {cameraState === "ready" && inferenceMode === "sam31" && <div className="overlay-key"><span><i className="raw-key" />SAM hand mask</span><span><i className="hand-key" />Second mask</span></div>}
      </div>
      <aside className="video-panel">
        <div className="inference-mode-switch" role="group" aria-label="Video inference model">
          <button type="button" className={inferenceMode === "pose" ? "active" : ""} aria-pressed={inferenceMode === "pose"} disabled={recording || videoProcessing} onClick={() => void selectInferenceMode("pose")}>Hand pose</button>
          <button type="button" className={inferenceMode === "sam31" ? "active experimental" : "experimental"} aria-pressed={inferenceMode === "sam31"} disabled={recording || videoProcessing} onClick={() => void selectInferenceMode("sam31")}><span>Experimental</span>SAM 3.1</button>
        </div>
        {inferenceMode === "sam31" && <button type="button" className={`sam-record-button ${videoOnlyRun?.active ? "recording" : ""}`} onClick={() => videoOnlyRun?.active ? stopVideoOnlyRecording() : void startVideoOnlyRecording()} disabled={videoProcessing || (!!captureRun && !videoOnlyRun?.active) || cameraState === "starting"}>{videoOnlyRun?.active ? "Stop video" : videoProcessing ? "Processing video…" : "Record video"}</button>}
        <div><span className="eyebrow">{inferenceMode === "sam31" ? "Segmentation tracking" : "Hand tracking"}</span><h2>Hand movement</h2><p>{inferenceMode === "sam31" ? "Live preview is optional. Record the smooth camera video first; SAM 3.1 analyzes it afterward into hand masks, boxes, and centroids." : "The yellow skeleton is stabilized for viewing. Cyan dots show each unsmoothed measurement; recording saves those raw timestamped landmarks."}</p></div>
        {modelError && inferenceMode === "sam31" && <div className="model-error" role="alert">{modelError}</div>}
        <dl><div><dt>Camera</dt><dd>{cameraState === "ready" ? `Connected${cameraFps ? ` · ${cameraFps} fps` : ""}` : "Not connected"}</dd></div><div><dt>{inferenceMode === "sam31" ? "Masks detected" : "Hands detected"}</dt><dd>{cameraState === "ready" ? hands : "—"}</dd></div><div><dt>Framing</dt><dd>{cameraState === "ready" ? framing === "ready" ? "Ready" : framing === "partial" ? "Partial" : "Waiting" : "—"}</dd></div><div><dt>Tracking rate</dt><dd>{cameraState === "ready" ? `${trackingFps} fps · ${inferenceMs} ms · ${trackingLatencyMs} ms total${inferenceMode === "pose" && inputMode !== "unknown" ? ` · ${inputMode}` : ""}${inferenceMode === "pose" && tierLabel !== "gpu" ? ` · ${tierLabel}` : ""}` : "—"}</dd></div></dl>
        <div className="capture-guide"><strong>Before recording</strong><span>Keep the hands visible and avoid moving the camera.</span></div>
        <p className="privacy-note">{inferenceMode === "sam31" ? "The annotated video, untouched source video, and mask data" : "Pose video and landmark data"} stay on this device. Tracking output is experimental and is not a diagnosis.</p>
      </aside>
      <footer className="record-bar video-record-bar"><div className="timer">{elapsed}</div><div className="ready-state"><span className={`status-dot ${cameraState==="ready"?"connected":""}`} />{status}</div><div className="video-downloads">{downloadUrl && <a className="download-link" href={downloadUrl} download={downloadFilename}>Download annotated video</a>}{rawDownloadUrl && <a className="download-link secondary" href={rawDownloadUrl} download={`${downloadBaseName}-sam31-raw.webm`}>Download raw video</a>}{trackingDownloadUrl && <a className="download-link" href={trackingDownloadUrl} download={trackingDownloadFilename}>Download tracking data</a>}</div><span className="capture-controlled">{recording ? (inferenceMode === "sam31" ? "Recording raw video · SAM runs after Stop" : "Recording with sensors") : videoProcessing ? "SAM 3.1 processing raw video" : inferenceMode === "sam31" ? "Ready for video-only capture" : "Ready for synchronized capture"}</span></footer>
    </section>
  );
}

