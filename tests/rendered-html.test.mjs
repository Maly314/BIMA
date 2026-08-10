import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render(pathname = "/") {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request(`http://localhost${pathname}`, { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server renders the movement capture application", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>BIMA<\/title>/i);
  assert.match(html, /Patients &amp; recordings/);
  assert.match(html, /Add patient/);
  assert.match(html, /Calibrate sensors/);
  assert.match(html, /Data archive/);
  assert.match(html, /Choose data folder/);
  assert.doesNotMatch(html, /Your site is taking shape|codex-preview/i);
});

test("desktop workspace fills the available viewport instead of collapsing to content height", async () => {
  const css = await readFile(new URL("../app/globals.css", import.meta.url), "utf8");

  assert.match(css, /\.app-window\s*\{[\s\S]*?min-height:calc\(100vh - 60px\)/);
  assert.match(css, /\.app-window\s*\{[\s\S]*?display:flex;\s*flex-direction:column/);
  assert.match(css, /\.landing-body\s*\{\s*flex:1;\s*display:flex;\s*flex-direction:column/);
  assert.match(css, /\.landing-empty\s*\{\s*flex:1;/);
});

test("hand capture includes synchronized hand, sensor, and privacy-preserving recording support", async () => {
  const [pageSource, landing, sensorView, videoView, recordingTypes, recordingDatabase, worker, sam31Client, sam31Recovery, cameraConfig] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/Landing.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/SensorView.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/VideoView.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/recording-types.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/recording-database.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/pose-worker.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/sam31-client.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/sam31-recovery.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/camera-config.ts", import.meta.url), "utf8"),
  ]);
  const page = `${pageSource}\n${landing}\n${sensorView}\n${videoView}`;
  const [desktop, sam31Service, sam31ChunkWorker] = await Promise.all([
    readFile(new URL("../desktop/main.cjs", import.meta.url), "utf8"),
    readFile(new URL("../desktop/sam31_service.py", import.meta.url), "utf8"),
    readFile(new URL("../desktop/sam31_chunk_worker.py", import.meta.url), "utf8"),
  ]);

  // Inference runs off the main thread. The compositor keeps live video fluid
  // while one latest frame is tracked at a time; display prediction is never
  // written into the raw training sidecar.
  assert.match(worker, /HandLandmarker/);
  assert.doesNotMatch(worker, /PoseLandmarker/);
  assert.match(worker, /VideoFrame keeps the camera texture/);
  assert.match(worker, /inputMode: detected\.mode/);
  assert.match(worker, /inferSampleMs/);
  assert.match(page, /pose-worker\.ts/);
  assert.match(page, /MediaStreamTrackProcessor/);
  assert.match(page, /predictHandsForDisplay/);
  assert.match(page, /pendingFrameRef/);
  assert.match(page, /flushPendingRef\.current/);
  assert.match(page, /maxBufferSize: 1/);
  assert.match(page, /pendingFrameRef\.current\.frame\.close/);
  assert.doesNotMatch(page, /transferFromImageBitmap/);
  assert.doesNotMatch(worker, /createImageBitmap/);
  assert.match(page, /Hand movement/);
  assert.match(page, /framing/);
  assert.match(page, /sidecarFilename/);
  assert.match(page, /sessionTimeMs/);
  assert.match(page, /chartDrawAtRef/);
  assert.match(page, /now - chartDrawAtRef\.current >= 66/);
  assert.match(page, /cameraMediaConstraints/);
  assert.match(cameraConfig, /CAMERA_WIDTH = 1280/);
  assert.match(cameraConfig, /CAMERA_HEIGHT = 720/);
  assert.match(cameraConfig, /frameRate: \{ ideal: TARGET_CAMERA_FPS, max: TARGET_CAMERA_FPS \}/);
  assert.match(cameraConfig, /TARGET_CAMERA_FPS = 30/);
  assert.match(page, /getVideoPlaybackQuality/);
  assert.match(page, /trackingLatencyMs/);
  assert.match(page, /inputMode/);
  assert.match(page, /p95InferMs/);
  assert.match(page, /maxInferMs/);
  assert.match(page, /skippedFrameCountRef/);
  assert.match(page, /worker-error/);
  assert.match(desktop, /getAppMetrics/);
  assert.match(desktop, /workingSetMb/);
  assert.match(page, /captureStream\(0\)/);
  assert.match(page, /requestFrame\(\)/);
  assert.match(page, /Record together/);
  assert.match(page, /Sensor CSV/);
  assert.match(page, /rawCameraStored:\s*false/);
  assert.match(page, /assessTrackingIntegrity/);
  assert.match(page, /integrity/);
  assert.match(page, /untouched source video/);
  assert.match(page, /Download annotated video/);
  assert.match(page, /Download raw video/);
  assert.match(page, /Download tracking data/);
  assert.match(page, />SAM 3\.1</);
  assert.match(page, /stopInferencePipeline\(\)/);
  assert.match(page, /workerRef\.current\?\.terminate\(\)/);
  assert.match(page, /samRequestAbortRef\.current\?\.abort\(\)/);
  assert.match(sam31Service, /build_sam3_predictor/);
  assert.match(sam31Service, /version="sam3\.1"/);
  assert.match(sam31Service, /_process_video/);
  assert.match(sam31Service, /process-video-full/);
  assert.match(sam31Service, /sam31-native-v12/);
  assert.match(sam31Service, /FULL_VIDEO_CHUNK_FRAMES/);
  assert.match(sam31Service, /sam31_chunk_worker\.py/);
  assert.match(sam31Service, /expandable_segments:True/);
  assert.match(sam31ChunkWorker, /torch\.inference_mode\(\), torch\.autocast/);
  assert.match(sam31Service, /propagate_in_video/);
  assert.match(page, /processSam31Video/);
  assert.match(sam31Client, /process-video-full/);
  assert.match(sam31Client, /sam31-native-v12/);
  assert.match(sam31Client, /BIMA version mismatch/);
  assert.match(page, /annotationStatus:"processing"/);
  assert.match(page, /await addRecording\(provisionalSamRecording\)/);
  assert.match(page, /if \(savedMode === "sam31"\)[\s\S]*?videoRef\.current\.srcObject = null/);
  assert.match(videoView, /recordingRef\.current \|\| videoProcessing/);
  assert.equal((videoView.match(/disabled=\{recording \|\| videoProcessing\}/g) ?? []).length, 2);
  assert.match(videoView, /onSaved\("pose", !annotationFailed\)/);
  assert.match(videoView, /streamRef\.current\?\.getTracks\(\)\.forEach\(\(track\)=>track\.stop\(\)\)/);
  assert.match(page, /samJobId, samPipelineVersion:"sam31-native-v12"/);
  assert.match(landing, /recoverSam31Recording/);
  assert.match(sam31Recovery, /recoveredAfterRendererRestart: true/);
  assert.match(sam31Recovery, /annotationStatus: "failed"/);
  assert.match(cameraConfig, /width: \{ ideal: CAMERA_WIDTH \}/);
  assert.doesNotMatch(cameraConfig, /width:\{exact:CAMERA_WIDTH\}/);
  assert.match(page, /annotationFailed \? ""/);
  assert.match(page, /Meta SAM 3\.1 native propagate_in_video/);
  assert.match(sam31Service, /calcOpticalFlowPyrLK/);
  assert.match(sam31Service, /estimateAffinePartial2D/);
  assert.match(sam31Service, /h264_nvenc/);
  assert.match(sam31Service, /yuv420p/);
  assert.doesNotMatch(sam31Service, /(?:import|from)\s+.*mediapipe/i);
  assert.match(desktop, /startSam31Server/);
  assert.match(page, /Tracking output is experimental and is not a diagnosis/);
  assert.match(recordingTypes, /sidecarBlob\?: Blob/);
  assert.match(recordingTypes, /rawBlob\?: Blob/);
  assert.match(recordingTypes, /captureSessionId\?: string/);
  assert.match(recordingDatabase, /capture-sessions/);
  assert.match(recordingTypes, /"pose"/);
});

test("four sensors are labelled by limb and bound to fixed physical IMUs", async () => {
  const page = await readFile(new URL("../app/SensorView.tsx", import.meta.url), "utf8");

  for (const limb of ["Left arm", "Right arm", "Left leg", "Right leg"]) {
    assert.match(page, new RegExp(`label: "${limb}"`));
  }
  for (const site of ["Left wrist", "Right wrist", "Left ankle", "Right ankle"]) {
    assert.match(page, new RegExp(`placement: "${site}"`));
  }

  // Rows must map to a fixed physical IMU. Binding to the detected-sensor list
  // would relabel every limb as soon as one sensor dropped out.
  assert.doesNotMatch(page, /const imu = imuMap\[index\]/);
  assert.doesNotMatch(page, /imuMapRef\.current\.forEach/);
  assert.match(page, /row\[`s\$\{idx \+ 1\}_imu`\] = idx \+ 1/);
});

test("the 3D board reports how far it is from its calibrated orientation", async () => {
  const [page, board, boardCss] = await Promise.all([
    readFile(new URL("../app/SensorView.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/SensorBoard3D.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/SensorBoard3D.module.css", import.meta.url), "utf8"),
  ]);

  assert.match(page, /function angleBetweenDeg/);
  assert.match(page, /LEVEL_ON_DEG/);
  assert.match(page, /LEVEL_NEAR_DEG/);
  // Falls back to true flat when the sensor has no usable calibration yet.
  assert.match(page, /calibrationIsUsable\(calibRef\.current\[i\]\)/);
  assert.match(page, /"LEVEL"/);

  assert.match(board, /levelRef/);
  assert.match(board, /data-level="off"/);
  assert.match(boardCss, /\[data-level="on"\]/);
  assert.match(boardCss, /\[data-level="near"\]/);
});

test("patient weight accepts kilograms or pounds and normalizes storage to kilograms", async () => {
  const [landing, sessionDomain] = await Promise.all([
    readFile(new URL("../app/Landing.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/session-domain.ts", import.meta.url), "utf8"),
  ]);
  assert.match(landing, /weightUnit/);
  assert.match(sessionDomain, /0\.45359237/);
  assert.match(landing, /setWeightUnit\("kg"\).*?>kg<\/button>/);
  assert.match(landing, /setWeightUnit\("lb"\).*?>lb<\/button>/);
  assert.match(sessionDomain, /weightKg/);
});

test("patient age is calculated from birth date and gestational age", async () => {
  const [page, correctedAge] = await Promise.all([
    readFile(new URL("../app/Landing.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/corrected-age.ts", import.meta.url), "utf8"),
  ]);
  assert.match(page, /Date of birth/);
  assert.match(page, /Gestational age at birth/);
  assert.match(page, /Corrected age/);
  assert.match(page, /postmenstrualAgeDays/);
  assert.match(correctedAge, /TERM_REFERENCE_DAYS\s*=\s*40\s*\*\s*7/);
  assert.match(correctedAge, /chronologicalAgeDays\s*-\s*prematurityCorrectionDays/);
  assert.match(correctedAge, /CORRECTED_AGE_USE_LIMIT_DAYS\s*=\s*2\s*\*\s*365/);
});
