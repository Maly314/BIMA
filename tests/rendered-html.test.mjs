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
  assert.doesNotMatch(html, /Your site is taking shape|codex-preview/i);
});

test("hand capture includes synchronized hand, sensor, and privacy-preserving recording support", async () => {
  const [page, recordings, worker] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/recordings.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/pose-worker.ts", import.meta.url), "utf8"),
  ]);

  // Inference runs off the main thread in the worker; the page feeds it
  // VideoFrames and draws each frame together with its own landmarks, so the
  // skeleton can never trail the image it is drawn on.
  assert.match(worker, /HandLandmarker/);
  assert.doesNotMatch(worker, /PoseLandmarker/);
  assert.match(page, /pose-worker\.ts/);
  assert.match(page, /MediaStreamTrackProcessor/);
  assert.match(page, /transferFromImageBitmap/);
  assert.match(page, /Hand movement/);
  assert.match(page, /framing/);
  assert.match(page, /sidecarFilename/);
  assert.match(page, /sessionTimeMs/);
  assert.match(page, /captureStream\(0\)/);
  assert.match(page, /requestFrame\(\)/);
  assert.match(page, /Record together/);
  assert.match(page, /Sensor CSV/);
  assert.match(page, /rawCameraStored:\s*false/);
  assert.match(page, /Raw camera frames are not saved/);
  assert.match(page, /Download pose video/);
  assert.match(page, /Download landmark data/);
  assert.match(page, /Tracking output is experimental and is not a diagnosis/);
  assert.match(recordings, /sidecarBlob\?: Blob/);
  assert.match(recordings, /captureSessionId\?: string/);
  assert.match(recordings, /capture-sessions/);
  assert.match(recordings, /"pose"/);
});

test("four sensors are labelled by limb and bound to fixed physical IMUs", async () => {
  const page = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");

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
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
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
  const page = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  assert.match(page, /weightUnit/);
  assert.match(page, /0\.45359237/);
  assert.match(page, /setWeightUnit\("kg"\).*?>kg<\/button>/);
  assert.match(page, /setWeightUnit\("lb"\).*?>lb<\/button>/);
  assert.match(page, /weightKg: normalizedWeightKg/);
});

test("patient age is calculated from birth date and gestational age", async () => {
  const [page, correctedAge] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
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
