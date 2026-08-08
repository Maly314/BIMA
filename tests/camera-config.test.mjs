import assert from "node:assert/strict";
import test from "node:test";
import {
  CAMERA_HEIGHT,
  CAMERA_WIDTH,
  TARGET_CAMERA_FPS,
  cameraMediaConstraints,
  cameraStartErrorMessage,
} from "../app/camera-config.ts";

test("camera requests the proven profile without brittle exact constraints", () => {
  const constraints = cameraMediaConstraints();
  assert.deepEqual(constraints, {
    video: {
      width: { ideal: CAMERA_WIDTH },
      height: { ideal: CAMERA_HEIGHT },
      frameRate: { ideal: TARGET_CAMERA_FPS, max: TARGET_CAMERA_FPS },
      facingMode: { ideal: "user" },
    },
    audio: false,
  });
  assert.equal("exact" in constraints.video.width, false);
});

test("camera failures retain distinct operator guidance", () => {
  assert.match(cameraStartErrorMessage({ name: "NotFoundError" }), /No camera found/);
  assert.match(cameraStartErrorMessage({ name: "NotReadableError" }), /Camera could not start/);
  assert.match(cameraStartErrorMessage({ name: "OverconstrainedError" }), /does not support/);
  assert.equal(cameraStartErrorMessage(new Error("denied")), "Camera unavailable or permission denied");
});
