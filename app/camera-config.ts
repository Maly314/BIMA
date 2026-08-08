export const CAMERA_WIDTH = 1280;
export const CAMERA_HEIGHT = 720;
export const TARGET_CAMERA_FPS = 30;
export const CAMERA_DSP_BRIGHTNESS = 160;

export function cameraMediaConstraints(): MediaStreamConstraints {
  return {
    video: {
      width: { ideal: CAMERA_WIDTH },
      height: { ideal: CAMERA_HEIGHT },
      frameRate: { ideal: TARGET_CAMERA_FPS, max: TARGET_CAMERA_FPS },
      facingMode: { ideal: "user" },
    },
    audio: false,
  };
}

export function cameraStartErrorMessage(error: unknown): string {
  const name = error && typeof error === "object" && "name" in error
    ? String((error as { name?: unknown }).name ?? "")
    : "";
  if (name === "NotFoundError") return "No camera found — connect one and try again";
  if (name === "NotReadableError") return "Camera could not start — reconnect it or close an application that may be using it";
  if (name === "OverconstrainedError") return "Camera does not support the requested video mode";
  return "Camera unavailable or permission denied";
}
