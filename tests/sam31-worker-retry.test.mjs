import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");
const python = path.join(root, ".sam31-venv", "Scripts", "python.exe");

function run(source) {
  return spawnSync(python, ["-c", source], {
    cwd: path.join(root, "desktop"),
    encoding: "utf8",
    windowsHide: true,
  });
}

test("retries a transient OpenCV video-open failure", () => {
  const result = run(`
from sam31_worker_retry import retry_transient_video_open
attempts = 0
def operation():
    global attempts
    attempts += 1
    if attempts < 3:
        raise ValueError("Could not open video: chunk.mp4")
    return "ready"
assert retry_transient_video_open(operation, attempts=3, delay_seconds=0) == "ready"
assert attempts == 3
`);
  assert.equal(result.status, 0, result.stderr);
});

test("does not retry CUDA, model, or corrupt-data failures", () => {
  const result = run(`
from sam31_worker_retry import retry_transient_video_open
attempts = 0
def operation():
    global attempts
    attempts += 1
    raise RuntimeError("CUDA out of memory")
try:
    retry_transient_video_open(operation, attempts=3, delay_seconds=0)
except RuntimeError as error:
    assert str(error) == "CUDA out of memory"
else:
    raise AssertionError("expected the non-transient failure")
assert attempts == 1
`);
  assert.equal(result.status, 0, result.stderr);
});
