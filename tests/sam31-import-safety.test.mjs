import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import os from "node:os";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");
const python = path.join(root, ".sam31-venv", "Scripts", "python.exe");

test("importing SAM helpers cannot delete an isolated worker's input", () => {
  const serviceTemp = mkdtempSync(path.join(os.tmpdir(), "bima-sam-import-safety-"));
  const sentinel = path.join(serviceTemp, "worker-input.mp4");
  writeFileSync(sentinel, "not a real video");
  const result = spawnSync(python, ["-c", `import pathlib, sam31_service; assert pathlib.Path(r'''${sentinel}''').is_file()`], {
    cwd: path.join(root, "desktop"),
    encoding: "utf8",
    windowsHide: true,
    env: { ...process.env, BIMA_SAM31_TEMP_ROOT: serviceTemp },
  });
  assert.equal(result.status, 0, result.stderr);
});

test("RLE decoding owns its NumPy dependency and renders a binary mask", () => {
  const result = spawnSync(python, ["-c", "import sam31_service; mask=sam31_service._decode_binary_rle([0,1,3],2,2); assert mask.shape == (2,2); assert mask.tolist() == [[1,0],[0,0]]"], {
    cwd: path.join(root, "desktop"),
    encoding: "utf8",
    windowsHide: true,
  });
  assert.equal(result.status, 0, result.stderr);
});

test("failed cleanup is reported truthfully instead of claiming patient data was deleted", () => {
  const scratch = mkdtempSync(path.join(os.tmpdir(), "bima-sam-cleanup-status-"));
  const result = spawnSync(python, ["-c", `
import pathlib, sam31_service
job = {}
def fail_cleanup(_directory):
    raise RuntimeError("file remains locked")
sam31_service._remove_tree_with_retries = fail_cleanup
sam31_service._publish_failed_job(job, pathlib.Path(r'''${scratch}'''), "worker failed")
assert job["status"] == "failed"
assert job["cleanupComplete"] is False
assert "file remains locked" in job["error"]
`], {
    cwd: path.join(root, "desktop"),
    encoding: "utf8",
    windowsHide: true,
  });
  rmSync(scratch, { recursive: true, force: true });
  assert.equal(result.status, 0, result.stderr);
});
