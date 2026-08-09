import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import test from "node:test";

const run = promisify(execFile);

test("isolated SAM worker reserves VRAM for the Electron renderer", async () => {
  const { stdout } = await run("python", ["desktop/sam31_chunk_worker.py", "--resource-policy"], {
    cwd: new URL("..", import.meta.url),
    env: { ...process.env, BIMA_SAM31_GPU_MEMORY_FRACTION: "0.86" },
    windowsHide: true,
  });
  assert.deepEqual(JSON.parse(stdout), {
    gpuMemoryFraction: 0.86,
    modelWeightDtype: "language-bfloat16",
    cudaModuleLoading: "LAZY",
  });
});
