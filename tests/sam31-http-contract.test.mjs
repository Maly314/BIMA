import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import net from "node:net";
import test from "node:test";

async function freePort() {
  return await new Promise((resolve, reject) => {
    const server = net.createServer();
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      server.close(() => resolve(address.port));
    });
  });
}

async function startService(t) {
  const port = await freePort();
  const child = spawn("python", ["desktop/sam31_service.py"], {
    cwd: new URL("..", import.meta.url),
    env: { ...process.env, BIMA_SAM31_PORT: String(port) },
    stdio: ["ignore", "pipe", "pipe"],
    windowsHide: true,
  });
  let stderr = "";
  child.stderr.on("data", (chunk) => { stderr += chunk; });
  t.after(() => child.kill());

  const deadline = Date.now() + 10_000;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(`http://127.0.0.1:${port}/health`);
      if (response.ok) return { port, response };
    } catch { /* service is still starting */ }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error(`SAM service did not start: ${stderr}`);
}

test("SAM service exposes its runtime and matching pipeline version", async (t) => {
  const { response } = await startService(t);
  const body = await response.json();
  assert.equal(body.service, "ready");
  assert.equal(body.pipelineVersion, "sam31-native-v12");
  assert.equal(body.runtime, "official facebookresearch/sam3");
  assert.match(body.model, /^(ready|not-loaded)$/);
  assert.deepEqual(body.resourcePolicy, {
    chunkFrames: 4,
    gpuMemoryFraction: 0.75,
    modelWeightDtype: "backbones-bfloat16",
    cudaModuleLoading: "LAZY",
  });
});

test("full-video endpoint rejects an empty recording without creating a job", async (t) => {
  const { port } = await startService(t);
  const response = await fetch(`http://127.0.0.1:${port}/process-video-full`, {
    method: "POST",
    headers: { "content-type": "video/webm" },
    body: new Uint8Array(),
  });
  assert.equal(response.status, 400);
  assert.deepEqual(await response.json(), { error: "invalid video payload" });
});

test("unknown SAM routes return the stable JSON 404 contract", async (t) => {
  const { port } = await startService(t);
  const response = await fetch(`http://127.0.0.1:${port}/not-a-route`);
  assert.equal(response.status, 404);
  assert.deepEqual(await response.json(), { error: "not found" });
});
