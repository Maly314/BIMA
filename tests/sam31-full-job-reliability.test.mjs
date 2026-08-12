import assert from "node:assert/strict";
import { execFileSync, spawn, spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, readdirSync } from "node:fs";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");
const python = existsSync(path.join(root, ".sam31-venv", "Scripts", "python.exe"))
  ? path.join(root, ".sam31-venv", "Scripts", "python.exe")
  : "python";
const fakeWorker = path.join(root, "tests", "fixtures", "fake-sam31-worker.py");

async function freePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address();
      server.close(() => resolve(port));
    });
  });
}

function stopTree(child) {
  if (!child?.pid) return;
  if (process.platform === "win32") spawnSync("taskkill", ["/pid", String(child.pid), "/T", "/F"], { windowsHide: true });
  else child.kill("SIGKILL");
}

async function startService(t, extraEnv = {}) {
  const port = await freePort();
  const tempRoot = mkdtempSync(path.join(os.tmpdir(), "bima-sam-reliability-"));
  const child = spawn(python, ["desktop/sam31_service.py"], {
    cwd: root,
    windowsHide: true,
    env: {
      ...process.env,
      BIMA_SAM31_PORT: String(port),
      BIMA_SAM31_TEMP_ROOT: tempRoot,
      BIMA_SAM31_CHUNK_WORKER: fakeWorker,
      BIMA_SAM31_VIDEO_ENCODER: "libx264",
      BIMA_SAM31_CHUNK_FRAMES: "4",
      ...extraEnv,
    },
    stdio: ["ignore", "pipe", "pipe"],
  });
  let logs = "";
  child.stdout.on("data", (chunk) => { logs += chunk; });
  child.stderr.on("data", (chunk) => { logs += chunk; });
  t.after(() => stopTree(child));
  const deadline = Date.now() + 15_000;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(`http://127.0.0.1:${port}/health`);
      if (response.ok) return { port, tempRoot, child, logs: () => logs };
    } catch { /* starting */ }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error(`service did not start: ${logs}`);
}

function makeVideo(directory, name = "input.webm") {
  const target = path.join(directory, name);
  execFileSync("ffmpeg", [
    "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=blue:s=320x180:r=30:d=0.266667",
    "-frames:v", "8", "-c:v", "libvpx-vp9", target,
  ], { windowsHide: true });
  return target;
}

async function startJob(port, input) {
  const response = await fetch(`http://127.0.0.1:${port}/process-video-full`, {
    method: "POST",
    headers: { "content-type": "video/webm" },
    body: await import("node:fs/promises").then(({ readFile }) => readFile(input)),
  });
  assert.equal(response.status, 202);
  return response.json();
}

async function waitForJob(port, jobId) {
  const deadline = Date.now() + 45_000;
  while (Date.now() < deadline) {
    const response = await fetch(`http://127.0.0.1:${port}/result/${jobId}/status`);
    const status = await response.json();
    if (status.status === "complete" || status.status === "failed") return status;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error(`job ${jobId} did not finish`);
}

function assertNoOrphans(tempRoot) {
  if (process.platform !== "win32") return;
  const escaped = tempRoot.replaceAll("'", "''");
  const output = execFileSync("powershell", ["-NoProfile", "-Command", `Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^(ffmpeg|python)' -and $_.CommandLine -like '*${escaped}*' } | Select-Object -ExpandProperty ProcessId`], { encoding: "utf8", windowsHide: true });
  assert.equal(output.trim(), "");
}

test("worker OOM cleans partial files and the next SAM job still completes", async (t) => {
  const markerRoot = mkdtempSync(path.join(os.tmpdir(), "bima-sam-fail-once-"));
  const marker = path.join(markerRoot, "failed.marker");
  const service = await startService(t, { BIMA_FAKE_WORKER_FAIL_ONCE: marker });
  const input = makeVideo(markerRoot);

  const failedJob = await startJob(service.port, input);
  const failed = await waitForJob(service.port, failedJob.jobId);
  assert.equal(failed.status, "failed");
  assert.match(failed.error, /chunk 1\/2 failed.*CUDA out of memory/is);
  assert.deepEqual(readdirSync(service.tempRoot), []);
  assertNoOrphans(service.tempRoot);

  const successfulJob = await startJob(service.port, input);
  const successful = await waitForJob(service.port, successfulJob.jobId);
  assert.equal(successful.status, "complete", service.logs());
  assert.equal(successful.processedFrames, 8);
  const metadataResponse = await fetch(`http://127.0.0.1:${service.port}/result/${successfulJob.jobId}/metadata`);
  const metadata = await metadataResponse.json();
  assert.equal(metadata.frameCount, 8);
  assert.equal(metadata.frames.length, 8);
  const videoResponse = await fetch(`http://127.0.0.1:${service.port}/result/${successfulJob.jobId}/video`);
  assert.equal(videoResponse.headers.get("content-type"), "video/mp4");
  assert.ok((await videoResponse.arrayBuffer()).byteLength > 1_000);

  const ack = await fetch(`http://127.0.0.1:${service.port}/result/${successfulJob.jobId}/ack`, { method: "POST" });
  assert.equal(ack.status, 200);
  assert.deepEqual(readdirSync(service.tempRoot), []);
  assert.equal((await fetch(`http://127.0.0.1:${service.port}/result/${successfulJob.jobId}/metadata`)).status, 404);
});

test("timed-out SAM workers are terminated, cleaned, and do not poison the service", async (t) => {
  const markerRoot = mkdtempSync(path.join(os.tmpdir(), "bima-sam-timeout-once-"));
  const service = await startService(t, {
    BIMA_FAKE_WORKER_SLEEP_ONCE: path.join(markerRoot, "slept.marker"),
    BIMA_FAKE_WORKER_SLEEP_SECONDS: "5",
    BIMA_SAM31_WORKER_TIMEOUT_SECONDS: "1",
  });
  const input = makeVideo(markerRoot);
  const timedOutJob = await startJob(service.port, input);
  const timedOut = await waitForJob(service.port, timedOutJob.jobId);
  assert.equal(timedOut.status, "failed");
  assert.match(timedOut.error, /timed out after 1 seconds and was terminated/i);
  assert.deepEqual(readdirSync(service.tempRoot), []);
  assertNoOrphans(service.tempRoot);

  const nextJob = await startJob(service.port, input);
  assert.equal((await waitForJob(service.port, nextJob.jobId)).status, "complete", service.logs());
});

test("completed SAM results are bounded and old patient-video temp data is evicted", async (t) => {
  const markerRoot = mkdtempSync(path.join(os.tmpdir(), "bima-sam-retention-"));
  const service = await startService(t, { BIMA_SAM31_MAX_RESULTS: "2" });
  const input = makeVideo(markerRoot);
  const jobIds = [];
  for (let index = 0; index < 3; index += 1) {
    const job = await startJob(service.port, input);
    jobIds.push(job.jobId);
    assert.equal((await waitForJob(service.port, job.jobId)).status, "complete", service.logs());
  }

  assert.equal((await fetch(`http://127.0.0.1:${service.port}/result/${jobIds[0]}/metadata`)).status, 404);
  assert.equal((await fetch(`http://127.0.0.1:${service.port}/result/${jobIds[1]}/metadata`)).status, 200);
  assert.equal((await fetch(`http://127.0.0.1:${service.port}/result/${jobIds[2]}/metadata`)).status, 200);
  assert.equal(readdirSync(service.tempRoot).length, 2);
});

test("persistent mode reuses one worker across multiple bounded video chunks", async (t) => {
  const markerRoot = mkdtempSync(path.join(os.tmpdir(), "bima-sam-persistent-"));
  const invocationLog = path.join(markerRoot, "worker-invocations.log");
  const service = await startService(t, {
    BIMA_SAM31_PERSISTENT_WORKER: "1",
    BIMA_SAM31_CHUNKS_PER_WORKER: "64",
    BIMA_FAKE_WORKER_INVOCATION_LOG: invocationLog,
  });
  const input = makeVideo(markerRoot);
  const job = await startJob(service.port, input);
  const completed = await waitForJob(service.port, job.jobId);
  assert.equal(completed.status, "complete", service.logs());
  assert.equal(completed.processedFrames, 8);
  assert.equal(readFileSync(invocationLog, "utf8").trim().split(/\r?\n/).length, 1);
});
