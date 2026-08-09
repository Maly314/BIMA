import assert from "node:assert/strict";
import { execFileSync, spawn, spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import net from "node:net";
import os from "node:os";
import path from "node:path";

const [inputPath, outputPath, metadataPath, reportPath] = process.argv.slice(2);
if (!inputPath || !outputPath || !metadataPath || !reportPath) {
  throw new Error("usage: node scripts/sam-electron-full-smoke.mjs INPUT OUTPUT_MP4 OUTPUT_METADATA_JSON REPORT_JSON");
}

const root = path.resolve(import.meta.dirname, "..");
const electron = path.join(root, "node_modules", "electron", "dist", "electron.exe");
const scratch = mkdtempSync(path.join(os.tmpdir(), "bima-electron-full-sam-"));
const debugLog = path.join(scratch, "desktop.log");
const stopFile = path.join(scratch, "stop");
const serviceTemp = path.join(scratch, "sam-temp");

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

async function waitFor(url, timeoutMs = 90_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url);
      if (response.ok) return response;
    } catch { /* starting */ }
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
  throw new Error(`timed out waiting for ${url}`);
}

async function waitExit(child, timeoutMs = 60_000) {
  if (child.exitCode !== null) return child.exitCode;
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error(`process ${child.pid} did not exit`)), timeoutMs);
    child.once("exit", (code) => { clearTimeout(timeout); resolve(code); });
  });
}

function stopTree(child) {
  if (!child?.pid) return;
  if (process.platform === "win32") spawnSync("taskkill", ["/pid", String(child.pid), "/T", "/F"], { windowsHide: true });
  else child.kill("SIGKILL");
}

const appPort = await freePort();
const samPort = await freePort();
const desktop = spawn(electron, [path.join(root, "desktop")], {
  cwd: root,
  windowsHide: true,
  env: {
    ...process.env,
    BIMA_PORT: String(appPort),
    BIMA_SAM31_PORT: String(samPort),
    BIMA_USER_DATA_DIR: path.join(scratch, "profile"),
    BIMA_DEBUG_LOG: debugLog,
    BIMA_HEADLESS_PROBE: "1",
    BIMA_STABILITY_PROBE_SECONDS: "3600",
    BIMA_STABILITY_PROBE_STOP_FILE: stopFile,
    BIMA_SAM31_TEMP_ROOT: serviceTemp,
  },
  stdio: "ignore",
});

let smoke;
const gpuSamples = [];
const startedAt = Date.now();
try {
  assert.equal((await waitFor(`http://127.0.0.1:${appPort}/`)).status, 200);
  const health = await (await waitFor(`http://127.0.0.1:${samPort}/health`)).json();
  assert.equal(health.pipelineVersion, "sam31-native-v12");
  assert.deepEqual(health.resourcePolicy, {
    chunkFrames: 4,
    gpuMemoryFraction: 0.75,
    modelWeightDtype: "backbones-bfloat16",
    cudaModuleLoading: "LAZY",
  });

  let smokeOutput = "";
  smoke = spawn(process.execPath, [path.join(root, "scripts", "sam-full-smoke.mjs"), inputPath, outputPath, metadataPath], {
    cwd: root,
    windowsHide: true,
    env: { ...process.env, BIMA_SAM31_SERVICE_URL: `http://127.0.0.1:${samPort}` },
    stdio: ["ignore", "pipe", "pipe"],
  });
  smoke.stdout.on("data", (chunk) => { smokeOutput += chunk; process.stdout.write(chunk); });
  smoke.stderr.on("data", (chunk) => { smokeOutput += chunk; process.stderr.write(chunk); });
  const gpuTimer = setInterval(() => {
    try {
      const [used, free] = execFileSync("nvidia-smi", ["--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"], { encoding: "utf8", windowsHide: true }).trim().split(",").map((value) => Number(value.trim()));
      if (Number.isFinite(used) && Number.isFinite(free)) gpuSamples.push({ atMs: Date.now() - startedAt, usedMiB: used, freeMiB: free });
    } catch { /* GPU telemetry is reported as unavailable below */ }
  }, 500);
  const smokeCode = await waitExit(smoke, 3_600_000);
  clearInterval(gpuTimer);
  assert.equal(smokeCode, 0, smokeOutput);
  assert.equal(existsSync(outputPath), true);
  assert.equal(existsSync(metadataPath), true);

  writeFileSync(stopFile, "complete");
  assert.equal(await waitExit(desktop), 0);
  const log = readFileSync(debugLog, "utf8");
  assert.match(log, /renderer-webgl .*NVIDIA GeForce RTX 4060 Ti/);
  assert.doesNotMatch(log, /renderer-loss|renderer-recovery-failed/);
  const stability = log.match(/stability-probe-complete ticks=(\d+) errors=(\d+)/);
  assert.ok(stability, "renderer stability result was not logged");
  assert.ok(Number(stability[1]) > 0);
  assert.equal(Number(stability[2]), 0);

  const probe = JSON.parse(execFileSync("ffprobe", ["-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=codec_name,profile,pix_fmt,width,height,r_frame_rate,duration,nb_read_frames", "-of", "json", outputPath], { encoding: "utf8", windowsHide: true }));
  const stream = probe.streams[0];
  const metadata = JSON.parse(readFileSync(metadataPath, "utf8"));
  assert.equal(stream.codec_name, "h264");
  assert.equal(stream.pix_fmt, "yuv420p");
  assert.equal(Number(stream.nb_read_frames), metadata.frames.length);
  assert.equal(metadata.frames.every((frame, index) => frame.frameIndex === index), true);
  assert.equal(metadata.frames.every((frame, index) => index === 0 || frame.sourceVideoTimeMs >= metadata.frames[index - 1].sourceVideoTimeMs), true);
  const framesWithSegments = metadata.frames.filter((frame) => frame.segments.length > 0).length;
  assert.ok(framesWithSegments >= metadata.frames.length / 2, `only ${framesWithSegments}/${metadata.frames.length} frames contained hand segments`);
  assert.equal(existsSync(serviceTemp) ? (await import("node:fs")).readdirSync(serviceTemp).length : 0, 0, "acknowledged SAM results left temporary patient data");

  const report = {
    passed: true,
    pipelineVersion: health.pipelineVersion,
    inputPath,
    outputPath,
    metadataPath,
    elapsedMs: Date.now() - startedAt,
    rendererTicks: Number(stability[1]),
    rendererErrors: Number(stability[2]),
    frameCount: metadata.frames.length,
    framesWithSegments,
    video: stream,
    gpu: gpuSamples.length ? {
      samples: gpuSamples.length,
      peakUsedMiB: Math.max(...gpuSamples.map((sample) => sample.usedMiB)),
      minimumFreeMiB: Math.min(...gpuSamples.map((sample) => sample.freeMiB)),
    } : { samples: 0 },
  };
  writeFileSync(reportPath, JSON.stringify(report, null, 2));
  console.log(`desktop SAM full smoke passed: ${metadata.frames.length} frames, ${framesWithSegments} segmented, renderer errors=0`);
} catch (error) {
  if (smoke) stopTree(smoke);
  stopTree(desktop);
  throw error;
}
