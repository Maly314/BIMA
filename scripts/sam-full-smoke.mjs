import { readFile, writeFile } from "node:fs/promises";
import { basename } from "node:path";

const [inputPath, outputPath, metadataPath] = process.argv.slice(2);
const serviceUrl = process.env.BIMA_SAM31_SERVICE_URL ?? "http://127.0.0.1:4831";
if (!inputPath || !outputPath || !metadataPath) {
  throw new Error("usage: node scripts/sam-full-smoke.mjs INPUT OUTPUT_MP4 OUTPUT_METADATA_JSON");
}

const healthResponse = await fetch(`${serviceUrl}/health`);
const health = await healthResponse.json();
if (!healthResponse.ok || health.pipelineVersion !== "sam31-native-v12") {
  throw new Error(`unexpected SAM service: ${JSON.stringify(health)}`);
}

const source = await readFile(inputPath);
const startedResponse = await fetch(`${serviceUrl}/process-video-full`, {
  method: "POST",
  headers: { "content-type": "video/webm" },
  body: source,
});
const started = await startedResponse.json();
if (!startedResponse.ok || !started.jobId) throw new Error(JSON.stringify(started));
console.log(`job=${started.jobId} input=${basename(inputPath)} bytes=${source.length}`);

let status;
do {
  await new Promise((resolve) => setTimeout(resolve, 2_000));
  const statusResponse = await fetch(`${serviceUrl}/result/${started.jobId}/status`);
  status = await statusResponse.json();
  if (!statusResponse.ok || status.status === "failed") throw new Error(JSON.stringify(status));
  console.log(`status=${status.status} frames=${status.processedFrames ?? 0}/${status.frameCount ?? "?"} phase=${status.phase ?? "?"}`);
} while (status.status !== "complete");

const [videoResponse, metadataResponse] = await Promise.all([
  fetch(`${serviceUrl}/result/${started.jobId}/video`),
  fetch(`${serviceUrl}/result/${started.jobId}/metadata`),
]);
if (!videoResponse.ok || !metadataResponse.ok) {
  throw new Error(`result retrieval failed: video=${videoResponse.status} metadata=${metadataResponse.status}`);
}
const video = Buffer.from(await videoResponse.arrayBuffer());
const metadata = Buffer.from(await metadataResponse.arrayBuffer());
if (video.length < 12 || video.subarray(4, 8).toString("ascii") !== "ftyp") {
  throw new Error("result was not a valid MP4");
}
await Promise.all([writeFile(outputPath, video), writeFile(metadataPath, metadata)]);
const ackResponse = await fetch(`${serviceUrl}/result/${started.jobId}/ack`, { method: "POST" });
if (!ackResponse.ok) throw new Error(`result cleanup acknowledgement failed: ${ackResponse.status}`);
console.log(`complete videoBytes=${video.length} metadataBytes=${metadata.length}`);
