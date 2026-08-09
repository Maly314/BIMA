import assert from "node:assert/strict";
import test from "node:test";
import { processSam31Video, resumeSam31Video, SAM_PIPELINE_VERSION } from "../app/sam31-client.ts";

const json = (body, init = {}) => new Response(JSON.stringify(body), {
  status: 200,
  headers: { "content-type": "application/json" },
  ...init,
});

const mp4Bytes = new Uint8Array([
  0, 0, 0, 24, 0x66, 0x74, 0x79, 0x70, 0x69, 0x73, 0x6f, 0x6d,
  0, 0, 0, 0, 0x69, 0x73, 0x6f, 0x6d, 0x6d, 0x70, 0x34, 0x32,
  0, 0, 0, 12, 0x6d, 0x64, 0x61, 0x74, 1, 2, 3, 4,
  0, 0, 0, 8, 0x6d, 0x6f, 0x6f, 0x76,
]);
const mp4 = () => new Response(mp4Bytes, { status: 200, headers: { "content-type": "video/mp4" } });

test("SAM client completes the versioned upload, progress, metadata, and video lifecycle", async () => {
  const calls = [];
  const responses = [
    json({ model: "ready", pipelineVersion: SAM_PIPELINE_VERSION }),
    json({ jobId: "job-1", status: "running", pipelineVersion: SAM_PIPELINE_VERSION }, { status: 202 }),
    json({ status: "running", phase: "tracking", progress: 50, processedFrames: 4, frameCount: 8 }),
    json({ status: "complete", phase: "complete", progress: 100, processedFrames: 8, frameCount: 8 }),
    json({ frames: [{ frameIndex: 0, sourceVideoTimeMs: 0, segments: [], source: "sam31-native-propagation" }], processingMs: 42 }),
    mp4(),
  ];
  const progress = [];
  const phases = [];
  const jobs = [];
  const result = await processSam31Video(new Blob(["video"], { type: "video/webm" }), {
    signal: new AbortController().signal,
    serviceUrl: "http://sam.test",
    sleep: async () => {},
    fetcher: async (url, init) => {
      calls.push({ url: String(url), method: init?.method ?? "GET" });
      return responses.shift();
    },
    onProgress: (job) => progress.push(job.progress),
    onPhase: (phase) => phases.push(phase),
    onJobStarted: async (jobId) => jobs.push(jobId),
  });

  assert.deepEqual(calls.map((call) => call.url), [
    "http://sam.test/load",
    "http://sam.test/process-video-full",
    "http://sam.test/result/job-1/status",
    "http://sam.test/result/job-1/status",
    "http://sam.test/result/job-1/metadata",
    "http://sam.test/result/job-1/video",
  ]);
  assert.deepEqual(progress, [50, 100]);
  assert.deepEqual(phases, ["retrieving-metadata", "retrieving-video"]);
  assert.deepEqual(jobs, ["job-1"]);
  assert.equal(result.jobId, "job-1");
  assert.equal(result.frames.length, 1);
  assert.equal(result.processingMs, 42);
  assert.equal(result.annotatedBlob.type, "video/mp4");
  assert.equal(result.annotatedBlob.size, mp4Bytes.length);
});

test("SAM client resumes a durable job without loading or uploading again", async () => {
  const calls = [];
  const responses = [
    json({ status: "complete", phase: "complete", progress: 100, processedFrames: 8, frameCount: 8 }),
    json({ frames: [], processingMs: 75 }),
    mp4(),
  ];
  const result = await resumeSam31Video("durable-job", {
    signal: new AbortController().signal,
    serviceUrl: "http://sam.test",
    sleep: async () => {},
    fetcher: async (url) => {
      calls.push(String(url));
      return responses.shift();
    },
  });

  assert.deepEqual(calls, [
    "http://sam.test/result/durable-job/status",
    "http://sam.test/result/durable-job/metadata",
    "http://sam.test/result/durable-job/video",
  ]);
  assert.equal(result.jobId, "durable-job");
  assert.equal(result.annotatedBlob.size, mp4Bytes.length);
});

test("SAM client validates MP4 structure through bounded slices", async () => {
  const originalArrayBuffer = Blob.prototype.arrayBuffer;
  let largestRead = 0;
  Blob.prototype.arrayBuffer = function arrayBuffer() {
    largestRead = Math.max(largestRead, this.size);
    return originalArrayBuffer.call(this);
  };
  try {
    const responses = [
      json({ status: "complete", progress: 100 }),
      json({ frames: [], processingMs: 10 }),
      mp4(),
    ];
    await resumeSam31Video("bounded-validation", {
      signal: new AbortController().signal,
      sleep: async () => {},
      fetcher: async () => responses.shift(),
    });
    assert.ok(largestRead <= 16, `MP4 validation read ${largestRead} bytes at once`);
  } finally {
    Blob.prototype.arrayBuffer = originalArrayBuffer;
  }
});

test("SAM client rejects a mislabeled or truncated annotated video", async () => {
  const responses = [
    json({ status: "complete", progress: 100 }),
    json({ frames: [], processingMs: 10 }),
    new Response(new Uint8Array([1, 2, 3, 4]), { status: 200, headers: { "content-type": "video/mp4" } }),
  ];
  await assert.rejects(
    resumeSam31Video("bad-video", {
      signal: new AbortController().signal,
      sleep: async () => {},
      fetcher: async () => responses.shift(),
    }),
    /invalid annotated MP4.*raw recording was preserved/,
  );
});

test("SAM client rejects an ftyp-only video that cannot contain annotations", async () => {
  const responses = [
    json({ status: "complete", progress: 100 }),
    json({ frames: [], processingMs: 10 }),
    new Response(mp4Bytes.slice(0, 24), { status: 200, headers: { "content-type": "video/mp4" } }),
  ];
  await assert.rejects(
    resumeSam31Video("header-only-video", {
      signal: new AbortController().signal,
      sleep: async () => {},
      fetcher: async () => responses.shift(),
    }),
    /invalid annotated MP4.*raw recording was preserved/,
  );
});

test("SAM client rejects a stale service before uploading patient video", async () => {
  let calls = 0;
  await assert.rejects(
    processSam31Video(new Blob(["video"]), {
      signal: new AbortController().signal,
      sleep: async () => {},
      fetcher: async () => {
        calls += 1;
        return json({ model: "ready", pipelineVersion: "sam31-native-old" });
      },
    }),
    /BIMA version mismatch/,
  );
  assert.equal(calls, 1);
});

test("SAM client surfaces the isolated worker failure returned by job status", async () => {
  const responses = [
    json({ model: "ready", pipelineVersion: SAM_PIPELINE_VERSION }),
    json({ jobId: "job-2", pipelineVersion: SAM_PIPELINE_VERSION }, { status: 202 }),
    json({ status: "failed", error: "SAM chunk 2 failed" }),
  ];
  await assert.rejects(
    processSam31Video(new Blob(["video"]), {
      signal: new AbortController().signal,
      sleep: async () => {},
      fetcher: async () => responses.shift(),
    }),
    /SAM chunk 2 failed/,
  );
});

test("SAM client rejects an unexpected job state instead of polling forever", async () => {
  const responses = [
    json({ model: "ready", pipelineVersion: SAM_PIPELINE_VERSION }),
    json({ jobId: "job-3", pipelineVersion: SAM_PIPELINE_VERSION }, { status: 202 }),
    json({ status: "paused" }),
  ];
  await assert.rejects(
    processSam31Video(new Blob(["video"]), {
      signal: new AbortController().signal,
      sleep: async () => {},
      fetcher: async () => responses.shift(),
    }),
    /unexpected job status: paused/,
  );
});

test("SAM client preserves cancellation during result polling", async () => {
  const controller = new AbortController();
  const responses = [
    json({ model: "ready", pipelineVersion: SAM_PIPELINE_VERSION }),
    json({ jobId: "job-4", pipelineVersion: SAM_PIPELINE_VERSION }, { status: 202 }),
  ];
  await assert.rejects(
    processSam31Video(new Blob(["video"]), {
      signal: controller.signal,
      sleep: async () => controller.abort(),
      fetcher: async (_url, init) => {
        if (init?.signal?.aborted) throw new DOMException("The operation was aborted", "AbortError");
        return responses.shift();
      },
    }),
    (error) => error instanceof DOMException && error.name === "AbortError",
  );
});
