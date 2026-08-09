export const SAM_PIPELINE_VERSION = "sam31-native-v12";
export const SAM_SERVICE_URL = "http://127.0.0.1:4831";

export type Sam31JobProgress = {
  status?: string;
  phase?: string;
  progress?: number;
  processedFrames?: number;
  frameCount?: number;
  error?: string;
};

export type Sam31NativeFrame<TSegment> = {
  frameIndex: number;
  sourceVideoTimeMs: number;
  segments: TSegment[];
  source: "sam31-native-propagation";
};

type ProcessOptions = {
  signal: AbortSignal;
  onJobStarted?: (jobId: string) => void | Promise<void>;
  onProgress?: (progress: Sam31JobProgress) => void;
  onPhase?: (phase: "retrieving-metadata" | "retrieving-video") => void;
  fetcher?: typeof fetch;
  sleep?: (milliseconds: number) => Promise<void>;
  serviceUrl?: string;
};

type ResumeOptions = Omit<ProcessOptions, "onJobStarted">;

async function jsonOrEmpty<T>(response: Response): Promise<T> {
  return response.json().catch(() => ({} as T));
}

async function validatedMp4(response: Response) {
  if (!response.ok) throw new Error("The annotated SAM video could not be retrieved");
  const blob = await response.blob();
  const header = new Uint8Array(await blob.slice(0, 12).arrayBuffer());
  const hasFtyp = header.length >= 8
    && header[4] === 0x66 && header[5] === 0x74 && header[6] === 0x79 && header[7] === 0x70;
  if (!response.headers.get("content-type")?.toLowerCase().startsWith("video/mp4") || !hasFtyp) {
    throw new Error("The SAM service returned an invalid annotated MP4; the raw recording was preserved");
  }
  return blob;
}

export async function processSam31Video<TSegment>(blob: Blob, options: ProcessOptions) {
  const fetcher = options.fetcher ?? fetch;
  const serviceUrl = options.serviceUrl ?? SAM_SERVICE_URL;

  const loadResponse = await fetcher(`${serviceUrl}/load`, { method: "POST", signal: options.signal });
  const loadResult = await jsonOrEmpty<{ error?: string; pipelineVersion?: string }>(loadResponse);
  if (!loadResponse.ok) throw new Error(loadResult.error || "SAM 3.1 could not load for post-processing");
  if (loadResult.pipelineVersion !== SAM_PIPELINE_VERSION) {
    throw new Error(`BIMA version mismatch: app expects ${SAM_PIPELINE_VERSION}, but the SAM service reports ${loadResult.pipelineVersion ?? "an older version"}. Restart BIMA and try again.`);
  }

  const response = await fetcher(`${serviceUrl}/process-video-full`, {
    method: "POST",
    headers: { "Content-Type": blob.type || "video/webm" },
    body: blob,
    signal: options.signal,
  });
  const started = await jsonOrEmpty<{ jobId?: string; error?: string; pipelineVersion?: string }>(response);
  if (!response.ok || !started.jobId) throw new Error(started.error || "SAM 3.1 native video propagation could not start");
  if (started.pipelineVersion !== SAM_PIPELINE_VERSION) throw new Error("BIMA changed versions while processing. Restart BIMA and record again.");

  await options.onJobStarted?.(started.jobId);
  return resumeSam31Video<TSegment>(started.jobId, options);
}

export async function resumeSam31Video<TSegment>(jobId: string, options: ResumeOptions) {
  const fetcher = options.fetcher ?? fetch;
  const sleep = options.sleep ?? ((milliseconds) => new Promise<void>((resolve) => setTimeout(resolve, milliseconds)));
  const serviceUrl = options.serviceUrl ?? SAM_SERVICE_URL;

  while (true) {
    await sleep(1000);
    const statusResponse = await fetcher(`${serviceUrl}/result/${jobId}/status`, { signal: options.signal });
    const job = await jsonOrEmpty<Sam31JobProgress>(statusResponse);
    if (!statusResponse.ok || job.status === "failed") throw new Error(job.error || "SAM 3.1 native propagation failed");
    options.onProgress?.(job);
    if (job.status === "complete") break;
    if (job.status !== "running") throw new Error(`SAM 3.1 returned an unexpected job status: ${job.status ?? "missing"}`);
  }

  options.onPhase?.("retrieving-metadata");
  const metadataResponse = await fetcher(`${serviceUrl}/result/${jobId}/metadata`, { signal: options.signal });
  const result = await jsonOrEmpty<{
    frames?: Sam31NativeFrame<TSegment>[];
    processingMs?: number;
    error?: string;
  }>(metadataResponse);
  if (!metadataResponse.ok) throw new Error(result.error || "SAM 3.1 native mask metadata could not be retrieved");

  options.onPhase?.("retrieving-video");
  const videoResponse = await fetcher(`${serviceUrl}/result/${jobId}/video`, { signal: options.signal });
  return {
    jobId,
    frames: result.frames ?? [],
    processingMs: result.processingMs ?? 0,
    annotatedBlob: await validatedMp4(videoResponse),
  };
}
