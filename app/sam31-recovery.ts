import { addCaptureAsset, addRecording, type Recording } from "./recordings.ts";
import { resumeSam31Video, type Sam31NativeFrame } from "./sam31-client.ts";

type Segment = Record<string, unknown>;
type ResumeResult = {
  jobId: string;
  frames: Sam31NativeFrame<Segment>[];
  processingMs: number;
  annotatedBlob: Blob;
};

type RecoveryDependencies = {
  resume?: (jobId: string, signal: AbortSignal) => Promise<ResumeResult>;
  saveRecording?: typeof addRecording;
  saveAsset?: typeof addCaptureAsset;
};

const activeRecoveries = new Map<string, Promise<Recording>>();

function trackedFilename(rawFilename: string) {
  return rawFilename.replace(/-sam31-raw\.webm$/i, "-sam31-tracked.mp4");
}

function sidecarFilename(rawFilename: string) {
  return rawFilename.replace(/-sam31-raw\.webm$/i, "-segments.json");
}

export async function recoverSam31Recording(
  recording: Recording,
  signal: AbortSignal,
  dependencies: RecoveryDependencies = {},
): Promise<Recording> {
  if (recording.annotationStatus !== "processing" || !recording.samJobId) return recording;
  const existing = activeRecoveries.get(recording.samJobId);
  if (existing) return existing;

  const recovery = (async () => {
    const saveRecording = dependencies.saveRecording ?? addRecording;
    const saveAsset = dependencies.saveAsset ?? addCaptureAsset;
    const resume = dependencies.resume ?? ((jobId, resumeSignal) => resumeSam31Video<Segment>(jobId, { signal: resumeSignal }));
    try {
      const result = await resume(recording.samJobId!, signal);
      const rawFilename = recording.rawFilename ?? recording.filename;
      const trackingFilename = sidecarFilename(rawFilename);
      const sidecar = new Blob([JSON.stringify({
        schemaVersion: 3,
        recoveredAfterRendererRestart: true,
        capture: {
          sessionId: recording.captureSessionId,
          patientNumber: recording.patientNumber,
          studyId: recording.studyId,
          note: recording.note,
          studyDate: recording.studyDate,
          startedAtEpochMs: recording.sync?.startedAtEpochMs,
        },
        tracking: {
          inferenceBackend: "Meta SAM 3.1 native propagate_in_video",
          processingMs: result.processingMs,
          frameCount: result.frames.length,
        },
        frames: result.frames,
      })], { type: "application/json" });
      const completed: Recording = {
        ...recording,
        blob: result.annotatedBlob,
        filename: trackedFilename(rawFilename),
        rawBlob: recording.blob,
        rawFilename,
        sidecarBlob: sidecar,
        sidecarFilename: trackingFilename,
        size: result.annotatedBlob.size + recording.blob.size + sidecar.size,
        annotationStatus: "complete",
        processingError: undefined,
        sync: recording.sync ? { ...recording.sync, sampleCount: result.frames.length } : undefined,
      };
      await saveRecording(completed);
      if (recording.captureSessionId) {
        await saveAsset(recording.captureSessionId, {
          recordingId: recording.id,
          kind: "pose",
          filename: completed.filename,
          sidecarFilename: completed.sidecarFilename,
          sampleCount: result.frames.length,
          streamStartOffsetMs: recording.sync?.streamStartOffsetMs ?? 0,
          size: completed.size,
          metadata: { recoveredAfterRendererRestart: true, samJobId: recording.samJobId },
        });
      }
      return completed;
    } catch (error) {
      if (signal.aborted || (error instanceof DOMException && error.name === "AbortError")) throw error;
      const failed: Recording = {
        ...recording,
        annotationStatus: "failed",
        processingError: error instanceof Error ? error.message : "SAM recovery failed",
      };
      await saveRecording(failed);
      return failed;
    }
  })().finally(() => activeRecoveries.delete(recording.samJobId!));
  activeRecoveries.set(recording.samJobId, recovery);
  return recovery;
}
