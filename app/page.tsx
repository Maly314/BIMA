"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { addCaptureSession, getCaptureSession, newId, updateCaptureSession } from "./recordings";
import { CAPTURE_SCHEMA_VERSION, captureDurationMs, captureElapsedMs, stopCaptureRun, type CaptureRun } from "./capture-sync";
import { ageRangeLabel, clinicalAgeMetadata, detailedAgeLabel, type Session } from "./session-domain";
import BrandLockup from "./BrandLockup";
import Landing from "./Landing";
import SensorView from "./SensorView";
import VideoView from "./VideoView";
function CaptureWindow({ session, onExit }: { session: Session; onExit: () => void }) {
  const [view, setView] = useState<"sensor" | "video">("sensor");
  const [studyId, setStudyId] = useState(`Patient ${session.patientNumber}`);
  const [note, setNote] = useState("");
  const [sensorReady, setSensorReady] = useState(false);
  const [videoReady, setVideoReady] = useState(false);
  const [captureRun, setCaptureRun] = useState<CaptureRun | null>(null);
  const [saving, setSaving] = useState(false);
  const [saveErrors, setSaveErrors] = useState<string[]>([]);
  const [savedKinds, setSavedKinds] = useState<Set<"sensor" | "pose">>(new Set());
  const [lastSessionId, setLastSessionId] = useState("");
  const [elapsed, setElapsed] = useState("00:00");
  const [startingCamera, setStartingCamera] = useState(false);
  const captureIdRef = useRef("");
  const posePreviewRef = useRef<HTMLCanvasElement>(null);
  const cameraControlRef = useRef<(() => Promise<boolean>) | null>(null);
  const registerCameraControl = useCallback((enable: () => Promise<boolean>) => { cameraControlRef.current = enable; }, []);
  const recording = !!captureRun?.active;
  const ready = sensorReady && videoReady;

  const handleSaved = useCallback((kind: "sensor" | "pose", ok: boolean) => {
    if (!ok) setSaveErrors((current) => [...current, kind]);
    setSavedKinds((current) => {
      const next = new Set(current);
      next.add(kind);
      if (next.size === 2) {
        setSaving(false);
        setLastSessionId(captureIdRef.current);
      }
      return next;
    });
  }, []);

  const startSynchronizedCapture = async () => {
    if (!sensorReady || recording || saving || startingCamera) return;

    // Record together owns the camera: if it is not on yet, switch it on and
    // wait, so the operator never has to visit the Video tab first. A capture
    // is not started at all if the camera cannot be opened — a sensor-only run
    // that looks like a paired one would be worse than no run.
    if (!videoReady) {
      setStartingCamera(true);
      const opened = await cameraControlRef.current?.();
      setStartingCamera(false);
      if (!opened) return;
    }

    const run: CaptureRun = {
      id: newId(), active: true, startedAtEpochMs: Date.now(), startedAtPerfMs: performance.now(),
      studyId: studyId.trim() || `Patient ${session.patientNumber}`, note: note.trim(),
    };
    captureIdRef.current = run.id;
    setSavedKinds(new Set());
    setSaveErrors([]);
    setLastSessionId("");
    setElapsed("00:00");
    addCaptureSession({
      id: run.id, schemaVersion: CAPTURE_SCHEMA_VERSION, patientNumber: session.patientNumber,
      suspected: session.suspected, ageYears: session.ageYears, ageMonths: session.ageMonths,
      ageDays: session.ageDays, studyDate: session.studyDate, weightKg: session.weightKg,
      ...clinicalAgeMetadata(session),
      studyId: run.studyId, note: run.note, startedAtEpochMs: run.startedAtEpochMs,
      clock: { type: "performance-time-origin", unit: "milliseconds", zero: "capture-start" },
      status: "recording", assets: [],
    }).catch(() => setSaveErrors(["session metadata"]));
    setCaptureRun(run);
  };

  const stopSynchronizedCapture = () => {
    if (!captureRun?.active) return;
    const stopped = stopCaptureRun(captureRun);
    setSaving(true);
    setCaptureRun(stopped);
    updateCaptureSession(stopped.id, {
      stoppedAtEpochMs: stopped.stoppedAtEpochMs,
      durationMs: captureDurationMs(stopped),
      status: "processing",
    }).catch(() => setSaveErrors((current) => [...current, "session metadata"]));
  };

  useEffect(() => {
    if (!recording || !captureRun) return;
    const timer = window.setInterval(() => {
      const total = Math.floor(captureElapsedMs(captureRun, performance.now()) / 1000);
      setElapsed(`${String(Math.floor(total / 60)).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}`);
    }, 250);
    return () => window.clearInterval(timer);
  }, [captureRun, recording]);

  const downloadSessionManifest = async () => {
    if (!lastSessionId) return;
    const stored = await getCaptureSession(lastSessionId);
    if (!stored) return;
    const blob = new Blob([JSON.stringify(stored, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `capture-${stored.id}-manifest.json`;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
  };

  const syncStatus = recording
    ? `Recording sensor + video · ${elapsed}`
    : startingCamera
      ? "Starting the camera for a paired capture…"
    : saving
      ? `Saving paired capture · ${savedKinds.size}/2 streams stored`
      : saveErrors.length
        ? `Capture finished with a storage error: ${saveErrors.join(", ")}`
        : lastSessionId
          ? "Paired capture saved locally"
          : ready
            ? "Sensors and video ready on one shared clock"
            : sensorReady
              ? "Sensors ready · Record together will start the camera"
              : videoReady
                ? "Camera ready · connect and calibrate sensors"
                : "Connect and calibrate the sensors to begin";

  return (
    <main className="page-shell"><section className="app-window" aria-labelledby="page-title">
      <div className="window-bar" aria-hidden="true"><span className="window-caption">BIMA</span></div>
      <header className="session-header">
        <div className="title-line">
          <div className="title-group">
            <button type="button" className="capture-back" onClick={onExit} disabled={recording || saving}>← Patients</button>
            <BrandLockup compact />
            <span className="title-divider" aria-hidden="true" />
            <h1 id="page-title">Movement capture</h1>
            <span className={`tag ${session.suspected ? "tag-susp" : "tag-non"}`}>{session.suspected ? "Suspected" : "Not suspected"}</span>
          </div>
          <nav className="view-switch" aria-label="Capture view"><button className={view==="sensor"?"active":""} onClick={()=>setView("sensor")}>Sensor</button><button className={view==="video"?"active":""} onClick={()=>setView("video")}>Video</button></nav>
        </div>
        <div className="field-row">
          <label><span>Study ID</span><input type="text" value={studyId} onChange={(event) => setStudyId(event.target.value)} autoComplete="off" disabled={recording} /></label>
          <label><span>{session.useCorrectedAge ? "Corrected age" : "Age"}</span><input type="text" value={`${detailedAgeLabel(session)} · ${ageRangeLabel(session)}`} readOnly /></label>
          <label><span>Note</span><input type="text" value={note} onChange={(event) => setNote(event.target.value)} placeholder="Add a note" autoComplete="off" disabled={recording} /></label>
        </div>
        <div className="sync-capture-bar">
          <div className="sync-state"><span className={`status-dot ${ready ? "connected" : sensorReady || videoReady ? "pending" : ""}`} /><div><strong>Synchronized capture</strong><span>{syncStatus}</span></div></div>
          <div className="sync-assets"><span>Sensor CSV</span><i /> <span>Pose video</span><i /> <span>Landmarks</span></div>
          {lastSessionId && !recording && !saving && <button type="button" className="manifest-button" onClick={downloadSessionManifest}>Session manifest</button>}
          <button type="button" className={`sync-record-button ${recording ? "recording" : ""}`} onClick={recording ? stopSynchronizedCapture : () => { void startSynchronizedCapture(); }} disabled={saving || startingCamera || (!recording && !sensorReady)}>{recording ? "Stop capture" : saving ? "Saving…" : startingCamera ? "Starting camera…" : "Record together"}</button>
        </div>
      </header>
          <div className={`capture-pane ${view === "sensor" ? "active" : ""}`} aria-hidden={view !== "sensor"}><SensorView session={session} captureRun={captureRun} onReadyChange={setSensorReady} onSaved={handleSaved} posePreviewRef={posePreviewRef} posePreviewActive={videoReady} active={view === "sensor"} /></div>
      <div className={`capture-pane ${view === "video" ? "active" : ""}`} aria-hidden={view !== "video"}><VideoView session={session} captureRun={captureRun} onReadyChange={setVideoReady} onSaved={handleSaved} posePreviewRef={posePreviewRef} registerCameraControl={registerCameraControl} /></div>
    </section></main>
  );
}

export default function Home() {
  const [session, setSession] = useState<Session | null>(null);
  if (!session) return <Landing onStart={setSession} />;
  return <CaptureWindow session={session} onExit={() => setSession(null)} />;
}
