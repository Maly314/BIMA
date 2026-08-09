"use client";

import { useCallback, useEffect, useState } from "react";
import { calculateCorrectedAge, formatAgeDays, formatPma } from "./corrected-age";
import { deleteRecording, getCalibration, getLastPatient, listRecordings, setCalibration, type Calibration, type Recording } from "./recordings";
import { ageWeeks, createPatientSession, detailedAgeLabel, type Session } from "./session-domain";
import { CALIBRATION_CAPTURE_MS, CALIBRATION_SETTLE_MS, INVALID_RE, buildCalibration, calibrationIsUsable, parseImuLine, type ImuSample } from "./sensor-calibration";
import BrandLockup from "./BrandLockup";
import { markInterruptedSam31Recordings, recoverSam31Recording } from "./sam31-recovery";

function formatDate(ms: number) {
  return new Date(ms).toLocaleString(undefined, { month: "short", day: "numeric", year: "numeric", hour: "numeric", minute: "2-digit" });
}

function RecordingCard({ recording, onDelete }: { recording: Recording; onDelete: (id: string) => void }) {
  const kindLabel = recording.kind === "pose" ? "Pose" : recording.kind === "video" ? "Video" : "Sensor";
  const download = (blob: Blob, filename: string) => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = filename; document.body.appendChild(a); a.click();
    a.remove(); URL.revokeObjectURL(url);
  };
  return (
    <div className="rec-card">
      <div className="rec-thumb">
        {recording.thumbnail ? <img src={recording.thumbnail} alt="" /> : <div className="rec-file-icon" />}
      </div>
      <div className="rec-meta">
        <div className="rec-title">Patient {recording.patientNumber}</div>
        <div className="rec-tags"><span className={`tag ${recording.suspected ? "tag-susp" : "tag-non"}`}>{recording.suspected ? "Suspected" : "Not suspected"}</span></div>
        <div className="rec-sub">{kindLabel} · {detailedAgeLabel(recording)}{recording.weightKg != null ? ` · ${recording.weightKg} kg` : ""}</div>
        <div className="rec-date">{formatDate(recording.date)}</div>
      </div>
      <div className="rec-actions">
        <button type="button" className="danger" onClick={() => onDelete(recording.id)}>Delete</button>
        <button type="button" onClick={() => download(recording.blob, recording.filename)}>{recording.annotationStatus === "failed" || recording.filename.endsWith("-sam31-raw.webm") ? "Raw video" : recording.rawBlob ? "Annotated video" : recording.kind === "pose" ? "Pose video" : recording.kind === "video" ? "Video" : "Download"}</button>
        {recording.rawBlob && recording.rawFilename && <button type="button" onClick={() => download(recording.rawBlob!, recording.rawFilename!)}>Raw video</button>}
        {recording.sidecarBlob && recording.sidecarFilename && <button type="button" onClick={() => download(recording.sidecarBlob!, recording.sidecarFilename!)}>Tracking data</button>}
      </div>
    </div>
  );
}

function AddPatientDialog({ nextPatient, onClose, onGo }: { nextPatient: number; onClose: () => void; onGo: (session: Session) => void }) {
  const [patientNumber, setPatientNumber] = useState(String(nextPatient));
  const [suspected, setSuspected] = useState(true);
  const localToday = () => {
    const now = new Date();
    now.setMinutes(now.getMinutes() - now.getTimezoneOffset());
    return now.toISOString().slice(0, 10);
  };
  const [dateOfBirth, setDateOfBirth] = useState(localToday);
  const [studyDate, setStudyDate] = useState(localToday);
  const [gestationalWeeks, setGestationalWeeks] = useState("40");
  const [gestationalDays, setGestationalDays] = useState("0");
  const [weightKg, setWeightKg] = useState("");
  const [weightUnit, setWeightUnit] = useState<"kg" | "lb">("kg");
  let correctedAge: ReturnType<typeof calculateCorrectedAge> | null = null;
  let ageError = "";
  try {
    correctedAge = calculateCorrectedAge(dateOfBirth, studyDate, Number(gestationalWeeks), Number(gestationalDays));
  } catch (error) {
    ageError = error instanceof Error ? error.message : "Check the age information";
  }
  const go = () => {
    if (!correctedAge) return;
    onGo(createPatientSession({
      patientNumber,
      nextPatient,
      suspected,
      dateOfBirth,
      studyDate,
      gestationalWeeks,
      gestationalDays,
      weight: weightKg,
      weightUnit,
    }));
  };
  return (
    <div className="dialog-overlay" role="dialog" aria-modal="true" aria-label="Add patient" onClick={onClose}>
      <div className="dialog" onClick={(e) => e.stopPropagation()}>
        <h2>Add patient</h2>
        <label><span>Patient number</span>
          <input type="number" min={1} value={patientNumber} onChange={(e) => setPatientNumber(e.target.value)} autoFocus />
        </label>
        <div className="dialog-field"><span>Category</span>
          <div className="toggle-group" role="group" aria-label="Category">
            <button type="button" className={suspected ? "active" : ""} onClick={() => setSuspected(true)}>Suspected</button>
            <button type="button" className={!suspected ? "active" : ""} onClick={() => setSuspected(false)}>Not suspected</button>
          </div>
        </div>
        <div className="patient-details-grid">
          <label><span>Date of birth</span><input type="date" value={dateOfBirth} max={studyDate} onChange={(e) => setDateOfBirth(e.target.value)} /></label>
          <label><span>Study date</span><input type="date" value={studyDate} min={dateOfBirth} onChange={(e) => setStudyDate(e.target.value)} /></label>
        </div>
        <div className="dialog-field"><span>Gestational age at birth</span>
          <div className="gestational-age-grid">
            <label className="age-cell"><span>Weeks</span><input type="number" min={20} max={45} inputMode="numeric" value={gestationalWeeks} onChange={(e) => setGestationalWeeks(e.target.value)} /></label>
            <label className="age-cell"><span>Days</span><input type="number" min={0} max={6} inputMode="numeric" value={gestationalDays} onChange={(e) => setGestationalDays(e.target.value)} /></label>
          </div>
        </div>
        <div className={`corrected-age-preview ${ageError ? "error" : ""}`} role="status">
          {correctedAge ? <><strong>{correctedAge.useCorrectedAge ? "Corrected age" : "Chronological age"}: {formatAgeDays(correctedAge.useCorrectedAge ? correctedAge.correctedAgeDays : correctedAge.chronologicalAgeDays)}</strong><span>{correctedAge.preterm ? `${formatAgeDays(correctedAge.chronologicalAgeDays)} chronological · ${formatPma(correctedAge.postmenstrualAgeDays)} · due ${correctedAge.expectedDueDate}` : "Term infant · no prematurity correction"}</span></> : <span>{ageError}</span>}
        </div>
        <label><span>Weight</span>
          <div className="weight-input-row">
            <input type="number" min={0} step="0.01" inputMode="decimal" placeholder="0.00" value={weightKg} onChange={(e) => setWeightKg(e.target.value)} aria-label={`Weight in ${weightUnit === "kg" ? "kilograms" : "pounds"}`} />
            <div className="weight-unit-toggle" role="group" aria-label="Weight unit">
              <button type="button" className={weightUnit === "kg" ? "active" : ""} aria-pressed={weightUnit === "kg"} onClick={() => setWeightUnit("kg")}>kg</button>
              <button type="button" className={weightUnit === "lb" ? "active" : ""} aria-pressed={weightUnit === "lb"} onClick={() => setWeightUnit("lb")}>lb</button>
            </div>
          </div>
        </label>
        <div className="dialog-actions">
          <button type="button" className="btn-secondary" onClick={onClose}>Cancel</button>
          <button type="button" className="btn-go" onClick={go} disabled={!correctedAge}>Go</button>
        </div>
      </div>
    </div>
  );
}

export default function Landing({ onStart }: { onStart: (session: Session) => void }) {
  const [recordings, setRecordings] = useState<Recording[]>([]);
  const [nextPatient, setNextPatient] = useState(1);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [filter, setFilter] = useState<"all" | "suspected" | "not">("all");
  const [confirmId, setConfirmId] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    const items = await listRecordings();
    setRecordings(items);
    const maxRec = items.reduce((m, r) => Math.max(m, r.patientNumber), 0);
    setNextPatient(Math.max(getLastPatient(), maxRec) + 1);
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void (async () => {
      await refresh();
      const stored = await markInterruptedSam31Recordings(await listRecordings());
      const pending = stored.filter((recording) => recording.annotationStatus === "processing" && recording.samJobId);
      if (!pending.length) return;
      await Promise.allSettled(pending.map((recording) => recoverSam31Recording(recording, controller.signal)));
      if (!controller.signal.aborted) await refresh();
    })();
    return () => controller.abort();
  }, [refresh]);

  const confirmTarget = recordings.find((r) => r.id === confirmId) || null;
  const doDelete = async () => { if (!confirmId) return; await deleteRecording(confirmId); setConfirmId(null); refresh(); };

  // One calibration per physical IMU, reused until the hardware is moved or
  // reconnected. Settling and capture are separate phases; captured samples are
  // then checked for motion and reduced with robust medians rather than averages.
  const [calRunning, setCalRunning] = useState(false);
  const [calMsg, setCalMsg] = useState("");
  const [calPhase, setCalPhase] = useState<"idle" | "countdown" | "capturing">("idle");
  const [calCountdown, setCalCountdown] = useState(5);
  const [calProgress, setCalProgress] = useState(0);
  const [calInfo, setCalInfo] = useState<{ count: number; date: number } | null>(null);
  useEffect(() => {
    const c = getCalibration();
    const keys = Object.keys(c).filter((key) => calibrationIsUsable(c[key]));
    if (keys.length) setCalInfo({ count: keys.length, date: Math.max(...keys.map((k) => c[k].date || 0)) });
  }, []);

  const runCalibration = async () => {
    const serial = (navigator as any).serial;
    if (!serial) { setCalMsg("Web Serial isn't available — use Chrome or Edge on desktop."); return; }
    let port: any, reader: any;
    try { port = await serial.requestPort(); await port.open({ baudRate: 921600 }); }
    catch { setCalMsg("Connection cancelled."); return; }
    setCalRunning(true);
    setCalPhase("countdown");
    setCalCountdown(Math.ceil(CALIBRATION_SETTLE_MS / 1000));
    setCalProgress(0);
    setCalMsg("Hold all four sensors down and completely still.");
    const samples: Record<string, ImuSample[]> = {};
    const invalidSeen = new Set<number>();
    let timedOut = false;
    let readFailure = "";
    try {
      reader = port.readable.getReader();
      const dec = new TextDecoder();
      let buf = "";
      const started = performance.now();
      const captureStarts = started + CALIBRATION_SETTLE_MS;
      const ends = captureStarts + CALIBRATION_CAPTURE_MS;
      while (performance.now() < ends) {
        const now = performance.now();
        if (now < captureStarts) {
          const seconds = Math.max(1, Math.ceil((captureStarts - now) / 1000));
          setCalPhase("countdown");
          setCalCountdown(seconds);
        } else {
          const percent = Math.min(100, Math.round(((now - captureStarts) / CALIBRATION_CAPTURE_MS) * 100));
          setCalPhase("capturing");
          setCalProgress(percent);
        }

        let timeoutId = 0;
        const outcome = await Promise.race([
          reader.read().then((result: any) => ({ kind: "read" as const, result })),
          new Promise<{ kind: "timeout" }>((resolve) => {
            timeoutId = window.setTimeout(() => resolve({ kind: "timeout" }), 900);
          }),
        ]);
        window.clearTimeout(timeoutId);
        if (outcome.kind === "timeout") { timedOut = true; break; }
        const { value, done } = outcome.result;
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let idx;
        while ((idx = buf.indexOf("\n")) >= 0) {
          const line = buf.slice(0, idx).trim(); buf = buf.slice(idx + 1);
          const parsed = parseImuLine(line);
          if (parsed && performance.now() >= captureStarts) {
            (samples[String(parsed.imu)] ||= []).push(parsed.sample);
          } else {
            const invalid = line.match(INVALID_RE);
            if (invalid) invalidSeen.add(Number.parseInt(invalid[1], 10) - 1);
          }
        }
      }
    } catch (error: any) {
      readFailure = error?.message || "Serial read failed";
    } finally {
      if (timedOut) { try { await reader?.cancel(); } catch { /* noop */ } }
      try { reader?.releaseLock(); } catch { /* noop */ }
      try { await port.close(); } catch { /* noop */ }
    }

    const result: Record<string, Calibration> = {};
    const rejected: string[] = [];
    const now = Date.now();
    for (const [key, imuSamples] of Object.entries(samples)) {
      const checked = buildCalibration(imuSamples, now);
      if (checked.ok) result[key] = checked.calibration;
      else rejected.push(`IMU${Number(key) + 1}: ${checked.reason}`);
    }

    const n = Object.keys(result).length;
    setCalRunning(false);
    setCalPhase("idle");
    setCalProgress(0);
    if (n > 0) {
      setCalibration(result);
      setCalInfo({ count: n, date: now });
      const unavailable = invalidSeen.size ? ` ${invalidSeen.size} hardware channel${invalidSeen.size === 1 ? " is" : "s are"} not reporting.` : "";
      const warning = rejected.length ? ` ${rejected.join("; ")}.` : "";
      setCalMsg(`${n} sensor${n === 1 ? "" : "s"} passed stability checks.${unavailable}${warning}`);
    } else {
      setCalibration({});
      setCalInfo(null);
      if (rejected.length) setCalMsg(`Calibration rejected — ${rejected.join("; ")}. Keep each board completely still and retry.`);
      else if (readFailure) setCalMsg(`Calibration failed — ${readFailure}.`);
      else if (timedOut) setCalMsg("Sensor stream stopped during calibration — reconnect the Teensy and retry.");
      else setCalMsg("No valid sensor data received — check the Teensy connection and retry.");
    }
  };

  const visible = recordings.filter((r) => filter === "all" ? true : filter === "suspected" ? r.suspected : !r.suspected);
  const groups = new Map<number, Recording[]>();
  for (const r of visible) { const w = ageWeeks(r); if (!groups.has(w)) groups.set(w, []); groups.get(w)!.push(r); }
  const weekKeys = [...groups.keys()].sort((a, b) => a - b);
  const filters: { key: typeof filter; label: string }[] = [
    { key: "all", label: "All" }, { key: "suspected", label: "Suspected" }, { key: "not", label: "Not suspected" },
  ];

  return (
    <main className="page-shell"><section className="app-window" aria-labelledby="page-title">
      <div className="window-bar" aria-hidden="true"><span className="window-caption">BIMA</span></div>
      <header className="landing-header">
        <div className="landing-title-block">
          <BrandLockup />
          <span className="brand-divider" aria-hidden="true" />
          <div><h1 id="page-title">Movement capture</h1><p>Patients &amp; recordings</p></div>
        </div>
        <button type="button" className="add-patient" onClick={() => setDialogOpen(true)}><span className="plus">+</span> Add patient</button>
      </header>
      <div className="setup-bar">
        <div className="setup-info">
          <span className={`status-dot ${calInfo ? "connected" : "pending"}`} />
          <span>{calInfo ? `${calInfo.count} sensor${calInfo.count === 1 ? "" : "s"} calibrated · ${formatDate(calInfo.date)}` : "Sensors not calibrated"}{calMsg && <span className="setup-msg"> — {calMsg}</span>}</span>
        </div>
        <button type="button" className="btn-cal" onClick={runCalibration} disabled={calRunning}>
          {calPhase === "countdown" ? "Get ready…" : calPhase === "capturing" ? "Calibrating…" : calInfo ? "Recalibrate sensors" : "Calibrate sensors"}
        </button>
      </div>
      {calRunning && (
        <div className={`calibration-stage ${calPhase}`} role="status" aria-live="polite">
          <div className="calibration-stage-value">{calPhase === "countdown" ? calCountdown : `${calProgress}%`}</div>
          <div className="calibration-stage-copy">
            <strong>{calPhase === "countdown" ? "Hold all four sensors down and keep them still" : "Calibrating — keep holding still"}</strong>
            <span>{calPhase === "countdown" ? "Baseline capture will begin after the countdown." : "Do not move or release the sensors until this finishes."}</span>
            <div className="calibration-progress" aria-hidden="true"><i style={{ width: calPhase === "countdown" ? `${((5 - calCountdown) / 5) * 100}%` : `${calProgress}%` }} /></div>
          </div>
        </div>
      )}
      <div className="landing-body">
        {recordings.length > 0 && (
          <div className="landing-toolbar">
            <nav className="view-switch" aria-label="Filter by category">
              {filters.map((f) => <button key={f.key} className={filter === f.key ? "active" : ""} onClick={() => setFilter(f.key)}>{f.label}</button>)}
            </nav>
          </div>
        )}
        {visible.length === 0
          ? <div className="landing-empty">{recordings.length === 0 ? "No recordings yet — add a patient to start capturing." : "No recordings in this category."}</div>
          : weekKeys.map((w) => (
            <div className="rec-group" key={w}>
              <div className="rec-group-title">{w}–{w + 1} {groups.get(w)!.some((recording) => recording.correctedAgeDays != null) ? "corrected weeks" : "weeks"}</div>
              <div className="rec-grid">{groups.get(w)!.map((r) => <RecordingCard key={r.id} recording={r} onDelete={setConfirmId} />)}</div>
            </div>
          ))}
      </div>
      {dialogOpen && <AddPatientDialog nextPatient={nextPatient} onClose={() => setDialogOpen(false)} onGo={onStart} />}
      {confirmTarget && (
        <div className="dialog-overlay" role="dialog" aria-modal="true" aria-label="Delete recording" onClick={() => setConfirmId(null)}>
          <div className="dialog confirm-dialog" onClick={(e) => e.stopPropagation()}>
            <h2>Delete recording?</h2>
            <p className="confirm-text">This permanently deletes the <strong>Patient {confirmTarget.patientNumber}</strong> {confirmTarget.kind} recording. This can’t be undone.</p>
            <div className="dialog-actions">
              <button type="button" className="btn-secondary" onClick={() => setConfirmId(null)}>Cancel</button>
              <button type="button" className="btn-danger" onClick={doDelete}>Delete</button>
            </div>
          </div>
        </div>
      )}
    </section></main>
  );
}

