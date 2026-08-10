"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { addCaptureAsset, addRecording, getCalibration, newId, type Calibration } from "./recordings";
import { CAPTURE_SCHEMA_VERSION, captureElapsedMs, captureEpochMs, type CaptureRun } from "./capture-sync";
import { ageWeeks, clinicalAgeMetadata } from "./session-domain";
import { parseTeensyDisplayState, writeTeensyDisplayState, type TeensyRequestedState } from "./teensy-control";
import { addClockPoint, newClockFit, parseDeviceClockLine, parseInvalidImuLine, solveClockFit, type DeviceClockStamp, calibratedMovement, calibrationIsUsable, parseImuLine, type ImuSample } from "./sensor-calibration";
import SensorBoard3D from "./SensorBoard3D";
import type { SensorViewProps } from "./capture-view-types";
import { buildSensorLongCsvParts, buildSensorWideCsvParts, type SensorRow } from "./sensor-export";

const sensors = [
  { key: "left-arm", label: "Left arm", placement: "Left wrist" },
  { key: "right-arm", label: "Right arm", placement: "Right wrist" },
  { key: "left-leg", label: "Left leg", placement: "Left ankle" },
  { key: "right-leg", label: "Right leg", placement: "Right ankle" },
];
function EmptyPlot() {
  return (
    <div className="plot" aria-label="Waiting for sensor data">
      <div className="plot-title">Movement</div>
      <div className="amplitude">— <span>m/s²</span></div>
      <div className="axis-label axis-top">2.0</div>
      <div className="axis-label axis-mid">1.0</div>
      <div className="axis-label axis-bottom">0.0</div>
      <div className="plot-area">
        <div className="grid-line top" /><div className="grid-line zero" /><div className="grid-line bottom" />
        <div className="waiting-label"><span className="spinner" />Waiting for data</div>
      </div>
    </div>
  );
}

const HIST = 6000;         // 60 s of history at the 100 Hz acquisition rate
const ACC_FULL = 2.0;      // m/s² dynamic acceleration at top of the waveform
const RAD2DEG = 180 / Math.PI;
const LEVEL_ON_DEG = 5;    // within this of the calibrated pose, call it level
const LEVEL_NEAR_DEG = 15; // amber guidance band while the operator seats it

// Angle between two vectors in degrees, or null if either has no length. Used
// against gravity, so it is insensitive to yaw — a board spun flat on the table
// still reads level, which is the intent.
function angleBetweenDeg(a: number[], b: number[]): number | null {
  const magA = Math.hypot(a[0], a[1], a[2]);
  const magB = Math.hypot(b[0], b[1], b[2]);
  if (!magA || !magB) return null;
  const cos = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (magA * magB);
  return Math.acos(Math.min(1, Math.max(-1, cos))) * RAD2DEG;
}

function LivePlot({ canvasRef, mov }: { canvasRef: { current: HTMLCanvasElement | null }; mov: number }) {
  return (
    <div className="plot" aria-label="Live movement">
      <div className="plot-title">Movement</div>
      <div className="amplitude">{mov.toFixed(2)} <span>m/s²</span></div>
      <div className="axis-label axis-top">2.0</div>
      <div className="axis-label axis-mid">1.0</div>
      <div className="axis-label axis-bottom">0.0</div>
      <div className="plot-area">
        <div className="grid-line top" /><div className="grid-line zero" /><div className="grid-line bottom" />
        <canvas ref={canvasRef} style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }} />
      </div>
    </div>
  );
}

export default function SensorView({ session, captureRun, onReadyChange, onSaved, posePreviewRef, posePreviewActive, active }: SensorViewProps) {
  const [placements, setPlacements] = useState(() => sensors.map((sensor) => sensor.placement));
  const updatePlacement = (index: number, value: string) => setPlacements((current) => current.map((item, i) => i === index ? value : item));

  const [connected, setConnected] = useState(false);
  const [status, setStatus] = useState("Waiting for sensor connection");
  const [portError, setPortError] = useState("");
  const [imuMap, setImuMap] = useState<number[]>([]);
  const [frame, setFrame] = useState<{ mov: number[]; valid: boolean[] }>({ mov: [0, 0, 0, 0], valid: [false, false, false, false] });
  const [recording, setRecording] = useState(false);
  const [elapsed, setElapsed] = useState("00:00");
  const [downloadUrl, setDownloadUrl] = useState("");

  const portRef = useRef<any>(null);
  const readerRef = useRef<any>(null);
  const keepReadingRef = useRef(false);
  const commandQueueRef = useRef<Promise<void>>(Promise.resolve());
  const canvas0 = useRef<HTMLCanvasElement>(null);
  const canvas1 = useRef<HTMLCanvasElement>(null);
  const canvas2 = useRef<HTMLCanvasElement>(null);
  const canvas3 = useRef<HTMLCanvasElement>(null);
  const canvasRefs = [canvas0, canvas1, canvas2, canvas3];
  const chartDrawAtRef = useRef(0);
  // Ring buffers: pushing 480 samples/s into plain arrays with shift() moved
  // ~11M elements per second on the main thread for nothing.
  const histRef = useRef(
    [0, 1, 2, 3].map(() => ({ data: new Float32Array(HIST), head: 0, count: 0 })),
  );
  const movRef = useRef<number[]>([0, 0, 0, 0]);
  const biasRef = useRef<number[][]>([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]);
  const lastSampleAtRef = useRef<number[]>([0, 0, 0, 0]);
  // Orientation (radians) per IMU from a gyro+accel complementary filter, and a
  // smoothed copy per row for rendering the 3D box.
  const orientRef = useRef([{ p: 0, r: 0, y: 0 }, { p: 0, r: 0, y: 0 }, { p: 0, r: 0, y: 0 }, { p: 0, r: 0, y: 0 }]);
  const renderOrientRef = useRef([{ p: 0, r: 0, y: 0 }, { p: 0, r: 0, y: 0 }, { p: 0, r: 0, y: 0 }, { p: 0, r: 0, y: 0 }]);
  const box0 = useRef<HTMLDivElement>(null);
  const box1 = useRef<HTMLDivElement>(null);
  const box2 = useRef<HTMLDivElement>(null);
  const box3 = useRef<HTMLDivElement>(null);
  const boxRefs = [box0, box1, box2, box3];
  const level0 = useRef<HTMLSpanElement>(null);
  const level1 = useRef<HTMLSpanElement>(null);
  const level2 = useRef<HTMLSpanElement>(null);
  const level3 = useRef<HTMLSpanElement>(null);
  const levelRefs = [level0, level1, level2, level3];
  const validRef = useRef<boolean[]>([false, false, false, false]);
  const latestRef = useRef<any[]>([null, null, null, null]);
  const imuMapRef = useRef<number[]>([]);
  const recordingRef = useRef(false);
  const samplesRef = useRef<any[]>([]);
  const captureRef = useRef<CaptureRun | null>(null);
  // Device-clock plumbing: the stamp closing the current record, the running
  // host-vs-device fit, and packet-loss tracking from the sequence counter.
  const deviceStampRef = useRef<DeviceClockStamp | null>(null);
  const clockFitRef = useRef(newClockFit());
  const lastSeqRef = useRef<number | null>(null);
  const droppedPacketsRef = useRef(0);
  const placementsRef = useRef(placements);
  useEffect(() => { placementsRef.current = placements; }, [placements]);

  // Calibration is held fixed during acquisition. Learning a new bias while a
  // session is running can erase the exact low-amplitude jitter being collected.
  const calibRef = useRef<Record<string, Calibration>>({});
  useEffect(() => { calibRef.current = getCalibration(); }, []);

  const sendTeensyState = useCallback((nextState: TeensyRequestedState) => {
    const command = commandQueueRef.current
      .catch(() => undefined)
      .then(() => writeTeensyDisplayState(portRef.current, nextState));
    commandQueueRef.current = command;
    command.catch((error: unknown) => {
      const detail = error instanceof Error ? error.message : "serial write failed";
      setPortError(`Could not set Teensy to ${nextState.toUpperCase()}: ${detail}`);
    });
    return command;
  }, []);

  const handleLine = useCallback((raw: string) => {
    const line = raw.trim();

    // Cheap single-character dispatch before any regex: at 480 Hz the port
    // delivers ~4,600 lines/s, over a quarter of which are the labeled
    // viewer stream (I1_AX:…) and STATS that this parser can never use.
    // Line types: "IMUn …" data / "IMUn INVALID", "T: …" record trailer,
    // "System state: …" confirmations, and the blank record terminator.
    if (line !== "") {
      const c0 = line.charCodeAt(0);
      if (c0 === 73 /* I */) {
        if (line.charCodeAt(1) !== 77 /* M */) return; // labeled viewer stream
        const parsed = parseImuLine(line);
        if (parsed) { handleImuSample(parsed.imu, parsed.sample); return; }
        // A failed I2C read is reported explicitly. Clear the cached sample so
        // the row records a gap instead of silently repeating the last value.
        const invalidImu = parseInvalidImuLine(line);
        if (invalidImu !== null) {
          latestRef.current[invalidImu] = null;
          validRef.current[invalidImu] = false;
        }
        return;
      }
      if (c0 === 84 /* T */) {
        // The firmware closes each record with its own clock and sample
        // counter, immediately before the blank line that ends the record.
        const stamp = parseDeviceClockLine(line);
        if (stamp) deviceStampRef.current = stamp;
        return;
      }
      const teensyState = parseTeensyDisplayState(line);
      if (teensyState !== "unknown") {
        setPortError("");
        setStatus(teensyState === "running"
          ? "Teensy RUNNING confirmed · OLED animation active"
          : "Teensy STANDBY confirmed");
      }
      return;
    }

    if (line === "") {
      const run = captureRef.current;
      if (recordingRef.current && run) {
        const receivedAtPerfMs = performance.now();
        const sessionTimeMs = captureElapsedMs(run, receivedAtPerfMs);
        const stampForRow = deviceStampRef.current;
        const row: Record<string, string | number> = {
          session_id: run.id,
          packet_index: samplesRef.current.length,
          session_time_ms: sessionTimeMs,
          epoch_ms: captureEpochMs(run, receivedAtPerfMs),
          t: +(sessionTimeMs / 1000).toFixed(4),
          patient_number: session.patientNumber,
          study_id: run.studyId,
          study_date: session.studyDate,
          age_days: session.correctedAgeDays,
          corrected_age_days: session.correctedAgeDays,
          chronological_age_days: session.chronologicalAgeDays,
          gestational_age_birth_days: session.gestationalAgeAtBirthDays,
          postmenstrual_age_days: session.postmenstrualAgeDays,
          age_basis: session.useCorrectedAge ? "corrected" : "chronological",
          weight_kg: session.weightKg,
        };
        // Column sN is always physical IMU N, never "the Nth sensor that
        // happened to be detected". If IMU 2 drops out mid-study its columns go
        // empty rather than silently shifting IMU 3's data — and every
        // placement label keeps pointing at the limb it was assigned to.
        for (let idx = 0; idx < sensors.length; idx++) {
          const r = latestRef.current[idx];
          row[`s${idx + 1}_placement`] = placementsRef.current[idx] || "unspecified";
          row[`s${idx + 1}_imu`] = idx + 1;
          // This IMU's own sample instant. The four are read sequentially, so
          // without the offset every sensor in a record would falsely claim the
          // cycle-start time.
          const offsetUs = stampForRow?.offsetsUs[idx];
          if (offsetUs !== undefined) {
            row[`s${idx + 1}_offset_us`] = offsetUs;
            row[`s${idx + 1}_device_us`] = stampForRow!.deviceUs + offsetUs;
          }
          if (r) { row[`s${idx + 1}_ax`] = r.ax; row[`s${idx + 1}_ay`] = r.ay; row[`s${idx + 1}_az`] = r.az; row[`s${idx + 1}_gx`] = r.gx; row[`s${idx + 1}_gy`] = r.gy; row[`s${idx + 1}_gz`] = r.gz; row[`s${idx + 1}_mov`] = +r.mov.toFixed(4); }
        }

        // Device clock alongside host time. Both are written so the archive can
        // be realigned later without trusting either one on its own.
        if (stampForRow) {
          row.device_us = stampForRow.deviceUs;
          row.seq = stampForRow.seq;
          const previousSeq = lastSeqRef.current;
          if (previousSeq !== null && stampForRow.seq > previousSeq + 1) {
            droppedPacketsRef.current += stampForRow.seq - previousSeq - 1;
          }
          lastSeqRef.current = stampForRow.seq;
          addClockPoint(clockFitRef.current, stampForRow.deviceUs / 1000, sessionTimeMs);
        }
        row.dropped_packets = droppedPacketsRef.current;

        samplesRef.current.push(row);
      }
      return;
    }
  }, [session]);

  // One decoded IMU sample: movement metric, gyro conditioning, orientation
  // integration, chart history. Refs only — safe to call at 480 Hz × 4.
  const handleImuSample = (imu: number, sample: ImuSample) => {
    {
      const { ax, ay, az, gx, gy, gz } = sample;
      const first = !validRef.current[imu];
      const calibration = calibRef.current[imu];

      // Magnitude against calibrated gravity avoids false movement when a limb
      // changes orientation, while retaining high-frequency translation/jitter.
      const mov = calibratedMovement(sample, calibration);

      // Keep the captured gyro zero fixed for the whole session. The former live
      // bias adaptation could absorb sustained small jitter as if it were drift.
      const b = biasRef.current[imu];
      if (first) {
        const cb = calibrationIsUsable(calibration) ? calibration.gyroBias : [gx, gy, gz];
        b[0] = cb[0]; b[1] = cb[1]; b[2] = cb[2];
      }
      let rx = gx - b[0], ry = gy - b[1], rz = gz - b[2];
      const gyroNoise = calibration?.gyroNoise ?? 0.012;
      const deadband = Math.min(0.03, Math.max(0.008, gyroNoise * 1.15));
      if (Math.abs(rx) < deadband) rx = 0;
      if (Math.abs(ry) < deadband) ry = 0;
      if (Math.abs(rz) < deadband) rz = 0;
      const still = mov < Math.max(0.04, (calibration?.accelNoise ?? 0.02) * 3)
        && Math.hypot(rx, ry, rz) < Math.max(0.025, gyroNoise * 1.5);

      // Orientation for the 3D box: integrate gyro, pull pitch/roll toward gravity
      // (so it's flat when flat), and bleed off yaw at rest (no absolute yaw ref).
      const o = orientRef.current[imu];
      const pitchAcc = Math.atan2(-ax, Math.hypot(ay, az));
      const rollAcc = Math.atan2(ay, az);
      const now = performance.now();
      const previous = lastSampleAtRef.current[imu];
      const dt = previous ? Math.min(0.25, Math.max(0.02, (now - previous) / 1000)) : 0.1;
      lastSampleAtRef.current[imu] = now;
      if (first) { o.p = pitchAcc; o.r = rollAcc; o.y = 0; }
      else {
        o.p = 0.92 * (o.p + ry * dt) + 0.08 * pitchAcc;
        o.r = 0.92 * (o.r + rx * dt) + 0.08 * rollAcc;
        o.y = o.y + rz * dt;
        if (still) o.y *= 0.94;
      }

      movRef.current[imu] = mov;
      validRef.current[imu] = true;
      latestRef.current[imu] = { ax, ay, az, gx: rx, gy: ry, gz: rz, mov };
      const h = histRef.current[imu];
      h.data[h.head] = mov;
      h.head = (h.head + 1) % HIST;
      if (h.count < HIST) h.count += 1;
    }
  };

  const drawAll = useCallback((now = performance.now()) => {
    const drawCharts = now - chartDrawAtRef.current >= 66;
    if (drawCharts) chartDrawAtRef.current = now;
    const dpr = window.devicePixelRatio || 1;
    if (drawCharts) for (let i = 0; i < sensors.length; i++) {
      const canvas = canvasRefs[i].current;
      if (!canvas) continue;
      const rect = canvas.getBoundingClientRect();
      if (!rect.width) continue;
      const W = Math.round(rect.width * dpr), H = Math.round(rect.height * dpr);
      if (canvas.width !== W) canvas.width = W;
      if (canvas.height !== H) canvas.height = H;
      const ctx = canvas.getContext("2d");
      if (!ctx) continue;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const w = rect.width, h = rect.height;
      ctx.clearRect(0, 0, w, h);
      const hist = histRef.current[i];
      ctx.strokeStyle = "#087886";
      ctx.lineWidth = 1.6;
      ctx.lineJoin = "round";
      // Decimate to the pixel grid with a min/max envelope per column. At
      // 6,000 samples in ~350 px the raw polyline was ~24,000 canvas ops per
      // chart per frame that all overdrew the same columns; the envelope is
      // visually identical (every spike survives — max is kept, not an
      // average) at ~2 ops per pixel.
      const columnMin = new Float32Array(Math.max(1, Math.ceil(w))).fill(Infinity);
      const columnMax = new Float32Array(Math.max(1, Math.ceil(w))).fill(-Infinity);
      const start = (hist.head - hist.count + HIST) % HIST;
      for (let j = 0; j < hist.count; j++) {
        const value = hist.data[(start + j) % HIST];
        const px = Math.min(columnMin.length - 1, Math.floor((j / (HIST - 1)) * w));
        if (value < columnMin[px]) columnMin[px] = value;
        if (value > columnMax[px]) columnMax[px] = value;
      }
      ctx.beginPath();
      let started = false;
      for (let px = 0; px < columnMin.length; px++) {
        if (columnMin[px] === Infinity) continue;
        const yTop = Math.max(0, h - Math.min(columnMax[px] / ACC_FULL, 1) * h);
        const yBottom = Math.max(0, h - Math.min(columnMin[px] / ACC_FULL, 1) * h);
        if (!started) { ctx.moveTo(px, yBottom); started = true; }
        ctx.lineTo(px, yBottom);
        if (yTop !== yBottom) ctx.lineTo(px, yTop);
      }
      if (started) ctx.stroke();
    }

    // Drive each 3D box from the smoothed orientation (eased toward the target).
    for (let i = 0; i < sensors.length; i++) {
      const box = boxRefs[i].current;
      if (!box) continue;
      const t = orientRef.current[i];
      const ro = renderOrientRef.current[i];
      ro.p += (t.p - ro.p) * 0.25;
      ro.r += (t.r - ro.r) * 0.25;
      ro.y += (t.y - ro.y) * 0.25;
      // Roll is rotation around the sensor's X axis; pitch is around Y.
      box.style.transform = `rotateZ(${(ro.y * RAD2DEG).toFixed(2)}deg) rotateX(${(ro.r * RAD2DEG).toFixed(2)}deg) rotateY(${(ro.p * RAD2DEG).toFixed(2)}deg)`;

      // Angle between the live gravity vector and the reference the sensor was
      // calibrated in, so the operator can seat the board the same way every
      // time. Falls back to true flat (+Z up) before a calibration exists.
      const badge = levelRefs[i].current;
      if (!badge) continue;
      const sample = latestRef.current[i];
      if (!validRef.current[i] || !sample) {
        badge.dataset.level = "off";
        badge.textContent = "—";
        continue;
      }
      const reference = calibrationIsUsable(calibRef.current[i])
        ? calibRef.current[i].gravity
        : [0, 0, 1];
      const deviation = angleBetweenDeg([sample.ax, sample.ay, sample.az], reference);
      if (deviation === null) {
        badge.dataset.level = "off";
        badge.textContent = "—";
        continue;
      }
      badge.dataset.level = deviation <= LEVEL_ON_DEG ? "on" : deviation <= LEVEL_NEAR_DEG ? "near" : "off";
      badge.textContent = deviation <= LEVEL_ON_DEG ? "LEVEL" : `${Math.round(deviation)}°`;
    }
  }, []); // reads only stable refs

  const readLoop = async () => {
    const port = portRef.current;
    if (!port?.readable) return;
    const reader = port.readable.getReader();
    readerRef.current = reader;
    const decoder = new TextDecoder();
    let buffer = "";
    try {
      while (keepReadingRef.current) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let idx;
        while ((idx = buffer.indexOf("\n")) >= 0) { handleLine(buffer.slice(0, idx)); buffer = buffer.slice(idx + 1); }
      }
    } catch { /* port closed / disconnected */ }
    finally {
      // Release the lock first, THEN close — closing a locked stream throws.
      try { reader.releaseLock(); } catch { /* noop */ }
      try { await port.close(); } catch { /* noop */ }
      if (portRef.current === port) portRef.current = null;
      readerRef.current = null;
    }
  };

  // Leaving the capture screen must also release the physical COM port.
  useEffect(() => () => {
    keepReadingRef.current = false;
    void readerRef.current?.cancel().catch(() => undefined);
  }, []);

  const connect = async () => {
    setPortError("");
    const serial = (navigator as any).serial;
    if (!serial) { setPortError("Web Serial isn't available here — open the app in Chrome or Edge on desktop."); return; }
    try {
      const port = await serial.requestPort();
      await port.open({ baudRate: 921600 });
      portRef.current = port;
      keepReadingRef.current = true;
      validRef.current = [false, false, false, false];
      lastSampleAtRef.current = [0, 0, 0, 0];
      setConnected(true);
      setStatus("Sensors connected · synchronizing Teensy standby…");
      readLoop();
      void sendTeensyState("standby");
    } catch (error: any) {
      const message = error?.message || "";
      setPortError(/failed to open serial port/i.test(message)
        ? "Connection failed: the Teensy port is already open in another BIMA, browser, Arduino Serial Monitor, or Teensy window. Close it there and retry."
        : message ? `Connection failed: ${message}` : "Connection cancelled");
    }
  };

  const disconnect = async () => {
    // Return the physical display to standby before closing Web Serial.
    try { await sendTeensyState("standby"); } catch { /* surfaced by sendTeensyState */ }
    keepReadingRef.current = false;
    // Cancel unblocks the pending read(); the read loop's finally releases the
    // lock and closes the port. Never close a locked stream here.
    try { await readerRef.current?.cancel(); } catch { /* noop */ }
    recordingRef.current = false;
    setRecording(false);
    setConnected(false);
    setStatus("Waiting for sensor connection");
    validRef.current = [false, false, false, false];
    setImuMap([]);
  };

  useEffect(() => {
    if (!captureRun) return;
    if (captureRun.active && !recordingRef.current) {
      if (downloadUrl) { URL.revokeObjectURL(downloadUrl); setDownloadUrl(""); }
      samplesRef.current = [];
      clockFitRef.current = newClockFit();
      lastSeqRef.current = null;
      droppedPacketsRef.current = 0;
      captureRef.current = captureRun;
      recordingRef.current = true;
      setRecording(true);
      setElapsed("00:00");
      setStatus("Capture started · waiting for Teensy RUNNING confirmation…");
      void sendTeensyState("running");
      return;
    }
    if (!captureRun.active && recordingRef.current) {
      recordingRef.current = false;
      setRecording(false);
      captureRef.current = captureRun;
      setStatus("Capture stopped · waiting for Teensy STANDBY confirmation…");
      void sendTeensyState("standby");
      const run = captureRun;
      const rows = samplesRef.current as SensorRow[];
      // Build bounded CSV string batches. A five-minute high-rate run can have
      // hundreds of thousands of rows; one giant joined string creates a large
      // temporary heap spike exactly when video post-processing also starts.
      const blob = new Blob(buildSensorWideCsvParts(rows), { type: "text/csv;charset=utf-8" });
      const analysisBlob = new Blob(buildSensorLongCsvParts(rows), { type: "text/csv;charset=utf-8" });
      const recordingId = newId();
      const baseName = `patient-${session.patientNumber}-${session.suspected ? "susp" : "non"}-wk${ageWeeks(session)}-${run.id.slice(0, 8)}`;
      const filename = `${baseName}-sensors.csv`;
      const analysisFilename = `${baseName}-sensors-long.csv`;
      const streamStartOffsetMs = Number(rows[0]?.session_time_ms ?? 0);
      setDownloadUrl(URL.createObjectURL(blob));
      const activeSensors = sensors.map((_, index) => ({ imu: index + 1, placement: placementsRef.current[index] || "unspecified", calibration: calibRef.current[index] }));
      // How well host arrival time tracked the Teensy's own clock, and whether
      // any samples were lost. Recorded per capture so the alignment quality of
      // a session is a stored fact rather than something re-derived later.
      const clockFit = solveClockFit(clockFitRef.current);
      const droppedPackets = droppedPacketsRef.current;
      addRecording({
        id: recordingId, patientNumber: session.patientNumber, suspected: session.suspected,
        ageYears: session.ageYears, ageMonths: session.ageMonths, ageDays: session.ageDays,
        ...clinicalAgeMetadata(session),
        studyDate: session.studyDate, weightKg: session.weightKg, studyId: run.studyId, note: run.note,
        kind: "sensor", date: run.stoppedAtEpochMs ?? Date.now(), blob, filename,
        size: blob.size + analysisBlob.size, thumbnail: canvas0.current?.toDataURL("image/png"), captureSessionId: run.id,
        sidecarBlob: analysisBlob, sidecarFilename: analysisFilename,
        sync: { schemaVersion: CAPTURE_SCHEMA_VERSION, clock: "performance-time-origin", startedAtEpochMs: run.startedAtEpochMs, streamStartOffsetMs, sampleCount: rows.length },
      }).then(async (archive) => {
        await addCaptureAsset(run.id, { recordingId, kind: "sensor", filename, sidecarFilename: analysisFilename, sampleCount: rows.length, streamStartOffsetMs, size: blob.size + analysisBlob.size, metadata: {
        sensors: activeSensors,
        csvTimeColumn: "session_time_ms",
        deviceClock: {
          column: "device_us",
          sequenceColumn: "seq",
          unit: "microseconds",
          source: "teensy-micros",
          droppedPackets,
          observedRateHz: rows.length > 1 && streamStartOffsetMs !== undefined
            ? Number((rows.length * 1000 / Math.max(1, Number(rows[rows.length - 1].session_time_ms) - streamStartOffsetMs)).toFixed(2))
            : 0,
          hostVsDeviceFit: clockFit,
        },
        } });
        if (archive.error) throw new Error(archive.error);
      })
        .then(() => onSaved("sensor", true)).catch(() => onSaved("sensor", false));
      captureRef.current = null;
    }
  }, [captureRun, downloadUrl, onSaved, sendTeensyState, session]);

  // Throttled UI sync from the read-loop refs.
  useEffect(() => {
    if (!connected) return;
    const id = window.setInterval(() => {
      const active = validRef.current.map((v, i) => (v ? i : -1)).filter((i) => i >= 0).slice(0, sensors.length);
      imuMapRef.current = active;
      setImuMap(active);
      setFrame({ mov: [...movRef.current], valid: [...validRef.current] });
    }, 120);
    return () => window.clearInterval(id);
  }, [connected]);

  // Draw loop while connected.
  useEffect(() => {
    if (!connected || !active) return;
    let raf = 0;
    const step = () => { drawAll(); raf = requestAnimationFrame(step); };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [active, connected, drawAll]);

  // Recording timer.
  useEffect(() => {
    if (!recording) return;
    const timer = window.setInterval(() => {
      const total = Math.floor(captureElapsedMs(captureRef.current!, performance.now()) / 1000);
      setElapsed(`${String(Math.floor(total / 60)).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}`);
    }, 250);
    return () => window.clearInterval(timer);
  }, [recording]);

  // Cleanup on unmount — cancel only; the read loop's finally closes the port
  // once the lock is released (closing a locked stream throws).
  useEffect(() => () => { keepReadingRef.current = false; try { readerRef.current?.cancel(); } catch { /* noop */ } }, []);

  const calibrationReady = imuMap.length > 0
    && imuMap.every((imu) => calibrationIsUsable(calibRef.current[imu]));
  const placementsReady = imuMap.length > 0 && imuMap.every((_, index) => !!placements[index]);
  const acquisitionStatus = connected && imuMap.length > 0 && !calibrationReady
    ? "Calibration required before recording"
    : connected && calibrationReady && !placementsReady
      ? "Select a placement for each sensor"
    : portError || status;

  useEffect(() => { onReadyChange(connected && calibrationReady && placementsReady); }, [calibrationReady, connected, onReadyChange, placementsReady]);
  useEffect(() => () => onReadyChange(false), [onReadyChange]);

  return (
    <div className={`sensor-layout ${posePreviewActive ? "with-preview" : ""}`}>
      <div className="sensors">
        {sensors.map((sensor, index) => {
          // Row N is physically IMU N. Binding to the detected-sensor list
          // instead would relabel every limb the moment one sensor drops out.
          const imu = index;
          const isValid = connected && frame.valid[imu];
          const mov = frame.mov[imu];
          const dotClass = isValid ? "connected" : connected ? "pending" : "";
          const calDone = isValid && !!calibRef.current[imu];
          const label = isValid ? `Connected · IMU${imu + 1}${calDone ? " · Calibrated ✓" : ""}` : connected ? "Sensor not detected" : "Waiting for sensor";
          return (
            <section className="sensor-row" key={sensor.key} aria-labelledby={`sensor-${index}`}>
              <div className="sensor-meta">
                <h2 id={`sensor-${index}`}>{sensor.label}</h2>
                {/* The heading above already names the limb, so the select
                    carries its caption as an aria-label to keep four rows on
                    screen without scrolling. */}
                <select aria-label={`${sensor.label} placement`} title="Placement" value={placements[index]} onChange={(event) => updatePlacement(index, event.target.value)}>
                  <option value="" disabled>Select placement</option><option>Left ankle</option><option>Right ankle</option>
                  <option>Left wrist</option><option>Right wrist</option><option>Chest</option><option>Other</option>
                </select>
                <div className="connection-status"><span className={`status-dot ${dotClass}`} />{label}</div>
              </div>
              <div className="box-cell"><SensorBoard3D boardRef={boxRefs[index]} levelRef={levelRefs[index]} live={isValid} sensorLabel={sensor.label} /></div>
              {isValid ? <LivePlot canvasRef={canvasRefs[index]} mov={mov} /> : <EmptyPlot />}
            </section>
          );
        })}
        <aside className="pose-preview" aria-label="Live pose preview">
          <span className="pose-preview-title">Pose preview</span>
          <canvas ref={posePreviewRef} />
          <span className="pose-preview-note">Skeleton only · camera not saved</span>
        </aside>
      </div>
      <div className="time-scale" aria-hidden="true"><div className="ticks" /><span>-60 s</span><span>-45 s</span><span>-30 s</span><span>-15 s</span><span>0 s</span></div>
      <footer className="record-bar sensor-record-bar">
        <div className="timer">{elapsed}</div>
        <div className="ready-state"><span className={`status-dot ${connected && calibrationReady ? "connected" : connected ? "pending" : ""}`} />{acquisitionStatus}</div>
        <div className="sensor-actions">
          {downloadUrl && <a className="download-link" href={downloadUrl} download={`movement-sensors-${Date.now()}.csv`}>Download CSV</a>}
          {connected ? (
            <>
              <button type="button" className="link-btn" onClick={() => { void disconnect(); }}>Disconnect</button>
              <span className="capture-controlled">{recording ? "Recording with video" : "Ready for synchronized capture"}</span>
            </>
          ) : (
            <button type="button" onClick={connect}>Connect sensors</button>
          )}
        </div>
      </footer>
    </div>
  );
}

