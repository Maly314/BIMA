"use client";

import { useCallback, useEffect, useRef, useState, type RefObject } from "react";
import { addCaptureAsset, addCaptureSession, addRecording, deleteRecording, getCalibration, getCaptureSession, getLastPatient, listRecordings, newId, setCalibration, updateCaptureSession, type Calibration, type Recording } from "./recordings";
import { CAPTURE_SCHEMA_VERSION, captureDurationMs, captureElapsedMs, captureEpochMs, stopCaptureRun, type CaptureRun } from "./capture-sync";
import { calculateCorrectedAge, formatAgeDays, formatPma } from "./corrected-age";
import { parseTeensyDisplayState, writeTeensyDisplayState, type TeensyRequestedState } from "./teensy-control";
import SensorBoard3D from "./SensorBoard3D";
import {
  addClockPoint,
  newClockFit,
  parseDeviceClockLine,
  parseInvalidImuLine,
  solveClockFit,
  type DeviceClockStamp,
  CALIBRATION_CAPTURE_MS,
  CALIBRATION_SETTLE_MS,
  INVALID_RE,
  buildCalibration,
  calibratedMovement,
  calibrationIsUsable,
  parseImuLine,
  type ImuSample,
} from "./sensor-calibration";

type Session = {
  patientNumber: number;
  suspected: boolean;
  ageYears: number;
  ageMonths: number;
  ageDays: number;
  studyDate: string;
  weightKg: number;
  dateOfBirth: string;
  gestationalAgeWeeks: number;
  gestationalAgeDays: number;
  gestationalAgeAtBirthDays: number;
  chronologicalAgeDays: number;
  prematurityCorrectionDays: number;
  correctedAgeDays: number;
  postmenstrualAgeDays: number;
  expectedDueDate: string;
  preterm: boolean;
  useCorrectedAge: boolean;
};
type AgeLike = {
  ageYears: number;
  ageMonths: number;
  ageDays: number;
  correctedAgeDays?: number;
  chronologicalAgeDays?: number;
  postmenstrualAgeDays?: number;
  preterm?: boolean;
  useCorrectedAge?: boolean;
};

function BrandLockup({ compact = false }: { compact?: boolean }) {
  return (
    <div className={`brand-lockup${compact ? " compact" : ""}`} aria-label="BIMA">
      <img className="brand-mark" src="/bima-icon-192.png" width={compact ? 32 : 44} height={compact ? 32 : 44} alt="" />
      <span className="brand-copy">
        <strong>BIMA</strong>
      </span>
    </div>
  );
}

// Age helpers — total days uses 365-day years / 30-day months for weekly bucketing.
const ageTotalDays = (a: AgeLike) => a.correctedAgeDays ?? (a.ageYears * 365 + a.ageMonths * 30 + a.ageDays);
const ageWeeks = (a: AgeLike) => Math.floor(ageTotalDays(a) / 7);
const ageRangeLabel = (a: AgeLike) => a.correctedAgeDays != null
  ? `${ageWeeks(a)}–${ageWeeks(a) + 1} corrected weeks`
  : `${ageWeeks(a)}–${ageWeeks(a) + 1} weeks`;
const formatAge = (a: AgeLike) => {
  if (a.correctedAgeDays != null && a.chronologicalAgeDays != null) {
    const days = a.useCorrectedAge ? a.correctedAgeDays : a.chronologicalAgeDays;
    return `${formatAgeDays(days)} ${a.useCorrectedAge ? "corrected" : "chronological"}`;
  }
  const parts: string[] = [];
  if (a.ageYears) parts.push(`${a.ageYears}y`);
  if (a.ageMonths) parts.push(`${a.ageMonths}mo`);
  if (a.ageDays) parts.push(`${a.ageDays}d`);
  return parts.length ? parts.join(" ") : "0d";
};
const detailedAgeLabel = (a: AgeLike) => a.preterm && a.postmenstrualAgeDays != null
  ? `${formatAge(a)} · ${formatPma(a.postmenstrualAgeDays)}`
  : formatAge(a);
const clinicalAgeMetadata = (session: Session) => ({
  dateOfBirth: session.dateOfBirth,
  gestationalAgeWeeks: session.gestationalAgeWeeks,
  gestationalAgeDays: session.gestationalAgeDays,
  gestationalAgeAtBirthDays: session.gestationalAgeAtBirthDays,
  chronologicalAgeDays: session.chronologicalAgeDays,
  prematurityCorrectionDays: session.prematurityCorrectionDays,
  correctedAgeDays: session.correctedAgeDays,
  postmenstrualAgeDays: session.postmenstrualAgeDays,
  expectedDueDate: session.expectedDueDate,
  preterm: session.preterm,
  useCorrectedAge: session.useCorrectedAge,
});

// One row per sensor, named for the limb it mounts on so the operator can place
// the board without cross-referencing anything. The placement select defaults to
// the matching anatomical site and stays editable.
const sensors = [
  { key: "left-arm", label: "Left arm", placement: "Left wrist" },
  { key: "right-arm", label: "Right arm", placement: "Right wrist" },
  { key: "left-leg", label: "Left leg", placement: "Left ankle" },
  { key: "right-leg", label: "Right leg", placement: "Right ankle" },
];
// Model URLs, tier logic, and all inference live in app/pose-worker.ts.

// Chromium API not yet in the TS dom lib: taps a MediaStreamTrack as a
// ReadableStream of VideoFrames.
declare class MediaStreamTrackProcessor {
  constructor(init: { track: MediaStreamTrack; maxBufferSize?: number });
  readable: ReadableStream<VideoFrame>;
}

const HAND_LINKS = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];

type TrackedPoint = { x: number; y: number; z: number; visibility?: number };
type TrackingFrame = {
  frameIndex: number;
  sessionTimeMs: number;
  epochMs: number;
  sourceVideoTimeMs: number;
  // Hands are measured fresh on every recorded frame — never carried forward.
  hands: TrackedPoint[][];
};
type CaptureChildProps = {
  session: Session;
  captureRun: CaptureRun | null;
  onReadyChange: (ready: boolean) => void;
  onSaved: (kind: "sensor" | "pose", ok: boolean) => void;
};

// The pose preview canvas is owned by CaptureWindow: VideoView paints into it
// every tracked frame and SensorView shows it, so the skeleton stays visible
// while the operator watches the sensor traces.
type SensorViewProps = CaptureChildProps & {
  posePreviewRef: RefObject<HTMLCanvasElement | null>;
  posePreviewActive: boolean;
};

type VideoViewProps = CaptureChildProps & {
  posePreviewRef: RefObject<HTMLCanvasElement | null>;
  // Lets "Record together" switch the camera on from the sensor page.
  registerCameraControl: (enable: () => Promise<boolean>) => void;
};

const csvCell = (value: unknown) => {
  const text = String(value ?? "");
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
};

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
const PREVIEW_WIDTH = 300; // backing-store width of the sensor-page pose preview
// The recorded skeleton does not need camera resolution. Encoding VP9 at 720p
// in real time was the single largest cost in the capture loop; the landmarks
// are stored at full precision in the sidecar regardless, so the video is a
// visual aid and 640 wide is ample.
const POSE_CANVAS_WIDTH = 640;
const POSE_VIDEO_BITRATE = 1_800_000;
// Ask for the camera's fastest mode: hand inference costs ~10 ms in the
// worker, so 60 fps is sustainable, and the drop-when-busy pipeline sheds
// frames harmlessly if a machine can't hold it. In dim light the camera
// itself extends exposure and delivers 15-30 fps regardless — lighting, not
// software, sets the ceiling there.
const TARGET_CAMERA_FPS = 60;
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

function SensorView({ session, captureRun, onReadyChange, onSaved, posePreviewRef, posePreviewActive }: SensorViewProps) {
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

  const drawAll = useCallback(() => {
    const dpr = window.devicePixelRatio || 1;
    for (let i = 0; i < sensors.length; i++) {
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
      const rows = samplesRef.current;
      const keys = rows.length
        ? Array.from(rows.reduce((set: Set<string>, row) => { Object.keys(row).forEach((key) => set.add(key)); return set; }, new Set<string>()))
        : ["session_id", "packet_index", "session_time_ms", "epoch_ms", "t", "patient_number", "study_id", "study_date", "age_days", "corrected_age_days", "chronological_age_days", "gestational_age_birth_days", "postmenstrual_age_days", "age_basis", "weight_kg"];
      const csv = [keys.map(csvCell).join(","), ...rows.map((row) => keys.map((key) => csvCell(row[key])).join(","))].join("\n");
      const blob = new Blob([csv], { type: "text/csv" });
      const recordingId = newId();
      const baseName = `patient-${session.patientNumber}-${session.suspected ? "susp" : "non"}-wk${ageWeeks(session)}-${run.id.slice(0, 8)}`;
      const filename = `${baseName}-sensors.csv`;
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
        size: blob.size, thumbnail: canvas0.current?.toDataURL("image/png"), captureSessionId: run.id,
        sync: { schemaVersion: CAPTURE_SCHEMA_VERSION, clock: "performance-time-origin", startedAtEpochMs: run.startedAtEpochMs, streamStartOffsetMs, sampleCount: rows.length },
      }).then(() => addCaptureAsset(run.id, { recordingId, kind: "sensor", filename, sampleCount: rows.length, streamStartOffsetMs, size: blob.size, metadata: {
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
      } }))
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
    if (!connected) return;
    let raf = 0;
    const step = () => { drawAll(); raf = requestAnimationFrame(step); };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [connected, drawAll]);

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

function VideoView({ session, captureRun, onReadyChange, onSaved, posePreviewRef, registerCameraControl }: VideoViewProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const frameCanvasRef = useRef<HTMLCanvasElement>(null);
  const poseCanvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const videoTrackRef = useRef<CanvasCaptureMediaStreamTrack | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const trackingFramesRef = useRef<TrackingFrame[]>([]);
  const frameIndexRef = useRef(0);
  const framingCandidateRef = useRef<{ value: "waiting" | "partial" | "ready"; since: number }>({ value: "waiting", since: 0 });
  const recordingRef = useRef(false);
  const captureRef = useRef<CaptureRun | null>(null);
  const recorderStartedOffsetRef = useRef(0);
  const [cameraState, setCameraState] = useState<"off" | "starting" | "ready" | "error">("off");
  // Mirrors cameraState for the imperative enable path, which must not depend on
  // a re-render having landed before the parent awaits it.
  const cameraStateRef = useRef<"off" | "starting" | "ready" | "error">("off");
  const [status, setStatus] = useState("Camera is off");
  const [hands, setHands] = useState(0);
  const [framing, setFraming] = useState<"waiting" | "partial" | "ready">("waiting");
  const [recording, setRecording] = useState(false);
  const [elapsed, setElapsed] = useState("00:00");
  const [downloadUrl, setDownloadUrl] = useState("");
  const [trackingDownloadUrl, setTrackingDownloadUrl] = useState("");
  const [downloadBaseName, setDownloadBaseName] = useState("movement-pose");
  const [trackingFps, setTrackingFps] = useState(0);

  // Cached contexts and the rate instrumentation.
  const overlayCtxRef = useRef<CanvasRenderingContext2D | null>(null);
  const poseCtxRef = useRef<CanvasRenderingContext2D | null>(null);
  const previewCtxRef = useRef<CanvasRenderingContext2D | null>(null);
  const frameCtxRef = useRef<ImageBitmapRenderingContext | null>(null);
  const trackedFrameCountRef = useRef(0);
  const fpsWindowStartRef = useRef(0);
  const backdropRef = useRef<HTMLCanvasElement | null>(null);
  const [inferenceMs, setInferenceMs] = useState(0);
  // All inference happens in app/pose-worker.ts — its own thread, its own GPU
  // context, self-healing, tier-demoting. The camera track itself is
  // transferred to the worker, which reads frames directly: the main thread
  // does NO per-frame capture work at all — it only draws results as they
  // arrive. This also makes tracking immune to every main-thread scheduling
  // hazard found along the way (rVFC's occluded-window 1 fps mode, rAF
  // starvation, compositor backpressure).
  const workerRef = useRef<Worker | null>(null);
  const frameReaderRef = useRef<ReadableStreamDefaultReader<VideoFrame> | null>(null);
  const inFlightRef = useRef(0);
  const tierRef = useRef("gpu");
  const lastResultAtRef = useRef(0);
  const restartingRef = useRef(false);
  const [tierLabel, setTierLabel] = useState("gpu");
  const [detectorResets, setDetectorResets] = useState(0);

  // Draws one camera frame WITH the landmarks measured on that exact frame.
  // This same-moment compositing is why the skeleton sits on the hand instead
  // of trailing it: the view runs one inference (~10-30 ms) behind reality —
  // imperceptible, like a mirror — but image and skeleton can never disagree.
  //
  // The image goes to a bitmaprenderer canvas (compositor-direct, no 2D
  // raster — drawing the frame through a 2D context was measured poisoning
  // the worker's GPU delegate when the window was maximised) and the strokes
  // go on a transparent 2D overlay above it. Both come from the same frame.
  const drawResult = useCallback((bitmap: ImageBitmap, detectedHands: TrackedPoint[][]) => {
    const frameCanvas = frameCanvasRef.current;
    const canvas = canvasRef.current;
    const poseCanvas = poseCanvasRef.current;
    if (!frameCanvas || !canvas || !poseCanvas) { bitmap.close(); return; }
    const frameWidth = bitmap.width || 1280;
    const frameHeight = bitmap.height || 720;

    // The live view only exists on the Video tab. While the operator is on
    // the Sensor tab the whole pane is display:none, so painting it every
    // tick is pure waste — and that is exactly when a capture is usually
    // running.
    const viewVisible = frameCanvas.offsetParent !== null;

    const overlayWidth = Math.min(960, frameWidth);
    const overlayHeight = Math.max(1, Math.round((overlayWidth * frameHeight) / frameWidth));
    if (viewVisible && (canvas.width !== overlayWidth || canvas.height !== overlayHeight)) {
      canvas.width = overlayWidth; canvas.height = overlayHeight;
    }
    const poseWidth = Math.min(POSE_CANVAS_WIDTH, frameWidth);
    const poseHeight = Math.max(1, Math.round((poseWidth * frameHeight) / frameWidth));
    if (poseCanvas.width !== poseWidth || poseCanvas.height !== poseHeight) {
      poseCanvas.width = poseWidth; poseCanvas.height = poseHeight;
    }
    const drawTrackedHands = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
      ctx.lineCap = "round"; ctx.lineJoin = "round";
      for (const landmarks of detectedHands) {
        ctx.lineWidth = Math.max(2, width / 420); ctx.strokeStyle = "#ffd166"; ctx.fillStyle = "#ffffff";
        ctx.beginPath();
        HAND_LINKS.forEach(([a,b]) => { const p=landmarks[a], q=landmarks[b]; ctx.moveTo(p.x*width,p.y*height); ctx.lineTo(q.x*width,q.y*height); });
        ctx.stroke();
        landmarks.forEach((point, index) => { ctx.beginPath(); ctx.arc(point.x*width, point.y*height, index===0?5:3, 0, Math.PI*2); ctx.fill(); ctx.stroke(); });
      }
    };

    // Contexts are cached: getContext is not free at 30–60 calls a second.
    if (!overlayCtxRef.current) overlayCtxRef.current = canvas.getContext("2d");
    if (!poseCtxRef.current) poseCtxRef.current = poseCanvas.getContext("2d", { alpha: false });
    if (!frameCtxRef.current) frameCtxRef.current = frameCanvas.getContext("bitmaprenderer");
    const ctx = overlayCtxRef.current;
    const poseCtx = poseCtxRef.current;
    if (!poseCtx) { bitmap.close(); return; }

    if (viewVisible && frameCtxRef.current) {
      if (frameCanvas.width !== frameWidth || frameCanvas.height !== frameHeight) {
        frameCanvas.width = frameWidth; frameCanvas.height = frameHeight;
      }
      // Consumes the bitmap — no close needed on this path. The canvas is
      // mirrored in CSS; the overlay mirrors its strokes to match.
      frameCtxRef.current.transferFromImageBitmap(bitmap);
    } else {
      bitmap.close();
    }

    if (ctx && viewVisible) {
      ctx.save();
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.translate(canvas.width, 0); ctx.scale(-1, 1);
      drawTrackedHands(ctx, canvas.width, canvas.height);
      ctx.restore();
    }

    // This second canvas is the only canvas sent to MediaRecorder. It contains
    // landmarks on a neutral background and never receives webcam pixels.
    // The recording canvas and its picture-in-picture copy only need painting
    // when something consumes them. On the Video tab, with no capture running,
    // nobody does — and that work was competing with inference for the frame.
    const preview = posePreviewRef.current;
    const previewVisible = !!preview && preview.offsetParent !== null;
    if (!recordingRef.current && !previewVisible) return;

    // The backdrop and its grid never change, so they are rasterised once and
    // blitted afterwards rather than re-stroked on every frame.
    let backdrop = backdropRef.current;
    if (!backdrop || backdrop.width !== poseCanvas.width || backdrop.height !== poseCanvas.height) {
      backdrop = document.createElement("canvas");
      backdrop.width = poseCanvas.width; backdrop.height = poseCanvas.height;
      const backdropCtx = backdrop.getContext("2d");
      if (backdropCtx) {
        backdropCtx.fillStyle = "#10171b";
        backdropCtx.fillRect(0, 0, backdrop.width, backdrop.height);
        // One path for the whole grid rather than ~25 separate stroke calls.
        backdropCtx.strokeStyle = "rgba(255,255,255,.06)";
        backdropCtx.lineWidth = 1;
        backdropCtx.beginPath();
        for (let x = 0; x <= backdrop.width; x += Math.max(40, backdrop.width / 16)) { backdropCtx.moveTo(x, 0); backdropCtx.lineTo(x, backdrop.height); }
        for (let y = 0; y <= backdrop.height; y += Math.max(40, backdrop.height / 9)) { backdropCtx.moveTo(0, y); backdropCtx.lineTo(backdrop.width, y); }
        backdropCtx.stroke();
      }
      backdropRef.current = backdrop;
    }

    poseCtx.save();
    poseCtx.setTransform(1, 0, 0, 1, 0, 0);
    poseCtx.drawImage(backdrop, 0, 0);
    poseCtx.translate(poseCanvas.width, 0); poseCtx.scale(-1, 1);
    drawTrackedHands(poseCtx, poseCanvas.width, poseCanvas.height);
    poseCtx.restore();

    // Mirror the skeleton into the sensor page's picture-in-picture. This is a
    // copy of the pose canvas, never the webcam, so the preview shows exactly
    // what gets recorded.
    if (preview && previewVisible && poseCanvas.width) {
      const targetHeight = Math.max(1, Math.round((PREVIEW_WIDTH * poseCanvas.height) / poseCanvas.width));
      if (preview.width !== PREVIEW_WIDTH || preview.height !== targetHeight) {
        preview.width = PREVIEW_WIDTH;
        preview.height = targetHeight;
      }
      if (!previewCtxRef.current) previewCtxRef.current = preview.getContext("2d");
      previewCtxRef.current?.drawImage(poseCanvas, 0, 0, preview.width, preview.height);
    }
  }, [posePreviewRef]);

  // Handles every message from the inference worker: draws, updates the
  // status panel, and appends to the recording sidecar.
  const handleWorkerMessage = useCallback((message: {
    type: string; tier?: string; error?: string; skipped?: boolean;
    ts?: number; videoTimeMs?: number; inferMs?: number;
    hands?: TrackedPoint[][]; bitmap?: ImageBitmap;
  }) => {
    if (message.type === "recovered") {
      tierRef.current = message.tier ?? tierRef.current;
      setTierLabel(tierRef.current);
      setDetectorResets((count) => count + 1);
      return;
    }
    if (message.type !== "result") return;
    lastResultAtRef.current = performance.now();
    inFlightRef.current = Math.max(0, inFlightRef.current - 1);
    const bitmap = message.bitmap;
    if (message.skipped) { bitmap?.close(); return; }

    const now = performance.now();
    const rawHands = message.hands ?? [];
    tierRef.current = message.tier ?? tierRef.current;

    // Rolling effective frame rate, for the readout beside the camera state.
    trackedFrameCountRef.current += 1;
    if (now - fpsWindowStartRef.current >= 1000) {
      setTrackingFps(Math.round((trackedFrameCountRef.current * 1000) / (now - fpsWindowStartRef.current)));
      setInferenceMs(message.inferMs ?? 0);
      setTierLabel(tierRef.current);
      trackedFrameCountRef.current = 0;
      fpsWindowStartRef.current = now;
    }

    // No smoothing of our own — what the model measured this frame is what
    // gets drawn (MediaPipe's VIDEO mode already filters internally), and the
    // landmarks are painted onto the exact frame they were measured on.
    const inFrame = rawHands.filter((landmarks) => landmarks.some((point) => point.x > .02 && point.x < .98 && point.y > .02 && point.y < .98)).length;
    const nextFraming = !rawHands.length ? "waiting" : inFrame === rawHands.length ? "ready" : "partial";
    if (framingCandidateRef.current.value !== nextFraming) framingCandidateRef.current = { value: nextFraming, since: now };
    else if (now - framingCandidateRef.current.since >= 350) setFraming(nextFraming);
    setHands(rawHands.length);
    if (bitmap) drawResult(bitmap, rawHands); // drawResult consumes/closes it

    const run = captureRef.current;
    if (recordingRef.current && run) {
      // Rounding by arithmetic rather than toFixed: this runs 42 landmarks x
      // 4 fields per recorded frame, and the string round-trip was showing up.
      const compact = (point: TrackedPoint): TrackedPoint => ({
        x: Math.round(point.x * 1e5) / 1e5, y: Math.round(point.y * 1e5) / 1e5, z: Math.round(point.z * 1e5) / 1e5,
        ...(point.visibility == null ? {} : { visibility: Math.round(point.visibility * 1e4) / 1e4 }),
      });
      // Timestamps are from frame CAPTURE, not result arrival — the sidecar
      // stays aligned to the camera even if inference cost varies.
      const capturedAt = message.ts ?? now;
      trackingFramesRef.current.push({
        frameIndex: frameIndexRef.current++,
        sessionTimeMs: captureElapsedMs(run, capturedAt),
        epochMs: captureEpochMs(run, capturedAt),
        sourceVideoTimeMs: message.videoTimeMs ?? 0,
        hands: rawHands.map((landmarks) => landmarks.map(compact)),
      });
      // captureStream(0) records only explicitly requested frames. This makes
      // the sidecar frame index and the pose-video frame order share one clock.
      videoTrackRef.current?.requestFrame();
    }
  }, [drawResult]);
  const handleWorkerMessageRef = useRef(handleWorkerMessage);
  handleWorkerMessageRef.current = handleWorkerMessage;

  // Spin up (or replace) the inference worker, wait for its models to come
  // online, then start feeding it camera frames. MediaStreamTrackProcessor
  // taps a clone of the camera track and each VideoFrame is TRANSFERRED to
  // the worker — zero copies, no per-frame canvas work. When the worker is
  // busy, frames are closed instead of queued, so it always sees the latest
  // camera image and backlog cannot build up. (The track itself would be the
  // cleaner transfer, but MediaStreamTrack is not transferable without a
  // Chromium feature flag; VideoFrame is.)
  const startWorker = async (): Promise<void> => {
    const track = streamRef.current?.getVideoTracks()[0];
    if (!track) throw new Error("camera track not available");
    workerRef.current?.terminate();
    frameReaderRef.current?.cancel().catch(() => {});
    // A CLASSIC worker, deliberately: MediaPipe's wasm bootstrap calls
    // importScripts, which module workers forbid ("ModuleFactory not set").
    // The bundler compiles this file and its imports into a single IIFE.
    const worker = new Worker(new URL("./pose-worker.ts", import.meta.url));
    workerRef.current = worker;
    worker.onmessage = (event) => handleWorkerMessageRef.current(event.data);
    await new Promise<void>((resolve, reject) => {
      const onInit = (event: MessageEvent) => {
        if (event.data?.type === "ready") {
          tierRef.current = event.data.tier; setTierLabel(event.data.tier);
          worker.removeEventListener("message", onInit); resolve();
        } else if (event.data?.type === "init-failed") {
          worker.removeEventListener("message", onInit); reject(new Error(event.data.error));
        }
      };
      worker.addEventListener("message", onInit);
      worker.postMessage({ type: "init" });
    });

    const processor = new MediaStreamTrackProcessor({ track: track.clone(), maxBufferSize: 1 });
    const reader = processor.readable.getReader();
    frameReaderRef.current = reader;
    inFlightRef.current = 0;
    lastResultAtRef.current = performance.now();
    (async () => {
      for (;;) {
        const { value: frame, done } = await reader.read();
        if (done) break;
        if (!frame) continue;
        // A replaced worker means this loop is stale — stop it.
        if (workerRef.current !== worker) { frame.close(); reader.cancel().catch(() => {}); break; }
        // Two frames in the pipeline: one being inferred, one in transit.
        if (inFlightRef.current >= 2) { frame.close(); continue; }
        inFlightRef.current += 1;
        worker.postMessage({ type: "frame", frame, ts: performance.now() }, [frame]);
      }
    })().catch(() => {});
  };
  const startWorkerRef = useRef(startWorker);
  startWorkerRef.current = startWorker;

  // Watchdog: results should arrive ~30×/s. Three silent seconds means the
  // worker died or wedged — replace it wholesale. The camera stream lives on
  // the main thread, so a fresh clone of the track restarts frame delivery.
  useEffect(() => {
    if (cameraState !== "ready") return;
    const id = window.setInterval(() => {
      if (performance.now() - lastResultAtRef.current > 3000 && !restartingRef.current) {
        restartingRef.current = true;
        console.warn("[tracking] worker went silent — replacing it");
        startWorkerRef.current()
          .catch((error) => console.error("[tracking] worker restart failed", error))
          .finally(() => { restartingRef.current = false; lastResultAtRef.current = performance.now(); });
      }
    }, 1000);
    return () => window.clearInterval(id);
  }, [cameraState]);

  const enableCamera = async (): Promise<boolean> => {
    if (cameraStateRef.current === "ready") return true;
    setCameraState("starting"); setStatus("Starting camera and movement tracking…");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width:{ideal:1280}, height:{ideal:720}, frameRate:{ideal:TARGET_CAMERA_FPS}, facingMode:"user" }, audio:false });
      streamRef.current = stream;
      if (videoRef.current) { videoRef.current.srcObject = stream; await videoRef.current.play(); }
      await startWorker();
      setCameraState("ready"); cameraStateRef.current = "ready";
      setStatus("Camera connected · movement tracking active");
      return true;
    } catch (error) {
      console.error(error);
      setCameraState("error"); cameraStateRef.current = "error";
      const reason = error instanceof DOMException && error.name === "NotFoundError"
        ? "No camera found — connect one and try again"
        : error instanceof DOMException && error.name === "NotReadableError"
          ? "Camera is in use by another application"
          : "Camera unavailable or permission denied";
      setStatus(reason);
      workerRef.current?.terminate(); workerRef.current = null;
      streamRef.current?.getTracks().forEach((track) => track.stop()); streamRef.current=null;
      return false;
    }
  };

  // The parent calls this to switch the camera on when a synchronized capture
  // starts from the sensor page. Registered once; the ref keeps it current.
  const enableCameraRef = useRef(enableCamera);
  enableCameraRef.current = enableCamera;
  useEffect(() => { registerCameraControl(() => enableCameraRef.current()); }, [registerCameraControl]);

  useEffect(() => {
    if (!captureRun) return;
    if (captureRun.active && !recordingRef.current) {
      const poseCanvas = poseCanvasRef.current;
      if (!poseCanvas || cameraState !== "ready") return;
      if (downloadUrl) URL.revokeObjectURL(downloadUrl);
      if (trackingDownloadUrl) URL.revokeObjectURL(trackingDownloadUrl);
      setDownloadUrl(""); setTrackingDownloadUrl(""); chunksRef.current=[]; trackingFramesRef.current=[];
      frameIndexRef.current = 0;
      captureRef.current = captureRun;
      const stream = poseCanvas.captureStream(0);
      videoTrackRef.current = stream.getVideoTracks()[0] as CanvasCaptureMediaStreamTrack;
      const mimeType = MediaRecorder.isTypeSupported("video/webm;codecs=vp9") ? "video/webm;codecs=vp9" : "video/webm";
      const recorder = new MediaRecorder(stream,{mimeType, videoBitsPerSecond: POSE_VIDEO_BITRATE}); recorderRef.current=recorder;
      recorderStartedOffsetRef.current = captureElapsedMs(captureRun, performance.now());
      recorder.ondataavailable=(event)=>{ if(event.data.size) chunksRef.current.push(event.data); };
      recorder.onstop=()=>{
        const run = captureRef.current ?? captureRun;
        const frames = trackingFramesRef.current;
        const durationMs = captureDurationMs(run);
        const stoppedAt = run.stoppedAtEpochMs ?? Date.now();
        const blob=new Blob(chunksRef.current,{type:mimeType});
        const sidecar = new Blob([JSON.stringify({
          schemaVersion: CAPTURE_SCHEMA_VERSION,
          capture: { sessionId: run.id, patientNumber: session.patientNumber, studyId: run.studyId, note: run.note, studyDate: session.studyDate, ...clinicalAgeMetadata(session), ageBasis: session.useCorrectedAge ? "corrected" : "chronological", weightKg: session.weightKg, suspected: session.suspected, startedAtEpochMs: run.startedAtEpochMs, startedAt: new Date(run.startedAtEpochMs).toISOString(), durationMs },
          synchronization: { clock: "performance-time-origin", unit: "milliseconds", zero: "capture-start", recorderStartedOffsetMs: recorderStartedOffsetRef.current, frameOrderMatchesPoseVideo: true },
          tracking: { observedFrameRateHz: durationMs ? Number((frames.length * 1000 / durationMs).toFixed(3)) : 0, coordinateSpace: "normalized-camera", handLandmarksPerHand: 21, handsMeasuredEveryFrame: true, rawCameraStored: false, visualization: "hands-only", inferenceBackend: tierRef.current },
          frames,
        })], { type: "application/json" });
        const baseName=`patient-${session.patientNumber}-${session.suspected?"susp":"non"}-wk${ageWeeks(session)}-${run.id.slice(0,8)}`;
        const filename = `${baseName}-pose.webm`;
        const sidecarFilename = `${baseName}-landmarks.json`;
        const recordingId = newId();
        setDownloadBaseName(baseName); setDownloadUrl(URL.createObjectURL(blob)); setTrackingDownloadUrl(URL.createObjectURL(sidecar)); stream.getTracks().forEach((track)=>track.stop());
        videoTrackRef.current = null;
        addRecording({ id:recordingId, patientNumber:session.patientNumber, suspected:session.suspected, ageYears:session.ageYears, ageMonths:session.ageMonths, ageDays:session.ageDays, ...clinicalAgeMetadata(session), studyDate:session.studyDate, weightKg:session.weightKg, studyId:run.studyId, note:run.note, kind:"pose", date:stoppedAt, blob, filename, size:blob.size, thumbnail:poseCanvasRef.current?.toDataURL("image/png"), sidecarBlob:sidecar, sidecarFilename, captureSessionId:run.id, sync:{ schemaVersion:CAPTURE_SCHEMA_VERSION, clock:"performance-time-origin", startedAtEpochMs:run.startedAtEpochMs, streamStartOffsetMs:recorderStartedOffsetRef.current, sampleCount:frames.length } })
          .then(() => addCaptureAsset(run.id, { recordingId, kind:"pose", filename, sidecarFilename, sampleCount:frames.length, streamStartOffsetMs:recorderStartedOffsetRef.current, size:blob.size + sidecar.size }))
          .then(() => onSaved("pose", true)).catch(() => onSaved("pose", false));
        captureRef.current = null;
      };
      recordingRef.current=true; recorder.start(1000); setRecording(true); setElapsed("00:00");
      return;
    }
    if (!captureRun.active && recordingRef.current) {
      recordingRef.current=false;
      captureRef.current = captureRun;
      if (recorderRef.current?.state === "recording") recorderRef.current.stop();
      setRecording(false);
    }
  }, [cameraState, captureRun, downloadUrl, onSaved, session, trackingDownloadUrl]);

  useEffect(() => {
    if (!recording) return;
    const timer = window.setInterval(() => { const total=Math.floor(captureElapsedMs(captureRef.current!, performance.now())/1000); setElapsed(`${String(Math.floor(total/60)).padStart(2,"0")}:${String(total%60).padStart(2,"0")}`); },250);
    return () => window.clearInterval(timer);
  }, [recording]);

  useEffect(() => () => {
    recordingRef.current = false;
    recorderRef.current?.state === "recording" && recorderRef.current.stop();
    frameReaderRef.current?.cancel().catch(() => {});
    streamRef.current?.getTracks().forEach((track) => track.stop());
    workerRef.current?.terminate(); workerRef.current = null;
  }, []);

  useEffect(() => { onReadyChange(cameraState === "ready"); }, [cameraState, onReadyChange]);
  useEffect(() => () => onReadyChange(false), [onReadyChange]);

  return (
    <section className="video-view" aria-label="Video movement tracking">
      <div className="video-stage">
        <video ref={videoRef} muted playsInline aria-hidden="true" />
        <canvas ref={frameCanvasRef} className="camera-frame" aria-hidden="true" />
        <canvas ref={canvasRef} />
        <canvas ref={poseCanvasRef} className="pose-recording-canvas" aria-hidden="true" />
        {cameraState !== "ready" && <div className="camera-empty"><div className="camera-outline" /><h2>Video movement capture</h2><p>Enable the camera to track the hands and finger landmarks.</p><button className="enable-camera" type="button" onClick={enableCamera} disabled={cameraState==="starting"}>{cameraState==="starting" ? "Starting…" : "Enable camera"}</button></div>}
        {cameraState === "ready" && <div className="tracking-badge"><span className="live-dot" />{framing === "ready" ? "Hands in frame" : framing === "partial" ? "Keep hands fully in view" : "Looking for hands"}</div>}
        {cameraState === "ready" && <div className="overlay-key"><span><i className="hand-key" />Hands</span></div>}
      </div>
      <aside className="video-panel">
        <div><span className="eyebrow">Hand tracking</span><h2>Hand movement</h2><p>The camera is used for live tracking only. Recording saves a skeleton-only video and timestamped finger landmarks.</p></div>
        <dl><div><dt>Camera</dt><dd>{cameraState === "ready" ? "Connected" : "Not connected"}</dd></div><div><dt>Hands detected</dt><dd>{cameraState === "ready" ? hands : "—"}</dd></div><div><dt>Framing</dt><dd>{cameraState === "ready" ? framing === "ready" ? "Ready" : framing === "partial" ? "Partial" : "Waiting" : "—"}</dd></div><div><dt>Tracking rate</dt><dd>{cameraState === "ready" ? `${trackingFps} fps · ${inferenceMs} ms${tierLabel !== "gpu" ? ` · ${tierLabel}` : ""}${detectorResets ? ` · recovered ×${detectorResets}` : ""}` : "—"}</dd></div></dl>
        <div className="capture-guide"><strong>Before recording</strong><span>Keep the hands visible and avoid moving the camera.</span></div>
        <p className="privacy-note">Raw camera frames are not saved. Pose video and landmark data stay on this device. Tracking output is experimental and is not a diagnosis.</p>
      </aside>
      <footer className="record-bar video-record-bar"><div className="timer">{elapsed}</div><div className="ready-state"><span className={`status-dot ${cameraState==="ready"?"connected":""}`} />{status}</div><div className="video-downloads">{downloadUrl && <a className="download-link" href={downloadUrl} download={`${downloadBaseName}-pose.webm`}>Download pose video</a>}{trackingDownloadUrl && <a className="download-link" href={trackingDownloadUrl} download={`${downloadBaseName}-landmarks.json`}>Download landmark data</a>}</div><span className="capture-controlled">{recording ? "Recording with sensors" : "Ready for synchronized capture"}</span></footer>
    </section>
  );
}

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
        <button type="button" onClick={() => download(recording.blob, recording.filename)}>{recording.kind === "pose" ? "Pose video" : recording.kind === "video" ? "Video" : "Download"}</button>
        {recording.sidecarBlob && recording.sidecarFilename && <button type="button" onClick={() => download(recording.sidecarBlob!, recording.sidecarFilename!)}>Landmarks</button>}
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
    const pn = parseInt(patientNumber, 10);
    const enteredWeight = Math.max(0, parseFloat(weightKg) || 0);
    const normalizedWeightKg = weightUnit === "lb"
      ? Number((enteredWeight * 0.45359237).toFixed(3))
      : enteredWeight;
    const years = Math.floor(correctedAge.chronologicalAgeDays / 365);
    const afterYears = correctedAge.chronologicalAgeDays - years * 365;
    const months = Math.floor(afterYears / 30);
    const days = afterYears - months * 30;
    onGo({
      patientNumber: Number.isFinite(pn) ? pn : nextPatient,
      suspected,
      ageYears: years,
      ageMonths: months,
      ageDays: days,
      studyDate,
      weightKg: normalizedWeightKg,
      dateOfBirth,
      gestationalAgeWeeks: Number(gestationalWeeks),
      gestationalAgeDays: Number(gestationalDays),
      ...correctedAge,
    });
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

function Landing({ onStart }: { onStart: (session: Session) => void }) {
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

  useEffect(() => { refresh(); }, [refresh]);

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
      <div className={`capture-pane ${view === "sensor" ? "active" : ""}`} aria-hidden={view !== "sensor"}><SensorView session={session} captureRun={captureRun} onReadyChange={setSensorReady} onSaved={handleSaved} posePreviewRef={posePreviewRef} posePreviewActive={videoReady} /></div>
      <div className={`capture-pane ${view === "video" ? "active" : ""}`} aria-hidden={view !== "video"}><VideoView session={session} captureRun={captureRun} onReadyChange={setVideoReady} onSaved={handleSaved} posePreviewRef={posePreviewRef} registerCameraControl={registerCameraControl} /></div>
    </section></main>
  );
}

export default function Home() {
  const [session, setSession] = useState<Session | null>(null);
  if (!session) return <Landing onStart={setSession} />;
  return <CaptureWindow session={session} onExit={() => setSession(null)} />;
}
