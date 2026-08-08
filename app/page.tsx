"use client";

import { useCallback, useEffect, useRef, useState, type RefObject } from "react";
import { addCaptureAsset, addCaptureSession, addRecording, deleteRecording, getCalibration, getCaptureSession, getLastPatient, listRecordings, newId, setCalibration, updateCaptureSession, type Calibration, type Recording } from "./recordings";
import { CAPTURE_SCHEMA_VERSION, captureDurationMs, captureElapsedMs, captureEpochMs, stopCaptureRun, type CaptureRun } from "./capture-sync";
import { calculateCorrectedAge, formatAgeDays, formatPma } from "./corrected-age";
import { parseTeensyDisplayState, writeTeensyDisplayState, type TeensyRequestedState } from "./teensy-control";
import { extrapolateHandsForDisplay, predictHandsForDisplay, type DisplayHandHistory, type TrackedPoint } from "./pose-display";
import { assessTrackingIntegrity } from "./tracking-integrity";
import { processSam31Video } from "./sam31-client";
import { CAMERA_DSP_BRIGHTNESS, cameraMediaConstraints, cameraStartErrorMessage } from "./camera-config";
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

function drawTrackedHands(ctx: CanvasRenderingContext2D, hands: TrackedPoint[][], width: number, height: number) {
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.lineWidth = Math.max(2, width / 420);
  ctx.strokeStyle = "#ffd166";
  ctx.fillStyle = "#ffffff";
  for (const landmarks of hands) {
    ctx.beginPath();
    HAND_LINKS.forEach(([a, b]) => {
      const p = landmarks[a], q = landmarks[b];
      ctx.moveTo(p.x * width, p.y * height);
      ctx.lineTo(q.x * width, q.y * height);
    });
    ctx.stroke();
    ctx.beginPath();
    landmarks.forEach((point, index) => {
      const radius = index === 0 ? 5 : 3;
      const x = point.x * width, y = point.y * height;
      ctx.moveTo(x + radius, y);
      ctx.arc(x, y, radius, 0, Math.PI * 2);
    });
    ctx.fill();
    ctx.stroke();
  }
}

// Raw measurements remain visible as small dots while the connected skeleton
// is display-stabilized. This makes subtle measured motion inspectable without
// turning detector shimmer into a violently twitching pose. It is a visual
// distinction only: recording already stores the raw coordinates directly.
function drawRawHandMeasurements(ctx: CanvasRenderingContext2D, hands: TrackedPoint[][], width: number, height: number) {
  ctx.fillStyle = "#24d2c1";
  ctx.globalAlpha = 0.9;
  for (const landmarks of hands) {
    ctx.beginPath();
    for (const point of landmarks) {
      const x = point.x * width, y = point.y * height;
      ctx.moveTo(x + 2, y);
      ctx.arc(x, y, 2, 0, Math.PI * 2);
    }
    ctx.fill();
  }
  ctx.globalAlpha = 1;
}

function paintBinaryRle(image: ImageData, runs: number[], color: [number, number, number, number]) {
  let index = 0;
  let foreground = false;
  for (const run of runs) {
    const end = Math.min(image.width * image.height, index + Math.max(0, run));
    if (foreground) {
      for (; index < end; index += 1) image.data.set(color, index * 4);
    } else index = end;
    foreground = !foreground;
  }
}
type Sam31Instance = {
  id: number;
  confidence?: number;
  bbox: [number, number, number, number];
  centroid: [number, number];
  maskWidth: number;
  maskHeight: number;
  rle: number[];
};
type PoseTrackingFrame = {
  frameIndex: number;
  sessionTimeMs: number;
  epochMs: number;
  sourceVideoTimeMs: number;
  // Hands are measured fresh on every recorded frame — never carried forward.
  hands: TrackedPoint[][];
};
type Sam31TrackingFrame = {
  frameIndex: number;
  sessionTimeMs: number;
  epochMs: number;
  sourceVideoTimeMs: number;
  segments: Sam31Instance[];
  source?: "sam31" | "optical-flow" | "sam31-native-propagation";
};
type TrackingFrame = PoseTrackingFrame | Sam31TrackingFrame;
type InferenceMode = "pose" | "sam31";
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
  active: boolean;
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
// The C922's native MJPEG mode is 1280x720 at 30 Hz. Requesting the earlier
// non-native 960x540 mode made Chromium choose its 1024x576 YUYV path, which
// is capped at 15 Hz. Inference still downsamples to 640 wide in the worker.
// The live C922 path on this machine delivers roughly 30 unique frames/second.
// Ask for that proven rate and let continuous exposure brighten the room rather
// than holding an underexposed 60 Hz shutter that the device cannot sustain.
// Modest UVC digital-brightness lift for the 60 Hz shutter. It does not alter
// exposure time or frame cadence; controlled front lighting remains the source
// of real signal and lower-noise landmark detail.
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

function SensorView({ session, captureRun, onReadyChange, onSaved, posePreviewRef, posePreviewActive, active }: SensorViewProps) {
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

function VideoView({ session, captureRun, onReadyChange, onSaved, posePreviewRef, registerCameraControl }: VideoViewProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
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
  const [videoOnlyRun, setVideoOnlyRun] = useState<CaptureRun | null>(null);
  const [videoProcessing, setVideoProcessing] = useState(false);
  const [elapsed, setElapsed] = useState("00:00");
  const [downloadUrl, setDownloadUrl] = useState("");
  const [rawDownloadUrl, setRawDownloadUrl] = useState("");
  const [trackingDownloadUrl, setTrackingDownloadUrl] = useState("");
  const [downloadBaseName, setDownloadBaseName] = useState("movement-pose");
  const [downloadFilename, setDownloadFilename] = useState("movement-pose.webm");
  const [trackingFps, setTrackingFps] = useState(0);
  const [cameraFps, setCameraFps] = useState(0);
  const [trackingLatencyMs, setTrackingLatencyMs] = useState(0);
  const [inputMode, setInputMode] = useState<"direct" | "scaled" | "unknown">("direct");
  const [inferenceMode, setInferenceMode] = useState<InferenceMode>("pose");
  const [modelState, setModelState] = useState<"idle" | "loading" | "running" | "error">("idle");
  const [modelError, setModelError] = useState("");

  // Cached contexts and the rate instrumentation.
  const overlayCtxRef = useRef<CanvasRenderingContext2D | null>(null);
  const poseCtxRef = useRef<CanvasRenderingContext2D | null>(null);
  const previewCtxRef = useRef<CanvasRenderingContext2D | null>(null);
  const trackedFrameCountRef = useRef(0);
  const skippedFrameCountRef = useRef(0);
  const inferSamplesRef = useRef<number[]>([]);
  const fpsWindowStartRef = useRef(0);
  const inputFrameCountRef = useRef(0);
  const inputDropCountRef = useRef(0);
  const inputWindowStartRef = useRef(0);
  const backdropRef = useRef<HTMLCanvasElement | null>(null);
  const displayHistoryRef = useRef<DisplayHandHistory | null>(null);
  const latestRawHandsRef = useRef<TrackedPoint[][]>([]);
  const handsCountRef = useRef(0);
  const framingRef = useRef<"waiting" | "partial" | "ready">("waiting");
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
  const processorTrackRef = useRef<MediaStreamTrack | null>(null);
  const inFlightRef = useRef(0);
  // The detector is synchronous and serial inside one worker. Keep one
  // replaceable latest-frame slot instead of a FIFO queue: the worker can be
  // busy, but it must never process a frame that is older than the newest one
  // waiting at the camera boundary.
  const pendingFrameRef = useRef<{ frame: VideoFrame; ts: number } | null>(null);
  const flushPendingRef = useRef<(() => void) | null>(null);
  const tierRef = useRef("gpu");
  const lastResultAtRef = useRef(0);
  const restartingRef = useRef(false);
  const cameraStartRef = useRef<Promise<boolean> | null>(null);
  const [tierLabel, setTierLabel] = useState("gpu");
  const inferenceModeRef = useRef<InferenceMode>("pose");
  const captureModeRef = useRef<InferenceMode>("pose");
  const samLoopTokenRef = useRef(0);
  const samRequestAbortRef = useRef<AbortController | null>(null);
  const samInputCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const samMaskCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const samMotionRef = useRef<{ centroid: [number, number]; at: number; velocity: [number, number] }[]>([]);
  const captureSourceRef = useRef<"sync" | "video">("sync");

  const stopInferencePipeline = useCallback(() => {
    samLoopTokenRef.current += 1;
    samRequestAbortRef.current?.abort();
    samRequestAbortRef.current = null;
    frameReaderRef.current?.cancel().catch(() => {});
    frameReaderRef.current = null;
    if (pendingFrameRef.current) {
      pendingFrameRef.current.frame.close();
      pendingFrameRef.current = null;
    }
    flushPendingRef.current = null;
    inFlightRef.current = 0;
    processorTrackRef.current?.stop();
    processorTrackRef.current = null;
    workerRef.current?.terminate();
    workerRef.current = null;
    displayHistoryRef.current = null;
    latestRawHandsRef.current = [];
    samMotionRef.current = [];
  }, []);

  const drawOverlay = useCallback((displayHands: TrackedPoint[][], rawHands = latestRawHandsRef.current) => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video || canvas.offsetParent === null) return;
    const frameWidth = video.videoWidth || 960;
    const frameHeight = video.videoHeight || 540;
    const overlayWidth = Math.min(960, frameWidth);
    const overlayHeight = Math.max(1, Math.round((overlayWidth * frameHeight) / frameWidth));
    if (canvas.width !== overlayWidth || canvas.height !== overlayHeight) {
      canvas.width = overlayWidth;
      canvas.height = overlayHeight;
    }
    if (!overlayCtxRef.current) {
      overlayCtxRef.current = canvas.getContext("2d", { alpha: true, desynchronized: true });
    }
    const ctx = overlayCtxRef.current;
    if (!ctx) return;
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    drawTrackedHands(ctx, displayHands, canvas.width, canvas.height);
    drawRawHandMeasurements(ctx, rawHands, canvas.width, canvas.height);
    ctx.restore();
  }, []);

  const drawSam31Result = useCallback((instances: Sam31Instance[], sourceAt = performance.now()) => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video || canvas.offsetParent === null) return;
    const frameWidth = video.videoWidth || 960;
    const frameHeight = video.videoHeight || 540;
    const overlayWidth = Math.min(960, frameWidth);
    const overlayHeight = Math.max(1, Math.round(overlayWidth * frameHeight / frameWidth));
    if (canvas.width !== overlayWidth || canvas.height !== overlayHeight) {
      canvas.width = overlayWidth;
      canvas.height = overlayHeight;
      overlayCtxRef.current = null;
    }
    if (!overlayCtxRef.current) overlayCtxRef.current = canvas.getContext("2d", { alpha: true, desynchronized: true });
    const ctx = overlayCtxRef.current;
    if (!ctx) return;
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    const renderAt = performance.now();
    instances.forEach((instance, index) => {
      const previous = samMotionRef.current[index];
      const sampleDelta = previous ? Math.max(16, sourceAt - previous.at) : 0;
      const observedVelocity: [number, number] = previous && sampleDelta
        ? [(instance.centroid[0] - previous.centroid[0]) / sampleDelta, (instance.centroid[1] - previous.centroid[1]) / sampleDelta]
        : [0, 0];
      const velocity: [number, number] = previous
        ? [observedVelocity[0] * 0.65 + previous.velocity[0] * 0.35, observedVelocity[1] * 0.65 + previous.velocity[1] * 0.35]
        : observedVelocity;
      // The segmentation is computed on a frame from the past. A bounded
      // constant-velocity projection keeps the overlay on the live hand while
      // preserving the raw SAM mask in the downloaded sidecar.
      const ageMs = Math.max(0, renderAt - sourceAt);
      const shiftX = Math.max(-0.14, Math.min(0.14, velocity[0] * ageMs));
      const shiftY = Math.max(-0.14, Math.min(0.14, velocity[1] * ageMs));
      const shiftedBox: [number, number, number, number] = [
        Math.max(0, Math.min(1, instance.bbox[0] + shiftX)),
        Math.max(0, Math.min(1, instance.bbox[1] + shiftY)),
        Math.max(0, Math.min(1, instance.bbox[2] + shiftX)),
        Math.max(0, Math.min(1, instance.bbox[3] + shiftY)),
      ];
      const shiftedCentroid: [number, number] = [
        Math.max(0, Math.min(1, instance.centroid[0] + shiftX)),
        Math.max(0, Math.min(1, instance.centroid[1] + shiftY)),
      ];
      samMotionRef.current[index] = { centroid: instance.centroid, at: sourceAt, velocity };
      ctx.save();
      ctx.translate(shiftX * canvas.width, shiftY * canvas.height);
      let maskCanvas = samMaskCanvasRef.current;
      if (!maskCanvas) {
        maskCanvas = document.createElement("canvas");
        samMaskCanvasRef.current = maskCanvas;
      }
      maskCanvas.width = instance.maskWidth;
      maskCanvas.height = instance.maskHeight;
      const maskCtx = maskCanvas.getContext("2d");
      if (!maskCtx) return;
      const pixels = maskCtx.createImageData(instance.maskWidth, instance.maskHeight);
      paintBinaryRle(pixels, instance.rle, index ? [255, 209, 102, 86] : [36, 210, 193, 86]);
      maskCtx.putImageData(pixels, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(maskCanvas, 0, 0, canvas.width, canvas.height);
      const [x0, y0, x1, y1] = shiftedBox;
      ctx.strokeStyle = index ? "#ffd166" : "#24d2c1";
      ctx.lineWidth = 2;
      ctx.strokeRect(x0 * canvas.width, y0 * canvas.height, (x1 - x0) * canvas.width, (y1 - y0) * canvas.height);
      ctx.beginPath();
      ctx.arc(shiftedCentroid[0] * canvas.width, shiftedCentroid[1] * canvas.height, 4, 0, Math.PI * 2);
      ctx.fillStyle = ctx.strokeStyle;
      ctx.fill();
      ctx.restore();
    });
    ctx.restore();

    const poseCanvas = poseCanvasRef.current;
    if (!poseCanvas || (!recordingRef.current && !posePreviewRef.current?.offsetParent)) return;
    poseCanvas.width = canvas.width;
    poseCanvas.height = canvas.height;
    if (!poseCtxRef.current) poseCtxRef.current = poseCanvas.getContext("2d", { alpha: false });
    poseCtxRef.current!.fillStyle = "#10171b";
    poseCtxRef.current!.fillRect(0, 0, poseCanvas.width, poseCanvas.height);
    poseCtxRef.current!.drawImage(canvas, 0, 0);
    const preview = posePreviewRef.current;
    if (preview?.offsetParent) {
      const height = Math.max(1, Math.round(PREVIEW_WIDTH * poseCanvas.height / poseCanvas.width));
      preview.width = PREVIEW_WIDTH;
      preview.height = height;
      if (!previewCtxRef.current) previewCtxRef.current = preview.getContext("2d");
      previewCtxRef.current?.drawImage(poseCanvas, 0, 0, preview.width, preview.height);
    }
  }, [posePreviewRef]);

  // The browser compositor paints the camera at camera rate. This callback
  // only paints the lightweight overlay and optional skeleton recording.
  const drawResult = useCallback((displayHands: TrackedPoint[][], recordedHands: TrackedPoint[][]) => {
    const poseCanvas = poseCanvasRef.current;
    const video = videoRef.current;
    if (!poseCanvas) return;
    const frameWidth = video?.videoWidth || 960;
    const frameHeight = video?.videoHeight || 540;
    drawOverlay(displayHands, recordedHands);

    // The live view only exists on the Video tab. While the operator is on
    // the Sensor tab the whole pane is display:none, so painting it every
    // tick is pure waste — and that is exactly when a capture is usually
    // running.
    const poseWidth = Math.min(POSE_CANVAS_WIDTH, frameWidth);
    const poseHeight = Math.max(1, Math.round((poseWidth * frameHeight) / frameWidth));
    if (poseCanvas.width !== poseWidth || poseCanvas.height !== poseHeight) {
      poseCanvas.width = poseWidth; poseCanvas.height = poseHeight;
    }
    // Contexts are cached: getContext is not free at 30–60 calls a second.
    if (!poseCtxRef.current) poseCtxRef.current = poseCanvas.getContext("2d", { alpha: false });
    const poseCtx = poseCtxRef.current;
    if (!poseCtx) return;

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
    drawTrackedHands(poseCtx, recordedHands, poseCanvas.width, poseCanvas.height);
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
  }, [drawOverlay, posePreviewRef]);

  // Handles every message from the inference worker: draws, updates the
  // status panel, and appends to the recording sidecar.
  const handleWorkerMessage = useCallback((message: {
    type: string; tier?: string; error?: string; skipped?: boolean;
    ts?: number; videoTimeMs?: number; inferMs?: number; inferSampleMs?: number; inputMode?: string;
    hands?: TrackedPoint[][];
  }) => {
    if (message.type === "recover-failed") {
      console.warn(`[tracking] worker-error ${message.error ?? "unknown"}`);
      return;
    }
    if (message.type !== "result") return;
    lastResultAtRef.current = performance.now();
    inFlightRef.current = Math.max(0, inFlightRef.current - 1);
    // Hand the newest waiting frame to the serial worker immediately. This is
    // deliberately before the skipped-result return so an inference error
    // cannot strand the reader with a permanently full handoff slot.
    flushPendingRef.current?.();
    if (message.skipped) { skippedFrameCountRef.current += 1; return; }

    const now = performance.now();
    const rawHands = message.hands ?? [];
    tierRef.current = message.tier ?? tierRef.current;
    const capturedAt = message.ts ?? now;
    const endToEndMs = Math.max(0, now - capturedAt);
    if (typeof message.inferSampleMs === "number" && Number.isFinite(message.inferSampleMs)) inferSamplesRef.current.push(message.inferSampleMs);

    // Rolling effective frame rate, for the readout beside the camera state.
    trackedFrameCountRef.current += 1;
    if (now - fpsWindowStartRef.current >= 1000) {
      setTrackingFps(Math.round((trackedFrameCountRef.current * 1000) / (now - fpsWindowStartRef.current)));
      setInferenceMs(message.inferMs ?? 0);
      setTierLabel(tierRef.current);
      setTrackingLatencyMs(Math.round(endToEndMs));
      setInputMode(message.inputMode === "scaled" ? "scaled" : message.inputMode === "direct" ? "direct" : "unknown");
      const inputWindowMs = now - inputWindowStartRef.current;
      const inferSamples = inferSamplesRef.current.slice().sort((a, b) => a - b);
      const p95Index = Math.min(inferSamples.length - 1, Math.floor(inferSamples.length * 0.95));
      const p95InferMs = inferSamples.length ? inferSamples[p95Index] : message.inferMs ?? 0;
      const maxInferMs = inferSamples.length ? inferSamples[inferSamples.length - 1] : message.inferMs ?? 0;
      console.info(`[tracking] pipeline ${JSON.stringify({
        inputFps: Math.round(inputFrameCountRef.current * 1000 / inputWindowMs),
        resultFps: Math.round(trackedFrameCountRef.current * 1000 / inputWindowMs),
        skipped: skippedFrameCountRef.current,
        dropped: inputDropCountRef.current,
        inferMs: message.inferMs ?? 0,
        p95InferMs,
        maxInferMs,
        tier: tierRef.current,
        inputMode: message.inputMode ?? "unknown",
      })}`);
      trackedFrameCountRef.current = 0;
      skippedFrameCountRef.current = 0;
      fpsWindowStartRef.current = now;
      inputFrameCountRef.current = 0;
      inputDropCountRef.current = 0;
      inputWindowStartRef.current = now;
      inferSamplesRef.current = [];
    }

    // The analysis path never receives display smoothing.
    // Raw measurements drive framing, recording, and the cyan measurement
    // dots. Only the connected yellow operator skeleton is display-stabilized.
    const inFrame = rawHands.filter((landmarks) => landmarks.some((point) => point.x > .02 && point.x < .98 && point.y > .02 && point.y < .98)).length;
    const nextFraming = !rawHands.length ? "waiting" : inFrame === rawHands.length ? "ready" : "partial";
    if (framingCandidateRef.current.value !== nextFraming) framingCandidateRef.current = { value: nextFraming, since: now };
    else if (now - framingCandidateRef.current.since >= 350 && framingRef.current !== nextFraming) {
      framingRef.current = nextFraming;
      setFraming(nextFraming);
    }
    if (handsCountRef.current !== rawHands.length) {
      handsCountRef.current = rawHands.length;
      setHands(rawHands.length);
    }
    const displayRaw = [...rawHands].sort((a, b) => (a[0]?.x ?? 0) - (b[0]?.x ?? 0));
    latestRawHandsRef.current = rawHands;
    const predicted = predictHandsForDisplay(displayHistoryRef.current, displayRaw, capturedAt, now);
    displayHistoryRef.current = predicted.history;
    drawResult(predicted.hands, rawHands);

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

  // The camera/model result cadence is hardware-dependent. Paint the latest
  // bounded prediction on every display tick so the visible skeleton stays
  // fluid without pretending that extra raw measurements were captured.
  useEffect(() => {
    let frame = 0;
    const paint = (now: number) => {
      const history = displayHistoryRef.current;
      if (history && inferenceModeRef.current === "pose") drawOverlay(extrapolateHandsForDisplay(history, now));
      frame = window.requestAnimationFrame(paint);
    };
    frame = window.requestAnimationFrame(paint);
    return () => window.cancelAnimationFrame(frame);
  }, [drawOverlay]);

  // Decode cadence is independent from inference cadence. Showing both makes
  // a camera/driver cap distinguishable from a slow model without adding a
  // per-frame React update or another paint loop.
  useEffect(() => {
    if (cameraState !== "ready") { setCameraFps(0); return; }
    const video = videoRef.current;
    if (!video?.getVideoPlaybackQuality) return;
    let lastFrames = video.getVideoPlaybackQuality().totalVideoFrames;
    let lastAt = performance.now();
    const id = window.setInterval(() => {
      const now = performance.now();
      const frames = video.getVideoPlaybackQuality().totalVideoFrames;
      const elapsedMs = now - lastAt;
      if (elapsedMs > 0) setCameraFps(Math.round((frames - lastFrames) * 1000 / elapsedMs));
      lastFrames = frames;
      lastAt = now;
    }, 1000);
    return () => window.clearInterval(id);
  }, [cameraState]);

  // Spin up (or replace) the inference worker, wait for its models to come
  // online, then start feeding it camera frames. MediaStreamTrackProcessor
  // taps a clone of the camera track and each VideoFrame is TRANSFERRED to
  // the worker — zero copies, no per-frame canvas work. When the worker is
  // busy, frames are closed instead of queued, so it always sees the latest
  // camera image and backlog cannot build up. (The track itself would be the
  // cleaner transfer, but MediaStreamTrack is not transferable without a
  // Chromium feature flag; VideoFrame is.)
  const startPoseWorker = async (): Promise<void> => {
    const track = streamRef.current?.getVideoTracks()[0];
    if (!track) throw new Error("camera track not available");
    stopInferencePipeline();
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

    const processorTrack = track.clone();
    processorTrackRef.current = processorTrack;
    // Keep a one-frame pull buffer. The reader remains active while inference
    // is busy, replacing a single pending frame so the worker always receives
    // the newest available camera image rather than building a stale queue.
    const processor = new MediaStreamTrackProcessor({ track: processorTrack, maxBufferSize: 1 });
    const reader = processor.readable.getReader();
    frameReaderRef.current = reader;
    inFlightRef.current = 0;
    lastResultAtRef.current = performance.now();
    inputFrameCountRef.current = 0;
    inputDropCountRef.current = 0;
    skippedFrameCountRef.current = 0;
    inferSamplesRef.current = [];
    inputWindowStartRef.current = performance.now();
    const send = (next: { frame: VideoFrame; ts: number }) => {
      if (workerRef.current !== worker) { next.frame.close(); return; }
      inFlightRef.current = 1;
      worker.postMessage({ type: "frame", frame: next.frame, ts: next.ts }, [next.frame]);
    };
    // Called after every inference. At most one latest frame is retained, and
    // an arriving frame replaces (and closes) the old pending one instead of
    // extending a stale FIFO queue.
    flushPendingRef.current = () => {
      if (inFlightRef.current !== 0) return;
      const pending = pendingFrameRef.current;
      if (!pending) return;
      pendingFrameRef.current = null;
      send(pending);
    };
    (async () => {
      for (;;) {
        const { value: frame, done } = await reader.read();
        if (done) break;
        if (!frame) continue;
        inputFrameCountRef.current += 1;
        // A replaced worker means this loop is stale — stop it.
        if (workerRef.current !== worker) { frame.close(); reader.cancel().catch(() => {}); processorTrack.stop(); break; }
        const next = { frame, ts: performance.now() };
        if (inFlightRef.current === 0 && !pendingFrameRef.current) send(next);
        else {
          // Keep the newest camera image, not the oldest queued image. The
          // replaced frame is explicitly closed so it cannot leak GPU memory.
          if (pendingFrameRef.current) {
            pendingFrameRef.current.frame.close();
            inputDropCountRef.current += 1;
          }
          pendingFrameRef.current = next;
        }
      }
    })().catch(() => {}).finally(() => {
      if (pendingFrameRef.current) {
        pendingFrameRef.current.frame.close();
        pendingFrameRef.current = null;
      }
      flushPendingRef.current = null;
      if (processorTrackRef.current === processorTrack) {
        processorTrack.stop();
        processorTrackRef.current = null;
      }
    });
  };
  const startPoseWorkerRef = useRef(startPoseWorker);
  startPoseWorkerRef.current = startPoseWorker;

  const startSam31 = async (): Promise<void> => {
    const video = videoRef.current;
    if (!video) throw new Error("camera video is not available");
    stopInferencePipeline();
    setModelState("loading");
    setModelError("");
    setStatus("Loading experimental SAM 3.1…");
    const loadController = new AbortController();
    samRequestAbortRef.current = loadController;
    const loadResponse = await fetch("http://127.0.0.1:4831/load", { method: "POST", signal: loadController.signal });
    const loadResult = await loadResponse.json().catch(() => ({})) as { error?: string };
    if (!loadResponse.ok) throw new Error(loadResult.error || "SAM 3.1 service could not load the model");
    if (inferenceModeRef.current !== "sam31") return;

    setModelState("running");
    setStatus("Camera connected · experimental SAM 3.1 active");
    const token = ++samLoopTokenRef.current;
    let resultCount = 0;
    let windowStarted = performance.now();
    const loop = async () => {
      if (token !== samLoopTokenRef.current || inferenceModeRef.current !== "sam31" || cameraStateRef.current !== "ready") return;
      let input = samInputCanvasRef.current;
      if (!input) {
        input = document.createElement("canvas");
        samInputCanvasRef.current = input;
      }
      input.width = 640;
      input.height = Math.max(1, Math.round(640 * (video.videoHeight || 720) / (video.videoWidth || 1280)));
      input.getContext("2d", { alpha: false })?.drawImage(video, 0, 0, input.width, input.height);
      const jpeg = await new Promise<Blob | null>((resolve) => input!.toBlob(resolve, "image/jpeg", 0.76));
      if (!jpeg || token !== samLoopTokenRef.current) return;
      const capturedAt = performance.now();
      const controller = new AbortController();
      samRequestAbortRef.current = controller;
      try {
        const response = await fetch("http://127.0.0.1:4831/infer", {
          method: "POST",
          headers: { "Content-Type": "image/jpeg" },
          body: jpeg,
          signal: controller.signal,
        });
        const result = await response.json() as { instances?: Sam31Instance[]; inferenceMs?: number; error?: string };
        if (!response.ok) throw new Error(result.error || "SAM 3.1 inference failed");
        if (token !== samLoopTokenRef.current) return;
        const instances = result.instances ?? [];
        drawSam31Result(instances, capturedAt);
        setHands(instances.length);
        setFraming(instances.length ? "ready" : "waiting");
        setInferenceMs(Math.round(result.inferenceMs ?? 0));
        setTrackingLatencyMs(Math.round(performance.now() - capturedAt));
        resultCount += 1;
        const now = performance.now();
        if (now - windowStarted >= 1000) {
          setTrackingFps(Math.round(resultCount * 1000 / (now - windowStarted)));
          resultCount = 0;
          windowStarted = now;
        }
        const run = captureRef.current;
        if (recordingRef.current && run) {
          trackingFramesRef.current.push({
            frameIndex: frameIndexRef.current++,
            sessionTimeMs: captureElapsedMs(run, capturedAt),
            epochMs: captureEpochMs(run, capturedAt),
            sourceVideoTimeMs: video.currentTime * 1000,
            segments: instances,
          });
          videoTrackRef.current?.requestFrame();
        }
      } catch (error) {
        if (controller.signal.aborted || token !== samLoopTokenRef.current) return;
        const message = error instanceof Error ? error.message : "SAM 3.1 inference failed";
        setModelState("error");
        setModelError(message);
        setStatus("SAM 3.1 unavailable");
        return;
      }
      window.setTimeout(loop, 0);
    };
    void loop();
  };
  const startSam31Ref = useRef(startSam31);
  startSam31Ref.current = startSam31;

  const processSamRecordedVideo = async (blob: Blob, run: CaptureRun): Promise<{ frames: Sam31TrackingFrame[]; annotatedBlob: Blob; processingMs: number; samKeyframes: number }> => {
    const controller = new AbortController();
    samRequestAbortRef.current = controller;
    setStatus("SAM 3.1 full propagation · uploading video");
    try {
      const result = await processSam31Video<Sam31Instance>(blob, {
        signal: controller.signal,
        onProgress: (job) => {
          setStatus(`SAM 3.1 full propagation · ${job.phase ?? "processing"} · ${Math.round(job.progress ?? 0)}%${job.processedFrames ? ` · ${job.processedFrames}/${job.frameCount ?? "?"} frames` : ""}`);
        },
        onPhase: (phase) => {
          setStatus(phase === "retrieving-video"
            ? "SAM 3.1 native tracking complete · retrieving annotated video"
            : "SAM 3.1 native tracking complete · retrieving tracking metadata");
        },
      });
      const frames = result.frames.map((frame) => ({
        ...frame,
        sessionTimeMs: frame.sourceVideoTimeMs,
        epochMs: run.startedAtEpochMs + frame.sourceVideoTimeMs,
      }));
      return { frames, annotatedBlob: result.annotatedBlob, processingMs: result.processingMs, samKeyframes: frames.length };
    } finally {
      if (samRequestAbortRef.current === controller) samRequestAbortRef.current = null;
    }
  };

  const startVideoOnlyRecording = async () => {
    if (inferenceModeRef.current !== "sam31" || recordingRef.current || captureRun || videoProcessing) return;
    if (cameraStateRef.current !== "ready") {
      const opened = await enableCamera();
      if (!opened) return;
    }
    const run: CaptureRun = {
      id: newId(),
      active: true,
      startedAtEpochMs: Date.now(),
      startedAtPerfMs: performance.now(),
      studyId: `Patient ${session.patientNumber} · video-only`,
      note: "SAM 3.1 video-only capture",
    };
    setVideoOnlyRun(run);
  };

  const stopVideoOnlyRecording = () => {
    if (!videoOnlyRun?.active) return;
    setVideoOnlyRun(stopCaptureRun(videoOnlyRun));
  };

  const startSelectedInference = async (mode: InferenceMode) => {
    if (mode === "sam31") await startSam31Ref.current();
    else {
      setModelState("running");
      setModelError("");
      await startPoseWorkerRef.current();
    }
  };
  const selectInferenceMode = async (mode: InferenceMode) => {
    if (mode === inferenceModeRef.current || recordingRef.current) return;
    stopInferencePipeline();
    inferenceModeRef.current = mode;
    setInferenceMode(mode);
    setHands(0);
    setFraming("waiting");
    setTrackingFps(0);
    setInferenceMs(0);
    setTrackingLatencyMs(0);
    const ctx = canvasRef.current?.getContext("2d");
    if (ctx && canvasRef.current) ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    if (cameraStateRef.current !== "ready") {
      setModelState("idle");
      setStatus("Camera is off");
      return;
    }
    setModelState("loading");
    setStatus(mode === "sam31" ? "Switching to experimental SAM 3.1…" : "Switching to hand pose…");
    try {
      await startSelectedInference(mode);
      if (mode === "pose") setStatus("Camera connected · hand pose active");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Inference could not start";
      setModelState("error");
      setModelError(message);
      setStatus(mode === "sam31" ? "SAM 3.1 unavailable" : "Hand pose unavailable");
    }
  };

  // Watchdog: results should arrive continuously. Three silent seconds means the
  // worker died or wedged — replace it wholesale. The camera stream lives on
  // the main thread, so a fresh clone of the track restarts frame delivery.
  useEffect(() => {
    if (cameraState !== "ready" || inferenceMode !== "pose") return;
    const id = window.setInterval(() => {
      if (performance.now() - lastResultAtRef.current > 3000 && !restartingRef.current) {
        restartingRef.current = true;
        console.warn("[tracking] worker went silent — replacing it");
        startPoseWorkerRef.current()
          .catch((error) => console.error("[tracking] worker restart failed", error))
          .finally(() => { restartingRef.current = false; lastResultAtRef.current = performance.now(); });
      }
    }, 1000);
    return () => window.clearInterval(id);
  }, [cameraState, inferenceMode]);

  const enableCamera = async (): Promise<boolean> => {
    if (cameraStateRef.current === "ready") return true;
    if (cameraStateRef.current === "starting") return cameraStartRef.current ?? false;
    cameraStateRef.current = "starting";
    const pending = (async () => {
    setCameraState("starting"); setStatus("Starting camera and movement tracking…");
    try {
      // `exact` mode requests make Chromium surface a generic NotReadableError
      // for several Windows driver negotiation failures. That was incorrectly
      // shown as "another application" even when Windows reported no active
      // camera client. Ask for the desired profile while allowing the camera
      // driver to return its closest native 720p/30 mode.
      const stream = await navigator.mediaDevices.getUserMedia(cameraMediaConstraints());
      streamRef.current = stream;
      const cameraTrack = stream.getVideoTracks()[0];
      const capabilities = cameraTrack?.getCapabilities?.() as MediaTrackCapabilities & {
        exposureMode?: string[];
        brightness?: { min: number; max: number; step: number };
      };
      // The C922 retains the prior hardware exposure mode across streams. Make
      // the bright 30 fps profile explicit; merely omitting the old manual
      // constraint leaves its short, underexposed shutter active.
      if (cameraTrack && capabilities?.exposureMode?.includes("continuous")) {
        await cameraTrack.applyConstraints({
          advanced: [{ exposureMode: "continuous" } as MediaTrackConstraintSet],
        });
      }
      if (cameraTrack && capabilities?.brightness) {
        const range = capabilities.brightness;
        const target = Math.min(range.max, Math.max(range.min, CAMERA_DSP_BRIGHTNESS));
        const brightness = range.min + Math.round((target - range.min) / range.step) * range.step;
        try {
          await cameraTrack.applyConstraints({ advanced: [{ brightness } as MediaTrackConstraintSet] });
        } catch (error) {
          // The camera stays usable if a driver exposes brightness capability
          // metadata but rejects the live control.
          console.warn("[camera] brightness control unavailable", error);
        }
      }
      console.info(`[camera] negotiation ${JSON.stringify({
        label: cameraTrack?.label,
        settings: cameraTrack?.getSettings?.(),
        capabilities,
      })}`);
      if (videoRef.current) { videoRef.current.srcObject = stream; await videoRef.current.play(); }
      setCameraState("ready"); cameraStateRef.current = "ready";
      try {
        await startSelectedInference(inferenceModeRef.current);
        setStatus(inferenceModeRef.current === "sam31" ? "Camera connected · experimental SAM 3.1 active" : "Camera connected · hand pose active");
      } catch (inferenceError) {
        const message = inferenceError instanceof Error ? inferenceError.message : "Inference could not start";
        setModelState("error");
        setModelError(message);
        if (inferenceModeRef.current === "sam31") {
          setStatus("Camera connected · SAM 3.1 unavailable");
          return true;
        }
        throw inferenceError;
      }
      return true;
    } catch (error) {
      console.error(error);
      setCameraState("error"); cameraStateRef.current = "error";
      setStatus(cameraStartErrorMessage(error));
      stopInferencePipeline();
      streamRef.current?.getTracks().forEach((track) => track.stop()); streamRef.current=null;
      return false;
    }
    })();
    cameraStartRef.current = pending;
    try { return await pending; }
    finally { if (cameraStartRef.current === pending) cameraStartRef.current = null; }
  };

  // The parent calls this to switch the camera on when a synchronized capture
  // starts from the sensor page. Registered once; the ref keeps it current.
  const enableCameraRef = useRef(enableCamera);
  enableCameraRef.current = enableCamera;
  useEffect(() => { registerCameraControl(() => enableCameraRef.current()); }, [registerCameraControl]);

  useEffect(() => {
    const activeRun = captureRun ?? videoOnlyRun;
    if (!activeRun) return;
    if (activeRun.active && !recordingRef.current) {
      const poseCanvas = poseCanvasRef.current;
      if (!poseCanvas || cameraState !== "ready") return;
      if (downloadUrl) URL.revokeObjectURL(downloadUrl);
      if (rawDownloadUrl) URL.revokeObjectURL(rawDownloadUrl);
      if (trackingDownloadUrl) URL.revokeObjectURL(trackingDownloadUrl);
      setDownloadUrl(""); setRawDownloadUrl(""); setTrackingDownloadUrl(""); chunksRef.current=[]; trackingFramesRef.current=[];
      frameIndexRef.current = 0;
      captureRef.current = activeRun;
      captureSourceRef.current = captureRun ? "sync" : "video";
      captureModeRef.current = inferenceModeRef.current;
      const savedMode = captureModeRef.current;
      if (savedMode === "sam31") {
        // SAM is analyzed after capture. Stop its live request loop so the
        // camera and MediaRecorder never wait on segmentation.
        samLoopTokenRef.current += 1;
        samRequestAbortRef.current?.abort();
        samRequestAbortRef.current = null;
      }
      // Use a cloned camera track for SAM recording so the recorder starts on
      // the Record press without taking ownership of the preview stream.
      const stream = savedMode === "sam31" ? streamRef.current?.clone() : poseCanvas.captureStream(0);
      if (!stream) return;
      videoTrackRef.current = savedMode === "sam31" ? null : stream.getVideoTracks()[0] as CanvasCaptureMediaStreamTrack;
      const mimeType = MediaRecorder.isTypeSupported("video/webm;codecs=vp9") ? "video/webm;codecs=vp9" : "video/webm";
      const recorder = new MediaRecorder(stream,{mimeType, videoBitsPerSecond: POSE_VIDEO_BITRATE}); recorderRef.current=recorder;
      recorderStartedOffsetRef.current = captureElapsedMs(activeRun, performance.now());
      recorder.ondataavailable=(event)=>{ if(event.data.size) chunksRef.current.push(event.data); };
      recorder.onstop=async()=>{
        const run = captureRef.current ?? activeRun;
        let frames = trackingFramesRef.current;
        const durationMs = captureDurationMs(run);
        const stoppedAt = run.stoppedAtEpochMs ?? Date.now();
        const blob=new Blob(chunksRef.current,{type:mimeType});
        let savedBlob = blob;
        let rawBlob: Blob | undefined;
        let samProcessingMs = 0;
        let samKeyframes = 0;
        let processingError = "";
        if (savedMode === "sam31") {
          setVideoProcessing(true);
          setStatus("SAM 3.1 post-processing · starting");
          try {
            const processed = await processSamRecordedVideo(blob, run);
            frames = processed.frames;
            savedBlob = processed.annotatedBlob;
            rawBlob = blob;
            samProcessingMs = processed.processingMs;
            samKeyframes = processed.samKeyframes;
            setStatus(`Annotated video saved · ${frames.length} tracked frames`);
          } catch (error) {
            if (error instanceof DOMException && error.name === "AbortError") { setVideoProcessing(false); return; }
            processingError = error instanceof Error ? error.message : "SAM 3.1 post-processing failed";
            setModelError(processingError);
            setStatus("Video saved · SAM post-processing failed");
          }
        }
        const integrity = savedMode === "pose"
          ? assessTrackingIntegrity(frames.filter((frame): frame is PoseTrackingFrame => "hands" in frame))
          : {
              rawLandmarks: false,
              frameCount: frames.length,
              frameIndexSequential: frames.every((frame, index) => frame.frameIndex === index),
              timestampsMonotonic: frames.every((frame, index) => index === 0 || frame.sessionTimeMs >= frames[index - 1].sessionTimeMs),
              framesWithSegments: frames.filter((frame) => "segments" in frame && frame.segments.length > 0).length,
            };
        const sidecar = new Blob([JSON.stringify({
          schemaVersion: CAPTURE_SCHEMA_VERSION,
          capture: { sessionId: run.id, patientNumber: session.patientNumber, studyId: run.studyId, note: run.note, studyDate: session.studyDate, ...clinicalAgeMetadata(session), ageBasis: session.useCorrectedAge ? "corrected" : "chronological", weightKg: session.weightKg, suspected: session.suspected, startedAtEpochMs: run.startedAtEpochMs, startedAt: new Date(run.startedAtEpochMs).toISOString(), durationMs },
          synchronization: { clock: "performance-time-origin", unit: "milliseconds", zero: "capture-start", recorderStartedOffsetMs: recorderStartedOffsetRef.current, frameOrderMatchesPoseVideo: true },
          tracking: savedMode === "pose"
            ? { observedFrameRateHz: durationMs ? Number((frames.length * 1000 / durationMs).toFixed(3)) : 0, coordinateSpace: "normalized-camera", handLandmarksPerHand: 21, handsMeasuredEveryFrame: true, rawCameraStored: false, visualization: "hands-only", inferenceBackend: tierRef.current, integrity }
            : { observedFrameRateHz: durationMs ? Number((frames.length * 1000 / durationMs).toFixed(3)) : 0, coordinateSpace: "normalized-camera", output: "native-hand-mask-rle-bbox-centroid", maskResolution: [640, 360], rawCameraStored: true, visualization: "native-mask-overlay", inferenceBackend: "Meta SAM 3.1 native propagate_in_video", prompt: "human hand on frame 0", nativeTrackedFrames: samKeyframes, processingMs: samProcessingMs, postProcessed: true, processingError: processingError || undefined, integrity },
          frames,
        })], { type: "application/json" });
        const baseName=`patient-${session.patientNumber}-${session.suspected?"susp":"non"}-wk${ageWeeks(session)}-${run.id.slice(0,8)}`;
        const filename = `${baseName}-${savedMode === "sam31" && !processingError ? "sam31-tracked.mp4" : savedMode === "sam31" ? "sam31-raw.webm" : "pose.webm"}`;
        const rawFilename = savedMode === "sam31" && rawBlob ? `${baseName}-sam31-raw.webm` : undefined;
        const sidecarFilename = `${baseName}-${savedMode === "sam31" ? "segments" : "landmarks"}.json`;
        const recordingId = newId();
        const annotationFailed = savedMode === "sam31" && Boolean(processingError);
        setDownloadBaseName(baseName); setDownloadFilename(filename); setDownloadUrl(annotationFailed ? "" : URL.createObjectURL(savedBlob)); setRawDownloadUrl(annotationFailed ? URL.createObjectURL(blob) : rawBlob ? URL.createObjectURL(rawBlob) : ""); setTrackingDownloadUrl(URL.createObjectURL(sidecar));
        stream.getTracks().forEach((track)=>track.stop());
        videoTrackRef.current = null;
        addRecording({ id:recordingId, patientNumber:session.patientNumber, suspected:session.suspected, ageYears:session.ageYears, ageMonths:session.ageMonths, ageDays:session.ageDays, ...clinicalAgeMetadata(session), studyDate:session.studyDate, weightKg:session.weightKg, studyId:run.studyId, note:run.note, kind:"pose", date:stoppedAt, blob:savedBlob, filename, size:savedBlob.size + (rawBlob?.size ?? 0), rawBlob, rawFilename, annotationStatus:savedMode === "sam31" ? (processingError ? "failed" : "complete") : undefined, processingError:processingError || undefined, thumbnail:poseCanvasRef.current?.toDataURL("image/png"), sidecarBlob:sidecar, sidecarFilename, captureSessionId:captureSourceRef.current === "sync" ? run.id : undefined, sync:{ schemaVersion:CAPTURE_SCHEMA_VERSION, clock:"performance-time-origin", startedAtEpochMs:run.startedAtEpochMs, streamStartOffsetMs:recorderStartedOffsetRef.current, sampleCount:frames.length } })
          .then(() => captureSourceRef.current === "sync" ? addCaptureAsset(run.id, { recordingId, kind:"pose", filename, sidecarFilename, sampleCount:frames.length, streamStartOffsetMs:recorderStartedOffsetRef.current, size:savedBlob.size + (rawBlob?.size ?? 0) + sidecar.size }) : undefined)
          .then(() => { setVideoProcessing(false); if (captureSourceRef.current === "sync") onSaved("pose", true); })
          .catch(() => { setVideoProcessing(false); if (captureSourceRef.current === "sync") onSaved("pose", false); });
        captureRef.current = null;
        if (captureSourceRef.current === "video") setVideoOnlyRun(null);
      };
      recordingRef.current=true;
      recorder.start(1000);
      setStatus(savedMode === "sam31" ? "Recording raw video · SAM runs after Stop" : "Recording video and sensors");
      setRecording(true); setElapsed("00:00");
      return;
    }
    if (!activeRun.active && recordingRef.current) {
      recordingRef.current=false;
      captureRef.current = activeRun;
      if (recorderRef.current?.state === "recording") recorderRef.current.stop();
      setRecording(false);
    }
  }, [cameraState, captureRun, downloadUrl, onSaved, rawDownloadUrl, session, trackingDownloadUrl, videoOnlyRun]);

  useEffect(() => {
    if (!recording) return;
    const timer = window.setInterval(() => { const total=Math.floor(captureElapsedMs(captureRef.current!, performance.now())/1000); setElapsed(`${String(Math.floor(total/60)).padStart(2,"0")}:${String(total%60).padStart(2,"0")}`); },250);
    return () => window.clearInterval(timer);
  }, [recording]);

  useEffect(() => () => {
    recordingRef.current = false;
    if (recorderRef.current?.state === "recording") recorderRef.current.stop();
    stopInferencePipeline();
    streamRef.current?.getTracks().forEach((track) => track.stop());
  }, [stopInferencePipeline]);

  useEffect(() => { onReadyChange(cameraState === "ready"); }, [cameraState, onReadyChange]);
  useEffect(() => () => onReadyChange(false), [onReadyChange]);

  return (
    <section className="video-view" aria-label="Video movement tracking">
      <div className="video-stage">
        <video ref={videoRef} muted playsInline aria-hidden="true" />
        <canvas ref={canvasRef} />
        <canvas ref={poseCanvasRef} className="pose-recording-canvas" aria-hidden="true" />
        {cameraState !== "ready" && <div className="camera-empty"><div className="camera-outline" /><h2>Video movement capture</h2><p>Choose an inference mode, then enable the camera.</p><button className="enable-camera" type="button" onClick={enableCamera} disabled={cameraState==="starting"}>{cameraState==="starting" ? "Starting…" : "Enable camera"}</button></div>}
        {cameraState === "ready" && <div className="tracking-badge"><span className={`live-dot ${modelState === "error" ? "error" : ""}`} />{modelState === "loading" ? "Loading model" : modelState === "error" ? "Inference unavailable" : framing === "ready" ? "Hands in frame" : framing === "partial" ? "Keep hands fully in view" : "Looking for hands"}</div>}
        {cameraState === "ready" && inferenceMode === "pose" && <div className="overlay-key"><span><i className="hand-key" />Stable pose</span><span><i className="raw-key" />Raw landmarks</span></div>}
        {cameraState === "ready" && inferenceMode === "sam31" && <div className="overlay-key"><span><i className="raw-key" />SAM hand mask</span><span><i className="hand-key" />Second mask</span></div>}
      </div>
      <aside className="video-panel">
        <div className="inference-mode-switch" role="group" aria-label="Video inference model">
          <button type="button" className={inferenceMode === "pose" ? "active" : ""} aria-pressed={inferenceMode === "pose"} disabled={recording} onClick={() => void selectInferenceMode("pose")}>Hand pose</button>
          <button type="button" className={inferenceMode === "sam31" ? "active experimental" : "experimental"} aria-pressed={inferenceMode === "sam31"} disabled={recording} onClick={() => void selectInferenceMode("sam31")}><span>Experimental</span>SAM 3.1</button>
        </div>
        {inferenceMode === "sam31" && <button type="button" className={`sam-record-button ${videoOnlyRun?.active ? "recording" : ""}`} onClick={() => videoOnlyRun?.active ? stopVideoOnlyRecording() : void startVideoOnlyRecording()} disabled={videoProcessing || (!!captureRun && !videoOnlyRun?.active) || cameraState === "starting"}>{videoOnlyRun?.active ? "Stop video" : videoProcessing ? "Processing video…" : "Record video"}</button>}
        <div><span className="eyebrow">{inferenceMode === "sam31" ? "Segmentation tracking" : "Hand tracking"}</span><h2>Hand movement</h2><p>{inferenceMode === "sam31" ? "Live preview is optional. Record the smooth camera video first; SAM 3.1 analyzes it afterward into hand masks, boxes, and centroids." : "The yellow skeleton is stabilized for viewing. Cyan dots show each unsmoothed measurement; recording saves those raw timestamped landmarks."}</p></div>
        {modelError && inferenceMode === "sam31" && <div className="model-error" role="alert">{modelError}</div>}
        <dl><div><dt>Camera</dt><dd>{cameraState === "ready" ? `Connected${cameraFps ? ` · ${cameraFps} fps` : ""}` : "Not connected"}</dd></div><div><dt>{inferenceMode === "sam31" ? "Masks detected" : "Hands detected"}</dt><dd>{cameraState === "ready" ? hands : "—"}</dd></div><div><dt>Framing</dt><dd>{cameraState === "ready" ? framing === "ready" ? "Ready" : framing === "partial" ? "Partial" : "Waiting" : "—"}</dd></div><div><dt>Tracking rate</dt><dd>{cameraState === "ready" ? `${trackingFps} fps · ${inferenceMs} ms · ${trackingLatencyMs} ms total${inferenceMode === "pose" && inputMode !== "unknown" ? ` · ${inputMode}` : ""}${inferenceMode === "pose" && tierLabel !== "gpu" ? ` · ${tierLabel}` : ""}` : "—"}</dd></div></dl>
        <div className="capture-guide"><strong>Before recording</strong><span>Keep the hands visible and avoid moving the camera.</span></div>
        <p className="privacy-note">{inferenceMode === "sam31" ? "The annotated video, untouched source video, and mask data" : "Pose video and landmark data"} stay on this device. Tracking output is experimental and is not a diagnosis.</p>
      </aside>
      <footer className="record-bar video-record-bar"><div className="timer">{elapsed}</div><div className="ready-state"><span className={`status-dot ${cameraState==="ready"?"connected":""}`} />{status}</div><div className="video-downloads">{downloadUrl && <a className="download-link" href={downloadUrl} download={downloadFilename}>Download annotated video</a>}{rawDownloadUrl && <a className="download-link secondary" href={rawDownloadUrl} download={`${downloadBaseName}-sam31-raw.webm`}>Download raw video</a>}{trackingDownloadUrl && <a className="download-link" href={trackingDownloadUrl} download={`${downloadBaseName}-${inferenceMode === "sam31" ? "segments" : "landmarks"}.json`}>Download tracking data</a>}</div><span className="capture-controlled">{recording ? (inferenceMode === "sam31" ? "Recording raw video · SAM runs after Stop" : "Recording with sensors") : inferenceMode === "sam31" ? "Ready for video-only capture" : "Ready for synchronized capture"}</span></footer>
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
