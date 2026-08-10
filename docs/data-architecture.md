# BIMA data architecture

How capture data should be stored so that (a) nothing is lost, (b) pose and sensor
streams stay aligned, and (c) the archive can be turned into training matrices later
without re-recording anything.

Written 2026-07-30 against the current implementation in `app/recordings.ts`,
`app/capture-sync.ts` and `app/page.tsx`.

## 2026-08-10 implementation update

The desktop app now implements the first practical disk-archive layer described
below. The landing screen lets the operator choose a persistent folder. Existing
recordings are backfilled immediately and future sensor/video recordings are
written automatically under `participants/sub-NNNN/sessions/ses-ID/` while the
IndexedDB copy remains available for renderer-crash recovery.

Disk writes are streamed in 4 MiB chunks, finalized through a temporary file,
restricted to relative paths under the selected root, and accompanied by
SHA-256 integrity metadata. Each synchronized session also has `manifest.json`,
`session_summary.csv`, participant metadata, and a data dictionary. Sensor data
is preserved as a stable wide acquisition table and additionally exported as a
long, one-row-per-sensor analysis CSV with explicit units and validity fields.
CSV files use UTF-8 BOM, CRLF rows, deterministic columns, and formula-injection
protection for operator-entered text so they open cleanly in Excel.

The longer-term incremental-flush and rebuildable-index work in sections 3.1 and
3.5 is still pending; IndexedDB is therefore still a recovery copy rather than a
fully disposable cache.

---

## 1. What the current build does

**Timing (good).** `CaptureRun` is a single shared clock: `performance.now()` origin,
zero = capture start. Both streams stamp every record with `sessionTimeMs` and
`epochMs` from that one origin. The pose recorder uses `captureStream(0)` +
`requestFrame()`, so webm frame *N* is exactly `frames[N]` in the sidecar. That is the
right design and should be kept.

**Privacy (good).** Raw camera frames are never persisted; only a skeleton-only
render and the landmark sidecar. Keep this. It does mean the sidecar is the *only*
archival record of the video, so its integrity matters more than usual.

**Storage (the weak part).** Everything lives in IndexedDB `movement-capture` v2,
stores `recordings` and `capture-sessions`, plus `localStorage` for calibration.
`db/schema.ts` is an empty stub and nothing imports it — the Drizzle/D1/Wrangler
dependencies are dead scaffolding.

### Confirmed problems

| # | Problem | Consequence |
|---|---------|-------------|
| 1 | `navigator.storage.persist()` is never called | The browser may evict the entire dataset under disk pressure, silently and without warning |
| 2 | `navigator.storage.estimate()` is never called | No quota warning; recording fails at the moment the disk fills, mid-session |
| 3 | Data is keyed to origin `http://127.0.0.1:4820`; port is hard-coded with no fallback | Change the port and the whole dataset becomes unreachable. Worse: `ping()` reuses *any* server answering on 4820, so a port conflict can silently load a different app |
| 4 | Whole session buffered in JS arrays until stop (`samplesRef`, `trackingFramesRef`, `chunksRef`) | A crash, refresh or OOM loses 100% of the session. A 10-minute capture is a multi-hundred-MB heap |
| 5 | `listRecordings()` calls `getAll()` | Every video Blob is loaded into memory just to render the list |
| 6 | No IndexedDB indexes | No efficient query by participant, session or date |
| 7 | Sensor and pose saves are independent promise chains | If either rejects, the session stays `status:"recording"` forever |
| 8 | Pose recording silently no-ops when the camera isn't ready (`if (!poseCanvas \|\| cameraState !== "ready") return`) | A sensor-only capture looks like a normal capture until you open it |
| 9 | Participant metadata (DOB, ages, weight) is copied onto every recording, every CSV row and the sidecar | No single source of truth; a corrected DOB cannot propagate |
| 10 | Sensor rows are timestamped on **host arrival**, not by the Teensy | USB/serial scheduling jitter (typ. 1–10 ms, occasionally 30 ms+) is baked in and unrecoverable |
| 11 | `packet_index` is just the array index | A dropped Teensy packet is invisible and silently compresses time |
| 12 | No export path beyond per-file download buttons | Building a training set means clicking through the UI file by file |

Items 10 and 11 are the ones that limit how well you can ever train on both modalities.
Items 1–4 are the ones that can lose data outright.

---

## 2. What the field does

Automated GMA is an active area and the conventions are fairly settled.

- **The clinical target.** Prechtl's General Movements Assessment reads spontaneous
  whole-body movement from fetal life to ~5 months post-term; *fidgety movements*
  (~9–20 weeks post-term) are the window with the strongest predictive value for
  cerebral palsy. This is why corrected age is already a first-class field in the app —
  keep it that way.
- **Multimodal beats single-modality.** The Communications Medicine sensor-fusion study
  recorded RGB video + a pressure mat + six IMUs, all synchronized, and reached **94.5%**
  fidgety-movement classification accuracy from three-sensor fusion — significantly above
  any single modality. That is the direct justification for the aligned pose + IMU archive.
- **The unit of analysis is a 5-second snippet.** Chosen because it is the minimum
  duration a human GMA assessor will commit to a judgement on. Their corpus was 19,451
  units from 51 biweekly-assessed participants.
- **Concrete feature shapes** (worth matching so results are comparable):
  - Pose: 17 keypoints → 15 after dropping ears; `250 × 60` per snippet
    (250 frames = 5 s × 50 fps; 60 = 15 kp × 2 coords × {position, velocity}).
  - IMU: `300 × 36` (300 frames = 5 s × 60 Hz; 36 = 6 sensors × {3 accel, 3 gyro}).
  - Pressure: `500 × 6` at 100 Hz.
- **Normalization conventions:** pose is median- and moving-average-filtered, centred on
  the hip keypoints, rotated so the shoulder–hip midline aligns to the Y axis, scaled to
  a fixed body length, then z-scored — positions and velocities z-scored *separately*.
  IMU uses raw accel/gyro with a 5-frame moving average, accel and gyro z-scored separately.
- **Labels are 3-way, not binary:** `FM+`, `FM−`, `not assessable`, from two independent
  assessors (they report κ = 0.97). *Not assessable* is **excluded**, never folded into the
  negative class. Your schema must be able to represent it.

Sources are listed at the end.

---

## 3. Target architecture

### 3.1 Move the bytes to disk; keep only the index in IndexedDB

This is a desktop Electron app — it has a filesystem, and should use it. IndexedDB is a
poor archive: evictable, origin-scoped, opaque to backup software, awkward to copy to a
training machine.

```
BIMA-data/                                  # user-chosen root, backed up like any folder
  participants.json                         # one record per infant — the source of truth
  sessions/
    sub-0007_ses-20260730T1412_run-01/
      manifest.json                         # contract for this capture
      sensor.csv                            # raw, native rate, never resampled
      pose.jsonl                            # one JSON object per frame, appendable
      pose.webm                             # skeleton-only video
      events.csv                            # annotations / labels
      exports/aligned_30hz.npz              # generated, disposable, reproducible
```

IndexedDB keeps a small metadata mirror for fast listing and search. It is a **cache**:
if it is lost, rebuild it by scanning `sessions/`. That single property removes items
1, 2, 3 and 5 from the table above.

`pose.jsonl` rather than one large JSON object matters twice: it can be appended during
capture (crash-safe, fixes #4) and streamed at read time without parsing hundreds of
megabytes.

### 3.2 The manifest is the contract

```jsonc
{
  "schemaVersion": 4,
  "sessionId": "…",
  "participant": { "id": "sub-0007" },      // pointer, NOT a copy — fixes #9
  "clock": {
    "type": "performance-time-origin", "unit": "ms", "zero": "capture-start",
    "startedAtEpochMs": 1785312720123, "durationMs": 300123
  },
  "streams": [
    { "name": "sensor", "role": "imu", "file": "sensor.csv",
      "timeColumn": "session_time_ms",
      "nominalRateHz": 50, "observedRateHz": 49.87,
      "startOffsetMs": 12, "sampleCount": 14962,
      "droppedPackets": 3, "gaps": [{ "fromMs": 81340, "toMs": 81402 }],
      "channels": [{ "name": "s1_ax", "unit": "g", "imu": 1, "placement": "left_wrist" }],
      "calibration": { /* per-IMU, as captured */ } },

    { "name": "pose", "role": "pose2d", "file": "pose.jsonl", "video": "pose.webm",
      "timeColumn": "sessionTimeMs",
      "observedRateHz": 29.9, "frameCount": 8970, "startOffsetMs": 40,
      "model": "mediapipe/pose_landmarker_lite@float16/1",
      "coordinateSpace": "normalized-camera",
      "poseLandmarks": 33, "handLandmarksPerHand": 21,
      "frameOrderMatchesVideo": true }
  ],
  "integrity": { "sensor.csv": { "bytes": 4821004, "sha256": "…" } },
  "status": "complete"                      // recording | partial | complete
}
```

Design rules:

1. **Participant data by reference.** Age is *derived* at export from
   `participants.json` + `studyDate`, not copied into 15,000 CSV rows.
2. **Per-stream blocks, not a flat bag.** Adding a pressure mat or a second camera is
   then purely additive — no schema migration.
3. **Integrity hashes** so a truncated write is detectable rather than silently wrong.
4. **`status` is crash-recoverable.** A session directory found with
   `status:"recording"` at startup means the app died mid-capture: salvage what is on
   disk and mark it `partial`. This fixes #7 and makes #8 visible — a session with no
   `pose` stream in `streams[]` is obviously sensor-only.

### 3.3 Timing — the part that decides how well fusion can ever work

Three changes, in descending order of value:

1. **Stamp the Teensy's own clock.** Add `micros()` to each serial packet and store both
   `device_us` and `host_recv_ms`. You can then fit `host ≈ a·device + b` by least
   squares and recover sub-millisecond alignment. Without a device timestamp your
   alignment error is bounded by USB scheduling jitter and — more importantly — is
   *unmeasurable*, so you cannot report it.
2. **Device-side packet counter,** so drops are detected instead of silently compressing
   the time axis.
3. **Never resample into the archive.** Store each stream at its native rate with its own
   timestamps. Resampling is an *export* concern. Once you have baked a resampled archive
   you cannot undo the choice.

Expected result: alignment residual reported in the manifest, and a real number to quote
in a methods section rather than "they were recorded together".

### 3.4 The export layer (currently missing entirely)

An "Export for training" step that reads the archive and emits fixed-rate arrays.

- **Target grid: 30 Hz.** Matches the pose rate, and 50 Hz IMU decimates to it cleanly.
- **IMU → decimate with an anti-alias filter,** not nearest-sample. Naive decimation
  folds exactly the high-frequency jitter you care about back into your band.
- **Pose → linear interpolation** between neighbouring frames, plus a `pose_valid` mask
  column set to 0 wherever the gap exceeds ~1.5 frame intervals or landmarks were absent.
  **Never interpolate across a detection dropout without flagging it** — models reliably
  learn the interpolation artifact instead of the movement.
- **Windowing:** 5-second snippets (the field's standard unit), stride 1 s for training,
  non-overlapping for evaluation.
- **Labels** joined from `events.csv`, 3-way `FM+ / FM− / not assessable`, with
  `not assessable` excluded rather than treated as negative.
- **Normalization parameters written into the export manifest** so inference reproduces
  training exactly.

Emit `.npz` or Parquet. Both load in one line from Python and neither needs the app.

### 3.5 Robustness items not covered above

- Call `navigator.storage.persist()` at startup and surface the result — without it the
  metadata mirror is evictable.
- Surface `navigator.storage.estimate()`; warn at 80%.
- Have `ping()` verify the responder is actually BIMA (a known header or `/healthz`
  route) before reusing a server on port 4820, and fall back to another port on conflict.
- Flush sensor rows and pose frames to disk every ~2 s during capture.
- Add IndexedDB indexes on `patientNumber`, `captureSessionId`, `date`; split metadata
  and blobs into separate stores so listing never touches video bytes.
- Decide on `db/schema.ts`: for a single capture station the disk + manifest design is
  strictly better, and the Drizzle/D1/Wrangler dependencies should be removed. Keep them
  only if multi-site sync is actually planned.

---

## 4. Suggested order of work

| Stage | Work | Why first |
|-------|------|-----------|
| 1 | `storage.persist()`, quota warning, `ping()` identity check, session finalizer + explicit sensor-only state | Small, self-contained, stops silent data loss today |
| 2 | Teensy `micros()` + packet counter; store `device_us` and `host_recv_ms` | Firmware change — the sooner it lands, the less data is captured without recoverable timing |
| 3 | Disk archive + manifest; IndexedDB demoted to a rebuildable index; incremental flush | The structural change |
| 4 | Export-for-training pipeline (resample, window, label, normalize) | Depends on 3 |
| 5 | Annotation UI for `events.csv` (3-way, per-snippet) | Needed before any supervised training |

Stages 1 and 2 are worth doing regardless of whether stage 3 is adopted — every session
recorded before stage 2 permanently lacks recoverable device timing.

---

## Sources

- [AI Approaches towards Prechtl's Assessment of General Movements: A Systematic Literature Review](https://www.mdpi.com/1424-8220/20/18/5321)
- [Deep learning empowered sensor fusion boosts infant movement classification (Communications Medicine)](https://www.nature.com/articles/s43856-024-00701-w) · [preprint with full preprocessing detail](https://arxiv.org/html/2406.09014v2)
- [Automating General Movements Assessment with quantitative deep learning (Nature Communications)](https://www.nature.com/articles/s41467-023-44141-x)
- [AGMA-PESS: deep-learning infant pose estimator and sequence selector for GMA](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11688355/)
- [Comparison of marker-less 2D image-based methods for infant pose estimation](https://www.nature.com/articles/s41598-025-96206-0)
- [3D pose estimators benchmarked against inertial sensors](https://link.springer.com/article/10.1007/s10462-026-11559-w)
