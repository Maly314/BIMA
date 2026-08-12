# BIMA project timeline

Generated from the local Git history and current working tree on 2026-08-11. All times are Central Time. The first defensible code timestamp is 2026-07-20 21:14:40 CDT. Work performed between commits can only be dated to its eventual commit unless another timestamped artifact exists.

## Complete timestamped change ledger

### 2026-07-20 — initial application

#### 2026-07-20 21:14:40 CDT — Initial movement sensor capture interface (`b6641a6`)

- Created the first application shell, page layout, global styling, metadata, icons, and build configuration.
- Added the initial movement-capture screen and its rendered-page characterization test.
- Added the initial database schema and worker/build scaffolding inherited by later versions.
- Established the project’s first version-controlled point. This is the earliest reliable project start date.

#### 2026-07-20 21:22:47 CDT — Webcam hand tracking and recording (`8f6b80c`)

- Added webcam acquisition to the movement-capture interface.
- Added live hand tracking and video recording controls.
- Added the browser-side packages required for camera inference and recording.
- Updated the visual layout for the camera/tracking surface.

### 2026-08-05 — consolidated desktop application

#### 2026-08-05 20:44:24 CDT — Made GitHub repo (`65229b5`)

This checkpoint also consolidated the working project into a complete desktop application:

- **Desktop runtime:** Added the Electron launcher, shortcut installers, Windows launch scripts, application manifest, icons, and desktop branding.
- **Four physical sensors:** Added the four-IMU acquisition interface, fixed limb-to-sensor mapping, serial communication, Teensy state commands, and hardware-aware calibration.
- **Sensor visualization:** Added interactive 3D sensor boards and orientation feedback used during calibration.
- **Synchronized capture:** Added the shared sensor/video capture clock and data structures for aligning recordings.
- **Patient metadata:** Added corrected-age calculation, study metadata, patient numbering, weight handling, and recording persistence.
- **Video pose path:** Added the hand-pose worker, live overlay, camera recording, and tracking data capture.
- **Movement model:** Added baseline training and inference code plus a separate live model-test page.
- **Synthetic data:** Added neonatal-style IMU generators, evaluation scripts, notebooks, manifests, example outputs, and baseline model artifacts.
- **Pose research:** Added KinePose/KineRes research architectures, datasets, training scripts, ablations, robustness benchmarks, reports, and visualization tools.
- **Firmware:** Added the four-LSM6DSV16X Teensy firmware, OLED boot frames, BIMA logo assets, and the hardware handoff package.
- **Documentation and tests:** Added data architecture documentation and tests for corrected age, sensor calibration, Teensy control, and the rendered application.

### 2026-08-08 — characterization and structural refactor

#### 2026-08-08 12:44:02 CDT — Verified SAM v7 baseline (`4e90bd9`)

- Captured the known-working SAM/video baseline before structural changes.
- Added the local SAM service and isolated chunk worker.
- Added pose-display prediction and tracking-integrity analysis.
- Updated the Electron runtime, recording schema, video worker, dependencies, and regression tests.

#### 2026-08-08 12:44:39 CDT — SAM HTTP contract tests (`74dc4ca`)

- Added executable tests for SAM health, version reporting, invalid uploads, and stable HTTP behavior.
- Included those contracts in the normal test command.

#### 2026-08-08 12:46:16 CDT — Extracted SAM client (`5633b6e`)

- Moved SAM upload, polling, metadata retrieval, and annotated-video retrieval out of the main page.
- Added focused client tests for request ordering and failure propagation.
- Reduced coupling between the React interface and the local inference service.

#### 2026-08-08 12:48:23 CDT — Centralized camera configuration (`acaf6d2`)

- Moved resolution, target frame rate, facing mode, brightness, and camera error messages into one module.
- Added camera contract tests so later tuning could not silently change capture behavior.

#### 2026-08-08 12:52:42 CDT — Restored SAM retrieval feedback (`8cb39e6`)

- Corrected the UI lifecycle so retrieval of the finished annotated video remained visible.
- Added client-phase reporting and tests around the retrieval transition.

#### 2026-08-08 14:28:18 CDT — Characterized recording persistence (`109add7`)

- Added behavioral IndexedDB tests before splitting persistence code.
- Covered schema upgrade, recording round trips, capture-session completion, deletion repair, and preferences.

#### 2026-08-08 14:31:01 CDT — Split persistence boundaries (`dc9e451`)

- Separated recording types, database opening/upgrades, recording CRUD, capture-session CRUD, and local preferences.
- Preserved the original `recordings.ts` API as a compatibility facade.

#### 2026-08-08 14:32:55 CDT — Centralized patient/session metadata (`602ddfc`)

- Extracted corrected-age and normalized patient metadata into a dedicated domain module.
- Added tests for premature and term-patient metadata generation.

#### 2026-08-08 14:34:46 CDT — Extracted patient and recording workspace (`a5c2e43`)

- Moved the brand lockup, patient dialog, recording cards, and landing workspace out of the main page.
- Kept the visible application behavior unchanged while reducing the main component’s responsibilities.

#### 2026-08-08 14:36:40 CDT — Extracted sensor acquisition view (`7ee83f0`)

- Moved serial parsing, calibration, sensor graphs, 3D boards, and sensor recording into `SensorView`.
- Added shared capture-view types used by the sensor and video panes.

#### 2026-08-08 14:37:51 CDT — Extracted video tracking view (`0600b41`)

- Moved camera acquisition, pose inference, video recording, overlays, and video-save lifecycle into `VideoView`.
- Reduced the main page to orchestration and synchronized-capture coordination.

#### 2026-08-08 14:38:59 CDT — Characterized Electron policies (`84a89da`)

- Added direct tests for Chromium GPU flags, listener parsing, Teensy USB IDs, serial selection, and application-origin permissions.
- Extracted runtime flags and device policy into focused CommonJS modules.

#### 2026-08-08 14:39:37 CDT — Extracted Electron runtime policy (`16dbc8a`)

- Rewired the Electron entry point to use the characterized runtime and device-policy modules.
- Preserved application ports, permission behavior, serial selection, and process lifecycle.

#### 2026-08-08 14:41:02 CDT — CommonJS lint alignment (`32a4d16`)

- Removed an unused sensor-view import.
- Scoped lint configuration appropriately for CommonJS Electron modules.

### 2026-08-09 — desktop and SAM reliability hardening

#### 2026-08-09 13:52:37 CDT — Full-height desktop workspace (`c473106`)

- Fixed the desktop interface collapsing vertically instead of filling the window.
- Added a layout regression test for the application frame.

#### 2026-08-09 15:40:02 CDT — Protected Electron from SAM GPU pressure (`cf480a6`)

- Added durable raw-first SAM recording state and renderer recovery support.
- Added GPU-memory policy controls and an isolated SAM worker.
- Added renderer crash detection, recovery tests, SAM recovery tests, and an Electron recovery smoke test.
- Ensured a white-screen renderer failure could reload without discarding the raw recording.

#### 2026-08-09 16:19:56 CDT — Reduced SAM VRAM pressure (`de2ff06`)

- Tightened SAM chunk size, GPU memory fraction, CUDA allocation policy, and model-weight precision.
- Added a real full-video smoke script and resource-policy tests.
- Improved status reporting for full-video post-processing.

#### 2026-08-09 16:28:15 CDT — Protected completed recovery results (`17b283f`)

- Prevented a later capture-manifest error from overwriting a successfully recovered annotated recording.
- Added recovery tests for secondary metadata failures.

#### 2026-08-09 16:31:09 CDT — Correct terminal state for SAM failure (`05863b6`)

- Prevented failed SAM annotation from marking a paired sensor/video session complete.
- Preserved raw video while reporting the capture as partial/failed.

#### 2026-08-09 16:37:51 CDT — Cleanup, output, and origin hardening (`2ae8a83`)

- Added bounded job/result retention and temporary-directory cleanup.
- Strengthened MP4 structure and server-side output validation.
- Restricted local-service access to the application origin.
- Added deterministic fake-worker failure tests covering OOM, timeout, cleanup, and subsequent-job recovery.

#### 2026-08-09 16:38:33 CDT — Renderer-recovery retry (`8af68d3`)

- Added bounded retry behavior when the first renderer reload fails.
- Added tests for successful cancellation and failed recovery attempts.

#### 2026-08-09 16:40:06 CDT — Electron teardown smoke test (`d507a32`)

- Added a desktop lifecycle test that closes Electron during active SAM work.
- Verified child Python/FFmpeg processes and service ports are torn down.

#### 2026-08-09 16:44:10 CDT — Full SAM processing inside Electron (`e507445`)

- Added a real desktop-plus-SAM smoke path rather than testing only the standalone service.
- Added runtime switches needed for bounded automated desktop verification.

#### 2026-08-09 16:48:41 CDT — Retried transient decoder startup (`924be13`)

- Added targeted retry logic for temporary OpenCV video-open failures.
- Explicitly avoided retrying corrupt input, model failures, or CUDA OOM.

#### 2026-08-09 16:51:08 CDT — Deterministic SAM video chunks (`6010639`)

- Replaced fragile worker input chunks with FFmpeg-generated H.264/yuv420p chunks.
- Forced deterministic keyframes and repaired invalid browser-recording timing metadata.

#### 2026-08-09 16:52:07 CDT — Accurate source-frame counting (`07ed674`)

- Used FFprobe-decoded frame counts rather than unreliable container metadata.
- Added source/chunk frame-count invariants.

#### 2026-08-09 16:55:36 CDT — Released decoder state before inference (`a2fc7de`)

- Moved chunk validation to short-lived FFprobe processes.
- Prevented long-lived decoder state from interfering with CUDA worker video access.

#### 2026-08-09 17:00:41 CDT — Separated tracking and rendering (`072314c`)

- Completed all CUDA tracking before initializing OpenCV and the final encoder.
- Prevented decoder/encoder allocations from competing with SAM inference.

#### 2026-08-09 17:03:28 CDT — Protected worker input (`0bbb238`)

- Prevented imported SAM helper code from deleting a worker’s input directory.
- Added an isolated import-safety regression test.

#### 2026-08-09 17:28:07 CDT — Annotation-render cleanup (`cad07bf`)

- Hardened encoder termination, pipe closing, failure cleanup, and RLE mask rendering.
- Extended fake-worker fixtures to exercise actual overlay rendering.

#### 2026-08-09 17:56:46 CDT — Truthful cleanup status (`2203dd7`)

- Reported `cleanupComplete: false` when temporary patient-video deletion fails.
- Added an injected cleanup-failure test.

#### 2026-08-09 17:59:13 CDT — Bounded client validation and terminal partial sessions (`a764953`)

- Replaced full-video renderer buffering with 8–16-byte MP4 box-header reads.
- Required valid `ftyp`, `moov`, and `mdat` structure without duplicating the video in memory.
- Marked failed paired SAM captures `partial` rather than leaving them permanently `processing`.

#### 2026-08-09 18:14:54 CDT — Independent four-IMU synthetic benchmark (`8b7f0ac`)

- Added two independently generated synthetic datasets: one for training and another for evaluation.
- Added still/nonperiodic, fine-tremor, and artifact/unusable classes across four sensors.
- Trained and saved a baseline model, validation report, manifests, example CSVs, and release tests.

#### 2026-08-09 21:55:15 CDT — Narrow-window layout repair (`505c6e5`)

- Restored the full BIMA desktop frame when viewed in narrower browser panes.
- Prevented the interface from appearing horizontally or vertically squashed.

#### 2026-08-09 23:07:31 CDT — Windows asset serving repair (`dd47a5b`)

- Added a deterministic patch for Vinext’s Windows asset-path behavior.
- Added a regression test and integrated the patch into install/build/start scripts.

### 2026-08-10 — durable storage and final SAM performance work

#### 2026-08-10 17:22:28 CDT — Configurable durable data archive (`74eac00`)

- Added a desktop folder picker and persisted archive destination.
- Added integrity-checked streaming writes with traversal protection and safe replacement.
- Added analysis-ready sensor CSV export and a data dictionary.
- Connected recordings and synchronized capture sessions to the selected archive.
- Updated the data architecture documentation and added storage/export tests.

#### 2026-08-10 19:24–19:37 CDT — SAM access and persistent-worker optimization (working tree)

- Authorized the exact local preview origins while retaining strict origin matching and rejecting lookalike origins.
- Replaced per-four-frame model reloads with persistent bounded GPU workers.
- Added atomic per-chunk result writes and explicit session cleanup between chunks.
- Prevented worker diagnostic output from filling a pipe and deadlocking on chunk 1.
- Kept four-frame chunks after an eight-frame test produced a measured CUDA OOM on the 8 GB GPU.
- Added persistent-worker characterization and expanded the fake worker to process manifests.
- Validated 61/61 and 296/296 real frames, H.264/yuv420p encoding, full decoding, failure recovery, and the complete application test suite.

### 2026-08-11 — product direction correction

#### 2026-08-11 16:28–16:30 CDT — Removed SAM from the active interface (working tree)

- Removed the Experimental SAM 3.1 button and the entire inference-model selector.
- Changed the empty camera message to lead directly into hand tracking.
- Removed selector-only CSS and the now-unused mode-switch handler.
- Kept the SAM backend, recovery code, and old recording compatibility so existing stored results are not corrupted.
- Updated UI characterization tests to require that no SAM button is rendered.
- Verified production build, TypeScript, UI tests, HTTP 200 desktop startup, and absence of the SAM button in built assets.

## Current system areas

| Area | What exists now | Introduced or substantially revised |
|---|---|---|
| Patient workflow | Patient creation, corrected age, weight normalization, study metadata, recording library | 2026-08-05; refactored 2026-08-08 |
| Four-IMU acquisition | Teensy serial acquisition, fixed sensor identities, timing, calibration, gaps, graphs, and 3D boards | 2026-08-05 |
| Video capture | 1280×720 camera target, 30 FPS request, MediaRecorder capture, synchronized clock | 2026-07-20; expanded 2026-08-05 |
| Active tracking | MediaPipe hand landmarks, raw measurements, stabilized display skeleton | 2026-07-20; expanded 2026-08-05 and 2026-08-08 |
| SAM compatibility | Backend and recovery retained for historical recordings; no active UI entry point | 2026-08-08 through 2026-08-10; UI removed 2026-08-11 |
| Storage | IndexedDB library plus user-selected durable desktop archive and analysis-ready CSV | 2026-08-05; refactored 2026-08-08; archive added 2026-08-10 |
| Sensor modelling | Baseline and independent four-IMU synthetic train/evaluation pipelines | 2026-08-05 and 2026-08-09 |
| Desktop runtime | Electron shell, device permissions, serial selection, recovery, GPU diagnostics, shortcut assets | 2026-08-05; hardened 2026-08-08 and 2026-08-09 |
| Firmware | Four LSM6DSV16X IMUs, Teensy state control, OLED boot animation and logo | 2026-08-05 |
| Verification | Build tests, UI contracts, storage tests, sensor tests, SAM reliability tests, Electron smoke tests | Continuous from 2026-07-20 onward |

## Evidence notes

- Exact timestamps through 2026-08-10 17:22:28 come from Git commit metadata.
- The two working-tree ranges use local modification timestamps and the verified order of work; they are not yet commits.
- The repository contains 39 committed checkpoints in this interval.
- This document intentionally contains no BIMA GitHub configuration, remote, branch, or publication details beyond the phrase “Made GitHub repo.”
