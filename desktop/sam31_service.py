"""Local-only SAM 3.1 inference bridge for the BIMA desktop app.

The service deliberately has no substitute model. If Meta's gated checkpoint is
not available, /load returns a precise setup error and the app stays in the
selected SAM mode without starting MediaPipe.
"""

from __future__ import annotations

import io
import json
import os
# Must be set before PyTorch is imported by the SAM package. Long-running video
# jobs otherwise lose usable VRAM to reserved block fragmentation on Windows.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

HOST = "127.0.0.1"
PORT = int(os.environ.get("BIMA_SAM31_PORT", "4831"))
APP_ORIGIN = os.environ.get("BIMA_APP_ORIGIN", "http://127.0.0.1:4820").rstrip("/")
PIPELINE_VERSION = "sam31-native-v12"
ROOT = Path(__file__).resolve().parents[1]
SERVICE_TEMP_ROOT = Path(os.environ.get("BIMA_SAM31_TEMP_ROOT", Path(tempfile.gettempdir()) / f"bima-sam31-service-{PORT}"))
CHECKPOINT_CANDIDATES = [
    Path(os.environ["BIMA_SAM31_CHECKPOINT"]) if os.environ.get("BIMA_SAM31_CHECKPOINT") else None,
    ROOT / "local-models" / "sam3" / "checkpoints" / "sam3.1_multiplex.pt",
    Path.home() / "Downloads" / "sam3.1_multiplex.pt",
]

_model: Any = None
_model_error = ""
_model_lock = threading.Lock()
_prompt_state_lock = threading.Lock()
_last_boxes: list[list[float]] = []
_tracked_prompt_frames = 0
_last_prompt_monotonic = 0.0
_video_process_lock = threading.Lock()
_video_results: dict[str, dict[str, Any]] = {}
_full_video_jobs: dict[str, dict[str, Any]] = {}
_video_results_lock = threading.Lock()
HAND_LINKS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)
MASK_WIDTH = 320
MASK_HEIGHT = 180
OUTPUT_PROB_THRESH = float(os.environ.get("BIMA_SAM31_OUTPUT_THRESH", "0.5"))
VIDEO_SAM_INTERVAL_SECONDS = float(os.environ.get("BIMA_SAM31_VIDEO_INTERVAL", "1.0"))
FULL_MASK_WIDTH = 640
FULL_MASK_HEIGHT = 360
# The multiplex checkpoint peaks near the full 8 GB capacity even at 640x360.
# Four 30-fps frames is the largest repeatedly verified state that preserves
# enough VRAM for Electron while the official multiplex model is active.
FULL_VIDEO_CHUNK_FRAMES = max(1, int(os.environ.get("BIMA_SAM31_CHUNK_FRAMES", "4")))
GPU_MEMORY_FRACTION = min(0.95, max(0.50, float(os.environ.get("BIMA_SAM31_GPU_MEMORY_FRACTION", "0.75"))))
MODEL_WEIGHT_DTYPE = os.environ.get("BIMA_SAM31_WEIGHT_DTYPE", "backbones-bfloat16").lower()
RESULT_TTL_SECONDS = max(60, int(os.environ.get("BIMA_SAM31_RESULT_TTL_SECONDS", "3600")))
MAX_VIDEO_RESULTS = max(1, int(os.environ.get("BIMA_SAM31_MAX_RESULTS", "8")))
CHUNK_WORKER_PATH = Path(os.environ.get("BIMA_SAM31_CHUNK_WORKER", Path(__file__).with_name("sam31_chunk_worker.py")))
CHUNK_WORKER_TIMEOUT_SECONDS = max(1, int(os.environ.get("BIMA_SAM31_WORKER_TIMEOUT_SECONDS", "600")))
VIDEO_ENCODER = os.environ.get("BIMA_SAM31_VIDEO_ENCODER", "h264_nvenc")


def _remove_video_result(job_id: str) -> None:
    with _video_results_lock:
        stored = _video_results.pop(job_id, None)
        _full_video_jobs.pop(job_id, None)
    if stored:
        _remove_tree_with_retries(stored["directory"])


def _remove_tree_with_retries(directory: Path, attempts: int = 20) -> None:
    for _attempt in range(attempts):
        shutil.rmtree(directory, ignore_errors=True)
        if not directory.exists():
            return
        time.sleep(0.1)
    raise RuntimeError(f"Could not remove SAM temporary data at {directory}")


def _prune_video_results() -> None:
    now = time.time()
    with _video_results_lock:
        ordered = sorted(_video_results.items(), key=lambda item: float(item[1].get("created", 0)))
        expired = [job_id for job_id, stored in ordered if now - float(stored.get("created", 0)) > RESULT_TTL_SECONDS]
        overflow = [job_id for job_id, _stored in ordered[:-MAX_VIDEO_RESULTS]] if len(ordered) > MAX_VIDEO_RESULTS else []
    for job_id in dict.fromkeys([*expired, *overflow]):
        _remove_video_result(job_id)


def _video_encoder_command(ffmpeg: str, width: int, height: int, fps: float, output_path: Path) -> list[str]:
    command = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
        "-r", f"{fps:.6f}", "-i", "-", "-an", "-c:v", VIDEO_ENCODER,
    ]
    if VIDEO_ENCODER == "h264_nvenc":
        command.extend(["-profile:v", "high", "-pix_fmt", "yuv420p", "-preset", "p4", "-tune", "hq", "-rc", "vbr", "-cq", "20", "-b:v", "0"])
    else:
        command.extend(["-profile:v", "high", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "20"])
    command.extend(["-movflags", "+faststart", str(output_path)])
    return command


def _decoded_frame_count(video_path: Path) -> int:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("FFprobe is required to count recorded video frames.")
    probe = subprocess.run(
        [ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames", "-of", "json", str(video_path)],
        capture_output=True,
        timeout=120,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    if probe.returncode != 0:
        raise RuntimeError(f"Recorded video frame-count probe failed: {probe.stderr.decode(errors='replace').strip()}")
    streams = json.loads(probe.stdout or b"{}").get("streams", [])
    frame_count = int((streams[0] if streams else {}).get("nb_read_frames", 0))
    if frame_count < 1:
        raise RuntimeError(f"Recorded video contained no countable frames: {streams!r}")
    return frame_count


def _source_video_timing(video_path: Path) -> tuple[int, float]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("FFprobe is required to inspect recorded video timing.")
    probe = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(video_path)],
        capture_output=True,
        timeout=120,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    if probe.returncode != 0:
        raise RuntimeError(f"Recorded video timing probe failed: {probe.stderr.decode(errors='replace').strip()}")
    frames = json.loads(probe.stdout or b"{}").get("frames", [])
    timestamps = [float(frame["best_effort_timestamp_time"]) for frame in frames if frame.get("best_effort_timestamp_time") is not None]
    frame_count = len(frames)
    if frame_count < 1:
        raise RuntimeError("Recorded video contained no decodable frames.")
    fps = (len(timestamps) - 1) / (timestamps[-1] - timestamps[0]) if len(timestamps) > 1 and timestamps[-1] > timestamps[0] else 30.0
    if not (1 <= fps <= 120):
        fps = 30.0
    return frame_count, fps


def _validate_encoded_video(video_path: Path, expected_frames: int) -> None:
    ffprobe = shutil.which("ffprobe")
    ffmpeg = shutil.which("ffmpeg")
    if not ffprobe or not ffmpeg:
        raise RuntimeError("FFprobe and FFmpeg are required to validate the annotated video.")
    probe = subprocess.run(
        [ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=codec_name,pix_fmt,nb_read_frames", "-of", "json", str(video_path)],
        capture_output=True,
        timeout=120,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    if probe.returncode != 0:
        raise RuntimeError(f"Annotated video probe failed: {probe.stderr.decode(errors='replace').strip()}")
    streams = json.loads(probe.stdout or b"{}").get("streams", [])
    stream = streams[0] if streams else {}
    if stream.get("codec_name") != "h264" or stream.get("pix_fmt") != "yuv420p" or int(stream.get("nb_read_frames", 0)) != expected_frames:
        raise RuntimeError(f"Annotated video integrity mismatch: {stream!r}; expected {expected_frames} frames.")
    decoded = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(video_path), "-f", "null", "-"],
        capture_output=True,
        timeout=120,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    if decoded.returncode != 0:
        raise RuntimeError(f"Annotated video decode failed: {decoded.stderr.decode(errors='replace').strip()}")


def _checkpoint_path() -> Path:
    checkpoint = next((path for path in CHECKPOINT_CANDIDATES if path and path.is_file()), None)
    if checkpoint is None:
        raise RuntimeError(
            "SAM 3.1 checkpoint not found. Expected sam3.1_multiplex.pt in Downloads "
            "or local-models/sam3/checkpoints."
        )
    return checkpoint


def _release_loaded_model() -> None:
    """Release the service's live-inference model before isolated video work."""
    global _model
    with _model_lock:
        loaded = _model
        _model = None
    if loaded is None:
        return
    import gc
    import torch

    del loaded
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _json_safe(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return value


def _load_model() -> Any:
    global _model, _model_error
    with _model_lock:
        if _model is not None:
            return _model
        try:
            import torch
            from sam3.model_builder import build_sam3_predictor
            import sam3.model.decoder as sam_decoder
            from torch.nn.attention import SDPBackend, sdpa_kernel as torch_sdpa_kernel

            if not torch.cuda.is_available():
                raise RuntimeError("SAM 3.1 requires the NVIDIA CUDA runtime on this machine.")
            # SAM 3.1's non-FA3 decoder currently hard-codes FLASH_ATTENTION.
            # Windows CUDA wheels on the target RTX 4060 Ti do not expose that
            # kernel for every propagation tensor shape and abort with "No
            # available kernel". Keep the exact official scaled-dot-product
            # attention operation while allowing PyTorch's efficient or math
            # CUDA kernels for unsupported shapes.
            sam_decoder.sdpa_kernel = lambda _requested: torch_sdpa_kernel(
                [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
            )
            # Flash Attention 3 is optional and is not installed on the BIMA
            # workstation. The official PyTorch attention path preserves the
            # model; it is not a fallback to a different detector.
            _model = build_sam3_predictor(
                checkpoint_path=str(_checkpoint_path()),
                version="sam3.1",
                compile=False,
                warm_up=False,
                # The released multiplex checkpoint is structurally trained
                # for 16 slots. Reducing these values changes tensor shapes;
                # cap displayed hand instances after inference instead.
                max_num_objects=16,
                multiplex_count=16,
                use_fa3=False,
                # The published checkpoint stores complex RoPE tensors as
                # `freqs_cis`; real/imaginary split tensors are only required
                # by the compiled model variant.
                use_rope_real=False,
                async_loading_frames=False,
            )
            if MODEL_WEIGHT_DTYPE == "backbones-bfloat16":
                # The official predictor already computes under BF16 autocast.
                # Store its visual and language feature backbones in that same
                # inference precision while leaving the downstream decoder in
                # FP32. This removes duplicated precision without quantizing or
                # replacing any layer of the official model.
                for backbone in (
                    _model.model.detector.backbone.language_backbone,
                    _model.model.detector.backbone.vision_backbone,
                ):
                    for parameter in backbone.parameters():
                        if parameter.is_floating_point():
                            parameter.data = parameter.data.to(dtype=torch.bfloat16)
            _model_error = ""
            return _model
        except Exception as exc:
            message = str(exc)
            if "403" in message or "gated" in message.lower() or "authorized" in message.lower():
                message = (
                    "This Hugging Face account's Meta SAM 3.1 access request is awaiting approval. "
                    "Check the terms at huggingface.co/facebook/sam3.1; once Meta approves access, restart BIMA."
                )
            _model_error = message
            raise RuntimeError(message) from exc


def _binary_rle(mask: Any) -> list[int]:
    """Encode row-major binary data as alternating zero/one run lengths."""
    flat = mask.reshape(-1).astype("uint8")
    runs: list[int] = []
    expected = 0
    count = 0
    for raw in flat:
        value = 1 if raw else 0
        if value == expected:
            count += 1
        else:
            runs.append(count)
            count = 1
            expected = value
    runs.append(count)
    return runs


def _decode_binary_rle(runs: list[int], width: int, height: int) -> Any:
    """Decode the compact alternating-run representation used by the web UI."""
    import numpy as np

    flat = np.zeros(width * height, dtype=np.uint8)
    cursor = 0
    value = 0
    for raw_count in runs:
        count = max(0, int(raw_count))
        if value and count:
            flat[cursor : min(flat.size, cursor + count)] = 1
        cursor += count
        if cursor >= flat.size:
            break
        value = 1 - value
    return flat.reshape((height, width))


def _mask_instance(mask: Any, instance_id: int) -> dict[str, Any] | None:
    """Create the same normalized, compact instance contract as image inference."""
    import cv2
    import numpy as np
    import torch

    binary = np.asarray(mask, dtype=np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return None
    # Reject tiny optical-flow flecks and retain the dominant connected hand.
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if int(stats[component, cv2.CC_STAT_AREA]) < 24:
        return None
    binary = (labels == component).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    ys, xs = np.nonzero(binary)
    if not len(xs):
        return None
    compact = cv2.resize(binary, (MASK_WIDTH, MASK_HEIGHT), interpolation=cv2.INTER_NEAREST) > 0
    height, width = binary.shape
    return {
        "id": instance_id,
        "bbox": [
            float(xs.min() / width),
            float(ys.min() / height),
            float((xs.max() + 1) / width),
            float((ys.max() + 1) / height),
        ],
        "centroid": [float(xs.mean() / width), float(ys.mean() / height)],
        "maskWidth": MASK_WIDTH,
        "maskHeight": MASK_HEIGHT,
        "rle": _binary_rle(compact),
    }


def _match_masks(previous: list[Any], acquired: list[Any]) -> list[Any]:
    """Keep hand identities stable when SAM changes its multiplex slot ordering."""
    import numpy as np

    if not previous or not acquired:
        return acquired
    remaining = list(range(len(acquired)))
    ordered: list[Any] = []
    for prior in previous:
        if not remaining:
            break
        py, px = np.nonzero(prior)
        if not len(px):
            break
        prior_center = np.array([px.mean(), py.mean()])
        selected = min(
            remaining,
            key=lambda idx: float(
                np.linalg.norm(
                    prior_center
                    - np.array(
                        [
                            np.nonzero(acquired[idx])[1].mean(),
                            np.nonzero(acquired[idx])[0].mean(),
                        ]
                    )
                )
            ),
        )
        ordered.append(acquired[selected])
        remaining.remove(selected)
    ordered.extend(acquired[index] for index in remaining)
    return ordered


def _process_video(video_bytes: bytes) -> dict[str, Any]:
    """Track SAM masks over every frame and render a crisp annotated MP4.

    SAM 3.1 periodically performs semantic re-acquisition. Between those
    authoritative masks, OpenCV DIS optical flow transports the mask at the
    source frame rate. This avoids independent-frame slot jitter and avoids the
    multi-minute cost of full multiplex propagation on the target RTX 4060 Ti.
    """
    import cv2
    import numpy as np

    with _video_process_lock:
        started = time.perf_counter()
        job_id = uuid.uuid4().hex
        job_dir = Path(tempfile.mkdtemp(prefix="live-", dir=SERVICE_TEMP_ROOT))
        source_path = job_dir / "source.webm"
        output_path = job_dir / "tracked.mp4"
        source_path.write_bytes(video_bytes)

        capture = cv2.VideoCapture(str(source_path))
        if not capture.isOpened():
            shutil.rmtree(job_dir, ignore_errors=True)
            raise RuntimeError("Recorded WebM could not be decoded for SAM tracking.")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not (1 <= fps <= 120):
            fps = 30.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        track_width = min(640, width)
        track_height = max(2, int(round(height * track_width / width)) // 2 * 2)
        key_interval = max(1, int(round(fps * VIDEO_SAM_INTERVAL_SECONDS)))

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            capture.release()
            shutil.rmtree(job_dir, ignore_errors=True)
            raise RuntimeError("FFmpeg is required to create the annotated tracking video.")
        encoder = subprocess.Popen(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{width}x{height}",
                "-r",
                f"{fps:.6f}",
                "-i",
                "-",
                "-an",
                "-c:v",
                "h264_nvenc",
                "-profile:v",
                "high",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "p4",
                "-tune",
                "hq",
                "-rc",
                "vbr",
                "-cq",
                "20",
                "-b:v",
                "0",
                "-movflags",
                "+faststart",
                str(output_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )

        previous_gray = None
        masks: list[Any] = []
        frames: list[dict[str, Any]] = []
        sam_keyframes = 0
        sam_inference_ms = 0.0
        motion_tracking_ms = 0.0
        frame_index = 0
        colors = [(193, 210, 36), (102, 209, 255)]  # BGR: teal, amber
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                tracking_frame = cv2.resize(frame, (track_width, track_height), interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(tracking_frame, cv2.COLOR_BGR2GRAY)
                should_acquire = frame_index == 0 or frame_index % key_interval == 0 or not masks
                if should_acquire:
                    encoded, jpeg = cv2.imencode(".jpg", tracking_frame, [cv2.IMWRITE_JPEG_QUALITY, 86])
                    if not encoded:
                        raise RuntimeError("Could not encode a SAM keyframe.")
                    operation_started = time.perf_counter()
                    # Recorded video favors correctness over the live preview's
                    # faster box prompt. Re-ground every keyframe so drift can
                    # never turn a nearby edge into the tracked anatomy.
                    result = _infer(jpeg.tobytes(), force_text=True)
                    sam_inference_ms += (time.perf_counter() - operation_started) * 1000
                    acquired = [
                        cv2.resize(
                            _decode_binary_rle(instance["rle"], instance["maskWidth"], instance["maskHeight"]),
                            (track_width, track_height),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(np.uint8)
                        for instance in result.get("instances", [])
                    ]
                    acquired = [mask for mask in acquired if int(mask.sum()) >= 24]
                    if acquired:
                        masks = _match_masks(masks, acquired)[:2]
                    sam_keyframes += 1
                elif previous_gray is not None and masks:
                    operation_started = time.perf_counter()
                    transported = []
                    for mask in masks:
                        # Track texture inside the segmented hand, then fit a
                        # RANSAC similarity transform. This is much faster than
                        # full-frame dense flow and deliberately keeps the SAM
                        # boundary rigid and crisp between semantic keyframes.
                        points = cv2.goodFeaturesToTrack(
                            previous_gray,
                            mask=(mask * 255).astype(np.uint8),
                            maxCorners=96,
                            qualityLevel=0.01,
                            minDistance=4,
                            blockSize=5,
                        )
                        moved = mask
                        if points is not None and len(points) >= 3:
                            next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                                previous_gray,
                                gray,
                                points,
                                None,
                                winSize=(21, 21),
                                maxLevel=3,
                                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.02),
                            )
                            if next_points is not None and status is not None:
                                valid = status.reshape(-1) > 0
                                if int(valid.sum()) >= 3:
                                    transform, _ = cv2.estimateAffinePartial2D(
                                        points.reshape(-1, 2)[valid],
                                        next_points.reshape(-1, 2)[valid],
                                        method=cv2.RANSAC,
                                        ransacReprojThreshold=2.0,
                                    )
                                    if transform is not None:
                                        moved = cv2.warpAffine(
                                            mask,
                                            transform,
                                            (track_width, track_height),
                                            flags=cv2.INTER_NEAREST,
                                            borderMode=cv2.BORDER_CONSTANT,
                                        )
                        transported.append(moved)
                    masks = transported
                    motion_tracking_ms += (time.perf_counter() - operation_started) * 1000

                instances = []
                display_masks: list[Any] = []
                for index, mask in enumerate(masks[:2]):
                    instance = _mask_instance(mask, index + 1)
                    if not instance:
                        continue
                    instances.append(instance)
                    clean = cv2.resize(
                        _decode_binary_rle(instance["rle"], MASK_WIDTH, MASK_HEIGHT),
                        (width, height),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(np.uint8)
                    display_masks.append(clean)
                    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(frame, contours, -1, colors[index], 3, cv2.LINE_AA)
                    cx = int(instance["centroid"][0] * width)
                    cy = int(instance["centroid"][1] * height)
                    cv2.circle(frame, (cx, cy), 4, colors[index], -1, cv2.LINE_AA)
                frames.append(
                    {
                        "frameIndex": frame_index,
                        "sourceVideoTimeMs": round(frame_index * 1000 / fps, 3),
                        "segments": instances,
                        "source": "sam31" if should_acquire else "optical-flow",
                    }
                )
                if encoder.stdin is None:
                    raise RuntimeError("Annotated video encoder did not start.")
                encoder.stdin.write(frame.tobytes())
                previous_gray = gray
                frame_index += 1
        finally:
            capture.release()
            if encoder.stdin:
                encoder.stdin.close()
            encoder.wait(timeout=60)

        if encoder.returncode != 0 or not output_path.is_file():
            error = encoder.stderr.read().decode("utf-8", errors="replace") if encoder.stderr else ""
            shutil.rmtree(job_dir, ignore_errors=True)
            raise RuntimeError(f"Annotated video encoding failed: {error.strip()}")
        result = {
            "jobId": job_id,
            "frames": frames,
            "frameCount": frame_index,
            "sourceFps": round(fps, 3),
            "samKeyframes": sam_keyframes,
            "samInferenceMs": round(sam_inference_ms),
            "motionTrackingMs": round(motion_tracking_ms),
            "processingMs": round((time.perf_counter() - started) * 1000),
            "videoMimeType": "video/mp4",
        }
        _video_results[job_id] = {"directory": job_dir, "video": output_path, "result": result, "created": time.time()}
        return result


def _native_instances(outputs: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert official SAM 3.1 propagation output without changing its masks."""
    import cv2
    import numpy as np

    masks = np.asarray(_json_safe(outputs.get("out_binary_masks", [])))
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None, ...]
    probs = np.asarray(_json_safe(outputs.get("out_probs", []))).reshape(-1)
    object_ids = np.asarray(_json_safe(outputs.get("out_obj_ids", []))).reshape(-1)
    instances = []
    for index, mask in enumerate(masks):
        binary = np.asarray(mask, dtype=np.uint8)
        ys, xs = np.nonzero(binary)
        if not len(xs):
            continue
        compact = cv2.resize(binary, (FULL_MASK_WIDTH, FULL_MASK_HEIGHT), interpolation=cv2.INTER_NEAREST) > 0
        instances.append(
            {
                "id": int(object_ids[index]) if index < len(object_ids) else index + 1,
                "confidence": float(probs[index]) if index < len(probs) else None,
                "bbox": [
                    float(xs.min() / binary.shape[1]),
                    float(ys.min() / binary.shape[0]),
                    float((xs.max() + 1) / binary.shape[1]),
                    float((ys.max() + 1) / binary.shape[0]),
                ],
                "centroid": [float(xs.mean() / binary.shape[1]), float(ys.mean() / binary.shape[0])],
                "maskWidth": FULL_MASK_WIDTH,
                "maskHeight": FULL_MASK_HEIGHT,
                "rle": _binary_rle(compact),
            }
        )
    return instances


def _run_full_video_job(job_id: str, source_path: Path, job_dir: Path) -> None:
    """Run native SAM propagation in bounded chunks and render one full video.

    The SAM video state retains features for every decoded frame. A normal
    15-second 720p recording was therefore large enough to terminate the
    service process on Windows before Python could report an exception. Each
    two-second chunk below uses the exact native add_prompt/propagate API, but
    releases its state before loading the next chunk.
    """
    job = _full_video_jobs[job_id]
    output_path = job_dir / "tracked.mp4"
    encoder: subprocess.Popen[bytes] | None = None
    failure_message = ""
    try:
        print(f"[sam31-full] job={job_id} version={PIPELINE_VERSION} started bytes={source_path.stat().st_size}", flush=True)
        with _video_process_lock:
            started = time.perf_counter()
            job.update({"phase": "preparing-isolated-workers", "progress": 1})
            # Do not initialize OpenCV in this parent process before the SAM
            # workers run. Its Windows decoder state can prevent a fresh child
            # process from opening an otherwise valid chunk.
            source_frame_count, fps = _source_video_timing(source_path)
            job.update({"phase": "preparing-video", "progress": 2, "frameCount": source_frame_count, "sourceFps": round(fps, 3)})

            ffmpeg = shutil.which("ffmpeg")
            if not ffmpeg:
                raise RuntimeError("FFmpeg is required to prepare and render native SAM video.")
            # Keep enough image detail for finger silhouettes while bounding
            # both VRAM and host RAM. Re-encoding also repairs MediaRecorder's
            # invalid 1000-fps WebM metadata before SAM sees it.
            width, height = 640, 360
            chunks_dir = job_dir / "chunks"
            chunks_dir.mkdir()
            # Build exact frame-count H.264 chunks. OpenCV's MP4V writer can
            # produce files that its parent process reads but a fresh Windows
            # worker cannot open after CUDA/model initialization. FFmpeg gives
            # each four-frame chunk a deterministic keyframe and portable
            # yuv420p container while correcting MediaRecorder's bogus 1000-fps
            # metadata.
            chunk_pattern = chunks_dir / "chunk-%05d.mp4"
            split_frames = ",".join(str(index) for index in range(FULL_VIDEO_CHUNK_FRAMES, source_frame_count, FULL_VIDEO_CHUNK_FRAMES))
            chunk_command = [
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(source_path),
                "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black",
                "-r", f"{fps:.6f}", "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
                "-pix_fmt", "yuv420p", "-g", str(FULL_VIDEO_CHUNK_FRAMES),
                "-keyint_min", str(FULL_VIDEO_CHUNK_FRAMES), "-sc_threshold", "0",
                "-force_key_frames", f"expr:gte(n,n_forced*{FULL_VIDEO_CHUNK_FRAMES})",
            ]
            if split_frames:
                chunk_command.extend(["-f", "segment", "-segment_frames", split_frames, "-reset_timestamps", "1", str(chunk_pattern)])
            else:
                chunk_command.append(str(chunks_dir / "chunk-00000.mp4"))
            prepared = subprocess.run(
                chunk_command,
                capture_output=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            if prepared.returncode != 0:
                details = prepared.stderr.decode(errors="replace").strip()
                raise RuntimeError(f"Could not create deterministic SAM video chunks: {details[-2000:]}")
            chunk_paths = sorted(chunks_dir.glob("chunk-*.mp4"))
            if not chunk_paths:
                raise RuntimeError("Recorded video contained no decodable frames.")
            # Do not pre-open every chunk through OpenCV in this long-lived
            # parent process. On Windows those decoder contexts can remain
            # allocated after release and prevent the isolated SAM worker from
            # opening its first file. FFprobe exits after each count and returns
            # all decoder resources to the OS.
            chunk_counts = [_decoded_frame_count(chunk_path) for chunk_path in chunk_paths]
            if any(count < 1 or count > FULL_VIDEO_CHUNK_FRAMES for count in chunk_counts):
                raise RuntimeError(f"Internal chunk-size invariant failed: expected 1-{FULL_VIDEO_CHUNK_FRAMES} frames, got {chunk_counts}.")
            frame_count = sum(chunk_counts)
            if frame_count != source_frame_count:
                raise RuntimeError(f"Internal chunk frame-count mismatch: source={source_frame_count}, chunks={frame_count}.")
            job.update({"phase": "prepared-video", "progress": 5, "frameCount": frame_count, "chunkCount": len(chunk_paths)})

            # `/load` may have populated the parent service for the live
            # preview. Video processing is offline, so release that model before
            # launching a worker; otherwise both processes compete for 8 GB.
            _release_loaded_model()

            processed = 0
            chunk_result_paths: list[Path] = []
            for chunk_number, (chunk_path, expected_chunk_frames) in enumerate(zip(chunk_paths, chunk_counts), start=1):
                job.update({"phase": f"tracking-chunk-{chunk_number}-of-{len(chunk_paths)}", "chunk": chunk_number})
                chunk_result_path = job_dir / f"chunk-{chunk_number:05d}-masks.json"
                try:
                    worker = subprocess.run(
                        [sys.executable, str(CHUNK_WORKER_PATH), str(chunk_path), str(chunk_result_path)],
                        capture_output=True,
                        timeout=CHUNK_WORKER_TIMEOUT_SECONDS,
                        env={
                            **os.environ,
                            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                            "CUDA_MODULE_LOADING": "LAZY",
                            "BIMA_SAM31_GPU_MEMORY_FRACTION": str(GPU_MEMORY_FRACTION),
                        },
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                    )
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError(f"SAM chunk {chunk_number}/{len(chunk_paths)} timed out after {CHUNK_WORKER_TIMEOUT_SECONDS} seconds and was terminated.") from exc
                if worker.returncode != 0 or not chunk_result_path.is_file():
                    details = worker.stderr.decode(errors="replace").strip()
                    raise RuntimeError(f"SAM chunk {chunk_number}/{len(chunk_paths)} failed in its isolated GPU worker: {details[-3000:]}")
                tracked = {int(index): instances for index, instances in json.loads(chunk_result_path.read_text(encoding="utf-8")).items()}
                if len(tracked) != expected_chunk_frames:
                    raise RuntimeError(f"SAM chunk {chunk_number} returned incomplete tracking ({len(tracked)}/{expected_chunk_frames} frames).")
                chunk_result_paths.append(chunk_result_path)
                processed += len(tracked)
                job.update({"processedFrames": processed, "progress": min(84, 6 + round(78 * processed / max(1, frame_count)))})
                print(f"[sam31-full] job={job_id} chunk={chunk_number}/{len(chunk_paths)} tracking complete", flush=True)

            # Only after every CUDA worker has exited do we initialize OpenCV
            # and the final encoder in the parent process. This prevents parent
            # decoder/encoder contexts from leaking into the tracking phase.
            import cv2
            import numpy as np

            output_path = job_dir / "tracked.mp4"
            encoder = subprocess.Popen(
                _video_encoder_command(ffmpeg, width, height, fps, output_path),
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            colors = [(193, 210, 36), (102, 209, 255), (255, 128, 128), (180, 120, 255)]
            rendered = 0
            sidecar_frames = []
            for chunk_number, (chunk_path, expected_chunk_frames, chunk_result_path) in enumerate(zip(chunk_paths, chunk_counts, chunk_result_paths), start=1):
                job.update({"phase": f"rendering-chunk-{chunk_number}-of-{len(chunk_paths)}", "progress": min(96, 85 + round(11 * rendered / max(1, frame_count)))})
                tracked = {int(index): instances for index, instances in json.loads(chunk_result_path.read_text(encoding="utf-8")).items()}
                capture = cv2.VideoCapture(str(chunk_path))
                local_index = 0
                try:
                    if not capture.isOpened():
                        raise RuntimeError(f"Video chunk {chunk_number} could not be reopened for rendering.")
                    while True:
                        ok, image = capture.read()
                        if not ok:
                            break
                        instances = tracked.get(local_index, [])
                        for instance_index, instance in enumerate(instances):
                            mask = cv2.resize(_decode_binary_rle(instance["rle"], instance["maskWidth"], instance["maskHeight"]), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
                            color = colors[instance_index % len(colors)]
                            overlay = image.copy(); overlay[mask] = color
                            image = cv2.addWeighted(overlay, 0.34, image, 0.66, 0)
                            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(image, contours, -1, color, 2, cv2.LINE_AA)
                        sidecar_frames.append(
                            {
                                "frameIndex": rendered,
                                "sourceVideoTimeMs": round(rendered * 1000 / fps, 3),
                                "segments": instances,
                                "source": "sam31-native-propagation",
                            }
                        )
                        if encoder.stdin is None:
                            raise RuntimeError("Native SAM video encoder did not start.")
                        encoder.stdin.write(image.tobytes())
                        rendered += 1
                        local_index += 1
                finally:
                    capture.release()
                if local_index != expected_chunk_frames:
                    raise RuntimeError(f"Video chunk {chunk_number} decoded incompletely ({local_index}/{expected_chunk_frames} frames).")
                print(f"[sam31-full] job={job_id} chunk={chunk_number}/{len(chunk_paths)} frames={local_index} rendered", flush=True)

            job.update({"phase": "finalizing-video", "progress": 97})
            if encoder.stdin:
                encoder.stdin.close()
            encoder.wait(timeout=120)
            if encoder.returncode != 0 or not output_path.is_file():
                error = encoder.stderr.read().decode("utf-8", errors="replace") if encoder.stderr else ""
                raise RuntimeError(f"Native SAM video encoding failed: {error.strip()}")
            _validate_encoded_video(output_path, rendered)
            result = {
                "jobId": job_id,
                "pipelineVersion": PIPELINE_VERSION,
                "frames": sidecar_frames,
                "frameCount": rendered,
                "sourceFps": round(fps, 3),
                "processingMs": round((time.perf_counter() - started) * 1000),
                "videoMimeType": "video/mp4",
                "inferenceBackend": "Meta SAM 3.1 native propagate_in_video",
            }
            with _video_results_lock:
                _video_results[job_id] = {"directory": job_dir, "video": output_path, "result": result, "created": time.time()}
            job.update({"phase": "complete", "progress": 100, "status": "complete"})
            _prune_video_results()
            print(f"[sam31-full] job={job_id} complete frames={rendered} processingMs={result['processingMs']}", flush=True)
    except Exception as exc:
        failure_message = str(exc)
        job.update({"phase": "cleaning-up", "status": "running"})
        print(f"[sam31-full] job={job_id} failed error={exc}", flush=True)
        traceback.print_exc()
    finally:
        if encoder is not None and encoder.poll() is None:
            try:
                if encoder.stdin and not encoder.stdin.closed:
                    encoder.stdin.close()
            except OSError:
                pass
            encoder.terminate()
            try:
                encoder.wait(timeout=5)
            except subprocess.TimeoutExpired:
                encoder.kill()
                encoder.wait(timeout=5)
        if encoder is not None:
            for pipe in (encoder.stdin, encoder.stdout, encoder.stderr):
                if pipe and not pipe.closed:
                    pipe.close()
        if job.get("status") != "complete":
            try:
                _remove_tree_with_retries(job_dir)
            except Exception as cleanup_error:
                failure_message = f"{failure_message}; temporary-data cleanup failed: {cleanup_error}" if failure_message else str(cleanup_error)
        if failure_message:
            job.update({"phase": "failed", "status": "failed", "error": failure_message, "cleanupComplete": True})


def _start_full_video_job(video_bytes: bytes) -> dict[str, Any]:
    job_id = uuid.uuid4().hex
    _prune_video_results()
    job_dir = Path(tempfile.mkdtemp(prefix="full-", dir=SERVICE_TEMP_ROOT))
    source_path = job_dir / "source.webm"
    source_path.write_bytes(video_bytes)
    _full_video_jobs[job_id] = {
        "jobId": job_id,
        "pipelineVersion": PIPELINE_VERSION,
        "status": "running",
        "phase": "queued",
        "progress": 0,
        "created": time.time(),
    }
    threading.Thread(target=_run_full_video_job, args=(job_id, source_path, job_dir), daemon=True).start()
    return {"jobId": job_id, "status": "running", "pipelineVersion": PIPELINE_VERSION}


def _render_landmark_video(video_bytes: bytes, landmark_payload: dict[str, Any]) -> dict[str, Any]:
    """Render real 21-point hand kinematics into a compatible MP4."""
    import bisect
    import cv2

    job_id = uuid.uuid4().hex
    job_dir = Path(tempfile.mkdtemp(prefix="hands-", dir=SERVICE_TEMP_ROOT))
    source_path = job_dir / "source.webm"
    output_path = job_dir / "tracked.mp4"
    source_path.write_bytes(video_bytes)
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        shutil.rmtree(job_dir, ignore_errors=True)
        raise RuntimeError("Recorded video could not be decoded for landmark rendering.")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not (1 <= fps <= 120):
        fps = 30.0
    frames = sorted(landmark_payload.get("frames", []), key=lambda frame: float(frame.get("sourceVideoTimeMs", 0)))
    times = [float(frame.get("sourceVideoTimeMs", 0)) for frame in frames]
    if not frames:
        capture.release()
        shutil.rmtree(job_dir, ignore_errors=True)
        raise RuntimeError("No hand landmarks were supplied for rendering.")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        capture.release()
        shutil.rmtree(job_dir, ignore_errors=True)
        raise RuntimeError("FFmpeg is required to render the hand-tracking video.")
    encoder = subprocess.Popen(
        [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
            "-r", f"{fps:.6f}", "-i", "-", "-an", "-c:v", "h264_nvenc",
            "-profile:v", "high", "-pix_fmt", "yuv420p", "-preset", "p4",
            "-tune", "hq", "-rc", "vbr", "-cq", "20", "-b:v", "0",
            "-movflags", "+faststart", str(output_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    colors = [(193, 210, 36), (102, 209, 255)]
    frame_index = 0
    started = time.perf_counter()
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            media_time_ms = frame_index * 1000 / fps
            position = bisect.bisect_left(times, media_time_ms)
            candidates = [index for index in (position - 1, position) if 0 <= index < len(frames)]
            selected = min(candidates, key=lambda index: abs(times[index] - media_time_ms)) if candidates else 0
            frame = frames[selected]
            # The raw measurements remain in JSON. A separately supplied
            # displayHands field may be adaptively stabilized for inspection.
            hands = frame.get("displayHands") or frame.get("hands") or []
            for hand_index, hand in enumerate(hands[:2]):
                if len(hand) < 21:
                    continue
                points = [
                    (
                        int(max(0, min(1, float(point.get("x", 0)))) * width),
                        int(max(0, min(1, float(point.get("y", 0)))) * height),
                    )
                    for point in hand
                ]
                color = colors[hand_index]
                for start, end in HAND_LINKS:
                    cv2.line(image, points[start], points[end], color, 3, cv2.LINE_AA)
                for point_index, point in enumerate(points):
                    radius = 5 if point_index == 0 else 3
                    cv2.circle(image, point, radius, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(image, point, radius, color, 1, cv2.LINE_AA)
            if encoder.stdin is None:
                raise RuntimeError("Landmark video encoder did not start.")
            encoder.stdin.write(image.tobytes())
            frame_index += 1
    finally:
        capture.release()
        if encoder.stdin:
            encoder.stdin.close()
        encoder.wait(timeout=60)
    if encoder.returncode != 0 or not output_path.is_file():
        error = encoder.stderr.read().decode("utf-8", errors="replace") if encoder.stderr else ""
        shutil.rmtree(job_dir, ignore_errors=True)
        raise RuntimeError(f"Landmark video encoding failed: {error.strip()}")
    result = {
        "jobId": job_id,
        "frameCount": frame_index,
        "landmarkFrameCount": len(frames),
        "sourceFps": round(fps, 3),
        "processingMs": round((time.perf_counter() - started) * 1000),
        "videoMimeType": "video/mp4",
    }
    _video_results[job_id] = {"directory": job_dir, "video": output_path, "result": result, "created": time.time()}
    return result


def _infer(jpeg: bytes, force_text: bool = False) -> dict[str, Any]:
    global _last_boxes, _tracked_prompt_frames, _last_prompt_monotonic
    import numpy as np
    from PIL import Image

    predictor = _load_model()
    image = Image.open(io.BytesIO(jpeg)).convert("RGB")
    started = time.perf_counter()
    # SAM 3.1's released multiplex model does not accept the
    # `offload_state_to_cpu` keyword that its shared predictor wrapper always
    # forwards. Start the official model state directly until Meta reconciles
    # those two public APIs; all prompting and cleanup still use the predictor.
    session_id = str(uuid.uuid4())
    inference_state = predictor.model.init_state(
        resource_path=[image],
        offload_video_to_cpu=False,
        async_loading_frames=predictor.async_loading_frames,
    )
    now = time.time()
    predictor._all_inference_states[session_id] = {
        "state": inference_state,
        "session_id": session_id,
        "start_time": now,
        "last_use_time": now,
    }
    try:
        with _prompt_state_lock:
            # A text grounding pass is needed to acquire the first hand. For
            # subsequent frames, feed SAM its last box as a native prompt. This
            # keeps the segmentation anchored to the hand instead of asking
            # the detector to re-ground the entire room every frame. Re-ground
            # periodically so a hand that leaves/re-enters the view can be
            # reacquired without another model or pose estimator.
            now_monotonic = time.monotonic()
            use_box = (
                not force_text
                and bool(_last_boxes)
                and _tracked_prompt_frames < 8
                and now_monotonic - _last_prompt_monotonic < 2.5
            )
            tracked_boxes = [box[:] for box in _last_boxes]
            prompt_kind = "tracked-box" if use_box else "human hand"
            if use_box:
                prompt = {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": 0,
                    "bounding_boxes": tracked_boxes,
                    "bounding_box_labels": [1] * len(tracked_boxes),
                    "output_prob_thresh": OUTPUT_PROB_THRESH,
                }
            else:
                prompt = {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": 0,
                    "text": "human hand",
                    "output_prob_thresh": OUTPUT_PROB_THRESH,
                }
            result = predictor.handle_request(prompt)["outputs"]
            _last_prompt_monotonic = now_monotonic
        raw_masks = _json_safe(result.get("out_binary_masks", []))
        raw_probs = np.asarray(_json_safe(result.get("out_probs", []))).reshape(-1)
        masks = np.asarray(raw_masks)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        if masks.ndim == 2:
            masks = masks[None, ...]
        instances = []
        ranked_indices = sorted(range(len(masks)), key=lambda idx: float(raw_probs[idx]) if idx < len(raw_probs) else 0.0, reverse=True)
        for index in ranked_indices[:2]:
            mask = masks[index]
            mask = np.asarray(mask, dtype=np.uint8)
            ys, xs = np.nonzero(mask)
            if not len(xs):
                continue
            # A compact mask keeps IPC and main-thread drawing bounded. Nearest
            # neighbour preserves the model boundary without smoothing it.
            compact = Image.fromarray(mask * 255).resize((MASK_WIDTH, MASK_HEIGHT), Image.Resampling.NEAREST)
            compact_mask = np.asarray(compact) > 0
            instances.append(
                {
                    "id": index + 1,
                    "confidence": float(raw_probs[index]) if index < len(raw_probs) else None,
                    "bbox": [
                        float(xs.min() / mask.shape[1]),
                        float(ys.min() / mask.shape[0]),
                        float((xs.max() + 1) / mask.shape[1]),
                        float((ys.max() + 1) / mask.shape[0]),
                    ],
                    "centroid": [float(xs.mean() / mask.shape[1]), float(ys.mean() / mask.shape[0])],
                    "maskWidth": MASK_WIDTH,
                    "maskHeight": MASK_HEIGHT,
                    "rle": _binary_rle(compact_mask),
                }
            )
        with _prompt_state_lock:
            _last_boxes = [
                [box[0], box[1], box[2] - box[0], box[3] - box[1]]
                for box in (instance["bbox"] for instance in instances)
            ]
            _tracked_prompt_frames = _tracked_prompt_frames + 1 if instances and use_box else (1 if instances else 0)
        return {
            "instances": instances,
            "inferenceMs": round((time.perf_counter() - started) * 1000, 1),
            "model": "Meta SAM 3.1",
            "prompt": prompt_kind,
        }
    finally:
        # The next frame reuses the same loaded model immediately. The official
        # close path defaults to gc.collect()+cuda.empty_cache(), which was
        # flushing ~3.3 GB after every frame and adding a visible stall. The
        # session state is still explicitly cleared by close_session; leave
        # PyTorch's allocator warm so the next request can reuse those blocks.
        predictor.handle_request(
            {
                "type": "close_session",
                "session_id": session_id,
                "run_gc_collect": False,
            }
        )


class Handler(BaseHTTPRequestHandler):
    server_version = "BIMA-SAM31/1"

    def _origin_allowed(self) -> bool:
        origin = self.headers.get("Origin")
        return origin is None or origin.rstrip("/") == APP_ORIGIN

    def _headers(self, status: int = 200, content_type: str = "application/json", content_length: int | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        if content_length is not None:
            self.send_header("Content-Length", str(content_length))
        self.send_header("Access-Control-Allow-Origin", APP_ORIGIN)
        self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-BIMA-Landmarks-Length")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def _reply(self, payload: dict[str, Any], status: int = 200) -> None:
        self._headers(status)
        self.wfile.write(json.dumps(payload, separators=(",", ":")).encode("utf-8"))

    def do_OPTIONS(self) -> None:  # noqa: N802
        if not self._origin_allowed():
            self._reply({"error": "origin not allowed"}, 403)
            return
        self._headers(204)

    def do_GET(self) -> None:  # noqa: N802
        if not self._origin_allowed():
            self._reply({"error": "origin not allowed"}, 403)
            return
        _prune_video_results()
        if self.path.startswith("/result/") and self.path.endswith("/status"):
            job_id = self.path.split("/")[2]
            job = _full_video_jobs.get(job_id)
            if not job:
                self._reply({"error": "native SAM job not found"}, 404)
                return
            self._reply({key: value for key, value in job.items() if key != "result"})
            return
        if self.path.startswith("/result/") and self.path.endswith("/metadata"):
            job_id = self.path.split("/")[2]
            stored = _video_results.get(job_id)
            if not stored:
                self._reply({"error": "native SAM metadata not found"}, 404)
                return
            self._reply(stored["result"])
            return
        if self.path.startswith("/result/") and self.path.endswith("/video"):
            job_id = self.path.split("/")[2]
            stored = _video_results.get(job_id)
            video_path = stored.get("video") if stored else None
            if not isinstance(video_path, Path) or not video_path.is_file():
                self._reply({"error": "video result not found"}, 404)
                return
            size = video_path.stat().st_size
            self._headers(200, "video/mp4", size)
            with video_path.open("rb") as source:
                shutil.copyfileobj(source, self.wfile)
            return
        if self.path != "/health":
            self._reply({"error": "not found"}, 404)
            return
        self._reply(
            {
                "service": "ready",
                "pipelineVersion": PIPELINE_VERSION,
                "model": "ready" if _model is not None else "not-loaded",
                "error": _model_error or None,
                "runtime": "official facebookresearch/sam3",
                "resourcePolicy": {
                    "chunkFrames": FULL_VIDEO_CHUNK_FRAMES,
                    "gpuMemoryFraction": GPU_MEMORY_FRACTION,
                    "modelWeightDtype": MODEL_WEIGHT_DTYPE,
                    "cudaModuleLoading": os.environ["CUDA_MODULE_LOADING"],
                },
                "checkpoint": str(_checkpoint_path()) if any(path and path.is_file() for path in CHECKPOINT_CANDIDATES) else None,
            }
        )

    def do_POST(self) -> None:  # noqa: N802
        try:
            if not self._origin_allowed():
                self._reply({"error": "origin not allowed"}, 403)
                return
            if self.path.startswith("/result/") and self.path.endswith("/ack"):
                job_id = self.path.split("/")[2]
                if job_id not in _full_video_jobs and job_id not in _video_results:
                    self._reply({"error": "native SAM job not found"}, 404)
                    return
                _remove_video_result(job_id)
                self._reply({"jobId": job_id, "status": "released"})
                return
            if self.path == "/load":
                _load_model()
                self._reply({"model": "ready", "name": "Meta SAM 3.1", "pipelineVersion": PIPELINE_VERSION})
                return
            if self.path == "/process-video":
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 1_000_000_000:
                    self._reply({"error": "invalid video payload"}, 400)
                    return
                self._reply(_process_video(self.rfile.read(length)))
                return
            if self.path == "/process-video-full":
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 1_000_000_000:
                    self._reply({"error": "invalid video payload"}, 400)
                    return
                self._reply(_start_full_video_job(self.rfile.read(length)), 202)
                return
            if self.path == "/render-landmarks":
                length = int(self.headers.get("Content-Length", "0"))
                metadata_length = int(self.headers.get("X-BIMA-Landmarks-Length", "0"))
                if length <= 0 or length > 1_000_000_000 or metadata_length <= 0 or metadata_length >= length:
                    self._reply({"error": "invalid landmark video payload"}, 400)
                    return
                body = self.rfile.read(length)
                metadata = json.loads(body[:metadata_length].decode("utf-8"))
                self._reply(_render_landmark_video(body[metadata_length:], metadata))
                return
            if self.path != "/infer":
                self._reply({"error": "not found"}, 404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 2_000_000:
                self._reply({"error": "invalid frame payload"}, 400)
                return
            self._reply(_infer(self.rfile.read(length)))
        except Exception as exc:
            self._reply({"error": str(exc)}, 503)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[sam31] {self.address_string()} {fmt % args}", flush=True)


if __name__ == "__main__":
    os.chdir(ROOT)
    # Startup cleanup belongs here rather than at module import. Isolated chunk
    # workers import helpers from this module and may receive an input file
    # inside the same service temp root; import-time cleanup would delete the
    # worker's own video before inference begins.
    shutil.rmtree(SERVICE_TEMP_ROOT, ignore_errors=True)
    SERVICE_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[sam31] service listening on http://{HOST}:{PORT}", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
