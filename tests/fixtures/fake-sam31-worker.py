from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def first_time(marker_name: str) -> bool:
    raw = os.environ.get(marker_name)
    if not raw:
        return False
    marker = Path(raw)
    if marker.exists():
        return False
    marker.write_text("triggered", encoding="utf-8")
    return True


if first_time("BIMA_FAKE_WORKER_FAIL_ONCE"):
    sys.stderr.write("torch.OutOfMemoryError: CUDA out of memory in deterministic test worker\n")
    raise SystemExit(9)

if first_time("BIMA_FAKE_WORKER_SLEEP_ONCE"):
    time.sleep(float(os.environ.get("BIMA_FAKE_WORKER_SLEEP_SECONDS", "5")))

import cv2

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
capture = cv2.VideoCapture(str(source))
frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
capture.release()
frames = {str(index): [] for index in range(frame_count)}
if frame_count:
    # Exercise the real RLE decode/overlay path instead of allowing tests to
    # pass with empty masks that skip all annotation rendering.
    frames["0"] = [{"rle": [0, 1, 57_599], "maskWidth": 320, "maskHeight": 180}]
destination.write_text(json.dumps(frames), encoding="utf-8")
