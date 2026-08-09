"""Narrow retry policy for transient Windows video-decoder startup failures."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar


T = TypeVar("T")


def retry_transient_video_open(
    operation: Callable[[], T],
    *,
    attempts: int = 3,
    delay_seconds: float = 0.5,
) -> T:
    """Retry only OpenCV's transient `Could not open video` failure.

    Freshly finalized MP4 chunks can remain temporarily unavailable to a new
    decoder process on Windows. Corrupt videos, CUDA failures, and every other
    exception remain immediate hard failures.
    """
    if attempts < 1:
        raise ValueError("attempts must be at least one")
    for attempt in range(attempts):
        try:
            return operation()
        except ValueError as exc:
            if "Could not open video" not in str(exc) or attempt + 1 >= attempts:
                raise
            time.sleep(delay_seconds * (attempt + 1))
    raise AssertionError("unreachable")
