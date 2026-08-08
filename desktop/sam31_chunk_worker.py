"""Isolated native SAM worker for one bounded video chunk.

The official multiplex runtime retains CUDA allocations after a session closes.
Running one chunk per process guarantees the CUDA context and all VRAM are
released by the OS before the next chunk begins.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from sam31_service import OUTPUT_PROB_THRESH, _load_model, _native_instances


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: sam31_chunk_worker.py INPUT_VIDEO OUTPUT_JSON")
    source = Path(sys.argv[1])
    destination = Path(sys.argv[2])

    import torch

    predictor = _load_model()
    session_id = str(uuid.uuid4())
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        state = predictor.model.init_state(
            resource_path=str(source),
            offload_video_to_cpu=True,
            async_loading_frames=False,
        )
        now = time.time()
        predictor._all_inference_states[session_id] = {
            "state": state,
            "session_id": session_id,
            "start_time": now,
            "last_use_time": now,
        }
        prompted = predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "text": "human hand",
                "output_prob_thresh": OUTPUT_PROB_THRESH,
            }
        )
        tracked = {str(int(prompted["frame_index"])): _native_instances(prompted["outputs"])}
        for response in predictor.handle_stream_request(
            {
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": "forward",
                "output_prob_thresh": OUTPUT_PROB_THRESH,
            }
        ):
            tracked[str(int(response["frame_index"]))] = _native_instances(response["outputs"])
    destination.write_text(json.dumps(tracked), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
