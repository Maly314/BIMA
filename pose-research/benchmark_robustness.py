from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from benchmark_adult_pose import load_model
from dual_domain_model import softargmax_2d
from kinepose_losses import torso_scale
from kinepose_model import decode_coordinates
from pose_data import COCO_KEYPOINTS, PoseDataset, load_coco_records


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("vitpose", "kinepose", "simplebaseline", "kineres"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def corruptions(images: torch.Tensor):
    yield "clean", images
    generator = torch.Generator(device=images.device).manual_seed(7001)
    noise = torch.randn(images.shape, generator=generator, device=images.device, dtype=images.dtype)
    yield "sensor_noise_2pct", images + noise * 0.02 / 0.225
    yield "sensor_noise_5pct", images + noise * 0.05 / 0.225
    yield "box_blur_5x5", F.avg_pool2d(F.pad(images, (2, 2, 2, 2), mode="reflect"), 5, stride=1)
    occluded = images.clone()
    height, width = images.shape[-2:]
    occluded[:, :, int(height * 0.38):int(height * 0.62), int(width * 0.30):int(width * 0.70)] = 0
    yield "center_occlusion", occluded


def empty_metric(device):
    return {
        "total": torch.zeros((), dtype=torch.long, device=device),
        "correct": torch.zeros((), dtype=torch.long, device=device),
        "error": torch.zeros((), device=device),
        "joint_total": torch.zeros(17, dtype=torch.long, device=device),
        "joint_correct": torch.zeros(17, dtype=torch.long, device=device),
    }


@torch.inference_mode()
def main():
    args = parse_args()
    device = torch.device("cuda")
    root = Path(__file__).resolve().parent
    records = load_coco_records(root / "data" / "coco", "val")
    if args.limit:
        records = records[:args.limit]
    loader = DataLoader(PoseDataset(records, augment=False), batch_size=args.batch_size, num_workers=2, pin_memory=True)
    model, model_info = load_model(SimpleNamespace(model=args.model, checkpoint=args.checkpoint), device)
    metrics = {name: empty_metric(device) for name in ("clean", "sensor_noise_2pct", "sensor_noise_5pct", "box_blur_5x5", "center_occlusion")}

    for batch in tqdm(loader, desc=f"robustness-{args.model}", unit="batch"):
        images = batch["pixel_values"].to(device, non_blocking=True)
        target = batch["keypoints"].to(device, non_blocking=True)
        visible = batch["visible"].to(device, non_blocking=True)
        scale, scale_valid = torso_scale(target, visible)
        valid = visible & scale_valid[:, None]
        for name, altered in corruptions(images):
            with torch.autocast("cuda", dtype=torch.float16):
                if args.model == "vitpose":
                    domain = torch.zeros(altered.shape[0], dtype=torch.long, device=device)
                    heatmaps = model(pixel_values=altered, dataset_index=domain).heatmaps
                    predicted = softargmax_2d(heatmaps.float(), temperature=0.08)
                else:
                    predicted = decode_coordinates(model(altered))
            normalized = torch.linalg.vector_norm(predicted - target, dim=-1) / scale[:, None]
            result = metrics[name]
            correct = (normalized <= 0.10) & valid
            result["total"] += valid.sum()
            result["correct"] += correct.sum()
            result["error"] += normalized[valid].sum()
            result["joint_total"] += valid.sum(dim=0)
            result["joint_correct"] += correct.sum(dim=0)

    report = {"model": model_info, "samples": len(records), "conditions": {}}
    for name, values in metrics.items():
        total = max(int(values["total"]), 1)
        report["conditions"][name] = {
            "pck@0.1": int(values["correct"]) / total,
            "normalized_mean_error": float(values["error"]) / total,
            "pck@0.1_by_joint": {
                joint: int(values["joint_correct"][index]) / max(int(values["joint_total"][index]), 1)
                for index, joint in enumerate(COCO_KEYPOINTS)
            },
        }
    clean = report["conditions"]["clean"]["pck@0.1"]
    for name, values in report["conditions"].items():
        values["absolute_pck_drop"] = clean - values["pck@0.1"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
