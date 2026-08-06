from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import VitPoseForPoseEstimation

from dual_domain_model import ADULT_EXPERT, BASE_MODEL_ID, DualDomainVitPose, softargmax_2d
from pose_data import (
    HEATMAP_HEIGHT, HEATMAP_WIDTH, INPUT_HEIGHT, INPUT_WIDTH,
    PoseDataset, load_babypose_records, load_coco_records,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("baseline", "checkpoint"), default="baseline")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--domain", choices=("adult", "infant", "both"), default="both")
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates the complete split")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def make_model(args, device):
    if args.model == "baseline":
        model = VitPoseForPoseEstimation.from_pretrained(BASE_MODEL_ID)
    else:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for checkpoint evaluation")
        model = DualDomainVitPose.from_pretrained(args.checkpoint)
    return model.eval().to(device)


def decode_to_original(heatmaps: torch.Tensor, crop: torch.Tensor):
    points = softargmax_2d(heatmaps.float(), temperature=0.08)
    points[:, :, 0] *= INPUT_WIDTH / HEATMAP_WIDTH
    points[:, :, 1] *= INPUT_HEIGHT / HEATMAP_HEIGHT
    points[:, :, 0] = crop[:, 0, None] + points[:, :, 0] * crop[:, 2, None] / INPUT_WIDTH
    points[:, :, 1] = crop[:, 1, None] + points[:, :, 1] * crop[:, 3, None] / INPUT_HEIGHT
    return points


def torso_lengths(points: torch.Tensor, visible: torch.Tensor):
    pairs = ((5, 11), (6, 12), (5, 12), (6, 11))
    values = []
    for shoulder, hip in pairs:
        valid = visible[:, shoulder] & visible[:, hip]
        distance = torch.linalg.vector_norm(points[:, shoulder, :2] - points[:, hip, :2], dim=-1)
        values.append(torch.where(valid, distance, torch.zeros_like(distance)))
    values = torch.stack(values, dim=-1)
    count = (values > 0).sum(dim=-1)
    scale = values.sum(dim=-1) / count.clamp_min(1)
    return scale, count > 0


@torch.inference_mode()
def evaluate(model, dataset, device, batch_size, workers, model_kind):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    thresholds = (0.05, 0.075, 0.10, 0.15)
    correct = {threshold: 0 for threshold in thresholds}
    total = 0
    normalized_error = 0.0
    router_correct = 0
    router_total = 0
    for batch in tqdm(loader, desc="benchmark", unit="batch"):
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        domain = batch["domain"].to(device)
        if model_kind == "baseline":
            indices = torch.full_like(domain, ADULT_EXPERT)
            heatmaps = model(pixel_values=pixels, dataset_index=indices).heatmaps
        else:
            output = model(pixels, domain=None)
            heatmaps = output.heatmaps
            router_correct += int((output.routed_domain == domain).sum())
            router_total += domain.numel()
        predicted = decode_to_original(heatmaps, batch["crop"].to(device))
        target = batch["original_keypoints"].to(device)
        visible = target[:, :, 2] > 0
        scale, scale_valid = torso_lengths(target, visible)
        valid = visible & scale_valid[:, None]
        error = torch.linalg.vector_norm(predicted - target[:, :, :2], dim=-1)
        normalized = error / scale[:, None].clamp_min(1e-6)
        total += int(valid.sum())
        normalized_error += float(normalized[valid].sum())
        for threshold in thresholds:
            correct[threshold] += int(((normalized <= threshold) & valid).sum())
    metrics = {
        "samples": len(dataset),
        "visible_keypoints": total,
        "normalized_mean_error": normalized_error / max(total, 1),
        **{f"pck@{threshold:g}": correct[threshold] / max(total, 1) for threshold in thresholds},
    }
    if router_total:
        metrics["router_accuracy"] = router_correct / router_total
    return metrics


@torch.inference_mode()
def throughput(model, device, model_kind):
    pixels = torch.randn(1, 3, INPUT_HEIGHT, INPUT_WIDTH, device=device)
    domain = torch.zeros(1, dtype=torch.long, device=device)
    for _ in range(10):
        if model_kind == "baseline":
            model(pixel_values=pixels, dataset_index=domain)
        else:
            model(pixels, domain=domain)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    iterations = 60
    for _ in range(iterations):
        if model_kind == "baseline":
            model(pixel_values=pixels, dataset_index=domain)
        else:
            model(pixels, domain=domain)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {"batch1_fps": iterations / elapsed, "batch1_latency_ms": elapsed * 1000 / iterations}


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(args, device)
    report = {
        "model": args.model,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "device": str(device),
        "throughput": throughput(model, device, args.model),
        "domains": {},
    }
    requested = ("adult", "infant") if args.domain == "both" else (args.domain,)
    for domain in requested:
        if domain == "adult":
            records = load_coco_records(root / "data" / "coco", "val")
        else:
            records = load_babypose_records(root / "data" / "babypose", "test")
        if args.limit:
            records = records[: args.limit]
        report["domains"][domain] = evaluate(
            model, PoseDataset(records, augment=False), device,
            args.batch_size, args.workers, args.model,
        )
    output = args.output or root / "runs" / f"benchmark-{args.model}-{int(time.time())}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()

