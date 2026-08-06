from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from kinepose_losses import kinepose_loss, torso_scale
from kinepose_model import KinePose, decode_coordinates, parameter_report
from pose_data import PoseDataset, load_coco_records


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("full", "no_graph", "no_fields", "plain", "hybrid", "hybrid_sharp", "hybrid_anchor", "hybrid_fast"), default="full")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--graph-depth", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=0.04)
    parser.add_argument("--train-limit", type=int, default=24000)
    parser.add_argument("--val-limit", type=int, default=1800)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def deterministic_subset(records, limit, seed):
    if not limit or limit >= len(records):
        return records
    generator = random.Random(seed)
    indices = generator.sample(range(len(records)), limit)
    return [records[index] for index in indices]


@torch.inference_mode()
def validate(model, dataset, device, batch_size):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    counts = {0.05: 0, 0.075: 0, 0.10: 0, 0.15: 0}
    total = 0
    normalized_sum = 0.0
    calibration_error = 0.0
    calibration_count = 0
    for batch in loader:
        images = batch["pixel_values"].to(device, non_blocking=True)
        target = batch["keypoints"].to(device, non_blocking=True)
        visible = batch["visible"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            output = model(images)
        predicted = decode_coordinates(output)
        scale, valid_scale = torso_scale(target, visible)
        valid = visible & valid_scale[:, None]
        normalized = torch.linalg.vector_norm(predicted - target, dim=-1) / scale[:, None]
        total += int(valid.sum())
        normalized_sum += float(normalized[valid].sum())
        for threshold in counts:
            counts[threshold] += int(((normalized <= threshold) & valid).sum())
        predicted_sigma = torch.exp(0.5 * output.log_variance.float())
        calibration_error += float((predicted_sigma[valid] - normalized[valid]).abs().sum())
        calibration_count += int(valid.sum())
    return {
        "samples": len(dataset),
        "visible_keypoints": total,
        "normalized_mean_error": normalized_sum / max(total, 1),
        **{f"pck@{threshold:g}": counts[threshold] / max(total, 1) for threshold in counts},
        "uncertainty_mae": calibration_error / max(calibration_count, 1),
    }


def checkpoint(path, model, optimizer, scheduler, scaler, epoch, best, args, metrics):
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
        "epoch": epoch, "best": best, "args": vars(args), "metrics": metrics,
        "architecture": {
            "name": "KinePose", "variant": args.variant, "width": args.width,
            "graph_depth": args.graph_depth, **parameter_report(model),
        },
    }, path)


def main():
    args = parse_args()
    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for KinePose training")
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    root = Path(__file__).resolve().parent
    run_name = args.run_name or f"kinepose-{args.variant}-w{args.width}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = root / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    print("Reading COCO keypoint annotations...")
    train_records = load_coco_records(root / "data" / "coco", "train")
    validation_records = load_coco_records(root / "data" / "coco", "val")
    train_records = deterministic_subset(train_records, args.train_limit, args.seed)
    validation_records = deterministic_subset(validation_records, args.val_limit, args.seed + 1)
    print(f"Training crops: {len(train_records):,}; validation crops: {len(validation_records):,}")
    train_set = PoseDataset(train_records, augment=True)
    validation_set = PoseDataset(validation_records, augment=False)
    loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0, drop_last=True,
    )

    model = KinePose(args.width, args.graph_depth, args.variant).to(device)
    print(json.dumps(parameter_report(model), indent=2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    total_steps = args.epochs * len(loader)
    warmup = max(200, total_steps // 15)

    def schedule(step):
        if step < warmup:
            return max(step, 1) / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.03 + 0.97 * (0.5 + 0.5 * math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler("cuda")
    start_epoch = 0
    best = -1.0
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        scaler.load_state_dict(payload["scaler"])
        start_epoch = payload["epoch"] + 1
        best = payload["best"]

    history = []
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running = {}
        started = time.perf_counter()
        progress = tqdm(loader, desc=f"{args.variant} epoch {epoch + 1}/{args.epochs}", unit="batch")
        for batch in progress:
            images = batch["pixel_values"].to(device, non_blocking=True)
            keypoints = batch["keypoints"].to(device, non_blocking=True)
            visible = batch["visible"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                output = model(images)
                loss, parts = kinepose_loss(output, keypoints, visible)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            for name, value in parts.items():
                number = float(value)
                running[name] = 0.97 * running.get(name, number) + 0.03 * number
            progress.set_postfix(loss=f"{running['loss']:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        metrics = validate(model, validation_set, device, args.batch_size * 2)
        report = {
            "epoch": epoch + 1, "seconds": time.perf_counter() - started,
            "train": running, "validation": metrics, "lr": scheduler.get_last_lr()[0],
        }
        history.append(report)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        score = metrics["pck@0.1"]
        checkpoint(run_dir / "last.pt", model, optimizer, scheduler, scaler, epoch, max(best, score), args, metrics)
        if score > best:
            best = score
            checkpoint(run_dir / "best.pt", model, optimizer, scheduler, scaler, epoch, best, args, metrics)
        print(json.dumps(report, indent=2))

    print(f"Completed {run_name}; best PCK@0.1={best:.6f}")


if __name__ == "__main__":
    main()
