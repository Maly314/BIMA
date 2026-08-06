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

from kinepose_losses import kinepose_loss
from kinepose_model import parameter_report
from pose_data import PoseDataset, load_coco_records
from simplebaseline_model import SimpleBaselineR18
from train_kinepose import deterministic_subset, seed_everything, validate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=0.04)
    parser.add_argument("--train-limit", type=int, default=24000)
    parser.add_argument("--val-limit", type=int, default=1800)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--run-name", default="baseline-simple-r18")
    return parser.parse_args()


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best, args, metrics):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best": best,
            "args": vars(args),
            "metrics": metrics,
            "architecture": {
                "name": "SimpleBaselineR18",
                "deconv_channels": 128,
                **parameter_report(model),
            },
        },
        path,
    )


def main():
    args = parse_args()
    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.backends.cudnn.benchmark = True
    root = Path(__file__).resolve().parent
    run_dir = root / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    train_records = deterministic_subset(
        load_coco_records(root / "data" / "coco", "train"), args.train_limit, args.seed
    )
    val_records = deterministic_subset(
        load_coco_records(root / "data" / "coco", "val"), args.val_limit, args.seed + 1
    )
    train_set = PoseDataset(train_records, augment=True)
    val_set = PoseDataset(val_records, augment=False)
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        drop_last=True,
    )

    device = torch.device("cuda")
    model = SimpleBaselineR18().to(device)
    print(json.dumps(parameter_report(model), indent=2))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
    )
    total_steps = args.epochs * len(loader)
    warmup = max(200, total_steps // 15)

    def schedule(step):
        if step < warmup:
            return max(step, 1) / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.03 + 0.97 * (0.5 + 0.5 * math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler("cuda")
    best = -1.0
    history = []
    for epoch in range(args.epochs):
        model.train()
        running = {}
        started = time.perf_counter()
        progress = tqdm(loader, desc=f"simple-r18 epoch {epoch + 1}/{args.epochs}", unit="batch")
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

        metrics = validate(model, val_set, device, args.batch_size * 2)
        report = {
            "epoch": epoch + 1,
            "seconds": time.perf_counter() - started,
            "train": running,
            "validation": metrics,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(report)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        score = metrics["pck@0.1"]
        save_checkpoint(run_dir / "last.pt", model, optimizer, scheduler, scaler, epoch, max(best, score), args, metrics)
        if score > best:
            best = score
            save_checkpoint(run_dir / "best.pt", model, optimizer, scheduler, scaler, epoch, best, args, metrics)
        print(json.dumps(report, indent=2))
    print(f"Completed {args.run_name}; best PCK@0.1={best:.6f}")


if __name__ == "__main__":
    main()

