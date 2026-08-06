from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dual_domain_model import DualDomainVitPose, softargmax_2d, training_loss
from pose_data import PoseDataset, load_babypose_records, load_coco_records


class BalancedDomains(Dataset):
    def __init__(self, adult: PoseDataset, infant: PoseDataset, samples_per_epoch: int):
        self.adult = adult
        self.infant = infant
        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, index):
        source = self.adult if index % 2 == 0 else self.infant
        # A large odd multiplier disperses sequential video frames and COCO IDs.
        source_index = ((index // 2) * 7919 + random.randrange(len(source))) % len(source)
        return source[source_index]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--steps-per-epoch", type=int, default=1200)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--adult-val-limit", type=int, default=1800)
    parser.add_argument("--infant-val-limit", type=int, default=0)
    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def validate(model, dataset: PoseDataset, device, batch_size: int):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    correct_05 = correct_10 = correct_15 = total = 0
    router_correct = router_total = 0
    for batch in loader:
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        target = batch["keypoints"].to(device)
        visible = batch["visible"].to(device)
        domain = batch["domain"].to(device)
        output = model(pixels, domain=None)
        predicted = softargmax_2d(output.heatmaps.float(), temperature=0.08)
        torso_values = []
        for shoulder, hip in ((5, 11), (6, 12), (5, 12), (6, 11)):
            pair_visible = visible[:, shoulder] & visible[:, hip]
            distance = torch.linalg.vector_norm(target[:, shoulder] - target[:, hip], dim=-1)
            torso_values.append(torch.where(pair_visible, distance, torch.zeros_like(distance)))
        torso_values = torch.stack(torso_values, dim=-1)
        torso_count = (torso_values > 0).sum(dim=-1)
        torso = torso_values.sum(dim=-1) / torso_count.clamp_min(1)
        valid = visible & (torso_count > 0)[:, None]
        normalized = torch.linalg.vector_norm(predicted - target, dim=-1) / torso[:, None].clamp_min(1e-6)
        total += int(valid.sum())
        correct_05 += int(((normalized <= 0.05) & valid).sum())
        correct_10 += int(((normalized <= 0.10) & valid).sum())
        correct_15 += int(((normalized <= 0.15) & valid).sum())
        router_correct += int((output.routed_domain == domain).sum())
        router_total += domain.numel()
    return {
        "pck@0.05": correct_05 / max(total, 1),
        "pck@0.1": correct_10 / max(total, 1),
        "pck@0.15": correct_15 / max(total, 1),
        "router_accuracy": router_correct / max(router_total, 1),
        "samples": len(dataset),
    }


def save_checkpoint(path: Path, model, optimizer, scheduler, scaler, epoch, best_score, args, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_score": best_score,
        "args": vars(args),
        "metrics": metrics,
    }, path)


def main():
    args = parse_args()
    seed_everything(args.seed)
    root = Path(__file__).resolve().parent
    run_dir = args.run_dir or root / "runs" / time.strftime("dual-domain-%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This experiment requires CUDA; CPU training is intentionally disabled.")
    torch.backends.cudnn.benchmark = True

    print("Loading dataset metadata...")
    adult_train_records = load_coco_records(root / "data" / "coco", "train")
    adult_val_records = load_coco_records(root / "data" / "coco", "val")
    infant_train_records = load_babypose_records(root / "data" / "babypose", "train")
    infant_val_records = load_babypose_records(root / "data" / "babypose", "test")
    if args.adult_val_limit:
        adult_val_records = adult_val_records[: args.adult_val_limit]
    if args.infant_val_limit:
        infant_val_records = infant_val_records[: args.infant_val_limit]
    train_dataset = BalancedDomains(
        PoseDataset(adult_train_records, augment=True),
        PoseDataset(infant_train_records, augment=True),
        args.steps_per_epoch * args.batch_size,
    )
    adult_validation = PoseDataset(adult_val_records, augment=False)
    infant_validation = PoseDataset(infant_val_records, augment=False)
    loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
        drop_last=True,
    )

    model = DualDomainVitPose.from_pretrained(args.resume if args.resume else None)
    model.freeze_for_adaptation()
    model.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    print(f"Trainable parameters: {sum(parameter.numel() for parameter in trainable):,}")
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(loader)
    warmup_steps = max(100, total_steps // 20)

    def lr_factor(step):
        if step < warmup_steps:
            return max(step, 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    start_epoch = 0
    best_score = -1.0
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        if "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
            scheduler.load_state_dict(payload["scheduler"])
            scaler.load_state_dict(payload["scaler"])
            start_epoch = int(payload.get("epoch", -1)) + 1
            best_score = float(payload.get("best_score", -1.0))

    history = []
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running = {}
        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}", unit="batch")
        for batch in progress:
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            keypoints = batch["keypoints"].to(device, non_blocking=True)
            visible = batch["visible"].to(device, non_blocking=True)
            domain = batch["domain"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
                output = model(pixels, domain=domain)
                loss, parts = training_loss(output, keypoints, visible, domain)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            for name, value in parts.items():
                running[name] = 0.96 * running.get(name, float(value)) + 0.04 * float(value)
            progress.set_postfix(loss=f"{running['loss']:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        model.eval()
        adult_metrics = validate(model, adult_validation, device, args.batch_size * 2)
        infant_metrics = validate(model, infant_validation, device, args.batch_size * 2)
        # Harmonic mean prevents a high score in one domain hiding collapse in the other.
        adult_score, infant_score = adult_metrics["pck@0.1"], infant_metrics["pck@0.1"]
        score = 2 * adult_score * infant_score / max(adult_score + infant_score, 1e-9)
        epoch_report = {
            "epoch": epoch + 1, "score": score,
            "adult": adult_metrics, "infant": infant_metrics,
            "loss": running, "lr": scheduler.get_last_lr()[0],
        }
        history.append(epoch_report)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(epoch_report, indent=2))
        save_checkpoint(run_dir / "last.pt", model, optimizer, scheduler, scaler, epoch, max(best_score, score), args, epoch_report)
        if score > best_score:
            best_score = score
            save_checkpoint(run_dir / "best.pt", model, optimizer, scheduler, scaler, epoch, best_score, args, epoch_report)
            print(f"New best checkpoint: {run_dir / 'best.pt'}")

    print(f"Training complete. Best harmonic PCK@0.1: {best_score:.6f}")
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()

