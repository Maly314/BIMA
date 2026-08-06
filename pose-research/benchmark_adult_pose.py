from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import VitPoseForPoseEstimation

from dual_domain_model import BASE_MODEL_ID, softargmax_2d
from kinepose_losses import torso_scale
from kinepose_model import KinePose, decode_coordinates, parameter_report
from kineres_model import KineResPose
from pose_data import COCO_KEYPOINTS, HEATMAP_HEIGHT, HEATMAP_WIDTH, INPUT_HEIGHT, INPUT_WIDTH, PoseDataset, load_coco_records
from simplebaseline_model import SimpleBaselineR18


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("vitpose", "kinepose", "simplebaseline", "kineres"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_model(args, device):
    if args.model == "vitpose":
        model = VitPoseForPoseEstimation.from_pretrained(BASE_MODEL_ID).to(device).eval()
        info = {"name": BASE_MODEL_ID, "parameters": sum(p.numel() for p in model.parameters())}
        return model, info
    if args.model == "simplebaseline":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for SimpleBaselineR18")
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model = SimpleBaselineR18(
            deconv_channels=payload["architecture"].get("deconv_channels", 128)
        )
        model.load_state_dict(payload["model"])
        return model.to(device).eval(), {
            **payload["architecture"],
            **parameter_report(model),
        }
    if args.model == "kineres":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for KineResPose")
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        architecture = payload["architecture"]
        model = KineResPose(
            graph_depth=architecture["graph_depth"],
            decoder_channels=architecture["decoder_channels"],
            variant=architecture["variant"],
        )
        model.load_state_dict(payload["model"])
        return model.to(device).eval(), {**architecture, **parameter_report(model)}
    if not args.checkpoint:
        raise ValueError("--checkpoint is required for KinePose")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    architecture = payload["architecture"]
    model = KinePose(
        width=architecture["width"], graph_depth=architecture["graph_depth"],
        variant=architecture["variant"],
    )
    model.load_state_dict(payload["model"])
    return model.to(device).eval(), {"name": "KinePose", **architecture, **parameter_report(model)}


def original_coordinates(points, crop):
    result = points.clone()
    result[:, :, 0] *= INPUT_WIDTH / HEATMAP_WIDTH
    result[:, :, 1] *= INPUT_HEIGHT / HEATMAP_HEIGHT
    result[:, :, 0] = crop[:, 0, None] + result[:, :, 0] * crop[:, 2, None] / INPUT_WIDTH
    result[:, :, 1] = crop[:, 1, None] + result[:, :, 1] * crop[:, 3, None] / INPUT_HEIGHT
    return result


@torch.inference_mode()
def inference(model, model_kind, loader, device, annotation_to_image):
    coco_results = []
    pck_counts = {0.05: 0, 0.075: 0, 0.10: 0, 0.15: 0}
    pck_total = 0
    normalized_sum = 0.0
    joint_total = torch.zeros(17, dtype=torch.long, device=device)
    joint_correct = torch.zeros(17, dtype=torch.long, device=device)
    image_ids = set()
    for batch in tqdm(loader, desc=model_kind, unit="batch"):
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        crop = batch["crop"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            if model_kind == "vitpose":
                domain = torch.zeros(pixels.shape[0], dtype=torch.long, device=device)
                heatmaps = model(pixel_values=pixels, dataset_index=domain).heatmaps
                points = softargmax_2d(heatmaps.float(), temperature=0.08)
            else:
                output = model(pixels)
                heatmaps = output.heatmaps
                points = decode_coordinates(output)
        predicted = original_coordinates(points, crop)
        scores = heatmaps.float().sigmoid().flatten(2).amax(dim=-1)
        target = batch["original_keypoints"].to(device)
        visible = target[:, :, 2] > 0
        scale, scale_valid = torso_scale(target[:, :, :2], visible)
        valid = visible & scale_valid[:, None]
        normalized = torch.linalg.vector_norm(predicted - target[:, :, :2], dim=-1) / scale[:, None]
        pck_total += int(valid.sum())
        normalized_sum += float(normalized[valid].sum())
        joint_total += valid.sum(dim=0)
        joint_correct += ((normalized <= 0.10) & valid).sum(dim=0)
        for threshold in pck_counts:
            pck_counts[threshold] += int(((normalized <= threshold) & valid).sum())

        for index, sample_id in enumerate(batch["sample_id"]):
            annotation_id = int(sample_id.split("-")[-1])
            image_id = annotation_to_image[annotation_id]
            image_ids.add(image_id)
            keypoints = []
            for joint in range(17):
                keypoints.extend((float(predicted[index, joint, 0]), float(predicted[index, joint, 1]), float(scores[index, joint])))
            coco_results.append({
                "image_id": image_id, "category_id": 1, "keypoints": keypoints,
                "score": float(scores[index].mean()),
            })
    pck = {
        "visible_keypoints": pck_total,
        "normalized_mean_error": normalized_sum / max(pck_total, 1),
        **{f"pck@{threshold:g}": pck_counts[threshold] / max(pck_total, 1) for threshold in pck_counts},
        "pck@0.1_by_joint": {
            name: int(joint_correct[index]) / max(int(joint_total[index]), 1)
            for index, name in enumerate(COCO_KEYPOINTS)
        },
    }
    return coco_results, sorted(image_ids), pck


@torch.inference_mode()
def measure_throughput(model, model_kind, device):
    pixels = torch.randn(1, 3, INPUT_HEIGHT, INPUT_WIDTH, device=device)
    domain = torch.zeros(1, dtype=torch.long, device=device)

    def measure(amp):
        with torch.autocast("cuda", enabled=amp, dtype=torch.float16):
            for _ in range(20):
                if model_kind == "vitpose":
                    model(pixel_values=pixels, dataset_index=domain)
                else:
                    model(pixels)
            torch.cuda.synchronize()
            started = time.perf_counter()
            iterations = 100
            for _ in range(iterations):
                if model_kind == "vitpose":
                    model(pixel_values=pixels, dataset_index=domain)
                else:
                    model(pixels)
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        return {"fps": iterations / elapsed, "latency_ms": elapsed * 1000 / iterations}

    return {"fp32": measure(False), "fp16": measure(True)}


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    annotation_path = root / "data" / "coco" / "annotations" / "person_keypoints_val2017.json"
    coco = COCO(str(annotation_path))
    annotation_to_image = {annotation["id"]: annotation["image_id"] for annotation in coco.dataset["annotations"]}
    records = load_coco_records(root / "data" / "coco", "val")
    if args.limit:
        records = records[: args.limit]
    loader = DataLoader(PoseDataset(records, augment=False), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    device = torch.device("cuda")
    model, model_info = load_model(args, device)
    results, image_ids, pck = inference(model, args.model, loader, device, annotation_to_image)
    result_object = coco.loadRes(results)
    evaluator = COCOeval(coco, result_object, "keypoints")
    evaluator.params.imgIds = image_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    names = ("AP", "AP50", "AP75", "AP_medium", "AP_large", "AR", "AR50", "AR75", "AR_medium", "AR_large")
    report = {
        "protocol": "COCO 2017 validation, ground-truth person boxes, identical 1.25x crops",
        "samples": len(records), "model": model_info,
        "coco": {name: float(value) for name, value in zip(names, evaluator.stats)},
        "pck": pck, "throughput": measure_throughput(model, args.model, device),
    }
    output = args.output or root / "runs" / f"benchmark-{args.model}-{int(time.time())}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
