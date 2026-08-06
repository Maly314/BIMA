from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode


INPUT_WIDTH = 192
INPUT_HEIGHT = 256
HEATMAP_WIDTH = 48
HEATMAP_HEIGHT = 64
COCO_KEYPOINTS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
FLIP_INDEX = torch.tensor([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15])
TORSO_PAIRS = ((5, 11), (6, 12), (5, 12), (6, 11))


@dataclass(frozen=True)
class PoseRecord:
    image_path: str
    bbox: tuple[float, float, float, float]
    keypoints: tuple[tuple[float, float, float], ...]
    domain: int
    sample_id: str


def _expand_box(box: tuple[float, float, float, float], scale: float, tx: float, ty: float):
    x, y, width, height = box
    center_x = x + width * (0.5 + tx)
    center_y = y + height * (0.5 + ty)
    aspect = INPUT_WIDTH / INPUT_HEIGHT
    if width > aspect * height:
        height = width / aspect
    else:
        width = height * aspect
    width *= scale
    height *= scale
    return center_x - width / 2, center_y - height / 2, width, height


def _rotate_points(points: torch.Tensor, angle_degrees: float) -> torch.Tensor:
    angle = math.radians(angle_degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    center = torch.tensor([(INPUT_WIDTH - 1) / 2, (INPUT_HEIGHT - 1) / 2], dtype=points.dtype)
    shifted = points - center
    # PIL uses counter-clockwise positive angles in image coordinates (y down).
    x = cosine * shifted[:, 0] + sine * shifted[:, 1]
    y = -sine * shifted[:, 0] + cosine * shifted[:, 1]
    return torch.stack((x, y), dim=-1) + center


class PoseDataset(Dataset):
    def __init__(self, records: list[PoseRecord], augment: bool = False):
        self.records = records
        self.augment = augment
        self.color = ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12, hue=0.025)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = Image.open(record.image_path).convert("RGB")
        points = torch.tensor([[p[0], p[1]] for p in record.keypoints], dtype=torch.float32)
        visible = torch.tensor([p[2] > 0 for p in record.keypoints], dtype=torch.bool)

        if self.augment:
            box_scale = random.uniform(1.12, 1.42)
            translate_x = random.uniform(-0.045, 0.045)
            translate_y = random.uniform(-0.045, 0.045)
        else:
            box_scale, translate_x, translate_y = 1.25, 0.0, 0.0
        crop_x, crop_y, crop_w, crop_h = _expand_box(record.bbox, box_scale, translate_x, translate_y)

        image = image.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
        image = image.resize((INPUT_WIDTH, INPUT_HEIGHT), Image.Resampling.BILINEAR)
        points[:, 0] = (points[:, 0] - crop_x) * INPUT_WIDTH / crop_w
        points[:, 1] = (points[:, 1] - crop_y) * INPUT_HEIGHT / crop_h

        angle = 0.0
        if self.augment:
            angle = random.uniform(-28.0, 28.0) if record.domain == 0 else random.uniform(-45.0, 45.0)
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
            points = _rotate_points(points, angle)
            if random.random() < 0.5:
                image = TF.hflip(image)
                points[:, 0] = INPUT_WIDTH - 1 - points[:, 0]
                points = points[FLIP_INDEX]
                visible = visible[FLIP_INDEX]
            image = self.color(image)

        inside = (
            (points[:, 0] >= 0) & (points[:, 0] < INPUT_WIDTH)
            & (points[:, 1] >= 0) & (points[:, 1] < INPUT_HEIGHT)
        )
        visible &= inside
        heatmap_points = points.clone()
        heatmap_points[:, 0] *= HEATMAP_WIDTH / INPUT_WIDTH
        heatmap_points[:, 1] *= HEATMAP_HEIGHT / INPUT_HEIGHT

        pixel_values = TF.pil_to_tensor(image).float() / 255.0
        pixel_values = TF.normalize(pixel_values, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        return {
            "pixel_values": pixel_values,
            "keypoints": heatmap_points,
            "visible": visible,
            "domain": torch.tensor(record.domain, dtype=torch.long),
            "sample_id": record.sample_id,
            "original_keypoints": torch.tensor(record.keypoints, dtype=torch.float32),
            "crop": torch.tensor((crop_x, crop_y, crop_w, crop_h), dtype=torch.float32),
        }


def load_coco_records(root: Path, split: str, minimum_keypoints: int = 6) -> list[PoseRecord]:
    annotation_path = root / "annotations" / f"person_keypoints_{split}2017.json"
    image_root = root / f"{split}2017"
    cache_path = root / "annotations" / f"pose_records_{split}_min{minimum_keypoints}.pt"
    if cache_path.exists() and cache_path.stat().st_mtime >= annotation_path.stat().st_mtime:
        return torch.load(cache_path, map_location="cpu", weights_only=False)
    with annotation_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    images = {item["id"]: item for item in payload["images"]}
    records: list[PoseRecord] = []
    for annotation in payload["annotations"]:
        if annotation.get("iscrowd") or annotation.get("num_keypoints", 0) < minimum_keypoints:
            continue
        image = images[annotation["image_id"]]
        image_path = image_root / image["file_name"]
        raw = annotation["keypoints"]
        keypoints = tuple((float(raw[i]), float(raw[i + 1]), float(raw[i + 2])) for i in range(0, 51, 3))
        records.append(PoseRecord(
            image_path=str(image_path),
            bbox=tuple(float(value) for value in annotation["bbox"]),
            keypoints=keypoints,
            domain=0,
            sample_id=f"coco-{annotation['id']}",
        ))
    torch.save(records, cache_path)
    return records


BABY_JOINT_ALIASES = {
    5: ("leftshoulder", "shoulderleft", "lshoulder", "shoulderl"),
    6: ("rightshoulder", "shoulderright", "rshoulder", "shoulderr"),
    7: ("leftelbow", "elbowleft", "lelbow", "elbowl"),
    8: ("rightelbow", "elbowright", "relbow", "elbowr"),
    9: ("leftwrist", "wristleft", "lwrist", "wristl", "lefthand", "handleft"),
    10: ("rightwrist", "wristright", "rwrist", "wristr", "righthand", "handright"),
    11: ("lefthip", "hipleft", "lhip", "hipl"),
    12: ("righthip", "hipright", "rhip", "hipr"),
    13: ("leftknee", "kneeleft", "lknee", "kneel"),
    14: ("rightknee", "kneeright", "rknee", "kneer"),
    15: ("leftankle", "ankleleft", "lankle", "anklel", "leftfoot", "footleft"),
    16: ("rightankle", "ankleright", "rankle", "ankler", "rightfoot", "footright"),
}


def _normalized_column(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _find_axis_column(columns: Iterable[str], aliases: tuple[str, ...], axis: str) -> str | None:
    normalized = {column: _normalized_column(column) for column in columns}
    candidates = []
    for column, compact in normalized.items():
        for alias in aliases:
            if alias in compact and (compact.endswith(axis) or compact.startswith(axis) or f"{axis}{alias}" in compact):
                candidates.append((len(compact), column))
    return min(candidates)[1] if candidates else None


def _resolve_baby_columns(frame: pd.DataFrame):
    columns = list(frame.columns)
    path_column = next((column for column in columns if any(word in _normalized_column(column) for word in ("filename", "filepath", "image"))), columns[0])
    mapping: dict[int, tuple[str, str]] = {}
    for keypoint_index, aliases in BABY_JOINT_ALIASES.items():
        x_column = _find_axis_column(columns, aliases, "x")
        y_column = _find_axis_column(columns, aliases, "y")
        if x_column and y_column:
            mapping[keypoint_index] = (x_column, y_column)
    if len(mapping) < 10:
        raise ValueError(f"Could not identify babyPose columns. Found {mapping}; columns={columns}")
    return path_column, mapping


def load_babypose_records(root: Path, split: str) -> list[PoseRecord]:
    workbook = next(iter(root.rglob(f"*{split}*.xlsx")), None)
    if workbook is None:
        raise FileNotFoundError(f"No babyPose {split} workbook under {root}")
    frame = pd.read_excel(workbook)
    path_column, mapping = _resolve_baby_columns(frame)
    records: list[PoseRecord] = []
    for row_index, row in frame.iterrows():
        relative = str(row[path_column]).replace("\\", "/").lstrip("./")
        filename = Path(relative).name
        candidates = list(root.rglob(filename))
        if not candidates:
            continue
        keypoints = [[0.0, 0.0, 0.0] for _ in COCO_KEYPOINTS]
        for keypoint_index, (x_column, y_column) in mapping.items():
            x, y = row[x_column], row[y_column]
            if pd.notna(x) and pd.notna(y) and float(x) >= 0 and float(y) >= 0:
                keypoints[keypoint_index] = [float(x), float(y), 2.0]
        visible_xy = np.array([[point[0], point[1]] for point in keypoints if point[2] > 0], dtype=np.float32)
        if len(visible_xy) < 6:
            continue
        low = visible_xy.min(axis=0)
        high = visible_xy.max(axis=0)
        width, height = np.maximum(high - low, np.array([24.0, 32.0]))
        bbox = (float(low[0] - width * 0.18), float(low[1] - height * 0.18), float(width * 1.36), float(height * 1.36))
        records.append(PoseRecord(
            image_path=str(candidates[0]),
            bbox=bbox,
            keypoints=tuple(tuple(point) for point in keypoints),
            domain=1,
            sample_id=f"baby-{split}-{row_index}",
        ))
    return records


def torso_scale(keypoints: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
    scales = []
    for shoulder, hip in TORSO_PAIRS:
        valid = visible[:, shoulder] & visible[:, hip]
        distance = torch.linalg.vector_norm(keypoints[:, shoulder] - keypoints[:, hip], dim=-1)
        scales.append(torch.where(valid, distance, torch.zeros_like(distance)))
    stacked = torch.stack(scales, dim=-1)
    count = (stacked > 0).sum(dim=-1).clamp_min(1)
    return stacked.sum(dim=-1) / count
