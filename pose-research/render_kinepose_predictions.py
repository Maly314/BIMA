from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from kinepose_model import KinePose, LIMBS, decode_coordinates
from pose_data import HEATMAP_HEIGHT, HEATMAP_WIDTH, INPUT_HEIGHT, INPUT_WIDTH, PoseDataset, load_coco_records


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=12)
    return parser.parse_args()


def draw_pose(draw, points, visible, color, width=3):
    for first, second in LIMBS:
        if visible[first] and visible[second]:
            draw.line((tuple(points[first]), tuple(points[second])), fill=color, width=width)
    for index, point in enumerate(points):
        if visible[index]:
            x, y = point
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)


@torch.inference_mode()
def main():
    args = parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    architecture = payload["architecture"]
    model = KinePose(architecture["width"], architecture["graph_depth"], architecture["variant"])
    model.load_state_dict(payload["model"])
    model.cuda().eval()
    root = Path(__file__).resolve().parent
    records = load_coco_records(root / "data" / "coco", "val")[: args.samples]
    dataset = PoseDataset(records, augment=False)
    tiles = []
    for record, sample in zip(records, dataset):
        with torch.autocast("cuda", dtype=torch.float16):
            output = model(sample["pixel_values"][None].cuda())
        predicted = decode_coordinates(output)[0].cpu()
        predicted[:, 0] *= INPUT_WIDTH / HEATMAP_WIDTH
        predicted[:, 1] *= INPUT_HEIGHT / HEATMAP_HEIGHT
        image = Image.open(record.image_path).convert("RGB")
        crop_x, crop_y, crop_w, crop_h = sample["crop"].tolist()
        predicted[:, 0] = crop_x + predicted[:, 0] * crop_w / INPUT_WIDTH
        predicted[:, 1] = crop_y + predicted[:, 1] * crop_h / INPUT_HEIGHT
        target = sample["original_keypoints"]
        visible = target[:, 2] > 0
        draw = ImageDraw.Draw(image)
        draw_pose(draw, target[:, :2].tolist(), visible.tolist(), "#3ddc84", 4)
        draw_pose(draw, predicted.tolist(), visible.tolist(), "#ff3da1", 3)
        panel = image.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h)).resize((288, 384), Image.Resampling.BILINEAR)
        caption = Image.new("RGB", (288, 28), "#10151a")
        ImageDraw.Draw(caption).text((8, 6), "green: label   magenta: KinePose", fill="white", font=ImageFont.load_default())
        tile = Image.new("RGB", (288, 412), "#10151a")
        tile.paste(panel, (0, 0))
        tile.paste(caption, (0, 384))
        tiles.append(tile)
    columns = 4
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 288, rows * 412), "#10151a")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * 288, (index // columns) * 412))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, quality=92)
    print(args.output)


if __name__ == "__main__":
    main()

