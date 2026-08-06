from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from kinepose_model import KinePose, decode_coordinates
from kineres_model import KineResPose
from pose_data import INPUT_HEIGHT, INPUT_WIDTH


class DeploymentWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor):
        output = self.model(pixel_values)
        coordinates = decode_coordinates(output)
        confidence = output.heatmaps.sigmoid().flatten(2).amax(dim=-1)
        uncertainty = torch.exp(0.5 * output.log_variance)
        return coordinates, confidence, uncertainty


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    architecture = payload["architecture"]
    if architecture.get("name") == "KinePose":
        model = KinePose(
            width=architecture["width"],
            graph_depth=architecture["graph_depth"],
            variant=architecture["variant"],
        )
    elif architecture.get("name") == "KineResPose":
        model = KineResPose(
            graph_depth=architecture["graph_depth"],
            decoder_channels=architecture["decoder_channels"],
            variant=architecture["variant"],
        )
    else:
        raise ValueError(f"Unsupported architecture: {architecture.get('name')}")
    model.load_state_dict(payload["model"])
    wrapper = DeploymentWrapper(model.eval()).eval()
    example = torch.randn(1, 3, INPUT_HEIGHT, INPUT_WIDTH)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torchscript_path = args.output_dir / "kinepose-deployment.pt"
    traced = torch.jit.trace(wrapper, example, strict=True)
    traced.save(str(torchscript_path))
    with torch.inference_mode():
        eager_outputs = wrapper(example)
        traced_outputs = traced(example)
    trace_errors = [
        float((expected - actual).abs().max())
        for expected, actual in zip(eager_outputs, traced_outputs)
    ]

    onnx_path = args.output_dir / "kinepose-deployment.onnx"
    torch.onnx.export(
        wrapper,
        (example,),
        str(onnx_path),
        input_names=["pixel_values"],
        output_names=["coordinates", "confidence", "uncertainty"],
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "coordinates": {0: "batch"},
            "confidence": {0: "batch"},
            "uncertainty": {0: "batch"},
        },
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )

    report = {
        "source_checkpoint": str(args.checkpoint.resolve()),
        "architecture": architecture,
        "input": {"name": "pixel_values", "shape": ["batch", 3, INPUT_HEIGHT, INPUT_WIDTH]},
        "outputs": {
            "coordinates": ["batch", 17, 2],
            "confidence": ["batch", 17],
            "uncertainty": ["batch", 17],
        },
        "torchscript_max_abs_error": trace_errors,
        "torchscript_bytes": torchscript_path.stat().st_size,
        "onnx_bytes": onnx_path.stat().st_size,
    }

    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        actual = session.run(None, {"pixel_values": example.numpy()})
        report["onnx_max_abs_error"] = [
            float(np.max(np.abs(expected.detach().numpy() - result)))
            for expected, result in zip(eager_outputs, actual)
        ]
        report["onnx_validated"] = True
    except ImportError:
        report["onnx_validated"] = False
        report["onnx_validation_note"] = "onnxruntime is not installed"

    report_path = args.output_dir / "export-report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
