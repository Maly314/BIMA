from __future__ import annotations

import math

import torch
from torch import nn

from kinepose_model import (
    JOINTS,
    LIMBS,
    KinePoseOutput,
    SkeletonGraphBlock,
    skeleton_adjacency,
)
from pose_data import HEATMAP_HEIGHT, HEATMAP_WIDTH


class ResidualUnit(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
        )
        self.skip = (
            nn.Identity()
            if stride == 1 and input_channels == output_channels
            else nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(output_channels),
            )
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor):
        return self.activation(self.skip(inputs) + self.body(inputs))


class KineResPose(nn.Module):
    """Compute-focused adult pose model with a location-aware skeleton head.

    The encoder keeps expensive convolutions at low resolution. A learned
    three-stage upsampler restores 64x48 features; auxiliary limb fields and
    sharp heatmap-pooled joint tokens then drive explicit skeleton reasoning.
    """

    def __init__(self, graph_depth: int = 3, decoder_channels: int = 96, variant: str = "graph"):
        super().__init__()
        if variant not in {"plain", "graph"}:
            raise ValueError(f"Unknown KineResPose variant: {variant}")
        self.variant = variant
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )
        self.stage1 = nn.Sequential(ResidualUnit(32, 32), ResidualUnit(32, 32))
        self.stage2 = nn.Sequential(ResidualUnit(32, 64, 2), ResidualUnit(64, 64))
        self.stage3 = nn.Sequential(ResidualUnit(64, 128, 2), ResidualUnit(128, 128))
        self.stage4 = nn.Sequential(ResidualUnit(128, 256, 2), ResidualUnit(256, 256))
        decoder = []
        input_channels = 256
        for _ in range(3):
            decoder.extend(
                (
                    nn.ConvTranspose2d(input_channels, decoder_channels, 4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(decoder_channels),
                    nn.SiLU(inplace=True),
                )
            )
            input_channels = decoder_channels
        self.decoder = nn.Sequential(*decoder)
        self.coarse_head = nn.Conv2d(decoder_channels, JOINTS, 1)
        self.field_head = nn.Conv2d(decoder_channels, len(LIMBS) * 2, 1)
        self.offset_head = nn.Conv2d(decoder_channels, JOINTS * 2, 1)
        self.field_embedding = nn.Conv2d(len(LIMBS) * 2, decoder_channels, 1)
        self.joint_embedding = nn.Parameter(torch.randn(1, JOINTS, decoder_channels) * 0.02)
        adjacency = skeleton_adjacency()
        self.graph = nn.ModuleList(
            SkeletonGraphBlock(decoder_channels, adjacency) for _ in range(graph_depth)
        )
        self.pixel_projection = nn.Conv2d(decoder_channels, decoder_channels, 1, bias=False)
        self.query_projection = nn.Linear(decoder_channels, decoder_channels, bias=False)
        self.uncertainty = nn.Sequential(nn.LayerNorm(decoder_channels), nn.Linear(decoder_channels, 1))
        self.refine_scale = nn.Parameter(torch.tensor(0.05))
        self._initialize_heads()

    def _initialize_heads(self):
        for module in (self.coarse_head, self.field_head, self.offset_head):
            nn.init.normal_(module.weight, std=0.001)
            nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor) -> KinePoseOutput:
        features = self.stem(images)
        features = self.stage1(features)
        features = self.stage2(features)
        features = self.stage3(features)
        features = self.stage4(features)
        high_resolution = self.decoder(features)
        if high_resolution.shape[-2:] != (HEATMAP_HEIGHT, HEATMAP_WIDTH):
            raise RuntimeError(f"Unexpected output shape: {tuple(high_resolution.shape)}")
        coarse = self.coarse_head(high_resolution)
        fields = self.field_head(high_resolution)
        offsets = self.offset_head(high_resolution).view(
            images.shape[0], JOINTS, 2, HEATMAP_HEIGHT, HEATMAP_WIDTH
        )
        probabilities = (coarse.flatten(2) / 0.18).softmax(dim=-1)
        pixels = (high_resolution + self.field_embedding(fields)).flatten(2).transpose(1, 2)
        tokens = torch.bmm(probabilities, pixels) + self.joint_embedding
        if self.variant == "graph":
            for block in self.graph:
                tokens = block(tokens)
            projected = self.pixel_projection(high_resolution)
            queries = self.query_projection(tokens)
            refinement = torch.einsum("bjc,bchw->bjhw", queries, projected) / math.sqrt(projected.shape[1])
            heatmaps = coarse + torch.tanh(self.refine_scale) * refinement
        else:
            heatmaps = coarse
        return KinePoseOutput(
            heatmaps=heatmaps,
            coarse_heatmaps=coarse,
            offsets=offsets,
            limb_fields=fields,
            log_variance=self.uncertainty(tokens).squeeze(-1).clamp(-5, 5),
        )

