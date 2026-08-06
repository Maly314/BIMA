from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from pose_data import HEATMAP_HEIGHT, HEATMAP_WIDTH


JOINTS = 17
LIMBS = (
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
)


class DirectionalMixer(nn.Module):
    """Local + horizontal + vertical spatial mixing with input-conditioned gates."""

    def __init__(self, channels: int, expansion: int = 3):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.local = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.horizontal = nn.Conv2d(channels, channels, (1, 9), padding=(0, 4), groups=channels)
        self.vertical = nn.Conv2d(channels, channels, (9, 1), padding=(4, 0), groups=channels)
        hidden = channels * expansion
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, max(channels // 4, 12), 1),
            nn.SiLU(), nn.Conv2d(max(channels // 4, 12), 3, 1),
        )
        self.channel_mix = nn.Sequential(
            nn.Conv2d(channels, hidden, 1), nn.GELU(), nn.Conv2d(hidden, channels, 1),
        )
        self.layer_scale = nn.Parameter(torch.full((1, channels, 1, 1), 1e-3))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(inputs)
        weights = self.gate(normalized).softmax(dim=1)
        mixed = (
            weights[:, 0:1] * self.local(normalized)
            + weights[:, 1:2] * self.horizontal(normalized)
            + weights[:, 2:3] * self.vertical(normalized)
        )
        return inputs + self.layer_scale * self.channel_mix(mixed)


class PlainMixer(nn.Module):
    """Conventional local residual block used as a from-scratch control."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(1, channels),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )
        self.layer_scale = nn.Parameter(torch.full((1, channels, 1, 1), 1e-3))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.layer_scale * self.block(inputs)


class FastPlainMixer(PlainMixer):
    """Same capacity as PlainMixer with a less conservative residual start."""

    def __init__(self, channels: int):
        super().__init__(channels)
        self.layer_scale.data.fill_(0.1)


class EncoderStage(nn.Sequential):
    def __init__(self, channels: int, depth: int, block=DirectionalMixer):
        super().__init__(*(block(channels) for _ in range(depth)))


class GatedFuse(nn.Module):
    def __init__(self, channels: int, block=DirectionalMixer):
        super().__init__()
        self.gate = nn.Sequential(nn.Conv2d(channels * 2, channels, 1), nn.Sigmoid())
        self.refine = nn.Sequential(block(channels), block(channels))

    def forward(self, lateral: torch.Tensor, top_down: torch.Tensor) -> torch.Tensor:
        top_down = F.interpolate(top_down, size=lateral.shape[-2:], mode="bilinear", align_corners=False)
        weight = self.gate(torch.cat((lateral, top_down), dim=1))
        return self.refine(weight * lateral + (1 - weight) * top_down)


class SkeletonGraphBlock(nn.Module):
    def __init__(self, channels: int, adjacency: torch.Tensor):
        super().__init__()
        self.register_buffer("adjacency", adjacency, persistent=False)
        self.message = nn.Sequential(nn.Linear(channels * 2, channels * 2), nn.GELU(), nn.Linear(channels * 2, channels))
        self.update = nn.GRUCell(channels, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        neighbor = torch.einsum("ij,bjc->bic", self.adjacency, tokens)
        message = self.message(torch.cat((tokens, neighbor), dim=-1))
        updated = self.update(message.flatten(0, 1), tokens.flatten(0, 1)).view_as(tokens)
        return self.norm(tokens + updated)


@dataclass
class KinePoseOutput:
    heatmaps: torch.Tensor
    coarse_heatmaps: torch.Tensor
    offsets: torch.Tensor
    limb_fields: torch.Tensor
    log_variance: torch.Tensor


def skeleton_adjacency() -> torch.Tensor:
    adjacency = torch.eye(JOINTS)
    # Face points connect to the nose and shoulders; body points use COCO limbs.
    extra = ((0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6))
    for first, second in tuple(LIMBS) + extra:
        adjacency[first, second] = 1
        adjacency[second, first] = 1
    return adjacency / adjacency.sum(dim=-1, keepdim=True)


class KinePose(nn.Module):
    """
    Original adult pose architecture trained from random initialization.

    It combines a directional convolutional pyramid, auxiliary limb vector
    fields, heatmap-weighted joint tokens, and explicit skeleton-graph message
    passing before producing refined high-resolution joint distributions.
    """

    def __init__(self, width: int = 48, graph_depth: int = 3, variant: str = "full"):
        super().__init__()
        if variant not in {"full", "no_graph", "no_fields", "plain", "hybrid", "hybrid_sharp", "hybrid_anchor", "hybrid_fast"}:
            raise ValueError(f"Unknown variant: {variant}")
        self.variant = variant
        if variant == "hybrid_fast":
            block = FastPlainMixer
        elif variant in {"plain", "hybrid", "hybrid_sharp", "hybrid_anchor"}:
            block = PlainMixer
        else:
            block = DirectionalMixer
        channels = (width, width * 2, width * 4, width * 6)
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], 4, stride=4),
            nn.GroupNorm(1, channels[0]), nn.GELU(),
        )
        self.stage1 = EncoderStage(channels[0], 3, block)
        self.down1 = nn.Conv2d(channels[0], channels[1], 2, stride=2)
        self.stage2 = EncoderStage(channels[1], 4, block)
        self.down2 = nn.Conv2d(channels[1], channels[2], 2, stride=2)
        self.stage3 = EncoderStage(channels[2], 5, block)
        self.down3 = nn.Conv2d(channels[2], channels[3], 2, stride=2)
        self.stage4 = EncoderStage(channels[3], 3, block)

        fused_channels = width * 2
        self.lateral1 = nn.Conv2d(channels[0], fused_channels, 1)
        self.lateral2 = nn.Conv2d(channels[1], fused_channels, 1)
        self.lateral3 = nn.Conv2d(channels[2], fused_channels, 1)
        self.lateral4 = nn.Conv2d(channels[3], fused_channels, 1)
        self.fuse43 = GatedFuse(fused_channels, block)
        self.fuse32 = GatedFuse(fused_channels, block)
        self.fuse21 = GatedFuse(fused_channels, block)

        self.coarse_head = nn.Sequential(block(fused_channels), nn.Conv2d(fused_channels, JOINTS, 1))
        self.field_head = nn.Sequential(block(fused_channels), nn.Conv2d(fused_channels, len(LIMBS) * 2, 1))
        self.offset_head = nn.Sequential(block(fused_channels), nn.Conv2d(fused_channels, JOINTS * 2, 1))
        self.field_embedding = nn.Conv2d(len(LIMBS) * 2, fused_channels, 1)
        self.pixel_projection = nn.Conv2d(fused_channels, fused_channels, 1, bias=False)
        self.joint_embedding = nn.Parameter(torch.randn(1, JOINTS, fused_channels) * 0.02)
        adjacency = skeleton_adjacency()
        self.graph = nn.ModuleList(SkeletonGraphBlock(fused_channels, adjacency) for _ in range(graph_depth))
        self.query_projection = nn.Linear(fused_channels, fused_channels, bias=False)
        self.uncertainty = nn.Sequential(nn.LayerNorm(fused_channels), nn.Linear(fused_channels, 1))
        self.refine_scale = nn.Parameter(torch.tensor(0.0))
        if variant == "hybrid_anchor":
            self.coordinate_embedding = nn.Linear(2, fused_channels)
            self.coordinate_delta = nn.Sequential(
                nn.LayerNorm(fused_channels), nn.Linear(fused_channels, 2)
            )
            self.prior_log_sigma = nn.Parameter(torch.full((1, JOINTS, 1, 1), math.log(1.5)))
            self.prior_gain = nn.Parameter(torch.tensor(-2.0))

        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor) -> KinePoseOutput:
        feature1 = self.stage1(self.stem(images))
        feature2 = self.stage2(self.down1(feature1))
        feature3 = self.stage3(self.down2(feature2))
        feature4 = self.stage4(self.down3(feature3))
        pyramid3 = self.fuse43(self.lateral3(feature3), self.lateral4(feature4))
        pyramid2 = self.fuse32(self.lateral2(feature2), pyramid3)
        high_resolution = self.fuse21(self.lateral1(feature1), pyramid2)

        coarse = self.coarse_head(high_resolution)
        fields = self.field_head(high_resolution)
        offsets = self.offset_head(high_resolution).view(images.shape[0], JOINTS, 2, HEATMAP_HEIGHT, HEATMAP_WIDTH)
        token_features = high_resolution if self.variant in {"no_fields", "plain"} else high_resolution + self.field_embedding(fields)

        # The Gaussian heatmap targets live in [0, 1]. Temperature-1 pooling is
        # consequently almost uniform and yields a global-image token. Preserve
        # that behavior in the original ablation, while the separately named
        # sharp variant pools genuinely joint-local evidence.
        token_temperature = 0.18 if self.variant in {"hybrid_sharp", "hybrid_anchor", "hybrid_fast"} else 1.0
        probabilities = (coarse.flatten(2) / token_temperature).softmax(dim=-1)
        pixels = token_features.flatten(2).transpose(1, 2)
        if self.variant == "hybrid_anchor":
            y_grid, x_grid = torch.meshgrid(
                torch.arange(HEATMAP_HEIGHT, device=coarse.device, dtype=coarse.dtype),
                torch.arange(HEATMAP_WIDTH, device=coarse.device, dtype=coarse.dtype),
                indexing="ij",
            )
            coarse_x = (probabilities * x_grid.flatten()).sum(dim=-1)
            coarse_y = (probabilities * y_grid.flatten()).sum(dim=-1)
            normalized_coordinates = torch.stack(
                (
                    coarse_x * (2.0 / (HEATMAP_WIDTH - 1)) - 1.0,
                    coarse_y * (2.0 / (HEATMAP_HEIGHT - 1)) - 1.0,
                ),
                dim=-1,
            )
            sampled = F.grid_sample(
                token_features,
                normalized_coordinates[:, :, None, :],
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            ).squeeze(-1).transpose(1, 2)
            tokens = sampled + self.coordinate_embedding(normalized_coordinates) + self.joint_embedding
        else:
            tokens = torch.bmm(probabilities, pixels) + self.joint_embedding
        if self.variant not in {"no_graph", "plain"}:
            for block in self.graph:
                tokens = block(tokens)
        if self.variant == "hybrid_anchor":
            delta = 3.0 * torch.tanh(self.coordinate_delta(tokens))
            refined_x = coarse_x + delta[:, :, 0]
            refined_y = coarse_y + delta[:, :, 1]
            squared_distance = (
                (x_grid[None, None] - refined_x[:, :, None, None]).square()
                + (y_grid[None, None] - refined_y[:, :, None, None]).square()
            )
            sigma = self.prior_log_sigma.exp().clamp(0.75, 4.0)
            kinematic_prior = torch.exp(-squared_distance / (2.0 * sigma.square()))
            heatmaps = coarse + F.softplus(self.prior_gain) * kinematic_prior
        else:
            projected_pixels = self.pixel_projection(high_resolution)
            queries = self.query_projection(tokens)
            refinement = torch.einsum("bjc,bchw->bjhw", queries, projected_pixels) / math.sqrt(projected_pixels.shape[1])
            heatmaps = coarse if self.variant == "plain" else coarse + torch.tanh(self.refine_scale) * refinement
        return KinePoseOutput(
            heatmaps=heatmaps,
            coarse_heatmaps=coarse,
            offsets=offsets,
            limb_fields=fields,
            log_variance=self.uncertainty(tokens).squeeze(-1).clamp(-5, 5),
        )


def decode_coordinates(output: KinePoseOutput, temperature: float = 0.18) -> torch.Tensor:
    heatmaps = output.heatmaps.float()
    batch, joints, height, width = heatmaps.shape
    probabilities = F.softmax(heatmaps.flatten(2) / temperature, dim=-1)
    y_grid, x_grid = torch.meshgrid(
        torch.arange(height, device=heatmaps.device, dtype=heatmaps.dtype),
        torch.arange(width, device=heatmaps.device, dtype=heatmaps.dtype),
        indexing="ij",
    )
    x = (probabilities * x_grid.flatten()).sum(dim=-1)
    y = (probabilities * y_grid.flatten()).sum(dim=-1)
    # The offset branch is intentionally limited to half a heatmap pixel. It
    # refines localization; it cannot replace a missing or incorrect heatmap.
    offsets = (0.5 * torch.tanh(output.offsets.float())).flatten(3)
    offset_x = (probabilities * offsets[:, :, 0]).sum(dim=-1)
    offset_y = (probabilities * offsets[:, :, 1]).sum(dim=-1)
    return torch.stack((x + offset_x, y + offset_y), dim=-1)


def parameter_report(model: nn.Module) -> dict[str, int]:
    return {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }
