from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import VitPoseForPoseEstimation

from pose_data import HEATMAP_HEIGHT, HEATMAP_WIDTH


BASE_MODEL_ID = "usyd-community/vitpose-plus-small"
ADULT_EXPERT = 0
INFANT_EXPERT = 6
SKELETON_EDGES = (
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
)


class HeatmapRefiner(nn.Module):
    """A zero-initialized residual structural prior over all body joints."""

    def __init__(self, joints: int = 17, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(joints, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.Conv2d(hidden, hidden, 1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, joints, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, heatmaps: torch.Tensor) -> torch.Tensor:
        return self.net(heatmaps)


class DomainRouter(nn.Module):
    """Small visual router used when age/profile metadata is unavailable."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(16), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(48, 2)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(pixel_values).flatten(1))


@dataclass
class DualPoseOutput:
    heatmaps: torch.Tensor
    base_heatmaps: torch.Tensor
    router_logits: torch.Tensor
    routed_domain: torch.Tensor


def _expand_with_infant_expert(model: VitPoseForPoseEstimation) -> None:
    for layer in model.backbone.encoder.layer:
        experts = layer.mlp.experts
        if len(experts) == INFANT_EXPERT:
            experts.append(copy.deepcopy(experts[ADULT_EXPERT]))
        experts.num_experts = len(experts)
        layer.mlp.num_experts = len(experts)
    model.config.backbone_config.num_experts = INFANT_EXPERT + 1
    model.backbone.config.num_experts = INFANT_EXPERT + 1


class DualDomainVitPose(nn.Module):
    """
    ViTPose++ with a seventh infant expert and domain-specific residual heads.

    The original adult MoE path stays frozen. The infant path starts as an
    exact copy of the COCO expert, avoiding a random loss of generic pose
    knowledge before adaptation.
    """

    def __init__(self, base: VitPoseForPoseEstimation):
        super().__init__()
        _expand_with_infant_expert(base)
        self.pose = base
        self.router = DomainRouter()
        self.refiners = nn.ModuleList((HeatmapRefiner(), HeatmapRefiner()))
        self.residual_strength = nn.Parameter(torch.tensor([-2.2, -2.2]))

    @classmethod
    def from_pretrained(cls, checkpoint: str | Path | None = None):
        model = cls(VitPoseForPoseEstimation.from_pretrained(BASE_MODEL_ID))
        if checkpoint:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
            model.load_state_dict(state)
        return model

    def freeze_for_adaptation(self) -> None:
        for parameter in self.pose.parameters():
            parameter.requires_grad = False
        for layer in self.pose.backbone.encoder.layer:
            for parameter in layer.mlp.experts[INFANT_EXPERT].parameters():
                parameter.requires_grad = True
        for parameter in self.router.parameters():
            parameter.requires_grad = True
        for parameter in self.refiners.parameters():
            parameter.requires_grad = True
        self.residual_strength.requires_grad = True

    def forward(self, pixel_values: torch.Tensor, domain: torch.Tensor | None = None) -> DualPoseOutput:
        router_logits = self.router(pixel_values)
        router_probabilities = router_logits.softmax(dim=-1)
        routed_domain = router_probabilities.argmax(dim=-1) if domain is None else domain
        expert_indices = torch.where(
            routed_domain == 1,
            torch.full_like(routed_domain, INFANT_EXPERT),
            torch.full_like(routed_domain, ADULT_EXPERT),
        )
        base_heatmaps = self.pose(pixel_values=pixel_values, dataset_index=expert_indices).heatmaps
        adult_residual = self.refiners[0](base_heatmaps)
        infant_residual = self.refiners[1](base_heatmaps)
        if domain is None:
            adult_weight = router_probabilities[:, 0, None, None, None]
            infant_weight = router_probabilities[:, 1, None, None, None]
        else:
            adult_weight = (domain == 0).to(base_heatmaps.dtype)[:, None, None, None]
            infant_weight = (domain == 1).to(base_heatmaps.dtype)[:, None, None, None]
        strengths = torch.sigmoid(self.residual_strength)
        refined = base_heatmaps + adult_weight * strengths[0] * adult_residual + infant_weight * strengths[1] * infant_residual
        return DualPoseOutput(refined, base_heatmaps, router_logits, routed_domain)


def make_target_heatmaps(keypoints: torch.Tensor, visible: torch.Tensor, sigma: float = 1.8) -> torch.Tensor:
    y_grid, x_grid = torch.meshgrid(
        torch.arange(HEATMAP_HEIGHT, device=keypoints.device, dtype=keypoints.dtype),
        torch.arange(HEATMAP_WIDTH, device=keypoints.device, dtype=keypoints.dtype),
        indexing="ij",
    )
    x_delta = x_grid[None, None] - keypoints[:, :, 0, None, None]
    y_delta = y_grid[None, None] - keypoints[:, :, 1, None, None]
    targets = torch.exp(-(x_delta.square() + y_delta.square()) / (2 * sigma * sigma))
    return targets * visible[:, :, None, None].to(targets.dtype)


def softargmax_2d(heatmaps: torch.Tensor, temperature: float = 0.12) -> torch.Tensor:
    batch, joints, height, width = heatmaps.shape
    probabilities = F.softmax(heatmaps.reshape(batch, joints, -1) / temperature, dim=-1)
    y_grid, x_grid = torch.meshgrid(
        torch.arange(height, device=heatmaps.device, dtype=heatmaps.dtype),
        torch.arange(width, device=heatmaps.device, dtype=heatmaps.dtype),
        indexing="ij",
    )
    x = (probabilities * x_grid.flatten()).sum(dim=-1)
    y = (probabilities * y_grid.flatten()).sum(dim=-1)
    return torch.stack((x, y), dim=-1)


def training_loss(output: DualPoseOutput, keypoints: torch.Tensor, visible: torch.Tensor, domain: torch.Tensor):
    targets = make_target_heatmaps(keypoints, visible)
    joint_mask = visible[:, :, None, None].to(output.heatmaps.dtype)
    denominator = (joint_mask.sum() * HEATMAP_HEIGHT * HEATMAP_WIDTH).clamp_min(1)
    heatmap = ((output.heatmaps - targets).square() * joint_mask).sum() / denominator

    predicted_points = softargmax_2d(output.heatmaps.float())
    coordinate_error = F.smooth_l1_loss(predicted_points, keypoints.float(), reduction="none").sum(dim=-1)
    coordinate = (coordinate_error * visible).sum() / visible.sum().clamp_min(1)

    topology_terms = []
    for first, second in SKELETON_EDGES:
        valid = visible[:, first] & visible[:, second]
        if valid.any():
            predicted_length = torch.linalg.vector_norm(predicted_points[:, first] - predicted_points[:, second], dim=-1)
            target_length = torch.linalg.vector_norm(keypoints[:, first] - keypoints[:, second], dim=-1)
            topology_terms.append(F.smooth_l1_loss(predicted_length[valid], target_length[valid]))
    topology = torch.stack(topology_terms).mean() if topology_terms else heatmap.new_zeros(())
    router = F.cross_entropy(output.router_logits, domain)

    adult = domain == 0
    preservation = (
        F.mse_loss(output.heatmaps[adult], output.base_heatmaps[adult].detach())
        if adult.any() else heatmap.new_zeros(())
    )
    total = heatmap + 0.035 * coordinate + 0.015 * topology + 0.08 * router + 0.12 * preservation
    return total, {
        "loss": total.detach(), "heatmap": heatmap.detach(), "coordinate": coordinate.detach(),
        "topology": topology.detach(), "router": router.detach(), "preservation": preservation.detach(),
    }

