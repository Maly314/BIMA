from __future__ import annotations

import torch
from torch.nn import functional as F

from kinepose_model import JOINTS, LIMBS, KinePoseOutput, decode_coordinates
from pose_data import HEATMAP_HEIGHT, HEATMAP_WIDTH


def coordinate_grid(device, dtype):
    return torch.meshgrid(
        torch.arange(HEATMAP_HEIGHT, device=device, dtype=dtype),
        torch.arange(HEATMAP_WIDTH, device=device, dtype=dtype),
        indexing="ij",
    )


def gaussian_heatmaps(keypoints: torch.Tensor, visible: torch.Tensor, sigma: float = 1.7):
    y_grid, x_grid = coordinate_grid(keypoints.device, keypoints.dtype)
    distance = (
        (x_grid[None, None] - keypoints[:, :, 0, None, None]).square()
        + (y_grid[None, None] - keypoints[:, :, 1, None, None]).square()
    )
    targets = torch.exp(-distance / (2 * sigma * sigma))
    return targets * visible[:, :, None, None].to(targets.dtype)


def limb_field_targets(keypoints: torch.Tensor, visible: torch.Tensor, radius: float = 1.8):
    y_grid, x_grid = coordinate_grid(keypoints.device, keypoints.dtype)
    grid = torch.stack((x_grid, y_grid), dim=-1)[None]
    fields = []
    masks = []
    for first, second in LIMBS:
        start = keypoints[:, first]
        end = keypoints[:, second]
        vector = end - start
        squared_length = vector.square().sum(dim=-1).clamp_min(1e-5)
        relative = grid - start[:, None, None]
        projection = (relative * vector[:, None, None]).sum(dim=-1) / squared_length[:, None, None]
        projection = projection.clamp(0, 1)
        closest = start[:, None, None] + projection[..., None] * vector[:, None, None]
        distance = torch.linalg.vector_norm(grid - closest, dim=-1)
        valid_limb = visible[:, first] & visible[:, second]
        mask = (distance <= radius) & valid_limb[:, None, None]
        unit = vector / torch.sqrt(squared_length)[:, None]
        field = unit[:, :, None, None].expand(-1, -1, HEATMAP_HEIGHT, HEATMAP_WIDTH)
        fields.append(field)
        masks.append(mask)
    return torch.cat(fields, dim=1), torch.stack(masks, dim=1)


def torso_scale(keypoints: torch.Tensor, visible: torch.Tensor):
    values = []
    for shoulder, hip in ((5, 11), (6, 12), (5, 12), (6, 11)):
        valid = visible[:, shoulder] & visible[:, hip]
        distance = torch.linalg.vector_norm(keypoints[:, shoulder] - keypoints[:, hip], dim=-1)
        values.append(torch.where(valid, distance, torch.zeros_like(distance)))
    values = torch.stack(values, dim=-1)
    count = (values > 0).sum(dim=-1)
    scale = values.sum(dim=-1) / count.clamp_min(1)
    return scale.clamp_min(1.0), count > 0


def kinepose_loss(output: KinePoseOutput, keypoints: torch.Tensor, visible: torch.Tensor):
    targets = gaussian_heatmaps(keypoints, visible)
    joint_mask = visible[:, :, None, None].to(output.heatmaps.dtype)
    heatmap_denominator = (joint_mask.sum() * HEATMAP_HEIGHT * HEATMAP_WIDTH).clamp_min(1)
    # Standard raw Gaussian regression avoids the degenerate all-background
    # solution produced by sigmoid loss on sparse heatmaps.
    heatmap = ((output.heatmaps.float() - targets).square() * joint_mask).sum() / heatmap_denominator
    coarse = ((output.coarse_heatmaps.float() - targets).square() * joint_mask).sum() / heatmap_denominator

    predicted = decode_coordinates(output)
    scale, scale_valid = torso_scale(keypoints, visible)
    coordinate_mask = visible & scale_valid[:, None]
    normalized_delta = (predicted - keypoints) / scale[:, None, None]
    coordinate_per_joint = F.smooth_l1_loss(normalized_delta, torch.zeros_like(normalized_delta), reduction="none").sum(dim=-1)
    coordinate = (coordinate_per_joint * coordinate_mask).sum() / coordinate_mask.sum().clamp_min(1)

    bone_terms = []
    for first, second in LIMBS:
        valid = visible[:, first] & visible[:, second] & scale_valid
        if valid.any():
            predicted_vector = (predicted[:, second] - predicted[:, first]) / scale[:, None]
            target_vector = (keypoints[:, second] - keypoints[:, first]) / scale[:, None]
            bone_terms.append(F.smooth_l1_loss(predicted_vector[valid], target_vector[valid]))
    bone = torch.stack(bone_terms).mean() if bone_terms else heatmap.new_zeros(())

    field_targets, field_mask = limb_field_targets(keypoints, visible)
    predicted_fields = output.limb_fields.view(output.limb_fields.shape[0], len(LIMBS), 2, HEATMAP_HEIGHT, HEATMAP_WIDTH)
    field_targets = field_targets.view_as(predicted_fields)
    field_mask = field_mask[:, :, None].to(predicted_fields.dtype)
    field = (F.smooth_l1_loss(predicted_fields, field_targets, reduction="none") * field_mask).sum()
    field = field / (field_mask.sum() * 2).clamp_min(1)

    squared_error = normalized_delta.square().sum(dim=-1).detach()
    uncertainty_nll = torch.exp(-output.log_variance) * squared_error + output.log_variance
    uncertainty = (uncertainty_nll * coordinate_mask).sum() / coordinate_mask.sum().clamp_min(1)

    offset_regularization = output.offsets.square().mean()
    total = (
        heatmap + 0.35 * coarse + 2.0 * coordinate + 0.45 * bone
        + 0.20 * field + 0.015 * uncertainty + 0.001 * offset_regularization
    )
    return total, {
        "loss": total.detach(), "heatmap": heatmap.detach(), "coarse": coarse.detach(),
        "coordinate": coordinate.detach(), "bone": bone.detach(), "field": field.detach(),
        "uncertainty": uncertainty.detach(), "offset_l2": offset_regularization.detach(),
    }
