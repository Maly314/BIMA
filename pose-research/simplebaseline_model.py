from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18

from kinepose_model import JOINTS, LIMBS, KinePoseOutput
from pose_data import HEATMAP_HEIGHT, HEATMAP_WIDTH


class SimpleBaselineR18(nn.Module):
    """ResNet-18 plus three deconvolutions, trained from random initialization.

    This is an established-style top-down heatmap control. Auxiliary heads are
    kept identical to KinePose so the two models can use the same supervision.
    """

    def __init__(self, deconv_channels: int = 128):
        super().__init__()
        backbone = resnet18(weights=None)
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        layers = []
        input_channels = 512
        for _ in range(3):
            layers.extend(
                (
                    nn.ConvTranspose2d(
                        input_channels,
                        deconv_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(deconv_channels),
                    nn.ReLU(inplace=True),
                )
            )
            input_channels = deconv_channels
        self.deconv = nn.Sequential(*layers)
        self.heatmap_head = nn.Conv2d(deconv_channels, JOINTS, 1)
        self.field_head = nn.Conv2d(deconv_channels, len(LIMBS) * 2, 1)
        self.offset_head = nn.Conv2d(deconv_channels, JOINTS * 2, 1)
        self.uncertainty_pool = nn.AdaptiveAvgPool2d(1)
        self.uncertainty = nn.Linear(deconv_channels, JOINTS)
        self._initialize_heads()

    def _initialize_heads(self):
        for module in self.deconv.modules():
            if isinstance(module, nn.ConvTranspose2d):
                nn.init.normal_(module.weight, std=0.001)
        for module in (self.heatmap_head, self.field_head, self.offset_head):
            nn.init.normal_(module.weight, std=0.001)
            nn.init.zeros_(module.bias)
        nn.init.normal_(self.uncertainty.weight, std=0.001)
        nn.init.zeros_(self.uncertainty.bias)

    def forward(self, images: torch.Tensor) -> KinePoseOutput:
        features = self.deconv(self.backbone(images))
        heatmaps = self.heatmap_head(features)
        if heatmaps.shape[-2:] != (HEATMAP_HEIGHT, HEATMAP_WIDTH):
            raise RuntimeError(f"Unexpected output shape: {tuple(heatmaps.shape)}")
        offsets = self.offset_head(features).view(
            images.shape[0], JOINTS, 2, HEATMAP_HEIGHT, HEATMAP_WIDTH
        )
        pooled = self.uncertainty_pool(features).flatten(1)
        return KinePoseOutput(
            heatmaps=heatmaps,
            coarse_heatmaps=heatmaps,
            offsets=offsets,
            limb_fields=self.field_head(features),
            log_variance=self.uncertainty(pooled).clamp(-5, 5),
        )

