from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet18, resnet34


class ResNetBaseline(nn.Module):
    def __init__(
        self,
        num_classes: int,
        cifar_stem: bool = True,
        dropout: float = 0.0,
        resnet_name: str = "resnet18",
    ):
        super().__init__()
        builders = {
            "resnet18": resnet18,
            "resnet34": resnet34,
        }
        if resnet_name not in builders:
            raise ValueError(f"Unsupported resnet_name: {resnet_name}. Choices: {sorted(builders.keys())}")

        self.backbone = builders[resnet_name](weights=None)

        if cifar_stem:
            self.backbone.conv1 = nn.Conv2d(
                3,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            self.backbone.maxpool = nn.Identity()

        in_features = self.backbone.fc.in_features
        if dropout > 0:
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
        else:
            self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
