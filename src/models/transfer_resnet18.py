"""
ResNet18 transfer learning model for EuroSAT RGB.
Backbone pretrained on ImageNet; classification head replaced for 10 classes.
"""

import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def build_resnet18(num_classes: int = 10, freeze_backbone: bool = True) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final fully-connected layer — this new layer is always trainable,
    # regardless of freeze_backbone, since it's randomly initialized and must learn.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model