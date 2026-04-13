"""
Model factory functions for CIFAR-scale experiments.

Both ResNet-18 and MobileNetV2 are adapted for 32×32 CIFAR images:
  - ResNet-18: The initial 7×7/stride-2 conv is replaced with a 3×3/stride-1
    conv, and the first max-pool is removed.  This is the standard CIFAR
    modification used in almost all published benchmarks.
  - MobileNetV2: Only the final classifier head is swapped; the backbone
    works fine on 32×32 images after the first stride is adjusted.

Usage:
    model = create_model("resnet18", num_classes=10)
    model = create_model("mobilenet_v2", num_classes=100)
"""

from __future__ import annotations

import torch.nn as nn
import torchvision.models as tv


# --------------------------------------------------------------------------- #
#  Individual factory functions                                                #
# --------------------------------------------------------------------------- #

def create_resnet18(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    """
    ResNet-18 adapted for CIFAR 32×32 images.

    Key changes vs ImageNet variant:
      1. conv1: 7×7 stride-2 → 3×3 stride-1 (preserves spatial resolution).
      2. maxpool: removed (replaced with Identity).
      3. fc: replaced to output `num_classes` logits.
    """
    weights = tv.ResNet18_Weights.DEFAULT if pretrained else None
    model = tv.resnet18(weights=weights)

    # Adapt for 32×32 CIFAR input
    model.conv1  = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    # Replace head
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def create_mobilenet_v2(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    """
    MobileNetV2 with replaced classifier head.

    The backbone's first conv uses stride=2 twice early on, which still
    produces a workable feature map from 32×32 CIFAR inputs (ends up at 1×1
    before the adaptive pool).  Performance is slightly lower than ResNet-18
    on CIFAR but the architecture is useful as a lightweight alternative.
    """
    weights = tv.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = tv.mobilenet_v2(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


# --------------------------------------------------------------------------- #
#  Unified registry + factory                                                  #
# --------------------------------------------------------------------------- #

_REGISTRY: dict[str, callable] = {
    "resnet18":      create_resnet18,
    "mobilenet_v2":  create_mobilenet_v2,
}


def create_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = False,
) -> nn.Module:
    """
    Create a model by name.

    Args:
        model_name:  One of "resnet18" or "mobilenet_v2" (case-insensitive).
        num_classes: Number of output classes (10 for CIFAR-10, 100 for CIFAR-100).
        pretrained:  Load ImageNet pre-trained weights (not recommended for
                     research baselines unless explicitly studying transfer).

    Returns:
        An nn.Module ready to be moved to a device and optionally wrapped in DDP.
    """
    key = model_name.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[key](num_classes=num_classes, pretrained=pretrained)


def count_parameters(model: nn.Module) -> int:
    """Returns total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
