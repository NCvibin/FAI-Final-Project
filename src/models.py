"""ImageNet-pretrained backbones with a single-logit head (binary task)."""
import torch.nn as nn
from torchvision import models


def build_model(name: str = "densenet169", pretrained: bool = True) -> nn.Module:
    if name == "densenet169":
        m = models.densenet169(weights="DEFAULT" if pretrained else None)
        m.classifier = nn.Linear(m.classifier.in_features, 1)
    elif name in ("resnet18", "resnet50"):
        m = getattr(models, name)(weights="DEFAULT" if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, 1)
    else:
        raise ValueError(f"unknown model {name}")
    return m


def gradcam_target_layer(model: nn.Module, name: str) -> nn.Module:
    """Last convolutional block, for Grad-CAM later (step 6)."""
    if name == "densenet169":
        return model.features.denseblock4
    return model.layer4
