from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision import models


SUPPORTED_ARCHITECTURES = {
    "vgg16",
    "mobilenet_v2",
    "resnet50",
    "mobilenet_v3_small",
    "resnet18",
}


def build_model(architecture: str, num_classes: int = 2) -> nn.Module:
    architecture = architecture.lower().replace("-", "_")
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"unsupported architecture '{architecture}'; choose {sorted(SUPPORTED_ARCHITECTURES)}")
    model = getattr(models, architecture)(weights=None)
    if architecture == "vgg16":
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif architecture == "mobilenet_v2":
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif architecture == "mobilenet_v3_small":
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    else:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError("checkpoint must be a state_dict or contain state_dict/model_state_dict")
    if not all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        raise ValueError("checkpoint dictionary contains non-tensor model values")
    cleaned = {}
    for key, value in checkpoint.items():
        for prefix in ("module.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        cleaned[key] = value
    return cleaned


def load_model(architecture: str, checkpoint_path: Path, num_classes: int = 2) -> nn.Module:
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    try:
        scripted = torch.jit.load(str(checkpoint_path), map_location="cpu")
        return scripted
    except Exception:
        pass
    model = build_model(architecture, num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = _extract_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise ValueError(
            "checkpoint does not exactly match architecture "
            f"{architecture}; missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    return model


def parameter_count(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable

