from pathlib import Path

import torch
from torch import nn
from torchvision import models, transforms


CLASSES = ("Non-defect", "Defect")
IMAGE_SIZE = 224
WEIGHTS = {
    "vgg16": models.VGG16_Weights.IMAGENET1K_V1,
    "mobilenet_v2": models.MobileNet_V2_Weights.IMAGENET1K_V1,
    "resnet50": models.ResNet50_Weights.IMAGENET1K_V1,
    "mobilenet_v3_small": models.MobileNet_V3_Small_Weights.IMAGENET1K_V1,
}
CHECKPOINT_FORMATS = {"m3-full-v1", "m3-head-v1", "m4-full-v1", "m4-head-v1"}


def image_transform():
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def _safe_torch_load(path, **kwargs):
    try:
        return torch.load(path, weights_only=True, **kwargs)
    except TypeError:  # pragma: no cover - compatibility with old PyTorch
        return torch.load(path, **kwargs)


class DefectClassifier(nn.Module):
    """Reconstruct the frozen-backbone architecture shared by M3 and M4."""

    def __init__(self, architecture, pretrained=False, weights_dir=None):
        super().__init__()
        if architecture not in WEIGHTS:
            raise ValueError(f"Unsupported architecture: {architecture}")
        weights = WEIGHTS[architecture]
        constructor = getattr(models, architecture)
        base = constructor(weights=weights if pretrained and weights_dir is None else None)
        if pretrained and weights_dir is not None:
            weight_path = Path(weights_dir) / weights.url.rsplit("/", 1)[-1]
            if not weight_path.is_file():
                raise FileNotFoundError(f"Official backbone weights not found: {weight_path}")
            base.load_state_dict(_safe_torch_load(weight_path, map_location="cpu"))

        if architecture == "resnet50":
            self.backbone = nn.Sequential(*list(base.children())[:-2])
            channels = base.fc.in_features
        else:
            self.backbone = base.features
            channels = 512 if architecture == "vgg16" else (
                base.last_channel if architecture == "mobilenet_v2" else base.classifier[0].in_features
            )
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.pool = nn.Sequential(nn.AvgPool2d(4), nn.Flatten())
        self.head = nn.Sequential(
            nn.Linear(channels, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, len(CLASSES)),
        )

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, images):
        return self.head(self.pool(self.backbone(images)))


def load_checkpoint(path, device="cpu", weights_dir=None):
    """Load an M3/M4 full or head-only checkpoint and return model plus metadata."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = _safe_torch_load(path, map_location="cpu")
    required = {"format", "architecture", "classes", "state_dict"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError(f"{path} is not a supported M3/M4 checkpoint")
    if checkpoint["format"] not in CHECKPOINT_FORMATS:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint['format']}")
    if tuple(checkpoint["classes"]) != CLASSES:
        raise ValueError(f"Unexpected class order: {checkpoint['classes']}")

    architecture = checkpoint["architecture"]
    is_head_only = checkpoint["format"].endswith("head-v1")
    model = DefectClassifier(
        architecture,
        pretrained=is_head_only,
        weights_dir=weights_dir,
    )
    module = model.head if is_head_only else model
    module.load_state_dict(checkpoint["state_dict"], strict=True)
    metadata = {key: value for key, value in checkpoint.items() if key != "state_dict"}
    return model.to(device).eval(), metadata


def default_target_layer(model, architecture):
    """Return the target layer recommended by the M3/M4 model owners."""
    if architecture == "vgg16":
        return model.backbone[28]
    if architecture in {"mobilenet_v2", "resnet50", "mobilenet_v3_small"}:
        return model.backbone[-1]
    raise ValueError(f"No default Grad-CAM target for {architecture}")


def resolve_target_layer(model, path):
    """Resolve a dotted module path such as ``backbone.28``."""
    module = model
    for part in path.split("."):
        if not part:
            raise ValueError(f"Invalid target-layer path: {path}")
        try:
            module = module[int(part)] if part.lstrip("-").isdigit() else getattr(module, part)
        except (AttributeError, IndexError, KeyError, TypeError) as error:
            raise ValueError(f"Cannot resolve target-layer path: {path}") from error
    if not isinstance(module, nn.Module):
        raise ValueError(f"Target-layer path does not resolve to a module: {path}")
    return module
