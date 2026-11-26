from pathlib import Path

import torch
from torch import nn
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

CLASSES = ("Non-defect", "Defect")
IMAGE_SIZE = 224
WEIGHTS = {
    "vgg16": models.VGG16_Weights.IMAGENET1K_V1,
    "mobilenet_v2": models.MobileNet_V2_Weights.IMAGENET1K_V1,
}


def resize_image(image):
    return transforms.functional.resize(image, (IMAGE_SIZE, IMAGE_SIZE), antialias=True)


def image_transform(training=False, resize=True):
    steps = [resize_image] if resize else []
    if training:
        steps.extend([
            transforms.Pad(40, padding_mode="edge"),
            transforms.RandomRotation(15, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(IMAGE_SIZE),
        ])
    return transforms.Compose([
        *steps,
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


class DefectClassifier(nn.Module):
    def __init__(self, architecture, pretrained=False, weights_dir=None):
        super().__init__()
        weights = WEIGHTS[architecture]
        constructor = getattr(models, architecture)
        base = constructor(weights=weights if pretrained and weights_dir is None else None)
        if pretrained and weights_dir is not None:
            filename = weights.url.rsplit("/", 1)[-1]
            base.load_state_dict(torch.load(Path(weights_dir) / filename, map_location="cpu", weights_only=True))
        # Both backbones emit a 7x7 feature map for 224x224 images.
        self.backbone = base.features
        channels = 512 if architecture == "vgg16" else base.last_channel
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.pool = nn.Sequential(nn.AvgPool2d(4), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(channels, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 2))
        for layer in (self.head[0], self.head[3]):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def features(self, images):
        return self.pool(self.backbone(images))

    def forward(self, images):
        # Keep this forward differentiable so a Grad-CAM consumer can hook the backbone.
        return self.head(self.features(images))


def load_checkpoint(path, device="cpu", weights_dir=None):
    """Return an eval-mode model and metadata from a full or head-only checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["format"] not in ("m3-full-v1", "m3-head-v1") or tuple(checkpoint["classes"]) != CLASSES:
        raise ValueError("Unsupported M3 checkpoint format or class order")
    head_only = checkpoint["format"] == "m3-head-v1"
    model = DefectClassifier(checkpoint["architecture"], pretrained=head_only, weights_dir=weights_dir)
    module = model.head if head_only else model
    module.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), {key: value for key, value in checkpoint.items() if key != "state_dict"}
