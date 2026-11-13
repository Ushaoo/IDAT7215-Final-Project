import pytest

torch = pytest.importorskip("torch")
from torch import nn

from m5.gradcam import GradCAM


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 5, 3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(5, 2)

    def forward(self, images):
        return self.head(self.pool(self.backbone(images)).flatten(1))


def test_gradcam_returns_finite_normalized_input_sized_maps():
    torch.manual_seed(7)
    model = TinyClassifier().eval()
    inputs = torch.randn(2, 3, 16, 12)
    with GradCAM(model, model.backbone[2]) as gradcam:
        maps, logits, classes = gradcam.generate(inputs)
    assert maps.shape == (2, 1, 16, 12)
    assert logits.shape == (2, 2)
    assert classes.shape == (2,)
    assert torch.isfinite(maps).all()
    assert maps.min() >= 0
    assert maps.max() <= 1


def test_gradcam_accepts_explicit_class_and_handles_flat_map():
    model = TinyClassifier().eval()
    for parameter in model.parameters():
        nn.init.zeros_(parameter)
    with GradCAM(model, model.backbone[2]) as gradcam:
        maps, _logits, classes = gradcam.generate(torch.ones(1, 3, 8, 8), 1)
    assert classes.tolist() == [1]
    assert torch.equal(maps, torch.zeros_like(maps))
