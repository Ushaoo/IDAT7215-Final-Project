from __future__ import annotations

import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from .data import SplitImageDataset
from .metrics import binary_metrics
from .models import parameter_count


def imagenet_transform(image_size: int = 224, perturbation=None):
    # Apply corruption at the actual model-input resolution. Applying a 5-pixel
    # blur to a 5184-pixel source and then downsampling would nearly erase it.
    steps = [transforms.Resize((image_size, image_size), antialias=True)]
    if perturbation is not None:
        steps.append(perturbation)
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return transforms.Compose(steps)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


@torch.inference_mode()
def evaluate_model(
    model,
    data_root: Path,
    split_csv: Path,
    split: str = "test",
    perturbation=None,
    batch_size: int = 32,
    workers: int = 0,
    device: str = "auto",
    image_size: int = 224,
) -> tuple[dict, list[dict]]:
    torch_device = resolve_device(device)
    dataset = SplitImageDataset(
        data_root=data_root,
        split_csv=split_csv,
        split=split,
        transform=imagenet_transform(image_size, perturbation=perturbation),
        perturbation=None,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch_device.type == "cuda",
    )
    model = model.to(torch_device).eval()
    truth, predicted, predictions = [], [], []
    elapsed = 0.0
    for images, labels, paths in loader:
        images = images.to(torch_device, non_blocking=True)
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
        started = time.perf_counter()
        logits = model(images)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
        elapsed += time.perf_counter() - started
        probabilities = torch.softmax(logits, dim=1)
        batch_pred = probabilities.argmax(dim=1).cpu().tolist()
        batch_labels = labels.tolist()
        truth.extend(batch_labels)
        predicted.extend(batch_pred)
        for path, label, pred, probability in zip(paths, batch_labels, batch_pred, probabilities.cpu().tolist()):
            predictions.append(
                {
                    "path": path,
                    "label": label,
                    "prediction": pred,
                    "prob_non_defect": probability[0],
                    "prob_defect": probability[1],
                }
            )
    metrics = binary_metrics(truth, predicted)
    total_params, trainable_params = parameter_count(model)
    metrics.update(
        {
            "parameter_count": total_params,
            "trainable_parameter_count": trainable_params,
            "inference_seconds": elapsed,
            "inference_ms_per_image": elapsed * 1000 / len(dataset),
            "device": str(torch_device),
            "split": split,
        }
    )
    return metrics, predictions

