from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageEnhance


def _to_rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


@dataclass(frozen=True)
class GaussianNoise:
    sigma: float
    seed: int = 2026

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.sigma < 0:
            raise ValueError("sigma must be non-negative")
        array = _to_rgb_array(image)
        digest = hashlib.sha256(image.convert("RGB").tobytes()).digest()
        image_seed = int.from_bytes(digest[:8], "little") ^ self.seed
        rng = np.random.default_rng(image_seed)
        noisy = np.clip(array + rng.normal(0.0, self.sigma, array.shape), 0.0, 1.0)
        return Image.fromarray(np.round(noisy * 255).astype(np.uint8), mode="RGB")


@dataclass(frozen=True)
class MotionBlur:
    kernel_size: int

    def __call__(self, image: Image.Image) -> Image.Image:
        k = int(self.kernel_size)
        if k < 1 or k % 2 == 0:
            raise ValueError("motion blur kernel_size must be a positive odd integer")
        if k == 1:
            return image.convert("RGB")
        array = _to_rgb_array(image)
        pad = k // 2
        padded = np.pad(array, ((0, 0), (pad, pad), (0, 0)), mode="reflect")
        cumsum = np.cumsum(padded, axis=1, dtype=np.float64)
        cumsum = np.concatenate([np.zeros_like(cumsum[:, :1]), cumsum], axis=1)
        blurred = (cumsum[:, k:] - cumsum[:, :-k]) / k
        return Image.fromarray(np.round(np.clip(blurred, 0.0, 1.0) * 255).astype(np.uint8), mode="RGB")


@dataclass(frozen=True)
class Brightness:
    factor: float

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.factor < 0:
            raise ValueError("brightness factor must be non-negative")
        return ImageEnhance.Brightness(image.convert("RGB")).enhance(self.factor)


def make_perturbation(kind: str, value: float, seed: int = 2026):
    kind = kind.lower().strip()
    if kind in {"none", "baseline", "clean"}:
        return None
    if kind in {"noise", "gaussian_noise"}:
        return GaussianNoise(float(value), seed=seed)
    if kind in {"blur", "motion_blur"}:
        return MotionBlur(int(value))
    if kind in {"brightness", "bright"}:
        return Brightness(float(value))
    raise ValueError(f"unknown perturbation kind: {kind}")

