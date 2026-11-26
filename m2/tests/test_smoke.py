from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from am_m2.data import assign_splits, collect_samples, write_split_csv
from am_m2.evaluation import evaluate_model
from am_m2.metrics import binary_metrics
from am_m2.perturbations import Brightness, GaussianNoise, MotionBlur


class TinyModel(nn.Module):
    def forward(self, images):
        score = images.mean(dim=(1, 2, 3))
        return torch.stack([-score, score], dim=1)


class SmokeTest(unittest.TestCase):
    def test_metrics(self):
        result = binary_metrics([0, 0, 1, 1], [0, 1, 1, 1])
        self.assertAlmostEqual(result["accuracy"], 0.75)
        self.assertGreater(result["mcc"], 0)

    def test_perturbations_and_pipeline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for class_name, value in (("Non-defect", 30), ("Defect", 220)):
                folder = root / class_name
                folder.mkdir()
                for index in range(10):
                    Image.fromarray(np.full((24, 24, 3), value + index, dtype=np.uint8)).save(folder / f"{index}.png")
            image = Image.open(root / "Defect" / "0.png")
            self.assertEqual(GaussianNoise(0.1)(image).size, image.size)
            self.assertEqual(MotionBlur(13)(image).size, image.size)
            self.assertEqual(Brightness(0.8)(image).size, image.size)
            rows = assign_splits(collect_samples(root), (0.7, 0.15, 0.15), 2026)
            split_csv = root / "split.csv"
            write_split_csv(rows, split_csv)
            metrics, predictions = evaluate_model(
                TinyModel(), root, split_csv, split="test", batch_size=4, image_size=32
            )
            self.assertEqual(metrics["n"], len(predictions))
            self.assertIn("f1_macro", metrics)


if __name__ == "__main__":
    unittest.main()

