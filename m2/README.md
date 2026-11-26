# M2 · Data, Unified Evaluation and Robustness · 范雨馨

This folder contains the fixed data split audit, shared evaluation code and completed robustness analysis for all four group models.

## Completed outputs

- Dataset: 271 images; 190 train, 41 validation, 40 test.
- Models: VGG16, MobileNetV2, ResNet50 and MobileNetV3-Small.
- Corruptions: Gaussian noise (0.05/0.10/0.20), horizontal motion blur (5/9/13), brightness (-40%/-20%/+20%/+40%).
- Metrics: Accuracy, Macro-F1, Defect F1, MCC, confusion counts and per-image predictions.
- Efficiency: common Windows CPU, FP32, batch size 1, 10 warm-ups and 60 timed forwards.

The primary result files are in `m2/results/`:

- `M2_最终报告与PPT素材.md`
- `robustness_results_all_models.csv`
- `predictions_all_models.csv`
- `efficiency_cpu.csv`
- `f1_robustness_all_models.png` and `.pdf`

## Main result

Clean accuracy alone did not predict robustness. At Gaussian noise sigma=0.20, Macro-F1 was 0.774 for MobileNetV2, 0.747 for MobileNetV3-Small, 0.601 for ResNet50 and 0.387 for VGG16. Motion blur and brightness caused substantially smaller changes.

## Method

Each image is converted to RGB, resized to the actual 224x224 model input, corrupted, converted to a tensor and ImageNet-normalized. Applying corruption after resize keeps the pixel-based blur kernels comparable and prevents a small kernel from disappearing during downsampling of the 5184x3456 originals. Gaussian noise is deterministic from the image bytes and seed 2026, so every model receives the same corrupted image.

The shared split is `data/split.csv`. Label 0 is Non-defect and label 1 is Defect. M3 checkpoints and loaders come from branch `m3/vgg16-mobilenetv2`; M4 inputs come from branch `m4/resnet50-mobilenetv3`. Only the small learned heads are stored in those branches; official Torchvision ImageNet V1 backbones are reconstructed on load.

## Reproduce

Use Python with the M3 and M4 requirements installed. After the M3 and M4 folders are available locally:

```bash
python m2/run_m3_robustness.py --m3-repo . --data-root /path/to/dataset --split-csv data/split.csv --output-dir m2/results/m3
python m2/run_m4_robustness.py --m4-repo . --data-root /path/to/dataset --split-csv data/split.csv --output-dir m2/results/m4
```

Combine the two result CSVs, then plot:

```bash
python m2/plot_robustness.py --results m2/results/robustness_results_all_models.csv --output m2/results/f1_robustness_all_models.png
```

Run the common CPU latency benchmark:

```bash
python m2/benchmark_all_models.py --m3-repo . --m4-repo . --data-root /path/to/dataset --split-csv data/split.csv --output m2/results/efficiency_cpu.csv
```

## Limitations

The test set contains only 40 images. Results use one fixed split, one training seed and one deterministic noise realization. They support fair comparison within this project but should not be presented as a general deployment guarantee.

