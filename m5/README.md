# M5 · Grad-CAM analysis · 夏立原

This folder generates Grad-CAM panels for the shared M3/M4 PyTorch checkpoints,
creates a blinded three-rater review bundle, and reveals the model identities only
when completed reviews are aggregated.

## Scope and interpretation

- Class order is fixed as `Non-defect=0`, `Defect=1`.
- The default sample contains up to 20 ground-truth Defect test images. A
  localization-success denominator is meaningful only when a defect exists to be
  localized. Add `--include-non-defect` for a separate qualitative attention audit;
  those images are excluded from the success-rate denominator.
- Misclassified images are selected first when available. If every submitted model
  is correct on the fixed test split, the tool reports only correct cases rather
  than manufacturing error cases.
- The default heatmap explains the model's predicted class. Use
  `--target-mode defect` for a consistent Defect-class sensitivity analysis.
- Grad-CAM is an attribution visualization, not a defect segmentation mask. The
  reported localization rate is a human-review result and must be described as such.

## Install

Use Python 3.11 or newer from the repository root:

```bash
python -m pip install -r m5/requirements.txt
```

Head-only checkpoints reconstruct their frozen backbone from official Torchvision
ImageNet V1 weights. The first run downloads those weights unless they are cached.
For an offline run, put the official files in one directory and pass
`--weights-dir`. Expected filenames are:

- `vgg16-397923af.pth`
- `mobilenet_v2-b0353104.pth`
- `resnet50-0676ba61.pth`
- `mobilenet_v3_small-047dcff4.pth`

## Generate heatmaps and the blind-review bundle

Run after the M3/M4 checkpoint folders are available in the working tree:

```bash
python -m m5.generate_gradcam \
  --model VGG16=m3/results/vgg16/best_head.pt \
  --model MobileNetV2=m3/results/mobilenet_v2/best_head.pt \
  --model ResNet50=m4/results/resnet50/best_head.pt \
  --model MobileNetV3-Small=m4/results/mobilenet_v3_small/best_head.pt \
  --data-root /path/to/AMdataset/dataset \
  --split-csv data/split.csv \
  --output /path/to/new/m5_output
```

The output directory must be empty. The generator checks every image against the
fixed split SHA-256 by default, uses the same 224×224 ImageNet normalization as M3
and M4, and chooses the target layers recommended by the model owners:

- VGG16: `backbone.28`
- MobileNetV2, ResNet50, MobileNetV3-Small: `backbone.-1`

Use `--target-layer MODEL=MODULE.PATH` only for an explicitly documented
sensitivity check.

### Output separation

- `blind/images/`: anonymous panels in the order Original, Grad-CAM, Overlay.
- `blind/review_M1.csv`, `review_M5.csv`, `review_M8.csv`: independent blank forms.
- `private/manifest.csv`: the identity key. Do not share it before scoring ends.
- `private/images/`: labeled panels for analysis and PPT selection after unblinding.
- `private/run_summary.json`: sampling and run metadata.

For genuine blinding, M8 or another non-rater should run the final generation and
retain `private/`. Each rater receives only `blind/images/` and their own CSV, fills
`covers_defect` with `yes/no` (or `1/0`), and must not see another rater's answers.
Each rater's CSV uses a different deterministic random order so the model blocks
cannot be inferred from row order.

## Aggregate completed reviews

```bash
python -m m5.summarize_reviews \
  --manifest /path/to/m5_output/private/manifest.csv \
  --review M1=/path/to/completed_M1.csv \
  --review M5=/path/to/completed_M5.csv \
  --review M8=/path/to/completed_M8.csv \
  --output /path/to/new/m5_summary
```

This writes item-level votes, `localization_success.csv`, and a PPT-ready bar chart.
The primary rate uses majority vote across the available independent ratings and
only ground-truth Defect images. The output also reports completion and unanimity so
missing reviews or disagreement remain visible.

## Test

```bash
python -m pytest m5/tests -q
```

## Attribution

The checkpoint reconstruction and preprocessing contract are adapted from the
group's M3 and M4 training branches. The backbones and pretrained weights are from
Torchvision. The Grad-CAM implementation, blinded material generation, sampling,
review aggregation and documentation in `m5/` are the M5 contribution.
