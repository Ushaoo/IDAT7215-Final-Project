# M3 · VGG16 and MobileNetV2 · 许哲熙

This folder contains the M3 training code, measured results and learned classification heads. It is adapted from the group M4 workflow; attribution and changes are recorded in [来源与修改说明](来源与修改说明.md).

## Measured results

Group fixed split: 190 training, 41 validation and 40 test images. Label 0 is Non-defect; label 1 is Defect.

| Model | Test accuracy | Defect F1 | MCC | Parameters | Full checkpoint MB | Mean forward ms |
|---|---:|---:|---:|---:|---:|---:|
| VGG16 | 100% | 1.0000 | 1.0000 | 14,747,650 | 59.00 | 23.17 |
| MobileNetV2 | 100% | 1.0000 | 1.0000 | 2,305,986 | 9.46 | 8.02 |

Both ran for 30 epochs, Adam lr=0.001, batch size 8, seed 42. The best checkpoint was selected by validation Defect F1, then validation loss; both selected epoch 30. Full saved models were reloaded before test evaluation. Latency was measured on the same M2 Mac using MPS, FP32, batch size 1, ten warm-ups and sixty synchronized forwards, excluding loading and preprocessing. These results cover only this 40-image test split and one seed.

[Comparison CSV](results/comparison.csv) · [Confusion matrices](results/confusion_matrices.png) · [Training curves](results/training_curves.png) · [Efficiency](results/efficiency.png) · [PPT素材与讲稿](results/PPT结果素材与讲稿.md)

## Load the shared checkpoints

Run from the repository root after installing `m3/requirements.txt`. The shared files contain the learned head; the frozen backbone is reconstructed with the specified official Torchvision ImageNet V1 weights. First loading requires network access or cached weights.

```python
from PIL import Image
import torch
from m3.models import load_checkpoint, image_transform

model, metadata = load_checkpoint('m3/results/vgg16/best_head.pt', device='cpu')
with Image.open('example.jpg') as image:
    batch = image_transform()(image.convert('RGB')).unsqueeze(0)
with torch.inference_mode():
    p_defect = model(batch).softmax(1)[:, 1]
```

For MobileNetV2 use `m3/results/mobilenet_v2/best_head.pt`. An optional `weights_dir` can point to the official weight files `vgg16-397923af.pth` and `mobilenet_v2-b0353104.pth`. Full checkpoints remain available in the local M3 handoff ZIP; the full checkpoint sizes and hashes in metrics.json refer to those artifacts, not the small shared head files.

Grad-CAM: enable gradients and set `batch.requires_grad_(True)` because the backbone is frozen. Suggested starting target: `model.backbone[28]` for VGG16 and `model.backbone[-1]` for MobileNetV2. Use the transform provided above and class index 1 for Defect.

## Reproduce

Use a separate Python 3.11–3.13 environment. From the repository root:

```bash
python -m pip install -r m3/requirements.txt
python -m m3.fetch_data --data-root m3_output/images
python -m m3.train --arch vgg16 --data-root m3_output/images --device mps --output m3_output/vgg16
python -m m3.train --arch mobilenet_v2 --data-root m3_output/images --device mps --output m3_output/mobilenet_v2
python -m m3.package_results --source m3_output --output m3_output/report
```

Each training and report output directory must be new. Use `--device cpu` without an Apple GPU or `--device cuda` on a compatible NVIDIA machine. Keep the repository's existing `data/split.csv` unchanged. The split SHA256 is `f7efba67d4802aa9feaca11cad5e39cd8d348c955148910a1be738f55e991a12`.

No raw image dataset, environment, official downloaded backbone files or duplicated ZIP is included in this folder. The backbone and preprocessing implementation follows the M4 PyTorch port, so this is an adapted reproduction rather than an identical Keras rerun. Robustness and Grad-CAM analyses are assigned to the other group members.
