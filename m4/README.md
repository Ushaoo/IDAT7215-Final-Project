# M4 · ResNet50 and MobileNetV3-Small

Run the commands from the repository root with Python 3.12. The training runs use the supplied [fixed split](../data/split.csv) with 190 train, 41 validation and 40 test images. Image bytes are checked against the manifest before training. The 271 full-resolution JPEGs take about 810 MiB. Label 0 means Non-defect, and label 1 means Defect.

```bash
python -m pip install -r m4/requirements.txt
python -m m4.fetch_data --data-root m4_output/images
python -m m4.train --arch resnet50 --data-root m4_output/images --device mps --output m4_output/resnet50
python -m m4.train --arch mobilenet_v3_small --data-root m4_output/images --device mps --output m4_output/mobilenet_v3_small
python -m m4.package_results --output m4_output/report
```

Set `--device cpu` for a CPU run or `--device cuda` on a machine with a compatible NVIDIA GPU. To use locally downloaded ImageNet weights, pass `--weights-dir` with the official Torchvision files `resnet50-0676ba61.pth` and `mobilenet_v3_small-047dcff4.pth`.

## Experiment

Both runs start from official ImageNet V1 weights. Each freezes the convolutional backbone and its batch-normalization statistics. The classification head follows the [reference notebook](../fault_3d_printing_experiment.ipynb): 4×4 average pooling, flatten, a 64-unit ReLU layer, dropout 0.5 and two class scores. PyTorch cross-entropy applies the equivalent log-softmax internally. The port uses 224×224 RGB images, Torchvision ImageNet normalization and training-only rotation up to 15 degrees with edge padding. MobileNetV3-Small uses the same head training method as ResNet50 for an architecture comparison.

The defaults are Adam, learning rate 0.001, batch size 8, 30 epochs and seed 42. The dataset caches only resized RGB images; training rotation is sampled on each visit. Validation defect F1 selects the checkpoint, with validation loss and then earlier epoch breaking ties. The selected full checkpoint is reloaded before evaluating the test set. The F1 positive class is Defect, and confusion-matrix rows are true labels. The output directory contains the run configuration, epoch history, per-image test predictions, aggregate metrics and both checkpoint forms. [Paper](https://doi.org/10.1016/j.eswa.2024.124293)

Latency is measured on prepared batch-size-one inputs already on the device, using FP32, ten warm-ups and sixty synchronized forward passes. It measures model latency on the recorded device. Compare with the other group models under the same device, timing boundary and precision.

## Loading for evaluation and Grad-CAM

```python
from PIL import Image
import torch
from m4.models import image_transform, load_checkpoint

model, metadata = load_checkpoint("m4_output/resnet50/best_full.pt", device="cpu")
with Image.open("example.jpg") as image:
    batch = image_transform()(image.convert("RGB")).unsqueeze(0)
with torch.inference_mode():
    probabilities = model(batch).softmax(dim=1)
```

For the compact `best_head.pt` format, the same loader reconstructs the frozen backbone with the official weights and loads the learned head. Pass `weights_dir` if the official files are already downloaded. Use `model.backbone[-1]` as a starting Grad-CAM target for both models, with gradients enabled and `batch.requires_grad_(True)` for the frozen backbone.

## Delivered results

The branch includes the learned heads for [ResNet50](results/resnet50/best_head.pt) and [MobileNetV3-Small](results/mobilenet_v3_small/best_head.pt). Loading either head with its official backbone reproduced every tensor of the corresponding full checkpoint. CPU inference from the shared head format reproduced all 40 MPS test labels for each model. Each run folder includes the epoch history, test predictions and metric definitions. [Comparison CSV](results/comparison.csv) · [Slide 8–9 materials](results/slides_08_09.md)

Implementation in this folder is the group's M4 training and evaluation workflow. The input split is supplied by M2, the backbones and pretrained weights are from Torchvision, and the head layout and rotation range follow the linked AM3d reference. [ResNet50 weights](https://docs.pytorch.org/vision/0.22/models/generated/torchvision.models.resnet50.html) · [MobileNetV3-Small weights](https://docs.pytorch.org/vision/0.22/models/generated/torchvision.models.mobilenet_v3_small.html)
