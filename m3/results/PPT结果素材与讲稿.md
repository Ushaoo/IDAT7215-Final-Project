# M3 result-page materials (generated from measured results)

## Page 1: How we trained the two models

- VGG16 and MobileNetV2 use official Torchvision ImageNet V1 weights.
- Frozen feature extractors; 4x4 average pooling, flatten, 64-unit ReLU, dropout 0.5, two logits.
- Adam, learning rate 0.001, batch size 8, 30 epochs, seed 42.
- Group fixed split: 190 train, 41 validation, 40 test. Defect is positive (label 1).
- Best validation defect F1 selects the checkpoint; validation loss breaks ties.
- The full saved model is reloaded before final test evaluation.

## Page 2: Results on the fixed test split

| Model | Accuracy | Defect F1 | MCC | Parameters | Full model MB | Mean latency ms |
|---|---:|---:|---:|---:|---:|---:|
| VGG16 | 100.0% | 1.0000 | 1.0000 | 14,747,650 | 59.00 | 23.17 |
| MobileNetV2 | 100.0% | 1.0000 | 1.0000 | 2,305,986 | 9.46 | 8.02 |

VGG16: 40/40 correct; 0 missed defects; 0 false alarms; selected epoch 30.
MobileNetV2: 40/40 correct; 0 missed defects; 0 false alarms; selected epoch 30.

Figures: confusion_matrices.png and training_curves.png.

## Page 3: Accuracy, size and latency

Figure: efficiency.png. Use the measured table above to describe the observed trade-off.
Latency device: mps; FP32, batch size 1, 10 warm-ups and 60 synchronized forwards; excludes file loading and preprocessing.
Cross-group speed comparisons must use the same hardware and timing boundaries.
Limitations: only 40 test images, one seed and one fixed split; no claim of general deployment reliability.
Robustness and Grad-CAM are evaluated separately by the designated teammates.
These are the group's measured results, not the original paper's reported scores.

## Short spoken explanation

I trained VGG16 and MobileNetV2 using the same fixed data split as the other group models. Both started from ImageNet weights. I froze the feature extractors and trained a small classification head for thirty epochs. I used the validation set to select the best checkpoint, then reloaded it before evaluating the test images.
VGG16: 40/40 correct; 0 missed defects; 0 false alarms; selected epoch 30.
MobileNetV2: 40/40 correct; 0 missed defects; 0 false alarms; selected epoch 30.
The table also shows the total parameters, saved model size and single-image forward time. These measurements help us compare classification performance with computational cost. However, the test set contains only forty images, so the results apply to this particular split. Our teammates will test noise, blur and brightness changes, and inspect the trained models using Grad-CAM.
