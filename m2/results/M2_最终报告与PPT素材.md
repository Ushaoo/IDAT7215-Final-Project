# M2 最终报告：四模型统一评估与鲁棒性

## 完成状态

M2 技术任务已完成：固定数据划分、统一评估、三类扰动、四模型批量推理、逐图预测、效率基准和 PPT 曲线均已生成。

## 数据与方法

- 数据：271 张，Defect 136、Non-defect 135。
- 固定划分：190 train、41 validation、40 test。
- 测试集：Defect 20、Non-defect 20。
- 模型：VGG16、MobileNetV2、ResNet50、MobileNetV3-Small。
- 输入规则：RGB 图片先缩放到 224x224，再施加扰动，最后做 ImageNet 归一化。
- 高斯噪声：sigma=0.05/0.10/0.20。
- 水平运动模糊：kernel=5/9/13。
- 亮度：-40%/-20%/+20%/+40%。
- 主指标：Macro-F1；同时保存 Accuracy、Defect F1、MCC 和混淆矩阵计数。
- 噪声固定种子：2026；同一图片对所有模型使用相同噪声。

## 干净测试集

| 模型 | Accuracy | Macro-F1 | Defect F1 | MCC |
|---|---:|---:|---:|---:|
| VGG16 | 1.000 | 1.000 | 1.000 | 1.000 |
| MobileNetV2 | 1.000 | 1.000 | 1.000 | 1.000 |
| ResNet50 | 1.000 | 1.000 | 1.000 | 1.000 |
| MobileNetV3-Small | 0.950 | 0.950 | 0.952 | 0.905 |

## 鲁棒性结果（Macro-F1）

| 条件 | VGG16 | MobileNetV2 | ResNet50 | MobileNetV3-Small |
|---|---:|---:|---:|---:|
| Clean | 1.000 | 1.000 | 1.000 | 0.950 |
| Noise 0.05 | 0.975 | 0.925 | 0.925 | 0.950 |
| Noise 0.10 | 0.925 | 0.873 | 0.763 | 0.899 |
| Noise 0.20 | 0.387 | 0.774 | 0.601 | 0.747 |
| Blur 5 | 0.950 | 0.975 | 0.975 | 0.975 |
| Blur 9 | 0.950 | 0.950 | 0.950 | 0.925 |
| Blur 13 | 0.925 | 0.925 | 0.950 | 0.873 |
| Brightness -40% | 1.000 | 1.000 | 1.000 | 0.975 |
| Brightness -20% | 1.000 | 1.000 | 1.000 | 0.975 |
| Brightness +20% | 0.975 | 1.000 | 0.975 | 0.975 |
| Brightness +40% | 0.975 | 1.000 | 0.975 | 0.975 |

全部 11 个条件的平均 Macro-F1：MobileNetV2 0.947、MobileNetV3-Small 0.929、ResNet50 0.919、VGG16 0.915。

## 强噪声重点结果

在 sigma=0.20 时：

| 模型 | Macro-F1 | MCC | FP | FN |
|---|---:|---:|---:|---:|
| MobileNetV2 | 0.774 | 0.556 | 6 | 3 |
| MobileNetV3-Small | 0.747 | 0.510 | 3 | 7 |
| ResNet50 | 0.601 | 0.420 | 0 | 14 |
| VGG16 | 0.387 | 0.160 | 19 | 0 |

VGG16 在强噪声下几乎把所有正常图片判为缺陷；ResNet50 则漏掉大量缺陷。两者的错误方向不同，仅报告 Accuracy 或总错误数会掩盖这一差异。

## 统一 CPU 效率基准

相同 Windows CPU、FP32、batch size 1、10 次预热、60 次前向；不包含读图和预处理。

| 模型 | 参数量 | 完整 checkpoint | 平均 ms/image | 中位数 ms | P95 ms |
|---|---:|---:|---:|---:|---:|
| VGG16 | 14.75M | 59.00 MB | 39.94 | 39.82 | 41.40 |
| MobileNetV2 | 2.31M | 9.46 MB | 21.14 | 20.88 | 23.05 |
| ResNet50 | 23.64M | 94.87 MB | 45.47 | 45.36 | 48.70 |
| MobileNetV3-Small | 0.96M | 3.98 MB | 13.37 | 13.28 | 14.82 |

## 最终结论（PPT 可直接使用）

1. **Clean accuracy is not enough.** 三个模型在干净集取得满分，但强噪声下 Macro-F1 分别降至 0.387、0.774 和 0.601。
2. **Gaussian noise is the dominant failure mode.** 模糊和亮度影响较小，所有模型的最差条件都是 noise sigma=0.20。
3. **Lightweight does not necessarily mean less robust.** MobileNetV2 的参数量远低于 VGG16 和 ResNet50，却取得最高的跨条件平均 F1，并在强噪声下最好。
4. **MobileNetV3-Small provides the lowest deployment cost.** 它只有约 0.96M 参数和 3.98 MB，CPU 推理最快，但干净集存在两个误报，强模糊下也下降较多。
5. **Failure direction matters.** 强噪声下 VGG16 主要误报缺陷，ResNet50 主要漏检缺陷；部署风险并不相同。

## 第 11 页建议布局

- 标题：`Clean Accuracy Is Not Enough: Robustness Under Corruptions`
- 主图：`f1_robustness_all_models.png`
- 图旁重点数字：`sigma=0.20: MobileNetV2 0.774 > MobileNetV3-Small 0.747 > ResNet50 0.601 > VGG16 0.387`
- 页脚限制：`40 test images, one split, one training seed and one fixed noise realization.`

## 约 60 秒讲稿

我们在完全相同的 40 张测试图片上，对四个模型加入高斯噪声、运动模糊和亮度变化。VGG16、MobileNetV2 和 ResNet50 在干净图片上都是满分，但这并不代表它们同样可靠。高斯噪声是影响最大的因素。当噪声强度增加到 0.20 时，VGG16 的 Macro-F1 降到 0.387，ResNet50 为 0.601；两个轻量模型更稳定，MobileNetV2 为 0.774，MobileNetV3-Small 为 0.747。进一步检查发现，VGG16 主要把正常件误报为缺陷，而 ResNet50 主要漏掉真实缺陷。相比之下，运动模糊和亮度变化造成的下降较小。结合统一 CPU 测速，MobileNetV2 在精度、鲁棒性和效率之间表现最平衡；MobileNetV3-Small 最快最小，但干净集和强模糊下略有精度损失。

## 限制

- 测试集只有 40 张，单张错误会造成 2.5 个百分点的 Accuracy 变化。
- 仅有一个固定切分、一个训练种子和一个固定噪声实现。
- 未计算多次噪声采样的均值和置信区间。
- 这些结果支持组内公平比较，不应宣称代表所有真实打印或拍摄环境。

