
# **Defect Analysis of 3D Printed Objects Using Transfer Learning Approaches**

Group 6 M4 results and checkpoint loading instructions are available in [M4 training and evaluation](m4/README.md). [Slide 8–9 materials](m4/results/slides_08_09.md)

## **Abstract**

Additive manufacturing (AM) is rapidly gaining traction across diverse industries, including healthcare, aerospace, and automotive, due to its ability to enhance production efficiency. However, a critical challenge in AM is the early detection of defects, which can significantly reduce production costs and improve productivity. Addressing this need, our study explored the effectiveness of machine learning (ML) approaches, focusing on transfer learning (TL) models, for defect detection in 3D-printed objects. We utilized a range of TL models such as Visual Geometry Group (VGG)-16, Inception-Residual Network (InceptionResNet)-V2, Residual Network (ResNet)-50, Mobile Network (MobileNet)-V2, VGG19, ResNet101, Extreme Inception (Xception), Efficient Network (EfficientNet)-B0, EfficientNetB7, EfficientNetV2M (EfficientNetV2M), and Neural Architecture Search Network Large (NASNetLarge), analyzing images of 3D-printed objects across three datasets using various statistical measures.

An ablation study was also conducted using Adaptive Moment Estimation (Adam), Stochastic Gradient Descent (SGD), and Root Mean Square Propagation (RMSprop) optimizers. We identified that TL models such as VGG16, MobileNetV2, InceptionResNetV2, and NASNetLarge significantly outperform others, especially when optimized with Adam and RMSprop. Conversely, models like EfficientNetB0, EfficientNetB7, and EfficientNetV2M exhibited lower performance when paired with the SGD optimizer. Additionally, Local Interpretable Model-agnostic Explanations (LIME)-based approaches were used to provide explanations of the models' predictions. These findings offer substantial insights into model optimization and integration, aiming to enhance AM product quality through advanced image-based monitoring and inspection.

<p align="center">
  <img src="https://github.com/mahsan2/AM3d/blob/main/figure/Proposedmethods3.PNG" alt="Flow diagram of the proposed transfer learning-based approaches with frozen pre-trained weights and updated layers for defect analysis">
  <br><strong>Flow diagram of the proposed transfer learning-based approaches with frozen pre-trained weights and updated layers for defect analysis</strong>
</p>



## **Citation**

If you use this dataset, please cite the following paper:

```bibtex
@article{ahsan2024defect,
  title={Defect analysis of 3D printed object using transfer learning approaches},
  author={Ahsan, Md Manjurul and Raman, Shivakumar and Liu, Yingtao and Siddique, Zahed},
  journal={Expert Systems with Applications},
  volume={253},
  pages={124293},
  year={2024},
  publisher={Elsevier}
}
```

**Paper link:** [https://doi.org/10.1016/j.eswa.2024.124293](https://doi.org/10.1016/j.eswa.2024.124293)

**Dataset link:** https://github.com/mahsan2/AMdataset
