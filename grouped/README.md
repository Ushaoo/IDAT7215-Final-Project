# Grouped transfer-learning experiment

This directory contains the November 26 experimental results and reproduction instructions for IDAT7215 Group 6, Topic 4. The task is image-level defect classification. Label 1 means defective and label 0 means non-defective.

## What was reproduced

The upstream [AM3d repository](https://github.com/mahsan2/AM3d/tree/dd1025d7180d91f1128c95817b4d3dd268626b1e) contains 413 files. All 413 Git blob hashes match our starting main commit `8902bcc`, which adds the group's split.csv. The first evaluation used the five team branches M2 `8378f83`, M3 `fd8e55d`, M4 `fffb74c`, M5 `3914906` and M8 `532834e`. Their source history is included in this experimental branch.

[Ahsan et al. (2024)](https://doi.org/10.1016/j.eswa.2024.124293) compared 11 transfer-learning models across three studies. Table 1 reports 272 laboratory images, 3,017 expanded laboratory images and 1,557 external images. The public [AMdataset snapshot](https://github.com/mahsan2/AMdataset/tree/52fe75dec70c1e8046493b4e3256f8f752a4d39d) has 271 raw JPGs (135 normal, 136 defect) and 3,017 augmented PNGs (2,070 normal, 947 defect). We verified all 271 raw SHA-256 hashes against the team's original manifest. All lab images and derivatives are training-only in the new mixed-source experiments. The Kaggle source contains 798 normal and 759 defective images.

The original Keras notebook basic_algorithm.ipynb uses a seeded image-level 70% training split and splits the remainder approximately 1:2 between validation and test. It uses frozen pretrained features, 4×4 average pooling and a small dense classification head. The group's PyTorch adaptation uses Torchvision ImageNet V1 weights, 224×224 RGB inputs, ImageNet normalization, a 64-unit ReLU head and dropout 0.5. For this experiment all cached representations use deterministic resizing, and no online random rotation is applied. We use Adam, learning rate 0.001, batch size 64, 30 epochs and validation Macro-F1 with minimum loss to resolve ties. Comparisons below cover four team architectures, three from the reference paper plus MobileNetV3-Small. Table 8 of the paper reports Study Three Adam test accuracy of 0.94 for VGG16 and 0.80 for ResNet50. Different sample ordering, initialization, preprocessing, training and evaluation prevent a direct superiority claim.

## Split and controls

The [manifest](results/split.csv) records source, path, label, file hash, augmentation status and all four fold roles. Each test fold holds out one whole defect prefix and one whole normal prefix from Justin Ruan's Kaggle data. The next pair cyclically forms validation. The other two pairs form the Kaggle portion of training. Across folds each Kaggle image has one out-of-fold test prediction per model, arm and seed. All 4,845 file hashes and all 4,845 resized RGB pixel hashes are distinct. The prefixes identify filename families. Physical print or production-run identifiers were unavailable. The group pairing had already been explored in the pilot, so the evaluation has no untouched third-site confirmation.

| Fold | Held-out Kaggle families | All training | Validation | Test |
|---|---|---:|---:|---:|
| 0 | bed_not_stick / scratch | 4,071 | 496 | 278 |
| 1 | leg_broken / scratch_2 | 4,043 | 306 | 496 |
| 2 | no_bottom / scratch_3 | 4,062 | 477 | 306 |
| 3 | no_support / scratch_4 | 4,090 | 278 | 477 |

The Kaggle-only reference controls use the original head on a seeded random image split (1,089/159/309) and on the four grouped splits. Moving to the grouped protocol changes both training size and test composition. The comparison measures sensitivity to protocol. The mixed-source ablation then holds groups, epochs, update budget, optimizer, seed and checkpoint-selection rule constant within each fold. Raw-only training repeats shuffled passes to match the larger arm's exposure count.

1. **Raw** uses 271 raw lab images and the Kaggle training groups with the team's original pooling.
2. **Expanded** adds all 3,017 lab augmentation files under the same pooling and training budget.
3. **Global pooling** averages every position of the 7×7 map. The earlier 4×4 pooling with stride 4 on a 7×7 map outputs only the first 4×4 window. Dense-layer dimensions stay fixed.
4. **Global pooling with balanced loss** gives equal total weight to each domain, class within domain and training family within class. The two lab folders for each class share one family. All expanded training images are visited each epoch.

Every arm trains a freshly initialized classification head on the frozen backbone. Each grouped result averages the two-class Macro-F1 over the four test folds equally, then over seeds 42, 43 and 44. The primary unit for comparing performance is the held-out family pair. Repeated seeds measure optimization variability on those same pairs.

## Results

| Backbone | Kaggle random reference | Kaggle grouped reference | Raw mixed | Expanded mixed | Global pooling | Global + balanced |
|---|---:|---:|---:|---:|---:|---:|
| VGG16 | .963 | .447 | .516 | .517 | **.635** | .625 |
| MobileNetV2 | .987 | .324 | .359 | .354 | **.530** | .515 |
| ResNet50 | .936 | .274 | .365 | .438 | .520 | **.570** |
| MobileNetV3-Small | .971 | .569 | .563 | .548 | .565 | **.580** |

The matched global-pooling change improves average Macro-F1 for every architecture over the expanded control. Adding augmentation alone produces mixed results. Balancing helps ResNet50 and MobileNetV3-Small beyond global pooling, and weakens the other two averages. VGG16 with global pooling scores .053, .989, .525 and .971 across folds. The weakest fold contains the unseen bed-adhesion failure and the scratch normal family. Fold-level results limit deployment claims. Macro-F1 and error costs matter together. For ResNet50, the final arm gives mean defect recall .634 and false-positive rate .397 versus .486 and .551 for the raw mixed arm.

[Primary metrics](results/summary.csv) · [Per-fold scores](results/fold_summary.csv) · [Ablation figure](results/ablation.png) · [Reference runs](results/reference_metrics.csv)

## Robustness and explanations

For seed 42 we replayed all 64 saved fold-specific heads on the original held-out RGB images. Labels matched the cached clean predictions for every architecture, fold and arm. We then applied M2's deterministic Gaussian noise (sigma .20), horizontal motion blur (kernel 13), and brightness factors .6 and 1.4. Under global pooling plus balanced loss, clean and strong-noise Macro-F1 averaged .601 and .341 for ResNet50, .588 and .414 for MobileNetV3-Small, .526 and .336 for MobileNetV2, and .619 and .305 for VGG16. These seed-42 values and the three-seed primary summaries have different denominators. [All stress conditions and arms](results/stress_summary.csv) · [Measured CPU efficiency](results/efficiency.csv)

Each architecture has a Grad-CAM grid of the median numbered normal and defective frame from each held-out fold. All panels explain the defect score. New-panel human ratings are pending. The efficiency file reports seed-42 global/weighted model forward latency on one Mac CPU using four threads, FP32, one prepared image, ten warm-up and sixty timed runs. The earlier Windows benchmark covers different hardware.

## Paired human review

The new review kit uses 20 labelled defect cases, five from each held-out Kaggle fold. A seeded sampler draws one image from each of five consecutive filename bands in a fold. Selection uses the manifest and filename order. For each case there are eight anonymous panels, one for each architecture and pooling arm. The 160 panels compare the expanded-data control with whole-map pooling. Both arms use the same expanded training data, loss and seed-42 checkpoint selection. Grad-CAM targets the defect score even when the model predicts normal.

Use one independently shuffled ZIP per reviewer. The coordinator holds the private ID key until all three forms have been returned. Reviewers evaluate the overlap of the strongest highlights and a visible defect with yes, no or unsure. Three completed forms feed the summary code. It reports vote counts and denominators, majority coverage by model and fold, unresolved cases, paired outcomes, and pairwise agreement including Cohen's kappa. A resolved majority requires at least two yes votes or at least two no votes. Human ratings are pending. The sample represents visual overlap on the 20 selected defect cases. Classification accuracy and recall use the full out-of-fold test predictions.

Generate a fresh kit locally from the repository root with the prepared image cache and official backbones. The generator creates new random panel IDs on every run. Use forms and key from the same run.

```sh
python -m grouped.blind_review.generate grouped/results "$cache" "$data" /path/to/new-kit
python -m grouped.blind_review.aggregate /path/to/new-kit/coordinator_only/blind_key.csv ratings_A.csv ratings_B.csv ratings_C.csv /path/to/new-summary
```

The team coordinator distributes only the three reviewer ZIPs. The kit and identity key live outside Git.

## Original responsibilities and new results for review

The November 26 experiment was implemented and run in this branch with Codex assistance at M4's request. The following assignments identify the owners of the existing workflows and natural reviewers for the extension.

| Existing role | New material available for the member to review |
|---|---|
| M1 · scope and conclusions | Paper comparison, scope and limitations above |
| M2 · data and robustness | New source manifests, grouped folds, out-of-fold predictions and stress-test results using the M2 perturbation functions |
| M3 · VGG16 and MobileNetV2 | Both architectures' four arms, head checkpoints and per-fold errors using the M3 backbone code |
| M4 · ResNet50 and MobileNetV3-Small | Both architectures' four arms, pooling change and domain/class/family weighting |
| M5 · explanations | Held-out Grad-CAM examples produced with the M5 implementation, using the defect target and the fold's seed-42 balanced head |
| M6 · presentation and efficiency | Same-machine CPU latency and parameter counts for the new balanced models |
| M7 · text integration | Verified results and limitations for later integration |
| M8 · figures and review | Reproducible ablation figure, Grad-CAM panels and source CSVs. The previous 80-panel human ratings apply to the original lab models only |

## Reproduction

Use Python 3.12 from the repository root. Install `grouped/requirements.txt`. The frozen feature pass used the Mac MPS device, and the heads ran on CPU. CPU feature extraction also works and can yield small numerical differences. Put data and caches outside Git. Allow several GB for weights, archives, prepared images and features.

```sh
python -m pip install -r grouped/requirements.txt
data=../idat_grouped_data
cache=../idat_grouped_cache
out=../idat_grouped_rerun
mkdir -p "$data" "$cache" "$out"
curl -fL https://codeload.github.com/mahsan2/AMdataset/zip/52fe75dec70c1e8046493b4e3256f8f752a4d39d -o "$data/smis_source.zip"
curl -fL https://www.kaggle.com/api/v1/datasets/download/justin900429/3d-printer-defected-dataset -o "$data/kaggle_source.zip"
for name in vgg16-397923af mobilenet_v2-b0353104 resnet50-0676ba61 mobilenet_v3_small-047dcff4; do
  curl -fL "https://download.pytorch.org/models/$name.pth" -o "$data/$name.pth"
done
python -m grouped.prepare "$data" "$cache" "$out"
python -m grouped.features "$data" "$cache" --device mps
python -m grouped.train "$cache" "$out" mobilenet_v3_small mobilenet_v2 resnet50 vgg16
python -m grouped.reference "$cache" "$out"
python -m grouped.summarize "$out"
python -m grouped.diagnostics "$data" "$cache" "$out" --device mps
python -m grouped.check_results "$out"
```

The expected manifest SHA-256 is `e3640d293ab43505beddaede1533022c77eed40c5c4474f92659f7bff67e331b`. Each saved head belongs to one architecture, arm, fold and seed (42). Load the matching official backbone, pooling and head as in diagnostics.py. Source data archives and pretrained backbones are retrieved from the respective providers. Kaggle examples in the Grad-CAM figures are resized images with heatmap overlays from Justin Ruan's [3D-Printer Defected Dataset v1](https://www.kaggle.com/datasets/justin900429/3d-printer-defected-dataset), licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The corresponding derived panels are distributed under that license.
