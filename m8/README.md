# IDAT7215 Topic 4: Defect Detection in 3D-Printed Objects via Transfer Learning

Group project for **IDAT7215 — Computer Programming for Product Development and Applications** (2026/27 Semester 1).

Reproduction of the paper:

> Ahsan, M. et al., *"Defect analysis of 3D printed object using transfer learning approaches."*
> Expert Systems with Applications (2024). DOI: [10.1016/j.eswa.2024.124293](https://doi.org/10.1016/j.eswa.2024.124293)

Reproduction study; full author list is in the submitted report.

## 1. Overview

One-line project summary: we reproduce the transfer-learning defect-detection pipeline from the paper on the AMdataset (3D-printed cylinders: good vs. bad), and extend it with three increments:

1. **Robustness evaluation** — Macro-F1 under 10 image corruptions (Gaussian noise σ = 0.05 / 0.1 / 0.2, motion blur kernel = 5 / 9 / 13, brightness ×0.6 / 0.8 / 1.2 / 1.4) against the clean baseline, for all four models on the same 40-image test split.
2. **Blind-review quantification** — a three-person blind review quantifies whether Grad-CAM heatmaps actually localise the defect region, with inter-rater agreement statistics.
3. **Lightweight model comparison** — a small footprint model (MobileNetV3-Small) benchmarked against the paper's representative models (VGG16, MobileNetV2, ResNet50) on accuracy, latency and size.

## 2. Repository Structure

```text
.
├── README.md                        # upstream paper README (unmodified)
├── data/
│   ├── split.csv                    # fixed train/val/test split (271 images, sha256 per image)
│   ├── efficiency_cpu.csv           # M2 unified CPU benchmark (params / latency / checkpoint)
│   └── robustness_results_all_models.csv   # M2: 4 models x 11 conditions
├── m2/                              # M2 — data card, evaluation, robustness, CPU benchmark
│   ├── am_m2/                       #   package: data / models / metrics / perturbations / evaluation
│   ├── split_dataset.py             #   builds data/split.csv (sha256 contract)
│   ├── run_m3_robustness.py         #   robustness harness for the M3 pair
│   ├── run_m4_robustness.py         #   robustness harness for the M4 pair
│   ├── benchmark_all_models.py      #   unified CPU latency/parameter/checkpoint benchmark
│   └── plot_robustness.py           #   M2's own robustness figure
├── m3/                              # M3 — VGG16 / MobileNetV2
│   ├── fetch_data.py, data.py, models.py
│   ├── train.py                     #   --arch vgg16 | mobilenet_v2
│   └── package_results.py
├── m4/                              # M4 — ResNet50 / MobileNetV3-Small
│   ├── fetch_data.py, data.py, models.py
│   ├── train.py                     #   --arch resnet50 | mobilenet_v3_small
│   └── package_results.py
├── m5/                              # M5 — Grad-CAM + blind-review bundle
│   ├── gradcam.py, modeling.py, data.py
│   ├── generate_gradcam.py          #   heatmaps -> blind/ + private/manifest.csv
│   └── summarize_reviews.py         #   M5's own review aggregation
└── m8/                              # M8 — report figures, blind-review statistics, reproducibility
    ├── README.md                    #   this file (project readme)
    ├── requirements.txt             #   consolidated environment (torch stack)
    ├── plot_data_card.py            #   p05 data card (downloads + SHA-256 verifies sample images)
    ├── plot_efficiency.py           #   p10 — reads m8/data/efficiency_cpu.csv
    ├── plot_robustness.py           #   p11 — reads m8/data/robustness_results_all_models.csv
    ├── plot_gradcam_panel.py        #   p12 — reads m8/results/per_image_votes.csv + m8/panels/
    ├── blind_review/                #   independent blind-review toolkit (statistics used for p13)
    │   ├── make_blind_sheets.py     #     manifest -> 3 blind sheets + secret key
    │   ├── stats_blind_review.py    #     filled sheets -> localization rate / agreement / p13
    │   └── heatmap_manifest_template.csv
    ├── data/                        #   copies of M2's result CSVs (see data/PROVENANCE.md)
    ├── results/                     #   statistics tables, unblinding key, filled review sheets, p13
    ├── panels/                      #   the 8 blinded heatmaps shown on slide 12
    ├── figures/                     #   p05 / p10 / p11 / p12 / p13
    └── M8_individual_contribution_statement.md
```

## 3. Environment Setup

Python 3.10+ with a single PyTorch stack for all four models (M2–M5 use `torch` / `torchvision`; no TensorFlow code is used).

```bash
python -m pip install -r m8/requirements.txt
```

- M2–M5 each ship their own narrower `requirements.txt` inside their branch (`m2/`, `m3/`, `m4/`, `m5/`); `m8/requirements.txt` is the union needed to run everything end to end.
- GPU is optional — every number reported here was produced on CPU or Apple MPS as noted in the tables.
- `scikit-learn` is needed for the Cohen's kappa in the blind-review statistics.

## 4. Dataset

- Source: [github.com/mahsan2/AMdataset](https://github.com/mahsan2/AMdataset)
- Classes: `badcylinder` (136 images) / `goodcylinder` (135 images); augmented set 947 / 2070.
- Split rule: **70% train / 10% val / 20% test**, `random_state=42` (fixed, reproducible).
- The exact split is committed in `data/split.csv` (271 images: 190 train / 41 val / 40 test, each row hashed with sha256 — the contract file all models share).

## 5. How to Run

All commands assume the repository root as the working directory. `--device cpu` replaces `mps` on a non-Apple machine.

```bash
# 5.1 Fixed split (produced once by M2, committed as data/split.csv)
python -m m2.split_dataset --data-root /path/to/AMdataset/dataset

# 5.2 Download and verify the 271 split images
python -m m3.fetch_data --data-root m3_output/images
python -m m4.fetch_data --data-root m4_output/images

# 5.3 Train the four models (Adam, lr 1e-3, batch 8, 30 epochs, seed 42;
#     checkpoint selected on validation defect-F1)
python -m m3.train --arch vgg16           --data-root m3_output/images --device mps --output m3_output/vgg16
python -m m3.train --arch mobilenet_v2    --data-root m3_output/images --device mps --output m3_output/mobilenet_v2
python -m m4.train --arch resnet50        --data-root m4_output/images --device mps --output m4_output/resnet50
python -m m4.train --arch mobilenet_v3_small --data-root m4_output/images --device mps --output m4_output/mobilenet_v3_small

# 5.4 Robustness on the 40-image test split (M2)
python m2/run_m3_robustness.py --m3-repo . --data-root /path/to/dataset --split-csv data/split.csv --output-dir m2/results/m3
python m2/run_m4_robustness.py --m4-repo . --data-root /path/to/dataset --split-csv data/split.csv --output-dir m2/results/m4

# 5.5 Unified CPU benchmark (params / latency / checkpoint size)
python m2/benchmark_all_models.py --m3-repo . --m4-repo . --data-root /path/to/dataset --split-csv data/split.csv --output m2/results/efficiency_cpu.csv

# 5.6 Grad-CAM + blind-review bundle (M5) — the output directory must be empty
python -m m5.fetch_data --split-csv data/split.csv --subset test --data-root /path/to/m5_data
python -m m5.generate_gradcam \
  --model VGG16=m3/results/vgg16/best_head.pt \
  --model MobileNetV2=m3/results/mobilenet_v2/best_head.pt \
  --model ResNet50=m4/results/resnet50/best_head.pt \
  --model MobileNetV3-Small=m4/results/mobilenet_v3_small/best_head.pt \
  --data-root /path/to/AMdataset/dataset --split-csv data/split.csv --output /path/to/m5_output

# 5.7 Blind-review statistics (M8) — independent implementation, seeds p13
python m8/blind_review/stats_blind_review.py \
  --sheets-dir m8/results \
  --mapping m8/results/blind_key.csv \
  --out-dir m8/results

# 5.8 Report figures (defaults are relative to m8/)
python m8/plot_data_card.py       # p05 — needs data/split.csv from main, downloads sample images
python m8/plot_efficiency.py      # p10 — reads m8/data/efficiency_cpu.csv
python m8/plot_robustness.py      # p11 — reads m8/data/robustness_results_all_models.csv
python m8/plot_gradcam_panel.py   # p12 — reads m8/results/ + m8/panels/
```

## 6. Open-Source Attribution

The following is reused from the upstream repository [mahsan2/AM3d](https://github.com/mahsan2/AM3d):

- Data loading and image preprocessing
- The transfer-learning classification head structure

Specifically, the upstream `Studyone/` notebooks (grid search over 11 backbones × Adam/SGD/RMSprop) informed our backbone choice and hyper-parameters; the upstream notebooks are kept in the repository for provenance and were not modified.

Parts developed by our group:

| Component | Files | Owner |
|---|---|---|
| Fixed split + sha256 contract, data card | `m2/split_dataset.py`, `m2/am_m2/`, `m2/data/data_card.*` | M2 |
| Evaluation metrics | `m2/am_m2/metrics.py`, `evaluation.py` | M2 |
| Robustness perturbations + harness | `m2/am_m2/perturbations.py`, `m2/run_m3_robustness.py`, `run_m4_robustness.py` | M2 |
| Unified CPU benchmark | `m2/benchmark_all_models.py` | M2 |
| VGG16 / MobileNetV2 pipeline | `m3/` | M3 |
| ResNet50 / MobileNetV3-Small pipeline | `m4/` | M4 |
| Grad-CAM + blind-review bundle | `m5/gradcam.py`, `generate_gradcam.py` | M5 |
| Blind-review statistics (independent re-implementation) | `m8/blind_review/stats_blind_review.py` | M8 |
| All report figures | `m8/plot_data_card.py`, `m8/plot_efficiency.py`, `m8/plot_robustness.py`, `m8/plot_gradcam_panel.py` | M8 |
| This README | `README.md` | M8 |

Model weights (checkpoints) are LLM-generated outputs of our own training runs and are not redistributed from the paper.

## 7. Results Summary

Clean test split (40 images, fixed `data/split.csv`):

| Model | Test acc | Macro-F1 | Params | CPU latency (ms/img) | Checkpoint | Localization rate |
|---|---|---|---|---|---|---|
| VGG16 | 40/40 | 1.000 | 14.75 M | 39.94 | 59.0 MB | **11.7%** |
| MobileNetV2 | 40/40 | 1.000 | 2.31 M | 21.14 | 9.46 MB | 95.0% |
| ResNet50 | 40/40 | 1.000 | 23.64 M | 45.47 | 94.87 MB | 88.3% |
| MobileNetV3-Small | 38/40 | 0.952 | 0.96 M | 13.37 | 3.98 MB | 93.3% |

- **Latency**: M2's unified CPU benchmark, batch size 1, FP32, warm-up + repeated timings (`m8/data/efficiency_cpu.csv`). Earlier Mac-MPS numbers are superseded.
- **Localization rate**: share of 20 blind Grad-CAM panels per model that the majority of three independent reviewers judged to cover the defect region (`m8/results/`).
- **Robustness** (Macro-F1 under σ = 0.2 Gaussian noise): VGG16 0.387, ResNet50 0.601, MobileNetV3-Small 0.747, MobileNetV2 0.774 — clean accuracy does not predict robustness (`m8/data/robustness_results_all_models.csv`).

Take-away for the report: all four models are near-ceiling on the clean split, so the discriminative evidence is in efficiency, robustness and explanation quality — not accuracy.

## 8. Contributors

| Member | Role |
|---|---|
| M1 | Shen Yuming (沈禹茗) — team lead, schedule, evaluation-script requirements, efficiency summary, PPT storyline and conclusion, final submission, Q&A |
| M2 | Fan Yuxin (范雨馨) — dataset/split, unified evaluation script, robustness perturbations, robustness curves |
| M3 | Xu Zhexi (许哲熙) — VGG16 + MobileNetV2 fine-tuning, result-page material |
| M4 | Fu Maojie (傅懋杰) — ResNet50 + MobileNetV3-Small training, efficiency metrics with M2 |
| M5 | Xia Liyuan (夏立原) — Grad-CAM generation, blind-review bundle, heatmap result pages |
| M6 | Chen Renke (陈人可) — PPT template, chart styling, animation, live control, rehearsal timing |
| M7 | Li Ziyu (李子妤) — PPT text integration, page structure, speaker notes, rehearsal log, Q&A preparation |
| M8 | Chen Yihuan (陈一桓) — data-card and result figures, blind-review statistics, README/reproducibility |

Individual contribution statements are in the written report; M8's is also in [`M8_individual_contribution_statement.md`](M8_individual_contribution_statement.md).
