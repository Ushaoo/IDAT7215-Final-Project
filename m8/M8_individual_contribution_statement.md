# Individual Contribution Statement

**Course:** IDAT7215 — Computer Programming for Product Development and Applications (2026/27 S1)
**Project:** Topic 4 — Defect Analysis of 3D Printed Objects Using Transfer Learning Approaches
**Member:** M8 — Chen Yihuan (陈一桓) · Student ID: 3036779633

## Scope of my contribution

My role was the **measurement and reporting layer** of the project: every figure in the slide deck, the blind-review protocol and its statistics, and the reproducibility materials. I did not train any of the four models; I consumed the checkpoints and outputs produced by M2, M3, M4 and M5, and I am responsible for the validity of every number shown on slides 5, 10, 11, 12 and 13.

## Concrete deliverables

### 1. Data card (slide 5) — `plot_data_card.py`, `figures/p05_*.png`
Three figures characterising the dataset: class composition (badcylinder 136 / goodcylinder 135, augmented 947 / 2070), a sample grid of both classes, and the fixed 70/10/20 split ratio (271 images: 190 / 41 / 40). All figures are generated from `data/split.csv` rather than hand-drawn, so the numbers on the slide cannot drift from the contract file.

### 2. Efficiency comparison (slide 10) — `plot_efficiency.py`, `figures/p10_efficiency.png`
Three-panel bar chart (parameters, CPU latency, checkpoint size) for the four models, built on M2's unified CPU benchmark (`data/efficiency_cpu.csv`). I made the decision to discard the earlier per-member Mac-MPS latency numbers and report only the single-hardware benchmark, because mixed hardware makes the comparison meaningless.

### 3. Robustness curve (slide 11) — `plot_robustness.py`, `figures/p11_robustness.png`
Small-multiple line chart (Gaussian noise / motion blur / brightness) of Macro-F1 versus perturbation level for all four models, built on M2's `data/robustness_results_all_models.csv`. This figure carries the project's main finding: clean-test accuracy is uninformative because all models are near-ceiling, while under σ = 0.2 noise VGG16 collapses to F1 = 0.387 and MobileNetV2 stays at 0.774.

### 4. Blind-review protocol (slide 13) — `blind_review/`, `blind_review/stats_blind_review.py`
I designed and implemented the blind-review protocol used to make the Grad-CAM claim quantitative rather than anecdotal:

- **Design:** 80 heatmap panels (20 per model, one per test image), blinded by cryptographically random IDs, presented to three independent raters (M1, M5, M8) in different row orders. The identity key (`manifest.csv`) was held separately and was not shared until all three sheets were returned.
- **Sheet generation:** `make_blind_sheets.py` splits the private key into three neutral review sheets so no rater can infer model identity from the sheet itself.
- **Statistics:** `stats_blind_review.py` independently re-implements the aggregation (I did not reuse M5's `summarize_reviews.py`, so the headline numbers are double-checked by two implementations). It computes per-model localization rate (yes / (yes + no), "cannot judge" excluded from the denominator and reported separately), full and pairwise agreement, and Cohen's κ, and emits both the CSV tables and the slide-13 figure.
- **Encoding reconciliation:** M1 returned 0/1 votes while M5 and M8 used yes/no. I verified the polarity empirically rather than assuming it — treating 1 as "yes" gives 93.8% agreement with my own sheet, treating it as "no" gives 6.2% — and normalised all three sheets to a single encoding before analysis.

**Result:** localization rate VGG16 11.7%, MobileNetV2 95.0%, ResNet50 88.3%, MobileNetV3-Small 93.3%; three-way agreement 85.0%; Cohen's κ 0.69–0.85 (substantial to almost perfect). The VGG16 collapse was independently corroborated by all three raters (19/20, 19/20 and 15/20 "no"), so it is a property of the heatmaps, not of the scoring.

I was also one of the three raters and scored all 80 panels myself (`review_M8.csv`).

### 5. Grad-CAM panel (slide 12) — `plot_gradcam_panel.py`, `figures/p12_gradcam_grid.png`
Selected the eight panels shown on the slide from the blind-review majority vote (one localized and one missed example per model, preferring unanimous votes), and laid them out as a 4×2 grid with the blind-revealed image name, the per-panel vote tally, and the model's full-sample localization rate printed on the row. Panels are pasted verbatim from M5's output — nothing is cropped or re-rendered — so the slide shows exactly the images the raters scored. Where a model had no genuinely unanimous example (VGG16 produced only one "covered" panel out of twenty), that is stated rather than papered over.

### 6. Reproducibility materials — `README.md`, `requirements.txt`
Wrote the project README (overview, repository structure, environment, dataset, end-to-end run commands for all four members' pipelines, open-source attribution, consolidated results table) and consolidated the four members' separate dependency files into one tested `requirements.txt`. Every figure script reads its data from a committed CSV or JSON, so the whole figure set regenerates from the repository without manual steps.

## What I did not do

Model training (M3, M4), dataset split and robustness/benchmark harnesses (M2), Grad-CAM generation (M5), slide design (M6) and report prose (M7). My contribution is bounded by what is listed above.

## Division for the presentation

I present slide 12 (Grad-CAM localization) and slide 13 (blind-review statistics) and answer questions on the review protocol, the agreement statistics, and how the figures were generated.
