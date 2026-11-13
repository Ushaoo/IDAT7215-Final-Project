# M5 real-model blind-review bundle

This bundle was generated from the fixed test split with the submitted M3 and M4
checkpoints. It contains 80 anonymous panels: 20 ground-truth Defect images for
each of VGG16, MobileNetV2, ResNet50 and MobileNetV3-Small.

## Run facts

- Fixed `data/split.csv` SHA-256: `f7efba67d4802aa9feaca11cad5e39cd8d348c955148910a1be738f55e991a12`
- M2 source commit: `8378f83`
- M3 source commit: `fd8e55d`
- M4 source commit: `fffb74c`
- AMdataset source commit: `52fe75dec70c1e8046493b4e3256f8f752a4d39d`
- Target mode: predicted class
- True and predicted class for all 80 items: Defect
- Baseline errors among the selected items: 0
- Private manifest SHA-256: `4cd57af2301cb89412ebc12513e3690d978eb7d3c0018c450ed501b727a74425`

All four submitted models classified every fixed-split test image correctly, so no
baseline misclassified examples were available. None were manufactured. This does
not establish localization quality; that result must come from the blind review.

## Blind-review procedure

1. M1, M5 and M8 each use only their assigned `review_*.csv`.
2. Open each referenced panel. Its columns are Original, Grad-CAM and Overlay.
3. Enter `yes` when the dominant warm Grad-CAM region overlaps the visually
   identifiable defect region. Enter `no` when it focuses on background, cables,
   fixtures or an unrelated object area.
4. If the defect cannot be judged confidently, leave the score blank and explain
   why in `comments`.
5. Work independently. Do not inspect another rater's answers or request the
   private identity manifest before all reviews are complete.
6. Do not commit completed review sheets to this public branch. Return them
   privately to M8 for aggregation.

The private model/image mapping is intentionally excluded from Git. Localization
success rates must not be reported until the three completed sheets are aggregated.
