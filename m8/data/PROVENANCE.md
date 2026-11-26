# Provenance of the CSV files in this folder

These two files are **verbatim copies** of M2's outputs, included so that the
`m8/` folder can regenerate every figure on its own branch. They are not
re-derived here; if M2 re-runs the benchmark or the robustness sweep, update
these copies from `m2/results/` rather than editing them.

| File | Source | Owner | Contents |
|---|---|---|---|
| `efficiency_cpu.csv` | `m2/results/efficiency_cpu.csv` (branch `m2/robustness-evaluation`) | M2 | One row per model: total/trainable parameters, full checkpoint size (MB, decimal), CPU mean/median/p95 latency, warm-up and repeat counts, batch size, precision |
| `robustness_results_all_models.csv` | `m2/results/robustness_results_all_models.csv` | M2 | 4 models × 11 conditions (baseline, Gaussian noise σ 0.05/0.1/0.2, motion blur 5/9/13, brightness 0.6/0.8/1.2/1.4): accuracy, macro-F1, defect-F1, MCC, F1 drop, confusion counts, inference time, n |

Synced: 2026-11-16.

`data/split.csv` is deliberately **not** duplicated here — the fixed split is
already committed at the repository root and is shared by all members.
`m8/plot_data_card.py` reads it from `<repo>/data/split.csv`.
