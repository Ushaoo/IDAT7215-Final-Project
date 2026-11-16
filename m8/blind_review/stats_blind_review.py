# -*- coding: utf-8 -*-
"""stats_blind_review.py — 盲评统计脚本（M8 任务③ 第二部分）

三位评审填完各自的 reviewer_*.csv 后运行本脚本，产出：
  results/blind_review_stats.csv   每模型定位成功率（+ 按预测对错分组）
  results/agreement_stats.csv      三人一致性 / 两两一致率与 Cohen's kappa
  results/per_image_votes.csv      每张热图的三人投票与多数结论
  results/p13_localization_rate.png  PPT 第 13 页柱状图

用法：
  python stats_blind_review.py --sheets-dir sheets
                               [--mapping sheets/_secret/blind_mapping.csv]
                               [--out-dir results]

统计口径（写进 PPT/报告时直接引用）：
  定位成功率 localization rate = 评审判"是"的数量 /（"是"+"否"的数量），
  NA（无法判断）不计入分母，但单独报告数量。
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from sklearn.metrics import cohen_kappa_score
    HAS_KAPPA = True
except ImportError:
    HAS_KAPPA = False

# ---- 与数据卡片图一致的配色（模型配色提案，待 M6 全局确认）----
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
MODEL_ORDER = ("VGG16", "MobileNetV2", "ResNet50", "MobileNetV3-Small")
MODEL_COLORS = {
    "VGG16": "#2a78d6",
    "MobileNetV2": "#eb6834",
    "ResNet50": "#1baf7a",
    "MobileNetV3-Small": "#eda100",
}
REVIEWERS = ("M1", "M5", "M8")


def load_mapping(path):
    """Read the secret mapping. Requires blind_id + model; `correct` optional
    (M5's private manifest has no `correct` column - all 80 panels were
    classified correctly, so no per-image correct/wrong grouping is possible)."""
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    if not rows:
        sys.exit("映射表为空：%s" % path)
    mapping = {}
    for row in rows:
        blind_id = (row.get("blind_id") or "").strip()
        if not blind_id:
            sys.exit("映射表缺少 blind_id 列：%s" % path)
        if "model" not in row:
            sys.exit("映射表缺少 model 列：%s" % path)
        row["correct"] = (row.get("correct") or "").strip()
        mapping[blind_id] = row
    return mapping


def parse_vote(value):
    """1 -> 1, 0 -> 0, NA/空白 -> None（无法判断）"""
    value = (value or "").strip().upper()
    if value in ("1", "Y", "YES"):
        return 1
    if value in ("0", "N", "NO"):
        return 0
    if value in ("NA", "N/A", "", "-"):
        return None
    sys.exit("covers 列出现无法识别的值：%r（允许 1 / 0 / NA）" % value)


def load_sheets(sheets_dir, mapping):
    """Accepts both naming schemes:
       reviewer_M1.csv  with column `covers`         (M8's original toolchain)
       review_M1.csv    with column `covers_defect`  (M5's bundle)
    """
    votes = {blind_id: {} for blind_id in mapping}
    paths = sorted(set(sheets_dir.glob("reviewer_*.csv")) | set(sheets_dir.glob("review_*.csv")))
    if not paths:
        sys.exit("找不到评审表：%s/review_*.csv" % sheets_dir)
    for path in paths:
        reviewer = path.stem.replace("reviewer_", "").replace("review_", "")
        if reviewer not in REVIEWERS:
            print("警告：忽略未知评审表 %s" % path)
            continue
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            columns = reader.fieldnames or []
            cover_col = next((c for c in ("covers", "covers_defect", "covers_defect_region")
                              if c in columns), None)
            if cover_col is None:
                sys.exit("%s 找不到评分列（covers / covers_defect）" % path.name)
            for row in reader:
                blind_id = (row.get("blind_id") or "").strip()
                if not blind_id:
                    continue
                if blind_id not in votes:
                    sys.exit("%s 里出现映射表没有的盲号：%s" % (path.name, blind_id))
                votes[blind_id][reviewer] = parse_vote(row.get(cover_col, ""))
    return votes


def rate(yes, no):
    return yes / (yes + no) if yes + no else None


def model_stats(votes, mapping):
    """每模型定位成功率；有 correct 列时按预测对错分组。"""
    per_model = {}
    for blind_id, row in mapping.items():
        model = row["model"]
        box = per_model.setdefault(model, {
            "n_heatmaps": 0, "yes": 0, "no": 0, "na": 0,
            "yes_correct": 0, "no_correct": 0, "yes_wrong": 0, "no_wrong": 0,
        })
        box["n_heatmaps"] += 1
        for reviewer, vote in votes[blind_id].items():
            if vote is None:
                box["na"] += 1
                continue
            if vote == 1:
                box["yes"] += 1
            else:
                box["no"] += 1
            c = (row.get("correct", "") or "").strip().lower()
            bucket = "correct" if c in ("1", "true", "yes") else ("wrong" if c in ("0", "false", "no") else None)
            if bucket:
                box["yes_%s" % bucket] += vote
                box["no_%s" % bucket] += 1 - vote
    rows = []
    for model in MODEL_ORDER:
        if model not in per_model:
            continue
        b = per_model[model]
        rows.append({
            "model": model,
            "n_heatmaps": b["n_heatmaps"],
            "n_rated": b["yes"] + b["no"],
            "n_yes": b["yes"],
            "n_no": b["no"],
            "n_na": b["na"],
            "localization_rate": rate(b["yes"], b["no"]),
            "rate_on_correct": rate(b["yes_correct"], b["no_correct"]),
            "rate_on_wrong": rate(b["yes_wrong"], b["no_wrong"]),
        })
    return rows


def agreement_stats(votes):
    """三人一致比例 + 两两一致率与 Cohen's kappa（只算双方都给了 0/1 的行）。"""
    all_equal = 0
    all_rated = 0
    pair_rows = {pair: [] for pair in ((0, 1), (0, 2), (1, 2))}
    for blind_id, by_reviewer in votes.items():
        values = [by_reviewer.get(r) for r in REVIEWERS]
        if all(v is not None for v in values):
            all_rated += 1
            if values[0] == values[1] == values[2]:
                all_equal += 1
        for (a, b), collected in pair_rows.items():
            va, vb = values[a], values[b]
            if va is not None and vb is not None:
                collected.append((va, vb))
    rows = [{
        "metric": "full_agreement_pct",
        "value": (100.0 * all_equal / all_rated) if all_rated else None,
        "detail": "%d/%d images rated by all three" % (all_equal, all_rated),
    }]
    names = ["%s-%s" % (REVIEWERS[a], REVIEWERS[b]) for a, b in pair_rows]
    for name, collected in zip(names, pair_rows.values()):
        agreed = sum(1 for va, vb in collected if va == vb)
        kappa = None
        if HAS_KAPPA and len(set(v for pair in collected for v in pair)) > 1 and collected:
            kappa = cohen_kappa_score([va for va, _ in collected], [vb for _, vb in collected])
        rows.append({
            "metric": "pair_agreement_pct_%s" % name,
            "value": (100.0 * agreed / len(collected)) if collected else None,
            "detail": "n=%d" % len(collected),
        })
        rows.append({
            "metric": "cohen_kappa_%s" % name,
            "value": kappa,
            "detail": "sklearn cohen_kappa_score" if HAS_KAPPA else "sklearn not installed",
        })
    return rows


def majority(votes, mapping):
    """每张热图的多数结论：covered / not covered / no consensus / no ratings。"""
    rows = []
    for blind_id in sorted(mapping):
        row = mapping[blind_id]
        values = [votes[blind_id].get(r) for r in REVIEWERS]
        yes = values.count(1)
        no = values.count(0)
        if yes > no:
            verdict = "covered"
        elif no > yes:
            verdict = "not covered"
        elif yes == no and yes > 0:
            verdict = "no consensus"
        else:
            verdict = "no ratings"
        rows.append({
            "blind_id": blind_id,
            "image_id": (row.get("image_id") or Path(row.get("image_path") or row.get("relative_path") or "").name),
            "model": row["model"],
            "correct": row.get("correct", ""),
            "votes_%s" % REVIEWERS[0]: values[0] if values[0] is not None else "NA",
            "votes_%s" % REVIEWERS[1]: values[1] if values[1] is not None else "NA",
            "votes_%s" % REVIEWERS[2]: values[2] if values[2] is not None else "NA",
            "majority": verdict,
        })
    return rows


def fmt(value, digits=3):
    return "%.3f" % value if value is not None else ""


def plot_rates(rows, out_path, note=None):
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    names = [r["model"] for r in rows]
    rates = [100.0 * r["localization_rate"] if r["localization_rate"] is not None else 0.0 for r in rows]
    ax.bar(names, rates, width=0.5, color=[MODEL_COLORS.get(n, "#898781") for n in names],
           edgecolor=SURFACE, linewidth=2, zorder=3)
    for name, value, row in zip(names, rates, rows):
        label = "%.0f%%" % value if row["localization_rate"] is not None else "n/a"
        ax.text(name, value + 2, label, ha="center", va="bottom",
                color=INK, fontsize=12, fontweight="bold")
        ax.text(name, -6, "n=%d" % row["n_heatmaps"], ha="center", va="top",
                color=MUTED, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7")
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Localization success rate (%)", color=INK_2, fontsize=11)
    ax.set_title("Blind review: does the heatmap cover the defect region?",
                 color=INK, fontsize=12.5, fontweight="bold")
    if note:
        fig.text(0.5, -0.06, note, ha="center", va="top", color=MUTED, fontsize=8.5)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="统计三人盲评结果")
    parser.add_argument("--sheets-dir", type=Path, default=here / "sheets")
    parser.add_argument("--mapping", type=Path, default=here / "sheets" / "_secret" / "blind_mapping.csv")
    parser.add_argument("--out-dir", type=Path, default=here / "results")
    args = parser.parse_args()

    mapping = load_mapping(args.mapping)
    votes = load_sheets(args.sheets_dir, mapping)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    stats = model_stats(votes, mapping)
    with (args.out_dir / "blind_review_stats.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(stats[0].keys()) if stats else [])
        writer.writeheader()
        for row in stats:
            writer.writerow({k: (fmt(v) if isinstance(v, float) and v is not None else v) for k, v in row.items()})

    agreement = agreement_stats(votes)
    with (args.out_dir / "agreement_stats.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", "value", "detail"])
        for row in agreement:
            writer.writerow([row["metric"], fmt(row["value"]) if isinstance(row["value"], float) else row["value"], row["detail"]])

    per_image = majority(votes, mapping)
    with (args.out_dir / "per_image_votes.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_image[0].keys()))
        writer.writeheader()
        writer.writerows(per_image)

    kappas = [r["value"] for r in agreement if r["metric"].startswith("cohen_kappa") and r["value"] is not None]
    pair_pct = [r["value"] for r in agreement if r["metric"].startswith("pair_agreement_pct") and r["value"] is not None]
    note = "Blind review by M1 / M5 / M8 · 20 heatmaps per model"
    if pair_pct and kappas:
        note += " · pairwise agreement %.1f-%.1f%%, Cohen's kappa %.2f-%.2f" % (
            min(pair_pct), max(pair_pct), min(kappas), max(kappas))
    plot_rates(stats, args.out_dir / "p13_localization_rate.png", note=note)
    figures_dir = here.parent / "figures"
    if figures_dir.is_dir():
        import shutil
        shutil.copyfile(args.out_dir / "p13_localization_rate.png",
                        figures_dir / "p13_localization_rate.png")
        print("copied -> %s" % (figures_dir / "p13_localization_rate.png"))

    print("== 定位成功率（localization rate = yes / (yes + no)）==")
    for row in stats:
        print("  %-20s rate=%s  (yes %d / no %d / NA %d, n=%d heatmaps)" % (
            row["model"], fmt(row["localization_rate"]),
            row["n_yes"], row["n_no"], row["n_na"], row["n_heatmaps"]))
        if row["rate_on_correct"] is not None or row["rate_on_wrong"] is not None:
            print("     判对图上 rate=%s，判错图上 rate=%s" % (
                fmt(row["rate_on_correct"]), fmt(row["rate_on_wrong"])))
    print("== 一致性 ==")
    for row in agreement:
        if row["value"] is not None:
            print("  %-28s %s  (%s)" % (row["metric"], fmt(row["value"]) if isinstance(row["value"], float) else row["value"], row["detail"]))
    print("输出目录：%s" % args.out_dir)


if __name__ == "__main__":
    main()
