"""Verify Codex panel ratings and publish summaries without panel identities."""

import argparse
import hashlib
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from grouped.blind_review.aggregate import read_csv, write_csv
from grouped.blind_review.codex_localization import summarize

THRESHOLD = 0.50


def validate(panels, key, annotations, case_key):
    expected = {row["blind_id"]: row for row in key}
    assert len(key) == len(expected) == 160
    assert {row["blind_id"] for row in panels} == expected.keys()
    assert len(panels) == 160
    case_sources = {row["case_id"]: row["source_path"] for row in case_key}
    assert len(case_key) == len(case_sources) == len(set(case_sources.values())) == 20
    assert len(annotations) == 20
    assert {row["case_id"] for row in annotations} == case_sources.keys()
    assessable = {case_sources[row["case_id"]] for row in annotations if row["boxes"]}
    for panel in panels:
        metadata = expected[panel["blind_id"]]
        assert all(
            panel[field] == metadata[field]
            for field in ("model", "arm", "fold", "source_path")
        )
        assert metadata["label"] == "1" and metadata["seed"] == "42"
        overlap = panel["hotspot_overlap"]
        assert bool(overlap) == (panel["source_path"] in assessable)
        if overlap:
            value = float(overlap)
            assert 0 <= value <= 1
            assert panel["covers_defect"] == ("yes" if value >= THRESHOLD else "no")
        else:
            assert panel["covers_defect"] == "unsure"
    pairs = Counter((r["source_path"], r["model"], r["arm"]) for r in panels)
    assert len(pairs) == 160 and set(pairs.values()) == {1}


def annotate_summaries(output, panels):
    group_counts = Counter(
        (p["model"], p["arm"], p["fold"], p["covers_defect"]) for p in panels
    )
    folds = []
    for model, arm, fold in sorted({key[:3] for key in group_counts}):
        yes, no, unsure = [
            group_counts[model, arm, fold, vote] for vote in ("yes", "no", "unsure")
        ]
        assert yes + no + unsure == 5
        folds.append(
            dict(
                reviewer="Codex (single AI reviewer)",
                model=model,
                arm=arm,
                fold=fold,
                assessable=yes + no,
                yes=yes,
                no=no,
                unsure=unsure,
            )
        )
    write_csv(output / "codex_localization_by_fold.csv", folds)

    # Match the pre-existing scoring function's numerical inputs.
    for panel in panels:
        value = panel["hotspot_overlap"]
        panel["hotspot_overlap"] = float(value) if value else None
    summarize(output, panels)
    rows = [
        r
        for r in read_csv(output / "codex_localization_summary.csv")
        if float(r["overlap_threshold"]) == THRESHOLD
    ]
    fig, ax = plt.subplots(figsize=(8, 4.9), layout="constrained")
    index = range(len(rows))
    width = 0.34
    left = ax.bar(
        [i - width / 2 for i in index],
        [int(r["control_yes"]) for r in rows],
        width,
        label="Local pooling",
        color="#64748b",
    )
    right = ax.bar(
        [i + width / 2 for i in index],
        [int(r["global_yes"]) for r in rows],
        width,
        label="Global pooling",
        color="#087f80",
    )
    for bars, field in ((left, "control_yes"), (right, "global_yes")):
        ax.bar_label(bars, [f"{r[field]}/{r['assessable']}" for r in rows], padding=4)
    names = {
        "mobilenet_v2": "MobileNetV2",
        "mobilenet_v3_small": "MobileNetV3-Small",
        "resnet50": "ResNet50",
        "vgg16": "VGG16",
    }
    ax.set_xticks(list(index), [names[r["model"]] for r in rows])
    ax.set(
        ylim=(0, 13), yticks=range(0, 14, 2), ylabel="Cases meeting the overlap rule"
    )
    ax.set_title(
        "Defect localization scored by Codex\n13 assessable cases of 20 paired images"
    )
    ax.legend(loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    fig.savefig(output / "codex_localization.png", dpi=180)
    plt.close(fig)


def finalize(private_dir, key_path, output):
    annotations = private_dir / "annotations.csv"
    case_key = private_dir / "case_key.csv"
    ratings = private_dir / "private_panel_ratings.csv"
    panels = read_csv(ratings)
    validate(panels, read_csv(key_path), read_csv(annotations), read_csv(case_key))
    output.mkdir(parents=True, exist_ok=True)
    annotate_summaries(output, panels)
    write_csv(
        output / "codex_localization_input_hashes.csv",
        [
            dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            for path in (key_path, case_key, annotations, ratings)
        ],
    )
    print("Verified 160 matched ratings and wrote the publication summaries.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("private_dir", "key_path", "output"):
        parser.add_argument(name, type=Path)
    finalize(**vars(parser.parse_args()))
