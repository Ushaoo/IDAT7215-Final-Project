import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


YES = {"1", "y", "yes", "true", "是", "覆盖"}
NO = {"0", "n", "no", "false", "否", "未覆盖"}


def parse_review(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use RATER=/path/to/completed_review.csv")
    rater, path = value.split("=", 1)
    if not rater.strip() or not path.strip():
        raise argparse.ArgumentTypeError("Rater name and review path are required")
    return rater.strip(), Path(path)


def parse_vote(value):
    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized in YES:
        return 1
    if normalized in NO:
        return 0
    raise ValueError(f"Unrecognized covers_defect value: {value!r}")


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def prepare_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def summarize(manifest_path, review_specs, output_path):
    output = prepare_output(output_path)
    manifest_rows = read_csv(manifest_path)
    manifest = {row["blind_id"]: row for row in manifest_rows}
    if len(manifest) != len(manifest_rows):
        raise ValueError("Duplicate blind_id in private manifest")

    votes = defaultdict(dict)
    for rater, review_path in review_specs:
        for row in read_csv(review_path):
            blind_id = row.get("blind_id", "")
            if blind_id not in manifest:
                raise ValueError(f"Unknown blind_id in {review_path}: {blind_id}")
            vote = parse_vote(row.get("covers_defect", ""))
            if vote is not None:
                if rater in votes[blind_id]:
                    raise ValueError(f"Duplicate vote for {rater}, {blind_id}")
                votes[blind_id][rater] = vote

    expected_raters = len({rater for rater, _path in review_specs})
    item_rows = []
    for blind_id, metadata in manifest.items():
        item_votes = votes[blind_id]
        yes_votes = sum(item_votes.values())
        no_votes = len(item_votes) - yes_votes
        majority = ""
        if yes_votes > no_votes:
            majority = "yes"
        elif no_votes > yes_votes:
            majority = "no"
        item_rows.append(
            {
                **metadata,
                "valid_votes": len(item_votes),
                "expected_votes": expected_raters,
                "yes_votes": yes_votes,
                "no_votes": no_votes,
                "majority_covers_defect": majority,
                "unanimous": bool(item_votes) and (yes_votes == 0 or no_votes == 0),
            }
        )

    by_model = defaultdict(list)
    for row in item_rows:
        if row["true_label"] == "1":
            by_model[row["model"]].append(row)
    model_rows = []
    for model, rows in sorted(by_model.items()):
        scored = [row for row in rows if row["majority_covers_defect"]]
        successes = sum(row["majority_covers_defect"] == "yes" for row in scored)
        complete = [row for row in scored if int(row["valid_votes"]) == expected_raters]
        unanimous = sum(bool(row["unanimous"]) for row in complete)
        model_rows.append(
            {
                "model": model,
                "defect_items": len(rows),
                "scored_items": len(scored),
                "localized_items": successes,
                "localization_success_rate": "" if not scored else f"{successes / len(scored):.6f}",
                "complete_items": len(complete),
                "unanimous_rate": "" if not complete else f"{unanimous / len(complete):.6f}",
            }
        )

    item_path = output / "item_scores.csv"
    with item_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=item_rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(item_rows)
    model_path = output / "localization_success.csv"
    with model_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=model_rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(model_rows)

    plotted = [row for row in model_rows if row["localization_success_rate"]]
    if plotted:
        figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
        names = [row["model"] for row in plotted]
        rates = [100 * float(row["localization_success_rate"]) for row in plotted]
        bars = axis.bar(names, rates, color="#2878B5")
        axis.set_ylim(0, 100)
        axis.set_ylabel("Localization success rate (%)")
        axis.set_title("Blind-review Grad-CAM localization")
        axis.bar_label(bars, fmt="%.1f%%", padding=3)
        axis.spines[["top", "right"]].set_visible(False)
        figure.savefig(output / "localization_success.png", dpi=240)
        plt.close(figure)
    return item_rows, model_rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Reveal and summarize completed blind reviews.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--review", required=True, action="append", type=parse_review, metavar="RATER=CSV")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    _items, models = summarize(args.manifest, args.review, args.output)
    for row in models:
        rate = row["localization_success_rate"] or "not available"
        print(f"{row['model']}: {row['localized_items']}/{row['scored_items']} ({rate})")


if __name__ == "__main__":
    main()
