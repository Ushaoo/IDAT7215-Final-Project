"""Unblind three completed sheets and summarize votes and paired outcomes."""

import argparse
import csv
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

VALID = {"yes", "no", "unsure"}
REVIEWERS = ("A", "B", "C")
ARMS = ("expanded", "global_pool")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_review(path, panels):
    votes = {}
    for row in read_csv(path):
        blind_id = row.get("blind_id")
        if blind_id in votes or blind_id not in panels:
            raise ValueError(f"Duplicate or unknown blind_id in {path}: {blind_id}")
        raw_vote = row.get("covers_defect") or ""
        vote = raw_vote.strip().lower()
        if vote not in VALID:
            raise ValueError(f"{path}: {blind_id} needs yes, no or unsure")
        votes[blind_id] = vote
    if votes.keys() != panels.keys():
        raise ValueError(f"{path}: expected {len(panels)} panels, got {len(votes)}")
    return votes


def ratio(numerator, denominator):
    return numerator / denominator if denominator else ""


def summarize_items(items):
    counts = Counter(item["majority"] for item in items)
    yes_votes = sum(item["yes_votes"] for item in items)
    no_votes = sum(item["no_votes"] for item in items)
    unsure_votes = sum(item["unsure_votes"] for item in items)
    resolved = counts["yes"] + counts["no"]
    return dict(
        panels=len(items),
        yes_votes=yes_votes,
        no_votes=no_votes,
        unsure_votes=unsure_votes,
        yes_share_binary_votes=ratio(yes_votes, yes_votes + no_votes),
        majority_covers_defect=counts["yes"],
        majority_does_not_cover=counts["no"],
        unresolved=counts["unresolved"],
        localized_share_resolved=ratio(counts["yes"], resolved),
        localized_share_all_panels=ratio(counts["yes"], len(items)),
    )


def summarize_pairs(items):
    pairs = defaultdict(dict)
    for item in items:
        key = item["model"], item["fold"], item["source_path"]
        arm = item["arm"]
        if arm in pairs[key]:
            raise ValueError(f"Duplicate arm for {key}: {arm}")
        pairs[key][arm] = item["majority"]
    counts = defaultdict(Counter)
    for (model, _, _), pair in pairs.items():
        if set(pair) != set(ARMS):
            raise ValueError(f"Missing matching arm for model {model}")
        counters = counts[model]
        counters["total"] += 1
        left, right = (pair[arm] for arm in ARMS)
        if "unresolved" in (left, right):
            counters["unresolved"] += 1
        else:
            counters[left, right] += 1
    return [
        dict(
            model=model,
            total_pairs=c["total"],
            both_localized=c["yes", "yes"],
            control_only=c["yes", "no"],
            global_only=c["no", "yes"],
            neither_localized=c["no", "no"],
            unresolved_pairs=c["unresolved"],
        )
        for model, c in sorted(counts.items())
    ]


def agreement(votes):
    rows = []
    for a, b in combinations(REVIEWERS, 2):
        left, right = votes[a], votes[b]
        agreement = sum(left[k] == right[k] for k in left) / len(left)
        left_counts, right_counts = Counter(left.values()), Counter(right.values())
        expected = sum(left_counts[k] * right_counts[k] for k in VALID) / len(left) ** 2
        rows.append(
            dict(
                reviewer_1=a,
                reviewer_2=b,
                panels=len(left),
                agreement=agreement,
                cohen_kappa=ratio(agreement - expected, 1 - expected),
            )
        )
    return rows


def aggregate(key, review_paths, output):
    key_rows = read_csv(key)
    panels = {r["blind_id"]: r for r in key_rows}
    if not panels or len(panels) != len(key_rows):
        raise ValueError("Expected a nonempty key with unique IDs")
    votes = {
        rater: load_review(path, panels) for rater, path in zip(REVIEWERS, review_paths)
    }
    items = []
    by_arm = defaultdict(list)
    by_fold = defaultdict(list)
    for blind_id, panel in panels.items():
        item_votes = {f"review_{r}": votes[r][blind_id] for r in REVIEWERS}
        count = Counter(item_votes.values())
        majority = next((x for x in ("yes", "no") if count[x] >= 2), "unresolved")
        item = dict(
            **panel,
            **item_votes,
            yes_votes=count["yes"],
            no_votes=count["no"],
            unsure_votes=count["unsure"],
            majority=majority,
        )
        items.append(item)
        by_arm[item["model"], item["arm"]].append(item)
        by_fold[item["model"], item["arm"], item["fold"]].append(item)
    pairs = summarize_pairs(items)
    summary = [
        dict(model=model, arm=arm, **summarize_items(panel_items))
        for (model, arm), panel_items in sorted(by_arm.items())
    ]
    folds = [
        dict(model=model, arm=arm, fold=fold, **summarize_items(panel_items))
        for (model, arm, fold), panel_items in sorted(by_fold.items())
    ]
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "panel_votes.csv", items)
    write_csv(output / "localization_by_model.csv", summary)
    write_csv(output / "localization_by_fold.csv", folds)
    write_csv(output / "matched_pooling_comparison.csv", pairs)
    write_csv(output / "reviewer_agreement.csv", agreement(votes))
    print(f"Aggregated {len(panels)} panels from {len(votes)} completed sheets")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("key", type=Path)
    parser.add_argument("review_a", type=Path)
    parser.add_argument("review_b", type=Path)
    parser.add_argument("review_c", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    aggregate(args.key, (args.review_a, args.review_b, args.review_c), args.output)
