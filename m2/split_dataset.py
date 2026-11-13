from __future__ import annotations

import argparse
from pathlib import Path

from am_m2.data import assign_splits, build_data_card, collect_samples, write_data_card, write_split_csv


def parse_args():
    parser = argparse.ArgumentParser(description="Create a deterministic, duplicate-aware AM dataset split.")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/data"))
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ratios = (args.train, args.val, args.test)
    samples = collect_samples(args.data_root)
    rows = assign_splits(samples, ratios, args.seed)
    split_path = args.output_dir / "split.csv"
    card_json = args.output_dir / "data_card.json"
    card_md = args.output_dir / "data_card.md"
    write_split_csv(rows, split_path)
    card = build_data_card(rows, args.data_root, args.seed, ratios)
    write_data_card(card, card_json, card_md)
    print(f"Wrote {len(rows)} samples to {split_path}")
    print(f"Data card: {card_md}")


if __name__ == "__main__":
    main()

