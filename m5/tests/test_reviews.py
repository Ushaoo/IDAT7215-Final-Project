import csv

from m5.summarize_reviews import parse_vote, summarize


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_parse_vote_accepts_english_chinese_and_binary():
    assert parse_vote("yes") == 1
    assert parse_vote("是") == 1
    assert parse_vote("0") == 0
    assert parse_vote("") is None


def test_summary_uses_majority_vote_and_defect_denominator(tmp_path):
    manifest = tmp_path / "manifest.csv"
    fields = ["blind_id", "model", "true_label"]
    write_csv(
        manifest,
        fields,
        [
            {"blind_id": "A", "model": "VGG16", "true_label": "1"},
            {"blind_id": "B", "model": "VGG16", "true_label": "0"},
        ],
    )
    specs = []
    for rater, votes in (("M1", ("yes", "yes")), ("M5", ("yes", "no")), ("M8", ("no", "no"))):
        review = tmp_path / f"{rater}.csv"
        write_csv(
            review,
            ["blind_id", "covers_defect"],
            [
                {"blind_id": "A", "covers_defect": votes[0]},
                {"blind_id": "B", "covers_defect": votes[1]},
            ],
        )
        specs.append((rater, review))
    _items, models = summarize(manifest, specs, tmp_path / "summary")
    assert models[0]["defect_items"] == 1
    assert models[0]["localized_items"] == 1
    assert models[0]["localization_success_rate"] == "1.000000"
