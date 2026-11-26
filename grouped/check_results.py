"""Independently check the reference and corruption outputs against their manifests."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from grouped.protocol import ARCHS, ARMS, SEEDS


def check_predictions(pred, metrics, keys, expected_for):
    assert not metrics.duplicated(keys).any()
    lookup = metrics.set_index(keys)
    groups = pred.groupby(keys)
    assert len(groups) == len(metrics)
    for key, frame in groups:
        expected = expected_for(dict(zip(keys, key)))
        actual = frame.set_index("path").label.sort_index()
        pd.testing.assert_series_equal(expected, actual)
        assert frame.p_defect.between(0, 1).all()
        outcome = frame.p_defect.gt(0.5)
        counts = confusion_matrix(frame.label, outcome, labels=[0, 1]).ravel()
        run = lookup.loc[key]
        assert np.array_equal(counts, run[["tn", "fp", "fn", "tp"]])
        assert np.isclose(f1_score(frame.label, outcome, average="macro"), run.macro_f1)


def main(root):
    manifest = pd.read_csv(root / "split.csv")
    assert len(manifest) == manifest.sha256.nunique() == 4845
    kaggle = manifest.query("source == 'kaggle'").set_index("path")
    assignments = pd.read_csv(root / "reference_split.csv")
    ref = pd.read_csv(root / "reference_metrics.csv")
    assert len(ref) == len(ARCHS) * len(SEEDS) * 5
    reference_tests = {}
    for k, g in assignments.groupby(["scheme", "fold"]):
        assert set(g.path) == set(kaggle.index) and g.path.is_unique
        reference_tests[k] = kaggle.loc[g[g.role == "test"].path].label.sort_index()
    check_predictions(
        pd.read_csv(root / "reference_predictions.csv.gz"),
        ref,
        ["model", "scheme", "fold", "seed"],
        lambda k: reference_tests[k["scheme"], k["fold"]],
    )
    ref.groupby(["model", "scheme"])[
        ["macro_f1", "accuracy", "recall", "fpr"]
    ].mean().to_csv(root / "reference_summary.csv")
    stress = pd.read_csv(root / "stress_metrics.csv")
    assert len(stress) == len(ARCHS) * len(ARMS) * 4 * 5
    folds = {
        i: kaggle[kaggle[f"fold{i}"] == "test"].label.sort_index() for i in range(4)
    }
    predictions = pd.read_csv(root / "stress_predictions.csv.gz")
    check_predictions(
        predictions,
        stress,
        ["model", "fold", "arm", "condition", "level"],
        lambda k: folds[k["fold"]],
    )
    clean = predictions.query("condition=='clean'")
    for arch in ARCHS:
        baseline = pd.read_csv(root / arch / "out_of_fold.csv.gz").query("seed==42")
        joined = clean[clean.model == arch].merge(
            baseline, on=["path", "arm", "fold"], validate="one_to_one"
        )
        assert len(joined) == 1557 * len(ARMS)
        print(
            arch,
            "clean label replay differences:",
            int((joined.p_defect_x.gt(0.5) != joined.p_defect_y.gt(0.5)).sum()),
        )
    stress.groupby(["model", "arm", "condition", "level"])[
        ["macro_f1", "accuracy", "recall", "fpr"]
    ].mean().to_csv(root / "stress_summary.csv")
    print(
        "Reference and stress result counts, predictions, labels, metrics and membership verified."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    main(parser.parse_args().results)
