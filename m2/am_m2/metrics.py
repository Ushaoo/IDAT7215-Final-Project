from __future__ import annotations

import math
from typing import Iterable


def binary_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> dict[str, float | int]:
    truth = [int(x) for x in y_true]
    pred = [int(x) for x in y_pred]
    if len(truth) != len(pred) or not truth:
        raise ValueError("y_true and y_pred must have the same non-zero length")
    if set(truth + pred) - {0, 1}:
        raise ValueError("binary metrics require labels 0 and 1")

    tp = sum(t == 1 and p == 1 for t, p in zip(truth, pred))
    tn = sum(t == 0 and p == 0 for t, p in zip(truth, pred))
    fp = sum(t == 0 and p == 1 for t, p in zip(truth, pred))
    fn = sum(t == 1 and p == 0 for t, p in zip(truth, pred))

    def f1(tp_: int, fp_: int, fn_: int) -> float:
        denominator = 2 * tp_ + fp_ + fn_
        return 0.0 if denominator == 0 else 2 * tp_ / denominator

    f1_pos = f1(tp, fp, fn)
    f1_neg = f1(tn, fn, fp)
    mcc_den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = 0.0 if mcc_den == 0 else (tp * tn - fp * fn) / mcc_den
    return {
        "n": len(truth),
        "accuracy": (tp + tn) / len(truth),
        "f1_macro": (f1_pos + f1_neg) / 2,
        "f1_defect": f1_pos,
        "mcc": mcc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }

