"""Evaluation metrics for hand assignment."""

from __future__ import annotations

import numpy as np


def switch_flags(labels: np.ndarray) -> np.ndarray:
    """1 where the hand differs from the previous note (a hand switch)."""
    flags = np.zeros(len(labels), dtype=np.int8)
    if len(labels) > 1:
        flags[1:] = (labels[1:] != labels[:-1]).astype(np.int8)
    return flags


def switch_prf(pred: np.ndarray, true: np.ndarray) -> tuple[float, float, float]:
    tp = int(((pred == 1) & (true == 1)).sum())
    fp = int(((pred == 1) & (true == 0)).sum())
    fn = int(((pred == 0) & (true == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return precision, recall, f1


def piece_metrics(pred: np.ndarray, labels: np.ndarray, pitch: np.ndarray) -> dict:
    """Accuracy, hand-switch F1 and the separation between the two hands."""
    accuracy = float((pred == labels).mean()) if len(labels) else 0.0
    _, _, f1 = switch_prf(switch_flags(pred), switch_flags(labels))
    right, left = pitch[labels == 0], pitch[labels == 1]
    separation = float(np.median(right) - np.median(left)) if len(right) and len(left) else 0.0
    return {"acc": accuracy, "sw_f1": f1, "sep": separation, "n": int(len(labels))}


def summarize(per_piece: list[dict], hard_threshold: float = 12.0) -> dict:
    if not per_piece:
        return {"acc": 0.0, "acc_macro": 0.0, "sw_f1": 0.0}
    accs = np.array([m["acc"] for m in per_piece])
    counts = np.array([m["n"] for m in per_piece])
    out = {
        "acc": float((accs * counts).sum() / max(counts.sum(), 1)),
        "acc_macro": float(accs.mean()),
        "sw_f1": float(np.mean([m["sw_f1"] for m in per_piece])),
    }
    hard = [m for m in per_piece if m["sep"] < hard_threshold]
    if hard:
        weights = np.array([m["n"] for m in hard])
        out["hard_n_pieces"] = len(hard)
        out["hard_acc"] = float((np.array([m["acc"] for m in hard]) * weights).sum()
                                / max(weights.sum(), 1))
    else:
        out["hard_n_pieces"] = 0
        out["hard_acc"] = float("nan")
    return out
