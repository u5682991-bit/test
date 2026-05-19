from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


def binary_metrics(y_true: list[float], y_score: list[float], threshold: float = 0.5) -> dict[str, float]:
    true = np.asarray(y_true).astype(int)
    score = np.asarray(y_score).astype(float)
    pred = (score >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(true, pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(true, score)
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
    }


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

