from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score


def evaluate_classification(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_score: pd.Series | None = None,
) -> dict[str, float]:
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    if y_score is not None and y_true.nunique() == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    return metrics


def evaluate_regression(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    rmse = math.sqrt(float(np.mean((y_true - y_pred) ** 2)))
    return {
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
    }


def summarize_fold_metrics(metrics_by_fold: list[dict[str, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(metrics_by_fold)
    summary = pd.DataFrame({
        "metric": frame.columns,
        "mean": [float(frame[c].mean()) for c in frame.columns],
        "std": [float(frame[c].std(ddof=0)) for c in frame.columns],
    })
    return summary
