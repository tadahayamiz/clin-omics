from __future__ import annotations

import pandas as pd

from clin_omics.dataset import CanonicalDataset
from clin_omics.exceptions import ClinOmicsError
from clin_omics.ml import (
    evaluate_classification,
    evaluate_regression,
    fit_linear_regressor,
    fit_logistic_classifier,
    get_cv_splitter,
    iter_cv_splits,
    predict_linear_regressor,
    predict_logistic_classifier,
    summarize_fold_metrics,
)


def _resolve_X(dataset: CanonicalDataset, feature_layer: str | None) -> pd.DataFrame:
    if feature_layer is None:
        return dataset.X
    if feature_layer not in dataset.layers:
        raise ClinOmicsError(f"Unknown feature layer: {feature_layer}")
    return dataset.layers[feature_layer]



def run_supervised_workflow(
    dataset: CanonicalDataset,
    target_col: str,
    task: str,
    feature_layer: str | None = None,
    groups_col: str | None = None,
    n_splits: int = 5,
    random_state: int | None = 0,
) -> dict[str, object]:
    X = _resolve_X(dataset, feature_layer)

    if target_col not in dataset.obs.columns:
        raise ClinOmicsError(f"Unknown target column: {target_col}")

    obs = dataset.obs.copy()
    if obs[target_col].isna().any():
        raise ClinOmicsError("Target column contains missing values.")

    y = pd.Series(obs[target_col].to_numpy(), index=obs["sample_id"].tolist(), name=target_col)
    groups = None
    if groups_col is not None:
        if groups_col not in obs.columns:
            raise ClinOmicsError(f"Unknown groups column: {groups_col}")
        groups = pd.Series(obs[groups_col].to_numpy(), index=obs["sample_id"].tolist(), name=groups_col)

    if task == "classification":
        method = "group_kfold" if groups_col is not None else "stratified_kfold"
    elif task == "regression":
        method = "group_kfold" if groups_col is not None else "kfold"
    else:
        raise ClinOmicsError(f"Unsupported task: {task}")

    splitter = get_cv_splitter(method=method, n_splits=n_splits, random_state=random_state)

    metrics_by_fold: list[dict[str, float]] = []
    prediction_frames: list[pd.DataFrame] = []
    models: list[object] = []

    for fold_idx, (train_idx, test_idx) in enumerate(iter_cv_splits(X, y, splitter, groups=groups), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if task == "classification":
            model = fit_logistic_classifier(X_train, y_train, random_state=random_state)
            y_score, y_pred = predict_logistic_classifier(model, X_test)
            metrics = evaluate_classification(y_true=y_test, y_pred=y_pred, y_score=y_score)
            pred_frame = pd.DataFrame(
                {
                    "sample_id": X_test.index,
                    "y_true": y_test.to_numpy(),
                    "y_pred": y_pred.to_numpy(),
                    "y_score": y_score.to_numpy(),
                    "fold": fold_idx,
                }
            )
        else:
            model = fit_linear_regressor(X_train, y_train)
            y_pred = predict_linear_regressor(model, X_test)
            metrics = evaluate_regression(y_true=y_test, y_pred=y_pred)
            pred_frame = pd.DataFrame(
                {
                    "sample_id": X_test.index,
                    "y_true": y_test.to_numpy(),
                    "y_pred": y_pred.to_numpy(),
                    "fold": fold_idx,
                }
            )

        metrics_by_fold.append(metrics)
        prediction_frames.append(pred_frame)
        models.append(model)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics_summary = summarize_fold_metrics(metrics_by_fold)

    return {
        "predictions": predictions,
        "metrics_by_fold": metrics_by_fold,
        "metrics_summary": metrics_summary,
        "models": models,
    }
