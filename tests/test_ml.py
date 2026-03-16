import pandas as pd

from clin_omics.dataset import CanonicalDataset
from clin_omics.exceptions import ClinOmicsError
from clin_omics.ml import get_cv_splitter, iter_cv_splits
from clin_omics.workflows import run_supervised_workflow


def make_classification_dataset() -> CanonicalDataset:
    X = pd.DataFrame(
        {
            "f1": [0.0, 0.1, 1.0, 1.2, 4.9, 5.1, 6.0, 6.2],
            "f2": [0.0, 0.2, 1.1, 1.0, 5.0, 4.8, 6.1, 5.9],
        },
        index=[f"s{i}" for i in range(1, 9)],
    )
    obs = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(1, 9)],
            "target": [0, 0, 0, 0, 1, 1, 1, 1],
            "subject_group": ["g1", "g2", "g3", "g4", "g1", "g2", "g3", "g4"],
        }
    )
    var = pd.DataFrame({"feature_id": ["f1", "f2"]})
    return CanonicalDataset(X=X, obs=obs, var=var)


def make_regression_dataset() -> CanonicalDataset:
    X = pd.DataFrame(
        {
            "f1": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "f2": [0.5, 1.5, 1.8, 3.1, 4.2, 5.1],
        },
        index=[f"r{i}" for i in range(1, 7)],
    )
    obs = pd.DataFrame(
        {
            "sample_id": [f"r{i}" for i in range(1, 7)],
            "target": [0.0, 1.1, 2.0, 3.2, 4.1, 5.0],
        }
    )
    var = pd.DataFrame({"feature_id": ["f1", "f2"]})
    return CanonicalDataset(X=X, obs=obs, var=var)


def test_group_kfold_does_not_split_groups_across_train_and_test() -> None:
    dataset = make_classification_dataset()
    X = dataset.X
    y = dataset.obs["target"]
    groups = dataset.obs["subject_group"]
    splitter = get_cv_splitter("group_kfold", n_splits=4)

    for train_idx, test_idx in iter_cv_splits(X, y, splitter, groups=groups):
        train_groups = set(groups.iloc[train_idx].tolist())
        test_groups = set(groups.iloc[test_idx].tolist())
        assert train_groups.isdisjoint(test_groups)


def test_classification_workflow_runs_end_to_end() -> None:
    dataset = make_classification_dataset()
    result = run_supervised_workflow(
        dataset,
        target_col="target",
        task="classification",
        groups_col="subject_group",
        n_splits=4,
    )

    assert len(result["metrics_by_fold"]) == 4
    assert set(result["predictions"].columns) == {"sample_id", "y_true", "y_pred", "y_score", "fold"}
    assert set(result["metrics_summary"]["metric"]) == {"accuracy", "roc_auc"}


def test_regression_workflow_runs_end_to_end() -> None:
    dataset = make_regression_dataset()
    result = run_supervised_workflow(
        dataset,
        target_col="target",
        task="regression",
        n_splits=3,
    )

    assert len(result["metrics_by_fold"]) == 3
    assert set(result["predictions"].columns) == {"sample_id", "y_true", "y_pred", "fold"}
    assert set(result["metrics_summary"]["metric"]) == {"rmse", "r2"}


def test_supervised_workflow_rejects_missing_target() -> None:
    dataset = make_classification_dataset()
    dataset.obs.loc[0, "target"] = None

    try:
        run_supervised_workflow(dataset, target_col="target", task="classification", n_splits=2)
    except ClinOmicsError as exc:
        assert "missing values" in str(exc)
    else:
        raise AssertionError("Expected ClinOmicsError for missing target values.")
