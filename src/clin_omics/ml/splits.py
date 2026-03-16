from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

from clin_omics.exceptions import ClinOmicsError


CVMethod = KFold | StratifiedKFold | GroupKFold


def get_cv_splitter(
    method: str,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int | None = 0,
) -> CVMethod:
    if n_splits < 2:
        raise ClinOmicsError("n_splits must be at least 2.")

    if method == "kfold":
        return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    if method == "stratified_kfold":
        return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    if method == "group_kfold":
        return GroupKFold(n_splits=n_splits)

    raise ClinOmicsError(f"Unsupported CV method: {method}")


def iter_cv_splits(
    X: pd.DataFrame,
    y: pd.Series,
    splitter: CVMethod,
    groups: pd.Series | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    if isinstance(splitter, GroupKFold):
        if groups is None:
            raise ClinOmicsError("groups must be provided for GroupKFold splits.")
        yield from splitter.split(X, y, groups)
        return

    if isinstance(splitter, StratifiedKFold):
        yield from splitter.split(X, y)
        return

    yield from splitter.split(X)
