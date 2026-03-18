from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from clin_omics.preprocess.base import BasePreprocessor


@dataclass
class ZScoreScaler(BasePreprocessor):
    """Column-wise z-score scaling for continuous matrices.

    This transform is generic and can be applied to ``X`` or to any numeric
    layer specified via ``source_layer`` / ``target``. For bulk RNA-seq, it is
    best used as a downstream visualization helper after ``LogCPMTransform``
    rather than as the primary normalization step for counts.
    """
    ddof: int = 0
    means_: pd.Series = field(init=False)
    stds_: pd.Series = field(init=False)

    def _fit_frame(self, frame: pd.DataFrame) -> None:
        numeric = frame.astype(float)
        self.means_ = numeric.mean(axis=0)
        self.stds_ = numeric.std(axis=0, ddof=self.ddof).replace(0.0, 1.0)

    def _transform_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        numeric = frame.astype(float)
        scaled = (numeric - self.means_) / self.stds_
        return pd.DataFrame(
            scaled.to_numpy(dtype=float),
            index=frame.index.copy(),
            columns=frame.columns.copy(),
        )
