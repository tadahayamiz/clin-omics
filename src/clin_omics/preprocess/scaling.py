from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from clin_omics.preprocess.base import BasePreprocessor


@dataclass
class ZScoreScaler(BasePreprocessor):
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
