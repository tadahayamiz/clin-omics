from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from clin_omics.preprocess.base import BasePreprocessor


@dataclass
class Log1pTransform(BasePreprocessor):
    def _transform_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if (frame < 0).any().any():
            raise ValueError("Log1pTransform requires non-negative values.")
        return pd.DataFrame(
            np.log1p(frame.to_numpy(dtype=float)),
            index=frame.index.copy(),
            columns=frame.columns.copy(),
        )
