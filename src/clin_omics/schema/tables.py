from __future__ import annotations

import pandas as pd

from clin_omics.constants import REQUIRED_OBS_ID_COLUMN, REQUIRED_VAR_ID_COLUMN
from clin_omics.exceptions import SchemaValidationError
from clin_omics.schema.ids import validate_feature_ids, validate_sample_ids


def validate_obs_table(obs: pd.DataFrame) -> pd.DataFrame:
    if REQUIRED_OBS_ID_COLUMN not in obs.columns:
        raise SchemaValidationError(
            f"obs must contain required column '{REQUIRED_OBS_ID_COLUMN}'."
        )
    validate_sample_ids(obs[REQUIRED_OBS_ID_COLUMN])
    return obs.copy()



def validate_var_table(var: pd.DataFrame) -> pd.DataFrame:
    if REQUIRED_VAR_ID_COLUMN not in var.columns:
        raise SchemaValidationError(
            f"var must contain required column '{REQUIRED_VAR_ID_COLUMN}'."
        )
    validate_feature_ids(var[REQUIRED_VAR_ID_COLUMN])
    return var.copy()
