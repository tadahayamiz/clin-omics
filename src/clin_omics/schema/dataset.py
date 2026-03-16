from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from clin_omics.exceptions import SchemaValidationError


@dataclass(frozen=True)
class LayerSpec:
    name: str
    n_obs: int
    n_var: int



def validate_layer_shapes(
    layers: Mapping[str, pd.DataFrame], *, n_obs: int, n_var: int
) -> list[LayerSpec]:
    specs: list[LayerSpec] = []
    for name, layer in layers.items():
        if layer.shape != (n_obs, n_var):
            raise SchemaValidationError(
                f"Layer '{name}' has shape {layer.shape}, expected {(n_obs, n_var)}."
            )
        specs.append(LayerSpec(name=name, n_obs=n_obs, n_var=n_var))
    return specs
