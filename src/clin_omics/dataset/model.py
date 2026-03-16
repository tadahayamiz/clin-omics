from __future__ import annotations

import json
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any

import h5py
import pandas as pd

from clin_omics.constants import REQUIRED_OBS_ID_COLUMN, REQUIRED_VAR_ID_COLUMN, SCHEMA_VERSION
from clin_omics.dataset.provenance import default_provenance, generate_dataset_id
from clin_omics.dataset.validate import validate_dataset_components


@dataclass(slots=True)
class CanonicalDataset:
    X: pd.DataFrame
    obs: pd.DataFrame
    var: pd.DataFrame
    layers: dict[str, pd.DataFrame] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=default_provenance)
    dataset_id: str = field(default_factory=generate_dataset_id)
    embeddings: dict[str, pd.DataFrame] = field(default_factory=dict)
    feature_scores: dict[str, pd.DataFrame] = field(default_factory=dict)
    assignments: dict[str, pd.Series] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_dataset_components(
            self.X,
            self.obs,
            self.var,
            self.layers,
            self.embeddings,
            self.feature_scores,
            self.assignments,
        )
        self.provenance.setdefault("schema_version", SCHEMA_VERSION)

    def save_h5(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(output_path, "w") as handle:
            handle.attrs["schema_version"] = SCHEMA_VERSION
            handle.attrs["dataset_id"] = self.dataset_id

            matrix_group = handle.create_group("matrix")
            matrix_group.create_dataset("X", data=self.X.to_numpy())

            layers_group = matrix_group.create_group("layers")
            for name, layer in self.layers.items():
                layers_group.create_dataset(name, data=layer.to_numpy())

            axis_group = handle.create_group("axis")
            axis_group.create_dataset(
                "obs_json",
                data=self.obs.to_json(orient="table").encode("utf-8"),
            )
            axis_group.create_dataset(
                "var_json",
                data=self.var.to_json(orient="table").encode("utf-8"),
            )

            derived_group = handle.create_group("derived")
            embeddings_group = derived_group.create_group("embeddings")
            for name, frame in self.embeddings.items():
                embeddings_group.create_dataset(
                    name,
                    data=frame.to_json(orient="table").encode("utf-8"),
                )
            feature_scores_group = derived_group.create_group("feature_scores")
            for name, frame in self.feature_scores.items():
                feature_scores_group.create_dataset(
                    name,
                    data=frame.to_json(orient="table").encode("utf-8"),
                )
            assignments_group = derived_group.create_group("assignments")
            for name, series in self.assignments.items():
                assignments_group.create_dataset(
                    name,
                    data=series.to_frame(name="value").to_json(orient="table").encode("utf-8"),
                )

            provenance_group = handle.create_group("provenance")
            provenance_group.create_dataset(
                "json",
                data=json.dumps(self.provenance, sort_keys=True).encode("utf-8"),
            )

        return output_path

    @classmethod
    def load_h5(cls, path: str | Path) -> "CanonicalDataset":
        input_path = Path(path)
        with h5py.File(input_path, "r") as handle:
            obs = pd.read_json(
                StringIO(handle["axis"]["obs_json"][()].decode("utf-8")), orient="table"
            )
            var = pd.read_json(
                StringIO(handle["axis"]["var_json"][()].decode("utf-8")), orient="table"
            )

            sample_ids = obs[REQUIRED_OBS_ID_COLUMN].tolist()
            feature_ids = var[REQUIRED_VAR_ID_COLUMN].tolist()

            X = pd.DataFrame(
                handle["matrix"]["X"][()], index=sample_ids, columns=feature_ids
            )

            layers: dict[str, pd.DataFrame] = {}
            if "layers" in handle["matrix"]:
                for name, dataset in handle["matrix"]["layers"].items():
                    layers[name] = pd.DataFrame(
                        dataset[()], index=sample_ids, columns=feature_ids
                    )

            embeddings: dict[str, pd.DataFrame] = {}
            feature_scores: dict[str, pd.DataFrame] = {}
            assignments: dict[str, pd.Series] = {}
            if "derived" in handle:
                if "embeddings" in handle["derived"]:
                    for name, dataset in handle["derived"]["embeddings"].items():
                        embeddings[name] = pd.read_json(
                            StringIO(dataset[()].decode("utf-8")), orient="table"
                        )
                if "feature_scores" in handle["derived"]:
                    for name, dataset in handle["derived"]["feature_scores"].items():
                        feature_scores[name] = pd.read_json(
                            StringIO(dataset[()].decode("utf-8")), orient="table"
                        )
                if "assignments" in handle["derived"]:
                    for name, dataset in handle["derived"]["assignments"].items():
                        frame = pd.read_json(
                            StringIO(dataset[()].decode("utf-8")), orient="table"
                        )
                        series = frame["value"].copy()
                        series.index = sample_ids
                        series.name = name
                        assignments[name] = series

            provenance = json.loads(handle["provenance"]["json"][()].decode("utf-8"))
            dataset_id = str(handle.attrs["dataset_id"])

        return cls(
            X=X,
            obs=obs,
            var=var,
            layers=layers,
            provenance=provenance,
            dataset_id=dataset_id,
            embeddings=embeddings,
            feature_scores=feature_scores,
            assignments=assignments,
        )
