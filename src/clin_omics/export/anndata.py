from __future__ import annotations

from typing import Any

from clin_omics.dataset import CanonicalDataset


def to_anndata(
    dataset: CanonicalDataset,
    *,
    include_layers: bool = True,
    include_embeddings: bool = True,
    include_feature_scores: bool = True,
    include_assignments: bool = True,
) -> Any:
    try:
        import anndata as ad
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError(
            "anndata is not installed. Install clin-omics with the 'anndata' extra to use to_anndata()."
        ) from exc

    obs = dataset.obs.set_index("sample_id", drop=False).copy()
    var = dataset.var.set_index("feature_id", drop=False).copy()
    adata = ad.AnnData(X=dataset.X.copy(), obs=obs, var=var)

    if include_layers:
        for name, layer in dataset.layers.items():
            adata.layers[name] = layer.to_numpy()

    if include_embeddings:
        for name, frame in dataset.embeddings.items():
            adata.obsm[name] = frame.to_numpy()

    if include_feature_scores:
        for name, frame in dataset.feature_scores.items():
            adata.varm[name] = frame.to_numpy()

    if include_assignments:
        for name, series in dataset.assignments.items():
            adata.obs[name] = series.reindex(dataset.obs["sample_id"]).to_numpy()

    adata.uns["dataset_id"] = dataset.dataset_id
    adata.uns["provenance"] = dict(dataset.provenance)
    return adata
