import pandas as pd

from clin_omics.dataset import CanonicalDataset, build_dataset


def test_build_dataset_from_matrix_and_clinical() -> None:
    omics = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "g1": [1.0, 3.0],
            "g2": [2.0, 4.0],
        }
    )
    clinical = pd.DataFrame(
        {
            "sid": ["s2", "s1"],
            "group": ["control", "case"],
        }
    )

    dataset = build_dataset(
        omics=omics,
        clinical=clinical,
        omics_format="matrix",
        clinical_sample_id_col="sid",
    )

    assert isinstance(dataset, CanonicalDataset)
    assert dataset.X.index.tolist() == ["s1", "s2"]
    assert dataset.obs["sample_id"].tolist() == ["s1", "s2"]
    assert dataset.obs["group"].tolist() == ["case", "control"]
    assert dataset.var["feature_id"].tolist() == ["g1", "g2"]
    assert dataset.provenance["build"]["omics_format"] == "matrix"


def test_build_dataset_from_long_without_clinical() -> None:
    omics = pd.DataFrame(
        {
            "sample_id": ["s1", "s1", "s2", "s2"],
            "feature_id": ["g1", "g2", "g1", "g2"],
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )

    dataset = build_dataset(omics=omics, omics_format="long")

    assert dataset.X.shape == (2, 2)
    assert dataset.obs["sample_id"].tolist() == ["s1", "s2"]
    assert dataset.var["feature_id"].tolist() == ["g1", "g2"]


def test_build_dataset_with_feature_meta_reindexed_to_matrix_order() -> None:
    omics = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "g1": [1.0, 3.0],
            "g2": [2.0, 4.0],
        }
    )
    feature_meta = pd.DataFrame(
        {
            "feature_id": ["g2", "g1"],
            "feature_name": ["gene2", "gene1"],
        }
    )

    dataset = build_dataset(omics=omics, feature_meta=feature_meta, omics_format="matrix")

    assert dataset.var["feature_id"].tolist() == ["g1", "g2"]
    assert dataset.var["feature_name"].tolist() == ["gene1", "gene2"]
