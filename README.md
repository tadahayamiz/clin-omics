# clin-omics

Reusable Python framework for clinical and omics data integration, canonical dataset construction, and downstream analysis.

This repository is a library / framework, not a project-specific experiment repository.
It does not store datasets, notebooks, or experiment-specific config files.
The intended usage is:

1. interactively curate and preprocess omics / clinical tables in your own project
2. convert the curated result into a canonical HDF5 dataset
3. run reusable downstream analysis and ML workflows through the library API or CLI

---

## Scope and design

`clin-omics` is organized around a canonical dataset object with:

- assay matrix `X` (`samples x features`)
- sample metadata `obs`
- feature metadata `var`
- optional `layers`
- derived artifacts such as embeddings, feature scores, and assignments
- provenance metadata

The repository intentionally separates:

- interactive curation / preprocessing: done mainly through library calls
- canonical dataset construction and downstream analysis: available both as library calls and CLI commands

In particular, trial-and-error-heavy preprocessing is not the responsibility of the CLI.
The CLI starts to become useful after `X / obs / var` are already reasonably well prepared.

### Bulk RNA-seq preprocessing notes

For bulk RNA-seq, this package now provides lightweight preprocessing helpers for
count matrices:

- `FilterLowExpression`: keep features with enough non-trivial counts
- `CPMNormalizer`: library-size normalization to CPM
- `LogCPMTransform`: exploratory `log2(CPM + prior_count)` representation
- `BulkRNASeqPreprocessor`: convenience wrapper that creates `counts_raw`,
  `counts_filtered`, `cpm`, `log_cpm`, and optional `zscore_log_cpm` layers

Recommended usage by task:

- DEG / differential expression: keep raw or filtered counts and use an
  external count-aware workflow such as DESeq2, edgeR, or limma-voom
- PCA / clustering / sample networks / gene co-expression: use `log_cpm`
- heatmaps of selected features: optionally use `zscore_log_cpm`

`ZScoreScaler` remains available as a generic transform, but for bulk RNA-seq it
should usually be treated as a visualization helper applied after `log_cpm`, not
as the primary normalization step for counts.

---

## Package layout

```text
src/clin_omics/
  io/
  schema/
  curation/
  dataset/
  preprocess/
  analysis/
  ml/
  export/
  workflows/
  utils/
```

The main responsibilities are:

- `schema/`: validation rules for IDs, tables, and canonical dataset structure
- `curation/`: omics / clinical table normalization and merging helpers
- `dataset/`: canonical dataset object and HDF5 persistence
- `preprocess/`: reusable matrix-level preprocessing transforms
- `analysis/`: embeddings, clustering, and QC helpers
- `ml/`: supervised learning utilities and workflows
- `export/`: export to tables and optional AnnData interop
- `workflows/`: higher-level entry points

---

## Installation

Editable install:

```bash
pip install -e .
```

With development dependencies:

```bash
pip install -e ".[dev]"
```

With optional AnnData support:

```bash
pip install -e ".[anndata]"
```

With optional Leiden support:

```bash
pip install -e ".[leiden]"
```

---

## Quick start

### 1. Build a canonical dataset from curated tables

At this stage, `X`, `obs`, and `var` are assumed to be already curated.
For example:

- `X`: samples x features matrix
- `obs`: sample metadata with `sample_id`
- `var`: feature metadata with `feature_id`

Library usage:

```python
import pandas as pd
from clin_omics.dataset import CanonicalDataset

X = pd.read_parquet("assay.parquet").set_index("sample_id")
obs = pd.read_parquet("obs.parquet")
var = pd.read_parquet("var.parquet")

dataset = CanonicalDataset(X=X, obs=obs, var=var)
dataset.save_h5("dataset.h5")
```

CLI usage:

```bash
clin-omics build-dataset \
  --x assay.parquet \
  --obs obs.parquet \
  --var var.parquet \
  --out dataset.h5
```

This command is intentionally narrow in scope.
It does not try to perform long-to-matrix conversion, metadata cleaning, or exploratory preprocessing.
Those should be done interactively beforehand.

### Bulk RNA-seq flow scripts

Reusable end-to-end flow logic lives under `src/clin_omics/workflows/`, while `scripts/` is reserved for thin shell runners.
They are intended for standard, repeatable table-to-dataset-to-analysis runs,
including Colab usage, rather than one-off experiment notebooks.

Current bulk RNA-seq flows:

- `clin_omics.workflows.bulk_rnaseq_basic`
  - canonical dataset construction
  - bulk RNA-seq preprocessing
  - PCA on `log_cpm`
  - optional k-means and optional UMAP
- `clin_omics.workflows.bulk_rnaseq_graph`
  - canonical dataset construction
  - bulk RNA-seq preprocessing
  - PCA on `log_cpm`
  - kNN graph + Leiden clustering
  - optional UMAP

The matching shell wrappers are:

- `scripts/run_bulk_rnaseq_basic.sh`
- `scripts/run_bulk_rnaseq_graph.sh`

HVG selection is intentionally **not** part of the default bulk RNA-seq flow.
For bulk RNA-seq, it is better treated as an optional refinement than as a required first step.

#### Basic flow

```bash
bash scripts/run_bulk_rnaseq_basic.sh   /content/project/assay.csv   /content/project/obs.csv   /content/project/var.csv   /content/project/basic_out   --min-count 10   --min-samples 2   --make-zscore
```

This helper will:

- build a canonical dataset from `X / obs / var`
- run `BulkRNASeqPreprocessor`
- run PCA on `log_cpm`
- run a small k-means clustering on the PCA embedding by default
- save `pca_basic` figures as both PNG and SVG
- write `bulk_rnaseq_basic_summary.json` and `cluster_kmeans_basic.csv`

Outputs are written under the requested output directory, including:

- `bulk_rnaseq_basic_input_dataset.h5`
- `bulk_rnaseq_basic_processed_dataset.h5`
- `bulk_rnaseq_basic_summary.json`
- `cluster_kmeans_basic.csv`
- `pca_basic.png`
- `pca_basic.svg`

Add `--run-umap` if you also want a quick UMAP check on `log_cpm`.
Add `--skip-kmeans` if you only want preprocessing + PCA outputs.

#### Graph flow

```bash
bash scripts/run_bulk_rnaseq_graph.sh   /content/project/assay.csv   /content/project/obs.csv   /content/project/var.csv   /content/project/graph_out   --min-count 10   --min-samples 2   --n-pca-components 10   --leiden-neighbors 15   --leiden-resolution 1.0   --run-umap
```

This flow adds graph-based clustering after PCA and writes `cluster_leiden_graph.csv`.
It requires the optional Leiden dependencies:

```bash
pip install -e ".[leiden]"
```

#### Colab-oriented one-cell snippets

Basic flow in one cell after cloning the repo and changing into the repo root:

```bash
pip install -e . && bash scripts/run_bulk_rnaseq_basic.sh /content/project/assay.csv /content/project/obs.csv /content/project/var.csv /content/project/basic_out --min-count 10 --min-samples 2 --make-zscore --run-umap
```

Graph flow in one cell:

```bash
pip install -e ".[leiden]" && bash scripts/run_bulk_rnaseq_graph.sh /content/project/assay.csv /content/project/obs.csv /content/project/var.csv /content/project/graph_out --min-count 10 --min-samples 2 --n-pca-components 10 --leiden-neighbors 15 --leiden-resolution 1.0 --run-umap
```

---

## CLI overview

Current CLI commands:

```bash
clin-omics version
clin-omics inspect <dataset.h5>
clin-omics validate <dataset.h5>
clin-omics build-dataset --x ... --obs ... --var ... --out ...
clin-omics pca --in ... --out ...
clin-omics factor-analysis --in ... --out ...
clin-omics umap --in ... --out ...
clin-omics cluster-kmeans --in ... --out ... --n-clusters ...
clin-omics cluster-knn-leiden --in ... --out ... --neighbors ... --resolution ...
clin-omics plot-embedding --in ... --embedding-key ... --out-prefix ...
clin-omics export-assignments --in ... --key ... --out ...
```

### `version`

```bash
clin-omics version
```

Shows the installed package version.

### `inspect`

```bash
clin-omics inspect dataset.h5
```

Prints a JSON summary including:

- dataset ID
- schema version
- matrix shape
- obs / var columns
- layers
- embeddings
- feature scores
- assignments
- provenance keys
- `n_samples` / `n_features`
- `layer_shapes`
- `embedding_shapes`
- `feature_score_shapes`
- `assignment_lengths`

### `validate`

```bash
clin-omics validate dataset.h5
```

Validates that the HDF5 dataset matches the canonical structure.
On success it prints `VALID`.
On failure it exits non-zero and prints an error message.

### `build-dataset`

```bash
clin-omics build-dataset \
  --x assay.parquet \
  --obs obs.parquet \
  --var var.parquet \
  --out dataset.h5
```

Creates a canonical dataset from curated tables.

Supported table formats:

- `.csv`
- `.tsv`
- `.txt`
- `.parquet`

### `pca`

```bash
clin-omics pca \
  --in dataset.h5 \
  --out dataset_pca.h5 \
  --n-components 10 \
  --key pca
```

Runs PCA and stores the resulting embedding and loadings in the output dataset.

### `cluster-kmeans`

Cluster on `X` directly:

```bash
clin-omics cluster-kmeans \
  --in dataset.h5 \
  --out dataset_kmeans.h5 \
  --n-clusters 5
```

Cluster on a stored embedding instead:

```bash
clin-omics cluster-kmeans \
  --in dataset_pca.h5 \
  --out dataset_kmeans.h5 \
  --n-clusters 5 \
  --embedding-key pca
```

### `cluster-knn-leiden`

```bash
clin-omics cluster-knn-leiden \
  --in dataset_pca.h5 \
  --out dataset_leiden.h5 \
  --embedding-key pca \
  --neighbors 15 \
  --resolution 1.0
```

This command is intended for the common `kNN -> Leiden` workflow.
It is particularly useful when you want to sweep parameters from shell scripts.

Example shell-side parameter sweep:

```bash
for k in 10 15 20; do
  for r in 0.5 1.0 1.5; do
    clin-omics cluster-knn-leiden \
      --in dataset_pca.h5 \
      --out "leiden_k${k}_r${r}.h5" \
      --embedding-key pca \
      --neighbors "$k" \
      --resolution "$r"
  done
done
```

Note: `cluster-knn-leiden` requires `igraph` and `leidenalg` at runtime.


### Recommended bulk-omics workflow

A practical default workflow for bulk omics is:

1. curate `X / obs / var`
2. build a canonical dataset
3. run PCA or factor analysis for compact representations
4. run `cluster-kmeans` or `cluster-knn-leiden`
5. run UMAP mainly for visualization
6. use `plot-embedding --color <assignment_key>` to inspect clusters
7. use `export-assignments` to recover a sample-level cluster table

Example:

```bash
clin-omics pca --in dataset.h5 --out dataset_pca.h5 --n-components 10 --key pca
clin-omics cluster-knn-leiden --in dataset_pca.h5 --out dataset_leiden.h5 --embedding-key pca --neighbors 15 --resolution 1.0
clin-omics umap --in dataset_pca.h5 --out dataset_umap.h5 --embedding-key pca --key umap --random-state 0
clin-omics plot-embedding --in dataset_umap.h5 --embedding-key umap --color cluster_knn_leiden --out-prefix out/umap_by_leiden
clin-omics export-assignments --in dataset_leiden.h5 --key cluster_knn_leiden --out cluster_knn_leiden.csv
```

---

## Main library usage patterns

### Canonical dataset

Core object:

```python
from clin_omics.dataset import CanonicalDataset
```

Typical usage:

```python
import pandas as pd
from clin_omics.dataset import CanonicalDataset

X = pd.DataFrame(
    [[1.0, 2.0], [3.0, 4.0]],
    index=["S1", "S2"],
    columns=["G1", "G2"],
)
obs = pd.DataFrame({"sample_id": ["S1", "S2"], "group": ["A", "B"]})
var = pd.DataFrame({"feature_id": ["G1", "G2"], "symbol": ["g1", "g2"]})

dataset = CanonicalDataset(X=X, obs=obs, var=var)
dataset.save_h5("dataset.h5")
loaded = CanonicalDataset.load_h5("dataset.h5")
```

---

### Curation helpers

Use these during interactive preparation.

```python
from clin_omics.curation import (
    curate_omics_long_to_matrix,
    curate_omics_matrix,
    curate_clinical_table,
    build_obs_from_clinical,
    build_var_from_matrix,
)
```

Example: long table to matrix.

```python
omics_long = pd.DataFrame(
    {
        "sample_id": ["S1", "S1", "S2", "S2"],
        "feature_id": ["G1", "G2", "G1", "G2"],
        "value": [1.0, 2.0, 3.0, 4.0],
    }
)

X = curate_omics_long_to_matrix(
    omics_long,
    sample_id_col="sample_id",
    feature_id_col="feature_id",
    value_col="value",
)
```

Example: clinical table normalization and alignment.

```python
clinical = pd.DataFrame(
    {
        "sample": ["S1", "S2"],
        "sex": ["M", "F"],
        "group": ["case", "control"],
    }
)

obs_curated = curate_clinical_table(
    clinical,
    sample_id_col="sample",
    rename_map={"sample": "sample_id"},
    category_maps={"sex": {"M": "male", "F": "female"}},
)
obs = build_obs_from_clinical(X, obs_curated)
var = build_var_from_matrix(X)
```

---

### Dataset build workflow

If you want library-level construction from omics / clinical inputs:

```python
from clin_omics.workflows import build_dataset

dataset = build_dataset(
    omics=omics_long,
    clinical=clinical,
    omics_format="long",
    sample_id_col="sample_id",
    feature_id_col="feature_id",
    value_col="value",
)
```

This is useful when you still want a structured build step in Python, even if the CLI `build-dataset` is intentionally narrower.

---

### Preprocessing

Available preprocessing components:

```python
from clin_omics.preprocess import (
    Log1pTransform,
    ZScoreScaler,
    VarianceFilter,
    PreprocessPipeline,
)
```

Example:

```python
pipeline = PreprocessPipeline(
    [
        Log1pTransform(target="layer", target_layer="log1p"),
        ZScoreScaler(source_layer="log1p"),
    ]
)

dataset2 = pipeline.fit_transform(dataset)
```

Use preprocessing interactively when trying out options.
If a preprocessing branch changes the sample / feature set substantially, it is usually better treated as a separate dataset artifact rather than as another internal layer.

---

### Analysis

Available analysis components:

```python
from clin_omics.analysis import (
    PCAEmbedding,
    FactorAnalysisEmbedding,
    UMAPEmbedding,
    KMeansClustering,
    KNNLeidenClustering,
    HierarchicalClustering,
    summarize_dataset_qc,
)
```

Example: PCA + k-means.

```python
from clin_omics.analysis import PCAEmbedding, KMeansClustering

pca_dataset = PCAEmbedding(n_components=10, key="pca").fit_transform(dataset)
clustered = KMeansClustering(n_clusters=5, embedding_key="pca").fit_predict(pca_dataset)
```

Example: factor analysis + UMAP for visualization.

```python
from clin_omics.analysis import FactorAnalysisEmbedding, UMAPEmbedding

fa_dataset = FactorAnalysisEmbedding(n_components=10, key="factor_analysis").fit_transform(dataset)
vis_dataset = UMAPEmbedding(n_components=2, source_layer="factor_analysis", key="umap").fit_transform(fa_dataset)
```

Example: QC summary.

```python
from clin_omics.analysis import summarize_dataset_qc

summary = summarize_dataset_qc(clustered)
print(summary)
```

---

### Supervised workflows

Available workflow:

```python
from clin_omics.workflows import run_supervised_workflow
```

Example: classification.

```python
result = run_supervised_workflow(
    dataset,
    target_col="response",
    task="classification",
    n_splits=5,
)

print(result["metrics_summary"])
```

Example: regression.

```python
result = run_supervised_workflow(
    dataset,
    target_col="score",
    task="regression",
    n_splits=5,
)
```

The current implementation intentionally keeps the first supervised layer minimal:

- classification: logistic regression
- regression: linear regression
- CV splitters: k-fold / stratified k-fold / group k-fold

---

### Export

Available export helpers:

```python
from clin_omics.export import (
    to_anndata,
    export_obs_table,
    export_var_table,
    export_embedding_table,
    export_feature_scores_table,
    export_assignments_table,
)
```

Example:

```python
obs_table = export_obs_table(dataset)
embedding_table = export_embedding_table(dataset, key="pca")
```

AnnData export:

```python
adata = to_anndata(dataset)
```

`anndata` is optional and only required when you actually call `to_anndata(...)`.

---

## Testing

Run the toy-data-based test suite with:

```bash
pytest -q
```

At the current stage, tests are based on toy data only.
Real-data tests can be added later in a separate project or an additional validation phase.

---

## Note
This repository is under construction and will be officially released by [Mizuno group](https://github.com/mizuno-group).  
Please contact tadahaya[at]gmail.com before publishing your paper using the contents of this repository.  

## Authors
- [Tadahaya Mizuno](https://github.com/tadahayamiz)  
    - correspondence  

## Contact
If you have any questions or comments, please feel free to create an issue on github here, or email us:  
- tadahaya[at]gmail.com  
    - lead contact
