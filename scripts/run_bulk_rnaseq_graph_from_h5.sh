#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/run_bulk_rnaseq_graph_from_h5.sh <DATASET_H5> <OUTDIR> [extra args ...]" >&2
  exit 1
fi

DATASET_H5="$1"
OUTDIR="$2"
shift 2

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
python -m clin_omics.workflows.bulk_rnaseq_graph_from_h5 \
  --dataset-h5 "$DATASET_H5" \
  --outdir "$OUTDIR" \
  "$@"
