#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: bash scripts/run_plot_obs_field_from_h5.sh <DATASET_H5> <FIELD> <OUTDIR> [extra args ...]" >&2
  exit 1
fi

DATASET_H5="$1"
FIELD="$2"
OUTDIR="$3"
shift 3

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
python -m clin_omics.workflows.plot_obs_field_from_h5 \
  --dataset-h5 "$DATASET_H5" \
  --field "$FIELD" \
  --outdir "$OUTDIR" \
  "$@"
