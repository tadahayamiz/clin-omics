#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: bash scripts/run_bulk_rnaseq_graph.sh <X.csv/tsv> <OBS.csv/tsv> <VAR.csv/tsv> <OUTDIR> [extra args...]" >&2
  exit 1
fi

X_PATH="$1"
OBS_PATH="$2"
VAR_PATH="$3"
OUTDIR="$4"
shift 4

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
python -m clin_omics.workflows.bulk_rnaseq_graph \
  --x "$X_PATH" \
  --obs "$OBS_PATH" \
  --var "$VAR_PATH" \
  --outdir "$OUTDIR" \
  "$@"
