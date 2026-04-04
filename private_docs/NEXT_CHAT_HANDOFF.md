# NEXT_CHAT_HANDOFF

## What was done in this turn

- Added the minimal public usage/example surface for the feature-vs-obs plotting path.
- Updated `README.md` so the new CLI command appears in the command list with concrete one-feature examples.
- Added a small Python workflow example using `run_plot_feature_vs_obs_from_h5`.
- Refreshed schedule and handoff to mark Phase A7 complete.

## Positioning of this turn

- docs refresh for Phase A7
- this turn intentionally improved discoverability rather than broadening features
- the plotting MVP now has a minimal public README surface in addition to the existing tests, workflow, and CLI

## Next one-theme task

Implement Phase B1 only if an actual use case requires it:

1. keep the current plotting path strict
2. avoid broad association-framework claims
3. do not start batch screening or multi-group inference yet

## Recommended implementation boundary

Touch only the smallest likely set:

- the smallest code/docs surface required by the next real use case
- schedule/handoff refresh
- no production-code broadening unless a real usability gap appears

## Do not do yet

- batch screening
- multi-feature report generation
- richer multi-group statistics
- project-specific convenience shortcuts

## Notes to preserve

Current strict behavior:

- supports `X` and explicit `layer`
- requires categorical-like grouping labels
- drops missing group / feature values with explicit counts
- aligns groups by `sample_id` rather than obs row order
- rejects duplicate feature indices
- top and right spines are hidden by default
- workflow entry point: `clin_omics.workflows.plot_feature_vs_obs_from_h5`
- CLI entry point: `clin-omics plot-feature-vs-obs`
- output remains one figure prefix plus one summary JSON per call
- optional `--annotate-mann-whitney` is available only for exact two-group plots
