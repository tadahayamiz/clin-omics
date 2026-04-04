# NEXT_CHAT_HANDOFF

## What was done in this turn

- Added a strict `feature-vs-feature` scatter plotting path for marker orthogonality inspection.
- Patients are plotted as points, with an optional categorical-like `label_field` used for point colors.
- Added explicit styling overrides for figure size, marker size, marker shape, marker edge width, alpha, and top/right spine visibility.
- Added an H5 workflow entry point and a narrow CLI command: `clin-omics plot-feature-vs-feature`.
- Refreshed the public README with minimal usage examples.

## Positioning of this turn

- strict specification for a new but still narrow plot family
- this turn did not broaden into screening, regression overlays, or richer multivariate marker analysis
- the plotting path remains two features, one optional label field, one output prefix per call

## Next one-theme task

Only move to the next theme if there is a real use case now requiring it:

1. keep the current plotting paths strict
2. consider regression / density / quadrant overlays only if they become necessary for interpretation
3. avoid broad association-framework claims

## Recommended implementation boundary

Touch only the smallest likely set:

- focused tests or docs refresh for the new scatter path
- optional overlays only if directly required
- schedule/handoff refresh
- no batch screening unless explicitly needed

## Do not do yet

- automatic quadrant calling or positivity thresholds
- regression or density overlays by default
- multi-feature report generation
- heuristic alias resolution across multiple `var` columns

## Notes to preserve

Current strict behavior:

- supports `X` and explicit `layer`
- supports exact `feature_id` lookup and optional lookup from one explicit `var` column per axis
- optional `label_field` must be categorical-like
- drops missing x / y / label values with explicit counts
- aligns labels by `sample_id` rather than obs row order
- top and right spines are hidden by default
- workflow entry point: `clin_omics.workflows.plot_feature_vs_feature_from_h5`
- CLI entry point: `clin-omics plot-feature-vs-feature`
- output remains one figure prefix plus one summary JSON per call


## 2026-04-04 bugfix
- Fixed feature-vs-feature CLI failure handling for non-categorical numeric label fields.
- Restored feature-vs-obs workflow compatibility when feature_lookup_col is absent on older callers/tests.


## 2026-04-04 axis-scale follow-up
- Added strict `xscale` / `yscale` support to `plot-feature-vs-feature` CLI, workflow, and visualization path.
- Related focused tests updated for CLI acceptance and scatter config propagation.
