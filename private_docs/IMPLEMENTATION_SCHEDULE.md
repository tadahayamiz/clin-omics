# IMPLEMENTATION_SCHEDULE

## Current interpretation lock

- **Current main theme**: add a minimal feature-vs-feature scatter plotting path for marker orthogonality inspection.
- **This turn scope**: one strict plotting path only. Support two selected features on x/y axes, optional categorical-like point labels, styling overrides, and narrow H5 workflow/CLI integration.
- **Touched this turn**: association extraction path, scatter visualization, H5 workflow/CLI surface, focused tests/smoke checks, and minimal public README refresh.
- **Do not touch yet**: regression overlays, multi-marker screening, or broader association generalization.
- **Current change type**: strict specification.

---

## Why this is the next priority

The current repo already supports:

- canonical dataset construction
- basic preprocessing and clustering flows
- embedding plots
- single `obs` field summarization plots

But it does **not** yet support the practical figure type needed for clinical interpretation:

- choose one clinical grouping label from `obs`
- choose one target feature from `X` or a specified layer
- compare distributions across groups
- render publication-style dot/strip-based plots with optional summary overlays and statistical annotation

This is the most immediate missing visualization bridge between `obs` and assay values.

---

## Priority roadmap

### Phase A. Feature-vs-obs plotting: strict MVP

Goal: establish one strict, reusable plotting path for the figure family typified by `clinical group vs feature value`.

#### A1. Analysis contract and plotting contract
- [x] Define a narrow analysis contract for extracting a single feature vector aligned to `obs`
- [x] Decide supported assay source for MVP:
  - strict implementation supports `X` and explicit `layer`
- [x] Define strict group handling for MVP:
  - categorical-like `obs` field only
  - missing values dropped with explicit count reporting
- [x] Define strict return schema for plot-ready comparison data
- [ ] Decide whether two-group statistics are MVP-strict or deferred

#### A2. Visualization module
- [x] Add `src/clin_omics/visualization/association.py`
- [x] Implement a plot function for `feature vs obs group`
- [x] Support publication-oriented defaults:
  - strip/dot style as primary visual element
  - optional box/summary overlay
  - no top spine
  - no right spine
- [x] Keep output routing compatible with existing save helpers

#### A3. Styling and override surface
- [x] Extend style handling so fine-grained overrides are possible without making the API project-specific
- [x] Allow explicit control of:
  - figure size
  - fontsize family
  - tick fontsize
  - marker size
  - line width
  - transparency / alpha
  - jitter strength
  - y-scale
  - spine visibility
- [x] Allow explicit color overrides per group
- [x] Set plotting defaults for colors:
  - control = gray
  - treated single group = blue-family
  - multiple treatment groups = Tol bright palette
- [x] Ensure defaults remain deterministic and reusable

#### A4. Workflow and CLI entry point
- [x] Add workflow entry point `plot_feature_vs_obs_from_h5.py`
- [x] Read dataset from H5 and route plotting args cleanly
- [x] Add narrow CLI command `plot-feature-vs-obs`
- [x] Keep CLI scope narrow: one feature, one obs field, one output prefix per call

#### A5. Statistical annotation
- [x] For strict MVP, support optional two-group Mann–Whitney annotation
- [x] Define behavior when group count is not two:
  - strict error when annotation is requested
- [x] Define label text formatting for p-values and test name
- [x] Defer multi-group inference unless needed for first use case

#### A6. Tests
- [x] Add focused tests for feature extraction and alignment
- [x] Add focused tests for group filtering and missing handling
- [x] Add visualization tests for:
  - color resolution
  - spine removal
  - override application
  - output generation
  - optional Mann-Whitney annotation
- [x] Add workflow/CLI smoke test for one end-to-end figure generation

---


### Phase C. Feature-vs-feature scatter for marker orthogonality

Goal: establish one strict, reusable plotting path for the figure family typified by `marker A vs marker B`, with patients as points and an optional clinical label as point color.

#### C1. Analysis contract and plotting contract
- [x] Define a narrow analysis contract for extracting two aligned feature vectors
- [x] Support `X` and explicit `layer` selection
- [x] Support optional categorical-like `label_field` from `obs` for point colors
- [x] Drop missing x / y / label values with explicit count reporting
- [x] Support exact feature IDs and explicit `var`-column lookup for each axis

#### C2. Visualization module
- [x] Add feature-vs-feature scatter plotting in `src/clin_omics/visualization/association.py`
- [x] Keep patients as points, with optional color by label
- [x] Hide top/right spines by default
- [x] Keep output routing compatible with existing save helpers

#### C3. Styling and override surface
- [x] Support figure size overrides
- [x] Support marker size overrides
- [x] Support marker shape overrides
- [x] Support marker edge width overrides
- [x] Support alpha overrides
- [x] Support explicit label color overrides
- [x] Use control gray / treated blue / Tol bright defaults when labels are present

#### C4. Workflow and CLI entry point
- [x] Add workflow entry point `plot_feature_vs_feature_from_h5.py`
- [x] Add narrow CLI command `plot-feature-vs-feature`
- [x] Keep CLI scope narrow: two features, one optional label field, one output prefix per call

#### C5. Validation surface
- [x] Add focused tests for feature resolution, alignment, and missing handling
- [x] Add visualization tests for style overrides and output generation
- [x] Add workflow and CLI smoke coverage
- [ ] Run related pytest cleanly in a stable environment

### Phase B. Generalize after the MVP is stable

Only start after Phase A is strict and tested.

#### B1. Broader data source support
- [ ] Support explicit `layer` selection
- [x] Consider `var`-based feature lookup helpers
- [ ] Consider protein / gene naming alias resolution only if the canonical schema already supports it cleanly

#### B2. Additional plot families
- [ ] box-only
- [ ] violin + strip
- [ ] paired plots if the repo later gains longitudinal semantics
- [ ] log-axis helpers with safer validation

#### B3. Richer statistics
- [ ] multi-group tests
- [ ] multiple-testing aware batch mode
- [ ] effect size reporting
- [ ] confidence interval overlays

#### B4. Batch reporting
- [ ] repeated plotting from a feature list
- [ ] table export of per-feature summary statistics
- [ ] summary report linking figures and test results

---

## Strict vs temporary decisions

### Strict for the planned MVP
- one feature at a time
- one clinical grouping field at a time
- explicit override surface for styling
- no top/right spines by default
- explicit group-color control
- deterministic default palette behavior
- focused workflow and focused CLI entry point

### Deferred, not temporary
- broad association-screening framework
- batch multi-feature inference
- complex multiple-testing workflows
- notebook-oriented convenience wrappers
- project-specific auto-formatting rules

### Avoid unless forced
- silent fallback from unknown group names to inferred control/treatment semantics
- overloaded CLI that performs screening and plotting in one command
- plot styling rules encoded ad hoc inside workflows instead of shared visualization config

---

## Proposed file touch plan for the future code turn

Expected minimal files for the first implementation turn:

- `src/clin_omics/analysis/association.py`
- `src/clin_omics/visualization/association.py`
- `src/clin_omics/workflows/plot_feature_vs_obs_from_h5.py`
- `src/clin_omics/cli.py`
- `src/clin_omics/visualization/style.py`
- directly related tests only

Keep all other modules untouched unless the first strict implementation proves this impossible.

---

## Current status ledger

- [x] Gap identified: repo lacks direct `obs group vs assay feature` plotting
- [x] Priority fixed: implement plotting path before broader association analysis
- [x] Visual requirements fixed at planning level:
  - fine-grained style overrides required
  - fine-grained color overrides required
  - default colors: control gray, treated blue-family, multi-group Tol bright
  - top/right spines removed by default
- [x] Private docs scaffold created
- [x] Code implementation started
- [x] Related tests added
- [x] H5 workflow and CLI path added
- [x] Remaining focused A6 tests added
- [x] Public README updated
- [x] Strict var-based feature lookup helper added for feature-vs-obs
- [x] Strict feature-vs-feature scatter path added for marker orthogonality inspection

---

## Next one-theme task

Keep the current plotting paths strict after C4/C5:

- do not broaden into regression overlays or batch screening yet
- consider label-shape or faceting only if a real use case now requires it
- avoid heuristic alias resolution beyond explicit `var` column lookup


## 2026-04-04 bugfix
- Fixed feature-vs-feature CLI failure handling for non-categorical numeric label fields.
- Restored feature-vs-obs workflow compatibility when feature_lookup_col is absent on older callers/tests.


## 2026-04-04 axis-scale follow-up
- Added strict `xscale` / `yscale` support to `plot-feature-vs-feature` CLI, workflow, and visualization path.
- Related focused tests updated for CLI acceptance and scatter config propagation.
