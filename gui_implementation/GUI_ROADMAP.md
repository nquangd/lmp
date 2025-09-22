# LMP GUI Roadmap

This document tracks the path to a first complete GUI for `lmp_pkg`. Update task status here as work progresses.

## Legend
- [x] Completed
- [ ] Pending / in progress

---

## 1. Core Infrastructure
- [x] Establish run-type metadata model (`RunRequest`, prefixes, labels).
- [x] Persist per-run metadata (labels, overrides, summary metrics) in workspace.
- [x] Ensure worker applies JSON parameter overrides and emits summary metrics.

## 2. Study Designer & Run Planning
- [x] Use study config name as default run label and propagate through run plans.
- [x] Parameter sweep planning: validate sweep JSON, generate manifest, queue child runs.
- [ ] Extend UI with dedicated panels for sensitivity, parameter estimation, virtual trials, VBE (config forms + validation).
- [ ] Support importing external manifest/CSV definitions for advanced sweeps.

## 3. Run Queue & Execution
- [x] Queue sweep runs with parent run ID, child overrides, and progress tracking.
- [ ] Add bulk controls (pause/resume/cancel group, rerun failed items).
- [ ] Surface warnings/errors from worker metadata in the queue table.

## 4. Results Viewer
- [x] Raw view: dataset switching, PBPK multi-select, existing plots.
- [x] Study summary mode with parameter/metric selectors, aggregated table, scaffolded plots.
- [ ] Populate study plots with richer chart types (contour, interactive Plotly) when ready.
- [ ] Provide export button for aggregated summaries (CSV/JSON).

## 5. Additional Run Types
- [ ] Sensitivity analysis: generate perturbation manifest, captures local gradients.
- [ ] Parameter estimation: integrate optimisation workflow, present fit metrics.
- [ ] Virtual trial: seed cohort generation, aggregate population metrics.
- [ ] Virtual bioequivalence: multi-product comparison, compute BE statistics.

## 6. UX & Polish
- [ ] Add empty-state guidance/tooltips across tabs.
- [ ] Theme/layout adjustments for consistent spacing and sizing.
- [ ] Optional Plotly integration for interactive plots (behind feature flag).

## 7. Testing & Packaging
- [ ] Expand automated tests for GUI orchestration (run plans, sweep manifests, metadata).
- [ ] Smoke-test on macOS/Windows, document setup steps.
- [ ] Prepare release notes / user guide for first full GUI version.

---

**Instructions:**
- Mark items with `[x]` when complete and add brief notes/dates if helpful.
- Append new tasks beneath relevant sections to keep the roadmap current.
