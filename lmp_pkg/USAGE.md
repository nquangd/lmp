% LMP Usage Guide

# LMP Usage Guide

This walk-through shows how to drive the Lung Modeling Platform (LMP) from Python or the CLI for common workflows:

1. create/load configs and run single simulations
2. generate parameter sweeps with manifests
3. run the stage pipeline in parallel via subject tasks
4. load observed data and perform parameter fitting
5. compute Sobol sensitivity indices
6. overlay observed vs predicted curves for plotting.

All examples assume `PYTHONPATH` points to `lmp/lmp_pkg/src`.

## Single Run

```python
from lmp_pkg import app_api

cfg = app_api.get_default_config()
cfg.product.ref = "reference_product"
cfg.run.workflow_name = "deposition_pbbm_pk"  # optional if you prefer configs to carry the workflow
result = app_api.run_single_simulation(cfg, run_id="demo_run", workflow_name="deposition_pbbm_pk")
print(result.runtime_seconds)
```

All bundled workflows now default to deterministic population variability. Unless you
explicitly opt in, every toggle is disabled so repeat runs match bit-for-bit. The `vbe`
workflow is the exception: it enables demographic, inhalation, and PK variability while
keeping lung generation and GI deterministic for QA. Override these defaults via the
`[population_variability]` block or the CLI flags shown below.

## Manifest / Parameter Sweeps

```python
from multiprocessing import Pool
from lmp_pkg import app_api

cfg = app_api.get_default_config()
axes = {
    "product.mass_median_diameter_um": [2.0, 2.5, 3.0],
    "pk.params.clearance": [20.0, 30.0],  
}
manifest = app_api.plan_simulation_manifest(cfg, axes)
# {"pk.params.clearance": [60.0, 90.0, 120.0]}
def run_row(row):
    overrides = {k: v for k, v in row.items() if k != "run_id"}
    return app_api.run_single_simulation(cfg, parameter_overrides=overrides, run_id=row["run_id"])

with Pool() as pool:
    results = pool.map(run_row, manifest.to_dict(orient="records"))
```

## Subject Tasks & Pipeline Runner

For large virtual BE workflows, pre-build subject tasks and use `run_tasks` or `run_pipeline_for_task`:

```python
from multiprocessing import Pool
from functools import partial
from lmp_pkg.simulation.subject_tasks import build_tasks
from lmp_pkg.simulation.pipeline_runner import run_pipeline_for_task
from lmp_pkg.engine.workflow import get_workflow, list_workflows

print(list_workflows())
tasks = build_tasks(["BD", "GP", "FF"], n_subjects=6, base_seed=1234)
workflow = get_workflow("deposition_pbbm_pk")

with Pool(processes=4) as pool:
    task_results = pool.map(partial(run_pipeline_for_task, workflow=workflow), tasks)

# selectively disable variability sources
from lmp_pkg.domain.entities import VariabilitySettings

no_pk_variability = VariabilitySettings(pk=False)
tasks = build_tasks(
    ["BD", "GP", "FF"],
    n_subjects=6,
    base_seed=1234,
    apply_variability=True,
    variability_settings=no_pk_variability,
)

Example TOML overrides:

```toml
[study]
study_type = "relative_ba"
design = "crossover"
n_subjects = 12
population = "copd_moderate"
charcoal_block = true

[pbbm]
model = "numba"
charcoal_block = false
suppress_et_absorption = true

[population_variability]
demographic = true
lung_regional = true
lung_generation = false
gi = false
pk = true
inhalation = true
```

## Observed Data & Fitting

```python
import numpy as np
from lmp_pkg import app_api
from lmp_pkg.solver.optimization import ParameterDefinition, ParameterFitter

cfg = app_api.get_default_config()
observed = app_api.load_observed_pk_csv("pk.csv", time_col="time_h", conc_col="ng_ml")
metric = app_api.build_pk_fitting_metric(observed)

params = [
    ParameterDefinition(name="CL", path="pk.params.clearance", bounds=(10.0, 150.0)),
    ParameterDefinition(name="MMAD", path="product.mass_median_diameter_um", bounds=(1.5, 4.0)),
]

fitter = ParameterFitter(
    base_config=cfg,
    parameters=params,
    observed_time_s=observed.time_s,
    observed_concentration=observed.concentration_ng_per_ml,
    metric=metric,
)

result = fitter.fit(initial_guess=np.array([50.0, 2.5]))
```

Overlay observed vs predicted:

```python
run = app_api.run_single_simulation(cfg)
df = app_api.pk_overlay_dataframe(run, observed)
# plot with pandas or seaborn
```

## Sobol Sensitivity

```python
import numpy as np
from lmp_pkg import app_api
from lmp_pkg.services.sensitivity_analysis import PipelineSobolAnalyzer, SensitivityParameter
from lmp_pkg.services.data_import import extract_predicted_pk

cfg = app_api.get_default_config()

def auc_metric(run):
    time_s, conc = extract_predicted_pk(run)
    return np.array([np.trapz(conc, time_s / 3600.0)])

params = [
    SensitivityParameter(name="CL", path="pk.params.clearance", bounds=(10, 150)),
    SensitivityParameter(name="MMAD", path="product.mass_median_diameter_um", bounds=(1.5, 4.0)),
]

analyzer = PipelineSobolAnalyzer(cfg, params, metric=auc_metric)
indices = analyzer.compute_indices(n_samples=200)
```

## CLI Usage

Basic examples:

- `lmp run -c configs/gp.toml --run-id run1`
- `lmp run -c configs/gp.toml --workflow deposition_pbbm_pk`
- `lmp plan configs/gp.toml --axes '{"product.mass_median_diameter_um": [2.0, 2.5]}' -o manifest.csv`
- `lmp list-models`
- `lmp list-workflows`
- `lmp run -c examples/basic.toml --pv-pk --pv-inhalation --no-pv-gi`

You can iterate `manifest.csv` and call the CLI per row or run them programmatically as shown above.
