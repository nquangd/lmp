"""Main API facade for LMP package.

This module provides the primary interface used by both Dash applications
and SLURM scripts. All high-level operations flow through these functions.
"""

from __future__ import annotations
import uuid
from pathlib import Path
from typing import Dict, List, Mapping, Iterable, Optional, Any, Union, Sequence, Tuple
import pandas as pd
import numpy as np
import structlog

from .config import (
    AppConfig, default_config, load_config, validate_config,
    hydrate_config, validate_hydrated_entities, get_entity_summary,
    DepositionConfig, PBBMConfig, PKConfig,
)
from .catalog import get_default_catalog
from .contracts import RunResult
from .contracts.errors import ConfigError, ValidationError
from .variability import (
    build_inter_subject,
    build_intra_subject,
    create_deterministic_rng,
    apply_population_variability_settings,
)
from .engine import get_registry
from .engine.workflow import Workflow, get_workflow
from .simulation.pipeline_runner import run_pipeline_for_task, run_tasks
from .simulation.subject_tasks import SubjectTask, build_tasks
from .services.data_import import ObservedPK, make_pk_fitting_metric, make_pk_overlay_data
from .data_structures.comprehensive_results import ComprehensivePBBMResults

logger = structlog.get_logger()


def get_default_config() -> AppConfig:
    """Get default configuration.
    
    Returns:
        Default configuration with sensible defaults
    """
    return default_config()


def load_config_from_file(path: Union[str, Path]) -> AppConfig:
    """Load and validate configuration from file.
    
    Args:
        path: Path to configuration file
        
    Returns:
        Loaded and validated configuration
        
    Raises:
        ConfigError: If configuration cannot be loaded or is invalid
    """
    config = load_config(path)
    validate_config(config)
    return config


def validate_configuration(config: AppConfig) -> None:
    """Validate configuration for common issues.
    
    Args:
        config: Configuration to validate
        
    Raises:
        ValidationError: If configuration has errors
    """
    validate_config(config)


def list_available_models() -> Dict[str, List[str]]:
    """List all available models by category.
    
    Returns:
        Dictionary mapping model categories to lists of available models
        
    Example:
        {
            "deposition": ["null", "clean_lung"],
            "lung_pbbm": ["numba"],
            "systemic_pk": ["null", "pk_1c", "pk_2c", "pk_3c"]
        }
    """
    registry = get_registry()
    return registry.list_models()


def _extract_population_variability_settings(
    config: AppConfig,
    workflow: Optional[Workflow] = None,
) -> Optional[Mapping[str, bool]]:
    """Get per-domain variability overrides from configuration and workflow defaults."""

    base_overrides: Dict[str, bool] = {}

    # Prefer explicit workflow if supplied, otherwise look up by name.
    selected_workflow = workflow
    if selected_workflow is None:
        workflow_name = getattr(getattr(config, "run", None), "workflow_name", None)
        if workflow_name:
            try:
                selected_workflow = get_workflow(workflow_name)
            except KeyError:
                selected_workflow = None
        else:
            selected_workflow = get_workflow("deposition_pbbm_pk")

    if selected_workflow and selected_workflow.population_variability_defaults:
        base_overrides.update(selected_workflow.population_variability_defaults.as_overrides())

    pop_var = getattr(config, "population_variability", None)
    if pop_var is not None:
        base_overrides.update(pop_var.as_overrides())

    return base_overrides or None


def _apply_pk_parameter_overrides(subject, api, pk_config: Optional["PKConfig"]) -> None:
    """Apply systemic PK parameter overrides onto hydrated entities."""

    if pk_config is None or not pk_config.params:
        return

    overrides = pk_config.params
    subject_pk = getattr(subject, 'pk', None)

    def _apply(target: Any, attr_candidates: Sequence[str], value: Any) -> None:
        for attr in attr_candidates:
            if hasattr(target, attr):
                try:
                    setattr(target, attr, float(value))
                except (TypeError, ValueError):
                    setattr(target, attr, value)
                break

    for key, value in overrides.items():
        if value is None:
            continue
        key_lower = str(key).lower()
        if key_lower in {"clearance", "clearance_l_h", "cl", "cl_systemic", "cl_systemic_l_h"}:
            attr_candidates = ["clearance_L_h", "cl_systemic_L_h"]
        elif key_lower in {"vd_central", "vd_central_l", "volume_central", "volume_central_l"}:
            attr_candidates = ["volume_central_L"]
        elif key_lower in {"vd_peripheral", "vd_peripheral_l", "volume_peripheral", "volume_peripheral_l"}:
            attr_candidates = ["volume_peripheral_L", "vd_peripheral_L"]
        elif key_lower in {"vd_peripheral1", "vd_peripheral1_l", "volume_peripheral1", "volume_peripheral1_l"}:
            attr_candidates = ["volume_peripheral1_L", "vd_peripheral1_L"]
        elif key_lower in {"vd_peripheral2", "vd_peripheral2_l", "volume_peripheral2", "volume_peripheral2_l"}:
            attr_candidates = ["volume_peripheral2_L", "vd_peripheral2_L"]
        elif key_lower in {"cl_distribution", "cl_distribution_l_h", "cl_distr", "cl_distribution1", "cl_distribution1_l_h"}:
            attr_candidates = ["cl_distribution_L_h", "cl_distribution1_L_h"]
        elif key_lower in {"cl_distribution2", "cl_distribution2_l_h"}:
            attr_candidates = ["cl_distribution2_L_h"]
        else:
            attr_candidates = [str(key)]

        if subject_pk is not None:
            _apply(subject_pk, attr_candidates, value)
        _apply(api, attr_candidates, value)


def _apply_config_overrides(config: AppConfig, overrides: Mapping[str, Any]) -> AppConfig:
    if not overrides:
        return config

    data = config.model_dump()

    def _set_nested(target: Dict[str, Any], parts: Sequence[str], value: Any) -> None:
        current = target
        for key in parts[:-1]:
            if isinstance(current, dict):
                current = current.setdefault(key, {})
            else:
                raise KeyError(f"Cannot set override at {'.'.join(parts)}")
        if isinstance(current, dict):
            current[parts[-1]] = value
        else:
            raise KeyError(f"Cannot set override at {'.'.join(parts)}")

    for dotted_key, value in overrides.items():
        if not dotted_key:
            continue
        parts = dotted_key.split('.')
        _set_nested(data, parts, value)

    return AppConfig.model_validate(data)


def _build_stage_overrides_from_config(config: AppConfig) -> Dict[str, Dict[str, Any]]:
    overrides: Dict[str, Dict[str, Any]] = {}

    registry = get_registry()
    available = registry.list_models()

    deposition_models = set(available.get("deposition", []))
    pbpk_models = set(available.get("lung_pbbm", []))
    pk_models = set(available.get("systemic_pk", []))

    # Deposition
    dep_cfg: Optional[DepositionConfig] = getattr(config, "deposition", None)
    if dep_cfg is not None:
        if dep_cfg.model and dep_cfg.model not in {"", "null"} and dep_cfg.model in deposition_models:
            overrides["deposition"] = {"model": dep_cfg.model}
        elif dep_cfg.model and dep_cfg.model not in deposition_models:
            logger.warning("Skipping deposition model override; model not registered", model=dep_cfg.model)

    # PBPK
    pbbm_cfg: Optional[PBBMConfig] = getattr(config, "pbbm", None)
    if pbbm_cfg is not None:
        pbbm_override: Dict[str, Any] = {}
        if pbbm_cfg.model and pbbm_cfg.model not in {"", "null"} and pbbm_cfg.model in pbpk_models:
            pbbm_override["model"] = pbbm_cfg.model
        elif pbbm_cfg.model and pbbm_cfg.model not in pbpk_models:
            logger.warning("Skipping PBPK model override; model not registered", model=pbbm_cfg.model)
        pbbm_params: Dict[str, Any] = {}
        if pbbm_cfg.params:
            pbbm_params.update(pbbm_cfg.params)
        if pbbm_cfg.solver:
            solver_options = pbbm_cfg.solver.model_dump()
            if solver_options:
                pbbm_params.setdefault("solver_options", solver_options)
        if pbbm_cfg.epi_layers:
            pbbm_params.setdefault("epi_layers", list(pbbm_cfg.epi_layers))
        if pbbm_cfg.charcoal_block:
            pbbm_params["charcoal_block"] = True
        if pbbm_cfg.suppress_et_absorption:
            pbbm_params["suppress_et_absorption"] = True
        if pbbm_params:
            pbbm_override.setdefault("params", {}).update(pbbm_params)
        if pbbm_override:
            overrides["pbbm"] = pbbm_override

    # PK
    pk_cfg: Optional[PKConfig] = getattr(config, "pk", None)
    if pk_cfg is not None:
        pk_override: Dict[str, Any] = {}
        if pk_cfg.model and pk_cfg.model not in {"", "null"} and pk_cfg.model in pk_models:
            pk_override["model"] = pk_cfg.model
        elif pk_cfg.model and pk_cfg.model not in pk_models:
            logger.warning("Skipping PK model override; model not registered", model=pk_cfg.model)
        if pk_cfg.params:
            pk_override.setdefault("params", {}).update(pk_cfg.params)
        if pk_override:
            overrides["pk"] = pk_override

    return overrides


def _resolve_config_value(config: AppConfig, dotted_path: str) -> Any:
    data: Any = config.model_dump()
    for part in dotted_path.split('.'):
        if isinstance(data, Mapping):
            data = data.get(part)
        else:
            data = getattr(data, part, None)
        if data is None:
            break
    return data


def plan_local_sensitivity_manifest(
    config: AppConfig,
    parameters: Mapping[str, Any],
    *,
    default_relative_delta: float = 0.1,
    zero_reference: float = 1.0,
) -> pd.DataFrame:
    """Create a manifest for finite-difference local sensitivity runs."""

    if not parameters:
        raise ValueError("parameters must contain at least one entry")

    parameter_columns = list(parameters.keys())
    base_values = {path: _resolve_config_value(config, path) for path in parameter_columns}

    run_prefix = f"sensitivity_{uuid.uuid4().hex[:8]}"
    rows: List[Dict[str, Any]] = []

    baseline_row: Dict[str, Any] = {
        "run_id": f"{run_prefix}_baseline",
        "variant": "baseline",
        "sensitivity_parameter": None,
        "direction": "baseline",
    }
    baseline_row.update(base_values)
    rows.append(baseline_row)

    for index, path in enumerate(parameter_columns, start=1):
        spec = parameters[path]
        if isinstance(spec, Mapping):
            delta_value = float(spec.get("delta", spec.get("value", default_relative_delta)))
            mode = str(spec.get("mode", spec.get("type", "relative"))).lower()
        else:
            delta_value = float(spec)
            mode = "relative"

        base_value = base_values.get(path)
        if base_value is None:
            continue
        if not isinstance(base_value, (int, float, np.integer, np.floating)):
            continue

        if mode == "absolute":
            change = delta_value
        else:
            reference = base_value if base_value != 0 else zero_reference
            change = reference * delta_value

        positive_value = base_value + change
        negative_value = base_value - change

        for direction, value in (("positive", positive_value), ("negative", negative_value)):
            row: Dict[str, Any] = {
                "run_id": f"{run_prefix}_{index}_{direction}",
                "variant": f"{path}_{direction}",
                "sensitivity_parameter": path,
                "direction": direction,
            }
            row.update(base_values)
            row[path] = value
            rows.append(row)

    return pd.DataFrame(rows)


def aggregate_local_sensitivity_results(
    manifest: pd.DataFrame,
    run_results: Mapping[str, RunResult],
) -> Dict[str, Any]:
    """Aggregate local sensitivity runs into derivative-style metrics."""

    required_cols = {"run_id", "sensitivity_parameter", "direction"}
    if not required_cols.issubset(manifest.columns):
        raise ValueError("manifest must include run_id, sensitivity_parameter, and direction columns")

    baseline_rows = manifest[manifest["direction"] == "baseline"]
    if baseline_rows.empty:
        raise ValueError("manifest does not contain a baseline row")

    baseline_row = baseline_rows.iloc[0]
    baseline_run_id = baseline_row["run_id"]
    if baseline_run_id not in run_results:
        raise KeyError(f"Baseline run '{baseline_run_id}' not found in run_results")

    parameter_columns = [col for col in manifest.columns if col not in {"run_id", "variant", "sensitivity_parameter", "direction"}]

    metrics_by_run: Dict[str, Dict[str, float]] = {}
    for run_id, result in run_results.items():
        metrics_by_run[run_id] = calculate_summary_metrics(result)

    baseline_metrics = metrics_by_run[baseline_run_id]

    parameter_summary: Dict[str, Dict[str, Any]] = {}
    metric_summary: Dict[str, Dict[str, Dict[str, float]]] = {}

    for path in parameter_columns:
        plus_row = manifest[(manifest["sensitivity_parameter"] == path) & (manifest["direction"] == "positive")]
        minus_row = manifest[(manifest["sensitivity_parameter"] == path) & (manifest["direction"] == "negative")]
        if plus_row.empty or minus_row.empty:
            continue

        plus_row = plus_row.iloc[0]
        minus_row = minus_row.iloc[0]
        plus_id = plus_row["run_id"]
        minus_id = minus_row["run_id"]

        if plus_id not in run_results or minus_id not in run_results:
            continue

        baseline_value = baseline_row.get(path)
        plus_value = plus_row.get(path)
        minus_value = minus_row.get(path)
        delta_value = None
        if plus_value is not None and minus_value is not None:
            delta_value = plus_value - minus_value

        parameter_summary[path] = {
            "baseline": baseline_value,
            "positive": plus_value,
            "negative": minus_value,
            "delta": delta_value,
        }

        plus_metrics = metrics_by_run[plus_id]
        minus_metrics = metrics_by_run[minus_id]

        for metric_name, baseline_metric in baseline_metrics.items():
            plus_metric = plus_metrics.get(metric_name)
            minus_metric = minus_metrics.get(metric_name)
            if plus_metric is None or minus_metric is None:
                continue
            sensitivity = None
            if delta_value and delta_value != 0:
                sensitivity = (plus_metric - minus_metric) / delta_value

            metric_summary.setdefault(metric_name, {})[path] = {
                "baseline": baseline_metric,
                "positive": plus_metric,
                "negative": minus_metric,
                "sensitivity": sensitivity,
            }

    return {
        "parameters": parameter_summary,
        "metrics": metric_summary,
        "baseline_run_id": baseline_run_id,
    }


def plan_parameter_estimation_runs(
    config: AppConfig,
    parameters: Mapping[str, Mapping[str, Any]],
    *,
    include_baseline: bool = True,
    default_relative_step: float = 0.1,
    zero_reference: float = 1.0,
) -> pd.DataFrame:
    """Create manifest rows for parameter estimation sweeps."""

    if not parameters:
        raise ValueError("parameters must contain at least one entry")

    parameter_paths = list(parameters.keys())
    baseline_values = {path: _resolve_config_value(config, path) for path in parameter_paths}

    run_prefix = f"estimate_{uuid.uuid4().hex[:8]}"
    rows: List[Dict[str, Any]] = []

    def _add_row(run_id: str, parameter: Optional[str], direction: str, value_map: Dict[str, Any]) -> None:
        row = {
            "run_id": run_id,
            "parameter": parameter,
            "direction": direction,
            **value_map,
        }
        if parameter:
            row["value"] = value_map.get(parameter)
        rows.append(row)

    if include_baseline:
        _add_row(f"{run_prefix}_baseline", None, "baseline", baseline_values.copy())

    for index, (path, spec) in enumerate(parameters.items(), start=1):
        settings = dict(spec or {})
        explicit_values = settings.get("values")

        if explicit_values is not None:
            for idx, value in enumerate(explicit_values, start=1):
                values = baseline_values.copy()
                values[path] = value
                _add_row(
                    f"{run_prefix}_{index}_{idx}",
                    path,
                    f"value_{idx}",
                    values,
                )
            continue

        mode = str(settings.get("mode", settings.get("type", "relative"))).lower()
        delta_setting = settings.get("delta", default_relative_step)
        if isinstance(delta_setting, (list, tuple)) and len(delta_setting) == 2:
            positive_delta, negative_delta = float(delta_setting[0]), float(delta_setting[1])
        else:
            positive_delta = float(delta_setting)
            negative_delta = float(delta_setting)

        baseline_value = baseline_values.get(path)
        if baseline_value is None or not isinstance(baseline_value, (int, float, np.integer, np.floating)):
            continue

        if mode == "absolute":
            plus_change = positive_delta
            minus_change = -negative_delta
        else:
            reference = baseline_value if baseline_value != 0 else zero_reference
            plus_change = reference * positive_delta
            minus_change = -reference * negative_delta

        value_plus = baseline_value + plus_change
        value_minus = baseline_value + minus_change

        plus_map = baseline_values.copy()
        plus_map[path] = value_plus
        _add_row(f"{run_prefix}_{index}_positive", path, "positive", plus_map)

        minus_map = baseline_values.copy()
        minus_map[path] = value_minus
        _add_row(f"{run_prefix}_{index}_negative", path, "negative", minus_map)

    return pd.DataFrame(rows)


def aggregate_parameter_estimation_results(
    manifest: pd.DataFrame,
    run_results: Mapping[str, RunResult],
    observed_time_s: Sequence[float],
    observed_values: Sequence[float],
) -> Dict[str, Any]:
    """Aggregate parameter estimation runs into error metrics and residuals."""

    if manifest.empty:
        raise ValueError("manifest must contain at least one row")

    observed_time = np.asarray(observed_time_s, dtype=float)
    observed = np.asarray(observed_values, dtype=float)
    if observed_time.shape != observed.shape:
        raise ValueError("observed_time_s and observed_values must have the same shape")

    baseline_rows = manifest[manifest["direction"] == "baseline"]
    baseline_run_id = None
    if not baseline_rows.empty:
        baseline_run_id = baseline_rows.iloc[0]["run_id"]

    summary: Dict[str, Any] = {
        "observed": {
            "time_s": observed_time.tolist(),
            "values": observed.tolist(),
        },
        "runs": {},
    }

    best_run_id = None
    best_error = None

    def _extract_pk_series(result: RunResult) -> Tuple[np.ndarray, np.ndarray]:
        if result.pbbk and result.pbbk.comprehensive is not None:
            comp = result.pbbk.comprehensive
            return np.asarray(comp.time_s, dtype=float), np.asarray(comp.pk_data.plasma_concentration_ng_per_ml, dtype=float)
        if result.pk is not None:
            return np.asarray(result.pk.t, dtype=float), np.asarray(result.pk.conc_plasma, dtype=float)
        raise ValueError("Simulation result does not contain PK output")

    def _evaluate(run_id: str) -> Tuple[float, np.ndarray]:
        result = run_results.get(run_id)
        if result is None:
            raise KeyError(f"Run result '{run_id}' not available")
        time, conc = _extract_pk_series(result)
        predicted = np.interp(observed_time, time, conc)
        residuals = observed - predicted
        error = float(np.sum(residuals ** 2))
        return error, residuals

    if baseline_run_id:
        baseline_error, baseline_residuals = _evaluate(baseline_run_id)
        summary["baseline"] = {
            "run_id": baseline_run_id,
            "sse": baseline_error,
            "residuals": baseline_residuals.tolist(),
        }
        best_run_id = baseline_run_id
        best_error = baseline_error

    for row in manifest.itertuples(index=False):
        run_id = getattr(row, "run_id")
        direction = getattr(row, "direction", None)
        parameter = getattr(row, "parameter", None)
        if direction == "baseline":
            continue

        error, residuals = _evaluate(run_id)
        value = getattr(row, "value", None)

        summary["runs"][run_id] = {
            "parameter": parameter,
            "direction": direction,
            "value": value,
            "sse": error,
            "residuals": residuals.tolist(),
        }

        if best_error is None or error < best_error:
            best_error = error
            best_run_id = run_id

    summary["best_run_id"] = best_run_id
    summary["best_error"] = best_error

    return summary


def plan_virtual_trial_tasks(
    config: AppConfig,
    *,
    n_subjects: int,
    base_seed: int = 1234,
    subject_name: Optional[str] = None,
    products: Optional[Sequence[str]] = None,
    apply_variability: bool = True,
) -> List[Dict[str, Any]]:
    """Build per-subject task specifications for virtual trials."""

    if n_subjects <= 0:
        raise ValueError("n_subjects must be at least 1")

    subject_ref = subject_name or config.subject.ref
    api_name = config.api.ref

    if products:
        product_list: Sequence[str] = tuple(products)
    else:
        product_list = (config.product.ref,)

    variability_settings = None
    if config.population_variability is not None:
        variability_settings = config.population_variability.model_dump()

    tasks = build_tasks(
        apis=[api_name],
        n_subjects=n_subjects,
        base_seed=base_seed,
        subject_name=subject_ref,
        products=product_list,
        apply_variability=apply_variability,
        variability_settings=variability_settings,
        study_type=getattr(config.study, "study_type", None),
        study_design=getattr(config.study, "design", None),
        charcoal_block=getattr(config.study, "charcoal_block", False),
        suppress_et_absorption=getattr(config.pbbm, "suppress_et_absorption", False),
    )

    task_specs: List[Dict[str, Any]] = []
    for task in tasks:
        task_specs.append(
            {
                "subject_index": task.subject_index,
                "subject_name": task.subject_name,
                "api": task.api_name,
                "products": list(task.products),
                "seed": task.seed,
                "apply_variability": task.apply_variability,
                "variability_settings": variability_settings,
                "study_type": task.study_type,
                "study_design": task.study_design,
                "charcoal_block": task.charcoal_block,
                "suppress_et_absorption": task.suppress_et_absorption,
            }
        )

    return task_specs


def plan_virtual_bioequivalence_tasks(
    config: AppConfig,
    *,
    n_subjects: int,
    test_products: Sequence[str],
    reference_product: Optional[str] = None,
    base_seed: int = 1234,
    subject_name: Optional[str] = None,
    apply_variability: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build task specifications for virtual bioequivalence studies."""

    if n_subjects <= 0:
        raise ValueError("n_subjects must be at least 1")
    if not test_products:
        raise ValueError("Provide at least one test product for VBE")

    subject_ref = subject_name or config.subject.ref
    api_name = config.api.ref
    reference = reference_product or config.product.ref
    product_entries = [
        {"name": reference, "role": "reference"},
        *[{"name": name, "role": "test"} for name in test_products],
    ]

    variability_settings = None
    if config.population_variability is not None:
        variability_settings = config.population_variability.model_dump()

    products_sequence = tuple(entry["name"] for entry in product_entries)

    tasks = build_tasks(
        apis=[api_name],
        n_subjects=n_subjects,
        base_seed=base_seed,
        subject_name=subject_ref,
        products=products_sequence,
        apply_variability=apply_variability,
        variability_settings=variability_settings,
        study_type=getattr(config.study, "study_type", None),
        study_design=getattr(config.study, "design", None),
        charcoal_block=getattr(config.study, "charcoal_block", False),
        suppress_et_absorption=getattr(config.pbbm, "suppress_et_absorption", False),
    )

    task_specs: List[Dict[str, Any]] = []
    for task in tasks:
        task_specs.append(
            {
                "subject_index": task.subject_index,
                "subject_name": task.subject_name,
                "api": task.api_name,
                "products": product_entries,
                "seed": task.seed,
                "apply_variability": task.apply_variability,
                "variability_settings": variability_settings,
                "study_type": task.study_type,
                "study_design": task.study_design,
                "charcoal_block": task.charcoal_block,
                "suppress_et_absorption": task.suppress_et_absorption,
            }
        )

    study_metadata = {
        "reference_product": reference,
        "test_products": list(test_products),
        "products": product_entries,
        "n_subjects": n_subjects,
        "base_seed": base_seed,
        "apply_variability": apply_variability,
        "subject_name": subject_ref,
    }

    return task_specs, study_metadata


def list_catalog_entries(category: str) -> List[str]:
    """List available catalog entries for a category.
    
    Args:
        category: Catalog category ("subject", "api", "product", "maneuver")
        
    Returns:
        List of available entry names
        
    Raises:
        ValueError: If category is not recognized
    """
    catalog = get_default_catalog()
    return catalog.list_entries(category)


def get_catalog_entry(category: str, name: str) -> Dict[str, Any]:
    """Get a specific catalog entry.
    
    Args:
        category: Catalog category
        name: Entry name
        
    Returns:
        Dictionary containing entry data
        
    Raises:
        ValueError: If category or entry name is not found
    """
    catalog = get_default_catalog()
    entity = catalog.get_entry(category, name)
    return entity.model_dump()


def plan_simulation_manifest(
    config: AppConfig, 
    parameter_axes: Mapping[str, Iterable[Any]]
) -> pd.DataFrame:
    """Generate simulation manifest for parameter sweeps.
    
    Args:
        config: Base configuration
        parameter_axes: Dictionary mapping parameter paths to value lists
                       e.g., {"pbbm.params.CL": [10, 20, 30], "subject.ref": ["adult", "child"]}
        
    Returns:
        DataFrame with columns: run_id, and one column per parameter
        
    Example:
        >>> axes = {"pbbm.params.duration_h": [12, 24], "pk.model": ["pk_1c", "pk_2c"]}
        >>> manifest = plan_simulation_manifest(config, axes)
        >>> print(manifest)
        run_id    pbbm.params.CL    pk.model
        run_001   10                pk_1c
        run_002   10                pk_2c  
        run_003   20                pk_1c
        run_004   20                pk_2c
    """
    import itertools
    
    if not parameter_axes:
        # Single run
        return pd.DataFrame({
            "run_id": [f"run_{uuid.uuid4().hex[:8]}"]
        })
    
    # Generate Cartesian product of all parameter combinations
    param_names = list(parameter_axes.keys())
    param_values = list(parameter_axes.values())
    
    combinations = list(itertools.product(*param_values))
    
    # Create DataFrame
    data = {
        "run_id": [f"run_{i+1:03d}" for i in range(len(combinations))]
    }
    
    for param_name, combo_values in zip(param_names, zip(*combinations)):
        data[param_name] = combo_values
    
    return pd.DataFrame(data)


def run_single_simulation(
    config: AppConfig,
    parameter_overrides: Optional[Mapping[str, Any]] = None,
    run_id: Optional[str] = None,
    artifact_directory: Optional[Union[str, Path]] = None,
    workflow: Optional[Workflow] = None,
    workflow_name: Optional[str] = None,
) -> RunResult:
    """Run a single simulation.
    
    Args:
        config: Configuration to use
        parameter_overrides: Optional parameter overrides to apply
        run_id: Optional run identifier (generated if not provided)  
        artifact_directory: Directory to save artifacts (uses config default if not provided)
        
    Returns:
        Complete simulation results
        
    Raises:
        ValidationError: If configuration is invalid
        ModelError: If simulation fails
    """
    if run_id is None:
        run_id = f"run_{uuid.uuid4().hex[:8]}"
    
    logger.info("Starting simulation", run_id=run_id)

    if parameter_overrides:
        config = _apply_config_overrides(config, parameter_overrides)

    # Hydrate configuration with real entities
    try:
        hydrated = hydrate_config(config)
        validate_hydrated_entities(hydrated)
        
        # Log entity summary
        summaries = get_entity_summary(hydrated)
        logger.info("Entities resolved", 
                   subject=summaries["subject"],
                   api=summaries["api"],
                   product=summaries["product"], 
                   maneuver=summaries["maneuver"])
        
    except Exception as e:
        logger.error("Entity resolution failed", error=str(e))
        raise ValidationError(f"Entity resolution failed: {e}")
    
    study_cfg = getattr(config, 'study', None)
    study_type = getattr(study_cfg, 'study_type', None) if study_cfg else None
    study_design = getattr(study_cfg, 'design', None) if study_cfg else None
    charcoal_block = getattr(study_cfg, 'charcoal_block', False) if study_cfg else False
    pbbm_cfg = getattr(config, 'pbbm', None)
    suppress_et_absorption = getattr(pbbm_cfg, 'suppress_et_absorption', False) if pbbm_cfg else False
    if pbbm_cfg:
        charcoal_block = charcoal_block or getattr(pbbm_cfg, 'charcoal_block', False)

    resolved_workflow_name = workflow_name or config.run.workflow_name
    selected_workflow = workflow or (
        get_workflow(resolved_workflow_name) if resolved_workflow_name else get_workflow("deposition_pbbm_pk")
    )

    variability_overrides = _extract_population_variability_settings(config, selected_workflow)

    _apply_pk_parameter_overrides(hydrated['subject'], hydrated['api'], config.pk)

    stage_overrides = _build_stage_overrides_from_config(config)

    product_name = getattr(hydrated['product'], 'name', config.product.ref or 'product')

    task = SubjectTask(
        subject_index=0,
        api_name=getattr(hydrated['api'], 'name', config.api.ref or 'api'),
        subject_name=getattr(hydrated['subject'], 'name', config.subject.ref or 'subject'),
        products=(product_name,),
        seed=config.run.seed,
        subject=hydrated['subject'],
        api=hydrated['api'],
        product_entities={product_name: hydrated['product']},
        apply_variability=False,
        variability_settings=variability_overrides,
        study_type=study_type,
        study_design=study_design,
        charcoal_block=charcoal_block,
        suppress_et_absorption=suppress_et_absorption,
    )

    task_result = run_pipeline_for_task(
        task,
        workflow=selected_workflow,
        stage_overrides=stage_overrides,
    )
    product_result = task_result.products[product_name]

    runtime = task_result.runtime_metadata.get('total_runtime_s', 0.0)
    product_stage_results = task_result.stage_results.get(product_name, {})
    stage_metadata = {
        stage: {
            "model": stage_result.model_name,
            "metadata": stage_result.metadata,
        }
        for stage, stage_result in product_stage_results.items()
    }

    result = RunResult(
        run_id=run_id,
        config=config.model_dump(),
        cfd=product_result.cfd,
        deposition=product_result.deposition,
        pbbk=product_result.pbpk,
        pk=product_result.pk,
        runtime_seconds=runtime,
        metadata={
            "status": "completed",
            "entities": summaries,
            "stage_times": task_result.runtime_metadata.get('stage_times', {}),
            "stage_results": stage_metadata,
        },
    )

    logger.info("Simulation completed", run_id=run_id, runtime=result.runtime_seconds)
    return result


def convert_results_to_dataframes(result: RunResult) -> Dict[str, pd.DataFrame]:
    """Convert simulation results to UI-friendly DataFrames.
    
    Args:
        result: Simulation results
        
    Returns:
        Dictionary mapping frame names to DataFrames with standardized columns:
        - "pk_curve": ["run_id", "t", "plasma_conc", "compartment"]  
        - "regional_auc": ["run_id", "region", "auc_elf", "auc_epi", "auc_tissue"]
        - "deposition_bins": ["run_id", "region", "particle_um", "amount_pmol", "fraction_of_dose"]
        - "solver_stats": ["run_id", "stage", "method", "rtol", "atol", "nfev", "njev", "status", "runtime_s"]
        - "subject_params": Wide one-row DataFrame for display/export
    """
    frames = {}
    stage_results = result.metadata.get("stage_results", {})

    # CFD summary data
    if result.cfd is not None:
        frames["cfd_summary"] = pd.DataFrame({
            "run_id": [result.run_id],
            "mmad_um": [result.cfd.mmad],
            "gsd": [result.cfd.gsd],
            "mt_deposition_fraction": [result.cfd.mt_deposition_fraction],
        })

    # PK curve data
    pk_data = result.pk
    if pk_data and hasattr(pk_data, 't') and hasattr(pk_data, 'conc_plasma'):
        # Real PK data
        n_points = len(pk_data.t)
        frames["pk_curve"] = pd.DataFrame({
            "run_id": [result.run_id] * n_points,
            "t": pk_data.t,
            "plasma_conc": pk_data.conc_plasma,
            "compartment": ["central"] * n_points
        })
    else:
        # Placeholder PK data
        frames["pk_curve"] = pd.DataFrame({
            "run_id": [result.run_id],
            "t": [0.0],
            "plasma_conc": [0.0],
            "compartment": ["central"]
        })

    # Regional AUC data from PBPK results and detailed PBPK time series
    pbbk_data = result.pbbk
    regional_rows: List[Dict[str, Any]] = []
    pbpk_timeseries_frames: List[pd.DataFrame] = []

    def _maybe_float(value: Any) -> float:
        if value is None:
            return float("nan")
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    if pbbk_data is not None:
        comprehensive = getattr(pbbk_data, "comprehensive", None)
        regional_data_map = getattr(comprehensive, "regional_data", {}) if comprehensive else {}

        if regional_data_map:
            time_reference_s = np.asarray(getattr(comprehensive, "time_s", []), dtype=float)

            for region_name, region_data in regional_data_map.items():
                region_time_s = np.asarray(getattr(region_data, "time_s", time_reference_s), dtype=float)
                region_time_h = np.asarray(getattr(region_data, "time_h", region_time_s / 3600.0), dtype=float)

                if region_time_h.size == 0 and time_reference_s.size:
                    region_time_s = time_reference_s
                    region_time_h = time_reference_s / 3600.0

                def _append_series(values: Any,
                                   compartment: str,
                                   quantity: str,
                                   binding: str,
                                   units: str) -> None:
                    if values is None:
                        return
                    arr = np.asarray(values, dtype=float).reshape(-1)
                    if arr.size == 0:
                        return
                    if region_time_h.size != arr.size:
                        return
                    frame = pd.DataFrame({
                        "run_id": result.run_id,
                        "region": region_name,
                        "compartment": compartment,
                        "quantity": quantity,
                        "binding": binding,
                        "units": units,
                        "time_s": region_time_s,
                        "time_h": region_time_h,
                        "value": arr
                    })
                    pbpk_timeseries_frames.append(frame)

                # Amount series (pmol)
                _append_series(getattr(region_data, "epithelium_amounts", None), "epithelium", "amount", "total", "pmol")
                _append_series(getattr(region_data, "tissue_amounts", None), "tissue", "amount", "total", "pmol")
                _append_series(getattr(region_data, "elf_amounts", None), "elf", "amount", "total", "pmol")
                _append_series(getattr(region_data, "solid_drug_amounts", None), "solid", "amount", "total", "pmol")
                _append_series(getattr(region_data, "total_amounts", None), "total", "amount", "total", "pmol")
                _append_series(getattr(region_data, "epithelium_unbound_amounts_pmol", None), "epithelium", "amount", "unbound", "pmol")
                _append_series(getattr(region_data, "tissue_unbound_amounts_pmol", None), "tissue", "amount", "unbound", "pmol")

                epi_binding = getattr(region_data, "epithelium_binding", None)
                if epi_binding is not None:
                    _append_series(getattr(epi_binding, "unbound_pmol", None), "epithelium", "amount", "unbound", "pmol")
                    _append_series(getattr(epi_binding, "bound_pmol", None), "epithelium", "amount", "bound", "pmol")
                    _append_series(getattr(epi_binding, "unbound_conc_pmol_per_ml", None), "epithelium", "concentration", "unbound", "pmol/mL")
                    _append_series(getattr(epi_binding, "bound_conc_pmol_per_ml", None), "epithelium", "concentration", "bound", "pmol/mL")
                    _append_series(getattr(epi_binding, "total_mass_pg", None), "epithelium", "mass", "total", "pg")
                    _append_series(getattr(epi_binding, "unbound_mass_pg", None), "epithelium", "mass", "unbound", "pg")
                    _append_series(getattr(epi_binding, "bound_mass_pg", None), "epithelium", "mass", "bound", "pg")

                tissue_binding = getattr(region_data, "tissue_binding", None)
                if tissue_binding is not None:
                    _append_series(getattr(tissue_binding, "unbound_pmol", None), "tissue", "amount", "unbound", "pmol")
                    _append_series(getattr(tissue_binding, "bound_pmol", None), "tissue", "amount", "bound", "pmol")
                    _append_series(getattr(tissue_binding, "unbound_conc_pmol_per_ml", None), "tissue", "concentration", "unbound", "pmol/mL")
                    _append_series(getattr(tissue_binding, "bound_conc_pmol_per_ml", None), "tissue", "concentration", "bound", "pmol/mL")
                    _append_series(getattr(tissue_binding, "total_mass_pg", None), "tissue", "mass", "total", "pg")
                    _append_series(getattr(tissue_binding, "unbound_mass_pg", None), "tissue", "mass", "unbound", "pg")
                    _append_series(getattr(tissue_binding, "bound_mass_pg", None), "tissue", "mass", "bound", "pg")

                _append_series(getattr(region_data, "epithelium_concentration_pmol_per_ml", None), "epithelium", "concentration", "total", "pmol/mL")
                _append_series(getattr(region_data, "tissue_concentration_pmol_per_ml", None), "tissue", "concentration", "total", "pmol/mL")
                _append_series(getattr(region_data, "epithelium_tissue_concentration_pmol_per_ml", None), "combined", "concentration", "total", "pmol/mL")
                _append_series(getattr(region_data, "epithelium_unbound_concentration_pmol_per_ml", None), "epithelium", "concentration", "unbound", "pmol/mL")
                _append_series(getattr(region_data, "tissue_unbound_concentration_pmol_per_ml", None), "tissue", "concentration", "unbound", "pmol/mL")
                _append_series(getattr(region_data, "epithelium_tissue_unbound_concentration_pmol_per_ml", None), "combined", "concentration", "unbound", "pmol/mL")

                auc_elf = float("nan")
                try:
                    elf_amounts = np.asarray(getattr(region_data, "elf_amounts", []), dtype=float)
                    if elf_amounts.size and region_time_s.size == elf_amounts.size:
                        auc_elf = float(np.trapz(elf_amounts, region_time_s) / 3600.0)
                except Exception:
                    auc_elf = float("nan")

                regional_rows.append({
                    "run_id": result.run_id,
                    "region": region_name,
                    "auc_elf": auc_elf,
                    "auc_epithelium_pmol_h_per_ml": _maybe_float(getattr(region_data, "auc_epithelium_pmol_h_per_ml", None)),
                    "auc_tissue_pmol_h_per_ml": _maybe_float(getattr(region_data, "auc_tissue_pmol_h_per_ml", None)),
                    "auc_epithelium_tissue_pmol_h_per_ml": _maybe_float(getattr(region_data, "auc_epithelium_tissue_pmol_h_per_ml", None)),
                    "auc_epithelium_unbound_pmol_h_per_ml": _maybe_float(getattr(region_data, "auc_epithelium_unbound_pmol_h_per_ml", None)),
                    "auc_tissue_unbound_pmol_h_per_ml": _maybe_float(getattr(region_data, "auc_tissue_unbound_pmol_h_per_ml", None)),
                    "auc_epithelium_tissue_unbound_pmol_h_per_ml": _maybe_float(getattr(region_data, "auc_epithelium_tissue_unbound_pmol_h_per_ml", None))
                })

        if not regional_rows:
            # Fallback to legacy aggregation when comprehensive output is missing
            regional_amounts = getattr(pbbk_data, "region_slices", None)
            solution = getattr(pbbk_data, "y", None)
            times = np.asarray(getattr(pbbk_data, "t", []), dtype=float)
            if regional_amounts and solution is not None and times.size:
                for region_name, slice_obj in regional_amounts.items():
                    try:
                        data = solution[:, slice_obj]
                        total_amount = np.sum(data, axis=1)
                        auc_total = float(np.trapz(total_amount, times) / 3600.0)
                    except Exception:
                        auc_total = float("nan")
                    regional_rows.append({
                        "run_id": result.run_id,
                        "region": region_name,
                        "auc_elf": float("nan"),
                        "auc_epithelium_pmol_h_per_ml": auc_total,
                        "auc_tissue_pmol_h_per_ml": float("nan"),
                        "auc_epithelium_tissue_pmol_h_per_ml": auc_total,
                        "auc_epithelium_unbound_pmol_h_per_ml": float("nan"),
                        "auc_tissue_unbound_pmol_h_per_ml": float("nan"),
                        "auc_epithelium_tissue_unbound_pmol_h_per_ml": float("nan")
                    })

    if not regional_rows:
        regional_rows.append({
            "run_id": result.run_id,
            "region": "unknown",
            "auc_elf": float("nan"),
            "auc_epithelium_pmol_h_per_ml": float("nan"),
            "auc_tissue_pmol_h_per_ml": float("nan"),
            "auc_epithelium_tissue_pmol_h_per_ml": float("nan"),
            "auc_epithelium_unbound_pmol_h_per_ml": float("nan"),
            "auc_tissue_unbound_pmol_h_per_ml": float("nan"),
            "auc_epithelium_tissue_unbound_pmol_h_per_ml": float("nan")
        })

    frames["regional_auc"] = pd.DataFrame(regional_rows)

    if pbpk_timeseries_frames:
        frames["pbpk_regional_timeseries"] = pd.concat(pbpk_timeseries_frames, ignore_index=True).drop_duplicates().reset_index(drop=True)

    # Deposition bins from deposition results
    deposition_data = result.deposition
    if deposition_data and hasattr(deposition_data, 'region_ids') and hasattr(deposition_data, 'elf_initial_amounts'):
        metadata = getattr(deposition_data, 'metadata', {}) or {}

        region_amount_map: Dict[str, float] = {}
        alias_lookup: Dict[str, str] = {}

        def register_region(name: Any, amount: Any) -> None:
            if name is None:
                return
            canonical = str(name)
            try:
                amount_value = float(amount)
            except (TypeError, ValueError):
                return
            region_amount_map[canonical] = amount_value
            alias_lookup[canonical.lower()] = canonical

        def resolve_region(name: str) -> Optional[str]:
            if name in region_amount_map:
                return name
            return alias_lookup.get(name.lower())

        regional_names = metadata.get('regional_names')
        regional_amounts = metadata.get('regional_amounts_pmol')
        if regional_names is not None:
            regional_names = list(regional_names)
        if regional_amounts is not None:
            regional_amounts = list(regional_amounts)

        if regional_names and regional_amounts and len(regional_names) == len(regional_amounts):
            for region_name, amount in zip(regional_names, regional_amounts):
                register_region(region_name, amount)

        if not region_amount_map and len(deposition_data.region_ids) == len(deposition_data.elf_initial_amounts):
            default_region_lookup = {0: "ET", 1: "BB", 2: "bb", 3: "Al"}
            for region_id, amount in zip(deposition_data.region_ids, deposition_data.elf_initial_amounts):
                canonical = default_region_lookup.get(int(region_id), str(region_id))
                register_region(canonical, amount)

        total_amount = sum(region_amount_map.values())

        bin_rows = []
        deposition_by_generation = metadata.get('deposition_by_generation')
        if deposition_by_generation is not None:
            depo_array = np.asarray(deposition_by_generation)
            particle_bins = np.array([0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0])

            region_gen_mapping = {
                "ET": [0],
                "TB": list(range(1, 10)),
                "BB": list(range(1, 10)),
                "bb": list(range(10, 17)),
                "P1": list(range(10, 17)),
                "Al": list(range(17, 25)),
                "P2": list(range(17, 25)),
            }

            def region_weight(key: str, particle_size: float) -> float:
                if key in {"ET"}:
                    return np.exp(-0.3 * (particle_size - 2.0) ** 2) if particle_size >= 1.0 else 0.1
                if key in {"TB", "BB"}:
                    return np.exp(-0.2 * (particle_size - 1.5) ** 2) if particle_size >= 0.7 else 0.2
                if key in {"bb", "P1"}:
                    return np.exp(-0.1 * (particle_size - 1.0) ** 2)
                return np.exp(-0.4 * (particle_size - 0.8) ** 2) if particle_size <= 3.0 else 0.1

            for mapping_key, generations in region_gen_mapping.items():
                resolved = resolve_region(mapping_key)
                if resolved is None:
                    continue

                region_amount = region_amount_map.get(resolved)
                if region_amount is None:
                    region_amount = float(sum(depo_array[g] for g in generations if g < len(depo_array)))
                if region_amount is None:
                    continue

                weights = np.array([max(region_weight(mapping_key, size), 1e-6) for size in particle_bins])
                weights = weights / weights.sum() if weights.sum() > 0 else np.full_like(particle_bins, 1.0 / len(particle_bins))

                for weight, particle_size in zip(weights, particle_bins):
                    amount_pmol = region_amount * weight
                    fraction = amount_pmol / total_amount if total_amount > 0 else 0.0
                    bin_rows.append({
                        "run_id": result.run_id,
                        "region": resolved,
                        "particle_um": particle_size,
                        "amount_pmol": amount_pmol,
                        "fraction_of_dose": fraction
                    })
        else:
            for region_name, amount in region_amount_map.items():
                fraction = amount / total_amount if total_amount > 0 else 0.0
                bin_rows.append({
                    "run_id": result.run_id,
                    "region": region_name,
                    "particle_um": np.nan,
                    "amount_pmol": amount,
                    "fraction_of_dose": fraction
                })

        frames["deposition_bins"] = pd.DataFrame(bin_rows)
    else:
        depo_rows = []
        regions = ["ET", "TB", "P1", "P2", "A"]
        particle_sizes = [0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0]
        region_preferences = {
            "ET": [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005],
            "TB": [0.25, 0.3, 0.35, 0.4, 0.35, 0.3, 0.25, 0.15, 0.1, 0.05],
            "P1": [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.35, 0.25, 0.15],
            "P2": [0.04, 0.04, 0.04, 0.04, 0.12, 0.2, 0.25, 0.35, 0.45, 0.5],
            "A": [0.01, 0.01, 0.01, 0.01, 0.03, 0.05, 0.05, 0.13, 0.19, 0.29]
        }

        for i, size in enumerate(particle_sizes):
            total = sum(region_preferences[region][i] for region in regions)
            if total > 0:
                for region in regions:
                    normalized_fraction = region_preferences[region][i] / total
                    if normalized_fraction > 0.001:
                        depo_rows.append({
                            "run_id": result.run_id,
                            "region": region,
                            "particle_um": size,
                            "amount_pmol": np.nan,
                            "fraction_of_dose": normalized_fraction
                        })

        frames["deposition_bins"] = pd.DataFrame(depo_rows)
    
    # Solver statistics from stage metadata
    solver_rows = []
    for stage, info in stage_results.items():
        meta = info.get("metadata", {}) if isinstance(info, Mapping) else {}
        solver_rows.append({
            "run_id": result.run_id,
            "stage": stage,
            "method": meta.get("solver_method", "unknown"),
            "rtol": meta.get("solver_rtol", 0.0),
            "atol": meta.get("solver_atol", 0.0),
            "nfev": meta.get("solver_nfev", 0),
            "njev": meta.get("solver_njev", 0),
            "status": meta.get("solver_status", "unknown"),
            "runtime_s": meta.get("runtime_s", 0.0)
        })
    
    if solver_rows:
        frames["solver_stats"] = pd.DataFrame(solver_rows)
    else:
        # Placeholder solver stats
        frames["solver_stats"] = pd.DataFrame({
            "run_id": [result.run_id],
            "stage": ["placeholder"],
            "method": ["unknown"],
            "rtol": [0.0],
            "atol": [0.0],
            "nfev": [0],
            "njev": [0], 
            "status": ["not_run"],
            "runtime_s": [0.0]
        })
    
    # Subject parameters from entities metadata
    entities = result.metadata.get("entities", {})
    subject_info = entities.get("subject", {})
    
    # Handle case where subject_info might be a string summary
    if isinstance(subject_info, str):
        # Parse basic info from string like "adult_70kg: 35.0y, 70.0kg, M, BMI=22.9"
        parts = subject_info.split(": ")
        name = parts[0] if parts else "unknown"
        details = parts[1].split(", ") if len(parts) > 1 else []
        
        # Extract values with defaults
        age = weight = height = bmi = bsa = 0.0
        sex = "unknown"
        
        for detail in details:
            if detail.endswith("y"):
                age = float(detail[:-1]) if detail[:-1].replace(".", "").isdigit() else 0.0
            elif detail.endswith("kg"):
                weight = float(detail[:-2]) if detail[:-2].replace(".", "").isdigit() else 0.0
            elif detail in ["M", "F"]:
                sex = detail
            elif detail.startswith("BMI="):
                bmi = float(detail[4:]) if detail[4:].replace(".", "").isdigit() else 0.0
        
        subject_data = {
            "subject_ref": name,
            "weight_kg": weight,
            "height_cm": height,
            "age_years": age,
            "sex": sex,
            "bmi_kg_m2": bmi,
            "bsa_m2": bsa
        }
    else:
        # Handle dictionary format
        subject_data = {
            "subject_ref": subject_info.get("name", "unknown"),
            "weight_kg": subject_info.get("weight_kg", 0.0),
            "height_cm": subject_info.get("height_cm", 0.0),
            "age_years": subject_info.get("age_years", 0.0),
            "sex": subject_info.get("sex", "unknown"),
            "bmi_kg_m2": subject_info.get("bmi_kg_m2", 0.0),
            "bsa_m2": subject_info.get("bsa_m2", 0.0)
        }
    
    frames["subject_params"] = pd.DataFrame({
        "run_id": [result.run_id],
        **{k: [v] for k, v in subject_data.items()}
    })
    
    return frames


def calculate_summary_metrics(result: RunResult) -> Dict[str, float]:
    """Calculate summary pharmacokinetic metrics.
    
    Args:
        result: Simulation results
        
    Returns:
        Dictionary of calculated metrics (Cmax, Tmax, AUC, etc.)
    """
    metrics: Dict[str, float] = {}

    if result.pk is not None and result.pk.t is not None and result.pk.conc_plasma is not None:
        try:
            t = np.asarray(result.pk.t, dtype=float)
            conc = np.asarray(result.pk.conc_plasma, dtype=float)
            if t.size and conc.size:
                idx = int(np.nanargmax(conc))
                metrics["pk_cmax"] = float(np.nanmax(conc))
                metrics["pk_tmax_h"] = float(t[idx]) if t.size > idx else float(idx)
                metrics["pk_auc_0_last"] = float(np.trapz(conc, t))
        except Exception:
            pass

    if result.deposition is not None and result.deposition.elf_initial_amounts is not None:
        try:
            total_elf = float(np.nansum(result.deposition.elf_initial_amounts))
            metrics["deposition_total_elf_pmol"] = total_elf
        except Exception:
            pass

    if result.cfd is not None:
        for attr in ("mmad", "gsd", "mt_deposition_fraction"):
            value = getattr(result.cfd, attr, None)
            if isinstance(value, (int, float, np.integer, np.floating)):
                metrics[f"cfd_{attr}"] = float(value)

    pbpk_res = result.pbbk
    comp = getattr(pbpk_res, "comprehensive", None) if pbpk_res is not None else None
    if isinstance(comp, ComprehensivePBBMResults):
        try:
            total_lung = comp.get_total_lung_amount()
            metrics["pbpk_lung_final_pmol"] = float(total_lung[-1])
            metrics["pbpk_lung_auc_pmol_h"] = float(np.trapz(total_lung, comp.time_h))
        except Exception:
            pass
        pk_data = getattr(comp, "pk_data", None)
        if pk_data is not None and getattr(pk_data, "plasma_concentration", None) is not None:
            plasma = np.asarray(pk_data.plasma_concentration, dtype=float)
            time_h = np.asarray(comp.time_h, dtype=float)
            if plasma.size and time_h.size:
                idx = int(np.nanargmax(plasma))
                metrics["pbpk_plasma_cmax_pmol_ml"] = float(np.nanmax(plasma))
                metrics["pbpk_plasma_tmax_h"] = float(time_h[idx]) if time_h.size > idx else float(idx)
                metrics["pbpk_plasma_auc_pmol_h_ml"] = float(np.trapz(plasma, time_h))

    return metrics


def try_load_cached_result(
    run_id: str, 
    artifact_directory: Union[str, Path]
) -> Optional[RunResult]:
    """Attempt to load cached simulation results.
    
    Args:
        run_id: Run identifier to look for
        artifact_directory: Directory containing artifacts
        
    Returns:
        Cached results if found, None otherwise
    """
    # TODO: This will be implemented with the artifact system
    return None


# ---------------------------------------------------------------------------
#  Observed data import / fitting utilities
# ---------------------------------------------------------------------------


def load_observed_pk_csv(
    path: Union[str, Path],
    time_col: str,
    conc_col: str,
    time_unit: str = "h",
    label: str = "observed",
    **read_csv_kwargs,
) -> ObservedPK:
    """Load observed PK data from a CSV file.

    Args:
        path: CSV path
        time_col: Column name for time
        conc_col: Column name for concentration
        time_unit: 'h', 'min', or 's' (default 'h')
        label: Series label for plotting
        **read_csv_kwargs: forwarded to pandas.read_csv
    """
    return ObservedPK.from_csv(path, time_col, conc_col, time_unit=time_unit, label=label, **read_csv_kwargs)


def build_pk_fitting_metric(dataset: ObservedPK, loss: str = "sse") -> Callable[[RunResult], float]:
    """Create a metric callable to compare pipeline outputs to observed PK."""
    return make_pk_fitting_metric(dataset, loss=loss)


def pk_overlay_dataframe(result: RunResult, dataset: ObservedPK) -> pd.DataFrame:
    """Return a tidy DataFrame for plotting observed vs predicted PK curves."""
    return make_pk_overlay_data(result, dataset)


def run_simulation_with_replicates(
    config: AppConfig,
    parameter_overrides: Optional[Mapping[str, Any]] = None,
    run_id: Optional[str] = None,
    artifact_directory: Optional[Union[str, Path]] = None,
    workflow: Optional[Workflow] = None,
    workflow_name: Optional[str] = None,
) -> List[RunResult]:
    """Run simulation with multiple replicates for variability sampling.
    
    Args:
        config: Configuration to use
        parameter_overrides: Optional parameter overrides to apply
        run_id: Optional base run identifier (generated if not provided)
        artifact_directory: Directory to save artifacts (uses config default if not provided)
        
    Returns:
        List of simulation results, one per replicate
        
    Raises:
        ValidationError: If configuration is invalid
        ModelError: If simulation fails
    """
    if run_id is None:
        run_id = f"run_{uuid.uuid4().hex[:8]}"
    
    n_replicates = config.run.n_replicates
    variability_spec = config.get_effective_variability()
    study_cfg = getattr(config, 'study', None)
    study_type = getattr(study_cfg, 'study_type', None) if study_cfg else None
    study_design = getattr(study_cfg, 'design', None) if study_cfg else None
    charcoal_block = getattr(study_cfg, 'charcoal_block', False) if study_cfg else False
    suppress_et_absorption = getattr(config.pbbm, 'suppress_et_absorption', False)
    charcoal_block = charcoal_block or getattr(config.pbbm, 'charcoal_block', False)

    resolved_workflow_name = workflow_name or config.run.workflow_name
    selected_workflow = workflow or (
        get_workflow(resolved_workflow_name) if resolved_workflow_name else get_workflow("deposition_pbbm_pk")
    )
    
    if parameter_overrides:
        config = _apply_config_overrides(config, parameter_overrides)

    logger.info("Starting multi-replicate simulation", 
                run_id=run_id, 
                n_replicates=n_replicates)
    
    # Hydrate configuration with real entities (base entities)
    try:
        hydrated = hydrate_config(config)
        validate_hydrated_entities(hydrated)
        
        base_entities = {
            "subject": hydrated["subject"],
            "api": hydrated["api"], 
            "product": hydrated["product"],
            "maneuver": hydrated["maneuver"]
        }

        _apply_pk_parameter_overrides(base_entities['subject'], base_entities['api'], config.pk)

    except Exception as e:
        logger.error("Entity resolution failed", error=str(e))
        raise ValidationError(f"Entity resolution failed: {e}")

    results = []
    
    variability_overrides = _extract_population_variability_settings(config, selected_workflow)
    effective_variability_spec = apply_population_variability_settings(
        variability_spec, variability_overrides
    )

    stage_overrides = _build_stage_overrides_from_config(config)

    # Generate Inter subject first (if variability enabled)
    inter_rng = create_deterministic_rng(config.run.seed, run_id)
    inter_entities, inter_factors = build_inter_subject(
        base_entities, effective_variability_spec, inter_rng, run_id
    )

    # Generate Intra subjects (one per replicate)
    for replicate_id in range(n_replicates):
        intra_rng = create_deterministic_rng(config.run.seed, run_id, replicate_id)
        final_entities = build_intra_subject(
            inter_entities, inter_factors, effective_variability_spec, intra_rng, replicate_id
        )

        _apply_pk_parameter_overrides(final_entities['subject'], final_entities['api'], config.pk)

        # Create replicate run ID
        if n_replicates == 1:
            replicate_run_id = run_id
        else:
            replicate_run_id = f"{run_id}_rep{replicate_id+1:03d}"
        
        logger.info("Processing replicate", 
                   run_id=replicate_run_id,
                   replicate=replicate_id+1,
                   total=n_replicates)
        
        product_name = getattr(base_entities['product'], 'name', config.product.ref or 'product')

        task = SubjectTask(
            subject_index=replicate_id,
            api_name=getattr(base_entities['api'], 'name', config.api.ref or 'api'),
            subject_name=getattr(base_entities['subject'], 'name', config.subject.ref or 'subject'),
            products=(product_name,),
            seed=config.run.seed + replicate_id,
            subject=final_entities['subject'],
            api=final_entities['api'],
            product_entities={product_name: final_entities['product']},
            apply_variability=False,
            variability_settings=variability_overrides,
            study_type=study_type,
            study_design=study_design,
            charcoal_block=charcoal_block,
            suppress_et_absorption=suppress_et_absorption,
        )

        try:
            task_result = run_pipeline_for_task(
                task,
                workflow=selected_workflow,
                stage_overrides=stage_overrides,
            )
        except Exception as e:
            logger.error(
                "Pipeline execution failed",
                run_id=replicate_run_id,
                replicate=replicate_id + 1,
                error=str(e),
            )
            raise

        product_result = task_result.products[product_name]
        runtime = task_result.runtime_metadata.get('total_runtime_s', 0.0)
        product_stage_results = task_result.stage_results.get(product_name, {})
        stage_metadata = {
            stage: {
                "model": stage_result.model_name,
                "metadata": stage_result.metadata,
            }
            for stage, stage_result in product_stage_results.items()
        }

        result = RunResult(
            run_id=replicate_run_id,
            config=config.model_dump(),
            cfd=product_result.cfd,
            deposition=product_result.deposition,
            pbbk=product_result.pbpk,
            pk=product_result.pk,
            runtime_seconds=runtime,
            metadata={
                "status": "completed",
                "message": "Pipeline executed with variability",
                "replicate_id": replicate_id,
                "base_run_id": run_id,
                "variability_applied": len(variability_spec.layers) > 0,
                "stage_times": task_result.runtime_metadata.get('stage_times', {}),
                "stage_results": stage_metadata,
            },
        )

        results.append(result)
    
    logger.info("Multi-replicate simulation completed", 
                run_id=run_id, 
                replicates=len(results))
    return results
