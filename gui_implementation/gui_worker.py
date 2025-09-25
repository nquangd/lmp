#!/usr/bin/env python3
"""GUI Worker Process Module.

This module implements the worker process interface for running simulations
from the GUI. Workers run in separate processes to avoid blocking the UI.

Based on the GUI plan requirements:
- Workers are launched via QProcess with `python -m lmp_pkg.gui_worker run --run-id <id> --config <path>`
- Progress is reported via JSONL to stdout
- Results are saved to runs/<run_id>/ directory
"""

import sys
import json
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, Mapping, List, Tuple
import structlog
import typer
import traceback
import numpy as np

# Import the existing app_api
sys.path.insert(0, str(Path(__file__).parent.parent / "lmp_pkg" / "src"))
from lmp_pkg import app_api
from lmp_pkg.config import AppConfig
from lmp_pkg.contracts import RunResult
from lmp_pkg.contracts.errors import LMPError
from lmp_pkg.engine.workflow import get_workflow
from lmp_pkg.run_types import normalise_run_label
from lmp_pkg.services.stage_summaries import build_stage_metrics, determine_stage_order
from lmp_pkg.simulation.pipeline_runner import run_pipeline_for_task
from lmp_pkg.simulation.subject_tasks import SubjectTask


app = typer.Typer(
    name="gui_worker",
    help="GUI Worker Process - Run simulations for the GUI frontend",
    no_args_is_help=True
)

# Configure structured logging for GUI consumption
logger = structlog.get_logger()


def emit_progress(event_type: str, **kwargs):
    """Emit progress event as JSONL to stdout for GUI consumption."""
    event = {
        "event": event_type,
        "timestamp": structlog.processors.TimeStamper()(None, None, {})["timestamp"],
        **kwargs
    }
    print(json.dumps(event), flush=True)


def emit_error(error_msg: str, details: Optional[str] = None):
    """Emit error event."""
    emit_progress("error", message=error_msg, details=details)


def emit_metric(name: str, value: float, units: Optional[str] = None):
    """Emit metric event."""
    emit_progress("metric", name=name, value=value, units=units)


def emit_checkpoint(path: str, stage: str):
    """Emit checkpoint save event."""
    emit_progress("checkpoint", path=path, stage=stage)


def _json_default(value):
    """Fallback serializer for types emitted by simulation results."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, set):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _set_by_path(target: Dict[str, Any], path_parts: list[str], value: Any) -> None:
    current = target
    for key in path_parts[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[path_parts[-1]] = value



def apply_parameter_overrides(cfg: AppConfig, overrides: Optional[Dict[str, Any]]) -> AppConfig:
    if not overrides:
        return cfg

    data = cfg.model_dump()
    for dotted_key, override_value in overrides.items():
        if not dotted_key:
            continue
        path_parts = dotted_key.split('.')
        _set_by_path(data, path_parts, override_value)

    return AppConfig.model_validate(data)


def _extract_pk_timeseries(result) -> tuple[np.ndarray, np.ndarray]:
    if getattr(result, "pbbk", None) is not None and getattr(result.pbbk, "comprehensive", None) is not None:
        comp = result.pbbk.comprehensive
        return np.asarray(comp.time_s, dtype=float), np.asarray(comp.pk_data.plasma_concentration_ng_per_ml, dtype=float)
    if getattr(result, "pk", None) is not None:
        return np.asarray(result.pk.t, dtype=float), np.asarray(result.pk.conc_plasma, dtype=float)
    raise ValueError("Simulation result does not contain PK output")


def _load_parent_estimation_settings(workspace: Path, parent_run_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if not parent_run_id:
        return None
    parent_dir = workspace / "runs" / parent_run_id
    metadata_path = parent_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        with open(metadata_path, "r") as handle:
            parent_meta = json.load(handle)
    except Exception:
        return None
    settings = parent_meta.get("estimation_settings") or parent_meta.get("parameter_estimation")
    if isinstance(settings, dict):
        return settings
    return None


SCALAR_SUMMARY_KEYS = {
    "pk_auc_0_t": "pk_auc_0_last",
    "cfd_mmad": "cfd_mmad",
    "cfd_gsd": "cfd_gsd",
    "cfd_mt_fraction": "cfd_mt_deposition_fraction",
}


def _evaluate_timeseries_target(result, target: Mapping[str, Any]) -> Dict[str, Any]:
    observed_spec = target.get("observed") or {}
    observed_time = observed_spec.get("time_s")
    observed_values = observed_spec.get("values")
    if observed_time is None or observed_values is None:
        raise ValueError("Observed dataset missing time_s or values")
    observed_time_arr = np.asarray(observed_time, dtype=float)
    observed_values_arr = np.asarray(observed_values, dtype=float)
    if observed_time_arr.size == 0 or observed_values_arr.size == 0:
        raise ValueError("Observed series is empty")

    pred_time, pred_conc = _extract_pk_timeseries(result)
    if pred_time.size == 0 or pred_conc.size == 0:
        raise ValueError("Predicted PK series is empty")

    predicted_interp = np.interp(observed_time_arr, pred_time, pred_conc)
    residuals = observed_values_arr - predicted_interp

    sse = float(np.sum(residuals ** 2))
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    loss_name = str(target.get("loss") or "sse").lower()
    objective_map = {
        "sse": sse,
        "mae": mae,
        "rmse": rmse,
    }
    objective_value = objective_map.get(loss_name, sse)

    return {
        "metric": "pk_concentration",
        "type": "timeseries",
        "loss": loss_name,
        "objective": objective_value,
        "sse": sse,
        "mae": mae,
        "rmse": rmse,
        "series": {
            "time_s": observed_time_arr.tolist(),
            "time_h": (observed_time_arr / 3600.0).tolist(),
            "observed": observed_values_arr.tolist(),
            "predicted": predicted_interp.tolist(),
            "residual": residuals.tolist(),
        },
        "observed_count": int(observed_time_arr.size),
    }


def _evaluate_scalar_target(summary_metrics: Mapping[str, Any], metric: str, observed_spec: Mapping[str, Any]) -> Dict[str, Any]:
    key = SCALAR_SUMMARY_KEYS.get(metric)
    if key is None:
        raise ValueError(f"Unsupported scalar metric '{metric}'")
    predicted = summary_metrics.get(key)
    if predicted is None:
        raise ValueError(f"Predicted value for metric '{metric}' not available")
    observed_value = observed_spec.get("value")
    if observed_value is None:
        raise ValueError(f"Observed value for metric '{metric}' missing")
    observed_float = float(observed_value)
    predicted_float = float(predicted)
    residual = predicted_float - observed_float
    objective = residual ** 2
    return {
        "metric": metric,
        "type": "scalar",
        "objective": objective,
        "predicted": predicted_float,
        "observed": observed_float,
        "residual": residual,
    }


def _evaluate_deposition_fraction_target(dataframes: Mapping[str, Any], observed_spec: Mapping[str, Any]) -> Dict[str, Any]:
    regions_spec = observed_spec.get("regions")
    if not isinstance(regions_spec, Mapping) or not regions_spec:
        raise ValueError("Deposition fraction targets require a regions mapping")

    df = dataframes.get("deposition_bins")
    if df is None or df.empty:
        raise ValueError("Deposition fraction data not available in results")

    grouped = df.groupby("region")["fraction_of_dose"].sum()
    per_region: List[Dict[str, Any]] = []
    objective = 0.0
    for region_name, observed_value in regions_spec.items():
        predicted_value = float(grouped.get(region_name, 0.0))
        observed_float = float(observed_value)
        residual = predicted_value - observed_float
        per_region.append({
            "region": region_name,
            "predicted": predicted_value,
            "observed": observed_float,
            "residual": residual,
        })
        objective += residual ** 2

    return {
        "metric": "deposition_fraction",
        "type": "regional",
        "objective": objective,
        "per_region": per_region,
    }


def _evaluate_parameter_estimation_targets(
    result,
    summary_metrics: Mapping[str, Any],
    dataframes: Mapping[str, Any],
    targets: List[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], float]:
    evaluations: List[Dict[str, Any]] = []
    combined_objective = 0.0

    for target in targets:
        metric = target.get("metric")
        weight = float(target.get("weight", 1.0))
        try:
            if metric == "pk_concentration":
                evaluation = _evaluate_timeseries_target(result, target)
            elif metric in SCALAR_SUMMARY_KEYS:
                evaluation = _evaluate_scalar_target(summary_metrics, metric, target.get("observed") or {})
            elif metric == "deposition_fraction":
                evaluation = _evaluate_deposition_fraction_target(dataframes, target.get("observed") or {})
            else:
                raise ValueError(f"Unsupported estimation metric '{metric}'")
        except Exception as exc:
            evaluations.append({
                "metric": metric,
                "weight": weight,
                "error": str(exc),
            })
            continue

        evaluation["metric"] = metric
        evaluation["weight"] = weight
        objective_value = evaluation.get("objective")
        if objective_value is not None:
            weighted_objective = float(objective_value) * weight
            evaluation["weighted_objective"] = weighted_objective
            combined_objective += weighted_objective
        evaluations.append(evaluation)

    return evaluations, combined_objective


def _sanitise_product_name(name: str) -> str:
    slug = normalise_run_label(name)
    return slug or name.replace(" ", "_").replace("/", "_")


def _run_virtual_subject(
    cfg: AppConfig,
    task_spec: Mapping[str, Any],
    run_dir: Path,
    run_id: str,
    run_type: str,
) -> Tuple[List[str], Dict[str, Any]]:
    """Execute a virtual subject task (trial or VBE) and persist results."""

    if not task_spec:
        raise ValueError("task_spec is required for virtual study runs")

    subject_index = int(task_spec.get("subject_index", 0))
    subject_name = str(task_spec.get("subject_name") or cfg.subject.ref)
    api_name = str(task_spec.get("api") or cfg.api.ref)
    seed = int(task_spec.get("seed", cfg.run.seed))
    apply_variability = bool(task_spec.get("apply_variability", True))
    variability_settings = task_spec.get("variability_settings")
    study_type = task_spec.get("study_type")
    study_design = task_spec.get("study_design")
    charcoal_block = bool(task_spec.get("charcoal_block", False))
    suppress_et_absorption = bool(task_spec.get("suppress_et_absorption", False))

    product_entries = task_spec.get("products")
    product_roles: Dict[str, Optional[str]] = {}
    if isinstance(product_entries, list) and product_entries and isinstance(product_entries[0], Mapping):
        product_names = [str(entry.get("name")) for entry in product_entries if entry.get("name")]
        product_roles = {str(entry.get("name")): entry.get("role") for entry in product_entries if entry.get("name")}
    else:
        products_raw = product_entries or [cfg.product.ref]
        product_names = [str(name) for name in products_raw]
        if run_type == "virtual_trial" and len(product_names) == 1:
            product_roles[product_names[0]] = "reference"

    if not product_names:
        raise ValueError("No products specified for virtual study task")

    stage_overrides = app_api._build_stage_overrides_from_config(cfg)  # type: ignore[attr-defined]

    workflow_name = task_spec.get("workflow_name") or cfg.run.workflow_name or "deposition_pbbm_pk"
    workflow = get_workflow(workflow_name)

    task = SubjectTask(
        subject_index=subject_index,
        api_name=api_name,
        subject_name=subject_name,
        products=tuple(product_names),
        seed=seed,
        apply_variability=apply_variability,
        variability_settings=variability_settings,
        study_type=study_type,
        study_design=study_design,
        charcoal_block=charcoal_block,
        suppress_et_absorption=suppress_et_absorption,
    )

    task_result = run_pipeline_for_task(
        task,
        workflow=workflow,
        stage_overrides=stage_overrides,
    )

    results_dir = run_dir / "results"
    results_dir.mkdir(exist_ok=True)

    saved_result_files: List[str] = []
    products_payload: Dict[str, Any] = {}

    total_runtime = task_result.runtime_metadata.get("total_runtime_s", 0.0)

    for product_name, product_result in task_result.products.items():
        product_slug = _sanitise_product_name(product_name)
        product_run_id = f"{run_id}_{product_slug}"
        run_result = RunResult(
            run_id=product_run_id,
            config=cfg.model_dump(),
            cfd=product_result.cfd,
            deposition=product_result.deposition,
            pbbk=product_result.pbpk,
            pk=product_result.pk,
            runtime_seconds=total_runtime,
            metadata={
                "subject_index": subject_index,
                "product": product_name,
                "product_role": product_roles.get(product_name),
                "seed": seed,
            },
        )

        frames = app_api.convert_results_to_dataframes(run_result)
        stage_metrics = build_stage_metrics(run_result)
        summary_metrics = app_api.calculate_summary_metrics(run_result)
        stage_order = determine_stage_order(run_result, stage_metrics)

        product_files: List[str] = []
        for frame_name, df in frames.items():
            df = df.copy()
            df["product"] = product_name
            if product_roles.get(product_name):
                df["product_role"] = product_roles[product_name]
            filename = f"{product_slug}__{frame_name}.parquet"
            parquet_path = results_dir / filename
            try:
                df.to_parquet(parquet_path, index=False)
            except ImportError:
                filename = f"{product_slug}__{frame_name}.csv"
                parquet_path = results_dir / filename
                df.to_csv(parquet_path, index=False)
            product_files.append(parquet_path.name)
            saved_result_files.append(parquet_path.name)
            emit_checkpoint(str(parquet_path), f"results_{product_slug}_{frame_name}")

        products_payload[product_name] = {
            "role": product_roles.get(product_name),
            "summary_metrics": summary_metrics,
            "stage_metrics": stage_metrics,
            "stage_order": stage_order,
            "result_files": product_files,
        }

    metadata = {
        "run_type": run_type,
        "subject_index": subject_index,
        "subject_name": subject_name,
        "api_name": api_name,
        "seed": seed,
        "apply_variability": apply_variability,
        "products": products_payload,
        "runtime_seconds": float(total_runtime),
        "task_spec": task_spec,
    }

    return saved_result_files, metadata
def load_worker_config(config_path: Path) -> AppConfig:
    """Load configuration for a worker run supporting JSON or TOML."""
    if config_path.suffix.lower() == ".json":
        with open(config_path, "r") as f:
            data = json.load(f)

        if isinstance(data, dict) and "config" in data:
            data = data["config"]

        return AppConfig.model_validate(data)

    return app_api.load_config_from_file(config_path)


@app.command()
def run(
    run_id: str = typer.Option(..., "--run-id", help="Unique run identifier"),
    config: Path = typer.Option(..., "--config", help="Configuration file path"),
    workspace: Optional[Path] = typer.Option(None, "--workspace", help="Workspace directory"),
    manifest_index: Optional[int] = typer.Option(None, "--manifest-index", help="Index in manifest for sweeps"),
    total_runs: Optional[int] = typer.Option(None, "--total-runs", help="Total runs in sweep"),
    run_type: str = typer.Option("single", "--run-type", help="Run type identifier"),
    run_label: Optional[str] = typer.Option(None, "--run-label", help="User-facing run label"),
    parent_run_id: Optional[str] = typer.Option(None, "--parent-run-id", help="Parent run grouping identifier"),
    parameter_overrides: Optional[str] = typer.Option(None, "--parameter-overrides", help="JSON encoded parameter overrides"),
    task_spec: Optional[str] = typer.Option(None, "--task-spec", help="JSON encoded subject/task specification"),
):
    """Run a single simulation for the GUI."""

    try:
        emit_progress("started", run_id=run_id, config=str(config))

        # Determine workspace and run directory
        if workspace is None:
            workspace = Path.cwd() / "workspace"

        workspace = Path(workspace)
        run_dir = workspace / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration
        emit_progress("progress", message="Loading configuration", pct=5.0)
        cfg = load_worker_config(config)

        overrides_dict: Optional[Dict[str, Any]] = None
        if parameter_overrides:
            try:
                overrides_dict = json.loads(parameter_overrides)
            except json.JSONDecodeError as exc:
                emit_error("Invalid parameter overrides", str(exc))
                sys.exit(1)

        if overrides_dict:
            cfg = apply_parameter_overrides(cfg, overrides_dict)

        # Save exact config used for this run
        config_copy = run_dir / "config.json"
        with open(config_copy, 'w') as f:
            json.dump(cfg.model_dump(), f, indent=2, default=_json_default)

        # Validate configuration
        emit_progress("progress", message="Validating configuration", pct=10.0)
        app_api.validate_configuration(cfg)

        # Determine task specification (if provided)
        task_spec_dict: Optional[Dict[str, Any]] = None
        if task_spec:
            try:
                task_spec_dict = json.loads(task_spec)
            except json.JSONDecodeError as exc:
                emit_error("Invalid task specification", str(exc))
                sys.exit(1)

        emit_progress("progress", message="Starting simulation pipeline", pct=15.0)

        results_dir = run_dir / "results"
        results_dir.mkdir(exist_ok=True)

        saved_result_files: List[str] = []
        metadata_payload: Dict[str, Any] = {}
        estimation_payload: Optional[Dict[str, Any]] = None
        dataframes: Dict[str, Any] = {}
        stage_metrics: Dict[str, Any] = {}
        summary_metrics: Dict[str, Any] = {}
        stage_order: List[str] = []

        if run_type in {"virtual_trial", "virtual_bioequivalence"}:
            if task_spec_dict is None:
                emit_error("Virtual study task specification missing", None)
                sys.exit(1)
            try:
                saved_result_files, metadata_payload = _run_virtual_subject(
                    cfg,
                    task_spec_dict,
                    run_dir,
                    run_id,
                    run_type,
                )
            except Exception as exc:
                error_details = traceback.format_exc()
                emit_error(f"Virtual study run failed: {exc}", error_details)
                sys.exit(1)
            emit_progress("progress", message="Virtual study completed", pct=95.0)
        else:
            # Run a single simulation
            result = app_api.run_single_simulation(
                cfg,
                run_id=run_id,
                artifact_directory=run_dir
            )

            emit_progress("progress", message="Simulation completed", pct=90.0)

            # Convert results to DataFrames and save
            emit_progress("progress", message="Converting results", pct=95.0)
            dataframes = app_api.convert_results_to_dataframes(result)

            for frame_name, df in dataframes.items():
                parquet_path = results_dir / f"{frame_name}.parquet"
                try:
                    df.to_parquet(parquet_path, index=False)
                    saved_result_files.append(parquet_path.name)
                    emit_checkpoint(str(parquet_path), f"results_{frame_name}")
                except ImportError:
                    csv_path = results_dir / f"{frame_name}.csv"
                    df.to_csv(csv_path, index=False)
                    saved_result_files.append(csv_path.name)
                    emit_checkpoint(str(csv_path), f"results_{frame_name}_csv")
                except Exception:
                    raise

            stage_metrics = build_stage_metrics(result)
            summary_metrics = app_api.calculate_summary_metrics(result)
            stage_order = determine_stage_order(result, stage_metrics)

            estimation_settings = _load_parent_estimation_settings(workspace, parent_run_id)
            if run_type == "parameter_estimation" and estimation_settings:
                targets = estimation_settings.get("targets") or []
                if isinstance(targets, list) and targets:
                    evaluations, combined_objective = _evaluate_parameter_estimation_targets(
                        result,
                        summary_metrics,
                        dataframes,
                        targets,
                    )
                    estimation_payload = {
                        "targets": evaluations,
                        "combined_objective": combined_objective,
                    }
                else:
                    logger.warning("Parameter estimation targets missing", run_id=run_id)

            metadata_payload.update({
                "summary_metrics": summary_metrics,
                "stage_metrics": stage_metrics,
                "stage_order": stage_order,
            })

        metadata_path = run_dir / "metadata.json"
        existing_metadata: Dict[str, Any] = {}
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as handle:
                    existing_metadata = json.load(handle)
            except Exception:
                existing_metadata = {}

        runtime_seconds_value: float
        if run_type in {"virtual_trial", "virtual_bioequivalence"}:
            runtime_seconds_value = float(metadata_payload.get("runtime_seconds", 0.0))
        else:
            runtime_seconds_value = float(result.runtime_seconds)

        metadata = dict(existing_metadata)
        metadata.update({
            "run_id": run_id,
            "config_path": str(config),
            "run_type": run_type,
            "label": normalise_run_label(run_label),
            "display_label": run_label,
            "parent_run_id": parent_run_id,
            "manifest_index": manifest_index,
            "total_runs": total_runs,
            "parameter_overrides": overrides_dict,
            "status": "completed",
            "runtime_seconds": runtime_seconds_value,
            "result_files": saved_result_files,
        })

        if run_type not in {"virtual_trial", "virtual_bioequivalence"}:
            metadata.update({
                "metadata": result.metadata,
                "stage_metrics": stage_metrics,
                "summary_metrics": summary_metrics,
                "stage_order": stage_order,
            })

        metadata.update(metadata_payload)

        if estimation_payload is not None:
            metadata["parameter_estimation"] = estimation_payload
            combined_objective = estimation_payload.get("combined_objective")
            if combined_objective is not None:
                emit_metric("objective_combined", float(combined_objective))
            for evaluation in estimation_payload.get("targets", []):
                if "error" in evaluation:
                    continue
                objective_value = evaluation.get("objective")
                metric_name = evaluation.get("metric")
                if metric_name and objective_value is not None:
                    emit_metric(f"objective_{metric_name}", float(objective_value))

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=_json_default)

        # Calculate and emit summary metrics
        if run_type in {"virtual_trial", "virtual_bioequivalence"}:
            products_payload = metadata_payload.get("products", {})
            for product_name, payload in products_payload.items():
                product_metrics = payload.get("summary_metrics") or {}
                for metric_name, value in product_metrics.items():
                    try:
                        emit_metric(f"{product_name}:{metric_name}", float(value))
                    except Exception:
                        continue
        else:
            for metric_name, value in summary_metrics.items():
                emit_metric(metric_name, value)

        emit_progress("completed",
                     run_id=run_id,
                     runtime=runtime_seconds_value,
                     results_dir=str(results_dir))

    except LMPError as e:
        emit_error(f"LMP Error: {e.message}", str(e.details) if e.details else None)
        sys.exit(1)
    except Exception as e:
        error_details = traceback.format_exc()
        emit_error(f"Unexpected error: {str(e)}", error_details)
        sys.exit(1)


@app.command()
def validate(
    config: Path = typer.Argument(..., help="Configuration file to validate")
):
    """Validate a configuration file for the GUI."""

    try:
        emit_progress("started", task="validation", config=str(config))

        cfg = app_api.load_config_from_file(config)
        app_api.validate_configuration(cfg)

        emit_progress("completed", task="validation", status="valid")

    except LMPError as e:
        emit_error(f"Validation failed: {e.message}", str(e.details) if e.details else None)
        sys.exit(1)
    except Exception as e:
        error_details = traceback.format_exc()
        emit_error(f"Validation error: {str(e)}", error_details)
        sys.exit(1)


@app.command()
def sweep(
    manifest: Path = typer.Option(..., "--manifest", help="Manifest parquet file"),
    workspace: Optional[Path] = typer.Option(None, "--workspace", help="Workspace directory"),
    max_parallel: int = typer.Option(1, "--max-parallel", help="Maximum parallel processes"),
    run_indices: Optional[str] = typer.Option(None, "--indices", help="Comma-separated run indices to execute")
):
    """Run a parameter sweep from a manifest file."""

    try:
        import pandas as pd

        emit_progress("started", task="sweep", manifest=str(manifest))

        # Load manifest
        manifest_df = pd.read_parquet(manifest)
        total_runs = len(manifest_df)

        # Filter by indices if specified
        if run_indices:
            indices = [int(i.strip()) for i in run_indices.split(",")]
            manifest_df = manifest_df.iloc[indices]
            emit_progress("progress", message=f"Filtered to {len(indices)} runs", pct=0)

        emit_progress("sweep_info", total_runs=len(manifest_df), max_parallel=max_parallel)

        # TODO: Implement actual sweep execution with process pool
        # For now, just report what would be done
        for idx, row in manifest_df.iterrows():
            run_id = row["run_id"]
            emit_progress("sweep_run", run_id=run_id, index=idx, total=len(manifest_df))

        emit_progress("completed", task="sweep", runs_completed=len(manifest_df))

    except Exception as e:
        error_details = traceback.format_exc()
        emit_error(f"Sweep error: {str(e)}", error_details)
        sys.exit(1)


if __name__ == "__main__":
    app()
