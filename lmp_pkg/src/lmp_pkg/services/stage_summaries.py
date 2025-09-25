"""Stage-level summary utilities for GUI consumption."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np

from ..contracts.types import CFDResult, DepositionResult, PBBKResult, PKResult, RunResult
from ..data_structures.comprehensive_results import ComprehensivePBBMResults


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return None


def normalize_stage_name(stage_name: Optional[str]) -> Optional[str]:
    """Return a normalised, lower-case identifier for a pipeline stage."""

    if not stage_name:
        return stage_name

    stage = stage_name.strip().lower().replace(" ", "_")
    if ":" in stage:
        prefix, suffix = stage.split(":", 1)
        if prefix == "analysis" and "bioequivalence" in suffix:
            return "vbe"
        stage = stage.replace(":", "_")
    if stage.startswith("analysis") and "bioequivalence" in stage:
        return "vbe"
    return stage


def _collect_numeric_scalars(value: Any) -> Dict[str, float]:
    """Extract numeric scalar values from nested structures for metrics."""

    scalars: Dict[str, float] = {}

    def _visit(current: Any, prefix: Optional[str] = None) -> None:
        numeric = _as_float(current)
        if numeric is not None:
            key = prefix if prefix else "value"
            scalars[key] = numeric
            return

        if current is None:
            return

        if isinstance(current, np.ndarray):
            if current.size == 1:
                single = _as_float(current.item())
                if single is not None and prefix:
                    scalars[prefix] = single
            return

        if isinstance(current, Mapping):
            module = type(current).__module__
            if module and module.startswith("pandas."):
                return
            for key, value in current.items():
                if key is None:
                    continue
                key_str = str(key).strip()
                if not key_str:
                    continue
                safe_key = key_str.replace(" ", "_")
                child_prefix = f"{prefix}_{safe_key}" if prefix else safe_key
                _visit(value, child_prefix)
            return

        if isinstance(current, (list, tuple)) and len(current) <= 4:
            for idx, item in enumerate(current):
                child_prefix = f"{prefix}_{idx}" if prefix else str(idx)
                _visit(item, child_prefix)

    _visit(value)
    return {k: v for k, v in scalars.items() if k}


def summarise_cfd(result: Optional[CFDResult]) -> Dict[str, float]:
    if result is None:
        return {}

    metrics: Dict[str, float] = {}

    for attr in ("mmad", "gsd", "mt_deposition_fraction"):
        value = getattr(result, attr, None)
        numeric = _as_float(value)
        if numeric is not None:
            metrics[attr] = numeric

    metadata = result.metadata or {}
    for key, value in metadata.items():
        numeric = _as_float(value)
        if numeric is not None:
            metrics[f"metadata_{key}"] = numeric

    return metrics


def summarise_deposition(result: Optional[DepositionResult]) -> Dict[str, float]:
    if result is None:
        return {}

    metrics: Dict[str, float] = {}

    try:
        elf_amounts = np.asarray(result.elf_initial_amounts, dtype=float)
    except Exception:
        elf_amounts = np.array([])

    if elf_amounts.size:
        metrics["total_elf_initial_pmol"] = float(np.nansum(elf_amounts))

    region_ids = result.region_ids if result.region_ids is not None else []
    regional_metadata = (result.metadata or {}).get("regional_amounts_pmol")
    regional_names = (result.metadata or {}).get("regional_names")

    region_map: Dict[str, float] = {}

    if regional_names is not None and regional_metadata is not None:
        try:
            for name, amount in zip(regional_names, regional_metadata):
                numeric = _as_float(amount)
                if numeric is not None:
                    region_map[str(name)] = numeric
        except Exception:
            region_map = {}

    if not region_map and len(region_ids) == len(elf_amounts):
        default_region_lookup = {0: "ET", 1: "BB", 2: "bb", 3: "Al"}
        for region_id, amount in zip(region_ids, elf_amounts):
            numeric = _as_float(amount)
            if numeric is None:
                continue
            region_name = default_region_lookup.get(int(region_id), str(region_id))
            region_map[region_name] = numeric

    for region, amount in region_map.items():
        metrics[f"region_{region}_elf_pmol"] = amount

    metadata = result.metadata or {}
    for key, value in metadata.items():
        numeric = _as_float(value)
        if numeric is not None:
            metrics[f"metadata_{key}"] = numeric

    return metrics


def _pbpk_region_metrics(comp: ComprehensivePBBMResults) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for region_name, region_data in comp.regional_data.items():
        try:
            final_total = float(region_data.total_amounts[-1])
            metrics[f"region_{region_name}_total_final_pmol"] = final_total
        except Exception:
            pass
        if region_data.auc_epithelium_pmol_h_per_ml is not None:
            metrics[f"region_{region_name}_auc_epithelium"] = float(region_data.auc_epithelium_pmol_h_per_ml)
        if region_data.auc_tissue_pmol_h_per_ml is not None:
            metrics[f"region_{region_name}_auc_tissue"] = float(region_data.auc_tissue_pmol_h_per_ml)
        if region_data.auc_epithelium_unbound_pmol_h_per_ml is not None:
            metrics[f"region_{region_name}_auc_epithelium_unbound"] = float(region_data.auc_epithelium_unbound_pmol_h_per_ml)
    return metrics


def summarise_pbpk(result: Optional[PBBKResult]) -> Dict[str, float]:
    if result is None:
        return {}

    metrics: Dict[str, float] = {}

    comp = result.comprehensive
    if isinstance(comp, ComprehensivePBBMResults):
        time_h = np.asarray(comp.time_h, dtype=float)
        try:
            total_lung = comp.get_total_lung_amount()
            metrics["lung_total_final_pmol"] = float(total_lung[-1])
            metrics["lung_total_auc_pmol_h"] = float(np.trapz(total_lung, time_h))
        except Exception:
            pass

        pk_data = getattr(comp, "pk_data", None)
        if pk_data is not None:
            plasma = getattr(pk_data, "plasma_concentration", None)
            if plasma is not None and len(plasma):
                plasma = np.asarray(plasma, dtype=float)
                idx = int(np.nanargmax(plasma))
                metrics["plasma_cmax_pmol_ml"] = float(np.nanmax(plasma))
                metrics["plasma_tmax_h"] = float(time_h[idx]) if time_h.size > idx else float(idx)
                metrics["plasma_auc_pmol_h_ml"] = float(np.trapz(plasma, time_h))
            total_systemic = getattr(pk_data, "total_systemic_amounts", None)
            if total_systemic is not None and len(total_systemic):
                total_systemic = np.asarray(total_systemic, dtype=float)
                metrics["systemic_total_final_pmol"] = float(total_systemic[-1])

        metrics.update(_pbpk_region_metrics(comp))

    metadata = result.metadata or {}
    for key, value in metadata.items():
        numeric = _as_float(value)
        if numeric is not None:
            metrics[f"metadata_{key}"] = numeric

    return metrics


def summarise_pk(result: Optional[PKResult]) -> Dict[str, float]:
    if result is None:
        return {}

    metrics: Dict[str, float] = {}

    try:
        t = np.asarray(result.t, dtype=float)
        conc = np.asarray(result.conc_plasma, dtype=float)
    except Exception:
        return metrics

    if t.size == 0 or conc.size == 0:
        return metrics

    cmax_idx = int(np.nanargmax(conc))
    metrics["cmax_conc_plasma"] = float(np.nanmax(conc))
    metrics["tmax_h"] = float(t[cmax_idx]) if t.size > cmax_idx else float(cmax_idx)
    metrics["auc_0_last"] = float(np.trapz(conc, t))

    metadata = result.metadata or {}
    for key, value in metadata.items():
        numeric = _as_float(value)
        if numeric is not None:
            metrics[f"metadata_{key}"] = numeric

    return metrics


def build_stage_metrics(result: RunResult) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    stage_results_info = (result.metadata or {}).get("stage_results", {})

    if result.cfd is not None:
        cfd_metrics = summarise_cfd(result.cfd)
        if cfd_metrics:
            metrics["cfd"] = cfd_metrics

    if result.deposition is not None:
        deposition_metrics = summarise_deposition(result.deposition)
        if deposition_metrics:
            metrics["deposition"] = deposition_metrics

    if result.pbbk is not None:
        pbpk_metrics = summarise_pbpk(result.pbbk)
        if pbpk_metrics:
            metrics["pbbm"] = pbpk_metrics

    if result.pk is not None:
        pk_metrics = summarise_pk(result.pk)
        if pk_metrics:
            metrics["pk"] = pk_metrics
            for extra_stage in ("iv_pk", "gi_pk"):
                if extra_stage in stage_results_info:
                    metrics.setdefault(extra_stage, {}).update(pk_metrics)

    for raw_stage, stage_info in stage_results_info.items():
        stage_key = normalize_stage_name(raw_stage)
        if not stage_key:
            continue
        metadata = (stage_info or {}).get("metadata") if isinstance(stage_info, Mapping) else None
        if metadata:
            scalar_meta = _collect_numeric_scalars(metadata)
            if scalar_meta:
                stage_metrics = metrics.setdefault(stage_key, {})
                for meta_key, meta_value in scalar_meta.items():
                    stage_metrics[f"metadata_{meta_key}"] = meta_value

    stage_times = (result.metadata or {}).get("stage_times", {})
    for stage, runtime in stage_times.items():
        numeric = _as_float(runtime)
        if numeric is None:
            continue
        stage_key = normalize_stage_name(stage)
        if not stage_key:
            continue
        metrics.setdefault(stage_key, {})["runtime_seconds"] = numeric

    summary_metrics = calculate_summary_metrics_safe(result)
    if summary_metrics:
        metrics.setdefault("overall", {}).update(summary_metrics)

    return metrics


def determine_stage_order(result: RunResult, stage_metrics: Mapping[str, Mapping[str, float]]) -> List[str]:
    """Derive an ordered list of stages present in the run."""

    ordered: List[str] = []
    seen = set()

    def _append(stage: Optional[str]) -> None:
        if not stage:
            return
        if stage not in stage_metrics:
            return
        if stage in seen:
            return
        ordered.append(stage)
        seen.add(stage)

    stage_results = (result.metadata or {}).get("stage_results", {})
    for stage_name in stage_results.keys():
        _append(normalize_stage_name(stage_name))

    stage_times = (result.metadata or {}).get("stage_times", {})
    for stage_name in stage_times.keys():
        _append(normalize_stage_name(stage_name))

    canonical = ["cfd", "deposition", "pbbm", "pbpk", "iv_pk", "gi_pk", "pk", "vbe"]
    for stage_name in canonical:
        _append(stage_name)

    for stage_name in stage_metrics.keys():
        if stage_name == "overall":
            continue
        _append(stage_name)

    if "overall" in stage_metrics and "overall" not in seen:
        ordered.append("overall")

    return ordered


def calculate_summary_metrics_safe(result: RunResult) -> Dict[str, float]:
    """Wrapper around calculate_summary_metrics to guard against import cycles."""
    from ..app_api import calculate_summary_metrics  # Local import to avoid circular dependency

    metrics = calculate_summary_metrics(result)
    cleaned: Dict[str, float] = {}
    for key, value in metrics.items():
        numeric = _as_float(value)
        if numeric is not None:
            cleaned[key] = numeric
    return cleaned
