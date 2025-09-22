from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Callable, Any
import numpy as np

from ..data_structures import ComprehensivePBBMResults


@dataclass
class MetricSpec:
    """Specification of which metric to extract for VBE.

    level:
      - 'Systemic': systemic PK (plasma)
      - 'Regional': region-wide (Epithelium, Tissue, or Combined)
      - 'Compartmental': synonym for Regional with explicit compartment target

    metric:
      - 'AUC', 'Cmax', 'Tmax', 'ConcAt'

    region: required for Regional/Compartmental (e.g., 'Al', 'BB', 'bb', 'ET')
    compartment: 'Epithelium' | 'Tissue' | 'Epithelium_Tissue' (Combined)
    unbound: if True, use unbound concentrations
    window_h: optional (t0_h, t1_h) for AUC; if None, use full profile
    time_h: for ConcAt (concentration at time)
    systemic_unit: 'pmol/mL' or 'ng/mL' for systemic concentration metrics
    """

    level: str
    metric: str
    region: Optional[str] = None
    compartment: Optional[str] = None
    unbound: bool = False
    window_h: Optional[Tuple[float, float]] = None
    time_h: Optional[float] = None
    systemic_unit: str = 'ng/mL'
    # Amount unit for deposition metrics: 'pmol', 'ng', or 'ug'
    amount_unit: str = 'ng'
    # Optional efficacy model selection and parameters
    efficacy_model: Optional[str] = None
    efficacy_params: Optional[Dict[str, Any]] = None


def _auc(y: np.ndarray, t_h: np.ndarray, window: Optional[Tuple[float, float]] = None) -> float:
    if y is None or t_h is None or len(y) == 0:
        return float('nan')
    if window is None:
        return float(np.trapz(y, t_h))
    t0, t1 = window
    # Clip to window via interpolation
    if t0 <= t_h[0] and t1 >= t_h[-1]:
        return float(np.trapz(y, t_h))
    # Build a masked/interpolated series on the window
    t_win = t_h[(t_h >= t0) & (t_h <= t1)]
    if t_win.size == 0:
        # Single point window: interpolate
        y0 = np.interp([t0, t1], t_h, y)
        return float(np.trapz(y0, [t0, t1]))
    # Ensure endpoints included
    tb = np.concatenate(([t0], t_win, [t1]))
    yb = np.interp(tb, t_h, y)
    return float(np.trapz(yb, tb))


def _conc_at(y: np.ndarray, t_h: np.ndarray, time_h: float) -> float:
    if y is None or t_h is None or len(y) == 0:
        return float('nan')
    return float(np.interp(time_h, t_h, y))


def _systemic_series(comp: ComprehensivePBBMResults, unit: str) -> np.ndarray:
    pk = comp.pk_data
    if unit == 'ng/mL' and getattr(pk, 'plasma_concentration_ng_per_ml', None) is not None:
        return pk.plasma_concentration_ng_per_ml
    return pk.plasma_concentration


def compute_metric(comp: ComprehensivePBBMResults, spec: MetricSpec) -> float:
    """Compute a scalar metric from ComprehensivePBBMResults for VBE.

    Returns a scalar suitable for per-subject ratio analysis (test/ref).
    """
    t_h = comp.time_s / 3600.0

    if spec.level.lower() == 'systemic':
        series = _systemic_series(comp, spec.systemic_unit)
        if series is None:
            return float('nan')
        if spec.metric.lower() == 'auc':
            return _auc(series, t_h, spec.window_h)
        if spec.metric.lower() == 'cmax':
            return float(np.nanmax(series))
        if spec.metric.lower() == 'tmax':
            idx = int(np.nanargmax(series))
            return float(t_h[idx])
        if spec.metric.lower() == 'concat':
            if spec.time_h is None:
                return float('nan')
            return _conc_at(series, t_h, spec.time_h)
        return float('nan')

    # Regional/Compartmental
    region = spec.region
    comp_name = (spec.compartment or 'Epithelium_Tissue')
    if not region or region not in comp.regional_data:
        return float('nan')
    r = comp.regional_data[region]

    # Special-case: regional deposition mass (initial)
    if spec.metric.lower() == 'deposition':
        initial_pmol = float(r.total_amounts[0]) if hasattr(r, 'total_amounts') and len(r.total_amounts) > 0 else float('nan')
        mw = float(comp.pk_data.molecular_weight) if comp.pk_data.molecular_weight else 250.0
        if spec.amount_unit.lower() == 'pmol':
            return initial_pmol
        if spec.amount_unit.lower() == 'ug':
            return initial_pmol * mw * 1e-6
        # default to ng
        return initial_pmol * mw * 1e-3

    # Select series by compartment and unbound flag
    def pick_series() -> Optional[np.ndarray]:
        if comp_name.lower() == 'epithelium':
            if spec.unbound and getattr(r, 'epithelium_unbound_concentration_pmol_per_ml', None) is not None:
                return r.epithelium_unbound_concentration_pmol_per_ml
            return r.epithelium_concentration_pmol_per_ml
        if comp_name.lower() == 'tissue':
            if spec.unbound and getattr(r, 'tissue_unbound_concentration_pmol_per_ml', None) is not None:
                return r.tissue_unbound_concentration_pmol_per_ml
            return r.tissue_concentration_pmol_per_ml
        # Combined
        if spec.unbound and getattr(r, 'epithelium_tissue_unbound_concentration_pmol_per_ml', None) is not None:
            return r.epithelium_tissue_unbound_concentration_pmol_per_ml
        return r.epithelium_tissue_concentration_pmol_per_ml

    series = pick_series()
    if series is None:
        return float('nan')

    if spec.metric.lower() == 'auc':
        return _auc(series, t_h, spec.window_h)
    if spec.metric.lower() == 'cmax':
        return float(np.nanmax(series))
    if spec.metric.lower() == 'tmax':
        idx = int(np.nanargmax(series))
        return float(t_h[idx])
    if spec.metric.lower() == 'concat':
        if spec.time_h is None:
            return float('nan')
        return _conc_at(series, t_h, spec.time_h)
    # Efficacy metric hook (scalar outcome)
    if spec.metric.lower() == 'efficacy':
        try:
            model_name = (spec.efficacy_model or 'fev1_copd_catalog').lower()
            if model_name == 'fev1_copd_catalog':
                from lmp_pkg.models.efficacy.fev1_model_refactored import CatalogFEV1EfficacyModel
                model = CatalogFEV1EfficacyModel()
                eff = model.predict_efficacy(exposure_data={}, patient_data=spec.efficacy_params or {})
                return float(eff.get('mean_effect_mL', float('nan')))
            return float('nan')
        except Exception:
            return float('nan')
    return float('nan')
