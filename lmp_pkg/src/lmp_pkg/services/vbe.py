from __future__ import annotations

from typing import Dict, Any, List, Tuple
import numpy as np
from scipy.stats import t

from ..data_structures import ComprehensivePBBMResults
from .vbe_helpers import MetricSpec, compute_metric


def _ratio_and_ci(test: np.ndarray, ref: np.ndarray, alpha: float = 0.10, non_parametric: bool = True,
                  n_bootstrap: int = 100) -> Tuple[float, float, float]:
    if len(test) == 0 or len(ref) == 0 or np.all(ref == 0):
        return np.nan, np.nan, np.nan

    logdiff = np.log(test) - np.log(ref)
    if non_parametric:
        gmr = float(np.exp(np.mean(logdiff)))
        n = len(logdiff)
        boot = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(np.arange(n), size=n, replace=True)
            boot.append(float(np.exp(np.mean(logdiff[idx]))))
        boot = np.array(boot, dtype=float)
        delta = boot - gmr
        lo_delta = np.nanpercentile(delta, 100 * alpha / 2)
        hi_delta = np.nanpercentile(delta, 100 * (1 - alpha / 2))
        return gmr, gmr - hi_delta, gmr - lo_delta
    else:
        n = len(logdiff)
        mean_ld = float(np.mean(logdiff))
        std_ld = float(np.std(logdiff, ddof=1))
        se = std_ld / np.sqrt(n)
        t_crit = float(t.ppf(1 - alpha / 2, n - 1))
        ci_lo = mean_ld - t_crit * se
        ci_hi = mean_ld + t_crit * se
        return float(np.exp(mean_ld)), float(np.exp(ci_lo)), float(np.exp(ci_hi))


def vbe_from_results(
    reference: Dict[str, ComprehensivePBBMResults],
    test: Dict[str, ComprehensivePBBMResults],
    metric: MetricSpec,
    alpha: float = 0.10,
    non_parametric: bool = True,
    n_bootstrap: int = 100
) -> Dict[str, Any]:
    """Compute VBE for a given metric using paired subject results.

    Returns a dict with per-subject values (ref/test/ratio), GMR, CI90, and pass flag (80–125%).
    """
    subjects = sorted(set(reference.keys()) & set(test.keys()))
    pairs: List[Dict[str, Any]] = []
    ref_vals: List[float] = []
    tst_vals: List[float] = []
    for sid in subjects:
        rv = compute_metric(reference[sid], metric)
        tv = compute_metric(test[sid], metric)
        ratio = (tv / rv) if (rv is not None and rv > 0) else np.nan
        pairs.append({'subject': sid, 'reference': rv, 'test': tv, 'ratio': ratio})
        ref_vals.append(rv)
        tst_vals.append(tv)

    ref_arr = np.array(ref_vals, dtype=float)
    tst_arr = np.array(tst_vals, dtype=float)
    mask = np.isfinite(ref_arr) & np.isfinite(tst_arr) & (ref_arr > 0) & (tst_arr > 0)
    ref_arr = ref_arr[mask]
    tst_arr = tst_arr[mask]

    gmr, lo, hi = _ratio_and_ci(tst_arr, ref_arr, alpha=alpha, non_parametric=non_parametric, n_bootstrap=n_bootstrap)
    return {
        'subjects': pairs,
        'gmr': gmr,
        'ci_lower': lo,
        'ci_upper': hi,
        'pass_80125': (not np.isnan(gmr)) and (lo >= 0.8) and (hi <= 1.25),
        'metric': metric,
    }

def _metric_id(spec: MetricSpec) -> str:
    parts: List[str] = []
    parts.append(str(spec.level))
    parts.append(str(spec.metric))
    if spec.region:
        parts.append(str(spec.region))
    if spec.compartment:
        parts.append(str(spec.compartment))
    if spec.unbound:
        parts.append('unbound')
    if spec.window_h:
        parts.append(f"win{spec.window_h}")
    if spec.time_h is not None:
        parts.append(f"t{spec.time_h}")
    if spec.metric.lower() == 'deposition':
        parts.append(f"{spec.amount_unit}")
    if spec.metric.lower() == 'efficacy' and spec.efficacy_model:
        parts.append(str(spec.efficacy_model))
    return ".".join(parts)

def vbe_multi_from_results(
    reference: Dict[str, ComprehensivePBBMResults],
    test: Dict[str, ComprehensivePBBMResults],
    metrics: List[MetricSpec],
    alpha: float = 0.10,
    non_parametric: bool = True,
    n_bootstrap: int = 100
) -> Dict[str, Any]:
    """Compute VBE for multiple metrics in one call.

    Returns a dict keyed by a human-readable metric id, each value containing
    the standard VBE result for that metric (same shape as vbe_from_results).
    """
    results: Dict[str, Any] = {}
    for spec in metrics:
        key = _metric_id(spec)
        results[key] = vbe_from_results(
            reference, test, spec,
            alpha=alpha, non_parametric=non_parametric, n_bootstrap=n_bootstrap
        )
    return results
