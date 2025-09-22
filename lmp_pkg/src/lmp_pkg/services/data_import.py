"""Data import helpers for observed datasets (in-vitro/in-vivo).

This module provides lightweight loaders for common tabular formats and
utilities to align observed data with model outputs for fitting and plots.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Sequence, Callable

import numpy as np
import pandas as pd

from ..contracts.types import RunResult


@dataclass
class ObservedPK:
    """Observed systemic PK dataset (single arm/series)."""

    time_s: np.ndarray
    concentration_ng_per_ml: np.ndarray
    label: str = "observed"
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def from_frame(
        cls,
        df: pd.DataFrame,
        time_col: str,
        conc_col: str,
        time_unit: str = "h",
        label: str = "observed",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ObservedPK":
        t = df[time_col].to_numpy(dtype=float)
        if time_unit.lower() in ("h", "hour", "hours"):
            time_s = t * 3600.0
        elif time_unit.lower() in ("min", "minute", "minutes"):
            time_s = t * 60.0
        else:  # assume seconds
            time_s = t
        conc = df[conc_col].to_numpy(dtype=float)
        return cls(time_s=time_s, concentration_ng_per_ml=conc, label=label, metadata=metadata)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        time_col: str,
        conc_col: str,
        time_unit: str = "h",
        label: str = "observed",
        **read_csv_kwargs,
    ) -> "ObservedPK":
        df = pd.read_csv(path, **read_csv_kwargs)
        return cls.from_frame(df, time_col, conc_col, time_unit=time_unit, label=label)

    def interpolate_predicted(self, time_s: np.ndarray, conc_ng_ml: np.ndarray) -> np.ndarray:
        """Interpolate predicted concentrations at the observed time grid."""
        return np.interp(self.time_s, time_s, conc_ng_ml)


def extract_predicted_pk(result: RunResult) -> tuple[np.ndarray, np.ndarray]:
    """Extract systemic PK (time_s, conc_ng_ml) from a RunResult."""
    if result.pbbk and result.pbbk.comprehensive is not None:
        comp = result.pbbk.comprehensive
        return comp.time_s, comp.pk_data.plasma_concentration_ng_per_ml
    if result.pk is not None:
        return result.pk.t, result.pk.conc_plasma
    raise ValueError("RunResult does not contain systemic PK output")


def make_pk_fitting_metric(dataset: ObservedPK, loss: str = "sse") -> Callable[[RunResult], float]:
    """Create a metric callable mapping RunResult -> scalar loss against observed PK.

    Args:
        dataset: Observed PK data
        loss: 'sse' (sum squared error), 'mae' (mean abs error), or 'rmse'
    """
    loss = loss.lower()

    def _metric(result: RunResult) -> float:
        t_pred, c_pred = extract_predicted_pk(result)
        c_on_obs = dataset.interpolate_predicted(t_pred, c_pred)
        res = dataset.concentration_ng_per_ml - c_on_obs
        if loss == "mae":
            return float(np.mean(np.abs(res)))
        if loss == "rmse":
            return float(np.sqrt(np.mean(res ** 2)))
        return float(np.sum(res ** 2))

    return _metric


def make_pk_overlay_data(result: RunResult, dataset: ObservedPK) -> pd.DataFrame:
    """Return a tidy DataFrame for plotting observed vs predicted PK."""
    t_pred, c_pred = extract_predicted_pk(result)
    df_pred = pd.DataFrame({
        'series': 'predicted',
        'time_h': t_pred / 3600.0,
        'concentration_ng_ml': c_pred,
    })
    df_obs = pd.DataFrame({
        'series': dataset.label,
        'time_h': dataset.time_s / 3600.0,
        'concentration_ng_ml': dataset.concentration_ng_per_ml,
    })
    return pd.concat([df_pred, df_obs], ignore_index=True)

