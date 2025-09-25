"""Standalone GI + systemic PK staging for oral dosing."""

from __future__ import annotations

from typing import Any, Dict, Optional, Set, Callable

import numpy as np

from ...contracts.stage import Stage
from ...contracts.types import PKInput, PKResult
from ...data_structures import build_comprehensive_results_from_orchestrator
from .pbpk_orchestrator_numba import ModelComponentType, PBPKOrchestratorNumba


class GIPKStage(Stage[PKInput, PKResult]):
    """Combined GI + systemic PK stage leveraging the numba orchestrator."""

    name = "gi_pk"

    def __init__(self, pk_model_type: str = "2c", gi_model_type: str = "default") -> None:
        if pk_model_type not in {"1c", "2c", "3c"}:
            raise ValueError("pk_model_type must be one of '1c', '2c', '3c'")
        self.pk_model_type = pk_model_type
        self.gi_model_type = gi_model_type

    @property
    def provides(self) -> Set[str]:
        return {"pk", "gi"}

    @property
    def requires(self) -> Set[str]:
        return set()

    def run(self, data: PKInput) -> PKResult:
        params = dict(data.params or {})
        subject = data.subject
        api = data.api

        duration_h = params.get("duration_h", 24.0)
        n_points = params.get("n_time_points", 1441)
        time_points = np.linspace(0.0, duration_h * 3600.0, int(n_points))

        orchestrator = PBPKOrchestratorNumba(
            subject_params=subject,
            api_params=api,
            components=[ModelComponentType.GI, ModelComponentType.PK],
            pk_model_type=self.pk_model_type,
        )

        initial_conditions = self._prepare_oral_initial_conditions(params, api)
        external_input_fn = self._prepare_oral_external_inputs(params, api)

        solver_options = {
            "method": params.get("method", "BDF"),
            "rtol": params.get("rtol", 1e-6),
            "atol": params.get("atol", 1e-9),
            "max_step": params.get("max_step", 3600.0),
        }

        solve_kwargs: Dict[str, Any] = dict(solver_options)
        solve_kwargs["initial_conditions"] = initial_conditions
        if external_input_fn is not None:
            solve_kwargs["external_input_fn"] = external_input_fn
        solution = orchestrator.solve(time_points, **solve_kwargs)

        mw = getattr(api, "molecular_weight", 500.0)
        comprehensive = build_comprehensive_results_from_orchestrator(solution, mw)

        pk_data = comprehensive.pk_data
        if pk_data is None:
            raise ValueError("GI PK orchestrator did not return PK data")

        plasma_ng = getattr(pk_data, "plasma_concentration_ng_per_ml", None)
        if plasma_ng is None:
            plasma_pmol = np.asarray(getattr(pk_data, "plasma_concentration", np.zeros_like(time_points)), dtype=float)
            plasma_ng = plasma_pmol * (mw / 1000.0)
        else:
            plasma_ng = np.asarray(plasma_ng, dtype=float)

        compartments: Dict[str, np.ndarray] = {
            "central": np.asarray(pk_data.central_amounts, dtype=float),
        }
        peripheral = getattr(pk_data, "peripheral_amounts", None)
        if isinstance(peripheral, dict):
            for name, values in peripheral.items():
                compartments[str(name)] = np.asarray(values, dtype=float)

        gi_data = getattr(comprehensive, "gi_data", None)
        if gi_data is not None:
            compartments["gi_stomach"] = np.asarray(getattr(gi_data, "stomach_amounts", np.zeros_like(time_points)), dtype=float)
            compartments["gi_duodenum"] = np.asarray(getattr(gi_data, "duodenum_amounts", np.zeros_like(time_points)), dtype=float)
            compartments["gi_jejunum"] = np.asarray(getattr(gi_data, "jejunum_amounts", np.zeros_like(time_points)), dtype=float)
            compartments["gi_ileum"] = np.asarray(getattr(gi_data, "ileum_amounts", np.zeros_like(time_points)), dtype=float)
            compartments["gi_colon"] = np.asarray(getattr(gi_data, "colon_amounts", np.zeros_like(time_points)), dtype=float)
        else:
            # Fallback: assume first five rows of state vector are GI compartments.
            if solution.y.shape[0] >= 5:
                compartments["gi_stomach"] = solution.y[0, :]
                compartments["gi_duodenum"] = solution.y[1, :]
                compartments["gi_jejunum"] = solution.y[2, :]
                compartments["gi_ileum"] = solution.y[3, :]
                compartments["gi_colon"] = solution.y[4, :]

        metrics = self._calculate_pk_metrics(time_points, plasma_ng)
        dose_mg = params.get("oral_dose_mg", params.get("dose_mg", 0.0))
        formulation = params.get("formulation", "immediate_release")
        metrics.update(
            {
                "dose_mg": float(dose_mg),
                "formulation": formulation,
                "dosing_type": "oral",
                "bioavailability": self._estimate_bioavailability(comprehensive, float(dose_mg), float(mw)),
            }
        )

        return PKResult(
            t=time_points,
            conc_plasma=plasma_ng,
            compartments=compartments,
            metadata={
                "pk_model_type": self.pk_model_type,
                "gi_model_type": self.gi_model_type,
                "pk_metrics": metrics,
                "units": "ng/mL",
                "source": "gi_pk_stage",
            },
        )

    @staticmethod
    def _prepare_oral_initial_conditions(
        params: Dict[str, Any],
        api: Dict[str, Any],
    ) -> Dict[str, float]:
        initial_conditions: Dict[str, float] = {}
        dose_mg = params.get("oral_dose_mg", params.get("dose_mg", 0.0))
        if dose_mg <= 0:
            return initial_conditions

        mw = getattr(api, "molecular_weight", 500.0)
        dose_pmol = float(dose_mg) * 1e12 / float(mw)
        formulation = params.get("formulation", "immediate_release")

        if formulation == "enteric_coated":
            initial_conditions["gi_1"] = dose_pmol
        else:
            initial_conditions["gi_0"] = dose_pmol

        return initial_conditions

    @staticmethod
    def _prepare_oral_external_inputs(
        params: Dict[str, Any],
        api: Dict[str, Any],
    ) -> Optional[Callable[[float], Dict[str, float]]]:
        formulation = params.get("formulation", "immediate_release")
        if formulation != "extended_release":
            return None

        dose_mg = params.get("oral_dose_mg", params.get("dose_mg", 0.0))
        release_duration_h = params.get("release_duration_h", 12.0)
        if dose_mg <= 0 or release_duration_h <= 0:
            return None

        mw = getattr(api, "molecular_weight", 500.0)
        release_rate_mg_h = float(dose_mg) / float(release_duration_h)
        release_rate_pmol_s = release_rate_mg_h * 1e12 / (float(mw) * 3600.0)

        duration_s = float(release_duration_h * 3600.0)

        def oral_release_rate(t_seconds: float) -> Dict[str, float]:
            rate = release_rate_pmol_s if t_seconds <= duration_s else 0.0
            return {"oral_dose_rate": rate}

        return oral_release_rate

    @staticmethod
    def _calculate_pk_metrics(time_points: np.ndarray, plasma_conc: np.ndarray) -> Dict[str, float]:
        if plasma_conc.size == 0:
            return {
                "cmax_ng_mL": 0.0,
                "tmax_h": 0.0,
                "auc_ng_h_mL": 0.0,
                "tlag_h": np.nan,
            }

        time_h = time_points / 3600.0
        cmax = float(np.nanmax(plasma_conc))
        idx = int(np.nanargmax(plasma_conc))
        tmax = float(time_h[idx]) if 0 <= idx < time_h.size else 0.0
        auc = float(np.trapz(plasma_conc, time_h))

        loq = 0.1
        above_loq = plasma_conc > loq
        tlag = float(time_h[int(np.argmax(above_loq))]) if np.any(above_loq) else np.nan

        return {
            "cmax_ng_mL": cmax,
            "tmax_h": tmax,
            "auc_ng_h_mL": auc,
            "tlag_h": tlag,
        }

    @staticmethod
    def _estimate_bioavailability(comprehensive, dose_mg: float, mw: float) -> float:
        if dose_mg <= 0:
            return 0.0

        pk_data = getattr(comprehensive, "pk_data", None)
        time_h = getattr(comprehensive, "time_h", None)
        if pk_data is None or time_h is None:
            return 0.0

        plasma = np.asarray(getattr(pk_data, "plasma_concentration", np.zeros(1)), dtype=float)
        time_h = np.asarray(time_h, dtype=float)
        if plasma.size == 0 or time_h.size == 0:
            return 0.0

        auc_pmol_h_per_ml = float(np.trapz(plasma, time_h))
        if auc_pmol_h_per_ml <= 0:
            return 0.0

        reference = dose_mg * 1e12 / mw
        return float(min(1.0, auc_pmol_h_per_ml / reference * 100.0))


class GIPKStage1C(GIPKStage):
    def __init__(self) -> None:
        super().__init__(pk_model_type="1c")


class GIPKStage2C(GIPKStage):
    def __init__(self) -> None:
        super().__init__(pk_model_type="2c")


class GIPKStage3C(GIPKStage):
    def __init__(self) -> None:
        super().__init__(pk_model_type="3c")
