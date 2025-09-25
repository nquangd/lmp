"""Standalone IV PK staging for direct intravenous dosing."""

from __future__ import annotations

from typing import Any, Dict, Optional, Set, Callable

import numpy as np

from ...contracts.stage import Stage
from ...contracts.types import PKInput, PKResult
from ...data_structures import build_comprehensive_results_from_orchestrator
from .pbpk_orchestrator_numba import ModelComponentType, PBPKOrchestratorNumba


class IVPKStage(Stage[PKInput, PKResult]):
    """Standalone IV systemic PK stage leveraging the numba orchestrator."""

    name = "iv_pk"

    def __init__(self, model_type: str = "2c") -> None:
        if model_type not in {"1c", "2c", "3c"}:
            raise ValueError("model_type must be one of '1c', '2c', '3c'")
        self.model_type = model_type

    @property
    def provides(self) -> Set[str]:
        return {"pk"}

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
            components=[ModelComponentType.PK],
            pk_model_type=self.model_type,
        )

        initial_conditions = self._prepare_iv_initial_conditions(params, api)
        external_input_fn = self._prepare_iv_external_inputs(params, api)

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
            raise ValueError("IV PK orchestrator did not return PK data")

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

        metrics = self._calculate_pk_metrics(time_points, plasma_ng)
        dose_mg = params.get("iv_dose_mg", params.get("dose_mg", 0.0))
        dose_duration_h = params.get("iv_dose_duration_h", params.get("dose_duration_h", 0.0))
        metrics.update(
            {
                "dose_mg": float(dose_mg),
                "dose_duration_h": float(dose_duration_h),
                "dosing_type": "iv_bolus" if dose_duration_h == 0 else "iv_infusion",
            }
        )

        return PKResult(
            t=time_points,
            conc_plasma=plasma_ng,
            compartments=compartments,
            metadata={
                "model_type": self.model_type,
                "pk_metrics": metrics,
                "units": "ng/mL",
                "source": "iv_pk_stage",
            },
        )

    @staticmethod
    def _prepare_iv_initial_conditions(
        params: Dict[str, Any],
        api: Dict[str, Any],
    ) -> Dict[str, float]:
        initial_conditions: Dict[str, float] = {}
        dose_mg = params.get("iv_dose_mg", params.get("dose_mg", 0.0))
        dose_duration_h = params.get("iv_dose_duration_h", params.get("dose_duration_h", 0.0))

        if dose_duration_h == 0 and dose_mg > 0:
            mw = getattr(api, "molecular_weight", 500.0)
            dose_pmol = float(dose_mg) * 1e12 / float(mw)
            initial_conditions["pk_central"] = dose_pmol

        return initial_conditions

    @staticmethod
    def _prepare_iv_external_inputs(
        params: Dict[str, Any],
        api: Dict[str, Any],
    ) -> Optional[Callable[[float], Dict[str, float]]]:
        dose_mg = params.get("iv_dose_mg", params.get("dose_mg", 0.0))
        dose_duration_h = params.get("iv_dose_duration_h", params.get("dose_duration_h", 0.0))

        if dose_duration_h > 0 and dose_mg > 0:
            mw = getattr(api, "molecular_weight", 500.0)
            infusion_rate_mg_h = float(dose_mg) / float(dose_duration_h)
            infusion_rate_pmol_s = infusion_rate_mg_h * 1e12 / (float(mw) * 3600.0)

            duration_s = float(dose_duration_h * 3600.0)

            def iv_dose_rate(t_seconds: float) -> Dict[str, float]:
                rate = infusion_rate_pmol_s if t_seconds <= duration_s else 0.0
                return {"iv_dose_rate": rate}

            return iv_dose_rate

        return None

    @staticmethod
    def _calculate_pk_metrics(time_points: np.ndarray, plasma_conc: np.ndarray) -> Dict[str, float]:
        if plasma_conc.size == 0:
            return {"cmax_ng_mL": 0.0, "tmax_h": 0.0, "auc_ng_h_mL": 0.0}

        time_h = time_points / 3600.0
        cmax = float(np.nanmax(plasma_conc))
        idx = int(np.nanargmax(plasma_conc))
        tmax = float(time_h[idx]) if 0 <= idx < time_h.size else 0.0
        auc = float(np.trapz(plasma_conc, time_h))

        return {
            "cmax_ng_mL": cmax,
            "tmax_h": tmax,
            "auc_ng_h_mL": auc,
        }


class IVPKStage1C(IVPKStage):
    def __init__(self) -> None:
        super().__init__(model_type="1c")


class IVPKStage2C(IVPKStage):
    def __init__(self) -> None:
        super().__init__(model_type="2c")


class IVPKStage3C(IVPKStage):
    def __init__(self) -> None:
        super().__init__(model_type="3c")
