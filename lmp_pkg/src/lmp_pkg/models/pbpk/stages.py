"""Pipeline stages wrapping the numba PBPK orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Set, Optional, Dict, Any
import numpy as np

from ...contracts.stage import Stage
from ...contracts.types import PBBKInput, PBBKResult
from ...models.pbpk.pbpk_orchestrator_numba import (
    PBPKOrchestratorNumba,
    ModelComponentType,
)
from ...data_structures import build_comprehensive_results_from_orchestrator


@dataclass
class NumbaPBPKConfig:
    """Configuration for the numba PBPK stage."""

    duration_h: float = 24.0
    n_time_points: int = 1441
    lung_model_type: str = "regional"
    pk_model_type: str = "3c"
    solve_dissolution: bool = True
    charcoal_block: bool = False
    suppress_et_absorption: bool = False
    solver_options: Optional[Dict[str, Any]] = None


class NumbaPBPKStage(Stage[PBBKInput, PBBKResult]):
    """Stage driving the PBPK orchestrator and returning comprehensive results."""

    name: str = "numba_pbpk"

    @property
    def provides(self) -> Set[str]:
        return {"pbbm", "pk"}

    @property
    def requires(self) -> Set[str]:
        return {"deposition"}

    def run(self, data: PBBKInput) -> PBBKResult:
        subject = data.subject
        api = data.api
        params = data.params or {}

        config = NumbaPBPKConfig(
            duration_h=params.get("duration_h", 24.0),
            n_time_points=params.get("n_time_points", 1441),
            lung_model_type=params.get("lung_model_type", "regional"),
            pk_model_type=params.get("pk_model_type", "3c"),
            solve_dissolution=params.get("solve_dissolution", True),
            charcoal_block=params.get("charcoal_block", False),
            suppress_et_absorption=params.get("suppress_et_absorption", False),
            solver_options=params.get("solver_options"),
        )

        orchestrator = PBPKOrchestratorNumba(
            subject_params=subject,
            api_params=api,
            components=[
                ModelComponentType.LUNG,
                ModelComponentType.GI,
                ModelComponentType.PK,
            ],
            lung_model_type=config.lung_model_type,
            pk_model_type=config.pk_model_type,
            solve_dissolution=config.solve_dissolution,
            charcoal_block=config.charcoal_block,
            suppress_et_absorption=config.suppress_et_absorption,
        )

        time_points = np.linspace(0.0, config.duration_h * 3600.0, config.n_time_points)

        initial_conditions: Dict[str, float] = {}
        if data.lung_seed is not None:
            regional_amounts = np.asarray(data.lung_seed.elf_initial_amounts)
            if regional_amounts.size >= 4:
                initial_conditions = {
                    'ET_deposition': float(regional_amounts[0]),
                    'BB_deposition': float(regional_amounts[1]),
                    'bb_deposition': float(regional_amounts[2]),
                    'Al_deposition': float(regional_amounts[3]),
                }

        solver_options = {
            'method': params.get('method', 'BDF'),
            'rtol': params.get('rtol', 1e-4),
            'atol': params.get('atol', 1e-8),
            'max_step': params.get('max_step', 3600.0),
        }
        if config.solver_options:
            solver_options.update(config.solver_options)

        solution = orchestrator.solve(
            time_points,
            initial_conditions=initial_conditions,
            **solver_options,
        )

        comprehensive = build_comprehensive_results_from_orchestrator(
            solution,
            api.molecular_weight,
        )

        return PBBKResult(
            t=comprehensive.time_s,
            y=None,
            region_slices={},
            pulmonary_outflow=comprehensive.pk_data.plasma_concentration_ng_per_ml,
            metadata={
                'solver_info': getattr(solution, 'model_info', {}),
            },
            comprehensive=comprehensive,
        )
