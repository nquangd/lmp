"""Workflow abstractions for preconfigured stage pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Sequence, Optional

from .pipeline import Pipeline
from .context import RunContext
from ..config.model import PopulationVariabilityConfig


@dataclass(frozen=True)
class Workflow:
    """Encapsulates a pipeline stage sequence and per-stage configuration."""

    name: str
    stages: Sequence[str]
    stage_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    population_variability_defaults: Optional[PopulationVariabilityConfig] = None

    def execute(
        self,
        entities: Dict[str, Any],
        context: RunContext,
        pipeline: Optional[Pipeline] = None,
        stage_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        pipe = pipeline or Pipeline()
        configs = dict(self.stage_configs)
        if stage_overrides:
            for stage, override in stage_overrides.items():
                merged = dict(configs.get(stage, {}))
                merged.update(override)
                configs[stage] = merged
        return pipe.run(
            stages=list(self.stages),
            entities=entities,
            stage_configs=configs,
            context=context,
        )


_BUILTIN_WORKFLOWS: Dict[str, Workflow] = {
    "deposition_pbbm_pk": Workflow(
        name="deposition_pbbm_pk",
        stages=["cfd", "deposition", "pbbm", "pk"],
        stage_configs={
            "cfd": {"model": "ml"},
            "deposition": {"model": "clean_lung"},
            "pbbm": {
                "model": "numba",
                "params": {"duration_h": 24.0, "n_time_points": 1441},
            },
            "pk": {"model": "pk_3c"},
        },
        population_variability_defaults=PopulationVariabilityConfig(
            demographic=False,
            lung_regional=False,
            lung_generation=False,
            gi=False,
            pk=False,
            inhalation=False,
        ),
    ),
    "deposition_pbbm": Workflow(
        name="deposition_pbbm",
        stages=["cfd", "deposition", "pbbm"],
        stage_configs={
            "cfd": {"model": "ml"},
            "deposition": {"model": "clean_lung"},
            "pbbm": {
                "model": "numba",
                "params": {"duration_h": 24.0, "n_time_points": 1441},
            },
        },
        population_variability_defaults=PopulationVariabilityConfig(
            demographic=False,
            lung_regional=False,
            lung_generation=False,
            gi=False,
            pk=False,
            inhalation=False,
        ),
    ),
    "pbbm_only": Workflow(
        name="pbbm_only",
        stages=["pbbm"],
        stage_configs={
            "pbbm": {
                "model": "numba",
                "params": {"duration_h": 24.0, "n_time_points": 1441},
            }
        },
        population_variability_defaults=PopulationVariabilityConfig(
            demographic=False,
            lung_regional=False,
            lung_generation=False,
            gi=False,
            pk=False,
            inhalation=False,
        ),
    ),
    "vbe": Workflow(
        name="vbe",
        stages=["cfd", "deposition", "pbbm", "pk"],
        stage_configs={
            "cfd": {"model": "ml"},
            "deposition": {"model": "clean_lung"},
            "pbbm": {
                "model": "numba",
                "params": {"duration_h": 24.0, "n_time_points": 1441},
            },
            "pk": {"model": "pk_3c"},
        },
        population_variability_defaults=PopulationVariabilityConfig(
            demographic=True,
            lung_regional=True,
            lung_generation=False,
            gi=False,
            pk=True,
            inhalation=True,
        ),
    ),
    "pe_pk_iv": Workflow(
        name="pe_pk_iv",
        stages=["pk"],
    ),
    "pe_gi_oral": Workflow(
        name="pe_gi_oral",
        stages=["gi", "pk"],
    ),
    "pe_full_pipeline": Workflow(
        name="pe_full_pipeline",
        stages=["cfd", "deposition", "pbbm", "gi", "pk"],
    ),
    "sa_full_pipeline": Workflow(
        name="sa_full_pipeline",
        stages=["cfd", "deposition", "pbbm", "gi", "pk"],
    ),
}


def list_workflows() -> Sequence[str]:
    return list(_BUILTIN_WORKFLOWS.keys())


def get_workflow(name: str) -> Workflow:
    if name not in _BUILTIN_WORKFLOWS:
        raise KeyError(f"Workflow '{name}' not found")
    return _BUILTIN_WORKFLOWS[name]


def register_workflow(workflow: Workflow) -> None:
    _BUILTIN_WORKFLOWS[workflow.name] = workflow
