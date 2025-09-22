"""Integration tests exercising predefined workflows with real catalog data."""

from __future__ import annotations

from pathlib import Path

import pytest

from lmp_pkg.domain.entities import Product
from lmp_pkg.engine.pipeline import Pipeline
from lmp_pkg.engine.context import RunContext
from lmp_pkg.engine.workflow import list_workflows, get_workflow
from lmp_pkg.contracts.types import DepositionResult, PBBKResult
from lmp_pkg.simulation.subject_tasks import SubjectTask, prepare_entities


@pytest.fixture(scope="module")
def base_task() -> SubjectTask:
    """Deterministic task used for workflow smoke tests."""
    return SubjectTask(
        subject_index=0,
        api_name="BD",
        subject_name="healthy_reference",
        products=("reference_product",),
        seed=1234,
        apply_variability=False,
    )


def _build_entities(task: SubjectTask) -> dict:
    entities = prepare_entities(task)
    product = Product.from_builtin(task.products[0]).get_final_values(task.api_name)
    entities["product"] = product
    return entities


def test_builtin_workflows_listed():
    names = list_workflows()
    assert {"deposition_pbbm_pk", "deposition_pbbm", "pbbm_only", "vbe"}.issubset(set(names))


def test_deposition_only_execution(base_task):
    entities = _build_entities(base_task)
    pipeline = Pipeline()
    context = RunContext(
        run_id="test_deposition",
        seed=base_task.seed,
        threads=1,
        enable_numba=False,
        artifact_dir=None,
        logger=None,
    )

    results = pipeline.run(
        stages=["deposition"],
        entities=entities,
        stage_configs={"deposition": {"model": "clean_lung"}},
        context=context,
    )

    depo_result = results["deposition"].data
    assert isinstance(depo_result, DepositionResult)
    assert depo_result.region_ids.size == 4


def test_deposition_then_pbpk_execution(base_task):
    entities = _build_entities(base_task)
    pipeline = Pipeline()
    context = RunContext(
        run_id="test_depo_pbpk",
        seed=base_task.seed,
        threads=1,
        enable_numba=False,
        artifact_dir=None,
        logger=None,
    )

    stage_configs = {
        "deposition": {"model": "clean_lung"},
        "pbbm": {
            "model": "numba",
            "params": {
                "duration_h": 0.05,
                "n_time_points": 61,
                "solve_dissolution": False,
            },
        },
    }

    results = pipeline.run(
        stages=["deposition", "pbbm"],
        entities=entities,
        stage_configs=stage_configs,
        context=context,
    )

    pbpk_result = results["pbbm"].data
    assert isinstance(pbpk_result, PBBKResult)
    assert pbpk_result.comprehensive is not None
    assert pbpk_result.pulmonary_outflow is not None
