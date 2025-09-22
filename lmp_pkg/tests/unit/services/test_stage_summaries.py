"""Tests for stage-level summarisation helpers."""

from __future__ import annotations

import numpy as np
import pytest

from lmp_pkg.contracts.types import (
    CFDResult,
    DepositionResult,
    PBBKResult,
    PKResult,
    RunResult,
)
from lmp_pkg.data_structures.comprehensive_results import (
    ComprehensivePBBMResults,
    FluxData,
    MassBalanceData,
    PKResultsData,
    RegionalAmountData,
)
from lmp_pkg.services.stage_summaries import build_stage_metrics, determine_stage_order


def _build_comprehensive_result(time_s: np.ndarray) -> ComprehensivePBBMResults:
    time_h = time_s / 3600.0

    region = RegionalAmountData(
        region_name="Al",
        time_s=time_s,
        time_h=time_h,
        epithelium_amounts=np.array([1.0, 2.0, 3.0]),
        tissue_amounts=np.array([0.5, 1.0, 1.5]),
        elf_amounts=np.array([0.2, 0.3, 0.4]),
        epithelium_volume_ml=1.0,
        tissue_volume_ml=1.0,
    )

    pk_data = PKResultsData(
        time_s=time_s,
        time_h=time_h,
        plasma_concentration=np.array([0.0, 1.0, 0.5]),
        central_amounts=np.array([0.0, 1.0, 1.5]),
    )

    flux_data = FluxData(
        time_s=time_s,
        time_h=time_h,
        systemic_absorption_rate=np.array([0.0, 0.1, 0.0]),
        mucociliary_clearance_rate=np.array([0.0, 0.05, 0.0]),
    )

    mass_balance = MassBalanceData(
        time_s=time_s,
        time_h=time_h,
        initial_deposition_pmol=5.0,
        lung_amounts=np.array([0.5, 1.5, 2.5]),
        systemic_amounts=np.array([0.1, 0.6, 1.0]),
        cumulative_elimination=np.array([0.0, 0.2, 0.4]),
    )

    return ComprehensivePBBMResults(
        time_s=time_s,
        regional_data={"Al": region},
        pk_data=pk_data,
        flux_data=flux_data,
        mass_balance=mass_balance,
        metadata={"solver": "ok"},
    )


def _build_run_result() -> RunResult:
    time_s = np.array([0.0, 3600.0, 7200.0])
    time_h = time_s / 3600.0

    cfd = CFDResult(
        mmad=1.5,
        gsd=2.1,
        mt_deposition_fraction=0.55,
        metadata={"loss": 0.1},
    )

    deposition = DepositionResult(
        region_ids=np.array([0, 1]),
        elf_initial_amounts=np.array([10.0, 5.0]),
        metadata={
            "regional_amounts_pmol": [10.0, 5.0],
            "regional_names": ["ET", "Al"],
            "mass_balance": 0.99,
        },
    )

    comprehensive = _build_comprehensive_result(time_s)
    pbbk = PBBKResult(
        t=time_h,
        y=np.zeros((3, 1)),
        metadata={"solver_steps": 42},
        comprehensive=comprehensive,
    )

    pk = PKResult(
        t=time_h,
        conc_plasma=np.array([0.0, 1.0, 0.5]),
        compartments={"central": np.array([0.0, 1.0, 1.5])},
        metadata={"objective": 0.2},
    )

    stage_metadata = {
        "cfd": {"model": "ml", "metadata": {"loss": 0.1}},
        "deposition": {"model": "clean", "metadata": {"mass_balance": 0.99}},
        "pbbm": {"model": "numba", "metadata": {"mass_error": 0.02}},
        "pk": {"model": "pk_3c", "metadata": {"solved": True}},
        "analysis:bioequivalence": {
            "model": "vbe",
            "metadata": {
                "gmr": 1.05,
                "ci": {"lower": 0.92, "upper": 1.18},
                "pass_80125": True,
            },
        },
    }

    return RunResult(
        run_id="run-1",
        config={},
        cfd=cfd,
        deposition=deposition,
        pbbk=pbbk,
        pk=pk,
        runtime_seconds=10.0,
        metadata={
            "status": "completed",
            "stage_times": {
                "cfd": 1.0,
                "deposition": 2.0,
                "pbbm": 3.0,
                "pk": 4.0,
                "analysis:bioequivalence": 5.0,
            },
            "stage_results": stage_metadata,
        },
    )


def test_build_stage_metrics_includes_all_stages():
    result = _build_run_result()

    stage_metrics = build_stage_metrics(result)

    assert stage_metrics["cfd"]["mmad"] == 1.5
    assert stage_metrics["cfd"]["metadata_loss"] == 0.1
    assert stage_metrics["cfd"]["runtime_seconds"] == 1.0

    assert "deposition" in stage_metrics
    assert stage_metrics["deposition"]["metadata_mass_balance"] == 0.99

    pbpk_metrics = stage_metrics["pbbm"]
    assert pbpk_metrics["lung_total_final_pmol"] == pytest.approx(4.9)
    assert "region_Al_auc_epithelium" in pbpk_metrics

    assert stage_metrics["pk"]["auc_0_last"] > 0

    vbe_metrics = stage_metrics["vbe"]
    assert vbe_metrics["metadata_gmr"] == 1.05
    assert vbe_metrics["metadata_ci_lower"] == 0.92
    assert vbe_metrics["metadata_pass_80125"] == 1.0

    assert "overall" in stage_metrics
    assert "pk_cmax" in stage_metrics["overall"]


def test_determine_stage_order_respects_pipeline_sequence():
    result = _build_run_result()
    stage_metrics = build_stage_metrics(result)

    order = determine_stage_order(result, stage_metrics)

    assert order == ["cfd", "deposition", "pbbm", "pk", "vbe", "overall"]
