"""Integration tests for the standalone IV and GI PK stages."""

from __future__ import annotations

import numpy as np
import pytest

from lmp_pkg.app_api import run_single_simulation
from lmp_pkg.config.model import AppConfig, EntityRef
from lmp_pkg.engine.workflow import get_workflow


@pytest.mark.integration
def test_iv_pk_bolus_stage_runs():
    cfg = AppConfig(
        subject=EntityRef(ref="healthy_reference"),
        api=EntityRef(ref="BD"),
        product=EntityRef(ref="reference_product"),
        maneuver=EntityRef(ref="pMDI_variable_trapezoid"),
    )

    params = {
        "iv_dose_mg": 100.0,
        "iv_dose_duration_h": 0.0,
        "duration_h": 24.0,
        "n_time_points": 241,
    }

    result = run_single_simulation(
        cfg,
        parameter_overrides={"iv_pk.params": params},
        workflow=get_workflow("pe_pk_iv"),
    )

    assert result.pk is not None
    assert result.deposition is None
    assert result.pbbk is None

    times_h = result.pk.t / 3600.0
    conc = result.pk.conc_plasma
    assert conc.size > 0
    assert times_h.size == conc.size
    cmax_idx = int(np.argmax(conc))
    assert times_h[cmax_idx] < 1.0

    metrics = (result.pk.metadata or {}).get("pk_metrics", {})
    assert metrics.get("dosing_type") == "iv_bolus"
    assert metrics.get("dose_mg") == 100.0


@pytest.mark.integration
def test_iv_pk_infusion_stage_runs():
    cfg = AppConfig(
        subject=EntityRef(ref="healthy_reference"),
        api=EntityRef(ref="BD"),
        product=EntityRef(ref="reference_product"),
        maneuver=EntityRef(ref="pMDI_variable_trapezoid"),
    )

    params = {
        "iv_dose_mg": 100.0,
        "iv_dose_duration_h": 2.0,
        "duration_h": 24.0,
        "n_time_points": 241,
    }

    result = run_single_simulation(
        cfg,
        parameter_overrides={"iv_pk.params": params},
        workflow=get_workflow("pe_pk_iv"),
    )

    assert result.pk is not None
    conc = result.pk.conc_plasma
    times_h = result.pk.t / 3600.0
    cmax_idx = int(np.argmax(conc))
    assert 1.5 <= times_h[cmax_idx] <= 3.0

    metrics = (result.pk.metadata or {}).get("pk_metrics", {})
    assert metrics.get("dosing_type") == "iv_infusion"


@pytest.mark.integration
def test_iv_pk_defaults_from_product_template():
    cfg = AppConfig(
        subject=EntityRef(ref="healthy_reference"),
        api=EntityRef(ref="BD"),
        product=EntityRef(ref="Default_IV_Template"),
        maneuver=EntityRef(ref="pMDI_variable_trapezoid"),
    )

    result = run_single_simulation(cfg, workflow=get_workflow("pe_pk_iv"))

    assert result.pk is not None
    metrics = (result.pk.metadata or {}).get("pk_metrics", {})
    assert metrics.get("dose_mg") == pytest.approx(0.32904, rel=1e-3)
    assert metrics.get("dosing_type") == "iv_bolus"


@pytest.mark.integration
def test_gi_pk_oral_stage_runs():
    cfg = AppConfig(
        subject=EntityRef(ref="healthy_reference"),
        api=EntityRef(ref="BD"),
        product=EntityRef(ref="reference_product"),
        maneuver=EntityRef(ref="pMDI_variable_trapezoid"),
    )

    params = {
        "oral_dose_mg": 50.0,
        "formulation": "immediate_release",
        "duration_h": 24.0,
        "n_time_points": 241,
    }

    result = run_single_simulation(
        cfg,
        parameter_overrides={"gi_pk.params": params},
        workflow=get_workflow("pe_gi_oral"),
    )

    assert result.pk is not None
    conc = result.pk.conc_plasma
    times_h = result.pk.t / 3600.0
    assert times_h[int(np.argmax(conc))] > 0.5

    compartments = result.pk.compartments
    assert "gi_stomach" in compartments
    assert "gi_duodenum" in compartments

    metrics = (result.pk.metadata or {}).get("pk_metrics", {})
    assert metrics.get("dosing_type") == "oral"
    assert metrics.get("formulation") == "immediate_release"


@pytest.mark.integration
def test_gi_pk_enteric_coated_formulation():
    cfg = AppConfig(
        subject=EntityRef(ref="healthy_reference"),
        api=EntityRef(ref="BD"),
        product=EntityRef(ref="reference_product"),
        maneuver=EntityRef(ref="pMDI_variable_trapezoid"),
    )

    params = {
        "oral_dose_mg": 50.0,
        "formulation": "enteric_coated",
        "duration_h": 24.0,
        "n_time_points": 241,
    }

    result = run_single_simulation(
        cfg,
        parameter_overrides={"gi_pk.params": params},
        workflow=get_workflow("pe_gi_oral"),
    )

    metrics = (result.pk.metadata or {}).get("pk_metrics", {})
    assert metrics.get("formulation") == "enteric_coated"


@pytest.mark.integration
def test_gi_pk_defaults_from_product_template():
    cfg = AppConfig(
        subject=EntityRef(ref="healthy_reference"),
        api=EntityRef(ref="BD"),
        product=EntityRef(ref="Default_Oral_Template"),
        maneuver=EntityRef(ref="pMDI_variable_trapezoid"),
    )

    result = run_single_simulation(cfg, workflow=get_workflow("pe_gi_oral"))

    assert result.pk is not None
    metrics = (result.pk.metadata or {}).get("pk_metrics", {})
    assert metrics.get("dose_mg") == pytest.approx(0.32904, rel=1e-3)
    assert metrics.get("formulation") == "immediate_release"
