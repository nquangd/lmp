import numpy as np
import pytest

from lmp_pkg.config.model import AppConfig, EntityRef, RunConfig
from lmp_pkg.app_api import run_single_simulation
from lmp_pkg.solver.optimization import ParameterFitter, ParameterDefinition


@pytest.mark.integration
def test_pe_pk_iv_runs_pk_stage_only():
    # Configure workflow to pk-only so CFD/Deposition/PBBM do not run
    cfg = AppConfig(
        run=RunConfig(workflow_name="pe_pk_iv"),
        subject=EntityRef(ref="healthy_reference"),
        api=EntityRef(ref="BD"),
        product=EntityRef(ref="reference_product"),
        maneuver=EntityRef(ref="pMDI_variable_trapezoid"),
        pk={"model": "pk_3c"},
    )

    result = run_single_simulation(cfg)

    # Only PK should be present
    assert result.pk is not None, "PK stage should produce output"
    assert result.deposition is None, "Deposition stage must not run in pe_pk_iv"
    assert result.pbbk is None, "PBBM stage must not run in pe_pk_iv"

    stage_results = result.metadata.get("stage_results", {})
    assert set(stage_results.keys()) == {"pk"}


@pytest.mark.integration
def test_parameter_fitter_pk_only_workflow():
    # Base config with pk-only workflow so fitter uses it internally
    cfg = AppConfig(
        run=RunConfig(workflow_name="pe_pk_iv"),
        subject=EntityRef(ref="healthy_reference"),
        api=EntityRef(ref="BD"),
        product=EntityRef(ref="reference_product"),
        maneuver=EntityRef(ref="pMDI_variable_trapezoid"),
        pk={"model": "pk_3c"},
    )

    # Synthetic observed data from a baseline run (pk-only)
    base = run_single_simulation(cfg)
    assert base.pk is not None
    t_obs = np.asarray(base.pk.t)
    c_obs = np.asarray(base.pk.conc_plasma)

    # Add small noise
    rng = np.random.default_rng(123)
    c_noisy = c_obs + rng.normal(0.0, 0.05 * (np.max(c_obs) if np.max(c_obs) > 0 else 1.0), size=c_obs.shape)

    params = [
        ParameterDefinition("volume_central_L", "pk.params.volume_central_L", (10.0, 60.0)),
        ParameterDefinition("clearance_L_h", "pk.params.clearance_L_h", (10.0, 200.0)),
    ]

    fitter = ParameterFitter(
        base_config=cfg,
        parameters=params,
        observed_time_s=t_obs,
        observed_concentration=c_noisy,
    )

    x0 = np.array([p.bounds[0] for p in params], dtype=float)
    res = fitter.fit(x0, method="L-BFGS-B", maxiter=5, ftol=1e-6)

    assert hasattr(res, "success")
    assert np.isfinite(getattr(res, "fun", np.nan))
