import numpy as np
import pytest

from lmp_pkg.config.model import AppConfig, EntityRef
from lmp_pkg.app_api import run_single_simulation
from lmp_pkg.solver.optimization import ParameterFitter, ParameterDefinition
from lmp_pkg.engine.workflow import get_workflow


@pytest.mark.integration
def test_pe_pk_iv_runs_pk_stage_only():
    # Build config and force workflow to pk-only
    cfg = AppConfig(
        subject=EntityRef(ref="healthy_reference"),
        api=EntityRef(ref="BD"),
        product=EntityRef(ref="reference_product"),
        maneuver=EntityRef(ref="pMDI_variable_trapezoid"),
        pk={"model": "pk_3c"},
    )
    # Use workflow override to avoid default full pipeline
    result = run_single_simulation(cfg, workflow=get_workflow("pe_pk_iv"))

    # Assertions: only PK should be present
    assert result.pk is not None, "PK stage should produce output"
    assert result.deposition is None, "Deposition stage must not run in pe_pk_iv"
    assert result.pbbk is None, "PBBM stage must not run in pe_pk_iv"

    # Stage results should only include 'pk'
    stage_results = result.metadata.get("stage_results", {})
    assert set(stage_results.keys()) == {"pk"}


@pytest.mark.integration
def test_parameter_fitter_on_pk_only_workflow():
    # Base config
    cfg = AppConfig(
        subject=EntityRef(ref="healthy_reference"),
        api=EntityRef(ref="BD"),
        product=EntityRef(ref="reference_product"),
        maneuver=EntityRef(ref="pMDI_variable_trapezoid"),
        pk={"model": "pk_3c"},
    )

    # Generate synthetic data using pk-only workflow
    base_result = run_single_simulation(cfg, workflow=get_workflow("pe_pk_iv"))
    assert base_result.pk is not None
    t_obs = np.asarray(base_result.pk.t)
    c_obs = np.asarray(base_result.pk.conc_plasma)

    # Slight noise to emulate observations
    rng = np.random.default_rng(123)
    c_noisy = c_obs + rng.normal(0, 0.05 * (np.max(c_obs) if np.max(c_obs) > 0 else 1.0), size=c_obs.shape)

    # Define a small parameter set to fit
    params = [
        ParameterDefinition(name="volume_central_L", path="pk.params.volume_central_L", bounds=(10.0, 60.0)),
        ParameterDefinition(name="clearance_L_h", path="pk.params.clearance_L_h", bounds=(10.0, 200.0)),
    ]

    fitter = ParameterFitter(
        base_config=cfg,
        parameters=params,
        observed_time_s=t_obs,
        observed_concentration=c_noisy,
    )

    # Limit optimizer work for test speed
    x0 = np.array([p.bounds[0] for p in params], dtype=float)
    res = fitter.fit(x0, method="L-BFGS-B", maxiter=5, ftol=1e-6)

    assert hasattr(res, "success")
    # Even if not converged, objective should be finite at last eval
    assert np.isfinite(getattr(res, "fun", np.nan))
