#!/usr/bin/env python3
import numpy as np
import pytest

#from lmp.lmp_pkg.src.lmp_pkg import app_api
from lmp_pkg.config.model import AppConfig, EntityRef, RunConfig
from lmp_pkg.app_api import run_single_simulation, pk_overlay_dataframe
from lmp_pkg.solver.optimization import ParameterFitter, ParameterDefinition
from lmp_pkg.engine.workflow import get_workflow
import pandas as pd

def test_pe_pk_iv_runs_pk_stage_only():
    # Build config and force workflow to pk-only
    cfg = AppConfig(
        run=RunConfig(workflow_name="pe_pk_iv"),
        subject=EntityRef(ref="healthy_reference"),
        api=EntityRef(ref="BD"),
        product=EntityRef(ref="reference_product"),
        maneuver=EntityRef(ref="pMDI_variable_trapezoid"),
        pk={"model": "pk_3c"},
    )
    # Use workflow override to avoid default full pipeline (redundant but explicit)
    result = run_single_simulation(cfg, workflow=get_workflow("pe_pk_iv"))

    # Assertions: only IV PK should be present
    assert result.pk is not None, "IV PK stage should produce output"
    assert result.deposition is None, "Deposition stage must not run in pe_pk_iv"
    assert result.pbbk is None, "PBBM stage must not run in pe_pk_iv"

    # Stage results should only include 'iv_pk'
    stage_results = result.metadata.get("stage_results", {})
    assert set(stage_results.keys()) == {"iv_pk"}



def test_parameter_fitter_on_pk_only_workflow():
    # Base config (pk-only workflow propagates into ParameterFitter)
    cfg = AppConfig(
        run=RunConfig(workflow_name="pe_pk_iv"),
        subject=EntityRef(ref="healthy_reference"),
        api=EntityRef(ref="BD"),
        product=EntityRef(ref="reference_product"),
        maneuver=EntityRef(ref="pMDI_variable_trapezoid"),
        pk={"model": "pk_3c"},
    )

    # Generate synthetic data using pk-only workflow
    base_result = run_single_simulation(cfg)
    assert base_result.pk is not None
    t_obs = np.asarray(base_result.pk.t)
    c_obs = np.asarray(base_result.pk.conc_plasma)

    # Slight noise to emulate observations
    rng = np.random.default_rng(123)
    c_noisy = c_obs + rng.normal(0, 0.05 * (np.max(c_obs) if np.max(c_obs) > 0 else 1.0), size=c_obs.shape)

    # Define a small parameter set to fit
    params = [
        ParameterDefinition(name="volume_central_L", path="iv_pk.params.volume_central_L", bounds=(10.0, 60.0)),
        ParameterDefinition(name="clearance_L_h", path="iv_pk.params.clearance_L_h", bounds=(10.0, 200.0)),
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
    print(res)
    assert hasattr(res, "success")
    # Even if not converged, objective should be finite at last eval
    assert np.isfinite(getattr(res, "fun", np.nan))
    # Predict with fitted params
    run = run_single_simulation(cfg)
    # Prepare overlay dataframe
    df = pk_overlay_dataframe(run, pd.DataFrame({"time_s": t_obs, "label": c_noisy}))
    import matplotlib.pyplot as plt
    plt.figure()
    plt.semilogy(df['time_s'], df['simulated_conc_plasma'], label='Simulated', marker='o')
    plt.semilogy(df['time_s'], df['observed_conc_plasma'], label='Observed', marker='x')
    plt.xlabel('Time (s)')
    plt.ylabel('Plasma Concentration')
    plt.title('PK Overlay: Simulated vs Observed')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    test_pe_pk_iv_runs_pk_stage_only()
    test_parameter_fitter_on_pk_only_workflow()
