#!/usr/bin/env python3
"""Example script for running a parameter estimation for PK parameters."""

import numpy as np

from lmp_pkg.config.model import AppConfig, EntityRef
from lmp_pkg.solver.optimization import ParameterFitter, ParameterDefinition
from lmp_pkg.app_api import run_single_simulation


def generate_pk_data(config: AppConfig, noise_level: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic PK data with noise."""
    print("Generating synthetic PK data...")
    result = run_single_simulation(config)
    time = result.pk.t
    conc = result.pk.conc_plasma
    
    # Add some noise
    noise = np.random.normal(0, noise_level * np.max(conc), len(conc))
    noisy_conc = conc + noise
    
    return time, noisy_conc


def main():
    """Run the pe_pk_iv workflow."""
    print("Running parameter estimation for IV PK example...")

    # 1. Create a base AppConfig for the 'BD' API
    config = AppConfig(
        subject=EntityRef(ref='healthy_reference'),
        api=EntityRef(ref='BD'),
        product=EntityRef(ref='reference_product'),
        maneuver=EntityRef(ref='pMDI_variable_trapezoid'),
        pk={"model": "pk_3c"}
    )

    # 2. Generate synthetic data to fit
    true_params = {
        "volume_central_L": 29.92,
        "clearance_L_h": 94.79,
        "k12_h": 6.656,
        "k21_h": 2.203,
        "k13_h": 1.623,
        "k31_h": 0.3901,
    }
    observed_time, observed_conc = generate_pk_data(config)

    # 3. Define the parameters to be fitted
    parameters = [
        ParameterDefinition(name="volume_central_L", path="pk.params.volume_central_L", bounds=(10, 50)),
        ParameterDefinition(name="clearance_L_h", path="pk.params.clearance_L_h", bounds=(50, 150)),
        ParameterDefinition(name="k12_h", path="pk.params.k12_h", bounds=(1, 10)),
        ParameterDefinition(name="k21_h", path="pk.params.k21_h", bounds=(1, 5)),
        ParameterDefinition(name="k13_h", path="pk.params.k13_h", bounds=(0.5, 3)),
        ParameterDefinition(name="k31_h", path="pk.params.k31_h", bounds=(0.1, 1.0)),
    ]

    # 4. Instantiate the ParameterFitter
    fitter = ParameterFitter(
        base_config=config,
        parameters=parameters,
        observed_time_s=observed_time,
        observed_concentration=observed_conc,
    )

    # 5. Run the fitting process
    initial_guess = [p.bounds[0] for p in parameters]
    result = fitter.fit(initial_guess)

    # 6. Print the results
    print("\nParameter Estimation Results:")
    if result.success:
        print("  Success: True")
        print(f"  Message: {result.message}")
        fitted_params = result.x
        if fitted_params is not None:
            print("  Fitted Parameters vs. True Parameters:")
            for param, (true_name, true_val), fitted_val in zip(parameters, true_params.items(), fitted_params):
                print(f"    {param.name}: True={true_val:.4f}, Fitted={fitted_val:.4f}")
    else:
        print("  Success: False")
        print(f"  Message: {result.message}")


if __name__ == "__main__":
    main()