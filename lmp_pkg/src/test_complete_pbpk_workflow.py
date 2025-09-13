#!/usr/bin/env python3
"""
Complete PBPK workflow test using new architecture:
1. Load entities using new domain system
2. Run CFD for mouth-throat deposition
3. Run regional deposition for lung regions
4. Run PBPK model for systemic exposure
5. Compare results with notebook
"""

import numpy as np
import sys
import warnings
import pandas as pd
from scipy.integrate import trapz

from lmp_pkg.domain.entities import Subject, API, Product
from lmp_pkg.models.cfd.ml_models import ML_CFD_MT_deposition
from lmp_pkg.models.deposition.clean_lung import CleanLungDeposition, CleanDepositionParameters
from lmp_pkg.contracts.types import DepositionInput
from lmp_pkg.models.pbpk.pbpk_orchestrator import PBPKOrchestrator, ModelComponentType

warnings.filterwarnings('ignore')

# Add both paths for accessing legacy PBPK
sys.path.insert(0, '/Users/duynguyen/Library/Mobile Documents/com~apple~CloudDocs/LMP/lmp/lmp_pkg/src')
sys.path.insert(0, '/Users/duynguyen/Library/Mobile Documents/com~apple~CloudDocs/LMP/lmp/lmp_pkg-to-refactor')

def run_and_analyze_pbpk(subject_params, api_params, regional_amounts, solve_dissolution, expected_pk=None):
    """Helper function to run PBPK and analyze results."""
    print("="*80)
    print(f"RUNNING PBPK SIMULATION (Dissolution: {solve_dissolution})")
    print("="*80)

    orchestrator = PBPKOrchestrator(
        subject_params=subject_params,
        api_params=api_params,
        components=[ModelComponentType.LUNG, ModelComponentType.GI, ModelComponentType.PK],
        lung_model_type="regional",
        pk_model_type="3c",
        solve_dissolution=solve_dissolution
    )

    simulation_time_hours = 24
    time_points = np.linspace(0, simulation_time_hours * 3600, simulation_time_hours * 60 + 1) # 1-minute intervals

    initial_conditions = {
        'ET_deposition': regional_amounts[0],
        'BB_deposition': regional_amounts[1],
        'bb_deposition': regional_amounts[2],
        'Al_deposition': regional_amounts[3],
    }

    results = orchestrator.solve(time_points, initial_conditions=initial_conditions)
    
    # Extract plasma concentration
    plasma_concentration = results.results_data['pk']['plasma_concentration_ng_ml']
    time_hours = results.time_points / 3600

    # Calculate PK parameters
    cmax = np.max(plasma_concentration)
    tmax_idx = np.argmax(plasma_concentration)
    tmax = time_hours[tmax_idx]
    auc = trapz(plasma_concentration, time_hours)

    print(f"\n--- PK Results (Dissolution: {solve_dissolution}) ---")
    print(f"Cmax: {cmax:.2f} ng/mL")
    print(f"Tmax: {tmax:.2f} hours")
    print(f"AUC: {auc:.2f} ng·h/mL")

    if expected_pk:
        print("\n--- Comparison with Notebook ---")
        cmax_expected = expected_pk['cmax']
        auc_expected = expected_pk['auc']
        
        cmax_error = (cmax - cmax_expected) / cmax_expected * 100
        auc_error = (auc - auc_expected) / auc_expected * 100

        print(f"Cmax: {cmax:.2f} (Expected: {cmax_expected:.2f}, Error: {cmax_error:.2f}%)")
        print(f"AUC: {auc:.2f} (Expected: {auc_expected:.2f}, Error: {auc_error:.2f}%)")
        
        assert abs(cmax_error) < 5, f"Cmax error ({cmax_error:.2f}%) exceeds tolerance"
        assert abs(auc_error) < 5, f"AUC error ({auc_error:.2f}%) exceeds tolerance"
        print("✓ PK parameters match notebook values within 5% tolerance.")

    return results, (cmax, tmax, auc)

def run_complete_workflow_new_architecture():
    print("=" * 80)
    print("COMPLETE PBPK WORKFLOW TEST WITH NEW ARCHITECTURE")
    print("=" * 80)
    
    try:
        # Step 1: Load entities
        subject = Subject.from_builtin("healthy_reference")
        api = API.from_builtin("BD")
        product = Product.from_builtin("reference_product")
        
        # Step 2: Get final transformed values
        subject_final = subject.get_final_values(apply_variability=False)
        product_final = product.get_final_values("BD")
        
        # Step 3: Run CFD
        mt_depo_fraction, mmad_cast, gsd_cast = ML_CFD_MT_deposition(
            mmad=product_final._final_mmad, gsd=product_final._final_gsd,
            propellant=product_final.propellant, usp_deposition=product_final._final_usp_depo_fraction,
            pifr=subject_final.inhalation_maneuver.pifr_Lpm, cast="medium", DeviceType="dfp"
        )
        mt_depo_fraction_scaled = mt_depo_fraction * subject_final.inhalation_maneuver.et_scale_factor
        
        # Step 4: Run regional deposition
        dose_pmol = product_final._final_dose_pg / api.molecular_weight
        
        clean_params = CleanDepositionParameters(
            frc_ml=subject_final.demographic.frc_ml,
            lung_geometry=subject_final._scaled_lung_geometry,
            regional_constants=dict(),
            peak_flow_l_min=subject_final.inhalation_maneuver.pifr_Lpm,
            inhaled_volume_ml=subject_final._scaled_inhaled_volume_L * 1000,
            breath_hold_time_s=subject_final.inhalation_maneuver.breath_hold_time_s,
            exhalation_flow_l_min=subject_final.inhalation_maneuver.exhalation_flow_Lpm,
            flow_profile=subject_final._flow_profile,
            bolus_volume_ml=subject_final.inhalation_maneuver.bolus_volume_ml,
            bolus_delay_s=subject_final.inhalation_maneuver.bolus_delay_s,
            mmad=mmad_cast, gsd=gsd_cast, mt_deposition_fraction=mt_depo_fraction_scaled,
            dose_pmol=dose_pmol, molecular_weight=api.molecular_weight
        )
        
        deposition_model = CleanLungDeposition("pbpk_test")
        deposition_input = DepositionInput(
            subject=subject_final.__dict__,
            product=product.__dict__,
            maneuver=subject_final.inhalation_maneuver.__dict__,
            params=clean_params.__dict__
        )
        
        deposition_result = deposition_model.run(deposition_input)
        regional_amounts = deposition_result.elf_initial_amounts

        # --- Run with dissolution ON and compare to notebook ---
        expected_pk_diss_on = {'cmax': 165.34, 'auc': 1312.54}
        run_and_analyze_pbpk(subject_final, api, regional_amounts, solve_dissolution=True, expected_pk=expected_pk_diss_on)

        # --- Run with dissolution OFF and report results ---
        run_and_analyze_pbpk(subject_final, api, regional_amounts, solve_dissolution=False, expected_pk=None)

    except Exception as e:
        print(f"❌ Workflow test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_complete_workflow_new_architecture()
    print("\n" + "=" * 80)
