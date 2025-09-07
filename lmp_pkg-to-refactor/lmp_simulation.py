import time
import copy
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count


from lmp_apps.pbbm.lung_pbbm import PBPK_Model_Virtual_Trial
from lmp_apps.pbbm.main import run_pbbm_simulations, process_simulation_solutions

from lmp_apps.lung_deposition.main import run_deposition_simulations
from lmp_apps.population.attributes import SubjectPhysiology
from lmp_apps.population.variability import Variability
from lmp_apps.api.api import API
from lmp_apps.study.study import Product, Study
from lmp_apps.vbe.bioequivalence import BioequivalenceAssessor
from lmp_apps.efficacy.fev1model import COPD_FEV1
from lmp_apps.visualisation.plots import TrialVisualizer
from lmp_apps.utils.config import simulation_settings

import warnings
warnings.filterwarnings('ignore')

# VBE package imports


#from vbe.utils.helpers import process_simulation_solutions, calculate_pk_and_regional_params

def main():
    """
    Main workflow for the virtual bioequivalence trial.
    """
    # --- Configuration ---
    ### solubv = 120 * 1e3 * 1e6, + rtol 1e-4 + atol 1e-8 + klump 5e-4 + cutoff 1e-6 works for gp ff

    # --- Configuration ---

    #ENABLE_VARIABILITY = True #False
    #DISSOLUTION_CUTOFF_RADIUS = 1e-6 # 1e-6
    #K_LUMP = 1e-4 * 1e6  # adjusted rate to pmol/s # 5e-4 #K_LUMP = 5e-4 * 1e6  # adjusted rate to pmol/s
    #RTOL = 1e-4 # 1e-6 # 1e-4
    #ATOL = 1e-8 # 1e-9 # 1e-8

    settings = simulation_settings(ENABLE_VARIABILITY = False)

    NUM_CORES = 8 #cpu_count() - 1 if cpu_count() > 1 else 1
    N_SUBJECTS = 8
    N_TRIALS_BE = 10



    study = Study(study_type='NON-CHARCOAL', trial_size = N_SUBJECTS, n_trials = N_TRIALS_BE)
        
    # --- Setup APIs and Products ---
    bd_base, gp_base, ff_base = API(name="BD"), API(name="GP"), API(name="FF")
    bd_base.solub = 25.8 * 1e6
    gp_base.solub = 120 * 1e3 * 1e6
    ff_base.solub = 120 * 1e3 * 1e6

    base_apis={"BD": bd_base, "GP": gp_base, "FF": ff_base}

    # Create deep copies of the API objects for each product to ensure independence
    test_apis = copy.deepcopy(base_apis)
    ref_apis = copy.deepcopy(base_apis)

    test_product = Product("Test", base_apis=[
        {'api': test_apis["BD"], 'dose_pg': 330.34e6, 'mmad': 4.06, 'gsd': 1.67, 'propellant': 'PT210', 'device': 'DFP', 'usp_depo_fraction': 41.08},
        {'api': test_apis["GP"], 'dose_pg': 14.42e6, 'mmad': 3.94, 'gsd': 1.67, 'propellant': 'PT210', 'device': 'DFP', 'usp_depo_fraction': 38.06},
        {'api': test_apis["FF"], 'dose_pg': 9.68e6, 'mmad': 4.0, 'gsd': 1.69, 'propellant': 'PT210', 'device': 'DFP', 'usp_depo_fraction': 39.50}
    ], study = study)
    ref_product = Product("Reference", base_apis=[
        {'api': ref_apis["BD"], 'dose_pg': 329.04e6, 'mmad': 3.53, 'gsd': 1.61, 'propellant': 'PT010', 'device': 'DFP', 'usp_depo_fraction': 41.26},
        {'api': ref_apis["GP"], 'dose_pg': 14.46e6, 'mmad': 3.4, 'gsd': 1.62, 'propellant': 'PT010', 'device': 'DFP', 'usp_depo_fraction': 38.18},
        {'api': ref_apis["FF"], 'dose_pg': 9.62e6, 'mmad': 3.4, 'gsd': 1.66, 'propellant': 'PT010', 'device': 'DFP', 'usp_depo_fraction': 38.28}
    ], study = study)
    products = [ref_product, test_product]

    print("--- Starting Virtual BE Trial ---")
    print(f"Variability Enabled: {settings.ENABLE_VARIABILITY}")

    # 1. Generate the virtual population
    variability = Variability(enable_variability=settings.ENABLE_VARIABILITY)
    subjects = [SubjectPhysiology(i, variability, enable_variability=settings.ENABLE_VARIABILITY) for i in range(1, study.trial_size + 1)]

    # 2. Pre-compute lung depositions in parallel
    deposition_pool = run_deposition_simulations(subjects, products, num_cores=NUM_CORES)




    # 5. Assemble and Process Results
    simulation_outputs = run_pbbm_simulations(subjects, products, deposition_pool, variability, settings, num_cores=NUM_CORES)
        
        
    final_pk_params = [item for output in simulation_outputs for item in output['pk_params']]
    results_df = pd.DataFrame(final_pk_params)
    
    processed_profiles_df = process_simulation_solutions(simulation_outputs)

    # 6. Bioequivalence Assessment and Visualization
    #be_assessor = BioequivalenceAssessor(n_trials=study.n_trials)
    be_assessor = BioequivalenceAssessor(study, residual_var = 0.042)
    visualizer = TrialVisualizer()

    # Systemic BE
    systemic_summary, systemic_trials = be_assessor.systemic_bioequiv(results_df)
    print("\n--- Systemic Bioequivalence Summary ---")
    print(systemic_summary)
    visualizer.plot_systemic_trials(systemic_trials)

    # Concentration-Time Profiles
    visualizer.plot_concentration_time_profiles(processed_profiles_df, plot_type='plasma', error_band='gcv')

if __name__ == "__main__":
    main()
