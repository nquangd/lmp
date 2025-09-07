import numpy as np
import time
from numba import jit, njit, prange
import pandas as pd
import os
from multiprocessing import Pool

from .Parameters_Settings import *
from .Lung_classes import *
from .PhysicalProp_classes import *
from .Calc_functions import *
from .Lung_deposition import *
from .deposition_PostCalc import *
from .Integral_flow import *
from .Lung_deposition_packageVariable_DN_edit import Lung_deposition_packageVariable
from ..cfd.ml_models import ML_CFD_MT_deposition
from .helper_functions import constructPSD, scale_lung




def lung_deposition_worker(args):
    """
    A simple Python wrapper to call the Numba-jitted worker function.
    This helps avoid pickling issues with multiprocessing.
    """
    try:
        return Lung_deposition_packageVariable(*args)
    except Exception as e:
        print(f"Exception caught in simulation: {e}")
        return None
    
    #return Lung_deposition_packageVariable(*args)

def run_deposition_simulations(subjects, products, num_cores = 1, distributed = False):
    """
    Prepares and runs all lung deposition simulations in parallel.
    """
    print("--- Starting precomputation of arguments for deposition model ---")
    
    tasks = []
    sim_map = {}
    sim_counter = 0

    for subject in subjects:
        maneuver = subject.inhalation_maneuver
        maneuver.inhale_profile()
        flowrate_profile = maneuver.flowprofile # trapezoid_waveform(maneuver.inhaled_volume_L * 1000, maneuver.pifr_Lpm, maneuver.rise_time_s, N_Q_max)
        flowrate_profile[:,1] = flowrate_profile[:,1] * 1e-3 / 60 # Convert to m3/s
        geometry_all = subject.physiology_df.values
        sub_lung = scale_lung(geometry_all, subject.FRC)

        var = subject.variability
        
        # Calculate inter-subject scaling factors using a multiplicative approach
        lung_dose_scale_inter = var.lung_dose_inter['mean'] * np.random.lognormal(mean=0, sigma=var.lung_dose_inter['std'])
        mmad_scale_inter = var.mmad_exMT_inter['mean'] * np.random.lognormal(mean=0, sigma=var.mmad_exMT_inter['std'])
        gsd_scale_inter = var.gsd_exMT_inter['mean'] * np.random.lognormal(mean=0, sigma=var.gsd_exMT_inter['std'])
        
        for product in products:
            for api_data in product.apis_data:
                sim_map[sim_counter] = {'subject_id': subject.id, 'product_name': product.name, 'api_name': api_data['api'].name}
                
                cfd_mtdepo, cfd_mmad, cfd_gsd = ML_CFD_MT_deposition(api_data['mmad'], api_data['gsd'], api_data['propellant'], api_data['usp_depo_fraction'], maneuver.pifr_Lpm, subject.MT_size, api_data['device'].lower() ) 
            
                # Calculate intra-subject factors and apply all variability
                lung_dose_factor_intra = np.random.lognormal(mean=0, sigma=var.lung_dose_intra['std'])
                mmad_factor_intra = np.random.lognormal(mean=0, sigma=var.mmad_exMT_intra['std'])
                gsd_factor_intra = np.random.lognormal(mean=0, sigma=var.gsd_exMT_intra['std'])

                mtdepo = (cfd_mtdepo) * lung_dose_scale_inter * lung_dose_factor_intra
                mmad = cfd_mmad * mmad_scale_inter * mmad_factor_intra
                gsd = cfd_gsd * gsd_scale_inter * gsd_factor_intra
                
                bins = np.linspace(min_bin_size, max_bin_size, N_bins_max)
                d_a, d_g, v_f = constructPSD(mmad, gsd, bins)
                psd_array = np.vstack((d_a, d_g, v_f)).T

                task_args = (
                    sub_lung, subject.FRC, psd_array, mtdepo,
                    flowrate_profile, maneuver.breath_hold_time_s,
                    maneuver.exhalation_flow_Lpm * 1e-3 / 60, # m3/s
                    maneuver.bolus_volume_ml * 1e-6, # m3
                    maneuver.bolus_delay_s
                )
                tasks.append(task_args)
                sim_counter += 1

    print(f"Running {len(tasks)} deposition simulations in parallel on {num_cores} cores...")
    start_time = time.time()
    if distributed:
        results = [lung_deposition_worker(args) for args in tasks]
    else:
        with Pool(processes=num_cores) as pool:
            results = pool.map(lung_deposition_worker, tasks)
    end_time = time.time()
    print(f"Total deposition simulation time: {end_time - start_time:.2f} seconds.")

    # Process results
    results = [o for o in results if o is not None]
    deposition_pool = {}
    #deposition_fraction_pool = {}
    
    pkg_dir = os.path.dirname(__file__)
    scint_keys_path = os.path.join(pkg_dir, '..', 'data', 'lung', 'scint_keys.csv')
    Scint_Keys = pd.read_csv(scint_keys_path).iloc[:, [2, 3, 4]].values

    for i, depo in enumerate(results):
        info = sim_map[i]
        s_id, p_name, a_name = info['subject_id'], info['product_name'], info['api_name']

        # Map deposition vector to regions
        cip = np.dot(depo[:-1], Scint_Keys) / 100
        ET = depo[0]
        BB = np.sum(depo[1:10])
        bb_b = np.sum(depo[10:17])
        Al =  np.sum(depo[17:-1])
        depo_results_arr = np.array((cip[0], cip[1], cip[2], ET, BB, bb_b, Al, depo[-1], Al/BB, cip[0] / cip[2]))
        
        # Depo results columns (0-based): 3:ET, 4:BB, 5:Bb, 6:Al, 7:Exhale
        et_frac, bb_frac, bbb_frac, al_frac, exhale = depo_results_arr[3:8]
        cpratio = depo_results_arr[-1]
        api_data = next(item for p in products if p.name == p_name for item in p.apis_data if item['api'].name == a_name)
        # Dose is in pg, MM is in pg/pmol -> total_dose_pmol
        total_dose_pmol = api_data['dose_pg'] / api_data['api'].MM

        if s_id not in deposition_pool: deposition_pool[s_id] = {}
        if p_name not in deposition_pool[s_id]: deposition_pool[s_id][p_name] = {}
        deposition_pool[s_id][p_name][a_name] = {'ET': total_dose_pmol * et_frac, 
                                                 'BB': total_dose_pmol * bb_frac, 
                                                 'bb_b': total_dose_pmol * bbb_frac, 
                                                 'Al': total_dose_pmol * al_frac,
                                                 }
        
        #deposition_fraction_pool[s_id][p_name][a_name] = depo_results_arr
      
    print("--- Deposition pre-computation complete ---\n")
    return deposition_pool