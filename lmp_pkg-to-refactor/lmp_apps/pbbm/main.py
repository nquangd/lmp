import time
import copy
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count

from .lung_pbbm import PBPK_Model_Virtual_Trial
from ..population.attributes import SubjectPhysiology, GITract
from ..population.variability import Variability
from ..api.api import API
from ..study.study import Product, Study
 

def calculate_pk_and_regional_params(solution, model, initial_dose_pmol):
    """
    Calculates Systemic PK and Regional AUCs from simulation results.
    Units: Systemic Cmax (pg/mL), AUCs (pg*h/mL), Dose (pg).
    """
    time_h = solution.t / 3600.0
    
    # --- Systemic PK (in pg/mL and pg*h/mL) ---
    V_central_L = model.api.V_central_L
    Cp_central_pmol = solution.y[-3, :]
    Cp_central_pmol_per_L = Cp_central_pmol / V_central_L
    # Convert pmol/L to pg/mL: (pmol/L) * (MM pg/pmol) * (1 L/1000 mL) = (pmol/L) * MM / 1000
    Cp_central_pg_per_ml = Cp_central_pmol_per_L * model.api.MM / 1000.0
    
    systemic_Cmax = np.max(Cp_central_pg_per_ml)
    systemic_AUC = np.trapz(Cp_central_pg_per_ml, time_h)

    # --- Regional AUCs (0-12h) in pg*h/mL ---
    time_12h_mask = time_h <= 12
    time_for_auc = time_h[time_12h_mask]
    
    y_lung_t = solution.y[:model.num_lung_states, time_12h_mask]

    auc_epi = np.zeros(len(model.region))
    auc_tissue = np.zeros(len(model.region))
    auc_epi_tissue = np.zeros(len(model.region))

    for r_idx, r_name in enumerate(model.region):
        offset = model.region_offsets[r_idx]
        n_epi = model.n_epi_layers[r_idx]
        
        epi_start_index = offset + 1
        Xg_Epi_local_shallow = y_lung_t[epi_start_index : epi_start_index + n_epi, :]
        Xg_Epi_local_deep = y_lung_t[epi_start_index + n_epi : epi_start_index + 2 * n_epi, :]
        
        tissue_offset = epi_start_index + 2 * n_epi
        Xg_Tissue_shallow = y_lung_t[tissue_offset, :]
        Xg_Tissue_deep = y_lung_t[tissue_offset + 1, :]

        Xg_Epi = np.sum(Xg_Epi_local_shallow, axis=0) + np.sum(Xg_Epi_local_deep, axis=0) # pmol
        Xg_Tissue = Xg_Tissue_shallow + Xg_Tissue_deep # pmol
        Xg_Epi_Tissue = Xg_Epi + Xg_Tissue # pmol
        
        V_Epi_r = model.numba_params['V_epi'][r_idx] # in cm^3 or mL
        V_Tissue_r = model.numba_params['V_tissue'][r_idx] # in cm^3 or mL
        V_Epi_Tissue_r = V_Epi_r + V_Tissue_r

        # Concentration in pmol/mL
        Cg_Epi_pmol_ml = Xg_Epi / V_Epi_r if V_Epi_r > 0.0 else np.zeros_like(Xg_Epi)
        Cg_Tissue_pmol_ml = Xg_Tissue / V_Tissue_r if V_Tissue_r > 0.0 else np.zeros_like(Xg_Tissue)
        Cg_Epi_Tissue_pmol_ml = Xg_Epi_Tissue / V_Epi_Tissue_r if V_Epi_Tissue_r > 0 else np.zeros_like(Xg_Epi_Tissue)

        # Convert pmol/mL to pg/mL: (pmol/mL) * (MM pg/pmol)
        Cg_Epi_pg_ml = Cg_Epi_pmol_ml * model.api.MM
        Cg_Tissue_pg_ml = Cg_Tissue_pmol_ml * model.api.MM
        Cg_Epi_Tissue_pg_ml = Cg_Epi_Tissue_pmol_ml * model.api.MM
        
        # Calculate AUCs in pg*h/mL
        auc_epi[r_idx] = np.trapz(Cg_Epi_pg_ml, time_for_auc)
        auc_tissue[r_idx] = np.trapz(Cg_Tissue_pg_ml, time_for_auc)
        auc_epi_tissue[r_idx] = np.trapz(Cg_Epi_Tissue_pg_ml, time_for_auc)
    
    regional_aucs = {
        'Epithelium': auc_epi,
        'Tissue': auc_tissue,
        'Epithelium_Tissue': auc_epi_tissue
    }
    
    # --- Assemble Results Table ---
    results_list = []
    total_dose_pg = sum(initial_dose_pmol.values()) * model.api.MM
    
    al_depo = initial_dose_pmol['Al']
    bb_depo = initial_dose_pmol['BB']
    cp_ratio = al_depo / bb_depo if bb_depo > 0.0 else 0.0

    for i, r in enumerate(model.region):
        for comp, auc_val_array in regional_aucs.items():
            results_list.append({
                'Region': r,
                'Compartment': comp,
                'Regional_AUC': auc_val_array[i], # Already in pg*h/mL
                'Systemic_AUC': systemic_AUC,     # Already in pg*h/mL
                'Systemic_Cmax': systemic_Cmax,    # Already in pg/mL
                'Dose': total_dose_pg,
                'ET': (initial_dose_pmol['ET'] * model.api.MM) / total_dose_pg if total_dose_pg > 0.0 else 0.0,
                'C/P': cp_ratio,
                'Regional Deposition': initial_dose_pmol[r] * model.api.MM # in pg
            })
    return results_list

def run_single_pbbm_simulation(args):
    """
    Worker function for parallel processing. Returns both PK params and the full solution.
    """
    subject, product_name, api_data, initial_dose_pmol, variability, settings = args
    #subject_intra = subject.get_subject_intra()
    # Create a deep copy of the API to avoid race conditions and state sharing in parallel execution
    sim_api_params = copy.deepcopy(api_data['api'])
    api_name = sim_api_params.name
    
    # Apply inter- and intra-subject variability for the current simulation run
    cl_inter_factor = subject.PK_Scale_Factor['CL'].get(api_name, 1.0) # subject.PK_Scale_Factor['CL'][api_name]
    cl_intra_factor = subject.get_scale_factor_pk(subject.variability.Intra)['CL'].get(api_name, 1.0) # subject_intra.PK_Scale_Factor['CL'].get(api_name, 1.0)
    sim_api_params.CL_h *= cl_inter_factor * cl_intra_factor
    
    eh_inter_factor = subject.PK_Scale_Factor['Eh'].get(api_name, 1.0)
    eh_intra_factor = subject.get_scale_factor_pk(subject.variability.Intra)['Eh'].get(api_name, 1.0) #subject_intra.PK_Scale_Factor['Eh'].get(api_name, 1.0)
    sim_api_params.Eh = min(100.00, sim_api_params.Eh * eh_inter_factor * eh_intra_factor)
    
    print(f"  Starting PBPK for Subject {subject.id}, Product {product_name}, API {api_name}...")
    model = PBPK_Model_Virtual_Trial(subject, sim_api_params, settings)
    #model.rtol = rtol
    #model.atol = atol
    #model.dissolution_cutoff_radius = dissolution_cutoff_radius
    #model.k_lump = k_lump
    
    try:
        solution = model.run_simulation(initial_dose_pmol)
        if not hasattr(solution, "success") or not solution.success:
            print(f"ODE failed for Subject {subject.id}, Product {product_name}, API {api_name}: {getattr(solution, 'message', 'Unknown error')}")
            return None
        
        # Calculate the PK parameter summary
        pk_params_list = calculate_pk_and_regional_params(solution, model, initial_dose_pmol)
        
        # Add identifiers to each dictionary in the pk_params_list
        for row in pk_params_list:
            row['Subject'] = subject.id
            row['Product'] = product_name
            row['API'] = api_name
            
        # Return a dictionary containing the now-identified PK parameters and the full solution for plotting
        return {
            'pk_params': pk_params_list, 
            'solution': solution, 
            'model': model, 
            'subject_id': subject.id, 
            'product_name': product_name, 
            'api_name': api_name
        }
    
    except Exception as e:
        print(f"Exception caught in simulation: {e}")
        return None
    

def process_simulation_solutions(simulation_outputs):
    """
    Processes the raw solution objects from simulations into a tidy DataFrame
    with concentrations in pg/mL for plotting.
    """
    all_profiles = []
    for output in simulation_outputs:
        sol = output['solution']
        model = output['model']
        time_h = sol.t / 3600
        
        # Process Plasma Profile (Concentration in pg/mL)
        Cp_central_pmol = sol.y[-3, :]
        Cp_central_pg_per_ml = (Cp_central_pmol / model.api.V_central_L) * model.api.MM / 1000.0
        for t, conc in zip(time_h, Cp_central_pg_per_ml):
            all_profiles.append({'Subject': output['subject_id'], 'Product': output['product_name'], 'API': output['api_name'], 'Time': t, 'Concentration': conc, 'ProfileType': 'Plasma', 'Region': 'Plasma', 'Compartment': 'Central'})

        # Process Regional Profiles (Concentration in pg/mL)
        y_lung_t = sol.y[:model.num_lung_states, :]
        for r_idx, r_name in enumerate(model.region):
            offset, n_epi = model.region_offsets[r_idx], model.n_epi_layers[r_idx]
            V_Epi_r, V_Tissue_r = model.numba_params['V_epi'][r_idx], model.numba_params['V_tissue'][r_idx]

            epi_start_index = offset + 1
            Xg_Epi_pmol = np.sum(y_lung_t[epi_start_index : epi_start_index + 2 * n_epi, :], axis=0)
            Cg_Epi_pmol_ml = Xg_Epi_pmol / V_Epi_r if V_Epi_r > 0 else np.zeros_like(Xg_Epi_pmol)
            Cg_Epi_pg_ml = Cg_Epi_pmol_ml * model.api.MM

            tissue_offset = epi_start_index + 2 * n_epi
            Xg_Tissue_pmol = np.sum(y_lung_t[tissue_offset : tissue_offset + 2, :], axis=0)
            Cg_Tissue_pmol_ml = Xg_Tissue_pmol / V_Tissue_r if V_Tissue_r > 0 else np.zeros_like(Xg_Tissue_pmol)
            Cg_Tissue_pg_ml = Cg_Tissue_pmol_ml * model.api.MM
            
            for t, conc in zip(time_h, Cg_Epi_pg_ml):
                all_profiles.append({'Subject': output['subject_id'], 'Product': output['product_name'], 'API': output['api_name'], 'Time': t, 'Concentration': conc, 'ProfileType': 'Regional', 'Region': r_name, 'Compartment': 'Epithelium'})
            for t, conc in zip(time_h, Cg_Tissue_pg_ml):
                all_profiles.append({'Subject': output['subject_id'], 'Product': output['product_name'], 'API': output['api_name'], 'Time': t, 'Concentration': conc, 'ProfileType': 'Regional', 'Region': r_name, 'Compartment': 'Tissue'})

    return pd.DataFrame(all_profiles)

def run_pbbm_simulations(subjects, products, deposition_pool, variability, settings, num_cores = 1, distributed = False):
    pbpk_tasks = []
    for subject in subjects:
        for product in products:
            for api_data in product.apis_data:
                api_name = api_data['api'].name
                if subject.id in deposition_pool and product.name in deposition_pool[subject.id] and api_name in deposition_pool[subject.id][product.name]:
                    initial_dose = deposition_pool[subject.id][product.name][api_name]
                    pbpk_tasks.append((subject, product.name, api_data, initial_dose, variability, settings))

    start_time = time.time()
    if distributed:
        simulation_outputs_all = [run_single_pbbm_simulation(args) for args in pbpk_tasks]
    else:
        with Pool(processes=num_cores) as pool:
            simulation_outputs_all = pool.map(run_single_pbbm_simulation, pbpk_tasks)
    end_time = time.time()
    print(f"\nTotal PBPK simulation time: {end_time - start_time:.2f} seconds.")
  
    simulation_outputs = [o for o in simulation_outputs_all if o is not None]
    
    return simulation_outputs