import numpy as np
from scipy.integrate import solve_ivp
from numba import jit
from ..population.attributes import SubjectPhysiology, GITract
from ..api.api import API


# ==============================================================================
# JIT-Compiled ODE System (The Core Solver Logic)
# ==============================================================================
@jit(nopython=True, cache=True)
def ode_system_numba_mechanistic(t:np.float64, y:np.float64,
                                 # State dimensions and helpers
                                 num_lung_states:np.int32, num_gi_states:np.int32, num_deposition_bins:np.int32,
                                 region_offsets:np.int32, n_epi_layers:np.int32, max_n_epi:np.int32,
                                 # Physico-chemical parameters
                                 Vm:np.float64, Sg:np.float64, D:np.float64, MM:np.float64, dissolution_cutoff_radius:np.float64, k_lump:np.float64,
                                 # General physiological parameters
                                 fu:np.float64, BP:np.float64, V_frac_g:np.float64,
                                 # Precalculated Lung Geometry/Flow arrays
                                 A_elf:np.float64, V_elf:np.float64, V_epi:np.float64, V_tissue:np.float64, Q_g_region:np.float64, tg_region:np.float64,
                                 # Precalculated Lung Permeability/Binding arrays
                                 Pg_in:np.float64, Pg_out:np.float64, Pg_para:np.float64,
                                 K_in_epi:np.float64, K_out_epi:np.float64, K_in_tissue:np.float64, K_out_tissue:np.float64,
                                 fu_epi_calc:np.float64, fu_tissue_calc:np.float64, cell_bind:np.int32,
                                 # GI parameters
                                 gi_area:np.float64, gi_tg:np.float64, gi_vol:np.float64, gi_peff:np.float64, gi_Eh:np.float64,
                                 # PK parameters
                                 V_central_L:np.float64, k10_s:np.float64, k12_s:np.float64, k13_s:np.float64, k21_s:np.float64, k31_s:np.float64,
                                 block_ET_GI:np.bool_):
    """
    Mechanistic dissolution model with lumping for small particles.
    Units: time=s, amount=pmol, volume=mL, length=cm
    """
    y = np.maximum(y, np.float64(0.0))
    y_lung = y[:num_lung_states]
    y_gi = y[num_lung_states:-3]
    y_pk = y[-3:]
    
    dydt = np.zeros_like(y)
    dydt_lung = np.zeros(num_lung_states)
    dydt_gi = np.zeros(num_gi_states)
    
    # Unpack central PK compartments
    Cp_central, Cp_p1, Cp_p2 = y_pk # Amount in pmol
    Cp_total_pmol_per_L = max(np.float64(0.0), Cp_central) / V_central_L if V_central_L > np.float64(0.0) else np.float64(0.0)
    
    # --- Lung Model Calculations ---
    total_systemic_absorption_lung = np.float64(0.0)
    mcc_to_gi = np.float64(0.0)
    
    for r_idx in range(len(region_offsets)):
        # Get offsets and layer counts for the current region
        offset = region_offsets[r_idx]
        n_epi = n_epi_layers[r_idx]
        
        n_states = 1 + 2 * n_epi + 2 + 2 * num_deposition_bins
        
        # ---- Disable ET region ----
        if block_ET_GI and r_idx == 0:
            dydt_lung[offset : offset + n_states] = 0.0
            continue
        
        # Unpack states for the current region (amounts in pmol)
        Xg_ELF_total = y_lung[offset]
        epi_start_index = offset + 1
        Xg_Epi_local_shallow = y_lung[epi_start_index : epi_start_index + n_epi]
        Xg_Epi_local_deep = y_lung[epi_start_index + n_epi : epi_start_index + 2 * n_epi]
        
        tissue_offset = epi_start_index + 2 * n_epi
        Xg_Tissue_shallow = y_lung[tissue_offset]
        Xg_Tissue_deep = y_lung[tissue_offset + 1]
        
        solid_offset = tissue_offset + 2
        Rsolid_bin = y_lung[solid_offset : solid_offset + num_deposition_bins]
        numpart_bin = y_lung[solid_offset + num_deposition_bins : solid_offset + 2 * num_deposition_bins]
        
        # Use pre-calculated parameters
        A_elf_r, V_elf_r, V_epi_r, Q_g_r, tg_r = A_elf[r_idx], V_elf[r_idx], V_epi[r_idx], Q_g_region[r_idx], tg_region[r_idx]
        fu_epi_calc_r, fu_tissue_calc_r = fu_epi_calc[r_idx], fu_tissue_calc[r_idx]
        V_epi_layer = V_epi_r / n_epi if n_epi > np.float64(0.0) else np.float64(0.0)

        # Conditional fluxes for tissue
        if cell_bind == 0:
            F_in_tissue = K_out_tissue[r_idx] * (1/(V_frac_g * fu[2]) - 1.0) * (Xg_Tissue_shallow)
            F_out_tissue = K_out_tissue[r_idx] * Xg_Tissue_deep
        else:  
            
            F_in_tissue = K_in_tissue[r_idx] * Xg_Tissue_shallow
            F_out_tissue = K_out_tissue[r_idx] * Xg_Tissue_deep

        # Concentrations (in pmol/mL)
        Cg_ELF_unbound = (Xg_ELF_total * fu[0]) / V_elf_r if V_elf_r > np.float64(0.0) else np.float64(0.0)
        Cg_Epi_local_unbound = (Xg_Epi_local_shallow * fu_epi_calc_r) / V_epi_layer if V_epi_layer > np.float64(0.0) else np.zeros(n_epi)
        Cg_Tissue_unbound = (Xg_Tissue_shallow * fu_tissue_calc_r) / V_tissue[r_idx] if V_tissue[r_idx] > np.float64(0.0) else np.float64(0.0)

        # Dissolution with Lumping Logic
        Rsolid_bin_floored = np.maximum(1e-9, Rsolid_bin)
        dRsolid_dt = -D * Vm / Rsolid_bin_floored * (Sg - Cg_ELF_unbound) # Vm is in cm^3/pmol
        # org bin size
        initial_rsolid = np.array([0.1643, 0.2063, 0.2591, 0.3255, 0.4088, 0.5135, 0.6449, 0.8100, 1.0173, 1.2777, 1.6048, 2.0156, 2.5315, 3.1795, 3.9933, 5.0155, 6.2993, 7.9116, 9.9368]) * np.float64(1e-4 / 2.0)
        #normed_radius = Rsolid_bin / initial_rsolid
        #small_particle_mask = normed_radius <= np.float64(0.1) #dissolution_cutoff_radius
        
        small_particle_mask = Rsolid_bin <= dissolution_cutoff_radius
        
        dissolution_per_bin = -dRsolid_dt * (4 * np.pi * Rsolid_bin_floored**2 / Vm) * np.maximum(np.float64(0.0), numpart_bin) # pmol/s
        dissolution_per_bin[small_particle_mask] = np.float64(0.0)

        # Lumping: convert particle mass to pmol and apply rate
        mass_to_lump_pg = numpart_bin * (4/3 * np.pi * Rsolid_bin**3) * MM / Vm # Mass in pg
        lumping_flux_per_bin = k_lump * mass_to_lump_pg / MM # pmol/s
        lumping_flux_per_bin[~small_particle_mask] = np.float64(0.0)
        total_lumping_flux = np.sum(lumping_flux_per_bin)
        
        dRsolid_dt[small_particle_mask] = np.float64(0.0)
        total_dissolution = np.sum(dissolution_per_bin)
        
        # Flows (pmol/s)
        elf_2_epi = 2.0 * Pg_in[r_idx] * A_elf_r * Cg_ELF_unbound
        epi_2_elf = 2.0 * Pg_out[r_idx] * A_elf_r * Cg_Epi_local_unbound[0]
        epi_2_tissue = 2.0 * Pg_out[r_idx] * A_elf_r * Cg_Epi_local_unbound[-1]
        tissue_2_epi = 2.0 * Pg_in[r_idx] * A_elf_r * Cg_Tissue_unbound
        net_para_flux = 2.0 * Pg_para[r_idx] * A_elf_r * (Cg_ELF_unbound - Cg_Tissue_unbound)
        
        blood_2_tissue = Q_g_r * BP * (Cp_total_pmol_per_L / 1000.0) # Convert L to mL for conc.
        tissue_2_blood = Q_g_r * BP * Cg_Tissue_unbound / fu[3]
        
        # Epithelium Layer Derivatives
        dXg_Epi_local_shallow_dt = np.zeros(n_epi)
        F_in_epi = K_in_epi[r_idx] * Xg_Epi_local_shallow
        F_out_epi = K_out_epi[r_idx] * Xg_Epi_local_deep
        dXg_Epi_local_deep_dt = F_in_epi - F_out_epi

        for epi_id in range(n_epi):
            if n_epi == 1:
                F_epi_flow_in = elf_2_epi + tissue_2_epi
                F_epi_flow_out = epi_2_elf + epi_2_tissue
            else:
                if epi_id == 0:
                    F_epi_flow_in = elf_2_epi + (2.0 * Pg_in[r_idx] * A_elf_r * Cg_Epi_local_unbound[epi_id + 1])
                    F_epi_flow_out = epi_2_elf + (2.0 * Pg_in[r_idx] * A_elf_r * Cg_Epi_local_unbound[epi_id])
                elif epi_id == n_epi - 1:
                    F_epi_flow_in = tissue_2_epi + (2.0 * Pg_in[r_idx] * A_elf_r * Cg_Epi_local_unbound[epi_id - 1])
                    F_epi_flow_out = epi_2_tissue + (2.0 * Pg_in[r_idx] * A_elf_r * Cg_Epi_local_unbound[epi_id])
                else:
                    F_epi_flow_in = (2.0 * Pg_in[r_idx] * A_elf_r * Cg_Epi_local_unbound[epi_id - 1]) + (2.0 * Pg_in[r_idx] * A_elf_r * Cg_Epi_local_unbound[epi_id + 1])
                    F_epi_flow_out = (2.0 * Pg_in[r_idx] * A_elf_r * Cg_Epi_local_unbound[epi_id]) * 2.0
            dXg_Epi_local_shallow_dt[epi_id] = F_epi_flow_in - F_epi_flow_out - F_in_epi[epi_id] + F_out_epi[epi_id]

        # Other compartment derivatives (all in pmol/s)
        MCC_dissolved = Cg_ELF_unbound * V_elf_r / tg_r if tg_r > np.float64(0.0) else np.float64(0.0)
        dnumpart_lump_term = k_lump * numpart_bin
        dnumpart_lump_term[~small_particle_mask] = np.float64(0.0)
        dnumpart_dt = -np.maximum(np.float64(0.0), numpart_bin) / tg_r - dnumpart_lump_term if tg_r > np.float64(0.0) else -dnumpart_lump_term
        
        dXg_ELF_total_dt = -MCC_dissolved + total_dissolution + total_lumping_flux - elf_2_epi + epi_2_elf - net_para_flux
        dXg_Tissue_shallow_dt = epi_2_tissue - tissue_2_epi - tissue_2_blood + blood_2_tissue - F_in_tissue + F_out_tissue + net_para_flux
        dXg_Tissue_deep_dt = F_in_tissue - F_out_tissue
        
        # MCC logic, debug try using dnumpartdt
        
        #MCC_solid = -np.sum(dnumpart_dt * (4/3 * np.pi * Rsolid_bin**3) / Vm) # Amount in pmol
        
        # MCC logic, org
        amount_pmol_solid_per_bin = numpart_bin * (4/3 * np.pi * Rsolid_bin**3) / Vm # Amount in pmol
        MCC_solid = np.sum(amount_pmol_solid_per_bin / tg_r) if tg_r > np.float64(0.0) else np.float64(0.0)
        
        if r_idx == 2: # 'bb_b'
            bb_offset = region_offsets[1]
            dydt_lung[bb_offset] += MCC_dissolved + MCC_solid
        if r_idx == 0 or r_idx == 1: # 'ET' or 'BB'
            mcc_to_gi += MCC_dissolved + MCC_solid

        total_systemic_absorption_lung += (tissue_2_blood - blood_2_tissue)
        
        # Assign derivatives
        dydt_lung[offset] = dXg_ELF_total_dt
        dydt_lung[epi_start_index : epi_start_index + n_epi] = dXg_Epi_local_shallow_dt
        dydt_lung[epi_start_index + n_epi : epi_start_index + 2 * n_epi] = dXg_Epi_local_deep_dt
        dydt_lung[tissue_offset] = dXg_Tissue_shallow_dt
        dydt_lung[tissue_offset + 1] = dXg_Tissue_deep_dt
        dydt_lung[solid_offset : solid_offset + num_deposition_bins] = dRsolid_dt
        dydt_lung[solid_offset + num_deposition_bins : solid_offset + 2 * num_deposition_bins] = dnumpart_dt
        
    # --- GI Tract Model Calculations ---
    if block_ET_GI:
        dydt_gi[:] = 0.0
        mcc_to_gi = 0.0  # Also block anything that sends solid from lung to GI
        gut_abs_net = 0.0
    else:
        Xg_gi = np.maximum(np.float64(0.0), y_gi) # pmol
        Cg_gi = Xg_gi / gi_vol # pmol/mL

        # Two-way permeability-limited transport for GI tract
        flux_gut_to_blood = gi_peff * gi_area * Cg_gi * fu[2] * BP / fu[3]
        flux_blood_to_gut = gi_peff * gi_area * BP * (Cp_total_pmol_per_L / 1000.0) # Convert L to mL
        absorption_gi_comp = flux_gut_to_blood - flux_blood_to_gut # pmol/s

        total_absorption_gi = np.sum(absorption_gi_comp)
        gut_abs_net = total_absorption_gi * (np.float64(1.0) - gi_Eh)
        
        transit_in = np.zeros(num_gi_states)
        transit_in[1:] = Xg_gi[:-1] / gi_tg[:-1]
        transit_in[0] = mcc_to_gi
        
        transit_out = Xg_gi / gi_tg
        dydt_gi = transit_in - transit_out - absorption_gi_comp
    
    # --- PK Model Calculations ---
    total_systemic_absorption = total_systemic_absorption_lung + gut_abs_net # pmol/s
    
    dCp_central_dt = total_systemic_absorption - (k10_s + k12_s + k13_s) * max(np.float64(0.0), Cp_central) + k21_s * max(np.float64(0.0), Cp_p1) + k31_s * max(np.float64(0.0), Cp_p2)
    dCp_p1_dt = k12_s * max(np.float64(0.0), Cp_central) - k21_s * max(np.float64(0.0), Cp_p1)
    dCp_p2_dt = k13_s * max(np.float64(0.0), Cp_central) - k31_s * max(np.float64(0.0), Cp_p2)
    
    # Combine all derivatives
    dydt[:num_lung_states] = dydt_lung
    dydt[num_lung_states:-3] = dydt_gi
    dydt[-3:] = [dCp_central_dt, dCp_p1_dt, dCp_p2_dt]
    
    return dydt

    



class PBPK_Model_Virtual_Trial:
    """PBPK model that accepts subject and API parameters for virtual trials."""
    def __init__(self, subject_phys, api, settings):
        self.physio, self.api, self.simtime = subject_phys, api, settings.simtime
        self.region = self.physio.region  # ['ET', 'BB', 'bb_b', 'Al']
        self.gi_tract = GITract(self.api)  # GI tract
        self.deposition = settings.particle_settings
        self.dissolution_cutoff_radius, self.k_lump = settings.pbbm_settings.dissolution_cutoff_radius, settings.pbbm_settings.k_lump #1E-6, 5E-4 * 1e6   # pmol
        
        self.rtol = settings.pbbm_settings.rtol #1e-4 #1e-5
        self.atol = settings.pbbm_settings.atol #1e-8 #1e-12
        self.block_ET_GI = settings.pbbm_settings.block_ET_GI
        self.precalculate_params_for_numba()
    def precalculate_params_for_numba(self):
        self.numba_params = {}
        num_regions = len(self.region)
        
        # Convert density from g/m^3 to g/cm^3
        density_g_cm3 = self.api.density / 1e6 
        # Calculate molar volume in cm^3/mol, then convert to cm^3/pmol
        Vm_cm3_mol = self.api.MM / density_g_cm3
        Vm_cm3_pmol = Vm_cm3_mol / 1e12

        self.numba_params.update({'num_deposition_bins': self.deposition.numbin_pbpk, 'dissolution_cutoff_radius': self.dissolution_cutoff_radius, 'k_lump': self.k_lump, 'A_elf': np.zeros(num_regions), 'V_elf': np.zeros(num_regions), 'V_epi': np.zeros(num_regions), 'Q_g_region': np.zeros(num_regions), 'tg_region': np.zeros(num_regions), 'Pg_in': np.zeros(num_regions), 'Pg_out': np.zeros(num_regions), 'Pg_para': np.zeros(num_regions), 'V_tissue': np.array([self.physio.V_tissue[r] for r in self.region]), 'fu_epi_calc': np.zeros(num_regions), 'fu_tissue_calc': np.zeros(num_regions), 'K_in_epi': np.zeros(num_regions), 'K_out_epi': np.zeros(num_regions), 'K_in_tissue': np.zeros(num_regions), 'K_out_tissue': np.zeros(num_regions)})
        self.numba_params.update({'n_epi_layers': np.array([self.physio.n_epi_layer[r] for r in self.region], dtype=np.int32), 'max_n_epi': max(self.physio.n_epi_layer.values()), 'Vm': Vm_cm3_pmol, 'Sg': self.api.solub / self.api.MM, 'D': self.api.D, 'V_frac_g': self.physio.V_frac_g, 'MM': self.api.MM})
        frc_factor = (self.physio.FRC / self.physio.FRC_ref)**(1/3)
        scaling_factor = {'ET': 1.0, 'BB': frc_factor,'bb_b' : frc_factor, 'Al': frc_factor}
        for i, r in enumerate(self.region):
            self.numba_params['A_elf'][i] = (self.physio.A_elf_ref[r] - self.physio.extra_area_ref[r]) * scaling_factor[r]**2 + self.physio.extra_area_ref[r] * scaling_factor[r]**3
            self.numba_params['V_elf'][i] = self.numba_params['A_elf'][i] * self.physio.d_elf[r]
            self.numba_params['V_epi'][i] = self.numba_params['A_elf'][i] * self.physio.d_epi[r]
            self.numba_params['Q_g_region'][i] = self.physio.Q_g[r] * scaling_factor[r]**3
            self.numba_params['tg_region'][i] = self.physio.tg[r]
            self.numba_params['V_tissue'][i] = self.physio.V_tissue[r] * scaling_factor[r]**3 # Already an array, but good practice
            self.numba_params['Pg_in'][i] = self.api.Peff['In'] * self.api.pscale[r]['In']
            self.numba_params['Pg_out'][i] = self.api.Peff['Out'] * self.api.pscale[r]['Out']
            self.numba_params['Pg_para'][i] = self.api.Peff_para * self.api.pscale_para[r]
            self.numba_params['K_in_epi'][i] = self.api.K_in['Epithelium'] * self.api.pscale_Kin[r]['Epithelium']
            self.numba_params['K_out_epi'][i] = self.api.K_out['Epithelium'] * self.api.pscale_Kout[r]['Epithelium']
            self.numba_params['K_in_tissue'][i] = self.api.K_in['Tissue'] * self.api.pscale_Kin[r]['Tissue']
            self.numba_params['K_out_tissue'][i] = self.api.K_out['Tissue'] * self.api.pscale_Kout[r]['Tissue']
            
            if self.api.K_in['Epithelium'] > 0 and self.api.K_out['Epithelium'] > 0:
                self.numba_params['fu_epi_calc'][i] = self.api.fu['Epithelium'] * (1 + self.api.K_in['Epithelium'] / self.api.K_out['Epithelium'])
            else:
                self.numba_params['fu_epi_calc'][i] = self.api.fu['Epithelium']

            if self.api.K_in['Tissue'] > 0 and self.api.K_out['Tissue'] > 0:
                if self.api.cell_bind == 0:
                    self.numba_params['fu_tissue_calc'][i] = self.api.fu['Tissue'] * (1 + self.api.K_in['Tissue'] / self.api.K_out['Tissue'])
                else:
                    self.numba_params['fu_tissue_calc'][i] = 1.0 / self.physio.V_frac_g
            else:
                self.numba_params['fu_tissue_calc'][i] = self.api.fu['Tissue']


        self.numba_params['fu'] = np.array([self.api.fu['ELF'], self.api.fu['Epithelium'], self.api.fu['Tissue'], self.api.fu['Plasma']])
        self.numba_params['cell_bind'] =  self.api.cell_bind
        self.numba_params['BP'] = self.api.BP
        self.numba_params['V_central_L'] = self.api.V_central_L
        self.numba_params.update({'k12_s': self.api.k12_h / 3600, 'k21_s': self.api.k21_h / 3600, 'k13_s': self.api.k13_h / 3600, 'k31_s': self.api.k31_h / 3600, 'k10_s': (self.api.CL_h / 3600) / self.api.V_central_L})
        self.numba_params['block_ET_GI'] = self.block_ET_GI #True if self.study_type.lower() == 'charcoal' else False

    def run_simulation(self, initial_dose_pmol):
        """Sets up and runs the ODE solver with amounts in pmol."""
        self.num_states_per_region, self.region_offsets, self.n_epi_layers, offset = {}, np.zeros(len(self.region), dtype=np.int32), np.zeros(len(self.region), dtype=np.int32), 0
        for cnt, r in enumerate(self.region):
            n_epi = self.physio.n_epi_layer[r]
            n_states = 1 + 2 * n_epi + 2 + 2 * self.deposition.numbin_pbpk
            self.num_states_per_region[r], self.region_offsets[cnt], self.n_epi_layers[cnt] = n_states, offset, n_epi
            offset += n_states
        self.num_lung_states, self.num_pk_states, self.num_gi_states = offset, 3, len(self.gi_tract.gi_area)
        y0 = np.zeros(self.num_lung_states + self.num_pk_states + self.num_gi_states)
        for i, r in enumerate(self.region):
            offset, n_epi = self.region_offsets[i], self.n_epi_layers[i]
            solid_offset = offset + 1 + 2 * n_epi + 2
            y0[solid_offset : solid_offset + self.deposition.numbin_pbpk] = self.deposition.binsize_pbpk
            
            # mass in mol = vol * density(g/cm3) / MM(g/mol)
            # mass in pmol = mass_in_mol * 1e12
            mass_per_particle_mol = (4/3 * np.pi * self.deposition.binsize_pbpk**3) * (self.api.density / 1e6) / self.api.MM
            mass_per_particle_pmol = mass_per_particle_mol * 1e12
            
            num_particles = np.divide(initial_dose_pmol[r] * self.deposition.massfraction_pbpk, mass_per_particle_pmol, out=np.zeros_like(mass_per_particle_pmol), where=mass_per_particle_pmol != 0)
            y0[solid_offset + self.deposition.numbin_pbpk : solid_offset + 2 * self.deposition.numbin_pbpk] = num_particles
        
        t_span = [0, self.simtime * 3600]
        t_eval = np.linspace(t_span[0], t_span[1], 300)
        
        args = (
            # State dimensions and helpers
            self.num_lung_states, self.num_gi_states, self.numba_params['num_deposition_bins'],
            self.region_offsets, self.n_epi_layers, self.numba_params['max_n_epi'],
            # Physico-chemical parameters
            self.numba_params['Vm'], self.numba_params['Sg'], self.numba_params['D'], self.numba_params['MM'],
            self.numba_params['dissolution_cutoff_radius'], self.numba_params['k_lump'],
            # General physiological parameters
            self.numba_params['fu'], self.numba_params['BP'], self.numba_params['V_frac_g'],
            # Precalculated Lung Geometry/Flow arrays
            self.numba_params['A_elf'], self.numba_params['V_elf'], self.numba_params['V_epi'],
            self.numba_params['V_tissue'], self.numba_params['Q_g_region'], self.numba_params['tg_region'],
            # Precalculated Lung Permeability/Binding arrays
            self.numba_params['Pg_in'], self.numba_params['Pg_out'], self.numba_params['Pg_para'],
            self.numba_params['K_in_epi'], self.numba_params['K_out_epi'],
            self.numba_params['K_in_tissue'], self.numba_params['K_out_tissue'],
            self.numba_params['fu_epi_calc'], self.numba_params['fu_tissue_calc'],
            self.numba_params['cell_bind'],
            # GI parameters
            self.gi_tract.gi_area, self.gi_tract.gi_tg, self.gi_tract.gi_vol, self.gi_tract.peff, self.gi_tract.Eh,
            # PK parameters
            self.numba_params['V_central_L'], self.numba_params['k10_s'], self.numba_params['k12_s'],
            self.numba_params['k13_s'], self.numba_params['k21_s'], self.numba_params['k31_s'],
            self.numba_params['block_ET_GI']
        )

        return solve_ivp(lambda t, y: ode_system_numba_mechanistic(t, y, *args), t_span, y0, method='BDF', t_eval=t_eval, rtol=self.rtol, atol=self.atol)
