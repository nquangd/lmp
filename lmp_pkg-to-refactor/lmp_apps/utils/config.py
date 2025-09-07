import numpy as np



        
class pbbm_settings:
    def __init__(self, study_type = 'NON_CHARCOAL'):
        # --- Configuration ---
        #ENABLE_VARIABILITY = False
        self.dissolution_cutoff_radius = 1e-6
        self.k_lump = 5e-4 * 1e6  # pmol/s
        self.rtol = 1e-4
        self.atol = 1e-8
        self.block_ET_GI = True if study_type.lower() == 'charcoal' else False

class deposition_settings: # unused for now
    def __init__(self):
        self.N_GIcomps_LungGens_max = 25
        self.N_bins_max = 20
        self.min_bin_size = 0.7 #um
        self.max_bin_size =  15 # um
        self.N_slices_max = 200
        self.N_Q_max = 60 # always 60 since working # 120
        self.N_steps = self.N_Q_max - 1 #119
        self.const_Flow = 500e-6
        self.V_slice_const = 20e-6
        

class particle_settings:
    """
    Manages the entire lung deposition calculation process by encapsulating
    helper methods and the main parallelized calculation function.
    """
    def __init__(self):
        # Fixed parameters for PBPK particle model
        self.numbin_pbpk = 19
        self.binsize_pbpk = np.array([0.1643, 0.2063, 0.2591, 0.3255, 0.4088, 0.5135, 0.6449, 0.8100, 1.0173, 1.2777, 1.6048, 2.0156, 2.5315, 3.1795, 3.9933, 5.0155, 6.2993, 7.9116, 9.9368])  * np.float64(1e-4 / 2.0)
        self.massfraction_pbpk = np.array([0.185, 0.4513, 0.9911, 1.96, 3.489, 5.594, 8.075, 10.5, 12.28, 12.95, 12.28, 10.5, 8.075, 5.594, 3.489, 1.96, 0.9911, 0.4513, 0.1842]) / np.float64(100.0)

        
class simulation_settings:
    def __init__(self, ENABLE_VARIABILITY = False, simtime = 24.0, study_type = 'NON_CHARCOAL' ):
        # --- Configuration ---
        self.ENABLE_VARIABILITY = ENABLE_VARIABILITY
        self.simtime = simtime
        self.pbbm_settings = pbbm_settings(study_type)
        self.deposition_settings = deposition_settings()
        self.particle_settings = particle_settings()
        