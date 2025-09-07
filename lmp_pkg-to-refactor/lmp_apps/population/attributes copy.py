import numpy as np
import pandas as pd
from scipy import stats
import random
import copy
import os

from ..lung_deposition.Parameters_Settings import *
from .variability import Variability
from ..api.api import API

class InhalationManeuver:
    """Holds inhalation maneuver parameters, sampled for a subject."""
    def __init__(self, frc_subject, variability, frc_ref=3300):
        # Base mean values for inhalation parameters
        pifr_mean = 30.0
        risetime_mean = 0.4

        # Get sigmas from the variability object, which will be 0 if variability is disabled
        pifr_sigma = variability.sigma_log['Inter']['Inhalation']['PIFR']
        risetime_sigma = variability.sigma_log['Inter']['Inhalation']['RiseTime']

        # Generate multiplicative factors. If sigma is 0, factor will be 1.
        pifr_factor = np.random.lognormal(0, pifr_sigma)
        risetime_factor = np.random.lognormal(0, risetime_sigma)
        
        # Sample parameters using the new multiplicative factor approach
        self.inhaled_volume_L = stats.norm.rvs(loc=2.0, scale=0.0) * (frc_subject / frc_ref)
        self.pifr_Lpm = pifr_mean * pifr_factor
        self.rise_time_s = risetime_mean * risetime_factor
        
        # Other parameters with no variability
        self.hold_time_s = stats.norm.rvs(loc=0.5, scale=0.0)
        self.breath_hold_time_s = stats.norm.rvs(loc=30, scale=0.0)
        self.exhalation_flow_Lpm = stats.norm.rvs(loc=30, scale=0.0)
        self.bolus_volume_ml = stats.norm.rvs(loc=200, scale=0.0)
        self.bolus_delay_s = stats.norm.rvs(loc=0, scale=0.0)
    
      
    def inhale_profile(self):
        Inhaled_Vol = self.inhaled_volume_L
        PIFR = self.pifr_Lpm
        Rise_Time = self.rise_time_s
        
        PIFR = PIFR / 60.0 # L/s
        if PIFR <= 0 or Rise_Time <= 0:
            Hold_Time = Inhaled_Vol / 1e-6 if PIFR <=0 else Inhaled_Vol/PIFR # Avoid division by zero
        else:
            Hold_Time = (Inhaled_Vol - 2 * (0.5 * PIFR * Rise_Time)) / PIFR

        if Hold_Time < 0: Hold_Time = 0
        
        Inhaled_Duration = Hold_Time + 2 * Rise_Time
        if Inhaled_Duration <= 0: Inhaled_Duration = 1e-6 # Avoid division by zero

        slope = PIFR / Rise_Time if Rise_Time > 0 else 0
        
        time_points = np.linspace(0, Inhaled_Duration, N_steps)
        flowrate = np.zeros_like(time_points)

        # Ramp up
        mask1 = time_points < Rise_Time
        flowrate[mask1] = time_points[mask1] * slope

        # Hold
        mask2 = (time_points >= Rise_Time) & (time_points < (Rise_Time + Hold_Time))
        flowrate[mask2] = PIFR

        # Ramp down
        mask3 = time_points >= (Rise_Time + Hold_Time)
        flowrate[mask3] = PIFR - (time_points[mask3] - (Rise_Time + Hold_Time)) * slope
        flowrate = np.maximum(0, flowrate) # Ensure no negative flow

        flowfile = np.zeros((len(flowrate),2))
        flowfile[:,0] = time_points
        flowfile[:,1] = np.array(flowrate) * 60.0 # Back to L/min
        
        self.flowprofile = flowfile
    

    
class SubjectPhysiology:
    """Holds physiological parameters for a single virtual subject."""
    def __init__(self, subject_id, variability, population = 'Healthy', enable_variability=True):
        self.id = subject_id
        self.region = ['ET', 'BB', 'bb_b', 'Al']
        
        frc_mean, frc_std = 3300, 600
        self.FRC = random.normalvariate(frc_mean, frc_std if enable_variability else 0)

        self.FRC_ref = 2999.6 
        self.population = population
        # Base Lung Geometry (generation-based, as needed for deposition model)
        cols = ['Multiplicity', 'Alveoli Vol', 'Length', 'Diameter', 'Gravity Angle', 'Branching Angle', 'Expansion Fraction']
        # In a real scenario, this would be loaded from a file (e.g., 'Healthy' sheet)
        # Using random data as a placeholder for the geometry dataframe.
        pop_options = ['Healthy', 'COPD_Opt_1','COPD_Opt_2','COPD_Opt_3','COPD_Opt_4','COPD_Opt_5' ]
        ref_dict = {}
        for pop in pop_options:
            pkg_dir = os.path.dirname(__file__)
            ref_phys_path = os.path.join(pkg_dir, '..', 'data', 'lung', 'Ref_Lung_Physiology.xlsx')
            ref = pd.read_excel(ref_phys_path, sheet_name=pop)
            
            ref.loc[0, 'Gravity Angle'] = 0.0
            ref.loc[0, 'Branching Angle'] = 0.0
            ref_dict[pop] = ref
        self.physiology_df = ref_dict[self.population][cols]
        self.physiology_df.iloc[0, [4, 5]] = 0.0

      
        
        # PBPK model parameters (regional, not generational)
        self.A_elf_ref = {'ET': 50.2, 'BB': 305.51, 'bb_b': 2772.4, 'Al': 1.43E+06}
        self.extra_area_ref = {'ET': 0, 'BB': 0, 'bb_b': 0, 'Al': 1330900.0}
        self.d_elf = {'ET': 0.027, 'BB': 0.0011, 'bb_b': 6.0E-4, 'Al': 7.0E-6}
        self.d_epi = {'ET': 0.06, 'BB': 0.0055812, 'bb_b': 0.0015, 'Al': 3.6E-5}
        self.V_tissue = {'ET': 1.004, 'BB': 20.9, 'bb_b': 7.6, 'Al': 432.6}
        self.Q_g = {'ET': 2.4, 'BB': 0.302333333, 'bb_b': 0.1685, 'Al': 86.65333333}
        self.tg = {'ET': 864.0, 'BB': 6855.0, 'bb_b': 62580.0, 'Al': 6.0E51}
        self.V_frac_g = 0.2
        self.n_epi_layer = {'ET': 1, 'BB': 1, 'bb_b': 1, 'Al': 1}
        
        # Inhalation maneuver is now sampled per subject
        self.inhalation_maneuver = InhalationManeuver(self.FRC, variability, self.FRC_ref)
        self.MT_size = 'medium'
        # Variability factors for PK/PD parameters
        self.inter_subject_factors = {
            'ET': {api: np.random.lognormal(0, sigma) for api, sigma in variability.sigma_log['Inter']['ET'].items()},
            'CL': {api: np.random.lognormal(0, sigma) for api, sigma in variability.sigma_log['Inter']['CL'].items()},
            'Eh': {api: np.random.lognormal(0, sigma) for api, sigma in variability.sigma_log['Inter']['Eh'].items()},
        }
        self.variability = variability

class GITract:
    """Holds physiological parameters of the gastrointestinal tract."""
    def __init__(self, api):
        self.num_comp = 9
        gi_area = {'BD': np.array([0, 0, 0, 0, 300, 300, 144.72, 280.02, 41.77]), 'GP': np.array([0, 80, 400, 422.82, 126.07, 226.32, 144.72, 28.02, 41.77]), 'FF': np.array([0, 0, 0, 0, 250, 150, 150, 28.02, 41.77])}
        gi_tg = {'BD': np.array([60.0, 600.0, 600.0, 600.0, 3000.0, 3600.0, 1044.0, 15084.0, 45252.0]), 'GP': np.array([12600.0, 5400.0, 3348.0, 2664.0, 2088.0, 1512.0, 1044.0, 15084.0, 45252.0]), 'FF': np.array([600.0, 600.0, 600.0, 600.0, 2088.0, 1512.0, 1044.0, 15084.0, 45252.0])}
        gi_vol = {'BD': np.array([46.56, 41.56, 154.2, 122.3, 94.29, 70.53, 49.8, 47.49, 50.33]), 'GP': np.array([46.56, 41.56, 154.2, 122.3, 94.29, 70.53, 49.8, 47.49, 50.33]), 'FF': np.array([46.56, 41.56, 154.2, 122.3, 94.29, 70.53, 49.8, 47.49, 50.33])}
        self.gi_area = gi_area.get(api.name, gi_area['BD'])
        self.gi_tg = gi_tg.get(api.name, gi_tg['BD'])
        self.gi_vol = gi_vol.get(api.name, gi_vol['BD'])
        self.peff = api.Peff_GI
        self.Eh = api.Eh / 100.0
