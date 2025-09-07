import numpy as np
from numba import jit, njit, prange

from .Parameters_Settings import *
from .Lung_deposition_packageVariable_DN_edit import Lung_deposition_packageVariable
from .helper_functions import *

@njit(parallel = True, cache = True)
def parallel_calculate_regional_deposition(n_subject, Scint_Keys, sub_lung_all, v_lung, psd_all, mtdepo_all, flowrate_all_sub, breath_hold_time_all, Flow_Exh_all, BolusVolume_all, Bolus_Delay_all):
    
    res = np.zeros((n_subject,10))
    
    deposition_gen = np.zeros((n_subject,N_GIcomps_LungGens_max+1))
    for id in prange(n_subject):
        
        
        depo = Lung_deposition_packageVariable(sub_lung_all[id*N_GIcomps_LungGens_max:(id+1)*N_GIcomps_LungGens_max, :], v_lung[id], psd_all[id*N_bins_max:(id+1)*N_bins_max, :], mtdepo_all[id], flowrate_all_sub[id*(N_Q_max-1):(id+1)*(N_Q_max-1), :], breath_hold_time_all[id], Flow_Exh_all[id], BolusVolume_all[id], Bolus_Delay_all[id])
        for i in prange(N_GIcomps_LungGens_max+1):
            deposition_gen[id, i] = depo[i]
        
        cip = np.dot(depo[:-1], Scint_Keys) / 100
        ET = depo[0]
        BB = np.sum(depo[1:10])
        bb_b = np.sum(depo[10:17])
        Al =  np.sum(depo[17:-1])
        res[id, :] = np.array((cip[0], cip[1], cip[2], ET, BB, bb_b, Al, depo[-1], Al/BB, cip[0] / cip[2]))
    return res, deposition_gen


def simulation_params_array(n_subject, geometry_all, flowrate_all, frc_all, psd, mtdepo, breath_hold_time, Flow_Exh, BolusVolume, Bolus_Delay):
    
    sub_lung_all = np.zeros((25 * n_subject, 8))
    v_lung = np.zeros(n_subject)
    mtdepo_all = np.zeros(n_subject)
    psd_all = np.zeros((N_bins_max * n_subject, 3))
    flowrate_all_sub = np.zeros(((N_Q_max-1) * n_subject, 2))
    breath_hold_time_all = np.zeros(n_subject)
    Flow_Exh_all = np.zeros(n_subject)
    BolusVolume_all = np.zeros(n_subject)
    Bolus_Delay_all = np.zeros(n_subject)
    for id in np.arange(n_subject):
        sub_lung = scale_lung(geometry_all[id], frc_all[id], N_GIcomps_LungGens_max)
        #v_lung = frc_all[id]
        flowrate_s = flowrate_all[id].copy()
        flowrate_s[:,1] =  flowrate_s[:,1] * 1e-3 / 60
    
        sub_lung_all[id*25:(id+1)*25, :] = sub_lung
        v_lung[id] = frc_all[id]
        psd_all[id*N_bins_max:(id+1)*N_bins_max, :] = psd[id].copy()
        v_lung[id] = frc_all[id]
        mtdepo_all[id] = mtdepo[id]
        #flowrate_all_sub[id*59:(id+1)*59, :] = flowrate_s
        flowrate_all_sub[id*(N_Q_max-1):(id+1)*(N_Q_max-1), :] = flowrate_s
        breath_hold_time_all[id] = breath_hold_time[id]
        Flow_Exh_all[id] = Flow_Exh[id] / 1000 / 60
        BolusVolume_all[id] = BolusVolume[id] / 1e6
        Bolus_Delay_all[id] = Bolus_Delay[id]
    return sub_lung_all, v_lung, psd_all, mtdepo_all, flowrate_all_sub, breath_hold_time_all, Flow_Exh_all, BolusVolume_all, Bolus_Delay_all

def simulation_params(id, mtdepo, breath_hold_time, Flow_Exh, BolusVolume):
    
    sub_lung = scale_lung(geometry_all[id], frc_all[id], N_GIcomps_LungGens_max)
    v_lung = frc_all[id]
    flowrate_s = flowrate_all[id].copy()
    flowrate_s[:,1] =  flowrate_s[:,1] * 1e-3 / 60
    
    return [sub_lung, v_lung, psd, mtdepo, flowrate_s, breath_hold_time, Flow_Exh, BolusVolume]

def calculate_regional_deposition(id, Scint_Keys):
    sub_lung = scale_lung(geometry_all[id], frc_all[id], N_GIcomps_LungGens_max)
    v_lung = frc_all[id]
    flowrate_s = flowrate_all[id].copy()
    flowrate_s[:,1] =  flowrate_s[:,1] * 1e-3 / 60
    depo = Lung_deposition_packageVariable(sub_lung, v_lung, psd, mtdepo, flowrate_s, breath_hold_time, Flow_Exh, BolusVolume)
    cip = np.dot(depo[:-1], Scint_Keys)
    #[deposition_tot_res[0],np.sum(deposition_tot_res[1:10]), np.sum(deposition_tot_res[10:17]),np.sum(deposition_tot_res[17:-1])]
    ET = depo[0]
    BB = np.sum(depo[1:10])
    bb_b = np.sum(depo[10:17])
    Al =  np.sum(depo[17:-1])
    return cip[0], cip[1], cip[2], ET, BB, bb_b, Al, Al/BB, cip[0] / cip[1]