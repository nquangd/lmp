import numpy as np
from numba import int32, int64, float64, void, types, typed, deferred_type, jit, njit    # import the types

from .Lung_classes import *
from .PhysicalProp_classes import *
from .Deposition_classes import *

@jit(cache=True, nopython=True)
def Convert_k_undiss_2_X_undiss_single(pi4_15Vm:int,
                                       Rmax:float,
                                       Rmin:float,
                                       k_undiss:float):
    dR5 = max(Rmax**5 - Rmin**5 ,0)
    X_undiss_bin = k_undiss * (pi4_15Vm * dR5)
    return float64(X_undiss_bin)

@jit(cache=True, nopython=True)
def Convert_k_undiss_2_NR2_single(Rmax:float,
                                  Rmin:float,
                                  k_undiss:float):
    dR4 = max(Rmax**4 - Rmin**4,0)
    val = k_undiss / 4 * dR4
    return float64(val)

@jit(cache=True, nopython=True)
def Convert_X_undiss_bin_2_k_undiss(N_bins:int,
                                    pi4_15Vm:int,
                                    Rmax_undiss:float,
                                    Rmin_undiss:float,
                                    X_undiss_bin:float):
    #k_undiss = [0.0] * N_bins
    k_undiss = np.zeros(N_bins)
    for j in range(0,N_bins):
        if Rmax_undiss[j] == 0.0:
            k_undiss[j] = 0.0
        else:
            dR5 = max(Rmax_undiss[j]**5 - Rmin_undiss[j]**5,0)
            if dR5 > 0:
                k_undiss[j] = X_undiss_bin[j] / (pi4_15Vm * dR5)
            else:
                k_undiss[j] = 0.0

    return k_undiss
            
