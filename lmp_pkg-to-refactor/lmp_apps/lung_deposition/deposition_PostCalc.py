import numpy as np
from numba import int32, int64, float64, void, types, typed, deferred_type, jit, njit    # import the types

from .Lung_classes import *
from .PhysicalProp_classes import *
from .Deposition_classes import *
from .SizeDistrLib import *

@jit(cache=True, nopython=True)
def Lung_deposition_PostCalc(N_bins: int,
                             N_generations: int,
                             Depo: LungDepo_Class,
                             r: r_Class,
                             UndissAero_GD_in: SizeDistr_in_Class,
                             Undiss_poly_in: SizeDistr_in_Class,
                             Aerosphere_alpha: float,
                             FLAG_Aerosphere_Area: int):
        
    
    if FLAG_Aerosphere_Area == 1:
        Aerosphere_Area(N_bins,
                        N_generations,
                        r,
                        UndissAero_GD_in,
                        Undiss_poly_in,
                        Aerosphere_alpha)
    else: 
        Aerosphere_Volume(N_bins,
                            N_generations,
                            r,
                            UndissAero_GD_in,
                            Undiss_poly_in,
                            Aerosphere_alpha)

    #if Depo.iFLAG_manual_deposition == 1:
    #    Manual_redistribution_Gen()

@jit(cache=True, nopython=True)
def Aerosphere_Volume(N_bins: int,
                    N_generations: int,
                    r: r_Class,
                    UndissAero_GD_in: SizeDistr_in_Class,
                    Undiss_poly_in: SizeDistr_in_Class,
                    Aerosphere_alpha: float):
    r_new = r_Class()
    sum_num_Tot = np.zeros(N_generations+1)
    sum_num_Inh = np.zeros(N_generations+1)
    sum_num_BH = np.zeros(N_generations+1)
    sum_num_Exh = np.zeros(N_generations+1)

    #sum_num_Tot = [0.0] * (N_generations+1)
    #sum_num_Inh = [0.0] * (N_generations+1)
    #sum_num_BH = [0.0] * (N_generations+1)
    #sum_num_Exh = [0.0] * (N_generations+1)

    for j in range(0,N_bins):
        R_mid = (Undiss_poly_in.Rmax[j] + Undiss_poly_in.Rmin[j]) / 2

    V_distr = 1
    alpha = Aerosphere_alpha
    for j in range(0,N_bins):
        for g in range(0, N_generations+1):
            add_term = 0.0
            if g == 0:
                sum_num_Tot[g] = add_term
                sum_num_Inh[g] = add_term
                sum_num_BH[g] = 0.0
                sum_num_Exh[g] = 0.0
            else:
                sum_num_Tot[g] = 0.0
                sum_num_Inh[g] = 0.0
                sum_num_BH[g] = 0.0
                sum_num_Exh[g] = 0.0
    
            for k in range(0,N_bins):
                sum_num_Tot[g] = sum_num_Tot[g] + Calc_BuildingBlock_AS_Vol(alpha,
                                                                        j,
                                                                        k,
                                                                        r.TOT.Tot[g,k],
                                                                        Undiss_poly_in,
                                                                        UndissAero_GD_in)
                sum_num_Inh[g] = sum_num_Inh[g] + Calc_BuildingBlock_AS_Vol(alpha,
                                                                        j,
                                                                        k,
                                                                        r.Inh.Tot[g,k],
                                                                        Undiss_poly_in,
                                                                        UndissAero_GD_in)
                sum_num_BH[g] = sum_num_BH[g] + Calc_BuildingBlock_AS_Vol(alpha,
                                                                        j,
                                                                        k,
                                                                        r.BH.Tot[g,k],
                                                                        Undiss_poly_in,
                                                                        UndissAero_GD_in)
                sum_num_Exh[g] = sum_num_Exh[g] + Calc_BuildingBlock_AS_Vol(alpha,
                                                                        j,
                                                                        k,
                                                                        r.Exh.Tot[g,k],
                                                                        Undiss_poly_in,
                                                                        UndissAero_GD_in)
        sum_denom_Tot = 0
        for g in range(0,N_generations+1):
            sum_denom_Tot = max(1e-100, sum_denom_Tot + sum_num_Tot[g])
        for g in range(0,N_generations+1):
            if sum_denom_Tot > 0:
                r_new.TOT.Tot[g,j] = sum_num_Tot[g]/sum_denom_Tot
                r_new.Inh.Tot[g,j] = sum_num_Inh[g]/sum_denom_Tot
                r_new.BH.Tot[g,j] = sum_num_BH[g]/sum_denom_Tot
                r_new.Exh.Tot[g,j] = sum_num_Exh[g]/sum_denom_Tot
            else:
                r_new.TOT.Tot[g,j] = 0.0
                r_new.Inh.Tot[g,j] = 0.0
                r_new.BH.Tot[g,j] = 0.0
                r_new.Exh.Tot[g,j] = 0.0
    
    for j in range(0,N_bins):
        for g in range(0,N_generations+1):
            r.TOT.Tot[g,j] = r_new.TOT.Tot[g,j]
            r.Inh.Tot[g,j] = r_new.Inh.Tot[g,j]
            r.BH.Tot[g,j] = r_new.BH.Tot[g,j]
            r.Exh.Tot[g,j] = r_new.Exh.Tot[g,j]


@jit(cache=True, nopython=True)            
def Aerosphere_Area(N_bins: int,
                    N_generations: int,
                    r: r_Class,
                    UndissAero_GD_in: SizeDistr_in_Class,
                    Undiss_poly_in: SizeDistr_in_Class,
                    Aerosphere_alpha: float):
    r_new = r_Class()
    sum_num_Tot = [0.0] * (N_generations+1)
    sum_num_Inh = [0.0] * (N_generations+1)
    sum_num_BH = [0.0] * (N_generations+1)
    sum_num_Exh = [0.0] * (N_generations+1)

    for j in range(0,N_bins):
        R_mid = (Undiss_poly_in.Rmax[j] + Undiss_poly_in.Rmin[j]) / 2

    V_distr = 1
    alpha = Aerosphere_alpha
    for j in range(0,N_bins):
        for g in range(0, N_generations+1):
            add_term = 0
            if g == 0:
                sum_num_Tot[g] = add_term
                sum_num_Inh[g] = add_term
                sum_num_BH[g] = 0
                sum_num_Exh[g] = 0
            else:
                sum_num_Tot[g] = 0
                sum_num_Inh[g] = 0
                sum_num_BH[g] = 0
                sum_num_Exh[g] = 0
    
            for k in range(0,N_bins):
                sum_num_Tot[g] = sum_num_Tot[g] + Calc_BuildingBlock_AS(alpha,
                                                                        j,
                                                                        k,
                                                                        r.TOT.Tot[g,k],
                                                                        Undiss_poly_in,
                                                                        UndissAero_GD_in)
                sum_num_Inh[g] = sum_num_Inh[g] + Calc_BuildingBlock_AS(alpha,
                                                                        j,
                                                                        k,
                                                                        r.Inh.Tot[g,k],
                                                                        Undiss_poly_in,
                                                                        UndissAero_GD_in)
                sum_num_BH[g] = sum_num_BH[g] + Calc_BuildingBlock_AS(alpha,
                                                                        j,
                                                                        k,
                                                                        r.BH.Tot[g,k],
                                                                        Undiss_poly_in,
                                                                        UndissAero_GD_in)
                sum_num_Exh[g] = sum_num_Exh[g] + Calc_BuildingBlock_AS(alpha,
                                                                        j,
                                                                        k,
                                                                        r.Exh.Tot[g,k],
                                                                        Undiss_poly_in,
                                                                        UndissAero_GD_in)
        sum_denom_Tot = 0
        for g in range(0,N_generations+1):
            sum_denom_Tot = max(1e-100, sum_denom_Tot + sum_num_Tot[g])
        for g in range(0,N_generations+1):
            if sum_denom_Tot > 0:
                r_new.TOT.Tot[g,j] = sum_num_Tot[g]/sum_denom_Tot
                r_new.Inh.Tot[g,j] = sum_num_Inh[g]/sum_denom_Tot
                r_new.BH.Tot[g,j] = sum_num_BH[g]/sum_denom_Tot
                r_new.Exh.Tot[g,j] = sum_num_Exh[g]/sum_denom_Tot
            else:
                r_new.TOT.Tot[g,j] = 0
                r_new.Inh.Tot[g,j] = 0
                r_new.BH.Tot[g,j] = 0
                r_new.Exh.Tot[g,j] = 0
    
    for j in range(0,N_bins):
        for g in range(0,N_generations+1):
            r.TOT.Tot[g,j] = r_new.TOT.Tot[g,j]
            r.Inh.Tot[g,j] = r_new.Inh.Tot[g,j]
            r.BH.Tot[g,j] = r_new.BH.Tot[g,j]
            r.Exh.Tot[g,j] = r_new.Exh.Tot[g,j]

"""
def nonuse():
    #Calculate new spray particle size distribution
    denominator = 0
    for j2 in range(0,N_bins):
        denominator_add =  Convert_k_undiss_2_NR2_single(UndissAero_GD_in.Rmax[0,j2],
                                                        UndissAero_GD_in.Rmin[0,j2],
                                                        UndissAero_GD_in.k[0,j2])
        denominator = denominator + denominator_add
    
    denominator = max(1e-100, denominator)
    k_undiss_in_new = [0.0]*N_bins

    for j in range(0,N_bins):
        Rmin = alpha * Undiss_poly_in.Rmin[j]
        numerator = 0.0

        j2 = 0
        while(True):
            j2 = j2 +1
            if UndissAero_GD_in.Rmax[j2] < Rmin and j2 < N_bins:
                continue
            else:
                break
        
        if UndissAero_GD_in.Rmax[j2] >= Rmin:
            numerator_add = Convert_k_undiss_2_NR2_single(UndissAero_GD_in.Rmax[0,j2],
                                                          Rmin,
                                                          UndissAero_GD_in.k[0,j2])
            
            numerator = numerator + numerator_add
            for j3 in range(j2+1,N_bins):
                numerator_add = Convert_k_undiss_2_NR2_single(UndissAero_GD_in.Rmax[0,j3],
                                                              UndissAero_GD_in.Rmin[0,j3],
                                                              UndissAero_GD_in)
                numerator = numerator + numerator_add
        
        if denominator > 0:
            k_undiss_in_new[j] = Undiss_poly_in.k[j] * numerator / denominator
        else:
            k_undiss_in_new[j] = 0
    
    #renormalize and replace size distribution
    factor1 = 0
    factor2 = 0
    for j in range(0,N_bins):
        factor_add = Convert_k_undiss_2_X_undiss_single(1,
                                           Undiss_poly_in.Rmax[0,j],
                                           Undiss_poly_in.Rmin[0,j],
                                           Undiss_poly_in.k[0,j])
        factor1 = factor1 + factor_add
        factor_add = Convert_k_undiss_2_X_undiss_single(1,
                                           Undiss_poly_in.Rmax[0,j],
                                           Undiss_poly_in.Rmin[0,j],
                                           k_undiss_in_new[j])
        factor2 = factor2 + factor_add

    for j in range(0, N_bins):
        if factor2 > 0:
            Undiss_poly_in.k[j] = k_undiss_in_new[j] * factor1 / factor2
        else:
            Undiss_poly_in.k[j] = 0
    
    for j in range(0,N_bins):
        R_mid = (Undiss_poly_in.Rmax[j] + Undiss_poly_in.Rmin[j]) / 2
    
    Undiss_poly_in, Undiss_poly = Copy_size_distribution(10,
                                                         N_bins,
                                                         1.0,
                                                         Undiss_poly_in)
"""

@jit(cache=True, nopython=True)
def Calc_BuildingBlock_AS_Vol(alpha,
                          j,
                          k,
                          r,
                          Undiss_in,
                          UndissAero_GD_in):
    kd=UndissAero_GD_in.k[0,k]
    Rp_max = Undiss_in.Rmax[0,j]
    Rp_min = Undiss_in.Rmin[0,j]
    Rd_max = UndissAero_GD_in.Rmax[0,k]
    Rd_min = UndissAero_GD_in.Rmin[0,k]

    if Rd_max <= alpha*Rp_min:
         Bb = 0.0
         
    elif Rd_max <= alpha*Rp_max and Rd_max >= alpha*Rp_min and Rd_min <= alpha*Rp_min:
         Bb = r*kd/50.0*(alpha**5 * Rp_min**10 + 1/alpha**5 * Rd_max**10 - 2.0* Rp_min**5 * Rd_max**5)
         
    elif Rd_max >= alpha*Rp_max and Rd_min <= alpha*Rp_max and Rd_min >= alpha*Rp_min:
         Bb = r*kd/50.0*(2 * Rd_max**5 * Rp_max**5 -2 * Rd_max**5 * Rp_min**5 + 2 * Rd_min**5 * Rp_min **5 - 1/alpha**5 * Rd_min**10 - alpha**5 * Rp_max**10)
         
    elif Rd_min >= alpha*Rp_max:
         Bb = r*kd/25.0*((Rd_max**5 - Rd_min**5)*(Rp_max**5 - Rp_min**5))
         
    elif Rd_max >= alpha*Rp_max and Rd_min <= alpha*Rp_min:
         Bb = r*kd/50.0*(2 * Rd_max**5 * Rp_max**5 - 2 * Rd_max**5 * Rp_min**5 - alpha**5 * Rp_max**10 + alpha**5 * Rp_min**10)
                  
    elif Rd_max <= Rp_max and Rd_min >= Rp_min:
         Bb = r*kd/50.0*(2 * Rd_min**5 * Rp_min**5 - 2 * Rd_max**5 * Rp_min**5 + 1/alpha**5 * Rd_max**10 - 1/alpha**5 * Rd_min**10)
         
    return Bb

@jit(cache=True, nopython=True)
def Calc_BuildingBlock_AS(alpha,
                          j,
                          k,
                          r,
                          Undiss_in,
                          UndissAero_GD_in):
    
    kd=UndissAero_GD_in.k[0,k]
    Rp_max = Undiss_in.Rmax[0,j]
    Rp_min = Undiss_in.Rmin[0,j]
    Rd_max = UndissAero_GD_in.Rmax[0,k]
    Rd_min = UndissAero_GD_in.Rmin[0,k]
    
    if Rd_max < alpha * Rp_min:
        Bb = 0
    elif Rd_max <= alpha * Rp_max and Rd_max >= alpha * Rp_min and Rd_min <= alpha * Rp_min:
        Bb = r * kd / 180 * (5 * alpha**4 * Rp_min**9 + 4/alpha**5 * Rd_max**9 - 9 * Rp_min**5 * Rd_max**4)
    elif Rd_max >= alpha * Rp_max and Rd_max <= alpha * Rp_min and Rd_min >= alpha * Rp_min: 
        Bb =  9 * Rd_max**4 * Rp_max**5 -9 * Rd_max**4 * Rp_min**5 + 9 * Rd_min**4 * Rp_min **5 - 4/alpha**5 * Rd_min**9 - 5 * alpha**4 * Rp_max**9
    elif Rd_min >= alpha * Rp_max: 
        Bb = r*kd/20 *((Rd_max**4 - Rd_min**4)*(Rp_max**5 - Rp_min**5))
    elif Rd_max >= alpha*Rp_max and Rd_min >= alpha*Rp_min:
        Bb = r*kd/180 *(9* Rd_max**4 * Rp_max**5 -9* Rd_max**4 * Rp_min**5 -5*alpha**4 * Rp_max**9+5*alpha**4 * Rp_min**9)
    elif Rd_max <= Rp_max and Rd_min >= Rp_min:
        Bb = r*kd/180*(9* Rd_min**4 * Rp_min**5 -9* Rd_max**4 * Rp_min**5 +4/alpha**5 * Rd_max**9 -4/alpha**5 * Rd_min**9)

    return Bb


def Copy_size_distribution(N_comps,
                           N_bins,
                           Scaling_factor,
                           Undiss_poly_in_A:SizeDistr_Class):
    Undiss_poly_in_B = SizeDistr_Class()
    Undiss_poly_B = SizeDistr_Class()

    for j in range(0, N_bins):
        Undiss_poly_in_B.Rmin[0,j] = Undiss_poly_in_A.Rmin[0,j]
        Undiss_poly_in_B.Rmax[0,j] = Undiss_poly_in_A.Rmax[0,j]

        for i in range(0,N_comps):
            Undiss_poly_B.Rmin[i,j] = Undiss_poly_in_B.Rmin[0,j]
            Undiss_poly_B.Rmax[i,j] = Undiss_poly_in_B.Rmax[0,j]
        Undiss_poly_in_B.k[0,j] = Undiss_poly_in_A.k[0,j] * Scaling_factor
        Undiss_poly_in_B.X_bin[0,j] = Undiss_poly_in_A.X_bin[0,j]

    return Undiss_poly_in_B, Undiss_poly_B

"""
def Manual_redistribution_Gen():
    LungBranches(N_Generations,
                 LungTube.Inflow_from_N_GenReg_DEP,
                 LungTube.OutFlow_to_GenReg_DEP)
    for j in range(0, N_bins):
        Rmax[j] = Undiss_poly_in.Rmax[j]
        Rmin[j] = Undiss_poly_in.Rmin[j]
        k_in[j] = UndissAero_GD_in.k[j]
"""