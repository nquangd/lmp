from numba import njit, jit, int32, float64

from .Lung_classes import *
from .PhysicalProp_classes import *
from .Deposition_classes import *
from .Lung_depEquations import *
from .SizeDistrLib import *

@jit(cache=True, nopython=True)
def Initialize_BH(N_slices: LungSliceB_Class,
                  V_slice: LungSliceReal8B_Class,
                  N_start_generation: LungSlice_Class):
    """Initializes breathhold parameters"""
    
    N_slices.BH = N_slices.Inh

    for i_slice in range(0,N_slices.Inh):
        V_slice.BH[i_slice] = V_slice.Inh[i_slice]
        N_start_generation.BH[i_slice] = N_start_generation.Inh[i_slice]
@jit(cache=True, nopython=True)
def Lung_deposition_Variable(j,
                             N_slices: LungSliceB_Class,
                             N_Generations: int,
                             Max_slice_reaches_generation: LungSlice_Class,
                             N_Start_generation: LungSlice_Class,
                             V_slice: LungSliceReal8B_Class,
                             Fraction_slice: float,
                             LungSliceTube: LungSliceTube_Class,
                             amountSl_still_in_air: LungSliceReal8_Class,
                             AirProp: AirProp_Class,
                             V_air_1: float,
                             V_air: float,
                             LungTube: LungTube_Class,
                             r: r_Class,
                             UndissAreo_AD: SizeDistr_Class,
                             UndissAreo_GD: SizeDistr_Class,
                             Depo: LungDepo_Class,
                             H2OProp: H2OProp_Class,
                             AreoProp: AeroProp_Class,
                             PhCo: PhCo_Class,
                             Body: Body_Class,
                             BH_fraction: float):
    """Calculates lung deposition for bin j"""
    #..... amount_entrance = amount of particles before entering a generation.
    #..... amountSl = amount of particles still in air in the slice.
    #..... amount = deposited amount of particles
    amount = AmountDepo_Class()
    Prob = ProbSlice_Class()
    

    amountSl_still_in_air.Inh[:,j] =  Amount_Set_Zero(N_Generations,
                                            N_slices.Inh,
                                            amount,
                                            amountSl_still_in_air.Inh[:,j])
    
    
    #Overlap_BHExh
    V_Cum_slice_BH=[0.0]*N_slices_max
    V_Cum_slice_BH[0] = 0 + V_slice.BH[0]

    for i_from in range(1,N_slices.BH):
        V_Cum_slice_BH[i_from] = V_Cum_slice_BH[i_from-1] + V_slice.BH[i_from]

    V_Cum_slice_Exh=[0.000]*N_slices_max
    V_Cum_slice_Exh[0] = 0 + V_slice.BH[0]

    for i_from in range(1,N_slices.BH):
        V_Cum_slice_Exh[i_from] = V_Cum_slice_Exh[i_from-1] + V_slice.Exh[i_from]

    OverlapBHExh = np.zeros((N_slices.BH,N_slices.Exh))
    for i_from in range(0,N_slices.BH):
        for i_to in range(0,N_slices.Exh):
            Vmin = max(V_Cum_slice_BH[i_from-1],V_Cum_slice_Exh[i_to-1])
            Vmax = min(V_Cum_slice_BH[i_from], V_Cum_slice_Exh[i_to])
            dV = max(0.0,Vmax-Vmin)

            OverlapBHExh[i_from,i_to] = dV / V_slice.BH[i_from]

    #Calculation of deposition probability
    #inhalation
    DepositionProb_Inhalation_Variable(j,
                                       N_slices.Inh,
                                       Prob.Inh,
                                       N_Start_generation,
                                       Max_slice_reaches_generation.Inh,
                                       AirProp,
                                       AreoProp,
                                       Depo,
                                       V_air_1,
                                       LungTube,
                                       LungSliceTube.Inh,
                                       UndissAreo_AD,
                                       UndissAreo_GD,
                                       Body,
                                       H2OProp,
                                       PhCo)
    #Breath hold
    DepositionProb_Breathhold_Variable(j,
                                       N_slices.BH,
                                       N_Generations,
                                       Prob.BH,
                                       Depo,
                                       LungTube,
                                       LungSliceTube.BH,
                                       UndissAreo_AD,
                                       UndissAreo_GD,
                                       Body,
                                       AirProp,
                                       AreoProp,
                                       H2OProp,
                                       PhCo,
                                       BH_fraction)
    
    #Exhalation
    DepositionProb_Exhalation_Variable(j,
                                       N_slices.Exh,
                                       Prob.Exh,
                                       N_Start_generation,
                                       Max_slice_reaches_generation.Exh,
                                       AirProp,
                                       AreoProp,
                                       Depo,
                                       V_air_1,
                                       LungTube,
                                       LungSliceTube.Exh,
                                       UndissAreo_AD,
                                       UndissAreo_GD,
                                       Body,
                                       H2OProp,
                                       PhCo)
    

    #add dose to Inh slices. All areaosol diametrs are added in the same proportions.
    for i_slice in range(0,N_slices.Inh):
        amountSl_still_in_air.Inh[i_slice,j] = Fraction_slice[i_slice]
        
    DepositionAmount_Inhalation_Variable(N_Generations,
                                             N_slices,
                                             amountSl_still_in_air,
                                             j,
                                             amount.Inh,
                                             Prob.Inh,
                                             N_Start_generation,
                                             Max_slice_reaches_generation)
    Move_From_Inh_to_BH(N_slices,
                            amountSl_still_in_air,
                            j)
    DepositionAmount_Breathhold_Variable(N_Generations,
                                             N_slices,
                                             amountSl_still_in_air,
                                             j,
                                             amount.BH,
                                             Prob.BH,
                                             Max_slice_reaches_generation,
                                             BH_fraction)
    Move_From_BH_to_Exh(N_slices,
                            amountSl_still_in_air,
                            j,
                            OverlapBHExh)
    DepositionAmount_Exhalation_Variable(N_slices,
                                             N_Generations,
                                             amountSl_still_in_air,
                                             j,
                                             amount.Exh,
                                             Prob.Exh,
                                             N_Start_generation,
                                             Max_slice_reaches_generation)


    sum_Inh = 0
    sum_BH = 0
    sum_Exh = 0
    sum_Tot = 0
    #Write results
    for g in range(0,N_Generations):
        r.Inh.Tot[g,j] = amount.Inh.Tot[g]
        r.Inh.Imp[g,j] = amount.Inh.Imp[g]
        r.Inh.Diff[g,j] = amount.Inh.Diff[g]
        r.Inh.Sed[g,j] = amount.Inh.Sed[g]

        r.BH.Tot[g,j] = amount.BH.Tot[g]
        r.BH.Imp[g,j] = amount.BH.Imp[g]
        r.BH.Diff[g,j] = amount.BH.Diff[g]
        r.BH.Sed[g,j] = amount.BH.Sed[g]

        r.Exh.Tot[g,j] = amount.Exh.Tot[g]
        r.Exh.Imp[g,j] = amount.Exh.Imp[g]
        r.Exh.Diff[g,j] = amount.Exh.Diff[g]
        r.Exh.Sed[g,j] = amount.Exh.Sed[g]

        #This is taken from Lungdeposition Postcalc:
        r.TOT.Tot[g,j] = r.Inh.Tot[g,j] + r.BH.Tot[g,j] + r.Exh.Tot[g,j]
        sum_Inh = sum_Inh + r.Inh.Tot[g,j]
        sum_BH = sum_BH + r.BH.Tot[g,j]
        sum_Exh = sum_Exh + r.Exh.Tot[g,j]
        sum_Tot = sum_Tot + r.TOT.Tot[g,j]
    #Still in air, taken from Postcalc file in Fortran implementation:  
    r.Inh.Tot[25,j] = 1 - sum_Inh
    r.BH.Tot[25,j] = 1 - sum_BH
    r.Exh.Tot[25,j] = 1 - sum_Exh
    r.TOT.Tot[25,j] = 1 - sum_Tot

@jit(cache=True, nopython=True)
def Amount_Set_Zero(N_Generations: int,
                    N_slices: int,
                    amount: AmountDepo_Class,
                    amountSl_still_in_air: float):
    """Initialize amount deposited in each generation data storage  variable"""

    for g in range(0,N_Generations):
        amount.Inh.Tot[g]=0.0
        amount.Inh.Imp[g]=0.0
        amount.Inh.Diff[g]=0.0
        amount.Inh.Sed[g]=0.0
        amount.Inh.Entrance[g]=0.0

        amount.BH.Tot[g]=0.0
        amount.BH.Imp[g]=0.0
        amount.BH.Diff[g]=0.0
        amount.BH.Sed[g]=0.0
        amount.BH.Entrance[g]=0.0

        amount.Exh.Tot[g]=0.0
        amount.Exh.Imp[g]=0.0
        amount.Exh.Diff[g]=0.0
        amount.Exh.Sed[g]=0.0
        amount.Exh.Entrance[g]=0.0
    
    for i_slice in range(0,N_slices):
        amountSl_still_in_air[i_slice] = 0.0
    return amountSl_still_in_air

@jit(cache=True, nopython=True)
def DepositionProb_Inhalation_Variable(j: int,
                                       N_slices: int,
                                       Prob_Inh: ProbSliceSingle_Class,
                                       N_start_generation: LungSlice_Class,
                                       Max_slice_reaches_generation: int,
                                       AirProp: AirProp_Class,
                                       AeroProp: AeroProp_Class,
                                       Depo: LungDepo_Class,
                                       V_air_1: float,
                                       LungTube: LungTube_Class,
                                       LungSliceTube: LungSliceTube_C_Class,
                                       UndissAero_AD: SizeDistr_Class,
                                       UndissAero_GD: SizeDistr_Class,
                                       Body: Body_Class,
                                       H2OProp: H2OProp_Class,
                                       PhCo: PhCo_Class):
    FLAG_Mode=1
    FLAG_DoImpaction = 1
    Depo.iFLAG_DoSedDiff = 1

    Deposition_parameters(j,
                          AeroProp,
                          AirProp,
                          UndissAero_AD,
                          UndissAero_GD,
                          Body,
                          PhCo)
    
    for i_slice in range(0,N_slices):
        for g in range(N_start_generation.Inh[i_slice],Max_slice_reaches_generation[i_slice]+1): 
            DepositionProb_InhExh_slice_generation(i_slice,
                                                   g,
                                                   Prob_Inh,
                                                   Depo,
                                                   V_air_1,
                                                   LungTube,
                                                   LungSliceTube,
                                                   Body,
                                                   H2OProp,
                                                   AeroProp,
                                                   AirProp,
                                                   PhCo,
                                                   FLAG_Mode,
                                                   FLAG_DoImpaction)

@jit(cache=True, nopython=True)
def DepositionProb_Breathhold_Variable(j: int,
                                       N_slices: int,
                                       N_Generations: int,
                                       Prob_BH: ProbSliceSingle_Class,
                                       Depo: LungDepo_Class,
                                       LungTube: LungTube_Class,
                                       LungSliceTube: LungSliceTube_C_Class,
                                       UndissAero_AD: SizeDistr_Class,
                                       UndissAero_GD: SizeDistr_Class,
                                       Body: Body_Class,
                                       AirProp: AirProp_Class,
                                       AeroProp: AeroProp_Class,
                                       H2OProp: H2OProp_Class,
                                       PhCo: PhCo_Class,
                                       BH_fraction: float):
    Deposition_parameters(j,
                          AeroProp,
                          AirProp,
                          UndissAero_AD,
                          UndissAero_GD,
                          Body,
                          PhCo)
    
    for i_slice in range(0,N_slices):
        for g in range(0,N_Generations):
            if BH_fraction[i_slice,g] > 0:
                DepositionProb_Breathhold_slice_generation(i_slice,
                                                           g,
                                                           Prob_BH,
                                                           Depo,
                                                           LungTube,
                                                           LungSliceTube,
                                                           Body,
                                                           H2OProp,
                                                           AirProp,
                                                           AeroProp,
                                                           PhCo)
            else:
                Prob_BH.Imp[i_slice,g] = 0.0
                Prob_BH.Diff[i_slice,g] = 0.0
                Prob_BH.Sed[i_slice,g] = 0.0
@jit(cache=True, nopython=True)
def DepositionProb_Exhalation_Variable(j: int,
                                       N_slices: int,
                                       Prob_Exh: ProbSliceSingle_Class,
                                       N_start_generation: LungSlice_Class,
                                       Max_slice_reaches_generation: int,
                                       AirProp: AirProp_Class,
                                       AeroProp: AeroProp_Class,
                                       Depo: LungDepo_Class,
                                       V_air_1: float,
                                       LungTube: LungTube_Class,
                                       LungSliceTube: LungSliceTube_C_Class,
                                       UndissAero_AD: SizeDistr_Class,
                                       UndissAero_GD: SizeDistr_Class,
                                       Body: Body_Class,
                                       H2OProp: H2OProp_Class,
                                       PhCo: PhCo_Class):
    FLAG_Mode = 2
    Deposition_parameters(j,
                          AeroProp,
                          AirProp,
                          UndissAero_AD,
                          UndissAero_GD,
                          Body,
                          PhCo)
    
    for i_slice in range(0,N_slices):
        for g in range(Max_slice_reaches_generation[i_slice],N_start_generation.Exh[i_slice]-1,-1):
            if g == Max_slice_reaches_generation[i_slice]:
                FLAG_DoImpaction = 0
                Depo.iFLAG_DoSedDiff = Depo.iFLAG_DoSedDiff_From_VB
            else:
                FLAG_DoImpaction = 1
                Depo.iFLAG_DoSedDiff = 1

            DepositionProb_InhExh_slice_generation(i_slice,
                                                   g,
                                                   Prob_Exh,
                                                   Depo,
                                                   V_air_1,
                                                   LungTube,
                                                   LungSliceTube,
                                                   Body,
                                                   H2OProp,
                                                   AeroProp,
                                                   AirProp,
                                                   PhCo,
                                                   FLAG_Mode,
                                                   FLAG_DoImpaction)
@jit(cache=True, nopython=True)    
def DepositionAmount_Inhalation_Variable(N_generations: int,
                                         N_slices: LungSliceB_Class,
                                         amountSL_still_in_air: LungSliceReal8_Class,
                                         j: int,
                                         amount_Inh: AmountDepoSingle_Class,
                                         Prob_Inh: ProbSliceSingle_Class,
                                         N_start_generation: LungSlice_Class,
                                         Max_slice_reaches_generation: LungSlice_Class):
    for g in range(0,N_generations):
        temp = DepositionAmount(g,
                                                N_slices.Inh,
                                                Max_slice_reaches_generation.Inh,
                                                N_start_generation.Inh,
                                                amount_Inh,
                                                amountSL_still_in_air.Inh[:,j],
                                                Prob_Inh)
        amountSL_still_in_air.Inh[:,j] = temp.copy()
@jit(cache=True, nopython=True)        
def DepositionAmount_Breathhold_Variable(N_generations: int,
                                         N_slices: LungSliceB_Class,
                                         amountSL_still_in_air: LungSliceReal8_Class,
                                         j: int,
                                         amount_BH: AmountDepoSingle_Class,
                                         Prob_BH: ProbSliceSingle_Class,
                                         max_slice_reaches_generation: LungSlice_Class,
                                         BH_fraction: float):
    amountSL_still_in_air.BH[:,j] = DepositionAmount_BH(N_generations,
                                                N_slices.BH,
                                                amount_BH,
                                                amountSL_still_in_air.BH[:,j],
                                                BH_fraction,
                                                Prob_BH)
@jit(cache=True, nopython=True)    
def DepositionAmount_Exhalation_Variable(N_slices: LungSliceB_Class,
                                         N_generations: int,
                                         amountSl_still_in_air: LungSliceReal8_Class,
                                         j: int,
                                         amount_Exh: AmountDepoSingle_Class,
                                         Prob_Exh: ProbSliceSingle_Class,
                                         N_start_generation: LungSlice_Class,
                                         Max_slice_reaches_generation: LungSlice_Class):
    for g in range(N_generations-1,-1,-1):
        amountSl_still_in_air.Exh[:,j] = DepositionAmount(g,
                                                        N_slices.Exh,
                                                        Max_slice_reaches_generation.Exh,
                                                        N_start_generation.Exh,
                                                        amount_Exh,
                                                        amountSl_still_in_air.Exh[:,j],
                                                        Prob_Exh)
@jit(cache=True, nopython=True)  
def Move_From_Inh_to_BH(N_slices: LungSliceB_Class,
                       amountSl_still_in_air: LungSliceReal8B_Class,
                       j: int):
    for i_slice in range(0,N_slices.BH):
        amountSl_still_in_air.BH[i_slice,j]=amountSl_still_in_air.Inh[i_slice,j]
@jit(cache=True, nopython=True)
def Move_From_BH_to_Exh(N_slices:LungSliceB_Class,
                        amountSl_still_in_air: LungSliceReal8B_Class,
                        j: int,
                        OverlapBHExh: float):
    for i_to in range(0,N_slices.Exh):
        amountSl_still_in_air.Exh[i_to,j]=0
    
    for i_from in range(0,N_slices.BH):
        for i_to in range(0,N_slices.Exh):
            delta_amount = amountSl_still_in_air.BH[i_from,j] * OverlapBHExh[i_from,i_to]
            amountSl_still_in_air.Exh[i_to,j] = amountSl_still_in_air.Exh[i_to,j] + delta_amount
    
    
"""
def Fraction_Still_in_Air_ET(N_bins: int,
                             Fraction_slice: float,
                             N_slices: LungSliceB_Class,
                             Max_slice_reaches_generation: LungSlice_Class,
                             LungTube: LungTube_Class,
                             UndissAreo_GD_in: SizeDistr_Class,
                             amountSL_still_in_air:LungSliceReal8B_Class):
    
    NormalizedSum_aerosols_still_in_air = 0.0
    NormalizedSum_aerosols_delivered = 0.0

    for i_slice in range(0,N_slices.Inh):
        for j in range(0,N_bins):
            temp = Convert_k_undiss_2_X_undiss_single(1,
                                               UndissAreo_GD_in.Rmax[j],
                                               UndissAreo_GD_in.Rmin[j],
                                               UndissAreo_GD_in.k[j])
            NormalizedSum_aerosols_delivered = NormalizedSum_aerosols_delivered + temp * Fraction_slice[i_slice]
        
        g = Max_slice_reaches_generation.Inh[i_slice]

        if LungTube.Region(g) == 'ET': #This slice only reaches generation 1, i.e. ET
            #Calculate how much aerosol volume drug there was in thsi slice after exhalation
            for j in range(0,N_bins):
                temp = Convert_k_undiss_2_X_undiss_single(1,
                                               UndissAreo_GD_in.Rmax[j],
                                               UndissAreo_GD_in.Rmin[j],
                                               UndissAreo_GD_in.k[j])
                NormalizedSum_aerosols_still_in_air = NormalizedSum_aerosols_still_in_air + temp * amountSL_still_in_air.Exh[i_slice,j]


     
    return NormalizedSum_aerosols_still_in_air/NormalizedSum_aerosols_delivered
"""
@jit(cache=True, nopython=True)
def DepositionProb_InhExh_slice_generation(i_slice: int,
                                           g: int,
                                           Prob: ProbSliceSingle_Class,
                                           Depo: LungDepo_Class,
                                           V_air_1: float,
                                           LungTube: LungTube_Class,
                                           LungSliceTube: LungSliceTube_C_Class,
                                           Body: Body_Class,
                                           H2OProp: H2OProp_Class,
                                           AeroProp: AeroProp_Class,
                                           AirProp: AirProp_Class,
                                           PhCo: PhCo_Class,
                                           FLAG_Mode: int,
                                           FLAG_DoImpaction: int):
    if Depo.iFLAG_KingsCollege == 1 and g < Depo.entrance_generation:
        SetprobZero_KingsCollege(i_slice, g, Prob)

    elif LungTube.Region(g) == 'ET':
        DepositionProb_InhExh_ET(i_slice,
                                 g,
                                 Prob,
                                 Depo,
                                 LungSliceTube,
                                 LungTube,
                                 V_air_1,
                                 FLAG_Mode,
                                 Body,
                                 H2OProp,
                                 AeroProp,
                                 AirProp,
                                 PhCo)
        
    elif LungTube.Region(g) == 'BB' or LungTube.Region(g) == 'bb' or LungTube.Region(g) == 'TB' or LungTube.Region(g) == 'Al':
        DepositionProb_InhExh_BB_bb_Al(i_slice,
                                       g,
                                       Prob,
                                       Depo,
                                       LungSliceTube,
                                       LungTube,
                                       FLAG_DoImpaction,
                                       FLAG_Mode,
                                       Body,
                                       H2OProp,
                                       AeroProp,
                                       AirProp,
                                       PhCo)

@jit(cache=True, nopython=True)        
def DepositionProb_Breathhold_slice_generation(i_slice: int,
                                               g: int,
                                               Prob: ProbSliceSingle_Class,
                                               Depo: LungDepo_Class,
                                               LungTube: LungTube_Class,
                                               LungSliceTube: LungSliceTube_C_Class,
                                               Body: Body_Class,
                                               H2OProp: H2OProp_Class,
                                               AirProp: AirProp_Class,
                                               AeroProp: AeroProp_Class,
                                               PhCo: PhCo_Class):
    if Depo.iFLAG_KingsCollege == 1 and g < Depo.entrance_generation:
        SetprobZero_KingsCollege(i_slice, g, Prob)
        
    elif LungTube.Region(g) == 'ET':
        DepositionProb_Breathhold_ET(i_slice,
                                     g,
                                     Prob,
                                     Depo,
                                     LungSliceTube,
                                     LungTube,
                                     Body,
                                     H2OProp,
                                     AeroProp,
                                     AirProp,
                                     PhCo)
        
    elif LungTube.Region(g) == 'BB' or LungTube.Region(g) == 'bb' or LungTube.Region(g) == 'TB' or LungTube.Region(g) == 'Al':
        DepositionProb_Breathhold_BB_bb_Al(i_slice,
                                           g,
                                           Prob,
                                           Depo,
                                           LungSliceTube,
                                           LungTube,
                                           Body,
                                           H2OProp,
                                           AeroProp,
                                           AirProp,
                                           PhCo)

@jit(cache=True, nopython=True)        
def SetprobZero_KingsCollege(i_slice: int,
                             g: int,
                             Prob: ProbSliceSingle_Class):
    Prob.Imp[i_slice,g] = 0
    Prob.Diff[i_slice,g] = 0
    Prob.Sed[i_slice,g] = 0

@jit(cache=True, nopython=True)
def DepositionProb_InhExh_ET(i_slice: int,
                             g: int,
                             Prob: ProbSliceSingle_Class,
                             Depo: LungDepo_Class,
                             LungSliceTube: LungSliceTube_C_Class,
                             LungTube: LungTube_Class,
                             V_air_1: float,
                             FLAG_Mode: int,
                             Body: Body_Class,
                             H2OProp: H2OProp_Class,
                             AeroProp: AeroProp_Class,
                             AirProp: AirProp_Class,
                             PhCo:PhCo_Class):
    #Impaction
    if Depo.ET_Model == 1:
        Prob.Imp[i_slice,g] = ETdep_NCRP96(AeroProp.DiffCoeff,
                                           AeroProp.D_Aero,
                                           LungSliceTube.Ave_Q[i_slice,g],
                                           FLAG_Mode,
                                           Depo)
    elif Depo.ET_Model == 3:
        Prob.Imp[i_slice,g] = ETdep_Manual_Deposition(Depo.ET_manual_deposition,
                                                      FLAG_Mode)
    #Diffusion and sedimentation
    Prob.Diff[i_slice,g] = 0
    Prob.Sed[i_slice,g]  = 0

    #For inhaltion only
    if FLAG_Mode == 1:
        Prob.Imp[i_slice,g] = Prob.Imp[i_slice,g] * (1 - Depo.CoarseFraction) + Depo.CoarseFraction
        p_between_0_1(Prob.Imp[i_slice,g])

@jit(cache=True, nopython=True)    
def DepositionProb_InhExh_BB_bb_Al(i_slice:int,
                                   g: int,
                                   Prob: ProbSliceSingle_Class,
                                   Depo: LungDepo_Class,
                                   LungSliceTube: LungSliceTube_C_Class,
                                   LungTube: LungTube_Class,
                                   FLAG_DoImpaction: int,
                                   FLAG_Mode: int,
                                   Body: Body_Class,
                                   H2OProp: H2OProp_Class,
                                   AeroProp: AeroProp_Class,
                                   AirProp: AirProp_Class,
                                   PhCo: PhCo_Class):
    #Check if manual deposition should be used for this generation
    FLAG_Do_Thoracic_manual_deposition = Assign_FLAG_Do_Thoracic_manual_deposition(g, Depo)

    #Impaction
    if FLAG_DoImpaction == 1:
        if FLAG_Do_Thoracic_manual_deposition == 0:
            Prob.Imp[i_slice,g] = Impaction_Yeh80_NCRP96(g,
                                                         AeroProp.D_Aero,
                                                         AeroProp.Density_aeroD_Particle,
                                                         AirProp,
                                                         LungTube,
                                                         LungSliceTube.Ave_Tube_Radius[i_slice,g],
                                                         LungSliceTube.Ave_Velocity[i_slice,g])
        else:
            if FLAG_Mode == 1:
                Prob.Imp[i_slice,g] = Depo.T_manual_deposition
            else:
                Prob.Imp[i_slice,g] = 0
    else:
        Prob.Imp[i_slice,g] = 0

    #Diffusion and Sedimentation
    if Depo.iFLAG_DoSedDiff == 1:
        if FLAG_Do_Thoracic_manual_deposition == 0:
            if Depo.T_model == 1:
                Prob.Diff[i_slice,g] = Diffusion_NCRP96(AeroProp.DiffCoeff,
                                                        LungSliceTube.ResidenceTime[i_slice,g],
                                                        LungSliceTube.Ave_Tube_Radius[i_slice,g],
                                                        LungSliceTube.Ave_Velocity[i_slice,g],
                                                        LungSliceTube.Ave_Reynold[i_slice,g],
                                                        LungTube.Angle_preceding[g])
                Prob.Sed[i_slice,g] = Sedimentation_Yeh80_NCRP96(AeroProp.v_settling,
                                                                 LungSliceTube.ResidenceTime[i_slice,g],
                                                                 LungSliceTube.Ave_Tube_Radius[i_slice,g],
                                                                 LungTube.Angle_gravity[g])
        else:
            Prob.Diff[i_slice,g] = 0
            Prob.Sed[i_slice,g] = 0
    else:
        Prob.Diff[i_slice,g] = 0
        Prob.Sed[i_slice,g] = 0

@jit(cache=True, nopython=True)
def DepositionProb_Breathhold_ET(i_slice: int,
                                 g: int,
                                 Prob: ProbSliceSingle_Class,
                                 Depo: LungDepo_Class,
                                 LungSliceTube: LungSliceTube_C_Class,
                                 LungTube: LungTube_Class,
                                 Body: Body_Class,
                                 H2OProp: H2OProp_Class,
                                 AeroProp: AeroProp_Class,
                                 AirProp: AirProp_Class,
                                 PhCo: PhCo_Class):
    #According to Bo Olsson. ETdep during BH is totally insignificant and there are no formulas in literature so far
    Prob.Imp[i_slice, g] = 0
    Prob.Diff[i_slice, g] = 0
    Prob.Sed[i_slice, g] = 0

@jit(cache=True, nopython=True)
def DepositionProb_Breathhold_BB_bb_Al(i_slice: int,
                                       g: int,
                                       Prob: ProbSliceSingle_Class,
                                       Depo: LungDepo_Class,
                                       LungSliceTube: LungSliceTube_C_Class,
                                       LungTube: LungTube_Class,
                                       Body: Body_Class,
                                       H2OProp: H2OProp_Class,
                                       AeroProp: AeroProp_Class,
                                       AirProp: AirProp_Class,
                                       PhCo: PhCo_Class):
    FLAG_Do_Thoracic_manual_deposition = Assign_FLAG_Do_Thoracic_manual_deposition(g, Depo)

    if FLAG_Do_Thoracic_manual_deposition == 0:
        #Impaction
        Prob.Imp[i_slice,g] = 0
        #Diffusion
        Prob.Diff[i_slice,g] = BHDiffusion(AeroProp.DiffCoeff,
                                           Depo.BHT,
                                           LungSliceTube.Ave_Tube_Radius[i_slice,g])
        #Sedimentation
        Prob.Sed[i_slice,g] = Sedimentation_Yeh80_NCRP96(AeroProp.v_settling,
                                                         Depo.BHT,
                                                         LungSliceTube.Ave_Tube_Radius[i_slice,g],
                                                         LungTube.Angle_gravity[g])
    else:
        #Impaction
        Prob.Imp[i_slice,g] = 0
        #Diffusion
        Prob.Diff[i_slice,g] = 0
        #Sedimentation
        Prob.Sed[i_slice,g] = 0

@jit(cache=True, nopython=True)
def Assign_FLAG_Do_Thoracic_manual_deposition(g: int,
                                              Depo: LungDepo_Class):
    
    Do_Thoracic_manual_deposition = 0
    if Depo.FLAG_Thoracic_manual_deposition == 1:
        if g <= Depo.Thoracic_manual_Generation_To:
            Do_Thoracic_manual_deposition = 1
    return Do_Thoracic_manual_deposition

@jit(cache=True, nopython=True)
def DepositionAmount(g: int,
                     N_slices: int,
                     Max_slice_reaches_generation: int,
                     N_start_generation: int,
                     amount: AmountDepoSingle_Class,
                     amountSl_still_in_air: float,
                     Prob: ProbSliceSingle_Class):
    """Calculates the amount of deposited particles in each generation g by looping over all slices"""
    for i_slice in range(0,N_slices):
        if Max_slice_reaches_generation[i_slice] >= g and N_start_generation[i_slice] <= g:
            amount.Entrance[g] = amount.Entrance[g] + amountSl_still_in_air[i_slice] #Amount that was still in air at entrance to generation, for all slices
            Prob_Imp = Prob.Imp[i_slice,g]
            Prob_Diff = Prob.Diff[i_slice,g]
            Prob_Sed = Prob.Sed[i_slice,g] 

            Prob_Tot = 1 - (1 - Prob_Imp) * (1 - Prob_Diff) * (1 - Prob_Sed)

            #Amount deposited in generation g due to slices:
            amount.Tot[g] = amount.Tot[g] + Prob_Tot * amountSl_still_in_air[i_slice]

            Calc_Separated_Depo(g,
                                amountSl_still_in_air[i_slice],
                                amount,
                                Prob_Tot,
                                Prob_Imp,
                                Prob_Diff,
                                Prob_Sed)
            
            amountSl_still_in_air[i_slice] = amountSl_still_in_air[i_slice] * (1 - Prob_Tot)
    return amountSl_still_in_air

@jit(cache=True, nopython=True)
def Calc_Separated_Depo(g: int,
                        amountSl_still_in_air: float,
                        amount: AmountDepoSingle_Class,
                        prob_Tot: float,
                        Prob_Imp, # probably needs float here
                        Prob_Diff,
                        Prob_Sed):
    """Calculates deposition due to the different deposition mechanics"""
    if Prob_Imp + Prob_Diff + Prob_Sed > 0:
        p_scaling = prob_Tot / (Prob_Imp + Prob_Diff + Prob_Sed)
    else:
        p_scaling = 1

    amount.Imp[g] = amount.Imp[g] + p_scaling * Prob_Imp * amountSl_still_in_air
    amount.Diff[g] = amount.Diff[g] + p_scaling * Prob_Diff * amountSl_still_in_air
    amount.Sed[g] = amount.Sed[g] + p_scaling * Prob_Sed * amountSl_still_in_air

@jit(cache=True, nopython=True)
def DepositionAmount_BH(N_generations: int,
                        N_slices: int,
                        amount: AmountDepoSingle_Class,
                        amountSl_still_in_air: float,
                        BH_fraction: float,
                        Prob: ProbSliceSingle_Class):
    for i_slice in range(0,N_slices):
        delta_amount = 0

        for g in range(0,N_generations):
            if BH_fraction[i_slice, g] > 0:
                amount.Entrance[g] = amount.Entrance[g] + amountSl_still_in_air[i_slice] * BH_fraction[i_slice,g]
                Prob_Imp = Prob.Imp[i_slice,g]
                Prob_Diff = Prob.Diff[i_slice,g]
                Prob_Sed = Prob.Sed[i_slice,g] 

                Prob_Tot = 1 - (1 - Prob_Imp) * (1 - Prob_Diff) * (1 - Prob_Sed)

                delta_amount = delta_amount + Prob_Tot * amountSl_still_in_air[i_slice] * BH_fraction[i_slice,g]

                amount.Tot[g] = amount.Tot[g] + Prob_Tot * amountSl_still_in_air[i_slice] * BH_fraction[i_slice,g]

                Calc_Separated_Depo(g,
                                    amountSl_still_in_air[i_slice] * BH_fraction[i_slice,g],
                                    amount,
                                    Prob_Tot,
                                    Prob_Imp,
                                    Prob_Diff,
                                    Prob_Sed)
                
        amountSl_still_in_air[i_slice] = amountSl_still_in_air[i_slice] -delta_amount
    return amountSl_still_in_air
