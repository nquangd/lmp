from numba import jit
from .Lung_classes import *
from .PhysicalProp_classes import *
from .Calc_functions import *
from .Lung_deposition import *
from .deposition_PostCalc import *
from .Integral_flow import *
from ...config.constants import *


@jit(cache=True, nopython=True)
def Lung_deposition_packageVariable(sub_lung:float, v_lung:float, psd:float, mtdepo:float, flowrate:float, breath_hold_time: float, Flow_Exh:float, BolusVolume:float, Bolus_Delay:float):
    #====================================
    #        Initiate variables
    #====================================
   

    Lung_size = v_lung / 1e6 #m^3 This is for other references

    # FLOW RATE SET UP
   
    
    #flowrate[:,1] =  flowrate[:,1] * 1e-3 / 60
    N_Q_Inh = len(flowrate) - 1   
    N_Q_Exh = len(flowrate)  - 1
    N_Q_max_len = len(flowrate)
   
              
    Inhaled_Volume = Integral_flow(N_Q_Inh, flowrate[:,1], flowrate[:,0], flowrate[-2,0])  #2078.205e-6 # Need to do the integral here to update
    
    
    dt_ex = Inhaled_Volume / Flow_Exh /  N_Q_Exh #const_Flow
    ex_time = np.arange(N_Q_Exh + 1) * dt_ex
    
    #bolusdelay = Bolus_Delay * 1.0
            
         
    #Flow_Inh = 500e-6 #m^3/s   # unused for variable flow
    #Flow_Exh = 500e-6 #m^3/s   # assume constant exhalation flowrate
    #BolusVolume = 200e-6  # 200ml

    # SIZE DISTRIBUTION

    #Size distributions: Aerodynamic and geometric radius for carrier particles. poly_in is the geometric radius for API
    
    N_bins = N_bins_max #20

    #bins = np.linspace(0.7, 15, N_bins)

    d_a, d_g, v_f = psd[:,0], psd[:,1], psd[:,2]

    ###############
    N_start_generation = LungSlice_Class()
    amountSl_still_in_air = LungSliceReal8_Class()

    LungTube = LungTube_Class(sub_lung[:,0], sub_lung[:,1], sub_lung[:,2], sub_lung[:,3], sub_lung[:,4], sub_lung[:,5], sub_lung[:,6], sub_lung[:,7] )
    
    LungFlow = LungFlow_Class(N_Q_max_len)
    V_slice = LungSliceReal8B_Class()
    N_slices = LungSliceB_Class()

    Max_slice_reaches_generation = LungSlice_Class()
    
    LungSliceTube = LungSliceTube_Class()

    Depo = LungDepo_Class()
    AirProp = AirProp_Class()
    H2OProp = H2OProp_Class()
    AeroProp = AeroProp_Class()
    PhCo = PhCo_Class()
    r = r_Class()
    UndissAero_AD = SizeDistr_Class()
    UndissAero_GD = SizeDistr_Class()
    Undiss_poly_in = SizeDistr_Class()
    Body = Body_Class()
    
    
    #====================================
    #        Parameter definition
    #==================================== 
    #Simulatin settings
    Depo.iFLAG_KingsCollege = 0
    Depo.iFLAG_DoSedDiff = 1
    Depo.iFLAG_DoSedDiff_From_VB = 1
    Depo.ET_Model = 3 #1: NCRP96, 3:Manual deposition
    Depo.CoarseFraction = 0.0
    Depo.FLAG_Thoracic_manual_deposition = 0

    Depo.ET_manual_deposition = mtdepo #0.5# 0.65
    Depo.BHT = breath_hold_time #10 #s


    # Lung settings (Lung physiology parameters are stored in Lung_classes.py)
    
    N_Generations = N_GIcomps_LungGens_max #25 #24 generations (0-23) + ET
    #V_slice.sliceVolume_const(15e-6) #Set same volume for all slices in m^3
    V_slice.sliceVolume_const(V_slice_const) #Set same volume for all slices in m^3

        
    #np.insert(d_a[:-1], 0, 0.5, axis = 0)
    adrmin = np.zeros(N_bins)
    adrmin[0] = 0.5
    adrmin[1:] = d_a[:-1]

    #np.insert(d_g[:-1], 0, 0.5/1.01, axis = 0)
    gdrmin = np.zeros(N_bins)
    gdrmin[0] = 0.5/1.01
    gdrmin[1:] = d_g[:-1]


    UndissAero_AD.Rmin[0, :] = 1/2 * 1e-6 * adrmin #np.insert(d_a[:-1], 0, 0.5, axis = 0) #np.array([0.5, 0.83026234,	0.96544454,	1.1226369,	1.3054232,	1.5179704,	1.7651243,	2.0525194,	2.3867079,	2.7753084,	3.2271805,	3.7526257,	4.3636231,	5.0741023,	5.9002608,	6.8609333,	7.9780212,	9.2769919,	10.787459, 12.543859])
    UndissAero_AD.Rmax[0, :] = 1/2 * 1e-6 * d_a # np.array([0.83026234,	0.96544454,	1.1226369,	1.3054232,	1.5179704,	1.7651243,	2.0525194,	2.3867079,	2.7753084,	3.2271805,	3.7526257,	4.3636231,	5.0741023,	5.9002608,	6.8609333,	7.9780212,	9.2769919,	10.787459,	12.543859,	14.5862339])
    UndissAero_AD.X_bin[0, :] = v_f # 1/100 * np.array([0.0, 0.1850, 0.4513, 0.9911, 1.960, 3.489, 5.594, 8.075, 10.50, 12.28, 12.95, 12.28, 10.50, 8.075, 5.594, 3.489, 1.960, 0.9911, 0.4513, 0.1842])
    
    UndissAero_GD.Rmin[0, :] = 1/2 * 1e-6 * gdrmin #np.insert(d_g[:-1], 0, 0.5/1.01, axis = 0) # np.array([0.4, 0.75126608,	0.87467377,	1.0181512,	1.1849766,	1.3789617,	1.6045366,	1.8668489,	2.1718817,	2.5265904,	2.9390621,	3.4187005,	3.9764404,	4.6249973,	5.3791571,	6.256113,	7.2758579,	8.46164,	9.8404926,	11.443851])
    UndissAero_GD.Rmax[0, :] = 1/2 * 1e-6 *  d_g #np.array([0.75126608,	0.87467377,	1.0181512,	1.1849766,	1.3789617,	1.6045366,	1.8668489,	2.1718817,	2.5265904,	2.9390621,	3.4187005,	3.9764404,	4.6249973,	5.3791571,	6.256113,	7.2758579,	8.46164,	9.8404926,	11.443851,	13.308268])
    UndissAero_GD.X_bin[0, :] = UndissAero_AD.X_bin[0, :]

    #Undiss_poly_in.Rmin[0,:] = 1/2 * 1e-6 * np.array([0.0,  0.5,  1.0,  1.5,  2.0,  2.5,  3.0,  3.5,  4.0,  4.5,  5.0,  6.0,  7.5,  9.0, 10.5, 12.0, 13.5, 15.0, 16.5, 18.0])
    #Undiss_poly_in.Rmax[0,:] = 1/2 * 1e-6 * np.array([0.5,  1.0,  1.5,  2.0,  2.5,  3.0,  3.5,  4.0,  4.5,  5.0,  6.0,  7.5,  9.0, 10.5, 12.0, 13.5, 15.0, 16.5, 18.0, 19.5])
    #Undiss_poly_in.X_bin[0,:] = np.array([0.14578960, 0.09285479, 0.06657523, 0.05171062, 0.04207024, 0.03529327, 0.03026654, 0.02639141, 0.02331560, 0.02081763, 0.03576589, 0.04296636, 0.03412255, 0.02796080, 0.02345031, 0.02002472, 0.01734760, 0.01520691, 0.01346270, 0.01201899+0.2225883])
    
    Undiss_poly_in.Rmin[0,:] = 1/2 * 1e-6 * adrmin
    Undiss_poly_in.Rmax[0,:] = 1/2 * 1e-6 * d_a
    Undiss_poly_in.X_bin[0,:] = v_f

    #====================================
    #        Start of program
    #====================================
    N_slices.set_N_slices(Inhaled_Volume, V_slice)
 

    LungFlow.Inh.Q[0:(N_Q_Inh+1)] = flowrate[:,1]  #* 1e-3 / 60
    LungFlow.Exh.Q[0:(N_Q_Exh+1)] = Flow_Exh 
    
    LungFlow.Inh.set_t(N_Q_Inh, Inhaled_Volume, flowrate[:,0], Depo.BHT, Flow_Exh, 0, 1)
    LungFlow.Exh.set_t(N_Q_Exh, Inhaled_Volume, ex_time, Depo.BHT, Flow_Exh, 0, 3)
    
    #LungFlow.Exh.set_t(N_Q_Exh, Inhaled_Volume, flowrate[:,0], Depo.BHT, Flow_Exh, 0, 3)
    
    LungFlow.Inh.set_kappa(N_Q_Inh, Lung_size, Inhaled_Volume, 1)
    LungFlow.Exh.set_kappa(N_Q_Exh, Lung_size, Inhaled_Volume, 3)
    LungTube.convert_k_expansion_frac() 

    
    UndissAero_GD.k[0,:] = Convert_X_undiss_bin_2_k_undiss(N_bins,
                                                            1,
                                                            UndissAero_GD.Rmax[0,:],
                                                            UndissAero_GD.Rmin[0,:],
                                                            UndissAero_AD.X_bin[0,:])

    
    

    #print("Calculating slice entrance times")
    #Calculate flow to generation entrances
    Q_flow_Inh_gen, Q_flow_Exh_gen = Calc_Flow_entrances(N_Q_Inh,
                                                         N_Q_Exh,
                                                         N_Generations,
                                                         LungTube,
                                                         LungFlow)
    
    

    #Calculate time for ET entrance
    InTime_ET_Inh = Calc_ET_entrance_time(N_Q_Inh,
                                          N_slices.Inh,
                                          LungFlow.Inh, 
                                          V_slice.Inh
                                          )
    InTime_ET_Exh = Calc_ET_entrance_time(N_Q_Exh,
                                          N_slices.Exh,
                                          LungFlow.Exh,
                                          V_slice.Exh
                                          )
   
    
    #print("Creating slices and allocating drug")
    #Calculate fraction of dose in each slice
    # This is where Bolus delayed accounted
    
    Fraction_slice = Bolus_dose_location(BolusVolume, V_slice.Inh, InTime_ET_Inh, Bolus_Delay, N_slices )
    
    #Fraction_slice = Bolus_dose_location(BolusVolume, V_slice.Inh, N_slices) # OLD
    
    
    #Calculate Start_generation
    N_start_generation.Inh = Calc_Start_generation(N_slices_max,
                                                   N_slices.Inh)
    N_start_generation.Exh = Calc_Start_generation(N_slices_max,
                                                   N_slices.Exh)
    
    
    #Initialize breath hold deposition
    Initialize_BH(N_slices, V_slice, N_start_generation)

    """
    Nils to Duy: The above are tested, except Q_flow_Inh_gen and Q_flow_Exh_gen which describe the flow into each generation which have been compared to manual calculation of the expressions in the help-document instead of fortran code values, since those were unavailable to me
    """
    
    #print("Calculating slice residence times")

  

    #Calculate residence time for Inh, BH & Exh in each generation, aswell as Max_slice_reaches_generation which describes how far each slice gets.
    BH_fraction = Calc_residence_time(N_Q_Inh,
                                      N_Q_Exh,
                                      N_slices,
                                      N_Generations,
                                      InTime_ET_Inh,
                                      Q_flow_Inh_gen,
                                      LungFlow,
                                      InTime_ET_Exh,
                                      Q_flow_Exh_gen,
                                      LungTube,
                                      Max_slice_reaches_generation,
                                      LungSliceTube,
                                      Depo,
                                      V_slice.BH)
    
    """
    Nils to Duy: I've tested Max_slice_reaches_generation, but haven't been able to test LungSliceTube.Inh/BH/Exh.ResidenceTime which describes for how long each slice stays in each generation since it can't be accesed in LungSIM and
                 I don't have acces to the entire LungSim code, just the algorithm, and can hence not run the Fortran code in parallell to inspect the values.

    """
    #print("Calculating average geometries")

    #Calculate average radii of generations, average_geometries
    Calc_average_geometries(N_Q_Inh,
                            N_Q_Exh,
                            N_slices,
                            N_Generations,
                            AirProp,
                            LungFlow,
                            LungTube,
                            LungSliceTube,
                            N_start_generation,
                            Max_slice_reaches_generation) 
    
    """
    Nils to Duy: I haven't been able to check the Ave_Q, Ave_Tube_Radius, etc. values stored in LungSliceTube. To counter this I've gone through the code for Calc_Average_geometries and Calc_residence_time and compared it to the Fortran code several times. 
                 If a better discretization doesn't reduce the discrepency this is a good place to start further testing, especially since the Ave_Tube_Radius is used as an input in probability calculations.
    """
    
    #print("Calculating lung deposition")

    #Calculate deposition in lung generations
    for j in range(0,N_bins):
        Lung_deposition_Variable(j,
                                 N_slices,
                                 N_Generations,
                                 Max_slice_reaches_generation,
                                 N_start_generation,
                                 V_slice,
                                 Fraction_slice,
                                 LungSliceTube,
                                 amountSl_still_in_air,
                                 AirProp,
                                 LungTube.V_air[0],
                                 LungTube.V_air,
                                 LungTube,
                                 r,
                                 UndissAero_AD,
                                 UndissAero_GD,
                                 Depo,
                                 H2OProp,
                                 AeroProp,
                                 PhCo,
                                 Body,
                                 BH_fraction)
        
    """
    Nils to Duy: The Lung deposition equations have been tested. 
    """

    #print("Lung deposition done :)")
    

    #Convert from carrier to API
    Lung_deposition_PostCalc(N_bins,
                             N_Generations,
                             Depo,
                             r,
                             UndissAero_GD,
                             Undiss_poly_in,
                             Aerosphere_alpha=0,
                             FLAG_Aerosphere_Area=0)
    """
    Postcalc have been tested
    """
    
    ##print("Post Postcalc")
    deposition_tot_res = summarize(r, Undiss_poly_in.X_bin, N_Generations)
    
    #return [deposition_tot_res[0],np.sum(deposition_tot_res[1:10]), np.sum(deposition_tot_res[10:17]),np.sum(deposition_tot_res[17:-1])]
    return deposition_tot_res

@jit(cache=True, nopython=True)
def summarize(r: r_Class, bins:float, N_generations:int):
    

    dep_Inh = np.zeros(N_GIcomps_LungGens_max+1)
    dep_BH = np.zeros(N_GIcomps_LungGens_max+1)
    dep_Exh = np.zeros(N_GIcomps_LungGens_max+1)
    dep_Tot = np.zeros(N_GIcomps_LungGens_max+1)

    for g in range(0,N_generations+1):
        dep_Inh[g] = np.sum(r.Inh.Tot[g,:] * bins[0,:])
        dep_BH[g] = np.sum(r.BH.Tot[g,:] * bins[0,:])
        dep_Exh[g] = np.sum(r.Exh.Tot[g,:] * bins[0,:])
        dep_Tot[g] = np.sum(r.TOT.Tot[g,:] * bins[0,:])

    
    ##print(dep_Tot)
    return dep_Tot
    

@jit(cache=True, nopython=True)
def Bolus_dose_location(BolusVolume, V_slice, InTime_ET_Inh, Bolus_Delay, N_slices: LungSliceB_Class ):
    
    cumsum = 0.0
    Fraction_slice = np.zeros(N_slices.Inh) #np.array([0.00]*N_slices.Inh)
    start_bolus = np.argwhere(InTime_ET_Inh > Bolus_Delay)[0][0] - 1
    
    for i in range(start_bolus, N_slices.Inh):
        cumsum = cumsum + V_slice[i]
        if BolusVolume >= cumsum:
            Fraction_slice[i] = V_slice[i]
        elif BolusVolume > cumsum - V_slice[i]:
            Fraction_slice[i] = BolusVolume - cumsum + V_slice[i]
        else:
            Fraction_slice[i] = 0
    Fraction_slice = Fraction_slice / BolusVolume
    return Fraction_slice
            
        
            
@jit(cache=True, nopython=True)
def backup_Bolus_dose_location(BolusVolume, V_slice, N_slices: LungSliceB_Class):
    
    cumsum = 0.0
    Fraction_slice = np.zeros(N_slices.Inh) #np.array([0.00]*N_slices.Inh)
    for i in range(0, N_slices.Inh):
        cumsum = cumsum + V_slice[i]
        if BolusVolume >= cumsum:
            Fraction_slice[i] = V_slice[i]
        elif BolusVolume > cumsum - V_slice[i]:
            Fraction_slice[i] = BolusVolume - cumsum + V_slice[i]
        else:
            Fraction_slice[i] = 0
    Fraction_slice = Fraction_slice / BolusVolume
    return Fraction_slice

