import numpy as np
from numba import njit, jit, int32, float64

from .Integral_flow import *
from .Deposition_classes import *
from .Lung_classes import *
from .PhysicalProp_classes import *
from .Parameters_Settings import * 




@jit(cache=True, nopython=True)
def Calc_Flow_entrances(N_Q_Inh: int, 
                        N_Q_Exh: int,
                        N_Generations: int, 
                        LungTube: LungTube_Class, 
                        LungFlow: LungFlow_Class):
    """
    Calculates the airflow into each generation from the expansion coefficient k and the lung volume when the flow into/out of ET is known. 
    The flows are calculated for each time point in the in- and outflow arrays. See section 3.3 and more specifically eq. 85, p. 20, in the lung deposition documentation for LungSIM.
    
    Inputs:
    N_Q_Inh, N_Q_Exh: Number of timepoints in in- and outflow arrays
    N_Generations: Number of lung generations
    LungTube: Object with Lung data
    LungFlow: Object with flow data

    Output:
    Q_flow_Inh_gen, Q_flow_Exh_gen: flow in each generation
    """
    sum_weight = 0.0
    weight=np.array([0.0]*N_Generations)
    cum_weight=np.array([0.0]*N_Generations)
    for g in range(0, N_Generations):
        weight[g] = LungTube.k_expansion_frac[g] * LungTube.V_air[g]
        cum_weight[g] = sum_weight
        sum_weight = sum_weight + weight[g]
    
    sum_weight = max(sum_weight, 1e-100)

    Q_flow_Inh_gen = np.zeros((N_Q_Inh, N_Generations))
    Q_flow_Exh_gen = np.zeros((N_Q_Inh, N_Generations))

    #Inhalation
    for g in range(0,N_Generations):
        for i in range(0,N_Q_Inh):
            Q_flow_Inh_gen[i,g] = LungFlow.Inh.Q[i] * (1-cum_weight[g]/sum_weight) #Inhaled lungflow minus the fraction thats been used to expand previous generations

    #Exhalation
    for g in range(0,N_Generations):
        for i in range(0,N_Q_Exh):
            Q_flow_Exh_gen[i,g] = LungFlow.Exh.Q[i] * (1-cum_weight[g]/sum_weight)

    return Q_flow_Inh_gen, Q_flow_Exh_gen



@jit(cache=True, nopython=True)
def Calc_ET_entrance_time(N_Q: int,
                          N_slices: int,
                          LungFlow: LungFlow_B_Class,
                          V_slice: float):
    """
    Calculates the time each slice enters ET. See section 3.3.1, eq. 87 from which we in this function calculate T.

    Inputs: 
    N_Q
    N_slices: number of slices inhaled
    LungFlow: Object with flow data and parameters
    V_slice: Volume of each slice, in array format

    Output:
    InTime_ET: Time T when each slice enters the ET region
    """
    InTime_ET = np.zeros(N_slices)
    

    i_slice = int(0)
    InTime_ET[i_slice] = 0.0
    V_slice_cum = 0.0
    
    for i_slice in range(1,N_slices):
        V_slice_cum = V_slice_cum + V_slice[i_slice-1]
        time_a = InTime_ET[i_slice-1]
        fa = Integral_flow(N_Q,
                           LungFlow.Q,
                           LungFlow.t,
                           time_a) - V_slice_cum
        

        time_b = LungFlow.t[N_Q-1]
        fb = Integral_flow(N_Q,
                           LungFlow.Q,
                           LungFlow.t,
                           time_b) - V_slice_cum

        
        while(True):
            time_m = (time_a + time_b)/2
            fm = Integral_flow(N_Q,
                                LungFlow.Q,
                                LungFlow.t,
                                time_m) - V_slice_cum
        
            if fa*fm >= 0: #Check if fm is the new upper or lower limit, since the true vale of fm would be 0
                time_a=time_m
                fa=fm
            else:
                time_b=time_m
                fb=fm
            
            if(np.abs(time_b - time_a) > 1e-10):
                continue
            else:
                break
        
        InTime_ET[i_slice] = time_m
    
    return InTime_ET

@jit(cache=True, nopython=True)
def Calc_Start_generation(N_slices_max: int,
                          N_slices: int):
    """
    Calculates where in the lung each slice is initiated. 
    Currently, the lung is assumed to be empty before inhalation and each slice is initiated in the ET
    """
    N_start_generation_val = np.array([0]*N_slices)
    for i_slice in range(0,N_slices):
        N_start_generation_val[i_slice]=0

    return N_start_generation_val
@jit(cache=True, nopython=True)
def Calc_residence_time(N_Q_Inh: int,
                        N_Q_Exh: int,
                        N_slices: LungSliceB_Class,
                        N_Generations: int,
                        InTime_ET_Inh: float,
                        Q_flow_Inh_gen: float,
                        LungFlow: LungFlow_Class,
                        InTime_ET_Exh: float,
                        Q_flow_Exh_gen: float,
                        LungTube: LungTube_Class,
                        Max_slice_reaches_generation: LungSlice_Class,
                        LungSliceTube: LungSliceTube_Class,
                        Depo: LungDepo_Class,
                        V_slice: float) -> float:
    """
    Calculates the time each slice stays in each generation.
    """
    
    #Adjust Lung geometry for manual deposition in upper regions
    Assign_V_air_local(N_Generations,
                       LungTube,
                       Depo) 
    
    #Inhalataion
    Max_slice_reaches_generation.Inh = Calc_residence_time_InhExh(N_Q_Inh,
                                                                  N_Generations,
                                                                  N_slices.Inh,
                                                                  LungSliceTube.Inh,
                                                                  InTime_ET_Inh,
                                                                  Q_flow_Inh_gen,
                                                                  LungFlow.Inh,
                                                                  LungTube) 
    
    #Breath hold
    for i_slice in range(0,N_slices.BH):
        Max_slice_reaches_generation.BH[i_slice] = Max_slice_reaches_generation.Inh[i_slice]
        for g in range(0,N_Generations):
            if Max_slice_reaches_generation.BH[i_slice] == g:
                LungSliceTube.BH.ResidenceTime[i_slice,g] = Depo.BHT
            else:
                LungSliceTube.BH.ResidenceTime[i_slice,g] = 0

    #Exhalation
    Max_slice_reaches_generation.Exh = Calc_residence_time_InhExh(N_Q_Exh,
                                                                  N_Generations,
                                                                  N_slices.Exh,
                                                                  LungSliceTube.Exh,
                                                                  InTime_ET_Exh,
                                                                  Q_flow_Exh_gen,
                                                                  LungFlow.Exh,
                                                                  LungTube)
    BH_fraction = np.zeros((N_slices.Inh, N_Generations))
    #Calculate breath hold fractional slices in generations
    for i_slice in range(0,N_slices.Inh):
        g = Max_slice_reaches_generation.BH[i_slice]
        time = LungSliceTube.Inh.OutTime[i_slice,g]
        V_flow = Integral_flow(N_Q_Inh,
                               Q_flow_Inh_gen[:,0],
                               LungFlow.Inh.t,
                               time)
        kappa = kappa_interpolate(N_Q_Inh,
                                  LungFlow.Inh,
                                  time)
        
        #Lite fel nedan. Jan har ej tagit hänsyn till att V_slice kan variera.
        V_forefront = V_flow - (i_slice)*V_slice[0] #flödet före slice i
        V_backfront = V_forefront - V_slice[i_slice] #flödet efter slice i

        V_cum1 = 0
        for g2 in range(0,N_Generations):
            V_cum2 = V_cum1 + LungTube.loc_V_air[g2] * (1 + LungTube.k_expansion_frac[g2] * kappa)
            V_min = max(V_cum1, V_backfront)
            V_max = min(V_cum2, V_forefront)
            dV = max(V_max - V_min, 0)
            BH_fraction[i_slice,g2] = dV/V_slice[i_slice]
            V_cum1 = V_cum2
    
    return BH_fraction

##@jit(cache=True)(float64[:](int32, int32, int32, float64[:], float64, float64,float64, float64))

#@jit(cache=True)(int32(int32, int32, int32, LungSliceTube_C_Class_type, float64, float64, LungFlow_B_Class_type, LungTube_Class_type))
@jit(cache=True, nopython=True)
def Calc_residence_time_InhExh(N_Q: int,
                               N_Generations: int,
                               N_slices: int,
                               LungSliceTube: LungSliceTube_C_Class,
                               InTime_ET: float,
                               Q_flow_gen: float,
                               LungFlow: LungFlow_B_Class,
                               LungTube: LungTube_Class) -> int:
    """Calculates the time each slice spends in each generation, and the maximal generation each slice reaches."""

    Max_slice_reaches_generation = np.array([0]*N_slices)
    #FLAG_slice_reaches_outside = 0
    for i_slice in range(0,N_slices):
        Max_slice_reaches_generation[i_slice] = 0
        LungSliceTube.InTime[i_slice,0] = InTime_ET[i_slice]

        for g in range(0,N_Generations):
            if Max_slice_reaches_generation[i_slice] == g:
                if LungTube.Radius[g] > 1.0 or LungTube.Length[g] > 1.0:
                    FLAG_slice_reaches_outside = Calc_Passage_time_V(i_slice,
                                                                     g,
                                                                     #N_Q_max, is in Jans calling of the function, but function dont have that many arguments
                                                                     N_Q,
                                                                     LungSliceTube,
                                                                     Q_flow_gen[:,g],
                                                                     LungFlow.t,
                                                                     LungTube.loc_V_air[g],
                                                                     0)
                else:
                    FLAG_slice_reaches_outside = Calc_Passage_time_V2(i_slice,
                                                                     g,
                                                                     N_Q,
                                                                     LungSliceTube,
                                                                     Q_flow_gen[:,g],
                                                                     LungFlow,
                                                                     LungTube.loc_V_air[g],
                                                                     0,
                                                                     LungTube.k_expansion_frac[g])

                LungSliceTube.ResidenceTime[i_slice,g] = LungSliceTube.OutTime[i_slice,g] - LungSliceTube.InTime[i_slice,g]
                if FLAG_slice_reaches_outside == 1:
                    Max_slice_reaches_generation[i_slice] = Max_slice_reaches_generation[i_slice] + 1
                    LungSliceTube.InTime[i_slice,g+1] = LungSliceTube.OutTime[i_slice,g]
            else:
                LungSliceTube.Ave_Q[i_slice,g] = 0
                LungSliceTube.ResidenceTime[i_slice,g] = 0
            
    return Max_slice_reaches_generation    



#@jit(cache=True)(int32(int32, int32, int32, LungSliceTube_Class.class_type.instance_type, float64, float64, float64, float64))
@jit(cache=True, nopython=True)
def Calc_Passage_time_V(i_slice: int,
                        g: int,
                        N_Q: int,
                        LungSliceTube: LungSliceTube_Class,
                        Q_flow: float,
                        LungFlow_t: float,
                        V0: float,
                        delta_V: float) -> int:
    """Calculates the time slice i leaves the generation g for non expanding regions"""
    
    Max_time = LungFlow_t[N_Q-1] # Nils
    #Max_time = LungFlow_t[-1]  #Duy

    #First trial
    OutTime_a = LungSliceTube.InTime[i_slice,g]
    V_outtime = V0 - delta_V
    f_a = Integral_flow(N_Q, Q_flow, LungFlow_t, OutTime_a) - Integral_flow(N_Q, Q_flow, LungFlow_t, LungSliceTube.InTime[i_slice,g]) - V_outtime

    #Second trial
    OutTime_b = Max_time
    f_b = Integral_flow(N_Q, Q_flow, LungFlow_t, OutTime_b) - Integral_flow(N_Q, Q_flow, LungFlow_t, LungSliceTube.InTime[i_slice,g]) - V_outtime
    
    if f_b < 0:
        #In this case, the slice did not reach outside of the generation
        LungSliceTube.OutTime[i_slice,g] = Max_time
        FLAG_slice_reaches_outside = 0
        LungSliceTube.TubeLength_Fraction[i_slice,g] = (f_b + V_outtime)/V_outtime
        LungSliceTube.Ave_Q[i_slice,g] = (f_b + V_outtime) / (LungSliceTube.OutTime[i_slice,g] - LungSliceTube.InTime[i_slice,g])
    else:
        #In this case, the slice did reach outside of the generation and we apply the bisection method
        while(True):
            OutTime_m = (OutTime_a + OutTime_b)/2
            f_m = Integral_flow(N_Q, Q_flow, LungFlow_t, OutTime_m) - Integral_flow(N_Q, Q_flow, LungFlow_t, LungSliceTube.InTime[i_slice,g]) - V_outtime

            if f_a*f_m > 0:
                OutTime_a = OutTime_m
                f_a = f_m
            else:
                OutTime_b = OutTime_m
                f_b = f_m
            
            if abs(OutTime_b-OutTime_a) > 1e-12:
                continue
            else:
                break

        #Finalizing
        LungSliceTube.OutTime[i_slice,g] = OutTime_m
        FLAG_slice_reaches_outside = 1
        LungSliceTube.TubeLength_Fraction[i_slice,g] = 1
        LungSliceTube.Ave_Q[i_slice,g] = (f_m + V_outtime) / (LungSliceTube.OutTime[i_slice,g] - LungSliceTube.InTime[i_slice,g])
    
    return FLAG_slice_reaches_outside   
#@jit(cache=True)(int32(int32, int32, int32, LungSliceTube_C_Class_type, float64, LungFlow_B_Class_type, float64, float64, float64))
@jit(cache=True, nopython=True)
def Calc_Passage_time_V2(i_slice: int,
                         g: int,
                         N_Q: int,
                         LungSliceTube: LungSliceTube_C_Class,
                         Q_flow: float,
                         LungFlow: LungFlow_B_Class,
                         V0: float,
                         delta_V: float,
                         k_expansion_frac: float) -> int:
    """Calculats the time slice i leavs generation g for expanding generations"""

    Max_time = LungFlow.t[N_Q-1] # Nils
    #Max_time = LungFlow.t[-1]  # duy
    InTime = LungSliceTube.InTime[i_slice,g]

    #First trial
    OutTime_a = InTime
    kappa_a = kappa_interpolate(N_Q,
                                LungFlow,
                                OutTime_a)
    V_outtime = V0 * (1 + k_expansion_frac * kappa_a) - delta_V

    f_a = Integral_flow(N_Q, Q_flow, LungFlow.t, OutTime_a) - Integral_flow(N_Q, Q_flow, LungFlow.t, InTime) - V_outtime

    #Second trial
    OutTime_b = Max_time

    kappa_b = kappa_interpolate(N_Q,
                                LungFlow,
                                OutTime_b)
    V_outtime = V0 * (1 + k_expansion_frac *kappa_b) - delta_V

    f_b = Integral_flow(N_Q, Q_flow, LungFlow.t, OutTime_b) - Integral_flow(N_Q, Q_flow, LungFlow.t, InTime) - V_outtime

    if f_b < 0:
        #In this case, the slice did not reach outside of the generation
        LungSliceTube.OutTime[i_slice,g] = Max_time
        FLAG_slice_reaches_outside = 0
        if abs(delta_V) < 1e-100:
            LungSliceTube.TubeLength_Fraction[i_slice,g] = (f_b + V_outtime)/V_outtime
        else:
            LungSliceTube.TubeLength_Fraction[i_slice,g] = (f_b + V_outtime - V0 * k_expansion_frac * kappa_b) / (V_outtime - V0 * k_expansion_frac * kappa_b)
     
        LungSliceTube.Ave_Q[i_slice,g] = (f_b + V_outtime) / (LungSliceTube.OutTime[i_slice,g] - InTime)
    else:
        #In this case the slice reached outside the generation.
        while(True):
            OutTime_m = (OutTime_a + OutTime_b)/2
            kappa_m = kappa_interpolate(N_Q,
                                        LungFlow,
                                        OutTime_m)
            V_outtime = V0 * (1+ k_expansion_frac * kappa_m) - delta_V

            f_m = Integral_flow(N_Q, Q_flow, LungFlow.t, OutTime_m) - Integral_flow(N_Q, Q_flow, LungFlow.t, InTime) - V_outtime

            if f_a*f_m > 0:
                OutTime_a = OutTime_m
                f_a = f_m
            else:
                OutTime_b = OutTime_m
                f_b = f_m
        
            if abs(OutTime_b-OutTime_a) > 1e-12:
                continue
            else:
                break

        #Finalizing
        LungSliceTube.OutTime[i_slice,g] = OutTime_m
        FLAG_slice_reaches_outside = 1
        LungSliceTube.TubeLength_Fraction[i_slice,g] = 1
        LungSliceTube.Ave_Q[i_slice,g] = (f_m + V_outtime) / (OutTime_m - InTime)

    return FLAG_slice_reaches_outside

@jit(cache=True, nopython=True)    
def Calc_average_geometries(N_Q_Inh: int,
                            N_Q_Exh: int,
                            N_slices: LungSliceB_Class,
                            N_Generations: int,
                            AirProp: AirProp_Class,
                            LungFlow: LungFlow_Class,
                            LungTube: LungTube_Class,
                            LungSliceTube: LungSliceTube_Class,
                            N_start_generation: LungSlice_Class,
                            Max_slice_reaches_generation: LungSlice_Class):
    """Calculates the average radius and length of generation g while slice i passes through it, and velocity of slice i"""

    #Inhalation
    for i_slice in range(0,N_slices.Inh):
        for g in range(N_start_generation.Inh[i_slice],Max_slice_reaches_generation.Inh[i_slice]+1):
            LungSliceTube.Inh.Ave_Tube_Radius[i_slice,g] = Calc_Ave_Tube_1D(i_slice,
                                                                            g,
                                                                            N_Q_Inh,
                                                                            LungSliceTube.Inh,
                                                                            LungFlow.Inh,
                                                                            LungTube.Radius[g],
                                                                            LungTube.k_expansion_frac[g])
            
            LungSliceTube.Inh.Ave_Tube_Length[i_slice,g] = Calc_Ave_Tube_1D(i_slice,
                                                                            g,
                                                                            N_Q_Inh,
                                                                            LungSliceTube.Inh,
                                                                            LungFlow.Inh,
                                                                            LungTube.Length[g],
                                                                            LungTube.k_expansion_frac[g])
            
            #Ave_Tube_RadiusII = Calc_Ave_Tube_1D(i_slice,
            #                                     g,
            #                                     N_Q_Inh,
            #                                     LungSliceTube.Inh,
            #                                     LungFlow.Inh,
            #                                     LungTube.RadiusII[g],
            #                                     LungTube.k_expansion_frac[g])#Not in use???
            
            if LungSliceTube.Inh.Ave_Tube_Length[i_slice,g] < 1:
                LungSliceTube.Inh.Ave_Velocity[i_slice,g] = LungSliceTube.Inh.TubeLength_Fraction[i_slice,g] * LungSliceTube.Inh.Ave_Tube_Length[i_slice,g] / (LungSliceTube.Inh.OutTime[i_slice,g] - LungSliceTube.Inh.InTime[i_slice,g])
                LungSliceTube.Inh.Ave_Reynold[i_slice,g] = Calc_Reynold(LungSliceTube.Inh.Ave_Velocity[i_slice,g],
                                                                          LungSliceTube.Inh.Ave_Tube_Radius[i_slice,g],
                                                                          AirProp)
            elif LungSliceTube.Inh.Ave_Tube_Length[i_slice,g] >= 1:
                LungSliceTube.Inh.Ave_Velocity[i_slice,g] = -1
                LungSliceTube.Inh.Ave_Reynold[i_slice,g] = -1

    #Breath hold
    for g in range(0,N_Generations):
        kappa = LungFlow.Inh.kappa[N_Q_Inh]
        LungSliceTube.BH.Ave_Tube_Radius[0,g] =Tube_RL(LungTube.Radius[g],
                                                             LungTube.k_expansion_frac[g],
                                                             kappa)
        LungSliceTube.BH.Ave_Tube_Length[0,g] = Tube_RL(LungTube.Length[g],
                                                              LungTube.k_expansion_frac[g],
                                                              kappa)
        for i_slice in range(1,N_slices.Inh):
            LungSliceTube.BH.Ave_Tube_Length[i_slice,g] = LungSliceTube.BH.Ave_Tube_Length[0,g]
            LungSliceTube.BH.Ave_Tube_Radius[i_slice,g] = LungSliceTube.BH.Ave_Tube_Radius[0,g]

    #Exhalation
    for i_slice in range(0,N_slices.Exh):
        for g in range(N_start_generation.Exh[i_slice], Max_slice_reaches_generation.Exh[i_slice]+1): 
            LungSliceTube.Exh.Ave_Tube_Radius[i_slice,g] = Calc_Ave_Tube_1D(i_slice,
                                                                            g,
                                                                            N_Q_Exh,
                                                                            LungSliceTube.Exh,
                                                                            LungFlow.Exh,
                                                                            LungTube.Radius[g],
                                                                            LungTube.k_expansion_frac[g])
            
            LungSliceTube.Exh.Ave_Tube_Length[i_slice,g] = Calc_Ave_Tube_1D(i_slice,
                                                                            g,
                                                                            N_Q_Exh,
                                                                            LungSliceTube.Exh,
                                                                            LungFlow.Exh,
                                                                            LungTube.Length[g],
                                                                            LungTube.k_expansion_frac[g])
            
        #    Ave_Tube_RadiusII = Calc_Ave_Tube_1D(i_slice,
        #                                         g,
        #                                         N_Q_Exh,
        #                                         LungSliceTube.Exh,
        #                                         LungFlow.Exh,
        #                                         LungTube.RadiusII[g],
        #                                         LungTube.k_expansion_frac[g])
            
            if LungSliceTube.Exh.Ave_Tube_Length[i_slice,g] < 1:
                LungSliceTube.Exh.Ave_Velocity[i_slice,g] = LungSliceTube.Exh.TubeLength_Fraction[i_slice,g] * LungSliceTube.Exh.Ave_Tube_Length[i_slice,g] / (LungSliceTube.Exh.OutTime[i_slice,g] - LungSliceTube.Exh.InTime[i_slice,g])
                LungSliceTube.Exh.Ave_Reynold[i_slice,g] = Calc_Reynold(LungSliceTube.Exh.Ave_Velocity[i_slice,g],
                                                                          LungSliceTube.Exh.Ave_Tube_Radius[i_slice,g],
                                                                          AirProp)
            elif LungSliceTube.Exh.Ave_Tube_Length[i_slice,g] >= 1:
                LungSliceTube.Exh.Ave_Velocity[i_slice,g] = -1
                LungSliceTube.Exh.Ave_Reynold[i_slice,g] = -1



@jit(cache=True, nopython=True)
def Calc_Ave_Tube_1D(i_slice,
                     g,
                     N_Q,
                     LungSliceTube: LungSliceTube_Class,
                     LungFlow,
                     RL0,
                     k_expansion_frac):
    """Calculates average tube radii and length"""
    #N_steps = 100
    dt = (LungSliceTube.OutTime[i_slice,g] - LungSliceTube.InTime[i_slice,g])/N_steps
    time = LungSliceTube.InTime[i_slice,g]
    kappa = kappa_interpolate(N_Q,
                              LungFlow,
                              time)
    
    sum_RL = 0.5 * Tube_RL(RL0,
                           k_expansion_frac,
                           kappa)
    
    for i in range(1,N_steps):
        time = time + dt
        kappa = kappa_interpolate(N_Q,
                                  LungFlow,
                                  time)
        sum_RL = sum_RL + Tube_RL(RL0,
                                  k_expansion_frac,
                                  kappa)
    time = time + dt 
    kappa = kappa_interpolate(N_Q,
                              LungFlow,
                              LungSliceTube.OutTime[i_slice,g])
    sum_RL = sum_RL + 0.5*Tube_RL(RL0,
                                  k_expansion_frac,
                                  kappa)
    Ave_Tube_1D = sum_RL * dt / (LungSliceTube.OutTime[i_slice,g] - LungSliceTube.InTime[i_slice,g])
    return Ave_Tube_1D

@jit(cache=True, nopython=True)
def Calc_Reynold(Velocity,
                 Radius,
                 AirProp: AirProp_Class):
    """Calculated Reynolds number"""
    Reynold = AirProp.Density_37C / AirProp.Dynamic_Visc_37C * 2 * Velocity * Radius
    return Reynold

