from numba import njit, jit, int32, float64

from .Deposition_classes import *
from .Lung_classes import *
from .PhysicalProp_classes import *
from ...config.constants import * 


@jit(cache=True, nopython=True)
def Integral_flow(N_Q: int,
                  Q_flow: float,
                  t_flow: float,
                  t_max: float):
    """Numerical integration of flow integral"""
    
    
    dt = t_max/N_steps

    t_flow_temp=0
    Q_f = Q_interpolate(N_Q,
                           Q_flow,
                           t_flow,
                           t_flow_temp)
    sum = 0.5*Q_f

    for i in range(1,N_steps):
        t_flow_temp = t_flow_temp + dt
        Q_f = Q_interpolate(N_Q,
                           Q_flow,
                           t_flow,
                           t_flow_temp)
        sum = sum + Q_f

    t_flow_temp = t_flow_temp + dt
    Q_f = Q_interpolate(N_Q,
                           Q_flow,
                           t_flow,
                           t_max)
    sum = sum + 0.5*Q_f
    
    Integral_flow_val = sum * dt

    return Integral_flow_val

@jit(cache=True, nopython=True)
def Q_interpolate(N_Q: int,
                  Q_flow: float,
                  t_flow: float,
                  t_flow_temp: float):
    """Helper function fo numerical integration of flow integral"""
    if t_flow_temp == t_flow[0]:
        Q_interpolate_val = Q_flow[0]
        return Q_interpolate_val
    else:
        for i in range(1,N_Q):
            if t_flow_temp <= t_flow[i]:
                Q_interpolate_val = Q_flow[i-1] + (Q_flow[i] - Q_flow[i-1]) * (t_flow_temp - t_flow[i-1])/(t_flow[i]-t_flow[i-1])
                return Q_interpolate_val 

@jit(cache=True, nopython=True)            
def Assign_V_air_local(N_Generations: int,
                       LungTube: LungTube_Class,
                       Depo: LungDepo_Class):
    """Adjusts Lung geometry for manual deposition in upper regions"""
    
    if (Depo.iFLAG_KingsCollege == 1): #Vad betyder denna flagga????
        for g in range(0,Depo.entrance_generation):
            LungTube.loc_V_air[g] = 1e-100 
        for g in range(Depo.entrance_generation, N_Generations):
            LungTube.loc_V_air[g] = LungTube.V_air[g]
    else:
        for g in range(0, N_Generations):
            LungTube.loc_V_air[g] = LungTube.V_air[g]
@jit(cache=True, nopython=True)    
def Tube_RL(RL0, 
            k_expansion_frac, 
            kappa):
    
    Tube_RL_val = RL0 * (1+ k_expansion_frac * kappa)**(1/3)
    return Tube_RL_val
@jit(cache=True, nopython=True)
def kappa_interpolate(N_Q,
                      LungFlow: LungFlow_B_Class,
                      time) -> float:
    if time == LungFlow.t[0]:
        kappa_interpolate_val = LungFlow.kappa[0]
    else:
        for i in range(1,N_Q):
            if time <= LungFlow.t[i]:
                kappa_interpolate_val = LungFlow.kappa[i-1] + (LungFlow.kappa[i] - LungFlow.kappa[i-1]) * (time - LungFlow.t[i-1]) / (LungFlow.t[i] - LungFlow.t[i-1])
                
    return kappa_interpolate_val


