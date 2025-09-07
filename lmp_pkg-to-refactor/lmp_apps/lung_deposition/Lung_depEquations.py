import numpy as np
from numba import int32, int64, float64, void, types, typed, deferred_type, jit, njit    # import the types

from .Lung_classes import *
from .PhysicalProp_classes import *
from .Deposition_classes import *

@jit(cache=True, nopython=True)
def Deposition_parameters(j: int,
                          AeroProp: AeroProp_Class,
                          AirProp: AirProp_Class,
                          UndissAero_AD: SizeDistr_Class,
                          UndissAero_GD: SizeDistr_Class,
                          Body: Body_Class,
                          PhCo: PhCo_Class):
    AeroProp.D_Geom = UndissAero_GD.Rmax[0,j] + UndissAero_GD.Rmin[0,j]
    AeroProp.D_Aero = UndissAero_AD.Rmax[0,j] + UndissAero_AD.Rmin[0,j]

    Body.kBT = PhCo.Boltzmann * Body.T

    #AeroProp.DiffCoeff = Body.kBT * Cun(AeroProp.D_Geom, Body.Cunningham_BodyTemperature_K) / (3 * np.pi * AirProp.Dynamic_Visc_37C * AeroProp.D_Geom)
    cun_val = Cun(AeroProp.D_Geom, Body.Cunningham_BodyTemperature_K)
    AeroProp.DiffCoeff = Body.kBT * cun_val / (3 * np.pi * AirProp.Dynamic_Visc_37C * AeroProp.D_Geom)
    cun_val_next  = Cun(AeroProp.D_Aero, Body.Cunningham_BodyTemperature_K)
    AeroProp.v_settling = AeroProp.Density_aeroD_Particle * PhCo.grav_const * AeroProp.D_Aero**2 * cun_val_next / (18.0 * AirProp.Dynamic_Visc_37C)
    #AeroProp.v_settling = AeroProp.Density_aeroD_Particle * PhCo.grav_const * AeroProp.D_Aero**2 * Cun(AeroProp.D_Aero, Body.Cunningham_BodyTemperature_K) / (18 * AirProp.Dynamic_Visc_37C)
@jit(cache=True, nopython=True)
def Cun(particle_diameter: float, Temperature: float):
    """Cunningham correction, according to Allan Raabe"""

    Lambda = 69.8267e-9 # @ 310.2K    
    tLambda = Lambda * Temperature / 310.2
    Cun_val = 1.0 + tLambda / particle_diameter * (2.43 + 1.05 * np.exp(-0.39 * particle_diameter / tLambda))
    return float64(Cun_val)

@jit(cache=True, nopython=True)
def ETdep_NCRP96(DiffCoeff,
                 dae,
                 Qflow,
                 Manoeuvre:int,
                 Depo: LungDepo_Class):
    
    #Nasal part
    if Manoeuvre == 1:
        #Inh
        a = 4600
        b = -18.2
    elif Manoeuvre == 2:
        #Exh
        a = 2300
        b = -21.3
    
    Nae = 1 / (1 + ((dae*1e6)**2 * (Qflow*1e6) / a)**(-0.94))
    Nth = 1 - np.exp(b * ((DiffCoeff*1e4) * (Qflow*1e6) ** (-0.25))**0.5)

    #oral part
    if Manoeuvre == 1:
        #inh
        a = 30000
        b = -14.6

        Oae = 1 / (1 + ((dae*1e6)**2 * (Qflow*1e6) / a) ** (-1.37))
    elif Manoeuvre == 2:
        b = -12.1
        Oae = 0
    
    Oth = 1 - np.exp(b * ((DiffCoeff*1e4) * (Qflow*1e6)**(-0.25))**0.5)

    ETdep = Depo.Fnf * np.sqrt(Nae**2 + Nth**2) + (1 - Depo.Fnf) * np.sqrt(Oae**2 + Oth**2)

    ETdep = p_between_0_1(ETdep)
    return float64(ETdep)

@jit(cache=True, nopython=True)
def ETdep_Manual_Deposition(ET_manual_deposition: float,
                            Manouvre: int):
    """Manual deposition to ET. Note: Same deposition probability regardless of size"""
    if Manouvre == 1:
        ETdep_Manual_Deposition_val = ET_manual_deposition
    elif Manouvre == 2:
        ETdep_Manual_Deposition_val = 0

    ETdep_Manual_Deposition_val = p_between_0_1(ETdep_Manual_Deposition_val)
    return ETdep_Manual_Deposition_val

@jit(cache=True, nopython=True)
def Impaction_Yeh80_NCRP96(g,
                           D_Aero,
                           Density_aeroD_particle,
                           AirProp: AirProp_Class,
                           LungTube: LungTube_Class,
                           Radius,
                           Velocity):
    Theta = LungTube.Angle_preceding[g] * np.pi/180
    
    Stoke = Density_aeroD_particle * D_Aero**2 * Velocity / (18.0*AirProp.Dynamic_Visc_37C * (2*Radius)) #Stokes number (dimensionless)
    if Stoke * Theta < 1:
        #Impaction_Yeh80_NCRP96 = max(0, 1 - 2 / np.pi * np.acos(Stoke * Theta) + 1/np.pi * np.sin(2 * np.acos(Stoke * Theta)))
        Impaction_Yeh80_NCRP96_val = max(0, 1 - 2 / np.pi * np.arccos(Stoke * Theta) + 1/np.pi * np.sin(2 * np.arccos(Stoke * Theta)))
    else:
        Impaction_Yeh80_NCRP96_val = 1.0

    Impaction_Yeh80_NCRP96_val = p_between_0_1(Impaction_Yeh80_NCRP96_val)

    return Impaction_Yeh80_NCRP96_val

@jit(cache=True, nopython=True)
def Diffusion_NCRP96(DiffCoeff,
                     ResidenceTime,
                     Radius,
                     Velocity,
                     Reynold,
                     Angle_preceding):
    Diffusion_NCRP96_val = Diffusion_Yeh80_sub(DiffCoeff,
                                           ResidenceTime,
                                           Radius,
                                           Reynold)
    Theta = Angle_preceding * np.pi/180

    c1 = (2.0 * Theta / np.pi) * (13.0 - 12.0 * Theta/np.pi)

    if Velocity * ResidenceTime / Radius > 10:
        c2 = 1.0 +c1 * (2 * Radius / (Velocity * ResidenceTime))
    else:
        c2 = 1.0 + c1 * 0.2

    Diffusion_NCRP96_val = 1.0 - (1 - Diffusion_NCRP96_val)**c2
    Diffusion_NCRP96_val = p_between_0_1(Diffusion_NCRP96_val)

    return Diffusion_NCRP96_val

@jit(cache=True, nopython=True)
def Diffusion_Yeh80_sub(DiffCoeff,
                        ResidenceTime,
                        Tube_Radius,
                        Reynold):
    h = DiffCoeff * ResidenceTime / (2 * Tube_Radius**2)

    if Reynold < 2300:
        Diffusion = 1.0 - 0.819 * np.exp(-7.315 * h) - 0.0976 * np.exp(-44.61 * h) - 0.0325 * np.exp(-114 * h) - 0.0509 * np.exp(-79.31 * h**0.66666666)
    else:
        if h > 0.0164:
            Diffusion = 1.0
        else:
            Diffusion = 4.0 * np.sqrt(h/2) * (1 - 0.444 * np.sqrt(h/2))
    
    return Diffusion

@jit(cache=True, nopython=True)
def Sedimentation_Yeh80_NCRP96(v_settling,
                               ResidenceTime,
                               Radius,
                               Angle_gravity):
    Phi = Angle_gravity * np.pi/180

    e1 = 3.0 * v_settling * ResidenceTime * np.cos(Phi) / (8 * Radius) #Dimensionless
    Sedimentation = 1.0 - np.exp(-1.69765 * e1)
    Sedimentation = p_between_0_1(Sedimentation)
    return Sedimentation

@jit(cache=True, nopython=True)
def BHDiffusion(DiffCoeff,
                ResidenceTime,
                Tube_Radius):
    h = DiffCoeff * ResidenceTime / (2 * Tube_Radius**2)

    Diffusion = 1.0  - np.exp(-5.784 * 2 * h)
    Diffusion = p_between_0_1(Diffusion)
    
    return Diffusion

@jit(cache=True, nopython=True)
def p_between_0_1(p: float):
    p = max(1e-12, p)
    p = min(1, p)
    return p

if __name__ == '__main__':
    print(ETdep_NCRP96(9.34268e-11,
                       4.14987e-7,
                       500e-6,
                       1))