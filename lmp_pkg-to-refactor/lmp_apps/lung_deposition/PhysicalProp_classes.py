from numba import int32, int64, float64, void, types, typed, deferred_type    # import the types
from numba.experimental import jitclass

AirProp_Class_spec = [
    
    ('Dynamic_Visc_37C', float64),
    ('Kinematic_Visc_37C', float64),
    ('Density_37C', float64),
]  
@jitclass(AirProp_Class_spec)
class AirProp_Class(object):
    """Physical properties of air at 37C"""
    def __init__(self):
        self.Dynamic_Visc_37C=0.0189e-3 #Pa s
        self.Kinematic_Visc_37C=16.64e-6 #m^2/s
        self.Density_37C=1.14 #kg/m^3

H2OProp_Class_spec = [
    
    ('Dynamic_Visc', float64),
    ('Density', float64),
]  
@jitclass(H2OProp_Class_spec)
class H2OProp_Class(object):
    """Physical properties of water at 37C"""
    def __init__(self):
        self.Dynamic_Visc=0.69e-3 #Pa s
        #self.Density=993.36 #kg/m^3
        self.Density=1000.0 #kg/m^3


AeroProp_Class_spec = [
    
    ('DiffCoeff', float64),
    ('D_Aero', float64),
    ('D_Geom', float64),
    ('Density_aeroD_Particle', float64),
    ('v_settling', float64),
]  
@jitclass(AeroProp_Class_spec)
class AeroProp_Class(object):
    "Areosol properties. Dependent on product platform"
    def __init__(self):
        self.DiffCoeff= 1e-6 # placer holder
        self.D_Aero= 3e-6 # None # placer holder #diam
        self.D_Geom=  3e-6 #None #diam
        self.Density_aeroD_Particle= 1.2e3 #kg/m^3
        self.v_settling= 0.0 #  None


PhCo_Class_spec = [
    
    ('gas_constant', float64),
    ('grav_const', float64),
    ('Avogadro', float64),
    ('Boltzmann', float64),

]  
@jitclass(PhCo_Class_spec)

class PhCo_Class(object):
    """Physical constants"""
    def __init__(self):
        self.gas_constant=8.31441 #J/mol/K
        self.grav_const=9.80665 #N/kg
        self.Avogadro=6.02205e23 #1/mol
        self.Boltzmann=1.38066e-23 #J/K

