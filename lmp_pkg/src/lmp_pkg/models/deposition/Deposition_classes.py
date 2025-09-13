import numpy as np
from numba import int32, int64, float64, void, types, typed, deferred_type    # import the types
from numba.experimental import jitclass

from ...config.constants import * 

AmountDepoSingleMatrix_Class_spec = [
    ('Tot', float64[:, :]),
    ('Imp', float64[:, :]),
    ('Diff', float64[:, :]),
    ('Sed', float64[:, :]),
    ('Entrance', float64[:, :]),
]  
@jitclass(AmountDepoSingleMatrix_Class_spec)
class AmountDepoSingleMatrix_Class(object):
    def __init__(self) -> float:
        self.Tot=np.zeros((N_GIcomps_LungGens_max,N_bins_max))
        self.Imp=np.zeros((N_GIcomps_LungGens_max,N_bins_max))
        self.Diff=np.zeros((N_GIcomps_LungGens_max,N_bins_max))
        self.Sed=np.zeros((N_GIcomps_LungGens_max,N_bins_max))
        self.Entrance=np.zeros((N_GIcomps_LungGens_max,N_bins_max))


AmountDepoSingleMatrix_Class_type = deferred_type()
AmountDepoSingleMatrix_Class_type.define(AmountDepoSingleMatrix_Class.class_type.instance_type)

AmountDepoMatrix_Class_spec = [
    

    ('Inh', AmountDepoSingleMatrix_Class_type),
    ('BH', AmountDepoSingleMatrix_Class_type),
    ('Exh', AmountDepoSingleMatrix_Class_type),
]        
@jitclass(AmountDepoMatrix_Class_spec)
class AmountDepoMatrix_Class(object):
    def __init__(self):
        self.Inh = AmountDepoSingleMatrix_Class()
        self.BH = AmountDepoSingleMatrix_Class()
        self.Exh = AmountDepoSingleMatrix_Class()




r_Single_Class_spec = [
    
    ('Tot', float64[:,:]),
    ('Imp', float64[:,:]),
    ('Diff', float64[:,:]),
    ('Sed', float64[:,:]),
]  
@jitclass(r_Single_Class_spec)
class r_Single_Class(object):
    def __init__(self):
        self.Tot=np.zeros((N_GIcomps_LungGens_max+1,N_bins_max))
        self.Imp=np.zeros((N_GIcomps_LungGens_max+1,N_bins_max))
        self.Diff=np.zeros((N_GIcomps_LungGens_max+1,N_bins_max))
        self.Sed=np.zeros((N_GIcomps_LungGens_max+1,N_bins_max))

r_Single_Class_type = deferred_type()
r_Single_Class_type.define(r_Single_Class.class_type.instance_type)

r_Class_spec = [
    
    ('TOT', r_Single_Class_type),
    ('Inh', r_Single_Class_type),
    ('BH', r_Single_Class_type),
    ('Exh', r_Single_Class_type),
]        
@jitclass(r_Class_spec)
class r_Class(object):
    def __init__(self):
        self.TOT=r_Single_Class()
        self.Inh=r_Single_Class()
        self.BH=r_Single_Class()
        self.Exh=r_Single_Class()


AmountDepoSingle_Class_spec = [
    
    ('Tot', float64[:]),
    ('Imp', float64[:]),
    ('Diff', float64[:]),
    ('Sed', float64[:]),
    ('Entrance', float64[:]),
]  
@jitclass(AmountDepoSingle_Class_spec)
class AmountDepoSingle_Class(object):
    def __init__(self):
        self.Tot=np.zeros(N_GIcomps_LungGens_max)
        self.Imp=np.zeros(N_GIcomps_LungGens_max)
        self.Diff=np.zeros(N_GIcomps_LungGens_max)
        self.Sed=np.zeros(N_GIcomps_LungGens_max)
        self.Entrance=np.zeros(N_GIcomps_LungGens_max)
        #self.Tot=np.array([0.00]*N_GIcomps_LungGens_max)
        #self.Imp=np.array([0.00]*N_GIcomps_LungGens_max)
        #self.Diff=np.array([0.00]*N_GIcomps_LungGens_max)
        #self.Sed=np.array([0.00]*N_GIcomps_LungGens_max)
        #self.Entrance=np.array([0.00]*N_GIcomps_LungGens_max)


AmountDepoSingle_Class_type = deferred_type()
AmountDepoSingle_Class_type.define(AmountDepoSingle_Class.class_type.instance_type)

AmountDepo_Class_spec = [
    ('Inh', AmountDepoSingle_Class_type),
    ('BH', AmountDepoSingle_Class_type),
    ('Exh', AmountDepoSingle_Class_type),
]        
@jitclass(AmountDepo_Class_spec)
class AmountDepo_Class(object):
    def __init__(self):
        self.Inh=AmountDepoSingle_Class()
        self.BH=AmountDepoSingle_Class()
        self.Exh=AmountDepoSingle_Class()


ProbSliceSingle_Class_spec = [
    ('Imp', float64[:,:]),
    ('Diff', float64[:,:]),
    ('Sed', float64[:,:]),
]  
@jitclass(ProbSliceSingle_Class_spec)
class ProbSliceSingle_Class(object):
    def __init__(self):
        self.Imp=np.zeros((N_slices_max,N_GIcomps_LungGens_max))
        self.Diff=np.zeros((N_slices_max,N_GIcomps_LungGens_max))
        self.Sed=np.zeros((N_slices_max,N_GIcomps_LungGens_max))

ProbSliceSingle_Class_type = deferred_type()
ProbSliceSingle_Class_type.define(ProbSliceSingle_Class.class_type.instance_type)

ProbSlice_Class_spec = [
    ('Inh', ProbSliceSingle_Class_type),
    ('BH', ProbSliceSingle_Class_type),
    ('Exh', ProbSliceSingle_Class_type),
]        
@jitclass(ProbSlice_Class_spec)
class ProbSlice_Class(object):
    def __init__(self):
        self.Inh=ProbSliceSingle_Class()
        self.BH=ProbSliceSingle_Class()
        self.Exh=ProbSliceSingle_Class()


SizeDistr_Class_spec = [
    ('Rmax', float64[:,:]),
    ('Rmin', float64[:,:]),
    ('k', float64[:,:]),
    ('X_bin', float64[:,:]),
]  

@jitclass(SizeDistr_Class_spec)
class SizeDistr_Class(object):
    def __init__(self):
        self.Rmax = np.zeros((N_GIcomps_LungGens_max, N_bins_max), dtype = float64)
        self.Rmin = np.zeros((N_GIcomps_LungGens_max, N_bins_max), dtype = float64)
        self.k = np.zeros((N_GIcomps_LungGens_max, N_bins_max), dtype = float64)
        self.X_bin = np.zeros((N_GIcomps_LungGens_max, N_bins_max), dtype = float64)


SizeDistr_in_Class_spec = [
    
    ('Rmax', float64),
    ('Rmin', float64),

    ('X_bin', float64),
]  
@jitclass(SizeDistr_in_Class_spec)
class SizeDistr_in_Class(object):
    def __init__(self):
        self.Rmax = 2e-6 #None
        self.Rmin = 2e-6 #None
        self.X_bin = 0.0 #None
