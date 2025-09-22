import numpy as np
from numba import int32, int64, float64, void, types, typed, deferred_type    # import the types
from numba.experimental import jitclass

from ...config.constants import * 

LungSlice_Class_spec = [
    
    ('Inh', int64[:]),
    ('BH', int64[:]),
    ('Exh', int64[:]),
]  
@jitclass(LungSlice_Class_spec)
class LungSlice_Class(object):
    def __init__(self):
        #self.Inh=np.array([None]*N_slices_max)
        #self.BH=np.array([None]*N_slices_max)
        #self.Exh=np.array([None]*N_slices_max)
        
        self.Inh=np.array([0]*N_slices_max)
        self.BH=np.array([0]*N_slices_max)
        self.Exh=np.array([0]*N_slices_max)

        #self.Inh=np.zeros(N_slices_max, dtype= int)
        #self.BH=np.zeros(N_slices_max, dtype= int)
        #self.Exh=np.zeros(N_slices_max, dtype= int)



LungSliceReal8_Class_spec = [
    
    ('Inh', float64[:,:]),
    ('BH', float64[:,:]),
    ('Exh', float64[:,:]),
]  
@jitclass(LungSliceReal8_Class_spec)
class LungSliceReal8_Class(object):
    def __init__(self):
        self.Inh=np.zeros((N_slices_max,N_bins_max))
        self.BH=np.zeros((N_slices_max,N_bins_max))
        self.Exh=np.zeros((N_slices_max,N_bins_max))


LungSliceReal8B_Class_spec = [
    
    ('Inh', float64[:]),
    ('BH', float64[:]),
    ('Exh', float64[:]),
]  
@jitclass(LungSliceReal8B_Class_spec)
class LungSliceReal8B_Class(object):
    def __init__(self):
        #self.Inh=np.array([None]*N_slices_max)
        #self.BH=np.array([None]*N_slices_max)
        #self.Exh=np.array([None]*N_slices_max)

        #self.Inh=np.array([0.0]*N_slices_max)
        #self.BH=np.array([0.0]*N_slices_max)
        #self.Exh=np.array([0.0]*N_slices_max)

        self.Inh=np.zeros(N_slices_max)
        self.BH=np.zeros(N_slices_max)
        self.Exh=np.zeros(N_slices_max)

    def sliceVolume_const(self, V):
        #self.Inh = np.array([V]*N_slices_max)
        #self.BH = np.array([V]*N_slices_max)
        #self.Exh = np.array([V]*N_slices_max)

        self.Inh = np.ones(N_slices_max) * V
        self.BH = np.ones(N_slices_max) * V
        self.Exh = np.ones(N_slices_max) * V

#class LungSlice_Class():
#    def __init__(self) -> int:
#        self.Inh=np.array([None]*N_slices_max)
#        self.BH=np.array([None]*N_slices_max)
#        self.Exh=np.array([None]*N_slices_max)


LungSliceB_Class_spec = [
    
    ('Inh', int32),
    ('BH', int32),
    ('Exh', int32),
]  
@jitclass(LungSliceB_Class_spec)
class LungSliceB_Class(object):
    def __init__(self):
        self.Inh= 0
        self.BH= 0
        self.Exh= 0
    
    def set_N_slices(self, Inhaled_Volume, V_slice):
        n = int(Inhaled_Volume/V_slice.Inh[0])+1
        self.Inh = n
        self.BH = n
        self.Exh = n


LungSliceTube_C_Class_spec = [
    ('Ave_Tube_Radius', float64[:,:]),          
    ('Ave_Tube_Length', float64[:,:]),     
    ('TubeLength_Fraction', float64[:,:]),          
    ('Ave_Velocity', float64[:,:]), 
    ('Ave_Reynold', float64[:,:]),          
    ('ResidenceTime', float64[:,:]),     
    ('Ave_Q', float64[:,:]),          
    ('InTime', float64[:,:]),    
    ('OutTime', float64[:,:]),   
]

@jitclass(LungSliceTube_C_Class_spec)
class LungSliceTube_C_Class(object):
    def __init__(self):
        self.Ave_Tube_Radius=np.zeros((N_slices_max,N_GIcomps_LungGens_max))
        self.Ave_Tube_Length=np.zeros((N_slices_max,N_GIcomps_LungGens_max))
        self.TubeLength_Fraction=np.zeros((N_slices_max,N_GIcomps_LungGens_max))
        self.Ave_Velocity=np.zeros((N_slices_max,N_GIcomps_LungGens_max))
        self.Ave_Reynold=np.zeros((N_slices_max,N_GIcomps_LungGens_max))
        self.ResidenceTime=np.zeros((N_slices_max,N_GIcomps_LungGens_max))
        self.Ave_Q=np.zeros((N_slices_max,N_GIcomps_LungGens_max))

        self.InTime=np.zeros((N_slices_max,N_GIcomps_LungGens_max))
        self.OutTime=np.zeros((N_slices_max,N_GIcomps_LungGens_max))



LungSliceTube_C_Class_type = deferred_type()
LungSliceTube_C_Class_type.define(LungSliceTube_C_Class.class_type.instance_type)

LungSliceTube_Class_spec = [
    
    ('Inh', LungSliceTube_C_Class_type),
    ('BH', LungSliceTube_C_Class_type),
    ('Exh', LungSliceTube_C_Class_type),
]        
@jitclass(LungSliceTube_Class_spec)
class LungSliceTube_Class(object):
    def __init__(self):
        self.Inh=LungSliceTube_C_Class()
        self.BH=LungSliceTube_C_Class()
        self.Exh=LungSliceTube_C_Class()


LungSliceTube_Class_type = deferred_type()
LungSliceTube_Class_type.define(LungSliceTube_Class.class_type.instance_type)
#class LungSliceTube_Class(LungSliceTube_C_Class):
#    def __init__(self):
#        self.Inh=LungSliceTube_C_Class()
#        self.BH=LungSliceTube_C_Class()
#        self.Exh=LungSliceTube_C_Class()


LungFlow_B_Class_spec = [
    ('Q', float64[:]),          
    ('t', float64[:]),     
    ('kappa', float64[:]),          
 
]

@jitclass(LungFlow_B_Class_spec)
class LungFlow_B_Class(object):
    def __init__(self, N_Q_max_len: int):
        #self.Q = np.array([0.0000]*N_Q_max_len)
        #self.t = np.array([0.0000]*N_Q_max_len)
        #self.kappa = np.array([0.0000]*N_Q_max_len)

        self.Q = np.zeros(N_Q_max_len)
        self.t = np.zeros(N_Q_max_len)
        self.kappa = np.zeros(N_Q_max_len)
    
    
    def set_t(self,N_Q, Inhaled_Volume, flowfile, BHT, Flow_Exh, t_inhalation, manouvre):
        self.t =   flowfile
    
    def set_t_bk(self,N_Q, Inhaled_Volume, flowfile, BHT, Flow_Exh, t_inhalation, manouvre):
        if manouvre ==1 :
            t_tot = np.max(flowfile) #Inhaled_Volume / const_Flow
            #dt=t_tot / (N_Q-1)
            
            #for i in range(0,N_Q):
                #self.t[i] = dt*i
            self.t =   flowfile

        elif manouvre == 3:
            #const_Flow = 300e-6
            t_tot = Inhaled_Volume / Flow_Exh #const_Flow
            dt = t_tot / (N_Q)
            t_prev = BHT + np.max(flowfile)
            ex_time = np.zeros(N_Q + 1)
            for i in range(0,N_Q+1):
                ex_time[i] = dt*i
            
                #self.t[i] = dt*i
            self.t = ex_time
    

    def set_kappa(self, N_Q, Lung_size, Inhaled_volume, manouvre):
        if manouvre == 1:
            self.kappa[0] = 0.0
            self.kappa[N_Q-1] = Inhaled_volume/Lung_size
        elif manouvre == 3:
            self.kappa[0] = Inhaled_volume/Lung_size
            self.kappa[N_Q-1] = 0.0
        
        k = (self.kappa[0]-self.kappa[N_Q-1]) / (self.t[0] - self.t[N_Q-1])

        for i in range(1,N_Q-1):
            self.kappa[i] = self.kappa[0] + k * (self.t[i]-self.t[0])



LungFlow_B_Class_type = deferred_type()
LungFlow_B_Class_type.define(LungFlow_B_Class.class_type.instance_type)

LungFlow_Class_spec = [
    
    ('Inh', LungFlow_B_Class_type),
    
    ('Exh', LungFlow_B_Class_type),
]        
@jitclass(LungFlow_Class_spec)
class LungFlow_Class(object):
    def __init__(self,N_Q_max_len:int):
        self.Inh = LungFlow_B_Class(N_Q_max_len)
        self.Exh = LungFlow_B_Class(N_Q_max_len)

#class LungFlow_Class(LungFlow_B_Class):
#    def __init__(self):
#        self.Inh = LungFlow_B_Class()
#        self.Exh = LungFlow_B_Class()


LungDepo_Class_spec = [
    ('ET_Model', int32),          
    ('BHT', float64),     
    ('iFLAG_KingsCollege', int32),          
    ('iFLAG_DoSedDiff', int32), 
    ('iFLAG_DoSedDiff_From_VB', int32),          
    ('entrance_generation', int32), 
    ('N_Breaths', int32), 
    ('ET_manual_deposition', float64),          
    ('CoarseFraction', float64), 
    ('T_manual_deposition', float64),          
    ('T_model', int32), 
    ('FLAG_Thoracic_manual_deposition', int32), 
    ('Fnf', int32),          
    ('Thoracic_manual_Generation_To', int32), 
]

@jitclass(LungDepo_Class_spec)

class LungDepo_Class(object):
    def __init__(self):
        self.ET_Model = 3 # None
        self.BHT = 0.0 #None #Breathhold time
        self.iFLAG_KingsCollege = 0 #None
        self.iFLAG_DoSedDiff = 1 #None
        self.iFLAG_DoSedDiff_From_VB = 1# None
        self.entrance_generation = 0
        self.N_Breaths = 1
        self.ET_manual_deposition = 0.0 #None
        self.CoarseFraction = 0.0 #None
        self.T_manual_deposition = 0.0 #None
        self.T_model = 1 #1 = NCRP96. Other models not implemented
        self.FLAG_Thoracic_manual_deposition = 0 #None
        self.Fnf = 0
        self.Thoracic_manual_Generation_To = 0



LungTube_Class_spec = [
  
    ('k_expansion_frac', float64[:]),          
    #('multi', int32[:]),     
    ('multi', float64[:]),  # just use float64 for simplicity
    ('V_alveoli', float64[:]),          
    ('loc_V_air', float64[:]), 
           
    ('Radius', float64[:]),     
    ('Length', float64[:]),          
    ('V_air', float64[:]),    
    ('Angle_preceding', float64[:]),   
    ('Angle_gravity', float64[:]), 
]

@jitclass(LungTube_Class_spec)
class LungTube_Class(object):#Weibel lung definition, 3300ml
   
    def __init__(self, k_expansion_frac, multi,  V_alveoli, Radius, Length, V_air, Angle_preceding , Angle_gravity ):
        
        
        self.k_expansion_frac = k_expansion_frac# physiology[:,11] #np.insert(physiology[1:,11], 0, 0.0, axis = 0) / 100 # np.array([0, 0.0416, 0.0417, 0.0147, 0.0416, 0.0417, 0.0147, 0.0416, 0.0417, 0.0147, 0.0416, 0.0417, 0.0147, 0.0416, 0.0417, 0.0147, 0.0416, 0.0417, 0.0147, 0.0416, 0.0417, 0.0147, 0.0416, 0.0417, 0.0147])
        self.multi = multi # physiology[:,2]
        self.V_alveoli = V_alveoli # physiology[:,4]  #np.array([50, 20.997, 7.6513, 2.8295, 1.0309, 2.2254, 2.2618, 2.4291, 2.7900, 3.0588, 3.5613, 4.2843, 5.0966, 6.5482, 8.0119, 11.069, 15.183, 20.767, 33.067, 56.425, 107.33, 247.29, 453.37, 846.14, 1436.6]) #mL
        self.loc_V_air = np.zeros(N_GIcomps_LungGens_max)
        
        self.Radius = Radius # physiology[:,13]  # np.array([0, 1.5887, 1.0767, 0.73295, 0.49448, 0.39745, 0.30867, 0.24673, 0.20337, 0.16414, 0.13627, 0.11459, 9.6006e-02, 8.3618e-02, 7.2263e-02, 6.5037e-02, 5.7810e-02, 5.2649e-02, 4.7487e-02, 4.4390e-02, 4.1293e-02, 3.9228e-02, 3.8196e-02, 3.6131e-02, 3.6131e-02])  
        self.Length = Length # physiology[:,5]  #np.array([0, 10.592, 4.2016, 1.6765, 0.67101, 1.1211 ,0.94458, 0.79386, 0.67101, 0.56468, 0.47693, 0.40570, 0.34376, 0.29112, 0.23847, 0.20337, 0.17653, 0.14556, 0.12491, 0.10323, 8.7748e-02, 7.3295e-02, 6.1940e-02, 5.1616e-02, 4.4390e-02])
        
        self.V_air = V_air # physiology[:,14] # (self.Radius ** 2) * np.pi * (self.Length) * self.multi + self.V_alveoli # np.array([[50e-6] + [i for i in V_air_tmp[1:]]])   #np.insert(V_air_tmp[1:], 0, 50e-6, axis = 0)
        self.Angle_preceding = Angle_preceding # physiology[:, 10] # np.insert(physiology[:, 10], 0, np.nan, axis = 0)  # np.array([None, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45])
        self.Angle_gravity = Angle_gravity #physiology[:, 9] # np.insert(physiology[:, 9], 0, np.nan, axis = 0)  #np.array([None, 90, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45])
    


    def set_k_expansion_frac(self, k, start_gen, end_gen):
        self.k_expansion_frac[start_gen:(end_gen+1)] = k

    def convert_k_expansion_frac(self):
        sum1 = sum(self.V_air)
        sum2 = sum(np.multiply(self.V_air,self.k_expansion_frac))
        self.k_expansion_frac = self.k_expansion_frac * sum1/sum2

    def Region(self, g):
        if g == 0:
            return 'ET'
        elif g > 0 and g <= 9:
            return 'BB'
        elif g > 9 and g <= 16:
            return 'bb'
        elif g > 16 and g <= 24:
            return 'Al'

LungTube_Class_type = deferred_type()
LungTube_Class_type.define(LungTube_Class.class_type.instance_type)

Body_Class_spec = [
    
    ('Cunningham_BodyTemperature_K', float64),
    
    ('kBT', float64),
    ('T', float64)
]        
@jitclass(Body_Class_spec)
class Body_Class(object):
    def __init__(self):
        self.Cunningham_BodyTemperature_K = 310.34 #K
        self.kBT = 1.38066e-23 * 300 # None  # place holder
        self.T = 310.15 #K








