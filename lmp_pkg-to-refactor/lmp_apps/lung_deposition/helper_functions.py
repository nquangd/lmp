import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import root_scalar
from numba import jit


@jit(cache = True, nopython=True)  # was without cache = True before
def parallel_scale_lung(ref_lung: float, v_lung:float, numgen = 25):
    Lung_size = v_lung / 1e6 #m^3

    sf = (Lung_size / 2999.60e-6 )**(1/3)

    # Columns to pass to lung class:
    # [(0, 'Generation'), (1, 'Region'), (2, 'Multiplicity'), (3, 'Total Airway Vol'), (4, 'Alveoli Vol'), (5, 'Length'), (6, 'Diameter'), (7, 'Diameter_2'), (8, 'Diameter_eff'), (9, 'Gravity Angle'), (10, 'Branching Angle'), (11, 'Expansion Fraction')]
    
    #ref_lung = pd.read_csv('./Ref_Lung_Physiology.csv')
    #ref_lung['Region'] = 0

    sub_lung = np.zeros((numgen, 8))
    #sub_lung = np.zeros((25,8))
    # sub lung order: 0: k_expansion_frac, 1: multi,  2: V_alveoli, 3: Radius, 4: Length, 5: V_air, 6: Angle_preceding , 7: Angle_gravity
    sub_lung[:,0] = ref_lung[:, 6] / 100
    sub_lung[:,1] = ref_lung[:, 0]
    sub_lung[:,2] = 1e-6 * ref_lung[:, 1] * (sf**3)
    sub_lung[:,3] = 1e-2 * 1/2 * ref_lung[:, 3] * sf 
    sub_lung[:,4] = 1e-2 * ref_lung[:, 2] * sf
    sub_lung[:,5] = (sub_lung[:,3] ** 2) * np.pi * (sub_lung[:,4]) * sub_lung[:,1] + sub_lung[:,2]
    sub_lung[0,5] = 50e-6
    sub_lung[:,6] = ref_lung[:, 5]
    sub_lung[:,7] = ref_lung[:, 4]

    return sub_lung

def scale_lung(ref_lung: float, v_lung:float, numgen = 25):
    Lung_size = v_lung / 1e6 #m^3

    sf = (Lung_size / 2999.60e-6 )**(1/3)

    # Columns to pass to lung class:
    # [(0, 'Generation'), (1, 'Region'), (2, 'Multiplicity'), (3, 'Total Airway Vol'), (4, 'Alveoli Vol'), (5, 'Length'), (6, 'Diameter'), (7, 'Diameter_2'), (8, 'Diameter_eff'), (9, 'Gravity Angle'), (10, 'Branching Angle'), (11, 'Expansion Fraction')]
    
    #ref_lung = pd.read_csv('./Ref_Lung_Physiology.csv')
    #ref_lung['Region'] = 0

    sub_lung = np.zeros((numgen, 8))
    #sub_lung = np.zeros((25,8))
    # sub lung order: 0: k_expansion_frac, 1: multi,  2: V_alveoli, 3: Radius, 4: Length, 5: V_air, 6: Angle_preceding , 7: Angle_gravity
    sub_lung[:,0] = ref_lung[:, 6] / 100
    sub_lung[:,1] = ref_lung[:, 0]
    sub_lung[:,2] = 1e-6 * ref_lung[:, 1] * (sf**3)
    sub_lung[:,3] = 1e-2 * 1/2 * ref_lung[:, 3] * sf 
    sub_lung[:,4] = 1e-2 * ref_lung[:, 2] * sf
    sub_lung[:,5] = (sub_lung[:,3] ** 2) * np.pi * (sub_lung[:,4]) * sub_lung[:,1] + sub_lung[:,2]
    sub_lung[0,5] = 50e-6
    sub_lung[:,6] = ref_lung[:, 5]
    sub_lung[:,7] = ref_lung[:, 4]

    return sub_lung


def cun(d):
    Temperature = 310.2 #K
    Lambda = 69.8267e-9 # @ 310.2K    
    tLambda = Lambda * Temperature / 310.2
    Cun_val = 1.0 + tLambda / d * (2.43 + 1.05 * np.exp(-0.39 * d / tLambda))
    #return 1 + 2 * 66e-9 / d * (1.17 + 0.525*np.exp(-0.39*d/66e-9))
    return Cun_val
def apsd_and_gsd(dg, ar:list):
                 
    da, rho_p, rho_zero, shape = ar[0], ar[1], ar[2], ar[3]
    fun = da * np.sqrt(rho_zero) - dg * np.sqrt(rho_p/shape * cun(dg)/cun(da))
    return fun



def constructPSD(MMAD:float, GSD:float, bins:float, density = 1200):
    dg = []
    for i in range(0, len(bins)): 
        res = root_scalar(apsd_and_gsd, x0 = bins[i] * 1.01, bracket= [bins[i] * 0.5, bins[i] *1.5], args = ([bins[i] , density, 1000, 1]), method='brentq', maxiter = 100)
        dg.append(res.root)
    # Fit PSD with binsize taken from CFD bins, take the middle value between two bins as size, add begining and end size
    dist = stats.lognorm(s=np.log(GSD), scale=MMAD)

    massfraction = Mass_Fraction_From_PSD(dist, bins)
    
    
    return np.array(bins), np.array(dg), np.array(massfraction)





def Mass_Fraction_From_PSD(dist, bins):

    first = round(bins[0] * 2  - (bins[0]+bins[1])/2,3)
    last = round(bins[-1] * 2  - (bins[-1]+bins[-2])/2,3)
    binsize = [first] + [round((bins[i] + bins[i+1])/2,3) for i in range(len(bins)-1)] + [last]
    
    # Cummulative
    cdf0 = dist.cdf(binsize[:-1])
    cdf1 = dist.cdf(binsize[1:])
    
    # Calculate Mass Fraction
    massfraction = [a - b for a, b in zip(cdf1, cdf0)]
    massfraction = [massfraction[0]  + cdf0[0]] + massfraction[1:]
    
    # Normalise Mass Fraction
    massfraction = np.array(massfraction) / np.sum(massfraction)
    
    return massfraction

# To construct PSD from MMAD/GSD of NGI test

