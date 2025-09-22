import os
from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from scipy import stats
from scipy.optimize import curve_fit
from scipy.stats import gstd



def ML_CFD_MT_deposition(
    mmad,
    gsd,
    propellant,
    usp_deposition,
    pifr,
    cast,
    DeviceType,
    *,
    data_root: Optional[Union[str, Path]] = None,
    database_filename: str = "DataBase_July_10.csv",
    model_root: Optional[Union[str, Path]] = None,
):
#def ML_CFD_MT_deposition(self):
    #mmad, gsd, propellant, usp_deposition, pifr, cast, DeviceType = product.api_data['mmad'], product.api_data['gsd'],product.api_data['propellant'], product.api_data['usp_depo_fraction'], subject.pifr_Lpm, subject.cast,product.api_data['device']  
    """Calculate mouth-throat deposition using the ML surrogate model.

    Parameters
    ----------
    data_root:
        Optional base directory containing the CFD surrogate assets. Defaults to
        the packaged ``data/cfd`` folder when ``None``.
    database_filename:
        Name (or absolute path) of the CSV database with particle distributions.
    model_root:
        Optional directory containing the fitted joblib models. Defaults to
        ``data_root`` when omitted.
    """
    # It's better to pass the path to the data files as an argument
    # For now, let's assume the data is in a 'data' subdirectory of the package
    pkg_dir = Path(__file__).resolve().parent
    default_root = pkg_dir / ".." / "data" / "cfd"
    data_root_path = Path(data_root) if data_root is not None else default_root

    database_path = Path(database_filename)
    if not database_path.is_absolute():
        database_path = data_root_path / database_filename

    if not database_path.exists():
        raise FileNotFoundError(f"CFD surrogate database not found at {database_path}")

    deposition = pd.read_csv(database_path)
    
    #deposition = pd.read_csv('C:/Users/krvt084/OneDrive - AZCollaboration/Emerging_Tech_Program/ETP_V2/LMP_Apps/CFD/DataBase.csv')

    # This correction is due to the use of wrong density in the CFD
    correction_factor = 1.0 # np.sqrt(1550/1000)
    deposition['Size'] = deposition['Size'] * 1e6

    deposition['Size_ORG'] = deposition['Size']
    deposition['Size'] = deposition['Size'] * correction_factor

    product_key = str(propellant or "").strip().upper()
    device_key = str(DeviceType or "").strip().lower()
    inlet_key = str(cast or "").strip().lower()

    if not product_key or not device_key or not inlet_key:
        raise ValueError("CFD surrogate requires propellant, device type, and inlet identifiers")

    prod_mask = (
        deposition['Product'].astype(str).str.upper() == product_key
    ) & (
        deposition['DeviceType'].astype(str).str.lower() == device_key
    )

    usp_mask = prod_mask & (deposition['Inlet'].astype(str).str.upper() == 'USP')
    df = deposition.loc[usp_mask]

    bins = np.unique(df['Size'].to_numpy())
    if bins.size == 0:
        raise ValueError(
            f"No CFD surrogate data for product={product_key}, device_type={device_key}, inlet=USP"
        )

    bins = np.sort(bins)
    prob = df['Mouth'].to_numpy(dtype=float)
    mf_out = constructPSD(mmad, gsd, bins)
    denom = 1 - prob
    if np.any(np.isclose(denom, 0.0)):
        raise ValueError("CFD surrogate encountered USP probabilities equal to 1.0, cannot rescale PSD")
    mf_in = mf_out / denom
    depo_fr = np.sum(mf_in * prob)
    coarse = (1-(1-usp_deposition/100)/(np.sum(mf_in)-depo_fr)) * 100

    # Reconstruct the inlet psd as log-normal
    psd_in = constructPSD_from_Mass_Fraction(bins, mf_in)

    # Move to CAST
    # Load the fitted for PIFR
    # This is to get the bins right, pifr = 30 here
    
    cast_mask = (
        prod_mask
        & (deposition['Inlet'].astype(str).str.lower() == inlet_key)
        & (deposition['Flowrate'] == 30)
    )
    bins_cast = np.unique(deposition.loc[cast_mask, 'Size'].to_numpy())
    if bins_cast.size == 0:
        raise ValueError(
            f"No CFD surrogate data for product={product_key}, device_type={device_key}, inlet={inlet_key}, flow=30"
        )

    bins_cast = np.sort(bins_cast)

    # Load the ML model
    model_filename = f'Fitted_CFD_{propellant}_{cast}_{DeviceType}.sav'
    model_root_path = Path(model_root) if model_root is not None else data_root_path
    model_path = model_root_path / model_filename

    if not model_path.exists():
        raise FileNotFoundError(f"ML model not found at {model_path}")

    model = joblib.load(model_path)
    prob_cast = model.predict(np.array([bins_cast, [1 / pifr] * len(bins_cast)]).T)

    # Re-calculate mf-in with bins for CAST, can either be the same or not the same as bins of USP
    mf_in = Mass_Fraction_From_PSD(psd_in, bins_cast)

    depo_fr_cast = np.sum(mf_in * prob_cast) * (100-coarse) + coarse  # in Percentage %
    mf_out_cast = mf_in * (1 - prob_cast)

    # try normalised
    mf_out_cast = mf_out_cast / np.sum(mf_out_cast)

    # calculate MMAD from here
    mmad_cast, gsd_cast = MMADfromCDF(bins_cast, mf_out_cast)
    
    #self.mtdepo = depo_fr_cast / 100
    #self.mmad = mmad_cast
    #self.gsd = gsd_cast
    return round(depo_fr_cast,3) / 100, round(mmad_cast,3), round(gsd_cast,3)


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

def constructPSD(MMAD, GSD, bins):
    # Fit PSD with binsize taken from CFD bins, take the middle value between two bins as size, add begining and end size
    dist = stats.lognorm(s=np.log(GSD), scale=MMAD)

    massfraction = Mass_Fraction_From_PSD(dist, bins)
    
    return massfraction


# Log-normal psd used to fit the mass fraction to    
def lognorm_fit(x, mu, sigma):
    y = (1/(sigma*x*np.sqrt(2*np.pi))) * np.exp(-1* ((np.log(x) - mu)**2/(2*sigma**2)))
    return y

# Fit the log-normal to mass fraction (normalised)
def constructPSD_from_Mass_Fraction(bins, mf):
    # Normalise again to make sure
    mf = mf/np.sum(mf)
    cdf = mf.cumsum()
    realx = [(a+b)/2 for a, b in zip(bins[1:], bins[0:-1])]
    realy = [(a - b) for a, b in zip(cdf[1:], cdf[0:-1])]

    x = realx + [0]
    y = realy
    y_bins = []
    for cnt, yi in enumerate(y):
        y_bins.append(yi / (x[cnt+1] - x[cnt]))

    (mu,sigma), _  = curve_fit(lognorm_fit, x[0:-1], y_bins, p0=[1, 0.5],maxfev=5000)
    dist = stats.lognorm(scale =np.exp(mu), s = sigma)
       
    return dist

# Calculated MMAD and GSD ex-cast  
def MMADfromCDF(bins, mf):

    dist = constructPSD_from_Mass_Fraction(bins, mf)
    samples = dist.rvs(size = 200000)
    samples = samples[samples > 0]
    
    return dist.ppf(0.5), gstd(samples)

# CFD based MT deposition with coarse fraction

