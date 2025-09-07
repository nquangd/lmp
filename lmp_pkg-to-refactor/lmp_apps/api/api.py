import numpy as np
import copy


class API:
    """
    Holds the parameters specific to a single API.
    Amount units converted to picograms (pg) and picomoles (pmol).
    MM (Molar Mass) is pg/pmol. Density is g/m^3. Solubility (solub) is pg/mL.
    """
    _defaults = {
        "BD": {"n_pk_compartments": 3, "MM": 430.54, "BP": 0.855, "cell_bind": 0, "Peff_para": 2.1E-7, "Eh": 87.0000, "Peff_GI": 3.5E-4, "pscale_para": {'ET': 0.0, 'BB': 0.0, 'bb_b': 0.0, 'Al': 0.0}, "pscale_Kin": {'ET': {'Epithelium': 1.0, 'Tissue': 1.0}, 'BB': {'Epithelium': 1.0, 'Tissue': 1.0}, 'bb_b': {'Epithelium': 1.0, 'Tissue': 1.0}, 'Al': {'Epithelium': 1.0, 'Tissue': 1.0}}, "pscale_Kout": {'ET': {'Epithelium': 1.0, 'Tissue': 1.0}, 'BB': {'Epithelium': 1.0, 'Tissue': 1.0}, 'bb_b': {'Epithelium': 1.0, 'Tissue': 1.0}, 'Al': {'Epithelium': 1.0, 'Tissue': 1.0}}, "fu": {'ELF': 1.0, 'Epithelium': 0.0433, 'Tissue': 0.0433, 'Plasma': 0.12}, "Peff": {'In': 2.1E-7, 'Out': 2.1E-7}, "pscale": {'ET': {'In': 125.0, 'Out': 125.0}, 'BB': {'In': 20.0, 'Out': 20.0}, 'bb_b': {'In': 10.0, 'Out': 10.0}, 'Al': {'In': 1.0, 'Out': 1.0}}, "K_in": {'Epithelium': 0.25/3600, 'Tissue': 0.25/3600}, "K_out": {'Epithelium': 0.4/3600, 'Tissue': 0.4/3600}, "V_central_L": 29.92, "CL_h": 94.79, "k12_h": 6.656, "k21_h": 2.203, "k13_h": 1.623, "k31_h": 0.3901, "D": 4.87E-6, "density": 1.2e6, "solub": 25.8 * 1e6},
        "GP": {"n_pk_compartments": 3, "MM": 318.43, "BP": 0.664, "cell_bind": 1, "Peff_para": 1.2E-7, "Eh": 96.5000, "Peff_GI": 1.535E-4, "pscale_para": {'ET': 3.5, 'BB': 1.0, 'bb_b': 1.0, 'Al': 5e-2}, "pscale_Kin": {'ET': {'Epithelium': 1, 'Tissue': 1}, 'BB': {'Epithelium': 1, 'Tissue': 1}, 'bb_b': {'Epithelium': 1, 'Tissue': 1}, 'Al': {'Epithelium': 1, 'Tissue': 1}}, "pscale_Kout": {'ET': {'Epithelium': 1, 'Tissue': 1}, 'BB': {'Epithelium': 1, 'Tissue': 1}, 'bb_b': {'Epithelium': 1, 'Tissue': 1}, 'Al': {'Epithelium': 1, 'Tissue': 1}}, "fu": {'ELF': 1.0, 'Epithelium': 0.217, 'Tissue': 0.217, 'Plasma': 0.516}, "Peff": {'In': 2.35E-7, 'Out': 2.0E-10}, "pscale": {'ET': {'In': 10.0, 'Out': 10.0}, 'BB': {'In': 1.0, 'Out': 1.0}, 'bb_b': {'In': 1.0, 'Out': 1.0}, 'Al': {'In': 1.0, 'Out': 1.0}}, "K_in": {'Epithelium': 0.0, 'Tissue': 7/3600}, "K_out": {'Epithelium': 0.0, 'Tissue': 0.48/3600}, "V_central_L": 5.6, "CL_h": 35.64, "k12_h": 7.38, "k21_h": 1.6, "k13_h": 2.41, "k31_h": 0.054, "D": 6.97E-6, "density": 1.2e6, "solub": 1270.0 * 1e6},
        "FF": {"n_pk_compartments": 3, "MM": 344.4, "BP": 0.9, "cell_bind": 0, "Peff_para": 2.0E-8, "Eh": 76.000, "Peff_GI": 5.0E-4, "pscale_para": {'ET': 0.0, 'BB': 0.0, 'bb_b': 0.0, 'Al': 0.0}, "pscale_Kin": {'ET': {'Epithelium': 0, 'Tissue': 0}, 'BB': {'Epithelium': 1, 'Tissue': 1}, 'bb_b': {'Epithelium': 1, 'Tissue': 1}, 'Al': {'Epithelium': 1, 'Tissue': 1}}, "pscale_Kout": {'ET': {'Epithelium': 0, 'Tissue': 0}, 'BB': {'Epithelium': 1, 'Tissue': 1}, 'bb_b': {'Epithelium': 1, 'Tissue': 1}, 'Al': {'Epithelium': 1, 'Tissue': 1}}, "fu": {'ELF': 1.0, 'Epithelium': 0.16, 'Tissue': 0.16, 'Plasma': 0.38}, "Peff": {'In': 8.0E-9, 'Out': 8.0E-9}, "pscale": {'ET': {'In': 1750.0, 'Out': 1750.0}, 'BB': {'In': 1.0, 'Out': 1.0}, 'bb_b': {'In': 1.0, 'Out': 1.0}, 'Al': {'In': 1.0, 'Out': 1.0}}, "K_in": {'Epithelium': 1495/3600, 'Tissue': 1495/3600}, "K_out": {'Epithelium': 0.3/3600, 'Tissue': 0.3/3600}, "V_central_L": 36.2638, "CL_h": 105.5188, "k12_h": 6.5213, "k21_h": 1.1265, "k13_h": 0.8745, "k31_h": 0.1682, "D": 6.97E-6, "density": 1.2e6, "solub": 1200.0 * 1e6}
    }
    
    def __init__(self, name, **kwargs):
        self.name = name
        if name in self._defaults:
            defaults = copy.deepcopy(self._defaults[name])
            self.__dict__.update(defaults)
        self.__dict__.update(kwargs)
        if hasattr(self, 'n_pk_compartments') and self.n_pk_compartments == 2:
            self.k13_h, self.k31_h = 0.0, 0.0
