import numpy as np
import copy

class Product:
    """Defines a product with APIs, doses, and aerosol properties."""
    def __init__(self, name, apis_data):
        self.name = name
        self.apis_data = apis_data

class Study:
    """Manages study conditions."""
    def __init__(self, study_type, base_apis):
        self.study_type = study_type
        self.apis = self.apply_study_conditions(base_apis)
        

    def apply_study_conditions(self, base_apis):
        study_apis = copy.deepcopy(base_apis)
        if self.study_type == 'CHARCOAL':
            for api in study_apis.values():
                api.pscale['ET'].update({'In': 0, 'Out': 0})
                api.pscale_para['ET'] = 0
                api.pscale_Kin['ET'].update({'Epithelium': 0, 'Tissue': 0})
                api.pscale_Kout['ET'].update({'Epithelium': 0, 'Tissue': 0})
                api.Eh = 100
        return study_apis


