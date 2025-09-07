import numpy as np
import copy

class Product:
    """Defines a product with APIs, doses, and aerosol properties."""
    def __init__(self, name, base_apis, study):
        self.name = name
        self.apis_data = copy.deepcopy(base_apis) 

class Study:
    """Manages study conditions."""
    def __init__(self, study_type = 'NON_CHARCOAL', trial_size = 100, n_trials = 50):
        self.study_type = study_type
        self.trial_size = trial_size
        self.n_trials = n_trials
        

    


