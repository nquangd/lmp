import numpy as np



class Variability:
    """Holds variability data for sampling."""
    def __init__(self, enable_variability=True):
        # Variability for lung dose scaling factors, from notebook
    

        # GCV data for PK/PD parameters, just library. this is not an input variables in the UI
        gcv_data = {
            'Intra': {
                #'ET': {'BD': 0.150, 'GP': 0.150, 'FF': 0.150},
                'CL': {'BD': 0.05, 'GP': 0.4, 'FF': 0.1},
                'Eh': {'BD': 0.10, 'GP': 0.4, 'FF': 0.2},
           
            },
            'Inter': {
                #'ET': {'BD': 0.234, 'GP': 0.234, 'FF': 0.234},
                'CL': {'BD': 0.26, 'GP': 0.5, 'FF': 0.3},
                'Eh': {'BD': 0.114, 'GP': 0.5, 'FF': 0.4},
         
            }
        }
        
       
        
        self.Inter = {'Inhalation': {'pifr_Lpm': self.convert_gcv_to_sigma_log(0.3),
                                     'rise_time_s' : self.convert_gcv_to_sigma_log(0.1),
                                     },
                      'PK': {param: {key: self.convert_gcv_to_sigma_log(gcv) for key, gcv in group.items()} for param, group in gcv_data['Inter'].items()}
                      }
        self.Intra = {'Inhalation': {'pifr_Lpm': self.convert_gcv_to_sigma_log(0.3),
                                     'rise_time_s' : self.convert_gcv_to_sigma_log(0.1),
                                     },
                      'PK': {param: {key: self.convert_gcv_to_sigma_log(gcv) for key, gcv in group.items()} for param, group in gcv_data['Intra'].items()}
                      }
        for params in ['hold_time_s', 'breath_hold_time_s', 'exhalation_flow_Lpm', 'bolus_volume_ml', 'bolus_delay_s', 'mmad', 'gsd']:
            self.Inter['Inhalation'][params] = 0.0
            self.Intra['Inhalation'][params] = 0.0
            
        
        self.Inter['Inhalation']['ET'] = 0.235
        self.Intra['Inhalation']['ET'] = 0.15
        
        self.Physiology = { 'FRC': {'Mean': 3300, 'Sigma': 600}}
        
        if not enable_variability:
            for key in self.Inter['Inhalation']:
                self.Inter['Inhalation'][key] = 0.0
                self.Intra['Inhalation'][key] = 0.0
            for param in self.Inter['PK']:
                for key in self.Inter['PK'][param]:
                    self.Inter['PK'][param][key] = 0.0
                    self.Intra['PK'][param][key] = 0.0
            self.Physiology['FRC']['Sigma'] = 0.0
        self.distribution = {'Inhalation': {key: 'lognormal' if key in ['pifr_Lpm', 'rise_time_s', 'ET' ] else 'normal' for key in self.Inter['Inhalation'] },
                             'PK': {param: {key: 'lognormal' for key, gcv in group.items()} for param, group in self.Inter['PK'].items()}
                             }
        
            
    def convert_gcv_to_sigma_log(self, gcv):
        """Converts a Geometric CV to the sigma of its underlying normal distribution."""
        if gcv <= 0: return 0.0
        return np.sqrt(np.log(gcv**2 + 1))

