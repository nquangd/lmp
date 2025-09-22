"""Data structures for simulation results.

Provides intuitive hierarchical access to PBBM and PK results.
"""

from .pbbm_results import (
    SubjectPBBMData, 
    RegionData,
    CompartmentData,
    ConcentrationData,
    AmountData,
    create_example_subject_data
)

from .comprehensive_results import (
    BindingStateData,
    RegionalAmountData,
    FluxData,
    PKResultsData,
    MassBalanceData,
    ComprehensivePBBMResults
)

from .adapters import (
    build_pk_results_from_orchestrator,
    build_regional_results_from_orchestrator,
    build_comprehensive_results_from_orchestrator,
)

from .subject_results import (
    Results_Deposition,
    Results_PK,
)

from .units import UnitContext

# Optional higher-level wrappers for study/product/API hierarchy
class Study:
    def __init__(self, name: str):
        self.name = name
        self.Product = None

class ProductNode:
    def __init__(self, name: str):
        self.name = name
        self.API = None

class APINode:
    def __init__(self, name: str, subject: SubjectPBBMData):
        self.name = name
        self.subject = subject

def build_study_hierarchy(study_name: str, product_name: str, api_name: str,
                          subject: SubjectPBBMData) -> Study:
    study = Study(study_name)
    study.Product = ProductNode(product_name)
    study.Product.API = APINode(api_name, subject)
    return study

__all__ = [
    'SubjectPBBMData',
    'RegionData', 
    'CompartmentData',
    'ConcentrationData',
    'AmountData',
    'create_example_subject_data',
    'UnitContext',
    'Study', 'ProductNode', 'APINode', 'build_study_hierarchy',
    # New architecture data structures
    'BindingStateData',
    'RegionalAmountData',
    'FluxData',
    'PKResultsData',
    'MassBalanceData',
    'ComprehensivePBBMResults',
    # Adapters
    'build_pk_results_from_orchestrator',
    'build_regional_results_from_orchestrator',
    'build_comprehensive_results_from_orchestrator',
    # Subject-scoped results wrappers
    'Results_Deposition',
    'Results_PK',
]
