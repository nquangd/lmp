"""Data structures for simulation results.

Provides intuitive hierarchical access to PBBM and PK results.
"""

from .pbbm_results import (
    SubjectPBBMData, 
    RegionData,
    CompartmentData,
    ConcentrationData,
    AmountData,
    ConcentrationAccessor,
    create_example_subject_data
)

from .comprehensive_results import (
    BindingStateData,
    RegionalAmountData,
    FluxData,
    PKResultsData,
    MassBalanceData
)

__all__ = [
    'SubjectPBBMData',
    'RegionData', 
    'CompartmentData',
    'ConcentrationData',
    'AmountData',
    'ConcentrationAccessor',
    'create_example_subject_data',
    # New architecture data structures
    'BindingStateData',
    'RegionalAmountData',
    'FluxData',
    'PKResultsData',
    'MassBalanceData'
]