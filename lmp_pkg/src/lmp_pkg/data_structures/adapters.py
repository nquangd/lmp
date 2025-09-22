"""Adapters to construct data structure objects from model results.

These helpers bridge outputs from the new PBPK orchestrator in
`lmp_pkg.models.pbpk.pbpk_orchestrator` to the data structures in this
package (PKResultsData, RegionalAmountData, ComprehensivePBBMResults).
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np

from .comprehensive_results import (
    PKResultsData,
    RegionalAmountData,
    FluxData,
    MassBalanceData,
    ComprehensivePBBMResults,
)
from .pbbm_results import SubjectPBBMData
from .units import UnitContext


def build_pk_results_from_orchestrator(results_data: Dict[str, np.ndarray],
                                       time_s: np.ndarray,
                                       molecular_weight_ug_per_umol: Optional[float]) -> PKResultsData:
    """Create PKResultsData from PBPKOrchestrator.extract_results output.

    Args:
        results_data: The `results['pk']` dict from orchestrator with keys like
                      'central', 'peripheral1', 'peripheral2', 'plasma_concentration_ng_ml'.
        time_s: Time vector in seconds corresponding to the series
        molecular_weight_ug_per_umol: API MW for unit conversions (μg/μmol)

    Returns:
        PKResultsData instance with normalized units (pmol/mL) and amounts.
    """
    time_h = time_s / 3600.0

    central = results_data.get('central', np.zeros_like(time_s))

    peripheral_amounts: Dict[str, np.ndarray] = {}
    if 'peripheral1' in results_data:
        peripheral_amounts['peripheral1'] = results_data['peripheral1']
    if 'peripheral2' in results_data:
        peripheral_amounts['peripheral2'] = results_data['peripheral2']

    pk = PKResultsData(
        time_s=time_s,
        time_h=time_h,
        plasma_concentration=None,  # will derive from ng/mL if needed
        central_amounts=central,
        peripheral_amounts=peripheral_amounts if peripheral_amounts else None,
        plasma_concentration_ng_per_ml=results_data.get('plasma_concentration_ng_ml'),
        molecular_weight=molecular_weight_ug_per_umol,
    )
    return pk


def build_regional_results_from_orchestrator(results_data: Dict[str, np.ndarray],
                                             time_s: np.ndarray) -> Dict[str, RegionalAmountData]:
    """Create RegionalAmountData dict from orchestrator lung results.

    The orchestrator flattens lung outputs with region prefixes, e.g.:
      'ET_elf_amount', 'ET_epithelium_total', 'ET_tissue_total', 'ET_particles_total'

    Args:
        results_data: The `results['lung']` dict from orchestrator
        time_s: Time vector in seconds

    Returns:
        Mapping of region name -> RegionalAmountData
    """
    time_h = time_s / 3600.0

    regions = {}
    # Discover regions by scanning keys with pattern '<Region>_...'
    for key in results_data.keys():
        if '_' in key:
            region_name = key.split('_', 1)[0]
            regions[region_name] = True

    regional: Dict[str, RegionalAmountData] = {}
    for region in regions.keys():
        epi = results_data.get(f'{region}_epithelium_total', np.zeros_like(time_s))
        tissue = results_data.get(f'{region}_tissue_total', np.zeros_like(time_s))
        epi_shallow = results_data.get(f'{region}_epithelium_shallow', None)
        tis_shallow = results_data.get(f'{region}_tissue_shallow', None)
        elf = results_data.get(f'{region}_elf_amount', None)
        solid = results_data.get(f'{region}_particles_total', None)

        # Volumes for concentration conversion
        epi_vol = results_data.get(f'{region}_epithelium_volume_ml', None)
        tis_vol = results_data.get(f'{region}_tissue_volume_ml', None)
        fu_epi_calc = results_data.get(f'{region}_fu_epithelium_calc', None)
        fu_tis_calc = results_data.get(f'{region}_fu_tissue_calc', None)

        regional[region] = RegionalAmountData(
            region_name=region,
            time_s=time_s,
            time_h=time_h,
            epithelium_amounts=epi,
            tissue_amounts=tissue,
            epithelium_shallow_amounts=epi_shallow if epi_shallow is not None else None,
            tissue_shallow_amounts=tis_shallow if tis_shallow is not None else None,
            elf_amounts=elf,
            solid_drug_amounts=solid,
            epithelium_volume_ml=float(epi_vol) if epi_vol is not None else None,
            tissue_volume_ml=float(tis_vol) if tis_vol is not None else None,
            fu_epithelium_calc=float(fu_epi_calc) if fu_epi_calc is not None else None,
            fu_tissue_calc=float(fu_tis_calc) if fu_tis_calc is not None else None,
        )

    return regional


def build_comprehensive_results_from_orchestrator(orchestrator_result,
                                                  molecular_weight_ug_per_umol: Optional[float]) -> ComprehensivePBBMResults:
    """Build a ComprehensivePBBMResults object from a PBPK orchestrator solve result.

    Args:
        orchestrator_result: Return value from PBPKOrchestrator.solve (scipy OdeResult-like)
        molecular_weight_ug_per_umol: API MW for unit conversions (μg/μmol)

    Returns:
        ComprehensivePBBMResults with PK, regional, flux and mass balance placeholders.
    """
    # Extract unified keys
    t_s = np.array(orchestrator_result.time_points)
    comp_results = orchestrator_result.results_data

    lung_data = comp_results.get('lung', {})
    pk_data = comp_results.get('pk', {})

    # Build PK and regional structures
    pk = build_pk_results_from_orchestrator(pk_data, t_s, molecular_weight_ug_per_umol)
    regional = build_regional_results_from_orchestrator(lung_data, t_s)

    # Build FluxData: use orchestrator-provided flux at t_eval if available
    flux_block = comp_results.get('flux', {}) if isinstance(comp_results, dict) else {}
    sys_abs = np.array(flux_block.get('total_lung_to_systemic', np.zeros_like(t_s)))
    gi_abs = np.array(flux_block.get('gi_to_systemic', np.zeros_like(t_s))) if isinstance(flux_block, dict) else None
    mcc_total = np.array(flux_block.get('mcc_total', np.zeros_like(t_s)))
    regional_mcc = flux_block.get('per_region_mcc', None) if isinstance(flux_block, dict) else None
    flux = FluxData(
        time_s=t_s,
        time_h=t_s / 3600.0,
        systemic_absorption_rate=sys_abs if sys_abs.size else np.zeros_like(t_s),
        mucociliary_clearance_rate=mcc_total if mcc_total.size else np.zeros_like(t_s),
        gi_absorption_rate=gi_abs if gi_abs.size else None,
        regional_systemic_absorption=flux_block.get('per_region', None) if isinstance(flux_block, dict) else None,
        regional_mcc_rates=regional_mcc
    )

    # Mass balance placeholders
    lung_total = np.zeros_like(t_s)
    for region in regional.values():
        lung_total = lung_total + region.total_amounts

    mass_balance = MassBalanceData(
        time_s=t_s,
        time_h=t_s / 3600.0,
        initial_deposition_pmol=float(lung_total[0]) if len(lung_total) else 0.0,
        lung_amounts=lung_total,
        systemic_amounts=pk.total_systemic_amounts,
        cumulative_elimination=np.zeros_like(t_s)
    )

    return ComprehensivePBBMResults(
        time_s=t_s,
        regional_data=regional,
        pk_data=pk,
        flux_data=flux,
        mass_balance=mass_balance,
        metadata={'source': 'pbpk_orchestrator'}
    )


def build_study_from_comprehensive(study_name: str,
                                   product_name: str,
                                   api_name: str,
                                   comp: ComprehensivePBBMResults,
                                   units: Optional[UnitContext] = None):
    """Create a Study.Product.API.Subject hierarchy from ComprehensivePBBMResults.

    Populates regions (Epithelium, Tissue) and Plasma systemic compartment with
    unit-aware Amount/Concentration accessors.
    """
    mw = float(comp.pk_data.molecular_weight) if comp.pk_data.molecular_weight else 250.0
    units = units or UnitContext(molecular_weight_ug_per_umol=mw,
                                 concentration_unit='ng/mL', amount_unit='ng')

    # Subject container
    subj = SubjectPBBMData("Subject", comp.time_s, molecular_weight=mw, units=units)

    # Regions
    for region_name, r in comp.regional_data.items():
        reg = subj.add_region(region_name)
        epi_vol = float(r.epithelium_volume_ml) if r.epithelium_volume_ml else 1.0
        tis_vol = float(r.tissue_volume_ml) if r.tissue_volume_ml else 1.0
        fu_epi = float(r.fu_epithelium_calc) if r.fu_epithelium_calc is not None else 1.0
        fu_tis = float(r.fu_tissue_calc) if r.fu_tissue_calc is not None else 1.0
        reg.add_compartment('Epithelium', r.epithelium_amounts, volume_ml=epi_vol, fu=fu_epi)
        reg.add_compartment('Tissue', r.tissue_amounts, volume_ml=tis_vol, fu=fu_tis)

    # Systemic Plasma (estimate Vc if not exposed)
    pk = comp.pk_data
    central = pk.central_amounts if pk.central_amounts is not None else None
    conc_pmol_ml = pk.plasma_concentration
    if central is not None and conc_pmol_ml is not None and np.any(conc_pmol_ml > 0):
        idx = conc_pmol_ml > 0
        vc_L = float(np.median(central[idx] / (conc_pmol_ml[idx] * 1000.0)))
        if not np.isfinite(vc_L) or vc_L <= 0:
            vc_L = 29.92
    else:
        vc_L = 29.92
    if central is None:
        central = np.zeros_like(comp.time_s)
    subj.add_systemic_compartment('Plasma', central, volume_L=vc_L, fu=1.0)

    # Wrap in Study hierarchy
    study = build_study_hierarchy(study_name, product_name, api_name, subj)
    return study
