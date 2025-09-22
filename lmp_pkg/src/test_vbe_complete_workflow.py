#!/usr/bin/env python3
"""End-to-end variability-enabled workflow using the stage pipeline."""

import os
import numpy as np
from multiprocessing import Pool, cpu_count

from lmp_pkg.services.vbe import vbe_from_results
from lmp_pkg.services.vbe_helpers import MetricSpec
from lmp_pkg.services.visualization import PBBMVisualizer, PlotConfig
from lmp_pkg.simulation.subject_tasks import build_tasks
from lmp_pkg.simulation.pipeline_runner import run_pipeline_for_task

import warnings
warnings.filterwarnings('ignore')


def execute_task(task):
    result = run_pipeline_for_task(task)
    return {
        'api': task.api_name,
        'subject_index': task.subject_index,
        'products': {
            name: product_result.pbpk.comprehensive
            for name, product_result in result.products.items()
        },
    }


def main(n_subjects: int = 6, enable_variability: bool = True):
    apis = ['BD', 'GP', 'FF']
    n_workers = min(cpu_count(), 6)

    tasks = build_tasks(
        apis,
        n_subjects,
        base_seed=1234,
        apply_variability=enable_variability,
        charcoal_block=False,
        suppress_et_absorption=False,
    )

    with Pool(processes=n_workers) as pool:
        raw_results = pool.map(execute_task, tasks)

    aggregated = {api: [] for api in apis}
    for entry in raw_results:
        subject_id = f"S{entry['subject_index'] + 1}"
        products = entry['products']
        aggregated[entry['api']].append({
            'subject_id': subject_id,
            'ref': products['reference_product'],
            'test': products['test_product'],
        })

    viz = PBBMVisualizer(PlotConfig(figsize=(10, 6)))
    MM = {'BD': 430, 'GP': 318, 'FF': 344}  # molecular mass for unit conversion
    for api in apis:
        print(f"\n=== API: {api} ===")
        results_list = aggregated[api]
        ref_dict = {r['subject_id']: r['ref'] for r in results_list}
        test_dict = {r['subject_id']: r['test'] for r in results_list}
        try:
            ref_auc =  [v.pk_data.auc_pmol_h_per_ml * MM[api] for v in ref_dict.values()]
            print(f"{api} Reference AUC (ng·h/mL) per subject: {ref_auc}")
        except Exception as e:
            pass
        any_comp = next(iter(ref_dict.values()))
        t_h = any_comp.time_s / 3600.0
        ref_plasma = np.stack([
            c.pk_data.plasma_concentration_ng_per_ml for c in ref_dict.values()
        ], axis=0)
        test_plasma = np.stack([
            c.pk_data.plasma_concentration_ng_per_ml for c in test_dict.values()
        ], axis=0)
        ref_auc = [np.trapz(c, t_h) * 1e3 for c in ref_plasma]
        test_auc = [np.trapz(c, t_h) * 1e3 for c in test_plasma]
        print(f"{api} Reference AUC (ng·h/mL) per subject: {ref_auc}")
        print(f"{api} Test AUC (ng·h/mL) per subject: {test_auc}")
        mean_ref = np.nanmean(ref_plasma, axis=0)
        mean_test = np.nanmean(test_plasma, axis=0)
        viz.plot_time_series(
            t_h,
            {f'{api} Reference': mean_ref, f'{api} Test': mean_test},
            title=f'{api} Plasma Concentration (mean)',
            ylabel='ng/mL',
        )

        metrics = [
            MetricSpec(level='Systemic', metric='AUC', systemic_unit='ng/mL', window_h=(0, 24)),
            MetricSpec(level='Systemic', metric='Cmax', systemic_unit='ng/mL'),
            MetricSpec(level='Regional', region='Al', metric='Deposition', amount_unit='ng'),
            MetricSpec(level='Systemic', metric='Efficacy', efficacy_model='fev1_copd_catalog', efficacy_params=None),
        ]

        for m in metrics:
            res = vbe_from_results(ref_dict, test_dict, m)
            print(
                f"{api} {m.metric} VBE: GMR={res['gmr']:.3f}, "
                f"CI90=[{res['ci_lower']:.3f}, {res['ci_upper']:.3f}], "
                f"pass={bool(res['pass_80125'])}"
            )


if __name__ == "__main__":
    os.environ.setdefault('PYTHONHASHSEED', '0')
    np.random.seed(1234)
    default_subjects = 2  # int(os.environ.get('LMP_VBE_SUBJECTS', '2'))
    enable_variability = False

    import time

    start_time = time.time()
    main(n_subjects=default_subjects, enable_variability=enable_variability)
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")
