"""Utilities for executing subject tasks through the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Optional, Any
import copy

import numpy as np

from ..engine.context import RunContext
from ..engine.workflow import Workflow, get_workflow
from ..contracts.types import CFDResult, PBBKResult, DepositionResult, PKResult
from ..domain.entities import Product
from .subject_tasks import SubjectTask, prepare_entities


@dataclass
class ProductResult:
    cfd: Optional[CFDResult]
    deposition: Optional[DepositionResult]
    pbpk: Optional[PBBKResult]
    pk: Optional[PKResult]


@dataclass
class TaskResult:
    task: SubjectTask
    products: Dict[str, ProductResult]
    stage_results: Dict[str, Dict[str, "StageResult"]]
    runtime_metadata: Dict[str, Any]


def run_pipeline_for_task(
    task: SubjectTask,
    pipeline: Optional["Pipeline"] = None,
    workflow: Optional[Workflow] = None,
    stage_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
) -> TaskResult:
    """Execute deposition and PBPK stages for a subject task."""
    from ..engine.pipeline import Pipeline, StageResult

    entities = prepare_entities(task)
    subject_final = entities['subject']
    api = entities['api']

    pipeline = pipeline or Pipeline()
    workflow = workflow or get_workflow("deposition_pbbm_pk")

    results: Dict[str, ProductResult] = {}
    stage_results_per_product: Dict[str, Dict[str, StageResult]] = {}

    for product_name in task.products:
        context = RunContext(
            run_id=f"task_{task.api_name}_{task.subject_index}_{product_name}",
            seed=task.seed,
            threads=1,
            enable_numba=True,
            artifact_dir=None,
            logger=None,
        )

        if task.product_entities and product_name in task.product_entities:
            product_entity = task.product_entities[product_name]
        else:
            product_entity = Product.from_builtin(product_name)

        entities_payload = {
            'subject': subject_final,
            'api': api,
            'product': product_entity,
            'maneuver': subject_final.inhalation_maneuver,
        }

        stage_overrides_map: Dict[str, Dict[str, Any]] = {}
        if stage_overrides:
            for stage_name, override in stage_overrides.items():
                if not isinstance(override, dict):
                    continue
                copied = dict(override)
                params = copied.get('params')
                if isinstance(params, dict):
                    copied['params'] = dict(params)
                stage_overrides_map[stage_name] = copied

        product_route = getattr(product_entity, "route", None) if isinstance(product_entity, Product) else None
        inferred_route = product_route or (Product.infer_route_from_stages(workflow.stages) if isinstance(product_entity, Product) else None)

        if isinstance(product_entity, Product) and inferred_route:
            route_stage_list = product_entity.get_route_stage_list(inferred_route)
            if route_stage_list:
                stage_configs = {
                    stage_name: workflow.stage_configs.get(stage_name, {})
                    for stage_name in route_stage_list
                }
                workflow = Workflow(
                    name=f"{workflow.name}__{inferred_route}",
                    stages=tuple(route_stage_list),
                    stage_configs=stage_configs,
                    population_variability_defaults=workflow.population_variability_defaults,
                )

            try:
                default_stage_overrides = product_entity.build_stage_overrides(
                    inferred_route,
                    api_name=task.api_name,
                )
            except Exception:
                default_stage_overrides = {}
            if default_stage_overrides:
                for stage_name, payload in default_stage_overrides.items():
                    existing = stage_overrides_map.get(stage_name)
                    if existing is None:
                        stage_overrides_map[stage_name] = copy.deepcopy(payload)
                        continue
                    if 'model' not in existing and 'model' in payload:
                        existing['model'] = payload['model']
                    payload_params = payload.get('params')
                    if isinstance(payload_params, dict):
                        existing_params = existing.setdefault('params', {})
                        for param_key, param_value in payload_params.items():
                            existing_params.setdefault(param_key, param_value)

        if 'pbbm' in workflow.stages:
            pbbm_config = stage_overrides_map.get('pbbm', {'params': {}})
            pbbm_params = dict(pbbm_config.get('params', {}))
            if getattr(task, 'charcoal_block', False):
                pbbm_params['charcoal_block'] = True
            if getattr(task, 'suppress_et_absorption', False):
                pbbm_params['suppress_et_absorption'] = True
            if pbbm_params:
                pbbm_config['params'] = pbbm_params
                stage_overrides_map['pbbm'] = pbbm_config

        stage_results = workflow.execute(
            entities=entities_payload,
            context=context,
            pipeline=pipeline,
            stage_overrides=stage_overrides_map,
        )

        cfd_stage = stage_results.get('cfd')
        deposition_stage = stage_results.get('deposition')
        pbbm_stage = stage_results.get('pbbm')
        pk_stage = stage_results.get('pk')
        iv_pk_stage = stage_results.get('iv_pk')
        gi_pk_stage = stage_results.get('gi_pk')

        cfd_result = cfd_stage.data if (cfd_stage and isinstance(cfd_stage.data, CFDResult)) else (
            cfd_stage.data if cfd_stage else None
        )

        deposition_result = None
        if deposition_stage:
            deposition_result = deposition_stage.data
        # Debugging check
        #print(f"Product: {product_name}, Deposition Result: {deposition_result}")
        pbpk_result = None
        if pbbm_stage:
            pbpk_data = pbbm_stage.data
            if isinstance(pbpk_data, PBBKResult):
                pbpk_result = pbpk_data
            else:
                pbpk_result = PBBKResult(comprehensive=pbpk_data)

        pk_result: Optional[PKResult] = None
        if pk_stage:
            pk_data = pk_stage.data
            if isinstance(pk_data, PKResult):
                pk_result = pk_data
        if pk_result is None and iv_pk_stage:
            pk_data = iv_pk_stage.data
            if isinstance(pk_data, PKResult):
                pk_result = pk_data
        if pk_result is None and gi_pk_stage:
            pk_data = gi_pk_stage.data
            if isinstance(pk_data, PKResult):
                pk_result = pk_data

        
        results[product_name] = ProductResult(
            cfd=cfd_result,
            deposition=deposition_result,
            pbpk=pbpk_result,
            pk=pk_result,
        )
        stage_results_per_product[product_name] = stage_results

    runtime_metadata = context.get_runtime_metadata()
    return TaskResult(
        task=task,
        products=results,
        stage_results=stage_results_per_product,
        runtime_metadata=runtime_metadata,
    )


def run_tasks(
    tasks: Sequence[SubjectTask],
    workflow: Optional[Workflow] = None,
) -> List[TaskResult]:
    """Run a collection of subject tasks sequentially."""
    from ..engine.pipeline import Pipeline

    pipeline = Pipeline()
    outputs: List[TaskResult] = []
    for task in tasks:
        outputs.append(run_pipeline_for_task(task, pipeline=pipeline, workflow=workflow))
    return outputs
