"""Pipeline execution engine with DAG-based stage orchestration."""

from __future__ import annotations
from typing import List, Dict, Set, Any, Optional, Type
from dataclasses import dataclass
import time

from ..contracts.stage import Stage
from ..contracts.types import (
    CFDInput, CFDResult,
    DepositionInput, DepositionResult,
    PBBKInput, PBBKResult,
    PKInput, PKResult
)
from ..contracts.errors import ModelError
from ..domain.entities import API, Product, InhalationManeuver
from .registry import get_registry, ModelInfo
from .context import RunContext


@dataclass
class StageResult:
    """Result from executing a pipeline stage."""
    stage_name: str
    model_name: str
    data: Any  # DepositionResult, PBBKResult, or PKResult
    metadata: Dict[str, Any]


class Pipeline:
    """Pipeline executor with DAG-based stage orchestration."""
    
    def __init__(self):
        self.registry = get_registry()
    
    def run(
        self,
        stages: List[str],
        entities: Dict[str, Any],
        stage_configs: Dict[str, Dict[str, Any]],
        context: RunContext
    ) -> Dict[str, StageResult]:
        """Run pipeline stages.
        
        Args:
            stages: List of stage names to execute ("deposition", "pbbm", "pk")
            entities: Dictionary with hydrated entities (subject, api, product, maneuver)
            stage_configs: Configuration for each stage
            context: Run context for logging and timing
            
        Returns:
            Dictionary mapping stage names to results
            
        Raises:
            ModelError: If pipeline execution fails
        """
        context.start_run()
        
        try:
            # Validate requested stages
            valid_stages = {"cfd", "deposition", "pbbm", "pk"}
            invalid_stages = set(stages) - valid_stages
            if invalid_stages:
                raise ModelError(f"Invalid stages: {invalid_stages}")
            
            # Execute stages in dependency order
            results: Dict[str, StageResult] = {}
            
            # Stage 1: CFD surrogate (optional)
            if "cfd" in stages:
                results["cfd"] = self._run_cfd_stage(
                    entities, stage_configs.get("cfd", {}), context
                )

            # Stage 2: Deposition (optional)
            if "deposition" in stages:
                results["deposition"] = self._run_deposition_stage(
                    entities,
                    stage_configs.get("deposition", {}),
                    results.get("cfd"),
                    context,
                )

            # Stage 3: Lung PBPK (optional, can use deposition results)
            if "pbbm" in stages:
                results["pbbm"] = self._run_pbbm_stage(
                    entities, stage_configs.get("pbbm", {}), 
                    results.get("deposition"), context
                )

            # Stage 4: Systemic PK (optional, can use PBPK results)
            if "pk" in stages:
                results["pk"] = self._run_pk_stage(
                    entities, stage_configs.get("pk", {}),
                    results.get("pbbm"), context
                )
            
            total_runtime = context.end_run()
            context.logger.info("Pipeline completed successfully", 
                              stages=stages, total_runtime_s=total_runtime)
            
            return results
            
        except Exception as e:
            context.end_run()
            context.logger.error("Pipeline failed", error=str(e))
            raise ModelError(f"Pipeline execution failed: {e}") from e
    
    def _run_cfd_stage(
        self,
        entities: Dict[str, Any],
        config: Dict[str, Any],
        context: RunContext
    ) -> StageResult:
        """Run CFD surrogate stage."""
        with context.time_stage("cfd"):
            model_name = config.get("model", "ml")
            model = self.registry.get_model("cfd", model_name)

            input_data = CFDInput(
                subject=entities["subject"],
                product=entities["product"],
                maneuver=entities["maneuver"],
                api=entities.get("api"),
                params=config.get("params", {}),
            )

            context.logger.info("Running CFD surrogate", model=model_name)
            result = model.run(input_data)

            return StageResult(
                stage_name="cfd",
                model_name=model_name,
                data=result,
                metadata=getattr(result, 'metadata', {}) or {},
            )

    def _run_deposition_stage(
        self,
        entities: Dict[str, Any],
        config: Dict[str, Any],
        cfd_result: Optional[StageResult],
        context: RunContext
    ) -> StageResult:
        """Run deposition stage."""
        with context.time_stage("deposition"):
            # Get model
            model_name = config.get("model", "null")
            model = self.registry.get_model("deposition", model_name)
            
            # Prepare input
            input_data = DepositionInput(
                subject=entities["subject"],
                product=entities["product"],
                maneuver=entities["maneuver"],
                api=entities.get("api"),
                particle_grid=config.get("particle_grid"),
                params=config.get("params", {}),
                cfd_result=cfd_result.data if cfd_result is not None else None,
            )
            
            # Execute
            context.logger.info("Running deposition", model=model_name)
            result = model.run(input_data)

            return StageResult(
                stage_name="deposition",
                model_name=model_name,
                data=result,
                metadata=getattr(result, 'metadata', {})
            )
    
    def _run_pbbm_stage(
        self,
        entities: Dict[str, Any],
        config: Dict[str, Any],
        deposition_result: Optional[StageResult],
        context: RunContext
    ) -> StageResult:
        """Run lung PBPK stage."""
        with context.time_stage("pbbm"):
            # Get model
            model_name = config.get("model", "null")
            model = self.registry.get_model("lung_pbbm", model_name)
            
            # Prepare input
            lung_seed = deposition_result.data if deposition_result else None
            
            input_data = PBBKInput(
                subject=entities["subject"],
                api=entities["api"],
                lung_seed=lung_seed,
                params=config.get("params", {})
            )
            
            # Execute
            context.logger.info("Running lung PBPK", model=model_name)
            result = model.run(input_data)

            return StageResult(
                stage_name="pbbm", 
                model_name=model_name,
                data=result,
                metadata=getattr(result, 'metadata', {})
            )
    
    def _run_pk_stage(
        self,
        entities: Dict[str, Any],
        config: Dict[str, Any],
        pbbm_result: Optional[StageResult],
        context: RunContext
    ) -> StageResult:
        """Run systemic PK stage."""
        with context.time_stage("pk"):
            # Get model
            model_name = config.get("model", "null")
            model = self.registry.get_model("systemic_pk", model_name)

            # Prepare input
            pulmonary_input = None
            if pbbm_result and hasattr(pbbm_result.data, 'pulmonary_outflow'):
                pulmonary_input = pbbm_result.data.pulmonary_outflow

            pk_params: Dict[str, Any] = dict(config.get("params", {}))
            if pbbm_result is not None:
                pk_params.setdefault('pbbm_result', pbbm_result.data)
                pk_params.setdefault('pbbm_comprehensive', getattr(pbbm_result.data, 'comprehensive', None))
                pk_params.setdefault('pbbm_time_s', getattr(pbbm_result.data, 't', None))

            input_data = PKInput(
                subject=entities["subject"],
                api=entities["api"],
                pulmonary_input=pulmonary_input,
                gi_input=None,  # Not supported yet
                params=pk_params
            )

            # Execute
            context.logger.info("Running systemic PK", model=model_name)
            result = model.run(input_data)

            return StageResult(
                stage_name="pk",
                model_name=model_name,
                data=result,
                metadata=getattr(result, 'metadata', {})
            )
    
    def validate_stage_sequence(self, stages: List[str]) -> None:
        """Validate that requested stages can be executed.
        
        Args:
            stages: List of stage names
            
        Raises:
            ModelError: If stage sequence is invalid
        """
        valid_stages = {"deposition", "pbbm", "pk"}
        invalid_stages = set(stages) - valid_stages
        if invalid_stages:
            raise ModelError(f"Invalid stages: {invalid_stages}")
        
        # For now, any combination is valid since stages are optional
        # Later PRs may add stricter dependency validation
