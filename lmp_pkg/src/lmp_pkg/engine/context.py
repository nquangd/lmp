"""Run context for pipeline execution."""

from __future__ import annotations
import time
from typing import Dict, Any, Optional
from pathlib import Path
import structlog
import numpy as np


class RunContext:
    """Context for pipeline execution with logging, timing, and RNG management."""
    
    def __init__(
        self,
        run_id: str,
        seed: int = 123,
        threads: int = 1,
        enable_numba: bool = False,
        artifact_dir: Optional[Path] = None,
        logger: Optional[structlog.BoundLogger] = None
    ):
        self.run_id = run_id
        self.seed = seed
        self.threads = threads
        self.enable_numba = enable_numba
        self.artifact_dir = artifact_dir or Path("results")
        
        # Set up logging
        if logger is None:
            self.logger = structlog.get_logger().bind(run_id=run_id)
        else:
            self.logger = logger.bind(run_id=run_id)
        
        # Initialize timing
        self._start_time: Optional[float] = None
        self._stage_times: Dict[str, float] = {}
        
        # Initialize RNG
        self.rng = np.random.default_rng(seed)
        
        # Runtime metadata
        self.metadata: Dict[str, Any] = {
            "run_id": run_id,
            "seed": seed,
            "threads": threads,
            "enable_numba": enable_numba
        }
    
    def start_run(self) -> None:
        """Mark start of run execution."""
        self._start_time = time.perf_counter()
        self.logger.info("Pipeline execution started")
    
    def end_run(self) -> float:
        """Mark end of run execution and return total runtime.
        
        Returns:
            Total runtime in seconds
        """
        if self._start_time is None:
            return 0.0
        
        runtime = time.perf_counter() - self._start_time
        self.logger.info("Pipeline execution completed", runtime_s=runtime)
        return runtime
    
    def time_stage(self, stage_name: str):
        """Context manager for timing a stage.
        
        Args:
            stage_name: Name of the stage being timed
        
        Returns:
            Context manager that tracks stage execution time
        """
        return _StageTimer(self, stage_name)
    
    def get_stage_rng(self, stage_name: str) -> np.random.Generator:
        """Get deterministic RNG for a stage.
        
        Args:
            stage_name: Name of the stage
            
        Returns:
            Deterministic RNG seeded for this stage
        """
        # Create deterministic seed from run seed and stage name
        stage_seed = self.seed ^ hash(f"{self.run_id}:{stage_name}") % (2**31)
        stage_seed = abs(stage_seed) % (2**31)
        
        return np.random.default_rng(stage_seed)
    
    def get_artifact_path(self, filename: str) -> Path:
        """Get path for an artifact file.
        
        Args:
            filename: Name of the artifact file
            
        Returns:
            Full path to the artifact file
        """
        run_dir = self.artifact_dir / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir / filename
    
    def get_runtime_metadata(self) -> Dict[str, Any]:
        """Get runtime metadata for this execution.
        
        Returns:
            Dictionary with runtime information
        """
        metadata = self.metadata.copy()
        metadata.update({
            "stage_times": self._stage_times.copy(),
            "total_runtime_s": sum(self._stage_times.values())
        })
        return metadata


class _StageTimer:
    """Context manager for timing stage execution."""
    
    def __init__(self, context: RunContext, stage_name: str):
        self.context = context
        self.stage_name = stage_name
        self.start_time: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        self.context.logger.info("Stage started", stage=self.stage_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            runtime = time.perf_counter() - self.start_time
            self.context._stage_times[self.stage_name] = runtime
            
            if exc_type is None:
                self.context.logger.info(
                    "Stage completed", 
                    stage=self.stage_name,
                    runtime_s=runtime
                )
            else:
                self.context.logger.error(
                    "Stage failed", 
                    stage=self.stage_name,
                    runtime_s=runtime,
                    error=str(exc_val)
                )