#!/usr/bin/env python3
"""SLURM array job runner.

This script is designed to be called by SLURM job arrays to execute
individual simulation runs from a manifest file.

Usage:
    python -m lmp_pkg.scripts.run_slurm \\
        --config lmp.toml \\
        --manifest manifest.csv \\
        --index $SLURM_ARRAY_TASK_ID \\
        --artifacts /scratch/$USER/lmp_runs

Example SLURM script:
    #!/bin/bash
    #SBATCH --job-name=lmp
    #SBATCH --array=1-1000
    #SBATCH --cpus-per-task=4
    #SBATCH --mem=8G
    #SBATCH --time=02:00:00
    #SBATCH --output=logs/lmp_%A_%a.out
    #SBATCH --error=logs/lmp_%A_%a.err
    
    python -m lmp_pkg.scripts.run_slurm \\
        --config lmp.toml \\
        --manifest manifest.csv \\
        --index $SLURM_ARRAY_TASK_ID \\
        --artifacts /scratch/$USER/lmp_runs
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any
import argparse
import time

import pandas as pd
import structlog

from .. import app_api
from ..contracts.errors import LMPError

# Set up structured logging for SLURM environment
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run single simulation from SLURM array manifest"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=Path,
        required=True,
        help="Base configuration file"
    )
    
    parser.add_argument(
        "--manifest", "-m", 
        type=Path,
        required=True,
        help="Simulation manifest CSV file"
    )
    
    parser.add_argument(
        "--index", "-i",
        type=int,
        required=True,
        help="Row index in manifest (1-indexed, SLURM_ARRAY_TASK_ID)"
    )
    
    parser.add_argument(
        "--artifacts", "-a",
        type=Path,
        help="Artifacts directory (overrides config)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def load_manifest_row(manifest_path: Path, index: int) -> Dict[str, Any]:
    """Load a specific row from the manifest file.
    
    Args:
        manifest_path: Path to manifest CSV
        index: Row index (1-indexed)
        
    Returns:
        Dictionary of parameter overrides for this row
        
    Raises:
        ValueError: If index is out of range
    """
    try:
        df = pd.read_csv(manifest_path)
        
        if index < 1 or index > len(df):
            raise ValueError(
                f"Index {index} out of range [1, {len(df)}] for manifest {manifest_path}"
            )
        
        # Convert to 0-indexed
        row = df.iloc[index - 1]
        
        # Convert row to parameter override dictionary
        overrides = {}
        for col, value in row.items():
            if col != "run_id" and pd.notna(value):
                overrides[col] = value
                
        logger.info(
            "Loaded manifest row",
            manifest=str(manifest_path),
            index=index,
            run_id=row.get("run_id", f"run_{index:03d}"),
            overrides=overrides
        )
        
        return {
            "run_id": row.get("run_id", f"run_{index:03d}"),
            "overrides": overrides
        }
        
    except Exception as e:
        logger.error(f"Failed to load manifest row: {e}")
        raise


def apply_parameter_overrides(config_dict: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Apply dotted parameter overrides to config dictionary.
    
    Args:
        config_dict: Base configuration
        overrides: Parameter overrides with dotted keys (e.g., "pbbm.params.CL": 20.0)
        
    Returns:
        Updated configuration dictionary
    """
    result = config_dict.copy()
    
    for key, value in overrides.items():
        # Split dotted key path
        path = key.split(".")
        
        # Navigate to parent dictionary
        current = result
        for part in path[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # Set the final value
        current[path[-1]] = value
        
        logger.debug("Applied override", key=key, value=value)
    
    return result


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.verbose:
        # Enable debug logging
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="ISO"),
                structlog.processors.add_log_level,
                structlog.processors.JSONRenderer()
            ],
            wrapper_class=structlog.make_filtering_bound_logger(10),  # DEBUG level
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
    
    start_time = time.time()
    
    try:
        # Load base configuration
        config = app_api.load_config_from_file(args.config)
        logger.info("Loaded base configuration", config_file=str(args.config))
        
        # Load manifest row
        manifest_data = load_manifest_row(args.manifest, args.index)
        run_id = manifest_data["run_id"]
        overrides = manifest_data["overrides"]
        
        # Apply artifacts directory override
        if args.artifacts:
            config.run.artifact_dir = str(args.artifacts)
            logger.info("Artifacts directory override", artifacts=str(args.artifacts))
        
        # Run simulation
        logger.info("Starting simulation", run_id=run_id)
        
        result = app_api.run_single_simulation(
            config,
            parameter_overrides=overrides,
            run_id=run_id,
            artifact_directory=args.artifacts
        )
        
        runtime = time.time() - start_time
        
        # Calculate summary metrics
        metrics = app_api.calculate_summary_metrics(result)
        
        # Print structured summary for SLURM logs
        summary = {
            "status": "success",
            "run_id": run_id,
            "index": args.index,
            "runtime_seconds": runtime,
            "simulation_time_seconds": result.runtime_seconds,
            "metrics": metrics
        }
        
        print(json.dumps(summary))
        logger.info("Simulation completed successfully", **summary)
        
    except LMPError as e:
        runtime = time.time() - start_time
        error_summary = {
            "status": "error",
            "error_type": type(e).__name__,
            "error_message": e.message,
            "index": args.index,
            "runtime_seconds": runtime
        }
        
        print(json.dumps(error_summary))
        logger.error("Simulation failed", **error_summary)
        sys.exit(1)
        
    except Exception as e:
        runtime = time.time() - start_time
        error_summary = {
            "status": "error", 
            "error_type": type(e).__name__,
            "error_message": str(e),
            "index": args.index,
            "runtime_seconds": runtime
        }
        
        print(json.dumps(error_summary))
        logger.error("Unexpected error", **error_summary)
        sys.exit(1)


if __name__ == "__main__":
    main()