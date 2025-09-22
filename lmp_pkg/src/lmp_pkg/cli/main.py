"""Main CLI application."""

from pathlib import Path
from typing import List, Optional, Dict, Any
import json

import typer
from rich.console import Console
from rich.table import Table
from rich import print as rprint
import pandas as pd

from .. import app_api
from ..engine.workflow import list_workflows as _list_workflows
from ..contracts.errors import LMPError
from ..config.model import PopulationVariabilityConfig

app = typer.Typer(
    name="lmp",
    help="Lung Modeling Platform - Modular PBPK simulation pipeline",
    no_args_is_help=True
)
console = Console()


@app.command()
def run(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Configuration file path"
    ),
    stages: Optional[str] = typer.Option(
        None, "--stages", help="Comma-separated list of stages to run"
    ),
    run_id: Optional[str] = typer.Option(
        None, "--run-id", help="Custom run identifier"
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output directory for artifacts"
    ),
    overrides: Optional[str] = typer.Option(
        None, "--set", help="Parameter overrides as JSON string"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Validate configuration without running"
    ),
    workflow: Optional[str] = typer.Option(
        None, "--workflow", help="Workflow name (e.g., deposition_pbbm_pk)"
    ),
    pv_demographic: Optional[bool] = typer.Option(
        None,
        "--pv-demographic/--no-pv-demographic",
        help="Toggle demographic variability",
    ),
    pv_lung_regional: Optional[bool] = typer.Option(
        None,
        "--pv-lung-regional/--no-pv-lung-regional",
        help="Toggle lung regional variability",
    ),
    pv_lung_generation: Optional[bool] = typer.Option(
        None,
        "--pv-lung-generation/--no-pv-lung-generation",
        help="Toggle lung generation variability",
    ),
    pv_gi: Optional[bool] = typer.Option(
        None,
        "--pv-gi/--no-pv-gi",
        help="Toggle GI variability",
    ),
    pv_pk: Optional[bool] = typer.Option(
        None,
        "--pv-pk/--no-pv-pk",
        help="Toggle PK variability",
    ),
    pv_inhalation: Optional[bool] = typer.Option(
        None,
        "--pv-inhalation/--no-pv-inhalation",
        help="Toggle inhalation variability",
    ),
):
    """Run a simulation with the given configuration."""
    
    try:
        # Load configuration
        if config:
            cfg = app_api.load_config_from_file(config)
            console.print(f"✓ Loaded configuration from {config}")
        else:
            cfg = app_api.get_default_config()
            console.print("✓ Using default configuration")
        
        # Apply command line overrides
        if stages:
            stage_list = [s.strip() for s in stages.split(",")]
            cfg.run.stages = stage_list
            console.print(f"✓ Stages override: {stage_list}")
            
        if output_dir:
            cfg.run.artifact_dir = str(output_dir)
            console.print(f"✓ Output directory: {output_dir}")
            
        # Apply parameter overrides
        param_overrides = None
        if overrides:
            try:
                param_overrides = json.loads(overrides)
                console.print(f"✓ Parameter overrides: {param_overrides}")
            except json.JSONDecodeError as e:
                console.print(f"❌ Invalid JSON in --set: {e}", style="red")
                raise typer.Exit(1)

        pv_options = {
            "demographic": pv_demographic,
            "lung_regional": pv_lung_regional,
            "lung_generation": pv_lung_generation,
            "gi": pv_gi,
            "pk": pv_pk,
            "inhalation": pv_inhalation,
        }
        pv_overrides = {key: value for key, value in pv_options.items() if value is not None}
        if pv_overrides:
            if cfg.population_variability is None:
                cfg.population_variability = PopulationVariabilityConfig(**pv_overrides)
            else:
                cfg.population_variability = cfg.population_variability.model_copy(update=pv_overrides)
            console.print(f"✓ Population variability overrides: {pv_overrides}")

        # Validate configuration
        app_api.validate_configuration(cfg)
        console.print("✓ Configuration validated")
        
        if dry_run:
            console.print("✓ Dry run completed successfully", style="green")
            return
            
        # Run simulation
        with console.status("Running simulation..."):
            result = app_api.run_single_simulation(
                cfg,
                parameter_overrides=param_overrides,
                run_id=run_id,
                artifact_directory=output_dir,
                workflow_name=workflow,
            )
        
        # Display results summary
        console.print(f"✅ Simulation completed: {result.run_id}", style="green")
        console.print(f"Runtime: {result.runtime_seconds:.2f}s")
        
        # Show summary metrics
        metrics = app_api.calculate_summary_metrics(result)
        if metrics:
            table = Table(title="Summary Metrics")
            table.add_column("Metric")
            table.add_column("Value")
            
            for metric, value in metrics.items():
                table.add_row(metric, f"{value:.4g}")
            
            console.print(table)
            
    except LMPError as e:
        console.print(f"❌ {e.message}", style="red")
        if e.details:
            console.print(f"Details: {e.details}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"❌ Unexpected error: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def validate(
    config: Path = typer.Argument(..., help="Configuration file to validate")
):
    """Validate a configuration file."""
    
    try:
        cfg = app_api.load_config_from_file(config)
        app_api.validate_configuration(cfg)
        console.print(f"✅ Configuration {config} is valid", style="green")
        
    except LMPError as e:
        console.print(f"❌ {e.message}", style="red") 
        raise typer.Exit(1)


@app.command("list-models")
def list_models():
    """List all available models by category."""
    
    try:
        models = app_api.list_available_models()
        
        for category, model_list in models.items():
            table = Table(title=f"{category.upper()} Models")
            table.add_column("Model Name")
            table.add_column("Description")
            
            for model in model_list:
                # TODO: Get actual descriptions from model registry
                desc = "Model implementation" if model != "null" else "No-op placeholder"
                table.add_row(model, desc)
            
            console.print(table)
            console.print()
            
    except Exception as e:
        console.print(f"❌ Error listing models: {e}", style="red")
        raise typer.Exit(1)


@app.command("list-catalog")
def list_catalog(
    category: str = typer.Argument(..., help="Catalog category (subject, api, product, maneuver)")
):
    """List available catalog entries."""
    
    try:
        entries = app_api.list_catalog_entries(category)
        
        table = Table(title=f"Catalog: {category.upper()}")
        table.add_column("Name")
        table.add_column("Description") 
        
        for entry in entries:
            # TODO: Get actual descriptions from catalog
            desc = f"Default {category} parameters"
            table.add_row(entry, desc)
        
        console.print(table)
        
    except ValueError as e:
        console.print(f"❌ {e}", style="red")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"❌ Error listing catalog: {e}", style="red")
        raise typer.Exit(1)


@app.command("list-workflows")
def list_workflows_cli():
    """List available preconfigured workflows."""

    names = _list_workflows()
    if not names:
        console.print("No workflows registered.")
        return

    table = Table(title="Workflows")
    table.add_column("Name")

    for name in names:
        table.add_row(name)

    console.print(table)


@app.command()
def plan(
    config: Path = typer.Argument(..., help="Base configuration file"),
    axes: Optional[str] = typer.Option(
        None, "--axes", help="Parameter axes as JSON (e.g., '{\"pk.model\": [\"pk_1c\", \"pk_2c\"]}')"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output manifest CSV file"
    )
):
    """Generate a simulation manifest for parameter sweeps."""
    
    try:
        # Load base configuration
        cfg = app_api.load_config_from_file(config)
        console.print(f"✓ Loaded base configuration from {config}")
        
        # Parse parameter axes
        parameter_axes: Dict[str, Any] = {}
        if axes:
            try:
                parameter_axes = json.loads(axes)
                console.print(f"✓ Parameter axes: {parameter_axes}")
            except json.JSONDecodeError as e:
                console.print(f"❌ Invalid JSON in --axes: {e}", style="red")
                raise typer.Exit(1)
        
        # Generate manifest
        manifest = app_api.plan_simulation_manifest(cfg, parameter_axes)
        console.print(f"✓ Generated manifest with {len(manifest)} runs")
        
        # Save or display
        if output:
            manifest.to_csv(output, index=False)
            console.print(f"✓ Manifest saved to {output}")
        else:
            console.print("\nManifest Preview:")
            console.print(manifest.head(10).to_string(index=False))
            if len(manifest) > 10:
                console.print(f"... ({len(manifest) - 10} more rows)")
                
    except LMPError as e:
        console.print(f"❌ {e.message}", style="red")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"❌ Error generating manifest: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def info():
    """Display package information and diagnostics."""
    
    from .. import __version__
    
    console.print(f"LMP Package v{__version__}")
    console.print()
    
    # Show available models
    try:
        models = app_api.list_available_models()
        total_models = sum(len(models[cat]) for cat in models)
        console.print(f"Available models: {total_models}")
        for cat, model_list in models.items():
            console.print(f"  {cat}: {len(model_list)} models")
    except Exception:
        console.print("Could not load model information")
    
    console.print()
    
    # Show catalog status
    try:
        for category in ["subject", "api", "product", "maneuver"]:
            entries = app_api.list_catalog_entries(category)
            console.print(f"Catalog {category}: {len(entries)} entries")
    except Exception:
        console.print("Could not load catalog information")


if __name__ == "__main__":
    app()
