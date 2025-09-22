# LMP Package

Lung Modeling Platform - A modular PBPK simulation pipeline for respiratory drug delivery modeling.

## Features

- **Modular pipeline**: Independently runnable stages (deposition → lung PBPK → systemic PK → analysis)
- **Model plug-ins**: Easy addition/replacement of models without engine changes
- **Catalog-driven**: Subject, API, Product, and Inhalation profiles configurable via data files
- **Dual execution**: Scripts/SLURM and Dash both use the same App API
- **Reproducible**: Typed, tested, with deterministic seeds

## Quick Start

```bash
pip install -e .

# Run a basic simulation
lmp run --config examples/basic.toml

# List available models
lmp list-models

# Plan a parameter sweep
lmp plan --config sweep.toml --output manifest.csv
```

## Development

```bash
pip install -e ".[dev]"
pre-commit install
pytest
```

## Architecture

```
lmp_pkg/
├── contracts/     # Stage protocols and types
├── config/        # Configuration and validation
├── domain/        # Entities and physiology
├── catalog/       # Default parameter data
├── engine/        # Pipeline execution
├── models/        # Pluggable model implementations
└── services/      # High-level operations
```

See `docs/` for detailed guides.