# Changelog

All notable changes to the LMP package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added - PR2: Entities + Catalog System
- **Entity Schemas**: Pydantic v2 models for Subject, API, Product, InhalationProfile
  - Comprehensive validation with field constraints
  - Computed properties (BMI, BSA, average flow rates)
  - Type safety with optional units support
- **Catalog System**: Extensible entity management
  - TOML/YAML/JSON file loaders with pluggable readers
  - Built-in catalog with 12 realistic entities (3 subjects, 3 APIs, 3 products, 3 maneuvers)
  - Search path hierarchy for user extensions
  - Validation and error handling
- **Configuration Hydration**: Resolve entity references with overrides
  - EntityRef system supporting both catalog references and inline overrides
  - Cross-entity validation with helpful warnings
  - Entity summary generation for logging/UI
- **Physiology Functions**: Pure calculation functions for respiratory modeling
  - Lung volume calculations with age/sex scaling
  - Airway dimension modeling (Weibel model)
  - Particle deposition fractions (ICRP 66)
  - Clearance rate calculations
  - Flow profile generation
- **Enhanced App API**: Real entity resolution and validation
- **Updated CLI**: Now shows resolved entity information during runs
- **Comprehensive Testing**: 41 new unit tests covering entities, catalog, and hydration

### Added - PR1: Foundation
- Initial package structure with modular architecture
- Core contracts: Stage protocol, typed data structures, error hierarchy
- Configuration system with TOML support and environment variable overrides
- App API facade for Dash and SLURM integration
- CLI with run, validate, list-models, list-catalog, plan, and info commands
- SLURM array job runner script
- Development tooling: ruff, black, mypy, pytest configuration
- Pre-commit hooks configuration

### Technical Details
- Python 3.9+ support (tested on 3.9, designed for 3.10+)
- Pydantic v2 for configuration validation and entity schemas
- Typer + Rich for CLI
- Structured logging with configurable JSON output
- Parameter sweep planning with manifest generation
- Environment variable configuration overrides
- Extensible catalog system with built-in entities

### Architecture
- Contracts layer with stable interfaces
- Entity-driven configuration with catalog resolution
- App API facade used by both CLI and Dash
- Modular stage-based pipeline design
- Pluggable model registry (ready for PR3)
- Catalog-based entity management with validation

## [0.1.0] - 2025-01-XX

### Added
- Initial release with foundation components
- Basic CLI functionality  
- Configuration system
- Testing infrastructure
- Documentation structure

### Known Limitations
- Models are placeholder implementations
- Catalog uses hardcoded data
- No actual simulation execution yet
- No Dash integration implemented
- No artifact storage implemented

### Next Steps (PR2)
- Implement catalog system with TOML/YAML loaders
- Add Pydantic entity schemas (Subject, API, Product, InhalationProfile)
- Implement configuration hydration with catalog resolution