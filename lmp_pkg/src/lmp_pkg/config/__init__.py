"""Configuration system with lazy attribute loading to avoid circular imports."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Tuple

from . import constants as _constants
from .constants import *  # noqa: F401,F403 - re-export constants

constants = _constants

_LAZY_ATTRS: Dict[str, Tuple[str, str]] = {
    # Core configuration models
    "AppConfig": (".model", "AppConfig"),
    "RunConfig": (".model", "RunConfig"),
    "DepositionConfig": (".model", "DepositionConfig"),
    "PBBMConfig": (".model", "PBBMConfig"),
    "PKConfig": (".model", "PKConfig"),
    "AnalysisConfig": (".model", "AnalysisConfig"),
    "SolverConfig": (".model", "SolverConfig"),
    "EntityRef": (".model", "EntityRef"),
    "PopulationVariabilityConfig": (".model", "PopulationVariabilityConfig"),
    "StudyConfig": (".model", "StudyConfig"),
    # Configuration helpers
    "load_config": (".load", "load_config"),
    "default_config": (".load", "default_config"),
    "validate_config": (".validation", "validate_config"),
    "hydrate_config": (".hydration", "hydrate_config"),
    "validate_hydrated_entities": (".hydration", "validate_hydrated_entities"),
    "get_entity_summary": (".hydration", "get_entity_summary"),
    "check_catalog_coverage": (".hydration", "check_catalog_coverage"),
}

_CONSTANT_NAMES = [name for name in dir(_constants) if not name.startswith("_")]
__all__ = sorted(set(["constants", *_CONSTANT_NAMES, *_LAZY_ATTRS.keys()]))


def __getattr__(name: str) -> Any:
    """Lazily load configuration helpers and models on first access."""

    try:
        module_name, attr_name = _LAZY_ATTRS[name]
    except KeyError as exc:  # pragma: no cover - defensive path
        raise AttributeError(f"module {__name__} has no attribute {name}") from exc

    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose constants and lazily loaded attributes to dir()."""

    return list(__all__)
