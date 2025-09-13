"""Catalog file loaders for different formats."""

from __future__ import annotations
from typing import Dict, Any
from pathlib import Path

from .base import CatalogReader


class TomlReader(CatalogReader):
    """Reader for TOML catalog files."""
    
    def can_read(self, path: Path) -> bool:
        """Check if file has .toml extension."""
        return path.suffix.lower() == ".toml"
    
    def read(self, path: Path) -> Dict[str, Any]:
        """Read TOML file."""
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
            
        with open(path, "rb") as f:
            return tomllib.load(f)


class YamlReader(CatalogReader):
    """Reader for YAML catalog files."""
    
    def can_read(self, path: Path) -> bool:
        """Check if file has .yml or .yaml extension.""" 
        return path.suffix.lower() in {".yml", ".yaml"}
    
    def read(self, path: Path) -> Dict[str, Any]:
        """Read YAML file."""
        try:
            import yaml
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        except ImportError:
            raise ImportError("PyYAML required for YAML catalog support")


class JsonReader(CatalogReader):
    """Reader for JSON catalog files."""
    
    def can_read(self, path: Path) -> bool:
        """Check if file has .json extension."""
        return path.suffix.lower() == ".json"
    
    def read(self, path: Path) -> Dict[str, Any]:
        """Read JSON file."""
        import json
        with open(path, "r") as f:
            return json.load(f)