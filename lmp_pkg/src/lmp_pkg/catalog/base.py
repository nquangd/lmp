"""Base catalog classes and interfaces."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, TYPE_CHECKING
from pathlib import Path

from ..domain.entities import Subject, API, Product, InhalationManeuver, BaseModel

if TYPE_CHECKING:
    from ..domain.entities import Subject


class CatalogError(Exception):
    """Exception raised by catalog operations."""
    pass


class CatalogReader(ABC):
    """Abstract base for catalog data readers."""
    
    @abstractmethod
    def can_read(self, path: Path) -> bool:
        """Check if this reader can handle the given file."""
        pass
    
    @abstractmethod
    def read(self, path: Path) -> Dict[str, Any]:
        """Read catalog data from file.""" 
        pass


class ReadOnlyCatalog:
    """Read-only catalog for entity lookup and listing.
    
    The catalog loads entity definitions from data files and provides
    a consistent interface for entity resolution. It supports:
    - Multiple file formats via pluggable readers
    - Search path hierarchies for extensibility  
    - Validation against entity schemas
    """
    
    def __init__(self, search_paths: List[Union[str, Path]], readers: Optional[List[CatalogReader]] = None):
        """Initialize catalog with search paths.
        
        Args:
            search_paths: Directories to search for catalog files
            readers: Optional list of file readers (defaults to built-in readers)
        """
        from .loaders import TomlReader, YamlReader
        self.search_paths = [Path(p) for p in search_paths]
        self.readers = readers or [TomlReader(), YamlReader()]
        self._cache: Dict[str, Dict[str, BaseModel]] = {}
        
        # Entity type mapping
        self._entity_types = {
            "subject": Subject,
            "api": API, 
            "product": Product,
            "maneuver": InhalationManeuver
        }
        
        self._load_all()
    
    def _load_all(self) -> None:
        """Load all catalog entries from search paths."""
        self._cache = {category: {} for category in self._entity_types.keys()}
        
        # Allow directory aliasing for categories (e.g., 'maneuver' <- 'inhalation')
        category_dir_aliases: Dict[str, List[str]] = {
            "maneuver": ["maneuver", "inhalation"],
        }
        
        for search_path in self.search_paths:
            if not search_path.exists():
                continue
                
            # Load each category directory (including aliases)
            for category in self._entity_types.keys():
                dir_names = category_dir_aliases.get(category, [category])
                for dir_name in dir_names:
                    category_path = search_path / dir_name
                    if category_path.exists() and category_path.is_dir():
                        self._load_category(category, category_path)
    
    def _load_category(self, category: str, category_path: Path) -> None:
        """Load all files in a category directory."""
        entity_cls = self._entity_types[category]
        
        for file_path in category_path.iterdir():
            if file_path.is_file():
                # Find compatible reader
                reader = self._find_reader(file_path)
                if reader:
                    try:
                        data = reader.read(file_path)
                        
                        # Handle both single entities and collections
                        if isinstance(data, dict) and "entries" in data:
                            # Collection format: {"entries": {"name1": {...}, "name2": {...}}}
                            for name, entity_data in data["entries"].items():
                                entity = entity_cls.model_validate(entity_data)
                                self._cache[category][name] = entity
                        else:
                            # Single entity format: use filename (without extension) as name
                            name = file_path.stem
                            entity = entity_cls.model_validate(data)
                            self._cache[category][name] = entity
                            
                    except Exception as e:
                        # Log warning but continue loading
                        import warnings
                        warnings.warn(f"Failed to load {file_path}: {e}")
    
    def _find_reader(self, path: Path) -> Optional[CatalogReader]:
        """Find a compatible reader for the given file."""
        for reader in self.readers:
            if reader.can_read(path):
                return reader
        return None
    
    def list_categories(self) -> List[str]:
        """List all available entity categories."""
        return list(self._entity_types.keys())
    
    def list_entries(self, category: str) -> List[str]:
        """List all entries in a category.
        
        Args:
            category: Entity category name
            
        Returns:
            List of entity names in the category
            
        Raises:
            ValueError: If category is not recognized
        """
        if category not in self._entity_types:
            raise ValueError(f"Unknown category: {category}")
            
        return list(self._cache[category].keys())
    
    def get_entry(self, category: str, name: str) -> BaseModel:
        """Get a specific catalog entry.
        
        Args:
            category: Entity category name
            name: Entity name
            
        Returns:
            Entity instance
            
        Raises:
            ValueError: If category or entry name is not found
        """
        if category not in self._entity_types:
            raise ValueError(f"Unknown category: {category}")
            
        if name not in self._cache[category]:
            raise ValueError(f"Entry '{name}' not found in category '{category}'")
            
        return self._cache[category][name]
    
    def has_entry(self, category: str, name: str) -> bool:
        """Check if an entry exists.
        
        Args:
            category: Entity category name
            name: Entity name
            
        Returns:
            True if entry exists, False otherwise
        """
        return (category in self._cache and 
                name in self._cache[category])
    
    def get_stats(self) -> Dict[str, int]:
        """Get catalog statistics.
        
        Returns:
            Dictionary mapping category names to entry counts
        """
        return {category: len(entities) 
                for category, entities in self._cache.items()}
    
    def validate_all(self) -> Dict[str, List[str]]:
        """Validate all catalog entries.
        
        Returns:
            Dictionary mapping categories to lists of validation errors
        """
        errors = {}
        
        for category, entities in self._cache.items():
            category_errors = []
            entity_cls = self._entity_types[category]
            
            for name, entity in entities.items():
                try:
                    # Re-validate the entity
                    entity_cls.model_validate(entity.model_dump())
                except Exception as e:
                    category_errors.append(f"{name}: {e}")
            
            if category_errors:
                errors[category] = category_errors
                
        return errors


def get_default_catalog() -> ReadOnlyCatalog:
    """Get default catalog with built-in search paths."""
    import importlib.util
    
    # Find built-in catalog path
    builtin_path = Path(__file__).parent / "builtin"
    search_paths = [builtin_path]
    
    # Add user catalog paths from entry points
    try:
        import pkg_resources
        for entry_point in pkg_resources.iter_entry_points("lmp_pkg.catalog_paths"):
            try:
                catalog_path = entry_point.load()
                if isinstance(catalog_path, (str, Path)):
                    search_paths.append(Path(catalog_path))
            except Exception:
                pass  # Skip invalid entry points
    except ImportError:
        pass  # pkg_resources not available
    
    # Add user home directory
    user_catalog = Path.home() / ".lmp" / "catalog"
    if user_catalog.exists():
        search_paths.append(user_catalog)
    
    return ReadOnlyCatalog(search_paths)
