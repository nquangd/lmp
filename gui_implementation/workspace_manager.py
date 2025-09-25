"""Workspace Manager for LMP GUI.

Handles workspace directory structure, run management, and file organization
according to the GUI plan requirements.
"""

import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import pandas as pd
import structlog


def _normalise_label(label: Optional[str]) -> Optional[str]:
    """Return a filesystem-friendly representation of the provided label."""

    if not label:
        return None

    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in label.strip())
    safe = safe.strip("-")
    return safe or None


def _slugify(name: str) -> str:
    """Provide a lowercase slug suitable for filenames."""

    safe = "".join(
        ch.lower() if ch.isalnum() else "-"
        for ch in name.strip()
    )
    # Collapse consecutive dashes
    parts = [chunk for chunk in safe.split("-") if chunk]
    slug = "-".join(parts)
    return slug or "entry"


_CATALOG_CATEGORIES = (
    "subject",
    "api",
    "product",
    "maneuver",
    "lung_geometry",
    "gi_tract",
)

logger = structlog.get_logger()


class WorkspaceManager:
    """Manages LMP workspace directory structure and run metadata."""

    def __init__(self, workspace_path: Union[str, Path]):
        """Initialize workspace manager.

        Args:
            workspace_path: Path to workspace directory
        """
        self.workspace_path = Path(workspace_path).resolve()
        self._ensure_workspace_structure()

    def _ensure_workspace_structure(self):
        """Create workspace directory structure if it doesn't exist."""
        required_dirs = [
            self.workspace_path,
            self.workspace_path / "configs",
            self.workspace_path / "runs",
            self.workspace_path / "logs",
            self.catalog_dir,
        ]

        for dir_path in required_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Ensure category subdirectories exist for saved catalog entries
        for category in _CATALOG_CATEGORIES:
            (self.catalog_dir / category).mkdir(parents=True, exist_ok=True)

        # Create workspace metadata if it doesn't exist
        workspace_info_file = self.workspace_path / "workspace.json"
        if not workspace_info_file.exists():
            workspace_info = {
                "created": datetime.now().isoformat(),
                "lmp_version": "0.1.0",  # TODO: Get from package
                "description": "LMP GUI Workspace"
            }
            with open(workspace_info_file, 'w') as f:
                json.dump(workspace_info, f, indent=2)

    @property
    def configs_dir(self) -> Path:
        """Get configs directory path."""
        return self.workspace_path / "configs"

    @property
    def runs_dir(self) -> Path:
        """Get runs directory path."""
        return self.workspace_path / "runs"

    @property
    def logs_dir(self) -> Path:
        """Get logs directory path."""
        return self.workspace_path / "logs"

    @property
    def catalog_dir(self) -> Path:
        """Directory for workspace-local catalog overrides."""
        return self.workspace_path / "catalog"

    def catalog_category_dir(self, category: str) -> Path:
        """Resolve category directory under workspace catalog storage."""
        if category not in _CATALOG_CATEGORIES:
            raise ValueError(f"Unsupported catalog category: {category}")
        path = self.catalog_dir / category
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ------------------------------------------------------------------
    # Workspace catalog helpers

    def save_catalog_entry(
        self,
        category: str,
        payload: Dict[str, Any],
        *,
        entry_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Persist a workspace catalog entry and return stored metadata.

        Args:
            category: Catalog category name (subject/api/product/maneuver)
            payload: Entry payload including at minimum a human-friendly name
            entry_id: Optional existing identifier; overwrites when provided

        Returns:
            Dictionary containing stored payload with metadata (id, path)
        """

        category_dir = self.catalog_category_dir(category)

        base_name = payload.get("name") or payload.get("ref") or "entry"
        slug = entry_id or payload.get("id") or _slugify(str(base_name))
        file_path = category_dir / f"{slug}.json"

        existing_created: Optional[str] = None
        if file_path.exists():
            try:
                with open(file_path, "r") as fh:
                    existing_payload = json.load(fh)
                existing_created = existing_payload.get("created")
            except Exception:
                existing_created = None

        now_iso = datetime.now().isoformat()
        stored_payload = {
            **payload,
            "id": slug,
            "category": category,
            "updated": now_iso,
        }
        if existing_created:
            stored_payload.setdefault("created", existing_created)
        else:
            stored_payload.setdefault("created", now_iso)

        with open(file_path, "w") as fh:
            json.dump(stored_payload, fh, indent=2)

        stored_payload["path"] = str(file_path)
        return stored_payload

    def load_catalog_entry(self, category: str, entry_id: str) -> Dict[str, Any]:
        """Load a previously stored catalog entry."""

        category_dir = self.catalog_category_dir(category)
        file_path = category_dir / f"{entry_id}.json"
        if not file_path.exists():
            raise FileNotFoundError(f"Catalog entry not found: {category}/{entry_id}")

        with open(file_path, "r") as fh:
            data = json.load(fh)
        data.setdefault("id", entry_id)
        data.setdefault("category", category)
        data["path"] = str(file_path)
        return data

    def list_catalog_entries(self, category: str) -> List[Dict[str, Any]]:
        """List catalog entries saved in this workspace for a category."""

        category_dir = self.catalog_category_dir(category)
        entries: List[Dict[str, Any]] = []
        for file_path in category_dir.glob("*.json"):
            try:
                with open(file_path, "r") as fh:
                    data = json.load(fh)
                data.setdefault("id", file_path.stem)
                data.setdefault("category", category)
                data.setdefault("name", data.get("display_name") or data.get("ref") or file_path.stem)
                data["path"] = str(file_path)
                entries.append(data)
            except Exception as exc:
                logger.warning(
                    "Could not read catalog entry",
                    category=category,
                    path=str(file_path),
                    error=str(exc),
                )

        entries.sort(key=lambda item: item.get("name", ""))
        return entries

    def delete_catalog_entry(self, category: str, entry_id: str) -> None:
        """Delete a workspace catalog entry if it exists."""

        category_dir = self.catalog_category_dir(category)
        file_path = category_dir / f"{entry_id}.json"
        if file_path.exists():
            file_path.unlink()

    def generate_run_id(self, prefix: Optional[str] = None, label: Optional[str] = None) -> str:
        """Generate a unique run ID with an optional prefix and label slug."""

        prefix = prefix or "run"
        components = [prefix]

        label_slug = _normalise_label(label)
        if label_slug:
            components.append(label_slug)

        components.append(uuid.uuid4().hex[:8])
        return "_".join(components)

    def generate_config_name(self, base_name: str = "config") -> str:
        """Generate a unique config filename."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base_name}_{timestamp}.json"

    def save_config(
        self,
        config_data: Dict[str, Any],
        name: Optional[str] = None,
        run_plan: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save configuration to configs directory.

        Args:
            config_data: Configuration dictionary
            name: Optional config filename (auto-generated if not provided)

        Returns:
            Path to saved config file
        """
        if name is None:
            name = self.generate_config_name()
        elif not name.endswith('.json'):
            name += '.json'

        config_path = self.configs_dir / name

        # Add metadata
        metadata_payload = {
            "created": datetime.now().isoformat(),
            "workspace": str(self.workspace_path),
            "config_name": name
        }
        if run_plan is not None:
            metadata_payload["run_plan"] = run_plan

        config_with_meta = {
            "metadata": metadata_payload,
            "config": config_data
        }

        with open(config_path, 'w') as f:
            json.dump(config_with_meta, f, indent=2)

        logger.info("Config saved", path=str(config_path))
        return config_path

    def load_config(self, name: str) -> Dict[str, Any]:
        """Load configuration from configs directory.

        Args:
            name: Config filename

        Returns:
            Configuration dictionary
        """
        config_path = self.configs_dir / name

        if config_path.suffix == "":
            # Prefer JSON default when no suffix is provided
            json_path = config_path.with_suffix('.json')
            toml_path = config_path.with_suffix('.toml')
            if json_path.exists():
                config_path = json_path
            elif toml_path.exists():
                config_path = toml_path
            else:
                config_path = json_path  # fall back for error path

        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        if config_path.suffix.lower() == '.toml':
            try:
                from lmp_pkg import app_api  # Local import to avoid path issues during module import
                cfg = app_api.load_config_from_file(config_path)
                return cfg.model_dump()
            except Exception as e:
                raise RuntimeError(f"Could not load TOML config {config_path}: {e}") from e

        with open(config_path, 'r') as f:
            config_data = json.load(f)

        if "config" in config_data and "metadata" in config_data:
            return config_data["config"]
        return config_data

    def list_configs(self) -> List[Dict[str, Any]]:
        """List all configurations in workspace.

        Returns:
            List of config info dictionaries
        """
        configs = []

        for config_file in self.configs_dir.iterdir():
            if config_file.suffix.lower() not in {'.json', '.toml'}:
                continue
            try:
                created = datetime.fromtimestamp(config_file.stat().st_mtime).isoformat()
                size_bytes = config_file.stat().st_size

                metadata: Dict[str, Any] = {}

                if config_file.suffix.lower() == '.json':
                    with open(config_file, 'r') as f:
                        config_data = json.load(f)

                    metadata = config_data.get("metadata", {}) if isinstance(config_data, dict) else {}
                    created = metadata.get("created", created)
                else:
                    metadata = {}

                config_name = metadata.get("config_name") if isinstance(metadata, dict) else None
                if not config_name:
                    config_name = config_file.name
                    if isinstance(metadata, dict):
                        metadata["config_name"] = config_name

                entry = {
                    "name": config_file.name,
                    "path": str(config_file),
                    "created": created,
                    "size_bytes": size_bytes
                }
                run_plan = metadata.get("run_plan") if isinstance(metadata, dict) else None
                if isinstance(run_plan, dict):
                    run_plan = dict(run_plan)
                    run_plan.setdefault("config_name", config_name)
                    metadata["run_plan"] = run_plan
                if run_plan is not None:
                    entry["run_plan"] = run_plan
                if metadata:
                    entry["metadata"] = metadata

                configs.append(entry)

            except Exception as e:
                logger.warning("Could not read config", path=str(config_file), error=str(e))

        return sorted(configs, key=lambda x: x["created"], reverse=True)

    def create_run_directory(
        self,
        run_id: str,
        *,
        run_type: str = "single",
        label: Optional[str] = None,
        parent_run_id: Optional[str] = None,
        request_metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Create directory structure for a new run.

        Args:
            run_id: Unique run identifier
            run_type: Run category string
            label: Optional user-facing label
            parent_run_id: Optional grouping identifier for multi-run studies
            request_metadata: Additional metadata captured during submission

        Returns:
            Path to run directory
        """
        run_dir = self.runs_dir / run_id

        # Create subdirectories
        subdirs = ["logs", "results", "artifacts"]
        for subdir in subdirs:
            (run_dir / subdir).mkdir(parents=True, exist_ok=True)

        # Create run metadata
        normalised_label = _normalise_label(label)

        run_metadata: Dict[str, Any] = {
            "run_id": run_id,
            "run_type": run_type,
            "label": normalised_label,
            "display_label": label,
            "parent_run_id": parent_run_id,
            "created": datetime.now().isoformat(),
            "status": "created",
            "workspace": str(self.workspace_path)
        }

        if request_metadata:
            for key, value in request_metadata.items():
                if value is not None:
                    run_metadata[key] = value

        metadata_path = run_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(run_metadata, f, indent=2)

        logger.info("Run directory created", run_id=run_id, path=str(run_dir))
        return run_dir

    def update_run_status(self, run_id: str, status: str, **kwargs):
        """Update run status and metadata.

        Args:
            run_id: Run identifier
            status: New status (created, running, completed, failed, cancelled)
            **kwargs: Additional metadata to update
        """
        run_dir = self.runs_dir / run_id
        metadata_path = run_dir / "metadata.json"

        if not metadata_path.exists():
            logger.warning("Run metadata not found", run_id=run_id)
            return

        # Load current metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Update status and timestamp
        metadata["status"] = status
        metadata["last_updated"] = datetime.now().isoformat()

        display_label = kwargs.pop("display_label", None)
        label_value = kwargs.get("label")
        if label_value is not None:
            kwargs["label"] = _normalise_label(label_value)

        metadata.update(kwargs)

        if display_label is not None:
            metadata["display_label"] = display_label
            if not metadata.get("label"):
                metadata["label"] = _normalise_label(display_label)

        # Save updated metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info("Run status updated", run_id=run_id, status=status)

    def list_runs(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all runs in workspace.

        Args:
            status_filter: Optional status to filter by

        Returns:
            List of run info dictionaries
        """
        runs = []

        for run_dir in self.runs_dir.iterdir():
            if not run_dir.is_dir():
                continue

            metadata_path = run_dir / "metadata.json"
            if not metadata_path.exists():
                continue

            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                # Apply status filter
                if status_filter and metadata.get("status") != status_filter:
                    continue

                # Add computed fields
                metadata["run_dir"] = str(run_dir)
                metadata["has_results"] = (run_dir / "results").exists()
                metadata["has_logs"] = any((run_dir / "logs").glob("*.log"))

                runs.append(metadata)

            except Exception as e:
                logger.warning("Could not read run metadata",
                             run_dir=str(run_dir), error=str(e))

        return sorted(runs, key=lambda x: x.get("created", ""), reverse=True)

    def get_run_info(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific run.

        Args:
            run_id: Run identifier

        Returns:
            Run info dictionary or None if not found
        """
        run_dir = self.runs_dir / run_id
        metadata_path = run_dir / "metadata.json"

        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except json.JSONDecodeError:
            logger.warning("Run metadata invalid", run_id=run_id, path=str(metadata_path))
            metadata = {"run_id": run_id, "status": "unknown"}

        # Add computed information
        results_dir = run_dir / "results"
        logs_dir = run_dir / "logs"

        metadata.update({
            "run_dir": str(run_dir),
            "results_dir": str(results_dir),
            "logs_dir": str(logs_dir),
            "has_results": results_dir.exists(),
            "result_files": list(results_dir.glob("*.parquet")) if results_dir.exists() else [],
            "log_files": list(logs_dir.glob("*.log")) if logs_dir.exists() else [],
            "config_file": str(run_dir / "config.json") if (run_dir / "config.json").exists() else None
        })

        return metadata

    def load_run_results(self, run_id: str) -> Dict[str, pd.DataFrame]:
        """Load result DataFrames for a run.

        Args:
            run_id: Run identifier

        Returns:
            Dictionary mapping result names to DataFrames
        """
        run_dir = self.runs_dir / run_id
        results_dir = run_dir / "results"

        if not results_dir.exists():
            raise FileNotFoundError(f"Results directory not found for run {run_id}")

        dataframes = {}

        # Prefer parquet outputs when available
        for parquet_file in results_dir.glob("*.parquet"):
            result_name = parquet_file.stem
            try:
                dataframes[result_name] = pd.read_parquet(parquet_file)
            except Exception as e:
                logger.warning(
                    "Could not load parquet result",
                    file=str(parquet_file),
                    error=str(e)
                )

        # Fallback to CSV files for runs generated without parquet support
        for csv_file in results_dir.glob("*.csv"):
            result_name = csv_file.stem
            if result_name in dataframes:
                continue
            try:
                dataframes[result_name] = pd.read_csv(csv_file)
            except Exception as e:
                logger.warning(
                    "Could not load CSV result",
                    file=str(csv_file),
                    error=str(e)
                )

        return dataframes

    def save_manifest(self, manifest_df: pd.DataFrame, name: str) -> Path:
        """Save a simulation manifest to the workspace.

        Args:
            manifest_df: Manifest DataFrame
            name: Manifest name

        Returns:
            Path to saved manifest file
        """
        if not name.endswith('.parquet'):
            name += '.parquet'

        manifest_path = self.workspace_path / "manifests" / name
        manifest_path.parent.mkdir(exist_ok=True)

        manifest_df.to_parquet(manifest_path, index=False)

        logger.info("Manifest saved", path=str(manifest_path), runs=len(manifest_df))
        return manifest_path

    def cleanup_run(self, run_id: str, keep_results: bool = True):
        """Clean up run directory, optionally keeping results.

        Args:
            run_id: Run identifier
            keep_results: Whether to keep result files
        """
        run_dir = self.runs_dir / run_id

        if not run_dir.exists():
            logger.warning("Run directory not found", run_id=run_id)
            return

        # Remove log files
        logs_dir = run_dir / "logs"
        if logs_dir.exists():
            for log_file in logs_dir.glob("*"):
                log_file.unlink()

        # Remove artifacts
        artifacts_dir = run_dir / "artifacts"
        if artifacts_dir.exists():
            for artifact_file in artifacts_dir.glob("*"):
                artifact_file.unlink()

        if not keep_results:
            # Remove results
            results_dir = run_dir / "results"
            if results_dir.exists():
                for result_file in results_dir.glob("*"):
                    result_file.unlink()

        logger.info("Run cleaned up", run_id=run_id, keep_results=keep_results)

    def export_workspace_summary(self) -> Dict[str, Any]:
        """Export summary of workspace contents.

        Returns:
            Workspace summary dictionary
        """
        summary = {
            "workspace_path": str(self.workspace_path),
            "created": datetime.now().isoformat(),
            "total_configs": len(list(self.configs_dir.glob("*.json"))),
            "total_runs": len(list(self.runs_dir.iterdir())),
            "runs_by_status": {},
            "total_size_mb": 0
        }

        # Count runs by status
        runs = self.list_runs()
        for run in runs:
            status = run.get("status", "unknown")
            summary["runs_by_status"][status] = summary["runs_by_status"].get(status, 0) + 1

        # Calculate workspace size
        total_size = 0
        for path in self.workspace_path.rglob("*"):
            if path.is_file():
                total_size += path.stat().st_size
        summary["total_size_mb"] = total_size / (1024 * 1024)

        return summary
