"""Run type definitions and helpers shared by GUI and CLI components."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class RunType(str, Enum):
    """Supported simulation orchestration modes."""

    SINGLE = "single"
    SWEEP = "sweep"
    SENSITIVITY = "sensitivity"
    PARAMETER_ESTIMATION = "parameter_estimation"
    VIRTUAL_TRIAL = "virtual_trial"
    VIRTUAL_BIOEQUIVALENCE = "virtual_bioequivalence"

    @property
    def prefix(self) -> str:
        """Return canonical run ID prefix for this run type."""

        mapping = {
            RunType.SINGLE: "single",
            RunType.SWEEP: "sweep",
            RunType.SENSITIVITY: "sens",
            RunType.PARAMETER_ESTIMATION: "pe",
            RunType.VIRTUAL_TRIAL: "vt",
            RunType.VIRTUAL_BIOEQUIVALENCE: "vbe",
        }
        return mapping[self]


def normalise_run_label(label: Optional[str]) -> Optional[str]:
    """Return a filesystem-friendly version of the provided label."""

    if not label:
        return None

    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in label.strip())
    safe = safe.strip("-")
    return safe or None


@dataclass
class RunRequest:
    """Container describing a simulation request submitted from the UI."""

    config_path: str
    run_type: RunType = RunType.SINGLE
    label: Optional[str] = None  # Normalised label used for IDs / paths
    display_label: Optional[str] = None  # Original user-supplied label for presentation
    run_id: Optional[str] = None
    parent_run_id: Optional[str] = None
    manifest_index: Optional[int] = None
    total_runs: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    parameter_overrides: Optional[Dict[str, Any]] = None
    task_spec: Optional[Dict[str, Any]] = None

    def as_metadata(self) -> Dict[str, Any]:
        """Return request metadata suitable for persisting to disk."""

        return {
            "run_type": self.run_type.value,
            "label": self.label,
            "display_label": self.display_label or self.label,
            "parent_run_id": self.parent_run_id,
            "manifest_index": self.manifest_index,
            "total_runs": self.total_runs,
            "parameter_overrides": self.parameter_overrides,
            "task_spec": self.task_spec,
            **self.metadata,
        }

    def with_label(self, raw_label: Optional[str]) -> "RunRequest":
        """Return a copy with the provided label normalised and stored."""

        normalised = normalise_run_label(raw_label)
        self.label = normalised
        self.display_label = raw_label.strip() if raw_label else None
        return self
