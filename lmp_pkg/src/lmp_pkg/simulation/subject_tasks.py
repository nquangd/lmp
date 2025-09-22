"""Helpers for preparing and executing subject-level simulation tasks.

These utilities mirror the way production jobs dispatch simulations via
SLURM arrays: each task contains the catalog identifiers and a deterministic
seed.  The worker reconstructs the required entities, applies variability
sampling, and runs the downstream models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Dict, Any, Optional, Mapping, Union
import numpy as np

from ..domain.entities import Subject, API, Product, VariabilitySettings


@dataclass(frozen=True)
class SubjectTask:
    """Descriptor for a single subject/API simulation run."""

    subject_index: int
    api_name: str
    subject_name: str = "healthy_reference"
    products: Sequence[str] = ("reference_product", "test_product")
    seed: int = 0
    subject: Optional[Subject] = None
    api: Optional[API] = None
    product_entities: Optional[Dict[str, Product]] = None
    apply_variability: bool = True
    variability_settings: Optional[Union[VariabilitySettings, Mapping[str, bool]]] = None
    study_type: Optional[str] = None
    study_design: Optional[str] = None
    charcoal_block: bool = False
    suppress_et_absorption: bool = False


def build_tasks(
    apis: Iterable[str],
    n_subjects: int,
    base_seed: int = 1234,
    subject_name: str = "healthy_reference",
    products: Sequence[str] = ("reference_product", "test_product"),
    apply_variability: bool = True,
    variability_settings: Optional[Union[VariabilitySettings, Mapping[str, bool]]] = None,
    study_type: Optional[str] = None,
    study_design: Optional[str] = None,
    charcoal_block: bool = False,
    suppress_et_absorption: bool = False,
) -> List[SubjectTask]:
    """Create deterministic subject tasks for a set of APIs.

    Args:
        apis: API names to simulate.
        n_subjects: Number of virtual subjects per API.
        base_seed: Base seed used to derive per-task seeds.
        subject_name: Catalog subject identifier to load.
        products: Products to simulate per subject.
        apply_variability: Whether to sample variability when hydrating each
            subject. Set to False for deterministic reference simulations.
        variability_settings: Optional per-domain toggle mapping forwarded to
            `Subject.get_final_values` for fine-grained variability control.
        study_type: Optional study type label for downstream metadata.
        study_design: Optional study design label for downstream metadata.
        charcoal_block: If True, disable GI absorption in PBPK stage.
        suppress_et_absorption: If True, disable ET systemic absorption in PBPK stage.

    Returns:
        List of tasks. Each task corresponds to one subject/API combination.
    """

    tasks: List[SubjectTask] = []
    for api_idx, api in enumerate(apis):
        api_offset = (api_idx + 1) * 10_000
        for subject_idx in range(n_subjects):
            seed = base_seed + api_offset + subject_idx
            tasks.append(
                SubjectTask(
                    subject_index=subject_idx,
                    api_name=api,
                    subject_name=subject_name,
                    products=tuple(products),
                    seed=seed,
                    apply_variability=apply_variability,
                    variability_settings=variability_settings,
                    study_type=study_type,
                    study_design=study_design,
                    charcoal_block=charcoal_block,
                    suppress_et_absorption=suppress_et_absorption,
                )
            )
    return tasks


def load_subject_for_task(task: SubjectTask) -> Tuple[Subject, API, Subject]:
    """Hydrate Subject/API entities for a task and apply variability.

    Returns the baseline subject (pre-variability), the API parameters, and
    the final subject with variability-driven attributes (scaled lung, flow
    profile, etc.).
    """

    # Ensure deterministic sampling regardless of ambient RNG state
    rng_state = np.random.get_state()
    np.random.seed(task.seed)
    try:
        if task.subject is not None:
            subject = task.subject
        else:
            subject = Subject.from_builtin(task.subject_name, api_name=task.api_name)

        if task.api is not None:
            api = task.api
        else:
            api = API.from_builtin(task.api_name)

        subject_final = subject.get_final_values(
            apply_variability=task.apply_variability,
            api_name=task.api_name,
            variability_settings=task.variability_settings,
        )
    finally:
        np.random.set_state(rng_state)
    #DEBUG:
    #print(f"Loaded subject {task.subject_index} for API {task.api_name} with FRC={subject_final.demographic.frc_ml:.1f} mL and region={subject_final.lung_regional}")
    return subject, api, subject_final


def prepare_entities(task: SubjectTask) -> Dict[str, Any]:
    """Build the entity dictionary expected by the pipeline stages."""

    subject, api, subject_final = load_subject_for_task(task)

    # Subject final already contains scaled geometry and inhalation profile
    maneuver = subject_final.inhalation_maneuver

    return {
        "subject": subject_final,
        "api": api,
        "maneuver": maneuver,
        # Products are resolved downstream per product name.
    }
