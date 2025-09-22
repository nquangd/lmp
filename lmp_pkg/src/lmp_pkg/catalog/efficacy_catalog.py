"""Efficacy model catalog for loading builtin efficacy parameters."""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
import toml
import warnings

from ..domain.efficacy import (
    EfficacyModelParameters, EfficacyModelType,
    FEV1BaselineParameters, FEV1DrugEffects, FEV1DiseaseProgression,
    FEV1PlaceboEffects, FEV1StudyDemographics, FEV1ResidualError,
    FEV1StudyConfig
)
from .base import CatalogError


class EfficacyCatalog:
    """Catalog for efficacy model parameters."""
    
    def __init__(self, builtin_path: Optional[Path] = None):
        """Initialize efficacy catalog.
        
        Args:
            builtin_path: Path to builtin efficacy configurations
        """
        if builtin_path is None:
            builtin_path = Path(__file__).parent / "builtin" / "efficacy"
        
        self.builtin_path = builtin_path
        self._baseline_cache: Dict[str, FEV1BaselineParameters] = {}
        self._drug_effects_cache: Dict[str, FEV1DrugEffects] = {}
        self._disease_progression_cache: Dict[str, FEV1DiseaseProgression] = {}
        self._placebo_effects_cache: Dict[str, FEV1PlaceboEffects] = {}
        self._demographics_cache: Dict[str, FEV1StudyDemographics] = {}
        self._residual_error_cache: Dict[str, FEV1ResidualError] = {}
        self._study_config_cache: Dict[str, FEV1StudyConfig] = {}
    
    def _load_toml_safe(self, file_path: Path) -> Dict:
        """Load TOML file with error handling."""
        try:
            with open(file_path, 'r') as f:
                return toml.load(f)
        except Exception as e:
            warnings.warn(f"Failed to load {file_path}: {e}")
            return {}
    
    def load_baseline_parameters(self, name: str = "fev1_baseline_copd") -> FEV1BaselineParameters:
        """Load FEV1 baseline parameters."""
        if name in self._baseline_cache:
            return self._baseline_cache[name]
        
        file_path = self.builtin_path / "fev1_baseline_parameters.toml"
        if not file_path.exists():
            raise CatalogError(f"Baseline parameters file not found: {file_path}")
        
        data = self._load_toml_safe(file_path)
        if not data:
            raise CatalogError(f"Failed to load baseline parameters from {file_path}")
        
        params = FEV1BaselineParameters(
            name=data.get('name', name),
            description=data.get('description', ''),
            typical_baseline_L=data.get('typical_baseline_L', 1.17),
            inter_study_variance=data.get('inter_study_variance', 0.0099),
            inter_arm_variance=data.get('inter_arm_variance', 0.0004),
            covariate_effects=data.get('covariate_effects', {}),
            reference_values=data.get('reference_values', {}),
            arm_size_reference=data.get('arm_size_reference', 200)
        )
        
        self._baseline_cache[name] = params
        return params
    
    def load_drug_effects(self, name: str = "fev1_drug_effects_triple") -> FEV1DrugEffects:
        """Load FEV1 drug effect parameters."""
        if name in self._drug_effects_cache:
            return self._drug_effects_cache[name]
        
        file_path = self.builtin_path / "fev1_drug_effects.toml"
        if not file_path.exists():
            raise CatalogError(f"Drug effects file not found: {file_path}")
        
        data = self._load_toml_safe(file_path)
        if not data:
            raise CatalogError(f"Failed to load drug effects from {file_path}")
        
        params = FEV1DrugEffects(
            name=data.get('name', name),
            description=data.get('description', ''),
            reference_doses=data.get('reference_doses', {}),
            budesonide=data.get('budesonide', {}),
            formoterol=data.get('formoterol', {}),
            glycopyrronium=data.get('glycopyrronium', {}),
            interactions=data.get('interactions', {}),
            baseline_interactions=data.get('baseline_interactions', {}),
            variability=data.get('variability', {})
        )
        
        self._drug_effects_cache[name] = params
        return params
    
    def load_disease_progression(self, name: str = "fev1_disease_progression_copd") -> FEV1DiseaseProgression:
        """Load FEV1 disease progression parameters."""
        if name in self._disease_progression_cache:
            return self._disease_progression_cache[name]
        
        file_path = self.builtin_path / "fev1_disease_progression.toml"
        if not file_path.exists():
            raise CatalogError(f"Disease progression file not found: {file_path}")
        
        data = self._load_toml_safe(file_path)
        if not data:
            raise CatalogError(f"Failed to load disease progression from {file_path}")
        
        params = FEV1DiseaseProgression(
            name=data.get('name', name),
            description=data.get('description', ''),
            progression_slope_L_per_year=data.get('progression_slope_L_per_year', -0.032),
            weeks_per_year=data.get('weeks_per_year', 52.14),
            baseline_normalization_L=data.get('baseline_normalization_L', 1.2),
            variability=data.get('variability', {})
        )
        
        self._disease_progression_cache[name] = params
        return params
    
    def load_placebo_effects(self, name: str = "fev1_placebo_effects_copd") -> FEV1PlaceboEffects:
        """Load FEV1 placebo effect parameters."""
        if name in self._placebo_effects_cache:
            return self._placebo_effects_cache[name]
        
        file_path = self.builtin_path / "fev1_placebo_effects.toml"
        if not file_path.exists():
            raise CatalogError(f"Placebo effects file not found: {file_path}")
        
        data = self._load_toml_safe(file_path)
        if not data:
            raise CatalogError(f"Failed to load placebo effects from {file_path}")
        
        params = FEV1PlaceboEffects(
            name=data.get('name', name),
            description=data.get('description', ''),
            placebo_emax_L=data.get('placebo_emax_L', 0.0359),
            placebo_et50_weeks=data.get('placebo_et50_weeks', 20),
            variability=data.get('variability', {})
        )
        
        self._placebo_effects_cache[name] = params
        return params
    
    def load_demographics(self, name: str = "fev1_study_demographics_pt009001") -> FEV1StudyDemographics:
        """Load FEV1 study demographics."""
        if name in self._demographics_cache:
            return self._demographics_cache[name]
        
        file_path = self.builtin_path / "fev1_study_demographics.toml"
        if not file_path.exists():
            raise CatalogError(f"Demographics file not found: {file_path}")
        
        data = self._load_toml_safe(file_path)
        if not data:
            raise CatalogError(f"Failed to load demographics from {file_path}")
        
        params = FEV1StudyDemographics(
            name=data.get('name', name),
            description=data.get('description', ''),
            age_distribution=data.get('age_distribution', {}),
            disease_severity=data.get('disease_severity', {}),
            exacerbation_history=data.get('exacerbation_history', {})
        )
        
        self._demographics_cache[name] = params
        return params
    
    def load_residual_error(self, name: str = "fev1_residual_error") -> FEV1ResidualError:
        """Load FEV1 residual error parameters."""
        if name in self._residual_error_cache:
            return self._residual_error_cache[name]
        
        file_path = self.builtin_path / "fev1_residual_error.toml"
        if not file_path.exists():
            raise CatalogError(f"Residual error file not found: {file_path}")
        
        data = self._load_toml_safe(file_path)
        if not data:
            raise CatalogError(f"Failed to load residual error from {file_path}")
        
        params = FEV1ResidualError(
            name=data.get('name', name),
            description=data.get('description', ''),
            residual_variance=data.get('residual_variance', 0.042),
            default_arm_size=data.get('default_arm_size', 204),
            reference_arm_size=data.get('reference_arm_size', 200),
            error_model=data.get('error_model', 'additive'),
            error_distribution=data.get('error_distribution', 'normal')
        )
        
        self._residual_error_cache[name] = params
        return params
    
    def load_study_config(self, name: str = "fev1_study_config_default") -> FEV1StudyConfig:
        """Load FEV1 study configuration."""
        if name in self._study_config_cache:
            return self._study_config_cache[name]
        
        file_path = self.builtin_path / "fev1_study_config.toml"
        if not file_path.exists():
            raise CatalogError(f"Study config file not found: {file_path}")
        
        data = self._load_toml_safe(file_path)
        if not data:
            raise CatalogError(f"Failed to load study config from {file_path}")
        
        params = FEV1StudyConfig(
            name=data.get('name', name),
            description=data.get('description', ''),
            n_trials=data.get('n_trials', 200),
            trial_size=data.get('trial_size', 204),
            trial_duration_weeks=data.get('trial_duration_weeks', 24),
            study_type=data.get('study_type', 'copd_efficacy'),
            primary_endpoint=data.get('primary_endpoint', 'fev1_change'),
            confidence_level=data.get('confidence_level', 0.95),
            alpha_level=data.get('alpha_level', 0.05)
        )
        
        self._study_config_cache[name] = params
        return params
    
    def create_complete_model_parameters(self,
                                       baseline_name: str = "fev1_baseline_copd",
                                       drug_effects_name: str = "fev1_drug_effects_triple", 
                                       disease_progression_name: str = "fev1_disease_progression_copd",
                                       placebo_effects_name: str = "fev1_placebo_effects_copd",
                                       demographics_name: str = "fev1_study_demographics_pt009001",
                                       residual_error_name: str = "fev1_residual_error",
                                       study_config_name: str = "fev1_study_config_default") -> EfficacyModelParameters:
        """Create complete efficacy model parameters by loading all components."""
        
        return EfficacyModelParameters(
            model_type=EfficacyModelType.FEV1_COPD,
            baseline=self.load_baseline_parameters(baseline_name),
            drug_effects=self.load_drug_effects(drug_effects_name),
            disease_progression=self.load_disease_progression(disease_progression_name),
            placebo_effects=self.load_placebo_effects(placebo_effects_name),
            demographics=self.load_demographics(demographics_name),
            residual_error=self.load_residual_error(residual_error_name),
            study_config=self.load_study_config(study_config_name)
        )
    
    def list_available_configs(self) -> Dict[str, List[str]]:
        """List all available efficacy configurations."""
        configs = {}
        
        if self.builtin_path.exists():
            for toml_file in self.builtin_path.glob("*.toml"):
                category = toml_file.stem
                try:
                    data = self._load_toml_safe(toml_file)
                    configs[category] = [data.get('name', category)] if data else []
                except Exception:
                    configs[category] = []
        
        return configs


def get_efficacy_catalog() -> EfficacyCatalog:
    """Get the default efficacy catalog instance."""
    return EfficacyCatalog()