"""Efficacy model domain entities."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import numpy as np
from enum import Enum


class EfficacyModelType(Enum):
    """Types of efficacy models."""
    FEV1_COPD = "fev1_copd"
    SYMPTOM_SCORE = "symptom_score"
    EXACERBATION_RATE = "exacerbation_rate"


@dataclass
class FEV1BaselineParameters:
    """FEV1 baseline model parameters."""
    name: str
    description: str
    typical_baseline_L: float
    inter_study_variance: float
    inter_arm_variance: float
    covariate_effects: Dict[str, float]
    reference_values: Dict[str, float]
    arm_size_reference: int


@dataclass
class FEV1DrugEffects:
    """FEV1 drug effect parameters."""
    name: str
    description: str
    reference_doses: Dict[str, float]
    budesonide: Dict[str, float]
    formoterol: Dict[str, float]
    glycopyrronium: Dict[str, float]
    interactions: Dict[str, float]
    baseline_interactions: Dict[str, float]
    variability: Dict[str, float]


@dataclass
class FEV1DiseaseProgression:
    """FEV1 disease progression parameters."""
    name: str
    description: str
    progression_slope_L_per_year: float
    weeks_per_year: float
    baseline_normalization_L: float
    variability: Dict[str, float]


@dataclass
class FEV1PlaceboEffects:
    """FEV1 placebo effect parameters."""
    name: str
    description: str
    placebo_emax_L: float
    placebo_et50_weeks: float
    variability: Dict[str, float]


@dataclass
class FEV1StudyDemographics:
    """FEV1 study demographics."""
    name: str
    description: str
    age_distribution: Dict[str, Any]
    disease_severity: Dict[str, Any]
    exacerbation_history: Dict[str, Any]


@dataclass
class FEV1ResidualError:
    """FEV1 residual error parameters."""
    name: str
    description: str
    residual_variance: float
    default_arm_size: int
    reference_arm_size: int
    error_model: str
    error_distribution: str


@dataclass
class FEV1StudyConfig:
    """FEV1 study configuration."""
    name: str
    description: str
    n_trials: int
    trial_size: int
    trial_duration_weeks: int
    study_type: str
    primary_endpoint: str
    confidence_level: float
    alpha_level: float


@dataclass
class EfficacyModelParameters:
    """Complete efficacy model parameter set."""
    model_type: EfficacyModelType
    baseline: FEV1BaselineParameters
    drug_effects: FEV1DrugEffects
    disease_progression: FEV1DiseaseProgression
    placebo_effects: FEV1PlaceboEffects
    demographics: FEV1StudyDemographics
    residual_error: FEV1ResidualError
    study_config: FEV1StudyConfig
    
    def get_variability_parameters(self) -> Dict[str, float]:
        """Extract all variability parameters."""
        return {
            # Baseline variability
            'baseline_isv': self.baseline.inter_study_variance,
            'baseline_iav': self.baseline.inter_arm_variance,
            
            # Drug effect variability
            'anti_inflammatory_cv': self.drug_effects.variability['anti_inflammatory_cv'],
            'bronchodilator_cv': self.drug_effects.variability['bronchodilator_cv'],
            
            # Disease progression variability
            'disease_progression_log_isv_cv': self.disease_progression.variability['log_isv_cv_disease_progression'],
            
            # Placebo variability
            'placebo_emax_isv_var': self.placebo_effects.variability['isv_variance_placebo_emax'],
            
            # Residual error
            'residual_variance': self.residual_error.residual_variance
        }
    
    def get_reference_doses(self) -> Dict[str, float]:
        """Get reference doses for all APIs."""
        return self.drug_effects.reference_doses.copy()
    
    def get_emax_parameters(self) -> Dict[str, Dict[str, float]]:
        """Get Emax model parameters for all APIs."""
        return {
            'budesonide': self.drug_effects.budesonide.copy(),
            'formoterol': self.drug_effects.formoterol.copy(),
            'glycopyrronium': self.drug_effects.glycopyrronium.copy()
        }