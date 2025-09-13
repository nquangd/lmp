"""FEV1 efficacy models for respiratory drug assessment."""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, NamedTuple
from scipy.stats import norm
from scipy import stats
from dataclasses import dataclass

from .base import EfficacyModel
from ...contracts.types import EfficacyInput, EfficacyResult


@dataclass
class VariabilityFEV1:
    """FEV1 variability parameters."""
    residual_var: float = 0.042
    disease_cov: float = 2
    exacerbation_cov: float = 0
    age_cov: float = 63.6


class FEV1StudyConfig(NamedTuple):
    """Configuration for FEV1 studies."""
    n_trials: int = 200
    trial_size: int = 204
    trial_duration_weeks: int = 24


class COPDREV1:
    """COPD FEV1 efficacy model.
    
    Based on the original fev1model.py, this class provides comprehensive
    FEV1 efficacy modeling for COPD patients including:
    - Baseline FEV1 simulation
    - Disease progression effects
    - Placebo effects with gradual onset
    - Multi-drug combination effects (Budesonide + Formoterol + Glycopyrronium)
    - Inter-study and intra-study variability
    """
    
    def __init__(self, study_config: FEV1StudyConfig, residual_var: float = 0.042):
        self.n_trials = study_config.n_trials
        self.trial_size = study_config.trial_size
        self.trial_duration_weeks = study_config.trial_duration_weeks
        self.residual_var = residual_var
        self.gmr_one_flag = False  # For testing equal efficacy
    
    @staticmethod
    def baseline_fev1(
        arm_size: int = 240,
        disease_cov: float = 2,
        exacerbation_cov: float = 0,
        age_cov: float = 63.6
    ) -> float:
        """Calculate baseline FEV1 for simulated patients.
        
        Parameters based on Table S2 from the supplementary material.
        
        Args:
            arm_size: Study arm size
            disease_cov: Disease severity covariate
            exacerbation_cov: Exacerbation history covariate
            age_cov: Age covariate
            
        Returns:
            Baseline FEV1 value [L]
        """
        # Typical baseline and variability parameters
        typical_bl = 1.17  # L
        bl_isv = 0.0099    # Inter-study variance
        bl_iav = 0.0004    # Inter-arm variance
        
        # Random effects
        eta_i = np.random.normal(0, np.sqrt(bl_isv))
        kappa_i = np.random.normal(0, np.sqrt(bl_iav))
        
        # Covariate effects
        disease_baseline = -0.163
        exacerbation_baseline = -0.0397
        age_baseline = -0.0091
        
        # Baseline FEV1 calculation (Eq. 3)
        baseline_i = (
            typical_bl *
            (1 + disease_baseline * (disease_cov - 2)) *
            (1 + exacerbation_baseline * (exacerbation_cov - 0)) *
            (1 + age_baseline * (age_cov - 63.6)) *
            np.exp(eta_i + kappa_i / np.sqrt(arm_size / 200))
        )
        
        return baseline_i
    
    @staticmethod
    def disease_progression(
        baseline: float,
        time_weeks: float,
        eta_dp: float = 0
    ) -> float:
        """Calculate FEV1 change due to disease progression.
        
        Based on Eq. 4 with inter-study variability.
        
        Args:
            baseline: Baseline FEV1 [L]
            time_weeks: Time elapsed [weeks]
            eta_dp: Inter-study variability for disease progression
            
        Returns:
            FEV1 change due to disease progression [L]
        """
        # Typical progression rate
        dp_slope_yearly = -0.032  # L/year
        dp_slope_weekly = dp_slope_yearly / 52.14  # L/week
        
        # Progression proportional to baseline, with variability
        progression = (
            dp_slope_weekly * (baseline / 1.2) * 
            time_weeks * np.exp(eta_dp)
        )
        
        return progression
    
    @staticmethod
    def placebo_effect(time_weeks: float, eta_pbo: float = 0) -> float:
        """Calculate placebo effect over time.
        
        Based on Eq. 5 with gradual onset model and inter-study variability.
        
        Args:
            time_weeks: Time elapsed [weeks]
            eta_pbo: Inter-study variability for placebo effect
            
        Returns:
            Placebo effect [L]
        """
        # Placebo parameters from Table S2
        pbo_emax = 0.0359  # Maximum placebo effect
        pbo_et50 = 20      # Time to 50% effect [weeks]
        
        # Gradual onset with variability
        effect = (pbo_emax + eta_pbo) * (time_weeks / (pbo_et50 + time_weeks))
        
        return effect
    
    @staticmethod
    def drug_effect(
        baseline: float = 1.17,
        bd_dose: float = 320,    # Budesonide [µg]
        ff_dose: float = 9.6,    # Formoterol [µg]
        gp_dose: float = 14.4,   # Glycopyrronium [µg]
        eta_ai: float = 0,       # Anti-inflammatory variability
        eta_bd: float = 0        # Bronchodilator variability
    ) -> float:
        """Calculate combined drug effect for triple therapy.
        
        Args:
            baseline: Baseline FEV1 [L]
            bd_dose: Budesonide dose [µg]
            ff_dose: Formoterol dose [µg]
            gp_dose: Glycopyrronium dose [µg]
            eta_ai: Anti-inflammatory effect variability
            eta_bd: Bronchodilator effect variability
            
        Returns:
            Combined drug effect [L]
        """
        # Drug-specific Emax model parameters
        formoterol = {'dref': ff_dose, 'e50': 1.5, 'emax': 82.3}
        glycopyrronium = {'dref': gp_dose * 2, 'e50': 10.1, 'emax': 144.3}
        budesonide = {'dref': bd_dose, 'e50': 169.8, 'emax': 63.2}
        
        # Interaction term for LABA+LAAC combination
        labalaacint = 1.35
        
        # Baseline interaction effects (Eq. 9)
        stepb = int(baseline >= 1.20)
        theta_ai_bl = 0.638
        theta_bd_bl = 0.268
        
        baseline_interaction_ai = (
            (1 - stepb) * (1 + theta_ai_bl * (baseline - 1.20)) + stepb
        )
        baseline_interaction_bd = (
            (1 - stepb) * (1 + theta_bd_bl * (baseline - 1.20)) + stepb
        )
        
        # Individual drug effects (Emax model)
        bd = (budesonide['emax'] * budesonide['dref']) / (
            budesonide['dref'] + budesonide['e50']
        )
        gp = (glycopyrronium['emax'] * glycopyrronium['dref']) / (
            glycopyrronium['dref'] + glycopyrronium['e50']
        )
        ff = (formoterol['emax'] * formoterol['dref']) / (
            formoterol['dref'] + formoterol['e50']
        )
        
        # Combined effects with interactions and variability
        broncho = (
            (gp ** labalaacint + ff ** labalaacint) ** (1 / labalaacint) *
            baseline_interaction_bd * (1 + eta_bd)
        )
        anti_inf = bd * baseline_interaction_ai * (1 + eta_ai)
        
        # Total drug effect
        bgf = broncho + anti_inf
        
        return bgf / 1000  # Convert from mL to L
    
    def residual_error(self, arm_size: int = 204) -> Dict[str, Any]:
        """Generate residual error for study arm.
        
        Args:
            arm_size: Study arm size
            
        Returns:
            Dictionary with arm-level and individual residual errors
        """
        # Individual residual errors
        res_person = np.random.normal(0, np.sqrt(self.residual_var), size=arm_size)
        
        # Study-arm level error (mean across individuals)
        resid_i = np.mean(res_person)
        
        return {"Resid_i": resid_i, "res_person": res_person}
    
    def simulate_trial_effects(
        self,
        n_arm: int = 204,
        n_sim: int = 1,
        dose_ratios: List[float] = [1, 1, 1],  # [BD, FF, GP] ratios
        trial_duration_weeks: int = 24
    ) -> Tuple[float, float]:
        """Simulate test vs reference drug effects.
        
        Args:
            n_arm: Arm size
            n_sim: Number of simulations
            dose_ratios: Dose ratios for [BD, FF, GP]
            trial_duration_weeks: Trial duration
            
        Returns:
            Tuple of (test_effect, reference_effect)
        """
        # Patient characteristics
        disease_dist = np.random.choice([1, 2, 3, 4], size=n_sim)
        exacerbation_dist = np.random.choice([0, 1, 2, 3, 4], size=n_sim)
        age_dist = np.random.normal(61.8, 8.5, size=n_sim)  # PT009001 demographics
        
        # Inter-study variabilities
        isv_ai_cv = 0.405  # Anti-inflammatory CV
        isv_bd_cv = 0.185  # Bronchodilator CV
        
        eta_ai = np.random.normal(0, np.sqrt(np.log(isv_ai_cv**2 + 1)), n_sim)
        eta_bd = np.random.normal(0, np.sqrt(np.log(isv_bd_cv**2 + 1)), n_sim)
        
        # Disease progression variability
        isv_dp_cv = np.exp(-0.611)
        var_dp = np.log(isv_dp_cv**2 + 1)
        eta_dp = np.random.normal(0, np.sqrt(var_dp), n_sim)
        
        # Placebo effect variability
        var_pbo = 0.0021
        eta_pbo = np.random.normal(0, np.sqrt(var_pbo), n_sim)
        
        # Simulate baseline FEV1
        baseline_simmed = np.array([
            self.baseline_fev1(n_arm, disease, exacerbation, age)
            for disease, exacerbation, age in zip(disease_dist, exacerbation_dist, age_dist)
        ])
        
        # Reference drug effect (standard doses)
        ref_drug_effect = np.array([
            self.drug_effect(baseline, 320, 9.6, 14.4, eta_ai_val, eta_bd_val)
            for baseline, eta_ai_val, eta_bd_val in zip(baseline_simmed, eta_ai, eta_bd)
        ])[0]
        
        # Test drug effect (modified doses)
        if self.gmr_one_flag:
            bd_ratio, ff_ratio, gp_ratio = 1, 1, 1  # Equal efficacy
        else:
            bd_ratio, ff_ratio, gp_ratio = dose_ratios
        
        test_drug_effect = np.array([
            self.drug_effect(baseline, 320 * bd_ratio, 9.6 * ff_ratio, 14.4 * gp_ratio, eta_ai_val, eta_bd_val)
            for baseline, eta_ai_val, eta_bd_val in zip(baseline_simmed, eta_ai, eta_bd)
        ])[0]
        
        # Disease progression
        disease_prog = np.array([
            self.disease_progression(baseline, trial_duration_weeks, eta_dp_val)
            for baseline, eta_dp_val in zip(baseline_simmed, eta_dp)
        ])
        
        # Placebo effect
        placebo_eff = np.array([
            self.placebo_effect(trial_duration_weeks, eta_pbo_val)
            for eta_pbo_val in eta_pbo
        ])
        
        # Residual errors (different for test and reference)
        ref_residual = self.residual_error(arm_size=n_arm)['Resid_i']
        test_residual = self.residual_error(arm_size=n_arm)['Resid_i']
        
        # Final FEV1 effects (change from baseline)
        ref_final_effect = (
            baseline_simmed + ref_drug_effect + 
            disease_prog + placebo_eff + ref_residual
        ) - baseline_simmed
        
        test_final_effect = (
            baseline_simmed + test_drug_effect + 
            disease_prog + placebo_eff + test_residual
        ) - baseline_simmed
        
        return test_final_effect[0], ref_final_effect[0]
    
    def run_dose_prediction_trial(
        self,
        n_arm: int = 204,
        n_sim: int = 200,
        trial_duration_weeks: int = 24,
        bd_dose: float = 320,
        gp_dose: float = 14.4,
        ff_dose: float = 9.6
    ) -> List[float]:
        """Run full trial simulation for dose prediction.
        
        Args:
            n_arm: Arm size
            n_sim: Number of simulations
            trial_duration_weeks: Trial duration
            bd_dose: Budesonide dose [µg]
            gp_dose: Glycopyrronium dose [µg]
            ff_dose: Formoterol dose [µg]
            
        Returns:
            List with [baseline_mean, effect_mean, effect_ci_low, effect_ci_high, effect_min, effect_max]
        """
        # Patient characteristics distributions
        disease_dist = np.random.choice([1, 2, 3, 4], size=n_sim)
        exacerbation_dist = np.random.choice([0, 1, 2, 3, 4], size=n_sim)
        age_dist = np.random.normal(61.8, 8.5, size=n_sim)
        
        # Inter-study variabilities
        isv_ai_cv = 0.405
        isv_bd_cv = 0.185
        eta_ai = np.random.normal(0, np.sqrt(np.log(isv_ai_cv**2 + 1)), n_sim)
        eta_bd = np.random.normal(0, np.sqrt(np.log(isv_bd_cv**2 + 1)), n_sim)
        
        # Disease progression variability
        isv_dp_cv = np.exp(-0.611)
        var_dp = np.log(isv_dp_cv**2 + 1)
        eta_dp = np.random.normal(0, np.sqrt(var_dp), n_sim)
        
        # Placebo effect variability
        var_pbo = 0.0021
        eta_pbo = np.random.normal(0, np.sqrt(var_pbo), n_sim)
        
        # Simulate all components
        baseline_simmed = np.array([
            self.baseline_fev1(n_arm, disease, exacerbation, age)
            for disease, exacerbation, age in zip(disease_dist, exacerbation_dist, age_dist)
        ])
        
        drug_effect_simmed = np.array([
            self.drug_effect(baseline, bd_dose, ff_dose, gp_dose, eta_ai_val, eta_bd_val)
            for baseline, eta_ai_val, eta_bd_val in zip(baseline_simmed, eta_ai, eta_bd)
        ])
        
        disease_prog_simmed = np.array([
            self.disease_progression(baseline, trial_duration_weeks, eta_dp_val)
            for baseline, eta_dp_val in zip(baseline_simmed, eta_dp)
        ])
        
        placebo_eff_simmed = np.array([
            self.placebo_effect(trial_duration_weeks, eta_pbo_val)
            for eta_pbo_val in eta_pbo
        ])
        
        # Residual errors
        residual_errors = [self.residual_error(n_arm) for _ in range(n_sim)]
        residual_simmed = np.array([x['Resid_i'] for x in residual_errors])
        
        # Final FEV1 change from baseline
        final_fev1_change = (
            baseline_simmed + drug_effect_simmed + 
            disease_prog_simmed + placebo_eff_simmed + residual_simmed
        ) - baseline_simmed
        
        # Convert to mL
        estimates_ml = final_fev1_change * 1000
        
        # Calculate confidence intervals
        residual_sds = np.array([np.std(x['res_person']) for x in residual_errors])
        pooled_sd = np.sqrt(np.mean((n_arm - 1) * residual_sds ** 2) / (n_arm - 1))
        bound_length = pooled_sd * np.sqrt(1/n_arm) * norm.ppf(0.975) * 1000  # 95% CI
        
        return [
            np.mean(baseline_simmed),     # Baseline mean
            estimates_ml.mean(),          # Effect mean
            estimates_ml.mean() - bound_length,  # CI lower
            estimates_ml.mean() + bound_length,  # CI upper
            estimates_ml.min(),           # Effect min
            estimates_ml.max()            # Effect max
        ]


class FEV1EfficacyModel(EfficacyModel):
    """FEV1 efficacy model implementing the Stage interface."""
    
    name: str = "fev1_copd"
    
    def __init__(self, 
                 study_config: Optional[FEV1StudyConfig] = None,
                 variability: Optional[VariabilityFEV1] = None):
        """Initialize FEV1 efficacy model.
        
        Args:
            study_config: Study configuration parameters
            variability: Variability parameters
        """
        if study_config is None:
            study_config = FEV1StudyConfig()
        if variability is None:
            variability = VariabilityFEV1()
            
        self.study_config = study_config
        self.variability = variability
        self.copd_model = COPDREV1(study_config, variability.residual_var)
    
    def predict_efficacy(self,
                        exposure_data: Dict[str, np.ndarray],
                        patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict FEV1 efficacy from drug exposure data.
        
        Args:
            exposure_data: Drug exposure metrics 
            patient_data: Patient characteristics
            
        Returns:
            FEV1 efficacy predictions
        """
        # Extract dose information from patient/exposure data
        dose_regimen = {
            'BD': patient_data.get('bd_dose', 320),
            'FF': patient_data.get('ff_dose', 9.6),
            'GP': patient_data.get('gp_dose', 14.4)
        }
        
        # Run prediction
        results = self.copd_model.run_dose_prediction_trial(
            n_arm=self.study_config.trial_size,
            n_sim=self.study_config.n_trials,
            trial_duration_weeks=self.study_config.trial_duration_weeks,
            bd_dose=dose_regimen['BD'],
            ff_dose=dose_regimen['FF'],
            gp_dose=dose_regimen['GP']
        )
        
        return {
            'dose_regimen': dose_regimen,
            'baseline_fev1_L': results[0],
            'mean_effect_mL': results[1],
            'ci_lower_mL': results[2],
            'ci_upper_mL': results[3],
            'min_effect_mL': results[4],
            'max_effect_mL': results[5]
        }
    
    def run(self, data: EfficacyInput) -> EfficacyResult:
        """Run FEV1 efficacy model."""
        # Extract relevant information from input
        exposure_data = data.pk_results if hasattr(data, 'pk_results') else {}
        patient_data = {
            'bd_dose': data.params.get('bd_dose', 320),
            'ff_dose': data.params.get('ff_dose', 9.6), 
            'gp_dose': data.params.get('gp_dose', 14.4)
        }
        
        # Predict efficacy
        efficacy_pred = self.predict_efficacy(exposure_data, patient_data)
        
        # Create result
        time_points = np.array([0, self.study_config.trial_duration_weeks])
        efficacy_values = np.array([0, efficacy_pred['mean_effect_mL'] / 1000])  # Convert to L
        
        return EfficacyResult(
            t=time_points,
            efficacy_endpoint=efficacy_values,
            endpoint_name="fev1_change_L",
            baseline_value=efficacy_pred['baseline_fev1_L'],
            metadata={
                'dose_regimen': efficacy_pred['dose_regimen'],
                'confidence_interval_mL': [
                    efficacy_pred['ci_lower_mL'],
                    efficacy_pred['ci_upper_mL']
                ],
                'study_config': {
                    'n_trials': self.study_config.n_trials,
                    'trial_size': self.study_config.trial_size,
                    'duration_weeks': self.study_config.trial_duration_weeks
                }
            }
        )