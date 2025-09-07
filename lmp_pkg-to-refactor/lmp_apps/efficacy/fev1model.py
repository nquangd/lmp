import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy import stats


class Variability_FEV1():
    def __init__(self):
        self.residual_var = 0.042
        self.disease_cov = 2
        self.exacerbation_cov=0
        self.age_cov = 63.6
    

class COPD_FEV1():
    def __init__(self, study, residual_var = 0.042):
        self.n_trials = study.n_trials
        self.trial_size = study.trial_size
        self.GMR_ONE_FLAG = False
        self.residual_var = residual_var
        
    @staticmethod
    def baseline_fev1(arm_size=240, disease_cov=2, exacerbation_cov=0, age_cov=63.6):
        """
        Calculates the baseline FEV1 for a simulated patient.
        Parameters are based on the "Augmented data (1996-2021)" column of Table S2
        in the supplementary material.
        """
        typical_bl = 1.17
        bl_isv = 0.0099  # Inter-study variance
        eta_i = np.random.normal(0, np.sqrt(bl_isv))
        bl_iav = 0.0004  # Inter-arm variance
        kappa_i = np.random.normal(0, np.sqrt(bl_iav))
        
        # Covariate effects
        disease_baseline = -0.163
        exacerbation_baseline = -0.0397
        age_baseline = -0.0091
        
        # Baseline FEV1 calculation based on Eq. 3 from the paper
        baseline_i = (typical_bl *
                    (1 + disease_baseline * (disease_cov - 2)) *
                    (1 + exacerbation_baseline * (exacerbation_cov - 0)) *
                    (1 + age_baseline * (age_cov - 63.6)) *
                    np.exp(eta_i + kappa_i / np.sqrt(arm_size / 200))
                    )
        
        return baseline_i

    # --- UPDATED FUNCTION ---
    @staticmethod
    def disease_progression(baseline, time_weeks, eta_dp=0):
        """
        Calculates the change in FEV1 due to disease progression over time.
        Based on Eq. 4 from the paper and parameters from Table S2.
        Now includes inter-study variability (eta_dp).
        """
        # Typical progression is -32 mL/year.
        dp_slope_yearly = -0.032 
        dp_slope_weekly = dp_slope_yearly / 52.14 # Convert to weekly rate
        
        # The effect is proportional to the baseline FEV1, normalized to a typical 1.2L baseline.
        # The exp(eta_dp) term introduces random variability between studies.
        progression = dp_slope_weekly * (baseline / 1.2) * time_weeks * np.exp(eta_dp)
        return progression

    # --- UPDATED FUNCTION ---
    @staticmethod
    def placebo_effect(time_weeks, eta_pbo=0):
        """
        Calculates the placebo effect over time.
        Based on Eq. 5 from the paper and parameters from Table S2.
        This implements the gradual onset model and now includes inter-study
        variability (eta_pbo).
        """
        # From Table S2 (Data 1996-2013 column, as new value isn't specified directly)
        pbo_emax = 0.0359
        # From Table S2, log(PT50) = 3.0, so PT50 is ~20 weeks.
        pbo_et50 = 20 

        # The eta_pbo term introduces random variability to the max placebo effect between studies.
        effect = (pbo_emax + eta_pbo) * (time_weeks / (pbo_et50 + time_weeks))
        return effect
    @staticmethod
    def drug_effect(baseline=1.17, bd_dose=320, ff_dose=9.6, gp_dose=14.4, eta_ai=0, eta_bd=0):
        """
        Calculates the combined drug effect from Budesonide (ICS), Formoterol (LABA),
        and Glycopyrronium (LAAC).
        """
        # Drug parameters from various sources in the paper, simplified for simulation
        formoterol = {
            'dref': ff_dose, 'e50': 1.5, 'emax': 82.3
        }
        glycopyrronium = {
            #'dref': gp_dose, 
            'dref': gp_dose * 2, 
            'e50': 10.1, 'emax': 144.3
        }
        budesonide = {
            #'dref': bd_dose * 2,  # bd is given as qd, hence the doubling in the model
            'dref': bd_dose ,  # bd is given as qd, hence the doubling in the model
            'e50': 169.8, 'emax': 63.2
        }
        
        # Interaction term for LABA+LAAC combination from Table S2
        labalaacint = 1.35 
        
        # Baseline interaction effect (Eq. 9)
        stepb = int(baseline >= 1.20)
        theta_ai_bl = 0.638
        baseline_interaction_ai = (1 - stepb) * (1 + theta_ai_bl * (baseline - 1.20)) + stepb
        theta_bd_bl = 0.268
        baseline_interaction_bd = (1 - stepb) * (1 + theta_bd_bl * (baseline - 1.20)) + stepb
        
        # Individual drug effects (Emax model)
        bd = (budesonide['emax'] * budesonide['dref']) / (budesonide['dref'] + budesonide['e50'])
        gp = (glycopyrronium['emax'] * glycopyrronium['dref']) / (glycopyrronium['dref'] + glycopyrronium['e50'])
        ff = (formoterol['emax'] * formoterol['dref']) / (formoterol['dref'] + formoterol['e50'])

        # Combined effects including interactions and random effects
        broncho = (gp ** labalaacint + ff ** labalaacint) ** (1 / labalaacint) * baseline_interaction_bd * (1 + eta_bd)
        anti_inf = bd * baseline_interaction_ai * (1 + eta_ai)
        bgf = broncho + anti_inf
        
        return bgf / 1000  # Convert from mL to Liters
    @staticmethod
    def residual_w(arm_size=204, residual_var = 0.042):
        """
        Generates residual error for a study arm.
        """
        # From Table S2, variance of additive residual error
        add_error_var = residual_var
        res_person = np.random.normal(0, np.sqrt(add_error_var), size=arm_size)
        # The paper models error at the study-arm level, so we take the mean
        eta = np.mean(res_person)
        return {"Resid_i": eta, "res_person": res_person}
    @staticmethod
    def residual_with_error(arm_size=204, add_error = 0.042):
        #add_error = 0.042
        res_person = np.random.normal(0, np.sqrt(add_error), size=arm_size)
        eta = np.mean(res_person)
        return pd.DataFrame({'Resid_i': eta, 'res_person': res_person})
    # --- UPDATED SIMULATION FUNCTION ---
    def run_trial_dose_prediction(self, n_arm=204, n_sim=200,  trial_duration_weeks=24, bd_dose=320, gp_dose=14.4, ff_dose=9.6):
        """
        
        Runs a full trial simulation, incorporating all model components to predict
        the final FEV1 at the end of the trial duration.
        """
        # Distributions for patient characteristics
        disease_dist = np.random.choice([1, 2, 3, 4], size=n_sim)
        exacerbation_dist = np.random.choice([0, 1, 2, 3, 4], size=n_sim)
        
        #disease_dist = np.random.choice([2, 4], size=n_sim)
        #exacerbation_dist = np.random.choice([0, 0], size=n_sim)
        
        #age_dist = np.random.normal(65, 7.5, size=n_sim)
        age_dist = np.random.normal(61.8, 8.5, size=n_sim)   # PT009001
        
        # --- Inter-study variabilities (ISV) from Table S2 ---
        # ISV for drug effects
        isv_ai_cv = 0.405 # CV for anti-inflammatory
        isv_bd_cv = 0.185 # CV for bronchodilator
        # Variance for log-normal is log(CV^2+1)
        eta_ai = np.random.normal(0, np.sqrt(np.log(isv_ai_cv**2 + 1)), n_sim)
        eta_bd = np.random.normal(0, np.sqrt(np.log(isv_bd_cv**2 + 1)), n_sim)
        
        # ISV for disease progression (NEWLY ADDED)
        # From "Log of ISV CV for the disease progression slope" = -0.611
        isv_dp_cv = np.exp(-0.611) 
        var_dp = np.log(isv_dp_cv**2 + 1)
        eta_dp = np.random.normal(0, np.sqrt(var_dp), n_sim)
        
        # ISV for placebo effect (NEWLY ADDED)
        # From "ISV variance for the placebo Emax"
        var_pbo = 0.0021
        eta_pbo = np.random.normal(0, np.sqrt(var_pbo), n_sim)
        
        # --- SIMULATE EACH COMPONENT ---
        
        # 1. Simulate Baseline FEV1 for each trial
        baseline_simmed = np.array([self.baseline_fev1(n_arm, disease, exacerbation, age) 
                                    for disease, exacerbation, age in zip(disease_dist, exacerbation_dist, age_dist)])
        
        # 2. Simulate Drug Effect for each trial
        ref_drug_effect_simmed = np.array([self.drug_effect(baseline, bd_dose, ff_dose, gp_dose, eta_ai_val, eta_bd_val)
                                        for baseline, eta_ai_val, eta_bd_val in zip(baseline_simmed, eta_ai, eta_bd)])
        
        # 3. Simulate Disease Progression for each trial (NOW WITH ISV)
        disease_prog_simmed = np.array([self.disease_progression(baseline, trial_duration_weeks, eta_dp_val) 
                                        for baseline, eta_dp_val in zip(baseline_simmed, eta_dp)])

        # 4. Calculate Placebo Effect for each trial (NOW WITH ISV)
        placebo_eff_simmed = np.array([self.placebo_effect(trial_duration_weeks, eta_pbo_val)
                                    for eta_pbo_val in eta_pbo])

        # 5. Simulate Residual Error for each trial
        RSR = [self.residual_w(n_arm, self.residual_var) for _ in range(n_sim)]
        ref_residual_simmed = np.array([x['Resid_i'] for x in RSR])

        # --- COMBINE ALL COMPONENTS FOR FINAL FEV1 ---
        # Final FEV1 = Baseline + Drug Effect + Disease Progression + Placebo + Residual Error
        final_fev1_simmed = (baseline_simmed + 
                            ref_drug_effect_simmed + 
                            disease_prog_simmed + 
                            placebo_eff_simmed + 
                            ref_residual_simmed) - baseline_simmed
        
        # Convert to mL for output consistency with original code
        estimates_simmed_ml = final_fev1_simmed * 1000

        # Calculate confidence interval bounds
        refsd = np.array([np.std(x['res_person']) for x in RSR])
        pooled_sd = np.sqrt(np.mean((n_arm - 1) * refsd ** 2) / (n_arm - 1))
        bound_length = pooled_sd * np.sqrt(1/n_arm) * norm.ppf(0.975) * 1000  # in mL
        
        outcome = pd.DataFrame({
            'est': estimates_simmed_ml,
            'low': estimates_simmed_ml - bound_length,
            'high': estimates_simmed_ml + bound_length,
            'ntrials': np.arange(1, n_sim + 1)
        })
        
        return [np.mean(baseline_simmed), outcome['est'].mean(), outcome['low'].mean(), outcome['high'].mean(), outcome['est'].min(), outcome['est'].max()]


    def calculate_effects(self,n_arm=204, n_sim=1, ratio = [1,1,1], trial_duration_weeks=24, bd_dose=320, gp_dose=14.4, ff_dose=9.6):
        """
        Runs a full trial simulation, incorporating all model components to predict
        the final FEV1 at the end of the trial duration.
        """
        
        # Distributions for patient characteristics
        disease_dist = np.random.choice([1, 2, 3, 4], size=n_sim)
        exacerbation_dist = np.random.choice([0, 1, 2, 3, 4], size=n_sim)
        
        #disease_dist = np.random.choice([2, 4], size=n_sim)
        #exacerbation_dist = np.random.choice([0, 0], size=n_sim)
        
        #age_dist = np.random.normal(65, 7.5, size=n_sim)
        age_dist = np.random.normal(61.8, 8.5, size=n_sim)   # PT009001
        
        # --- Inter-study variabilities (ISV) from Table S2 ---
        # ISV for drug effects
        isv_ai_cv = 0.405 # CV for anti-inflammatory
        isv_bd_cv = 0.185 # CV for bronchodilator
        # Variance for log-normal is log(CV^2+1)
        eta_ai = np.random.normal(0, np.sqrt(np.log(isv_ai_cv**2 + 1)), n_sim)
        eta_bd = np.random.normal(0, np.sqrt(np.log(isv_bd_cv**2 + 1)), n_sim)
        
        # ISV for disease progression (NEWLY ADDED)
        # From "Log of ISV CV for the disease progression slope" = -0.611
        isv_dp_cv = np.exp(-0.611) 
        var_dp = np.log(isv_dp_cv**2 + 1)
        eta_dp = np.random.normal(0, np.sqrt(var_dp), n_sim)
        
        # ISV for placebo effect (NEWLY ADDED)
        # From "ISV variance for the placebo Emax"
        var_pbo = 0.0021
        eta_pbo = np.random.normal(0, np.sqrt(var_pbo), n_sim)
        
        # --- SIMULATE EACH COMPONENT ---
        
        # 1. Simulate Baseline FEV1 for each trial
        baseline_simmed = np.array([self.baseline_fev1(n_arm, disease, exacerbation, age) 
                                    for disease, exacerbation, age in zip(disease_dist, exacerbation_dist, age_dist)])
        
        # 2. Simulate Drug Effect for each trial
        ref_drug_effect_simmed = np.array([self.drug_effect(baseline, bd_dose, ff_dose, gp_dose, eta_ai_val, eta_bd_val)
                                        for baseline, eta_ai_val, eta_bd_val in zip(baseline_simmed, eta_ai, eta_bd)])[0]
        if self.GMR_ONE_FLAG:
            bd_ratio, ff_ratio, gp_ratio = 1, 1, 1 #ratio[0], ratio[1], ratio[2]   # GMR = 1     
        else:
            bd_ratio, ff_ratio, gp_ratio = ratio[0], ratio[1], ratio[2]
        
        

        test_drug_effect_simmed = np.array([self.drug_effect(baseline, 320 * bd_ratio, 9.6 * ff_ratio, 14.4 * gp_ratio, eta_ai_val, eta_bd_val)
                                        for baseline, eta_ai_val, eta_bd_val in zip(baseline_simmed, eta_ai, eta_bd)])[0]
        # 3. Simulate Disease Progression for each trial (NOW WITH ISV)
        disease_prog_simmed = np.array([self.disease_progression(baseline, trial_duration_weeks, eta_dp_val) 
                                        for baseline, eta_dp_val in zip(baseline_simmed, eta_dp)])

        # 4. Calculate Placebo Effect for each trial (NOW WITH ISV)
        placebo_eff_simmed = np.array([self.placebo_effect(trial_duration_weeks, eta_pbo_val)
                                    for eta_pbo_val in eta_pbo])

        # 5. Simulate Residual Error for each trial
        # Sample residual differently for T & R. Alternative is the same, as all other variability
        
        RSR = self.residual_with_error(arm_size = n_arm, add_error=self.residual_var )['Resid_i'].values[0] #/ np.sqrt(n_arm) # take the mean is correct statistics -> std error
        TSR = self.residual_with_error(arm_size = n_arm, add_error=self.residual_var )['Resid_i'].values[0] #/ np.sqrt(n_arm)

        # --- COMBINE ALL COMPONENTS FOR FINAL FEV1 ---
        # Final FEV1 = Baseline + Drug Effect + Disease Progression + Placebo + Residual Error
        ref_final_fev1_simmed = (baseline_simmed + 
                            ref_drug_effect_simmed + 
                            disease_prog_simmed + 
                            placebo_eff_simmed + 
                            RSR) - baseline_simmed
        test_final_fev1_simmed = (baseline_simmed + 
                            test_drug_effect_simmed + 
                            disease_prog_simmed + 
                            placebo_eff_simmed + 
                            TSR) - baseline_simmed
        
        
        return test_final_fev1_simmed, ref_final_fev1_simmed

    def calculate_effects_ORG(self,n_arm=204,  n_sim=1, ratio = [1,1,1]):

        bd_ratio, ff_ratio, gp_ratio = ratio[0], ratio[1], ratio[2]
        disease_dist = np.random.choice([1, 2, 3, 4], size=n_sim)
        exacerbation_dist = np.random.choice([0, 1, 2, 3, 4], size=n_sim)
        age_dist = np.random.normal(63.6, 3, size=n_sim)
        
        isv_ai = 0.405
        eta_ai = np.random.normal(0, np.sqrt(isv_ai), n_sim)
        isv_bd = 0.185
        eta_bd = np.random.normal(0, np.sqrt(isv_bd), n_sim)
        
        baseline_simmed = np.array([self.baseline_fev1(n_arm, disease, exacerbation, age) 
                                    for disease, exacerbation, age in zip(disease_dist, exacerbation_dist, age_dist)])
        
        ref_drug_effect_simmed = np.array([self.drug_effect(baseline, 320, 9.6, 14.4, eta_ai_val, eta_bd_val)
                                            for baseline, eta_ai_val, eta_bd_val in zip(baseline_simmed, eta_ai, eta_bd)])[0]
        
        test_drug_effect_simmed = np.array([self.drug_effect(baseline, 320 * bd_ratio, 9.6 * ff_ratio, 14.4 * gp_ratio, eta_ai_val, eta_bd_val)
                                            for baseline, eta_ai_val, eta_bd_val in zip(baseline_simmed, eta_ai, eta_bd)])[0]
        
        
        # Sample residual differently for T & R. Alternative is the same, as all other variability
        
        RSR = self.residual_with_error(arm_size = n_arm, add_error=self.residual_var )['Resid_i'].values[0] #/ np.sqrt(n_arm) # take the mean is correct statistics -> std error
        TSR = self.residual_with_error(arm_size = n_arm, add_error=self.residual_var )['Resid_i'].values[0] #/ np.sqrt(n_arm)
        
        #RSR = np.std(residual(n_arm)['res_person']) / np.sqrt(n_arm)
        #TSR = np.std(residual(n_arm)['res_person']) / np.sqrt(n_arm)
        
        return test_drug_effect_simmed + TSR, ref_drug_effect_simmed + RSR
        #return test_drug_effect_simmed + TSR, ref_drug_effect_simmed + TSR

    def run_trial_deposition_dist(self,n_arm=204, n_sim=200,  bd_ratio_arm = np.ones(204), ff_ratio_arm = np.ones(204), gp_ratio_arm = np.ones(204)):
        disease_dist = np.random.choice([1, 2, 3, 4], size=n_sim)
        exacerbation_dist = np.random.choice([0, 1, 2, 3, 4], size=n_sim)
        age_dist = np.random.normal(63.6, 3, size=n_sim)
        
        isv_ai = 0.405
        eta_ai = np.random.normal(0, np.sqrt(isv_ai), n_sim)
        isv_bd = 0.185
        eta_bd = np.random.normal(0, np.sqrt(isv_bd), n_sim)
        
        baseline_simmed = np.array([self.baseline_fev1(n_arm, disease, exacerbation, age) 
                                    for disease, exacerbation, age in zip(disease_dist, exacerbation_dist, age_dist)])
        
        ref_drug_effect_simmed = np.array([self.drug_effect(baseline, 320, 9.6, 14.4, eta_ai_val, eta_bd_val)
                                            for baseline, eta_ai_val, eta_bd_val in zip(baseline_simmed, eta_ai, eta_bd)])
        
        #test_drug_effect_simmed = np.array([drug_effect(baseline, 320 * bd_ratio, 9.6 * ff_ratio, 14.4 * gp_ratio, eta_ai_val, eta_bd_val)
        #                                     for baseline, eta_ai_val, eta_bd_val in zip(baseline_simmed, eta_ai, eta_bd)]) 
        #test_drug_effect_simmed = np.array([drug_effect(baseline, 320 * bd_ratio, 9.6 * ff_ratio, 14.4 * gp_ratio, eta_ai_val, eta_bd_val)
        #                                    for bd_ratio, ff_ratio, gp_ratio, baseline, eta_ai_val, eta_bd_val in zip(bd_ratio_arm, ff_ratio_arm, gp_ratio_arm, baseline_simmed, eta_ai, eta_bd)]) 
        
        test_drug_effect_simmed = np.array([stats.gmean([self.drug_effect(baseline, 320 * bd_ratio, 9.6 * ff_ratio, 14.4 * gp_ratio, eta_ai_val, eta_bd_val)
                                        for bd_ratio, ff_ratio, gp_ratio in zip(bd_ratio_arm, ff_ratio_arm, gp_ratio_arm)]) for baseline, eta_ai_val, eta_bd_val in zip(baseline_simmed, eta_ai, eta_bd)]) 
        
        RSR = [self.residual(n_arm, self.residual_var) for _ in range(n_sim)]
        TSR = [self.residual(n_arm, self.residual_var) for _ in range(n_sim)]

        ref_residual_simmed = np.array([x['Resid_i'].values[0] for x in RSR])
        test_residual_simmed = np.array([x['Resid_i'].values[0] for x in TSR])
        
        estimates_simmed = ((ref_drug_effect_simmed + ref_residual_simmed) - (test_drug_effect_simmed + test_residual_simmed)) * 1000  # in mL

        refsd = np.array([np.std(x['res_person']) for x in RSR])
        testsd = np.array([np.std(x['res_person']) for x in TSR])
        pooled_sd = np.sqrt(((n_arm - 1) * refsd ** 2 + (n_arm - 1) * testsd ** 2) / (n_arm + n_arm - 2))
        bound_length = pooled_sd * np.sqrt(1/n_arm + 1/n_arm) * norm.ppf(0.975) * 1000  # Using norm.ppf for 95% CI bounds
        
        outcome = pd.DataFrame({
            'est': estimates_simmed,
            'low': estimates_simmed - bound_length,
            'high': estimates_simmed + bound_length,
            'ntrials': np.arange(1, n_sim + 1)
        })
        
        return outcome
