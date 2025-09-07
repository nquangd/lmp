import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import t
from multiprocessing import Pool

class BioequivalenceAssessor:
    """
    A class to handle bioequivalence (BE) statistical assessments, with parallel trial-level bootstrapping.
    """
    def __init__(self, study, residual_var = 0.013, inner_boot=100, alpha_pk=0.1, alpha_pd=0.1, alpha_metrics=0.05, seed=1000):
        self.n_trials = study.n_trials
        self.trial_size = study.trial_size
        self.inner_boot = inner_boot
        self.alpha_metrics = alpha_metrics
        self.alpha_pk = alpha_pk
        self.alpha_pd = alpha_pd
        self.non_parametric_pd_flag = False
        self.non_parametric_pk_flag = False
        self.seed = seed

    @staticmethod
    def bootstrap_ci(data, func, n_bootstrap=100, alpha=0.1):
        n = len(data)
        bootstraps = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            bootstraps.append(func(sample))
        bootstraps = np.array(bootstraps)
        lower = np.percentile(bootstraps, 100*alpha/2)
        upper = np.percentile(bootstraps, 100*(1-alpha/2))
        return func(data), lower, upper

    @staticmethod
    def geometric_mean(x):
        x = np.array(x)
        x = x[x > 0]
        if len(x) == 0: return np.nan
        return np.exp(np.mean(np.log(x)))

    @staticmethod
    def ratio_and_ci(test, ref, func, n_bootstrap=100, alpha=0.1, non_parametric=True):
        if non_parametric:
            if len(test)==0 or len(ref)==0 or np.all(ref==0): return np.nan, np.nan, np.nan
            logdiff = np.log(test) - np.log(ref)
            gmr = np.exp(np.mean(logdiff))
            n = len(logdiff)
            boot_gmr = []
            for _ in range(n_bootstrap):
                idx_boot = np.random.choice(np.arange(n), size=n, replace=True)
                boot_ldiff = logdiff[idx_boot]
                boot_gmr.append(np.exp(np.mean(boot_ldiff)))
            boot_gmr = np.array(boot_gmr)
            delta = boot_gmr - gmr
            lo = np.nanpercentile(delta, 100*alpha/2)
            hi = np.nanpercentile(delta, 100*(1-alpha/2))
            lower_adj = gmr - hi
            upper_adj = gmr - lo
            return gmr, lower_adj, upper_adj
        else:
            if len(test) == 0 or len(ref) == 0 or np.all(ref == 0): return np.nan, np.nan, np.nan
            logdiff = np.log(test) - np.log(ref)
            n = len(logdiff)
            mean_ld = np.mean(logdiff)
            std_ld = np.std(logdiff, ddof=1)
            se = std_ld / np.sqrt(n)
            t_crit = t.ppf(1-alpha/2, n-1)
            ci_lo = mean_ld - t_crit*se
            ci_hi = mean_ld + t_crit*se
            gmr = np.exp(mean_ld)
            lower = np.exp(ci_lo)
            upper = np.exp(ci_hi)
            return gmr, lower, upper

    @staticmethod
    def bootstrap_ci_fixed_effects(effects, lowers, uppers, alpha=0.1, logscale=False):
        effects = np.asarray(effects)
        lowers = np.asarray(lowers)
        uppers = np.asarray(uppers)
        if logscale:
            effects = np.log(effects)
            lowers = np.log(lowers)
            uppers = np.log(uppers)
        z = 1.96 if alpha==0.05 else 1.645
        variances = ((uppers-lowers)/(2*z))**2
        variances = np.clip(variances, 1e-12, None)
        weights = 1/variances
        theta_FE = np.sum(weights*effects)/np.sum(weights)
        se_FE = np.sqrt(1/np.sum(weights))
        ci_low = theta_FE - z*se_FE
        ci_upp = theta_FE + z*se_FE
        if logscale:
            return np.exp(theta_FE), np.exp(ci_low), np.exp(ci_upp)
        else:
            return theta_FE, ci_low, ci_upp

    @staticmethod
    def cv_percent(x):
        x = np.array(x)
        m = np.mean(x)
        s = np.std(x, ddof=1)
        if m == 0: return np.nan
        return (s/m)*100

    @staticmethod
    def cv_geometric_percent(x):
        x = np.array(x)
        x = x[x>0]
        if len(x) == 0: return np.nan
        log_x = np.log(x)
        s = np.std(log_x, ddof=1)
        return (np.exp(s)-1)*100

    @staticmethod
    def holistic_effect_ratio_and_ci(
        all_test_arrays, all_ref_arrays, func, calculate_effects, residual_var, n_bootstrap=100, alpha=0.1, non_parametric = True):
        api_order = ['BD', 'FF', 'GP']  # to match the order of BD, FF, GP in the effects calculation func
        apis = [api for api in api_order if api in all_test_arrays.keys()]
        ratios = []
        for api in api_order:
            test_vals = all_test_arrays[api]
            ref_vals = all_ref_arrays[api]
            ratio = func(test_vals) / func(ref_vals) if len(ref_vals) > 0 and func(ref_vals)!=0 else np.nan
            ratios.append(ratio)
        effect_test, effect_ref = calculate_effects(n_arm = len(ref_vals) , n_sim = 1, ratio = ratios)
        effect_point_est = effect_test / effect_ref

        
        if non_parametric:
            nsub = len(ref_vals)
            # Bootstrap
            effect_ratios = []
            for _ in range(n_bootstrap):
                idx_boot = np.random.choice(np.arange(nsub), size=nsub, replace=True)
                
                boot_test_arrays = {
                    api: all_test_arrays[api][idx_boot]
                    for api in api_order
                }
                boot_ref_arrays = {
                    api: all_ref_arrays[api][idx_boot]
                    for api in api_order
                }
                boot_ratios = []
                skip = False
                for api in api_order:
                    if len(boot_ref_arrays[api]) == 0 or func(boot_ref_arrays[api]) == 0:
                        skip = True
                        break
                    boot_ratios.append(func(boot_test_arrays[api]) / func(boot_ref_arrays[api]))
                if skip:
                    continue
                e_test, e_ref = calculate_effects(n_arm = len(ref_vals), n_sim = 1, ratio = boot_ratios)
                effect_ratios.append(e_test / e_ref)
            effect_ratios = np.array(effect_ratios)
            # Classic bootstrap CI
            delta = effect_ratios - effect_point_est
            lo = np.nanpercentile(delta, 100*alpha/2)
            hi = np.nanpercentile(delta, 100*(1-alpha/2))
            
            lower_adj = effect_point_est - hi
            upper_adj = effect_point_est - lo
            return effect_point_est, lower_adj, upper_adj

        else:

            log_gmr_observed = np.log(effect_test) - np.log(effect_ref)
            n_arm = len(ref_vals)
            refsd = np.std(np.log(residual_with_error(arm_size=n_arm, add_error=residual_var)['res_person']))
            testsd = np.std(np.log(residual_with_error(arm_size=n_arm, add_error=residual_var)['res_person']))
            pooled_var = ((n_arm - 1) * refsd ** 2 + (n_arm - 1) * testsd ** 2) / (n_arm + n_arm - 2)
            se_diff = np.sqrt(pooled_var * (1/n_arm + 1/n_arm))
                # Find the critical t-value for the 90% confidence interval
            #alpha = 1 - CONFIDENCE_LEVEL
            t_val = stats.t.ppf(1 - alpha / 2, df=(n_arm + n_arm - 2))

            # Calculate the confidence interval on the log scale
            margin_of_error = t_val * se_diff
            lower_ci_log = log_gmr_observed - margin_of_error
            upper_ci_log = log_gmr_observed + margin_of_error

            # --- 3. Convert Results Back to Original Scale ---
            gmr_observed = np.exp(log_gmr_observed)
            lower_ci = np.exp(lower_ci_log)
            upper_ci = np.exp(upper_ci_log)
            return gmr_observed, lower_ci, upper_ci
    


    @staticmethod
    def _systemic_bioequiv_single_trial(args):
        df, trial_size, inner_boot, alpha_pk, non_parametric_pk_flag, seed = args
        np.random.seed(seed)  # for reproducibility per worker
        systemic = df[['Subject', 'Product', 'API', 'Systemic_AUC', 'Systemic_Cmax']].drop_duplicates()
        sampled_subjects = np.random.choice(systemic['Subject'].unique(), size=trial_size, replace=True)
        sys_sample = systemic[systemic['Subject'].isin(sampled_subjects)]
        trial_rows = []
        for api, gdf in sys_sample.groupby('API'):
            dfp = pd.merge(
                gdf[(gdf['API'] == api) & (gdf['Product'] == 'Test')][['Subject','Systemic_AUC', 'Systemic_Cmax']],
                gdf[(gdf['API'] == api) & (gdf['Product'] == 'Reference')][['Subject','Systemic_AUC', 'Systemic_Cmax']],
                left_on='Subject', right_on='Subject', how='inner'
            )
            dfp.columns = ['Subject', 'Systemic_AUC_Test', 'Systemic_Cmax_Test', 'Systemic_AUC_Reference', 'Systemic_Cmax_Reference']
            test = dfp[['Subject', 'Systemic_AUC_Test', 'Systemic_Cmax_Test']]
            test.columns = ['Subject', 'Systemic_AUC', 'Systemic_Cmax']
            ref = dfp[['Subject', 'Systemic_AUC_Reference', 'Systemic_Cmax_Reference']]
            ref.columns = ['Subject', 'Systemic_AUC', 'Systemic_Cmax']

            auc_m, auc_m_low, auc_m_up = BioequivalenceAssessor.ratio_and_ci(
                test['Systemic_AUC'].values, ref['Systemic_AUC'].values,
                np.mean, inner_boot, alpha=alpha_pk, non_parametric=non_parametric_pk_flag
            )
            auc_g, auc_g_low, auc_g_up = BioequivalenceAssessor.ratio_and_ci(
                test['Systemic_AUC'].values, ref['Systemic_AUC'].values,
                BioequivalenceAssessor.geometric_mean, inner_boot, alpha=alpha_pk, non_parametric=non_parametric_pk_flag
            )
            cmax_m, cmax_m_low, cmax_m_up = BioequivalenceAssessor.ratio_and_ci(
                test['Systemic_Cmax'].values, ref['Systemic_Cmax'].values,
                np.mean, inner_boot, alpha=alpha_pk, non_parametric=non_parametric_pk_flag
            )
            cmax_g, cmax_g_low, cmax_g_up = BioequivalenceAssessor.ratio_and_ci(
                test['Systemic_Cmax'].values, ref['Systemic_Cmax'].values,
                BioequivalenceAssessor.geometric_mean, inner_boot, alpha=alpha_pk, non_parametric=non_parametric_pk_flag
            )
            trial_rows.append({'API': api, 'Parameter': 'AUC', 'Mean_Ratio': auc_m, 'Mean_CI_Lower': auc_m_low, 'Mean_CI_Upper': auc_m_up, 'GeoMean_Ratio': auc_g, 'GeoMean_CI_Lower': auc_g_low, 'GeoMean_CI_Upper': auc_g_up})
            trial_rows.append({'API': api, 'Parameter': 'Cmax', 'Mean_Ratio': cmax_m, 'Mean_CI_Lower': cmax_m_low, 'Mean_CI_Upper': cmax_m_up, 'GeoMean_Ratio': cmax_g, 'GeoMean_CI_Lower': cmax_g_low, 'GeoMean_CI_Upper': cmax_g_up})
        return trial_rows

    def systemic_bioequiv_parallel(self, df, pool_cores=8):
        """
        Distributes bootstrap trials in parallel using Pool.
        Returns: summary_df, trial_df
        """
        args_list = [
            (df.copy(), self.trial_size, self.inner_boot, self.alpha_pk, self.non_parametric_pk_flag, self.seed+i)
            for i in range(self.n_trials)
        ]
        with Pool(processes=pool_cores) as pool:
            results = pool.map(BioequivalenceAssessor._systemic_bioequiv_single_trial, args_list)

        # Flatten and index trials
        trial_rows = []
        for trial_idx, rows in enumerate(results):
            for row in rows:
                r = row.copy()
                r['Trial'] = trial_idx
                trial_rows.append(r)
        trial_df = pd.DataFrame(trial_rows)

        # Summary aggregation as before
        summary = []
        for key, g in trial_df.groupby(['API', 'Parameter']):
            vals, geo_vals = g['Mean_Ratio'].values, g['GeoMean_Ratio'].values
            mean, mean_low, mean_up = self.bootstrap_ci_fixed_effects(vals, g['Mean_CI_Lower'].values, g['Mean_CI_Upper'].values, alpha=self.alpha_pk, logscale=True)
            geo, geo_low, geo_up = self.bootstrap_ci_fixed_effects(geo_vals, g['GeoMean_CI_Lower'].values, g['GeoMean_CI_Upper'].values, alpha=self.alpha_pk, logscale=True)
            summary.append({'API': key[0], 'Parameter': key[1], 'Mean_Ratio': mean, 'Mean_CI_Lower': mean_low, 'Mean_CI_Upper': mean_up, 'GeoMean_Ratio': geo, 'GeoMean_CI_Lower': geo_low, 'GeoMean_CI_Upper': geo_up})
        summary_df = pd.DataFrame(summary)
        return summary_df, trial_df



    @staticmethod
    def _systemic_product_metrics_single_trial(args):
        df, trial_size, inner_boot, alpha_metrics, seed = args
        np.random.seed(seed)
        systemic = df[['Subject', 'Product', 'API', 'Systemic_AUC', 'Systemic_Cmax', 'ET', 'C/P']].drop_duplicates()
        systemic['Total_LungDeposition'] = 1 - systemic['ET']
        products, apis = systemic['Product'].unique(), systemic['API'].unique()
        parameters = ['Systemic_AUC', 'Systemic_Cmax', 'Total_LungDeposition', 'C/P']
        trial_rows = []
        sampled_subjects = np.random.choice(systemic['Subject'].unique(), size=trial_size, replace=True)
        sys_sample = systemic[systemic['Subject'].isin(sampled_subjects)]
        for product in products:
            for api in apis:
                for param in parameters:
                    values = sys_sample[(sys_sample['Product'] == product) & (sys_sample['API'] == api)][param].values
                    val_mean, mean_low, mean_up = BioequivalenceAssessor.bootstrap_ci(values, np.mean, n_bootstrap=inner_boot, alpha=alpha_metrics)
                    val_geo, geo_low, geo_up = BioequivalenceAssessor.bootstrap_ci(values, BioequivalenceAssessor.geometric_mean, n_bootstrap=inner_boot, alpha=alpha_metrics)
                    trial_rows.append({
                        'Product': product, 'API': api,
                        'Parameter': 'AUC' if param == 'Systemic_AUC' else 'Cmax' if param == 'Systemic_Cmax' else param,
                        'Mean': val_mean, 'Mean_CI_Lower': mean_low, 'Mean_CI_Upper': mean_up,
                        'GeoMean': val_geo, 'GeoMean_CI_Lower': geo_low, 'GeoMean_CI_Upper': geo_up,
                        'CV_percent': BioequivalenceAssessor.cv_percent(values),
                        'CV_Geo_percent': BioequivalenceAssessor.cv_geometric_percent(values)
                    })
        return trial_rows

    def systemic_product_metrics_parallel(self, df, pool_cores=8):
        args_list = [(df, self.trial_size, self.inner_boot, self.alpha_metrics, self.seed + i) for i in range(self.n_trials)]
        with Pool(processes=pool_cores) as pool:
            results = pool.map(BioequivalenceAssessor._systemic_product_metrics_single_trial, args_list)

        # Flatten and index trials
        trial_rows = []
        for trial_idx, rows in enumerate(results):
            for row in rows:
                r = row.copy()
                r['Trial'] = trial_idx
                trial_rows.append(r)
        trial_df = pd.DataFrame(trial_rows)

        # Summary aggregation as before
        summary = []
        for (product, api, param), g in trial_df.groupby(['Product', 'API', 'Parameter']):
            cv, cv_low, cv_up = self.bootstrap_ci(g['CV_percent'].values, np.mean, alpha=self.alpha_metrics)
            cvgeo, cvgeo_low, cvgeo_up = self.bootstrap_ci(g['CV_Geo_percent'].values, np.mean, alpha=self.alpha_metrics)
            mean, mean_low, mean_up = self.bootstrap_ci_fixed_effects(
                g['Mean'].values, g['Mean_CI_Lower'].values, g['Mean_CI_Upper'].values, alpha=self.alpha_metrics, logscale=False
            )
            geo, geo_low, geo_up = self.bootstrap_ci_fixed_effects(
                g['GeoMean'].values, g['GeoMean_CI_Lower'].values, g['GeoMean_CI_Upper'].values, alpha=self.alpha_metrics, logscale=False
            )
            summary.append({
                'Product': product, 'API': api, 'Parameter': param,
                'Mean': mean, 'Mean_CI_Lower': mean_low, 'Mean_CI_Upper': mean_up,
                'GeoMean': geo, 'GeoMean_CI_Lower': geo_low, 'GeoMean_CI_Upper': geo_up,
                'CV_percent': cv, 'CV_percent_CI_Lower': cv_low, 'CV_percent_CI_Upper': cv_up,
                'CV_Geo_percent': cvgeo, 'CV_Geo_CI_Lower': cvgeo_low, 'CV_Geo_CI_Upper': cvgeo_up
            })
        summary_df = pd.DataFrame(summary)
        return summary_df, trial_df

    # -------------------------------------------------------------------------------------------------

    @staticmethod
    def _regional_bioequiv_single_trial(args):
        df, trial_size, inner_boot, alpha_pk, alpha_pd, non_parametric_pk_flag, non_parametric_pd_flag, residual_var, seed, calculate_effects = args
        np.random.seed(seed)

        df = df.dropna(subset=['Regional_AUC', 'Regional_AUC'])
        df = df.dropna(subset=['Regional Deposition', 'Regional Deposition'])
        required_apis = {"BD", "GP", "FF"}
        required_prod = {"Test", "Reference"}

        subject_api_sets = df.groupby("Subject")["API"].agg(set)
        subjects_with_all_apis = subject_api_sets[subject_api_sets.apply(lambda apis: required_apis.issubset(apis))].index
        subject_prod_sets = df.groupby("Subject")["Product"].agg(set)
        subjects_with_all_products = subject_prod_sets[subject_prod_sets.apply(lambda prods: required_prod.issubset(prods))].index
        eligible_subjects = set(subjects_with_all_apis).intersection(subjects_with_all_products)
        df = df[df["Subject"].isin(eligible_subjects)]
        subjects = df['Subject'].unique()
        apis = sorted(df['API'].unique())

        trial_rows = []
        sampled_subjects = np.random.choice(subjects, size=trial_size, replace=True)
        df_sample = df[df['Subject'].isin(sampled_subjects)]
        df_sample = df_sample.sort_values(['Subject'])

        # Regional_AUC
        for (region, compartment), group_df in df_sample.groupby(['Region', 'Compartment']):
            all_test, all_ref  = {}, {}
            for api in apis:
                dfp = pd.merge(
                    group_df[(group_df['API']==api)&(group_df['Product']=='Test')][['Subject','Regional_AUC']],
                    group_df[(group_df['API']==api)&(group_df['Product']=='Reference')][['Subject','Regional_AUC']],
                    left_on='Subject', right_on='Subject', how = 'inner'
                )
                dfp.columns = ['Subject', 'Test', 'Reference']
                all_test[api] = dfp['Test'].values
                all_ref[api] = dfp['Reference'].values
                tvals, rvals = all_test[api], all_ref[api]
                m, m_low, m_up = BioequivalenceAssessor.ratio_and_ci(tvals, rvals, np.mean, inner_boot, alpha=alpha_pk, non_parametric=non_parametric_pk_flag)
                g, g_low, g_up = BioequivalenceAssessor.ratio_and_ci(tvals, rvals, BioequivalenceAssessor.geometric_mean, inner_boot, alpha=alpha_pk, non_parametric=non_parametric_pk_flag)
                trial_rows.append({'Region': region, 'Compartment': compartment, 'API': api, 'Parameter': 'Regional_AUC',
                                'Mean_Ratio': m, 'Mean_CI_Lower': m_low, 'Mean_CI_Upper': m_up,
                                'GeoMean_Ratio': g, 'GeoMean_CI_Lower': g_low, 'GeoMean_CI_Upper': g_up})
                e_m, e_mlow, e_mup = BioequivalenceAssessor.holistic_effect_ratio_and_ci(
                    all_test, all_ref, np.mean, calculate_effects, residual_var, inner_boot, alpha=alpha_pd, non_parametric=non_parametric_pd_flag)
                e_g, e_glow, e_gup = BioequivalenceAssessor.holistic_effect_ratio_and_ci(
                    all_test, all_ref, BioequivalenceAssessor.geometric_mean, calculate_effects, residual_var, inner_boot, alpha=alpha_pd, non_parametric=non_parametric_pd_flag)
                trial_rows.append({'Region': region, 'Compartment': compartment, 'API': 'all', 'Parameter': 'Regional_AUC_Effect',
                                'Mean_Effect': e_m, 'Mean_Effect_CI_Lower': e_mlow, 'Mean_Effect_CI_Upper': e_mup,
                                'GeoMean_Effect': e_g, 'GeoMean_Effect_CI_Lower': e_glow, 'GeoMean_Effect_CI_Upper': e_gup})

        # Deposition (compartment=ELF)
        dedup = df_sample.drop_duplicates(subset=['Subject', 'Product', 'API', 'Region'])
        dedup['DepositionDose'] = dedup['Regional Deposition'] * dedup['Dose']
        for region_val, group_df in dedup.groupby(['Region']):
            region = region_val[0] if isinstance(region_val, tuple) else region_val
            all_test, all_ref = {}, {}
            for api in apis:
                dfp = pd.merge(
                    group_df[(group_df['API']==api)&(group_df['Product']=='Test')][['Subject','DepositionDose']],
                    group_df[(group_df['API']==api)&(group_df['Product']=='Reference')][['Subject','DepositionDose']],
                    left_on='Subject', right_on='Subject', how = 'inner'
                )
                dfp.columns = ['Subject', 'Test', 'Reference']
                all_test[api] = dfp['Test'].values
                all_ref[api] = dfp['Reference'].values
                tvals, rvals = all_test[api], all_ref[api]
                m, m_low, m_up = BioequivalenceAssessor.ratio_and_ci(tvals, rvals, np.mean, inner_boot, alpha=alpha_pk, non_parametric=non_parametric_pk_flag)
                g, g_low, g_up = BioequivalenceAssessor.ratio_and_ci(tvals, rvals, BioequivalenceAssessor.geometric_mean, inner_boot, alpha=alpha_pk, non_parametric=non_parametric_pk_flag)
                trial_rows.append({'Region': region, 'Compartment': 'ELF', 'API': api, 'Parameter': 'Deposition',
                                'Mean_Ratio': m, 'Mean_CI_Lower': m_low, 'Mean_CI_Upper': m_up,
                                'GeoMean_Ratio': g, 'GeoMean_CI_Lower': g_low, 'GeoMean_CI_Upper': g_up})
                e_m, e_mlow, e_mup = BioequivalenceAssessor.holistic_effect_ratio_and_ci(
                    all_test, all_ref, np.mean, calculate_effects, residual_var, inner_boot, alpha=alpha_pd, non_parametric=non_parametric_pd_flag)
                e_g, e_glow, e_gup = BioequivalenceAssessor.holistic_effect_ratio_and_ci(
                    all_test, all_ref, BioequivalenceAssessor.geometric_mean, calculate_effects, residual_var, inner_boot, alpha=alpha_pd, non_parametric=non_parametric_pd_flag)
                trial_rows.append({'Region': region, 'Compartment': 'ELF', 'API': 'all', 'Parameter': 'Deposition_Effect',
                                'Mean_Effect': e_m, 'Mean_Effect_CI_Lower': e_mlow, 'Mean_Effect_CI_Upper': e_mup,
                                'GeoMean_Effect': e_g, 'GeoMean_Effect_CI_Lower': e_glow, 'GeoMean_Effect_CI_Upper': e_gup})
        return trial_rows

    def regional_bioequiv_parallel(self, df, pool_cores=8, residual_var=None, calculate_effects=None):
        if residual_var is None: residual_var = self.fevmodel.residual_var
        if calculate_effects is None: calculate_effects = self.fevmodel.calculate_effects
        args_list = [
            (df, self.trial_size, self.inner_boot, self.alpha_pk, self.alpha_pd,
            self.non_parametric_pk_flag, self.non_parametric_pd_flag, residual_var, self.seed + i, calculate_effects)
            for i in range(self.n_trials)
        ]
        with Pool(processes=pool_cores) as pool:
            results = pool.map(BioequivalenceAssessor._regional_bioequiv_single_trial, args_list)

        # Flatten and index trials
        trial_rows = []
        for trial_idx, rows in enumerate(results):
            for row in rows:
                r = row.copy()
                r['Trial'] = trial_idx
                trial_rows.append(r)
        trial_df = pd.DataFrame(trial_rows)

        effect_params = ['Regional_AUC_Effect', 'Deposition_Effect']
        all_params = ['Regional_AUC', 'Deposition']
        groupers = ['API', 'Region', 'Compartment', 'Parameter']
        summary = []
        for key, g in trial_df.groupby(groupers):
            row = dict(zip(groupers, key))
            if key[-1] in all_params:
                for p, c_low, c_up in [('Mean_Ratio', 'Mean_CI_Lower', 'Mean_CI_Upper'), ('GeoMean_Ratio', 'GeoMean_CI_Lower', 'GeoMean_CI_Upper')]:
                    m, m_low, m_up = self.bootstrap_ci_fixed_effects(g[p].values, g[c_low].values, g[c_up].values, alpha=self.alpha_pk, logscale=True)
                    row[p], row[c_low], row[c_up] = m, m_low, m_up
            if key[-1] in effect_params:
                for p, c_low, c_up in [('Mean_Effect', 'Mean_Effect_CI_Lower', 'Mean_Effect_CI_Upper'), ('GeoMean_Effect', 'GeoMean_Effect_CI_Lower', 'GeoMean_Effect_CI_Upper')]:
                    m, m_low, m_up = self.bootstrap_ci_fixed_effects(g[p].values, g[c_low].values, g[c_up].values, alpha=self.alpha_pd, logscale=True)
                    row[p], row[c_low], row[c_up] = m, m_low, m_up
            summary.append(row)
        summary_df = pd.DataFrame(summary)
        return summary_df, trial_df