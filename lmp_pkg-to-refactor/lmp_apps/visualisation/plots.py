import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import t
from IPython.display import clear_output

class TrialVisualizer:
    """
    A class to handle plotting of virtual trial results.
    All methods are static as they don't rely on class instance state.
    """
    @staticmethod
    def plot_systemic_trials(systemic_trials, mean_type='Mean', apis=None, parameters=('AUC', 'Cmax'), ref_interval=(0.8, 1.25), clinical = None, studycode = None, xlim=(0.6, 1.4), figsize_per_subplot=(4, 8)):
        clear_output(wait=True)
        plot_df = systemic_trials.copy()
        if apis is None: apis = plot_df['API'].unique()
        if parameters is None: parameters = plot_df['Parameter'].unique()
        plot_df = plot_df[plot_df['API'].isin(apis) & plot_df['Parameter'].isin(parameters)]

        param_palette = dict(zip(parameters, sns.color_palette("Set1", n_colors=len(parameters))))
        api_markers = ['o', '^', 's', 'D', 'X', 'P', '*', 'v', '<', '>']
        api_marker_dict = {api: api_markers[i % len(api_markers)] for i, api in enumerate(apis)}

        n_api, n_param = len(apis), len(parameters)
        ncols = n_api * n_param
        fig, axes = plt.subplots(1, ncols, figsize=(figsize_per_subplot[0] * ncols, figsize_per_subplot[1]), sharey=True)
        if ncols == 1: axes = [axes]

        mean_col = f'{mean_type}_Ratio'
        lower_col = f'{mean_type}_CI_Lower'
        upper_col = f'{mean_type}_CI_Upper'

        for ia, api in enumerate(apis):
            for ip, param in enumerate(parameters):
                idx = ia * n_param + ip
                ax = axes[idx]
                df_sp = plot_df[(plot_df['API'] == api) & (plot_df['Parameter'] == param)].sort_values('Trial').copy()
                df_sp['y'] = df_sp['Trial'].rank(method='first') - 1

                color, marker = param_palette[param], api_marker_dict[api]

                ax.errorbar(x=df_sp[mean_col], y=df_sp['y'], xerr=[df_sp[mean_col] - df_sp[lower_col], df_sp[upper_col] - df_sp[mean_col]], fmt=marker, color=color, ecolor=color, capsize=2, markersize=5, linestyle='none')
                ax.set_title(f'{api}\n{param}', fontsize=13)
                ax.set_xlabel('Ratio')
                ax.set_xlim(*xlim)
                ax.axvline(ref_interval[0], color='red', ls='dashed')
                ax.axvline(ref_interval[1], color='red', ls='dashed')
                if clinical is not None:
                    ax.axvline(clinical[studycode][param][api][1], color='black', ls='dashed', lw=1, label="Clinical CI")
                    ax.axvline(clinical[studycode][param][api][0], color='black', ls='-', lw=1, label="Clinical Obs")
                    ax.axvline(clinical[studycode][param][api][2], color='black', ls='dashed', lw=1, label="Clinical CI")
                ax.set_ylabel('Trials' if idx == 0 else '')
                ax.invert_yaxis()
                ax.grid(axis='x', linestyle=':', alpha=0.5)

        param_handles = [Line2D([0], [0], color=param_palette[p], lw=4, label=p) for p in parameters]
        api_handles = [Line2D([0], [0], color='gray', marker=api_marker_dict[a], linestyle='None', markersize=8, label=a) for a in apis]
        clinical_handle = [Line2D([0], [0], color='black',ls='dashed', lw=1, label='Clinical CI'), Line2D([0], [0], color='black', ls='-', lw=1, label='Clinical Obs'), Line2D([0], [0], color='black',ls='dashed', lw=1, label='Clinical CI')]
        plt.gcf().legend(handles=param_handles + api_handles + clinical_handle, bbox_to_anchor=(0.5, 1.04), loc='upper center', ncol=max(len(parameters), len(apis)))
        plt.suptitle("Systemic Bioequivalence Ratios by API and Endpoint", fontsize=17, y=1.10)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.show()

    @staticmethod
    def plot_regional_trials(trials_df, parameter, mean_type='Mean', apis=None, regions=None, compartment=None, ref_interval=(0.8, 1.25), xlim=(0.4, 1.6), figsize_per_subplot=(4, 8)):
        clear_output(wait=True)
        plot_df = trials_df.copy()
        plot_df = plot_df[plot_df['Parameter'] == parameter]

        if compartment is not None and 'Compartment' in plot_df.columns:
            plot_df = plot_df[plot_df['Compartment'] == compartment]
        if apis is None:
            apis = sorted(plot_df['API'].unique())
        if regions is None and 'Region' in plot_df.columns:
            regions = sorted(plot_df['Region'].unique())
        if regions is None:
            regions = [None]
        plot_df = plot_df[plot_df['API'].isin(apis)]
        if 'Region' in plot_df.columns:
            plot_df = plot_df[plot_df['Region'].isin(regions)]

        region_palette = dict(zip(regions, sns.color_palette("Set1", n_colors=len(regions))))
        api_markers = ['o', 's', '^', 'D', 'X', 'P', '*', 'v', '<', '>']
        api_marker_dict = {api: api_markers[i % len(api_markers)] for i, api in enumerate(apis)}

        n_api = len(apis)
        n_reg = len(regions)
        ncols = n_api * n_reg
        fig, axes = plt.subplots(1, ncols, figsize=(figsize_per_subplot[0]*ncols, figsize_per_subplot[1]), sharey=True)
        if ncols == 1:
            axes = [axes]

        mean_col = f'{mean_type}_Ratio'
        lower_col = f'{mean_type}_CI_Lower'
        upper_col = f'{mean_type}_CI_Upper'

        for ia, api in enumerate(apis):
            for ir, region in enumerate(regions):
                idx = ia * n_reg + ir
                ax = axes[idx]
                if 'Region' in plot_df.columns:
                    df_sp = plot_df[(plot_df['API'] == api) & (plot_df['Region'] == region)]
                else:
                    df_sp = plot_df[plot_df['API'] == api]
                df_sp = df_sp.sort_values('Trial').copy()
                df_sp['y'] = df_sp['Trial'].rank(method='first') - 1

                color = region_palette[region]
                marker = api_marker_dict[api]

                ax.errorbar(x=df_sp[mean_col], y=df_sp['y'], xerr=[df_sp[mean_col] - df_sp[lower_col], df_sp[upper_col] - df_sp[mean_col]], fmt=marker, color=color, ecolor=color, capsize=2, markersize=5, linestyle='none', label=f'{api}, {region}')
                title = f'{api}\nRegion: {region}' if 'Region' in plot_df.columns else api
                ax.set_title(title, fontsize=13)
                ax.set_xlabel('Ratio')
                ax.set_xlim(*xlim)
                ax.axvline(ref_interval[0], color='red', ls='dashed')
                ax.axvline(ref_interval[1], color='red', ls='dashed')
                ax.set_ylabel('Trials' if idx == 0 else '')
                ax.invert_yaxis()
                ax.grid(axis='x', linestyle=':', alpha=0.5)

        region_handles = [Line2D([0], [0], color=region_palette[r], lw=4, label=f'Region: {r}') for r in regions]
        api_handles = [Line2D([0], [0], color='gray', marker=api_marker_dict[a], linestyle='None', markersize=8, label=f'API: {a}') for a in apis]

        plt.gcf().legend(handles=region_handles + api_handles, bbox_to_anchor=(0.5, 1.03), loc='upper center', ncol=max(len(apis), len(regions)))
        plt.suptitle(f"{parameter} Bioequivalence Ratios by API and Region", fontsize=17, y=1.10)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.show()
        
    @staticmethod
    def plot_systemic_product_trials(trial_df, product, apis, mean_type='Mean', parameters=('AUC', 'Cmax'), ref_interval=None, clinical = None, studycode = None, xlim=None, figsize=(18, 10)):
        clear_output(wait=True)
        param_palette = dict(zip(parameters, sns.color_palette("Set1", n_colors=len(parameters))))
        api_markers = ['o', '^', 's', 'D', 'X', 'P', '*', 'v', '<', '>']
        api_marker_dict = {api: api_markers[i % len(api_markers)] for i, api in enumerate(apis)}
        
        n_api, n_param = len(apis), len(parameters)
        n_cols = n_api * n_param
        fig, axes = plt.subplots(1, n_cols, figsize=figsize, sharey=True)
        if n_cols == 1: axes = [axes]

        mean_col = f'{mean_type}'
        lower_col = mean_col + '_CI_Lower'
        upper_col = mean_col + '_CI_Upper'

        for ia, api in enumerate(apis):
            for ip, param in enumerate(parameters):
                idx = ia * n_param + ip
                ax = axes[idx]
                df_sub = trial_df[(trial_df['Product'] == product) & (trial_df['API'] == api) & (trial_df['Parameter'] == param)].sort_values('Trial')
                if df_sub.empty: continue
                
                y, vals, lowers, uppers = df_sub['Trial'], df_sub[mean_col], df_sub[lower_col], df_sub[upper_col]
                color, marker = param_palette[param], api_marker_dict[api]

                ax.errorbar(x=vals, y=y, xerr=[vals - lowers, uppers - vals], fmt=marker, color=color, ecolor=color, capsize=2, markersize=5, linestyle='none')
                ax.set_title(f'{api}\n{param}', fontsize=13)
                ax.set_xlabel(param)
                if xlim: ax.set_xlim(xlim)
                
                if clinical is not None:
                    ax.axvline(clinical[studycode][param][api][0], color='black', ls='-', lw=2, label="Clinical Obs")
                if ref_interval is not None:
                    ax.axvline(ref_interval[0], color='red', ls='dashed')
                    ax.axvline(ref_interval[1], color='red', ls='dashed')
                
                ax.set_ylabel('Trials' if idx == 0 else '')
                ax.invert_yaxis()
                ax.grid(axis='x', linestyle=':', alpha=0.4)

        param_handles = [Line2D([0], [0], color=param_palette[p], lw=4, label=p) for p in parameters]
        api_handles = [Line2D([0], [0], color='gray', marker=api_marker_dict[a], linestyle='None', markersize=8, label=a) for a in apis]
        handles = param_handles + api_handles
        if clinical is not None:
            handles.append(Line2D([0], [0], color='black', lw=1, label='Clinical Obs'))
        plt.gcf().legend(handles=handles, bbox_to_anchor=(0.5, 1.04), loc='upper center', ncol=n_cols)
        plt.suptitle(f"{product}:{parameters[0]} & {parameters[1]} across Simulations", fontsize=16, y=1.10)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.show()

    @staticmethod
    def plot_systemic_product_trials_cv(trial_df, product, apis, cv_type='CV_percent', parameters=('AUC', 'Cmax'), ref_lines=(), xlim=None, figsize=(18, 10)):
        clear_output(wait=True)
        param_palette = dict(zip(parameters, sns.color_palette("Set1", n_colors=len(parameters))))
        api_markers = ['o', '^', 's', 'D', 'X', 'P', '*', 'v', '<', '>']
        api_marker_dict = {api: api_markers[i % len(api_markers)] for i, api in enumerate(apis)}
        
        n_api, n_param = len(apis), len(parameters)
        n_cols = n_api * n_param
        fig, axes = plt.subplots(1, n_cols, figsize=figsize, sharey=True)
        if n_cols == 1: axes = [axes]

        for ia, api in enumerate(apis):
            for ip, param in enumerate(parameters):
                idx = ia * n_param + ip
                ax = axes[idx]
                df_sub = trial_df[(trial_df['Product'] == product) & (trial_df['API'] == api) & (trial_df['Parameter'] == param)].sort_values('Trial')
                if df_sub.empty: continue
                
                y, vals = df_sub['Trial'], df_sub[cv_type]
                color, marker = param_palette[param], api_marker_dict[api]

                ax.scatter(vals, y, color=color, marker=marker, s=40, alpha=0.7)
                ax.set_title(f'{api}\n{param}', fontsize=13)
                ax.set_xlabel(f"{'Geometric ' if 'Geo' in cv_type else ''}CV%")
                if xlim: ax.set_xlim(xlim)
                
                for line in ref_lines:
                    ax.axvline(line, color='red', ls='dashed', alpha=0.5)
                
                ax.set_ylabel('Trials' if idx == 0 else '')
                ax.invert_yaxis()
                ax.grid(axis='x', linestyle=':', alpha=0.4)

        param_handles = [Line2D([0], [0], color=param_palette[p], lw=4, label=p) for p in parameters]
        api_handles = [Line2D([0], [0], color='gray', marker=api_marker_dict[a], linestyle='None', markersize=8, label=a) for a in apis]
        plt.gcf().legend(handles=param_handles + api_handles, bbox_to_anchor=(0.5, 1.04), loc='upper center', ncol=n_cols)
        plt.suptitle(f"{product}: {cv_type.replace('_', ' ')} across Simulations", fontsize=16, y=1.10)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.show()

    @staticmethod
    def plot_concentration_time_profiles(processed_profiles_df, plot_type='plasma', region=None, compartment=None, products_to_plot=None, error_band='none', apis=None, log_scale=True, xlim=(0, 24), figsize_per_subplot=(6, 5)):
        """
        Plots geometric mean concentration-time profiles for one or two products.
        Can display error bands for gCV or 90% CI.
        """
        clear_output(wait=True)
        
        df = processed_profiles_df.copy()
        
        # Filter for the type of profile to plot
        if plot_type == 'plasma':
            df = df[df['ProfileType'] == 'Plasma']
            plot_title = "Plasma"
        elif plot_type == 'regional':
            if not region or not compartment:
                raise ValueError("Both 'region' and 'compartment' must be specified for regional plots.")
            df = df[(df['ProfileType'] == 'Regional') & (df['Region'] == region) & (df['Compartment'] == compartment)]
            plot_title = f"{region} - {compartment}"
        else:
            raise ValueError("plot_type must be 'plasma' or 'regional'")

        if apis is None:
            apis = sorted(df['API'].unique())
        if products_to_plot is None:
            products_to_plot = sorted(df['Product'].unique())

        df = df[df['API'].isin(apis) & df['Product'].isin(products_to_plot)]

        # Setup plot
        n_cols = len(apis)
        fig, axes = plt.subplots(1, n_cols, figsize=(figsize_per_subplot[0] * n_cols, figsize_per_subplot[1]), sharey=False)
        if n_cols == 1:
            axes = [axes]

        product_styles = {
            products_to_plot[0]: {'color': 'blue', 'linestyle': '-'},
        }
        if len(products_to_plot) > 1:
            product_styles[products_to_plot[1]] = {'color': 'red', 'linestyle': '--'}
        else:
            product_styles[products_to_plot[0]]['color'] = 'black'


        for i, api in enumerate(apis):
            ax = axes[i]
            for product in products_to_plot:
                style = product_styles[product]
                df_plot = df[(df['API'] == api) & (df['Product'] == product)]
                if df_plot.empty:
                    continue

                # Calculate geometric mean and error bands at each time point
                summary_data = []
                for time_point, group in df_plot.groupby('Time'):
                    conc = group['Concentration'].values
                    conc = conc[conc > 0]
                    if len(conc) < 2: continue
                    
                    log_conc = np.log(conc)
                    gmean = np.exp(np.mean(log_conc))
                    
                    res = {'Time': time_point, 'gmean': gmean}
                    
                    if error_band == 'gcv':
                        gstd = np.exp(np.std(log_conc))
                        res['lower'] = gmean / gstd
                        res['upper'] = gmean * gstd
                    elif error_band == 'ci':
                        n = len(conc)
                        sem_log = np.std(log_conc, ddof=1) / np.sqrt(n)
                        t_crit = t.ppf(0.95, df=n-1) # 90% CI -> 5% in each tail
                        ci_half_width = t_crit * sem_log
                        res['lower'] = np.exp(np.mean(log_conc) - ci_half_width)
                        res['upper'] = np.exp(np.mean(log_conc) + ci_half_width)
                    
                    summary_data.append(res)
                
                summary_df = pd.DataFrame(summary_data)
                if summary_df.empty: continue

                # Plotting
                ax.plot(summary_df['Time'], summary_df['gmean'], label=f'{product}', **style)
                if error_band != 'none':
                    ax.fill_between(summary_df['Time'], summary_df['lower'], summary_df['upper'], color=style['color'], alpha=0.2)

            ax.set_title(api)
            ax.set_xlabel("Time (hours)")
            if i == 0:
                ax.set_ylabel("Concentration (pg/mL)")
            if log_scale:
                ax.set_yscale('log')
            ax.set_xlim(xlim)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.legend()
            
        fig.suptitle(f"Geometric Mean {plot_title} Concentration-Time Profiles", fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()
