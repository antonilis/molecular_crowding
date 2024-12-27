import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import utils as uts
import ssDNA_hybdrization_equillibrium_constant



def adjust_dataframe():
    df = pd.read_csv('results/K_DNA-DNA_in_crowder_solutions.csv')

    df[['K', 'K_err']] = df['K'].str.split('±', expand=True)
    df[['D', 'D_err']] = df['D'].str.split('±', expand=True)
    df['K'] = pd.to_numeric(df['K'], errors='coerce')
    df['K_err'] = pd.to_numeric(df['K_err'], errors='coerce')
    df['D'] = pd.to_numeric(df['D'], errors='coerce')
    df['D_err'] = pd.to_numeric(df['D_err'], errors='coerce')
    return(df)


def K_DNA_vs_crowder_weight_percent():
    df = adjust_dataframe()
    styles = {
        'buffer': {'x': [0], 'y': [1760000000], 'yerr': [311872000], 'color': '#868788', 'marker': '*', 'ms': 9.5, 'mec': '#000000', 'label': 'buffer'},
        'EGly': {'color': '#2d642a', 'marker': '.', 'ms': 12, 'mec': '#000000', 'label': 'EGly'},
        'PEG200': {'color': '#b3daff', 'marker': 'd', 'ms': 7, 'mec': '#000000', 'label': 'PEG 200'},
        'PEG400': {'color': '#640024', 'marker': '^', 'ms': 7, 'mec': '#000000', 'label': 'PEG 400'},
        'PEG600': {'color': '#ff730f', 'marker': 'h', 'ms': 7, 'mec': '#000000', 'label': 'PEG 600'},
        'PEG1000': {'color': '#fed85d', 'marker': 'P', 'ms': 6.5, 'mec': '#000000', 'label': 'PEG 1k'},
        'PEG1500': {'color': '#142cd7', 'marker': 'p', 'ms': 7, 'mec': '#000000', 'label': 'PEG 1.5k'},
        'PEG3000': {'color': '#ac1416', 'marker': 's', 'ms': 6, 'mec': '#000000', 'label': 'PEG 3k'},
        'PEG6000': {'color': '#674ea7', 'marker': 'D', 'ms': 5.5, 'mec': '#000000', 'label': 'PEG 6k'},
        'PEG12000': {'color': '#709d74', 'marker': 'v', 'ms': 7, 'mec': '#000000', 'label': 'PEG 12k'},
        'PEG20000': {'color': '#00cccc', 'marker': 'X', 'ms': 7, 'mec': '#000000', 'label': 'PEG 20k'},
        'PEG35000': {'color': '#b28092', 'marker': 'h', 'ms': 7, 'mec': '#000000', 'label': 'PEG 35k'}
    }

    plt.figure(figsize=(7, 5))

    # Plot data for each 'crowder'
    for crowder, style in styles.items():
        if crowder == 'buffer':
            plt.errorbar(style['x'], style['y'], yerr=style['yerr'], color=style['color'], marker=style['marker'],
                         mec=style['mec'], linestyle='none', lw=1, ms=style['ms'], elinewidth=1, capsize=3, capthick=1, label=style['label'])
        else:
            subset = df[df['crowder'] == crowder]
            plt.errorbar(subset['wt_%'], subset['K'], yerr=subset['K_err'], color=style['color'], marker=style['marker'],
                         mec=style['mec'], linestyle='none', lw=1, ms=style['ms'], elinewidth=1, capsize=3, capthick=1, label=style['label'])

    # Plot formatting
    plt.title('$K_{DNA-DNA}$ vs crowder wt.% change', fontsize=18)
    plt.xlabel('crowder wt.%', fontsize=16)
    plt.ylabel('$K_{DNA-DNA}$ [M$^{-1}$]', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.yscale('log')
    plt.xlim(-2, 52)
    plt.legend(frameon=True, loc='lower left', fontsize=8)
    plt.savefig('plots/K_DNA_vs_crowder_weight_percent.png', bbox_inches='tight', transparent=True, dpi=300)
    return plt.show()


def D_DNA_vs_crowder_weight_percent():
    df = adjust_dataframe()
    styles = {
        'buffer': {'x': [0], 'y': [151], 'yerr': [2.1], 'color': '#868788', 'marker': '*', 'ms': 9.5, 'mec': '#000000', 'label': 'buffer'},
        'EGly': {'color': '#2d642a', 'marker': '.', 'ms': 12, 'mec': '#000000', 'label': 'EGly'},
        'PEG200': {'color': '#b3daff', 'marker': 'd', 'ms': 7, 'mec': '#000000', 'label': 'PEG 200'},
        'PEG400': {'color': '#640024', 'marker': '^', 'ms': 7, 'mec': '#000000', 'label': 'PEG 400'},
        'PEG600': {'color': '#ff730f', 'marker': 'h', 'ms': 7, 'mec': '#000000', 'label': 'PEG 600'},
        'PEG1000': {'color': '#fed85d', 'marker': 'P', 'ms': 6.5, 'mec': '#000000', 'label': 'PEG 1k'},
        'PEG1500': {'color': '#142cd7', 'marker': 'p', 'ms': 7, 'mec': '#000000', 'label': 'PEG 1.5k'},
        'PEG3000': {'color': '#ac1416', 'marker': 's', 'ms': 6, 'mec': '#000000', 'label': 'PEG 3k'},
        'PEG6000': {'color': '#674ea7', 'marker': 'D', 'ms': 5.5, 'mec': '#000000', 'label': 'PEG 6k'},
        'PEG12000': {'color': '#709d74', 'marker': 'v', 'ms': 7, 'mec': '#000000', 'label': 'PEG 12k'},
        'PEG20000': {'color': '#00cccc', 'marker': 'X', 'ms': 7, 'mec': '#000000', 'label': 'PEG 20k'},
        'PEG35000': {'color': '#b28092', 'marker': 'h', 'ms': 7, 'mec': '#000000', 'label': 'PEG 35k'}
    }

    plt.figure(figsize=(7, 5))

    # Plot data for each 'crowder'
    for crowder, style in styles.items():
        if crowder == 'buffer':
            plt.errorbar(style['x'], style['y'], yerr=style['yerr'], color=style['color'], marker=style['marker'],
                         mec=style['mec'], linestyle='none', lw=1, ms=style['ms'], elinewidth=1, capsize=3, capthick=1, label=style['label'])
        else:
            subset = df[df['crowder'] == crowder]
            plt.errorbar(subset['wt_%'], subset['D'], yerr=subset['D_err'], color=style['color'], marker=style['marker'],
                         mec=style['mec'], linestyle='none', lw=1, ms=style['ms'], elinewidth=1, capsize=3, capthick=1, label=style['label'])

    # Plot formatting
    plt.title('$D_{DNA}$ vs crowder wt.% change', fontsize=18)
    plt.xlabel('crowder wt.%', fontsize=16)
    plt.ylabel('$D_{DNA}$ [µm$^{2}$/s]', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.yscale('log')
    plt.xlim(-2, 52)
    plt.legend(frameon=True, loc='lower left', fontsize=8)
    plt.savefig('plots/D_DNA_vs_crowder_weight_percent.png', bbox_inches='tight', transparent=True, dpi=300)
    return plt.show()


def D_Na_vs_crowder_weight_percent():

    # Load CSV files into global variables and dictionary
    path = 'source_data/D_Na_and_D_crowder'
    crowders = {}
    for file in os.listdir(path):
        if file.endswith('.csv'):
            var_name = os.path.splitext(file)[0]
            df = pd.read_csv(os.path.join(path, file))
            globals()[var_name] = df
            crowders[var_name] = df

    # Define styles
    styles = {
        'buffer': {'x': [0], 'y': [EGly['D_Na_[um2/s]'][0]], 'yerr': [EGly['D_Na_err_[um2/s]'][0]], 'color': '#868788', 'marker': '*', 'ms': 9.5, 'mec': '#000000', 'label': 'buffer'},
        'EGly': {'color': '#2d642a', 'marker': '.', 'ms': 12, 'mec': '#000000', 'label': 'EGly'},
        'PEG200': {'color': '#b3daff', 'marker': 'd', 'ms': 7, 'mec': '#000000', 'label': 'PEG 200'},
        'PEG400': {'color': '#640024', 'marker': '^', 'ms': 7, 'mec': '#000000', 'label': 'PEG 400'},
        'PEG600': {'color': '#ff730f', 'marker': 'h', 'ms': 7, 'mec': '#000000', 'label': 'PEG 600'},
        'PEG1000': {'color': '#fed85d', 'marker': 'P', 'ms': 6.5, 'mec': '#000000', 'label': 'PEG 1k'},
        'PEG1500': {'color': '#142cd7', 'marker': 'p', 'ms': 7, 'mec': '#000000', 'label': 'PEG 1.5k'},
        'PEG3000': {'color': '#ac1416', 'marker': 's', 'ms': 6, 'mec': '#000000', 'label': 'PEG 3k'},
        'PEG6000': {'color': '#674ea7', 'marker': 'D', 'ms': 5.5, 'mec': '#000000', 'label': 'PEG 6k'},
        'PEG12000': {'color': '#709d74', 'marker': 'v', 'ms': 7, 'mec': '#000000', 'label': 'PEG 12k'},
        'PEG20000': {'color': '#00cccc', 'marker': 'X', 'ms': 7, 'mec': '#000000', 'label': 'PEG 20k'},
        'PEG35000': {'color': '#b28092', 'marker': 'h', 'ms': 7, 'mec': '#000000', 'label': 'PEG 35k'}
    }

    # Diffusion coefficient of Na in buffer
    Na_0 = (0, EGly['D_Na_[um2/s]'][0])
    plt.figure(figsize=(7, 5))

    # Loop through styles and crowders
    for name, style in styles.items():
        if name in crowders:  # Check if the crowder exists in the data
            df = crowders[name]
            slope, b_x, b_y, r_squared = uts.linear_fit_with_fixed_point(df['wt_%'], df['D_Na_[um2/s]'], Na_0)
            x_range = np.linspace(np.min(df['wt_%']), 19, 100)
            regression_line = slope * (x_range - b_x) + b_y
            plt.plot(x_range, regression_line, color=style['color'], linestyle='--', lw=1.5)
            plt.errorbar(df['wt_%'][1:], df['D_Na_[um2/s]'][1:], yerr=df['D_Na_err_[um2/s]'][1:],
                         color=style['color'], marker=style['marker'], mfc=style['color'], mec=style['mec'],
                         linestyle='none', lw=1, ms=style['ms'], elinewidth=1, capsize=3, capthick=1,
                         label=f"{style['label']}, R$^{2}$ = {r_squared:.3f}")
        elif 'x' in style and 'y' in style:  # Handle special cases like "buffer"
            plt.errorbar(style['x'], style['y'], yerr=style.get('yerr', None),
                         color=style['color'], marker=style['marker'], mfc=style['color'], mec=style['mec'],
                         linestyle='none', lw=1, ms=style['ms'], elinewidth=1, capsize=3, capthick=1,
                         label=style['label'])

    # Final plot styling and saving
    plt.title('$D_{Na^{+}}$ vs crowder wt.% change', fontsize=18)
    plt.xlabel('crowder wt.%', fontsize=16)
    plt.ylabel('$D_{Na^{+}}$ [µm$^{2}$/s]', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(-1, 19)
    plt.ylim(250,1550)
    plt.legend(frameon=True, loc='lower left', fontsize=8)
    plt.savefig('Plots/D_Na_vs_crowder_weight_percent.png', bbox_inches='tight', transparent=True, dpi=300)
    return plt.show()


def crowder_self_diffusion_coefficients_in_their_mass_functions():

    # Load CSV files into global variables and dictionary
    path = 'source_data/D_Na_and_D_crowder'
    crowders = {}
    for file in os.listdir(path):
        if file.endswith('.csv'):
            var_name = os.path.splitext(file)[0]
            df = pd.read_csv(os.path.join(path, file))
            globals()[var_name] = df
            crowders[var_name] = df

    # Define styles
    styles = {
        'EGly': {'color': '#2d642a', 'marker': '.', 'ms': 12, 'mec': '#000000', 'label': 'EGly'},
        'PEG200': {'color': '#b3daff', 'marker': 'd', 'ms': 7, 'mec': '#000000', 'label': 'PEG 200'},
        'PEG400': {'color': '#640024', 'marker': '^', 'ms': 7, 'mec': '#000000', 'label': 'PEG 400'},
        'PEG600': {'color': '#ff730f', 'marker': 'h', 'ms': 7, 'mec': '#000000', 'label': 'PEG 600'},
        'PEG1000': {'color': '#fed85d', 'marker': 'P', 'ms': 6.5, 'mec': '#000000', 'label': 'PEG 1k'},
        'PEG1500': {'color': '#142cd7', 'marker': 'p', 'ms': 7, 'mec': '#000000', 'label': 'PEG 1.5k'},
        'PEG3000': {'color': '#ac1416', 'marker': 's', 'ms': 6, 'mec': '#000000', 'label': 'PEG 3k'},
        'PEG6000': {'color': '#674ea7', 'marker': 'D', 'ms': 5.5, 'mec': '#000000', 'label': 'PEG 6k'},
        'PEG12000': {'color': '#709d74', 'marker': 'v', 'ms': 7, 'mec': '#000000', 'label': 'PEG 12k'},
        'PEG20000': {'color': '#00cccc', 'marker': 'X', 'ms': 7, 'mec': '#000000', 'label': 'PEG 20k'},
        'PEG35000': {'color': '#b28092', 'marker': 'h', 'ms': 7, 'mec': '#000000', 'label': 'PEG 35k'}
    }

    plt.figure(figsize=(7, 5))

    # Loop through styles and crowders
    for name, style in styles.items():
        df = crowders[name]
        a, b, c, r_squared = uts.exponential_fit_with_r_squared(df['wt_%'][1:], df['D_crowder_[um2/s]'][1:])
        x_range = np.linspace(np.min(df['wt_%']), 19, 100)
        plt.plot(x_range, uts.exponential_model(x_range, a, b, c), color=style['color'], linestyle='--', lw=1.5)
        plt.errorbar(df['wt_%'][1:], df['D_crowder_[um2/s]'][1:], yerr=df['D_crowder_err_[um2/s]'][1:],
                     color=style['color'], marker=style['marker'], mfc=style['color'], mec=style['mec'],
                     linestyle='none', lw=1, ms=style['ms'], elinewidth=1, capsize=3, capthick=1,
                     label=f"{style['label']}, R$^{2}$ = {r_squared:.3f}")

    # Final plot styling and saving
    plt.title('$D_{crowder}$ in their wt.% functions', fontsize=18)
    plt.xlabel('crowder wt.%', fontsize=16)
    plt.ylabel('$D_{crowder}$ [µm$^{2}$/s]', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(-1, 19)
    plt.legend(frameon=True, loc='upper right', fontsize=8)
    plt.savefig('Plots/crowder_self_diffusion_coefficients_in_their_mass_functions.png', bbox_inches='tight', transparent=True, dpi=300)
    return plt.show()


def kappa_results_for_PEG():
    # transforming data
    kappa_fitting_output = ssDNA_hybdrization_equillibrium_constant.kappa_fitting()
    df_kappa = pd.DataFrame([{
        "crowder": k,
        "kappa": float(v.split('±')[0]),
        "kappa_err": float(v.split('±')[1])
    } for k, v in kappa_fitting_output.items()])

    # choosing only values for PEGs and sorting data
    required_order = [
        "EGly", "PEG200", "PEG400", "PEG600", "PEG1000", "PEG1500",
        "PEG3000", "PEG6000", "PEG12000", "PEG20000", "PEG35000"]
    sorted_kappa = df_kappa[df_kappa['crowder'].isin(required_order)]
    sorted_kappa = sorted_kappa.set_index('crowder').loc[required_order].reset_index()

    # calculating weighted average of kappa for PEGs
    kappa_avg, kappa_avg_err =  uts.weighted_average_with_error(sorted_kappa['kappa'], sorted_kappa['kappa_err'])

    # plotting
    plt.figure(figsize=(7,5))

    plt.bar(sorted_kappa['crowder'], sorted_kappa['kappa'], yerr=sorted_kappa['kappa_err'],
            color='#dfb37f', capsize=3, error_kw={'ecolor': '#505050', 'elinewidth': 1.2}, label = '$κ$ values with \nrelated errors')
    plt.axhline(y=kappa_avg, color='black', linestyle='--', label = '$κ_{avg}$')
    plt.axhline(y = kappa_avg + kappa_avg_err, color='black', linestyle='-', label = '$κ_{err}$')
    plt.axhline(y = kappa_avg - kappa_avg_err, color='black', linestyle='-')
    x_min, x_max = plt.gca().get_xlim()  # Get current x-axis limits
    plt.fill_betweenx(y=[kappa_avg + kappa_avg_err, kappa_avg - kappa_avg_err], x1=x_min, x2=x_max, color='grey', alpha=0.4)

    plt.title('$κ_{crowder-Na^{+}}$ for EGly and PEGs', fontsize=18)
    plt.xlabel('', fontsize=16)
    plt.ylabel('$κ_{crowder-Na^{+}}$ [M$^{-1}$]', fontsize=16)
    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=12)
    plt.xlim(x_min, x_max)
    plt.ylim(0,0.31)
    plt.legend(frameon=True, loc='upper right', fontsize=8)
    plt.savefig('Plots/kappa_results_for_PEGs.png', bbox_inches='tight', transparent=True, dpi=300)
    return plt.show()


def PEG_coil_to_mesh_transition():
    value = uts.crowders_properties()

    plt.figure(figsize=(7, 5))

    a, b, r_squared = uts.power_fit_with_r_squared(value['MW_[g/mol]'].iloc[1:-3], value['c*_[g/cm3]'].iloc[1:-3])
    x_range = np.linspace(62.07, 50000, 5000)
    plt.errorbar(x_range, uts.power_model(x_range, a, b), color='#142cd7', marker='', linestyle='-')

    plt.errorbar(value['MW_[g/mol]'].iloc[:-3], value['c*_[g/cm3]'].iloc[:-3], color='#142cd7', marker='.',
                             mec='#000000', linestyle='none', lw=1, ms=12, elinewidth=1, capsize=3, capthick=1)

    plt.text(1500, 0.45, '$C_{crowder}$ > $C^{*}$\n     mesh', fontsize=14)
    plt.text(150, 0.1, '$C_{crowder}$ < $C^{*}$\n     coil', fontsize=14)

    plt.title('$C^{*}$ vs $MW_{crowder}$ for PEGs', fontsize=18)
    plt.xlabel('$MW_{crowder}$ [g/mol]', fontsize=16)
    plt.ylabel('$C^{*}$ [g/cm$^{3}$]', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xscale('log')
    plt.xlim(90, 50000)
    plt.ylim(0, 1.05)
    # plt.legend(frameon=True, loc='upper right', fontsize=8)
    plt.savefig('Plots/PEG_coil_to_mesh_transition.png', bbox_inches='tight', transparent=True, dpi=300)
    return plt.show()


def refractive_index_of_crowder_solutions():
    df = ssDNA_hybdrization_equillibrium_constant.calculate_average_RI_with_error_of_sample()

    # Adjust the plotting styles
    styles = {
        'EGly': {'color': '#2d642a', 'marker': '.', 'ms': 12, 'mec': '#000000', 'label': 'EGly'},
        'PEG200': {'color': '#b3daff', 'marker': 'd', 'ms': 7, 'mec': '#000000', 'label': 'PEG 200'},
        'PEG400': {'color': '#640024', 'marker': '^', 'ms': 7, 'mec': '#000000', 'label': 'PEG 400'},
        'PEG600': {'color': '#ff730f', 'marker': 'h', 'ms': 7, 'mec': '#000000', 'label': 'PEG 600'},
        'PEG1000': {'color': '#fed85d', 'marker': 'P', 'ms': 6.5, 'mec': '#000000', 'label': 'PEG 1k'},
        'PEG1500': {'color': '#142cd7', 'marker': 'p', 'ms': 7, 'mec': '#000000', 'label': 'PEG 1.5k'},
        'PEG3000': {'color': '#ac1416', 'marker': 's', 'ms': 6, 'mec': '#000000', 'label': 'PEG 3k'},
        'PEG6000': {'color': '#674ea7', 'marker': 'D', 'ms': 5.5, 'mec': '#000000', 'label': 'PEG 6k'},
        'PEG12000': {'color': '#709d74', 'marker': 'v', 'ms': 7, 'mec': '#000000', 'label': 'PEG 12k'},
        'PEG20000': {'color': '#00cccc', 'marker': 'X', 'ms': 7, 'mec': '#000000', 'label': 'PEG 20k'},
        'PEG35000': {'color': '#b28092', 'marker': 'h', 'ms': 7, 'mec': '#000000', 'label': 'PEG 35k'}}

    # plotting
    plt.figure(figsize=(7, 5))
    for crowder, style in styles.items():
        values = df[crowder].str.split('±').str[0].astype(float)  # Extract values before ±
        errors = df[crowder].str.split('±').str[1].astype(float)  # Extract errors after ±

        x_0 = [0, 1.337]
        slope, b_x, b_y, r_squared = uts.linear_fit_with_fixed_point(df['wt_%'], values, x_0)
        x_range = np.linspace(0, 42, 100)
        regression_line = slope * (x_range - b_x) + b_y
        plt.plot(x_range, regression_line, color=style['color'], linestyle='--', lw=1.5)

        plt.errorbar(df['wt_%'], values, yerr=errors, color=style['color'], marker=style['marker'],
                     mec=style['mec'], linestyle='none', lw=1, ms=style['ms'], elinewidth=1, capsize=3,
                     capthick=1, label=f''
                                       f"{style['label']}, R²={r_squared:.3f}")

    plt.title('$RI$ of crowder solutions', fontsize=18)
    plt.xlabel('crowder wt.%', fontsize=16)
    plt.ylabel('$RI$', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(-2, 42)
    plt.legend(frameon=True, loc='upper left', fontsize=8)
    plt.savefig('plots/refractive_index_of_crowder_solutions.png', bbox_inches='tight', transparent=True, dpi=300)
    return plt.show()


def gibbs_free_energy_of_DNA_DNA_reaction_vs_crowder_weight_percent():
    # Load and process data
    df = pd.read_csv('results/K_DNA-DNA_in_crowder_solutions.csv')
    K_0 = unc.ufloat(1760000000, 311872000)  # Reference K
    df["K"] = df["K"].apply(uts.convert_to_ufloat)
    df["D"] = df["D"].apply(uts.convert_to_ufloat)
    df["G"] = df["K"].apply(lambda K: -uts.R * uts.T * umath.log(K / K_0) / 1000)  # ΔG in kJ/mol
    df = df[df["G"].apply(lambda g: not unp.isnan(g))]  # Filter NaN G values

    # Define styles for plotting
    styles = {
        'buffer': {'x': [0], 'y': [0], 'yerr': [0], 'color': '#868788', 'marker': '*', 'ms': 9.5, 'mec': '#000000', 'label': 'buffer'},
        'EGly': {'color': '#2d642a', 'marker': '.', 'ms': 12, 'mec': '#000000', 'label': 'EGly'},
        'PEG200': {'color': '#b3daff', 'marker': 'd', 'ms': 7, 'mec': '#000000', 'label': 'PEG 200'},
        'PEG400': {'color': '#640024', 'marker': '^', 'ms': 7, 'mec': '#000000', 'label': 'PEG 400'},
        'PEG600': {'color': '#ff730f', 'marker': 'h', 'ms': 7, 'mec': '#000000', 'label': 'PEG 600'},
        'PEG1000': {'color': '#fed85d', 'marker': 'P', 'ms': 6.5, 'mec': '#000000', 'label': 'PEG 1k'},
        'PEG1500': {'color': '#142cd7', 'marker': 'p', 'ms': 7, 'mec': '#000000', 'label': 'PEG 1.5k'},
        'PEG3000': {'color': '#ac1416', 'marker': 's', 'ms': 6, 'mec': '#000000', 'label': 'PEG 3k'},
        'PEG6000': {'color': '#674ea7', 'marker': 'D', 'ms': 5.5, 'mec': '#000000', 'label': 'PEG 6k'},
        'PEG12000': {'color': '#709d74', 'marker': 'v', 'ms': 7, 'mec': '#000000', 'label': 'PEG 12k'},
        'PEG20000': {'color': '#00cccc', 'marker': 'X', 'ms': 7, 'mec': '#000000', 'label': 'PEG 20k'},
        'PEG35000': {'color': '#b28092', 'marker': 'h', 'ms': 7, 'mec': '#000000', 'label': 'PEG 35k'},
    }

    plt.figure(figsize=(7, 5))

    for crowder, style in styles.items():
        if crowder == 'buffer':
            plt.errorbar(style['x'], style['y'], yerr=style['yerr'], color=style['color'], marker=style['marker'],
                         mec=style['mec'], linestyle='none', lw=1, ms=style['ms'], elinewidth=1, capsize=3, capthick=1, label=style['label'])
        else:
            subset = df[df['crowder'] == crowder]
            wt_percent = subset['wt_%'].values
            G_nominal = unp.nominal_values(subset['G'])
            G_error = unp.std_devs(subset['G'])
            plt.errorbar(wt_percent, G_nominal, yerr=G_error, color=style['color'], marker=style['marker'],
                         mec=style['mec'], linestyle='none', lw=1, ms=style['ms'], elinewidth=1, capsize=3, capthick=1, label=style['label'])

    # Formatting to match the original
    plt.title('$ΔG_{DNA-DNA}$ vs crowder wt.% change', fontsize=18)
    plt.xlabel('Crowder wt.%', fontsize=16)
    plt.ylabel('$ΔG_{DNA-DNA}$ [kJ/mol]', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(-2, 52)
    plt.legend(frameon=True, loc='upper left', fontsize=8)
    plt.savefig('plots/gibbs_free_energy_vs_crowder_weight_percent.png', bbox_inches='tight', transparent=True, dpi=300)
    plt.show()


def kinetics_DNA_DNA_in_EGly_40wt():
    df = pd.read_csv('source_data/kinetics_DNA_DNA_in_EGly_40%wt.csv', sep=',', names=['t', 'It'], header=0)
    # apply Savitzky-Golay filter for denoising
    df['It_denoised'] = savgol_filter(df['It'], window_length=37, polyorder=2)

    # calculating k and error
    If = 1 # final fluorescence intensity [a.u.]
    a = 50 # Acc/Don ratio
    df['k'] = df['It_denoised'] / (If * (If - df['It_denoised']) * df['t'] * a)  # k with a factor
    df['ΔIt'] = (1 / (df['t'] * (If - df['It_denoised'])**2 * a)) * 0.1 * df['It_denoised']
    df['ΔIf'] = (df['It_denoised'] * (df['It_denoised'] - 2 * If)) / (If**2 * df['t'] * (If - df['It_denoised'])**2 * a) * 0.1 * If
    df['Δk'] = np.sqrt(df['ΔIt']**2 + df['ΔIf']**2)  # k error
    df['k/Δk^2'] = df['k'] / df['Δk']**2  # Calculation for weighted mean
    df['1/Δk^2'] = 1 / df['Δk']**2  # Calculation for weighted mean
    k_avg_no_multiply = (df["k/Δk^2"].sum() / df["1/Δk^2"].sum())  # Weighted mean k
    k_avg = k_avg_no_multiply * 10**8  # Weighted mean k
    k_error = (1 / df['1/Δk^2'].sum())**0.5 * 10**8  # Weighted mean error k

    # plot
    plt.figure(figsize=(7, 5))

    plt.errorbar(df['t']/60, df['It_denoised'], color='#142cd7', marker='.', linestyle='none', ms=6, zorder=1)
    fit = (k_avg_no_multiply * (If**2) * df['t'] * a) / (1 + k_avg_no_multiply * If * df['t'] * a)
    plt.plot(df['t']/60, fit, color='#dfb37f', linewidth=2.5, zorder=2)

    legend_label = ('$k_{a} = $' + f'{k_avg:.2e} M$^{{-1}}$s$^{{-1}}$\nerror = {k_error/k_avg*100:.2f}%')
    plt.legend(frameon=True, loc='lower right', fontsize=12, labels=[legend_label, 'data'])
    plt.title('DNA-DNA association rate in 40 wt.% EGly', fontsize=18)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.xlabel('time [min]', fontsize=16)
    plt.ylabel('fluorescence [a.u.]', fontsize=16)
    plt.savefig('plots/kinetics_DNA_DNA_in_EGly_40wt.png', bbox_inches='tight', transparent=True, dpi=300)
    return plt.show()


def denoising_kinetics_DNA_DNA_in_EGly_40wt():
    df = pd.read_csv('source_data/kinetics_DNA_DNA_in_EGly_40%wt.csv', sep=',', names=['t', 'It'], header=0)
    # apply Savitzky-Golay filter for denoising
    df['It_denoised'] = savgol_filter(df['It'], window_length=37, polyorder=2)

    # plot
    plt.figure(figsize=(7, 5))

    plt.errorbar(df['t'] / 60, df['It'], color='#dfb37f', marker='.', linestyle='none', ms=6, zorder=1, label = 'source data')
    plt.errorbar(df['t']/60, df['It_denoised'], color='#142cd7', marker='.', linestyle='none', ms=6, zorder=1, label = 'denoised data')

    plt.legend(frameon=True, loc='lower right', fontsize=12)
    plt.title('denoised vs source data of\nDNA-DNA association in 40 wt.% EGly', fontsize=18)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.xlabel('time [min]', fontsize=16)
    plt.ylabel('fluorescence [a.u.]', fontsize=16)
    plt.savefig('plots/denoising_kinetics_DNA_DNA_in_EGly_40wt.png', bbox_inches='tight', transparent=True, dpi=300)
    return plt.show()















