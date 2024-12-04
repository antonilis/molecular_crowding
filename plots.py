import numpy as np
import pandas as pd
import utils as uts
import os
import matplotlib.pyplot as plt
import utils as uts


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
    plt.ylabel('$K_{DNA-DNA}$ [$M^{-1}$]', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.yscale('log')
    plt.xlim(-2, 52)
    plt.legend(frameon=True, loc='lower left', fontsize=7.5)
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
    plt.ylabel('$D_{DNA}$ [$µm^{2}/s$]', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.yscale('log')
    plt.xlim(-2, 52)
    plt.legend(frameon=True, loc='lower left', fontsize=7.5)
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
    plt.title('D$_{Na^{+}}$ vs crowder wt.% change', fontsize=18)
    plt.xlabel('crowder wt.%', fontsize=16)
    plt.ylabel('D$_{Na^{+}}$ [$µm^{2}/s$]', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(-1, 19)
    plt.legend(frameon=True, loc='lower left', fontsize=7.5)
    plt.savefig('Plots/D_Na_vs_crowder_weight_percent.png', bbox_inches='tight', transparent=True, dpi=300)
    return plt.show()






# def crowder_self_diffusion_coefficients_in_their_mass_functions():
#
#     # Load CSV files into global variables and dictionary
#     path = 'source_data/D_Na_and_D_crowder'
#     crowders = {}
#     for file in os.listdir(path):
#         if file.endswith('.csv'):
#             var_name = os.path.splitext(file)[0]
#             df = pd.read_csv(os.path.join(path, file))
#             globals()[var_name] = df
#             crowders[var_name] = df
#
#     # Define styles
#     styles = {
#         'EGly': {'color': '#2d642a', 'marker': '.', 'ms': 12, 'mec': '#000000', 'label': 'EGly'},
#         'PEG200': {'color': '#b3daff', 'marker': 'd', 'ms': 7, 'mec': '#000000', 'label': 'PEG 200'},
#         'PEG400': {'color': '#640024', 'marker': '^', 'ms': 7, 'mec': '#000000', 'label': 'PEG 400'},
#         'PEG600': {'color': '#ff730f', 'marker': 'h', 'ms': 7, 'mec': '#000000', 'label': 'PEG 600'},
#         'PEG1000': {'color': '#fed85d', 'marker': 'P', 'ms': 6.5, 'mec': '#000000', 'label': 'PEG 1k'},
#         'PEG1500': {'color': '#142cd7', 'marker': 'p', 'ms': 7, 'mec': '#000000', 'label': 'PEG 1.5k'},
#         'PEG3000': {'color': '#ac1416', 'marker': 's', 'ms': 6, 'mec': '#000000', 'label': 'PEG 3k'},
#         'PEG6000': {'color': '#674ea7', 'marker': 'D', 'ms': 5.5, 'mec': '#000000', 'label': 'PEG 6k'},
#         'PEG12000': {'color': '#709d74', 'marker': 'v', 'ms': 7, 'mec': '#000000', 'label': 'PEG 12k'},
#         'PEG20000': {'color': '#00cccc', 'marker': 'X', 'ms': 7, 'mec': '#000000', 'label': 'PEG 20k'},
#         'PEG35000': {'color': '#b28092', 'marker': 'h', 'ms': 7, 'mec': '#000000', 'label': 'PEG 35k'}
#     }
#
#     plt.figure(figsize=(7, 5))
#
#     # Loop through styles and crowders
#     for name, style in styles.items():
#         df = crowders[name]
#         a, b, c, r_squared = uts.exponential_fit_with_r_squared(df['wt_%'][1:], df['D_crowder_[um2/s]'][1:])
#         x_range = np.linspace(np.min(df['wt_%']), 19, 100)
#         plt.plot(x_range, uts.exponential_model(x_range, a, b, c), color=style['color'], linestyle='--', lw=1.5)
#         plt.errorbar(df['wt_%'][1:], df['D_crowder_[um2/s]'][1:], yerr=df['D_crowder_err_[um2/s]'][1:],
#                      color=style['color'], marker=style['marker'], mfc=style['color'], mec=style['mec'],
#                      linestyle='none', lw=1, ms=style['ms'], elinewidth=1, capsize=3, capthick=1,
#                      label=f"{style['label']}, R$^{2}$ = {r_squared:.3f}")
#
#     # Final plot styling and saving
#     plt.title('D$_{Na^{+}}$ vs crowder wt.% change', fontsize=18)
#     plt.xlabel('crowder wt.%', fontsize=16)
#     plt.ylabel('D$_{Na^{+}}$ [$µm^{2}/s$]', fontsize=16)
#     plt.xticks(fontsize=12)
#     plt.yticks(fontsize=12)
#     plt.xlim(-1, 19)
#     plt.legend(frameon=True, loc='lower left', fontsize=7.5)
#     plt.savefig('Plots/xxx.png', bbox_inches='tight', transparent=True, dpi=300)
#     return plt.show()
#
#
# crowder_self_diffusion_coefficients_in_their_mass_functions()