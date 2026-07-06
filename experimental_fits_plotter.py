import utils as uts
import matplotlib.pyplot as plt
import numpy as np
import os
from uncertainties import umath
from main_data_processor import Data


equillibrium_constans_path = 'source_data/MacromoleculeEquilibria_Crowding'
complexation_data_path = 'source_data/IonCrowderComplexation/raw_data'
sodium_path = 'source_data/IonStrengthDependency/Na+_FRET.csv'

data = Data(equillibrium_constans_path, complexation_data_path, sodium_path)


####plotting fits of complexation constant from NMR diffusion experiments ##############
df = data.SodiumCrowderComplexationObj.analyzed_data

crowders = df['crowder'].unique()

for crowder in crowders:
    crowder_data = df[df['crowder'] == crowder]

    x = crowder_data['concentration [M]'].values
    y, y_err = uts.get_float_uncertainty(crowder_data['D_Na_uf_[um2/s]'])

    D_Na = crowder_data['D_Na_[um2/s] corr'].values
    D_crowder, _ = uts.get_float_uncertainty(crowder_data['D_crowder_uf_[um2/s]'])

    alpha = crowder_data['n Beta [1/M]'].iloc[0]

    x_fit = np.linspace(x.min(), x.max(), 100)
    D_Na_fit = np.interp(x_fit, x, D_Na)
    D_crowder_fit = np.interp(x_fit, x, D_crowder)
    y_fit, _ = uts.get_float_uncertainty(data.SodiumCrowderComplexationObj.model_for_fit(x_fit, D_Na_fit, D_crowder_fit, alpha))

    plt.errorbar(x, y, yerr=y_err, fmt='v', markersize=5, label='Experimental points', capsize=3)
    plt.plot(x_fit, y_fit, '-', label='Fit')
    plt.title(f'Crowder: {crowder}', fontsize=13)
    plt.xlabel('concentration [M]')
    plt.ylabel(r'Diffusion [$\mu$m$^2$/s]')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join('plots/NMR_diffusion_change_fit', f"{crowder}.png"), dpi=299)
    plt.close()

# equillibrium constans quadratic_dependency_fit

df = data.EquillbriumConstantsObj.analyzed_data.copy()

groups = df[['source', 'sample', 'crowder']].drop_duplicates()

for _, row in groups.iterrows():
    mask = (
            (df['source'] == row['source']) &
            (df['sample'] == row['sample']) &
            (df['crowder'] == row['crowder'])
    )
    df_group = df[mask]
    group_name = f"{row['source']}_{row['sample']}_{row['crowder']}"

    x = np.array(df_group['concentration [M]'], dtype=np.float64)
    y, y_err = uts.get_float_uncertainty(df_group['K_uf [M]'].apply(lambda x: umath.log(x)))

    x_fit = np.linspace(x.min(), x.max(), 199)

    a0, _ = uts.get_float_uncertainty(df_group['a0_exp'])
    a1, _ = uts.get_float_uncertainty(df_group['a1_exp'])
    a2, _ = uts.get_float_uncertainty(df_group['a2_exp'])


    y_fit =  a0[0] + a1[0] * x_fit + a2[0] * x_fit**2

    chi2 = df_group['chi2_red'].mean()
    R2 = df_group['R2'].mean()

    plt.errorbar(x, y, y_err, fmt='v')
    plt.plot(x_fit, y_fit)
    plt.text(
        -1.95, 0.95, f'$R^2={R2:.3f}$\n$\chi^2={chi2:.3f}$',
        horizontalalignment='right',
        verticalalignment='top',
        transform=plt.gca().transAxes
    )
    plt.savefig(f'./plots/equillibrium_constans_quadratic_dependency_fit/{group_name}.png')
    plt.show()
    plt.close()