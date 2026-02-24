from source_data_processors.reacting_probes_equillibrium_constants import EquillibriumConstants
from source_data_processors.sodium_crowder_complexation_constant import SodiumComplexation
from calculations.equllibrium_constant_theoretical_dependency import TheoreticalModel


class Data:

    def __init__(self, source_equillbrium_constants_path, source_complexation_data_path):
        self.EquillbriumConstantsObj = EquillibriumConstants(source_equillbrium_constants_path)
        self.SodiumCrowderComplexationObj = SodiumComplexation(source_complexation_data_path)
        self.theoretical_calculations = TheoreticalModel()

        self.data = self.merge_data()

    def merge_data(self):
        eq_data = self.EquillbriumConstantsObj.analyzed_data.copy()
        eq_data['n Beta [1/M]'] = eq_data['No_mono'] * self.SodiumCrowderComplexationObj.n_complexation

        theory_data = self.theoretical_calculations.compute_a1_a2_theory(eq_data.copy())

        final = self.theoretical_calculations.calculate_total_Gibbs_energy(theory_data.copy())

        return final


if __name__ == '__main__':
    import utils as uts
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from uncertainties import umath

    equillibrium_constans_path = 'source_data/MacromoleculeEquilibria_Crowding'
    complexation_data_path = 'source_data/IonCrowderComplexation/raw_data'

    data = Data(equillibrium_constans_path, complexation_data_path)

    df = data.data.copy()

    dat = df.copy()
    dat = dat[np.sign(dat['charge 1']) == np.sign(dat['charge 2'])]

    def filter_quadratic_fits(dat):
        filtering_mask = (dat['chi2_red'] > 0.2) & (dat['chi2_red'] < 10) & (dat['R2'] > 0.5) & (dat['a1_exp'] > 0) & (
                    dat['a2_exp'] < 0)
        return dat[filtering_mask]

    dat = filter_quadratic_fits(dat)

    df = dat[dat['sample'] == 'ssDNA_13bp']

    df['n Beta conc'] = df['n Beta [1/M]'] * df['concentration [M]']
    df['curve corr'] = -df['a1_exp'] / (2 * df['a2_exp'])

    df['curve corr theory'] = -df['theory_a1'] / (2 * df['theory_a2'])


    wrong = df[df['concentration [M]'] > df['curve corr']]




    # dat = df[df['crowder wt. [%]'] == 0]
    # result = df.merge(
    #     dat[['sample', 'crowder', 'source', 'Delta_G_exp_[J/mol]']].rename(columns={'Delta_G_exp_[J/mol]': 'Delta_G0_exp_[J/mol]'}),
    #     on=['sample', 'crowder', 'source'],
    #     how='left'
    # )
    #
    # wrong = result[result['Delta_G0_exp_[J/mol]'].isna()]

    # plotting fits of complexation constant from NMR diffusion experiments
    # df = data.SodiumCrowderComplexationObj.analyzed_data
    #
    # crowders = df['crowder'].unique()
    #
    # for crowder in crowders:
    #     crowder_data = df[df['crowder'] == crowder]
    #
    #     x = crowder_data['concentration [M]'].values
    #     y, y_err = uts.get_float_uncertainty(crowder_data['D_Na_uf_[um2/s]'])
    #
    #     D_Na = crowder_data['D_Na_[um2/s] corr'].values
    #     D_crowder, _ = uts.get_float_uncertainty(crowder_data['D_crowder_uf_[um2/s]'])
    #
    #     alpha = crowder_data['n Beta [1/M]'].iloc[0]
    #
    #     x_fit = np.linspace(x.min(), x.max(), 10)
    #     D_Na_fit = np.interp(x_fit, x, D_Na)
    #     D_crowder_fit = np.interp(x_fit, x, D_crowder)
    #     y_fit, _ = uts.get_float_uncertainty(data.SodiumCrowderComplexationObj.model_for_fit(x_fit, D_Na_fit, D_crowder_fit, alpha))
    #
    #     plt.errorbar(x, y, yerr=y_err, fmt='v', markersize=6, label='Experimental points', capsize=3)
    #     plt.plot(x_fit, y_fit, '-', label='Fit')
    #     plt.title(f'Crowder: {crowder}', fontsize=14)
    #     plt.xlabel('Concentration [M]')
    #     plt.ylabel('Diffusion [um^2/s]')
    #     plt.legend()
    #
    #     plt.tight_layout()
    #     plt.savefig(os.path.join('plots\\NMR_diffusion_change_fit', f"{crowder}.png"), dpi=300)
    #     plt.close()
    #
    # # equillibrium constans quadratic_dependency_fit
    #
    #
    # df = data.EquillbriumConstantsObj.analyzed_data.copy()
    #
    # groups = df[['source', 'sample', 'crowder']].drop_duplicates()
    #
    # for _, row in groups.iterrows():
    #     mask = (
    #             (df['source'] == row['source']) &
    #             (df['sample'] == row['sample']) &
    #             (df['crowder'] == row['crowder'])
    #     )
    #     df_group = df[mask]
    #     group_name = f"{row['source']}_{row['sample']}_{row['crowder']}"
    #
    #     x = np.array(df_group['concentration [M]'], dtype=np.float64)
    #     y, y_err = uts.get_float_uncertainty(df_group['K_uf [M]'].apply(lambda x: umath.log(x)))
    #
    #     x_fit = np.linspace(x.min(), x.max(), 200)
    #
    #     a0, _ = uts.get_float_uncertainty(df_group['a0_exp'])
    #     a1, _ = uts.get_float_uncertainty(df_group['a1_exp'])
    #     a2, _ = uts.get_float_uncertainty(df_group['a2_exp'])
    #
    #
    #     y_fit =  a0[0] + a1[0] * x_fit + a2[0] * x_fit**2
    #
    #     chi2 = df_group['chi2_red'].mean()
    #     R2 = df_group['R2'].mean()
    #
    #     plt.errorbar(x, y, y_err, fmt='v')
    #     plt.plot(x_fit, y_fit)
    #     plt.text(
    #         0.95, 0.95, f'$R^2={R2:.3f}$\n$\chi^2={chi2:.3f}$',
    #         horizontalalignment='right',
    #         verticalalignment='top',
    #         transform=plt.gca().transAxes
    #     )
    #     plt.savefig(f'.\\plots\\equillibrium_constans_quadratic_dependency_fit\\{group_name}.png')
    #     plt.close()
