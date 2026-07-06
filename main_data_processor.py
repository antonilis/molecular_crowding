from source_data_processors.reacting_probes_equillibrium_constants import EquillibriumConstants
from source_data_processors.sodium_crowder_complexation_constant import SodiumComplexation
from source_data_processors.sodium_dependency import SodiumDependency
from depletion_model import DepletionCalculations
from uncertainties import umath


class Data:

    def __init__(self, source_equillbrium_constants_path, source_complexation_data_path, source_sodium_dep):
        self.EquillbriumConstantsObj = EquillibriumConstants(source_equillbrium_constants_path)
        self.SodiumCrowderComplexationObj = SodiumComplexation(source_complexation_data_path)
        self.DepletionObj = DepletionCalculations()  # probably wrong right now
        self.SodiumDependencyObj = SodiumDependency(source_sodium_dep)

        self.data = self.process_data()

    def calculate_free_ions(self, df, nBeta_col, col_c_Na, col_c_peg):
        Bn = df[nBeta_col]
        c_free = df[col_c_Na] / 1000 / (Bn * df[col_c_peg] + 1)  # Na+ in mM concentration

        return c_free * 1000

    def process_data(self):

        eq_data = self.EquillbriumConstantsObj.analyzed_data.copy()
        eq_data['nBeta_[1/M]'] = eq_data['No_mono'] * self.SodiumCrowderComplexationObj.n_complexation

        eq_data['Na_free_[mM]'] = self.calculate_free_ions(eq_data, 'nBeta_[1/M]', 'Na conc. [mM]', 'concentration [M]')
        eq_data['dG_Na_[kJ/mol]'] = self.SodiumDependencyObj.calculate_sodium_dG(eq_data['Na_free_[mM]'])

        eq_data['dG_exp_[kJ/mol]'] = - self.DepletionObj.R * eq_data['T [K]'] * eq_data['K_uf [M]'].apply(
            lambda x: umath.log(x)) / 1000

        eq_data['dG_depl_dil_[kJ/mol]'] = self.DepletionObj.calculate_depletion_gibbs_energy_diluted(self.DepletionObj.R,
                                                                                               self.DepletionObj.R,
                                                                                               eq_data['Rg_[nm]'],
                                                                                               eq_data['T [K]'],
                                                                                               eq_data['concentration [M]'])


        eq_data['dG_depl_semi_dil_[kJ/mol]'] = eq_data.apply(
            lambda row: self.DepletionObj.calculate_depletion_gibbs_energy_semidiluted(
                R1=self.DepletionObj.ssDNA_radius,
                R2=self.DepletionObj.ssDNA_radius,
                Rg_crowder=row['Rg_[nm]'],
                T=row['T [K]'],
                c_star=row['c*_[M]'],
                c_max=row['concentration [M]']
            ),
            axis=1
        )

        return eq_data


if __name__ == '__main__':
    equillibrium_constans_path = 'source_data/MacromoleculeEquilibria_Crowding'
    complexation_data_path = 'source_data/IonCrowderComplexation/raw_data'
    sodium_path = 'source_data/IonStrengthDependency/Na+_FRET.csv'

    data = Data(equillibrium_constans_path, complexation_data_path, sodium_path)
