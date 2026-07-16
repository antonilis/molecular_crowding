from source_data_processors.reacting_probes_equillibrium_constants import EquillibriumConstants
from source_data_processors.sodium_crowder_complexation_constant import SodiumComplexation
from source_data_processors.sodium_dependency import SodiumDependency
from ssDNA_parameters import ssDNAParameters
from uncertainties import umath


# TODO: dodaj tutaj liczenie ciśnienia osmotycznego PI z eksperymentu


class Data:

    def __init__(self, source_equillbrium_constants_path, source_complexation_data_path, source_sodium_dep):


        # Physical constants, useful mostly in jupyter
        self.eps = 8.8541878188e-12 * 81
        self.Na = 6.02214076e23
        self.qe = 1.60217663e-19
        self.kb = 1.380649e-23
        self.R = 8.31446261815324

        self.EquillbriumConstantsObj = EquillibriumConstants(source_equillbrium_constants_path)  # Here are done all the calculation for derivation of the ssDNA hybrydyzation reaction
        self.SodiumCrowderComplexationObj = SodiumComplexation(source_complexation_data_path) # Here are calculation regarding the sodium complexation
        self.DepletionObj = ssDNAParameters(13)  # Here are ssDNA WCM size parameters, for 13 bp
        self.SodiumDependencyObj = SodiumDependency(source_sodium_dep) # Here we read how the ssDNA eq. const. depends on the sodium concentration

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

        eq_data['dG_exp_[kJ/mol]'] = - self.R * eq_data['T [K]'] * eq_data['K_uf [M]'].apply(
            lambda x: umath.log(x)) / 1000

        return eq_data


if __name__ == '__main__':
    equillibrium_constans_path = 'source_data/MacromoleculeEquilibria_Crowding'
    complexation_data_path = 'source_data/IonCrowderComplexation/raw_data'
    sodium_path = 'source_data/IonStrengthDependency/Na+_FRET.csv'

    data = Data(equillibrium_constans_path, complexation_data_path, sodium_path)
