import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


class Depletion:

    def __init__(self):
        self.sodium_concentration = 0.03508
        self.EG_molar_mas = 62.07
        self.Rg_ssDNA = 0.55
        self.R = 8.314
        self.T = 298.15
        self.kb = 1.38 * 10 ** (-23)
        self.N_av = 6.02 * 10 ** 23
        self.data = self.get_data()
        self.depletion_volume = self.calc_depletion_volume()
        self.polynomial_coeff = self.fit_polynomial_to_K()

    @staticmethod
    def calc_Rg(Mmol):
        Rg = 0.215 * Mmol ** (0.583) / 10  # result is in [nm]

        return Rg

    @staticmethod
    def calculate_PEG_Rh(Mmol):
        Rh = 0.145 * (Mmol) ** (0.571) / 10

        return Rh

    #
    # def calc_ksi(self,Mmol):
    #     ksi = self.calc_Rg(Mmol)
    #
    #
    def calc_c_star(self, Rg):
        c_star = 1 / (self.N_av * 4 / 3 * np.pi * Rg ** 3) * 10 ** 24  # getting liters from nm^3

        return c_star

    # def calc_c_star(self, Mmol):
    #     niu = 0.004 * Mmol**0.8 * 1e-1
    #     c_star = 1/niu
    #     return c_star/Mmol
    def number_of_monomers(self, Mmas):
        polymer_number = (Mmas - 18.02) / (self.EG_molar_mas - 18.02)  # number of monomers in polymer
        return polymer_number

    def calc_free_sodium(self, Mmas, K_complex, concentration):
        polymer_number = self.number_of_monomers(Mmas)  # number of monomers in polymer

        free_sodium = self.sodium_concentration - (
                concentration * polymer_number * K_complex * self.sodium_concentration) / (
                              1 + concentration * polymer_number * K_complex)

        return free_sodium

    @staticmethod
    def K_sodium_dependency(sodium_conentration):
        K_predicted = 4.09 * 10 ** (13) * sodium_conentration ** (3.14)

        return K_predicted

    def get_data(self):
        # reading data from excel files
        dat = pd.read_excel('depletion_table.xlsx')

        # calculating R_g

        dat['Rg [nm]'] = self.calc_Rg(dat['molar mas'])

        # calculating the molar concentration of pegs: muliplying by 1000 to change the density unit
        # and dividing by 100 do get % mas concentration -> multiplying by 10

        dat['concentration [M]'] = dat['mas concentration'] * dat['density'] / dat['molar mas'] * 10

        # calculating concentration of free sodium

        free_sodium = self.calc_free_sodium(dat['molar mas'], dat['K complex'], dat['concentration [M]'])

        # K  due to Na ions influence
        K_predicted = self.K_sodium_dependency(free_sodium)

        dat['G'] = self.R * self.T * (np.log(K_predicted / dat['K hybr'])) / 1000  # Disivion by 1000 gives kJ

        dat['c*'] = self.calc_c_star(dat['Rg [nm]'])

        dat['Rh [nm]'] = self.calculate_PEG_Rh(dat['molar mas'])

        dat['monomers number'] = self.number_of_monomers(dat['molar mas'])

        # changing slitghly order of columns for convenience
        columns = ['peg', 'molar mas', 'mas concentration', 'concentration [M]', 'c*', 'monomers number', 'density',
                   'Rg [nm]', 'Rh [nm]',
                   'K hybr',
                   'K hybr error', 'K complex', 'K  complex error',
                   'G']

        dat = dat[columns]

        return dat

    @staticmethod
    def calculate_depletion_layer(Rg):
        delta = 1.07 * Rg

        frac = delta / Rg

        delta_s = Rg * ((1 + 3 * frac + 2.273 * frac ** 2 - 0.0975 * frac ** (3)) ** (1 / 3) - 1)

        return delta_s

    def calc_interlap_volume(self, Rg):
        r = self.Rg_ssDNA + self.calculate_depletion_layer(Rg)
        R = self.Rg_ssDNA + self.calculate_depletion_layer(Rg)
        d = 2 * self.Rg_ssDNA

        overlap_volume = (np.pi * (r + R - d) ** 2 * (d ** 2 - 3 * (r - R) ** 2 + 2 * d * (r + R))) / (12 * d)

        return overlap_volume

    def calc_depletion_volume(self):
        peg = self.data['peg'].unique()
        linear_limit = [6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]

        molar_mas = self.data.groupby(by='peg', sort=False).mean()['molar mas']

        slope_list, overlap_volumes = np.array([]), np.array([])

        linear_function = lambda x, a: a * x

        for polimer, limit in zip(peg, linear_limit):
            df = self.data[self.data['peg'] == polimer].iloc[0:limit]

            x_to_fit, y_to_fit = df['concentration [M]'], df['G']

            params, _ = curve_fit(linear_function, x_to_fit, y_to_fit)

            volume = self.calc_interlap_volume(df['Rg [nm]'].mean())

            slope_list = np.append(slope_list, params[0])
            overlap_volumes = np.append(overlap_volumes, volume)

        volume = slope_list / (-self.kb * self.N_av ** 2 * self.T) * 10 ** (27)

        dat = pd.DataFrame({'peg': peg, 'slope': slope_list, 'volume [nm]': volume, 'molar mas': molar_mas,
                            'calc volume [nm]': overlap_volumes})

        return dat

    def plot_delta_G(self, peg):
        df = self.data[self.data['peg'] == peg]

        x = df['concentration [M]']
        y = df['G']

        plt.plot(x, y, '.', label=peg)
        plt.legend()
        plt.show()

    def choose_peg(self, peg):
        df = self.data[self.data['peg'] == peg]
        return df

    def fit_polynomial_to_K(self):
        names = self.data['peg'].unique()

        # Using a dictionary comprehension
        coeffs_dict = {name: np.polyfit(
            np.array(self.choose_peg(name)['concentration [M]'], dtype=np.float64),
            np.log(self.choose_peg(name)['K hybr']),
            2
        ) for name in names}

        # Convert the dictionary to a DataFrame
        coeffs_df = pd.DataFrame.from_dict(coeffs_dict, orient='index',
                                           columns=['a2', 'a1', 'a0'])

        # Reset the index to make 'peg' a column
        coeffs_df = coeffs_df.reset_index().rename(columns={'index': 'peg'})

        return coeffs_df


def fit_test(peg):
    x = np.linspace(0, 2)

    slope = depletion.depletion_volume.loc[depletion.depletion_volume['peg'] == peg, 'slope'].values

    y = slope * x

    plt.plot(x, y, label='fit')
    depletion.plot_delta_G(peg)




if __name__ == '__main__':
    depletion = Depletion()

    df = depletion.depletion_volume

    depletion.plot_delta_G('EG')

    # plt.plot(df['molar mas'], df['volume [nm]'], '.', label='experimental')
    # plt.plot(df['molar mas'], df['calc volume [nm]'], label='calculated')
    # plt.ylabel(r'interlaping volume [nm$^3$]')
    # plt.legend()
    # plt.show()
    # plt.savefig('interlaping_volume.png')
