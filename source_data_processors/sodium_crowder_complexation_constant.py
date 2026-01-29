import os
import pandas as pd
import uncertainties as unc
from scipy.optimize import least_squares
from scipy.optimize import curve_fit
import utils as uts
import numpy as np
from uncertainties import umath
import matplotlib.pyplot as plt


class SodiumComplexation:

    def __init__(self, path, calculate=True):

        self.path = path
        self.raw_data = None
        self.D_Na0 = None
        self.analyzed_data = None
        self.n_complexation = None

        if calculate:
            self.read_data()
            self.analyze_raw_data()
            self.fit_n_complexation_slope()

    def read_data(self):

        combined_list = []
        files = [f for f in os.listdir(self.path) if f.endswith('.csv')]

        for file in files:
            file_path = os.path.join(self.path, file)
            var_name = os.path.splitext(file)[0]
            df = pd.read_csv(file_path)
            df['crowder'] = var_name
            combined_list.append(df)

        raw_data = pd.concat(combined_list, ignore_index=True)

        raw_data = uts.combine_to_ufloat(raw_data, 'D_Na_[um2/s]', 'D_Na_err_[um2/s]', 'D_Na_uf_[um2/s]')

        self.D_Na0 = raw_data[raw_data['wt_%'] == 0]['D_Na_[um2/s]'].mean()

        raw_data.dropna(ignore_index=True, inplace=True)

        self.raw_data = uts.combine_to_ufloat(raw_data, 'D_crowder_[um2/s]', 'D_crowder_err_[um2/s]',
                                              'D_crowder_uf_[um2/s]')

        return self.raw_data

    @staticmethod
    def calculate_solution_properties(df):

        df = pd.merge(df, uts.crowders_properties(), left_on='crowder', right_index=True)

        df['density'] = 0.997 + 0.0017441 * df['wt_%']

        mass_conc = df['wt_%'] * df['density'] / 100  # g/cm3
        molar_conc = mass_conc * 1000 / df['MW_[g/mol]']  # mol/L (M)

        df['mass concentration [g/cm3]'] = mass_conc
        df['concentration [M]'] = molar_conc
        df['monomers concentration [M]'] = molar_conc * df['No_mono']

        return df

    def calculate_viscosity_correction(self, df):
        # average diffusion coefficient of the peg

        rh = uts.calculate_hydrodynamic_radius(self.D_Na0)

        b = 1.75

        df['ksi'] = df['Rg_[nm]'] * (df['mass concentration [g/cm3]'] / df['c*_[g/cm3]']) ** (-0.75)

        def viscosity_correction(row):

            Reff = ((row['Rh_[nm]'] ** 2 * rh ** 2) / (row['Rh_[nm]'] ** 2 + rh ** 2)) ** (0.5)

            critical_point = Reff / row['ksi']

            if critical_point < 1:
                a = 1.29
            else:
                a = 0.78

            return self.D_Na0 / np.exp(b * (Reff / row['ksi']) ** a)

        df['D_Na_[um2/s] corr'] = df.apply(viscosity_correction, axis=1)

        return df

    @staticmethod
    def model_for_fit(x, D_Na, D_crowder, alpha):
        return (D_Na + D_crowder * alpha * x) / (1 + alpha * x)

    def residual(self, alpha, x, y, D_Na, D_crowder, y_err):
        model = self.model_for_fit(x, D_Na, D_crowder, alpha[0])
        return (model - y) / y_err

    def equillibrium_constant_fit(self, df):
        nBeta_col = []
        chi2_col = []
        redchi2_col = []
        r2_col = []

        crowders = df['crowder'].unique()

        for crowder in crowders:
            data = df[df['crowder'] == crowder]

            x_data = data['concentration [M]'].values
            y_data, y_err = uts.get_float_uncertainty(data['D_Na_uf_[um2/s]'])

            D_Na = data['D_Na_[um2/s] corr'].values
            D_crowder, _ = uts.get_float_uncertainty(data['D_crowder_uf_[um2/s]'])

            res = least_squares(
                self.residual,
                x0=[0.1],
                args=(x_data, y_data, D_Na, D_crowder, y_err)
            )

            alpha_fit = res.x[0]

            # obliczanie błędu
            chi2 = 2 * res.cost
            dof = len(y_data) - len(res.x)
            sigma2 = chi2 / dof
            cov = np.linalg.inv(res.jac.T @ res.jac) * sigma2
            alpha_err = np.sqrt(cov[0, 0])

            # ufloat
            alpha_uf = unc.ufloat(alpha_fit, alpha_err)

            y_model = self.model_for_fit(x_data, D_Na, D_crowder, alpha_fit)

            ss_res = np.sum((y_data - y_model) ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r2 = 1 - ss_res / ss_tot
            redchi2 = chi2 / dof

            nBeta_col.extend([alpha_uf] * len(data))
            chi2_col.extend([chi2] * len(data))
            redchi2_col.extend([redchi2] * len(data))
            r2_col.extend([r2] * len(data))

        df['n Beta [1/M]'] = nBeta_col
        df['chi2'] = chi2_col
        df['redchi2'] = redchi2_col
        df['R^2'] = r2_col

        return df

    def analyze_raw_data(self):

        df = self.calculate_solution_properties(self.raw_data)

        df = self.calculate_viscosity_correction(df)

        self.analyzed_data = self.equillibrium_constant_fit(df)

        return self.analyzed_data


    def fit_n_complexation_slope(self):

        x = np.array(self.analyzed_data['No_mono'].unique())
        y, y_err = uts.get_float_uncertainty(self.analyzed_data['n Beta [1/M]'].unique())


        # doing linear with with y errors
        weights = 1 / y_err ** 2
        a = np.sum(weights * x * y) / np.sum(weights * x ** 2)
        a_err = np.sqrt(1 / np.sum(weights * x ** 2))

        n_uf = unc.ufloat(a, a_err)

        self.n_complexation = n_uf

        return n_uf

def plot_fits(df):
    crowders = df['crowder'].unique()
    for crowder in crowders:
        crowder_data = df[df['crowder'] == crowder]  # Filter data for this specific crowder

        x = crowder_data['concentration [M]']
        y, y_err = uts.get_float_uncertainty(crowder_data['D_Na_[um2/s]'])

        D_Na = crowder_data['D_Na_[um2/s] corr']

        D_crowder, err = uts.get_float_uncertainty(crowder_data['D_crowder_[um2/s]'])

        alpha = crowder_data['n Beta [1/M]'].iloc[0]

        # Interpolation grid for smooth plotting
        x_fit = np.linspace(min(x), max(x), 200)
        D_crowder_fit = np.interp(x_fit, x, D_crowder)  # interpolate B to match x_fit

        D_Na_fit = np.interp(x_fit, x, D_Na)

        y_fit = model_for_fit(x_fit, D_Na_fit, D_crowder_fit, alpha)

        plt.plot(x, y, 'v', label='Experimental point')

        plt.plot(x_fit, y_fit, label='fit')
        plt.title(crowder)
        plt.legend()
        plt.savefig(os.path.join('../plots', f"{crowder}.png"))
        plt.close()





if __name__ == "__main__":
    path = '../source_data/IonCrowderComplexation/raw_data'

    data = SodiumComplexation(path)
