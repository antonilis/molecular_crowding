import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import unumpy as unp
import utils as uts



class SodiumDependency:

    def __init__(self, raw_path):

        self.R = 8.31446261815324
        self.T = 273.15 + 25
        self.raw_data_path = raw_path
        self.raw_data = self.read_raw_data()

        self.popt_eff, self.pcov_eff, self.red_chi2, self.residuals = self.fit_phenomenological_equation()


    def read_raw_data(self):

        df = pd.read_csv(self.raw_data_path)

        uts.combine_to_ufloat(df, 'K_Na+_[1/M]', 'K_err_Na+_[1/M]', out_col='K_uf_Na+_[1/M]')

        df['dG_uf_[kJ/mol]'] = - self.R * self.T  * unp.log(df['K_uf_Na+_[1/M]'])/1000

        return df

    @staticmethod
    def model(x, A, B, k):

        if any(
                hasattr(i, "nominal_value") and hasattr(i, "std_dev")
                for i in x
        ):
            result = A + B * unp.exp(- k * unp.sqrt(x))
        else:
            result = A + B * np.exp(-k * np.sqrt(x))

        return result



    def fit_phenomenological_equation(self):


        dG, dG_err = uts.get_float_uncertainty(self.raw_data['dG_uf_[kJ/mol]'])

        popt_eff, pcov_eff = curve_fit(self.model, self.raw_data['C_Na+_[mM]'], dG, sigma=dG_err, absolute_sigma=True, p0=[-50, 50, 0.2])

        parameters_err = np.sqrt(pcov_eff)

        dG_fit = self.model(self.raw_data['C_Na+_[mM]'], *popt_eff)

        chi2 = np.sum(((dG - dG_fit) / dG_err) ** 2)

        residuals = (dG - dG_fit) / dG_err

        dof = len(dG) - len(popt_eff)

        chi2_red = chi2 / dof

        return popt_eff, parameters_err, chi2_red, residuals

    def calculate_sodium_dG(self, CNa):

        deltaG = self.model(CNa, self.popt_eff[0], self.popt_eff[1], self.popt_eff[2])

        return deltaG


if '__main__' == __name__:

   #path = '../source_data/IonStrengthDependency/Na+_FRET.csv'

   path = '../source_data/IonStrengthDependency/K_vs_K+_and_Na+_FRET.csv'
   ionic_strength =SodiumDependency(path)
