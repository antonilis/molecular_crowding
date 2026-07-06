import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import unumpy as unp



class SodiumDependency:

    def __init__(self, raw_path):

        self.R = 8.31446261815324
        self.T = 273.15 + 25
        self.raw_data_path = raw_path
        self.raw_data = self.read_raw_data()

        self.popt_eff, self.pcov_eff = self.fit_phenomenological_equation()


    def read_raw_data(self):

        df = pd.read_csv(self.raw_data_path)
        df['dG_[kJ/mol]'] = - self.R * self.T  * np.log(df['K_Na+_[1/M]'])/1000

        return df

    @staticmethod
    def model(x, A, B, k):

        if any(
                hasattr(i, "nominal_value") and hasattr(i, "std_dev")
                for i in x
        ):
            result = A + B * unp.exp(-k * x)
        else:
            result = A + B * np.exp(-k * x)

        return result

    def fit_phenomenological_equation(self):
        popt_eff, pcov_eff = curve_fit(self.model, self.raw_data['C_Na+_[mM]'], self.raw_data['dG_[kJ/mol]'],
                                       p0=[-140, 100, 0.2])

        return popt_eff, pcov_eff

    def calculate_sodium_dG(self, CNa):

        deltaG = self.model(CNa, self.popt_eff[0], self.popt_eff[1], self.popt_eff[2])


        return deltaG




if '__main__' == __name__:

   path = '../source_data/IonStrengthDependency/Na+_FRET.csv'

   ionic_strength =SodiumDependency(path)
