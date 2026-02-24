import os
import pandas as pd

import utils as uts
import uncertainties as unc
from uncertainties import umath
import numpy as np


class EquillibriumConstants:

    def __init__(self, source_data_path):
        self.raw_data = self.read_reactions_equillibrium_data(source_data_path)
        self.analyzed_data = self.calculate_parabolic_fit()

    @staticmethod
    def calculate_solution_properties(data):
        data['crowder'] = data['crowder'].bfill()
        data['molar mass [g/mol]'] = data['molar mass [g/mol]'].bfill()

        data['density'] = 0.99707 + 0.0017441 * data['crowder wt. [%]']  # g/cm^3

        data = pd.merge(data, pd.DataFrame(uts.crowders_properties()), left_on='crowder', right_index=True)

        data['concentration [M]'] = data['crowder wt. [%]'] * data['density'] / data['MW_[g/mol]'] * 10

        data['c*_[M]'] = data['c*_[g/cm3]'] / data['MW_[g/mol]'] * 10 ** 3

        data['c/c*'] = data['concentration [M]'] / data['c*_[M]']

        return data

    def read_reactions_equillibrium_data(self, path):
        files = os.listdir(path)
        files_no_txt = [file for file in files if file.endswith('xlsx')]

        df_list = []

        for file in files_no_txt:
            df = pd.read_excel(os.path.join(path, file))
            df['source'] = [file.split('.')[0]] * df.shape[0]

            df_list.append(df)

        df_concat = pd.concat(df_list)

        df_K_error = uts.combine_to_ufloat(df_concat, 'K [M]', 'K error [M]', 'K_uf [M]', drop=True)
        df_solution = self.calculate_solution_properties(df_K_error)

        return df_solution

    def fit_quadratic(self, data):
        x = np.array(data['concentration [M]'], dtype=np.float64)
        y, y_err = uts.get_float_uncertainty(data['K_uf [M]'].apply(lambda x: umath.log(x)))

        coefficients, cov_matrix = np.polyfit(x, y, 2, w=1 / y_err, cov=True)

        errors = np.sqrt(np.diag(cov_matrix))

        y_fit = np.polyval(coefficients, x)

        # chi-square
        chi2 = np.sum(((y - y_fit) / y_err) ** 2)
        dof = len(x) - 3  # 3 parameters: a, b, c
        chi2_red = chi2 / dof if dof > 0 else np.nan

        ss_res = np.sum((y - y_fit) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot

        return pd.Series(
            {
                "a2_exp": unc.ufloat(coefficients[0], errors[0]),
                "a1_exp": unc.ufloat(coefficients[1], errors[1]),
                "a0_exp": unc.ufloat(coefficients[2], errors[2]),
                "chi2": chi2,
                "chi2_red": chi2_red,
                "R2": r2
            }
        )

    @staticmethod
    def filter_quadratic_fits(dat):
        filtering_mask = (dat['chi2_red'] > 0.2) & (dat['chi2_red'] < 10) & (dat['R2'] > 0.5) & (dat['a1_exp'] > 0) & (dat['a2_exp'] < 0)

        return dat[filtering_mask]

    def calculate_parabolic_fit(self):
        df = self.raw_data.copy()

        grouping_columns = ['source', 'sample', 'crowder']

        fitted = df.groupby(by=grouping_columns).apply(self.fit_quadratic).reset_index()

        merged = pd.merge(df, fitted, on=grouping_columns)
        #filtered = self.filter_quadratic_fits(merged)

        return merged


