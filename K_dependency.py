import pandas as pd
import numpy as np
import uncertainties as unc
from uncertainties import umath
import utils as uts


def choose_peg(data, peg):
    df = data[data['crowder'] == peg]
    return df


def fit_quadratic_to_experimental_K(data):
    names = data['crowder'].unique()

    coeffs_dict = {}

    for name in names:
        x = np.array(choose_peg(data, name)['concentration [M]'], dtype=np.float64)
        y, y_err = uts.get_float_uncertainty(choose_peg(data, name)['K'].apply(lambda x: umath.log(x)))

        coefficients, cov_matrix = np.polyfit(x, y, 2, w=1 / y_err, cov=True)

        errors = np.sqrt(np.diag(cov_matrix))

        ufloat_coefficients = [unc.ufloat(coefficients[i], errors[i]) for i in range(len(coefficients))]

        coeffs_dict[name] = ufloat_coefficients

    # Convert the dictionary to a DataFrame
    coeffs_df = pd.DataFrame.from_dict(coeffs_dict, orient='index',
                                       columns=['a2', 'a1', 'a0'])

    # Reset the index to make 'peg' a column
    coeffs_df = coeffs_df.reset_index().rename(columns={'index': 'crowder'})

    return coeffs_df


def prepare_and_fit_experimental_data():
    data = pd.read_csv('./results/K_DNA-DNA_in_crowder_solutions.csv')

    data.dropna(inplace=True)

    # changing +- sign to the unc float objects
    for col in data.columns:
        data[col] = data[col].map(uts.convert_to_ufloat)

    data = pd.merge(data, pd.DataFrame(uts.crowders_properties()), left_on='crowder', right_index=True)

    data['density'] = 0.997 + data['d_coef'] * data['wt_%']

    data['concentration [M]'] = data['wt_%'] * data['density'] / data['MW_[g/mol]'] * 10

    coefficients = fit_quadratic_to_experimental_K(data)
    coefficients = pd.merge(coefficients, pd.DataFrame(uts.crowders_properties()), left_on='crowder', right_index=True)

    return (data, coefficients)


###########  theretical calculations ##########################
def calculate_a1mon_a2mon_theoretical_values(c0=uts.c0, zi=uts.zi, a=uts.a, Beta_m=uts.Beta_m, reagent_Rg=uts.Rg_ssDNA,
                                             Temperature=uts.T):
    alpha = np.sqrt((uts.eps * uts.kb * Temperature) / (2 * uts.Na * uts.qe ** 2))

    Zi = zi * uts.qe / (4 * np.pi * uts.eps)

    numerator = uts.Na * zi * uts.qe * Zi * np.sqrt(c0)
    denominator = 3 * np.sqrt(6) * (3 * a + 2) ** (3 / 2) * np.e * alpha

    lambd = numerator / denominator

    theory_a2 = -(4 * a + 2) * lambd / (uts.R * Temperature) * Beta_m ** 2

    theory_a1_monomer = 2 * (5 * a + 3) * lambd / (uts.R * Temperature) * Beta_m

    theory_a1_Rg = 4 * np.pi * uts.Rg_ssDNA ** 2 * uts.Na

    result_df = pd.DataFrame(
        {'theory a1 monomer': [theory_a1_monomer], 'theory a1 Rg': theory_a1_Rg, 'theory a2m': [theory_a2]})

    return result_df


if __name__ == '__main__':
    dat, fit = prepare_and_fit_experimental_data()

    theory = calculate_a1mon_a2mon_theoretical_values()

# pd.set_option('display.max_rows', None)  # Show all rows
# pd.set_option('display.max_columns', None)  # Show all columns
# pd.set_option('display.width', None)  # Ensure the table fits the screen width
