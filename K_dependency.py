import pandas as pd
import numpy as np
import uncertainties as unc
from uncertainties import umath
import utils as uts


def get_molar_mass(probe):
    if probe == 'EGly':
        return 62.07

    else:
        return float(probe.split('G')[1])

def calculate_Rg(Mmol):
    Rg = 0.215 * Mmol ** (0.583) / 10  # result is in [nm]

    return Rg

def choose_peg(data, peg):
    df = data[data['crowder'] == peg]
    return df

def calculate_number_of_monomers(Mmas):
    polymer_number = (Mmas - 18.02) / (62.07 - 18.02)  # number of monomers in polymer
    return polymer_number

def fit_polynomial_to_K(data):
    names = data['crowder'].unique()

    coeffs_dict = {}

    for name in names:

        x = np.array(choose_peg(data, name)['concentration [M]'], dtype=np.float64)
        y, y_err = uts.get_float_uncertainty(choose_peg(data, name)['K'].apply(lambda x: umath.log(x)))

        coefficients, cov_matrix = np.polyfit(x, y, 2, w= 1/y_err, cov=True)

        errors = np.sqrt(np.diag(cov_matrix))

        ufloat_coefficients = [unc.ufloat(coefficients[i], errors[i]) for i in range(len(coefficients))]

        coeffs_dict[name] = ufloat_coefficients

    # Convert the dictionary to a DataFrame
    coeffs_df = pd.DataFrame.from_dict(coeffs_dict, orient='index',
                                       columns=['a2', 'a1', 'a0'])

    # Reset the index to make 'peg' a column
    coeffs_df = coeffs_df.reset_index().rename(columns={'index': 'crowder'})

    return coeffs_df


data = pd.read_csv('./results/K_DNA-DNA_in_crowder_solutions.csv')

data.dropna(inplace=True)

data['molar mass'] = data['crowder'].apply(get_molar_mass)

density = 1

data['concentration [M]'] = data['mass concentration'] * density / data['molar mass'] * 10

for col in data.columns:
    data[col] = data[col].map(uts.convert_to_ufloat)


coefficients = fit_polynomial_to_K(data)

coefficients['molar mass'] = coefficients['crowder'].apply(get_molar_mass)
coefficients['Rg'] = calculate_Rg(coefficients['molar mass'])
coefficients['monomer numbers'] = calculate_number_of_monomers(coefficients['molar mass'])


###########  theretical calculations ##########################

#
eps = 8.8541878188 * 10**(-12) * 81 # vacuum * water relative permittivity
Na = 6.02214076 * 10**(23) # avogadro number
qe = 1.60217663 * 10**(-19) # elementary charge of electrion in Culomb
kb = 1.380649 * 10**(-23) # boltzman constant
R =  8.31446261815324 #
T = 298.15

Rg_ssDNA = 0.55 # radius of ssDNA

#experimental data
c0 = 35 # concentration of Na+ in mmol which is equal to mol/m^3
zi = 13 # charge of the ssDNA 13bp
a = 0.0754/0.1754 * 4 + 0.0246/0.1754 # the anions part of the ionic strength HPO42- and H2PO4-
Km = 0.14 # complexation constant of the PEG - Na complexation