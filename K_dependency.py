import pandas as pd
import numpy as np
import uncertainties as unc
from uncertainties import umath
import utils as uts
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


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

    data['c*_[M]'] = data['c*_[g/cm3]'] / data['MW_[g/mol]'] * 10 ** 3

    data['c/c*'] = data['concentration [M]'] / data['c*_[M]']

    data.sort_values(by=['MW_[g/mol]', 'concentration [M]'], inplace=True)

    coefficients = fit_quadratic_to_experimental_K(data)
    coefficients = pd.merge(coefficients, pd.DataFrame(uts.crowders_properties()), left_on='crowder', right_index=True)

    return (data, coefficients)


###########  theretical calculations ##########################


# Electrostatic interactions
def calculate_a1mon_a2mon_theoretical_values(c0=uts.c0, zi=uts.zi, a=uts.a, Beta_m=uts.Beta_m, reagent_Rg=uts.Rg_ssDNA,
                                             Temperature=uts.T):
    alpha = np.sqrt((uts.eps * uts.kb * Temperature) / (2 * uts.Na * uts.qe ** 2))

    Zi = zi * uts.qe / (4 * np.pi * uts.eps)

    numerator = uts.Na * zi * uts.qe * Zi * np.sqrt(c0)
    denominator = 3 * np.sqrt(6) * (3 * a + 2) ** (3 / 2) * np.e * alpha

    lambd = numerator / denominator

    theory_a2 = -(4 * a + 2) * lambd / (uts.R * Temperature) * Beta_m ** 2

    theory_a1_monomer = 2 * (5 * a + 3) * lambd / (uts.R * Temperature) * Beta_m

    # theory_a1_Rg = 4 * np.pi * uts.Rg_ssDNA ** 2 * uts.Na

    result_df = pd.DataFrame({'theory a1 electrostatic': [theory_a1_monomer], 'theory a2m': [theory_a2]})

    return result_df


# Depletion interactions
def calculate_depletion_layer(Rg_particle, Rg_crowder, phi):
    delta_0 = 1.07 * Rg_crowder

    if phi > 1:
        ksi = Rg_crowder * phi ** (-0.77)

        delta = (delta_0 * ksi) / (delta_0 ** 2 + ksi ** 2) ** (0.5)

    else:
        delta = delta_0

    frac = delta / Rg_particle

    delta_s = Rg_particle * ((1 + 3 * frac + 2.273 * frac ** 2 - 0.0975 * frac ** (3)) ** (1 / 3) - 1)

    return delta_s


def calc_interlap_volume(Rg_crowder, particle_size, phi):
    # r = uts.Rg_ssDNA + calculate_depletion_layer(Rg_crowder, phi)
    # R = particle_size + calculate_depletion_layer(Rg_crowder, phi)

    r = particle_size + calculate_depletion_layer(particle_size, Rg_crowder, phi)

    R = r

    d = 2 * particle_size

    overlap_volume = (np.pi * (r + R - d) ** 2 * (d ** 2 - 3 * (r - R) ** 2 + 2 * d * (r + R))) / (12 * d)

    return overlap_volume


def calculate_Pi(phi):
    numerator = 1 + 3.25 * phi + 4.15 * np.square(phi)
    denominator = 1 + 1.48 * phi
    fraction = numerator / denominator
    Pi = 1 + 2.63 * phi * np.power(fraction, 0.309)
    return Pi


def calc_potential(pi, dphi):
    potential = np.zeros(len(pi) - 1)
    for index in range(len(pi) - 1):
        potential[index] = np.sum(pi[0:index + 1]) * dphi

    return potential


def calculate_interpolation(inter_phi):
    # phi is ratio: c/c*
    phi, dphi = np.linspace(0, 30, 5000, retstep=True)

    pi = calculate_Pi(phi)
    potential = calc_potential(pi, dphi)

    potential_interpolator = interp1d(phi[0:-1], potential, kind='linear', fill_value='extrapolate')

    return potential_interpolator(inter_phi)


def calculate_depletion_gibbs_energy(Rg_crowder, phi, c_star, particle_size=uts.Rg_ssDNA,
                                     T=uts.T):  # phi is ratio: c/c*, radius are in nm and concentration in M

    volume = calc_interlap_volume(Rg_crowder, particle_size, phi)

    if phi > 1:
        potential = calculate_interpolation(phi)
    else:
        potential = 1

    gibbs_energy = - uts.kb * T * uts.Na ** 2 * volume * c_star * potential * 10 ** (-24)  # the units: J/mol

    return gibbs_energy


if __name__ == '__main__':
    dat, fit = prepare_and_fit_experimental_data()

    theory = calculate_a1mon_a2mon_theoretical_values()

    dat['G_depl [J/mol]'] = dat.apply(
        lambda row: calculate_depletion_gibbs_energy(
            Rg_crowder=row['Rg_[nm]'],
            phi=row['c/c*'],
            c_star=row['c*_[M]']
        ),
        axis=1  # Apply row-wise
    )

# pd.set_option('display.max_rows', None)  # Show all rows
# pd.set_option('display.max_columns', None)  # Show all columns
# pd.set_option('display.width', None)  # Ensure the table fits the screen width
