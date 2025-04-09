import os
import pandas as pd
import uncertainties as unc

import utils as uts
import numpy as np
from uncertainties import umath
import matplotlib.pyplot as plt


def load_and_combine_csv_files(path):
    """Load CSV files from the specified directory and combine them into a single DataFrame."""
    combined_list = []
    files = [f for f in os.listdir(path) if f.endswith('.csv')]

    for file in files:
        file_path = os.path.join(path, file)
        var_name = os.path.splitext(file)[0]  # Extract the file name without extension
        df = pd.read_csv(file_path)
        df['crowder'] = var_name  # Add a column to store the source
        combined_list.append(df)

    return pd.concat(combined_list, ignore_index=True)


def merge_with_crowder_properties(data, properties_func):
    """Merge the data DataFrame with additional crowder properties."""
    additional_df = properties_func().reset_index()
    additional_df.rename(columns={'index': 'crowder'}, inplace=True)
    return pd.merge(data, additional_df, on='crowder', how='left')


def calculate_density_and_concentration(df):
    """Calculate density and concentration for the merged DataFrame."""
    df['density'] = 0.997 + df['d_coef'] * df['wt_%']
    df['monomers concentration [M]'] = df['wt_%'] * df['density'] / df['MW_[g/mol]'] * 10 * df['No_mono']
    df['mass concentration [g/cm3]'] = df['wt_%'] * df['density'] / 100

    return df


def calculate_viscosity_correction(df):
    # average diffusion coefficient of the peg
    D_Na0 = df[df['wt_%'] == 0]['D_Na_[um2/s]'].mean()

    rh = uts.calculate_hydrodynamic_radius(D_Na0)

    b = 1.75

    df['ksi'] = df['Rg_[nm]'] * (df['mass concentration [g/cm3]'] / df['c*_[g/cm3]']) ** (-0.75)

    def viscosity_correction(row):

        Reff = ((row['Rh_[nm]'] ** 2 * rh ** 2) / (row['Rh_[nm]'] ** 2 + rh ** 2)) ** (0.5)

        critical_point = Reff / row['ksi']

        if critical_point < 1:
            a = 1.29
        else:
            a = 0.78

        return D_Na0 / np.exp(b * (Reff / row['ksi']) ** a)

    df['D_Na_[um2/s] corr'] = df.apply(viscosity_correction, axis=1)

    return df, D_Na0


def calculate_columns_for_fit(df):
    df['x axis'] = np.log(df['monomers concentration [M]'])

    def columns_calculation(row):

        B = row['D_crowder_[um2/s]']
        D0 = row['D_Na_[um2/s] corr']

        if row['monomers concentration [M]'] < 2:

            return (row['D_Na_[um2/s]'] - D0) / (B - D0)

        else:

            return 9 / 4 * (row['D_Na_[um2/s]'] - (2 / 3) * (D0 + B / 2)) / (B - D0) + 1 / 2

    df['y axis'] = df.apply(columns_calculation, axis=1)

    # Remove rows where column 'y axis' is negative
    df = df[df['y axis'] >= 0]

    df.loc[:, 'y axis'] = df['y axis'].apply(lambda x: umath.log(x))

    return df


def equillibrium_constant_fit(df):
    # Create lists to store the new columns' data
    slopes = []
    intercepts = []

    # Filter out crowders with less than 2 data points
    valid_crowders = df.groupby('crowder').filter(lambda x: len(x) >= 2)

    crowders = valid_crowders['crowder'].unique()

    for crowder in crowders:
        crowder_data = valid_crowders[valid_crowders['crowder'] == crowder]  # Filter data for this specific crowder

        slope, intercept = uts.linear_fit_with_y_err(crowder_data['x axis'],
                                                     crowder_data['y axis'])

        # Append values to the lists, repeated for all rows of this crowder
        slopes.extend([slope] * len(crowder_data))
        intercepts.extend([intercept] * len(crowder_data))

    # Add new columns to the DataFrame
    valid_crowders['slope'] = slopes
    valid_crowders['intercept'] = intercepts
    valid_crowders['Beta complex'] = valid_crowders['intercept'].apply(lambda x: umath.exp(x))
    valid_crowders['real x'] = (valid_crowders['Beta complex'] * valid_crowders['monomers concentration [M]']) ** (
        valid_crowders['slope'])

    return valid_crowders


def plot_fits(df):
    crowders = df['crowder'].unique()
    for crowder in crowders:
        crowder_data = df[df['crowder'] == crowder]  # Filter data for this specific crowder

        x = crowder_data['x axis']
        y, y_err = uts.get_float_uncertainty(crowder_data['y axis'])
        slope = uts.get_float_uncertainty(crowder_data['slope'])[0]
        intercept = uts.get_float_uncertainty(crowder_data['intercept'])[0]

        plt.errorbar(x, y, yerr=y_err, fmt='o', label='Experimental point', capsize=3)
        plt.plot(x, slope * x + intercept, label='fit')
        plt.title(crowder)
        plt.legend()
        plt.savefig(os.path.join('./plots', f"{crowder}.png"))
        plt.close()


def final_values_of_beta_n(df):

    not_pegs = ['Dextran6000', 'Dextran70000', 'Ficoll400000']

    filtered_data = df[~df['crowder'].isin(not_pegs)]

    Beta_complex_peg = sum(filtered_data['Beta complex']) / filtered_data.shape[0]
    n_complexation_peg = sum(filtered_data['slope']) / filtered_data.shape[0]

    Beta_complex_ficoll = df[df['crowder'] == 'Ficoll400000']['Beta complex'].iloc[0]
    n_complexation_ficoll = df[df['crowder'] == 'Ficoll400000']['slope'].iloc[0]

    Beta_complex_dex = df[df['crowder'] == 'Dextran70000']['Beta complex'].iloc[0]
    n_complexation_dex = df[df['crowder'] == 'Dextran70000']['slope'].iloc[0]

    results_df = pd.DataFrame({
        'Crowder': ['PEG', 'Ficoll', 'Dextran'],
        'Beta Complex': [Beta_complex_peg, Beta_complex_ficoll, Beta_complex_dex],
        'n Complexation': [n_complexation_peg, n_complexation_ficoll, n_complexation_dex]
    })

    return results_df


def calc_sodium_crowder_eq_constant():
    path = 'source_data/D_Na_and_D_crowder'

    # Load and combine data
    raw_data = load_and_combine_csv_files(path)

    # Merge with crowder properties
    data = merge_with_crowder_properties(raw_data, uts.crowders_properties)

    # Calculate density and concentration
    data = calculate_density_and_concentration(data)

    # Calculate Na0 (mean D_Na at wt_% == 0)
    # D_Na0 = 1425
    data, D_Na0 = calculate_viscosity_correction(data)
    # data['D_Na_[um2/s] corr'] = [D_Na0] * data.shape[0]

    # Clean data from NaN values
    data.dropna(inplace=True)

    # Combine errors and values as one unc.float()
    data = uts.combine_value_and_error(data, 'D_Na_[um2/s]', 'D_Na_err_[um2/s]')
    data = uts.combine_value_and_error(data, 'D_crowder_[um2/s]', 'D_crowder_err_[um2/s]')

    # # get x and y axis for fit
    data = calculate_columns_for_fit(data)

    # perform linear fit for each crowder
    data = equillibrium_constant_fit(data)

    data.sort_values(by=['MW_[g/mol]', 'monomers concentration [M]'], inplace=True)

    # plot_fits(data)

    final_complexation_slope_values = final_values_of_beta_n(data)

    return (data, raw_data, D_Na0, final_complexation_slope_values)


if __name__ == "__main__":
    dat = calc_sodium_crowder_eq_constant()
