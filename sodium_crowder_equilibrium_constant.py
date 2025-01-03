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
    # correctioin taken from paper: https://doi.org/10.1039/B908386C
    a, b = 0.7, 1.45
    ksi = df['Rg_[nm]'] * (df['mass concentration [g/cm3]'] / df['c*_[g/cm3]']) ** (-0.75)

    def calculate_corrected_diffusion(row):
        if row['Rg_[nm]'] < 0.5:
            # Apply macroviscosity formula
            macro_viscosity = np.exp(b * (row['Rg_[nm]'] / ksi[row.name]) ** a)
            return D_Na0 / macro_viscosity
        else:
            # Apply nanoviscosity correction
            nano_viscosity = np.exp(b * (rh / ksi[row.name]) ** a)
            return D_Na0 / nano_viscosity

    # Compute corrected diffusion coefficient for each row
    df['D_Na_[um2/s] corr'] = df.apply(calculate_corrected_diffusion, axis=1)
    print(rh)
    return df


def calculate_columns_for_fit(df):
    df['x axis'] = np.log(df['monomers concentration [M]'])

    df['x axis'] = df['monomers concentration [M]']

    B = df['D_crowder_[um2/s]']

    D0 = df['D_Na_[um2/s] corr']

    df['y axis'] = (df['D_Na_[um2/s]'] - D0) / (B - D0)  # around 0

    # df['y axis'] = 9/4 * (df['D_Na_[um2/s]'] - (2/3) * (D0 + B/2)) / (B - D0) + 1/2    # around 1/2

    # df['y axis'] = 4 * (df['D_Na_[um2/s]'] - 1/2 * (D0 + B)) / (B - D0) + 1     # around 1

    # Remove rows where column 'y axis' is negative
    df = df[df['y axis'] >= 0]

    df.loc[:, 'y axis'] = df['y axis'].apply(lambda x: umath.log(x))

    return df


def equillibrium_constant_fit(df):
    # Create lists to store the new columns' data
    slopes = []
    intercepts = []

    crowders = df['crowder'].unique()

    for crowder in crowders:
        crowder_data = df[df['crowder'] == crowder]  # Filter data for this specific crowder
        slope, intercept = uts.linear_fit_with_y_err(crowder_data['x axis'],
                                                     crowder_data['y axis'])

        # Append values to the lists, repeated for all rows of this crowder
        slopes.extend([slope] * len(crowder_data))
        intercepts.extend([intercept] * len(crowder_data))

        print(crowder)

    # Add new columns to the DataFrame
    df['slope'] = slopes
    df['intercept'] = intercepts
    df['K complex'] = df['intercept'].apply(lambda x: umath.exp(x))
    df['real x'] = (df['K complex'] * df['monomers concentration [M]']) ** (df['slope'])

    return df


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


def calc_sodium_crowder_eq_constant():
    path = 'source_data/D_Na_and_D_crowder'

    # Load and combine data
    data = load_and_combine_csv_files(path)

    # Merge with crowder properties
    data = merge_with_crowder_properties(data, uts.crowders_properties)

    # Calculate density and concentration
    data = calculate_density_and_concentration(data)

    # Calculate Na0 (mean D_Na at wt_% == 0)
    data = calculate_viscosity_correction(data)

    # Clean data from NaN values
    data.dropna(inplace=True)


    # Combine errors and values as one unc.float()
    data = uts.combine_value_and_error(data, 'D_Na_[um2/s]', 'D_Na_err_[um2/s]')
    data = uts.combine_value_and_error(data, 'D_crowder_[um2/s]', 'D_crowder_err_[um2/s]')

    # # get x and y axis for fit
    data = calculate_columns_for_fit(data)

    # perform linear fit for each crowder
    data = equillibrium_constant_fit(data)

    plot_fits(data)

    return data


if __name__ == "__main__":
    final_df = calc_sodium_crowder_eq_constant()


