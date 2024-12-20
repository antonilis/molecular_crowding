import os
import pandas as pd
import uncertainties

import utils as uts
import numpy as np
from uncertainties import umath


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
    df['concentration [M]'] = df['wt_%'] * df['density'] / df['MW_[g/mol]'] * 10
    return df


def calculate_columns_for_fit(df, D0):
    df['x axis'] = df['concentration [M]'].apply(lambda x: umath.log(x))

    df['y axis'] = (df['D_Na_[um2/s]'] - D0) / (df['D_crowder_[um2/s]'] - D0)

    # Remove rows where column '' is negative
    df_filtered = df[df['y axis'] >= 0]

    df_filtered.loc[:, 'y axis'] = df_filtered['y axis'].apply(lambda x: umath.log(x))

    return df_filtered


# def equillibrium_constant_fit(df):
#     crowders = df['crowders'].unique()
#
#     for crowder in crowders:
#         slope, intercept, slope_err, intercept_err = uts.linear_fit_with_x_y_errors(crowder['x axis'],
#                                                                                     crowder['y axis'])
#         slope = unc.ufloat(slope, slope_err)
#         intercept = unc.


def calc_sodium_crowder_eq_constant():
    path = 'source_data/D_Na_and_D_crowder'

    # Load and combine data
    data = load_and_combine_csv_files(path)

    # Merge with crowder properties
    data = merge_with_crowder_properties(data, uts.crowders_properties)

    # Calculate Na0 (mean D_Na at wt_% == 0)
    Na0 = data[data['wt_%'] == 0]['D_Na_[um2/s]'].mean()

    # Clean data from NaN values
    data.dropna(inplace=True)

    # Calculate density and concentration
    data = calculate_density_and_concentration(data)

    # Combine errors and values as one unc.float()
    data = uts.combine_value_and_error(data, 'D_Na_[um2/s]', 'D_Na_err_[um2/s]')
    data = uts.combine_value_and_error(data, 'D_crowder_[um2/s]', 'D_crowder_err_[um2/s]')

    # get x and y axis for fit
    data = calculate_columns_for_fit(data, Na0)

    return data


if __name__ == "__main__":
    final_df = calc_sodium_crowder_eq_constant()
