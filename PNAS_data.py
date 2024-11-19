import pandas as pd
import uncertainties as unc
from uncertainties import unumpy as unp
import numpy as np
import utils as uts


def get_molar_mass(probe):
    options = {'Ethylene glycol': 62.07, 'Diethylene glycol': 106.12, 'Triethylene glycol': 150.17}
    if probe in options:
        return options[probe]

    else:
        return float(probe.split(' ')[1])


def get_number_of_monomers(Mmas):
    polymer_number = (Mmas - 18.02) / (62.07 - 18.02)  # number of monomers in polymer
    return polymer_number


def combine_dataframes(df_dic):
    """Combine all DataFrames in the dictionary into a single DataFrame with an additional 'probe' column."""
    combined_dfs = []

    for key, df in df_dic.items():
        # Add a new column 'probe' with the current key
        df_with_probe = df.copy()  # Make a copy of the DataFrame
        df_with_probe['probe'] = key  # Assign the probe key to the new column
        combined_dfs.append(df_with_probe)  # Append to the list

    # Concatenate all DataFrames in the list into one
    final_df = pd.concat(combined_dfs, ignore_index=True)

    return final_df


data = pd.read_excel('PNAS_SI_data.xlsx', sheet_name='Sheet1', dtype=str)

# indices of rows with NaN values
df_nan_rows = list(data.loc[data.isnull().any(axis=1)].index)

df_dic = {}

for i in range(len(df_nan_rows)):
    # Get the start index from df_nan_rows
    start_index = df_nan_rows[i]

    # Determine the end index
    if i + 1 < len(df_nan_rows):
        end_index = df_nan_rows[i + 1]
    else:
        end_index = len(data)  # To include all remaining rows

    # Use the first column value as the key and slice accordingly
    key = data.iloc[start_index, 0]

    # Slice the DataFrame
    temp_df = data.iloc[start_index:end_index, :]  # Slice all columns for the specified rows

    # Reset index
    temp_df.reset_index(drop=True, inplace=True)

    # Check if there are at least 3 rows before accessing them
    if len(temp_df) > 2:
        new_column_names = temp_df.iloc[1]  # Get the third row for new column names
        temp_df = temp_df[2:]  # Keep only rows after the third row
        temp_df.columns = new_column_names  # Set new column names
    else:
        continue  # Skip this iteration if there aren't enough rows

    # Store in dictionary
    df_dic[key] = temp_df

final_df = combine_dataframes(df_dic)

for col in final_df.columns:
    final_df[col] = final_df[col].map(uts.convert_to_ufloat)

final_df['K [M]'] = 1 / final_df['KD (nM)'] * 10 ** 9

final_df['molar mass'] = final_df['probe'].apply(get_molar_mass)

final_df['c [M]'] = final_df['C (g/ml)'] / final_df['molar mass'] * 1000

final_df['monomers number'] = final_df['molar mass'].apply(get_number_of_monomers)


class PNAS_data:

    def __init__(self):
        self.data = final_df
        self.coeff = self.fit_coefficients()

    def get_probe(self, name):
        mask = self.data['probe'] == name
        df = self.data[mask]

        return df


    def fit_coefficients(self):
        names = self.data['probe'].unique()

        coeff_dict = {}
        for name in names:
            x = np.array(self.get_probe(name)['c [M]'], dtype=np.float64)
            y, y_error = uts.get_float_uncertainty(unp.log(self.get_probe(name)['K [M]']))
            coeff = np.polyfit(x, y, 2, w=1 / y_error)
            coeff_dict[name] = coeff

        # Convert the dictionary to a DataFrame
        coeffs_df = pd.DataFrame.from_dict(coeff_dict, orient='index',
                                           columns=['a2', 'a1', 'a0'])

        # Reset the index to make 'peg' a column
        coeffs_df = coeffs_df.reset_index().rename(columns={'index': 'probe'})

        return coeffs_df


PNAS = PNAS_data()

