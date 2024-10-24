import pandas as pd
import uncertainties as unc
import pickle


# Function to convert strings with uncertainties to ufloat
def convert_to_ufloat(value):
    if isinstance(value, str):  # Check if the value is a string
        # Normalize the string by replacing "±" and "+-" with spaces
        value = value.replace('±', ' ').replace('+-', ' ')
        parts = value.split()  # Split by whitespace

        # Check if we have two parts (nominal and uncertainty)
        if len(parts) == 2:
            try:
                return unc.ufloat(float(parts[0]), float(parts[1]))  # Create ufloat from parts
            except ValueError:
                return value  # Return original value if conversion fails
        else:
            try:
                return float(value)  # If it's just a number, convert directly
            except ValueError:
                return value  # Return original value if conversion fails
    return value  # Return as is if it's already a number


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
    final_df[col] = final_df[col].map(convert_to_ufloat)

final_df['K [M]'] = 1 / final_df['KD (nM)'] * 10 ** 9

final_df['molar mass'] = final_df['probe'].apply(get_molar_mass)

final_df['c [M]'] = final_df['C (g/ml)'] / final_df['molar mass'] * 1000

final_df['monomers number'] = final_df['molar mass'].apply(get_number_of_monomers)

class PNAS_data:

    def __init__(self):
        self.data = final_df

    def get_probe(self, name):
        mask = self.data['probe'] == name
        df = self.data[mask]

        return df

    # TODO   fitting, saving, create github
