import numpy as np
import pandas as pd
import utils as uts
import os
from lmfit import Model

AVOGADRO = 6.02e23
RI_WATER = 1.333


# returns a dataframe with average RI values and related errors of PEG solutions
def calculate_average_RI_with_error_of_sample():
    # Provide a csv. file as a source.
    df = pd.read_csv('RI_of_PEG_solutions.csv')
    names = ['EGly', 'PEG200', 'PEG400', 'PEG600', 'PEG1000', 'PEG1500', 'PEG3000', 'PEG6000', 'PEG12000', 'PEG20000',
             'PEG35000']
    data = {name: [] for name in names}  # Initialize a dictionary to store the data, where each key is a column name

    for name in names:
        for i in range(0, len(df[f'{name}_wt_%'])):
            one_sample = df.loc[
                i, f'{name}_RI_1':f'{name}_RI_4']  # Ensure columns are consecutively ordered in DataFrame
            average_RI_per_sample = np.average(one_sample)
            std_RI_per_sample = np.std(one_sample)
            data[name].append(f'{average_RI_per_sample:.5f}±{std_RI_per_sample:.5f}')

    df_result = pd.DataFrame(data)  # Convert the dictionary to a DataFrame
    df_result.insert(0, 'wt_%', df['PEG200_wt_%'])  # inserting a column with wt%
    return df_result


# Returns a dataframe with slopes for RI values of PEG solutions
def calculate_RI_slope_with_error():
    # Collect results in a list
    results = []
    names = ['EGly', 'PEG200', 'PEG400', 'PEG600', 'PEG1000', 'PEG1500', 'PEG3000', 'PEG6000', 'PEG12000', 'PEG20000',
             'PEG35000']
    RI_avg_dataframe = calculate_average_RI_with_error_of_sample()

    # Convert columns to `ufloat`
    for col in RI_avg_dataframe.columns:
        RI_avg_dataframe[col] = RI_avg_dataframe[col].map(uts.convert_to_ufloat)
    # Loop through each name and calculate slope, intercept, and r_squared
    for name in names:
        if str(RI_avg_dataframe[name].iloc[-1]) == 'nan+/-nan':  # Checks if last cell in column is nan+/-nan
            slope, intercept, r_squared = uts.linear_fit_normal(
                RI_avg_dataframe.loc[0:RI_avg_dataframe[name].last_valid_index() - 1, 'wt_%'],
                RI_avg_dataframe.loc[0:RI_avg_dataframe[name].last_valid_index() - 1, name])
        else:
            slope, intercept, r_squared = uts.linear_fit_normal(
                RI_avg_dataframe.loc[0:RI_avg_dataframe[name].last_valid_index(), 'wt_%'],
                RI_avg_dataframe.loc[0:RI_avg_dataframe[name].last_valid_index(), name])

        # Append the results to the list
        results.append({'name': name, 'slope': slope, 'intercept': intercept, 'r_squared': r_squared})

    # Create a DataFrame from the results list
    results_df = pd.DataFrame(results)

    # Set 'name' as the index
    results_df.set_index('name', inplace=True)
    return results_df


# extract name of crowder and its concentration based on the filename within a specific path
def extract_name_and_value_from_a_file(file_path):
    base_name = os.path.splitext(file_path)[0]  # Remove the .csv extension
    parts = base_name.split("_")  # Split by underscore
    if len(parts) == 2 and "p" in parts[1]:
        name = parts[0]  # Extract the name (part before the underscore)
        number1, number2 = parts[1].split("p")  # Split the second part by 'p'
        value = f"{int(number1)}.{int(number2)}"  # Convert to number1.number2 format
        return name, value, base_name
    return None, None, None  # Return None for both if the filename doesn't match the expected format


# --- Model definition (outside loop!) ---
def K_binding_model(x, K, D, alfa, gamma, v0, n=1):
    term = x / n + D + 1 / K
    delta = term ** 2 - 4 * (x / n) * D
    sqrt_delta = np.sqrt(delta)

    binding = 0.5 * (term - sqrt_delta)

    return alfa * v0 * (
            D - binding
    ) * (
            1 + (gamma / alfa) * K * (x / n - binding)
    )


# --- Helpers ---
def load_dataset(filepath):
    return pd.read_csv(filepath)


def compute_refractive_index(value, name, df_RI):
    return value * df_RI.at[name, 'slope'] + df_RI.at[name, 'intercept']


def compute_effective_donor_concentration(wt_percent, C_nominal=1e-8):
    rho_water = 0.99707  # g/mL
    rho_mix = 0.99707 + 0.0017441 * wt_percent

    V_water = 0.2 / rho_water

    m_mix = 0.2/(100 - wt_percent) * 100

    V_mix = m_mix / rho_mix

    volume_factor = V_mix / V_water

    C_eff = C_nominal / volume_factor

    return C_eff


def compute_physical_parameters(df, value, RI):
    z_correction = RI / RI_WATER
    C_Don = compute_effective_donor_concentration(value, 1e-8)

    # C_Don = 1e-8 * ((100 - value) / 100) # old way of doing stuff

    alfa = df['a'].iloc[0]
    gamma = alfa * df['y'].iloc[0]

    v0 = AVOGADRO * 1e-16 * 10 * df['Vef_[fl]'].iloc[0] * z_correction

    return alfa, gamma, v0, C_Don


def fit_single_dataset(df, alfa, gamma, v0, C_Don):
    x = df['ratio'] * C_Don
    y = df['value']

    model = Model(
        lambda x, K, D: K_binding_model(x, K, D, alfa, gamma, v0)
    )

    model.set_param_hint('K', value=1e6, min=1e2, max=5e13)
    model.set_param_hint('D', value=1e-8, min=1.1e-9, max=2e-8)

    result = model.fit(y, x=x)

    return {
        'K': result.params['K'].value,
        'K_err': result.params['K'].stderr,
        'D_fit': result.params['D'].value,
        'D_fit_err': result.params['D'].stderr
    }


def extract_metadata(directory):
    names, base_names, values = [], [], []

    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            name, value, base_name = extract_name_and_value_from_a_file(filename)
            if name and value:
                names.append(name)
                base_names.append(base_name)
                values.append(float(value))

    return list(zip(base_names, names, values))


# --- Main function ---
def K_DNA_DNA_fitting(data_dir='DNA-DNA_countrate_vs_crowder_concentration'):
    results = []

    df_RI = calculate_RI_slope_with_error()
    datasets = extract_metadata(data_dir)

    for base_name, name, value in datasets:
        filepath = os.path.join(data_dir, f"{base_name}.csv")
        df = load_dataset(filepath)

        # --- Physical parameters ---
        RI = compute_refractive_index(value, name, df_RI)
        alfa, gamma, v0, C_Don = compute_physical_parameters(df, value, RI)

        # --- Fit ---
        fit = fit_single_dataset(df, alfa, gamma, v0, C_Don)

        result_row = {
            'name': base_name,
            'crowder': name,
            'wt_%': value,
            'K [M]': fit['K'],
            'K error [M]': fit['K_err'],
            'D_exp [um^2/s]': df['D_[um^2/s]'].iloc[0],
            'D_err [um^2/s]': df['D_err'].iloc[0]
        }
        results.append(result_row)

    return pd.DataFrame(results)


def format_df(df):
    dat = df[['crowder', 'wt_%', 'K [M]', 'K error [M]']]

    dat['sample'] = ['ssDNA_13bp'] * dat.shape[0]
    dat['charge 1'] = dat['charge 2'] = [-13] * dat.shape[0]
    dat['Rg 1 [nm]'] = dat['Rg 2 [nm]'] = [13 * 0.6 / np.sqrt(12)] * dat.shape[0]
    dat['T [K]'] = [298.15] * dat.shape[0]

    dat['Na conc. [mM]'] = dat["wt_%"].apply(
        lambda x: compute_effective_donor_concentration(x, C_nominal=35)
    )

    dat['I [mM]'] = 1.38 * dat['Na conc. [mM]']

    data = uts.crowders_properties()['MW_[g/mol]']

    df = pd.merge(dat, data, left_on='crowder', right_index=True)

    df.dropna(inplace=True)

    df = df.rename(columns={
        "wt_%": "crowder wt. [%]",
        "MW_[g/mol]": "molar mass [g/mol]"
    })

    crowders = df['crowder'].unique()

    zero_df = pd.DataFrame({
        'crowder': crowders,
        'crowder wt. [%]': 0,
        'K [M]': 1760000000,
        'K error [M]': 311872000,
        'sample': 'ssDNA_13bp',
        'charge 1': -13,
        'charge 2': -13,
        'Rg 1 [nm]': 13 * 0.6 / np.sqrt(12),
        'Rg 2 [nm]': 13 * 0.6 / np.sqrt(12),
        'T [K]': 298.15,
        'Na conc. [mM]': 35,
        'I [mM]': 1.38 * 35
    })

    zero_df = zero_df.merge(
        df[['crowder', 'molar mass [g/mol]']].drop_duplicates(),
        on='crowder',
        how='left'
    )

    df = pd.concat([df, zero_df], ignore_index=True)
    return df


def save_to_csv(df, saving_dierectory):
    for crowder in df['crowder'].unique():
        dat = df[df['crowder'] == crowder]

        dat = dat.sort_values(by='crowder wt. [%]')

        file_name = f'OriginalData_{crowder}.xlsx'

        saving_path = os.path.join(saving_dierectory, file_name)

        dat.to_excel(saving_path, index=False)


if __name__ == '__main__':
    # here powinna być jedna funkcja, robi csv-y

    results = K_DNA_DNA_fitting()

    formated = format_df(results).sort_values(by=['crowder', 'crowder wt. [%]'])

    path = '..'

    save_to_csv(formated, path)
