import numpy as np
import pandas as pd
import utils as uts
import os
from lmfit import Model


# returns a dataframe with average RI values and related errors of PEG solutions
def calculate_average_RI_with_error_of_sample():
    # Provide a csv. file as a source.
    df = pd.read_csv('source_data/RI_of_PEG_solutions.csv')
    names = ['EGly', 'PEG200', 'PEG400', 'PEG600', 'PEG1000', 'PEG1500', 'PEG3000', 'PEG6000', 'PEG12000', 'PEG20000',
             'PEG35000']
    data = {name: [] for name in names}  # Initialize a dictionary to store the data, where each key is a column name

    for name in names:
        for i in range(0, len(df[f'{name}_wt_%'])):
            one_sample = df.loc[i, f'{name}_RI_1':f'{name}_RI_4']  # Ensure columns are consecutively ordered in DataFrame
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
    names = ['EGly', 'PEG200', 'PEG400', 'PEG600', 'PEG1000', 'PEG1500', 'PEG3000', 'PEG6000', 'PEG12000', 'PEG20000', 'PEG35000']
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


# equation to calculate κ of ion crowder interactions
def equation_to_calculate_κ(wt_percent, MW, solution_density,  no_monomers, Na_D, PEG_self_D, Na_0):
    mass = (wt_percent / 100) / (1 - (wt_percent / 100))
    c_mono = (mass / MW) / ((mass + 1) / solution_density * 0.001) * no_monomers
    kappa = (Na_D - Na_0) / (PEG_self_D - Na_0) / c_mono
    return kappa.mean(), kappa.std()


# calculates complexation constants of sodium ions by specific crowder per monomer
def kappa_fitting():
    # returns crowders names from directory and allows to call csv files by crowders names
    crowders = []
    crowders_names = []
    path = 'source_data/D_Na_and_D_crowder'
    files = os.listdir(path)
    for file in files:
        if file.endswith('.csv'):
            file_path = os.path.join(path, file)
            var_name = os.path.splitext(file)[0]
            df = pd.read_csv(file_path)
            globals()[var_name] = df
            crowders.append(df)
            crowders_names.append(var_name)

    # Diffusion coeffcient of Na in buffer
    Na_0 = EGly['D_Na_[um2/s]'][0]  # everywhere the same

    value = uts.crowders_properties()
    kappa_results = {}
    kappa_errors = {}

    # Loop through crowders and their names
    for crowder, crowder_name in zip(crowders, crowders_names):
        # Calculate kappa and kappa error
        kappa, kappa_err = equation_to_calculate_κ(
            crowder['wt_%'],
            value.loc[crowder_name]['MW_[g/mol]'],
            0.997 +  value.loc[crowder_name]['d_coef'] * crowder['wt_%'], #density of specific crowder solution
            value.loc[crowder_name]['No_mono'],
            crowder['D_Na_[um2/s]'],
            crowder['D_crowder_[um2/s]'],
            Na_0
        )
        if crowder_name == 'EGly':  # divided by 2 because EGly has two 'monomers'
            kappa_results[crowder_name] = f'{kappa / 2}±{kappa_err / 2}'
        elif crowder_name == 'Dextran6000' or crowder_name == 'Dextran70000':  # divided by 5 because we assumed one Na+ per one oxygen atom'
            kappa_results[crowder_name] = f'{kappa / 5}±{kappa_err / 5}'
        elif crowder_name == 'Ficoll400000':  # divided by 5 because we assumed one Na+ per one oxygen atom'
            kappa_results[crowder_name] = f'{kappa / 11}±{kappa_err / 11}' # divided by 11 because we assumed one Na+ per one oxygen atom'
        else:
            kappa_results[crowder_name] = f'{kappa}±{kappa_err}'

    return kappa_results


# calculates equilibrium constants of DNA-DNA interactions in the presence of crowders
def K_DNA_DNA_fitting():
    K_results = pd.DataFrame(columns=[
        'name', 'crowder', 'wt_%', 'K', 'D'])

    names = []  # crowder name
    base_names = []  # file name
    values = []  # crowder concentration

    # Loop through all files in the directory
    for filename in os.listdir('source_data/DNA-DNA_countrate_vs_crowder_concentration'):
        if filename.endswith(".csv"):
            name, value, base_name = extract_name_and_value_from_a_file(filename)
            if name and value:
                names.append(name)
                base_names.append(base_name)
                values.append(float(value))

    for i in range(len(base_names)):
        df = pd.read_csv(f'source_data/DNA-DNA_countrate_vs_crowder_concentration/{base_names[i]}.csv')
        df_RI = calculate_RI_slope_with_error()
        RI = values[i] * df_RI.at[names[i], 'slope'] + df_RI.at[names[i], 'intercept']  # refractive index of specific solution
        RI_water = 1.333
        z_correction = RI / RI_water
        C_Don = 1e-8 * ((100 - values[i]) / 100)  # nM correction for PEG concentration in solution
        alfa = df['a'][0]
        gamma = alfa * df['y'][0]
        v0 = 6.02 * 1e23 * 1e-16 * 10 * df['Vef_[fl]'][0] * z_correction
        n = 1
        x = df['ratio'] * C_Don
        y = df['value']
        xmin = df['ratio'].min() * C_Don
        xmax = df['ratio'].max() * C_Don
        xx = np.linspace(0, xmax, num=500)

        # equation to calculate K of DNA-DNA interactions
        def K_fitting(x, K, D):
            return alfa * v0 * (
                        D - (0.5 * (x / n + D + 1 / K - np.sqrt(((-x / n - D - 1 / K) ** 2) - 4 * x / n * D)))) * (
                        1 + (gamma / alfa) * K * (
                            x / n - (0.5 * (x / n + D + 1 / K - np.sqrt(((-x / n - D - 1 / K) ** 2) - 4 * x / n * D)))))

        # Fitting parameters
        funmodel = Model(K_fitting)
        funmodel.set_param_hint('K', value=1e6, min=1e2, max=5e13)
        funmodel.set_param_hint('D', value=1e-8, min=1.1e-9, max=2e-8)
        pars = funmodel.make_params()
        result = funmodel.fit(y, x=x)
        fitK1 = result.params['K'].value
        fitK1err = result.params['K'].stderr
        fitD1 = result.params['D'].value
        fitD1err = result.params['D'].stderr
        fitt1 = ((fitD1 * 1e9,) + (uts.logdef(fitK1)) + (gamma,))

        # Saving data
        if fitK1 and fitK1err and df['D_[um^2/s]'][0]:  # Ensure no invalid or empty values
            if fitK1 > 100000:
                data_out = {
                    'name': base_names[i],
                    'crowder': names[i],
                    'wt_%': values[i],
                    'K': f'{fitK1:.0f}±{fitK1err:.0f}',
                    'D': f"{df['D_[um^2/s]'][0]}±{df['D_err'][0]}"}
            else:
                data_out = {
                    'name': base_names[i],
                    'crowder': names[i],
                    'wt_%': values[i],
                    'K': None,
                    'D': None}

            K_results = pd.concat([K_results, pd.DataFrame([data_out])], ignore_index=True)
            K_results.to_csv('results/K_DNA-DNA_in_crowder_solutions.csv', index=False)  # Save to a CSV file
    return K_results

print(kappa_fitting())









