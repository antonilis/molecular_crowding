import numpy as np
import pandas as pd
import utils as uts
import os
import math
import matplotlib.pyplot as plt
from scipy.stats import linregress
from lmfit import Model


# standard linear fit
def linear_fit_normal(x, y):
    y_value, y_error = uts.get_float_uncertainty(y)
    slope, intercept, r_value, p_value, std_err = linregress(x, y_value)
    r_squared = r_value ** 2
    return slope, intercept, r_squared

# returns a dataframe with average RI values and related errors of PEG solutions
def calculate_average_RI_with_error_of_sample(df):
    # Provide a csv. file as a source.
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
def calculate_RI_slope_with_error(df):
    # Collect results in a list
    results = []
    names = ['EGly', 'PEG200', 'PEG400', 'PEG600', 'PEG1000', 'PEG1500', 'PEG3000', 'PEG6000', 'PEG12000', 'PEG20000', 'PEG35000']
    RI_avg_dataframe = calculate_average_RI_with_error_of_sample(df)

    # Convert columns to `ufloat`
    for col in RI_avg_dataframe.columns:
        RI_avg_dataframe[col] = RI_avg_dataframe[col].map(uts.convert_to_ufloat)

    # Loop through each name and calculate slope, intercept, and r_squared
    for name in names:
        if str(RI_avg_dataframe[name].iloc[-1]) == 'nan+/-nan':  # Checks if last cell in column is nan+/-nan
            slope, intercept, r_squared = linear_fit_normal(
                RI_avg_dataframe.loc[0:RI_avg_dataframe[name].last_valid_index() - 1, 'wt_%'],
                RI_avg_dataframe.loc[0:RI_avg_dataframe[name].last_valid_index() - 1, name])
        else:
            slope, intercept, r_squared = linear_fit_normal(
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

# model for K fitting based on countrate
def finalfunfit1(x,K,D):
    return alfa*v0*(D-(0.5*(x/n+D+1/K - np.sqrt(((-x/n-D-1/K)**2)-4*x/n*D))))*(1+(gamma/alfa)*K*(x/n-(0.5*(x/n+D+1/K - np.sqrt(((-x/n-D-1/K)**2)-4*x/n*D)))))

# log function for K fitting
def logdef(x):
    b=math.floor(math.log10(abs(x)))
    a=x*10**(-b)
    c=(a,b)
    return c





# Initialize an empty DataFrame to store the results
K_results = pd.DataFrame(columns=[
    'name', 'crowder', 'wt_%', 'K', 'K_err', 'K_err_%',
    'D', 'D_err', 'D_err_%'])

# Lists to store extracted names and values
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
    df_RI = calculate_RI_slope_with_error(pd.read_csv('source_data/RI_of_PEG_solutions.csv'))

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

    # Fitting parameters
    funmodel = Model(finalfunfit1)
    funmodel.set_param_hint('K', value=1e6, min=1e2, max=5e13)
    funmodel.set_param_hint('D', value=1e-8, min=1.1e-9, max=2e-8)
    pars = funmodel.make_params()
    result = funmodel.fit(y, x=x)
    fitK1 = result.params['K'].value
    fitK1err = result.params['K'].stderr
    fitD1 = result.params['D'].value
    fitD1err = result.params['D'].stderr
    fitt1 = ((fitD1 * 1e9,) + (logdef(fitK1)) + (gamma,))

    # Saving data
    if fitK1 and fitK1err and df['D_[um^2/s]'][0]:  # Ensure no invalid or empty values
        data_out = {
            'name': base_names[i],
            'crowder': names[i],
            'wt_%': values[i],
            'K': fitK1,
            'K_err': fitK1err,
            'K_err_%': fitK1err / fitK1 * 100 if fitK1 != 0 else None,
            'D': df['D_[um^2/s]'][0],
            'D_err': df['D_err'][0],
            'D_err_%': df['D_err'][0] / df['D_[um^2/s]'][0] * 100 if df['D_[um^2/s]'][0] != 0 else None
        }

        K_results = pd.concat([K_results, pd.DataFrame([data_out])], ignore_index=True)
        # K_results.to_csv('output.csv', index=False)  # Save to a CSV file
print(K_results)










