import numpy as np
import pandas as pd
import utils as uts
from scipy.stats import linregress


# standard linear fit
def linear_fit_normal(x, y):
    y_value, y_error = uts.get_float_uncertainty(y)

    slope, intercept, r_value, p_value, std_err = linregress(x, y_value)
    r_squared = r_value ** 2
    return slope, intercept, r_squared


# returns a dataframe with average RI values and related errors of PEG solutions
def calculate_average_RI_with_error_of_sample(df):
    # Provide a csv. file as a source.
    df = pd.read_csv('source_data/RI_of_PEG_solutions.csv')
    names = ['EG', 'PEG200', 'PEG400', 'PEG600', 'PEG1000', 'PEG1500', 'PEG3000', 'PEG6000', 'PEG12000', 'PEG20000',
             'PEG35000']
    data = {name: [] for name in names}  # Initialize a dictionary to store the data, where each key is a column name

    for i in range(0, 10):
        for name in names:
            one_sample = df.loc[i,
                         f'{name}_RI_1':f'{name}_RI_4']  # Ensure columns are consecutively ordered in DataFrame
            average_RI_per_sample = np.average(one_sample)
            std_RI_per_sample = np.std(one_sample)
            data[name].append(f'{average_RI_per_sample:.5f}±{std_RI_per_sample:.5f}')

    df_result = pd.DataFrame(data)  # Convert the dictionary to a DataFrame
    df_result.insert(0, 'wt_%', df['PEG200_wt_%'])  # inserting a column with wt%
    return df_result


RI_avg_dataframe = calculate_average_RI_with_error_of_sample(pd.read_csv('source_data/RI_of_PEG_solutions.csv'))

for col in RI_avg_dataframe.columns:
    RI_avg_dataframe[col] = RI_avg_dataframe[col].map(uts.convert_to_ufloat)

slope, intercept, smth = linear_fit_normal(RI_avg_dataframe['wt_%'], RI_avg_dataframe['PEG200'])
