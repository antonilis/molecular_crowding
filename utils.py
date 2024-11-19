import numpy as np
import uncertainties as unc
import math
from scipy.stats import linregress
import pandas as pd


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


# from unc.float extracts value and error
def get_float_uncertainty(column):
    values = np.array([x.nominal_value for x in column], dtype=np.float64)
    uncertainty = np.array([x.std_dev for x in column], dtype=np.float64)

    return values, uncertainty


# standard linear fit
def linear_fit_normal(x, y):
    y_value, y_error = get_float_uncertainty(y)
    slope, intercept, r_value, p_value, std_err = linregress(x, y_value)
    r_squared = r_value ** 2
    return slope, intercept, r_squared


# log function for K fitting
def logdef(x):
    b = math.floor(math.log10(abs(x)))
    a = x * 10 ** (-b)
    c = (a, b)
    return c


# returns all properties of crowders, such as MW, No of monomers, Rg, Rh, etc.
def crowders_properties():
    data = {
        'MW_[g/mol]': [62.07, 200, 400, 600, 1000, 1500, 3000, 6000, 12000, 20000, 35000],
        'No_mono': [1, 4.131, 8.672, 13.212, 22.292, 33.643, 67.695, 135.800, 272.009, 453.620, 794.143],
        'd_coef': [0.00094, 0.0012, 0.0013, 0.00135, 0.0014, 0.00145, 0.0015, 0.00155, 0.0016, 0.00165,
                   0.0017]}  # ρ=ρ0+A⋅C, where C is PEG wt.% and ρ0 = 0.997 g/cm3
    index = ["EGly", "PEG200", "PEG400", "PEG600", "PEG1000", "PEG1500", "PEG3000", "PEG6000", "PEG12000", "PEG20000",
             "PEG35000"]
    value = pd.DataFrame(data, index=index)
    value['Rg_[nm]'] = 0.215 * value['MW_[g/mol]'] ** 0.583 / 10
    value['Rg_err_[nm]'] = 0.215 * value['MW_[g/mol]'] ** 0.031 / 10
    value['Rh_[nm]'] = 0.145 * value['MW_[g/mol]'] ** 0.571 / 10
    value['Rh_err_[nm]'] = 0.145 * value['MW_[g/mol]'] ** 0.009 / 10
    value['V_Rg_[nm]'] = 4 / 3 * np.pi * value['Rg_[nm]'] ** 3
    value['V_Rg_err_[nm]'] = 4 * np.pi * value['Rg_[nm]'] ** 2 * value['Rg_err_[nm]']
    value['η_[dm3/g]'] = 0.004 * value['MW_[g/mol]'] ** 0.8 * 1e-1
    value['C*_[g/dm3]'] = 1 / value['η_[dm3/g]']
    return (value)