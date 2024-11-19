import numpy as np
import uncertainties as unc
import math
from scipy.stats import linregress

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
