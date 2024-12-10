import numpy as np
import uncertainties as unc
import math
from scipy.stats import linregress
import pandas as pd
from scipy.optimize import curve_fit


# physical constants
eps = 8.8541878188 * 10 ** (-12) * 81  # vacuum * water relative permittivity
Na = 6.02214076 * 10 ** (23)  # avogadro number
qe = 1.60217663 * 10 ** (-19)  # elementary charge of electrion in Culomb
kb = 1.380649 * 10 ** (-23)  # boltzman constant
R = 8.31446261815324  #

# experimental data from our studies
c0 = 35  # concentration of Na+ in mmol which is equal to mol/m^3
zi = 13  # charge of the ssDNA 13bp
a = 0.0754 / 0.1754 * 4 + 0.0246 / 0.1754  # the anions part of the ionic strength HPO42- and H2PO4-
Km = 0.14  # complexation constant of the PEG - Na complexation
Rg_ssDNA = 1.3  # radius of ssDNA, I have chosen this arbitraly
T = 298.15  # temperature for the reaction


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


def weighted_average_with_error(data, errors):
    weights = 1 / errors ** 2
    weighted_avg = np.sum(data * weights) / np.sum(weights)
    weighted_avg_error = np.sqrt(1 / np.sum(weights))
    return weighted_avg, weighted_avg_error


# standard linear fit
def linear_fit_normal(x, y):
    y_value, y_error = get_float_uncertainty(y)
    slope, intercept, r_value, p_value, std_err = linregress(x, y_value)
    r_squared = r_value ** 2
    return slope, intercept, r_squared


def linear_fit_with_fixed_point(x, y, fixed_point):
    b_x, b_y = fixed_point
    a = np.sum((x - b_x) * (y - b_y)) / np.sum((x - b_x) ** 2)
    y_pred = a * (x - b_x) + b_y
    y_mean = np.mean(y)
    sst = np.sum((y - y_mean) ** 2)
    ssr = np.sum((y - y_pred) ** 2)
    r_squared = 1 - (ssr / sst)
    return a, b_x, b_y, r_squared


def exponential_model(x, a, b, c):
    return a * np.exp(b * x) + c


def exponential_fit_with_r_squared(x, y):
    popt, pcov = curve_fit(exponential_model, x, y, p0=(1, 0.1, 1), maxfev=10000)
    a, b, c = popt
    y_fit = exponential_model(x, *popt)
    residuals = y - y_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    return a, b, c, r_squared


def power_model(x, a, b):
    return a * x**b


def power_fit_with_r_squared(x, y):
    popt, pcov = curve_fit(power_model, x, y, p0=(1, 1), maxfev=10000)
    a, b = popt
    y_fit = power_model(x, a, b)
    residuals = y - y_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    return a, b, r_squared


# log function for K fitting
def logdef(x):
    b = math.floor(math.log10(abs(x)))
    a = x * 10 ** (-b)
    c = (a, b)
    return c


# returns all properties of crowders, such as MW, No of monomers, Rg, Rh, etc.
def crowders_properties():
    data = {
        'MW_[g/mol]': [62.07, 200, 400, 600, 1000, 1500, 3000, 6000, 12000, 20000, 35000, 6000, 70000, 400000], # crowder molecular weight
        'No_mono': [1, 4.131, 8.672, 13.212, 22.292, 33.643, 67.695, 135.800, 272.009, 453.620, 794.143, 37.005, 431.726, 1168.566], # no of crowder monomers
        'd_coef': [0.00094, 0.0012, 0.0013, 0.00135, 0.0014, 0.00145, 0.0015, 0.00155, 0.0016, 0.00165, 0.0017, 0.0004, 0.00055, 0.00035]} # coefficient for density of crowder solutions ρ=ρ0+A⋅C, where C is crowder wt.% and ρ0 = 0.997 g/cm3
    index = ["EGly", "PEG200", "PEG400", "PEG600", "PEG1000", "PEG1500", "PEG3000", "PEG6000", "PEG12000", "PEG20000", "PEG35000", "Dextran6000", "Dextran70000", "Ficoll400000"]
    value = pd.DataFrame(data, index=index)
    value['Rg_[nm]'] = 0.215 * value['MW_[g/mol]'] ** 0.583 / 10 # crowder radius of gyration
    value.loc['Dextran6000', 'Rg_[nm]'] = 1.75
    value.loc['Dextran70000', 'Rg_[nm]'] = 7.5
    value.loc['Ficoll400000', 'Rg_[nm]'] = 12
    value['Rg_err_[nm]'] = 0.215 * value['MW_[g/mol]'] ** 0.031 / 10 # crowder error for radius of gyration
    value.loc['Dextran6000', 'Rg_err_[nm]'] = 0.25
    value.loc['Dextran70000', 'Rg_err_[nm]'] = 0.5
    value.loc['Ficoll400000', 'Rg_err_[nm]'] = 1.5
    value['Rh_[nm]'] = 0.145 * value['MW_[g/mol]'] ** 0.571 / 10 # crowder hydrodynamic radius
    value.loc['Dextran6000', 'Rh_[nm]'] = 1.15
    value.loc['Dextran70000', 'Rh_[nm]'] = 5.25
    value.loc['Ficoll400000', 'Rh_[nm]'] = 9
    value['Rh_err_[nm]'] = 0.145 * value['MW_[g/mol]'] ** 0.009 / 10 # crowder error for hydrodynamic radius
    value.loc['Dextran6000', 'Rh_err_[nm]'] = 0.15
    value.loc['Dextran70000', 'Rh_err_[nm]'] = 0.75
    value.loc['Ficoll400000', 'Rh_err_[nm]'] = 1
    value['V_Rg_[nm3]'] = 4/3 * np.pi * value['Rg_[nm]'] ** 3 # volumes of crowder coils if they are a coil
    value['V_Rg_err_[nm3]'] = 4 * np.pi * value['Rg_[nm]'] ** 2 * value['Rg_err_[nm]'] # error for volumes of crowder coils if they are a coil
    value['c*_[g/cm3]'] = value['MW_[g/mol]'] / (Na * value['V_Rg_[nm3]']) * 1e21 # concentration at which polymer starts to overlap calculated using scaling theories for polymer solutions from de Gennes, P. G. "Scaling Concepts in Polymer Physics." Cornell University Press, 1979.
    return(value)







