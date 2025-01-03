import numpy as np
import uncertainties as unc
import math
from scipy.stats import linregress
import pandas as pd
from scipy.optimize import curve_fit
from scipy import odr


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
T = 298.15  # temperature for the reaction and all the experimets


# calculatinh radius of the sodium cation

def calculate_hydrodynamic_radius(D):
    viscosity = 0.0008900 # of water in 25 degrees
    rh = (kb * T) / (6 * np.pi * viscosity * D * 10**(-12)) * 10**9
    return rh



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

def linear_fit_with_y_err(x, y):


    y_value, y_error = get_float_uncertainty(y)
    coeff, cov_matrix = np.polyfit(x, y_value , 1, w=1 / y_error, cov=True)

    errors = np.sqrt(np.diag(cov_matrix))

    ufloat_coefficients = [unc.ufloat(coeff[i], errors[i]) for i in range(len(coeff))]

    slope, intercept = ufloat_coefficients[0], ufloat_coefficients[1]

    return slope, intercept



def linear_fit_with_fixed_point(x, y, fixed_point):
    b_x, b_y = fixed_point
    a = np.sum((x - b_x) * (y - b_y)) / np.sum((x - b_x) ** 2)
    y_pred = a * (x - b_x) + b_y
    y_mean = np.mean(y)
    sst = np.sum((y - y_mean) ** 2)
    ssr = np.sum((y - y_pred) ** 2)
    r_squared = 1 - (ssr / sst)
    return a, b_x, b_y, r_squared


def combine_value_and_error(df, value_col, error_col):
    """
    Combine two columns (value and error) into one column with ufloat objects,
    and remove the original value and error columns.

    Parameters:
    - df: DataFrame containing the value and error columns.
    - value_col: Name of the column with values.
    - error_col: Name of the column with errors.
    - new_col_name: Name of the new column to store the ufloat values.

    Returns:
    - The DataFrame with the combined ufloat column and original value/error columns removed.
    """

    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    df[error_col] = pd.to_numeric(df[error_col], errors='coerce')

    # Create the new column with ufloat objects
    df[value_col] = df.apply(lambda row: unc.ufloat(row[value_col], row[error_col]), axis=1)

    # Drop the original value and error columns
    df.drop(columns=[error_col], inplace=True)

    return df


def gibbs_free_energy_from_K(K_0, K):
    return -R * T * np.log(K / K_0)


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
    return (value)

# mesh sizes for PEGs if mesh
def mesh_sizes():
    properties = crowders_properties().loc['EGly':'PEG35000', ]

    # crowders
    crowders = [
        "EGly", "PEG200", "PEG400", "PEG600", "PEG1000", "PEG1500",
        "PEG3000", "PEG6000", "PEG12000", "PEG20000", "PEG35000"]

    # weight percents of crowders
    wt_percents = [2.5, 5 , 7.5, 10, 12.5, 15, 20, 25, 30, 40]

    row_names = [
        "2.5_wt%", "5_wt%", "7.5_wt%", "10_wt%", "12.5_wt%",
        "15_wt%", "20_wt%", "25_wt%", "30_wt%", "40_wt%"]

    # mesh sizes [nm] calculated from equation ξ = Rg * (c/c*)**-β where β is 0.75 from Ulrich R.D. (1978) P. J. Flory. In: Ulrich R.D. (eds) Macromolecular Science. Contemporary Topics in Polymer Science, vol 1. Springer, Boston, MA. https://doi.org/10.1007/978-1-4684-2853-7_5
    data = [[properties.loc[crowder, 'Rg_[nm]'] * ((wt_percent / (100 / (0.997 + wt_percent * properties.loc[crowder, 'd_coef']))) / properties.loc[crowder, 'c*_[g/cm3]']) ** -0.75 for crowder in crowders] for wt_percent in wt_percents]
    df = pd.DataFrame(data, index=row_names, columns=crowders)

    # assigns 'Rg' to points where polymers are still coils (in this regime polymer is described by radius of gyration)
    df = df.astype(object)
    df.loc[:,'EGly'] = 'Rg'
    df.loc[:,'PEG200'] = 'Rg'
    df.loc[:,'PEG400'] = 'Rg'
    df.loc[:'30_wt%','PEG600'] = 'Rg'
    df.loc[:'20_wt%','PEG1000'] = 'Rg'
    df.loc[:'15_wt%','PEG1500'] = 'Rg'
    df.loc[:'7.5_wt%','PEG3000'] = 'Rg'
    df.loc[:'5_wt%','PEG6000'] = 'Rg'
    df.loc[:'2.5_wt%','PEG12000'] = 'Rg'
    df.loc[:'2.5_wt%','PEG20000'] = 'Rg'
    return df



df = crowders_properties()

















