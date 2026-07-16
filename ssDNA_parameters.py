import numpy as np
import utils as uts
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
from uncertainties import umath
from uncertainties import unumpy as unp

from scipy.integrate import simpson


class ssDNAParameters:

    def __init__(self, bp_num):


        self.bp_num = bp_num
         # physical constants
        self.ssDNA_radius, self.ssDNA_parameters = self.calculate_ssDNA_radius()  # number of baise pairs


    @staticmethod
    def ssDNA_cylinder_params(N):

        b0, Lp = 0.63, 3
        
        L = b0 * N

        # <R_ee^2> dla WLC (dokładny wzór)
        Ree2 = 2 * Lp * L * (1 - (Lp / L) * (1 - np.exp(-L / Lp)))
        Ree = np.sqrt(Ree2)

        # Rg^2 dla WLC (dokładny wzór)
        Rg2 = (Lp * L) / 3 - (Lp ** 2 / 3) * (1 - np.exp(-L / Lp))
        Rg = np.sqrt(Rg2)

        # parametry walca
        H = Ree
        D = 2 * Rg
        V = np.pi * (Rg ** 2) * H

        return {
            "L_contour_nm": L,
            "R_end_to_end_nm": Ree,
            "Rg_nm": Rg,
            "Cylinder_height_nm": H,
            "Cylinder_diameter_nm": D,
            "Cylinder_volume_nm3": V
        }

    def calculate_ssDNA_radius(self):

        ssDNA_parameters = self.ssDNA_cylinder_params(self.bp_num)

        ssDNA_R = (3 * ssDNA_parameters['Cylinder_volume_nm3']/(4 * np.pi)) ** (1/3)  # radius in nm
        
        return ssDNA_R, ssDNA_parameters
        





