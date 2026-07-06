import numpy as np
import utils as uts
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
from uncertainties import umath
from uncertainties import unumpy as unp

from scipy.integrate import simpson


class DepletionCalculations:

    def __init__(self):

         # physical constants
         self.eps = 8.8541878188e-12 * 81
         self.Na = 6.02214076e23
         self.qe = 1.60217663e-19
         self.kb = 1.380649e-23
         self.R = 8.31446261815324
         self.ssDNA_radius, self.ssDNA_parameters = self.calculate_ssDNA_radius(13)  # number of baise pairs

    @staticmethod
    def depletion_layer_sphere(R_particle, Rg_crowder, regime='diluted'):

        delta_0 = 1.07 * Rg_crowder
        x = delta_0 / R_particle

        delta_s = R_particle * (
                (1 + 3 * x + 2.273 * x ** 2 - 0.0975 * x ** 3) ** (1 / 3) - 1
        )
        return delta_s

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

    def calculate_ssDNA_radius(self, bp_number):

        ssDNA_parameters = self.ssDNA_cylinder_params(bp_number)  # we have

        ssDNA_R = (3 * ssDNA_parameters['Cylinder_volume_nm3']/(4 * np.pi)) ** (1/3)  # radius in nm
        
        return ssDNA_R, ssDNA_parameters
        

    def overlap_volume_diluted(self, R1, R2, Rg_crowder):
        delta1 = self.depletion_layer_sphere(R1, Rg_crowder)
        delta2 = self.depletion_layer_sphere(R2, Rg_crowder)

        r = R1 + delta1
        R = R2 + delta2
        d = R1 + R2

        V_overlap = (
                            np.pi * (r + R - d) ** 2 *
                            (d ** 2 - 3 * (r - R) ** 2 + 2 * d * (r + R))
                    ) / (12 * d)

        return V_overlap * 10 ** (-24) * self.Na  # from nm3 to liter/mol


    @staticmethod
    def depletion_layer_sphere_semidiluted(R_particle, Rg_crowder, phi):

        delta_0 = 1.07 * Rg_crowder

        ksi = Rg_crowder * phi ** (-0.77)

        delta = delta_0 * ksi / np.sqrt(delta_0 ** 2 + ksi ** 2)

        x = delta/R_particle

        delta_s = R_particle * (
                (1 + 3 * x + 2.273 * x ** 2 - 0.0975 * x ** 3) ** (1 / 3) - 1
        )
        return delta_s


    def overlap_volume_semidiluted(self, R1, R2, Rg_crowder, phi):
        delta1 = self.depletion_layer_sphere_semidiluted(R1, Rg_crowder, phi)
        delta2 = self.depletion_layer_sphere_semidiluted(R2, Rg_crowder, phi)

        r = R1 + delta1
        R = R2 + delta2
        d = R1 + R2

        V_overlap = (
                            np.pi * (r + R - d) ** 2 *
                            (d ** 2 - 3 * (r - R) ** 2 + 2 * d * (r + R))
                    ) / (12 * d)

        return V_overlap * 10 ** (-24) * self.Na  # from nm3 to liter/mol

    @staticmethod
    def calculate_Pi(phi):
        numerator = 1 + 3.25 * phi + 4.15 * np.square(phi)
        denominator = 1 + 1.48 * phi
        fraction = numerator / denominator
        Pi = 1 + 2.63 * phi * np.power(fraction, 0.309)
        return Pi

    def calculate_depletion_gibbs_energy_semidiluted(self, R1, R2, Rg_crowder, T, c_star, c_max, n_grid=2000):
        if c_max == 0:
            return 0.0

        c_grid, dc = np.linspace(1e-12, c_max, n_grid, retstep=True)
        phi_grid = c_grid / c_star

        Pi = self.calculate_Pi(phi_grid)
        V = self.overlap_volume_semidiluted(R1, R2, Rg_crowder, phi_grid)

        #integral = simpson(Pi * V, x = c_grid)
        integral = np.sum(Pi * V) * dc
        return -self.Na * self.kb * T * integral/1000

    def calculate_depletion_gibbs_energy_diluted(self,R1, R2, Rg_crowder, T, c):

        depl_volume = self.overlap_volume_diluted(R1, R2, Rg_crowder)

        G = - self.R * T * c * depl_volume / 1000

        return G




