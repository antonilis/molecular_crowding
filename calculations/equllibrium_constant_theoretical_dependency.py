import numpy as np
import utils as uts
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
from uncertainties import umath
from uncertainties import unumpy as unp


class TheoreticalModel:
    # physical constants
    eps = 8.8541878188e-12 * 81
    Na = 6.02214076e23
    qe = 1.60217663e-19
    kb = 1.380649e-23
    R = 8.31446261815324

    @staticmethod
    def depletion_layer_sphere(R_particle, Rg_crowder, regime='diluted'):

        delta_0 = 1.07 * Rg_crowder
        x = delta_0 / R_particle

        delta_s = R_particle * (
                (1 + 3 * x + 2.273 * x ** 2 - 0.0975 * x ** 3) ** (1 / 3) - 1
        )
        return delta_s



    def overlap_volume(self, R1, R2, Rg_crowder):
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

    def compute_a1_a2_theory(self, df):
        df = df.copy()

        a = (df['I [mM]'] - 0.5 * df['Na conc. [mM]']) * 2 / df['Na conc. [mM]']

        alpha = np.sqrt((self.eps * self.kb * df['T [K]']) /
                        (2 * self.Na * self.qe ** 2))

        Zi = df['charge 1'] * self.qe / (4 * np.pi * self.eps)

        lambd = (
                        self.Na * df['charge 2'] * self.qe * Zi * np.sqrt(df['Na conc. [mM]'])
                # mM is equivalent ot mol/m^3 in SI units
                ) / (
                        3 * np.sqrt(6) * (3 * a + 2) ** (3 / 2) * np.e * alpha
                )

        depletion_volume = self.overlap_volume(df['Rg 1 [nm]'], df['Rg 2 [nm]'], df['Rg_[nm]'])

        df['theory_a2'] = (
                -(4 * a + 2) * lambd /
                (self.R * df['T [K]']) *
                df['n Beta [1/M]'] ** 2
        )

        df['theory_a1'] = (
                2 * (5 * a + 3) * lambd /
                (self.R * df['T [K]']) *
                df['n Beta [1/M]']
                + depletion_volume
        )

        return df

    def compute_ddG_elec(self, df):
        df = df.copy()

        # parametr 'a'
        a = (df['I [mM]'] - 0.5 * df['Na conc. [mM]']) / (0.5 * df['Na conc. [mM]'])


        alpha = unp.sqrt(self.eps * self.kb * df['T [K]'] / (2 * self.Na * self.qe ** 2))


        Zi = df['charge 2'] * self.qe / (4 * np.pi * self.eps)

        # prefaktor z r0 = 0.2 nm
        r0 = 0.2*1e-9
        prefactor = self.Na * df['charge 1'] * self.qe * Zi / r0

        exponent_cPEG = - (r0 / alpha) * unp.sqrt(df['Na conc. [mM]'] / 2) * unp.sqrt(
            1 / (df['n Beta [1/M]'] * df['concentration [M]'] + 1) + a)
        G_cPEG = prefactor * unp.exp(exponent_cPEG)

        # argument wykładnika przy cPEG=0
        exponent_0 = - (r0 / alpha) * unp.sqrt(df['Na conc. [mM]'] / 2) * unp.sqrt(1 + a)
        G_0 = prefactor * unp.exp(exponent_0)

        # ΔΔG = różnica względem 0
        ddG = G_cPEG - G_0

        r_D = alpha/(unp.sqrt(df['Na conc. [mM]'] / 2) * unp.sqrt(
            1 / (df['n Beta [1/M]'] * df['concentration [M]'] + 1) + a))


        return ddG, r_D

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

    def compute_depletion_gibbs_energy(self, df):
        results = []

        for _, row in df.iterrows():
            R1 = row['Rg 1 [nm]']
            R2 = row['Rg 2 [nm]']
            Rg_crowder = row['Rg_[nm]']
            T = row['T [K]']
            c_star = row['c*_[M]']
            c_max = row['concentration [M]']

            if c_max == 0:
                deltaG = 0

            else:
                c_grid = np.linspace(1e-12, c_max, 2000)
                phi_grid = c_grid / c_star

                Pi = self.calculate_Pi(phi_grid)

                V = self.overlap_volume_semidiluted(R1, R2, Rg_crowder, phi_grid)

                integrand = Pi * V

                integral = cumulative_trapezoid(integrand, c_grid, initial=0)

                deltaG = - self.Na * self.kb * T * integral[-1] #

            results.append(deltaG)

        return results

    def calculate_total_Gibbs_energy(self, df):


        df['Delta_G_exp_[kJ/mol]'] = - self.R * df['T [K]'] * df['K_uf [M]'].apply(lambda x: umath.log(x))/1000

        df['Depl_Volume_[L/mol]'] = self.overlap_volume(2.8422678979129627,2.8422678979129627 ,df['Rg_[nm]'])

        dat = df[df['crowder wt. [%]'] == 0]
        df = df.merge(
            dat[['sample', 'crowder', 'source', 'Delta_G_exp_[kJ/mol]']].rename(
                columns={'Delta_G_exp_[kJ/mol]': 'Delta_G0_exp_[kJ/mol]'}),
            on=['sample', 'crowder', 'source'],
            how='left'
        )


        return df



