"""Q5: order-of-magnitude estimate of the purely dielectric (electrostatic) contribution
to ddG of 13-bp duplex formation when the solvent permittivity drops (EG / PEG 400 at 40 wt%).

Model: nonlinear Poisson-Boltzmann for uniformly charged cylinders in a 1:1 salt reservoir
(infinite length, end effects ignored). Electrostatic free energy per molecule from the
charging integral F = int_0^1 u_s(t*nu) * nu * L dt  [kT], valid for fixed charges.
dsDNA: radius 1.0 nm, 24 phosphates over 13*0.34 nm.
ssDNA (x2): radius a_ss, 12 phosphates over L_ss = 12*b_ss; b_ss is CALIBRATED so that the
model reproduces the measured salt slope d lnK / d ln[Na+] around 35 mM (our Na_FRET.csv).
Only epsilon is changed afterwards (salt fixed at 35 mM): ddG_diel = dF(eps) - dF(78.4).
Assumptions (critical): no ion specificity, no Born/solvation change of Na+, uniform bulk
permittivity up to the DNA surface, no change of strand geometry, activity = concentration.
"""
import numpy as np
from scipy.integrate import solve_bvp
from scipy.special import k0, k1
from scipy.optimize import brentq
import pandas as pd, os

kT_kJ = 8.314462618e-3 * 298.15
lB_w = 0.7155  # nm, Bjerrum length in water, eps=78.4, 25 C (e^2/(4 pi eps0 eps kT))
NA = 6.02214076e23

def lB(eps):
    return lB_w * 78.4 / eps

def kappa(c_M, eps):
    # kappa^2 = 8 pi lB n ; n in nm^-3
    n = c_M * NA * 1e3 / 1e27
    return np.sqrt(8 * np.pi * lB(eps) * n)

def surface_potential(nu, a, c_M, eps):
    """dimensionless surface potential u(a) for line charge nu (e/nm, signed)."""
    if nu == 0:
        return 0.0
    k = kappa(c_M, eps)
    R = a + 20.0 / k
    r = np.geomspace(a, R, 400)
    g = -2 * lB(eps) * nu / a  # du/dr at r=a
    # linear DH guess
    u_lin = -2 * lB(eps) * nu * k0(k * r) / (k * a * k1(k * a))
    def f(r, y):
        return np.vstack([y[1], -y[1] / r + k**2 * np.sinh(y[0])])
    def bc(ya, yb):
        return np.array([ya[1] - g, yb[1] + k * k1(k * R) / k0(k * R) * yb[0]])
    y0 = np.vstack([np.clip(u_lin, -15, 15), np.gradient(np.clip(u_lin, -15, 15), r)])
    sol = solve_bvp(f, bc, r, y0, tol=1e-6, max_nodes=200000)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.sol(a)[0]

def F_el(Q, L, a, c_M, eps, nt=24):
    """electrostatic free energy (kT) of cylinder with total charge Q (signed, e) and length L."""
    nu = Q / L
    t = np.linspace(0, 1, nt + 1)
    us = np.array([surface_potential(ti * nu, a, c_M, eps) for ti in t])
    return np.trapezoid(us, t) * nu * L

def dF_hyb(c_M, eps, b_ss, a_ss=0.7):
    Fds = F_el(-24, 13 * 0.34, 1.0, c_M, eps)
    Fss = F_el(-12, 12 * b_ss, a_ss, c_M, eps)
    return Fds - 2 * Fss  # kT; electrostatic part of dG_hyb / kT

def salt_slope(c_M, eps, b_ss, a_ss=0.7):
    h = 0.15
    return -(dF_hyb(c_M * np.exp(h), eps, b_ss, a_ss) - dF_hyb(c_M * np.exp(-h), eps, b_ss, a_ss)) / (2 * h)

if __name__ == '__main__':
    root = os.path.dirname(os.path.abspath(__file__))
    csv = os.path.join(root, '..', '..', 'source_data', 'IonStrengthDependency', 'Na_FRET.csv')
    d = pd.read_csv(csv)[['C_Na+_[mM]', 'K_Na+_[1/M]']].sort_values('C_Na+_[mM]')
    sel = d[(d['C_Na+_[mM]'] >= 17) & (d['C_Na+_[mM]'] <= 53)]
    S_exp = np.polyfit(np.log(sel['C_Na+_[mM]']), np.log(sel['K_Na+_[1/M]']), 1)[0]
    print(f'Measured d lnK/d ln[Na+] (17.5-52.6 mM, Na_FRET.csv, FRET pair): {S_exp:.2f}')
    c0 = 0.035
    eps_cases = {'water': 78.4, 'EG 40 wt% (est. 67.5)': 67.5, 'EG 40 wt% low (65)': 65.0,
                 'PEG400 40 wt% (est. 59)': 59.0, 'PEG400 40 wt% low (56.7)': 56.7}
    for a_ss in (0.5, 0.7, 0.9):
        try:
            b_cal = brentq(lambda b: salt_slope(c0, 78.4, b, a_ss) - S_exp, 0.3, 2.5, xtol=1e-3)
        except ValueError:
            print(f'a_ss={a_ss}: cannot calibrate in b_ss range'); continue
        print(f'\na_ss = {a_ss} nm -> calibrated b_ss = {b_cal:.3f} nm/nt '
              f'(model slope at 35 mM = {salt_slope(c0, 78.4, b_cal, a_ss):.2f})')
        ref = dF_hyb(c0, 78.4, b_cal, a_ss)
        for name, eps in eps_cases.items():
            ddG = (dF_hyb(c0, eps, b_cal, a_ss) - ref) * kT_kJ
            print(f'  {name:28s} eps={eps:5.1f}: ddG_diel = {ddG:+6.2f} kJ/mol '
                  f'(slope there {salt_slope(c0, eps, b_cal, a_ss):.2f})')
    # sensitivity: what Na+ dilution alone does at eps=78.4 (35 -> 22.5 mM)
    lnr = np.log(35.0 / 22.47)
    print(f'\nFor comparison, empirical salt slope x ln(35/22.47) = {S_exp*lnr:.2f} lnK units '
          f'= {S_exp*lnr*kT_kJ:.1f} kJ/mol (if Na+ really falls to 22.5 mM at 40 wt%).')
