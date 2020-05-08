# coding: utf-8
from scipy import constants
import math

import numpy as np

from scipy.integrate import simps

m_p_amu = constants.value('proton mass in u')
m_e_amu = constants.value('electron mass in u')
# Avogadro's number 6.022140857e+23
N_0 = constants.value('Avogadro constant')
c = constants.value('speed of light in vacuum') # m/s

# specify projectile and target charges
Z_1 = 1 # an incoming plasma proton
B8_Z = 5 # 8B nuclide has 5 protons
Z_2 = B8_Z # nuclide about to have a collision
# K, temperature at r=.0460, times 10^6 K (i.e., MK)
T_MK = 14.9420

# the reduced mass in atomic unit code
A_1 = m_p_amu  # A_1 incoming nucleon is a proton of mass 1.00727647.. u
# 8/5 nuclide mass per Krane Intro Nuclear Phys table of nuc props
B8_atomic_mass = 8.024606 # atomic mass units u, must subtract 5 electron mass
B8_electron_shell_mass = (B8_Z*m_e_amu)
B8_nuc_mass = B8_atomic_mass - B8_electron_shell_mass
# ignore positive correction for lost electron binding E, O(1e-6)
A_2 = B8_nuc_mass
mu_hat = (A_1 * A_2)/(A_1 + A_2)


# NACRE II equation 3 from
# http://arxiv.org/abs/1310.7099v1
# first take care of the constants out front of integral
# convert our operating temperature from million K to billion K as required
# mu_hat reduced mass u calculated earlier above; T_9 is global used below also
T_9 = T_MK / 1e3
in_front = 3.73e10 * mu_hat**(-1/2) * T_9**(-3/2)

# get Gamow peak; uses temperature in MK; Adelberger 1998 eq 2
# https://arxiv.org/abs/astro-ph/9805121
E0_keV = 1.2204*( Z_1**2 * Z_2**2 * mu_hat * T_MK**2  )**(1/3)

# construct array of integrand function values to call in the integration
# make array of dE steps, original limits around E0; the step is returned->dE
# we will use Adelberger 1998 pg 6 typical E solar fusion 5 kev - 30 keV
NACRE_low_lim_MeV = 0.005 # 5 keV in units of MeV
NACRE_high_lim_MeV = 0.030 # 30 keV in units of MeV
E_steps, dE = np.linspace(NACRE_low_lim_MeV, NACRE_high_lim_MeV, num=100, retstep=True)

# create simulated astrophysical factor S

# let us use simply the geometrical cross section, pi R^2 
# Be7 to 8B astrophys reaction.pdf gives 2.84 fm for this weakly bound system
eff_radius = 2.84

S0_18_fudged  = math.pi * (eff_radius*1e-15)**2 # result in m^2

# now per Krane a measured cross section is multiplied by E of measurement
# to give S factor; we want our S factor to come to max at E = E0, so
# let us say the cross section was measured there
S0_18_fudged = S0_18_fudged * E0_keV # now have keV m^2

# we want the cross section sigma(E) with E in MeV and sigma in b
# so let us convert S0_17 to Mev b here
# times 1e28 converts m^2 to barn; divide by 1e3 if convert keV to Mev
S0 = (S0_18_fudged*1e28) / 1e3 # convert keV m^2 to MeV b

eta_frontend = 0.1575*Z_1*Z_2 # get constant part of eta calculation
def E_sigmaE(Energy):
    """calculate and return E * sigma(E) given E in MeV"""
    S_of_E = S0
    eta_of_E = eta_frontend * (mu_hat/Energy)**(1/2)
    return(S_of_E*math.exp(-2*math.pi*eta_of_E))

# returns E sigma(E) = S exp(-2pi eta) defined by Adelberger 1998 pg 6, eq 7


# will use T_9 global temperature in units 10^9 K
# this function populates the integrand array 
def integrand_step(Energy):
    """accept E in MeV and calculate the integrand function value """
    return(math.exp(-11.605*Energy/T_9)*E_sigmaE(Energy))


# we provide f(E) as precalculated
# array integrand_fcn_values, E values in E_steps
iteration_limit = int(E_steps.shape[0])
integrand_fcn_values = np.zeros(iteration_limit)

for indx_1219 in range(0, iteration_limit):
    integrand_fcn_values[indx_1219] = integrand_step(E_steps[indx_1219])


integration_14GK = simps(  integrand_fcn_values, x=E_steps )

# and finish the calculation by applying the front-end to the integration
N_A_sigmav_8Bp_14MK = integration_14GK * in_front


# divide by Avogadro number to see sigma v (cross section * velocity)
sigmav_8Bp_14MK = N_A_sigmav_8Bp_14MK/N_0

# at solar r = 0.0460 number density of electrons ( same ~protons, neutral)
n_e_r0460 = 5.4670e25 # cm^-3
# don't know 7Be, so use 3He as proxy (same radius, from same SSM data)
n_3He_r0460 = 3.4914e20

# 16.7% of the 3He fuses with alpha's to produce 7Be
Be7_from_3He = n_3He_r0460 * 16.7 * 1e-2

# 0.12% of that 7Be captures a proton to from 8B (briefly)
B8_from_7Be = Be7_from_3He * 0.12 * 1e-2

collision_freq_8Bp = sigmav_8Bp_14MK * n_e_r0460 * B8_from_7Be
collision_interval_8Bp = 1 / collision_freq_8Bp


# parameters used, results
print("Temperature {0:.4f} GK".format(T_9))
print("reduced mass (mu hat): {0:.8f} u".format(mu_hat))
print("8B atomic mass (Krane): {0:.6f} u".format(B8_atomic_mass))
print("8B atom electron mass: {0:.6f} u".format( B8_electron_shell_mass ))
print("8B nuclear mass minus electron mass: {0:.6f} u".format(A_2))

print("electron mass: {0:.6f} u".format(m_e_amu))
print("proton mass: {0:.7f} u".format(m_p_amu))

print("simulated S factor for 8B p collisions: {0:.4f} keV b".     format(S0*1e3))

print("NACRE II E0 (Gamow peak) = {0:.4f} keV".format(E0_keV))
print("calculated N_A <sigma v> 8B->p: {0:.4e} cm^3 mol^-1 s^-1".     format(N_A_sigmav_8Bp_14MK))
print("integrated over E range {0:.3f} to {1:.3f} keV".     format(NACRE_low_lim_MeV*1e3 ,NACRE_high_lim_MeV*1e3))
print("calculated sigma v: {0:.4e} cm^3 s^-1".     format(sigmav_8Bp_14MK))

print("3He density: {0:.4e} cm^-3".format(n_3He_r0460))
print("7Be density: {0:.4e} cm^-3".format(Be7_from_3He))
print("8B resulting density: {0:.4e} cm^-3".format(B8_from_7Be))
print("nucleon projectile density: {0:.4e} cm^-3".format(n_e_r0460))
print("estimated collision frequency p on 8B: {0:.4e} s^-1".      format(collision_freq_8Bp))
print("estimated collision interval p on 8B {0:.4e} s".     format(collision_interval_8Bp))
# estimate neutrino packet length if constrained emission on that interval
neutrino_packet_8Bp = c*collision_interval_8Bp
print("corresponding neutrino packet len c*t then {0:.4f} m".      format(neutrino_packet_8Bp))


# bonus: cross check integration reaction rate using
# equations 6 and 7 per Longland 2010 
# http://arxiv.org/abs/1004.4136v1
tau_Longland_0512 = 4.2487*(Z_1**2 * Z_2**2 * mu_hat * (T_MK*1e-3)**(-1))**(1/3)
# our simulated S0 is in MeV b as required by Longland equation
N_A_sigma_v_Longland = ( 4.339e8 / (Z_1*Z_2) ) * (1/mu_hat) * S0 * math.exp(-tau_Longland_0512) * tau_Longland_0512**2
print("cross check above integrated result:")
print("tau: {0:.4f}".format(tau_Longland_0512))
print("NA sigma v: {0:.4e} cm^3 mol^-1 s^-1".format(N_A_sigma_v_Longland))


# more bonus code not needed for integration earlier
# eta sommerfeld parameter calculator
print("reduced mass: {0:.4f} u".format(mu_hat))
print("target charge: {0:.1f}".format(Z_2))
E_eta = E0_keV/1e3 # they want MeV for eta_s formula
print("energy: {0:.4f} keV".format(E0_keV))
# eta_S for Sommerfeld parameter; equation from NACRE II paper
# http://arxiv.org/abs/1310.7099v1
eta_S = 0.1575*Z_1*Z_2*(mu_hat / E_eta)**(1/2)
print("eta at Gamow peak E0: {0:.4f}".format(eta_S) )
print("eta * 2 * pi = {0:.4f}".format(eta_S*2*math.pi))
# per Adelberger 1998 2pi eta should be gte 1 for WKB approx validity
# Gamow penetration factor then:
print("tunneling probability: {0:.4e}".     format( math.exp(-eta_S*2*math.pi)  ))

