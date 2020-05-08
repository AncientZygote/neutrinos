# coding: utf-8

from scipy import constants
# we want to be forced to prepend math. to all Python math functions
import math

import numpy as np

import matplotlib.pyplot as plt

import matplotlib as mpl

# Fermi coupling constant in GeV^-2
G_f = constants.value('Fermi coupling constant')
# speed of light in m/s
c = constants.value('speed of light in vacuum')
hbar = constants.value('Planck constant over 2 pi in eV s')
hbar_GeVs = hbar / 1e9
c_cm_s = c * 1e2
hbarcGeVcm = hbar_GeVs * c_cm_s
root2 = math.sqrt(2)
# Avogadro's number 6.022140857e+23
N_0 = constants.value('Avogadro constant')

# mass split 2-1, eV^2
delta_m_sqr = 7.5e-5
# mass split 2-1, GeV^2
delta_m_sqr_GeV = 7.5e-5 / 1e18
# vacuum mix angle theta 1-2
theta_12 = math.radians(33.48)

sinsqr2theta12 = math.sin(2*theta_12)**2
sin2theta = math.sin(2*theta_12)
cos2theta = math.cos(2*theta_12)

tan2theta = math.tan(2*theta_12)
tanSqrd2theta = tan2theta**2

# number of points/sample steps in the specified interval on x axis x_ax
n_E_steps = 1000

# per PDG 2016 eq (14.58): gen array of MSW N_e densities in e/cm^3
# for each neutrino production energy scanned, GeV in, convert to MeV locally
# equation expects mass split in eV^2
def NRESF (energy_sequence):
    "scan neutrino energy array GeV->MeV, gen array of N_res MSW e/cm^3"
    return ( 6.56e6 * delta_m_sqr * cos2theta * N_0 / (energy_sequence * 1e3) )
NRESFvec = np.vectorize(NRESF)

# per PDG 2016 eq (14.57): create array of cos2theta_m from res density array
# and N_e electron density for the target neutrino species in Sun
def PDGCST2M (MSWresMtrx,N_e):
    "scan N_res MSW e/cm^3 array, gen cos2theta matter angle for each"
    return ( ((1 - (N_e / MSWresMtrx)) / math.sqrt((1 - (N_e / MSWresMtrx))**2 + tanSqrd2theta)) )
PDGCST2Mvec = np.vectorize(PDGCST2M)

# energy scan range, GeV
E_min = 0.150e-3
E_max = 100e-3

# x_ax contains the energy scan steps, GeV
x_ax = np.linspace(E_min,E_max, n_E_steps, endpoint=True)

# scan neutrino energies and calculate the MSW resonance density e/cm^3for each
NeRes_matrix = NRESFvec(x_ax)

# generate cos2theta_m values using PDG 2016 tangent and density ratio equation
# only target density n_e changes on each species
N_e_pp = 4.0249e25 # pp neutrino max flux at 0.0990 radisu, e density in cm^-3
# provide index for this used to produce graph array below
n_e_pp_index = 0
N_e_8B = 5.4670e25 # 8B neutrino max flux at 0.0460 radisu, e density in cm^-3
# provide index for this used to produce graph array below
n_e_8B_index = 1

n_e = np.array( [N_e_pp, N_e_8B])
# want to loop over the number of different neutrino species densities avail
work_index = n_e.shape[0]
# create zeroed array of arrays to accept an array for each cos 2theta_m
# array computed for each density (all over same energy range)
PDGcos2thetamMtrx = np.zeros( (work_index,n_E_steps) )

# loop over the available neutrino species densities creating cos 2theta_m
# for each
for indx in range(work_index):
    PDGcos2thetamMtrx[indx] = PDGCST2Mvec(NeRes_matrix,n_e[indx])

# using PDG 2016 eq (14.84) with PDG matter angle cos2theta
# is assumed SNO detector at Earth, so you have no explicit baseline
def PVEVES13 (pdgcos2thmtrx):
    "return list of survival probabilities per 2013 solar eq as array"
    return(  ( cos2theta * pdgcos2thmtrx * 0.5) + 0.5 )
PVEVES13vec = np.vectorize(PVEVES13)

# graph surv prob with 2013 solar equation, using PDG cos2theta matter angle
# created with tangent and density ratios, passing in the cos2theta_m array
# for each neutrino species.
y_ax_pp = PVEVES13vec(PDGcos2thetamMtrx[n_e_pp_index])
y_ax_8B = PVEVES13vec(PDGcos2thetamMtrx[n_e_8B_index])

# ditch previous runs or end up with multiple plots or pieces of plots
plt.clf()
# tell Matplotlib to use LaTex fonts throughout figures
# (other than label, legend etc which we have to specify); expect wait!
mpl.rcParams.update({'font.size': 12, 'text.usetex': True})
# create a matplotlib figure, accept default aspect ration, dpi
fg1 = plt.figure()
# create an Axes object on it explicitly; graphs are drawn on it
ax1 = fg1.add_subplot(111)

ax1.set(xlabel='E (MeV)', ylabel=r'$P (\nu_e \rightarrow \nu_{e})$')

# want to plot MeV rather than GeV so adjust
# by multiplying the original GeV array by 1e3 element-wise to new array
x_ax_MeV = np.multiply(1e3, x_ax)

# plot first neutrino species data, pp
ax1.plot(x_ax_MeV,y_ax_pp, color='black',label='pp')
# plot next neutrino species data, 8B
ax1.plot(x_ax_MeV,y_ax_8B, color='red',label='8B')

ax1.set_xscale('log')
ax1.legend()

# add annotation giving baseline used; data coordinates for position
ax1.annotate( r'implicit baseline$=1\; \mathrm{au}$',xy=(6.0, 0.50), xycoords='data')
# add annotation giving electron density used; data coordinates for position
ax1.annotate( r'$N_e(pp) = 4.0249 \; \times 10^{25} \; \mathrm{cm}^{-3}$',xy=(6.0, 0.48), xycoords='data')
ax1.annotate( r'$N_e(^8B) = 5.4670 \; \times 10^{25} \; \mathrm{cm}^{-3}$', xy=(6.0, 0.46), xycoords='data')

ax1.set_title(r'Solar electron neutrino survival vs energy')
fg1.tight_layout()
plt.show()
plt.close()

