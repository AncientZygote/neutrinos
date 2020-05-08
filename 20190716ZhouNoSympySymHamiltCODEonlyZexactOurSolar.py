# coding: utf-8

from scipy import constants
# we want to be forced to prepend math. to all Python math functions
import math
import cmath

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

# values from 2016 global fit
# eV^2 converted to GeV^2 since energy in GeV below
delta_m12_sqr = 7.5e-5 * 1e-18
delta_m31_sqr = 2.457e-3 * 1e-18
delta_m32_sqr = delta_m31_sqr - delta_m12_sqr

# want to use radian version of theta12 PMNS mix angle
theta_12 = math.radians(33.48)
theta_23 = math.radians(42.3)
theta_13 = math.radians(8.5)

# use Li, Zhang, Zhou nomenclature for variable dense equations
# c12, for example, is cos theta_12, etc.

c12 = math.cos(theta_12)
c12_squared = c12**2
c13 = math.cos(theta_13)
s13 = math.sin(theta_13)
s12 = math.sin(theta_12)
s12_squared = s12**2
s13_squared = s13**2
c13_squared = c13**2

s23_squared = math.sin(theta_23)**2 # not used, but wanted to see it

Delta_32 = delta_m32_sqr
Delta_31 = delta_m31_sqr
Delta_21 = delta_m12_sqr

# first row of PMNS matrix is used, squared
Ue1_squared = c12_squared*c13_squared
Ue2_squared = s12_squared*c13_squared
Ue3_squared = s13_squared

print("sin^2 theta_12 {0:.4f}".format(s12_squared))
print("sin^2 theta_13 {0:.4f}".format(s13_squared))
print("sin^2 theta_23 {0:.4f}".format(s23_squared))

n_e = 5.4670e25 #8B max emit point density cm^-3

# energy scan range in GeV; number of steps
E_min = 0.350e-3 # 350 keV in units GeV
#E_max = 10.5e-3  # 10.5 MeV in units Gev
E_max = 100e-3  # 100 MeV in units Gev

n_E_steps = 10000

# x_ax contains the energy scan steps, GeV
x_ax = np.linspace(E_min,E_max, n_E_steps, endpoint=True)

# 2*sqrt2 * G_f * n_e, will be multiplied by E on each step below
# returns GeV
# n_e in cm^-3 , hbarc in GeV cm
Matter_potential = 2.0 * root2 * G_f * n_e * hbarcGeVcm**3

# make the delta V sequence used to get sin2theta_m and delta m_m
# scan neutrino energies and calculate delta V for each
# recall that delta V = 2*sqrt2 * G_f * E * n_e
# returns GeV^2, expecting GeV in for potential and energy steps
# Li, Zhang, Zhou variable name in equations below
A = np.multiply(x_ax, Matter_potential)

# select your scheme here by assigning eta the desired scheme parameter

eta = 1.0 # normal scheme
eta2 = 2*eta

# Delta_ast is a constant, a scalar; it varies depending on scheme eta
Delta_ast = eta * Delta_31 + (1-eta) * Delta_32

# alpha_c is a constant, a scalar
alpha_ast = Delta_21 / Delta_ast

# *** A_hat_ast is a VECTOR (array) consisting of 2EV(steps) / Delta_ast ***
A_hat_ast = np.divide(A,Delta_ast)

# b is a VECTOR (array) since A_hat_ast is
# get the constant part done prior to the array operation
b_constant_part = -1 - alpha_ast*(eta2 - 1)
# subtract array A_hat_ast from constant part
b = b_constant_part - A_hat_ast

# x is a VECTOR (array) since its input A_hat_ast is of that ilk
# get some of the constant parts done prior to array operation
x_constant_part = Delta_ast * (  1 + ( 2- eta) * alpha_ast)
# so we have Delta_ast*A_hat_ast[], then add to the constant part
x =  np.add(np.multiply(A_hat_ast,Delta_ast),x_constant_part)

# y will be a VECTOR (array) since it consumes A_hat_ast
# some constant parts
Delta_ast_squared = Delta_ast**2
one_minus_Ue3_sqrd = 1 - math.fabs(Ue3_squared)
eta_min_Ue3_fract = eta - math.fabs(Ue1_squared)/one_minus_Ue3_sqrd
alpha_sqr_one_min_eta = alpha_ast**2 * (1 - eta)

y = Delta_ast_squared * ( np.multiply(A_hat_ast,one_minus_Ue3_sqrd) + \
        alpha_ast * ( np.add(A_hat_ast,1) - \
        np.multiply(A_hat_ast,one_minus_Ue3_sqrd) * eta_min_Ue3_fract )\
                          + alpha_sqr_one_min_eta )


# z will be VECTOR (array) since contains matrix A_hat_ast
# work out from the A_hat_ast array part
z_Ahat_factor = 27 * Delta_ast**3 * alpha_ast
z_Ahat_component_outer = np.multiply(A_hat_ast, z_Ahat_factor)
z_Ahat_first_expression = (1 + alpha_ast - eta*alpha_ast)*\
                          math.fabs(Ue1_squared)
z_Ahat_consolidate_1 =  \
    np.multiply(z_Ahat_component_outer,z_Ahat_first_expression)
# prepare a couple of bracket terms to add to the consolidated A_hat_ast
nine_xy = np.multiply(np.multiply(x,y),9)
two_x_cubed = np.multiply(np.power(x,3),2)
# sum the numerator stuff so far, subtracting last to avoid negatives
z_numerator_part_1 = two_x_cubed + z_Ahat_consolidate_1 - nine_xy
# multiply by the Delta_ast factor
z_numerator = np.multiply(z_numerator_part_1, Delta_ast)
# prepare denominator, evil square root first
z_denom_radicand = np.power( np.square(x) - np.multiply(y,3), 3 )
z_denom_sqrrt = np.sqrt( z_denom_radicand )
z_denominator = np.multiply(z_denom_sqrrt, (2*np.absolute(Delta_ast)) )
z_fract_arg_of_accos = np.divide( z_numerator, z_denominator )
# z_fract_arg_of_accos should be clean domain [-1.0,+1.0] now
z_arccos_part = np.arccos(z_fract_arg_of_accos)
# apply the 1/3 factor to that
z_braced_part = np.multiply(z_arccos_part, (1.0/3.0))
# take the cosine of that result and ship it out
z = np.cos(z_braced_part)

# do the part common to all three eigenvalues
L_common_b_by_3 = np.divide(b,3) # a matrix
L_common_one_by_3Delta = -1.0 / (3.0*Delta_ast) # a scalar NEGATIVE

L_common_xy_sqrt = np.sqrt( np.square(x) - np.multiply(y,3) )

# and a part common to first two eigenvalues
one_minus_zsqr_part = np.subtract(1.0, np.square(z))
L12_common_root = np.sqrt( np.multiply(3.0, one_minus_zsqr_part) )

# assemble eigenvalue 1, the last shall be first
lam_1 = np.add(z,L12_common_root) * L_common_xy_sqrt *\
        L_common_one_by_3Delta - L_common_b_by_3


# assemble eigenvalue 2, the rightmost first, polarity here only difference
lam_2 = np.subtract(z,L12_common_root) *\
        L_common_xy_sqrt * L_common_one_by_3Delta - L_common_b_by_3


# assemble eigenvalue 3, somewhat different than first two
two_by_3delta = 2.0 / (3*Delta_ast)
lam_3 = np.multiply(z,L_common_xy_sqrt) * two_by_3delta - L_common_b_by_3


# ditch previous runs or end up with multiple plots or pieces of plots
plt.clf()

fg1, axes = plt.subplots(1, 1) # create a figure

# scale x up to MeV from GeV
x_ax_MeV = np.multiply(x_ax,1e3)

# plot eigenvalue 1 first
axes.plot(x_ax_MeV, lam_1, color='r')

# eigenvalue 2
axes.plot(x_ax_MeV, lam_2, color='b', linestyle='dashed')

# eigenvalue 3 (plotting in invisible white to tweak plot of other 2)
axes.plot(x_ax_MeV, lam_3, color='w', linestyle='dotted')

# want to see the tick markers (not labels) on left and right
axes.yaxis.set_ticks_position('both')

axes.legend([r'$\lambda_1$', r'$\lambda_2$'])

axes.set_yscale('log')
axes.set_xscale('log')

axes.set_title(r'$\lambda_{1,2}$ vs E')
axes.set_xlabel('E (MeV)')

plt.show()
plt.close()

