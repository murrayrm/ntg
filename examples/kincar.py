# kincar.py - Python interface to NTG for kinematic car
# RMM, 9 Jul 2022
#
# This file illustrates the use of NTG to create a trajectory for a
# kinematic car (bicycle) model via the Python interface.  The code follows
# the basic format of the `kincar.c` file in this same directory, to show
# how to map things over.  This means that it uses the low-level interface
# to NTG.  See `steering.py` for an example using the high-level interface.

import numpy as np
import ctypes
import ntg

verbose = False                 # turn on when debugging
if verbose:
    ntg.print_banner()          # make sure things are working

#
# Vehicle dynamics
#
# These functions define the transformations between the states and inputs
# of the system and the flat outputs (and their derivatives).  NTG carries
# out all of its operations in terms of the flat outputs, so these functions
# are mainly used to transform the inputs and outputs to NTG into
# user-friendly states and inputs.
#

# Function to take states, inputs and return the flat flag
def kincar_flat_forward(x, u, params={}):
    # Get the parameter values
    b = params.get('wheelbase', 3.)
    #! TODO: add dir processing

    # Create a list of arrays to store the flat output and its derivatives
    zflag = [np.zeros(3), np.zeros(3)]

    # Flat output is the x, y position of the rear wheels
    zflag[0][0] = x[0]
    zflag[1][0] = x[1]

    # First derivatives of the flat output
    zflag[0][1] = u[0] * np.cos(x[2])  # dx/dt
    zflag[1][1] = u[0] * np.sin(x[2])  # dy/dt

    # First derivative of the angle
    thdot = (u[0]/b) * np.tan(u[1])

    # Second derivatives of the flat output (setting vdot = 0)
    zflag[0][2] = -u[0] * thdot * np.sin(x[2])
    zflag[1][2] =  u[0] * thdot * np.cos(x[2])

    return zflag

# Function to take the flat flag and return states, inputs
def kincar_flat_reverse(zflag, params={}):
    # Get the parameter values
    b = params.get('wheelbase', 3.)
    dir = params.get('dir', 'f')

    # Create a vector to store the state and inputs
    x = np.zeros(3)
    u = np.zeros(2)

    # Given the flat variables, solve for the state
    x[0] = zflag[0][0]  # x position
    x[1] = zflag[1][0]  # y position
    if dir == 'f':
        x[2] = np.arctan2(zflag[1][1], zflag[0][1])  # tan(theta) = ydot/xdot
    elif dir == 'r':
        # Angle is flipped by 180 degrees (since v < 0)
        x[2] = np.arctan2(-zflag[1][1], -zflag[0][1])
    else:
        raise ValueError("unknown direction:", dir)

    # And next solve for the inputs
    u[0] = zflag[0][1] * np.cos(x[2]) + zflag[1][1] * np.sin(x[2])
    thdot_v = zflag[1][2] * np.cos(x[2]) - zflag[0][2] * np.sin(x[2])
    u[1] = np.arctan2(thdot_v, u[0]**2 / b)

    return x, u

#
# Cost function
#
# The cost function for the system is implemented as a C function in the
# file `kincar.c`.  This function needs to be compiled into a shared object
# (.so) file and then is imported here using the `ctypes` package.
#

kincar = ctypes.cdll.LoadLibrary('kincar.so')
trajectorycostav = [ntg.actvar(0, 2), ntg.actvar(1, 2)]

#
# Parameter definitions
#
# The low-level interface to NTG requires keeping track of all of the
# details of the B-splines, constraints, and cost functions.  This section
# defines all of the parameters that are used to keep track of these
# details.  These are copied over directly from `kincar.c`.
#

# NTG parameters
NOUT = 2                        # number of flat outputs, j
NINTERV = 2                     # number of intervals, lj
MULT = 3                        # regularity of splits, mj
ORDER = 5                       # degree of split polynomial, kj
MAXDERIV = 3                    # highest derivative required + 1
NCOEF = 14                      # total # of coeffs

# number linear constraints
NLIC = 6                        # linear initial constraints
NLTC = 0                        # linear trajectory constraints
NLFC = 6                        # linear final constraints

# number nonlinear constraints
NNLIC = 0                       # nonlinear initial constraints
NNLTC = 0                       # nonlinear trajectory constraints
NNLFC = 0                       # nonlinear final constraints

#
# Initial and final conditions
#
# We how set up the trajectory generation problem by defining the initial
# sates and inputs, final states and inputs, and duration of the trajectory.
#

x0, u0 = np.array([0.0, -2.0, 0.0]), np.array([8.0, 0])
xf, uf = np.array([40.0, 2.0, 0.0]), np.array([8.0, 0])
Tf = 5                          # number of second to complete manuever

# Define the time points to be used in evaluating the trajectory
nbps = 20                       # number of breakpoints
bps = np.linspace(0, Tf, nbps)  # breakpoint values

# Convert to flat flag coordinates
zflag_0 = np.array(kincar_flat_forward(x0, u0)).reshape(-1)
zflag_f = np.array(kincar_flat_forward(xf, uf)).reshape(-1)

#
# Set up constraints and bounds
#
# In the low level interface, we have to specify the upper and lower bounds
# for all of the constraints by creating vectors of concatenated bounds, in
# the right order.  The order of the bounds is:
#
#   * linear initial constraints
#   * linear trajectory constraints
#   * linear final constraints
#   * nonlinear initial constraints
#   * nonlinear trajectory constraints
#   * nonlinear final constraints
#
# For this system we constrain the initial and final conditions using linear
# constraints.
#

state_constraint_matrix = np.zeros((NLIC, MAXDERIV * NOUT))
lowerb = np.zeros(NLIC + NLTC + NLFC + NNLIC + NNLTC + NNLFC)
upperb = np.zeros(NLIC + NLTC + NLFC + NNLIC + NNLTC + NNLFC)
for i in range(NLIC):
    state_constraint_matrix[i, i] = 1
    lowerb[i] = upperb[i] = zflag_0[i]
    lowerb[NLIC + i] = upperb[NLIC + i] = zflag_f[i]

if verbose:
    print("\nState constraint matrix:\n", state_constraint_matrix)
    print("\nUpper and lower bounds:\n", lowerb)

#
# Call NTG
#
# We are now ready to call NTG to solve the trajectory generation problem.
# We do some playing around with the NPSOL options to show different levels
# of detail, depending on the verbose flag (at the top of this file).
#

# Play around with NPSOL output (similar to kincar.c)
ntg.npsol_option("nolist")      # turn off NPSOL listing
if not verbose:
    ntg.npsol_option("print level 0")
ntg.npsol_option("summary file = 0")

coefs = ntg.ntg(
    NOUT, bps, [NINTERV, NINTERV], [ORDER, ORDER],
    [MULT, MULT], [MAXDERIV, MAXDERIV],
    lic=state_constraint_matrix, lfc=state_constraint_matrix,
    lowerb=lowerb, upperb=upperb,
    tcf=kincar.tcf, tcf_av=trajectorycostav
)

# Print the raw coefficients
if verbose:
    print("coefs = ", coefs)

#
# Print (and plot) the trajectory
#
# Finally, we take the coefficients and use them to compute out the
# trajectory for the system.  In principle we should be able to use the
# SciPy BSpline implementation, but this does not seem to match the way that
# NTG implements B-splines, so we use the NTG spline interpoloation function
# instead.
#

# import scipy as sp
# import scipy.interpolate

# Create splines corresponding to the solution
# bsp = []
knots = np.empty((NOUT, NINTERV + 1))
for j in range(NOUT):
    knots[j] = np.linspace(0, Tf, NINTERV + 1)
#     bsp.append(sp.interpolate.BSpline(
#         knots, coefs.reshape(NOUT, -1)[j], ORDER))

# Print the resulting values
zflag = np.zeros((NOUT, MAXDERIV))
timepts = np.linspace(0, Tf, 30)
x = np.empty((3, timepts.size))
u = np.empty((2, timepts.size))
for i, t in enumerate(timepts):
    for j in range(NOUT):
#        zflag[j] = bsp[j](t)
        zflag[j] = ntg.spline_interp(
            t, knots[j], NINTERV, coefs.reshape(NOUT, -1)[j],
            ORDER, MULT, MAXDERIV)
    x[:, i], u[:, i] = kincar_flat_reverse(zflag)
    print(t, x[:, i], u[:, i])

# Function to plot lane change manuever (from CDS 112 notes)
import matplotlib.pyplot as plt
def plot_lanechange(t, y, u, figure=None, yf=None):
    # Plot the xy trajectory
    plt.subplot(3, 1, 1, label='xy')
    plt.plot(y[0], y[1])
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    if yf:
        plt.plot(yf[0], yf[1], 'ro')

    # Plot the inputs as a function of time
    plt.subplot(3, 1, 2, label='v')
    plt.plot(t, u[0])
    plt.xlabel("t [sec]")
    plt.ylabel("velocity [m/s]")

    plt.subplot(3, 1, 3, label='delta')
    plt.plot(t, u[1])
    plt.xlabel("t [sec]")
    plt.ylabel("steering [rad/s]")

    plt.suptitle("Lane change manuever")
    plt.tight_layout()

# Plot the trajectory
plot_lanechange(timepts, x, u)
plt.show()
