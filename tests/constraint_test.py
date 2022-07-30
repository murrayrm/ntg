# constraint_test.py - test constraint functionality
# RMM, 22 Jul 2022

import ntg
import numba
import numpy as np
import scipy as sp
import scipy.optimize
import pytest

# System setup
nout = 2                    # 2 flat outputs
flaglen = [3, 3]            # 2 derivatives in each output
sys = ntg.FlatSystem(nout, flaglen)

# Time points: linearly spaced
Tf = 5
timepts = np.linspace(0, Tf, 30)

# Spline definition (default values)
degree = [5, 5]
smooth = [3, 3]
breakpoints = [0, Tf/2, Tf]
basis = ntg.BSplineFamily(breakpoints, degree, smooth, vars=nout)

# Initial and final conditions
z0 = np.array([[0., 8., 0.], [-2., 0., 0.]])
zf = np.array([[40., 8., 0.], [2., 0., 0.]])

# Trajectory cost function
from numba import types
@numba.cfunc(ntg.numba_trajectory_cost_signature)
def tcf_2d_curvature(mode, nstate, i, f, df, zp):
    if mode == 0 or mode == 2:
        f[0] = zp[0][2]**2 + zp[1][2]**2

    if mode == 1 or mode == 2:
        df[0] = 0; df[1] = 0; df[2] = 2 * zp[0][2];
        df[3] = 0; df[4] = 0; df[5] = 2 * zp[1][2];

    return 0


def test_linear_constraints():
    # Cost function: curvature
    c_tcf = tcf_2d_curvature.ctypes
    tcf_av = [ntg.actvar(0, 2), ntg.actvar(1, 2)]

    # Linear constraint functions for initial and final condition
    state_constraint_matrix = np.eye(6, 6)
    initial_val = z0.reshape(-1)
    final_val = zf.reshape(-1)

    # Set up initial, trajectory, and final constraints
    initial_constraints = sp.optimize.LinearConstraint(
        state_constraint_matrix, initial_val, initial_val)

    final_constraints = sp.optimize.LinearConstraint(
        state_constraint_matrix, final_val, final_val)

    # Compute the optimal trajectory
    systraj, cost, inform = ntg.solve_flat_ocp(
        sys,  timepts, basis,
        initial_constraints=initial_constraints,
        final_constraints=final_constraints,
        trajectory_cost=c_tcf, trajectory_cost_av=tcf_av, verbose=True)
    coefs = systraj.coefs

    # Make sure the optimization succeeded
    assert inform == 0 or inform == 1

    # Get the coefficients into a more useful form
    ninterv = len(breakpoints) - 1
    ncoefs = [ninterv * (degree[i] + 1 - smooth[i]) + smooth[i]
              for i in range(nout)]
    coef_list = [coefs[:ncoefs[0]], coefs[ncoefs[0]:]]

    # Make sure we started and stopped in the right place
    knots = [np.linspace(0, Tf, ninterv + 1) for i in range(nout)]

    z0_check = [ntg.spline_interp(
        0., knots[i], ninterv, coef_list[i], degree[i] + 1,
        smooth[i], flaglen[i]) for i in range(nout)]
    np.testing.assert_almost_equal(np.hstack(z0_check), z0.reshape(-1))

    zf_check = [ntg.spline_interp(
        Tf, knots[i], ninterv, coef_list[i], degree[i] + 1,
        smooth[i], flaglen[i]) for i in range(nout)]
    np.testing.assert_almost_equal(np.hstack(zf_check), zf.reshape(-1))


def test_nonlinear_constraints():
    # Cost function: curvature
    c_tcf = tcf_2d_curvature.ctypes
    tcf_av = [ntg.actvar(0, 2), ntg.actvar(1, 2)]

    # Linear constraint functions for initial and final condition
    state_constraint_matrix = np.eye(6, 6)
    initial_val = z0.reshape(-1)
    final_val = zf.reshape(-1)

    # Set up initial, trajectory, and final constraints
    initial_constraints = sp.optimize.LinearConstraint(
        state_constraint_matrix, initial_val, initial_val)

    final_constraints = sp.optimize.LinearConstraint(
        state_constraint_matrix, final_val, final_val)

    # Compute the optimal trajectory
    systraj, cost, inform = ntg.solve_flat_ocp(
        sys,  timepts, basis,
        initial_constraints=initial_constraints,
        final_constraints=final_constraints,
        trajectory_cost=c_tcf, trajectory_cost_av=tcf_av, verbose=True)

    # Make sure the optimization succeeded
    assert inform == 0 or inform == 1

    # Define a nonlinear constraint on the inputs
    input_limit = 0.90 * max(systraj.eval(timepts)[1, 2] ** 2)
    @numba.cfunc(ntg.numba_trajectory_constraint_signature)
    def bounded_input(mode, nstate, i, f, df, zp):
        if mode ==0 or mode == 2:
            f[0] = zp[1][2]**2

        if mode == 1 or mode == 2:
            df[0][0] = 0; df[0][1] = 0; df[0][2] = 0
            df[0][3] = 0; df[0][4] = 0; df[0][5] = 2 * zp[1][2]

        return 0

    trajectory_constraints = sp.optimize.NonlinearConstraint(
        bounded_input.ctypes, 0, input_limit)

    # Re-solve with nonlinear constraint and active variables specified
    avs_systraj, avs_cost, avs_inform = ntg.solve_flat_ocp(
        sys, timepts, basis,
        initial_constraints=initial_constraints,
        final_constraints=final_constraints,
        trajectory_constraints=trajectory_constraints,
        trajectory_constraint_av=[ntg.actvar(1, 2)],
        trajectory_cost=c_tcf, trajectory_cost_av=tcf_av, verbose=True)

    # Make sure the constraints were satisfied (with some slop)
    assert inform == 0 or inform == 1
    assert all(avs_systraj.eval(timepts)[1, 2] ** 2 <= input_limit * 1.0001)

