# constraint_test.py - test constrain functionality
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

# Spline definition (default values)
ninterv = [2, 2]
mult = [3, 3]
order = [6, 6]

# Initial and final conditions
z0 = np.array([[0., 8., 0.], [-2., 0., 0.]])
zf = np.array([[40., 8., 0.], [2., 0., 0.]])

# Breakpoints: linearly spaced
Tf = 5
bps = np.linspace(0, Tf, 30)

# Trajectory cost function
from numba import types
@numba.cfunc(
    types.void(
        types.CPointer(types.intc),       # int *mode
        types.CPointer(types.intc),       # int *nstate
        types.CPointer(types.intc),       # int *i
        types.CPointer(types.double),     # double *f
        types.CPointer(types.double),     # double *df
        types.CPointer(                   # double **zp
            types.CPointer(types.double))))
def tcf_2d_curvature(mode, nstate, i, f, df, zp):
    if mode[0] == 0 or mode[0] == 2:
        f[0] = zp[0][2]**2 + zp[1][2]**2

    if mode[0] == 1 or mode[0] == 2:
        df[0] = 0; df[1] = 0; df[2] = 2 * zp[0][2];
        df[3] = 0; df[4] = 0; df[5] = 2 * zp[1][2];


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
    systraj, cost, inform = ntg.ntg(
        nout, bps, ninterv, order, mult, flaglen,
        initial_constraints=initial_constraints,
        final_constraints=final_constraints,
        tcf=c_tcf, tcf_av=tcf_av, verbose=True)
    coefs = systraj.coefs

    # Make sure the optimization succeeded
    assert inform == 0 or inform == 1

    # Get the coefficients into a more useful form
    ncoefs = [ninterv[i] * (order[i] - mult[i]) + mult[i]
              for i in range(nout)]
    coef_list = [coefs[:ncoefs[0]], coefs[ncoefs[0]:]]

    # Make sure we started and stopped in the right place
    knots = [np.linspace(0, Tf, ninterv[i] + 1) for i in range(nout)]

    z0_check = [ntg.spline_interp(0., knots[i], ninterv[i],
                                 coef_list[i], order[i], mult[i], flaglen[i])
                for i in range(nout)]
    np.testing.assert_almost_equal(np.hstack(z0_check), z0.reshape(-1))

    zf_check = [ntg.spline_interp(Tf, knots[i], ninterv[i],
                                 coef_list[i], order[i], mult[i], flaglen[i])
                for i in range(nout)]
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
    systraj, cost, inform = ntg.ntg(
        nout, bps, ninterv, order, mult, flaglen,
        initial_constraints=initial_constraints,
        final_constraints=final_constraints,
        tcf=c_tcf, tcf_av=tcf_av, verbose=True)

    # Make sure the optimization succeeded
    assert inform == 0 or inform == 1

    # Define a nonlinear constraint on the inputs
    input_limit = 0.9 * max(systraj.eval(bps)[1, 2] ** 2)
    @numba.cfunc(ntg.numba_trajectory_constraint_signature)
    def bounded_input(mode, nstate, i, f, df, zp):
        if mode[0] ==0 or mode[0] == 2:
            f[0] = zp[1][2]**2

        if mode[0] == 1 or mode[0] == 2:
            df[0][0] = 0; df[0][1] = 0; df[0][2] = 0
            df[0][3] = 0; df[0][4] = 0; df[0][5] = 2 * zp[1][2]

    trajectory_constraints = sp.optimize.NonlinearConstraint(
        bounded_input.ctypes, 0, input_limit)

    # Re-solve with nonlinear constraint and active variables specified
    avs_systraj, avs_cost, avs_inform = ntg.ntg(
        nout, bps, ninterv, order, mult, flaglen,
        initial_constraints=initial_constraints,
        final_constraints=final_constraints,
        trajectory_constraints=trajectory_constraints,
        trajectory_constraint_actvars=[ntg.actvar(1, 2)],
        tcf=c_tcf, tcf_av=tcf_av, verbose=True)

    # Make sure the constraints were satisfied (with some slop)
    assert inform == 0 or inform == 1
    assert all(avs_systraj.eval(bps)[1, 2] ** 2 <= input_limit * 1.0001)

def test_constraint_errors():
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

    # Set up low-level constraints as well
    bounds = np.hstack([initial_val, final_val])

    with pytest.raises(TypeError):
        # Specifying the same constraint in two ways
        systraj, cost, inform = ntg.ntg(
            nout, bps, ninterv, order, mult, flaglen,
            initial_constraints=initial_constraints,
            lic=state_constraint_matrix, lowerb=bounds, upperb=bounds,
            tcf=c_tcf, tcf_av=tcf_av, verbose=True)

    with pytest.raises(TypeError):
        # Mixing up types of constraints
        systraj, cost, inform = ntg.ntg(
            nout, bps, ninterv, order, mult, flaglen,
            initial_constraints=initial_constraints,
            lfc=state_constraint_matrix, lowerb=bounds, upperb=bounds,
            tcf=c_tcf, tcf_av=tcf_av, verbose=True)

    with pytest.raises(TypeError):
        # Specifiying the same active variable in two ways
        systraj, cost, inform = ntg.ntg(
            nout, bps, ninterv, order, mult, flaglen,
            initial_constraints=initial_constraints,
            initial_constraint_actvars=[ntg.actvar(0,1)],
            nlicf_avs=[ntg.actvar(0,1)],
            lowerb=bounds, upperb=bounds,
            tcf=c_tcf, tcf_av=tcf_av, verbose=True)
