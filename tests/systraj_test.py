# systraj_test.py - test system trajectory
# RMM, 22 Jul 2022

import ntg
import numba
import numpy as np
import scipy as sp
import scipy.optimize
import pytest

def test_uniform_flag():
    # System setup
    nout = 2                    # 2 flat outputs
    flaglen = [3, 3]            # 2 derivatives in each output

    # Spline definition (default values)
    ninterv = [2, 3]
    mult = [3, 4]
    order = [6, 7]

    # Initial and final conditions
    z0 = np.array([[0., 8., 0.], [-2., 0., 0.]])
    zf = np.array([[40., 8., 0.], [2., 0., 0.]])

    # Breakpoints: linearly spaced
    Tf = 5
    bps = np.linspace(0, Tf, 30)

    # Cost function: curvature
    @numba.cfunc(ntg.numba_trajectory_cost_signature)
    def tcf_2d_curvature(mode, nstate, i, f, df, zp):
        if mode[0] == 0 or mode[0] == 2:
            f[0] = zp[0][2]**2 + zp[1][2]**2

        if mode[0] == 1 or mode[0] == 2:
            df[0] = 0; df[1] = 0; df[2] = 2 * zp[0][2];
            df[3] = 0; df[4] = 0; df[5] = 2 * zp[1][2];
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
        tcf=c_tcf, tcf_av=tcf_av)

    # Make sure the optimization succeeded
    assert inform == 0 or inform == 1

    # Create the system trajectory manually
    coefs = systraj.coefs
    coef_list = []
    offset = 0
    for i in range(nout):
        ncoefs = ninterv[i] * (order[i] - mult[i]) + mult[i]
        coef_list.append(coefs[offset:offset + ncoefs])
        offset += ncoefs

    # Get the full trajectory at a set of time points
    knots = [np.linspace(0, Tf, ninterv[i] + 1) for i in range(nout)]
    timepts = np.linspace(0, Tf, 100)
    ztraj = np.empty((nout, max(flaglen), timepts.size))
    ztraj.fill(np.nan)          # keep track of unused entries
    for i in range(nout):
        ztraj[i, 0:flaglen[i]] = np.array([
            ntg.spline_interp(t, knots[i], ninterv[i], coef_list[i],
                              order[i], mult[i], flaglen[i])
            for t in timepts]).transpose()

    # Compare to system trajectory
    np.testing.assert_equal(systraj.eval(timepts), ztraj)

def test_nonuniform_flag():
    # System setup
    nout = 2                    # 2 flat outputs
    flaglen = [2, 3]            # different numbers of derivatives

    # Spline definition (default values)
    ninterv = [2, 2]
    mult = [3, 3]
    order = [6, 6]

    # Initial and final conditions
    z0 = np.array([0., 8., -2., 0., 0.])
    zf = np.array([40., 8., 2., 0., 0.])

    # Breakpoints: linearly spaced
    Tf = 5
    bps = np.linspace(0, Tf, 30)

    # Cost function: inputs at end of each flag
    @numba.cfunc(ntg.numba_trajectory_cost_signature)
    def tcf(mode, nstate, i, f, df, zp):
        if mode[0] == 0 or mode[0] == 2:
            f[0] = zp[0][1]**2 + zp[1][2]**2

        if mode[0] == 1 or mode[0] == 2:
            df[0] = 0; df[1] = 2 * zp[0][1];
            df[2] = 0; df[3] = 0; df[4] = 2 * zp[1][2];
    c_tcf = tcf.ctypes
    tcf_av = [ntg.actvar(0, 1), ntg.actvar(1, 2)]

    # Set up initial, trajectory, and final constraints
    initial_constraints = sp.optimize.LinearConstraint( np.eye(5), z0, z0)
    final_constraints = sp.optimize.LinearConstraint(np.eye(5), zf, zf)

    # Compute the optimal trajectory
    systraj, cost, inform = ntg.ntg(
        nout, bps, ninterv, order, mult, flaglen,
        initial_constraints=initial_constraints,
        final_constraints=final_constraints,
        tcf=c_tcf, tcf_av=tcf_av, verbose=True)

    # Make sure the optimization succeeded
    assert inform == 0 or inform == 1

    # Create the system trajectory manually
    coefs = systraj.coefs
    coef_list = []
    offset = 0
    for i in range(nout):
        ncoefs = ninterv[i] * (order[i] - mult[i]) + mult[i]
        coef_list.append(coefs[offset:offset + ncoefs])
        offset += ncoefs

    # Get the full trajectory at a set of time points
    knots = [np.linspace(0, Tf, ninterv[i] + 1) for i in range(nout)]
    timepts = np.linspace(0, Tf, 100)
    ztraj = np.empty((nout, max(flaglen), timepts.size))
    ztraj.fill(np.nan)          # keep track of unused entries
    for i in range(nout):
        ztraj[i, 0:flaglen[i]] = np.array([
            ntg.spline_interp(t, knots[i], ninterv[i], coef_list[i],
                              order[i], mult[i], flaglen[i])
            for t in timepts]).transpose()

    # Compare to system trajectory
    np.testing.assert_equal(systraj.eval(timepts), ztraj)
