# keyword_test.py - test keyword functionality
# RMM, 22 Jul 2022

import ntg
import numba
import numpy as np
import pytest

# System setup
nout = 2                    # 2 flat outputs
maxderiv = [3, 3]           # 3 derivatives in each output

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

# Trajecotry costs function
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

def test_icf_keywords():
    # Cost function: curvature
    c_tcf = tcf_2d_curvature.ctypes
    tcf_av = [ntg.actvar(0, 2), ntg.actvar(1, 2)]

    # Linear constraint functions for initial and final condition
    state_constraint_matrix = np.eye(6, 6)
    initial_val = z0.reshape(-1)
    final_val = zf.reshape(-1)

    # Create the bounds matrix
    bounds = np.hstack([initial_val, final_val])

    # Compute the optimal trajectory
    systraj, cost, inform = ntg.ntg(
        nout, bps, ninterv, order, mult, maxderiv,
        lic=state_constraint_matrix, lfc=state_constraint_matrix,
        lowerb=bounds, upperb=bounds,
        tcf=c_tcf, tcf_av=tcf_av)
    coefs = systraj.coefs

    # Make sure the optimization succeeded
    assert inform == 0 or inform == 1

    # Recompute using different keywords: icf -> cost
    alt_systraj, alt_cost, alt_inform = ntg.ntg(
        nout, bps, ninterv, order, mult, maxderiv,
        lic=state_constraint_matrix, lfc=state_constraint_matrix,
        lowerb=bounds, upperb=bounds,
        cost=c_tcf, cost_actvars=None)
    alt_coefs = alt_systraj.coefs
    np.testing.assert_array_equal(alt_coefs, coefs)
    assert alt_cost == cost
    assert alt_inform == inform

    # Recompute using different keywords: icf -> trajectory_cost
    alt_systraj, alt_cost, alt_inform = ntg.ntg(
        nout, bps, ninterv, order, mult, maxderiv,
        lic=state_constraint_matrix, lfc=state_constraint_matrix,
        lowerb=bounds, upperb=bounds,
        trajectory_cost=c_tcf, trajectory_cost_actvars=tcf_av)
    alt_coefs = alt_systraj.coefs
    np.testing.assert_array_equal(alt_coefs, coefs)
    assert alt_cost == cost
    assert alt_inform == inform

    # Pass redundant arguments
    with pytest.raises(TypeError, match='redundant'):
        ntg.ntg(
            nout, bps, ninterv, order, mult, maxderiv,
            lic=state_constraint_matrix, lfc=state_constraint_matrix,
            lowerb=bounds, upperb=bounds,
            trajectory_cost=c_tcf, tcf=c_tcf)

# Context manager for not raising an exception
from contextlib import contextmanager
@contextmanager
def does_not_raise():
    yield

@pytest.mark.parametrize("knotpoints, nintervals, exception", [
    (None, 0, ValueError),
    (None, 1, None),
    ([0, Tf], 1, None),
    (None, 2, None),
    (None, [2, 3], None),
    ([[0, Tf/2, Tf], [0, Tf/2, Tf]], None, None),
    ([[0, Tf/2, Tf], [0, Tf/2, Tf]], 2, None),
    ([0, Tf/2, Tf], None, None),
    ([0, Tf/2, Tf], 2, None),
    ([[0, Tf/3, 2*Tf/3, Tf], [0, Tf/2, Tf]], [3, 2], None),
    ([[0, Tf/3, 2*Tf/3, Tf], [0, Tf/2, Tf]], [2, 2], ValueError),
    ([[0, Tf/2, Tf]], 2, ValueError),
    ([0, Tf/4, Tf/2], 2, ValueError),
    ([Tf/4, Tf/2, Tf], 2, ValueError),
    ([[0, Tf/2, Tf], [Tf/4, Tf/2, Tf]], 2, ValueError),
    ])
def test_nintervals_knotpoints(knotpoints, nintervals, exception):
    # Cost function: curvature
    c_tcf = tcf_2d_curvature.ctypes
    tcf_av = [ntg.actvar(0, 2), ntg.actvar(1, 2)]

    # Linear constraint functions for initial and final condition
    state_constraint_matrix = np.eye(6, 6)
    initial_val = z0.reshape(-1)
    final_val = zf.reshape(-1)

    # Create the bounds matrix
    bounds = np.hstack([initial_val, final_val])

    # Set the exception manager
    expectation = does_not_raise() if exception is None else \
        pytest.raises(exception)

    # Compute the optimal trajectory
    with expectation:
        systraj, cost, inform = ntg.ntg(
            nout, bps, nintervals, order, mult, maxderiv, knotpoints=knotpoints,
            lic=state_constraint_matrix, lfc=state_constraint_matrix,
            lowerb=bounds, upperb=bounds, tcf=c_tcf, tcf_av=tcf_av)

        # Make sure the optimization succeeded
        assert inform == 0 or inform == 1


# TODO: figure out why commented out cases crash
@pytest.mark.parametrize("kwargs, exception, match", [
    ({}, ValueError, r"missing value\(s\) for flaglen"),
    ({'flaglen': 3, 'order': 6, 'multiplicity': 3}, None, None),
    ({'flaglen': 3, 'order': [2, 6], 'multiplicity': 3}, None, None),
    # ({'flaglen': 3, 'verbose': True}, None, None),
    ({'flaglen': 3, 'order': [2, 6, 1], 'multiplicity': 3}, ValueError, None),
    ])
def test_spline_parameters(kwargs, exception, match):
    # Cost function: curvature
    c_tcf = tcf_2d_curvature.ctypes
    tcf_av = [ntg.actvar(0, 2), ntg.actvar(1, 2)]

    # Linear constraint functions for initial and final condition
    state_constraint_matrix = np.eye(6, 6)
    initial_val = z0.reshape(-1)
    final_val = zf.reshape(-1)

    # Create the bounds matrix
    bounds = np.hstack([initial_val, final_val])

    # Set the exception manager
    expectation = does_not_raise() if exception is None else \
        pytest.raises(exception, match=match)

    # Compute the optimal trajectory
    with expectation:
        systraj, cost, inform = ntg.ntg(
            nout, bps, **kwargs,
            lic=state_constraint_matrix, lfc=state_constraint_matrix,
            lowerb=bounds, upperb=bounds, tcf=c_tcf, tcf_av=tcf_av)

        # Make sure the optimization succeeded
        assert inform == 0 or inform == 1
        
