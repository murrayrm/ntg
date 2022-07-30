# keyword_test.py - test keyword functionality
# RMM, 22 Jul 2022

import ntg
import numba
import numpy as np
import scipy as sp
import pytest

# System setup
nout = 2                    # 2 flat outputs
flaglen = [3, 3]            # 2 derivatives in each output
sys = ntg.FlatSystem(nout, flaglen)

# Breakpoints: linearly spaced
Tf = 5
bps = np.linspace(0, Tf, 30)

# Spline definition (default values)
degree = [5, 5]
smooth = [3, 3]
knotpoints = [0, Tf/2, Tf]
basis = ntg.BSplineFamily(knotpoints, degree, smooth, vars=nout)

# Initial and final conditions
z0 = np.array([[0., 8., 0.], [-2., 0., 0.]])
zf = np.array([[40., 8., 0.], [2., 0., 0.]])

# Trajectory costs function
@numba.cfunc(ntg.numba_trajectory_cost_signature)
def tcf_2d_curvature(mode, nstate, i, f, df, zp):
    if mode == 0 or mode == 2:
        f[0] = zp[0][2]**2 + zp[1][2]**2

    if mode == 1 or mode == 2:
        df[0] = 0; df[1] = 0; df[2] = 2 * zp[0][2];
        df[3] = 0; df[4] = 0; df[5] = 2 * zp[1][2];

    return 0

def test_icf_keywords():
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
        sys,  bps, basis,
        initial_constraints=initial_constraints,
        final_constraints=final_constraints,
        trajectory_cost=c_tcf, trajectory_cost_av=tcf_av, verbose=True)
    coefs = systraj.coefs

    # Make sure the optimization succeeded
    assert inform == 0 or inform == 1

    # Recompute using different keywords: trajectory_cost -> cost
    alt_systraj, alt_cost, alt_inform = ntg.solve_flat_ocp(
        sys,  bps, basis,
        initial_constraints=initial_constraints,
        final_constraints=final_constraints,
        cost=c_tcf, cost_av=tcf_av, verbose=True)
    alt_coefs = alt_systraj.coefs
    np.testing.assert_array_equal(alt_coefs, coefs)
    assert alt_cost == cost
    assert alt_inform == inform

    # Pass redundant arguments
    with pytest.raises(TypeError, match='redundant'):
        ntg.solve_flat_ocp(
            sys,  bps, basis,
            initial_constraints=initial_constraints,
            final_constraints=final_constraints,
            cost=c_tcf, trajectory_cost=c_tcf, verbose=True)

# Context manager for not raising an exception
from contextlib import contextmanager
@contextmanager
def does_not_raise():
    yield

@pytest.mark.parametrize("knotpoints, exception", [
    (None, ValueError),
    ([0, Tf], None),
    ([0, Tf/2, Tf], None),
    ([0, Tf, Tf/2], ValueError),
    ])
def test_knotpoints(knotpoints, exception):
    # Set the exception manager
    expectation = does_not_raise() if exception is None else \
        pytest.raises(exception)

    # Compute the basis function
    with expectation:
        basis = ntg.BSplineFamily(
            knotpoints, degree, smooth, vars=nout)


# TODO: figure out why commented out cases crash
@pytest.mark.parametrize("kwargs, exception, match", [
    ({}, TypeError, "invalid flat system"),
    ({'flaglen': 3, 'degree': 5, 'smoothness': 3}, None, None),
    ({'flaglen': 3, 'degree': [4, 5], 'smoothness': 3}, None, None),
    # ({'flaglen': 3, 'verbose': True}, None, None),
    ({'flaglen': 3, 'degree': [1, 5, 1], 'smoothness': 3}, ValueError, None),
    ])
def test_spline_parameters(kwargs, exception, match):
    # Set the exception manager
    expectation = does_not_raise() if exception is None else \
        pytest.raises(exception, match=match)

    # Compute the optimal trajectory
    with expectation:
        # System setup
        args = [nout]
        if 'flaglen' in kwargs:
            args.append(kwargs.pop('flaglen'))
        new_sys = ntg.FlatSystem(*args)

        # Spline setup
        new_basis = ntg.BSplineFamily(
            knotpoints, kwargs.pop('degree', basis.degree),
            kwargs.pop('smoothness', basis.smoothness), vars=nout)
