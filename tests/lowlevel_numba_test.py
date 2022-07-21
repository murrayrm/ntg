# lowlevel_numba_test.py - test low-level NTG interface using numbda
# RMM, 16 Jul 2022

import numba
import numpy as np
import itertools
import ntg
import pytest

# Numba version
from numba import types, carray
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

@pytest.mark.parametrize("ninterv", [[2, 2], [2, 3]]) # [1, 1]
@pytest.mark.parametrize("mult", [[2, 2], [3, 4]])
@pytest.mark.parametrize("order", [[5, 5], [6, 5]])
@pytest.mark.parametrize(
    "zf_0, zf_f, Tf", [
        ([0, 0, 0, 0, 0, 0], [1, 0, 0, 1, 0, 0], 1),
        ([1, 2, 3, 4, 5, 6], [5, 4, 3, 2, 1, 0], 3.1415)
    ])
def test_2d_curvature_p2p(zf_0, zf_f, Tf, ninterv, mult, order):
    # System setup
    nout = 2                    # 2 flat outputs
    maxderiv = [3, 3]           # 3 derivatives in each output

    # Breakpoints: linearly spaced
    bps = np.linspace(0, Tf, 30)

    # Cost function: curvature
    c_tcf = tcf_2d_curvature.ctypes
    tcf_av = [ntg.actvar(0, 2), ntg.actvar(1, 2)]

    # Linear constraint functions for initial and final condition
    state_constraint_matrix = np.eye(6, 6)
    initial_val = np.array(zf_0)
    final_val = np.array(zf_f)

    # Create the bounds matrix
    bounds = np.hstack([initial_val, final_val])

    # Compute the optimal trajectory
    coefs, cost, inform = ntg.ntg(
        nout, bps, ninterv, order, mult, maxderiv,
        lic=state_constraint_matrix, lfc=state_constraint_matrix,
        lowerb=bounds, upperb=bounds,
        tcf=c_tcf, tcf_av=tcf_av)

    # Make sure the optimization succeedd
    assert inform == 0 or inform == 1

    # Get the coefficients into a more useful form
    ncoefs = [ninterv[i] * (order[i] - mult[i]) + mult[i]
              for i in range(nout)]
    coef_list = [coefs[:ncoefs[0]], coefs[ncoefs[0]:]]

    # Make sure we started and stopped in the right place
    knots = [np.linspace(0, Tf, ninterv[i] + 1) for i in range(nout)]

    z0_check = [ntg.spline_interp(0., knots[i], ninterv[i],
                                 coef_list[i], order[i], mult[i], maxderiv[i])
                for i in range(nout)]
    np.testing.assert_almost_equal(np.hstack(z0_check), zf_0)

    zf_check = [ntg.spline_interp(Tf, knots[i], ninterv[i],
                                 coef_list[i], order[i], mult[i], maxderiv[i])
                for i in range(nout)]
    np.testing.assert_almost_equal(np.hstack(zf_check), zf_f)

#
# Initial/final cost testing
#

NOUT = 2                        # number of outputs
MAXDERIV = 3                    # maximum number of derivatives
ifc_weight = 100                # initial/final cost weight

z0 = np.array([[0., 0., 0.], [-2., 0., 0.]])
zf = np.array([[40., 0., 0.], [2., 0., 0.]])

@numba.cfunc(
    types.void(
        types.CPointer(types.intc),       # int *mode
        types.CPointer(types.intc),       # int *nstate
        types.CPointer(types.double),     # double *f
        types.CPointer(types.double),     # double *df
        types.CPointer(                   # double **zp
            types.CPointer(types.double))))
def nl_2d_initial_cost(mode, nstate, f, df, zp):
    if mode[0] == 0 or mode[0] == 2:
        # compute cost function: square distance from initial value
        f[0] = 0
        for i in range(NOUT):
            for j in range(MAXDERIV):
                f[0] += (zp[i][j] - z0[i][j])**2 * ifc_weight

    if mode[0] == 1 or mode[0] == 2:
        # compute gradient of cost function (index = active variables)
        for i in range(NOUT):
            for j in range(MAXDERIV):
                df[i * MAXDERIV + j] = 2 * ifc_weight * \
                         (zp[i][j] - z0[i][j]);

@numba.cfunc(
    types.void(
        types.CPointer(types.intc),       # int *mode
        types.CPointer(types.intc),       # int *nstate
        types.CPointer(types.double),     # double *f
        types.CPointer(types.double),     # double *df
        types.CPointer(                   # double **zp
            types.CPointer(types.double))))
def nl_2d_final_cost(mode, nstate, f, df, zp):
    if mode[0] == 0 or mode[0] == 2:
        # compute cost function: square distance from initial value
        f[0] = 0
        for i in range(NOUT):
            for j in range(MAXDERIV):
                f[0] += (zp[i][j] - zf[i][j])**2 * ifc_weight

    if mode[0] == 1 or mode[0] == 2:
        # compute gradient of cost function (index = active variables)
        for i in range(NOUT):
            for j in range(MAXDERIV):
                df[i * MAXDERIV + j] = 2 * ifc_weight * \
                         (zp[i][j] - zf[i][j]);

def test_2d_curvature_ifc_ltc():
    """This unit test replaces initial and final constraints with cost
    functions, which should allow for reduced cost, and adds constraints on
    the inputs, which should increase the cost.
    """

    # System definition
    nout = 2                    # 2 flat outputs
    maxderiv = [3, 3]           # 3 derivatives in each output

    # Spline definition
    ninterv = [2, 2]
    mult = [3, 3]
    order = [6, 6]

    # Problem definition
    z0 = [0,  0, 0, -2, 0, 0]
    zf = [40, 0, 0,  2, 0, 0]
    Tf = 5

    # Breakpoints: linearly spaced
    bps = np.linspace(0, Tf, 30)

    # Cost function: curvature
    c_tcf = tcf_2d_curvature.ctypes
    tcf_av = [ntg.actvar(0, 2), ntg.actvar(1, 2)]

    #
    # Start by solving a point to point problem to get a baseline
    #

    # Linear constraint functions for initial and final condition
    state_constraint_matrix = np.eye(6, 6)
    initial_val = np.array(z0)
    final_val = np.array(zf)
    bounds = np.hstack([initial_val, final_val])

    # Compute the optimal trajectory
    p2p_coefs, p2p_cost, p2p_inform = ntg.ntg(
        nout, bps, ninterv, order, mult, maxderiv,
        lic=state_constraint_matrix, lfc=state_constraint_matrix,
        lowerb=bounds, upperb=bounds,
        tcf=c_tcf, tcf_av=tcf_av)
    assert p2p_inform in [0, 1]

    #
    # Now change the initial/final constraint to cost functions
    #

    c_icf = nl_2d_initial_cost.ctypes
    icf_av = [ntg.actvar(i, j) for i in range(nout) for j in range(maxderiv[i])]
    c_fcf = nl_2d_final_cost.ctypes
    fcf_av = [ntg.actvar(i, j) for i in range(nout) for j in range(maxderiv[i])]

    # Re-solve the problem with initial and final cost
    ifc_coefs, ifc_cost, ifc_inform = ntg.ntg(
        nout, bps, ninterv, order, mult, maxderiv,
        icf=c_icf, icf_av=icf_av,
        fcf=c_fcf, fcf_av=fcf_av,
        tcf=c_tcf, tcf_av=tcf_av)
    assert ifc_inform in [0, 1]

    #
    # Make sure the new cost is lower
    #
    assert ifc_cost < p2p_cost

    #
    # Make sure the initial conditions are close
    #

    # Figure out how much endpoints should have varied (based on cost)
    eps = pow(abs(ifc_cost - p2p_cost) / ifc_weight, 0.5)

    # Get the coefficients into a more useful form
    ncoefs = [ninterv[i] * (order[i] - mult[i]) + mult[i]
              for i in range(nout)]
    ifc_coef_list = [ifc_coefs[:ncoefs[0]], ifc_coefs[ncoefs[0]:]]

    # Figure out starting and ending points
    knots = [np.linspace(0, Tf, ninterv[i] + 1) for i in range(nout)]
    ifc_z0 = [ntg.spline_interp(0., knots[i], ninterv[i], ifc_coef_list[i],
                                order[i], mult[i], maxderiv[i])
                for i in range(nout)]
    ifc_zf = [ntg.spline_interp(Tf, knots[i], ninterv[i], ifc_coef_list[i],
                                order[i], mult[i], maxderiv[i])
                for i in range(nout)]

    k = 0
    for i in range(nout):
        for j in range(maxderiv[i]):
            assert abs(ifc_z0[i][j] - z0[k]) <= eps
            k += 1

    #
    # Add input constraints (second derivative of flat outputs)
    #

    # Get the full trajectory
    ztraj = list()
    for i in range(nout):
        ztraj.append(np.array([
            ntg.spline_interp(t, knots[i], ninterv[i], ifc_coef_list[i],
                              order[i], mult[i], maxderiv[i])
            for t in np.linspace(0, Tf, 100)]))

    # Find the max of each each variable
    u1_max = np.max(np.abs(ztraj[0][:, 2]))
    u2_max = np.max(np.abs(ztraj[1][:, 2]))

    # Set up linear constraints on the inputs
    alpha = 0.98
    input_constraint_matrix = np.array([
        [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1]])
    input_constraint_lowerb = np.array([-alpha * u1_max, -alpha * u2_max])
    input_constraint_upperb = np.array([ alpha * u1_max,  alpha * u2_max])

    # Create the new bound matrix
    ltc_lowerb = np.hstack([initial_val, input_constraint_lowerb, final_val])
    ltc_upperb = np.hstack([initial_val, input_constraint_upperb, final_val])

    # Resolve with constrainted inputs
    ltc_coefs, ltc_cost, ltc_inform = ntg.ntg(
        nout, bps, ninterv, order, mult, maxderiv,
        lic=state_constraint_matrix,
        ltc=input_constraint_matrix,
        lfc=state_constraint_matrix,
        lowerb=ltc_lowerb, upperb=ltc_upperb,
        tcf=c_tcf, tcf_av=tcf_av)
    assert ltc_inform in [0, 1]

    #
    # Make sure the new cost is higher
    #
    assert ltc_cost > p2p_cost

    # Get the coefficients into a more useful form
    ltc_coef_list = [ltc_coefs[:ncoefs[0]], ltc_coefs[ncoefs[0]:]]

    # Make sure the constraints were satisfied (up to some numerical precision)
    ltc_ztraj = list()
    for i in range(nout):
        ltc_ztraj.append(np.array([
            ntg.spline_interp(t, knots[i], ninterv[i], ltc_coef_list[i],
                              order[i], mult[i], maxderiv[i])
            for t in bps]))
    assert np.max(np.abs(ltc_ztraj[0][:, 2])) <= alpha * u1_max * 1.00001
    assert np.max(np.abs(ltc_ztraj[1][:, 2])) <= alpha * u2_max * 1.00001

#
# State constraints
#

@pytest.mark.parametrize("nltcf_avs", [
    [(0, 0), (1, 0)],
    None,
    ])
def test_2d_curvature_corridor_single(nltcf_avs):
    """This unit test adds a set of "corridor" constraints to the 2D curve
    example, testing out the use of a (nonlinear) trajectory constraint.
    """

    # System definition
    nout = 2                    # 2 flat outputs
    maxderiv = [3, 3]           # 3 derivatives in each output

    # Spline definition
    ninterv = [2, 2]
    mult = [3, 3]
    order = [6, 6]

    # Problem definition
    z0 = np.array([0.,  8., 0., -2., 0., 0.])
    zf = np.array([40., 8., 0.,  2., 0., 0.])
    Tf = 5.

    # Breakpoints: linearly spaced
    bps = np.linspace(0, Tf, 20)

    # Cost function: curvature
    c_tcf = tcf_2d_curvature.ctypes
    tcf_av = [ntg.actvar(0, 2), ntg.actvar(1, 2)]

    #
    # Start by solving a point to point problem to get a baseline
    #

    # Linear constraint functions for initial and final condition
    state_constraint_matrix = np.eye(6, 6)
    initial_val = np.array(z0)
    final_val = np.array(zf)
    bounds = np.hstack([initial_val, final_val])

    # Compute the optimal trajectory
    p2p_coefs, p2p_cost, p2p_inform = ntg.ntg(
        nout, bps, ninterv, order, mult, maxderiv,
        lic=state_constraint_matrix, lfc=state_constraint_matrix,
        lowerb=bounds, upperb=bounds,
        tcf=c_tcf, tcf_av=tcf_av)
    assert p2p_inform in [0, 1]

    # Get the coefficients into a more useful form
    ncoefs = [ninterv[i] * (order[i] - mult[i]) + mult[i]
              for i in range(nout)]
    p2p_coef_list = [p2p_coefs[:ncoefs[0]], p2p_coefs[ncoefs[0]:]]

    # Get the full trajectory (at breakpoints)
    knots = [np.linspace(0, Tf, ninterv[i] + 1) for i in range(nout)]
    p2p_ztraj = np.empty((nout, max(maxderiv), bps.size))
    for i in range(nout):
        p2p_ztraj[i] = np.array([
            ntg.spline_interp(t, knots[i], ninterv[i], p2p_coef_list[i],
                              order[i], mult[i], maxderiv[i])
            for t in bps]).transpose()

    #
    # Now impose a "corridor" constraint that forces the trajectory to cost
    # a bit more.
    #
    @numba.cfunc(
        types.void(
            types.CPointer(types.intc),       # int *mode
            types.CPointer(types.intc),       # int *nstate
            types.CPointer(types.intc),       # int *i
            types.CPointer(types.double),     # double *f
            types.CPointer(                   # double **df
                types.CPointer(types.double)),
            types.CPointer(                   # double **zp
                types.CPointer(types.double))))
    def nltcf_corridor(mode, nstate, i, f, df, zp):
        m = (zf[3] - z0[3]) / (zf[0] - z0[0])
        b = z0[3];

        if mode[0] == 0 or mode[0] == 2:
            # Compute the distance from the line connecting start to end
            d = m * (zp[0][0] - z0[0]) + b - zp[1][0]
            f[0] = d

        if mode[0] == 1 or mode[0] == 2:
            # Compute gradient of constraint fcn (2nd index = flat variables)
            df[0][0] = m;  df[0][1] = df[0][2] = 0;
            df[0][3] = -1; df[0][4] = df[0][5] = 0;

    c_nltcf_corridor = nltcf_corridor.ctypes
    nltcf_av = None if nltcf_avs is None else \
        [ntg.actvar(*args) for args in nltcf_avs]

    # Add corridor constraints
    corridor_radius = 0.45
    lowerb = np.hstack(
        [initial_val, final_val, np.array([-corridor_radius])])
    upperb = np.hstack(
        [initial_val, final_val, np.array([corridor_radius])])

    # Re-solve the problem with initial and final cost
    nltcf_coefs, nltcf_cost, nltcf_inform = ntg.ntg(
        nout, bps, ninterv, order, mult, maxderiv,
        lic=state_constraint_matrix, lfc=state_constraint_matrix,
        nltcf=c_nltcf_corridor, nltcf_av=nltcf_av, nltcf_num=1,
        lowerb=lowerb, upperb=upperb,
        tcf=c_tcf, tcf_av=tcf_av, verbose=True)
    assert nltcf_inform in [0, 1]

    # Get the coefficients into a more useful form
    ncoefs = [ninterv[i] * (order[i] - mult[i]) + mult[i]
              for i in range(nout)]
    nltcf_coef_list = [nltcf_coefs[:ncoefs[0]], nltcf_coefs[ncoefs[0]:]]

    # Get the full trajectory (at breakpoints)
    nltcf_ztraj = np.empty((nout, max(maxderiv), bps.size))
    for i in range(nout):
        nltcf_ztraj[i] = np.array([
            ntg.spline_interp(t, knots[i], ninterv[i], nltcf_coef_list[i],
                              order[i], mult[i], maxderiv[i])
            for t in bps]).transpose()

    # Make sure the new cost is higher
    assert nltcf_cost > p2p_cost

    # Make sure the initial and final conditions are close
    np.testing.assert_almost_equal(nltcf_ztraj[:, :, 0], z0.reshape(nout, -1))
    np.testing.assert_almost_equal(nltcf_ztraj[:, :, -1], zf.reshape(nout, -1))

    # Make sure the corridor constraints are (approximately) satisfied
    m = (zf[3] - z0[3]) / (zf[0] - z0[0])
    b = z0[3]
    assert all(abs(
        m * (nltcf_ztraj[0, 0, :]) + b - nltcf_ztraj[1, 0, :])
               <=  corridor_radius * 1.00001)


def test_2d_curvature_corridor_multiple():
    """This unit test adds a set of "corridor" constraints to the 2D curve
    example, testing out the use of (multiple, nonlinear) trajectory
    constraints.
    """

    # System definition
    nout = 2                    # 2 flat outputs
    maxderiv = [3, 3]           # 3 derivatives in each output

    # Spline definition
    ninterv = [2, 2]
    mult = [3, 3]
    order = [6, 6]

    # Problem definition
    z0 = np.array([0.,  8., 0., -2., 0., 0.])
    zf = np.array([40., 8., 0.,  2., 0., 0.])
    Tf = 5.

    # Breakpoints: linearly spaced
    bps = np.linspace(0, Tf, 20)

    # Cost function: curvature
    c_tcf = tcf_2d_curvature.ctypes
    tcf_av = [ntg.actvar(0, 2), ntg.actvar(1, 2)]

    #
    # Start by solving a point to point problem to get a baseline
    #

    # Linear constraint functions for initial and final condition
    state_constraint_matrix = np.eye(6, 6)
    initial_val = np.array(z0)
    final_val = np.array(zf)
    bounds = np.hstack([initial_val, final_val])

    # Compute the optimal trajectory
    p2p_coefs, p2p_cost, p2p_inform = ntg.ntg(
        nout, bps, ninterv, order, mult, maxderiv,
        lic=state_constraint_matrix, lfc=state_constraint_matrix,
        lowerb=bounds, upperb=bounds,
        tcf=c_tcf, tcf_av=tcf_av)
    assert p2p_inform in [0, 1]

    # Get the coefficients into a more useful form
    ncoefs = [ninterv[i] * (order[i] - mult[i]) + mult[i]
              for i in range(nout)]
    p2p_coef_list = [p2p_coefs[:ncoefs[0]], p2p_coefs[ncoefs[0]:]]

    # Get the full trajectory (at breakpoints)
    knots = [np.linspace(0, Tf, ninterv[i] + 1) for i in range(nout)]
    p2p_ztraj = np.empty((nout, max(maxderiv), bps.size))
    for i in range(nout):
        p2p_ztraj[i] = np.array([
            ntg.spline_interp(t, knots[i], ninterv[i], p2p_coef_list[i],
                              order[i], mult[i], maxderiv[i])
            for t in bps]).transpose()

    #
    # Now impose a "corridor" constraint that forces the trajectory to cost
    # a bit more.
    #
    @numba.cfunc(
        types.void(
            types.CPointer(types.intc),       # int *mode
            types.CPointer(types.intc),       # int *nstate
            types.CPointer(types.intc),       # int *i
            types.CPointer(types.double),     # double *f
            types.CPointer(                   # double **df
                types.CPointer(types.double)),
            types.CPointer(                   # double **zp
                types.CPointer(types.double))))
    def nltcf_corridor(mode, nstate, i, f, df, zp):
        m = (zf[3] - z0[3]) / (zf[0] - z0[0])
        b = z0[3];

        if mode[0] == 0 or mode[0] == 2:
            # Compute the distance from the line connecting start to end
            d = m * (zp[0][0] - z0[0]) + b - zp[1][0]
            f[0] = d; f[1] = d

        if mode[0] == 1 or mode[0] == 2:
            # Compute gradient of constraint fcn (2nd index = flat variables)
            df[0][0] = m;  df[0][1] = df[0][2] = 0;
            df[0][3] = -1; df[0][4] = df[0][5] = 0;

            df[1][0] = m;  df[1][1] = df[1][2] = 0;
            df[1][3] = -1; df[1][4] = df[1][5] = 0;

    c_nltcf_corridor = nltcf_corridor.ctypes
    nltcf_av = [ntg.actvar(0, 0), ntg.actvar(1, 0)]

    # Add corridor constraints
    corridor_radius = 0.45
    lowerb = np.hstack(
        [initial_val, final_val, np.array([-corridor_radius, -1e10])])
    upperb = np.hstack(
        [initial_val, final_val, np.array([1e10, corridor_radius])])

    # Re-solve the problem with initial and final cost
    nltcf_coefs, nltcf_cost, nltcf_inform = ntg.ntg(
        nout, bps, ninterv, order, mult, maxderiv,
        lic=state_constraint_matrix, lfc=state_constraint_matrix,
        nltcf=c_nltcf_corridor, nltcf_av=nltcf_av, nltcf_num=2,
        lowerb=lowerb, upperb=upperb,
        tcf=c_tcf, tcf_av=tcf_av, verbose=True)
    assert nltcf_inform in [0, 1]

    # Get the coefficients into a more useful form
    ncoefs = [ninterv[i] * (order[i] - mult[i]) + mult[i]
              for i in range(nout)]
    nltcf_coef_list = [nltcf_coefs[:ncoefs[0]], nltcf_coefs[ncoefs[0]:]]

    # Get the full trajectory (at breakpoints)
    nltcf_ztraj = np.empty((nout, max(maxderiv), bps.size))
    for i in range(nout):
        nltcf_ztraj[i] = np.array([
            ntg.spline_interp(t, knots[i], ninterv[i], nltcf_coef_list[i],
                              order[i], mult[i], maxderiv[i])
            for t in bps]).transpose()

    # Make sure the new cost is higher
    assert nltcf_cost > p2p_cost

    # Make sure the initial and final conditions are close
    np.testing.assert_almost_equal(nltcf_ztraj[:, :, 0], z0.reshape(nout, -1))
    np.testing.assert_almost_equal(nltcf_ztraj[:, :, -1], zf.reshape(nout, -1))

    # Make sure the corridor constraints are (approximately) satisfied
    m = (zf[3] - z0[3]) / (zf[0] - z0[0])
    b = z0[3]
    assert all(abs(
        m * (nltcf_ztraj[0, 0, :]) + b - nltcf_ztraj[1, 0, :])
               <=  corridor_radius * 1.00001)
