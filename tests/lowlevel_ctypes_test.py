# lowlevel_ctypes_test.py - test low-level NTG interface using ctypes
# RMM, 16 Jul 2022

import ctypes
import numpy as np
import itertools
import ntg
import pytest

cfcns = ctypes.cdll.LoadLibrary('lowlevel.so')

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
    flaglen = [3, 3]            # 2 derivatives in each output

    # Breakpoints: linearly spaced
    bps = np.linspace(0, Tf, 30)

    # Cost function: curvature
    c_tcf = cfcns.tcf_2d_curvature
    tcf_av = [ntg.actvar(0, 2), ntg.actvar(0, 2)]

    # Linear constraint functions for initial and final condition
    state_constraint_matrix = np.eye(6, 6)
    initial_val = np.array(zf_0)
    final_val = np.array(zf_f)

    # Create the bounds matrix
    bounds = np.hstack([initial_val, final_val])

    # Compute the optimal trajectory
    systraj, cost, inform = ntg.ntg(
        nout, bps, ninterv, order, mult, flaglen,
        lic=state_constraint_matrix, lfc=state_constraint_matrix,
        lowerb=bounds, upperb=bounds,
        tcf=c_tcf, tcf_av=tcf_av)
    coefs = systraj.coefs

    # Make sure the optimization succeedd
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
    np.testing.assert_almost_equal(np.hstack(z0_check), zf_0)

    zf_check = [ntg.spline_interp(Tf, knots[i], ninterv[i],
                                 coef_list[i], order[i], mult[i], flaglen[i])
                for i in range(nout)]
    np.testing.assert_almost_equal(np.hstack(zf_check), zf_f)

def test_2d_curvature_ifc_ltc():
    """This unit test replaces initial and final constraints with cost
    functions, which should allow for reduced cost, and adds linear
    trajectory constraints, which should increase the cost.
    """

    # System definition
    nout = 2                    # 2 flat outputs
    flaglen = [3, 3]           # 3 derivatives in each output

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
    c_tcf = cfcns.tcf_2d_curvature
    tcf_av = [ntg.actvar(0, 2), ntg.actvar(0, 2)]

    #
    # Start by solving a point to point problem to get a baseline
    #

    # Linear constraint functions for initial and final condition
    state_constraint_matrix = np.eye(6, 6)
    initial_val = np.array(z0)
    final_val = np.array(zf)
    bounds = np.hstack([initial_val, final_val])

    # Compute the optimal trajectory
    p2p_systraj, p2p_cost, p2p_inform = ntg.ntg(
        nout, bps, ninterv, order, mult, flaglen,
        lic=state_constraint_matrix, lfc=state_constraint_matrix,
        lowerb=bounds, upperb=bounds,
        tcf=c_tcf, tcf_av=tcf_av)
    p2p_coefs = p2p_systraj.coefs
    assert p2p_inform in [0, 1]

    #
    # Now change the initial/final constraint to cost functions
    #

    c_icf = cfcns.nl_2d_initial_cost
    icf_av = [ntg.actvar(i, j) for i in range(nout) for j in range(flaglen[i])]
    c_fcf = cfcns.nl_2d_final_cost
    fcf_av = [ntg.actvar(i, j) for i in range(nout) for j in range(flaglen[i])]

    # Re-solve the problem with initial and final cost
    ifc_systraj, ifc_cost, ifc_inform = ntg.ntg(
        nout, bps, ninterv, order, mult, flaglen,
        icf=c_icf, icf_av=icf_av,
        fcf=c_fcf, fcf_av=fcf_av,
        tcf=c_tcf, tcf_av=tcf_av)
    ifc_coefs = ifc_systraj.coefs
    assert ifc_inform in [0, 1]

    #
    # Make sure the new cost is lower
    #
    assert ifc_cost < p2p_cost

    #
    # Make sure the initial conditions are close
    #

    # Figure out how much endpoints should have varied (based on cost)
    ifc_weight = ctypes.c_double.in_dll(cfcns, 'ifc_weight').value
    eps = pow(abs(ifc_cost - p2p_cost) / ifc_weight, 0.5)

    # Get the coefficients into a more useful form
    ncoefs = [ninterv[i] * (order[i] - mult[i]) + mult[i]
              for i in range(nout)]
    ifc_coef_list = [ifc_coefs[:ncoefs[0]], ifc_coefs[ncoefs[0]:]]

    # Figure out starting and ending points
    knots = [np.linspace(0, Tf, ninterv[i] + 1) for i in range(nout)]
    ifc_z0 = [ntg.spline_interp(0., knots[i], ninterv[i], ifc_coef_list[i],
                                order[i], mult[i], flaglen[i])
                for i in range(nout)]
    ifc_zf = [ntg.spline_interp(Tf, knots[i], ninterv[i], ifc_coef_list[i],
                                order[i], mult[i], flaglen[i])
                for i in range(nout)]

    k = 0
    for i in range(nout):
        for j in range(flaglen[i]):
            assert abs(ifc_z0[i][j] - z0[k]) <= eps
            k += 1

    #
    # Add trajectory constraints on the inputs (second derivative of outputs)
    #

    # Get the full trajectory
    ztraj = list()
    for i in range(nout):
        ztraj.append(np.array([
            ntg.spline_interp(t, knots[i], ninterv[i], ifc_coef_list[i],
                              order[i], mult[i], flaglen[i])
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
    ltc_systraj, ltc_cost, ltc_inform = ntg.ntg(
        nout, bps, ninterv, order, mult, flaglen,
        lic=state_constraint_matrix,
        ltc=input_constraint_matrix,
        lfc=state_constraint_matrix,
        lowerb=ltc_lowerb, upperb=ltc_upperb,
        tcf=c_tcf, tcf_av=tcf_av)
    ltc_coefs = ltc_systraj.coefs
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
                              order[i], mult[i], flaglen[i])
            for t in bps]))
    assert np.max(np.abs(ltc_ztraj[0][:, 2])) <= alpha * u1_max * 1.00001
    assert np.max(np.abs(ltc_ztraj[1][:, 2])) <= alpha * u2_max * 1.00001
