# lowlevel_test.py - test low-level NTG interface
# RMM, 16 Jul 2022

import ctypes
import numpy as np
import ntg
import pytest

cfcns = ctypes.cdll.LoadLibrary('lowlevel.so')

@pytest.mark.parametrize(
    "zf_0, zf_f, Tf", [
        ([0, 0, 0, 0, 0, 0], [1, 0, 0, 1, 0, 0], 1),
        ([1, 2, 3, 4, 5, 6], [5, 4, 3, 2, 1, 0], 3.1415)
    ])
def test_2d_curvature_p2p(zf_0, zf_f, Tf):
    # System setup
    nout = 2                    # 2 flat outputs
    maxderiv = [3, 3]           # 3 derivatives in each output
    ninterv = [2, 3]            # number of intervals in each variable
    mult = [3, 4]               # multiplicity at knot points
    order = [6, 5]              # order of polynomials in each interval

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
    coefs = ntg.ntg(
        nout, bps, ninterv, order, mult, maxderiv,
        lic=state_constraint_matrix, lfc=state_constraint_matrix,
        lowerb=bounds, upperb=bounds,
        tcf=c_tcf, tcf_av=tcf_av)
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
    
