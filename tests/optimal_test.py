# Test function in the optimal module
# RMM, 30 Jul 2022

import ntg
import numpy as np
import scipy as sp
import pytest

@pytest.mark.parametrize("Tf, basis", [
    (5, ntg.BSplineFamily([0, 2.5, 5], [5, 6], [3, 4], vars=2)),
    (5, ntg.BSplineFamily([0, 2.5, 5], [5, 5], [4, 4], vars=2)),
    (5, ntg.BSplineFamily([0, 2.5, 5], 5, 4, vars=2)),
    (5, ntg.BSplineFamily([0, 2.5, 5], 5, 4)),
    ])
def test_quadratic_cost(Tf, basis):
    # System setup
    nout = 2                    # 2 flat outputs
    flaglen = [3, 3]            # 2 derivatives in each output
    sys = ntg.FlatSystem(nout, flaglen)

    # Initial and final conditions
    z0 = np.array([[0., 8., 0.], [-2., 0., 0.]])
    zf = np.array([[40., 8., 0.], [2., 0., 0.]])

    # Time points: linearly spaced
    timepts = np.linspace(0, Tf, 30)

    # Cost function: curvature
    costfun = ntg.quadratic_cost(sys, [np.diag([0, 0, 1]), np.diag([0, 0, 1])])
    cost_av = [ntg.actvar(0, 2), ntg.actvar(1, 2)]

    # Constraint functions for initial and final condition
    initial_constraints = ntg.flag_equality_constraint(sys, z0)
    final_constraints = ntg.flag_equality_constraint(sys, zf)

    # Compute the optimal trajectory
    systraj, cost, inform = ntg.solve_flat_ocp(
        sys, timepts, basis,
        initial_constraints=initial_constraints,
        final_constraints=final_constraints,
        trajectory_cost=costfun, trajectory_cost_av=cost_av)

    # Make sure the optimization succeeded
    assert inform == 0 or inform == 1

    # Make sure we solved the optimal control problem
    ztraj = systraj.eval(timepts)
    np.testing.assert_almost_equal(ztraj[:, :, 0], z0)
    np.testing.assert_almost_equal(ztraj[:, :, -1], zf)
    assert cost < 2            # minimum cost with good basis is ~1.75

    # Replace terminal constraint with terminal cost
    final_cost = ntg.quadratic_cost(
        sys, [np.eye(3) * 1e2, np.eye(3) * 1e2], zf, type='endpoint')
    systraj_fincost, fincost, inform = ntg.solve_flat_ocp(
        sys, timepts, basis,
        initial_constraints=initial_constraints,
        final_cost=final_cost,
        trajectory_cost=costfun, trajectory_cost_av=cost_av)

    # Make sure we solved the optimal control problem
    ztraj = systraj_fincost.eval(timepts)
    np.testing.assert_almost_equal(ztraj[:, :, 0], z0)
    np.testing.assert_almost_equal(ztraj[:, :, -1], zf, decimal=2)
    assert fincost < cost

    # Replace initial constraint with initial cost
    initial_cost = ntg.quadratic_cost(
        sys, [np.eye(3) * 1e3, np.eye(3) * 1e3], z0, type='endpoint')
    systraj_endcost, endcost, inform = ntg.solve_flat_ocp(
        sys, timepts, basis,
        initial_cost=initial_cost, final_cost=final_cost,
        trajectory_cost=costfun, trajectory_cost_av=cost_av)

    # Make sure we solved the optimal control problem
    ztraj = systraj_fincost.eval(timepts)
    np.testing.assert_almost_equal(ztraj[:, :, 0], z0, decimal=3)
    np.testing.assert_almost_equal(ztraj[:, :, -1], zf, decimal=2)
    assert endcost < fincost
