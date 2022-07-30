************
Introduction
************

Overview of differential flatness
=================================

A nonlinear differential equation of the form 

.. math::
    \dot x = f(x, u), \qquad x \in R^n, u \in R^m

is *differentially flat* if there exists a function :math:`\alpha` such that

.. math::
    z = \alpha(x, u, \dot u\, \dots, u^{(p)})

and we can write the solutions of the nonlinear system as functions of
:math:`z` and a finite number of derivatives

.. math::
    x &= \beta(z, \dot z, \dots, z^{(q)}) \\
    u &= \gamma(z, \dot z, \dots, z^{(q)}).
    :label: flat2state

For a differentially flat system, all of the feasible trajectories for
the system can be written as functions of a flat output :math:`z(\cdot)` and
its derivatives.  The number of flat outputs is always equal to the
number of system inputs.

Differentially flat systems are useful in situations where explicit
trajectory generation is required. Since the behavior of a flat system
is determined by the flat outputs, we can plan trajectories in output
space, and then map these to appropriate inputs.  Suppose we wish to
generate a feasible trajectory for the the nonlinear system

.. math::
    \dot x = f(x, u), \qquad x(0) = x_0,\, x(T) = x_f.

If the system is differentially flat then

.. math::
    x(0) &= \beta\bigl(z(0), \dot z(0), \dots, z^{(q)}(0) \bigr) = x_0, \\
    x(T) &= \gamma\bigl(z(T), \dot z(T), \dots, z^{(q)}(T) \bigr) = x_f,

and we see that the initial and final condition in the full state
space depends on just the output :math:`z` and its derivatives at the
initial and final times.  Thus any trajectory for :math:`z` that satisfies
these boundary conditions will be a feasible trajectory for the
system, using equation :eq:`flat2state` to determine the
full state space and input trajectories.

In particular, given initial and final conditions on :math:`z` and its
derivatives that satisfy the initial and final conditions any curve
:math:`z(\cdot)` satisfying those conditions will correspond to a feasible
trajectory of the system.  We can parameterize the flat output trajectory
using a set of smooth basis functions :math:`\psi_i(t)`:

.. math::
  z(t) = \sum_{i=1}^N c_i \psi_i(t), \qquad c_i \in R

We seek a set of coefficients :math:`c_i`, :math:`i = 1, \dots, N` such
that :math:`z(t)` satisfies the boundary conditions for :math:`x(0)` and
:math:`x(T)`.  The derivatives of the flat output can be computed in terms of
the derivatives of the basis functions:

.. math::
  \dot z(t) &= \sum_{i=1}^N c_i \dot \psi_i(t) \\
  &\,\vdots \\
  \dot z^{(q)}(t) &= \sum_{i=1}^N c_i \psi^{(q)}_i(t).

We can thus write the conditions on the flat outputs and their
derivatives as

.. math::
  \begin{bmatrix}
    \psi_1(0) & \psi_2(0) & \dots & \psi_N(0) \\
    \dot \psi_1(0) & \dot \psi_2(0) & \dots & \dot \psi_N(0) \\
    \vdots & \vdots & & \vdots \\
    \psi^{(q)}_1(0) & \psi^{(q)}_2(0) & \dots & \psi^{(q)}_N(0) \\[1ex]
    \psi_1(T) & \psi_2(T) & \dots & \psi_N(T) \\
    \dot \psi_1(T) & \dot \psi_2(T) & \dots & \dot \psi_N(T) \\
    \vdots & \vdots & & \vdots \\
    \psi^{(q)}_1(T) & \psi^{(q)}_2(T) & \dots & \psi^{(q)}_N(T) \\
  \end{bmatrix}
  \begin{bmatrix} c_1 \\ \vdots \\ c_N \end{bmatrix} =
  \begin{bmatrix}
    z(0) \\ \dot z(0) \\ \vdots \\ z^{(q)}(0) \\[1ex]
    z(T) \\ \dot z(T) \\ \vdots \\ z^{(q)}(T) \\
  \end{bmatrix}

This equation is a *linear* equation of the form 

.. math::
   M c = \begin{bmatrix} \bar z(0) \\ \bar z(T) \end{bmatrix}

where :math:`\bar z` is called the *flat flag* for the system.
Assuming that :math:`M` has a sufficient number of columns and that it is full
column rank, we can solve for a (possibly non-unique) :math:`\alpha` that
solves the trajectory generation problem.

TODO: add more about optimization and constraints

Introduction to NTG
===================

The NTG package can be used to generate optimal trajectories for
differentially flat systems with constraints.  NTG works completely in the
flat coordinates of the system, so that all costs and constriants are
written in terms of :math:`\bar z`, representing the flat outputs and a
finite number of derivatives.

NTG solves an optimal control problem of the form

.. math::

   \min_{z(\cdot)}\:
   V_0 \bigl( \bar z(0) \bigr) + \int_0^{T_\text{f}} L(\bar z(t))\, dt +
   V_f \bigl( \bar z(T_\text{f}) \bigr)

subject to a collection of linear and nonlinear constraints at the
initial, intermediate, and final time points:

.. math::

   L_0 \leq \begin{bmatrix}
       A_0\, \bar z (T_0) \\ F_0(\bar z(T_0))
     \end{bmatrix} \leq U_0, \qquad
   L_i \leq \begin{bmatrix}
       A_i\, \bar z (T_i) \\ F_i(\bar z(T_i))
     \end{bmatrix} \leq U_i, \qquad
   L_\text{f} \leq \begin{bmatrix}
     A_\text{f}\, \bar z (T_\text{f}) \\ F_\text{f}(\bar z(T_\text{f}))
   \end{bmatrix} \leq U_f.

NTG represents the flat outputs of the system using B-splines, which
form a basis for piecewise smooth polynomials that have a specified
level of smoothness at the breakpoints between intervals.

To create a trajectory for a differentially flat system, a
:class:`~ntg.FlatSystem` object must be created.  This is done by
specifying the :class:`~ntg:FlatSystem` constructor with the number of
flat outputs and the number of derivatives for each output::

    sys = ntg.FlatSystem(nout, flaglen)

In addition to the flat system description, a set of basis functions
:math:`\phi_i(t)` must be chosen.  The :class:`~ntg.BSplineFamily`
class is used to represent the basis functions::

    basis = ntg.BSpline(breakpoints, order[, smoothness)

Once the system and basis function have been defined, the
:func:`~ntg.solve_flat_ocp` function can be used to solve an optimal
control problem::

    traj = ntg.solve_flat_ocp(
        sys, timepts, initial_constraints=initial,
	trajectory_cost=cost, final_constraints=final, basis=basis)

The `cost` parameter is a function function with call signature
`cost(zflag)` and should return the (incremental) cost at the given
value of the flat output and its derivatives.  It will be evaluated at
each point in the `timepts` vector.  The `initial_constraints` and
`terminal_constraints` parameters can be used to specify the initial
and final conditions.

A typical usage is to constraint the initial and final values of the
flat flag and place a cost function on higher derivatives of the flag.
This can be achieved using the :func:`~ntg.flag_value_constraint`
and :func:`~ntg.quadratic_cost` functions::

    initial = ntg.flag_equality_constraint(sys, Z0)
    final = ntg.flag_equality_constraint(sys, Zf)
    cost = ntg.quadratic_cost(sys, [Q_1, ..., Q_m], Zd)

The returned object from :func:`~ntg.solve_flat_ocp` has class
:class:`~ntg.SystemTrajectory` and can be used to compute the state
and input trajectory between the initial and final condition::

    ztraj = traj.eval(T)

where `T` is a list of times on which the trajectory should be evaluated
(e.g., `T = numpy.linspace(0, Tf, M)`.

The :func:`~ntg.solve_flat_ocp` function also allows the specification
of a initial and terminal cost function as well as trajectory
constraints.  Constraints can either be linear or nonlinear.


Example
=======

To illustrate how we can use NTG to compute an optimal trajectory for
a nonlinear system, consider the problem of steering a car to change
lanes on a road.  A more complete description of the system can be
found in the course notes *Optimization-Based Control*
[http://fbswiki.org/OBC].

.. code-block:: python

    import ntg
    import numpy as np

    # The system has two flat outputs with flag of length 3 in each
    vehicle_flat = ntg.FlatSystem(2, [3, 3])

To find a trajectory from an initial flag value :math:`Z_0` to a final
flag value of :math:`F_\text{f}` in time :math:`T_\text{f}` we solve a
point-to-point trajectory generation problem while minimizing the
curvature of the trajectory (corresponding to minimizing the steering
wheel angle :math:`\delta` along the trajectory).

.. code-block:: python

    # Define the endpoints of the trajectory
    Z0 = ([  0, 10, 0], [-2, 0, 0])
    Zf = ([100, 10, 0], [ 2, 0, 0])
    Tf = 10

    # Define a set of basis functions to use for the trajectories
    # TODO: update to python-control signature
    basis = ntg.BSplineFamily(2, [0, Tf/2, Tf], 6)

    # Define the initial and final states
    initial = ntg.flag_equality_constraint(vehicle_flat, Z0)
    final = ntg.flag_equality_constraint(vehicle_flat, Zf)

    # Define the cost along the trajectory: penalize steering angle
    costfun = ntg.quadratic_cost(
        vehicle_flat, [np.diag([0, 0, 1]), np.diag([0, 0, 1])])

    # Define the time points and solve the optimal control problem
    timepts = np.linspace(0, Tf, 10)
    traj, cost, _ = ntg.solve_flat_ocp(		# TODO: redo signature
        vehicle_flat, timepts, basis=basis, trajectory_cost=costfun,
	initial_constraints=initial, final_constraints=final)

    T = np.linspace(0, Tf, 100)
    ztraj = traj.eval(T)
