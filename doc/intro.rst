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

subject to the constraints

.. math::

   \begin{aligned}
     &L_0 \leq A_0\, \bar z (T_0) \leq U_0 \\
     &L_i \leq A_i\, \bar z (T_i) \leq U_i \\
     &L_f \leq A_f\, \bar z (T_f) \leq U_f \\
   \end{aligned}
