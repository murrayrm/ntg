.. _package-ref:

.. currentmodule:: ntg

*********************
Package Documentation
*********************

Functions
=========

.. autosummary::
   :toctree: generated/

   call_ntg
   solve_flat_ocp

The :func:`~ntg.solve_flat_ocp` function is the standard interface to NTG.
The arguments specify the flat system description, the time points that will
be used to evaluate the cost and constriants, the basis function (B-spline
parameters) used to parameterize the flat outputs, and the initial,
trajectory, and final costs and constraints that define the optimization
problem to be solved.  Cost functions and constraints should be specified as
`ctypes` functions with the appropriate calling signatures.  These will
usually be created as `numba.cfunc` functions, as described in more detail
below.

The :func:`~ntg.call_ntg` provides a low-level interface to NTG, with the
arguments undergoing very minor processing before calling NTG.

Classes
=======

.. autosummary::
   :toctree: generated/
   :recursive:

   ActiveVariable
   FlatSystem
   OptimalControlResult
   SystemTrajectory


Numba Signatures
================

The NTG package defines a set of Numba signatures that can be used when
defining callback functions.

The primary numba callback signatures are used when passing functions to
`~ntg.solve_flat_ocp`:

+---------------------------------------+-----------------------------+
| Signature                             | Purpose                     |
+=======================================+=============================+
| numba_endpoint_cost_signature         | Initial/final cost function |
+---------------------------------------+-----------------------------+
| numba_endpoint_constraint_signature   | Initial/final constraints   |
+---------------------------------------+-----------------------------+
| numba_trajectory_cost_signature       | Trajectory cost function    |
+---------------------------------------+-----------------------------+
| numba_trajectory_constraint_signature | Trajectory constraints      |
+---------------------------------------+-----------------------------+

For the endpoint cost functions, the C signature is given by::

  retval = cost(int mode, int nstate, double *f, double *df, double **zflag)

where `mode` indicates whether the cost or constraint should be satisfied,
`nstate` is set to zero if this is the first call to the function, `f` is a
a pointer to a double that can be used to return the value of the cost
function, `df` is a pointer to a list of doubles that give the gradient of
the cost, and `zflag` is a pointer to a list of doubles that provide the
flag of the flat variables.  The gradient `df` is with respect to the
(stacked) flat flag variable.

The mode variable has the following meaning

* `mode = 0`: compute just the value of the cost, `f`
* `mode = 1`: compute just the graduate of the cost, `df`
* `mode = 2`: compute both the cost `f` and its gradient `df`

The return value of the cost function should be zero if the cost function
was successfully computed, otherwise -1 (which will terminate the
optimization).

For trajectory cost functions, the C signature is given by::

  retval = cost(int mode, int i, int nstate, double *f, double *df, double **zflag)

The additional argument `i` specifies the breakpoint index at which the
function is being evaluated.

For example, to define a cost function along a trajectory for a system with
2 flat outputs each having a flag of length 3, the following code can be
used::

  # Cost function: curvature
  @numba.cfunc(ntg.numba_trajectory_cost_signature)
  def tcf_2d_curvature(mode, nstate, i, f, df, zp):
      if mode == 0 or mode == 2:
          f[0] = zp[0][2]**2 + zp[1][2]**2

      if mode == 1 or mode == 2:
          df[0] = 0; df[1] = 0; df[2] = 2 * zp[0][2];
          df[3] = 0; df[4] = 0; df[5] = 2 * zp[1][2];

      return 0

Constraints functions have the same call signature except that the `df`
entry has type `double **`, with the first index representing the constraint
and the second index representing the flat flag.

For the low-level interface to NTG, via `~ngg.call_ntg`, the following
signatures are used:

+-------------------------------------------+---------------------------+
| Signature                                 + Purpose                   |
+===========================================+===========================+
| numba_ntg_endpoint_cost_signature         | Initial/final cost        |
+-------------------------------------------+---------------------------+
| numba_ntg_endpoint_constraint_signature   | Initial/final constraints |
+-------------------------------------------+---------------------------+
| numba_ntg_trajectory_cost_signature       | Trajectory cost           |
+-------------------------------------------+---------------------------+
| numba_ntg_trajectory_constraint_signature | Trajectory constraints    |
+-------------------------------------------+---------------------------+

.. autosummary::
   :toctree: generated/

