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
defining callback functions:

.. autosummary::
   :toctree: generated/

   numba_endpoint_cost_signature
   numba_trajectory_constraint_signature
