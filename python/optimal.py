# optimal.py - optimal control program interface to NTG
# RMM, 30 Jul 2022

import numpy as np
import scipy as sp
import scipy.optimize
from scipy.interpolate import BSpline
import ctypes
import numba
from warnings import warn
from .ntg import SystemTrajectory, call_ntg
from .ntg import numba_ntg_trajectory_cost_signature, \
    numba_ntg_endpoint_cost_signature, \
    numba_ntg_endpoint_constraint_signature, \
    numba_ntg_trajectory_constraint_signature

class FlatSystem:
    """FlatSystem(nout, flaglen)

    Class representing a flat system.

    This class is used to represent a differentially flat system.

    Attributes
    ----------
    nout : int
        Number of flat outputs.

    flaglen : 1D array of int
        Length of the flag for each output (number of derivatives + 1).

    """
    def __init__(self, *args):
        """FlatSystem(nout, flaglen)

        Create an object representing a flat system.

        Parameters
        ----------
        nout : int, optional
            Number of outputs of the flat system.  If not specified, determined
            based on the number of entries in flaglen (must be a list).

        flaglen : int or 1D-array
            Length of the flat flag for each output.  If nout is given and
            flaglen is an integer, then each output is assumed to have the
            given length.  If a list is given, then each entry indicates the
            length of the flat for that output.

        Returns
        -------
        sys : FlatSystem
            Object representing the flat system.

        """
        if len(args) == 1 and isinstance(args[0], list):
            self.nout = len(args[0])
            self.flaglen = args[0]
        elif len(args) == 2 and isinstance(args[0], int) and \
             isinstance(args[1], (int, list)):
            self.nout = args[0]
            if isinstance(args[1], int):
                self.flaglen = [args[1] for i in range(self.nout)]
            else:
                self.flaglen = args[1]
        else:
            raise TypeError("invalid flat system specification")


class BSplineFamily():
    """B-spline basis functions.

    This class represents a B-spline basis for piecewise polynomials defined
    across a set of breakpoints with given degree and smoothness.

    """
    def __init__(self, breakpoints, degree, smoothness=None, vars=None):
        """Create a B-spline basis for piecewise smooth polynomials

        Define B-spline polynomials for a set of one or more variables.
        B-splines are used as a basis for a set of piecewise smooth
        polynomials joined at breakpoints. On each interval we have a
        polynomial of a given degree and the spline is continuous up to a
        given smoothness at interior breakpoints.

        Parameters
        ----------
        breakpoints : 1D array or 2D array of float
            The breakpoints for the spline(s).

        degree : int or list of ints
            For each spline variable, the degree of the polynomial between
            break points.  If a single number is given and more than one
            spline variable is specified, the same degree is used for each
            spline variable.

        smoothness : int or list of ints
            For each spline variable, the smoothness at breakpoints (number
            of derivatives that should match).

        vars : None or int, optional
            The number of spline variables.  If specified as None (default),
            then the spline basis describes a single variable, with no
            indexing.  If the number of spine variables is > 0, then the
            spline basis is index using the `var` keyword.

        """
        # Process the breakpoints for the spline */
        breakpoints = np.array(breakpoints, dtype=float)
        if breakpoints.ndim == 2:
            raise NotImplementedError(
                "breakpoints for each spline variable not yet supported")
        elif breakpoints.ndim != 1:
            raise ValueError("breakpoints must be convertable to a 1D array")
        elif breakpoints.size < 2:
            raise ValueError("break point vector must have at least 2 values")
        elif np.any(np.diff(breakpoints) <= 0):
            raise ValueError("break points must be strictly increasing values")

        # Decide on the number of spline variables
        if vars is None:
            nvars = 1
            self.nvars = None           # track as single variable
        elif not isinstance(vars, int):
            raise TypeError("vars must be an integer")
        else:
            nvars = vars
            self.nvars = nvars

        #
        # Process B-spline parameters (degree, smoothness)
        #
        # B-splines are defined on a set of intervals separated by
        # breakpoints.  On each interval we have a polynomial of a certain
        # degree and the spline is continuous up to a given smoothness at
        # breakpoints.  The code in this section allows some flexibility in
        # the way that all of this information is supplied, including using
        # scalar values for parameters (which are then broadcast to each
        # output) and inferring values and dimensions from other
        # information, when possible.
        #

        # Utility function for broadcasting spline params (degree, smoothness)
        def process_spline_parameters(
            values, length, allowed_types, minimum=0,
            default=None, name='unknown'):

            # Preprocessing
            if values is None and default is None:
                return None
            elif values is None:
                values = default
            elif isinstance(values, np.ndarray):
                # Convert ndarray to list
                values = values.tolist()

            # Figure out what type of object we were passed
            if isinstance(values, allowed_types):
                # Single number of an allowed type => broadcast to list
                values = [values for i in range(length)]
            elif all([isinstance(v, allowed_types) for v in values]):
                # List of values => make sure it is the right size
                if len(values) != length:
                    raise ValueError(f"length of '{name}' does not match"
                                     f" number of variables")
            else:
                raise ValueError(f"could not parse '{name}' keyword")

            # Check to make sure the values are OK
            if values is not None and any([val < minimum for val in values]):
                raise ValueError(
                    f"invalid value for '{name}'; must be at least {minimum}")

            return values

        # Degree of polynomial
        degree = process_spline_parameters(
            degree, nvars, (int), name='degree', minimum=1)

        # Smoothness at breakpoints; set default to degree - 1 (max possible)
        smoothness = process_spline_parameters(
            smoothness, nvars, (int), name='smoothness', minimum=0,
            default=[d - 1 for d in degree])

        # Make sure degree is sufficent for the level of smoothness
        if any([degree[i] - smoothness[i] < 1 for i in range(nvars)]):
            raise ValueError("degree must be greater than smoothness")

        # Store the parameters for the spline (self.nvars already stored)
        self.breakpoints = breakpoints
        self.degree = degree
        self.smoothness = smoothness
        self.nintervals = breakpoints.size - 1

        #
        # Compute parameters for a SciPy BSpline object
        #
        # To create a B-spline, we need to compute the knotpoints, keeping
        # track of the use of repeated knotpoints at the initial knot and
        # final knot as well as repeated knots at intermediate points
        # depending on the desired smoothness.
        #

        # Store the coefficients for each output (useful later)
        self.coef_offset, self.coef_length, offset = [], [], 0
        for i in range(nvars):
            # Compute number of coefficients for the piecewise polynomial
            ncoefs = (self.degree[i] + 1) * (len(self.breakpoints) - 1) - \
                (self.smoothness[i] + 1) * (len(self.breakpoints) - 2)

            self.coef_offset.append(offset)
            self.coef_length.append(ncoefs)
            offset += ncoefs
        self.N = offset         # save the total number of coefficients

        # Create knotpoints for each spline variable
        # TODO: extend to multi-dimensional breakpoints
        self.knotpoints = []
        for i in range(nvars):
            # Allocate space for the knotpoints
            self.knotpoints.append(np.empty(
                (self.degree[i] + 1) + (len(self.breakpoints) - 2) * \
                (self.degree[i] - self.smoothness[i]) + (self.degree[i] + 1)))

            # Initial knotpoints (multiplicity = order = degree + 1)
            self.knotpoints[i][0:self.degree[i] + 1] = self.breakpoints[0]
            offset = self.degree[i] + 1

            # Interior knotpoints (multiplicity = degree - smoothness)
            nknots = self.degree[i] - self.smoothness[i]
            assert nknots > 0           # just in case
            for j in range(1, self.breakpoints.size - 1):
                self.knotpoints[i][offset:offset+nknots] = self.breakpoints[j]
                offset += nknots

            # Final knotpoint (multiplicity = order)
            self.knotpoints[i][offset:offset + self.degree[i] + 1] = \
                self.breakpoints[-1]

    def __repr__(self):
        return f'<{self.__class__.__name__}: nvars={self.nvars}, ' + \
            f'degree={self.degree}, smoothness={self.smoothness}>'


#
# OptimalControlResult class
#
# This class is used to store the result of an optimal control problem for a
# flat system, modeled on the python.optimal class of the same name.
#

class OptimalControlResult:
    """Result from solving an optimal control problem for a flat system

    This class contains the result of solving an optimal control problem for
    a differentially flat system.

    Attributes
    ----------
    systraj : SystemTrajectory
        The optimal trajectory computed by the solver.

    cost : float
        Final cost of the return solution.

    inform : int
        Optimizer status (NPSOL)

    """
    def __init__(self, systraj, cost, inform):
        self.systraj = systraj
        self.cost = cost
        self.inform = inform

    # Implement iter to allow assigning to a tuple
    def __iter__(self):
        return iter((self.systraj, self.cost, self.inform))

# Define a class to define active variables
class ActiveVariable(object):
    """Active variable for cost/constraint computation

    This class specifies an active variable used to compute NTG costs or
    constraints.

    Parameters
    ----------
    output : int
        The output index of the active variable.

    deriv : int
        The derivative of the active variable.

    """
    def __init__(self, output, deriv):
        self.output = output
        self.deriv = deriv
actvar = ActiveVariable         # short form

#
# High level NTG interface: solve_flat_ocp
#
# Simplified interface to specify and solve an optimal control problem for a
# differentially flat system.  This interface is modeled after the
# control.flatsys and control.optimal modules in python-control.
#
def solve_flat_ocp(
    sys,                   # flat system
    timepts,               # time points
    basis,                 # (B-spline) basis function description
    initial_cost=None,          # initial cost
    initial_cost_av=None,       # initial cost active variables
    trajectory_cost=None,       # cost along the trajectory
    trajectory_cost_av=None,    # trajectory cost active variables
    final_cost=None,            # final cost
    final_cost_av=None,         # final cost active variables
    initial_constraints=None,       # initial constraints
    initial_constraint_av=None,     # initial constraint active variables
    trajectory_constraints=None,    # trajectory constraints
    trajectory_constraint_av=None,  # trajectory constraint active variables
    final_constraints=None,         # final constraints
    final_constraint_av=None,       # final constraint active variables
    verbose=False,         # turn on verbose message
    **kwargs,              # allow some alternative keywords
):
    """Constrained, optimal trajectory generation

    Compute an optimal trajectory for a differentially flat system with
    initial, trajectory, and final costs and constraints.

    Parameters
    ----------
    sys : FlatSystem
        Flat system description.

    timepts : 1D array-like
        Array of times at which constraints should be evaluated and integrated
        costs are computed.

    basis : BSplineFamily
        Description of the B-spline basis to use for flat output trajectories.

    Returns
    -------
    result : :class:`~ntg.OptimalControlResult` object
        The result of the optimal control problem, including the system
        trajectory (`result.systraj`), optimal cost (`result.cost`), and
        optimizer status (`result.inform`).

    """
    # Make sure basis is consistent system
    if basis.nvars is not None and sys.nout != basis.nvars:
        raise ValueError(
            f"system size ({sys.nout}) and basis size "
            f"({basis.nvars}) must match")

    # If we were given a 1D basis, expand it to multi-dimensional basis
    if basis.nvars is None:
        basis = BSplineFamily(
            basis.breakpoints, basis.degree[0],
            basis.smoothness[0], vars=sys.nout)

    # Process keywords
    def process_alt_kwargs(kwargs, primary, other, name):
        for kw in other:
            if kw in kwargs:
                if primary is not None:
                    raise TypeError(f"redundant keywords: {name}, {kw}")
                primary = kwargs.pop(kw)
                name = kw
        return primary

    trajectory_cost = process_alt_kwargs(
        kwargs, trajectory_cost, ['cost'], 'trajectory_cost')
    trajectory_cost_av = process_alt_kwargs(
        kwargs, trajectory_cost_av, ['cost_av'], 'trajectory_cost_av')

    # Make sure there were no additional keyword arguments
    if kwargs:
        raise TypeError("unrecognized keyword arguments", kwargs)

    #
    # Time points
    #
    # Process the time points next.  These determine the points at which we
    # evaluate the cost functions and constraints.
    #
    timepts = np.atleast_1d(timepts)
    if timepts.ndim != 1:
        raise ValueError("timepts must be a 1D array")
    elif np.any(np.diff(timepts) <= 0):
        raise ValueError("timepts must be strictly increasing")

    #
    # Process cost functions
    #
    # Cost functions are represented by ctypes functions that use the
    # scipy.optimize.minimize format (and signature [TODO]).  These are
    # called using a wrapper function to convert function arguments.
    #

    #! TODO: refactor using a single utility function (ala constraints)?
    if initial_cost:
        # Define a wrapper around the function to change the arguments
        @numba.cfunc(numba_ntg_endpoint_cost_signature)
        def _call_initial_cost(mode, nstate, f, df, zp):
            retval = initial_cost(mode[0], nstate[0], f, df, zp)
            if retval < 0:
                mode[0] = retval
        initial_cost = _call_initial_cost.ctypes

    if trajectory_cost:
        # Define a wrapper around the function to change the arguments
        @numba.cfunc(numba_ntg_trajectory_cost_signature)
        def _call_trajectory_cost(mode, nstate, i, f, df, zp):
            retval = trajectory_cost(mode[0], nstate[0], i[0], f, df, zp)
            if retval < 0:
                mode[0] = retval
        trajectory_cost = _call_trajectory_cost.ctypes

    if final_cost:
        # Define a wrapper around the function to change the arguments
        @numba.cfunc(numba_ntg_endpoint_cost_signature)
        def _call_final_cost(mode, nstate, f, df, zp):
            retval = final_cost(mode[0], nstate[0], f, df, zp)
            if retval < 0:
                mode[0] = retval
        final_cost = _call_final_cost.ctypes

    #
    # Process constraints
    #
    # This section of the code parses the initial_constraints,
    # trajectory_constraints, and final_constraints keywords, which is the
    # preferred (and easier) way to specify constraints.  These constraints
    # are evetnually called via the NTG nlicf, nltcf, and nlfcf function
    # pointers within NTG.
    #

    # Figure out the dimension of the flat flag
    zflag_size = sum([sys.flaglen[i] for i in range(sys.nout)])

    # Initialize linear constraint matrices and bounds (if needed)
    lic = np.empty((0, zflag_size))
    ltc = np.empty((0, zflag_size))
    lfc = np.empty((0, zflag_size))
    lowerb = np.empty(0)
    upperb = np.empty(0)

    # Utility function to process linear constraints
    # TODO: look at simplifying since Amat and lower/upper start empty??
    def process_linear_constraints(
            constraint_list, Amat, lb, ub, name='unknown'):
        if constraint_list is None:
            # Nothing to do
            return Amat, lb, ub
        elif Amat.size > 0:
            # Can't use low-level and high-level interfaces at the same time
            raise TypeError(
                f"invalid mixture of {name} constraint types detected")

        # Turn constraint list into a list, if it isn't already
        if not isinstance(constraint_list, list):
            constraint_list = [constraint_list]

        # Go through each constraint & create low-level linear constraint
        for constraint in constraint_list:
            if isinstance(constraint, sp.optimize.LinearConstraint):
                # Set up a linear constraint
                Amat = np.vstack([Amat, constraint.A])
                lb = np.hstack([lb, constraint.lb])
                ub = np.hstack([ub, constraint.ub])

            elif isinstance(constraint, sp.optimize.NonlinearConstraint):
                # Nonlinear constraints are handled later
                pass

            else:
                raise ValueError(f"unregonized {name} constraint")
        return Amat, lb, ub

    # Process linear constraints (does nothing if low-level interface is used)
    lic, lowerb, upperb = process_linear_constraints(
        initial_constraints, lic, lowerb, upperb, name='initial')
    ltc, lowerb, upperb = process_linear_constraints(
        trajectory_constraints, ltc, lowerb, upperb, name='trajectory')
    lfc, lowerb, upperb = process_linear_constraints(
        final_constraints, lfc, lowerb, upperb, name='final')

    # Utility function to process nonlinear constraints
    # TODO: simplify based on the lack of lower-level functionality
    def process_nonlinear_constraints(
        constraint_list, lb, ub, type='endpoint', name='unknown'):
        if constraint_list is None:
            # Nothing to do
            return None, 0, lb, ub

        # Turn constraint list into a list, if it isn't already
        if not isinstance(constraint_list, list):
            constraint_list = [constraint_list]

        # Go through each constraint & create low-level nonlinear constraint
        funclist, nconstraints, offset = [], 0, []
        for constraint in constraint_list:
            if isinstance(constraint, sp.optimize.NonlinearConstraint):
                # Set up a nonlinear constraint
                funclist.append(constraint.fun)
                offset.append(nconstraints)

                # Convert upper and lower bounds to ndarray's
                constraint_lb = np.atleast_1d(constraint.lb)
                constraint_ub = np.atleast_1d(constraint.ub)
                if constraint_lb.ndim > 1 or constraint_ub.ndim > 1:
                    raise ValueError("upper/lower bounds must be 1D array-like")

                # Update the upper and lower bounds + constraint count
                lb = np.hstack([lb, constraint_lb])
                ub = np.hstack([ub, constraint_ub])
                nconstraints += constraint_lb.size

            elif isinstance(constraint, sp.optimize.LinearConstraint):
                # Linear constraints have already been handled
                pass

            else:
                raise ValueError(f"unregonized {name} constraint")

        # TODO: add support for multiple nonlinear constraints
        if len(funclist) > 1:
            raise ValueError(
                f"multiple nonlinear {name} constraints not yet supported")

        elif len(funclist) ==1 and type == 'endpoint':
            func = funclist[0]
            @numba.cfunc(numba_ntg_endpoint_constraint_signature)
            def _call_endpoint_constraint(mode, nstate, f, df, zp):
                retval = func(mode[0], nstate[0], f, df, zp)
                if retval < 0:
                    mode[0] = retval
            func = _call_endpoint_constraint.ctypes

        elif len(funclist) == 1 and type == 'trajectory':
            func = funclist[0]
            @numba.cfunc(numba_ntg_trajectory_constraint_signature)
            def _call_trajectory_constraint(mode, nstate, i, f, df, zp):
                retval = func(mode[0], nstate[0], i[0], f, df, zp)
                if retval < 0:
                    mode[0] = retval
            func = _call_trajectory_constraint.ctypes

        else:
            # No nonlinear constraints found
            func = None

        return func, nconstraints, lb, ub

    # Process nonlinear constraints
    nlicf, nlicf_num, lowerb, upperb = process_nonlinear_constraints(
        initial_constraints, lowerb, upperb, name='initial')
    nltcf, nltcf_num, lowerb, upperb = process_nonlinear_constraints(
        trajectory_constraints, lowerb, upperb, type='trajectory',
        name='trajectory')
    nlfcf, nlfcf_num, lowerb, upperb = process_nonlinear_constraints(
        final_constraints, lowerb, upperb, name='final')

    # Print the shapes of things if we need to know what is happening
    if verbose:
        print(f"lic.shape = {lic.shape}")
        print(f"ltc.shape = {ltc.shape}")
        print(f"lfc.shape = {lfc.shape}")
        print(f"lowerb.shape = {lowerb.shape}")
        print(f"upperb.shape = {upperb.shape}")

    # Call NTG
    systraj, cost, inform = call_ntg(
        sys.nout, timepts,
        flaglen=sys.flaglen, smoothness=basis.smoothness,
        order=[d + 1 for d in basis.degree],
        knotpoints=[basis.breakpoints for i in range(sys.nout)],
        nintervals=[basis.nintervals for i in range(sys.nout)],
        icf=initial_cost, icf_av=initial_cost_av,
        tcf=trajectory_cost, tcf_av=trajectory_cost_av,
        fcf=final_cost, fcf_av=final_cost_av,
        lic=lic, ltc=ltc, lfc=lfc, lowerb=lowerb, upperb=upperb,
        nlicf=nlicf, nlicf_num=nlicf_num, nlicf_av=initial_constraint_av,
        nltcf=nltcf, nltcf_num=nltcf_num, nltcf_av=trajectory_constraint_av,
        nlfcf=nlfcf, nlfcf_num=nlfcf_num, nlfcf_av=final_constraint_av,
        verbose=verbose)
    return OptimalControlResult(systraj, cost, inform)



def quadratic_cost(sys, Qlist, Z0=0, type='trajectory'):
    """Create quadratic cost function

    Returns a quadratic cost function that can be used for an optimal control
    problem.  The cost function is of the form

      cost = (zflag - Z0)^T diag(Qlist) (zflag - Z0)

    Parameters
    ----------
    sys : FlatSystem
        Flat system for which the cost function is being defined.
    Qlist : list of 2D array_like
        Weighting matrix for each flat output.  Dimensions much match
        the flag structure.
    Z0 : 1D array
        Nominal value of the system flag (for which cost should be zero).

    Returns
    -------
    cost_fun : NonlinearConstraint
        NonlinearConstraint object that can be used to evaluate the
        cost at a given output flag.

    """
    # Make sure list of weighting matrices has the right length
    if sys.nout != len(Qlist):
        raise ValueError("Length of Qlist must equal number of flat outputs")

    # Check (and/or convert) size of Z0
    if isinstance(Z0, (int, float)):
        # Convert Z0 to a list of arrays of the right shape
        Z0 = [np.ones(sys.flaglen[i]) * Z0 for i in range(sys.nout)]

    if sys.nout != len(Z0):
        raise ValueError("Length of Z0 must equal number of flat outputs")

    # Process the input arguments
    for i, length in enumerate(sys.flaglen):
        Qlist[i] = np.atleast_2d(Qlist[i]).astype(float)
        if Qlist[i].size == 1:      # allow scalar weights
            Qlist[i] = np.eye(length) * Qlist[i].item()
        elif Qlist[i].shape != (sys.flaglen[i], sys.flaglen[i]):
            raise ValueError(f"Qlist[{i}] matrix is the wrong shape")

        Z0[i] = np.atleast_1d(Z0[i])
        if Z0[i].shape != (sys.flaglen[i], ):
            raise ValueError(f"Z0[{i}] is the wrong shape")

    # Create variables to numba can work with
    nout = sys.nout
    flaglen = tuple(sys.flaglen)
    Qlist = tuple(Qlist)
    Z0 = tuple(Z0)

    if type == 'trajectory':
        # Create a trajectory constraint
        @numba.cfunc(numba_trajectory_cost_signature)
        def _quadratic_cost(mode, nstate, j, f, df, zp):
            if mode == 0 or mode == 2:
                cost = 0.
                for i in range(nout):
                    zflag_i = numba.carray(zp[i], flaglen[i])
                    cost += (zflag_i - Z0[i]) @ Qlist[i] @ (zflag_i - Z0[i])
                f[0] = cost

            if mode == 1 or mode == 2:
                offset = 0
                for i in range(nout):
                    zflag_i = numba.carray(zp[i], flaglen[i])
                    grad = 2 * Qlist[i] @ (zflag_i - Z0[i])
                    for j in range(flaglen[i]):
                        df[offset] = grad[j]
                        offset += 1
            return 0

    elif type == 'endpoint':
        # Create an endpoint constraint
        @numba.cfunc(numba_endpoint_cost_signature)
        def _quadratic_cost(mode, nstate, f, df, zp):
            if mode == 0 or mode == 2:
                cost = 0.
                for i in range(nout):
                    zflag_i = numba.carray(zp[i], flaglen[i])
                    cost += (zflag_i - Z0[i]) @ Qlist[i] @ (zflag_i - Z0[i])
                f[0] = cost

            if mode == 1 or mode == 2:
                offset = 0
                for i in range(nout):
                    zflag_i = numba.carray(zp[i], flaglen[i])
                    grad = 2 * Qlist[i] @ (zflag_i - Z0[i])
                    for j in range(flaglen[i]):
                        df[offset] = grad[j]
                        offset += 1
            return 0

    else:
        raise ValueError(f"unknown type '{type}'")

    return _quadratic_cost.ctypes


def flag_equality_constraint(sys, zflag):
    zflag = np.hstack([zi for zi in zflag])
    zlen = sum([sys.flaglen[i] for i in range(sys.nout)])
    constraint_matrix = np.eye(zlen)    # constrain all values in flag

    return sp.optimize.LinearConstraint(constraint_matrix, zflag, zflag)

#
# Numba signatures
#
# The following signatures can be used to create numba functions for costs
# and constraints.  Note that functions evaluated along the trajectory have
# an extra argument (i = time point number).
#

# Cost function and constraint signatures
from numba import types
numba_trajectory_cost_signature = types.intc(
    types.intc,                       # int mode
    types.intc,                       # int nstate
    types.intc,                       # int i
    types.CPointer(types.double),     # double *f
    types.CPointer(types.double),     # double *df
    types.CPointer(                   # double **zp
        types.CPointer(types.double)))

numba_endpoint_cost_signature = types.intc(
    types.intc,                       # int mode
    types.intc,                       # int nstate
    types.CPointer(types.double),     # double *f
    types.CPointer(types.double),     # double *df
    types.CPointer(                   # double **zp
        types.CPointer(types.double)))

numba_trajectory_constraint_signature = types.intc(
    types.intc,                       # int mode
    types.intc,                       # int nstate
    types.intc,                       # int i
    types.CPointer(types.double),     # double *f
    types.CPointer(                   # double **df
        types.CPointer(types.double)),
    types.CPointer(                   # double **zp
        types.CPointer(types.double)))

numba_endpoint_constraint_signature = types.intc(
    types.intc,                       # int mode
    types.intc,                       # int nstate
    types.CPointer(types.double),     # double *f
    types.CPointer(                   # double **df
        types.CPointer(types.double)),
    types.CPointer(                   # double **zp
        types.CPointer(types.double)))

