# ntg.pyx - Cython interface to NTG
# RMM, 15 Jul 2022
#
# This module provides a Cython interface to the NTG library.  The following
# functions and classes are provided:
#
#   actvar() - define active variables
#   npsol_option() - set NPSOL options
#   ntg() - main function to call NTG
#   spline_interp() - spline interpolation
#   print_banner() - print NTG banner
#

cimport numpy as np
import numpy as np
import scipy as sp
import scipy.optimize
import ctypes
from warnings import warn
from libc.stdlib cimport malloc, calloc, free
cimport ntg as ntg


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


class BSplineFamily:
    """B-spline basis functions.

    This class represents a vector-valued set of B-spline polynomials.
    B-splines are characterized by a set of intervals separated by knot
    points.  On each interval we have a polynomial of a certain order and
    the spline is continuous up to a given number of derivatives at the knot
    points.

    Attributes
    ----------
    nvars : int
        Dimension of the space in with the B-splines take their values.

    knotpoints : 1D array
        For each B-spline, the knot points for the B-spline.

    order : list of ints
        For each B-spline, the order of the Bezier polynomial.

    smoothness : list of ints
        For each B-spline, the number of derivatives of smoothness at the
        knot points.

    nintervals : list of ints
        For each B-spline, the number of intervals (= len(knotpoints) + 1)

    """
    def __init__(
        self, nvars, knotpoints, order, smoothness):
        """Define a set of B-spline polyomials

        Define B-spline polynomials for a set of variables.  B-splines are
        characterized by a set of intervals separated by knot points.  On
        each interval we have a polynomial of a certain order and the spline
        is continuous up to a given smoothness at interior knot points.

        Parameters
        ----------
        nvars : int
            Number of B-splines variables (output dimension).

        knotpoints : 1D array of float
            For each B-spline variable, the knot points for the B-spline.

        order : list of ints
            For each B-spline variable, the order of the Bezier polynomial.

        smoothness : list of ints
            For each B-spline variable, the smoothness at knot points.

        """
        #
        # Process knot points
        #
        # The knot points for the B-spline.  NTG allows these to be
        # different for each output variable, but here we assume they are
        # the same for all outputs (the most common case).
        #

        knotpoints = np.array(knotpoints, dtype=float)
        if knotpoints.ndim != 1:
            raise ValueError("knot points must be convertable to a 1D array")
        elif knotpoints.size < 2:
            raise ValueError("knot point vector must have at least 2 values")
        elif np.any(np.diff(knotpoints) <= 0):
            raise ValueError("knot points must strictly increasing values")

        #
        # Process B-spline parameters (order, smoothness)
        #
        # B-splines are characterized by a set of intervals separated by
        # knot points.  On each interval we have a polynomial of a certain
        # order and the spline is continuous up to a given smoothness at
        # knot points.  The code in this section allows some flexibility in
        # the way that all of this information is supplied, including using
        # scalar values for parameters (which are then broadcast to each
        # output) and inferring values and dimensions from other
        # information, when possible.
        #

        # Utility function for broadcasting spline parameters (order,
        # ninterv, mult)
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
                    raise ValueError(f"length of '{name}' does not match n")
            else:
                raise ValueError(f"could not parse '{name}' keyword")

            # Check to make sure the values are OK
            if values is not None and any([val < minimum for val in values]):
                raise ValueError(
                    f"invalid value for {name}; must be at least {minimum}")

            return values

        # Order of polynomial; set default to maximum number of derivativs
        order = process_spline_parameters(
            order, nvars, (int), name='order', minimum=1)

        # Smoothness at knotpoints; set default to maximum number of derivs
        smoothness = process_spline_parameters(
            smoothness, nvars, (int), name='smoothness', minimum=1)

        # Store the parameters and process them in call_ntg()
        self.nvars = nvars
        self.knotpoints = knotpoints
        self.order = order
        self.smoothness = smoothness
        self.nintervals = knotpoints.size - 1


class SystemTrajectory:
    """Class representing a system trajectory.

    The `SystemTrajectory` class is used to represent the trajectory of a
    (differentially flat) system.  Used by the :func:`~ntg.ntg` function to
    return a trajectory.

    Attributes
    ----------
    coefs : 1D array
        Flat output coefficient list, represented as a stacked array.

    nout : integer
        Number of independent flat output.

    flaglen : list of ints
        For each flat output, the number of derivatives of the flat
        output used to define the trajectory.

    knotpoints : list of 1D arrays
        For each flat output, the knot points for the B-spline.

    order : list of ints
        For each flat output, the order of the Bezier polynomial.

    smoothness : list of ints
        For each flat output, the smoothness at the knot points.

    """
    # TODO: update to accept basis as input
    def __init__(self, coefs, nout, flaglen, knotpoints, order, smoothness):
        """Initilize a system trajectory object."""
        # Save the elements
        self.coefs = coefs
        self.nout = nout
        self.flaglen = flaglen
        self.knotpoints = knotpoints
        self.order = order
        self.smoothness = smoothness

        # Keep track of the maximum length of a flag
        self.maxflag = max(flaglen)

        # Store the coefficients for each output (useful later)
        self.coef_offset, self.coef_length, offset = [], [], 0
        for i in range(nout):
            ncoefs = (len(knotpoints[i]) - 1) * \
                (order[i] - smoothness[i]) + smoothness[i]
            self.coef_offset.append(offset)
            self.coef_length.append(ncoefs)
            offset += ncoefs

    # Evaluate the trajectory over a list of time points
    def eval(self, tlist):
        """Evalulate the flat flag for a trajectory at a list of times

        Evaluate the trajectory at a list of time points, returning the state
        and input vectors for the trajectory:

            zflag = traj.eval(tlist)

        Parameters
        ----------
        tlist : 1D array
            List of times to evaluate the trajectory.

        Returns
        -------
        zflag : 3D array
            For each output, the value of the flat flag at the given times.
            The indices are output, derivative, and time index.

        """
        # Go through each time point and compute flat variables
        # TODO: make this more pythonic
        zflag = np.empty((self.nout, self.maxflag, len(tlist)))
        zflag.fill(np.nan)
        for i in range(self.nout):
            nintervals = self.knotpoints[i].size - 1
            zflag[i, :self.flaglen[i]] = np.array([spline_interp(
                t, self.knotpoints[i], nintervals,
                self.coefs[self.coef_offset[i]:
                           self.coef_offset[i] + self.coef_length[i]],
                self.order[i], self.smoothness[i], self.flaglen[i])
                                 for t in tlist]).transpose()

        return zflag

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
    breakpoints,           # break points
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

    breakpoints : 1D array-like
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
    if sys.nout != basis.nvars:
        raise ValueError(
            f"system size ({sys.nout}) and basis size "
            f"({basis.nvars}) must match")

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
    # Break points
    #
    # Process the break points next since this will tell us what the time
    # points are that we need to define things over.
    #
    breakpoints = np.atleast_1d(breakpoints)
    if breakpoints.ndim != 1:
        raise ValueError("breakpoints must be a 1D array")
    elif np.any(np.diff(breakpoints) <= 0):
        raise ValueError("breakpoints must be strictly increasing")

    #
    # Process constraints
    #
    # There are currently two (incompatible) ways to specify constraints:
    # directly, using lic, ltc, and lfc for linear constraints and nlicf,
    # nltcf, and nlfcf for nonlinear constraints (with lowerb and upperb set
    # appropriately) or via the initial_constraints, trajectory_constraints,
    # and final_constraints keywords, which use scipy.optimal's
    # LinearConstraints and NonlinearConstraints classes.
    #
    # This section of the code parses the initial_constraints,
    # trajectory_constraints, and final_constraints keywords, which is the
    # preferred (and easier) way to specify constraints.  If constraints
    # specified using nlicf, nltcf, and nlfcf instead, these are passed
    # through unchanged and converted directly to C data structures below.
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
            constraint_list, func, nconstraints, lb, ub, name='unknown'):
        if constraint_list is None:
            # Nothing to do
            return func, nconstraints, lb, ub
        elif func is not None or nconstraints is not None:
            # Can't use low-level and high-level interfaces at the same time
            raise TypeError(
                f"invalid mixture of {name} constraint types detected")

        # Turn constraint list into a list, if it isn't already
        if not isinstance(constraint_list, list):
            constraint_list = [constraint_list]

        # Go through each constraint & create low-level linear constraint
        funclist, nconstraints = [], 0
        for constraint in constraint_list:
            if isinstance(constraint, sp.optimize.NonlinearConstraint):
                # Set up a nonlinear constraint
                funclist.append(constraint.fun)

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
        elif len(funclist) == 1:
            # Return the single nonlinear constraint function
            func = funclist[0]
        else:
            # No nonlinear constraints found
            func = None

        return func, nconstraints, lb, ub

    # Process nonlinear constraints
    nlicf, nlicf_num, lowerb, upperb = process_nonlinear_constraints(
        initial_constraints, None, None, lowerb, upperb, name='initial')
    nltcf, nltcf_num, lowerb, upperb = process_nonlinear_constraints(
        trajectory_constraints, None, None, lowerb, upperb,
        name='trajectory')
    nlfcf, nlfcf_num, lowerb, upperb = process_nonlinear_constraints(
        final_constraints, None, None, lowerb, upperb, name='final')

    # Print the shapes of things if we need to know what is happening
    if verbose:
        print(f"lic.shape = {lic.shape}")
        print(f"ltc.shape = {ltc.shape}")
        print(f"lfc.shape = {lfc.shape}")
        print(f"lowerb.shape = {lowerb.shape}")
        print(f"upperb.shape = {upperb.shape}")

    # Call NTG
    systraj, cost, inform = call_ntg(
        sys.nout, breakpoints,
        flaglen=sys.flaglen, smoothness=basis.smoothness, order=basis.order,
        knotpoints=[basis.knotpoints for i in range(sys.nout)],
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


#
# Main ntg() function
#
# For now this function looks more or less like the C version of ntg(), but
# with Python objects as arguments (so we don't have to separately pass
# arguments) and with keyword arguments for anything that is optional.
#
def call_ntg(
    nout,                       # number of outputs
    breakpoints,                # break points
    nintervals,                 # number of intervals
    order,                      # order of polynomial (for each output)
    smoothness,                 # smoothness at knot points (for each output)
    flaglen,                    # max number of derivs + 1 (for each output)
    knotpoints=None,            # knot points for each output
    icf=None, icf_av=None,      # initial cost function, active vars
    tcf=None, tcf_av=None,      # trajectory cost function, active vars
    fcf=None, fcf_av=None,      # final cost function, active vars
    initial_guess=None,         # initial guess for coefficients
    lic=None, ltc=None, lfc=None,       # linear init, traj, final constraints
    nlicf=None, nlicf_num=None, nlicf_av=None, # NL initial constraints
    nltcf=None, nltcf_num=None, nltcf_av=None, # NL trajectory constraints
    nlfcf=None, nlfcf_num=None, nlfcf_av=None, # NL final constraints
    lowerb=None, upperb=None,   # upper and lower bounds for constraints
    verbose=False,              # turn on verbose messages
    **kwargs                    # additional arguments
):
    """Low-level interface to NTG

    Compute an optimal trajectory for a differentially flat system with
    initial, trajectory, and final costs and constraints.

    Parameters
    ----------
    nout : int
        Number of flat outputs

    breakpoints : 1D array-like
        Array of times at which constraints should be evaluated and integrated
        costs are computed.

    Returns
    -------
    systraj : :class:`~ntg.SystemTrajectory` object
        The system trajectory is returned as an object that implements the
        `eval()` function, we can be used to compute the value of the state
        and input and a given time t.

    """
    #
    # Process parameters, just in case someone tried to slip something by
    #

    if knotpoints is None:
        knotpoints = [np.linspace(0, breakpoints[-1], nintervals[i] + 1)
                      for i in range(nout)]

    #
    # Create the C data structures needed for ntg()
    #
    # Finally we create all of the C data structures and functions required
    # to all ntg().  All error checking should be done prior to this point,
    # we we assume everything here makes sense.
    #

    # Utility functions to check dimensions and set up C arrays
    def init_c_array_1d(array, size, type):
        array = np.atleast_1d(array)
        assert array.size == size
        return array.astype(type)

    # Set up spline dimensions
    cdef int [:] c_ninterv = init_c_array_1d(nintervals, nout, np.intc)
    cdef int [:] c_order = init_c_array_1d(order, nout, np.intc)
    cdef int [:] c_mult = init_c_array_1d(smoothness, nout, np.intc)
    cdef int [:] c_flaglen = init_c_array_1d(flaglen, nout, np.intc)

    # Breakpoints
    cdef int nbps = breakpoints.shape[0]
    cdef double [:] c_bps = init_c_array_1d(breakpoints, nbps, np.double)

    cdef double **c_knots = <double **> malloc(nout * sizeof(double *))
    for out in range(nout):
        assert len(knotpoints[out]) == nintervals[out] + 1
        c_knots[out] = <double *> malloc(len(knotpoints[out]) * sizeof(double))
        for time in range(len(knotpoints[out])):
            c_knots[out][time] = knotpoints[out][time]

    # Figure out the number of coefficients
    ncoef = 0
    for i in range(nout):
        ncoef += nintervals[i] * (order[i] - smoothness[i]) + smoothness[i]

    # Coefficients: initial guess + return result
    if initial_guess is not None:
        coefs = np.atleast_1d(initial_guess)
        assert coefs.ndim == 1
        assert coefs.size == ncoef
    else:
        coefs = np.ones(ncoef)
    cdef double [:] c_coefs = init_c_array_1d(coefs, ncoef, np.double)

    # Process linear constraints
    nlic, c_lic = _convert_linear_constraint(lic, name='initial', verbose=verbose)
    nltc, c_ltc = _convert_linear_constraint(
        ltc, name='trajectory', verbose=verbose)
    nlfc, c_lfc = _convert_linear_constraint(lfc, name='final', verbose=verbose)

    #
    # Callback functions
    #
    # Functions defining costs and constraints should be passed as ctypes
    # functions (so that they can be loaded dynamically).  The pointer to
    # this callback function is passed to NTG directly, after some
    # typcasting to keep Cython happy.
    #

    #
    # Constraint callbacks
    #
    # Nonlinear initial condition constraints
    nnlic, nlic_addr, ninitialconstrav, c_initialconstrav = \
        _parse_callback(nlicf, nlicf_av, nout, c_flaglen, num=nlicf_num,
                        name='initial constraint')
    c_nlic = (<ntg_vector_cbf *> nlic_addr)[0]

    # Nonlinear trajectory constraints
    nnltc, nltc_addr, ntrajectoryconstrav, c_trajectoryconstrav = \
        _parse_callback(nltcf, nltcf_av, nout, c_flaglen, num=nltcf_num,
                        name='trajectory_constraint')
    c_nltc = (<ntg_vector_traj_cbf *> nltc_addr)[0]

    nnlfc, nlfc_addr, nfinalconstrav, c_finalconstrav = \
        _parse_callback(nlfcf, nlfcf_av, nout, c_flaglen, num=nlfcf_num,
                        name='final constraint')
    c_nlfc = (<ntg_vector_cbf *> nlfc_addr)[0]

    # Bounds on the constraints
    nil = np.array([0.])        # Use as "empty" constraint matrix
    cdef double [:] c_lowerb = nil if lowerb is None or lowerb.size == 0 else \
        lowerb.astype(np.double)
    cdef double [:] c_upperb = nil if upperb is None or upperb.size == 0 else \
        upperb.astype(np.double)
    if verbose:
        print("  lower bounds = ", lowerb)
        print("  upper bounds = ", upperb)

    # Check to make sure dimensions are consistent
    if nlic + nltc + nlfc + nnlic + nnltc + nnlfc > 0:
        assert lowerb.size == nlic + nltc + nlfc + nnlic + nnltc + nnlfc
        assert upperb.size == nlic + nltc + nlfc + nnlic + nnltc + nnlfc

    #
    # Cost function callbacks
    #
    nicf, icf_addr, ninitialcostav, c_initialcostav = \
        _parse_callback(icf, icf_av, nout, c_flaglen, num=1, name='initial cost')
    c_icf = (<ntg_scalar_cbf *> icf_addr)[0]

    ntcf, tcf_addr, ntrajectorycostav, c_trajectorycostav = \
        _parse_callback(tcf, tcf_av, nout, c_flaglen, num=1,
                        name='trajectory cost')
    c_tcf = (<ntg_scalar_traj_cbf *> tcf_addr)[0]

    nfcf, fcf_addr, nfinalcostav, c_finalcostav = \
        _parse_callback(fcf, fcf_av, nout, c_flaglen, num=1, name='final cost')
    c_fcf = (<ntg_scalar_cbf *> fcf_addr)[0]

    #
    # NTG internal memory
    #
    cdef int *istate = <int *> calloc(
        ncoef + nlic + nltc * nbps + nlfc +
        nnlic + nnltc * nbps + nnlfc, sizeof(int))
    cdef double *clambda = <double *> calloc(
        ncoef + nlic + nltc * nbps + nlfc +
        nnlic + nnltc * nbps + nnlfc, sizeof(double))
    cdef double *R = <double *> calloc((ncoef+1)**2, sizeof(double))

    if verbose:
        print(f"Initialization finished; ncoef={ncoef}")

    # Return values
    cdef int inform
    cdef double objective

    # Call the NTG function to compute the trajectory (finally!)
    ntg.c_ntg(
        nout, &c_bps[0], nbps, &c_ninterv[0], &c_knots[0],
        &c_order[0], &c_mult[0], &c_flaglen[0], &c_coefs[0],
        nlic,                c_lic,
        nltc,                c_ltc,
        nlfc,                c_lfc,
        nnlic,               c_nlic,
        nnltc,               c_nltc,
        nnlfc,               c_nlfc,
        ninitialconstrav,    c_initialconstrav,
        ntrajectoryconstrav, c_trajectoryconstrav,
        nfinalconstrav,      c_finalconstrav,
        &c_lowerb[0], &c_upperb[0],
        nicf,		     c_icf,
        ntcf,                c_tcf,
        nfcf,		     c_fcf,
        ninitialcostav,      c_initialcostav,
        ntrajectorycostav,   c_trajectorycostav,
        nfinalcostav,	     c_finalcostav,
        istate, clambda, R, &inform, &objective);

    if verbose:
        print(f"NTG returned inform={inform}, objective={objective}")

    # Copy the coefficients back into our NumPy array
    cdef int k
    for k in range(coefs.size):
        coefs[k] = c_coefs[k]

    # Create a system trajectory object to store the result
    # TODO: move this functionality to solve_flat_ocp
    systraj = SystemTrajectory(
        coefs, nout, flaglen, knotpoints, order, smoothness)

    return systraj, objective, inform

# Evaluate the value of a spline at a given point `x`
def spline_interp(x, knots, ninterv, coefs, order, mult, flaglen):
    cdef double [:] c_knots = knots
    cdef double [:] c_coefs = coefs
    cdef int ncoefs = len(coefs)
    cdef np.ndarray[double, ndim=1, mode='c'] fz = np.empty(flaglen)
    ntg.SplineInterp(
        &fz[0], x, &c_knots[0], ninterv, &c_coefs[0], len(coefs),
        order, mult, flaglen)

    # Store results in an ndarray and free up memory
    return fz

#
# Cython utility functions
#

# Cython function to parse linear constraints
cdef (int, double **) _convert_linear_constraint(
    cmatrix, name='unknown', verbose=False):
    cdef int nlic = 0
    cdef double **c_lic = NULL

    if cmatrix is not None:
        cmatrix = np.atleast_2d(cmatrix)
        assert cmatrix.ndim == 2
        nlic = cmatrix.shape[0]
        c_lic = <double **> malloc(nlic * sizeof(double *))
        c_lic[0] = <double *> malloc(nlic * cmatrix.shape[1] * sizeof(double))
        for i in range(nlic):
            c_lic[i] = &c_lic[0][i * cmatrix.shape[1]]
            for j in range(cmatrix.shape[1]):
                c_lic[i][j] = cmatrix[i, j]

        if verbose:
            print(f"  {nlic} {name} constraints of size {cmatrix.shape[1]}")
            print("  cmatrix = ", cmatrix)

    return nlic, c_lic

# Dummy function to use as default callback (noop)
cdef void noop() nogil:
    pass

# Cython function to parse callbacks
cdef (int, size_t, int, AV *) _parse_callback(
        ctypes_fcn, avlist, nout, flaglen, num=None, name='unknown'):
    if ctypes_fcn is None:
        if avlist is not None:
            raise ValueError(
                f"active variables specified for {name} callback, "
                f"but no callback function given")
        # nfcn, fcn, nav, avlist
        return 0, <size_t> noop, 0, NULL

    cdef size_t c_fcn
    if num is None:
        # If the number of functions was not given, assume scalar and warn
        warn(f"Number of {name} not given; assuming scalar")
        nfcn = 1
        c_fcn = <size_t> ctypes.addressof(ctypes_fcn)
    else:
        nfcn = num
        c_fcn = <size_t> ctypes.addressof(ctypes_fcn)

    # Figure out the active variables to send back
    if avlist is not None:
        nav = len(avlist)
        c_av = <AV *> calloc(nav, sizeof(AV))
        for i, av in enumerate(avlist):
            c_av[i].output = av.output
            c_av[i].deriv = av.deriv
    else:
        # Figure out how many entries we need
        nav = 0
        for i in range(nout):
            nav += flaglen[i]

        # Make all variables active
        c_av = <AV *> calloc(nav, sizeof(AV))
        k = 0
        for i in range(nout):
            for j in range(flaglen[i]):
                c_av[k].output = i
                c_av[k].deriv = j
                k = k + 1

    return <int> nfcn, c_fcn, <int> nav, c_av

#
# Numba signatures
#
# The following signatures can be used to create numba functions for costs
# and constraints.  Note that functions evaluated along the trajectory have
# an extra argument (i = breakpoint number).
#

# Cost function and constraint signatures
from numba import types
numba_trajectory_cost_signature = types.void(
    types.CPointer(types.intc),       # int *mode
    types.CPointer(types.intc),       # int *nstate
    types.CPointer(types.intc),       # int *i
    types.CPointer(types.double),     # double *f
    types.CPointer(types.double),     # double *df
    types.CPointer(                   # double **zp
        types.CPointer(types.double)))

numba_endpoint_cost_signature = types.void(
    types.CPointer(types.intc),       # int *mode
    types.CPointer(types.intc),       # int *nstate
    types.CPointer(types.double),     # double *f
    types.CPointer(types.double),     # double *df
    types.CPointer(                   # double **zp
        types.CPointer(types.double)))

numba_trajectory_constraint_signature = types.void(
    types.CPointer(types.intc),       # int *mode
    types.CPointer(types.intc),       # int *nstate
    types.CPointer(types.intc),       # int *i
    types.CPointer(types.double),     # double *f
    types.CPointer(                   # double **df
        types.CPointer(types.double)),
    types.CPointer(                   # double **zp
        types.CPointer(types.double)))

numba_endpoint_constraint_signature = types.void(
    types.CPointer(types.intc),       # int *mode
    types.CPointer(types.intc),       # int *nstate
    types.CPointer(types.double),     # double *f
    types.CPointer(                   # double **df
        types.CPointer(types.double)),
    types.CPointer(                   # double **zp
        types.CPointer(types.double)))
