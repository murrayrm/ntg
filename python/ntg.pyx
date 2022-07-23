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

# Define a class to define active variables
class actvar(object):
    def __init__(self, output, deriv):
        self.output = output
        self.deriv = deriv

# Print the NTG banner
def print_banner():
    ntg.printNTGBanner()

# Define NPSOL options
def npsol_option(s):
    b = s.encode('utf-8')
    cdef char *c_string = b
    ntg.npsoloption(c_string)

#
# Main ntg() function
#
# For now this function looks more or less like the C version of ntg(), but
# with Python objects as arguments (so we don't have to separately pass
# arguments) and with keyword arguments for anything that is optional.
#
def ntg(
    nout,                       # number of outputs
    breakpoints,                # break points
    nintervals=None,            # number of intervals
    order=None,                 # order of polynomial (for each output)
    multiplicity=None,          # multiplicity at knot points (for each output)
    flaglen=None,               # max number of derivatives + 1 (for each output)
    knotpoints=None,            # knot points
    icf=None, icf_av=None,      # initial cost function, active vars
    tcf=None, tcf_av=None,      # trajectory cost function, active vars
    fcf=None, fcf_av=None,      # final cost function, active vars
    initial_guess=None,         # initial guess for coefficients
    initial_constraints=None,    # initial constraints (scipy.optimize form)
    trajectory_constraints=None, # trajectroy constraints (scipy.optimize form)
    final_constraints=None,      # initial constraints (scipy.optimize form)
    lic=None, ltc=None, lfc=None,       # linear init, traj, final constraints
    nlicf=None, nlicf_num=None, nlicf_av=None, # NL initial constraints
    nltcf=None, nltcf_num=None, nltcf_av=None, # NL trajectory constraints
    nlfcf=None, nlfcf_num=None, nlfcf_av=None, # NL final constraints
    lowerb=None, upperb=None,   # upper and lower bounds for constraints
    verbose=False,              # turn on verbose messages
    **kwargs                    # additional arguments
):
    # Process keywords
    def process_alt_kwargs(kwargs, primary, other, name):
        for kw in other:
            if kw in kwargs:
                if primary is not None:
                    raise TypeError(f"redundant keywords: {primary}, {kw}")
                primary = kwargs.pop(kw)
                name = kw
        return primary

    # Alternative versions of keywords (for python-control compatibility)
    icf = process_alt_kwargs(kwargs, icf, ['initial_cost'], 'icf')
    icf_av = process_alt_kwargs(
        kwargs, icf_av, ['initial_cost_actvars'], 'icf_av')
    tcf = process_alt_kwargs(kwargs, tcf, ['cost', 'trajectory_cost'], 'tcf')
    tcf_av = process_alt_kwargs(
        kwargs, tcf_av, ['cost_actvars', 'trajectory_cost_actvars'], 'tcf_av')
    fcf = process_alt_kwargs(kwargs, fcf, ['final_cost', 'terminal_cost'], 'fcf')
    fcf_av = process_alt_kwargs(
        kwargs, fcf_av,
        ['final_cost_actvars', 'terminal_cost_actvars'], 'fcf_av')

    # Make sure there were no additional keyword arguments
    if kwargs:
        raise TypeError("unrecognized keyword arguments", kwargs)

    #
    # Process B-spline parameters
    #
    # B-splines are characterized by a set of intervals separated by knot
    # points.  One each interval we have a polynomial of a certain order and
    # the spline is continuous up to a given multiplicity at interior knot
    # points.  The code in this section allows some flexibility in the way
    # that all of this information is supplied, including using scalar
    # values for parameters (which are then broadcast to each output) and
    # inferring values and dimensions from other information, when possible.
    #

    # Utility function for broadcasting spline parameters (order, flaglen,
    # ninterv, mult)
    def process_spline_parameters(
            values, nout, allowed_types, minimum=0, default=None, name='unknown'):
        # Preprosing
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
            values = [values for i in range(nout)]
        elif all([isinstance(v, allowed_types) for v in values]):
            # List of values => make sure it is the right size
            if len(values) != nout:
                raise ValueError(f"length of '{name}' does not match nout")
        else:
            raise ValueError(f"could not parse '{name}' keyword")

        # Check to make sure the values are OK
        if values is not None and any([val < minimum for val in values]):
            raise ValueError(
                f"invalid value for {name}; must be at least {minimum}")

        if verbose:
            print(f"{name} = {values}")
        return values

    #
    # Number of intervals
    #
    # We start by processing the number of intervals, if present, to make
    # sure these are ready for processing of knot points (next).
    #
    nintervals = process_spline_parameters(
        nintervals, nout, (int), name='nintervals', minimum=1)

    #
    # Break points
    #
    # Process the breakpoints next since this will tell us what the time
    # points are that we need to define things over.
    #
    breakpoints = np.atleast_1d(breakpoints)
    if breakpoints.ndim != 1:
        raise ValueError("breakpoints must be a 1D array")

    #
    # Knot points
    #
    # If given, the knot points specify the points in time that separate the
    # intervals of the B-spline.  They can be the same for each output or
    # different for each output, and there can be a different number of
    # intervals for each output.  If the knotpoints are not given, then they
    # will be inferred from the number of intervals (nintervals) and equally
    # spaced.  If the knotpoints are given, this is used to determine the
    # number of intervals (w/ no error if a consistent nintervals argument
    # is also supplied).
    #
    if knotpoints:
        # Convert the argument to a list, to allow for output-specific intervals
        if isinstance(knotpoints, np.ndarray):
            knotpoints = knotpoints.tolist()
        else:
            # Convert to a list (eg, versus a tuple)
            knotpoints = list(knotpoints)

        # Figure out if the list is 1D or 2D (and convert to 2D)
        if all([isinstance(pt, (int, long, float)) for pt in knotpoints]):
            # 1D list of knotpoints => convert to 2D
            knotpoints = [knotpoints for i in range(nout)]
        elif not all([isinstance(pt, list) for pt in knotpoints]):
            raise ValueError(
                "can't parse knot points (should be 1D or 2D array/list)")

        # Make sure the list makes sense
        if len(knotpoints) != nout:
            raise ValueError("number of knot point vectors must equal nout")

        print("knot points = ", knotpoints)
        print("nintervals = ", nintervals)

        # Process knot points for each output (and convert to 1D ndarrays)
        for i, knot in enumerate(knotpoints):
            knot = np.atleast_1d(knot)
            if knot.ndim > 1:
                raise ValueError("knot points should be 1D array for each output")
            if nintervals and knot.size != nintervals[i] + 1:
                raise ValueError(
                    f"for output {i}, number of knot points ({knot.size}) and "
                    f"nintervals ({nintervals[i]}) are inconsistent")
            knotpoints[i] = knot

        # Set up nintervals to match knot points
        if nintervals is None:
            nintervals = [knot.size -1 for knot in knotpoints]

    else:
        # If nintervals was not specified, use one interval per output
        if nintervals is None:
            nintervals = np.ones(nout, dtype=int).tolist()

        # Set up equally space knot points for reach output
        knotpoints = [np.linspace(0, breakpoints[-1], nintervals[out] + 1)
                 for out in range(nout)]

    # Make sure break point values make sense
    if any([knot[0] > breakpoints[0] or knot[-1] < breakpoints[-1]
            for knot in knotpoints]):
        raise ValueError(
            "initial and final knot points must be outside of break point range")

    # Maximum number of derivatives for each flat output
    flaglen = process_spline_parameters(
        flaglen, nout, (int), name='flaglen', minimum=0)
    if flaglen is None:
        raise ValueError("missing value(s) for flaglen")

    # Order of polynomial; set default to maximum number of derivativs
    order = process_spline_parameters(
        order, nout, (int), name='order', minimum=1,
        default=[derivs + 1 for derivs in flaglen])

    # Multiplicity at knotpoints; set default to maximum number of derives
    multiplicity = process_spline_parameters(
        multiplicity, nout, (int), name='multiplicity',
        minimum=1, default=flaglen)

    #
    # Process constraints
    #
    # There are currently two ways to specify constraints: directly, using
    # lic, ltc, and lfc for linear constraints and nlicf, nltcf, and nlfcf
    # for nonlinear constraints (with lowerb and upperb set appropriately)
    # or via the initial_constraints, trajectory_constraints, and
    # final_constraints keywords, which use scipy.optimal's
    # LinearConstraints and NonlinearConstraints classes.
    #
    # These two methods are currently incompatible.
    #
    # This section of the code parses the initial_constraints,
    # trajectory_constraints, and final_constraints keywords, which is the
    # preferred (and easier) way to specify constraints.
    #

    # Make sure we aren't mixing up constraint types
    if lowerb is not None or upperb is not None:
        for ctype, name in zip(
                [initial_constraints, trajectory_constraints, final_constraints],
                ['initial', 'trajectory', 'final']):
            if ctype is not None:
                raise TypeError(
                    f"invalid mixture of {name} constraint types detected")

    # Figure out the dimension of the flat flag
    zflag_size = sum([flaglen[i] for i in range(nout)])

    # Initialize linear constraint matrices and bounds (if needed)
    # (if low-level interface is used; these just convert to ndarrays)
    lic = np.empty((0, zflag_size)) if lic is None else np.atleast_2d(lic)
    ltc = np.empty((0, zflag_size)) if ltc is None else np.atleast_2d(ltc)
    lfc = np.empty((0, zflag_size)) if lfc is None else np.atleast_2d(lfc)
    lowerb = np.empty(0) if lowerb is None else np.atleast_1d(lowerb)
    upperb = np.empty(0) if upperb is None else np.atleast_1d(upperb)

    # Utility function to process linear constraints
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

    # Process nonlinear constraints (does nothing if low-level interface is used)
    nlicf, nlicf_num, lowerb, upperb = process_nonlinear_constraints(
        initial_constraints, nlicf, nlicf_num, lowerb, upperb, name='initial')
    nltcf, nltcf_num, lowerb, upperb = process_nonlinear_constraints(
        trajectory_constraints, nltcf, nltcf_num, lowerb, upperb,
        name='trajectory')
    nlfcf, nlfcf_num, lowerb, upperb = process_nonlinear_constraints(
        final_constraints, nlfcf, nlfcf_num, lowerb, upperb, name='final')

    # Print the shapes of things if we need to know what is happening
    if verbose:
        print(f"lic.shape = {lic.shape}")
        print(f"ltc.shape = {ltc.shape}")
        print(f"lfc.shape = {lfc.shape}")
        print(f"lowerb.shape = {lowerb.shape}")
        print(f"upperb.shape = {upperb.shape}")

    #
    # Create the C data structures needed for ntg()
    #

    # Utility functions to check dimensions and set up C arrays
    def init_c_array_1d(array, size, type):
        array = np.atleast_1d(array)
        assert array.size == size
        return array.astype(type)

    # Set up spline dimensions
    cdef int [:] c_ninterv = init_c_array_1d(nintervals, nout, np.intc)
    cdef int [:] c_order = init_c_array_1d(order, nout, np.intc)
    cdef int [:] c_mult = init_c_array_1d(multiplicity, nout, np.intc)
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
        ncoef += nintervals[i] * (order[i] - multiplicity[i]) + multiplicity[i]

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
        _parse_callback(nlicf, nlicf_av, nout, c_flaglen, num=nlicf_num)
    c_nlic = (<ntg_vector_cbf *> nlic_addr)[0]

    # Nonlinear trajectory constraints
    nnltc, nltc_addr, ntrajectoryconstrav, c_trajectoryconstrav = \
        _parse_callback(nltcf, nltcf_av, nout, c_flaglen, num=nltcf_num)
    c_nltc = (<ntg_vector_traj_cbf *> nltc_addr)[0]

    nnlfc, nlfc_addr, nfinalconstrav, c_finalconstrav = \
        _parse_callback(nlfcf, nlfcf_av, nout, c_flaglen, num=nlfcf_num)
    c_nlfc = (<ntg_vector_cbf *> nlfc_addr)[0]

    # Bounds on the constraints
    nil = np.array([0.])        # Use as "empty" constraint matrix
    cdef double [:] c_lowerb = nil if lowerb.size == 0 else \
        lowerb.astype(np.double)
    cdef double [:] c_upperb = nil if upperb.size == 0 else \
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
        _parse_callback(icf, icf_av, nout, c_flaglen, num=1)
    c_icf = (<ntg_scalar_cbf *> icf_addr)[0]

    ntcf, tcf_addr, ntrajectorycostav, c_trajectorycostav = \
        _parse_callback(tcf, tcf_av, nout, c_flaglen, num=1)
    c_tcf = (<ntg_scalar_traj_cbf *> tcf_addr)[0]

    nfcf, fcf_addr, nfinalcostav, c_finalcostav = \
        _parse_callback(fcf, fcf_av, nout, c_flaglen, num=1)
    c_fcf = (<ntg_scalar_cbf *> fcf_addr)[0]

    # NTG internal memory
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

    return coefs, objective, inform

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
                f"active variables specified for callback {name}, "
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
