# ntg.pyx - Cython interface to NTG
# RMM, 15 Jul 2022
#
# This module provides a Cython interface to the NTG library.  The following
# functions and classes are provided:
#
#   npsol_option() - set NPSOL options [TODO: missing]
#   ntg() - main function to call NTG
#   spline_interp() - spline interpolation
#
# This module provides the minimal support necessary to interface to the
# underlying NTG C code.  Higher level functions are in the .py files in
# this same directory (eg, optimal.py).
#

cimport numpy as np
import numpy as np
import ctypes
from warnings import warn
from libc.stdlib cimport malloc, calloc, free
cimport ntg as ntg


class SystemTrajectory:
    """Class representing a system trajectory.

    The `SystemTrajectory` class is used to represent the trajectory of a
    (differentially flat) system.  Used by the :func:`~ntg.call_ntg`
    function to return a trajectory.

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
        For each flat output, the order of the B-spline (= degree + 1)

    smoothness : list of ints
        For each flat output, the smoothness at the knot points.

    """
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
# Main ntg() function
#
# For now this function looks more or less like the C version of ntg(), but
# with Python objects as arguments (so we don't have to separately pass
# object sizes) and with keyword arguments for anything that is optional.
#

def call_ntg(
    nout,                       # number of outputs
    breakpoints,                # breakpoints (actually collocation points)
    nintervals,                 # number of intervals
    order,                      # order of B-spline (for each output)
    smoothness,                 # smoothness at knot points (for each output)
    flaglen,                    # max number of derivs + 1 (for each output)
    knotpoints=None,            # knot points (breakpoints) for each output
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
        Array of times at which constraints should be evaluated and
        integrated costs are computed.

    TODO: finish documentation

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
    nlic, c_lic = _convert_linear_constraint(
        lic, name='initial', verbose=verbose)
    nltc, c_ltc = _convert_linear_constraint(
        ltc, name='trajectory', verbose=verbose)
    nlfc, c_lfc = _convert_linear_constraint(
        lfc, name='final', verbose=verbose)

    #
    # Callback functions
    #
    # Functions defining costs and constraints should be passed as ctypes
    # functions (so that they can be loaded dynamically).  The pointer to
    # this callback function is passed to NTG directly, after some
    # typcasting to keep Cython happy.
    #

    # Nonlinear initial condition constraints
    nnlic, nlic_addr, ninitialconstrav, c_initialconstrav = \
        _parse_callback(nlicf, nlicf_av, nout, c_flaglen, num=nlicf_num,
                        name='initial constraint')
    c_nlic = (<ntg_vector_endpoint_cbf *> nlic_addr)[0]

    # Nonlinear trajectory constraints
    nnltc, nltc_addr, ntrajectoryconstrav, c_trajectoryconstrav = \
        _parse_callback(nltcf, nltcf_av, nout, c_flaglen, num=nltcf_num,
                        name='trajectory_constraint')
    c_nltc = (<ntg_vector_trajectory_cbf *> nltc_addr)[0]

    # Nonlinear final constraints
    nnlfc, nlfc_addr, nfinalconstrav, c_finalconstrav = \
        _parse_callback(nlfcf, nlfcf_av, nout, c_flaglen, num=nlfcf_num,
                        name='final constraint')
    c_nlfc = (<ntg_vector_endpoint_cbf *> nlfc_addr)[0]

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

    # Initial cost
    nicf, icf_addr, ninitialcostav, c_initialcostav = \
        _parse_callback(icf, icf_av, nout, c_flaglen, num=1,
                        name='initial cost')
    c_icf = (<ntg_scalar_endpoint_cbf *> icf_addr)[0]

    # Trajectory cost
    ntcf, tcf_addr, ntrajectorycostav, c_trajectorycostav = \
        _parse_callback(tcf, tcf_av, nout, c_flaglen, num=1,
                        name='trajectory cost')
    c_tcf = (<ntg_scalar_trajectory_cbf *> tcf_addr)[0]

    # Final cost
    nfcf, fcf_addr, nfinalcostav, c_finalcostav = \
        _parse_callback(fcf, fcf_av, nout, c_flaglen, num=1, name='final cost')
    c_fcf = (<ntg_scalar_endpoint_cbf *> fcf_addr)[0]

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


# Cost function and constraint signatures
from numba import types
numba_ntg_trajectory_cost_signature = types.void(
    types.CPointer(types.intc),       # int *mode
    types.CPointer(types.intc),       # int *nstate
    types.CPointer(types.intc),       # int *i
    types.CPointer(types.double),     # double *f
    types.CPointer(types.double),     # double *df
    types.CPointer(                   # double **zp
        types.CPointer(types.double)))

numba_ntg_endpoint_cost_signature = types.void(
    types.CPointer(types.intc),       # int *mode
    types.CPointer(types.intc),       # int *nstate
    types.CPointer(types.double),     # double *f
    types.CPointer(types.double),     # double *df
    types.CPointer(                   # double **zp
        types.CPointer(types.double)))

numba_ntg_trajectory_constraint_signature = types.void(
    types.CPointer(types.intc),       # int *mode
    types.CPointer(types.intc),       # int *nstate
    types.CPointer(types.intc),       # int *i
    types.CPointer(types.double),     # double *f
    types.CPointer(                   # double **df
        types.CPointer(types.double)),
    types.CPointer(                   # double **zp
        types.CPointer(types.double)))

numba_ntg_endpoint_constraint_signature = types.void(
    types.CPointer(types.intc),       # int *mode
    types.CPointer(types.intc),       # int *nstate
    types.CPointer(types.double),     # double *f
    types.CPointer(                   # double **df
        types.CPointer(types.double)),
    types.CPointer(                   # double **zp
        types.CPointer(types.double)))
