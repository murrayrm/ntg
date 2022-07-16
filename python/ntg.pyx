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
    nintervals,                 # number of intervals
    order,                      # order of polynomial (for each output)
    multiplicity,               # multiplicity at knot points (for each output)
    maxderivs,                  # max number of derivatives (for each output)
    knotpoints=None,            # knot points
    initialguess=None,          # initial guess for coefficients
    lic=None, ltc=None, lfc=None,       # linear init, traj, final constraints
    nlicf=None, nlicf_num=None, nlicf_av=None, # NL initial constraints
    nltcf=None, nltcf_num=None, nltcf_av=None, # NL trajectory constraints
    nlfcf=None, nlfcf_num=None, nlfcf_av=None, # NL final constraints
    lowerb=None, upperb=None,   # upper and lower bounds for constraints
    icf=None, icf_av=None,      # initial cost function, active vars
    tcf=None, tcf_av=None,      # trajectory cost function, active vars
    fcf=None, fcf_av=None,      # final cost function, active vars
    verbose=False               # turn on verbose messages
):
    # Utility functions to check dimensions and set up C arrays
    def init_c_array_1d(array, size, type):
        array = np.atleast_1d(array)
        assert array.size == size
        return array.astype(type)

    # Set up spline dimensions
    cdef int [:] c_ninterv = init_c_array_1d(nintervals, nout, np.intc)
    cdef int [:] c_order = init_c_array_1d(order, nout, np.intc)
    cdef int [:] c_mult = init_c_array_1d(multiplicity, nout, np.intc)
    cdef int [:] c_maxderiv = init_c_array_1d(maxderivs, nout, np.intc)

    # Breakpoints
    breakpoints = np.atleast_1d(breakpoints)
    assert breakpoints.ndim == 1
    cdef int nbps = breakpoints.shape[0]
    cdef double [:] c_bps = init_c_array_1d(breakpoints, nbps, np.double)

    # Knot points (for each output)
    if knotpoints is None:
        # Set up equally space knot points for reach output
        knotpoints = [np.linspace(0, breakpoints[-1], nintervals[out] + 1)
                 for out in range(nout)]

    cdef double **c_knots = <double **> malloc(nout * sizeof(double *))
    for out in range(nout):
        assert len(knotpoints[out]) == nintervals[out] + 1
        c_knots[out] = <double *> malloc(len(knotpoints[out]) * sizeof(double))
        for time in range(len(knotpoints[out])):
            c_knots[out][time] = knotpoints[out][time]

    # Coefficients: initial guess + return result
    ncoef = 0
    for i in range(nout):
        ncoef += nintervals[i] * (order[i] - multiplicity[i]) + multiplicity[i]

    if initialguess is not None:
        coefs = np.atleast_1d(initialguess)
        assert coefs.ndim == 1
        assert coefs.size == ncoef
    else:
        coefs = np.ones(ncoef)
    cdef double [:] c_coefs = init_c_array_1d(coefs, ncoef, np.double)

    # Process linear constraints
    nlic, c_lic = _parse_linear_constraint(lic, name='initial', verbose=verbose)
    nltc, c_ltc = _parse_linear_constraint(
        ltc, name='trajectory', verbose=verbose)
    nlfc, c_lfc = _parse_linear_constraint(lfc, name='final', verbose=verbose)

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
        _parse_callback(nlicf, nlicf_av, nout, c_maxderiv, num=nlicf_num)
    c_nlic = (<ntg_vector_cbf *> nlic_addr)[0]

    # Nonlinear trajectory constraints
    nnltc, nltc_addr, ntrajectoryconstrav, c_trajectoryconstrav = \
        _parse_callback(nltcf, nltcf_av, nout, c_maxderiv, num=nltcf_num)
    c_nltc = (<ntg_vector_cbf *> nltc_addr)[0]

    nnlfc, nlfc_addr, nfinalconstrav, c_finalconstrav = \
        _parse_callback(nlfcf, nlfcf_av, nout, c_maxderiv, num=nlfcf_num)
    c_nlfc = (<ntg_vector_cbf *> nlfc_addr)[0]

    # Bounds on the constraints
    cdef double [:] c_lowerb = lowerb
    cdef double [:] c_upperb = upperb
    if verbose:
        print("  lower bounds = ", lowerb)
        print("  upper bounds = ", upperb)

    # Check to make sure dimensions are consistent
    assert lowerb.size == nlic + nltc + nlfc + nnlic + nnltc + nnlfc
    assert upperb.size == nlic + nltc + nlfc + nnlic + nnltc + nnlfc

    #
    # Cost function callbacks
    #
    nicf, icf_addr, ninitialcostav, c_initialcostav = \
        _parse_callback(icf, icf_av, nout, c_maxderiv, num=1)
    c_icf = (<ntg_scalar_cbf *> icf_addr)[0]

    ntcf, tcf_addr, ntrajectorycostav, c_trajectorycostav = \
        _parse_callback(tcf, tcf_av, nout, c_maxderiv, num=1)
    c_tcf = (<ntg_scalar_cbf *> tcf_addr)[0]

    nfcf, fcf_addr, nfinalcostav, c_finalcostav = \
        _parse_callback(fcf, fcf_av, nout, c_maxderiv, num=1)
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
        &c_order[0], &c_mult[0], &c_maxderiv[0], &c_coefs[0],
        nlic,                &c_lic[0],
        nltc,                NULL,
        nlfc,                &c_lfc[0],
        nnlic,               NULL,
        nnltc,               c_nltc,
        nnlfc,               NULL,
        ninitialconstrav,    NULL,
        ntrajectoryconstrav, NULL,
        nfinalconstrav,      NULL,
        &c_lowerb[0], &c_upperb[0],
        nicf,		     NULL,
        ntcf,                c_tcf,
        nfcf,		     NULL,
        ninitialcostav,      NULL,
        ntrajectorycostav,   c_trajectorycostav,
        nfinalcostav,	     NULL,
        istate, clambda, R, &inform, &objective);

    if verbose:
        print(f"NTG returned inform={inform}, objective={objective}")

    # Copy the coefficients back into our NumPy array
    cdef int k
    for k in range(coefs.size):
        coefs[k] = c_coefs[k]

    return coefs

def spline_interp(x, knots, ninterv, coefs, order, mult, maxderiv):
    cdef double [:] c_knots = knots
    cdef double [:] c_coefs = coefs
    cdef int ncoefs = len(coefs)
    cdef np.ndarray[double, ndim=1, mode='c'] fz = np.empty(maxderiv)
    ntg.SplineInterp(
        &fz[0], x, &c_knots[0], ninterv, &c_coefs[0], len(coefs),
        order, mult, maxderiv)

    # Store results in an ndarray and free up memory
    return fz

#
# Utility functions
#

# Cython function to parse linear constraints
cdef (int, double **) _parse_linear_constraint(
    cmatrix, name='unknown', verbose=False):
    cdef int nlic = 0
    cdef double ** c_lic = NULL
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
        ctypes_fcn, avlist, nout, maxderiv, num=None, name='unknown'):
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
            nav += maxderiv[i]

        # Make all variables active
        c_av = <AV *> calloc(nav, sizeof(AV))
        k = 0
        for i in range(nout):
            for j in range(maxderiv[i]):
                c_av[k].output = i
                c_av[k].deriv = j
                k = k + 1

    return <int> nfcn, c_fcn, <int> nav, c_av
