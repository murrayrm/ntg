# distutils: include_dirs = ../include

import numpy as np
import ctypes

from libc.stdlib cimport malloc, calloc, free
cimport ntg as ntg

class actvar(object):
    def __init__(self, output, deriv):
        self.output = output
        self.deriv = deriv

def print_banner():
    ntg.printNTGBanner()

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
    bps,                        # break points
    ninterv,                    # number of intervals
    order,                      # order of polynomial (for each output)
    mult,                       # multiplicity at knot points (for each output)
    maxderiv,                   # max number of derivatives (for each output)
    knots=None,                 # knot points
    initialguess=None,          # initial guess for coefficients
    lic=None, ltc=None,         # linear init, traj, final constraints
        lfc=None,
    nlicf=None, nltcf=None,     # nonln init, traj, final constraints
        nlfcf=None,
    initialconstav=None,        # (nonlinear) initial constraint active vars
    trajectoryconstav=None,     # (nonlinear) initial constraint active vars
    finalconstav=None,          # (nonlinear) initial constraint active vars
    lowerb=None, upperb=None,   # upper and lower bounds for constraints
    icf=None, tcf=None,         # initial, traj, final cost functions
        fcf=None,
    icf_av=None, tcf_av=None,   # init, traj, final cost active vars
        fcf_av=None,
    verbose=False               # turn on verbose messages
):
    # Utility functions to check dimensions and set up C arrays
    def init_c_array_1d(array, size, type):
        array = np.atleast_1d(array)
        assert array.size == size
        return array.astype(type)

    # Set up spline dimensions
    cdef int [:] c_ninterv = init_c_array_1d(ninterv, nout, np.intc)
    cdef int [:] c_order = init_c_array_1d(order, nout, np.intc)
    cdef int [:] c_mult = init_c_array_1d(mult, nout, np.intc)
    cdef int [:] c_maxderiv = init_c_array_1d(maxderiv, nout, np.intc)

    # Breakpoints
    bps = np.atleast_1d(bps)
    assert bps.ndim == 1
    cdef int nbps = bps.shape[0]
    cdef double [:] c_bps = init_c_array_1d(bps, nbps, np.double)

    # Knot points (for each output)
    cdef double **c_knots = <double **> malloc(nout * sizeof(double *))
    if knots is None:
        # Set up equally space knot points for reach output
        knots = \
            [np.linspace(0, bps[-1], ninterv[out] + 1) for out in range(nout)]
    for out in range(nout):
        assert len(knots[out]) == ninterv[out] + 1
        c_knots[out] = <double *> malloc(len(knots[out]) * sizeof(double))
        for time in range(len(knots[out])):
            c_knots[out][time] = knots[out][time]

    # Coefficients: initial guess + return result
    ncoef = 0
    for i in range(nout):
        ncoef += ninterv[i] * (order[i] - mult[i]) + mult[i]
        
    if initialguess is not None:
        coefs = np.atleast_1d(initialguess)
        assert coefs.ndim == 1
        assert coefs.size == ncoef
    else:
        coefs = np.ones(ncoef)
    cdef double [:] c_coefs = init_c_array_1d(coefs, ncoef, np.double)

    # Process linear constraints
    cdef int nlic = 0
    cdef double **c_lic = NULL
    if lic is not None:
        lic = np.atleast_2d(lic)
        assert lic.ndim == 2
        nlic = lic.shape[0]
        c_lic = <double **> malloc(nlic * sizeof(double *))
        c_lic[0] = <double *> malloc(nlic * lic.shape[1] * sizeof(double))
        for i in range(nlic):
            c_lic[i] = &c_lic[0][i * lic.shape[1]]
            for j in range(lic.shape[1]):
                c_lic[i][j] = lic[i, j]

        if verbose:
            print(f"  {nlic} initial constraints of size {lic.shape[1]}")
            print("  lic = ", lic)
            
    cdef int nltc = 0
    cdef double **c_ltc = NULL
    if ltc is not None:
        ltc = np.atleast_2d(ltc)
        assert ltc.ndim == 2
        nltc = ltc.shape[0]
        c_ltc = <double **> calloc(nltc, sizeof(double *))
        c_ltc[0] = <double *> malloc(nltc * ltc.shape[1] * sizeof(double))
        for i in range(nltc):
            c_ltc[i] = &c_ltc[0][i * ltc.shape[1]]
            for j in range(ltc.shape[1]):
                c_ltc[i][j] = ltc[i, j]

        if verbose:
            print(f"  {nltc} trajectory constraints of size {ltc.shape[1]}")
                
    cdef int nlfc = 0
    cdef double **c_lfc = NULL
    if lfc is not None:
        lfc = np.atleast_2d(lfc)
        assert lfc.ndim == 2
        nlfc = lfc.shape[0]
        c_lfc = <double **> calloc(nlfc, sizeof(double *))
        c_lfc[0] = <double *> malloc(nlfc * lfc.shape[1] * sizeof(double))
        for i in range(nlfc):
            c_lfc[i] = &c_lfc[0][i * lfc.shape[1]]
            for j in range(lfc.shape[1]):
                c_lfc[i][j] = lfc[i, j]

        if verbose:
            print(f"  {nlfc} final constraints of size {lfc.shape[1]}")
            print("  lfc = ", lfc)

    # Bounds on the constraints
    cdef double [:] c_lowerb = lowerb
    cdef double [:] c_upperb = upperb
    if verbose:
        print("  lower bounds = ", lowerb)
        print("  upper bounds = ", upperb)
    
    # Nonlinear constraint callbacks
    cdef int nnlic = 0
    cdef void (*nl_initial_constraint)()
    cdef int ninitialconstrav = 0

    cdef int nnltc = 0
    cdef void (*nl_trajectory_constraint)()
    cdef int ntrajectoryconstrav = 0

    cdef int nnlfc = 0
    cdef void (*nl_final_constraint)()
    cdef int nfinalconstrav = 0

    # Cost function callbacks
    cdef int nicf = 0
    cdef int ninitialcostav = 0

    cdef int ntcf = 0
    cdef int ntrajectorycostav = 0
    if tcf is not None:
        global ctypes_tcf
        ctypes_tcf = tcf
        ntcf = 1

        if tcf_av is not None:
            ntrajectorycostav = len(tcf_av)
            c_trajectorycostav = <AV *> calloc(ntrajectorycostav, sizeof(AV))
            for i, av in enumerate(tcf_av):
                c_trajectorycostav[i].output = av.output
                c_trajectorycostav[i].deriv = av.deriv
    
    cdef int nfcf = 0
    cdef int nfinalcostav = 0

    # Check to make sure dimensions are consistent
    assert lowerb.size == nlic + nltc + nlfc + nnlic + nnltc + nnlfc
    assert upperb.size == nlic + nltc + nlfc + nnlic + nnltc + nnlfc
    
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
        nnltc,               nltc_callback,
        nnlfc,               NULL,
        ninitialconstrav,    NULL,
        ntrajectoryconstrav, NULL,
        nfinalconstrav,      NULL,
        &c_lowerb[0], &c_upperb[0],
        nicf,		     NULL,
        ntcf,                tcf_callback,
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

#
# Callback functions
#
# Functions defining costs and constraints should be passed as ctypes
# functions (so that they can be loaded dynamically).  We pass a Cython
# callback function to NTG and then the callback function calls the ctypes
# function that were were passed (which is stored in a module variable of
# the form 'ctypes_<ntgfcn>'.
#
# Note that the callback functions don't check to make sure that the ctypes
# callback function is not None.  We use the fact that the number of
# functions/constraints will be passed as zero and so the callback will
# never actually be called.
#

# Trajectory cost function
ctypes_tcf = None
cdef void tcf_callback(
    int *mode, int *nstate, int *i, double *f, double *df, double **zp):
    # Convert the ctypes function pointer into a Cython funciton pointer
    cdef ntg_scalar_cbf cy_tcf = \
        (<ntg_scalar_cbf *> <size_t> ctypes.addressof(ctypes_tcf))[0]
    with nogil:
        cy_tcf(mode, nstate, i, f, df, zp)

# Trajectory constraint function
ctypes_nltc = None
cdef void nltc_callback(
    int *mode, int *nstate, int *nnltc,
    double *nlc, double **dnlc, double **zp):
    # Convert the ctypes function pointer into a Cython funciton pointer
    cdef ntg_vector_cbf cy_nltc = \
        (<ntg_vector_cbf *> <size_t> ctypes.addressof(ctypes_nltc))[0]
    with nogil:
        cy_nltc(mode, nstate, nnltc, nlc, dnlc, zp)
