# ntg.pxd: Cython API defintion for NTG
# RMM, 9 Jul 2022

cdef extern from "../src/av.h":
    ctypedef struct AV:
        int output
        int deriv

cdef extern from "../src/colloc.h":
    void SplineInterp(
        double *f, double x, double *knots, int ninterv,
        double *coeffs, int ncoeffs,
        int order,int mult,int maxderiv);

cdef extern from "../src/ntg.h":
    void printNTGBanner()
    void npsoloption(char *option)
    void c_ntg "ntg" (
        int nout,
        double *bps,
        int nbps,
        int *kninterv,
        double **knots,
        int *order,int *mult,int *max_deriv,
        double *initialguess,

        int nlic,double **lic,
        int nltc,double **ltc,
        int nlfc,double **lfc,

        int nnlic, void (*nlicf)(int *, int *, double *, double **, double **),
        int nnltc, void (*nltcf)(
            int *, int *, int *, double *, double **, double **),
        int nnlfc, void (*nlfcf)(int *, int *, double *, double **, double **),
        int ninitialconstrav,AV *initialconstrav,
        int ntrajectoryconstrav,AV *trajectoryconstrav,
        int nfinalconstrav,AV *finalconstrav,

        double *lowerb,double *upperb,

        int nicf,void (*icf)(int *, int *, double *, double *, double **),
        int nucf,void (*ucf)(int *, int *, int *, double *, double *, double **),
        int nfcf,void (*fcf)(int *, int *, double *, double *, double **),
        int ninitialcostav,AV *initialcostav,
        int ntrajectorycostav,AV *trajectorycostav,
        int nfinalcostav,AV *finalcostav,

        int *istate,double *clambda,double *R,
        int *inform,double *objective
    )

# Define a type for call back functions
ctypedef void (*ntg_scalar_endpoint_cbf)(
    int *mode, int *nstate, double *f, double *df, double **zp) nogil

ctypedef int (*ocp_scalar_endpoint_cbf)(
    int mode, int nstate, double *f, double *df, double **zp) nogil

ctypedef void (*ntg_scalar_trajectory_cbf)(
    int *mode, int *nstate, int *i, double *f, double *df, double **zp) nogil

ctypedef int (*ocp_scalar_trajectory_cbf)(
    int mode, int nstate, int i, double *f, double *df, double **zp) nogil

ctypedef void (*ntg_vector_endpoint_cbf)(
    int *mode, int *nstate, double *f, double **df, double **zp) nogil

ctypedef int (*ocp_vector_endpoint_cbf)(
    int mode, int nstate, double *f, double **df, double **zp) nogil

ctypedef void (*ntg_vector_trajectory_cbf)(
    int *mode, int *nstate, int *i, double *f, double **df, double **zp) nogil

ctypedef int (*ocp_vector_trajectory_cbf)(
    int mode, int nstate, int i, double *f, double **df, double **zp) nogil

