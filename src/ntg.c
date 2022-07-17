/*********************************************************/
/*                                                       */
/*                     NTG 2.2                           */
/*              Copyright (c) 2000 by                    */
/*      California Institute of Technology               */
/*          Control and Dynamical Systems                */
/*    Mark Milam, Kudah Mushambi, and Richard Murray     */
/*               All right reserved.                     */
/*                                                       */
/*********************************************************/


#include "ntg.h"

/*
 * Static ("global") variables
 *
 * To allow the callback functions to access the arguments that are
 * passed to ntg(), we store everything in a static variable for later
 * access.  (Not thread safe.)
 */

static ConcatColloc *Gccolloc;
static double *GZ;
static double *Gbps;

static FMatrix *GcJac;

static int Gnlic, Gnltc, Gnlfc;

static AV *Ginitialconstrav; static int Gninitialconstrav;
static AV *Gtrajectoryconstrav; static int Gntrajectoryconstrav;
static AV *Gfinalconstrav; static int Gnfinalconstrav;

static AV *Ginitialcostav; static int Gninitialcostav;
static AV *Gtrajectorycostav; static int Gntrajectorycostav;
static AV *Gfinalcostav; static int Gnfinalcostav;

static int Gnnlic, Gnnltc, Gnnlfc;
static void (*Gnlicf)(int *, int *, double *, double **, double **);
static void (*Gnltcf)(int *, int *, int *, double *, double **, double **);
static void (*Gnlfcf)(int *, int *, double *, double **, double **);

static int Gnicf, Gnucf, Gnfcf;
static void (*Gicf)(int *, int *, double *, double *, double **);
static void (*Gucf)(int *, int *, int *, double *, double *, double **);
static void (*Gfcf)(int *, int *, double *, double *, double **);

/*
 * External (FORTRAN) functions
 *
 * This is the list of external functions that we call to interface with
 * NPSOL and others solvers.
 *
 */

extern void npsol_();
extern void npfile_();
extern void npoptn_();
extern void fopen_();
extern void fclose_();

/*
 * Call back function declations
 *
 * These are the functions that we pass to NPSOL to evaluate the constraints
 * and objective (cost) function.
 *
 */

static void NPfuncon();
static void NPfunobj();

/*
 * Main NTG function
 *
 */

void ntg(
  int nout,		 /* number of outputs */
  double *bps,		 /* breakpoint sequence */
  int nbps,		 /* number of breakpoints (same for all outputs ) */
  int *kninterv,	 /* knot point intervals */
  double **knots,	 /* Knot points for each output */
  int *order,		 /* orders of the outputs */
  int *mult,		 /* multiplicities of the outputs */
  int *maxderiv,	 /* Max derivative + 1  occurring in each output */
  double *initialguess,	 /* initial b-spline coefficient guess */
  int nlic, double **lic,
  int nltc, double **ltc,
  int nlfc, double **lfc,
  int nnlic, void (*nlicf)(int *, int *, double *, double **, double **),
  int nnltc, void (*nltcf)(int *, int *, int *, double *, double **, double **),
  int nnlfc, void (*nlfcf)(int *, int *, double *, double **, double **),
  int ninitialconstrav,    AV *initialconstrav,
  int ntrajectoryconstrav, AV *trajectoryconstrav,
  int nfinalconstrav,      AV *finalconstrav,
  double *lowerb,
  double *upperb,
  int nicf, void (*icf)(int *, int *, double *, double *, double **),
  int nucf, void (*ucf)(int *, int *, int *, double *, double *, double **),
  int nfcf, void (*fcf)(int *, int *, double *, double *, double **),
  int ninitialcostav,    AV *initialcostav,
  int ntrajectorycostav, AV *trajectorycostav,
  int nfinalcostav,      AV *finalcostav,
  int *istate, double *clambda, double *R,
  int *inform, double *objective)
{
  int i;

  /*
   * NPSOL variables
   *
   * These variables are passed to NPSOL
   *
   */

  int NPn;
  int NPnclin;
  int NPncnln;
  int NPldA;
  int NPldJ;
  int NPldR;
  int *NPinform;
  int NPiter;
  int NPleniw;
  int NPlenw;
  int *NPistate;
  int *NPiw;

  double *NPf;
  double *NPA;
  double *NPbl;
  double *NPbu;
  double *NPc;
  double *NPcJac;
  double *NPclambda;
  double *NPg;
  double *NPR;
  double *NPx=initialguess;
  double *NPw;

  /* Temporary (floating point) matrices used to construct constraints */
  FMatrix *Matlc, *Matlic, *Matltc, *Matlfc;

  /*
   * Allocate space for the collocation matrix and the flat flag
   */

  Gccolloc = ConcatCollocMatrix(nout, knots, kninterv, bps,
			      nbps, maxderiv, order, mult);
  /*cbug*/
  /*PrintColloc("colloc", Gccolloc->colloc[0]); */

  GZ = calloc(Gccolloc->nZ, sizeof(double));

  /*
   * Store all of the arguments so that we can acccess them in the
   * callback functions (below)
   */

  Gbps = bps;

  Gnlic = nlic;
  Gnltc = nltc;
  Gnlfc = nlfc;
  Gnnlic = nnlic;
  Gnnltc = nnltc;
  Gnnlfc = nnlfc;

  Gnlicf = nlicf;
  Gnltcf = nltcf;
  Gnlfcf = nlfcf;

  Ginitialconstrav = initialconstrav;
  Gtrajectoryconstrav = trajectoryconstrav;
  Gfinalconstrav = finalconstrav;
  Gninitialconstrav = ninitialconstrav;
  Gntrajectoryconstrav = ntrajectoryconstrav;
  Gnfinalconstrav = nfinalconstrav;

  Gnicf = nicf;
  Gnucf = nucf;
  Gnfcf = nfcf;
  Gicf = icf;
  Gucf = ucf;
  Gfcf = fcf;

  Ginitialcostav = initialcostav;
  Gtrajectorycostav = trajectorycostav;
  Gfinalcostav = finalcostav;
  Gninitialcostav = ninitialcostav;
  Gntrajectorycostav = ntrajectorycostav;
  Gnfinalcostav = nfinalcostav;

  /*
   * NPSOL parameters
   */

  NPn = Gccolloc->nC;
  NPnclin = Gnlic + Gnltc*nbps + Gnlfc;
  NPncnln = Gnnlic + Gnnltc*nbps + Gnnlfc;
  NPinform = inform;
  NPf = objective;

  /*
   * Contruct NPSOL matrices for linear contraints
   *
   */

  if (NPnclin == 0) {
    NPldA = 1;
    Matlc = MakeFMatrix(NPldA, 1);
  } else {
    NPldA = NPnclin;
    Matlc = MakeFMatrix(NPldA, NPn);

    if (nlic != 0) {
      Matlic = malloc(sizeof(FMatrix));
      Matlic->elements = lic;
      Matlic->rows = Gccolloc->nz;
      Matlic->cols = nlic;
    } else {
      Matlic = NULL;
    }

    if (nltc != 0) {
      Matltc = malloc(sizeof(FMatrix));
      Matltc->elements = ltc;
      Matltc->rows = Gccolloc->nz;
      Matltc->cols = nltc;
    } else {
      Matltc = NULL;
    }

    if (nlfc != 0) {
      Matlfc = malloc(sizeof(FMatrix));
      Matlfc->elements = lfc;
      Matlfc->rows = Gccolloc->nz;
      Matlfc->cols = nlfc;
    } else {
	Matlfc = NULL;
    }

    LinearConstraintsMatrix(Matlc, Matlic, Matltc, Matlfc, Gccolloc);
    if (nlic != 0) free(Matlic);
    if (nltc != 0) free(Matltc);
    if (nlfc != 0) free(Matlfc);
  }
  NPA = Matlc->elements[0];
  /*cbug*/
  /*PrintFMatrix("NPA", Matlc); */

  if (NPncnln == 0) {
    NPldJ = 1;
    GcJac = MakeFMatrix(NPldJ, 1);
  } else {
    NPldJ = NPncnln;
    GcJac = MakeFMatrix(NPldJ, NPn);
  }
  NPcJac = GcJac->elements[0];

  NPldR = NPn;

  /*
   * Create NPSOL upper and lower bound vectors
   */

  NPbu = calloc((NPn + NPnclin + NPncnln), sizeof(double));
  NPbl = calloc((NPn + NPnclin + NPncnln), sizeof(double));
  bounds(NPbu, upperb, Gccolloc->nC,
	 nlic, nltc, nlfc, nnlic, nnltc, nnlfc, Gccolloc->nbps, DBL_MAX);
  bounds(NPbl, lowerb, Gccolloc->nC,
	 nlic, nltc, nlfc, nnlic, nnltc, nnlfc, Gccolloc->nbps, -DBL_MAX);

  NPistate = istate;
  NPc = calloc(NPncnln, sizeof(double));
  NPclambda = clambda;
  NPg = calloc(NPn, sizeof(double));
  NPR = R;

  NPleniw = (3*NPn) + NPnclin + (2*NPncnln);
  NPiw = calloc(NPleniw, sizeof(int));
  if (NPnclin == 0 && NPncnln == 0)
    NPlenw = 20*NPn;
  else if (NPncnln == 0)
    NPlenw = (2*NPn*NPn) + (20*NPn) + (11*NPnclin);
  else
    NPlenw = (2*NPn*NPn) + (NPn*NPnclin) + (2*NPn*NPncnln) + (20*NPn)\
      + (11*NPnclin) + (21*NPncnln);
  NPw = calloc(NPlenw, sizeof(double));

  /*
   * Call NPSOL
   */

  npsoloption("nolist");
  npsoloption("derivative level = 3");
  npsol_(&NPn, &NPnclin, &NPncnln, &NPldA, &NPldJ, &NPldR, NPA, NPbl,
	 NPbu, NPfuncon, NPfunobj, NPinform, &NPiter, NPistate,
	 NPc, NPcJac, NPclambda, NPf, NPg, NPR, NPx, NPiw, &NPleniw,
	 NPw, &NPlenw);

  /* Free up storage that we don't need */
  /* TODO: updated for RHC implementation */
  free(NPbu);
  free(NPbl);
  free(NPc);
  free(NPg);
  free(NPiw);
  free(NPw);

  FreeConcatColloc(Gccolloc);
  FreeFMatrix(Matlc);
  FreeFMatrix(GcJac);

  free(GZ);
}

/* Utility function to set NPSOL options (call FORTRAN function) */
void npsoloption(char *option) {
	npoptn_(option, (long)strlen(option));
}

/* Objective function callback */
void NPfunobj(int *mode, int *n, double *x, double *y,
	      double *yprime, int *nstate) {
  size_t i1;
  double I = 0.0, *dI;
  double In = 0.0, *dIn;
  double F = 0.0, *dF;

  if (Gnicf != 0)
    updateZ(GZ, Gccolloc, x, Ginitialcostav, Gninitialcostav, AVINITIAL);
  if (Gnucf != 0)
    updateZ(GZ, Gccolloc, x, Gtrajectorycostav, Gntrajectorycostav, AVTRAJECTORY);
  if (Gnfcf != 0)
    updateZ(GZ, Gccolloc, x, Gfinalcostav, Gnfinalcostav, AVFINAL);

  switch(*mode) {
    case 0:			 /* only compute objective function */
      if (Gnicf == 1)
	InitialCost(mode, nstate, &I, NULL, Gicf, Gccolloc, GZ);
      if (Gnucf == 1)
	IntegratedCost(mode, nstate, &In, NULL, Gbps, Gucf, Gccolloc, GZ);
      if (Gnfcf == 1)
	FinalCost(mode, nstate, &F, NULL, Gfcf, Gccolloc, GZ);
      *y = I + In + F;
      break;

    case 1:			 /* only compute objective gradient */
      dI = calloc(Gccolloc->nC, sizeof(double));
      dIn = calloc(Gccolloc->nC, sizeof(double));
      dF = calloc(Gccolloc->nC, sizeof(double));
      if (Gnicf != 0)
	InitialCost(mode, nstate, NULL, dI, Gicf, Gccolloc, GZ);
      if (Gnucf != 0)
	IntegratedCost(mode, nstate, NULL, dIn, Gbps, Gucf, Gccolloc, GZ);
      if (Gnfcf != 0)
	FinalCost(mode, nstate, NULL, dF, Gfcf, Gccolloc, GZ);
      Vector3Add(yprime, dI, dIn, dF, Gccolloc->nC);
      free(dI); free(dIn); free(dF);
      break;

    case 2:		 /* compute both obj function and gradients */
      dI = calloc(Gccolloc->nC, sizeof(double));
      dIn = calloc(Gccolloc->nC, sizeof(double));
      dF = calloc(Gccolloc->nC, sizeof(double));
      if (Gnicf != 0)
	InitialCost(mode, nstate, &I, dI, Gicf, Gccolloc, GZ);
      if (Gnucf != 0)
	IntegratedCost(mode, nstate, &In, dIn, Gbps, Gucf, Gccolloc, GZ);
      if (Gnfcf != 0)
	FinalCost(mode, nstate, &F, dF, Gfcf, Gccolloc, GZ);
      *y = I + In + F;
      Vector3Add(yprime, dI, dIn, dF, Gccolloc->nC);
      free(dI); free(dIn); free(dF);
      break;
    default:
      *nstate = -1;
  }
}

/* Constraint function callback */
void NPfuncon(
  int *mode,
  int *ncnln,
  int *n,
  int *nrowj,
  int *needc,
  double *x,
  double *C,
  double *Cjac,
  int *nstate)
{
  if (Gnnlic != 0)
    updateZ(GZ, Gccolloc, x, Ginitialconstrav, Gninitialconstrav, AVINITIAL);
  if (Gnnltc != 0)
    updateZ(GZ, Gccolloc, x, Gtrajectoryconstrav, Gntrajectoryconstrav,
	    AVTRAJECTORY);
  if (Gnnlfc != 0)
    updateZ(GZ, Gccolloc, x, Gfinalconstrav, Gnfinalconstrav, AVFINAL);

  switch(*mode) {
  case 0:		       /* only compute constraint functions */
    NonLinearConstraints(mode, nstate, C, NULL,
			 Gnnlic, Gnlicf, Gnnltc, Gnltcf,
			 Gnnlfc, Gnlfcf, Gccolloc, GZ);
    break;

  case 1:			/* only compute constraint gradient */
    NonLinearConstraints(mode, nstate, NULL, GcJac,
			 Gnnlic, Gnlicf, Gnnltc, Gnltcf,
			 Gnnlfc, Gnlfcf, Gccolloc, GZ);
    break;

  case 2:		     /* compute both function and gradients */
    NonLinearConstraints(mode, nstate, C, GcJac,
			 Gnnlic, Gnlicf, Gnnltc, Gnltcf,
			 Gnnlfc, Gnlfcf, Gccolloc, GZ);
    break;

  default:
    *mode = -1;
  }
}

/* MATLAB linspace() */
void linspace(double *v, double d0, double d1, int n)
{
  int i;
  double interv;
  if (d0 == d1) {
    for (i = 0; i<n; i++)
      v[i] = d0;
    return;
  }

  interv = (d1-d0)/(n-1);
  v[0] = d0;
  for (i = 1; i<n; i++)
    v[i] = v[i-1] + interv;
}

void printNTGBanner(void)
{
  printf("\n");
  printf("\n");
  printf("                                   NTG 2.2                       \n");
  printf("                            Copyright (c) 2000 by                \n");
  printf("                    California Institute of Technology           \n");
  printf("                       Control and Dynamical Systems             \n");
  printf("              Mark Milam, Kudah Mushambi, and Richard Murray     \n");
  printf("                             All right reserved.                 \n");
  printf("\n");
  printf("          *******************************************************\n");
  printf("\n");
  printf("\n");
}
