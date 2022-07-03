/*
 * kincar.c - kinematic car example for NTG
 * RMM, 27 Jun 2022
 *
 * This file is an example of how to use NTG to generate an optimal
 * trajectory for a kinematic car performing a lane change operation.
 *
 */

#include <stdlib.h>
#include <math.h>
#include "ntg.h"

/*
 * Vehicle dynamics 
 *
 */

struct kincar_params {
  double b;
  char dir;
};
struct kincar_params default_params = {3.0, 'f'};

/* Function to take states, inputs and return the flat flag */
void kincar_flat_forward(double *x, double *u, struct kincar_params *params,
			  double **zflag) {
  double b = params->b;
  double thdot;

  /* Flat output is the x, y position of the rear wheels */
  zflag[0][0] = x[0];
  zflag[1][0] = x[1];

  /* First derivatives of the flat output */
  zflag[0][1] = u[0] * cos(x[2]);  /* dx/dt */
  zflag[1][1] = u[0] * sin(x[2]);  /* dy/dt */

  /* First derivative of the angle */
  thdot = (u[0]/b) * tan(u[1]);

  /* Second derivatives of the flat output (setting vdot = 0) */
  zflag[0][2] = -u[0] * thdot * sin(x[2]);
  zflag[1][2] =  u[0] * thdot * cos(x[2]);
}

/* Function to take the flat flag and return states, inputs */
void kincar_flat_reverse(double **zflag, struct kincar_params *params,
			 double *x, double *u) {
  double b = params->b;
  char dir = params->dir;
  double thdot_v;

  /* Given the flat variables, solve for the state */
  x[0] = zflag[0][0];			/* x position */
  x[1] = zflag[1][0];			/* y position */
  if (dir == 'f') {
    x[2] = atan2(zflag[1][1], zflag[0][1]);	/* tan(theta) = ydot/xdot */
  } else if (dir == 'r') {
    /* Angle is flipped by 180 degrees (since v < 0) */;
    x[2] = atan2(-zflag[1][1], -zflag[0][1]);
  } else {
    fprintf(stderr, "unknown direction: %c", dir);
  }

  /* Next solve for the inputs */
  u[0] = zflag[0][1] * cos(x[2]) + zflag[1][1] * sin(x[2]);
  thdot_v = zflag[1][2] * cos(x[2]) - zflag[0][2] * sin(x[2]);
  u[1] = atan2(thdot_v, pow(u[0], 2.0) / b);
}

/* Trajectory cost function (unintegrated) */
void tcf(int *mode, int *nstate, int *i, double *f, double *df, double **zp)
{
  if (*mode == 0 || *mode == 2) {
    /* compute cost function: curvature */
    *f = zp[0][2] * zp[0][2] + zp[1][2] * zp[1][2];
  }

  if (*mode == 1 || *mode == 2) {
    /* compute gradient of cost function (index = active variables) */
    df[0] = 0;
    df[1] = 0;
    df[2] = 2 * zp[0][2];
    df[3] = 0;
    df[4] = 0;
    df[5] = 2 * zp[1][2];
  }
}

#define NOUT		2		/* number of flat outputs, j */
#define NINTERV		2		/* number of intervals, lj */
#define MULT		3		/* regularity of splits, mj */
#define ORDER		5		/* degree of split polynomial, kj */
#define MAXDERIV	3		/* highest derivative required + 1 */

/*
 * Total number of coefficients required
 *
 * Sum_j [lj*(kj-mj) + mj]
 *
 */
#define NCOEF		14		/* total # of coeffs */

/* number linear constraints */
#define NLIC		6		/* linear initial constraints */
#define NLTC		0		/* linear trajectory constraints */
#define NLFC		6		/* linear final constraints */

/* number nonlinear constraints */
#define NNLIC		0		/* nonlinear initial constraints */
#define NNLTC		0		/* nonlinear trajectory constraints */
#define NNLFC		0		/* nonlinear final constraints */

/* 
 * Nonlinear constraint function active variables
 * 
 * This is the number of active variables that show up in the nonlinear
 * constraints.  The active variables are defined by their indices into
 * the zp[][] array
 *
 */
#define NINITIALCONSTRAV	0	/* active variables, initial */
#define NTRAJECTORYCONSTRAV	0	/* active variables, trajectory */
#define NFINALCONSTRAV		0	/* active variables, final */

/* number of cost functions */
#define NICF		0		/* initial */
#define NTCF		1		/* trajectory (unintegrated) */
#define NFCF		0		/* final */

/* Same as above, now for the cost functions */
#define NINITIALCOSTAV		0	/* initial */
#define NTRAJECTORYCOSTAV	6	/* trajectory */
#define NFINALCOSTAV		0	/* final */

/*
 * Now declare all of the active variables that are required for the
 * nonlinear constraints [none here] and the cost functions
 *
 */
static AV trajectorycostav[NTRAJECTORYCOSTAV] =
  {{0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 1}, {1, 2}};


int main(void)
{
  double *knots[NOUT];			/* knot points, list of times for
					   each output */

  /*
   * Now define all of the NTG parameters that were listed above
   */
  int ninterv[NOUT] =	{NINTERV, NINTERV};
  int mult[NOUT] =	{MULT, MULT};
  int maxderiv[NOUT] =	{MAXDERIV, MAXDERIV};
  int order[NOUT] =	{ORDER, ORDER};
  int nbps;
  double *bps;
  double **lic, **lfc;

  /* initial guess size = sum over each output ninterv*(order-mult)+mult */
  int ncoef;
  double *coefficients;
  int *istate;
  double *clambda;
  double *R;
	
  double lowerb[NLIC + NLTC + NLFC + NNLIC + NNLTC + NNLFC];
  double upperb[NLIC + NLTC + NLFC + NNLIC + NNLTC + NNLFC];
  int inform;
  double objective;

  int i, j;	/* indices for iterating */

  /*
   * Allocate space and initialize the knot points
   *
   * For this problem, regularly spaced from t=0 to t=5
   *
   */
  for (i = 0; i < NOUT; ++i) {
    knots[i] = calloc((ninterv[i] + 1), sizeof(double));
    linspace(knots[i], 0, 5, ninterv[i] + 1);
  }

  /*
   * Compute the number of spline coefficients required
   *
   * WARNING: this formula was computed above and the formula below
   * is specific to this particular case
   */
  ncoef = 0;
  for (i = 0; i < NOUT; ++i) {
    ncoef += ninterv[i] * (order[i] - mult[i]) + mult[i];
  }
  assert(ncoef == NCOEF);
  coefficients = calloc(ncoef, sizeof(double));

  /* Initial guess for coefficients (all 1s) */
  linspace(coefficients, 1, 1, ncoef);

  /* Allocate space for breakpoints and initialize */
  nbps = 20;
  bps = calloc(nbps, sizeof(double));
  linspace(bps, 0, 5, nbps);

  /* 
   * NTG internal memory
   *
   * These calculations do not need to be changed.
   */
  istate = calloc((ncoef + NLIC + NLFC + NLTC * nbps + NNLIC +
		   NNLTC * nbps + NNLFC), sizeof(int));
  clambda = calloc((ncoef + NLIC + NLFC + NLTC * nbps + NNLIC +
		    NNLTC * nbps + NNLFC), sizeof(double));
  R = calloc((ncoef + 1) * (ncoef + 1), sizeof(double));

  lic = DoubleMatrix(NLIC, MAXDERIV * NOUT);
  lfc = DoubleMatrix(NLFC, MAXDERIV * NOUT);
  double **bounds = DoubleMatrix(NOUT, MAXDERIV);

  /* 
   * Define the constraints
   *
   * This section defines the various constraints for the problem.
   * For the nonlinear constraints, these are indexed by the active
   * variable number.
   *
   * Format: [constraint number][active variable number]
   */

  /* Lane change manuever */
  double x0[3] = {0.0, -2.0, 0.0}, u0[2] = {8.0, 0};
  double xf[3] = {40.0, 2.0, 0.0}, uf[2] = {8.0, 0};

  /* Initial condition constraint: zflag given */
  for (i = 0; i < NLIC; ++i) { lic[i][i] = 1.0; }
  kincar_flat_forward(x0, u0, &default_params, bounds);
  for (i = 0; i < NOUT; ++i) {
    for (j = 0; j < MAXDERIV; ++j) {
      lowerb[i * MAXDERIV + j] = upperb[i * MAXDERIV + j] = bounds[i][j];
    }
  }
			       
  /* Final condition constraint: zflag given */
  for (i = 0; i < NLFC; ++i) { lfc[i][i] = 1.0; }
  kincar_flat_forward(xf, uf, &default_params, bounds);
  for (i = 0; i < NOUT; ++i) {
    for (j = 0; j < MAXDERIV; ++j) {
      lowerb[NLIC + i * MAXDERIV + j] =
	upperb[NLIC + i * MAXDERIV + j] = bounds[i][j];
    }
  }

  // PrintVector("lowerb", lowerb, NLIC + NLTC + NLFC + NNLIC + NNLTC + NNLFC);
  // PrintVector("upperb", upperb, NLIC + NLTC + NLFC + NNLIC + NNLTC + NNLFC);

  npsoloption("summary file = 0");
  ntg(NOUT, bps, nbps, ninterv, knots, order, mult, maxderiv,
      coefficients,
      NLIC,                lic,
      NLTC,                NULL,
      NLFC,                lfc,
      NNLIC,               NULL,
      NNLTC,               NULL,
      NNLFC,               NULL,
      NINITIALCONSTRAV,    NULL,
      NTRAJECTORYCONSTRAV, NULL,
      NFINALCONSTRAV,      NULL,
      lowerb, upperb,
      NICF,		   NULL,
      NTCF,		   tcf,
      NFCF,		   NULL,
      NINITIALCOSTAV,      NULL,
      NTRAJECTORYCOSTAV,   trajectorycostav,
      NFINALCOSTAV,	   NULL,
      istate, clambda, R, &inform, &objective);
	
  PrintVector("coef1", coefficients, ncoef);

  #define PRINTCOEFS
  #ifdef PRINTCOEFS
  /* Print out the trajectory for the states and inputs */
  double **fz = DoubleMatrix(NOUT, MAXDERIV);
  double x[3], u[2];
  int ntimepts = 30;
  for (i = 0; i < ntimepts; ++i) {
    double time = 5.0 * (double) i / (ntimepts-1);
    for (j = 0; j < NOUT; ++j) {
      SplineInterp(fz[j], time, knots[j], ninterv[j],
		   &coefficients[j * ncoef/2], ncoef/2,
		   order[j], mult[j], maxderiv[j]);
    }
    kincar_flat_reverse(fz, &default_params, x, u);
    printf("%8.3g %8.3g %8.3g %8.3g %8.3g %8.3g\n",
	   time, x[0], x[1], x[2], u[0], u[1]);
  }
  FreeDoubleMatrix(fz);
  #endif
	
  FreeDoubleMatrix(lic);
  FreeDoubleMatrix(lfc);
  FreeDoubleMatrix(bounds);
  free(istate);	
  free(clambda);
  free(R);
  free(bps);	
  free(knots[0]); free(knots[1]);
  free(coefficients);
	
  return 0;
}
