/*
 * kincar.c - kinematic car example for NTG
 * RMM, 27 Jun 2022
 *
 * This file is an example of how to use NTG to generate an optimal
 * trajectory for a kinematic car performing a lane change operation.  A
 * description of the examples worked out here is available in Chapter 2 of
 * the FBS Optimization-Based Control Supplement:
 *
 *  https://fbswiki.org/wiki/index.php/OBC
 * 
 * Usage:
 *
 *   ./kincar [-i]
 *
 * Options:
 *   -i       interactive mode, prompts for initial and final conditions
 *   -o file  save optimal trajectory to file
 *   -v       turn out verbose output (NTG and solver)
 *   --ipopt  solve using IPOPT (instead of NPLSOL) [not implemented]
 *
 */

#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include "ntg.h"

/*
 * Vehicle dynamics 
 *
 * This section contains the description of the dynamics of the vehicle in
 * terms of the differentially flat outputs, allowing conversion from the
 * flat flag to the system state and input.
 *
 */

/* Vehicle dynamics parameters */
struct kincar_params {
  double b;				/* wheel based, default = 3 m */
  char dir;				/* travel direction ('f' or 'r') */
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

  /* Vehicle angle determined by the tangent of the flat output curve */
  if (dir == 'f') {
    x[2] = atan2(zflag[1][1], zflag[0][1]);	/* tan(theta) = ydot/xdot */
  } else if (dir == 'r') {
    /* Angle is flipped by 180 degrees (since v < 0) */;
    x[2] = atan2(-zflag[1][1], -zflag[0][1]);
  } else {
    fprintf(stderr, "unknown direction: %c", dir);
  }

  /* Next solve for the inputs, based on curvature */
  thdot_v = zflag[1][2] * cos(x[2]) - zflag[0][2] * sin(x[2]);
  u[0] = zflag[0][1] * cos(x[2]) + zflag[1][1] * sin(x[2]);
  u[1] = atan2(thdot_v, pow(u[0], 2.0) / b);
}

/*
 * Optimal control problem setup
 * 
 * For this system we solve a point-to-point optimal control problem in
 * which we minmimze the integrated curvature of the trajectory between the
 * inital and final positions.  This corresponds roughly to minimizing the
 * inputs to the vehicle (velocity and steering angle).
 *
 */

/* Lane change manuever */
double x0[3] = {0.0, -2.0, 0.0}, u0[2] = {8.0, 0};
double xf[3] = {40.0, 2.0, 0.0}, uf[2] = {8.0, 0};
double Tf = 5.0;

/* Trajectory cost function (unintegrated) */
void tcf(int *mode, int *nstate, int *i, double *f, double *df, double **zp)
{
  if (*mode == 0 || *mode == 2) {
    /* compute cost function: curvature */
    *f = zp[0][2] * zp[0][2] + zp[1][2] * zp[1][2];
  }

  if (*mode == 1 || *mode == 2) {
    /* compute gradient of cost function (index = flat variables) */
    df[0] = 0; df[1] = 0; df[2] = 2 * zp[0][2];
    df[3] = 0; df[4] = 0; df[5] = 2 * zp[1][2];
  }
}

/* Nonlinear constraint: corridor between start and end */

void nltcf_corridor(
int *mode, int *nstate, int *i, double *f, double **df, double **zp)
{
  double m = (xf[1] - x0[1]) / (xf[0] - x0[0]);
  double b = x0[1];

  if (*mode == 0 || *mode == 2) {
    /* Compute the distance from the line connecting start to end */
    double d = m * (zp[0][0] - x0[0]) + b - zp[1][0];
    f[0] = d; f[1] = d;
  }

  if (*mode == 1 || *mode == 2) {
    /* Compute gradient of constraint function (2nd index = flat variables) */
    df[0][0] = m;  df[0][1] = df[0][2] = 0;
    df[0][3] = -1; df[0][4] = df[0][5] = 0;

    df[1][0] = m;  df[1][1] = df[1][2] = 0;
    df[1][3] = -1; df[1][4] = df[1][5] = 0;
  }
}

/* Nonlinear constraint: two obstacles along the path */

/* 
 * NTG problem setup
 *
 * This section defaults all of the parameters that are used to set up the
 * optimization problem that NTG solves, including the number of intervals,
 * order of the splines, and number of free variables.
 *
 * Note: NTG allows each of the outputs to have different values for the
 * number of intervals, multiplicity, order, etc.  Here we will use the same
 * values for both outputs (the arrays are defined at the top of the main()
 * function, below).
 *
 */

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
#define NNLTC		2		/* nonlinear trajectory constraints */
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
#define NTRAJECTORYCONSTRAV	2	/* active variables, trajectory */
#define NFINALCONSTRAV		0	/* active variables, final */

/* number of cost functions */
#define NICF		0		/* initial */
#define NTCF		1		/* trajectory (unintegrated) */
#define NFCF		0		/* final */

/* Same as above, now for the cost functions */
#define NINITIALCOSTAV		0	/* initial */
#define NTRAJECTORYCOSTAV	2	/* trajectory */
#define NFINALCOSTAV		0	/* final */

/*
 * Now declare all of the active variables that are required for the
 * nonlinear constraints and the cost functions
 *
 */

static AV trajectorycostav[NTRAJECTORYCOSTAV] =
  {{0, 2}, {1, 2}};			/* second deriv's of flat outputs */

static AV trajectoryconstrav[NTRAJECTORYCONSTRAV] =
  {{0, 0}, {1, 0}};			/* second deriv's of flat outputs */

/*
 * Main program
 *
 * This is the actual program that is called to compute the optimal
 * trajectory and print out the result.
 *
 */

int main(int argc, char **argv)
{
  int i, j;				/* indices for iterating */

  /*
   * Now define all of the NTG parameters that were listed above
   *
   */
  int ninterv[NOUT] =	{NINTERV, NINTERV};
  int mult[NOUT] =	{MULT, MULT};
  int maxderiv[NOUT] =	{MAXDERIV, MAXDERIV};
  int order[NOUT] =	{ORDER, ORDER};
  double *knots[NOUT];			/* knot points for each output */
  int nbps = 20;			/* number of breakpoints */
  double *bps;				/* breakpoint times */
  double **lic, **lfc;			/* linear initial/final constraints */

  int ncoef;				/* number of B-spline coefficients */
  double *coefficients;			/* vector of coefficients (stacked) */
  int *istate;				/* NTG internal memory */
  double *clambda;			/* NTG internal memory */
  double *R;				/* NTG internal memory */

  /* Upper and lower bound vectors (stacked) */
  double lowerb[NLIC + NLTC + NLFC + NNLIC + NNLTC + NNLFC];
  double upperb[NLIC + NLTC + NLFC + NNLIC + NNLTC + NNLFC];

  int inform;				/* output from NPSOL */
  double objective;			/* optimial cost value */

  /*
   * Process command line arguments
   *
   */
  int opt, interactive = 0, verbose = 0;
  FILE *outfp = stdout;
  while ((opt = getopt(argc, argv, "io:v")) != -1) {
    switch (opt) {
    case 'i':				/* interactive mode */
      interactive = 1;
      break;

    case 'o':				/* set output file */
      outfp = fopen(optarg, "w");
      if (outfp == NULL) {
	fprintf(stderr, "%s: can't open output file '%s'\n", argv[0], optarg);
	exit(EXIT_FAILURE);
      }
      break;

    case 'v':
      verbose = 1;
      break;

    default:
      fprintf(stderr, "Usage: %s [-i] [-o file] [-v]\n", argv[0]);
      exit(EXIT_FAILURE);
    }
  }

  /*
   * Allocate space and initialize the knot points
   *
   * For this problem, regularly spaced from t=0 to t=5
   *
   */
  for (i = 0; i < NOUT; ++i) {
    knots[i] = calloc((ninterv[i] + 1), sizeof(double));
    linspace(knots[i], 0, Tf, ninterv[i] + 1);
  }

  /*
   * Compute the number of spline coefficients required
   *
   * This formula was given above (manually) and so this serves as a check
   * to make sure everything lines up.
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
  bps = calloc(nbps, sizeof(double));
  linspace(bps, 0, Tf, nbps);

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

  Matrix *lic_m = MakeMatrix(NLIC, MAXDERIV * NOUT); lic = lic_m->elements;
  Matrix *lfc_m = MakeMatrix(NLFC, MAXDERIV * NOUT); lfc = lfc_m->elements;
  Matrix *bounds_m = MakeMatrix(NOUT, MAXDERIV);
  double **bounds = bounds_m->elements;

  /* 
   * Define the initial and final conditions
   *
   * This section defines the initial and final conditions by setting up a
   * set of linear initial/final constraints.  Since the flat outputs and
   * their derivatives are determine by the state and inputs, constraining
   * the initial and final state corresponds to constraining all of the flat
   * outputs and their derivatives up to third order.
   *
   * TODO: add trajectory constraints on the input variables (indexed by
   * active variable).
   */

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

  if (verbose) {
    fprintf(stdout, "\nLinear initial constraints (LIC) matrix:\n");
    PrintMatrix("stdout", lic_m);
    fprintf(stdout, "\nLinear final constraints (LFC) matrix:\n");
    PrintMatrix("stdout", lfc_m);
    fprintf(stdout, "\nUpper and Lower Bounds:\n");
    PrintVector("stdout", lowerb, NLIC + NLTC + NLFC + NNLIC + NNLTC + NNLFC);
  }

  /*
   * Define the trajectory constraints
   *
   */
  double corridor_radius = 2;
  lowerb[NLIC + NLFC + 0] = -corridor_radius;
  upperb[NLIC + NLFC + 0] = 1e10;	/* no bound */
  lowerb[NLIC + NLFC + 1] = -1e10;	/* no bound */
  upperb[NLIC + NLFC + 1] = corridor_radius;

  /*
   * Call NTG
   *
   * Now that the problem is set up, we call the main NTG function.  The
   * coefficients array contains the initial guess (all ones, set above) and
   * returns the final value of the optimal solution (if successful).
   *
   */

  /* Set NPSOL options */
  npsoloption("nolist");		/* turn off NPSOL listing */
  if (!verbose) {
    npsoloption("print level 0");	/* only print the final solution */
  }
  npsoloption("summary file = 0");	/* ??? */

  /* Call NTG */
  ntg(NOUT, bps, nbps, ninterv, knots, order, mult, maxderiv,
      coefficients,
      NLIC,                lic,
      NLTC,                NULL,
      NLFC,                lfc,
      NNLIC,               NULL,
      NNLTC,               nltcf_corridor,
      NNLFC,               NULL,
      NINITIALCONSTRAV,    NULL,
      NTRAJECTORYCONSTRAV, trajectoryconstrav,
      NFINALCONSTRAV,      NULL,
      lowerb, upperb,
      NICF,		   NULL,
      NTCF,		   tcf,
      NFCF,		   NULL,
      NINITIALCOSTAV,      NULL,
      NTRAJECTORYCOSTAV,   trajectorycostav,
      NFINALCOSTAV,	   NULL,
      istate, clambda, R, &inform, &objective);

  /*
   * Print out/store the results for later use.
   *
   */

  /* Print out the trajectory for the states and inputs */
  double **fz = DoubleMatrix(NOUT, MAXDERIV);
  double x[3], u[2];
  int ntimepts = 30;
  for (i = 0; i < ntimepts; ++i) {
    double time = Tf * (double) i / (ntimepts-1);
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

  /* Free up storage we used (not really needed here, but why not) */
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
