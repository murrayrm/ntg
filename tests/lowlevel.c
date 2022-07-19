/*
 * lowlevel.c - C code for low level tests
 * RMM, 16 Jul 2022
 *
 * This file contains cost & constraint functions for use with lowlevel_test.py
 */

#include <math.h>

#define NOUT	    2			/* number of outputs */
#define MAXDERIV    3			/* number of derivatives + 1 */

double ifc_weight = 100.0;		/* weight for initial/final cost */

/* Planar system with curvature as the cost function */
void tcf_2d_curvature(
int *mode, int *nstate, int *i, double *f, double *df, double **zp)
{
  if (*mode == 0 || *mode == 2) {
    /* compute cost function: curvature */
    *f = zp[0][2] * zp[0][2] + zp[1][2] * zp[1][2];
  }

  if (*mode == 1 || *mode == 2) {
    /* compute gradient of cost function (index = active variables) */
    df[0] = 0; df[1] = 0; df[2] = 2 * zp[0][2];
    df[3] = 0; df[4] = 0; df[5] = 2 * zp[1][2];
  }
}

/* Utility function to define quadratic cost */
void nl_2d_quadratic_cost(
double zd[NOUT][MAXDERIV], int *mode, double *f, double *df, double **zp)
{
  if (*mode == 0 || *mode == 2) {
    /* compute cost function: square distance from initial value */
    *f = 0;
    for (int i = 0; i < NOUT; ++i)
      for (int j = 0; j < MAXDERIV; ++j)
	*f += pow(zp[i][j] - zd[i][j], 2.0) * ifc_weight;
  }

  if (*mode == 1 || *mode == 2) {
    /* compute gradient of cost function (index = active variables) */
    for (int i = 0; i < NOUT; ++i)
      for (int j = 0; j < MAXDERIV; ++j)
	df[i * MAXDERIV + j] = 2 * ifc_weight * (zp[i][j] - zd[i][j]);
  }
}

/* Planar system with final condition at the origin */
void nl_2d_initial_cost(
int *mode, int *nstate, double *f, double *df, double **zp)
{
  double z0[NOUT][MAXDERIV] = {{0, 0, 0}, {-2, 0, 0}};
  nl_2d_quadratic_cost(z0, mode, f, df, zp);
}

/* Planar system with final condition at the origin */
void nl_2d_final_cost(
int *mode, int *nstate, double *f, double *df, double **zp)
{
  double zf[NOUT][MAXDERIV] = {{40, 0, 0}, {2, 0, 0}};
  nl_2d_quadratic_cost(zf, mode, f, df, zp);
}
