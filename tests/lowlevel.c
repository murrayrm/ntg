/*
 * lowlevel.c - C code for low level tests
 * RMM, 16 Jul 2022
 *
 * This file contains cost & constraint functions for use with lowlevel_test.py
 */

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
