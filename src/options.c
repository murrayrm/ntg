#define MXPARM 30

extern struct NPPAR2
{
  double rpsvnp[MXPARM],
  double cdint,    /* Central difference interval */
  double ctol,     /* Nonlinear feasibility */
  double dxlim,    /* Step Limit */
  double epsrf,    /* Function Precision */
  double eta,      /* Line search tolerance */
  double fdint,    /* Difference interval */
  double ftol,     /* Optimality tolerance */
  double hcndbd,
  double rpadnp[22]
} nppar2_
