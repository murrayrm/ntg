/*
 * vanderpol.c - example optimization for NTG
 * MBM, ??
 *
 * This file is an example of how to use NTG.  It contains at a 
 * model for a driven van der Pol oscillator and tries to minimize
 * a quadratic cost subject to a set of boundary conditions.  See the
 * README file for more information.
 * 
 * RMM, 1 Feb 02: added comments to example file
 */

#include <stdlib.h>
#include <math.h>			/* math functions */
#include "ntg.h"			/* main NTG declarations */

#define NOUT		1		/* number of (flat) outputs */
#define NINTERV	        2		/* number of intervals, lj */
#define MULT		3		/* regularity of splines, mj */
#define ORDER		5		/* degree of spline polynomial, kj */
#define MAXDERIV	3		/* highest derived required + 1 */

/*
 * Total number of coefficients required
 *
 * Sum_j [lj*(kj-mj) + mj]
 *
 */
#define NCOEF		7		/* total # of coeffs */

/*
 * Give names to the different outputs and their derivatives
 *
 * The zp label is used in the function calls below.  This allows us to
 * use simpler labelling.
 */
#define z		zp[0][0]
#define zd		zp[0][1]
#define zdd		zp[0][2]

/* number linear constraints */
#define NLIC		2		/* linear initial constraints */
#define NLTC		0		/* linear trajectory constraints */
#define NLFC		1		/* linear final constraints */

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
#define NUCF		1		/* unintegrated */
#define NFCF		0		/* final */

/* Same as above, now for the cost functions */
#define NINITIALCOSTAV		0	/* initial */
#define NTRAJECTORYCOSTAV	3	/* trajectory */
#define NFINALCOSTAV		0	/* final */

/*
 * Now declare all of the active variables that are required for the
 * nonlinear constraints [none here] and the cost functions
 *
 */
static AV trajectorycostav[NTRAJECTORYCOSTAV]={{0,0},{0,1},{0,2}};

/* Declare the function used to compute the unintegrated cost */
static void ucf(int *mode,int *nstate,int *i,double *f,double *df,double **zp);

int main(void)
{
  double *knots[NOUT];			/* knot points, list of times for
					   each output */

  /*
   * Now define all of the NTG parameters that were listed above
   */
  int ninterv[NOUT]=	{NINTERV};
  int mult[NOUT]=	{MULT};
  int maxderiv[NOUT]=	{MAXDERIV};
  int order[NOUT]=	{ORDER};
  int nbps;
  double *bps;
  double **lic,**lfc;

  /* initial guess size = sum over each output ninterv*(order-mult)+mult */
  int ncoef;
  double *coefficients;
  int *istate;
  double *clambda;
  double *R;
	
  double lowerb[NLIC+NLTC+NLFC+NNLIC+NNLTC+NNLFC];
  double upperb[NLIC+NLTC+NLFC+NNLIC+NNLTC+NNLFC];
  int inform;
  double objective;

  /*
   * Allocate space and initialize the knot points
   *
   * For this problem, regularly spaced from t=0 to t=5
   *
   */
  knots[0]=calloc((ninterv[0]+1), sizeof(double));
  linspace(knots[0], 0, 5, ninterv[0]+1);

  /*
   * Compute the number of spline coefficients required
   *
   * WARNING: this formula was computed above and the formula below
   * is specific to this particular case
   */
  ncoef=ninterv[0]*(order[0]-mult[0])+mult[0];
  coefficients=calloc(ncoef, sizeof(double));

  /* Initial guess for coefficients (all 1s) */
  linspace(coefficients,1,1,ncoef);

  /* Allocate space for breakpoints and initialize */
  nbps=20;
  bps=calloc(nbps, sizeof(double));
  linspace(bps,0,5,nbps);

  /* 
   * NTG internal memory
   *
   * These calculations do not need to be changed.
   */
  istate=calloc((ncoef+NLIC+NLFC+NLTC*nbps+NNLIC+NNLTC*nbps+NNLFC),sizeof(int));
  clambda=calloc((ncoef+NLIC+NLFC+NLTC*nbps+NNLIC+NNLTC*nbps+NNLFC),sizeof(double));
  R=calloc((ncoef+1)*(ncoef+1),sizeof(double));

  lic = DoubleMatrix(NLIC,maxderiv[0]);
  lfc = DoubleMatrix(NLFC,maxderiv[0]);


  /* 
   * Define the constraints
   *
   * This section defines the various constraints for the problem.  For
   * the linear constraints, these are indexed by the active variable number.
   *
   * Format: [constraint number][active variable number]
   */

  /* Initial condition constraints */
  lic[0][0]=1.0;			/* z */
  lowerb[0]=upperb[0]=1.0;		/* equality constraint */

  lic[1][1]=1.0;			/* zdot */
  lowerb[1]=upperb[1]=0.0;		/* equality constraint */

  /* Final constraint: -x1(5) + x2(5) - 1 = 0 */
  lfc[0][0]=-1.0;
  lfc[0][1]=1.0;
  lowerb[2]=upperb[2]=1.0;		/* equality constraint */

  npsoloption("summary file = 0");
  ntg(NOUT,bps,nbps,ninterv,knots,order,mult,maxderiv,
		coefficients,
		NLIC,						lic,
		NLTC,						NULL,
		NLFC,						lfc,
		NNLIC,					NULL,
		NNLTC,					NULL,
		NNLFC,					NULL,
		NINITIALCONSTRAV,		NULL,
		NTRAJECTORYCONSTRAV,	NULL,
		NFINALCONSTRAV,		NULL,
		lowerb,upperb,
		NICF,						NULL,
		NUCF,						ucf,
		NFCF,						NULL,
		NINITIALCOSTAV,		NULL,
		NTRAJECTORYCOSTAV,	trajectorycostav,
		NFINALCOSTAV,			NULL,
		istate,clambda,R,&inform,&objective);
	
	PrintVector("coef1",coefficients,ncoef);
	
	FreeDoubleMatrix(lic);
	FreeDoubleMatrix(lfc);
	free(istate);	
	free(clambda);
	free(R);
	free(bps);	
	free(knots[0]);
	free(coefficients);
	
	return 0;
}

void ucf(int *mode,int *nstate,int *i,double *f,double *df,double **zp)
{
	double t1,t2,t4,t6;
	switch(*mode)
	{
		case 0:
      t1 = z*z;
      t2 = zd*zd;
      t6 = pow(zdd+z-(1.0-t1)*zd,2.0);
      *f = t1/2.0+t2/2.0+t6/2.0;
		break;
		
		case 1:
      t1 = z*z;
      t2 = 1.0-t1;
      t4 = zdd+z-t2*zd;
      df[0] = z+t4*(1.0+2.0*z*zd);
      df[1] = zd-t4*t2;
      df[2] = t4;
		break;
		
		case 2:
      t1 = z*z;
      t2 = zd*zd;
      t6 = pow(zdd+z-(1.0-t1)*zd,2.0);
      *f = t1/2.0+t2/2.0+t6/2.0;

      /*t1 = z*z; this is calculated above */
      t2 = 1.0-t1;
      t4 = zdd+z-t2*zd;
      df[0] = z+t4*(1.0+2.0*z*zd);
      df[1] = zd-t4*t2;
      df[2] = t4;
		break;
	}
}
