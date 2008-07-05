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



/*
* ntg.h
*
* Trajectory generation parameters
*/

#ifndef _NTJ_H_
#define _NTJ_H_
#define MAXNOUT 5


#include <stdio.h>
#include <float.h>
#include "av.h"
#include "colloc.h"
#include "constraints.h"
#include "cost.h"
#include "matrix.h"

/*
nout: number of outputs
bps,nbps: an array with the breakpoint sequence and the number of breakpoints
ninterv: number of intervals for each output
knots: knot points for each output
order: order of spline outputs
mult: continuity condition of the outputs
maxderiv: maximum derivative required for each output
initialguess: self-explanatory
nlic: number of linear initial constraints.
nlic,lic: linear initial constraints. size[nlic,sum(maxderiv,over all outputs)]
nltc,ltc: linear trajectory constraints. size(nltc,sum(maxderiv));
nlfc,lfc: linear final constraints. size(nlfc,sum(maxderiv));

nnlic: no. of nonlinear initial constraints.
nlicf: nonlinear initial constraint function.
nnltc: no. of nonlinear trajectory constraints.
nltcf: nonlinear trajectory constraint function.
nnlfc: no. of nonlinear final constraints.
nlfcf: nonlinear final constraint function.
constrAV,nconstrAV: the "active" variables used in evaluating the
  constraints.

upperb,lowerb: upper and lower bounds on the constraints.
  size[nlic+nltc+nlfc+nnlic+nnltc+nnlfc]
 
icf: initial cost function
ucf: unintegrated cost function
fcf: final cost function
costAV,ncostAV: the variables used in evaluating the costs.

istate (int *) : an array of length
   ncoeff + nlic+nltc*nbps+nlfc + nnlic+nnltc*nbps+nnlfc
clambda (double *): an array of the same length as istate.
R (double *)   :
	an array length ncoef*ncoef

*/

void ntg(
int nout,double *bps, int nbps,int *kninterv, double **knots,
int *order,int *mult,int *max_deriv,
double *initialguess,

int nlic,double **lic,
int nltc,double **ltc,
int nlfc,double **lfc,

int nnlic,void (*nlicf)(int *,int *,double *,double **,double **),
int nnltc,void (*nltcf)(int *,int *,int *,double *,double **,double **),
int nnlfc,void (*nlfcf)(int *,int *,double *,double **,double **),
int ninitialconstrav,AV *initialconstrav,
int ntrajectoryconstrav,AV *trajectoryconstrav,
int nfinalconstrav,AV *finalconstrav,

double *lowerb,double *upperb,

int nicf,void (*icf)(int *,int *,double *,double *,double **),
int nucf,void (*ucf)(int *,int *,int *,double *,double *,double **),
int nfcf,void (*fcf)(int *,int *,double *,double *,double **),
int ninitialcostav,AV *initialcostav,
int ntrajectorycostav,AV *trajectorycostav,
int nfinalcostav,AV *finalcostav,

int *istate,double *clambda,double *R,
int *inform,double *objective
);
void npsoloption(char *type);

/* see matlab's linspace() */
void linspace(double *v,double d0,double d1,int n);
void printNTGBanner(void);

#endif
