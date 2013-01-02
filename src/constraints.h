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
* constraints.h
*
* functions to calculate the linear constraint matrix and bounds
*
*/

#ifndef _CONSTRAINTS_H_
#define _CONSTRAINTS_H_

#include <sys/malloc.h>
#include "colloc.h"
#include "matrix.h"

void bounds(
double *bbar,double *b,int nC,
int nlic,int nltc,int nlfc,
int nnlic,int nnltc,int nnlfc,
int nbps,double bigbnd);

void NonLinearConstraints(
int *mode,int *nstate,
double *nlc,FMatrix *dnlc,
int nnlic,void (*nlicf)(int *,int *,double *,double **,double **),
int nnltc,void (*nltcf)(int *,int *,int *,double *,double **,double **),
int nnlfc,void (*nlfcf)(int *,int *,double *,double **,double **),
ConcatColloc *ccolloc,double *Z);

void NonLinearInitialConstraints(
int *mode,int *nstate,
int nnlic,double *nlic,FMatrix *dIdC,
void (*nlicf)(int *,int *,double *,double **,double **),
ConcatColloc *ccolloc,double *Z);

void NonLinearTrajectoryConstraints(
int *mode,int *nstate,
int nnltc,double *nltc,FMatrix *dnltc,
void (*nltcf)(int *,int *,int *,double *,double **,double **),
ConcatColloc *ccolloc,double *Z);

void NonLinearFinalConstraints(
int *mode,int *nstate,int nnlfc,double *nlfc,FMatrix *dnlfc,
void (*nlfcf)(int *,int *,double *,double **,double **),
ConcatColloc *ccolloc,double *Z);

void LinearConstraintsMatrix(
FMatrix *lc,FMatrix *lic,FMatrix *ltc,FMatrix *lfc,ConcatColloc *ccolloc);

void InitialConstraintsMatrix(FMatrix *dIdC,FMatrix *dIdz,ConcatColloc *ccolloc);
void TrajectoryConstraintsMatrix(FMatrix *dIdC,FMatrix *dIdz,
	ConcatColloc *ccolloc);
void FinalConstraintsMatrix(FMatrix *dIdC,FMatrix *dIdz,ConcatColloc *colloc);

/*
bounds()
--------
void bounds(double *bbar,double *b,int nc,int nlic,int nltc,int nlfc,
int nnlic,int nnltc,int nnlfc,int nbps,double bigbnd)
fills the bounds vector that will be passed to NPSOL
recall the actual values of the coefficients are unconstrained so
set the bounds to +/-bigbnd

bbar: vector to be filled. size [nc+nlic+nltc*nbps+nlfc+nnlic+nnltc*nbps+nnlfc]
b: user defined bounds. size [nlic+nltc+nlfc+nnlic+nnltc+nnlfc]
nc: number of coefficients
nlic,nltc,nlfc: number of linear (initial,trajectory,final) constraints.
nnlic,nnltc,nnlfc: number of nonlinear (initial,trajectory,final) constraints.
nbps: number of breakpoints
bigbnd: the value to set the first nc elements to.

NonLinearConstraints()
----------------------
fills the nonlinear constraints vector with values of the constraints

mode: action to take. If mode is 0...
nlc: values of the nonlinear constraints. size(nlc)=[nnlic+nnltc*nbps+nnlfc].
dnlc: Jacobian of nlc. size(dnlc)=[size(nlc),ccolloc->nC]
nnlic, nlicf: number of nonlinear initial constraints and the user defined
  nonlinear intial constraints function.
nnltc, nltcf: number of nonlinear trajectory constraints and the user defined
  nonlinear trajectory constraints function.
nnlfc, nlfcf: number of nonlinear final constraints and the user defined
  nonlinear final constraints function.
ccolloc,Z: the concatenated collocation matrix and the vector of output Z's

NonLinearInitialConstraints()
-----------------------------
fills the nonlinear initial constraints vector values of the constraints

mode: action to take
nnlic,nlic,dnlic: place to store function values and their derivatives.
  size(nlic)=nnlic. size(dnlic)=[nnlic,ccolloc->nC]
nlicf: user defined function to calculate nonlinear initial constraints
ccolloc,Z: concatenated collocation matrix and vector of output Z's

NonLinearTrajectoryConstraints()
--------------------------------
fills a vector with the values of the nonlinear trajectory constraints.

mode: action to take
nnltc,nltc,dnltc: place to store function values and their derivatives.
  size(nltc)=nbps*nnltc. size(dnltc)=[nbps*nnltc,ccolloc->nC]
nltcf: user defined function to calculate nonlinear trajectory constraints
ccolloc,Z: concatenated collocation matrix and vector of output Z's

NonLinearFinalConstraints
-------------------------
fills a vector with the values of the nonlinear final constraints.

mode: action to take
nnlfc,nlfc,dnlfc: place to store function values and their derivatives.
  size(nlfc)=nnlfc. size(dnlfc)=[nnlfc,ccolloc->nC]
nlfcf: user defined function to calculate nonlinear final constraints.
ccolloc,Z: collocation matrix and vector of output Z's

LinearConstraintsMatrix()
-----------------
fills the linear constraint matrix that will be passed to npsol
arguments
lc: pointer to the matrix to be filled. Must be of
  size [nlic+nltc*nbps+nlfc][nout*colloc->cols]
lic: initial time linear constraints. size [nlci,ccolloc->nz]. 
ltc: trajectory linear constraints. size [nlcj,colloc->nz]. 
lfc: final time linear constraints. size [nlcf,colloc->nz]. 
ccolloc: concatenated collocation matrix

InitialConstraintsMatrix()
-----------------------------
takes the user specified trajectory constraint matrix
and returns the linear trajectory constraint matrix

dIdC: (output) linear constraints matrix. Must point to a matrix of size
  [nbps*nlic,ccolloc->nC]
dIdz: (input) user specified constraint matrix. These
  linear constraints are applied only to the first breakpoint.
  Assumed to be of size [nlic,ccolloc->nz]
ccolloc: (input) the collocation matrix

TrajectoryConstraintsMatrix()
-----------------------------
takes the user specified trajectory constraint matrix
and returns the linear trajectory constraint matrix

dIdC: (output) linear tj constr mat. Must point to a matrix of size
  [nbps*nltj,ccolloc->nC]
dIdz: (input) user specified constraint matrix. These
  linear constraints are applied to every breakpoint.
  Assumed to be of size [nltj,ccolloc->nz]
ccolloc: (input) the concatenated collocation matrix
nout: (input) the number of outputs

FinalConstraintsMatrix()
-----------------------------
takes the user specified trajectory constraint matrix
and returns the linear trajectory constraint matrix

dIdC: (output) linear constraints matrix. Must point to a matrix of size
  [nbps*nlfc,ccolloc->nC]
dIdz: (input) user specified constraint matrix. These
  linear constraints are applied only to the last breakpoint of each output.
  Assumed to be of size [nlic,ccolloc->nz]
ccolloc: (input) the concatenated collocation matrix
*/

#endif
