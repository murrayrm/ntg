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
costs.h
*/

#ifndef _COSTS_H_
#define _COSTS_H_

#include "av.h"
#include "colloc.h"
#include "integrator.h"
#include "matrix.h"


void InitialCost(
int *mode,int *nstate,
double *I,double *dI,
void (*func)(int *,int *,double *,double *,double **),
ConcatColloc *ccolloc,double *Z);

void IntegratedCost(
int *mode,int *nstate,
double *I,double *dI,double *bps,
void (*func)(int *,int *,int *,double *,double *,double **),
ConcatColloc *ccolloc,double *Z);

void FinalCost(
int *mode,int *nstate,
double *F,double *dF,
void (*func)(int *,int *,double *,double *,double **),
ConcatColloc *ccolloc,double *Z);

/*
InitialCost()
-------------
computes the value of the initial cost function

mode: action to take
nstate: is this first time the cost is being calculated?
I: value of the initial cost functions size(I)=[1]
dI: Jacobian of I size(dI)=[1,nout*colloc->cols]
func: the initial cost function
colloc,Z: the collocation matrix and the vector of outputs Z.
nout: number of outputs
nz: Sum of the number of variables (sum over i max_deriv(i))
nZ: Sum of the number of breakpoints (nz*nbps)
nc: Sum of the number of B-spline coefficients

IntegratedCost()
----------------
computes the value of the integrated cost function

mode: action to take
I: value of the integrated cost functions size(I)=[1]
dI: place to store Jacobian of I. size(dI)=[1,nout*colloc->cols]
bps: the breakpoints. size [colloc->nbps]
func: the unintegrated cost function
colloc,Z: the collocation matrix and the vector of outputs Z.
bps: breakpoints vector. size [colloc->nbps]
nout: number of outputs
nz: Sum of the number of variables (sum over i max_deriv(i))
nc: Sum of the number of B-spline coefficients


FinalCost()
-----------
computes the value of the final cost function

mode: action to take
F: place to store value of the final cost function size(F)=1
dF: place to store jacobian of F. size(dF)=[1,nout*colloc->cols]
func: fincal cost function
Colloc,Z: the collocation Matrix and the vector of outputs Z.
nout: number of outputs
nz: Sum of the number of variables (sum over i max_deriv(i))
nZ: Sum of the number of breakpoints (nz*nbps)
nc: Sum of the number of B-spline coefficients


*/

#endif
