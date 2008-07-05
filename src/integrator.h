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



#ifndef _INTEGRATOR_H_
#define _INTEGRATOR_H_

#include "matrix.h"

#define FEULER		0
#define BEULER		1
#define TRAPEZOID	2

void IntegrateVector(double *I,double *f,double *t,int n,int type);
void IntegrateFMatrixCols(double *I,FMatrix *f,double *t,int type);

/*
IntegrateVector()
-----------------
IntegrateVector(double *I,double *f,double *t,int n,int type)
integrate a vector of values

I: place to store the integral. size(I)=[1]
f: function values size(f)=[n]
t: independent variable values size(t)=[n]
n: length of the arrays f and x
type: type of integrator to use

IntegrateMatrixCols()
---------------------
IntegrateMatrixCols(double *I,Matrix *f,double *t,int type)

integrate each column of f

I: place to store the integral. size(I)=[matrix->cols]
f: function values
t: independent variable values size(t)=[matrix->rows]
type: type of integrator

*/


#endif
