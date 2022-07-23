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
* colloc.h
*
* routines for the collocation matrix
*
*/

#ifndef _COLLOC_H_
#define _COLLOC_H_

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "matrix.h"
#include "av.h"

extern void interv_();
extern void knots_();
extern void bsplvd_();

extern struct side
{
	int m;
	int iside;
	double xside[10];
}side_;

typedef struct BlockStruct
{
	FMatrix *matrix;
	int offset;
}Block;

typedef struct CollocStruct
{
	Block *block;
	int ninterv;
	int order;
	int mult;
	int flaglen;
	int nbps;
	int rows;
	int cols;
} Colloc;

typedef struct ConcatCollocStruct
{
	Colloc **colloc;
	int nout;
	int nbps;
	int nz;
	int nZ;
	int nC;
	int *iZ;
	int *iz;
	int *iC;
} ConcatColloc;

ConcatColloc *ConcatCollocMatrix(
	int nout,double **knots,int *ninterv, double *bps, int nbps,
	int *flaglen,int *order,int *mult);
Colloc *CollocMatrix(
	double *knots,int ninterv,double *bps,
	int nbps,int flaglen,int order,int mult);
void FreeConcatColloc(ConcatColloc *ccolloc);
void FreeColloc(Colloc *colloc);
void CollocMult(FMatrix *B,FMatrix *A,Colloc *colloc);
void CollocConcatMult(FMatrix *B,FMatrix *A,ConcatColloc *ccolloc);
void CollocConcatMultT(FMatrix *dIdC,FMatrix *dIdZ,ConcatColloc *ccolloc);
void CollocConcatMultI(FMatrix *dIdC,FMatrix *dIdZ,ConcatColloc *ccolloc);
void CollocConcatMultF(FMatrix *dIdC,FMatrix *dIdZ,ConcatColloc *colloc);
double CollocElement(Colloc *colloc,int row,int col);
void CollocBlockMultVC(double *vout,double *v,Colloc *colloc,int b);
void PrintColloc(char *filename,Colloc *colloc);

double Zvalue(ConcatColloc *ccolloc,double *C,int output,int deriv,int bp);
int odb2lin(ConcatColloc *ccolloc,int output,int deriv,int bp);
int db2lin(Colloc *colloc,int deriv,int bp);
void lin2db(int *deriv,int *bp,Colloc *colloc,int i);
void updateZ(double *Z,ConcatColloc *ccolloc,double *C,AV *av,int nav,int type);

void dIdz2dIdZI(FMatrix *dIdZ,FMatrix *dIdz,ConcatColloc *ccolloc);
void dIdz2dIdZT(FMatrix *dIdZ,FMatrix *dIdz,ConcatColloc *ccolloc,int bp);
void dIdz2dIdZF(FMatrix *dIdZ,FMatrix *dIdz,ConcatColloc *ccolloc);
void Z2zpI(double **zp,double *Z,ConcatColloc *ccolloc);
void Z2zpT(double **zp,double *Z,ConcatColloc *ccolloc,int bp);
void Z2zpF(double **zp,double *Z,ConcatColloc *ccolloc);

void SplineInterp(
double *f,double x,double *knots,int ninterv,double *coeffs,int ncoeffs,
int order,int mult,int flaglen);

/*
CollocMatrix()
--------------
returns an array of sub blocks representing the collocation
matrix. The number of sub blocks is equal to the number
of breakpoints.

FreeColloc()
------------
frees memory allocated for a collocation matrix created using
CollocMatrix().

CollocElement()
---------------
double CollocElement(Colloc *colloc,int row,int col);
return the element of the collocation matrix at position
row,col

CollocMultAC1()
--------------
void CollocMultAC1(Matrix *B,Matrix *A,Colloc *colloc)
multiplies the matrix A and the collocation matrix. stores the result in B.

so, A->cols=colloc->rows and B assumed to be of size [A->rows,colloc->cols]

CollocMultAC2()
--------------
void CollocMultAC2(Matrix *B,Matrix *A,Colloc *colloc,int *k,int nk)
multiplies a matrix of special structure A and the collocation matrix
stores the result in B.

The structure of A is such that only certain columns in A have non-zero
elements. All the elements the other columns are.

cols[ncols] is an array of integers that describe which columns in A
are non-zero. The list of columns does not have to be ordered, but
of course all integers in the list must be 0 <= k < A->cols.

so, A->cols=colloc->rows and B assumed to be of size [A->rows,colloc->cols]

CollocConcatMult()
--------------
void CollocConcatMult(Matrix *B,Matrix *A,Colloc *colloc,int n)
multiplies a matrix A by the concatenated collocation matrix.
stores thre result in B. This is the brute force method. Assumes
the matrices to be dense. Use this routine if you think one
of the sparse routines might be wrong.

The concatenated collocation matrix is the matrix with the collocation
matrix n times on the diagonal, with zero entries elsewhere.

so, A->cols=n*colloc->rows and B assumed to be of size [A->rows,n*colloc->cols]

CollocConcatMultT()
--------------
void CollocConcatMultT(Matrix *B,Matrix *A,Colloc **colloc,int n, int nC)
multiplies a matrix A by the concatenated collocation matrix.
stores the result in B.

This method assumes A is sparse (ie doesn't perform every multiplication
Matrix A should be of the form created of dIdZ in
NonLinearTrajectoryConstraints() or TrajectoryConstraintsMatrix()

The concatenated collocation matrix is the matrix with the collocation
matrix n times on the diagonal, with zero entries elsewhere.

so, A->cols=n*colloc->rows and B assumed to be of size [A->rows,n*colloc->cols]

CollocConcatMultI()
--------------
void CollocConcatMultI(Matrix *dIdc,Matrix *dIdZ,Colloc **colloc,int nout)
multiplies a matrix A by the concatenated collocation matrix.
stores the result in B.

This method assumes A is sparse (ie doesn't perform every multiplication
Matrix A should be of the form created of dIdZ in
NonLinearInitialConstraints() or InitialConstraintsMatrix()

The concatenated collocation matrix is the matrix with the collocation
matrix n times on the diagonal, with zero entries elsewhere.

so, A->cols=n*colloc->rows and B assumed to be of size [A->rows,n*colloc->cols]

CollocConcatMultF()
---------------------
void CollocConcatMultF(Matrix *B,Matrix *A,Colloc **colloc,int nout);
multiplies a matrix A by the concatenated collocation matrix.
stores the result in B.

This method assumes A is sparse (ie doesn't perform every multiplication
Matrix A should be of the form created of dIdZ in
NonLinearFinalConstraints() or FinalConstraintsMatrix()

The concatenated collocation matrix is the matrix with the collocation
matrix n times on the diagonal, with zero entries elsewhere.

so, A->cols=n*colloc->rows and B assumed to be of size [A->rows,n*colloc->cols]

CollocBlockMultVC()
-------------------
void CollocBlockMultVC(double *vout,double *v,Colloc *colloc,int b);
multiplies the vector by the submatrix of colloc that is all the columns in the
collocation matrix by the same rows that contain block b.

vector: place to store the result. size [colloc->cols]
v: vector to multiply. size [colloc->flaglen]
colloc: the collocation matrix
b: the number of the block in the collocation matrix

Zvalue()
--------
double zvalue(ConcatColloc *ccolloc,double *C,int output,int deriv,int bp)
calculates a specific value in the z vector.

inputs
colloc: the collocation matrix. The matrix multiplying c is assumed
  to consist of the collocation matrix noutput times on the diagonal
  with zeros off the diagonal
c: the bspline coefficient vector for noutput outputs.
  Assumed to be of size [colloc->rows*noutputs,1]
output,deriv,breakpoint: of the value of z that you want

output
the value z.

use the funtion odb2lin() to find the correct position of the
value in the z vector.

odb2lin()
---------
int odb2lin(ConcatColloc *ccolloc,int output,int deriv,int bp)

converts an output deriv breakpoint specification into a linear
array index

lin2db()
--------
void lin2db(int *deriv,int *bp,Colloc *colloc,int i);
converts a linear array index into a breakpoint deriv specification.

updateZ()
---------
void updateZ(double *Z,ConcatColloc *ccolloc,double *C,AV *av,int nav,int type)
calculates the values of z specified in the array av

Z: the z vector of all the outputs. values of the active variables
  are stored in their appropriate places and can be accessed using odb2lin().
ccolloc: the collocation matrix
C: the b-spline coefficients
av: the z's to calculate
nav: the number of active variables
type:
  AVINTIAL to evaluate the active variables at the first breakpoint
  of each output.
  AVTRAJECTORY all the breakpoints
  AVFINAL the final breakpoint
*/

#endif
