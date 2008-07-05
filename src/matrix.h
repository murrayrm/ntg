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
* matrix.h
* matrix handling routines
*/

#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

typedef struct MatrixStruct
{
	double **elements;
	int rows,cols;
} Matrix;

void MatrixCopy(
  Matrix *B,int rowb,int colb,
  Matrix *A,int rowa,int cola,int rows,int cols);
void MatrixSet(Matrix *m,double d,int row,int col,int rows,int cols);
Matrix *MakeMatrix(int rows,int cols);
void MatrixMult(Matrix *mout,Matrix *m1,Matrix *m2);
void FreeMatrix(Matrix *matrix);
void PrintMatrix(char *filename,Matrix *matrix);
void PrintVector(char *filename,double *f,int nf);
void PrintiVector(char *filename,int *f,int nf);

double **DoubleMatrix(int rows,int cols);
void FreeDoubleMatrix(double **d);

/*
* stores and manipulate matrices columnwise (fortran style).
*/

typedef struct FMatrixStruct
{
	double **elements;
	int rows,cols;
} FMatrix;

void FMatrixCopy(
	FMatrix *B,int rowb,int colb,
	FMatrix *A,int rowa,int cola,int rows,int cols);
void FMatrixSet(FMatrix *m,double d,int row,int col,int rows,int cols);
FMatrix *ResizeFMatrix(FMatrix *m,int rows,int cols);
FMatrix *MakeFMatrix(int rows,int cols);
FMatrix *SubFMatrix(FMatrix *m,int row,int col,int rows,int cols);
void FreeSubFMatrix(FMatrix *m);
void FMatrixMult(FMatrix *mout,FMatrix *m1,FMatrix *m2);
void FreeFMatrix(FMatrix *matrix);
void PrintFMatrix(char *filename,FMatrix *matrix);

double **DoubleFMatrix(int rows,int cols);
void FreeDoubleFMatrix(double **d);

int FSub2ind(int rows,int cols,int i,int j);
void FTranspose(FMatrix *out,FMatrix *in);
void F2CMatrix(Matrix *out,FMatrix *in);
void C2FMatrix(FMatrix *out,Matrix *in);

void Vector3Add(double *vout,double *v1,double *v2,double *v3,int n);
double dot(double *v1,double *v2,int n);

/**************************************************

Description of functions
-----------------------

Matrix *MakeMatrix(int rows,int cols);
void FreeMatrix(Matrix *matrix);
FMatrix *MakeFMatrix(int rows,int cols);
void FreeFMatrix(FMatrix *matrix);

Make/Free a matrix of size [rows,cols]. and initialises all elements to 0.
--------------------------------------------------------------

MatrixCopy()
FMatrixCopy()

copy the submatrix in A of size [rows,cols] at position [rowa,cola]
in the the matrix B at position [rowb,colb]
make sure B has enough memory already allocated
-------------------------------------------------------

void MatrixSet(Matrix *m,double d,int row,int col,int rows,int cols);
void FMatrixSet(FMatrix *m,double d,int row,int col,int rows,int cols);

sets all the elements in the submatrix of size [rows,cols] positioned
at [row,col] in m to the number d
-------------------------------------------------------

void Transpose(Matrix *out,Matrix *in);
void FTranspose(FMatrix *out,FMatrix *in);

returns the transpose of in in out.
------------------------------------------

int Fsub2ind(int rows,int cols,int i,int j);

determines the equivalent single index corresponding
subscript values row i, column j for a matrix of size
"rows","cols" stored columnwise in a 1D array
(ie Fortran style).
see matlabs function sub2ind
--------------------------------------------


void F2CMatrix(Matrix *out,FMatrix *in)
void C2FMatrix(FMatrix *out,Matrix *in)

copies data between the two types of matrices. in and out must
be of the same size. ie out->cols=in->cols and out->rows=in->rows
---------------------------------------

void MatrixMult(Matrix *mout,Matrix *m1,Matrix *m2);
void FMatrixMult(FMatrix *mout,FMatrix *m1,FMatrix *m2);

multiplies the two matrices
mout: result stored here size(mout)=[m1->rows,m2->cols]
m1,m2: matrices to multiply. note this should hold m1->cols=m2->rows
-------------------------------------------------------

void Vector3Add(double *vout,double *v1,double *v2,double *v3,int n)

adds three vectors element by elements
vout: space to store result
v1,v2,v3: the 3 vectors to add
n: the length of vout,v1,v2,v3

********************************************************/

#endif
