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



#include "matrix.h"

FMatrix *ResizeFMatrix(FMatrix *m,int rows,int cols)
{
	double *d1;
	size_t s1;
	int i,j;

	s1=rows*cols*sizeof(double);
	d1=realloc(m->elements[0],s1);
  memset(d1,0,s1); /*bzero(d1,s1);*/
	m->elements=realloc(m->elements,cols*sizeof(double *));
	m->elements[0]=d1;
	m->rows=rows;
	m->cols=cols;
	for(j=1,i=rows;j<cols;j++,i+=rows)
		m->elements[j]=&(m->elements[0][i]);
	return m;
}
	
FMatrix *SubFMatrix(FMatrix *m,int row,int col,int rows,int cols)
{
	FMatrix *m1;
	int j;

	m1=malloc(sizeof(Matrix));
	m1->rows=rows;
	m1->cols=cols;
	m1->elements=calloc(cols,sizeof(double *));

	for(j=0;j<cols;j++)
		m1->elements[j]=&(m->elements[j+col][row]);
	return m1;
}

void FreeSubFMatrix(FMatrix *m)
{
	free(m->elements);
	free(m);
}

void FMatrixSet(FMatrix *m,double d,int row,int col,int rows,int cols)
{
	int i,j;
	for(i=0;i<rows;i++)
	for(j=0;j<cols;j++)
		m->elements[col+j][row+i]=d;
}

void FMatrixCopy(
  FMatrix *B,int rowb,int colb,
  FMatrix *A,int rowa,int cola,int rows,int cols)
{
	int j;
	for(j=0;j<cols;j++)
		memcpy(
		&(B->elements[colb+j][rowb]),
		&(A->elements[cola+j][rowa]),
		rows*sizeof(double));
}

void FTranspose(FMatrix *out,FMatrix *in)
{
	int i,j;

	for(i=0;i<in->rows;i++)
		for(j=0;j<in->cols;j++)
			out->elements[i][j]=in->elements[j][i];
}

int Fsub2ind(int rows,int cols,int i,int j)
{
	return j*rows+i;
}

FMatrix *MakeFMatrix(int rows,int cols)
{
	FMatrix *m;

	m=malloc(sizeof(FMatrix));
	m->elements=DoubleFMatrix(rows,cols);
	m->rows=rows;
	m->cols=cols;
	return m;
}

void FreeFMatrix(FMatrix *m)
{
	FreeDoubleFMatrix(m->elements);
	free(m);
}

double **DoubleFMatrix(int rows,int cols)
{
	double **ppd;
	int i,j;
	size_t s1;

	ppd=malloc(cols*sizeof(double *));
	ppd[0]=calloc(rows*cols,sizeof(double));
	for(j=1,i=rows;j<cols;j++,i+=rows)
		ppd[j]=&(ppd[0][i]);
	return ppd;
}

void FreeDoubleFMatrix(double **d)
{
	free(d[0]);
	free(d);
}

void F2CMatrix(Matrix *out,FMatrix *in)
{
	int i,j;
	for(j=0;j<in->cols;j++)
		for(i=0;i<in->rows;i++)
			out->elements[i][j]=in->elements[j][i];
}

void C2FMatrix(FMatrix *out,Matrix *in)
{
	int i,j;
	for(j=0;j<in->cols;j++)
		for(i=0;i<in->rows;i++)
			out->elements[j][i]=in->elements[i][j];
}

void PrintFMatrix(char *filename,FMatrix *matrix)
{
	int i,j;
	FILE *file;

	if(!strcmp(filename,"stdout"))
		file=stdout;
	else if(!strcmp(filename,"stderr"))
		file=stderr;
	else
	{
		if((file=fopen(filename,"w"))==NULL)
		return;
	}

	for(i=0;i<matrix->rows;i++)
	{
		for(j=0;j<matrix->cols;j++)
			fprintf(file,"%.16e ",matrix->elements[j][i]);
		fprintf(file,"\n");
	}
	fprintf(file,"\n\n\n");

	if(file!=stdout && file!=stderr)
		fclose(file);
}

void FMatrixMult(FMatrix *mout,FMatrix *m1,FMatrix *m2)
{
	int i,j,k;
	for(i=0;i<m1->rows;i++)
		for(j=0;mout->elements[j][i]=0,j<m2->cols;j++)
			for(k=0;k<m1->cols;k++)
				mout->elements[j][i]+=m1->elements[k][i]*m2->elements[j][k];
}

void Vector3Add(double *vout,double *v1,double *v2,double *v3,int n)
{
   int i;
   for(i=0;i<n;i++)
      vout[i]=v1[i]+v2[i]+v3[i];
}

/* C Matrix routines */
void MatrixSet(Matrix *m,double d,int row,int col,int rows,int cols)
{
	int i,j;
	for(i=0;i<rows;i++)
	for(j=0;j<cols;j++)
		m->elements[row+i][col+j]=d;
}

void MatrixCopy(
  Matrix *B,int rowb,int colb,
  Matrix *A,int rowa,int cola,int rows,int cols)
{
	int i;
	for(i=0;i<rows;i++)
		memcpy(
		&(B->elements[rowb+i][colb]),
		&(A->elements[rowa+i][cola]),
		cols*sizeof(double));
}

Matrix *MakeMatrix(int rows,int cols)
{
	Matrix *m;

	m=malloc(sizeof(Matrix));					assert(m!=NULL);
	m->elements=DoubleMatrix(rows,cols);
	m->rows=rows;
	m->cols=cols;
	return m;
}

void FreeMatrix(Matrix *matrix)
{
	FreeDoubleMatrix(matrix->elements);
	assert(matrix!=NULL);		free(matrix);
}

double **DoubleMatrix(int rows,int cols)
{
	double **tmp;
	size_t sizeofmatrix;
	int i,j;

	tmp=malloc(rows*sizeof(double *));			assert(tmp!=NULL);
	tmp[0]=calloc(rows*cols,sizeof(double));					assert(tmp[0]!=NULL);
	for(i=1,j=cols;i<rows;i++,j+=cols)
		tmp[i]=&(tmp[0][j]);
	return tmp;
}

void FreeDoubleMatrix(double **d)
{
	assert(d[0]!=NULL);			free(d[0]);
	assert(d!=NULL);				free(d);
}

double dot(double *v1,double *v2,int n)
{
	int i;
	double tmp=0.0;
	for(i=0;i<n;i++)
		tmp+=v1[i]*v2[i];
	return tmp;
}

void PrintMatrix(char *filename,Matrix *matrix)
{
	int i,j;
	FILE *file;

	if(!strcmp(filename,"stdout"))
		file=stdout;
	else if(!strcmp(filename,"stderr"))
		file=stderr;
	else
	{
		if((file=fopen(filename,"w"))==NULL)
		return;
	}

	for(i=0;i<matrix->rows;i++)
	{
		for(j=0;j<matrix->cols;j++)
			fprintf(file,"%f ",matrix->elements[i][j]);
		fprintf(file,"\n");
	}
	fprintf(file,"\n\n\n");

	if(file!=stdout && file!=stderr)
		fclose(file);
}

void PrintVector(char *filename,double *f,int nf)
{
	int i;
	FILE *file;

	if(!strcmp(filename,"stdout"))
		file=stdout;
	else if(!strcmp(filename,"stderr"))
		file=stderr;
	else
	{
		if((file=fopen(filename,"w"))==NULL)
		return;
	}
	for(i=0;i<nf;i++)
		fprintf(file,"%.18e ",f[i]);
	fprintf(file,"\n");
	if(file!=stdout && file!=stderr)
		fclose(file);
}

void PrintiVector(char *filename,int *f,int nf)
{
	int i;
	FILE *file;

	if(!strcmp(filename,"stdout"))
		file=stdout;
	else if(!strcmp(filename,"stderr"))
		file=stderr;
	else
	{
		if((file=fopen(filename,"w"))==NULL)
		return;
	}
	for(i=0;i<nf;i++)
		fprintf(file,"%d\n",f[i]);
	if(file!=stdout && file!=stderr)
		fclose(file);
}

void MatrixMult(Matrix *mout,Matrix *m1,Matrix *m2)
{
	int i,j,k;

	for(i=0;i<m1->rows;i++)
		for(j=0;j<m2->cols;j++)
		{
			mout->elements[i][j]=0.0;
			for(k=0;k<m1->cols;k++)
				mout->elements[i][j]+=m1->elements[i][k]*m2->elements[k][j];
		}
}
