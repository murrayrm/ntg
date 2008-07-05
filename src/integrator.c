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



#include "integrator.h"

void IntegrateVector(double *I,double *f,double *t,int n,int type)
{
	int i;
	switch(type)
	{
		case TRAPEZOID:
			for(i=0,*I=0.0;i<n-1;i++)
				*I+=(t[i+1]-t[i])*(f[i+1]+f[i])/2;
			break;
		/* forward euler */
		case FEULER:
			for(i=0,*I=0.0;i<n-1;i++)
				*I+=(t[i+1]-t[i])*f[i];
			break;
		/* backward euler */
		case BEULER:
			for(i=0,*I=0.0;i<n-1;i++)
				*I+=(t[i+1]-t[i])*f[i+1];
			break;
	}
}

void IntegrateFMatrixCols(double *I,FMatrix *f,double *t,int type)
{
	int i,j;
		
	switch(type)
	{
		case TRAPEZOID:
			for(i=0;i<f->cols;i++)
				for(j=0,I[i]=0.0;j<f->rows-1;j++)
					I[i]+=(t[j+1]-t[j])*(f->elements[i][j+1]+f->elements[i][j])/2;
			break;
		/* forward euler */
		case FEULER:
			for(i=0;i<f->cols;i++)
				for(j=0,I[i]=0.0;j<f->rows-1;j++)
					I[i]+=(t[j+1]-t[j])*f->elements[i][j];
			break;
		/* backward euler */
		case BEULER:
			for(i=0;i<f->cols;i++)
				for(j=0,I[i]=0.0;j<f->rows-1;j++)
					I[i]+=(t[j+1]-t[j])*f->elements[i][j+1];
			break;
	}
}
