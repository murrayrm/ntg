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


#include "colloc.h"

ConcatColloc *ConcatCollocMatrix(
int nout,
double **knots,
int *ninterv,
double *bps,
int nbps,
int *flaglen,
int *order,
int *mult)
{
	ConcatColloc *ccolloc;
	int i;

	ccolloc=malloc(sizeof(ConcatColloc));			assert(ccolloc!=NULL);
	ccolloc->colloc=malloc(nout*sizeof(Colloc *));assert(ccolloc->colloc!=NULL);
	ccolloc->iZ=malloc(nout*sizeof(int));			assert(ccolloc->iZ!=NULL);
	ccolloc->iz=malloc(nout*sizeof(int));			assert(ccolloc->iz!=NULL);
	ccolloc->iC=malloc(nout*sizeof(int));			assert(ccolloc->iC!=NULL);

	for(i=0,ccolloc->nz=0,ccolloc->iZ[0]=0,ccolloc->iC[0]=0,ccolloc->nC=0,
		ccolloc->iz[0]=0;
		i<nout;i++)
	{
		ccolloc->colloc[i]=CollocMatrix(knots[i],ninterv[i],bps,nbps,
			flaglen[i],order[i],mult[i]);
		assert(ccolloc->colloc[i]!=NULL);
		if(i!=0)
		{
			ccolloc->iZ[i]=ccolloc->iZ[i-1]+ccolloc->colloc[i-1]->rows;
			ccolloc->iz[i]=ccolloc->iz[i-1]+ccolloc->colloc[i-1]->flaglen;
			ccolloc->iC[i]=ccolloc->iC[i-1]+ccolloc->colloc[i-1]->cols;
		}
		ccolloc->nz+=flaglen[i];
		ccolloc->nC+=ccolloc->colloc[i]->cols;
	}
	ccolloc->nZ=ccolloc->nz*nbps;
	ccolloc->nout=nout;
	ccolloc->nbps=nbps;

	return ccolloc;
}

Colloc *CollocMatrix(
double *knots,
int ninterv,
double *bps,
int nbps,
int flaglen,
int order,
int mult)
{
	Colloc *colloc;
	int n=ninterv*(order-mult)+mult;
	int nknots=ninterv+1;
	int naugknots=n+order;
	double *augknots;
	double *a;
	FMatrix *dbiatx;
	double x;
	int left;
	int mflag;
	int i;

	dbiatx=MakeFMatrix(order,flaglen);
	augknots=malloc(naugknots*sizeof(double));		assert(augknots!=NULL);
	a=malloc(order*order*sizeof(double));				assert(a!=NULL);

	colloc=malloc(sizeof(Colloc));						assert(colloc!=NULL);
	colloc->block=malloc(nbps*sizeof(Block));			assert(colloc->block!=NULL);
	colloc->ninterv=ninterv;
	colloc->order=order;
	colloc->mult=mult;
	colloc->flaglen=flaglen;
	colloc->nbps=nbps;
	colloc->rows=flaglen*nbps;
	colloc->cols=n;
	
	side_.m=mult;
	knots_(knots,&ninterv,&order,augknots,&n);

	for(i=0;i<nbps;i++)
	{
		x=bps[i];
		interv_(augknots,&naugknots,&x,&left,&mflag);
		bsplvd_(augknots,&order,&x,&left,a,dbiatx->elements[0],&flaglen);
		colloc->block[i].matrix=MakeFMatrix(flaglen,order);
		FTranspose(colloc->block[i].matrix,dbiatx);
	}
	/* place in a separate for loop so interv_ doesn't have to search so much*/
	for(i=0;i<nbps;i++)
	{
		x=bps[i];
		interv_(knots,&nknots,&x,&left,&mflag);
		colloc->block[i].offset=(left-1)*(order-mult);
		/* subtract 1 from left because interv_() is a fortran subroutine
		which assumes arrays are indexed from 1*/
	}
	free(augknots);
	free(a);
	FreeFMatrix(dbiatx);

	return colloc;
}

void FreeConcatColloc(ConcatColloc *ccolloc)
{
	int i;
   for(i=0;i<ccolloc->nout;i++)
      FreeColloc(ccolloc->colloc[i]);
   assert(ccolloc->iZ!=NULL);			free(ccolloc->iZ);
	assert(ccolloc->iz!=NULL);			free(ccolloc->iz);
   assert(ccolloc->iC!=NULL);			free(ccolloc->iC);
   assert(ccolloc->colloc!=NULL);	free(ccolloc->colloc);
   assert(ccolloc!=NULL);				free(ccolloc);
}

void FreeColloc(Colloc *colloc)
{
	int i;

	for(i=0;i<colloc->nbps;i++)
		FreeFMatrix(colloc->block[i].matrix);
	assert(colloc->block!=NULL);		free(colloc->block);
	assert(colloc!=NULL);				free(colloc);
}

void CollocMult(FMatrix *B,FMatrix *A,Colloc *colloc)
{
	int i,j,k;
	for(i=0;i<A->rows;i++)
	for(j=0,B->elements[j][i]=0;j<colloc->cols;j++)
	for(k=0;k<colloc->rows;k++)
		B->elements[j][i]+=A->elements[k][i]*CollocElement(colloc,k,j);
}

void CollocConcatMult(FMatrix *B,FMatrix *A,ConcatColloc *ccolloc)
{
	int i;
	FMatrix *m1,*m2;

	if(ccolloc->nout==1)
	{
		CollocMult(B,A,ccolloc->colloc[0]);
		return;
	}

	for(i=0;i<ccolloc->nout;i++)
	{
		if(i==0)
		{
			m1=MakeFMatrix(A->rows,ccolloc->colloc[0]->rows);
			m2=MakeFMatrix(A->rows,ccolloc->colloc[0]->cols);
		}
		else
		{
			m1=ResizeFMatrix(m1,A->rows,ccolloc->colloc[i]->rows);
			m2=ResizeFMatrix(m2,A->rows,ccolloc->colloc[i]->cols);
		}
		FMatrixCopy(m1,0,0,A,0,ccolloc->iZ[i],m1->rows,m1->cols);
		CollocMult(m2,m1,ccolloc->colloc[i]);
		FMatrixCopy(B,0,ccolloc->iC[i],m2,0,0,m2->rows,m2->cols);
	}
	FreeFMatrix(m1);
	FreeFMatrix(m2);
}

double CollocElement(Colloc *colloc,int row,int col)
{
	int deriv;
	int bp;

	lin2db(&deriv,&bp,colloc,row);
	if(col<colloc->block[bp].offset)
		return 0.0;
	else if(col>=colloc->block[bp].offset+colloc->order)
		return 0.0;
	else
		return colloc->block[bp].matrix->elements[col-colloc->block[bp].offset][deriv];
}

void PrintColloc(char *filename,Colloc *colloc)
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

	for(i=0;i<colloc->rows;i++)
	{
		for(j=0;j<colloc->cols;j++)
			fprintf(file,"%.16e ",CollocElement(colloc,i,j));
		fprintf(file,"\n");
	}
	fprintf(file,"\n\n\n");
	if(file!=stdout && file!=stderr)
		fclose(file);
}

/*void CollocBlockMultVC(double *vout,double *v,Colloc *colloc,int b)
{
	int offset=colloc->block[b].offset;
	int i;
	FMatrix *m1,*m2;

	for(i=0;i<offset;i++)
		vout[i]=0.0;
	for(i=offset+colloc->order;i<colloc->cols;i++)
		vout[i]=0.0;

	m1=MakeFMatrix(1,colloc->order);
	m2=MakeFMatrix(1,colloc->flaglen);
	memcpy(m2->elements[0],v,colloc->flaglen*sizeof(double));
	FMatrixMult(m1,m2,colloc->block[b].matrix);
	memcpy(vout+offset,m1->elements[0],colloc->order*sizeof(double));

	FreeFMatrix(m1);
	FreeFMatrix(m2);
}*/

/* for the sparse matrices of the form given by the initial constraints */
void CollocConcatMultI(FMatrix *dIdC,FMatrix *dIdZ,ConcatColloc *ccolloc)
{
	int I,J;
	int j,k,l;

	assert(dIdZ->cols==ccolloc->nZ);

	for(I=0;I<dIdZ->rows;I++)
	for(j=0;j<ccolloc->nout;j++)
	for(k=0;k<ccolloc->colloc[j]->order;k++)
	{
     J=ccolloc->iC[j]+k;
     for(l=0,dIdC->elements[J][I]=0;l<ccolloc->colloc[j]->flaglen;l++)
       dIdC->elements[J][I]+=
       dIdZ->elements[ccolloc->iZ[j]+l][I]*
       ccolloc->colloc[j]->block[0].matrix->elements[k][l];
   }
}

/* for the sparse matrices of the form given by the trajectory constraints */
void CollocConcatMultT(FMatrix *dIdC,FMatrix *dIdZ,ConcatColloc *ccolloc)
{
	int I;
	int i,j,k,l;

	assert(dIdZ->cols==ccolloc->nZ);

	for(I=0;I<dIdZ->rows;I++)
	{
		i=I%ccolloc->nbps;
		for(j=0;j<ccolloc->nout;j++)
      for(k=ccolloc->colloc[j]->block[i].offset;
		k<ccolloc->colloc[j]->block[i].offset+ccolloc->colloc[j]->order;k++)
		for(l=0,dIdC->elements[ccolloc->iC[j]+k][I]=0;
		l<ccolloc->colloc[j]->flaglen;l++)
		{
			dIdC->elements[ccolloc->iC[j]+k][I]+=
			dIdZ->elements[ccolloc->iZ[j]+i*ccolloc->colloc[j]->flaglen+l][I]*
			ccolloc->colloc[j]->block[i].matrix->elements
			[k-ccolloc->colloc[j]->block[i].offset][l];
		}
	}
}

void CollocConcatMultF(FMatrix *dIdC,FMatrix *dIdZ,ConcatColloc *ccolloc)
{
	int I,J;
	int j,k,l;

	assert(dIdZ->cols==ccolloc->nZ);

	for(I=0;I<dIdZ->rows;I++)
	for(j=0;j<ccolloc->nout;j++)
   for(k=0;k<ccolloc->colloc[j]->order;k++)
   {
     J=ccolloc->iC[j]+ccolloc->colloc[j]->block[ccolloc->nbps-1].offset+k;
     for(l=0,dIdC->elements[J][I]=0;l<ccolloc->colloc[j]->flaglen;l++)
     {
       if(j==ccolloc->nout-1)
		   dIdC->elements[J][I]+=dIdZ->elements
				[ccolloc->iZ[j]+ccolloc->colloc[j]->flaglen*
				ccolloc->nbps-ccolloc->colloc[j]->flaglen+l][I]*
		   	ccolloc->colloc[j]->block[ccolloc->nbps-1].matrix->elements[k][l];
       else
		   dIdC->elements[J][I]+=dIdZ->elements
				[ccolloc->iZ[j+1]-ccolloc->colloc[j]->flaglen+l][I]*
		   	ccolloc->colloc[j]->block[ccolloc->nbps-1].matrix->elements[k][l];
	  }
	}
	/*for(j=0;j<ccolloc->nout;j++)
    FMatrixSet(dIdC,0.0,
    0,ccolloc->iC[j],
    dIdZ->rows,ccolloc->colloc[j]->cols-ccolloc->colloc[j]->order);*/
}

double Zvalue(ConcatColloc *ccolloc,double *C,int output,int deriv,int bp)
{
	int j;
	double d;
	for(j=0,d=0.0;j<ccolloc->colloc[output]->order;j++)
		d+=ccolloc->colloc[output]->block[bp].matrix->elements[j][deriv]*
		C[ccolloc->iC[output]+ccolloc->colloc[output]->block[bp].offset+j];
	return d;
}

int odb2lin(ConcatColloc *ccolloc,int output,int deriv,int bp)
{
	return ccolloc->iZ[output]+ccolloc->colloc[output]->flaglen*bp+deriv;
}

int db2lin(Colloc *colloc,int deriv,int bp)
{
	return bp*colloc->flaglen+deriv;
}

void lin2db(int *deriv,int *bp,Colloc *colloc,int i)
{
	*deriv=i%(colloc->flaglen);	/* row in the block */
	*bp=i/(colloc->flaglen);		/* the number of the block i'm in */
}

void updateZ(double *Z,ConcatColloc *ccolloc,double *C,AV *av,int nav,int type)
{
	int i,j;

	switch(type)
	{
		case AVINITIAL:
			for(i=0;i<nav;i++)
			  Z[odb2lin(ccolloc,av[i].output,av[i].deriv,0)]=
			    Zvalue(ccolloc,C,av[i].output,av[i].deriv,0);
			break;
		case AVTRAJECTORY:
			for(i=0;i<nav;i++)
			for(j=0;j<ccolloc->nbps;j++)
				Z[odb2lin(ccolloc,av[i].output,av[i].deriv,j)]=
					Zvalue(ccolloc,C,av[i].output,av[i].deriv,j);
			break;
		case AVFINAL:
			for(i=0;i<nav;i++)
				Z[odb2lin(ccolloc,av[i].output,av[i].deriv,ccolloc->nbps-1)]=
					Zvalue(ccolloc,C,av[i].output,av[i].deriv,ccolloc->nbps-1);
			break;
	}
}

void dIdz2dIdZI(FMatrix *dIdZ,FMatrix *dIdz,ConcatColloc *ccolloc)
{
	int i,j,k;

	assert(dIdZ->rows==dIdz->cols);
	assert(dIdZ->cols==ccolloc->nZ);
	assert(dIdz->rows==ccolloc->nz);

	for(i=0;i<dIdZ->rows;i++)
	for(j=0;j<ccolloc->nout;j++)
	for(k=0;k<ccolloc->colloc[j]->flaglen;k++)
		dIdZ->elements[ccolloc->iZ[j]+k][i]=
		dIdz->elements[i][ccolloc->iz[j]+k];
}

void dIdz2dIdZT(FMatrix *dIdZ,FMatrix *dIdz,ConcatColloc *ccolloc,int bp)
{
	int i,j,k;

	assert(bp<ccolloc->nbps);
	assert(dIdZ->rows==ccolloc->nbps*dIdz->cols);
	assert(dIdZ->cols==ccolloc->nZ);
	assert(dIdz->rows==ccolloc->nz);

	for(i=0;i<dIdz->cols;i++)
	for(j=0;j<ccolloc->nout;j++)
	for(k=0;k<ccolloc->colloc[j]->flaglen;k++)
		dIdZ->elements
		[ccolloc->iZ[j]+bp*ccolloc->colloc[j]->flaglen+k][i*ccolloc->nbps+bp]=
		dIdz->elements[i][ccolloc->iz[j]+k];
}

void dIdz2dIdZF(FMatrix *dIdZ,FMatrix *dIdz,ConcatColloc *ccolloc)
{
	int i,j,k;

	assert(dIdZ->rows==dIdz->cols);
	assert(dIdZ->cols==ccolloc->nZ);
	assert(dIdz->rows==ccolloc->nz);

	for(i=0;i<dIdZ->rows;i++)
	{
		if(ccolloc->nout!=1)
		{
			for(j=0;j<ccolloc->nout-1;j++)
			for(k=0;k<ccolloc->colloc[j]->flaglen;k++)
				dIdZ->elements[ccolloc->iZ[j+1]-ccolloc->colloc[j]->flaglen+k][i]=
				dIdz->elements[i][ccolloc->iz[j]+k];
		}
		for(k=0;k<ccolloc->colloc[ccolloc->nout-1]->flaglen;k++)
			dIdZ->elements
			[ccolloc->nZ-ccolloc->colloc[ccolloc->nout-1]->flaglen+k][i]=
			dIdz->elements[i][ccolloc->iz[ccolloc->nout-1]+k];
	}
}

void Z2zpI(double **zp,double *Z,ConcatColloc *ccolloc)
{
	int i;
	for(i=0;i<ccolloc->nout;i++)
		zp[i]=&(Z[ccolloc->iZ[i]]);
}

void Z2zpT(double **zp,double *Z,ConcatColloc *ccolloc,int bp)
{
	int i;
	for(i=0;i<ccolloc->nout;i++)
		zp[i]=&(Z[ccolloc->iZ[i]+bp*ccolloc->colloc[i]->flaglen]);
}

void Z2zpF(double **zp,double *Z,ConcatColloc *ccolloc)
{
	int i;
	if(ccolloc->nout!=1)
		for(i=0;i<ccolloc->nout-1;i++)
			zp[i]=&(Z[ccolloc->iZ[i+1]-ccolloc->colloc[i]->flaglen]);
	zp[ccolloc->nout-1]=
	&(Z[ccolloc->nZ-ccolloc->colloc[ccolloc->nout-1]->flaglen]);
}

void SplineInterp(
double *f,double x,double *knots,int ninterv,double *coefs,int ncoefs,
int order,int mult,int flaglen)
{
	int nknots=ninterv+1;
	int n=ninterv*(order-mult)+mult;
	int naugknots=n+order;
	double *augknots=malloc(naugknots*sizeof(double));
	double *a=malloc(order*order*sizeof(double));
	double *dbiatx=malloc(order*flaglen*sizeof(double));
	static int left1;
	static int left2;
	int mflag;
	int i,j;
	int offset,i1;
	
	assert(n==ncoefs);
	side_.m=mult;
	knots_(knots,&ninterv,&order,augknots,&n);

	interv_(augknots,&naugknots,&x,&left1,&mflag);
	bsplvd_(augknots,&order,&x,&left1,a,dbiatx,&flaglen);

	interv_(knots,&nknots,&x,&left2,&mflag);
	offset=(left2-1)*(order-mult);

	for(i=0;i<flaglen;i++)
	{
		i1=i*order;
		for(j=0,f[i]=0.0;j<order;j++)
			f[i]+=dbiatx[i1+j]*coefs[offset+j];
	}
	free(augknots);
	free(a);
	free(dbiatx);
}
