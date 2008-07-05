
#include "constraints.h"

/* checked */
void bounds(
double *bbar,double *b,int nc,
int nlic,int nltc,int nlfc,
int nnlic,int nnltc,int nnlfc,
int nbps,double bigbnd)
{
	int i,j;

	/* the values of the coefficients are unbounded */
	for(i=0;i<nc;i++)
		bbar[i]=bigbnd;

	/* first the linear constraint bounds */
	memcpy(bbar+nc,b,nlic*sizeof(double));
	for(i=0;i<nltc;i++)
		for(j=0;j<nbps;j++)
			bbar[nc+nlic+i*nbps+j]=b[nlic+i];
	memcpy(bbar+nc+nlic+nbps*nltc,b+nlic+nltc,nlfc*sizeof(double));

	/* now the nonlinear constraint bounds */
	memcpy(bbar+nc+nlic+nbps*nltc+nlfc,b+nlic+nltc+nlfc,nnlic*sizeof(double));
	for(i=0;i<nnltc;i++)
		for(j=0;j<nbps;j++)
			bbar[nc+nlic+nbps*nltc+nlfc+nnlic+i*nbps+j]=b[nlic+nltc+nlfc+nnlic+i];
	memcpy(bbar+nc+nlic+nbps*nltc+nlfc+nnlic+nbps*nnltc,
		b+nlic+nltc+nlfc+nnlic+nnltc,nnlfc*sizeof(double));

/*PrintVector(bbar,nc+nlic+nbps*nltc+nlfc+nnlic+nbps*nnltc+nnlfc);*/
}

/* checked */
void NonLinearConstraints(
int *mode,int *nstate,
double *nlc,FMatrix *dnlc,
int nnlic,void (*nlicf)(int *,int *,double *,double **,double **),
int nnltc,void (*nltcf)(int *,int *,int *,double *,double **,double **),
int nnlfc,void (*nlfcf)(int *,int *,double *,double **,double **),
ConcatColloc *ccolloc,double *Z)
{
	FMatrix *m1;
	int i1=0;

	if(*mode==0)
	{
		if(nnlic!=0)
		{
			NonLinearInitialConstraints(mode,nstate,nnlic,nlc,NULL,nlicf,ccolloc,Z);
			i1=nnlic;
		}
		if(nnltc!=0)
		{
			NonLinearTrajectoryConstraints(mode,nstate,nnltc,nlc+i1,NULL,nltcf,ccolloc,Z);
			i1+=nnltc*ccolloc->nbps;
		}
		if(nnlfc!=0)
			NonLinearFinalConstraints(mode,nstate,nnlfc,nlc+i1,NULL,nlfcf,ccolloc,Z);
	}
	else
	{
		assert(dnlc->rows==nnlic+nnltc*ccolloc->nbps+nnlfc);
		assert(dnlc->cols==ccolloc->nC);
		if(nnlic!=0)
		{
			NonLinearInitialConstraints(mode,nstate,nnlic,nlc,dnlc,nlicf,ccolloc,Z);
			i1=nnlic;
		}
		if(nnltc!=0)
		{
			m1=SubFMatrix(dnlc,i1,0,nnltc*ccolloc->nbps,ccolloc->nC);
			NonLinearTrajectoryConstraints(mode,nstate,nnltc,nlc+i1,m1,nltcf,ccolloc,Z);
			FreeSubFMatrix(m1);
			i1+=nnltc*ccolloc->nbps;
		}
		if(nnlfc!=0)
		{
			m1=SubFMatrix(dnlc,i1,0,nnlfc,ccolloc->nC);
			NonLinearFinalConstraints(mode,nstate,nnlfc,nlc+i1,m1,nlfcf,ccolloc,Z);
			FreeSubFMatrix(m1);
		}
	}
}

/* checked */
void NonLinearInitialConstraints(
int *mode,int *nstate,
int nnlic,double *nlic,FMatrix *dIdC,
void (*nlicf)(int *,int *,double *,double **,double **),
ConcatColloc *ccolloc,double *Z)
{
	FMatrix *dIdz;
	FMatrix *dIdZ;
	double **zp=malloc(ccolloc->nout*sizeof(double *));

	Z2zpI(zp,Z,ccolloc);

	if(*mode==0)
	{
		(*nlicf)(mode,nstate,nlic,NULL,zp);
		free(zp);
	}
	else if(*mode==1 || *mode==2)
	{
		dIdz=MakeFMatrix(ccolloc->nz,nnlic);
		(*nlicf)(mode,nstate,nlic,dIdz->elements,zp);
		free(zp);

		dIdZ=MakeFMatrix(nnlic,ccolloc->nZ);
		dIdz2dIdZI(dIdZ,dIdz,ccolloc);
		FreeFMatrix(dIdz);
		CollocConcatMultI(dIdC,dIdZ,ccolloc);
		FreeFMatrix(dIdZ);
	}
}

/* checked */
void NonLinearTrajectoryConstraints(
int *mode,int *nstate,
int nnltc,double *nltc,FMatrix *dIdC,
void (*nltcf)(int *,int *,int *,double *,double **,double **),
ConcatColloc *ccolloc,double *Z)
{
	FMatrix *dIdz;
	FMatrix *dIdZ;
	double *tmp=malloc(nnltc*sizeof(double));
	double **zp=malloc(ccolloc->nout*sizeof(double *));
	int i,j;

	if(*mode==0)
	{
		for(i=0;i<ccolloc->nbps;i++)
		{
			Z2zpT(zp,Z,ccolloc,i);
			(*nltcf)(mode,nstate,&i,tmp,NULL,zp);
			for(j=0;j<nnltc;j++)
				nltc[j*ccolloc->nbps+i]=tmp[j];
		}
		free(tmp);
		free(zp);
	}
	else if(*mode==1 || *mode==2)
	{
		dIdz=MakeFMatrix(ccolloc->nz,nnltc);
		dIdZ=MakeFMatrix(ccolloc->nbps*nnltc,ccolloc->nZ);
		for(i=0;i<ccolloc->nbps;i++)
		{
			Z2zpT(zp,Z,ccolloc,i);
			(*nltcf)(mode,nstate,&i,tmp,dIdz->elements,zp);
			for(j=0;j<nnltc;j++)
				nltc[j*ccolloc->nbps+i]=tmp[j];
			dIdz2dIdZT(dIdZ,dIdz,ccolloc,i);
		}
		FreeFMatrix(dIdz);
		free(tmp);
		free(zp);
		CollocConcatMultT(dIdC,dIdZ,ccolloc);
		FreeFMatrix(dIdZ);
	}
}

/* checked */
void NonLinearFinalConstraints(
int *mode,int *nstate,
int nnlfc,double *nlfc,FMatrix *dIdC,
void (*nlfcf)(int *,int *,double *,double **,double **),
ConcatColloc *ccolloc,double *Z)
{
	FMatrix *dIdz;
	FMatrix *dIdZ;
	double **zp=malloc(ccolloc->nout*sizeof(double *));

	Z2zpF(zp,Z,ccolloc);

	if(*mode==0)
	{
		(*nlfcf)(mode,nstate,nlfc,NULL,zp);
		free(zp);
	}
	else if(*mode==1 || *mode==2)
	{
		dIdz=MakeFMatrix(ccolloc->nz,nnlfc);
		(*nlfcf)(mode,nstate,nlfc,dIdz->elements,zp);
		free(zp);

		dIdZ=MakeFMatrix(nnlfc,ccolloc->nZ);
		dIdz2dIdZF(dIdZ,dIdz,ccolloc);
		FreeFMatrix(dIdz);

		CollocConcatMultF(dIdC,dIdZ,ccolloc);
		FreeFMatrix(dIdZ);
	}
}

/* checked */
void LinearConstraintsMatrix(
FMatrix *lc,FMatrix *lic,FMatrix *ltc,FMatrix *lfc,ConcatColloc *ccolloc)
{
	FMatrix *m1;
	int i1=0;

	if(lic!=NULL)
	{
		InitialConstraintsMatrix(lc,lic,ccolloc);
		i1=lic->cols;
	}
	if(ltc!=NULL)
	{
		m1=SubFMatrix(lc,i1,0,ccolloc->nbps*ltc->cols,ccolloc->nC);
		TrajectoryConstraintsMatrix(m1,ltc,ccolloc);
		FreeSubFMatrix(m1);
		i1+=ltc->cols*ccolloc->nbps;
	}
	if(lfc!=NULL)
	{
		m1=SubFMatrix(lc,i1,0,lfc->cols,ccolloc->nC);
		FinalConstraintsMatrix(m1,lfc,ccolloc);
		FreeSubFMatrix(m1);
	}
}

/* checked */
void InitialConstraintsMatrix(FMatrix *dIdC,FMatrix *dIdz,ConcatColloc *ccolloc)
{
	FMatrix *dIdZ=MakeFMatrix(dIdz->cols,ccolloc->nZ);

	dIdz2dIdZI(dIdZ,dIdz,ccolloc);
/*cbug*/
/*PrintFMatrix("stdout",dIdZ);*/
	CollocConcatMultI(dIdC,dIdZ,ccolloc);
	FreeFMatrix(dIdZ);
}

/* checked */
void TrajectoryConstraintsMatrix(FMatrix *dIdC,FMatrix *dIdz,
	ConcatColloc *ccolloc)
{
	FMatrix *dIdZ=MakeFMatrix(ccolloc->nbps*dIdz->cols,ccolloc->nZ);
	int i;
	
	for(i=0;i<ccolloc->nbps;i++)
		dIdz2dIdZT(dIdZ,dIdz,ccolloc,i);
/*cbug*/
/*PrintFMatrix("stdout",dIdZ);*/
	CollocConcatMultT(dIdC,dIdZ,ccolloc);
	FreeFMatrix(dIdZ);
}

/* checked */
void FinalConstraintsMatrix(FMatrix *dIdC,FMatrix *dIdz,ConcatColloc *ccolloc)
{
	FMatrix *dIdZ=MakeFMatrix(dIdz->cols,ccolloc->nZ);
	
	dIdz2dIdZF(dIdZ,dIdz,ccolloc);
/*cbug*/
/*PrintFMatrix("stdout",dIdZ);*/
	CollocConcatMultF(dIdC,dIdZ,ccolloc);
	FreeFMatrix(dIdZ);
}
