/*
 * cost.c: Functions to compute the cost of a trajectory
 *
 */
#include "cost.h"

void InitialCost(
int *mode, int *nstate, double *I, double *dI,
void (*func)(int *, int *, double *, double *, double **),
ConcatColloc *ccolloc, double *Z)
{
  FMatrix *dIdz;
  FMatrix *dIdZ;
  FMatrix *dIdC;
  double **zp = malloc(ccolloc->nout*sizeof(double *));

  Z2zpI(zp, Z, ccolloc);
  if(*mode == 0) {
    (*func)(mode, nstate, I, NULL, zp);
    free(zp);

  } else if(*mode == 1 || *mode == 2) {
    dIdz = MakeFMatrix(ccolloc->nz, 1);
    (*func)(mode, nstate, I, dIdz->elements[0], zp);
    free(zp);

    dIdZ = MakeFMatrix(1, ccolloc->nZ);
    dIdz2dIdZI(dIdZ, dIdz, ccolloc);
    FreeFMatrix(dIdz);

    dIdC = MakeFMatrix(1, ccolloc->nC);
    CollocConcatMultI(dIdC, dIdZ, ccolloc);
    memcpy(dI, dIdC->elements[0], ccolloc->nC*sizeof(double));
    FreeFMatrix(dIdZ);
    FreeFMatrix(dIdC);
  }
}

void IntegratedCost(
int *mode, int *nstate,
double *I, double *dI, double *bps,
void (*func)(int *, int *, int *, double *, double *, double **),
ConcatColloc *ccolloc, double *Z)
{
  double **zp = malloc(ccolloc->nout*sizeof(double *));
  double *d1;
  double *f;

  FMatrix *dIdC;
  FMatrix *dIdz;
  int i, j, k, l, offset;

  switch(*mode) {
  case 0:
    f = malloc(ccolloc->nbps*sizeof(double));
    for(i = 0; i < ccolloc->nbps; i++) {
      Z2zpT(zp, Z, ccolloc, i);
      (*func)(mode, nstate, &i, f+i, NULL, zp);
    }
    IntegrateVector(I, f, bps, ccolloc->nbps, TRAPEZOID);
    /*cbug
      PrintVector("data/Z", Z, ccolloc->nZ);
      PrintVector("data/f", f, ccolloc->nbps);
      PrintVector("data/I", I, 1); */
    free(f);
    break;

  case 1:
    d1 = malloc(ccolloc->nz*sizeof(double));
    dIdz = MakeFMatrix(ccolloc->nbps, ccolloc->nz);
    for(i = 0; i < ccolloc->nbps; i++) {
      Z2zpT(zp, Z, ccolloc, i);
      (*func)(mode, nstate, &i, NULL, d1, zp);
      for(j = 0; j < ccolloc->nz; j++)
	dIdz->elements[j][i] = d1[j];
    }
    free(d1);

    dIdC = MakeFMatrix(ccolloc->nbps, ccolloc->nC);
    for(i = 0; i < ccolloc->nout; i++)
      for(j = 0; j < ccolloc->nbps; j++) {
	offset = ccolloc->colloc[i]->block[j].offset;
	for(k = 0; k < offset; k++)
	  dIdC->elements[ccolloc->iC[i]+k][j] = 0;
	for(; k < offset+ccolloc->colloc[i]->order; k++)
	  for(l = 0, dIdC->elements[ccolloc->iC[i]+k][j] = 0;
	      l < ccolloc->colloc[i]->flaglen; l++)
	    dIdC->elements[ccolloc->iC[i]+k][j] +=
	      dIdz->elements[ccolloc->iz[i]+l][j]*
	      ccolloc->colloc[i]->block[j].matrix->elements[k-offset][l];
	for(; k < ccolloc->colloc[i]->cols; k++)
	  dIdC->elements[ccolloc->iC[i]+k][j] = 0;
      }
    FreeFMatrix(dIdz);
    IntegrateFMatrixCols(dI, dIdC, bps, TRAPEZOID);
    FreeFMatrix(dIdC);
    break;

  case 2:
    d1 = malloc(ccolloc->nz*sizeof(double));
    dIdz = MakeFMatrix(ccolloc->nbps, ccolloc->nz);
    f = malloc(ccolloc->nbps*sizeof(double));
    for(i = 0; i < ccolloc->nbps; i++) {
      Z2zpT(zp, Z, ccolloc, i);
      (*func)(mode, nstate, &i, f+i, d1, zp);
      for(j = 0; j < ccolloc->nz; j++)
	dIdz->elements[j][i] = d1[j];
    }
    free(d1);
    IntegrateVector(I, f, bps, ccolloc->nbps, TRAPEZOID);
    /*cbug
      PrintVector("data/Z", Z, ccolloc->nZ);
      PrintVector("data/f", f, ccolloc->nbps);
      PrintVector("data/I", I, 1); */
    free(f);
    dIdC = MakeFMatrix(ccolloc->nbps, ccolloc->nC);
    for(i = 0; i < ccolloc->nout; i++)
      for(j = 0; j < ccolloc->nbps; j++) {
	offset = ccolloc->colloc[i]->block[j].offset;
	for(k = 0; k < offset; k++)
	  dIdC->elements[ccolloc->iC[i]+k][j] = 0;
	for(; k < offset+ccolloc->colloc[i]->order; k++)
	  for(l = 0, dIdC->elements[ccolloc->iC[i]+k][j] = 0;
	      l < ccolloc->colloc[i]->flaglen; l++)
	    dIdC->elements[ccolloc->iC[i]+k][j] +=
	      dIdz->elements[ccolloc->iz[i]+l][j]*
	      ccolloc->colloc[i]->block[j].matrix->elements[k-offset][l];
	for(; k < ccolloc->colloc[i]->cols; k++)
	  dIdC->elements[ccolloc->iC[i]+k][j] = 0;
      }
    FreeFMatrix(dIdz);
    IntegrateFMatrixCols(dI, dIdC, bps, TRAPEZOID);
    FreeFMatrix(dIdC);
    break;
  }
  free(zp);
}

void FinalCost(
int *mode, int *nstate, double *I, double *dI,
void (*func)(int *, int *, double *, double *, double **),
ConcatColloc *ccolloc, double *Z)
{
  double **zp = malloc(ccolloc->nout*sizeof(double *));
  FMatrix *dIdz;
  FMatrix *dIdZ;
  FMatrix *dIdC;

  Z2zpF(zp, Z, ccolloc);

  if(*mode == 0) {
    (*func)(mode, nstate, I, NULL, zp);
    free(zp);

  } else if(*mode == 1 || *mode == 2) {
    dIdz = MakeFMatrix(ccolloc->nz, 1);
    (*func)(mode, nstate, I, dIdz->elements[0], zp);
    free(zp);

    dIdZ = MakeFMatrix(1, ccolloc->nZ);
    dIdz2dIdZF(dIdZ, dIdz, ccolloc);
    FreeFMatrix(dIdz);

    dIdC = MakeFMatrix(1, ccolloc->nC);
    CollocConcatMultF(dIdC, dIdZ, ccolloc);
    memcpy(dI, dIdC->elements[0], ccolloc->nC*sizeof(double));
    FreeFMatrix(dIdC);
    FreeFMatrix(dIdZ);
  }
}
