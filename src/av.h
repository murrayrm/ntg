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


#ifndef _AV_H_
#define _AV_H_

/* active variable */

#define AVINITIAL	0
#define AVTRAJECTORY	1
#define AVFINAL		2

typedef struct AVStruct
{
	int output;
	int deriv;
} AV;

#endif
