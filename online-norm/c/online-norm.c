/* 
 * Released under BSD 3-Clause License,
 * Copyright (c) 2019 Cerebras Systems Inc.
 * All rights reserved.
 *
 * This is a C implementation of the Online Normalization algorithm. 
 */

#include <math.h>

typedef struct {
	float mu;   /* mu      */
	float s2;   /* sigma^2 */
	float a;    /* alpha   */
	float ap;   /* alpha'  */
	float e1;   /* e^(1)   */
	float ey;   /* e^(y)   */
} onlinenorm;

void onlinenorm_init(onlinenorm *on, float a, float ap) {
	on->mu = 0;  /* assume initial input distribution is zero mean  */
	on->s2 = 1;  /* assume initial input distribution is unit variance */
	on->a = a;
	on->e1 = 0;  /* initialize to no accumulated error */
	on->ey = 0;  /* initialize to no accumulated error */
	on->ap = ap;
}

float onlinenorm_fprop(float x, onlinenorm *on) {
	const float a = on->a;
	const float mu = on->mu;
	const float s2 = on->s2;
	const float y = (x - mu)/sqrt(s2);      /* Equation 8a */ 
	on->mu = a*mu + (1-a)*x;                /* Equation 8b */
	on->s2 = a*s2 + a*(1-a)*(x-mu)*(x-mu);  /* Equation 8c */
	return y;
}

float onlinenorm_bprop(float yp, float y, onlinenorm *on) {
	const float s2 = on->s2;
	const float ap = on->ap;
	const float ey = on->ey;
	const float e1 = on->e1;
	const float xpp = yp - (1-ap)*ey*y;        /* Equation 11a */
	const float xp = xpp/sqrt(s2) - (1-ap)*e1; /* Equation 12a */
	on->ey = ey + xpp * y;                     /* Equation 11b */
	on->e1 = e1 + xp;                          /* Equation 12b */
	return xp;
}

