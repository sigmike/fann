/*
Fast Artificial Neural Network Library (fann)
Copyright (C) 2003 Steffen Nissen (lukesky@diku.dk)

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#ifndef __fann_internal_h__
#define __fann_internal_h__
/* internal include file, not to be included directly
 */

#include <math.h>
#include "fann_data.h"

#define FANN_FIX_VERSION "FANN_FIX_1.0"
#define FANN_FLO_VERSION "FANN_FLO_1.0"

#ifdef FIXEDFANN
#define FANN_CONF_VERSION FANN_FIX_VERSION
#else
#define FANN_CONF_VERSION FANN_FLO_VERSION
#endif

struct fann * fann_allocate_structure(float learning_rate, unsigned int num_layers);
void fann_allocate_neurons(struct fann *ann);

void fann_allocate_connections(struct fann *ann);

int fann_save_internal(struct fann *ann, const char *configuration_file, unsigned int save_as_fixed);
void fann_save_train_internal(struct fann_train_data* data, char *filename, unsigned int save_as_fixed, unsigned int decimal_point);

int fann_compare_connections(const void* c1, const void* c2);
void fann_seed_rand();

/* called fann_max, in order to not interferre with predefined versions of max */
#define fann_max(x, y) (((x) > (y)) ? (x) : (y))
#define fann_min(x, y) (((x) < (y)) ? (x) : (y))

#define fann_rand(min_value, max_value) (((double)(min_value))+(((double)(max_value)-((double)(min_value)))*rand()/(RAND_MAX+1.0)))

#define fann_abs(value) (((value) > 0) ? (value) : -(value))

#ifdef FIXEDFANN

#define fann_mult(x,y) ((x*y) >> decimal_point)
#define fann_div(x,y) (((x) << decimal_point)/y)
#define fann_random_weight() (fann_type)(fann_rand(-multiplier/10,multiplier/10))
/* sigmoid calculated with use of floats, only as reference */
#define fann_sigmoid(steepness, value) ((fann_type)(0.5+((1.0/(1.0 + exp(-2.0 * ((float)steepness/multiplier) * ((float)value/multiplier))))*multiplier)))
/* sigmoid as a stepwise linear function */
#define fann_linear(v1, r1, v2, r2, value) ((((r2-r1) * (value-v1))/(v2-v1)) + r1)
#define fann_sigmoid_stepwise(v1, v2, v3, v4, v5, v6, r1, r2, r3, r4, r5, r6, value, multiplier) (value < v5 ? (value < v3 ? (value < v2 ? (value < v1 ? 0 : fann_linear(v1, r1, v2, r2, value)) : fann_linear(v2, r2, v3, r3, value)) : (value < v4 ? fann_linear(v3, r3, v4, r4, value) : fann_linear(v4, r4, v5, r5, value))) : (value < v6 ? fann_linear(v5, r5, v6, r6, value) : multiplier))
#else

#define fann_mult(x,y) (x*y)
#define fann_div(x,y) (x/y)
#define fann_random_weight() (fann_rand(-0.1,0.1))
#define fann_sigmoid(steepness, value) (1.0/(1.0 + exp(-2.0 * steepness * value)))
#define fann_sigmoid_derive(steepness, value) (2.0 * steepness * value * (1.0 - value))

#endif

#endif
