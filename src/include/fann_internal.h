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
#include <stdio.h>
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
int fann_save_internal_fd(struct fann *ann, FILE *conf, const char *configuration_file, unsigned int save_as_fixed);
void fann_save_train_internal(struct fann_train_data* data, char *filename, unsigned int save_as_fixed, unsigned int decimal_point);
void fann_save_train_internal_fd(struct fann_train_data* data, FILE *file, char *filename, unsigned int save_as_fixed, unsigned int decimal_point);

void fann_seed_rand();

void fann_update_stepwise_hidden(struct fann *ann);
void fann_update_stepwise_output(struct fann *ann);

void fann_error(struct fann_error *errdat, unsigned int errno, ...);
void fann_init_error_data(struct fann_error *errdat);

struct fann * fann_create_from_fd(FILE *conf, const char *configuration_file);
struct fann_train_data* fann_read_train_from_fd(FILE *file, char *filename);

/* called fann_max, in order to not interferre with predefined versions of max */
#define fann_max(x, y) (((x) > (y)) ? (x) : (y))
#define fann_min(x, y) (((x) < (y)) ? (x) : (y))
#define fann_safe_free(x) if(x) free(x)
#define fann_clip(x, lo, hi) (((x) < (lo)) ? (lo) : (((x) > (hi)) ? (hi) : (x)))

#define fann_rand(min_value, max_value) (((double)(min_value))+(((double)(max_value)-((double)(min_value)))*rand()/(RAND_MAX+1.0)))

#define fann_abs(value) (((value) > 0) ? (value) : -(value))

#ifdef FIXEDFANN

#define fann_mult(x,y) ((x*y) >> decimal_point)
#define fann_div(x,y) (((x) << decimal_point)/y)
#define fann_random_weight() (fann_type)(fann_rand(-multiplier/10,multiplier/10))
/* sigmoid calculated with use of floats, only as reference */

#else

#define fann_mult(x,y) (x*y)
#define fann_div(x,y) (x/y)
#define fann_random_weight() (fann_rand(-0.1,0.1))

#endif

#endif
