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

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

#include "config.h"
#include "fann.h"
#include "fann_errno.h"

FANN_EXTERNAL void FANN_API fann_print_parameters(struct fann *ann)
{
	struct fann_layer *layer_it;

	printf("Input layer                :%4d neurons, 1 bias\n", ann->num_input);
	for(layer_it = ann->first_layer + 1; layer_it != ann->last_layer - 1; layer_it++)
	{
		if(ann->shortcut_connections)
		{
			printf("  Hidden layer             :%4d neurons, 0 bias\n",
				   layer_it->last_neuron - layer_it->first_neuron);
		}
		else
		{
			printf("  Hidden layer             :%4d neurons, 1 bias\n",
				   layer_it->last_neuron - layer_it->first_neuron - 1);
		}
	}
	printf("Output layer               :%4d neurons\n", ann->num_output);
	printf("Total neurons and biases   :%4d\n", fann_get_total_neurons(ann));
	printf("Total connections          :%4d\n", ann->total_connections);
	printf("Connection rate            :  %5.2f\n", ann->connection_rate);
	printf("Shortcut connections       :%4d\n", ann->shortcut_connections);
	printf("Training algorithm         :   %s\n", FANN_TRAIN_NAMES[ann->training_algorithm]);
	printf("Learning rate              :  %5.2f\n", ann->learning_rate);
/*	printf("Activation function hidden :   %s\n", FANN_ACTIVATION_NAMES[ann->activation_function_hidden]);
	printf("Activation function output :   %s\n", FANN_ACTIVATION_NAMES[ann->activation_function_output]);
*/
#ifndef FIXEDFANN
/*
	printf("Activation steepness hidden:  %5.2f\n", ann->activation_steepness_hidden);
	printf("Activation steepness output:  %5.2f\n", ann->activation_steepness_output);
*/
#else
/*
	printf("Activation steepness hidden:  %d\n", ann->activation_steepness_hidden);
	printf("Activation steepness output:  %d\n", ann->activation_steepness_output);
*/
	printf("Decimal point              :%4d\n", ann->decimal_point);
	printf("Multiplier                 :%4d\n", ann->multiplier);
#endif
	printf("Training error function    :   %s\n", FANN_ERRORFUNC_NAMES[ann->train_error_function]);
	printf("Quickprop decay            :  %9.6f\n", ann->quickprop_decay);
	printf("Quickprop mu               :  %5.2f\n", ann->quickprop_mu);
	printf("RPROP increase factor      :  %5.2f\n", ann->rprop_increase_factor);
	printf("RPROP decrease factor      :  %5.2f\n", ann->rprop_decrease_factor);
	printf("RPROP delta min            :  %5.2f\n", ann->rprop_delta_min);
	printf("RPROP delta max            :  %5.2f\n", ann->rprop_delta_max);
	printf("Cascade change fraction    :  %9.6f\n", ann->cascade_change_fraction);
	printf("Cascade stagnation epochs  :%4d\n", ann->cascade_stagnation_epochs);
	printf("Cascade no. of candidates  :%4d\n", fann_get_cascade_num_candidates(ann));
}

#define FANN_GET(type, name) \
FANN_EXTERNAL type FANN_API fann_get_ ## name(struct fann *ann) \
{ \
	return ann->name; \
}

#define FANN_SET(type, name) \
FANN_EXTERNAL void FANN_API fann_set_ ## name(struct fann *ann, type value) \
{ \
	ann->name = value; \
}

#define FANN_GET_SET(type, name) \
FANN_GET(type, name) \
FANN_SET(type, name)

FANN_GET_SET(enum fann_train_enum, training_algorithm)
FANN_GET_SET(float, learning_rate)

FANN_EXTERNAL void FANN_API fann_set_activation_function_hidden(struct fann *ann,
																enum fann_activationfunc_enum activation_function)
{
	struct fann_neuron *last_neuron, *neuron_it;
	struct fann_layer *layer_it;
	struct fann_layer *last_layer = ann->last_layer - 1;	/* -1 to not update the output layer */

	for(layer_it = ann->first_layer + 1; layer_it != last_layer; layer_it++)
	{
		last_neuron = layer_it->last_neuron;
		for(neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++)
		{
			neuron_it->activation_function = activation_function;
		}
	}
}

FANN_EXTERNAL void FANN_API fann_set_activation_function_output(struct fann *ann,
																enum fann_activationfunc_enum activation_function)
{
	struct fann_neuron *last_neuron, *neuron_it;
	struct fann_layer *last_layer = ann->last_layer - 1;

	last_neuron = last_layer->last_neuron;
	for(neuron_it = last_layer->first_neuron; neuron_it != last_neuron; neuron_it++)
	{
		neuron_it->activation_function = activation_function;
	}
}

FANN_EXTERNAL void FANN_API fann_set_activation_steepness_hidden(struct fann *ann,
																 fann_type steepness)
{
	struct fann_neuron *last_neuron, *neuron_it;
	struct fann_layer *layer_it;
	struct fann_layer *last_layer = ann->last_layer - 1;	/* -1 to not update the output layer */

	for(layer_it = ann->first_layer + 1; layer_it != last_layer; layer_it++)
	{
		last_neuron = layer_it->last_neuron;
		for(neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++)
		{
			neuron_it->activation_steepness = steepness;
		}
	}
}

FANN_EXTERNAL void FANN_API fann_set_activation_steepness_output(struct fann *ann,
																 fann_type steepness)
{
	struct fann_neuron *last_neuron, *neuron_it;
	struct fann_layer *last_layer = ann->last_layer - 1;

	last_neuron = last_layer->last_neuron;
	for(neuron_it = last_layer->first_neuron; neuron_it != last_neuron; neuron_it++)
	{
		neuron_it->activation_steepness = steepness;
	}
}

FANN_GET(unsigned int, num_input)
FANN_GET(unsigned int, num_output)

FANN_EXTERNAL unsigned int FANN_API fann_get_total_neurons(struct fann *ann)
{
	if(ann->shortcut_connections)
	{
		return ann->total_neurons;
	}
	else
	{
		/* -1, because there is always an unused bias neuron in the last layer */
		return ann->total_neurons - 1;
	}
}

FANN_GET(unsigned int, total_connections)
FANN_GET_SET(enum fann_errorfunc_enum, train_error_function)
FANN_GET_SET(float, quickprop_decay)
FANN_GET_SET(float, quickprop_mu)
FANN_GET_SET(float, rprop_increase_factor)
FANN_GET_SET(float, rprop_decrease_factor)
FANN_GET_SET(float, rprop_delta_min)
FANN_GET_SET(float, rprop_delta_max)
FANN_GET_SET(enum fann_stopfunc_enum, train_stop_function)

FANN_EXTERNAL unsigned int FANN_API fann_get_cascade_num_candidates(struct fann *ann)
{
	return ann->cascade_activation_functions_count *
		ann->cascade_activation_steepnesses_count *
		ann->cascade_num_candidate_groups;
}

FANN_GET_SET(float, cascade_change_fraction)
FANN_GET_SET(unsigned int, cascade_stagnation_epochs)
FANN_GET_SET(unsigned int, cascade_num_candidate_groups)
FANN_GET_SET(fann_type, cascade_weight_multiplier)
FANN_GET_SET(fann_type, cascade_candidate_limit)
FANN_GET_SET(unsigned int, cascade_max_out_epochs)
FANN_GET_SET(unsigned int, cascade_max_cand_epochs)

FANN_GET(unsigned int, cascade_activation_functions_count)
FANN_GET(enum fann_activationfunc_enum *, cascade_activation_functions)

FANN_EXTERNAL void fann_set_cascade_activation_functions(struct fann *ann,
														 enum fann_activationfunc_enum *
														 cascade_activation_functions,
														 unsigned int 
														 cascade_activation_functions_count)
{
	if(ann->cascade_activation_functions_count != cascade_activation_functions_count)
	{
		ann->cascade_activation_functions_count = cascade_activation_functions_count;
		
		/* reallocate mem */
		ann->cascade_activation_functions = 
			(enum fann_activationfunc_enum *)realloc(ann->cascade_activation_functions, 
			ann->cascade_activation_functions_count * sizeof(enum fann_activationfunc_enum));
		if(ann->cascade_activation_functions == NULL)
		{
			fann_error((struct fann_error*)ann, FANN_E_CANT_ALLOCATE_MEM);
			return;
		}
	}
	
	memmove(ann->cascade_activation_functions, cascade_activation_functions, 
		ann->cascade_activation_functions_count * sizeof(enum fann_activationfunc_enum));
}

FANN_GET(unsigned int, cascade_activation_steepnesses_count)
FANN_GET(fann_type *, cascade_activation_steepnesses)

FANN_EXTERNAL void fann_set_cascade_activation_steepnesses(struct fann *ann,
														   fann_type *
														   cascade_activation_steepnesses,
														   unsigned int 
														   cascade_activation_steepnesses_count)
{
	if(ann->cascade_activation_steepnesses_count != cascade_activation_steepnesses_count)
	{
		ann->cascade_activation_steepnesses_count = cascade_activation_steepnesses_count;
		
		/* reallocate mem */
		ann->cascade_activation_steepnesses = 
			(fann_type *)realloc(ann->cascade_activation_steepnesses, 
			ann->cascade_activation_steepnesses_count * sizeof(fann_type));
		if(ann->cascade_activation_steepnesses == NULL)
		{
			fann_error((struct fann_error*)ann, FANN_E_CANT_ALLOCATE_MEM);
			return;
		}
	}
	
	memmove(ann->cascade_activation_steepnesses, cascade_activation_steepnesses, 
		ann->cascade_activation_steepnesses_count * sizeof(fann_type));
}

#ifdef FIXEDFANN

FANN_GET(unsigned int, decimal_point)
FANN_GET(unsigned int, multiplier)

/* INTERNAL FUNCTION
   Adjust the steepwise functions (if used)
*/
void fann_update_stepwise(struct fann *ann)
{
	unsigned int i = 0;

	/* Calculate the parameters for the stepwise linear
	 * sigmoid function fixed point.
	 * Using a rewritten sigmoid function.
	 * results 0.005, 0.05, 0.25, 0.75, 0.95, 0.995
	 */
	ann->sigmoid_results[0] = fann_max((fann_type) (ann->multiplier / 200.0 + 0.5), 1);
	ann->sigmoid_results[1] = (fann_type) (ann->multiplier / 20.0 + 0.5);
	ann->sigmoid_results[2] = (fann_type) (ann->multiplier / 4.0 + 0.5);
	ann->sigmoid_results[3] = ann->multiplier - (fann_type) (ann->multiplier / 4.0 + 0.5);
	ann->sigmoid_results[4] = ann->multiplier - (fann_type) (ann->multiplier / 20.0 + 0.5);
	ann->sigmoid_results[5] =
		fann_min(ann->multiplier - (fann_type) (ann->multiplier / 200.0 + 0.5),
				 ann->multiplier - 1);

	ann->sigmoid_symmetric_results[0] =
		fann_max((fann_type) ((ann->multiplier / 100.0) - ann->multiplier - 0.5),
				 (fann_type) (1 - (fann_type) ann->multiplier));
	ann->sigmoid_symmetric_results[1] =
		(fann_type) ((ann->multiplier / 10.0) - ann->multiplier - 0.5);
	ann->sigmoid_symmetric_results[2] =
		(fann_type) ((ann->multiplier / 2.0) - ann->multiplier - 0.5);
	ann->sigmoid_symmetric_results[3] = ann->multiplier - (fann_type) (ann->multiplier / 2.0 + 0.5);
	ann->sigmoid_symmetric_results[4] =
		ann->multiplier - (fann_type) (ann->multiplier / 10.0 + 0.5);
	ann->sigmoid_symmetric_results[5] =
		fann_min(ann->multiplier - (fann_type) (ann->multiplier / 100.0 + 1.0),
				 ann->multiplier - 1);

	for(i = 0; i < 6; i++)
	{
		ann->sigmoid_values[i] =
			(fann_type) (((log(ann->multiplier / (float) ann->sigmoid_results[i] - 1) *
						   (float) ann->multiplier) / -2.0) * (float) ann->multiplier);
		ann->sigmoid_symmetric_values[i] =
			(fann_type) (((log
						   ((ann->multiplier -
							 (float) ann->sigmoid_symmetric_results[i]) /
							((float) ann->sigmoid_symmetric_results[i] +
							 ann->multiplier)) * (float) ann->multiplier) / -2.0) *
						 (float) ann->multiplier);
	}
}
#endif
