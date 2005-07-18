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

/* Prints all of the parameters and options of the ANN */
FANN_EXTERNAL void FANN_API fann_print_parameters(struct fann *ann)
{
	struct fann_layer *layer_it;
	
	printf("Input layer                :%4d neurons, 1 bias\n", ann->num_input);
	for(layer_it = ann->first_layer+1; layer_it != ann->last_layer-1; layer_it++){
		if(ann->shortcut_connections){
			printf("  Hidden layer             :%4d neurons, 0 bias\n",
				layer_it->last_neuron - layer_it->first_neuron);
		} else {
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
	printf("Cascade no. of candidates  :%4d\n", ann->cascade_num_candidates);
}

FANN_EXTERNAL unsigned int FANN_API fann_get_training_algorithm(struct fann *ann)
{
	return ann->training_algorithm;
}

FANN_EXTERNAL void FANN_API fann_set_training_algorithm(struct fann *ann, unsigned int training_algorithm)
{
	ann->training_algorithm = training_algorithm;
}

FANN_EXTERNAL void FANN_API fann_set_learning_rate(struct fann *ann, float learning_rate)
{
	ann->learning_rate = learning_rate;
}

FANN_EXTERNAL void FANN_API fann_set_activation_function_hidden(struct fann *ann, unsigned int activation_function)
{
	struct fann_neuron *last_neuron, *neuron_it;
	struct fann_layer *layer_it;
	struct fann_layer *last_layer = ann->last_layer-1; /* -1 to not update the output layer */
	for(layer_it = ann->first_layer+1; layer_it != last_layer; layer_it++)
	{
		last_neuron = layer_it->last_neuron;
		for(neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++)
		{
			neuron_it->activation_function = activation_function;
		}
	}
}

FANN_EXTERNAL void FANN_API fann_set_activation_function_output(struct fann *ann, unsigned int activation_function)
{
	struct fann_neuron *last_neuron, *neuron_it;
	struct fann_layer *last_layer = ann->last_layer-1;

	last_neuron = last_layer->last_neuron;
	for(neuron_it = last_layer->first_neuron; neuron_it != last_neuron; neuron_it++)
	{
		neuron_it->activation_function = activation_function;
	}
}

FANN_EXTERNAL void FANN_API fann_set_activation_steepness_hidden(struct fann *ann, fann_type steepness)
{
	struct fann_neuron *last_neuron, *neuron_it;
	struct fann_layer *layer_it;
	struct fann_layer *last_layer = ann->last_layer-1; /* -1 to not update the output layer */
	for(layer_it = ann->first_layer+1; layer_it != last_layer; layer_it++)
	{
		last_neuron = layer_it->last_neuron;
		for(neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++)
		{
			neuron_it->activation_steepness = steepness;
		}
	}
}

FANN_EXTERNAL void FANN_API fann_set_activation_steepness_output(struct fann *ann, fann_type steepness)
{
	struct fann_neuron *last_neuron, *neuron_it;
	struct fann_layer *last_layer = ann->last_layer-1;

	last_neuron = last_layer->last_neuron;
	for(neuron_it = last_layer->first_neuron; neuron_it != last_neuron; neuron_it++)
	{
		neuron_it->activation_steepness = steepness;
	}
}

FANN_EXTERNAL void FANN_API fann_set_activation_hidden_steepness(struct fann *ann, fann_type steepness)
{
	fann_set_activation_steepness_hidden(ann, steepness);
}

FANN_EXTERNAL void FANN_API fann_set_activation_output_steepness(struct fann *ann, fann_type steepness)
{
	fann_set_activation_steepness_output(ann, steepness);
}

FANN_EXTERNAL float FANN_API fann_get_learning_rate(struct fann *ann)
{
	return ann->learning_rate;
}

FANN_EXTERNAL unsigned int FANN_API fann_get_num_input(struct fann *ann)
{
	return ann->num_input;
}

FANN_EXTERNAL unsigned int FANN_API fann_get_num_output(struct fann *ann)
{
	return ann->num_output;
}

/*
FANN_EXTERNAL unsigned int FANN_API fann_get_activation_function_hidden(struct fann *ann)
{
	return ann->activation_function_hidden;
}

FANN_EXTERNAL unsigned int FANN_API fann_get_activation_function_output(struct fann *ann)
{
	return ann->activation_function_output;
}

FANN_EXTERNAL fann_type FANN_API fann_get_activation_hidden_steepness(struct fann *ann)
{
	return ann->activation_steepness_hidden;
}

FANN_EXTERNAL fann_type FANN_API fann_get_activation_output_steepness(struct fann *ann)
{
	return ann->activation_steepness_output;
}

FANN_EXTERNAL fann_type FANN_API fann_get_activation_steepness_hidden(struct fann *ann)
{
	return ann->activation_steepness_hidden;
}

FANN_EXTERNAL fann_type FANN_API fann_get_activation_steepness_output(struct fann *ann)
{
	return ann->activation_steepness_output;
}
*/

FANN_EXTERNAL unsigned int FANN_API fann_get_total_neurons(struct fann *ann)
{
	if(ann->shortcut_connections){
		return ann->total_neurons;
	} else {
		/* -1, because there is always an unused bias neuron in the last layer */
		return ann->total_neurons - 1;
	}
}

FANN_EXTERNAL unsigned int FANN_API fann_get_total_connections(struct fann *ann)
{
	return ann->total_connections;
}

/* When using this, training is usually faster. (default ).
   Makes the error used for calculating the slopes
   higher when the difference is higher.
 */
FANN_EXTERNAL void FANN_API fann_set_train_error_function(struct fann *ann, unsigned int train_error_function)
{
	ann->train_error_function = train_error_function;
}

/* Decay is used to make the weights do not go so high (default -0.0001). */
FANN_EXTERNAL void FANN_API fann_set_quickprop_decay(struct fann *ann, float quickprop_decay)
{
	ann->quickprop_decay = quickprop_decay;
}
	
/* Mu is a factor used to increase and decrease the stepsize (default 1.75). */
FANN_EXTERNAL void FANN_API fann_set_quickprop_mu(struct fann *ann, float quickprop_mu)
{
	ann->quickprop_mu = quickprop_mu;
}

/* Tells how much the stepsize should increase during learning (default 1.2). */
FANN_EXTERNAL void FANN_API fann_set_rprop_increase_factor(struct fann *ann, float rprop_increase_factor)
{
	ann->rprop_increase_factor = rprop_increase_factor;
}

/* Tells how much the stepsize should decrease during learning (default 0.5). */
FANN_EXTERNAL void FANN_API fann_set_rprop_decrease_factor(struct fann *ann, float rprop_decrease_factor)
{
	ann->rprop_decrease_factor = rprop_decrease_factor;
}

/* The minimum stepsize (default 0.0). */
FANN_EXTERNAL void FANN_API fann_set_rprop_delta_min(struct fann *ann, float rprop_delta_min)
{
	ann->rprop_delta_min = rprop_delta_min;
}

/* The maximum stepsize (default 50.0). */
FANN_EXTERNAL void FANN_API fann_set_rprop_delta_max(struct fann *ann, float rprop_delta_max)
{
	ann->rprop_delta_max = rprop_delta_max;
}

/* When using this, training is usually faster. (default ).
   Makes the error used for calculating the slopes
   higher when the difference is higher.
 */
FANN_EXTERNAL unsigned int FANN_API fann_get_train_error_function(struct fann *ann)
{
	return ann->train_error_function;
}

/* Decay is used to make the weights do not go so high (default -0.0001). */
FANN_EXTERNAL float FANN_API fann_get_quickprop_decay(struct fann *ann)
{
	return ann->quickprop_decay;
}
	
/* Mu is a factor used to increase and decrease the stepsize (default 1.75). */
FANN_EXTERNAL float FANN_API fann_get_quickprop_mu(struct fann *ann)
{
	return ann->quickprop_mu;
}

/* Tells how much the stepsize should increase during learning (default 1.2). */
FANN_EXTERNAL float FANN_API fann_get_rprop_increase_factor(struct fann *ann)
{
	return ann->rprop_increase_factor;
}

/* Tells how much the stepsize should decrease during learning (default 0.5). */
FANN_EXTERNAL float FANN_API fann_get_rprop_decrease_factor(struct fann *ann)
{
	return ann->rprop_decrease_factor;
}

/* The minimum stepsize (default 0.0). */
FANN_EXTERNAL float FANN_API fann_get_rprop_delta_min(struct fann *ann)
{
	return ann->rprop_delta_min;
}

/* The maximum stepsize (default 50.0). */
FANN_EXTERNAL float FANN_API fann_get_rprop_delta_max(struct fann *ann)
{
	return ann->rprop_delta_max;
}

#ifdef FIXEDFANN
/* returns the position of the fix point.
 */
FANN_EXTERNAL unsigned int FANN_API fann_get_decimal_point(struct fann *ann)
{
	return ann->decimal_point;
}

/* returns the multiplier that fix point data is multiplied with.
 */
FANN_EXTERNAL unsigned int FANN_API fann_get_multiplier(struct fann *ann)
{
	return ann->multiplier;
}

/* INTERNAL FUNCTION
   Adjust the steepwise functions (if used)
*/
void fann_update_stepwise(struct fann *ann)
{
	unsigned int i = 0;
	/* Calculate the parameters for the stepwise linear
	   sigmoid function fixed point.
	   Using a rewritten sigmoid function.
	   results 0.005, 0.05, 0.25, 0.75, 0.95, 0.995
	*/
	ann->sigmoid_results[0] = fann_max((fann_type)(ann->multiplier/200.0+0.5), 1);
	ann->sigmoid_results[1] = (fann_type)(ann->multiplier/20.0+0.5);
	ann->sigmoid_results[2] = (fann_type)(ann->multiplier/4.0+0.5);
	ann->sigmoid_results[3] = ann->multiplier - (fann_type)(ann->multiplier/4.0+0.5);
	ann->sigmoid_results[4] = ann->multiplier - (fann_type)(ann->multiplier/20.0+0.5);
	ann->sigmoid_results[5] = fann_min(ann->multiplier - (fann_type)(ann->multiplier/200.0+0.5), ann->multiplier-1);

	ann->sigmoid_symmetric_results[0] = fann_max((fann_type)((ann->multiplier/100.0) - ann->multiplier-0.5), (fann_type)(1-(fann_type)ann->multiplier));
	ann->sigmoid_symmetric_results[1] = (fann_type)((ann->multiplier/10.0) - ann->multiplier-0.5);
	ann->sigmoid_symmetric_results[2] = (fann_type)((ann->multiplier/2.0) - ann->multiplier-0.5);
	ann->sigmoid_symmetric_results[3] = ann->multiplier - (fann_type)(ann->multiplier/2.0+0.5);
	ann->sigmoid_symmetric_results[4] = ann->multiplier - (fann_type)(ann->multiplier/10.0+0.5);
	ann->sigmoid_symmetric_results[5] = fann_min(ann->multiplier - (fann_type)(ann->multiplier/100.0+1.0), ann->multiplier-1);

	/*DEBUG
	ann->sigmoid_results[0] = (fann_type)(ann->multiplier/200.0+0.5);
	ann->sigmoid_results[1] = (fann_type)(ann->multiplier/20.0+0.5);
	ann->sigmoid_results[2] = (fann_type)(ann->multiplier/4.0+0.5);
	ann->sigmoid_results[3] = ann->multiplier - (fann_type)(ann->multiplier/4.0+0.5);
	ann->sigmoid_results[4] = ann->multiplier - (fann_type)(ann->multiplier/20.0+0.5);
	ann->sigmoid_results[5] = ann->multiplier - (fann_type)(ann->multiplier/200.0+0.5);

	ann->sigmoid_symmetric_results[0] = (fann_type)((ann->multiplier/100.0) - ann->multiplier + 0.5);
	ann->sigmoid_symmetric_results[1] = (fann_type)((ann->multiplier/10.0) - ann->multiplier + 0.5);
	ann->sigmoid_symmetric_results[2] = (fann_type)((ann->multiplier/2.0) - ann->multiplier + 0.5);
	ann->sigmoid_symmetric_results[3] = ann->multiplier - (fann_type)(ann->multiplier/2.0+0.5);
	ann->sigmoid_symmetric_results[4] = ann->multiplier - (fann_type)(ann->multiplier/10.0+0.5);
	ann->sigmoid_symmetric_results[5] = ann->multiplier - (fann_type)(ann->multiplier/100.0+0.5);
	*/
	
	for(i = 0; i < 6; i++){
		ann->sigmoid_values[i] = (fann_type)(((log(ann->multiplier/(float)ann->sigmoid_results[i] -1)*(float)ann->multiplier) / -2.0)*(float)ann->multiplier);
		ann->sigmoid_symmetric_values[i] = (fann_type)(((log((ann->multiplier - (float)ann->sigmoid_symmetric_results[i])/((float)ann->sigmoid_symmetric_results[i] + ann->multiplier))*(float)ann->multiplier) / -2.0)*(float)ann->multiplier);
	}
}
#endif
