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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <stdarg.h>

#include "compat_time.h"
#include "fann.h"
#include "fann_internal.h"
#include "fann_errno.h"

/* Allocates the main structure and sets some default values.
 */
struct fann * fann_allocate_structure(float learning_rate, unsigned int num_layers)
{
	struct fann *ann;
	
	if(num_layers < 2){
#ifdef DEBUG
		printf("less than 2 layers - ABORTING.\n");
#endif
		return NULL;
	}

	/* allocate and initialize the main network structure */
	ann = (struct fann *)malloc(sizeof(struct fann));
	if(ann == NULL){
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		return NULL;
	}

	ann->learning_rate = learning_rate;
	ann->total_neurons = 0;
	ann->total_connections = 0;
	ann->num_input = 0;
	ann->num_output = 0;
	ann->train_deltas = NULL;
	ann->num_errors = 0;
	ann->error_value = 0;
	ann->errstr = NULL;
	ann->errno_f = 0;
	ann->error_log = stderr;

#ifdef FIXEDFANN
	/* these values are only boring defaults, and should really
	   never be used, since the real values are always loaded from a file. */
	ann->decimal_point = 8;
	ann->multiplier = 256;
#endif
	
	ann->activation_function_hidden = FANN_SIGMOID_STEPWISE;
	ann->activation_function_output = FANN_SIGMOID_STEPWISE;
#ifdef FIXEDFANN
	ann->activation_hidden_steepness = ann->multiplier/2;
	ann->activation_output_steepness = ann->multiplier/2;
#else
	ann->activation_hidden_steepness = 0.5;
	ann->activation_output_steepness = 0.5;
#endif

	/* allocate room for the layers */
	ann->first_layer = (struct fann_layer *)calloc(num_layers, sizeof(struct fann_layer));
	if(ann->first_layer == NULL){
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		free(ann);
		return NULL;
	}
	
	ann->last_layer = ann->first_layer + num_layers;

	return ann;
}

/* Allocates room for the neurons.
 */
void fann_allocate_neurons(struct fann *ann)
{
	struct fann_layer *layer_it;
	struct fann_neuron *neurons;
	unsigned int num_neurons_so_far = 0;
	unsigned int num_neurons = 0;

	/* all the neurons is allocated in one long array */
	neurons = (struct fann_neuron *)calloc(ann->total_neurons, sizeof(struct fann_neuron));
	if(neurons == NULL){
		fann_error(ann, FANN_E_CANT_ALLOCATE_MEM);
		return;
	}
	
	/* clear data, primarily to make the input neurons cleared */
	memset(neurons, 0, ann->total_neurons * sizeof(struct fann_neuron));
	
	for(layer_it = ann->first_layer; layer_it != ann->last_layer; layer_it++){
		num_neurons = layer_it->last_neuron - layer_it->first_neuron;
		layer_it->first_neuron = neurons+num_neurons_so_far;
		layer_it->last_neuron = layer_it->first_neuron+num_neurons;
		num_neurons_so_far += num_neurons;
	}

	ann->output = (fann_type *)calloc(num_neurons, sizeof(fann_type));
	if(ann->output == NULL){
		fann_error(ann, FANN_E_CANT_ALLOCATE_MEM);
		return;
	}
}

/* Allocate room for the connections.
 */
void fann_allocate_connections(struct fann *ann)
{
	struct fann_layer *layer_it, *last_layer;
	struct fann_neuron *neuron_it, *last_neuron;
	fann_type *weights;
	struct fann_neuron **connected_neurons = NULL;
	unsigned int connections_so_far = 0;
	
	weights = (fann_type *)calloc(ann->total_connections, sizeof(fann_type));
	if(weights == NULL){
		fann_error(ann, FANN_E_CANT_ALLOCATE_MEM);
		return;
	}
	
	/* TODO make special cases for all places where the connections
	   is used, so that it is not needed for fully connected networks.
	*/
	connected_neurons = (struct fann_neuron **) calloc(ann->total_connections, sizeof(struct fann_neuron*));
	if(connected_neurons == NULL){
		fann_error(ann, FANN_E_CANT_ALLOCATE_MEM);
		return;
	}
	

	last_layer = ann->last_layer;
	for(layer_it = ann->first_layer+1; layer_it != ann->last_layer; layer_it++){
		last_neuron = layer_it->last_neuron-1;
		for(neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++){
			neuron_it->weights = weights+connections_so_far;
			neuron_it->connected_neurons = connected_neurons+connections_so_far;
			connections_so_far += neuron_it->num_connections;
		}
	}

	if(connections_so_far != ann->total_connections){
		fann_error(ann, FANN_E_WRONG_NUM_CONNECTIONS, connections_so_far, ann->total_connections);
		return;
	}
}

/* Used to save the network to a file.
 */
int fann_save_internal(struct fann *ann, const char *configuration_file, unsigned int save_as_fixed)
{
	int retval;
	FILE *conf = fopen(configuration_file, "w+");
	if(!conf){
		fann_error(NULL, FANN_E_CANT_OPEN_CONFIG_W, configuration_file);
		return -1;
	}
	retval = fann_save_internal_fd(ann, conf, configuration_file, save_as_fixed);
	fclose(conf);
	return retval;
}

/* Used to save the network to a file descriptor.
 */
int fann_save_internal_fd(struct fann *ann, FILE *conf, const char *configuration_file, unsigned int save_as_fixed)
{
	struct fann_layer *layer_it;
	int calculated_decimal_point = 0;
	struct fann_neuron *neuron_it, *first_neuron;
	fann_type *weights;
	struct fann_neuron **connected_neurons;
	unsigned int i = 0;
#ifndef FIXEDFANN
	/* variabels for use when saving floats as fixed point variabels */
	unsigned int decimal_point = 0;
	unsigned int fixed_multiplier = 0;
	fann_type max_possible_value = 0;
	unsigned int bits_used_for_max = 0;
	fann_type current_max_value = 0;
#endif

#ifndef FIXEDFANN
	if(save_as_fixed){
		/* save the version information */
		fprintf(conf, FANN_FIX_VERSION"\n");
	}else{
		/* save the version information */
		fprintf(conf, FANN_FLO_VERSION"\n");
	}
#else
	/* save the version information */
	fprintf(conf, FANN_FIX_VERSION"\n");
#endif
	
#ifndef FIXEDFANN
	if(save_as_fixed){
		/* calculate the maximal possible shift value */

		for(layer_it = ann->first_layer+1; layer_it != ann->last_layer; layer_it++){
			for(neuron_it = layer_it->first_neuron; neuron_it != layer_it->last_neuron; neuron_it++){
				/* look at all connections to each neurons, and see how high a value we can get */
				current_max_value = 0;
				for(i = 0; i != neuron_it->num_connections; i++){
					current_max_value += fann_abs(neuron_it->weights[i]);
				}

				if(current_max_value > max_possible_value){
					max_possible_value = current_max_value;
				}
			}
		}

		for(bits_used_for_max = 0; max_possible_value >= 1; bits_used_for_max++){
			max_possible_value /= 2.0;
		}

		/* The maximum number of bits we shift the fix point, is the number
		   of bits in a integer, minus one for the sign, one for the minus
		   in stepwise, and minus the bits used for the maximum.
		   This is devided by two, to allow multiplication of two fixed
		   point numbers.
		*/
		calculated_decimal_point = (sizeof(int)*8-2-bits_used_for_max)/2;

		if(calculated_decimal_point < 0){
			decimal_point = 0;
		}else{
			decimal_point = calculated_decimal_point;
		}
		
		fixed_multiplier = 1 << decimal_point;

#ifdef DEBUG
		printf("calculated_decimal_point=%d, decimal_point=%u, bits_used_for_max=%u\n", calculated_decimal_point, decimal_point, bits_used_for_max);
#endif
		
		/* save the decimal_point on a seperate line */
		fprintf(conf, "%u\n", decimal_point);
		
		/* save the number layers "num_layers learning_rate connection_rate activation_function_hidden activation_function_output activation_hidden_steepness activation_output_steepness" */	
		fprintf(conf, "%u %f %f %u %u %d %d\n", ann->last_layer - ann->first_layer, ann->learning_rate, ann->connection_rate, ann->activation_function_hidden, ann->activation_function_output, (int)(ann->activation_hidden_steepness * fixed_multiplier), (int)(ann->activation_output_steepness * fixed_multiplier));
	}else{
		/* save the number layers "num_layers learning_rate connection_rate activation_function_hidden activation_function_output activation_hidden_steepness activation_output_steepness" */	
		fprintf(conf, "%u %f %f %u %u "FANNPRINTF" "FANNPRINTF"\n", ann->last_layer - ann->first_layer, ann->learning_rate, ann->connection_rate, ann->activation_function_hidden, ann->activation_function_output, ann->activation_hidden_steepness, ann->activation_output_steepness);
	}
#else
	/* save the decimal_point on a seperate line */
	fprintf(conf, "%u\n", ann->decimal_point);
	
	/* save the number layers "num_layers learning_rate connection_rate activation_function_hidden activation_function_output activation_hidden_steepness activation_output_steepness" */	
	fprintf(conf, "%u %f %f %u %u "FANNPRINTF" "FANNPRINTF"\n", ann->last_layer - ann->first_layer, ann->learning_rate, ann->connection_rate, ann->activation_function_hidden, ann->activation_function_output, ann->activation_hidden_steepness, ann->activation_output_steepness);	
#endif

	for(layer_it = ann->first_layer; layer_it != ann->last_layer; layer_it++){
		/* the number of neurons in the layers (in the last layer, there is always one too many neurons, because of an unused bias) */
		fprintf(conf, "%u ", layer_it->last_neuron - layer_it->first_neuron);
	}
	fprintf(conf, "\n");

	
	for(layer_it = ann->first_layer; layer_it != ann->last_layer; layer_it++){
		/* the number of connections to each neuron */
		for(neuron_it = layer_it->first_neuron; neuron_it != layer_it->last_neuron; neuron_it++){
			fprintf(conf, "%u ", neuron_it->num_connections);
		}
		fprintf(conf, "\n");
	}

	connected_neurons = (ann->first_layer+1)->first_neuron->connected_neurons;
	weights = (ann->first_layer+1)->first_neuron->weights;
	first_neuron = ann->first_layer->first_neuron;
	
	/* Now save all the connections.
	   We only need to save the source and the weight,
	   since the destination is given by the order.

	   The weight is not saved binary due to differences
	   in binary definition of floating point numbers.
	   Especially an iPAQ does not use the same binary
	   representation as an i386 machine.
	 */
	for(i = 0; i < ann->total_connections; i++){
#ifndef FIXEDFANN
		if(save_as_fixed){
			/* save the connection "(source weight) "*/
			fprintf(conf, "(%u %d) ",
				connected_neurons[i] - first_neuron,
				(int)floor((weights[i]*fixed_multiplier) + 0.5));
		}else{
			/* save the connection "(source weight) "*/
			fprintf(conf, "(%u "FANNPRINTF") ",
				connected_neurons[i] - first_neuron, weights[i]);
		}
#else
		/* save the connection "(source weight) "*/
		fprintf(conf, "(%u "FANNPRINTF") ",
			connected_neurons[i] - first_neuron, weights[i]);
#endif
		
	}
	fprintf(conf, "\n");

	return calculated_decimal_point;
}

/* Save the train data structure.
 */
void fann_save_train_internal(struct fann_train_data* data, char *filename, unsigned int save_as_fixed, unsigned int decimal_point)
{	
	FILE *file = fopen(filename, "w");
	if(!file){
		fann_error(NULL, FANN_E_CANT_OPEN_TD_W, filename);
		return;
	}
	fann_save_train_internal_fd(data, file, filename, save_as_fixed, decimal_point);
	fclose(file);
}

/* Save the train data structure.
 */
void fann_save_train_internal_fd(struct fann_train_data* data, FILE *file, char *filename, unsigned int save_as_fixed, unsigned int decimal_point)
{
	unsigned int num_data = data->num_data;
	unsigned int num_input = data->num_input;
	unsigned int num_output = data->num_output;
	unsigned int i, j;
#ifndef FIXEDFANN
	unsigned int multiplier = 1 << decimal_point;
#endif
	
	fprintf(file, "%u %u %u\n", data->num_data, data->num_input, data->num_output);

	for(i = 0; i < num_data; i++){
		for(j = 0; j < num_input; j++){
#ifndef FIXEDFANN
			if(save_as_fixed){
				fprintf(file, "%d ", (int)(data->input[i][j]*multiplier));
			}else{
				fprintf(file, FANNPRINTF" ", data->input[i][j]);
			}
#else
			fprintf(file, FANNPRINTF" ", data->input[i][j]);
#endif
		}
		fprintf(file, "\n");

		for(j = 0; j < num_output; j++){
#ifndef FIXEDFANN
			if(save_as_fixed){
				fprintf(file, "%d ", (int)(data->output[i][j]*multiplier));
			}else{
				fprintf(file, FANNPRINTF" ", data->output[i][j]);
			}
#else
			fprintf(file, FANNPRINTF" ", data->output[i][j]);
#endif
		}
		fprintf(file, "\n");
	}
}

/* Adjust the steepwise functions (if used) */
void fann_update_stepwise_hidden(struct fann *ann)
{
	unsigned int i = 0;
#ifndef FIXEDFANN
	/* For use in stepwise linear activation function.
	   results 0.005, 0.05, 0.25, 0.75, 0.95, 0.995
	*/
	switch(ann->activation_function_hidden){
		case FANN_SIGMOID:
		case FANN_SIGMOID_STEPWISE:
			ann->activation_hidden_results[0] = 0.005;
			ann->activation_hidden_results[1] = 0.05;
			ann->activation_hidden_results[2] = 0.25;
			ann->activation_hidden_results[3] = 0.75;
			ann->activation_hidden_results[4] = 0.95;
			ann->activation_hidden_results[5] = 0.995;	
			break;
		case FANN_SIGMOID_SYMMETRIC:
		case FANN_SIGMOID_SYMMETRIC_STEPWISE:
			ann->activation_hidden_results[0] = -0.99;
			ann->activation_hidden_results[1] = -0.9;
			ann->activation_hidden_results[2] = -0.5;
			ann->activation_hidden_results[3] = 0.5;
			ann->activation_hidden_results[4] = 0.9;
			ann->activation_hidden_results[5] = 0.99;
			break;
		default:
			/* the actiavation functions which do not have a stepwise function
			   should not have it calculated */
			return;
	}
#else
	/* Calculate the parameters for the stepwise linear
	   sigmoid function fixed point.
	   Using a rewritten sigmoid function.
	   results 0.005, 0.05, 0.25, 0.75, 0.95, 0.995
	*/
	switch(ann->activation_function_hidden){
		case FANN_SIGMOID:
		case FANN_SIGMOID_STEPWISE:
			ann->activation_hidden_results[0] = (fann_type)(ann->multiplier/200.0+0.5);
			ann->activation_hidden_results[1] = (fann_type)(ann->multiplier/20.0+0.5);
			ann->activation_hidden_results[2] = (fann_type)(ann->multiplier/4.0+0.5);
			ann->activation_hidden_results[3] = ann->multiplier - (fann_type)(ann->multiplier/4.0+0.5);
			ann->activation_hidden_results[4] = ann->multiplier - (fann_type)(ann->multiplier/20.0+0.5);
			ann->activation_hidden_results[5] = ann->multiplier - (fann_type)(ann->multiplier/200.0+0.5);
			break;
		case FANN_SIGMOID_SYMMETRIC:
		case FANN_SIGMOID_SYMMETRIC_STEPWISE:
			ann->activation_hidden_results[0] = (fann_type)((ann->multiplier/100.0) - ann->multiplier + 0.5);
			ann->activation_hidden_results[1] = (fann_type)((ann->multiplier/10.0) - ann->multiplier + 0.5);
			ann->activation_hidden_results[2] = (fann_type)((ann->multiplier/2.0) - ann->multiplier + 0.5);
			ann->activation_hidden_results[3] = ann->multiplier - (fann_type)(ann->multiplier/2.0+0.5);
			ann->activation_hidden_results[4] = ann->multiplier - (fann_type)(ann->multiplier/10.0+0.5);
			ann->activation_hidden_results[5] = ann->multiplier - (fann_type)(ann->multiplier/100.0+0.5);
			break;
		default:
			/* the actiavation functions which do not have a stepwise function
			   should not have it calculated */
			return;
	}			
#endif

	for(i = 0; i < 6; i++){
#ifndef FIXEDFANN
		switch(ann->activation_function_hidden){
			case FANN_SIGMOID:
				break;
			case FANN_SIGMOID_STEPWISE:
				ann->activation_hidden_values[i] = ((log(1.0/ann->activation_hidden_results[i] -1.0) * 1.0/-2.0) * 1.0/ann->activation_hidden_steepness);
				break;
			case FANN_SIGMOID_SYMMETRIC:
			case FANN_SIGMOID_SYMMETRIC_STEPWISE:
				ann->activation_hidden_values[i] = ((log((1.0-ann->activation_hidden_results[i]) / (ann->activation_hidden_results[i]+1.0)) * 1.0/-2.0) * 1.0/ann->activation_hidden_steepness);
				break;
		}
#else
		switch(ann->activation_function_hidden){
			case FANN_SIGMOID:
			case FANN_SIGMOID_STEPWISE:
				ann->activation_hidden_values[i] = (fann_type)((((log(ann->multiplier/(float)ann->activation_hidden_results[i] -1)*(float)ann->multiplier) / -2.0)*(float)ann->multiplier) / ann->activation_hidden_steepness);
				break;
			case FANN_SIGMOID_SYMMETRIC:
			case FANN_SIGMOID_SYMMETRIC_STEPWISE:
				ann->activation_hidden_values[i] = (fann_type)((((log((ann->multiplier - (float)ann->activation_hidden_results[i])/((float)ann->activation_hidden_results[i] + ann->multiplier))*(float)ann->multiplier) / -2.0)*(float)ann->multiplier) / ann->activation_hidden_steepness);
				break;
		}
#endif
	}
}

/* Adjust the steepwise functions (if used) */
void fann_update_stepwise_output(struct fann *ann)
{
	unsigned int i = 0;
#ifndef FIXEDFANN
	/* For use in stepwise linear activation function.
	   results 0.005, 0.05, 0.25, 0.75, 0.95, 0.995
	*/
	switch(ann->activation_function_output){
		case FANN_SIGMOID:
		case FANN_SIGMOID_STEPWISE:
			ann->activation_output_results[0] = 0.005;
			ann->activation_output_results[1] = 0.05;
			ann->activation_output_results[2] = 0.25;
			ann->activation_output_results[3] = 0.75;
			ann->activation_output_results[4] = 0.95;
			ann->activation_output_results[5] = 0.995;	
			break;
		case FANN_SIGMOID_SYMMETRIC:
		case FANN_SIGMOID_SYMMETRIC_STEPWISE:
			ann->activation_output_results[0] = -0.99;
			ann->activation_output_results[1] = -0.9;
			ann->activation_output_results[2] = -0.5;
			ann->activation_output_results[3] = 0.5;
			ann->activation_output_results[4] = 0.9;
			ann->activation_output_results[5] = 0.99;
			break;
		default:
			/* the actiavation functions which do not have a stepwise function
			   should not have it calculated */
			return;
	}
#else
	/* Calculate the parameters for the stepwise linear
	   sigmoid function fixed point.
	   Using a rewritten sigmoid function.
	   results 0.005, 0.05, 0.25, 0.75, 0.95, 0.995
	*/
	switch(ann->activation_function_output){
		case FANN_SIGMOID:
		case FANN_SIGMOID_STEPWISE:
			ann->activation_output_results[0] = (fann_type)(ann->multiplier/200.0+0.5);
			ann->activation_output_results[1] = (fann_type)(ann->multiplier/20.0+0.5);
			ann->activation_output_results[2] = (fann_type)(ann->multiplier/4.0+0.5);
			ann->activation_output_results[3] = ann->multiplier - (fann_type)(ann->multiplier/4.0+0.5);
			ann->activation_output_results[4] = ann->multiplier - (fann_type)(ann->multiplier/20.0+0.5);
			ann->activation_output_results[5] = ann->multiplier - (fann_type)(ann->multiplier/200.0+0.5);
			break;
		case FANN_SIGMOID_SYMMETRIC:
		case FANN_SIGMOID_SYMMETRIC_STEPWISE:
			ann->activation_output_results[0] = (fann_type)((ann->multiplier/100.0) - ann->multiplier + 0.5);
			ann->activation_output_results[1] = (fann_type)((ann->multiplier/10.0) - ann->multiplier + 0.5);
			ann->activation_output_results[2] = (fann_type)((ann->multiplier/2.0) - ann->multiplier + 0.5);
			ann->activation_output_results[3] = ann->multiplier - (fann_type)(ann->multiplier/2.0+0.5);
			ann->activation_output_results[4] = ann->multiplier - (fann_type)(ann->multiplier/10.0+0.5);
			ann->activation_output_results[5] = ann->multiplier - (fann_type)(ann->multiplier/100.0+0.5);
			break;
		default:
			/* the actiavation functions which do not have a stepwise function
			   should not have it calculated */
			return;
	}			
#endif

	for(i = 0; i < 6; i++){
#ifndef FIXEDFANN
		switch(ann->activation_function_output){
			case FANN_SIGMOID:
				break;
			case FANN_SIGMOID_STEPWISE:
				ann->activation_output_values[i] = ((log(1.0/ann->activation_output_results[i] -1.0) * 1.0/-2.0) * 1.0/ann->activation_output_steepness);
				break;
			case FANN_SIGMOID_SYMMETRIC:
			case FANN_SIGMOID_SYMMETRIC_STEPWISE:
				ann->activation_output_values[i] = ((log((1.0-ann->activation_output_results[i]) / (ann->activation_output_results[i]+1.0)) * 1.0/-2.0) * 1.0/ann->activation_output_steepness);
				break;
		}
#else
		switch(ann->activation_function_output){
			case FANN_SIGMOID:
			case FANN_SIGMOID_STEPWISE:
				ann->activation_output_values[i] = (fann_type)((((log(ann->multiplier/(float)ann->activation_output_results[i] -1)*(float)ann->multiplier) / -2.0)*(float)ann->multiplier) / ann->activation_output_steepness);
				break;
			case FANN_SIGMOID_SYMMETRIC:
			case FANN_SIGMOID_SYMMETRIC_STEPWISE:
				ann->activation_output_values[i] = (fann_type)((((log((ann->multiplier - (float)ann->activation_output_results[i])/((float)ann->activation_output_results[i] + ann->multiplier))*(float)ann->multiplier) / -2.0)*(float)ann->multiplier) / ann->activation_output_steepness);
				break;
		}
#endif
	}
}


/* Seed the random function.
 */
void fann_seed_rand()
{
	FILE *fp = fopen("/dev/urandom", "r");
	unsigned int foo;
	struct timeval t;
	if(!fp){
		gettimeofday(&t, NULL);
		foo = t.tv_usec;
#ifdef DEBUG
		printf("unable to open /dev/urandom\n");
#endif
	}else{
		fread(&foo, sizeof(foo), 1, fp);
		fclose(fp);
	}
	srand(foo);
}

/* Populate the error information
 */
void fann_error(struct fann *ann, const unsigned int errno, ...)
{
	va_list ap;
	char * errstr;

	if(ann != NULL)	ann->errno_f = errno;
	
	if(ann != NULL && ann->errstr != NULL){
		errstr = ann->errstr;
	}else{
		errstr = (char *)malloc(FANN_ERRSTR_MAX);
		if(errstr == NULL){
			fprintf(stderr, "Unable to allocate memory.\n");
			return;
		}
	}

	va_start(ap, errno);
	switch ( errno ) {
	case FANN_E_NO_ERROR:
		break;
	case FANN_E_CANT_OPEN_CONFIG_R:
		vsnprintf(errstr, FANN_ERRSTR_MAX, "Unable to open configuration file \"%s\" for reading.\n", ap);
		break;
	case FANN_E_CANT_OPEN_CONFIG_W:
		vsnprintf(errstr, FANN_ERRSTR_MAX, "Unable to open configuration file \"%s\" for writing.\n", ap);
		break;
	case FANN_E_WRONG_CONFIG_VERSION:
		vsnprintf(errstr, FANN_ERRSTR_MAX, "Wrong version of configuration file, aborting read of configuration file \"%s\".\n", ap);
		break;
	case FANN_E_CANT_READ_CONFIG:
		vsnprintf(errstr, FANN_ERRSTR_MAX, "Error reading info from configuration file \"%s\".\n", ap);
		break;
	case FANN_E_CANT_READ_NEURON:
		vsnprintf(errstr, FANN_ERRSTR_MAX, "Error reading neuron info from configuration file \"%s\".\n", ap);
		break;
	case FANN_E_CANT_READ_CONNECTIONS:
		vsnprintf(errstr, FANN_ERRSTR_MAX, "Error reading connections from configuration file \"%s\".\n", ap);
		break;
	case FANN_E_WRONG_NUM_CONNECTIONS:
		vsnprintf(errstr, FANN_ERRSTR_MAX, "ERROR connections_so_far=%d, total_connections=%d\n", ap);
		break;
	case FANN_E_CANT_OPEN_TD_W:
		vsnprintf(errstr, FANN_ERRSTR_MAX, "Unable to open train data file \"%s\" for writing.\n", ap);
		break;
	case FANN_E_CANT_OPEN_TD_R:
		vsnprintf(errstr, FANN_ERRSTR_MAX, "Unable to open train data file \"%s\" for writing.\n", ap);
		break;
	case FANN_E_CANT_READ_TD:
		vsnprintf(errstr, FANN_ERRSTR_MAX, "Error reading info from train data file \"%s\", line: %d.\n", ap);
		break;
	case FANN_E_CANT_ALLOCATE_MEM:
		snprintf(errstr, FANN_ERRSTR_MAX, "Unable to allocate memory.\n");
		break;
	case FANN_E_CANT_TRAIN_ACTIVATION:
		snprintf(errstr, FANN_ERRSTR_MAX, "Unable to train with the selected activation function.\n");
		break;
	case FANN_E_CANT_USE_ACTIVATION:
		snprintf(errstr, FANN_ERRSTR_MAX, "Unable to use the selected activation function.\n");
		break;
	case FANN_E_TRAIN_DATA_MISMATCH:
		snprintf(errstr, FANN_ERRSTR_MAX, "Training data must be of equivalent structure.");
		break;
	default:
		vsnprintf(errstr, FANN_ERRSTR_MAX, "Unknown error.\n", ap);
		break;
	}
	va_end(ap);

	if ( ann == NULL ) {
		fprintf(stderr, "FANN Error %d: %s", errno, errstr);
	} else {
		ann->errstr = errstr;
		if ( ann->error_log != NULL ) {
			fprintf(ann->error_log, "FANN Error %d: %s", errno, errstr);
		}
	}
}

/* Create a network from a configuration file descriptor.
 */
struct fann * fann_create_from_fd(FILE *conf, const char *configuration_file)
{
	unsigned int num_layers, layer_size, activation_function_hidden, activation_function_output, input_neuron, i;
#ifdef FIXEDFANN
	unsigned int decimal_point, multiplier;
#endif
	fann_type activation_hidden_steepness, activation_output_steepness;
	float learning_rate, connection_rate;
	struct fann_neuron *first_neuron, *neuron_it, *last_neuron, **connected_neurons;
	fann_type *weights;
	struct fann_layer *layer_it;
	struct fann *ann;
	
	char *read_version;
	
	read_version = (char *)calloc(strlen(FANN_CONF_VERSION"\n"), 1);
	if(read_version == NULL){
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		return NULL;
	}
	
	fread(read_version, 1, strlen(FANN_CONF_VERSION"\n"), conf); /* reads version */
	
	/* compares the version information */
	if(strncmp(read_version, FANN_CONF_VERSION"\n", strlen(FANN_CONF_VERSION"\n")) != 0){
		fann_error(NULL, FANN_E_WRONG_CONFIG_VERSION, configuration_file);
		free(read_version);
		return NULL;
	}

	free(read_version);
	
#ifdef FIXEDFANN
	if(fscanf(conf, "%u\n", &decimal_point) != 1){
		fann_error(NULL, FANN_E_CANT_READ_CONFIG, configuration_file);
		return NULL;
	}
	multiplier = 1 << decimal_point;
#endif
	
	if(fscanf(conf, "%u %f %f %u %u "FANNSCANF" "FANNSCANF"\n", &num_layers, &learning_rate, &connection_rate, &activation_function_hidden, &activation_function_output, &activation_hidden_steepness, &activation_output_steepness) != 7){
		fann_error(NULL, FANN_E_CANT_READ_CONFIG, configuration_file);
		return NULL;
	}
	
	ann = fann_allocate_structure(learning_rate, num_layers);
	if(ann == NULL){
		return NULL;
	}
	ann->connection_rate = connection_rate;

#ifdef FIXEDFANN
	ann->decimal_point = decimal_point;
	ann->multiplier = multiplier;
#endif

	ann->activation_hidden_steepness = activation_hidden_steepness;
	ann->activation_output_steepness = activation_output_steepness;
	ann->activation_function_hidden = activation_function_hidden;
	ann->activation_function_output = activation_function_output;
	fann_update_stepwise_hidden(ann);
	fann_update_stepwise_output(ann);
	
#ifdef DEBUG
	printf("creating network with learning rate %f\n", learning_rate);
	printf("input\n");
#endif
	
	/* determine how many neurons there should be in each layer */
	for(layer_it = ann->first_layer; layer_it != ann->last_layer; layer_it++){
		if(fscanf(conf, "%u ", &layer_size) != 1){
			fann_error(ann, FANN_E_CANT_READ_NEURON, configuration_file);
			fann_destroy(ann);
			return NULL;
		}
		/* we do not allocate room here, but we make sure that
		   last_neuron - first_neuron is the number of neurons */
		layer_it->first_neuron = NULL;
		layer_it->last_neuron = layer_it->first_neuron + layer_size;
		ann->total_neurons += layer_size;
#ifdef DEBUG
		printf("  layer       : %d neurons, 1 bias\n", layer_size);
#endif
	}
	
	ann->num_input = ann->first_layer->last_neuron - ann->first_layer->first_neuron - 1;
	ann->num_output = ((ann->last_layer-1)->last_neuron - (ann->last_layer-1)->first_neuron) - 1;
	
	/* allocate room for the actual neurons */
	fann_allocate_neurons(ann);
	if(ann->errno_f == FANN_E_CANT_ALLOCATE_MEM){
		fann_destroy(ann);
		return NULL;
	}
	
	last_neuron = (ann->last_layer-1)->last_neuron;
	for(neuron_it = ann->first_layer->first_neuron;
		neuron_it != last_neuron; neuron_it++){
		if(fscanf(conf, "%u ", &neuron_it->num_connections) != 1){
			fann_error(ann, FANN_E_CANT_READ_NEURON, configuration_file);
			fann_destroy(ann);
			return NULL;
		}
		ann->total_connections += neuron_it->num_connections;
	}
	
	fann_allocate_connections(ann);
	if(ann->errno_f == FANN_E_CANT_ALLOCATE_MEM){
		fann_destroy(ann);
		return NULL;
	}
	
	connected_neurons = (ann->first_layer+1)->first_neuron->connected_neurons;
	weights = (ann->first_layer+1)->first_neuron->weights;
	first_neuron = ann->first_layer->first_neuron;
	
	for(i = 0; i < ann->total_connections; i++){
		if(fscanf(conf, "(%u "FANNSCANF") ", &input_neuron, &weights[i]) != 2){
			fann_error(ann, FANN_E_CANT_READ_CONNECTIONS, configuration_file);
			fann_destroy(ann);
			return NULL;
		}
		connected_neurons[i] = first_neuron+input_neuron;
	}	
	
#ifdef DEBUG
	printf("output\n");
#endif
	return ann;
}

/* Reads training data from a file descriptor.
 */
struct fann_train_data* fann_read_train_from_fd(FILE *file, char *filename)
{
	unsigned int num_input, num_output, num_data, i, j;
	unsigned int line = 1;
	struct fann_train_data* data = (struct fann_train_data *)malloc(sizeof(struct fann_train_data));

	if(data == NULL){
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		return NULL;
	}
	
	if(fscanf(file, "%u %u %u\n", &num_data, &num_input, &num_output) != 3){
		fann_error(NULL, FANN_E_CANT_READ_TD, filename, line);
		fann_destroy_train(data);
		return NULL;
	}
	line++;
	
	data->num_data = num_data;
	data->num_input = num_input;
	data->num_output = num_output;
	data->input = (fann_type **)calloc(num_data, sizeof(fann_type *));
	if(data->input == NULL){
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		fann_destroy_train(data);
		return NULL;
	}
	
	data->output = (fann_type **)calloc(num_data, sizeof(fann_type *));
	if(data->output == NULL){
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		fann_destroy_train(data);
		return NULL;
	}
	
	for(i = 0; i != num_data; i++){
		data->input[i] = (fann_type *)calloc(num_input, sizeof(fann_type));
		if(data->input[i] == NULL){
			fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
			fann_destroy_train(data);
			return NULL;
		}
		
		for(j = 0; j != num_input; j++){
			if(fscanf(file, FANNSCANF" ", &data->input[i][j]) != 1){
				fann_error(NULL, FANN_E_CANT_READ_TD, filename, line);
				fann_destroy_train(data);
				return NULL;
			}
		}
		line++;
		
		data->output[i] = (fann_type *)calloc(num_output, sizeof(fann_type));
		if(data->output[i] == NULL){
			fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
			fann_destroy_train(data);
			return NULL;
		}

		for(j = 0; j != num_output; j++){
			if(fscanf(file, FANNSCANF" ", &data->output[i][j]) != 1){
				fann_error(NULL, FANN_E_CANT_READ_TD, filename, line);
				fann_destroy_train(data);
				return NULL;
			}
		}
		line++;
	}
	return data;
}
