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
#include <time.h>
#include <math.h>

#include "config.h"
#include "fann.h"
#include "fann_errno.h"

/* create a neural network.
 */
FANN_EXTERNAL struct fann * FANN_API fann_create(float connection_rate, float learning_rate,
	unsigned int num_layers, /* the number of layers, including the input and output layer */


	...) /* the number of neurons in each of the layers, starting with the input layer and ending with the output layer */
{
	struct fann *ann;
	va_list layer_sizes;
	unsigned int *layers = (unsigned int *)calloc(num_layers, sizeof(unsigned int));
	if(layers == NULL){
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		return NULL;
	}
	int i = 0;

	va_start(layer_sizes, num_layers);
	for ( i=0 ; i<(int)num_layers ; i++ ) {
		layers[i] = va_arg(layer_sizes, unsigned int);
	}
	va_end(layer_sizes);

	ann = fann_create_array(connection_rate, learning_rate, num_layers, layers);

	free(layers);

	return ann;
}

/* create a neural network.
 */
FANN_EXTERNAL struct fann * FANN_API fann_create_array(float connection_rate, float learning_rate, unsigned int num_layers, unsigned int * layers)
{
	struct fann_layer *layer_it, *last_layer, *prev_layer;
	struct fann *ann;
	struct fann_neuron *neuron_it, *first_neuron, *last_neuron, *random_neuron, *bias_neuron;
	unsigned int prev_layer_size, i, j;
	unsigned int num_neurons_in, num_neurons_out;
	unsigned int min_connections, max_connections, num_connections;
	unsigned int connections_per_neuron, allocated_connections;
	unsigned int random_number, found_connection;
	
#ifdef FIXEDFANN
	unsigned int decimal_point;
	unsigned int multiplier;
#endif
	if(connection_rate > 1){
		connection_rate = 1;
	}
	
	/* seed random */
	fann_seed_rand();
	
	/* allocate the general structure */
	ann = fann_allocate_structure(learning_rate, num_layers);
	if(ann == NULL){
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		return NULL;
	}

	ann->connection_rate = connection_rate;
#ifdef FIXEDFANN
	decimal_point = ann->decimal_point;
	multiplier = ann->multiplier;
#endif
	fann_update_stepwise_hidden(ann);
	fann_update_stepwise_output(ann);

	/* determine how many neurons there should be in each layer */
	i = 0;
	for(layer_it = ann->first_layer; layer_it != ann->last_layer; layer_it++){
		/* we do not allocate room here, but we make sure that
		   last_neuron - first_neuron is the number of neurons */
		layer_it->first_neuron = NULL;
		layer_it->last_neuron = layer_it->first_neuron + layers[i++] +1; /* +1 for bias */
		ann->total_neurons += layer_it->last_neuron - layer_it->first_neuron;
	}
	
	ann->num_output = (ann->last_layer-1)->last_neuron - (ann->last_layer-1)->first_neuron -1;
	ann->num_input = ann->first_layer->last_neuron - ann->first_layer->first_neuron -1;
	
	/* allocate room for the actual neurons */
	fann_allocate_neurons(ann);
	if(ann->errno_f == FANN_E_CANT_ALLOCATE_MEM){
		fann_destroy(ann);
		return NULL;
	}
	
#ifdef DEBUG
	printf("creating network with learning rate %f and connection rate %f\n", learning_rate, connection_rate);
	printf("input\n");
	printf("  layer       : %d neurons, 1 bias\n", ann->first_layer->last_neuron - ann->first_layer->first_neuron - 1);
#endif
	
	num_neurons_in = ann->num_input;
	for(layer_it = ann->first_layer+1; layer_it != ann->last_layer; layer_it++){
		num_neurons_out = layer_it->last_neuron - layer_it->first_neuron - 1;
		/* if all neurons in each layer should be connected to at least one neuron
		  in the previous layer, and one neuron in the next layer.
		  and the bias node should be connected to the all neurons in the next layer.
		  Then this is the minimum amount of neurons */
		min_connections = fann_max(num_neurons_in, num_neurons_out) + num_neurons_out;
		max_connections = num_neurons_in * num_neurons_out; /* not calculating bias */
		num_connections = fann_max(min_connections,
			(unsigned int)(0.5+(connection_rate * max_connections)) + num_neurons_out);
				
		connections_per_neuron = num_connections/num_neurons_out;
		allocated_connections = 0;
		/* Now split out the connections on the different neurons */
		for(i = 0; i != num_neurons_out; i++){
			layer_it->first_neuron[i].first_con = ann->total_connections + allocated_connections;
			allocated_connections += connections_per_neuron;
			layer_it->first_neuron[i].last_con = ann->total_connections + allocated_connections;
			
			if(allocated_connections < (num_connections*(i+1))/num_neurons_out){
				layer_it->first_neuron[i].last_con++;
				allocated_connections++;
			}
		}
		
		/* bias neuron also gets stuff */
		layer_it->first_neuron[i].first_con = ann->total_connections + allocated_connections;
		layer_it->first_neuron[i].last_con = ann->total_connections + allocated_connections;

		ann->total_connections += num_connections;
		
		/* used in the next run of the loop */
		num_neurons_in = num_neurons_out;
	}
	
	fann_allocate_connections(ann);
	if(ann->errno_f == FANN_E_CANT_ALLOCATE_MEM){
		fann_destroy(ann);
		return NULL;
	}

	first_neuron = ann->first_layer->first_neuron;
	
	if(connection_rate >= 1){
		prev_layer_size = ann->num_input+1;
		prev_layer = ann->first_layer;
		last_layer = ann->last_layer;
		for(layer_it = ann->first_layer+1; layer_it != last_layer; layer_it++){
			last_neuron = layer_it->last_neuron-1;
			for(neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++){
				for(i = neuron_it->first_con; i != neuron_it->last_con; i++){
					ann->weights[i] = (fann_type)fann_random_weight();
					/* these connections are still initialized for fully connected networks, to allow
					   operations to work, that are not optimized for fully connected networks.
					*/
					ann->connections[i] = prev_layer->first_neuron + (i - neuron_it->first_con);
				}
			}
			prev_layer_size = layer_it->last_neuron - layer_it->first_neuron;
			prev_layer = layer_it;
#ifdef DEBUG
			printf("  layer       : %d neurons, 1 bias\n", prev_layer_size-1);
#endif
		}
	} else {
		/* make connections for a network, that are not fully connected */
		
		/* generally, what we do is first to connect all the input
		   neurons to a output neuron, respecting the number of
		   available input neurons for each output neuron. Then
		   we go through all the output neurons, and connect the
		   rest of the connections to input neurons, that they are
		   not allready connected to.
		*/
		
		/* All the connections are cleared by calloc, because we want to
		   be able to see which connections are allready connected */
		
		for(layer_it = ann->first_layer+1;
			layer_it != ann->last_layer; layer_it++){
			
			num_neurons_out = layer_it->last_neuron - layer_it->first_neuron - 1;
			num_neurons_in = (layer_it-1)->last_neuron - (layer_it-1)->first_neuron - 1;
			
			/* first connect the bias neuron */
			bias_neuron = (layer_it-1)->last_neuron-1;
			last_neuron = layer_it->last_neuron-1;
			for(neuron_it = layer_it->first_neuron;
				neuron_it != last_neuron; neuron_it++){

				ann->connections[neuron_it->first_con] = bias_neuron;
				ann->weights[neuron_it->first_con] = (fann_type)fann_random_weight();
			}
			
			/* then connect all neurons in the input layer */
			last_neuron = (layer_it-1)->last_neuron - 1;
			for(neuron_it = (layer_it-1)->first_neuron;
				neuron_it != last_neuron; neuron_it++){
				
				/* random neuron in the output layer that has space
				   for more connections */
				do {
					random_number = (int) (0.5+fann_rand(0, num_neurons_out-1));
					random_neuron = layer_it->first_neuron + random_number;
					/* checks the last space in the connections array for room */
				}while(ann->connections[random_neuron->last_con-1]);
				
				/* find an empty space in the connection array and connect */
				for(i = random_neuron->first_con; i < random_neuron->last_con; i++){
					if(ann->connections[i] == NULL){
						ann->connections[i] = neuron_it;
						ann->weights[i] = (fann_type)fann_random_weight();
						break;
					}
				}
			}
			
			/* then connect the rest of the unconnected neurons */
			last_neuron = layer_it->last_neuron - 1;
			for(neuron_it = layer_it->first_neuron;
				neuron_it != last_neuron; neuron_it++){
				/* find empty space in the connection array and connect */
				for(i = neuron_it->first_con; i < neuron_it->last_con; i++){
					/* continue if allready connected */
					if(ann->connections[i] != NULL) continue;
					
					do {
						found_connection = 0;
						random_number = (int) (0.5+fann_rand(0, num_neurons_in-1));
						random_neuron = (layer_it-1)->first_neuron + random_number;
						
						/* check to see if this connection is allready there */
						for(j = neuron_it->first_con; j < i; j++){
							if(random_neuron == ann->connections[j]){
								found_connection = 1;
								break;
							}
						}
						
					}while(found_connection);
					
					/* we have found a neuron that is not allready
					   connected to us, connect it */
					ann->connections[i] = random_neuron;
					ann->weights[i] = (fann_type)fann_random_weight();
				}
			}
			
#ifdef DEBUG
			printf("  layer       : %d neurons, 1 bias\n", num_neurons_out);
#endif
		}
		
		/* TODO it would be nice to have the randomly created
		   connections sorted for smoother memory access.
		*/
	}
	
#ifdef DEBUG
	printf("output\n");
#endif
	
	return ann;
}

 
/* create a neural network with shortcut connections.
 */
FANN_EXTERNAL struct fann * FANN_API fann_create_shortcut(float learning_rate,
	unsigned int num_layers, /* the number of layers, including the input and output layer */


	...) /* the number of neurons in each of the layers, starting with the input layer and ending with the output layer */
{
	struct fann *ann;
	va_list layer_sizes;
	unsigned int *layers = (unsigned int *)calloc(num_layers, sizeof(unsigned int));
	if(layers == NULL){
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		return NULL;
	}
	
	int i = 0;

	va_start(layer_sizes, num_layers);
	for ( i=0 ; i<(int)num_layers ; i++ ) {
		layers[i] = va_arg(layer_sizes, unsigned int);
	}
	va_end(layer_sizes);

	ann = fann_create_shortcut_array(learning_rate, num_layers, layers);

	free(layers);

	return ann;
}

/* create a neural network with shortcut connections.
 */
FANN_EXTERNAL struct fann * FANN_API fann_create_shortcut_array(float learning_rate, unsigned int num_layers, unsigned int * layers)
{
	struct fann_layer *layer_it, *layer_it2, *last_layer;
	struct fann *ann;
	struct fann_neuron *neuron_it, *neuron_it2 = 0;
	unsigned int i;
	unsigned int num_neurons_in, num_neurons_out;
	
#ifdef FIXEDFANN
	unsigned int decimal_point;
	unsigned int multiplier;
#endif
	/* seed random */
	fann_seed_rand();
	
	/* allocate the general structure */
	ann = fann_allocate_structure(learning_rate, num_layers);
	if(ann == NULL){
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		return NULL;
	}

	ann->connection_rate = 1;
	ann->shortcut_connections = 1;
#ifdef FIXEDFANN
	decimal_point = ann->decimal_point;
	multiplier = ann->multiplier;
#endif
	fann_update_stepwise_hidden(ann);
	fann_update_stepwise_output(ann);

	/* determine how many neurons there should be in each layer */
	i = 0;
	for(layer_it = ann->first_layer; layer_it != ann->last_layer; layer_it++){
		/* we do not allocate room here, but we make sure that
		   last_neuron - first_neuron is the number of neurons */
		layer_it->first_neuron = NULL;
		layer_it->last_neuron = layer_it->first_neuron + layers[i++];
		if(layer_it == ann->first_layer){
			/* there is a bias neuron in the first layer */
			layer_it->last_neuron++;
		}
		
		ann->total_neurons += layer_it->last_neuron - layer_it->first_neuron;
	}
	
	ann->num_output = (ann->last_layer-1)->last_neuron - (ann->last_layer-1)->first_neuron;
	ann->num_input = ann->first_layer->last_neuron - ann->first_layer->first_neuron -1;
	
	/* allocate room for the actual neurons */
	fann_allocate_neurons(ann);
	if(ann->errno_f == FANN_E_CANT_ALLOCATE_MEM){
		fann_destroy(ann);
		return NULL;
	}
	
#ifdef DEBUG
	printf("creating fully shortcut connected network with learning rate %f.\n", learning_rate);
	printf("input\n");
	printf("  layer       : %d neurons, 1 bias\n", ann->first_layer->last_neuron - ann->first_layer->first_neuron - 1);
#endif
	
	num_neurons_in = ann->num_input;
	last_layer = ann->last_layer;
	for(layer_it = ann->first_layer+1; layer_it != last_layer; layer_it++){
		num_neurons_out = layer_it->last_neuron - layer_it->first_neuron;
		
		/* Now split out the connections on the different neurons */
		for(i = 0; i != num_neurons_out; i++){
			layer_it->first_neuron[i].first_con = ann->total_connections;
			ann->total_connections += num_neurons_in+1;
			layer_it->first_neuron[i].last_con = ann->total_connections;
		}
		
#ifdef DEBUG
		printf("  layer       : %d neurons, 0 bias\n", num_neurons_out);
#endif
		/* used in the next run of the loop */
		num_neurons_in += num_neurons_out;
	}
	
	fann_allocate_connections(ann);
	if(ann->errno_f == FANN_E_CANT_ALLOCATE_MEM){
		fann_destroy(ann);
		return NULL;
	}

	/* Connections are created from all neurons to all neurons in later layers
	 */
	num_neurons_in = ann->num_input+1;
	for(layer_it = ann->first_layer+1; layer_it != last_layer; layer_it++){
		for(neuron_it = layer_it->first_neuron; neuron_it != layer_it->last_neuron; neuron_it++){

			i = neuron_it->first_con;
			for(layer_it2 = ann->first_layer; layer_it2 != layer_it; layer_it2++){
				for(neuron_it2 = layer_it2->first_neuron; neuron_it2 != layer_it2->last_neuron; neuron_it2++){
					
					ann->weights[i] = (fann_type)fann_random_weight();
					ann->connections[i] = neuron_it2;
					i++;
				}
			}
		}
		num_neurons_in += layer_it->last_neuron - layer_it->first_neuron;
	}

#ifdef DEBUG
	printf("output\n");
#endif
	
	return ann;
}

/* runs the network.
 */
FANN_EXTERNAL fann_type * FANN_API fann_run(struct fann *ann, fann_type *input)
{
	struct fann_neuron *neuron_it, *last_neuron, *neurons, **neuron_pointers;
	unsigned int activation_function, i, num_connections, num_input, num_output;
	fann_type neuron_value, *output;
	fann_type *weights;
	struct fann_layer *layer_it, *last_layer;
	
	
	/* store some variabels local for fast access */
#ifndef FIXEDFANN
	fann_type steepness;
	const fann_type activation_steepness_output = ann->activation_steepness_output;
	const fann_type activation_steepness_hidden = ann->activation_steepness_hidden;
#endif
	
	unsigned int activation_function_output = ann->activation_function_output;
	unsigned int activation_function_hidden = ann->activation_function_hidden;
	struct fann_neuron *first_neuron = ann->first_layer->first_neuron;
#ifdef FIXEDFANN
	int multiplier = ann->multiplier;
	unsigned int decimal_point = ann->decimal_point;
#endif
	
	/* values used for the stepwise linear sigmoid function */
	fann_type rh1 = 0, rh2 = 0, rh3 = 0, rh4 = 0, rh5 = 0, rh6 = 0;
	fann_type h1 = 0, h2 = 0, h3 = 0, h4 = 0, h5 = 0, h6 = 0;

	switch(ann->activation_function_hidden){
#ifdef FIXEDFANN
		case FANN_SIGMOID:
		case FANN_SIGMOID_SYMMETRIC:
#endif
		case FANN_SIGMOID_STEPWISE:
		case FANN_SIGMOID_SYMMETRIC_STEPWISE:			
			/* the hidden results */
			rh1 = ann->activation_results_hidden[0];
			rh2 = ann->activation_results_hidden[1];
			rh3 = ann->activation_results_hidden[2];
			rh4 = ann->activation_results_hidden[3];
			rh5 = ann->activation_results_hidden[4];
			rh6 = ann->activation_results_hidden[5];
			
			/* the hidden parameters */
			h1 = ann->activation_values_hidden[0];
			h2 = ann->activation_values_hidden[1];
			h3 = ann->activation_values_hidden[2];
			h4 = ann->activation_values_hidden[3];
			h5 = ann->activation_values_hidden[4];
			h6 = ann->activation_values_hidden[5];
			break;
		default:
			break;
	}
			
	/* first set the input */
	num_input = ann->num_input;
	for(i = 0; i != num_input; i++){
#ifdef FIXEDFANN
		if(fann_abs(input[i]) > multiplier){
			printf("Warning input number %d is out of range -%d - %d with value %d, integer overflow may occur.\n", i, multiplier, multiplier, input[i]);
		}
#endif
		first_neuron[i].value = input[i];
	}
	/* Set the bias neuron in the input layer */
#ifdef FIXEDFANN
	(ann->first_layer->last_neuron-1)->value = multiplier;
#else
	(ann->first_layer->last_neuron-1)->value = 1;
#endif
	
	last_layer = ann->last_layer;
	for(layer_it = ann->first_layer+1; layer_it != last_layer; layer_it++){
		
#ifndef FIXEDFANN
		steepness = (layer_it == last_layer-1) ? 
			activation_steepness_output : activation_steepness_hidden;
#endif
		
		activation_function = (layer_it == last_layer-1) ?
			activation_function_output : activation_function_hidden;

		if(layer_it == layer_it-1){
			switch(ann->activation_function_output){
#ifdef FIXEDFANN
				case FANN_SIGMOID:
				case FANN_SIGMOID_SYMMETRIC:
#endif
				case FANN_SIGMOID_STEPWISE:
				case FANN_SIGMOID_SYMMETRIC_STEPWISE:			
					/* the output results */
					rh1 = ann->activation_results_output[0];
					rh2 = ann->activation_results_output[1];
					rh3 = ann->activation_results_output[2];
					rh4 = ann->activation_results_output[3];
					rh5 = ann->activation_results_output[4];
					rh6 = ann->activation_results_output[5];
			
					/* the output parameters */
					h1 = ann->activation_values_output[0];
					h2 = ann->activation_values_output[1];
					h3 = ann->activation_values_output[2];
					h4 = ann->activation_values_output[3];
					h5 = ann->activation_values_output[4];
					h6 = ann->activation_values_output[5];
					break;
				default:
					break;
			}
		}
		
		last_neuron = layer_it->last_neuron;
		for(neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++){
			if(neuron_it->first_con == neuron_it->last_con){
				/* bias neurons */
				neuron_it->value = 1;
				continue;
			}
			
			neuron_value = 0;
			num_connections = neuron_it->last_con - neuron_it->first_con;
			weights = ann->weights + neuron_it->first_con;
			
			if(ann->connection_rate >= 1){
				if(ann->shortcut_connections){
					neurons = ann->first_layer->first_neuron;
				} else {
					neurons = (layer_it-1)->first_neuron;
				}
				
				i = num_connections & 3; /* same as modulo 4 */
				switch(i) {
					case 3:
						neuron_value += fann_mult(weights[2], neurons[2].value);
					case 2:
						neuron_value += fann_mult(weights[1], neurons[1].value);
					case 1:
						neuron_value += fann_mult(weights[0], neurons[0].value);
					case 0:
						break;
				}
				
				for(;i != num_connections; i += 4){
					neuron_value +=
						fann_mult(weights[i], neurons[i].value) +
						fann_mult(weights[i+1], neurons[i+1].value) +
						fann_mult(weights[i+2], neurons[i+2].value) +
						fann_mult(weights[i+3], neurons[i+3].value);
				}
			} else {
				neuron_pointers = ann->connections + neuron_it->first_con;
				
				i = num_connections & 3; /* same as modulo 4 */
				switch(i) {
					case 3:
						neuron_value += fann_mult(weights[2], neuron_pointers[2]->value);
					case 2:
						neuron_value += fann_mult(weights[1], neuron_pointers[1]->value);
					case 1:
						neuron_value += fann_mult(weights[0], neuron_pointers[0]->value);
					case 0:
						break;
				}
				
				for(;i != num_connections; i += 4){
					neuron_value +=
						fann_mult(weights[i], neuron_pointers[i]->value) +
						fann_mult(weights[i+1], neuron_pointers[i+1]->value) +
						fann_mult(weights[i+2], neuron_pointers[i+2]->value) +
						fann_mult(weights[i+3], neuron_pointers[i+3]->value);
				}
			}
			
			switch(activation_function){
#ifdef FIXEDFANN
				case FANN_SIGMOID:
				case FANN_SIGMOID_STEPWISE:
					neuron_it->value = (fann_type)fann_stepwise(h1, h2, h3, h4, h5, h6, rh1, rh2, rh3, rh4, rh5, rh6, 0, multiplier, neuron_value);
					break;
				case FANN_SIGMOID_SYMMETRIC:
				case FANN_SIGMOID_SYMMETRIC_STEPWISE:
					neuron_it->value = (fann_type)fann_stepwise(h1, h2, h3, h4, h5, h6, rh1, rh2, rh3, rh4, rh5, rh6, -multiplier, multiplier, neuron_value);
					break;
#else
				case FANN_LINEAR:
					neuron_it->value = (fann_type)fann_linear(steepness, neuron_value);
					break;
					
				case FANN_SIGMOID:
					neuron_it->value = (fann_type)fann_sigmoid(steepness, neuron_value);
					break;
					
				case FANN_SIGMOID_SYMMETRIC:
					neuron_it->value = (fann_type)fann_sigmoid_symmetric(steepness, neuron_value);
					break;
					
				case FANN_SIGMOID_STEPWISE:
					neuron_it->value = (fann_type)fann_stepwise(h1, h2, h3, h4, h5, h6, rh1, rh2, rh3, rh4, rh5, rh6, 0, 1, neuron_value);
					break;
				case FANN_SIGMOID_SYMMETRIC_STEPWISE:
					neuron_it->value = (fann_type)fann_stepwise(h1, h2, h3, h4, h5, h6, rh1, rh2, rh3, rh4, rh5, rh6, -1, 1, neuron_value);
					break;
#endif
				case FANN_THRESHOLD:
					neuron_it->value = (fann_type)((neuron_value < 0) ? 0 : 1);
					break;
				case FANN_THRESHOLD_SYMMETRIC:
					neuron_it->value = (fann_type)((neuron_value < 0) ? -1 : 1);
					break;
				default:
					fann_error((struct fann_error *)ann, FANN_E_CANT_USE_ACTIVATION);
			}
		}
	}
	
	/* set the output */
	output = ann->output;
	num_output = ann->num_output;
	neurons = (ann->last_layer-1)->first_neuron;
	for(i = 0; i != num_output; i++){
		output[i] = neurons[i].value;
	}
	return ann->output;
}

/* deallocate the network.
 */
FANN_EXTERNAL void FANN_API fann_destroy(struct fann *ann)
{
	if(ann == NULL) return;
	fann_safe_free(ann->weights);
	fann_safe_free(ann->connections);
	fann_safe_free(ann->first_layer->first_neuron);
	fann_safe_free(ann->first_layer);
	fann_safe_free(ann->output);
	fann_safe_free(ann->train_errors);
	fann_safe_free(ann->train_slopes);
	fann_safe_free(ann->prev_train_slopes);
	fann_safe_free(ann->prev_steps);
	fann_safe_free(ann->errstr);
	fann_safe_free(ann);
}

FANN_EXTERNAL void FANN_API fann_randomize_weights(struct fann *ann, fann_type min_weight, fann_type max_weight)
{
	fann_type *last_weight;
	fann_type *weights = ann->weights;
	last_weight = weights + ann->total_connections;
	for(;weights != last_weight; weights++){
		*weights = (fann_type)(fann_rand(min_weight, max_weight));
	}
}

FANN_EXTERNAL void FANN_API fann_print_connections(struct fann *ann)
{
	struct fann_layer *layer_it;
	struct fann_neuron *neuron_it;
	unsigned int i, value;
	char *neurons;
	unsigned int num_neurons = fann_get_total_neurons(ann) - fann_get_num_output(ann);
	neurons = (char *)malloc(num_neurons+1);
	if(neurons == NULL){
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		return;
	}
	neurons[num_neurons] = 0;

	printf("Layer / Neuron ");
	for(i = 0; i < num_neurons; i++){
		printf("%d", i%10);
	}
	printf("\n");
	
	for(layer_it = ann->first_layer+1; layer_it != ann->last_layer; layer_it++){
		for(neuron_it = layer_it->first_neuron;
			neuron_it != layer_it->last_neuron; neuron_it++){
			
			memset(neurons, (int)'.', num_neurons);
			for(i = neuron_it->first_con; i < neuron_it->last_con; i++){
#ifdef FIXEDFANN
				value = (unsigned int)(fann_abs(ann->weights[i]/(double)ann->multiplier)+0.5);
#else
				value = (unsigned int)(fann_abs(ann->weights[i])+0.5);
#endif
				
				if(value > 25) value = 25;
				neurons[ann->connections[i] - ann->first_layer->first_neuron] = 'a' + value;
			}
			printf("L %3d / N %4d %s\n", layer_it - ann->first_layer,
				neuron_it - ann->first_layer->first_neuron, neurons);
		}
	}

	free(neurons);
}

/* Initialize the weights using Widrow + Nguyen's algorithm.
*/
FANN_EXTERNAL void FANN_API fann_init_weights(struct fann *ann, struct fann_train_data *train_data)
{
	fann_type smallest_inp, largest_inp;
	unsigned int dat = 0, elem, num_connect, num_hidden_neurons;
	struct fann_layer *layer_it;
	struct fann_neuron *neuron_it, *last_neuron, *bias_neuron;
#ifdef FIXEDFANN
	unsigned int multiplier = ann->multiplier;
#endif
	float scale_factor;

	for ( smallest_inp = largest_inp = train_data->input[0][0] ; dat < train_data->num_data ; dat++ ) {
		for ( elem = 0 ; elem < train_data->num_input ; elem++ ) {
			if ( train_data->input[dat][elem] < smallest_inp )
				smallest_inp = train_data->input[dat][elem];
			if ( train_data->input[dat][elem] > largest_inp )
				largest_inp = train_data->input[dat][elem];
		}
	}

	num_hidden_neurons = ann->total_neurons - (ann->num_input + ann->num_output + (ann->last_layer - ann->first_layer));
	scale_factor = (float)(pow((double)(0.7f * (double)num_hidden_neurons),
				  (double)(1.0f / (double)ann->num_input)) / (double)(largest_inp - smallest_inp));

#ifdef DEBUG
	printf("Initializing weights with scale factor %f\n", scale_factor);
#endif
	bias_neuron = ann->first_layer->last_neuron-1;
	for ( layer_it = ann->first_layer+1; layer_it != ann->last_layer ; layer_it++) {
		last_neuron = layer_it->last_neuron;

		if(!ann->shortcut_connections){
			bias_neuron = (layer_it-1)->last_neuron-1;
		}

		for(neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++) {
			for ( num_connect = neuron_it->first_con; num_connect < neuron_it->last_con ; num_connect++ ) {
				if ( bias_neuron == ann->connections[num_connect] ) {
#ifdef FIXEDFANN
					ann->weights[num_connect] = (fann_type)fann_rand(-scale_factor, scale_factor * multiplier);
#else
					ann->weights[num_connect] = (fann_type)fann_rand(-scale_factor, scale_factor);
#endif
				} else {
#ifdef FIXEDFANN
					ann->weights[num_connect] = (fann_type)fann_rand(0, scale_factor * multiplier);
#else
					ann->weights[num_connect] = (fann_type)fann_rand(0, scale_factor);
#endif
				}
			}
		}
	}
}

/* INTERNAL FUNCTION
   Allocates the main structure and sets some default values.
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

	ann->errno_f = 0;
	ann->error_log = NULL;
	ann->errstr = NULL;
	ann->learning_rate = learning_rate;
	ann->total_neurons = 0;
	ann->total_connections = 0;
	ann->num_input = 0;
	ann->num_output = 0;
	ann->train_errors = NULL;
	ann->train_slopes = NULL;
	ann->prev_steps = NULL;
	ann->prev_train_slopes = NULL;
	ann->training_algorithm = FANN_TRAIN_RPROP;
	ann->num_MSE = 0;
	ann->MSE_value = 0;
	ann->shortcut_connections = 0;
	ann->train_error_function = FANN_ERRORFUNC_TANH;

	/* variables used for cascade correlation (reasonable defaults) */
	ann->cascade_change_fraction = 0.001;
	ann->cascade_stagnation_epochs = 64;
	ann->cascade_num_candidates = 8;
	ann->cascade_candidate_scores = NULL;

	/* Variables for use with with Quickprop training (reasonable defaults) */
	ann->quickprop_decay = (float)-0.0001;
	ann->quickprop_mu = 1.75;

	/* Variables for use with with RPROP training (reasonable defaults) */
	ann->rprop_increase_factor = (float)1.2;
	ann->rprop_decrease_factor = 0.5;
	ann->rprop_delta_min = 0.0;
	ann->rprop_delta_max = 50.0;

	fann_init_error_data((struct fann_error *)ann);

#ifdef FIXEDFANN
	/* these values are only boring defaults, and should really
	   never be used, since the real values are always loaded from a file. */
	ann->decimal_point = 8;
	ann->multiplier = 256;
#endif
	
	ann->activation_function_hidden = FANN_SIGMOID_STEPWISE;
	ann->activation_function_output = FANN_SIGMOID_STEPWISE;
#ifdef FIXEDFANN
	ann->activation_steepness_hidden = ann->multiplier/2;
	ann->activation_steepness_output = ann->multiplier/2;
#else
	ann->activation_steepness_hidden = 0.5;
	ann->activation_steepness_output = 0.5;
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

/* INTERNAL FUNCTION
   Allocates room for the neurons.
 */
void fann_allocate_neurons(struct fann *ann)
{
	struct fann_layer *layer_it;
	struct fann_neuron *neurons;
	unsigned int num_neurons_so_far = 0;
	unsigned int num_neurons = 0;

	/* all the neurons is allocated in one long array (calloc clears mem) */
	neurons = (struct fann_neuron *)calloc(ann->total_neurons, sizeof(struct fann_neuron));
	ann->total_neurons_allocated = ann->total_neurons;
	
	if(neurons == NULL){
		fann_error((struct fann_error *)ann, FANN_E_CANT_ALLOCATE_MEM);
		return;
	}
		
	for(layer_it = ann->first_layer; layer_it != ann->last_layer; layer_it++){
		num_neurons = layer_it->last_neuron - layer_it->first_neuron;
		layer_it->first_neuron = neurons+num_neurons_so_far;
		layer_it->last_neuron = layer_it->first_neuron+num_neurons;
		num_neurons_so_far += num_neurons;
	}

	ann->output = (fann_type *)calloc(num_neurons, sizeof(fann_type));
	if(ann->output == NULL){
		fann_error((struct fann_error *)ann, FANN_E_CANT_ALLOCATE_MEM);
		return;
	}
}

/* INTERNAL FUNCTION
   Allocate room for the connections.
 */
void fann_allocate_connections(struct fann *ann)
{
	ann->weights = (fann_type *)calloc(ann->total_connections, sizeof(fann_type));
	if(ann->weights == NULL){
		fann_error((struct fann_error *)ann, FANN_E_CANT_ALLOCATE_MEM);
		return;
	}
	ann->total_connections_allocated = ann->total_connections;
	
	/* TODO make special cases for all places where the connections
	   is used, so that it is not needed for fully connected networks.
	*/
	ann->connections = (struct fann_neuron **) calloc(ann->total_connections_allocated, sizeof(struct fann_neuron*));
	if(ann->connections == NULL){
		fann_error((struct fann_error *)ann, FANN_E_CANT_ALLOCATE_MEM);
		return;
	}
}

/* INTERNAL FUNCTION
   Seed the random function.
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
