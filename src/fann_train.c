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

#ifndef FIXEDFANN
/* Trains the network with the backpropagation algorithm.
 */
void fann_train(struct fann *ann, fann_type *input, fann_type *desired_output)
{
	struct fann_neuron *neuron_it, *last_neuron, *neurons;
	fann_type neuron_value, *delta_it, *delta_begin, tmp_delta;
	struct fann_layer *layer_it;
	unsigned int i, shift_prev_layer;
	
	/* store some variabels local for fast access */
	const float learning_rate = ann->learning_rate;
	const fann_type activation_output_steepness = ann->activation_output_steepness;
	const fann_type activation_hidden_steepness = ann->activation_hidden_steepness;
	const struct fann_neuron *first_neuron = ann->first_layer->first_neuron;
	
	const struct fann_neuron *last_layer_begin = (ann->last_layer-1)->first_neuron;
	const struct fann_neuron *last_layer_end = last_layer_begin + ann->num_output;
	struct fann_layer *first_layer = ann->first_layer;
	struct fann_layer *last_layer = ann->last_layer;
	
	fann_run(ann, input);
	/* if no room allocated for the delta variabels, allocate it now */
	if(ann->train_deltas == NULL){
		ann->train_deltas = (fann_type *)calloc(ann->total_neurons, sizeof(fann_type));
		if(ann->train_deltas == NULL){
			fann_error((struct fann_error *)ann, FANN_E_CANT_ALLOCATE_MEM);
			return;
		}
	}
	delta_begin = ann->train_deltas;
	
	/* clear the delta variabels */
	memset(delta_begin, 0, (ann->total_neurons) * sizeof(fann_type));
	
#ifdef DEBUGTRAIN
	printf("calculate deltas\n");
#endif
	/* calculate the error and place it in the output layer */
	delta_it = delta_begin + (last_layer_begin - first_neuron);

	for(; last_layer_begin != last_layer_end; last_layer_begin++){
		neuron_value = last_layer_begin->value;
		switch(ann->activation_function_output){
			case FANN_LINEAR:
				*delta_it = (fann_type)fann_linear_derive(activation_output_steepness, neuron_value) * (*desired_output - neuron_value);
				break;
			case FANN_SIGMOID:
			case FANN_SIGMOID_STEPWISE:
				*delta_it = (fann_type)fann_sigmoid_derive(activation_output_steepness, neuron_value) * (*desired_output - neuron_value);
				break;
			case FANN_SIGMOID_SYMMETRIC:
			case FANN_SIGMOID_SYMMETRIC_STEPWISE:
				*delta_it = (fann_type)fann_sigmoid_symmetric_derive(activation_output_steepness, neuron_value) * (*desired_output - neuron_value);
				break;
			default:
				fann_error((struct fann_error *)ann, FANN_E_CANT_TRAIN_ACTIVATION);
				return;
		}
		
		ann->error_value += (*desired_output - neuron_value) * (*desired_output - neuron_value);
		
#ifdef DEBUGTRAIN
		printf("delta1[%d] = "FANNPRINTF"\n", (delta_it - delta_begin), *delta_it);
#endif
		desired_output++;
		delta_it++;
	}
	ann->num_errors++;
	
	
	/* go through all the layers, from last to first. And propagate the error backwards */
	for(layer_it = last_layer-1; layer_it != first_layer; --layer_it){
		last_neuron = layer_it->last_neuron;
		
		/* for each connection in this layer, propagate the error backwards*/
		if(ann->connection_rate == 1){ /* optimization for fully connected networks */
			shift_prev_layer = (layer_it-1)->first_neuron - first_neuron;
			for(neuron_it = layer_it->first_neuron;
				neuron_it != last_neuron; neuron_it++){
				tmp_delta = *(delta_begin + (neuron_it - first_neuron));
				for(i = 0; i < neuron_it->num_connections; i++){
					*(delta_begin + i + shift_prev_layer) += tmp_delta * neuron_it->weights[i];
#ifdef DEBUGTRAIN
					printf("delta2[%d] = "FANNPRINTF" += ("FANNPRINTF" * "FANNPRINTF")\n", (i + shift_prev_layer), *(delta_begin + i + shift_prev_layer), tmp_delta, neuron_it->weights[i]);
#endif
				}
			}
		}else{
			for(neuron_it = layer_it->first_neuron;
				neuron_it != last_neuron; neuron_it++){
				tmp_delta = *(delta_begin + (neuron_it - first_neuron));
				for(i = 0; i < neuron_it->num_connections; i++){
					*(delta_begin + (neuron_it->connected_neurons[i] - first_neuron)) +=
						tmp_delta * neuron_it->weights[i];
				}
			}
		}
		
		/* then calculate the actual errors in the previous layer */
		delta_it = delta_begin + ((layer_it-1)->first_neuron - first_neuron);
		last_neuron = (layer_it-1)->last_neuron;
		
		switch(ann->activation_function_hidden){
			case FANN_LINEAR:
				for(neuron_it = (layer_it-1)->first_neuron;
					neuron_it != last_neuron; neuron_it++){
					neuron_value = neuron_it->value;
					*delta_it *= (fann_type)fann_linear_derive(activation_hidden_steepness, neuron_value) * learning_rate;
					delta_it++;
				}
				break;
			case FANN_SIGMOID:
			case FANN_SIGMOID_STEPWISE:
				for(neuron_it = (layer_it-1)->first_neuron;
					neuron_it != last_neuron; neuron_it++){
					neuron_value = neuron_it->value;
					neuron_value = fann_clip(neuron_value, 0.01f, 0.99f);
					*delta_it *= (fann_type)fann_sigmoid_derive(activation_hidden_steepness, neuron_value);
#ifdef DEBUGTRAIN
					printf("delta3[%d] = "FANNPRINTF" *= fann_sigmoid_derive(%f, %f) * %f\n", (delta_it - delta_begin), *delta_it, activation_hidden_steepness, neuron_value, learning_rate);
#endif
					delta_it++;
				}
				break;
			case FANN_SIGMOID_SYMMETRIC:
			case FANN_SIGMOID_SYMMETRIC_STEPWISE:
				for(neuron_it = (layer_it-1)->first_neuron;
					neuron_it != last_neuron; neuron_it++){
					neuron_value = neuron_it->value;
					neuron_value = fann_clip(neuron_value, -0.98f, 0.98f);
					*delta_it *= (fann_type)fann_sigmoid_symmetric_derive(activation_hidden_steepness, neuron_value);
#ifdef DEBUGTRAIN
					printf("delta3[%d] = "FANNPRINTF" *= fann_sigmoid_symmetric_derive(%f, %f) * %f\n", (delta_it - delta_begin), *delta_it, activation_hidden_steepness, neuron_value, learning_rate);
#endif
					delta_it++;
				}
				break;
			default:
				fann_error((struct fann_error *)ann, FANN_E_CANT_TRAIN_ACTIVATION);
				return;
		}
	}
	
#ifdef DEBUGTRAIN
	printf("\nupdate weights\n");
#endif
	
	for(layer_it = (first_layer+1); layer_it != last_layer; layer_it++){
#ifdef DEBUGTRAIN
		printf("layer[%d]\n", layer_it - first_layer);
#endif
		last_neuron = layer_it->last_neuron;
		if(ann->connection_rate == 1){ /* optimization for fully connected networks */
			neurons = (layer_it-1)->first_neuron;
			for(neuron_it = layer_it->first_neuron;
				neuron_it != last_neuron; neuron_it++){
				tmp_delta = *(delta_begin + (neuron_it - first_neuron));
				for(i = 0; i < neuron_it->num_connections; i++){
#ifdef DEBUGTRAIN
					printf("weights[%d] += "FANNPRINTF" = %f * %f\n", i, tmp_delta * neurons[i].value, tmp_delta, neurons[i].value);
#endif
					neuron_it->weights[i] += learning_rate * tmp_delta * neurons[i].value;
				}
			}
		}else{
			for(neuron_it = layer_it->first_neuron;
				neuron_it != last_neuron; neuron_it++){
				tmp_delta = *(delta_begin + (neuron_it - first_neuron));
				for(i = 0; i < neuron_it->num_connections; i++){
					neuron_it->weights[i] += learning_rate * tmp_delta * neuron_it->connected_neurons[i]->value;
				}
			}
		}
	}
}
#endif

/* Tests the network.
 */
fann_type *fann_test(struct fann *ann, fann_type *input, fann_type *desired_output)
{
	fann_type neuron_value;
	fann_type *output_begin = fann_run(ann, input);
	fann_type *output_it;
	const fann_type *output_end = output_begin + ann->num_output;
	
	/* calculate the error */
	for(output_it = output_begin;
		output_it != output_end; output_it++){
		neuron_value = *output_it;
		
#ifdef FIXEDFANN
		ann->error_value += ((*desired_output - neuron_value)/(float)ann->multiplier) * ((*desired_output - neuron_value)/(float)ann->multiplier);
#else
		ann->error_value += (*desired_output - neuron_value) * (*desired_output - neuron_value);
#endif
		
		desired_output++;
	}
	ann->num_errors++;
	
	return output_begin;
}

/* get the mean square error.
   (obsolete will be removed at some point, use fann_get_MSE)
 */
float fann_get_error(struct fann *ann)
{
	return fann_get_MSE(ann);
}

/* get the mean square error.
 */
float fann_get_MSE(struct fann *ann)
{
	if(ann->num_errors){
		return ann->error_value/(float)ann->num_errors;
	}else{
		return 0;
	}
}

/* reset the mean square error.
   (obsolete will be removed at some point, use fann_reset_MSE)
 */
void fann_reset_error(struct fann *ann)
{
	fann_reset_MSE(ann);
}

/* reset the mean square error.
 */
void fann_reset_MSE(struct fann *ann)
{
	ann->num_errors = 0;
	ann->error_value = 0;
}

