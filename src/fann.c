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

/* create a neural network.
 */
struct fann * fann_create_array(float connection_rate, float learning_rate, unsigned int num_layers, unsigned int * layers)
{
	struct fann_layer *layer_it, *last_layer, *prev_layer;
	struct fann *ann;
	struct fann_neuron *neuron_it, *last_neuron, *random_neuron, *bias_neuron;
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
	ann->connection_rate = connection_rate;
#ifdef FIXEDFANN
	decimal_point = ann->decimal_point;
	multiplier = ann->multiplier;
#endif
	fann_initialise_result_array(ann);
	
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
		
		ann->total_connections += num_connections;
		
		connections_per_neuron = num_connections/num_neurons_out;
		allocated_connections = 0;
		/* Now split out the connections on the different neurons */
		for(i = 0; i != num_neurons_out; i++){
			layer_it->first_neuron[i].num_connections = connections_per_neuron;
			allocated_connections += connections_per_neuron;
			
			if(allocated_connections < (num_connections*(i+1))/num_neurons_out){
				layer_it->first_neuron[i].num_connections++;
				allocated_connections++;
			}
		}
		
		/* used in the next run of the loop */
		num_neurons_in = num_neurons_out;
	}
	
	fann_allocate_connections(ann);
	
	if(connection_rate == 1){
		prev_layer_size = ann->num_input+1;
		prev_layer = ann->first_layer;
		last_layer = ann->last_layer;
		for(layer_it = ann->first_layer+1; layer_it != last_layer; layer_it++){
			last_neuron = layer_it->last_neuron-1;
			for(neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++){
				for(i = 0; i != prev_layer_size; i++){
					neuron_it->weights[i] = fann_random_weight();
					/* these connections are still initialized for fully connected networks, to allow
					   operations to work, that are not optimized for fully connected networks.
					*/
					neuron_it->connected_neurons[i] = prev_layer->first_neuron+i;
				}
				
			}
			prev_layer_size = layer_it->last_neuron - layer_it->first_neuron;
			prev_layer = layer_it;
#ifdef DEBUG
			printf("  layer       : %d neurons, 1 bias\n", prev_layer_size-1);
#endif
		}
	}else{
		/* make connections for a network, that are not fully connected */
		
		/* generally, what we do is first to connect all the input
		   neurons to a output neuron, respecting the number of
		   available input neurons for each output neuron. Then
		   we go through all the output neurons, and connect the
		   rest of the connections to input neurons, that they are
		   not allready connected to.
		*/
		
		/* first clear all the connections, because we want to
		   be able to see which connections are allready connected */
		memset((ann->first_layer+1)->first_neuron->connected_neurons, 0, ann->total_connections * sizeof(struct fann_neuron*));
		
		for(layer_it = ann->first_layer+1;
			layer_it != ann->last_layer; layer_it++){
			
			num_neurons_out = layer_it->last_neuron - layer_it->first_neuron - 1;
			num_neurons_in = (layer_it-1)->last_neuron - (layer_it-1)->first_neuron - 1;
			
			/* first connect the bias neuron */
			bias_neuron = (layer_it-1)->last_neuron-1;
			last_neuron = layer_it->last_neuron-1;
			for(neuron_it = layer_it->first_neuron;
				neuron_it != last_neuron; neuron_it++){
				
				neuron_it->connected_neurons[0] = bias_neuron;
				neuron_it->weights[0] = fann_random_weight();
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
				}while(random_neuron->connected_neurons[random_neuron->num_connections-1]);
				
				/* find an empty space in the connection array and connect */
				for(i = 0; i < random_neuron->num_connections; i++){
					if(random_neuron->connected_neurons[i] == NULL){
						random_neuron->connected_neurons[i] = neuron_it;
						random_neuron->weights[i] = fann_random_weight();
						break;
					}
				}
			}
			
			/* then connect the rest of the unconnected neurons */
			last_neuron = layer_it->last_neuron - 1;
			for(neuron_it = layer_it->first_neuron;
				neuron_it != last_neuron; neuron_it++){
				/* find empty space in the connection array and connect */
				for(i = 0; i < neuron_it->num_connections; i++){
					/* continue if allready connected */
					if(neuron_it->connected_neurons[i] != NULL) continue;
					
					do {
						found_connection = 0;
						random_number = (int) (0.5+fann_rand(0, num_neurons_in-1));
						random_neuron = (layer_it-1)->first_neuron + random_number;
						
						/* check to see if this connection is allready there */
						for(j = 0; j < i; j++){
							if(random_neuron == neuron_it->connected_neurons[j]){
								found_connection = 1;
								break;
							}
						}
						
					}while(found_connection);
					
					/* we have found a neuron that is not allready
					   connected to us, connect it */
					neuron_it->connected_neurons[i] = random_neuron;
					neuron_it->weights[i] = fann_random_weight();
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

struct fann * fann_create(float connection_rate, float learning_rate,
	unsigned int num_layers, /* the number of layers, including the input and output layer */


	...) /* the number of neurons in each of the layers, starting with the input layer and ending with the output layer */
{
	va_list layer_sizes;
	unsigned int layers[num_layers];
	int i = 0;

	va_start(layer_sizes, num_layers);
	for ( i=0 ; i<num_layers ; i++ ) {
		layers[i] = va_arg(layer_sizes, unsigned int);
	}
	va_end(layer_sizes);

	return fann_create_array(connection_rate, learning_rate, num_layers, layers);
}

/* Create a network from a configuration file.
 */
struct fann * fann_create_from_file(const char *configuration_file)
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
	FILE *conf = fopen(configuration_file, "r");
	
	if(!conf){
		printf("Unable to open configuration file \"%s\" for reading.\n", configuration_file);
		return NULL;
	}
	
	read_version = (char *)calloc(strlen(FANN_CONF_VERSION"\n"), 1);
	fread(read_version, 1, strlen(FANN_CONF_VERSION"\n"), conf); /* reads version */
	
	/* compares the version information */
	if(strncmp(read_version, FANN_CONF_VERSION"\n", strlen(FANN_CONF_VERSION"\n")) != 0){
		printf("Wrong version of configuration file, aborting read of configuration file \"%s\".\n", configuration_file);
		return NULL;
	}
	
#ifdef FIXEDFANN
	if(fscanf(conf, "%u\n", &decimal_point) != 1){
		printf("Error reading info from configuration file \"%s\".\n", configuration_file);
		return NULL;
	}
	multiplier = 1 << decimal_point;
#endif
	
	if(fscanf(conf, "%u %f %f %u %u "FANNSCANF" "FANNSCANF"\n", &num_layers, &learning_rate, &connection_rate, &activation_function_hidden, &activation_function_output, &activation_hidden_steepness, &activation_output_steepness) != 7){
		printf("Error reading info from configuration file \"%s\".\n", configuration_file);
		return NULL;
	}
	
	ann = fann_allocate_structure(learning_rate, num_layers);
	ann->connection_rate = connection_rate;
	
#ifdef FIXEDFANN
	ann->decimal_point = decimal_point;
	ann->multiplier = multiplier;
#endif
	fann_initialise_result_array(ann);
	
	fann_set_activation_hidden_steepness(ann, activation_hidden_steepness);
	fann_set_activation_output_steepness(ann, activation_output_steepness);
	fann_set_activation_function_hidden(ann, activation_function_hidden);
	fann_set_activation_function_output(ann, activation_function_output);
	
#ifdef DEBUG
	printf("creating network with learning rate %f\n", learning_rate);
	printf("input\n");
#endif
	
	/* determine how many neurons there should be in each layer */
	for(layer_it = ann->first_layer; layer_it != ann->last_layer; layer_it++){
		if(fscanf(conf, "%u ", &layer_size) != 1){
			printf("Error reading neuron info from configuration file \"%s\".\n", configuration_file);
			return ann;
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
	
	ann->num_input = ann->first_layer->last_neuron - ann->first_layer->first_neuron;
	ann->num_output = ((ann->last_layer-1)->last_neuron - (ann->last_layer-1)->first_neuron) - 1;
	
	/* allocate room for the actual neurons */
	fann_allocate_neurons(ann);
	
	last_neuron = (ann->last_layer-1)->last_neuron;
	for(neuron_it = ann->first_layer->first_neuron;
		neuron_it != last_neuron; neuron_it++){
		if(fscanf(conf, "%u ", &neuron_it->num_connections) != 1){
			printf("Error reading neuron info from configuration file \"%s\".\n", configuration_file);
			return ann;
		}
		ann->total_connections += neuron_it->num_connections;
	}
	
	fann_allocate_connections(ann);
	
	connected_neurons = (ann->first_layer+1)->first_neuron->connected_neurons;
	weights = (ann->first_layer+1)->first_neuron->weights;
	first_neuron = ann->first_layer->first_neuron;
	
	for(i = 0; i < ann->total_connections; i++){
		if(fscanf(conf, "(%u "FANNSCANF") ", &input_neuron, &weights[i]) != 2){
			printf("Error reading connections from configuration file \"%s\".\n", configuration_file);
			return ann;
		}
		connected_neurons[i] = first_neuron+input_neuron;
	}	
	
#ifdef DEBUG
	printf("output\n");
#endif
	fclose(conf);
	return ann;
}


/* deallocate the network.
 */
void fann_destroy(struct fann *ann)
{
	free((ann->first_layer+1)->first_neuron->weights);
	free((ann->first_layer+1)->first_neuron->connected_neurons);
	free(ann->first_layer->first_neuron);
	free(ann->first_layer);
	free(ann->output);
	if(ann->train_deltas != NULL) free(ann->train_deltas);
	free(ann);
}

/* Save the network.
 */
void fann_save(struct fann *ann, const char *configuration_file)
{
	fann_save_internal(ann, configuration_file, 0);
}

/* Save the network as fixed point data.
 */
int fann_save_to_fixed(struct fann *ann, const char *configuration_file)
{
	return fann_save_internal(ann, configuration_file, 1);
}

void fann_set_learning_rate(struct fann *ann, float learning_rate)
{
	ann->learning_rate = learning_rate;
}

void fann_set_activation_function_hidden(struct fann *ann, unsigned int activation_function)
{
	ann->activation_function_hidden = activation_function;
	fann_update_stepwise_hidden(ann);
}

void fann_set_activation_function_output(struct fann *ann, unsigned int activation_function)
{
	ann->activation_function_output = activation_function;
	fann_update_stepwise_output(ann);
}

void fann_set_activation_hidden_steepness(struct fann *ann, fann_type steepness)
{
	ann->activation_hidden_steepness = steepness;
	fann_update_stepwise_hidden(ann);
}

void fann_set_activation_output_steepness(struct fann *ann, fann_type steepness)
{
	ann->activation_output_steepness = steepness;
	fann_update_stepwise_output(ann);
}

float fann_get_learning_rate(struct fann *ann)
{
	return ann->learning_rate;
}

unsigned int fann_get_num_input(struct fann *ann)
{
	return ann->num_input;
}

unsigned int fann_get_num_output(struct fann *ann)
{
	return ann->num_output;
}

unsigned int fann_get_activation_function_hidden(struct fann *ann)
{
	return ann->activation_function_hidden;
}

unsigned int fann_get_activation_function_output(struct fann *ann)
{
	return ann->activation_function_output;
}

fann_type fann_get_activation_hidden_steepness(struct fann *ann)
{
	return ann->activation_hidden_steepness;
}

fann_type fann_get_activation_output_steepness(struct fann *ann)
{
	return ann->activation_output_steepness;
}

unsigned int fann_get_total_neurons(struct fann *ann)
{
	/* -1, because there is always an unused bias neuron in the last layer */
	return ann->total_neurons - 1;
}

unsigned int fann_get_total_connections(struct fann *ann)
{
	return ann->total_connections;
}

void fann_randomize_weights(struct fann *ann, fann_type min_weight, fann_type max_weight)
{
	fann_type *last_weight;
	fann_type *weights = (ann->first_layer+1)->first_neuron->weights;
	last_weight = weights + ann->total_connections;
	for(;weights != last_weight; weights++){
		*weights = (fann_type)(fann_rand(min_weight, max_weight));
	}
}

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
		/* TODO add switch the minute there are other activation functions */
		*delta_it = fann_sigmoid_derive(activation_output_steepness, neuron_value) * (*desired_output - neuron_value);
		
		ann->error_value += (*desired_output - neuron_value) * (*desired_output - neuron_value);
		
#ifdef DEBUGTRAIN
		printf("delta[%d] = "FANNPRINTF"\n", (delta_it - delta_begin), *delta_it);
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
		for(neuron_it = (layer_it-1)->first_neuron;
			neuron_it != last_neuron; neuron_it++){
			neuron_value = neuron_it->value;
			/* TODO add switch the minute there are other activation functions */
			*delta_it *= fann_sigmoid_derive(activation_hidden_steepness, neuron_value) * learning_rate;
			
#ifdef DEBUGTRAIN
			printf("delta[%d] = "FANNPRINTF"\n", delta_it - delta_begin, *delta_it);
#endif
			delta_it++;
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
					neuron_it->weights[i] += tmp_delta * neurons[i].value;
				}
			}
		}else{
			for(neuron_it = layer_it->first_neuron;
				neuron_it != last_neuron; neuron_it++){
				tmp_delta = *(delta_begin + (neuron_it - first_neuron));
				for(i = 0; i < neuron_it->num_connections; i++){
					neuron_it->weights[i] += tmp_delta * neuron_it->connected_neurons[i]->value;
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

/* Reads training data from a file.
 */
struct fann_train_data* fann_read_train_from_file(char *filename)
{
	unsigned int num_input, num_output, num_data, i, j;
	unsigned int line = 1;
	struct fann_train_data* data;
	
	FILE *file = fopen(filename, "r");
	
	data = (struct fann_train_data *)malloc(sizeof(struct fann_train_data));
	
	if(!file){
		printf("Unable to open train data file \"%s\" for reading.\n", filename);
		return NULL;
	}
	
	if(fscanf(file, "%u %u %u\n", &num_data, &num_input, &num_output) != 3){
		printf("Error reading info from train data file \"%s\", line: %d.\n", filename, line);
		return NULL;
	}
	line++;
	
	data->num_data = num_data;
	data->num_input = num_input;
	data->num_output = num_output;
	data->input = (fann_type **)calloc(num_data, sizeof(fann_type *));
	data->output = (fann_type **)calloc(num_data, sizeof(fann_type *));
	
	for(i = 0; i != num_data; i++){
		data->input[i] = (fann_type *)calloc(num_input, sizeof(fann_type));
		for(j = 0; j != num_input; j++){
			if(fscanf(file, FANNSCANF" ", &data->input[i][j]) != 1){
				printf("Error reading info from train data file \"%s\", line: %d.\n", filename, line);
				return NULL;
			}
		}
		line++;
		
		data->output[i] = (fann_type *)calloc(num_output, sizeof(fann_type));
		for(j = 0; j != num_output; j++){
			if(fscanf(file, FANNSCANF" ", &data->output[i][j]) != 1){
				printf("Error reading info from train data file \"%s\", line: %d.\n", filename, line);
				return NULL;
			}
		}
		line++;
	}
	
	return data;
}

/* Save training data to a file
 */
void fann_save_train(struct fann_train_data* data, char *filename)
{
	fann_save_train_internal(data, filename, 0, 0);
}

/* Save training data to a file in fixed point algebra.
   (Good for testing a network in fixed point)
*/
void fann_save_train_to_fixed(struct fann_train_data* data, char *filename, unsigned int decimal_point)
{
	fann_save_train_internal(data, filename, 1, decimal_point);
}

/* deallocate the train data structure.
 */
void fann_destroy_train(struct fann_train_data *data)
{
	unsigned int i;
	for(i = 0; i != data->num_data; i++){
		free(data->input[i]);
		free(data->output[i]);
	}
	free(data->input);
	free(data->output);
	free(data);
}

#ifndef FIXEDFANN

/* Train directly on the training data.
 */
void fann_train_on_data_callback(struct fann *ann, struct fann_train_data *data, unsigned int max_epochs, unsigned int epochs_between_reports, float desired_error, int (*callback)(unsigned int epochs, float error))
{
	float error;
	unsigned int i, j;
	
	if(epochs_between_reports && callback == NULL){
		printf("Max epochs %8d. Desired error: %.10f\n", max_epochs, desired_error);
	}
	
	for(i = 1; i <= max_epochs; i++){
		/* train */
		fann_reset_error(ann);
		
		for(j = 0; j != data->num_data; j++){
			fann_train(ann, data->input[j], data->output[j]);
		}
		
		error = fann_get_error(ann);
		
		/* print current output */
		if(epochs_between_reports &&
			(i % epochs_between_reports == 0
				|| i == max_epochs
				|| i == 1
				|| error < desired_error)){
			if (callback == NULL) {
				printf("Epochs     %8d. Current error: %.10f\n", i, error);
			} else if((*callback)(i, error) == -1){
				/* you can break the training by returning -1 */
				break;
			}
		}
		
		if(error < desired_error){
			break;
		}
	}
}

void fann_train_on_data(struct fann *ann, struct fann_train_data *data, unsigned int max_epochs, unsigned int epochs_between_reports, float desired_error)
{
	fann_train_on_data_callback(ann, data, max_epochs, epochs_between_reports, desired_error, NULL);
}


/* Wrapper to make it easy to train directly on a training data file.
 */
void fann_train_on_file_callback(struct fann *ann, char *filename, unsigned int max_epochs, unsigned int epochs_between_reports, float desired_error, int (*callback)(unsigned int epochs, float error))
{
	struct fann_train_data *data = fann_read_train_from_file(filename);
	fann_train_on_data_callback(ann, data, max_epochs, epochs_between_reports, desired_error, callback);
	fann_destroy_train(data);
}

void fann_train_on_file(struct fann *ann, char *filename, unsigned int max_epochs, unsigned int epochs_between_reports, float desired_error)
{
	fann_train_on_file_callback(ann, filename, max_epochs, epochs_between_reports, desired_error, NULL);
}


#endif

/* get the mean square error.
 */
float fann_get_error(struct fann *ann)
{
	if(ann->num_errors){
		return ann->error_value/(float)ann->num_errors;
	}else{
		return 0;
	}
}

/* reset the mean square error.
 */
void fann_reset_error(struct fann *ann)
{
	ann->num_errors = 0;
	ann->error_value = 0;
}

#ifdef FIXEDFANN
/* returns the position of the fix point.
 */
unsigned int fann_get_decimal_point(struct fann *ann)
{
	return ann->decimal_point;
}

/* returns the multiplier that fix point data is multiplied with.
 */
unsigned int fann_get_multiplier(struct fann *ann)
{
	return ann->multiplier;
}

#endif

/* runs the network.
 */
fann_type* fann_run(struct fann *ann, fann_type *input)
{
	struct fann_neuron *neuron_it, *last_neuron, *neurons, **neuron_pointers;
	unsigned int activation_function, i, num_connections, num_input, num_output;
	fann_type neuron_value, *weights, *output;
	struct fann_layer *layer_it, *last_layer;
	
	
	/* store some variabels local for fast access */
#ifndef FIXEDFANN
	fann_type steepness;
	const fann_type activation_output_steepness = ann->activation_output_steepness;
	const fann_type activation_hidden_steepness = ann->activation_hidden_steepness;
#endif
	
	unsigned int activation_function_output = ann->activation_function_output;
	unsigned int activation_function_hidden = ann->activation_function_hidden;
	struct fann_neuron *first_neuron = ann->first_layer->first_neuron;
#ifdef FIXEDFANN
	unsigned int multiplier = ann->multiplier;
	unsigned int decimal_point = ann->decimal_point;
#endif
	
	/* values used for the stepwise linear sigmoid function */
	fann_type r1, r2, r3, r4, r5, r6;
	fann_type h1, h2, h3, h4, h5, h6;
	fann_type o1, o2, o3, o4, o5, o6;
	
#ifdef FIXEDFANN
	if(activation_function_output == FANN_SIGMOID_STEPWISE ||
		activation_function_hidden == FANN_SIGMOID_STEPWISE ||
		activation_function_output == FANN_SIGMOID ||
		activation_function_hidden == FANN_SIGMOID){
#else
		if(activation_function_output == FANN_SIGMOID_STEPWISE ||
			activation_function_hidden == FANN_SIGMOID_STEPWISE){
#endif
			/* the results */
			r1 = ann->activation_results[0];
			r2 = ann->activation_results[1];
			r3 = ann->activation_results[2];
			r4 = ann->activation_results[3];
			r5 = ann->activation_results[4];
			r6 = ann->activation_results[5];
			
			/* the hidden parameters */
			h1 = ann->activation_hidden_values[0];
			h2 = ann->activation_hidden_values[1];
			h3 = ann->activation_hidden_values[2];
			h4 = ann->activation_hidden_values[3];
			h5 = ann->activation_hidden_values[4];
			h6 = ann->activation_hidden_values[5];
			
			/* the output parameters */
			o1 = ann->activation_output_values[0];
			o2 = ann->activation_output_values[1];
			o3 = ann->activation_output_values[2];
			o4 = ann->activation_output_values[3];
			o5 = ann->activation_output_values[4];
			o6 = ann->activation_output_values[5];
#ifdef FIXEDFANN /* just to make autoindent happy */
		}
#else
	}
#endif
	
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
	
	last_layer = ann->last_layer;
	for(layer_it = ann->first_layer+1; layer_it != last_layer; layer_it++){
		
#ifdef FIXEDFANN
		((layer_it-1)->last_neuron-1)->value = multiplier;
#else
		/* set the bias neuron */
		((layer_it-1)->last_neuron-1)->value = 1;
		
		steepness = (layer_it == last_layer-1) ? 
			activation_output_steepness : activation_hidden_steepness;
#endif
		
		activation_function = (layer_it == last_layer-1) ?
			activation_function_output : activation_function_hidden;
		
		last_neuron = layer_it->last_neuron-1;
		for(neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++){
			neuron_value = 0;
			num_connections = neuron_it->num_connections;
			weights = neuron_it->weights;
			if(ann->connection_rate == 1){
				neurons = (layer_it-1)->first_neuron;
				
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
			}else{
				neuron_pointers = neuron_it->connected_neurons;
				
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
					if(layer_it == last_layer-1){
						neuron_it->value = fann_sigmoid_stepwise(o1, o2, o3, o4, o5, o6, r1, r2, r3, r4, r5, r6, neuron_value, multiplier);
					}else{
						neuron_it->value = fann_sigmoid_stepwise(h1, h2, h3, h4, h5, h6, r1, r2, r3, r4, r5, r6, neuron_value, multiplier);
					}
					break;
#else
				case FANN_SIGMOID:
					neuron_it->value = fann_sigmoid(steepness, neuron_value);
					break;
					
				case FANN_SIGMOID_STEPWISE:
					if(layer_it == last_layer-1){
						neuron_it->value = fann_sigmoid_stepwise(o1, o2, o3, o4, o5, o6, r1, r2, r3, r4, r5, r6, neuron_value, 1);
					}else{
						neuron_it->value = fann_sigmoid_stepwise(h1, h2, h3, h4, h5, h6, r1, r2, r3, r4, r5, r6, neuron_value, 1);
					}
					break;
#endif
				case FANN_THRESHOLD:
					neuron_it->value = (neuron_value < 0) ? 0 : 1;
					break;
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
