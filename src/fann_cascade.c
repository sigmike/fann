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

#include "fann.h"
#include "fann_errno.h"

#ifndef FIXEDFANN

/* #define CASCADE_DEBUG */

void fann_cascadetrain_on_data_callback(struct fann *ann, struct fann_train_data *data, float desired_error, int (*callback)(unsigned int epochs, float error), unsigned int max_out_epochs, unsigned int max_cand_epochs, unsigned int max_neurons, unsigned int neurons_between_reports);

int fann_train_outputs(struct fann *ann, struct fann_train_data *data, float desired_error, unsigned int max_epochs);

float fann_train_outputs_epoch(struct fann *ann, struct fann_train_data *data);

int fann_train_candidates(struct fann *ann, struct fann_train_data *data, unsigned int max_epochs);

float fann_train_candidates_epoch(struct fann *ann, struct fann_train_data *data);

void fann_install_candidate(struct fann *ann);

int fann_initialize_candidates(struct fann *ann);

void fann_set_shortcut_connections(struct fann *ann);

/* Cascade training directly on the training data.
   The connected_neurons pointers are not valid during training,
   but they will be again after training.
 */
void fann_cascadetrain_on_data_callback(struct fann *ann, struct fann_train_data *data, float desired_error, int (*callback)(unsigned int epochs, float error), unsigned int max_out_epochs, unsigned int max_cand_epochs, unsigned int max_neurons, unsigned int neurons_between_reports)
{
	float error;
	unsigned int i;
	unsigned int total_epochs = 0;

	if(neurons_between_reports && callback == NULL){
		printf("Max neurons %6d. Desired error: %.6f\n", max_neurons, desired_error);
	}
	
	for(i = 1; i <= max_neurons; i++){
		/* train output neurons */		
		total_epochs += fann_train_outputs(ann, data, desired_error, max_out_epochs);

		error = fann_get_MSE(ann);

		/* print current error */
		if(neurons_between_reports &&
			(i % neurons_between_reports == 0
				|| i == max_neurons
				|| i == 1
				|| error < desired_error)){
			if (callback == NULL) {
				printf("Neurons     %6d. Current error: %.6f. Epochs %6d\n", i, error, total_epochs);
			} else if((*callback)(i, error) == -1){
				/* you can break the training by returning -1 */
				break;
			}
		}
		
		if(error < desired_error){
			break;
		}
		
		if(fann_initialize_candidates(ann) == -1){
			/* Unable to initialize room for candidates */
			break;
		}
		
		/* train new candidates */
		total_epochs += fann_train_candidates(ann, data, max_cand_epochs);

		/* this installs the best candidate */
		fann_install_candidate(ann);
	}

	/* Train outputs one last time */
	total_epochs += fann_train_outputs(ann, data, desired_error, max_out_epochs);

	/* Set pointers in connected_neurons
	   This is ONLY done in the end of cascade training,
	   since there is no need for them during training.
	*/
	fann_set_shortcut_connections(ann);
}

int fann_train_outputs(struct fann *ann, struct fann_train_data *data, float desired_error, unsigned int max_epochs)
{
	float error, initial_error, error_improvement;
	float target_improvement = 0.0;
	float backslide_improvement = 0.0;
	unsigned int i;
	unsigned int stagnation = max_epochs;

	fann_clear_train_arrays(ann);
	
	/* run an initial epoch to set the initital error */
	initial_error = fann_train_outputs_epoch(ann, data);

	if(initial_error < desired_error){
		return 1;
	}
	
	for(i = 1; i < max_epochs; i++){
		error = fann_train_outputs_epoch(ann, data);

		if(error < desired_error){
			printf("Error %f < %f (%f)\n", error, desired_error, fann_get_MSE(ann));
			return i+1;
		}
		
		/* Improvement since start of train */
		error_improvement = initial_error - error;
		
		/* After any significant change, set a new goal and
		   allow a new quota of epochs to reach it */  
		if ((error_improvement > target_improvement) ||
			(error_improvement < backslide_improvement))
		{
			/*printf("error_improvement=%f, target_improvement=%f, backslide_improvement=%f, stagnation=%d\n", error_improvement, target_improvement, backslide_improvement, stagnation);*/

			target_improvement = error_improvement * (ann->cascade_change_fraction + 1);
			backslide_improvement = error_improvement * (ann->cascade_change_fraction - 1);
			stagnation = i + ann->cascade_stagnation_epochs;
		}
		
		/* No improvement in allotted period, so quit */
		if (i >= stagnation)
		{
			return i+1;
		}
	}

	return max_epochs;
}

float fann_train_outputs_epoch(struct fann *ann, struct fann_train_data *data)
{	
	return fann_train_epoch_quickprop(ann, data); /* TODO remove this line */
	unsigned int i;
	fann_reset_MSE(ann);
	
	for(i = 0; i < data->num_data; i++){
		fann_run(ann, data->input[i]);
		fann_compute_MSE(ann, data->output[i]);
 		fann_update_slopes_batch(ann, ann->last_layer-1, ann->last_layer-1);
	}
	/* TODO this should actually use the algorithm selected by
	   ann->training_algorithm
	*/
	fann_update_weights_quickprop(ann, data->num_data, ann->last_layer-1, ann->last_layer-1);

	return fann_get_MSE(ann);
}

int fann_reallocate_connections(struct fann *ann, unsigned int total_connections)
{
	/* The connections are allocated, but the pointers inside are
	   first moved in the end of the cascade training session.
	*/
	ann->connections = (struct fann_neuron **)realloc(ann->connections, total_connections * sizeof(struct fann_neuron *));
	if(ann->connections == NULL){
		fann_error((struct fann_error *)ann, FANN_E_CANT_ALLOCATE_MEM);
		return -1;
	}

	ann->weights = (fann_type *)realloc(ann->weights, total_connections * sizeof(fann_type));
	if(ann->weights == NULL){
		fann_error((struct fann_error *)ann, FANN_E_CANT_ALLOCATE_MEM);
		return -1;
	}

	ann->train_slopes = (fann_type *)realloc(ann->train_slopes, total_connections * sizeof(fann_type));
	if(ann->train_slopes == NULL){
		fann_error((struct fann_error *)ann, FANN_E_CANT_ALLOCATE_MEM);
		return -1;
	}

	ann->prev_steps = (fann_type *)realloc(ann->prev_steps, total_connections * sizeof(fann_type));
	if(ann->prev_steps == NULL){
		fann_error((struct fann_error *)ann, FANN_E_CANT_ALLOCATE_MEM);
		return -1;
	}

	ann->prev_train_slopes = (fann_type *)realloc(ann->prev_train_slopes, total_connections * sizeof(fann_type));
	if(ann->prev_train_slopes == NULL){
		fann_error((struct fann_error *)ann, FANN_E_CANT_ALLOCATE_MEM);
		return -1;
	}

	ann->total_connections_allocated = total_connections;

	return 0;
}

int fann_reallocate_neurons(struct fann *ann, unsigned int total_neurons)
{
	struct fann_layer *layer_it;
	struct fann_neuron *neurons;
	unsigned int num_neurons = 0;
	unsigned int num_neurons_so_far = 0;

	neurons = (struct fann_neuron *)realloc(ann->first_layer->first_neuron, total_neurons * sizeof(struct fann_neuron));
	ann->total_neurons_allocated = total_neurons;
	
	if(neurons == NULL){
		fann_error((struct fann_error *)ann, FANN_E_CANT_ALLOCATE_MEM);
		return -1;
	}

	/* Also allocate room for more train_errors */
	ann->train_errors = realloc(ann->train_errors, total_neurons * sizeof(fann_type));
	if(ann->train_errors == NULL){
		fann_error((struct fann_error *)ann, FANN_E_CANT_ALLOCATE_MEM);
		return -1;
	}

	if(neurons != ann->first_layer->first_neuron){
		/* Then the memory has moved, also move the pointers */

#ifdef CASCADE_DEBUG	
		printf("Moving neuron pointers\n");
#endif

		/* Move pointers from layers to neurons */
		for(layer_it = ann->first_layer; layer_it != ann->last_layer; layer_it++){
			num_neurons = layer_it->last_neuron - layer_it->first_neuron;
			layer_it->first_neuron = neurons+num_neurons_so_far;
			layer_it->last_neuron = layer_it->first_neuron+num_neurons;
			num_neurons_so_far += num_neurons;
		}
	}

	return 0;
}

int fann_initialize_candidates(struct fann *ann)
{
	/* The candidates are allocated after the normal neurons and connections,
	   but there is an empty place between the real neurons and the candidate neurons,
	   so that it will be possible to make room when the chosen candidate are copied in
	   on the desired place.
	*/
	unsigned int neurons_to_allocate, connections_to_allocate;
	unsigned int num_neurons = ann->total_neurons + ann->cascade_num_candidates + 1;
	/* the number of connections going into a and out of a candidate is at maximum
	   ann->total_neurons */
	unsigned int candidate_connections = ann->total_neurons * (ann->cascade_num_candidates + 1);
	unsigned int num_connections = ann->total_connections + candidate_connections;
	unsigned int first_candidate_connection = ann->total_connections + ann->total_neurons;
	unsigned int first_candidate_neuron = ann->total_neurons + 1;
	unsigned int connection_it, i;
	struct fann_neuron *neurons;

	/* First make sure that there is enough room, and if not then allocate a
	   bit more so that we do not need to allocate more room each time.
	*/
	if(num_neurons > ann->total_neurons_allocated){
		/* Then we need to allocate more neurons
		   Allocate half as many neurons as already exist (at least ten)
		*/
		neurons_to_allocate = num_neurons + num_neurons/2;
		if(neurons_to_allocate < num_neurons + 10){
			neurons_to_allocate = num_neurons + 10;
		}

		if(fann_reallocate_neurons(ann, neurons_to_allocate) == -1){
			return -1;
		}
	}

	if(num_connections > ann->total_connections_allocated){
		/* Then we need to allocate more connections
		   Allocate half as many connections as already exist
		   (at least enough for ten neurons)
		*/
		connections_to_allocate = num_connections + num_connections/2;
		if(connections_to_allocate < num_connections + ann->total_neurons * 10){
			connections_to_allocate = num_connections + ann->total_neurons * 10;
		}

		if(fann_reallocate_connections(ann, connections_to_allocate) == -1){
			return -1;
		}
	}

	/* Set the neurons.
	 */
	connection_it = first_candidate_connection;
	neurons = ann->first_layer->first_neuron;
	for(i = first_candidate_neuron; i < num_neurons; i++){
		/* TODO candidates should actually be created both in
		   the last layer before the output layer, and in a new layer.
		*/
		neurons[i].value = 0;
		neurons[i].first_con = connection_it;
		connection_it += candidate_connections;
		neurons[i].last_con = connection_it;
		ann->train_errors[i] = 0;
	}

	/* Now randomize the weights and zero out the arrays that needs zeroing out.
	 */
	printf("random cand weight [%d ... %d]\n", first_candidate_connection, num_connections-1);
	for(i = first_candidate_connection; i < num_connections; i++){
		ann->weights[i] = fann_random_weight();
		ann->train_slopes[i] = 0;
		ann->prev_steps[i] = 0;
		ann->prev_train_slopes[i] = 0;
	}

	return 0;
}

int fann_train_candidates(struct fann *ann, struct fann_train_data *data, unsigned int max_epochs)
{
	float best_cand_score;
	float target_cand_score = 0.0;
	float backslide_cand_score = 0.0;
	unsigned int i;
	unsigned int stagnation = max_epochs;

	if(ann->cascade_candidate_scores == NULL){
		ann->cascade_candidate_scores = (fann_type *)malloc(ann->cascade_num_candidates * sizeof(fann_type));
		if(ann->cascade_candidate_scores == NULL){
			fann_error((struct fann_error *)ann, FANN_E_CANT_ALLOCATE_MEM);
			return 0;
		}
	}

	/* TODO remove, just sets to first candidate neuron and returns.
	 */
	ann->cascade_best_candidate = ann->total_neurons+1;
	return 0;

	for(i = 0; i < max_epochs; i++){
		best_cand_score = fann_train_candidates_epoch(ann, data);

		if ((best_cand_score > target_cand_score) ||
			(best_cand_score < backslide_cand_score))
		{
			/*printf("best_cand_score=%f, target_cand_score=%f, backslide_cand_score=%f, stagnation=%d\n", best_cand_score, target_cand_score, backslide_cand_score, stagnation);*/

			target_cand_score = best_cand_score * (ann->cascade_change_fraction + 1);
			backslide_cand_score = best_cand_score * (ann->cascade_change_fraction - 1);
			stagnation = i + ann->cascade_stagnation_epochs;
		}
		
		/* No improvement in allotted period, so quit */
		if (i >= stagnation)
		{
			return i+1;
		}
	}

	return max_epochs;
}

void fann_update_candidate_slopes(struct fann *ann)
{
	struct fann_neuron * neurons = ann->first_layer->first_neuron;
	struct fann_neuron * first_cand = neurons + ann->total_neurons + 1;
	struct fann_neuron * last_cand = first_cand + ann->cascade_num_candidates;
	struct fann_neuron * neuron_it;	
	unsigned int i, num_connections;
	fann_type neuron_value, activation, derived;
	fann_type *weights;
	
	for(neuron_it = first_cand; neuron_it < last_cand; neuron_it++){

		/* code more or less stolen from fann_run to fast forward pass
		 */

		neuron_value = 0.0;
		num_connections = neuron_it->last_con - neuron_it->first_con;
		weights = ann->weights + neuron_it->first_con;
				
		i = num_connections & 3; /* same as modulo 4 */
		switch(i) {
			case 3:
				neuron_value += weights[2] * neurons[2].value;
			case 2:
				neuron_value += weights[1] * neurons[1].value;
			case 1:
				neuron_value += weights[0] * neurons[0].value;
			case 0:
				break;
		}
				
		for(;i != num_connections; i += 4){
			neuron_value +=
				weights[i] * neurons[i].value +
				weights[i+1] * neurons[i+1].value +
				weights[i+2] * neurons[i+2].value +
				weights[i+3] * neurons[i+3].value;
		}
	}

	activation = fann_activation(ann, 0, neuron_value);
	derived = fann_activation_derived(ann->activation_function_hidden,
		ann->activation_steepness_hidden, activation);

	/* BIG TODO add more here do stuff for the output */

}

float fann_train_candidates_epoch(struct fann *ann, struct fann_train_data *data)
{
	/* TODO this should actually train the candidates, but first I will need to decide how the candidates should be allocated */
	
	unsigned int i;
	float MSE = fann_get_MSE(ann);

	unsigned int num_cand = ann->cascade_num_candidates;
	for(i = 0; i < num_cand; i++){
		ann->cascade_candidate_scores[i] = (fann_type)MSE;
	}
	
	fann_reset_MSE(ann);
	
	for(i = 0; i < data->num_data; i++){
		fann_run(ann, data->input[i]);
		fann_compute_MSE(ann, data->output[i]);
		fann_update_candidate_slopes(ann);
	}

	/* fann_update_candidate_weights */
	
	return fann_get_MSE(ann); /* TODO return the score of the best candidate */
}

/* add a layer ad the position pointed to by *layer */
struct fann_layer *fann_add_layer(struct fann *ann, struct fann_layer *layer)
{
	int layer_pos = layer - ann->first_layer;
	int num_layers = ann->last_layer - ann->first_layer + 1;
	int i;

	/* allocate the layer */
	struct fann_layer *layers = realloc(ann->first_layer, num_layers * sizeof(struct fann_layer));
	if(layers == NULL){
		fann_error((struct fann_error *)ann, FANN_E_CANT_ALLOCATE_MEM);
		return NULL;
	}

	/* copy layers so that the free space is at the right location */
	for(i = num_layers-1; i >= layer_pos; i--){
		layers[i] = layers[i-1];
	}

	/* the newly allocated layer is empty */
	layers[layer_pos].first_neuron = layers[layer_pos+1].first_neuron;
	layers[layer_pos].last_neuron = layers[layer_pos+1].first_neuron;

	/* Set the ann pointers correctly */
	ann->first_layer = layers;
	ann->last_layer = layers + num_layers;

#ifdef CASCADE_DEBUG	
	printf("add layer at pos %d\n", layer_pos);
#endif
	
	return layers + layer_pos;
}

void fann_set_shortcut_connections(struct fann *ann)
{
	struct fann_layer *layer_it;
	struct fann_neuron *neuron_it, **neuron_pointers, *neurons;
	unsigned int num_connections = 0, i;
	neuron_pointers = ann->connections;
	neurons = ann->first_layer->first_neuron;
	
	for(layer_it = ann->first_layer+1; layer_it != ann->last_layer; layer_it++){
		for(neuron_it = layer_it->first_neuron;
			neuron_it != layer_it->last_neuron; neuron_it++){

			neuron_pointers += num_connections;
			num_connections = neuron_it->last_con - neuron_it->first_con;

			for(i = 0; i != num_connections; i++){
				neuron_pointers[i] = neurons + i;
			}
		}
	}
}

void fann_add_shortcut_neuron(struct fann *ann, struct fann_layer *layer)
{
	unsigned int num_connections_in = layer->first_neuron - ann->first_layer->first_neuron;
	unsigned int num_connections_out = (ann->last_layer-1)->last_neuron - (layer+1)->first_neuron;
	unsigned int num_connections_move = num_connections_out + num_connections_in;
	unsigned int neurons_to_allocate = 0;
	unsigned int connections_to_allocate = 0;
	int i, candidate_con;

	struct fann_layer *layer_it;
	struct fann_neuron *neuron_it, *neuron_place;

	/* We know that there is enough room for the new neuron
	   (the candidates are in the same arrays), so move
	   the last neurons to make room for this neuron.
	*/

	/* first move the pointers to neurons in the layer structs */
	for(layer_it = ann->last_layer-1; layer_it != layer; layer_it--){
#ifdef CASCADE_DEBUG	
		printf("move neuron pointers in layer %d, first(%d -> %d), last(%d -> %d)\n",
			layer_it - ann->first_layer,
			layer_it->first_neuron - ann->first_layer->first_neuron,
			layer_it->first_neuron - ann->first_layer->first_neuron + 1,
			layer_it->last_neuron - ann->first_layer->first_neuron,
			layer_it->last_neuron - ann->first_layer->first_neuron + 1);
#endif
		layer_it->first_neuron++;
		layer_it->last_neuron++;
	}

	/* also move the last neuron in the layer that needs the neuron added */
	layer->last_neuron++;

	/* this is the place that should hold the new neuron */
	neuron_place = layer->last_neuron-1;

	printf("num_connections_in=%d, num_connections_out=%d, neurons_to_allocate=%d, connections_to_allocate=%d\n", num_connections_in, num_connections_out, neurons_to_allocate, connections_to_allocate);	

	/* move the actual neurons and the indexes to the connection arrays */
	for(neuron_it = (ann->last_layer-1)->last_neuron-1;
		neuron_it != neuron_place; neuron_it--){
#ifdef CASCADE_DEBUG	
		printf("move neuron %d -> %d\n", neuron_it - ann->first_layer->first_neuron -1,
			neuron_it - ann->first_layer->first_neuron);
#endif
		*neuron_it = *(neuron_it-1);

#ifdef CASCADE_DEBUG	
		printf("move connection first(%d -> %d), last(%d -> %d)\n", neuron_it->first_con, neuron_it->first_con + num_connections_move-1, neuron_it->last_con, neuron_it->last_con + num_connections_move);
#endif

		/* move the weights */
		printf("move weight[%d ... %d] -> weight[%d ... %d]\n", neuron_it->first_con, neuron_it->last_con-1, neuron_it->first_con + num_connections_move - 1, neuron_it->last_con + num_connections_move - 2);
		for(i = neuron_it->last_con - 1; i >= (int)neuron_it->first_con; i--){
#ifdef CASCADE_DEBUG	
			printf("move weight[%d] = weight[%d]\n", i + num_connections_move - 1, i);
#endif
			ann->weights[i + num_connections_move - 1] = ann->weights[i];
		}

		/* move the indexes to weights */
		neuron_it->last_con += num_connections_move;
		num_connections_move--;
		neuron_it->first_con += num_connections_move;

		/* set the new weight to the newly allocated neuron */
		printf("random weight[%d]\n", neuron_it->last_con-1);
#ifdef CASCADE_DEBUG	
		printf("random weight[%d]\n", neuron_it->last_con-1);
#endif
		/* TODO this should be the weights into the candidate
		   neuron, don't really know how to get this.
		*/
		ann->weights[neuron_it->last_con-1] = (fann_type)fann_random_weight();
	}

	/* Now inititalize the actual neuron */
	neuron_place->value = 0;
	neuron_place->last_con = (neuron_place+1)->first_con;
	neuron_place->first_con = neuron_place->last_con - num_connections_in;
#ifdef CASCADE_DEBUG	
	printf("neuron[%d] = (%d - %d)\n", neuron_place - ann->first_layer->first_neuron, neuron_place->first_con, neuron_place->last_con);
#endif

	candidate_con = ann->first_layer->first_neuron[ann->cascade_best_candidate].first_con;
	/* initialize the input weights at random */
	printf("move cand weights[%d ... %d] -> [%d ... %d]\n", candidate_con, candidate_con + num_connections_in-1, neuron_place->first_con, neuron_place->last_con-1);

	for(i = 0; i < num_connections_in; i++){
		ann->weights[i + neuron_place->first_con] = ann->weights[i + candidate_con];
#ifdef CASCADE_DEBUG	
		printf("move weights[%d] -> weights[%d] (%f)\n", i + candidate_con, i + neuron_place->first_con, ann->weights[i + neuron_place->first_con]);
#endif
	}

	/* Change some of main variables */
	ann->total_neurons++;
	ann->total_connections += num_connections_in + num_connections_out;

	return;
}

void fann_install_candidate(struct fann *ann)
{
	struct fann_layer *layer;
	layer = fann_add_layer(ann, ann->last_layer-1);
	fann_add_shortcut_neuron(ann, layer);
	return;
}



#endif /* FIXEDFANN */
