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

	/* Train outputs one last time but without any desired error */
	total_epochs += fann_train_outputs(ann, data, 0.0, max_out_epochs);

	if(neurons_between_reports && callback == NULL){
		printf("Train outputs       Current error: %.6f. Epochs %6d\n", fann_get_MSE(ann), total_epochs);
	}

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
#ifdef CASCADE_DEBUG	
			printf("Error %f < %f\n", error, desired_error);
#endif
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
	fann_update_weights_quickprop(ann, data->num_data, (ann->last_layer-1)->first_neuron->first_con, ann->total_connections);

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
	unsigned int candidate_connections_in = ann->total_neurons - ann->num_output;
	unsigned int candidate_connections_out = ann->num_output;
	/* the number of connections going into a and out of a candidate is
	   ann->total_neurons */
	unsigned int num_connections = ann->total_connections + (ann->total_neurons * (ann->cascade_num_candidates + 1));
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
		connection_it += candidate_connections_in;
		neurons[i].last_con = connection_it;
		/* We have no specific pointers to the output weights, but they are
		   available after last_con */
		connection_it += candidate_connections_out;
		ann->train_errors[i] = 0;
	}

	/* Now randomize the weights and zero out the arrays that needs zeroing out.
	 */
#ifdef CASCADE_DEBUG	
	printf("random cand weight [%d ... %d]\n", first_candidate_connection, num_connections-1);
#endif
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

	for(i = 0; i < max_epochs; i++){
		best_cand_score = fann_train_candidates_epoch(ann, data);

		if ((best_cand_score > target_cand_score) ||
			(best_cand_score < backslide_cand_score))
		{
#ifdef CASCADE_DEBUG	
			printf("best_cand_score=%f, target_cand_score=%f, backslide_cand_score=%f, stagnation=%d\n", best_cand_score, target_cand_score, backslide_cand_score, stagnation);
#endif

			target_cand_score = best_cand_score * (ann->cascade_change_fraction + 1.0);
			backslide_cand_score = best_cand_score * (1.0 - ann->cascade_change_fraction);
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
	struct fann_neuron *neurons = ann->first_layer->first_neuron;
	struct fann_neuron *first_cand = neurons + ann->total_neurons + 1;
	struct fann_neuron *last_cand = first_cand + ann->cascade_num_candidates;
	struct fann_neuron *cand_it;	
	unsigned int i, j, num_connections;
	unsigned int num_output = ann->num_output;
	fann_type cand_value, activation, derived, error_value, diff, cand_score;
	fann_type *weights, *out_weights, *cand_slopes;
	fann_type *output_train_errors = ann->train_errors + (ann->total_neurons - ann->num_output);

	for(cand_it = first_cand; cand_it < last_cand; cand_it++){
		cand_score = 0.0;
		error_value = 0.0;

		/* code more or less stolen from fann_run to fast forward pass
		 */
		cand_value = 0.0;
		num_connections = cand_it->last_con - cand_it->first_con;
		weights = ann->weights + cand_it->first_con;

		/* unrolled loop start */
		i = num_connections & 3; /* same as modulo 4 */
		switch(i) {
			case 3:
				cand_value += weights[2] * neurons[2].value;
			case 2:
				cand_value += weights[1] * neurons[1].value;
			case 1:
				cand_value += weights[0] * neurons[0].value;
			case 0:
				break;
		}
				
		for(;i != num_connections; i += 4){
			cand_value +=
				weights[i] * neurons[i].value +
				weights[i+1] * neurons[i+1].value +
				weights[i+2] * neurons[i+2].value +
				weights[i+3] * neurons[i+3].value;
		}
		/* unrolled loop end */

		activation = fann_activation(ann, 0, cand_value);
		derived = fann_activation_derived(ann->activation_function_hidden,
			ann->activation_steepness_hidden, activation);

		/* The output weights is located right after the input weights in
		   the weight array.
		*/
		out_weights = weights + num_connections;
		
		for(j = 0; j < num_output; j++){
			diff = (activation * out_weights[j]) - output_train_errors[j];
			/*printf("%f = (%f * %f) - %f;\n", diff, activation, out_weights[j], output_train_errors[j]);*/
			cand_score += (diff * diff);
			error_value += diff * out_weights[j];
		}

		ann->cascade_candidate_scores[cand_it - first_cand] = cand_score;
		error_value *= derived;

		cand_slopes = ann->train_slopes + cand_it->first_con;
		for(i = 0; i < num_connections; i++){
			cand_slopes[i] += error_value * neurons[i].value;
		}
	}
}

void fann_update_candidate_weights(struct fann *ann, unsigned int num_data)
{
	struct fann_neuron *first_cand = (ann->last_layer-1)->last_neuron + 1; /* there is an empty neuron between the actual neurons and the candidate neuron */
	struct fann_neuron *last_cand = first_cand + ann->cascade_num_candidates-1;

	fann_update_weights_quickprop(ann, num_data, first_cand->first_con, last_cand->last_con+ann->num_output);
}

float fann_train_candidates_epoch(struct fann *ann, struct fann_train_data *data)
{
	unsigned int i;
	fann_type best_score = ann->cascade_candidate_scores[0];
	unsigned int best_candidate = 0;
	unsigned int num_cand = ann->cascade_num_candidates;
	float MSE = fann_get_MSE(ann);

	for(i = 0; i < num_cand; i++){
		ann->cascade_candidate_scores[i] = (fann_type)MSE;
	}
	
	fann_reset_MSE(ann);
	
	for(i = 0; i < data->num_data; i++){
		fann_run(ann, data->input[i]);
		fann_compute_MSE(ann, data->output[i]);
		fann_update_candidate_slopes(ann);
	}

	fann_update_candidate_weights(ann, data->num_data);

	/* find the best candidate score */
	for(i = 1; i < num_cand; i++){
		if(ann->cascade_candidate_scores[i] > best_score){
			best_candidate = i;
			best_score = ann->cascade_candidate_scores[i];
		}
	}

	ann->cascade_best_candidate = ann->total_neurons + best_candidate + 1;
	/*printf("Best candidate: %d(%d) with score %f\n", ann->cascade_best_candidate, best_candidate, best_score);*/
	
	return best_score;

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

void fann_add_candidate_neuron(struct fann *ann, struct fann_layer *layer)
{
	unsigned int num_connections_in = layer->first_neuron - ann->first_layer->first_neuron;
	unsigned int num_connections_out = (ann->last_layer-1)->last_neuron - (layer+1)->first_neuron;
	unsigned int num_connections_move = num_connections_out + num_connections_in;

	int i, candidate_con, candidate_output_weight;

	struct fann_layer *layer_it;
	struct fann_neuron *neuron_it, *neuron_place, *candidate;

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

#ifdef CASCADE_DEBUG	
	printf("num_connections_in=%d, num_connections_out=%d\n", num_connections_in, num_connections_out);
#endif

	candidate = ann->first_layer->first_neuron + ann->cascade_best_candidate;
	
	/* the output weights for the candidates are located after the input weights */
	candidate_output_weight = candidate->last_con;
	
	/* move the actual neurons and the indexes to the connection arrays */
	for(neuron_it = (ann->last_layer-1)->last_neuron-1;
		neuron_it != neuron_place; neuron_it--){
#ifdef CASCADE_DEBUG	
		printf("move neuron %d -> %d\n", neuron_it - ann->first_layer->first_neuron -1,
			neuron_it - ann->first_layer->first_neuron);
#endif
		*neuron_it = *(neuron_it-1);

		/* move the weights */
#ifdef CASCADE_DEBUG	
		printf("move weight[%d ... %d] -> weight[%d ... %d]\n", neuron_it->first_con, neuron_it->last_con-1, neuron_it->first_con + num_connections_move - 1, neuron_it->last_con + num_connections_move - 2);
#endif
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
#ifdef CASCADE_DEBUG	
		printf("cadidate output weight set to weight[%d] = weight[%d] = %f\n", neuron_it->last_con-1, candidate_output_weight, 0.0 - ann->weights[candidate_output_weight]);
#endif

		ann->weights[neuron_it->last_con-1] = 0.0 - ann->weights[candidate_output_weight];
		candidate_output_weight++;
	}

	/* Now inititalize the actual neuron */
	neuron_place->value = 0;
	neuron_place->last_con = (neuron_place+1)->first_con;
	neuron_place->first_con = neuron_place->last_con - num_connections_in;
#ifdef CASCADE_DEBUG	
	printf("neuron[%d] = weights[%d ... %d]\n", neuron_place - ann->first_layer->first_neuron, neuron_place->first_con, neuron_place->last_con-1);
#endif

	candidate_con = candidate->first_con;
	/* initialize the input weights at random */
#ifdef CASCADE_DEBUG	
	printf("move cand weights[%d ... %d] -> [%d ... %d]\n", candidate_con, candidate_con + num_connections_in-1, neuron_place->first_con, neuron_place->last_con-1);
#endif
	
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
	/* TODO the candidate should be installed correctly,
	   with input and output weights.
	*/
	
	struct fann_layer *layer;
	layer = fann_add_layer(ann, ann->last_layer-1);
	fann_add_candidate_neuron(ann, layer);
	return;
}



#endif /* FIXEDFANN */
