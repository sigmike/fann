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

void fann_cascadetrain_on_data_callback(struct fann *ann, struct fann_train_data *data, float desired_error, int (*callback)(unsigned int epochs, float error), unsigned int max_out_epochs, unsigned int max_neurons, unsigned int neurons_between_reports);

int fann_train_outputs(struct fann *ann, struct fann_train_data *data, float desired_error, unsigned int max_epochs);

float fann_train_outputs_epoch(struct fann *ann, struct fann_train_data *data);

/* Train directly on the training data.
 */
void fann_cascadetrain_on_data_callback(struct fann *ann, struct fann_train_data *data, float desired_error, int (*callback)(unsigned int epochs, float error), unsigned int max_out_epochs, unsigned int max_neurons, unsigned int neurons_between_reports)
{
	float error;
	unsigned int i;
	unsigned int total_epochs = 0;
	
	if(neurons_between_reports && callback == NULL){
		printf("Max neurons %6d. Desired error: %.6f\n", max_neurons, desired_error);
	}
	
	for(i = 1; i <= max_neurons; i++){
		/* train */
		
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

		/* fann_train_candidate */
		/* fann_install_candidate */
	}
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

			target_improvement = error_improvement * (ann->change_fraction + 1);
			backslide_improvement = error_improvement * (ann->change_fraction - 1);
			stagnation = i + ann->stagnation_epochs;
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
		/* TODO this should be real quickprop training and only on the output layer */
		/*fann_train(ann, data->input[i], data->output[i]);*/

		fann_run(ann, data->input[i]);
		fann_compute_MSE(ann, data->output[i]);
		fann_backpropagate_MSE(ann);
		/*fann_update_weights(ann);*/
		fann_update_slopes_batch(ann);
	}
	fann_update_weights_quickprop(ann, data->num_data);
	/*fann_update_weights_batch(ann, data->num_data);*/

	/*fann_update_output_weights(ann);*/

	return fann_get_MSE(ann);
}

void fann_update_output_weights(struct fann *ann)
{
	printf("fann_update_output_weights not implemented\n");
}
