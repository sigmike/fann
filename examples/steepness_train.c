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
#include <stdio.h>

void train_on_steepness_file(struct fann *ann, char *filename,
	unsigned int max_epochs, unsigned int epochs_between_reports,
	float desired_error, float steepness_start,
	float steepness_step, float steepness_end)
{
	float error;
	unsigned int i, j;

	struct fann_train_data *data = fann_read_train_from_file(filename);
	if(epochs_between_reports){
		printf("Max epochs %8d. Desired error: %.10f\n",
			max_epochs, desired_error);
	}

	fann_set_activation_hidden_steepness(ann, steepness_start);
	fann_set_activation_output_steepness(ann, steepness_start);
	for(i = 1; i <= max_epochs; i++){
		/* train */
		fann_reset_MSE(ann);

		for(j = 0; j != data->num_data; j++){
			fann_train(ann, data->input[j], data->output[j]);
		}

		error = fann_get_MSE(ann);

		/* print current output */
		if(epochs_between_reports &&
			(i % epochs_between_reports == 0
				|| i == max_epochs
				|| i == 1
				|| error < desired_error)){
			printf("Epochs     %8d. Current error: %.10f\n", i, error);
		}

		if(error < desired_error){
			steepness_start += steepness_step;
			if(steepness_start <= steepness_end){
				printf("Steepness: %f\n", steepness_start);
				fann_set_activation_hidden_steepness(ann, steepness_start);
				fann_set_activation_output_steepness(ann, steepness_start);
			}else{
				break;
			}
		}
	}
	fann_destroy_train(data);
}

int main()
{
	const float connection_rate = 1;
	const float learning_rate = 0.7;
	const unsigned int num_input = 2;
	const unsigned int num_output = 1;
	const unsigned int num_layers = 3;
	const unsigned int num_neurons_hidden = 4;
	const float desired_error = 0.0001;
	const unsigned int max_iterations = 500000;
	const unsigned int iterations_between_reports = 1000;
	unsigned int i;
	fann_type *calc_out;

	struct fann_train_data *data;
	
	struct fann *ann = fann_create(connection_rate,
		learning_rate, num_layers,
		num_input, num_neurons_hidden, num_output);

	data = fann_read_train_from_file("xor.data");
	
	train_on_steepness_file(ann, "xor.data", max_iterations,
		iterations_between_reports, desired_error, 0.5, 0.1, 20.0);

	fann_set_activation_function_hidden(ann, FANN_THRESHOLD);
	fann_set_activation_function_output(ann, FANN_THRESHOLD);

	for(i = 0; i != data->num_data; i++){
		calc_out = fann_run(ann, data->input[i]);
		printf("XOR test (%f, %f) -> %f, should be %f, difference=%f\n",
		data->input[i][0], data->input[i][1], *calc_out, data->output[i][0],
			(float)fann_abs(*calc_out - data->output[i][0]));
	}
	
	
	fann_save(ann, "xor_float.net");
	
	fann_destroy(ann);
	fann_destroy_train(data);

	return 0;
}
