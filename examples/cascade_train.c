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

#include "fann.h"

int main()
{
	const float learning_rate = (const float)0.7;
	const float desired_error = (const float)0.001;
	unsigned int max_out_epochs = 10000;
	unsigned int max_cand_epochs = 10000;
	unsigned int max_neurons = 50;
	unsigned int neurons_between_reports = 1;
	unsigned int i = 0;
	fann_type *calc_out;
	struct fann *ann;
	struct fann_train_data *train_data, *test_data;
	
	printf("Reading data.\n");

	train_data = fann_read_train_from_file("../benchmarks/datasets/building.train");
	test_data = fann_read_train_from_file("../benchmarks/datasets/building.test");

	printf("Creating network.\n");

	ann = fann_create_shortcut(learning_rate, 2, train_data->num_input, train_data->num_output);

	fann_set_activation_function_hidden(ann, FANN_SIGMOID);
	fann_set_activation_function_output(ann, FANN_SIGMOID);
	
	/*fann_print_connections(ann);*/
	fann_print_parameters(ann);

	printf("Training network.\n");
	
	/*fann_train_on_data(ann, train_data, 300, 1, desired_error);*/
	printf("\nTrain error: %f, Test error: %f\n\n", fann_test_data(ann, train_data), fann_test_data(ann, test_data));

	fann_cascadetrain_on_data_callback(ann, train_data, desired_error, NULL, max_out_epochs, max_cand_epochs, max_neurons, neurons_between_reports);

	printf("\nTrain error: %f, Test error: %f\n\n", fann_test_data(ann, train_data), fann_test_data(ann, test_data));

	fann_print_connections(ann);
	/*fann_print_parameters(ann);*/

	/*
	printf("\nTesting network.\n");
	
	for(i = 0; i < test_data->num_data; i++){
		calc_out = fann_run(ann, test_data->input[i]);
		printf("XOR test (%f,%f) -> %f, should be %f, difference=%f\n",
		test_data->input[i][0], test_data->input[i][1], *calc_out, test_data->output[i][0], fann_abs(*calc_out - test_data->output[i][0]));
	}
	*/
	
	printf("Saving network.\n");

	fann_save(ann, "xor_float.net");
	
	/*fann_randomize_weights(ann, -0.1, 0.1);
	  fann_train_on_data(ann, train_data, max_out_epochs, 1, desired_error);
	
	printf("\nTrain error: %f, Test error: %f\n\n", fann_test_data(ann, train_data), fann_test_data(ann, test_data));

	fann_print_connections(ann);
	fann_print_parameters(ann);*/

	printf("Cleaning up.\n");
	fann_destroy_train(train_data);
	fann_destroy_train(test_data);
	fann_destroy(ann);
	
	return 0;
}
