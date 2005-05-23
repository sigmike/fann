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
	unsigned int max_out_epochs = 100;
	unsigned int max_cand_epochs = 100;
	unsigned int max_neurons = 40;
	unsigned int neurons_between_reports = 1;
	struct fann *ann;
	struct fann_train_data *train_data, *test_data;
	
	printf("Reading data.\n");

	/*
	*/

	/* this is in range -1 to 1 */
	/*
	*/
	
	train_data = fann_read_train_from_file("../benchmarks/datasets/parity4.train");
	test_data = fann_read_train_from_file("../benchmarks/datasets/parity4.test");
	
	train_data = fann_read_train_from_file("xor.data");
	test_data = fann_read_train_from_file("xor.data");

	train_data = fann_read_train_from_file("../benchmarks/datasets/two-spiral2.train");
	test_data = fann_read_train_from_file("../benchmarks/datasets/two-spiral2.test");
	
	printf("Creating network.\n");

	ann = fann_create_shortcut(learning_rate, 2, train_data->num_input, train_data->num_output);
	
	fann_set_training_algorithm(ann, FANN_TRAIN_QUICKPROP);
	fann_set_activation_steepness_hidden(ann, 1);
	fann_set_activation_steepness_output(ann, 1);
	fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_function_output(ann, FANN_LINEAR);
	fann_set_train_error_function(ann, FANN_ERRORFUNC_LINEAR);
	
	fann_print_parameters(ann);
	/*fann_print_connections(ann);*/

	printf("Training network.\n");

	fann_cascadetrain_on_data_callback(ann, train_data, desired_error, NULL, max_out_epochs, max_cand_epochs, max_neurons, neurons_between_reports);

	/*fann_train_on_data(ann, train_data, 300, 1, desired_error);*/
	/*printf("\nTrain error: %f, Test error: %f\n\n", fann_test_data(ann, train_data), fann_test_data(ann, test_data));*/

	fann_print_connections(ann);
	/*fann_print_parameters(ann);*/

	printf("\nTrain error: %f, Test error: %f\n\n", fann_test_data(ann, train_data), fann_test_data(ann, test_data));

	printf("Saving network.\n");

	fann_save(ann, "xor_float.net");

	printf("Cleaning up.\n");
	fann_destroy_train(train_data);
	fann_destroy_train(test_data);
	fann_destroy(ann);
	
	return 0;
}
