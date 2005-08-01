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

struct fann *ann;
struct fann_train_data *train_data, *test_data;

int print_callback(unsigned int epochs, float error)
{
	printf("Epochs     %8d. Current MSE-Error: %.10f ", epochs, error);
	printf("Train error: %f, Test error: %f\n\n", fann_test_data(ann, train_data), fann_test_data(ann, test_data));
	return 0;
}

int main()
{
	const float learning_rate = (const float)1.7;
	const float desired_error = (const float)0.00001;
	unsigned int max_out_epochs = 1500;
	unsigned int max_cand_epochs = 1500;
	unsigned int max_neurons = 40;
	unsigned int neurons_between_reports = 1;
	/*int i;
	fann_type number, steepness, v1, v2;*/
	
	printf("Reading data.\n");

	/*
	*/

	/* this is in range -1 to 1 */
	/*
	*/
		
	train_data = fann_read_train_from_file("../benchmarks/datasets/parity8.train");
	test_data = fann_read_train_from_file("../benchmarks/datasets/parity8.test");

	train_data = fann_read_train_from_file("../benchmarks/datasets/parity13.test");
	test_data = fann_read_train_from_file("../benchmarks/datasets/parity13.test");

	train_data = fann_read_train_from_file("../benchmarks/datasets/mushroom.train");
	test_data = fann_read_train_from_file("../benchmarks/datasets/mushroom.train");

	train_data = fann_read_train_from_file("../benchmarks/datasets/pumadyn-32fm.train");
	test_data = fann_read_train_from_file("../benchmarks/datasets/pumadyn-32fm.test");

	train_data = fann_read_train_from_file("../benchmarks/datasets/gene.train");
	test_data = fann_read_train_from_file("../benchmarks/datasets/gene.test");
	
	train_data = fann_read_train_from_file("../benchmarks/datasets/two-spiral.train");
	test_data = fann_read_train_from_file("../benchmarks/datasets/two-spiral.test");

	train_data = fann_read_train_from_file("../benchmarks/datasets/soybean.train");
	test_data = fann_read_train_from_file("../benchmarks/datasets/soybean.test");

	train_data = fann_read_train_from_file("../benchmarks/datasets/thyroid.train");
	test_data = fann_read_train_from_file("../benchmarks/datasets/thyroid.test");

	train_data = fann_read_train_from_file("../benchmarks/datasets/robot.train");
	test_data = fann_read_train_from_file("../benchmarks/datasets/robot.test");

	train_data = fann_read_train_from_file("xor.data");
	test_data = fann_read_train_from_file("xor.data");

	train_data = fann_read_train_from_file("../benchmarks/datasets/two-spiral2.train");
	test_data = fann_read_train_from_file("../benchmarks/datasets/two-spiral2.test");

	train_data = fann_read_train_from_file("../benchmarks/datasets/building.train");
	test_data = fann_read_train_from_file("../benchmarks/datasets/building.test");

	fann_scale_train_data(train_data, -1, 1);
	fann_scale_train_data(test_data, -1, 1);
	
	printf("Creating network.\n");

	ann = fann_create_shortcut(learning_rate, 2, train_data->num_input, train_data->num_output);
	
	fann_set_training_algorithm(ann, FANN_TRAIN_BATCH);
	fann_set_training_algorithm(ann, FANN_TRAIN_QUICKPROP);
	fann_set_training_algorithm(ann, FANN_TRAIN_RPROP);
	
	fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_function_output(ann, FANN_LINEAR);
	fann_set_activation_steepness_hidden(ann, 1);
	fann_set_activation_steepness_output(ann, 1);


	fann_set_train_error_function(ann, FANN_ERRORFUNC_LINEAR);

	fann_set_rprop_increase_factor(ann, 1.2);
	fann_set_rprop_decrease_factor(ann, 0.5);
	fann_set_rprop_delta_min(ann, 0.0);
	fann_set_rprop_delta_max(ann, 50.0);

	ann->cascade_change_fraction = 0.001;
	ann->cascade_stagnation_epochs = 120;
	ann->cascade_num_candidates = 16;
	ann->cascade_weight_multiplier = 0.7;
	
	fann_print_parameters(ann);
	/*fann_print_connections(ann);*/

	printf("Training network.\n");

	fann_cascadetrain_on_data_callback(ann, train_data, desired_error, print_callback, max_out_epochs, max_cand_epochs, max_neurons, neurons_between_reports);

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


	/*
	for(i = 0; i < 6; i++){
		printf("%.20e, ", ann->activation_values_hidden[i]);
	}
	for(i = 0; i < 6; i++){
		printf("%.20e, ", ann->activation_results_hidden[i]);
	}
	printf("\n");

	for(i = 0; i < 100000; i++)
	{
		number = fann_rand(-10.0,10.0);
		steepness = fann_rand(0.0,2.0);
		fann_set_activation_steepness_hidden(ann, steepness);
		fann_set_activation_steepness_output(ann, steepness);
		v1 = fann_stepwise(
			   ann->activation_values_hidden[0],
			   ann->activation_values_hidden[1],
			   ann->activation_values_hidden[2],
			   ann->activation_values_hidden[3],
			   ann->activation_values_hidden[4],
			   ann->activation_values_hidden[5],
			   ann->activation_results_hidden[0],
			   ann->activation_results_hidden[1],
			   ann->activation_results_hidden[2],
			   ann->activation_results_hidden[3],
			   ann->activation_results_hidden[4],
			   ann->activation_results_hidden[5],
			   -1, 1, number);
		v1 = fann_activation_new(ann, ann->activation_function_hidden, ann->activation_steepness_hidden, number);
		number = number*steepness;
		v2 = fann_stepwise(-2.64665246009826660156e+00, -1.47221946716308593750e+00, -5.49306154251098632812e-01, 5.49306154251098632812e-01, 1.47221934795379638672e+00, 2.64665293693542480469e+00, 4.99999988824129104614e-03, 5.00000007450580596924e-02, 2.50000000000000000000e-01, 7.50000000000000000000e-01, 9.49999988079071044922e-01, 9.95000004768371582031e-01, 0, 1, number);
		v2 = fann_stepwise(-2.64665293693542480469e+00, -1.47221934795379638672e+00, -5.49306154251098632812e-01, 5.49306154251098632812e-01, 1.47221934795379638672e+00, 2.64665293693542480469e+00, -9.90000009536743164062e-01, -8.99999976158142089844e-01, -5.00000000000000000000e-01, 5.00000000000000000000e-01, 8.99999976158142089844e-01, 9.90000009536743164062e-01, -1, 1, number);
		if((int)floor(v1*10000.0+0.5) != (int)floor(v2*10000.0+0.5))
		{
			printf("steepness = %f, number = %f, v1 = %f, v2 = %f", steepness, number, v1, v2);
			printf(" **********************");
			printf("\n");
		}
	}
	
	exit(0);
	*/
