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

/* In this file I do not need to include floatfann or doublefann,
   because it is included in the makefile. Normaly you would need
   to do a #include "floatfann.h".
*/

int main()
{
	fann_type *calc_out;
	const float connection_rate = 1;
	const float learning_rate = 0.7;
	const unsigned int num_input = 2;
	const unsigned int num_output = 1;
	const unsigned int num_layers = 3;
	const unsigned int num_neurons_hidden = 4;
	const float desired_error = 0.0001;
	const unsigned int max_iterations = 500000;
	const unsigned int iterations_between_reports = 1000;
	struct fann *ann;
	struct fann_train_data *data;
	
	unsigned int i = 0;
	unsigned int decimal_point;

	printf("Creating network.\n");

	ann = fann_create(connection_rate, learning_rate, num_layers,
		num_input,
		num_neurons_hidden,
		num_output);

	printf("Training network.\n");

	data = fann_read_train_from_file("test/xor.data");
	fann_train_on_data(ann, data, max_iterations, iterations_between_reports, desired_error);

	printf("Testing network.\n");

	for(i = 0; i < data->num_data; i++){
		calc_out = fann_run(ann, data->input[i]);
		printf("XOR test (%f,%f) -> %f, should be %f, difference=%f\n",
		data->input[i][0], data->input[i][1], *calc_out, data->output[i][0], fann_abs(*calc_out - data->output[i][0]));
	}
	
	printf("Saving network.\n");

	fann_save(ann, "xor_float.net");

	decimal_point = fann_save_to_fixed(ann, "test/xor_fixed.net");
	fann_save_train_to_fixed(data, "test/xor_fixed.data", decimal_point);
	
	printf("Cleaning up.\n");
	fann_destroy_train(data);
	fann_destroy(ann);
	
	return 0;
}
