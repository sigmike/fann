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

/* In this file I do not need to include floatfann or doublefann,
   because it is included in the makefile. Normaly you would need
   to do a #include "floatfann.h".
*/

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

	struct fann *ann = fann_create(connection_rate, learning_rate, num_layers,
		num_input, num_neurons_hidden, num_output);
	
	fann_train_on_file(ann, "xor.data", max_iterations,
		iterations_between_reports, desired_error);
	
	fann_save(ann, "xor_float.net");
	
	fann_destroy(ann);

	return 0;
}
