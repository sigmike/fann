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

#ifndef __fann_data_h__
#define __fann_data_h__

/* ----- Data structures -----
 * No data within these structures should be altered directly by the user.
 */

struct fann_neuron
{
	fann_type *weights;
	struct fann_neuron **connected_neurons;
	unsigned int num_connections;
	fann_type value;
#ifdef __GNUC__
}__attribute__((packed));
#else
};
#endif

/* A single layer in the neural network.
 */
struct fann_layer
{
	/* A pointer to the first neuron in the layer 
	 * When allocated, all the neurons in all the layers are actually
	 * in one long array, this is because we wan't to easily clear all
	 * the neurons at once.
	 */
	struct fann_neuron *first_neuron;

	/* A pointer to the neuron past the last neuron in the layer */
	/* the number of neurons is last_neuron - first_neuron */
	struct fann_neuron *last_neuron;
};

/* The fast artificial neural network(fann) structure
 */
struct fann
{
	/* the learning rate of the network */
	float learning_rate;

	/* the connection rate of the network
	 * between 0 and 1, 1 meaning fully connected
	 */
	float connection_rate;

	/* pointer to the first layer (input layer) in an array af all the layers,
	 * including the input and outputlayers 
	 */
	struct fann_layer *first_layer;

	/* pointer to the layer past the last layer in an array af all the layers,
	 * including the input and outputlayers 
	 */
	struct fann_layer *last_layer;

	/* Total number of neurons.
	 * very usefull, because the actual neurons are allocated in one long array
	 */
	unsigned int total_neurons;

	/* Number of input neurons (not calculating bias) */
	unsigned int num_input;

	/* Number of output neurons (not calculating bias) */
	unsigned int num_output;

	/* Used to contain the error deltas used during training
	 * Is allocated during first training session,
	 * which means that if we do not train, it is never allocated.
	 */
	fann_type *train_deltas;

	/* Used to choose which activation function to use
	   
	   Sometimes it can be smart, to set the activation function for the hidden neurons
	   to FANN_THRESHOLD and the activation function for the output neurons to FANN_SIGMOID,
	   in this way you get a very fast network, that is still cabable of
	   producing real valued output.
	 */
	unsigned int activation_function_hidden, activation_function_output;

	/* Parameters for the activation function */
	fann_type activation_hidden_steepness;
	fann_type activation_output_steepness;

#ifdef FIXEDFANN
	/* the decimal_point, used for shifting the fix point
	   in fixed point integer operatons.
	*/
	unsigned int decimal_point;
	
	/* the multiplier, used for multiplying the fix point
	   in fixed point integer operatons.
	   Only used in special cases, since the decimal_point is much faster.
	*/
	unsigned int multiplier;
#endif

	/* When in choosen (or in fixed point), the sigmoid function is
	   calculated as a stepwise linear function. In the
	   activation_results array, the result is saved, and in the
	   two values arrays, the values that gives the results are saved.
	 */
	fann_type activation_results[6];
	fann_type activation_hidden_values[6];
	fann_type activation_output_values[6];

	/* Total number of connections.
	 * very usefull, because the actual connections
	 * are allocated in one long array
	 */
	unsigned int total_connections;

	/* used to store outputs in */
	fann_type *output;

	/* the number of data used to calculate the error.
	 */
	unsigned int num_errors;

	/* the total error value.
	   the real mean square error is error_value/num_errors
	 */
	float error_value;

	/* The type of error that last occured. */
	unsigned int errno;

	/* A string representation of the last error. */
	char * errstr;
};

/* Structure used to store data, for use with training. */
struct fann_train_data
{
	unsigned int num_data;
	unsigned int num_input;
	unsigned int num_output;
	fann_type **input;
	fann_type **output;
};

#endif
