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

/* This file defines the user interface to the fann library.
   It is included from fixedfann.h, floatfann.h and doublefann.h and should
   NOT be included directly.
*/

#ifndef FANN_INCLUDE
#include "floatfann.h"
#else


#include "fann_data.h"
#include "fann_internal.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */


/* ----- Initialisation and configuration ----- */

/* Constructs a backpropagation neural network, from an connection rate,
   a learning rate, the number of layers and the number of neurons in each
   of the layers.

   The connection rate controls how many connections there will be in the
   network. If the connection rate is set to 1, the network will be fully
   connected, but if it is set to 0.5 only half of the connections will be set.

   There will be a bias neuron in each layer (except the output layer),
   and this bias neuron will be connected to all neurons in the next layer.
   When running the network, the bias nodes always emits 1
 */
struct fann * fann_create(float connection_rate, float learning_rate,
	/* the number of layers, including the input and output layer */
	unsigned int num_layers,
	/* the number of neurons in each of the layers, starting with
	   the input layer and ending with the output layer */
	...);

/* Constructs a backpropagation neural network from a configuration file.
 */
struct fann * fann_create_from_file(const char *configuration_file);

/* Destructs the entire network.
   Be sure to call this function after finished using the network.
 */
void fann_destroy(struct fann *ann);

/* Save the entire network to a configuration file.
 */
void fann_save(struct fann *ann, const char *configuration_file);

/* Saves the entire network to a configuration file.
   But it is saved in fixed point format no matter which
   format it is currently in.

   This is usefull for training a network in floating points,
   and then later executing it in fixed point.

   The function returns the bit position of the fix point, which
   can be used to find out how accurate the fixed point network will be.
   A high value indicates high precision, and a low value indicates low
   precision.

   A negative value indicates very low precision, and a very
   strong possibility for overflow.
   (the actual fix point will be set to 0, since a negative
   fix point does not make sence).

   Generally, a fix point lower than 6 is bad, and should be avoided.
   The best way to avoid this, is to have less connections to each neuron,
   or just less neurons in each layer.

   The fixed point use of this network is only intended for use on machines that
   have no floating point processor, like an iPAQ. On normal computers the floating
   point version is actually faster.
*/
int fann_save_to_fixed(struct fann *ann, const char *configuration_file);

/* ----- Some stuff to set options on the network on the fly. ----- */

/* Set the learning rate.
 */
void fann_set_learning_rate(struct fann *ann, float learning_rate);

/* The possible activation functions.
   Threshold can not be used, when training the network.
 */
#define FANN_SIGMOID 1
#define FANN_THRESHOLD 2

/* Set the activation function for the hidden layers (default SIGMOID).
 */
void fann_set_activation_function_hidden(struct fann *ann, unsigned int activation_function);

/* Set the activation function for the output layer (default SIGMOID).
 */
void fann_set_activation_function_output(struct fann *ann, unsigned int activation_function);

/* Set the steepness of the sigmoid function used in the hidden layers.
   Only usefull if sigmoid function is used in the hidden layers (default 0.5).
 */
void fann_set_activation_hidden_steepness(struct fann *ann, fann_type steepness);

/* Set the steepness of the sigmoid function used in the output layer.
   Only usefull if sigmoid function is used in the output layer (default 0.5).
 */
void fann_set_activation_output_steepness(struct fann *ann, fann_type steepness);

/* ----- Some stuff to read network options from the network. ----- */

/* Get the learning rate.
 */
float fann_get_learning_rate(struct fann *ann);

/* Get the number of input neurons.
 */
unsigned int fann_get_num_input(struct fann *ann);

/* Get the number of output neurons.
 */
unsigned int fann_get_num_output(struct fann *ann);

/* Get the activation function used in the hidden layers.
 */
unsigned int fann_get_activation_function_hidden(struct fann *ann);

/* Get the activation function used in the output layer.
 */
unsigned int fann_get_activation_function_output(struct fann *ann);

/* Get the steepness parameter for the sigmoid function used in the hidden layers.
 */
fann_type fann_get_activation_hidden_steepness(struct fann *ann);

/* Get the steepness parameter for the sigmoid function used in the output layer.
 */
fann_type fann_get_activation_output_steepness(struct fann *ann);

/* Get the total number of neurons in the entire network.
 */
unsigned int fann_get_total_neurons(struct fann *ann);

/* Get the total number of connections in the entire network.
 */
unsigned int fann_get_total_connections(struct fann *ann);

/* Randomize weights (from the beginning the weights are random between -0.1 and 0.1)
 */
void fann_randomize_weights(struct fann *ann, fann_type min_weight, fann_type max_weight);

/* ----- Training ----- */

#ifndef FIXEDFANN
/* Train one iteration with a set of inputs, and a set of desired outputs.
 */
void fann_train(struct fann *ann, fann_type *input, fann_type *desired_output);
#endif /* NOT FIXEDFANN */

/* Test with a set of inputs, and a set of desired outputs.
   This operation updates the mean square error, but does not
   change the network in any way.
*/
fann_type *fann_test(struct fann *ann, fann_type *input, fann_type *desired_output);

/* Reads a file that stores training data, in the format:
   num_train_data num_input num_output\n
   inputdata seperated by space\n
   outputdata seperated by space\n

   .
   .
   .
   
   inputdata seperated by space\n
   outputdata seperated by space\n
*/
struct fann_train_data* fann_read_train_from_file(char *filename);

/* Destructs the training data
   Be sure to call this function after finished using the training data.
 */
void fann_destroy_train(struct fann_train_data* train_data);

#ifndef FIXEDFANN
/* Trains on an entire dataset, for a maximum of max_epochs
   epochs or until mean square error is lower than desired_error.
   Reports about the progress is given every
   epochs_between_reports epochs.
   If epochs_between_reports is zero, no reports are given.
*/
void fann_train_on_data(struct fann *ann, struct fann_train_data *data, unsigned int max_epochs, unsigned int epochs_between_reports, float desired_error);

/* Does the same as train_on_data, but reads the data directly from a file.
 */
void fann_train_on_file(struct fann *ann, char *filename, unsigned int max_epochs, unsigned int epochs_between_reports, float desired_error);
#endif /* NOT FIXEDFANN */

/* Save the training structure to a file.
 */
void fann_save_train(struct fann_train_data* data, char *filename);

/* Saves the training structure to a fixed point data file.
 *  (Very usefull for testing the quality of a fixed point network).
 */
void fann_save_train_to_fixed(struct fann_train_data* data, char *filename, unsigned int decimal_point);

/* Reads the mean square error from the network.
 */
float fann_get_error(struct fann *ann);

/* Resets the mean square error from the network.
 */
void fann_reset_error(struct fann *ann);

/* ----- Running ----- */

/* Runs a input through the network, and returns the output.
 */
fann_type* fann_run(struct fann *ann, fann_type *input);

#ifdef FIXEDFANN

/* returns the position of the decimal point.
 */
unsigned int fann_get_decimal_point(struct fann *ann);

/* returns the multiplier that fix point data is multiplied with.
 */
unsigned int fann_get_multiplier(struct fann *ann);
#endif /* FIXEDFANN */

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* NOT FANN_INCLUDE */
