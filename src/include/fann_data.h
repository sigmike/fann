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

#include <stdio.h>
#include "fann_activation.h"
#include "fann_errno.h"

enum fann_train_enum
{
	/* Standard backpropagation incremental or online training */
	FANN_TRAIN_INCREMENTAL = 0,
	/* Standard backpropagation batch training */
	FANN_TRAIN_BATCH,
	/* The iRprop- training algorithm */
	FANN_TRAIN_RPROP,
	/* The quickprop training algorithm */
	FANN_TRAIN_QUICKPROP
};

static char const *const FANN_TRAIN_NAMES[] = {
	"FANN_TRAIN_INCREMENTAL",
	"FANN_TRAIN_BATCH",
	"FANN_TRAIN_RPROP",
	"FANN_TRAIN_QUICKPROP"
};

/* Error function used during training */
enum fann_errorfunc_enum
{
	/* Standard linear error function */
	FANN_ERRORFUNC_LINEAR = 0,
	/* Tanh error function, usually better but can require
	 * a lower learning rate */
	FANN_ERRORFUNC_TANH
};

static char const *const FANN_ERRORFUNC_NAMES[] = {
	"FANN_ERRORFUNC_LINEAR",
	"FANN_ERRORFUNC_TANH"
};

/* Stop function used during training */
enum fann_stopfunc_enum
{
	/* Stop criteria is MSE value */
	FANN_STOPFUNC_MSE = 0,
	/* Stop criteria is number of bits that fail */
	FANN_STOPFUNC_BIT
};

static char const *const FANN_STOPFUNC_NAMES[] = {
	"FANN_STOPFUNC_MSE",
	"FANN_STOPFUNC_BIT"
};

/* ----- Data structures -----
 * No data within these structures should be altered directly by the user.
 */

struct fann_neuron
{
	/* Index to the first and last connection
	 * (actually the last is a past end index)
	 */
	unsigned int first_con;
	unsigned int last_con;
	/* The sum of the inputs multiplied with the weights */
	fann_type sum;
	/* The value of the activation function applied to the sum */
	fann_type value;
	/* The steepness of the activation function */
	fann_type activation_steepness;
	/* Used to choose which activation function to use */
	enum fann_activationfunc_enum activation_function;
#ifdef __GNUC__
} __attribute__ ((packed));
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

/* Structure used to store error-related information */
struct fann_error
{
	enum fann_errno_enum errno_f;
	FILE *error_log;
	char *errstr;
};


/* The fast artificial neural network(fann) structure
 */
struct fann
{
	/* The type of error that last occured. */
	enum fann_errno_enum errno_f;

	/* Where to log error messages. */
	FILE *error_log;

	/* A string representation of the last error. */
	char *errstr;

	/* the learning rate of the network */
	float learning_rate;

	/* the connection rate of the network
	 * between 0 and 1, 1 meaning fully connected
	 */
	float connection_rate;

	/* is 1 if shortcut connections are used in the ann otherwise 0
	 * Shortcut connections are connections that skip layers.
	 * A fully connected ann with shortcut connections are a ann where
	 * neurons have connections to all neurons in all later layers.
	 */
	unsigned int shortcut_connections;

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

	/* The weight array */
	fann_type *weights;

	/* The connection array */
	struct fann_neuron **connections;

	/* Used to contain the errors used during training
	 * Is allocated during first training session,
	 * which means that if we do not train, it is never allocated.
	 */
	fann_type *train_errors;

	/* Training algorithm used when calling fann_train_on_..
	 */
	enum fann_train_enum training_algorithm;

#ifdef FIXEDFANN
	/* the decimal_point, used for shifting the fix point
	 * in fixed point integer operatons.
	 */
	unsigned int decimal_point;

	/* the multiplier, used for multiplying the fix point
	 * in fixed point integer operatons.
	 * Only used in special cases, since the decimal_point is much faster.
	 */
	unsigned int multiplier;

	/* When in choosen (or in fixed point), the sigmoid function is
	 * calculated as a stepwise linear function. In the
	 * activation_results array, the result is saved, and in the
	 * two values arrays, the values that gives the results are saved.
	 */
	fann_type sigmoid_results[6];
	fann_type sigmoid_values[6];
	fann_type sigmoid_symmetric_results[6];
	fann_type sigmoid_symmetric_values[6];
#endif

	/* Total number of connections.
	 * very usefull, because the actual connections
	 * are allocated in one long array
	 */
	unsigned int total_connections;

	/* used to store outputs in */
	fann_type *output;

	/* the number of data used to calculate the mean square error.
	 */
	unsigned int num_MSE;

	/* the total error value.
	 * the real mean square error is MSE_value/num_MSE
	 */
	float MSE_value;

	/* The number of outputs which would fail (only valid for classification problems)
	 */
	unsigned int num_bit_fail;

	/* The maximum difference between the actual output and the expected output 
	 * which is accepted when counting the bit fails.
	 * This difference is multiplied by two when dealing with symmetric activation functions,
	 * so that symmetric and not symmetric activation functions can use the same limit.
	 */
	fann_type bit_fail_limit;

	/* The error function used during training. (default FANN_ERRORFUNC_TANH)
	 */
	enum fann_errorfunc_enum train_error_function;
	
	/* The stop function used during training. (default FANN_STOPFUNC_MSE)
	*/
	enum fann_stopfunc_enum train_stop_function;

	/* Variables for use with Cascade Correlation */

	/* The error must change by at least this
	 * fraction of its old value to count as a
	 * significant change.
	 */
	float cascade_change_fraction;

	/* No change in this number of epochs will cause
	 * stagnation.
	 */
	unsigned int cascade_stagnation_epochs;

	/* The current best candidate, which will be installed.
	 */
	unsigned int cascade_best_candidate;

	/* The upper limit for a candidate score
	 */
	fann_type cascade_candidate_limit;

	/* Scale of copied candidate output weights
	 */
	fann_type cascade_weight_multiplier;
	
	/* Maximum epochs to train the output neurons during cascade training
	 */
	unsigned int cascade_max_out_epochs;
	
	/* Maximum epochs to train the candidate neurons during cascade training
	 */
	unsigned int cascade_max_cand_epochs;	

	/* An array consisting of the activation functions used when doing
	 * cascade training.
	 */
	enum fann_activationfunc_enum *cascade_activation_functions;
	
	/* The number of elements in the cascade_activation_functions array.
	*/
	unsigned int cascade_activation_functions_count;
	
	/* An array consisting of the steepnesses used during cascade training.
	*/
	fann_type *cascade_activation_steepnesses;

	/* The number of elements in the cascade_activation_steepnesses array.
	*/
	unsigned int cascade_activation_steepnesses_count;
	
	/* The number of candidates of each type that will be present.
	 * The actual number of candidates is then 
	 * cascade_activation_functions_count * 
	 * cascade_activation_steepnesses_count *
	 * cascade_num_candidate_groups
	*/
	unsigned int cascade_num_candidate_groups;
	
	/* An array consisting of the score of the individual candidates,
	 * which is used to decide which candidate is the best
	 */
	fann_type *cascade_candidate_scores;
	
	/* The number of allocated neurons during cascade correlation algorithms.
	 * This number might be higher than the actual number of neurons to avoid
	 * allocating new space too often.
	 */
	unsigned int total_neurons_allocated;

	/* The number of allocated connections during cascade correlation algorithms.
	 * This number might be higher than the actual number of neurons to avoid
	 * allocating new space too often.
	 */
	unsigned int total_connections_allocated;

	/* Variables for use with Quickprop training */

	/* Decay is used to make the weights not go so high */
	float quickprop_decay;

	/* Mu is a factor used to increase and decrease the stepsize */
	float quickprop_mu;

	/* Variables for use with with RPROP training */

	/* Tells how much the stepsize should increase during learning */
	float rprop_increase_factor;

	/* Tells how much the stepsize should decrease during learning */
	float rprop_decrease_factor;

	/* The minimum stepsize */
	float rprop_delta_min;

	/* The maximum stepsize */
	float rprop_delta_max;

	/* The initial stepsize */
	float rprop_delta_zero;

	/* Used to contain the slope errors used during batch training
	 * Is allocated during first training session,
	 * which means that if we do not train, it is never allocated.
	 */
	fann_type *train_slopes;

	/* The previous step taken by the quickprop/rprop procedures.
	 * Not allocated if not used.
	 */
	fann_type *prev_steps;

	/* The slope values used by the quickprop/rprop procedures.
	 * Not allocated if not used.
	 */
	fann_type *prev_train_slopes;
};

/* Structure used to store data, for use with training. */
struct fann_train_data
{
	enum fann_errno_enum errno_f;
	FILE *error_log;
	char *errstr;

	unsigned int num_data;
	unsigned int num_input;
	unsigned int num_output;
	fann_type **input;
	fann_type **output;
};

#endif
