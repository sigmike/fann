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

#ifndef __fann_train_h__
#define __fann_train_h__

/* Section: FANN Training */

/* Struct: struct fann_train_data
	Structure used to store data, for use with training.
	
	The data inside this structure should never be manipulated directly, but should use some 
	of the supplied functions in <Training Data>.
	
	See also:
	<fann_read_train_from_file>, <fann_train_on_data>, <fann_destroy_train>
*/
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

/* Section: FANN Training */

/* Group: Training */

#ifndef FIXEDFANN
/* Function: fann_train
   Train one iteration with a set of inputs, and a set of desired outputs.
 */ 
FANN_EXTERNAL void FANN_API fann_train(struct fann *ann, fann_type * input,
									   fann_type * desired_output);

#endif	/* NOT FIXEDFANN */
	
/* Function: fann_test
   Test with a set of inputs, and a set of desired outputs.
   This operation updates the mean square error, but does not
   change the network in any way.
*/ 
	FANN_EXTERNAL fann_type * FANN_API fann_test(struct fann *ann, fann_type * input,
												 fann_type * desired_output);

/* Function: fann_get_MSE
   Reads the mean square error from the network.
 */ 
FANN_EXTERNAL float FANN_API fann_get_MSE(struct fann *ann);


/* Function: fann_reset_MSE
   Resets the mean square error from the network.
 */ 
FANN_EXTERNAL void FANN_API fann_reset_MSE(struct fann *ann);

/* Group: Training Data */

/* Function: fann_read_train_from_file
   Reads a file that stores training data, in the format:
   num_train_data num_input num_output\n
   inputdata seperated by space\n
   outputdata seperated by space\n

   .
   .
   .
   
   inputdata seperated by space\n
   outputdata seperated by space\n
*/ 
FANN_EXTERNAL struct fann_train_data *FANN_API fann_read_train_from_file(char *filename);


/* Function: fann_destroy_train
   Destructs the training data
   Be sure to call this function after finished using the training data.
 */ 
FANN_EXTERNAL void FANN_API fann_destroy_train(struct fann_train_data *train_data);


#ifndef FIXEDFANN
	
/* Function: fann_train_epoch
   Train one epoch with a set of training data.
 */ 
FANN_EXTERNAL float FANN_API fann_train_epoch(struct fann *ann, struct fann_train_data *data);


/* Function: fann_test_data
   Test a set of training data and calculate the MSE
 */ 
FANN_EXTERNAL float FANN_API fann_test_data(struct fann *ann, struct fann_train_data *data);


/* Function: fann_train_on_data

   Trains on an entire dataset, for a maximum of max_epochs
   epochs or until mean square error is lower than desired_error.
   Reports about the progress is given every
   epochs_between_reports epochs.
   If epochs_between_reports is zero, no reports are given.
*/ 
FANN_EXTERNAL void FANN_API fann_train_on_data(struct fann *ann, struct fann_train_data *data,
											   unsigned int max_epochs,
											   unsigned int epochs_between_reports,
											   float desired_error);


/* Function: fann_train_on_data_callback
   
   Same as fann_train_on_data, but a callback function is given,
   which can be used to print out reports. (effective for gui programming).
   If the callback returns -1, then the training is terminated, otherwise
   it continues until the normal stop criteria.
*/ 
FANN_EXTERNAL void FANN_API fann_train_on_data_callback(struct fann *ann,
														struct fann_train_data *data,
														unsigned int max_epochs,
														unsigned int epochs_between_reports,
														float desired_error,
														int (FANN_API *
															 callback) (unsigned int epochs,
																		float error));


/* Function: fann_train_on_file
   
   Does the same as train_on_data, but reads the data directly from a file.
 */ 
FANN_EXTERNAL void FANN_API fann_train_on_file(struct fann *ann, char *filename,
											   unsigned int max_epochs,
											   unsigned int epochs_between_reports,
											   float desired_error);


/* Function: fann_train_on_file_callback
   
   Does the same as train_on_data_callback, but
   reads the data directly from a file.
 */ 
FANN_EXTERNAL void FANN_API fann_train_on_file_callback(struct fann *ann, char *filename,
														unsigned int max_epochs,
														unsigned int epochs_between_reports,
														float desired_error,
														int (FANN_API *
															 callback) (unsigned int epochs,
																		float error));


/* Function: fann_shuffle_train_data
   
   shuffles training data, randomizing the order
 */ 
FANN_EXTERNAL void FANN_API fann_shuffle_train_data(struct fann_train_data *train_data);


/* Function: fann_scale_input_train_data
   
   Scales the inputs in the training data to the specified range
 */ 
FANN_EXTERNAL void FANN_API fann_scale_input_train_data(struct fann_train_data *train_data,
														fann_type new_min, fann_type new_max);


/* Function: fann_scale_output_train_data
   
   Scales the inputs in the training data to the specified range
 */ 
FANN_EXTERNAL void FANN_API fann_scale_output_train_data(struct fann_train_data *train_data,
														 fann_type new_min, fann_type new_max);


/* Function: fann_scale_train_data
   
   Scales the inputs in the training data to the specified range
 */ 
FANN_EXTERNAL void FANN_API fann_scale_train_data(struct fann_train_data *train_data,
												  fann_type new_min, fann_type new_max);


/* Function: fann_merge_train_data
   
   merges training data into a single struct.
 */ 
FANN_EXTERNAL struct fann_train_data *FANN_API fann_merge_train_data(struct fann_train_data *data1,
																	 struct fann_train_data *data2);


/* Function: fann_duplicate_train_data
   
   return a copy of a fann_train_data struct
 */ 
FANN_EXTERNAL struct fann_train_data *FANN_API fann_duplicate_train_data(struct fann_train_data
																		 *data);


#endif	/* NOT FIXEDFANN */
	
/* Function: fann_save_train
   
   Save the training structure to a file.
 */ 
FANN_EXTERNAL void FANN_API fann_save_train(struct fann_train_data *data, char *filename);


/* Function: fann_save_train_to_fixed
   
   Saves the training structure to a fixed point data file.
 *  (Very usefull for testing the quality of a fixed point network).
 */ 
FANN_EXTERNAL void FANN_API fann_save_train_to_fixed(struct fann_train_data *data, char *filename,
													 unsigned int decimal_point);


/* Group: Parameters */

/* Function: fann_get_training_algorithm

   Get the training algorithm.
 */ 
FANN_EXTERNAL enum fann_train_enum FANN_API fann_get_training_algorithm(struct fann *ann);


/* Function: fann_set_training_algorithm

   Set the training algorithm.
 */ 
FANN_EXTERNAL void FANN_API fann_set_training_algorithm(struct fann *ann,
														enum fann_train_enum training_algorithm);


/* Function: fann_get_learning_rate

   Get the learning rate.
 */ 
FANN_EXTERNAL float FANN_API fann_get_learning_rate(struct fann *ann);


/* Function: fann_set_learning_rate

   Set the learning rate.
 */ 
FANN_EXTERNAL void FANN_API fann_set_learning_rate(struct fann *ann, float learning_rate);


/* Function: fann_set_activation_function_hidden

   Set the activation function for the hidden layers.
 */ 
FANN_EXTERNAL void FANN_API fann_set_activation_function_hidden(struct fann *ann,
																enum fann_activationfunc_enum
																activation_function);


/* Function: fann_set_activation_function_output

   Set the activation function for the output layer.
 */ 
FANN_EXTERNAL void FANN_API fann_set_activation_function_output(struct fann *ann,
																enum fann_activationfunc_enum
																activation_function);

/* Function: fann_set_activation_steepness_hidden

   Set the steepness of the sigmoid function used in the hidden layers.
    (default 0.5).
 */ 
FANN_EXTERNAL void FANN_API fann_set_activation_steepness_hidden(struct fann *ann,
																 fann_type steepness);


/* Function: fann_set_activation_steepness_output

   Set the steepness of the sigmoid function used in the output layer.
   Only usefull if sigmoid function is used in the output layer (default 0.5).
 */ 
FANN_EXTERNAL void FANN_API fann_set_activation_steepness_output(struct fann *ann,
																 fann_type steepness);


/* Function: fann_set_train_error_function

   Set the error function used during training. (default FANN_ERRORFUNC_TANH)
 */ 
FANN_EXTERNAL void FANN_API fann_set_train_error_function(struct fann *ann,
														  enum fann_errorfunc_enum 
														  train_error_function);


/* Function: fann_get_train_error_function

   Get the error function used during training.
 */ 
FANN_EXTERNAL enum fann_errorfunc_enum FANN_API fann_get_train_error_function(struct fann *ann);


/* Function: fann_set_train_stop_function

   Set the stop function used during training. (default FANN_STOPFUNC_MSE)
 */ 
FANN_EXTERNAL void FANN_API fann_set_train_stop_function(struct fann *ann,
														 enum fann_stopfunc_enum train_stop_function);


/* Function: fann_get_train_stop_function

   Get the stop function used during training.
 */ 
FANN_EXTERNAL enum fann_stopfunc_enum FANN_API fann_get_train_stop_function(struct fann *ann);

/* Function: fann_get_bit_fail_limit

   Get the bit fail limit used during training.
 */ 
FANN_EXTERNAL fann_type FANN_API fann_get_bit_fail_limit(struct fann *ann);

/* Function: fann_set_bit_fail_limit

   Set the bit fail limit used during training. (default 0.35)
  
   The bit fail limit is the maximum difference between the actual output and the desired output 
   which is accepted when counting the bit fails.
   This difference is multiplied by two when dealing with symmetric activation functions,
   so that symmetric and not symmetric activation functions can use the same limit.
 */ 
FANN_EXTERNAL void FANN_API fann_set_bit_fail_limit(struct fann *ann, fann_type bit_fail_limit);

/* Function: fann_get_quickprop_decay

   Decay is used to make the weights do not go so high (default -0.0001). 
 */
FANN_EXTERNAL float FANN_API fann_get_quickprop_decay(struct fann *ann);


/* Function: fann_set_quickprop_decay

   Decay is used to make the weights do not go so high (default -0.0001). */ 
FANN_EXTERNAL void FANN_API fann_set_quickprop_decay(struct fann *ann, float quickprop_decay);


/* Function: fann_get_quickprop_mu

   Mu is a factor used to increase and decrease the stepsize (default 1.75). */ 
FANN_EXTERNAL float FANN_API fann_get_quickprop_mu(struct fann *ann);


/* Function: fann_set_quickprop_mu

   Mu is a factor used to increase and decrease the stepsize (default 1.75). */ 
FANN_EXTERNAL void FANN_API fann_set_quickprop_mu(struct fann *ann, float quickprop_mu);


/* Function: fann_get_rprop_increase_factor

   Tells how much the stepsize should increase during learning (default 1.2). */ 
FANN_EXTERNAL float FANN_API fann_get_rprop_increase_factor(struct fann *ann);


/* Function: fann_set_rprop_increase_factor

   Tells how much the stepsize should increase during learning (default 1.2). */ 
FANN_EXTERNAL void FANN_API fann_set_rprop_increase_factor(struct fann *ann,
														   float rprop_increase_factor);


/* Function: fann_get_rprop_decrease_factor

   Tells how much the stepsize should decrease during learning (default 0.5). */ 
FANN_EXTERNAL float FANN_API fann_get_rprop_decrease_factor(struct fann *ann);


/* Function: fann_set_rprop_decrease_factor

   Tells how much the stepsize should decrease during learning (default 0.5). */ 
FANN_EXTERNAL void FANN_API fann_set_rprop_decrease_factor(struct fann *ann,
														   float rprop_decrease_factor);


/* Function: fann_get_rprop_delta_min

   The minimum stepsize (default 0.0). */ 
FANN_EXTERNAL float FANN_API fann_get_rprop_delta_min(struct fann *ann);


/* Function: fann_set_rprop_delta_min

   The minimum stepsize (default 0.0). */ 
FANN_EXTERNAL void FANN_API fann_set_rprop_delta_min(struct fann *ann, float rprop_delta_min);


/* Function: fann_get_rprop_delta_max

   The maximum stepsize (default 50.0). */ 
FANN_EXTERNAL float FANN_API fann_get_rprop_delta_max(struct fann *ann);


/* Function: fann_set_rprop_delta_max

   The maximum stepsize (default 50.0). */ 
FANN_EXTERNAL void FANN_API fann_set_rprop_delta_max(struct fann *ann, float rprop_delta_max);


#endif
