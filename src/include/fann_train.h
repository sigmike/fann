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

/* Section: FANN Training 
 
 	There are many different ways of training neural networks and the FANN library supports
 	a number of different approaches. 
 	
 	Two fundementally different approaches are the most commonly used:
 	
 		Fixed topology training - The size and topology of the ANN is determined in advance
 			and the training alters the weights in order to minimize the difference between
 			the desired output values and the actual output values. This kind of training is 
 			supported by <fann_train_on_data>.
 			
 		Evolving topology training - The training start out with an empty ANN, only consisting
 			of input and output neurons. Hidden neurons and connections is the added during training,
 			in order to reach the same goal as for fixed topology training. This kind of training
 			is supported by <FANN Cascade Training>.
 */

/* Struct: struct fann_train_data
	Structure used to store data, for use with training.
	
	The data inside this structure should never be manipulated directly, but should use some 
	of the supplied functions in <Training Data>.
	
	The training data structure is very usefull for storing data during training and testing of a
	neural network.
   
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
   This training is always incremental training (see <fann_train_enum>), since
   only one pattern is presented.
   
   Parameters:
   	ann - The neural network structure
   	input - an array of inputs. This array must be exactly <fann_get_num_input> long.
   	desired_output - an array of desired outputs. This array must be exactly <fann_get_num_output> long.
   	
   	See also:
   		<fann_train_on_data>, <fann_train_epoch>
   	
   	This function appears in FANN >= 1.0.0.
 */ 
FANN_EXTERNAL void FANN_API fann_train(struct fann *ann, fann_type * input,
									   fann_type * desired_output);

#endif	/* NOT FIXEDFANN */
	
/* Function: fann_test
   Test with a set of inputs, and a set of desired outputs.
   This operation updates the mean square error, but does not
   change the network in any way.
   
   See also:
   		<fann_test_data>, <fann_train>
   
   This function appears in FANN >= 1.0.0.
*/ 
FANN_EXTERNAL fann_type * FANN_API fann_test(struct fann *ann, fann_type * input,
												 fann_type * desired_output);

/* Function: fann_get_MSE
   Reads the mean square error from the network.
   
   Reads the mean square error from the network. This value is calculated during 
   training or testing, and can therefore sometimes be a bit off if the weights 
   have been changed since the last calculation of the value.
   
   See also:
   	<fann_test_data>

	This function appears in FANN >= 1.1.0.
 */ 
FANN_EXTERNAL float FANN_API fann_get_MSE(struct fann *ann);

/* Function: fann_get_bit_fail
	
	The number of fail bits; means the number of output neurons which differ more 
	than the bit fail limit (see <fann_get_bit_fail_limit>, <fann_set_bit_fail_limit>). 
	The bits are counted in all of the training data, so this number can be higher than
	the number of training data.
	
	This value is reset by <fann_reset_MSE> and updated by all the same functions which also
	updates the MSE value (e.g. <fann_test_data>, <fann_train_epoch>)
	
	See also:
		<fann_stopfunc_enum>, <fann_get_MSE>

	This function appears in FANN >= 2.0.0
*/
FANN_EXTERNAL unsigned int fann_get_bit_fail(struct fann *ann);

/* Function: fann_reset_MSE
   Resets the mean square error from the network.
   
   This function also resets the number of bits that fail.
   
   See also:
   	<fann_get_MSE>, <fann_get_bit_fail_limit>
   
    This function appears in FANN >= 1.1.0
 */ 
FANN_EXTERNAL void FANN_API fann_reset_MSE(struct fann *ann);

/* Group: Training Data Training */

#ifndef FIXEDFANN
	
/* Function: fann_train_on_data

   Trains on an entire dataset, for a period of time. 
   
   This training uses the training algorithm chosen by <fann_set_training_algorithm>,
   and the parameters set for these training algorithms.
   
   Parameters:
   		ann - The neural network
   		data - The data, which should be used during training
   		max_epochs - The maximum number of epochs the training should continue
   		epochs_between_reports - The number of epochs between printing a status report to stdout.
   			A value of zero means no reports should be printed.
   		desired_error - The desired <fann_get_MSE> or <fann_get_bit_fail>, depending on which stop function
   			is chosen by <fann_set_train_stop_function>.

	Instead of printing out reports every epochs_between_reports, a callback function can be called 
	(see <fann_set_callback>).
	
	See also:
		<fann_train_on_file>, <fann_train_epoch>, <Parameters>

	This function appears in FANN >= 1.0.0.
*/ 
FANN_EXTERNAL void FANN_API fann_train_on_data(struct fann *ann, struct fann_train_data *data,
											   unsigned int max_epochs,
											   unsigned int epochs_between_reports,
											   float desired_error);

/* Function: fann_train_on_file
   
   Does the same as <fann_train_on_data>, but reads the training data directly from a file.
   
   See also:
   		<fann_train_on_data>

	This function appears in FANN >= 1.0.0.
*/ 
FANN_EXTERNAL void FANN_API fann_train_on_file(struct fann *ann, char *filename,
											   unsigned int max_epochs,
											   unsigned int epochs_between_reports,
											   float desired_error);

/* Function: fann_train_epoch
   Train one epoch with a set of training data.
   
    Train one epoch with the training data stored in data. One epoch is where all of 
    the training data is considered exactly once.

	This function returns the MSE error as it is calculated either before or during 
	the actual training. This is not the actual MSE after the training epoch, but since 
	calculating this will require to go through the entire training set once more, it is 
	more than adequate to use this value during training.

	The training algorithm used by this function is chosen by the <fann_set_training_algorithm> 
	function.
	
	See also:
		<fann_train_on_data>, <fann_test_data>
		
	This function appears in FANN >= 1.2.0.
 */ 
FANN_EXTERNAL float FANN_API fann_train_epoch(struct fann *ann, struct fann_train_data *data);

/* Function: fann_test_data
  
   Test a set of training data and calculates the MSE for the training data. 
   
   This function updates the MSE and the bit fail values.
   
   See also:
 	<fann_test>, <fann_get_MSE>, <fann_get_bit_fail>

	This function appears in FANN >= 1.2.0.
 */ 
FANN_EXTERNAL float FANN_API fann_test_data(struct fann *ann, struct fann_train_data *data);

/* Group: Training Data Manipulation */

/* Function: fann_read_train_from_file
   Reads a file that stores training data.
   
   The file must be formatted like:
   >num_train_data num_input num_output
   >inputdata seperated by space
   >outputdata seperated by space
   >
   >.
   >.
   >.
   >
   >inputdata seperated by space
   >outputdata seperated by space
   
   See also:
   	<fann_train_on_data>, <fann_destroy_train>, <fann_save_train>

    This function appears in FANN >= 1.0.0
*/ 
FANN_EXTERNAL struct fann_train_data *FANN_API fann_read_train_from_file(char *filename);


/* Function: fann_destroy_train
   Destructs the training data and properly deallocates all of the associated data.
   Be sure to call this function after finished using the training data.

    This function appears in FANN >= 1.0.0
 */ 
FANN_EXTERNAL void FANN_API fann_destroy_train(struct fann_train_data *train_data);


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

/* Function: fann_set_callback
 	Sets the callback function for use during training.
 	
 	See <fann_callback_type> for more information about the callback function.
 */
FANN_EXTERNAL void fann_set_callback(struct fann *ann, fann_callback_type callback);

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
