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

/* Package: FANN Training */

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
	
/* Package: Parameters */

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


/* Function: fann_get_train_stop_function

   Set the stop function used during training. (default FANN_STOPFUNC_MSE)
 */ 
FANN_EXTERNAL void FANN_API fann_set_train_stop_function(struct fann *ann,
														 enum fann_stopfunc_enum train_stop_function);


/* Function: fann_get_train_stop_function

   Get the stop function used during training.
 */ 
FANN_EXTERNAL enum fann_stopfunc_enum FANN_API fann_get_train_stop_function(struct fann *ann);


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
