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
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

#include "config.h"
#include "fann.h"
#include "fann_errno.h"

void fann_set_learning_rate(struct fann *ann, float learning_rate)
{
	ann->learning_rate = learning_rate;
}

void fann_set_activation_function_hidden(struct fann *ann, unsigned int activation_function)
{
	ann->activation_function_hidden = activation_function;
	fann_update_stepwise_hidden(ann);
}

void fann_set_activation_function_output(struct fann *ann, unsigned int activation_function)
{
	ann->activation_function_output = activation_function;
	fann_update_stepwise_output(ann);
}

void fann_set_activation_hidden_steepness(struct fann *ann, fann_type steepness)
{
	ann->activation_hidden_steepness = steepness;
	fann_update_stepwise_hidden(ann);
}

void fann_set_activation_output_steepness(struct fann *ann, fann_type steepness)
{
	ann->activation_output_steepness = steepness;
	fann_update_stepwise_output(ann);
}

float fann_get_learning_rate(struct fann *ann)
{
	return ann->learning_rate;
}

unsigned int fann_get_num_input(struct fann *ann)
{
	return ann->num_input;
}

unsigned int fann_get_num_output(struct fann *ann)
{
	return ann->num_output;
}

unsigned int fann_get_activation_function_hidden(struct fann *ann)
{
	return ann->activation_function_hidden;
}

unsigned int fann_get_activation_function_output(struct fann *ann)
{
	return ann->activation_function_output;
}

fann_type fann_get_activation_hidden_steepness(struct fann *ann)
{
	return ann->activation_hidden_steepness;
}

fann_type fann_get_activation_output_steepness(struct fann *ann)
{
	return ann->activation_output_steepness;
}

unsigned int fann_get_total_neurons(struct fann *ann)
{
	/* -1, because there is always an unused bias neuron in the last layer */
	return ann->total_neurons - 1;
}

unsigned int fann_get_total_connections(struct fann *ann)
{
	return ann->total_connections;
}

#ifdef FIXEDFANN
/* returns the position of the fix point.
 */
unsigned int fann_get_decimal_point(struct fann *ann)
{
	return ann->decimal_point;
}

/* returns the multiplier that fix point data is multiplied with.
 */
unsigned int fann_get_multiplier(struct fann *ann)
{
	return ann->multiplier;
}

#endif

/* INTERNAL FUNCTION
   Adjust the steepwise functions (if used)
*/
void fann_update_stepwise_hidden(struct fann *ann)
{
	unsigned int i = 0;
#ifndef FIXEDFANN
	/* For use in stepwise linear activation function.
	   results 0.005, 0.05, 0.25, 0.75, 0.95, 0.995
	*/
	switch(ann->activation_function_hidden){
		case FANN_SIGMOID:
		case FANN_SIGMOID_STEPWISE:
			ann->activation_hidden_results[0] = (fann_type)0.005;
			ann->activation_hidden_results[1] = (fann_type)0.05;
			ann->activation_hidden_results[2] = (fann_type)0.25;
			ann->activation_hidden_results[3] = (fann_type)0.75;
			ann->activation_hidden_results[4] = (fann_type)0.95;
			ann->activation_hidden_results[5] = (fann_type)0.995;	
			break;
		case FANN_SIGMOID_SYMMETRIC:
		case FANN_SIGMOID_SYMMETRIC_STEPWISE:
			ann->activation_hidden_results[0] = (fann_type)-0.99;
			ann->activation_hidden_results[1] = (fann_type)-0.9;
			ann->activation_hidden_results[2] = (fann_type)-0.5;
			ann->activation_hidden_results[3] = (fann_type)0.5;
			ann->activation_hidden_results[4] = (fann_type)0.9;
			ann->activation_hidden_results[5] = (fann_type)0.99;
			break;
		default:
			/* the actiavation functions which do not have a stepwise function
			   should not have it calculated */
			return;
	}
#else
	/* Calculate the parameters for the stepwise linear
	   sigmoid function fixed point.
	   Using a rewritten sigmoid function.
	   results 0.005, 0.05, 0.25, 0.75, 0.95, 0.995
	*/
	switch(ann->activation_function_hidden){
		case FANN_SIGMOID:
		case FANN_SIGMOID_STEPWISE:
			ann->activation_hidden_results[0] = (fann_type)(ann->multiplier/200.0+0.5);
			ann->activation_hidden_results[1] = (fann_type)(ann->multiplier/20.0+0.5);
			ann->activation_hidden_results[2] = (fann_type)(ann->multiplier/4.0+0.5);
			ann->activation_hidden_results[3] = ann->multiplier - (fann_type)(ann->multiplier/4.0+0.5);
			ann->activation_hidden_results[4] = ann->multiplier - (fann_type)(ann->multiplier/20.0+0.5);
			ann->activation_hidden_results[5] = ann->multiplier - (fann_type)(ann->multiplier/200.0+0.5);
			break;
		case FANN_SIGMOID_SYMMETRIC:
		case FANN_SIGMOID_SYMMETRIC_STEPWISE:
			ann->activation_hidden_results[0] = (fann_type)((ann->multiplier/100.0) - ann->multiplier + 0.5);
			ann->activation_hidden_results[1] = (fann_type)((ann->multiplier/10.0) - ann->multiplier + 0.5);
			ann->activation_hidden_results[2] = (fann_type)((ann->multiplier/2.0) - ann->multiplier + 0.5);
			ann->activation_hidden_results[3] = ann->multiplier - (fann_type)(ann->multiplier/2.0+0.5);
			ann->activation_hidden_results[4] = ann->multiplier - (fann_type)(ann->multiplier/10.0+0.5);
			ann->activation_hidden_results[5] = ann->multiplier - (fann_type)(ann->multiplier/100.0+0.5);
			break;
		default:
			/* the actiavation functions which do not have a stepwise function
			   should not have it calculated */
			return;
	}			
#endif

	for(i = 0; i < 6; i++){
#ifndef FIXEDFANN
		switch(ann->activation_function_hidden){
			case FANN_SIGMOID:
				break;
			case FANN_SIGMOID_STEPWISE:
				ann->activation_hidden_values[i] = (fann_type)((log(1.0/ann->activation_hidden_results[i] -1.0) * 1.0/-2.0) * 1.0/ann->activation_hidden_steepness);
				break;
			case FANN_SIGMOID_SYMMETRIC:
			case FANN_SIGMOID_SYMMETRIC_STEPWISE:
				ann->activation_hidden_values[i] = (fann_type)((log((1.0-ann->activation_hidden_results[i]) / (ann->activation_hidden_results[i]+1.0)) * 1.0/-2.0) * 1.0/ann->activation_hidden_steepness);
				break;
		}
#else
		switch(ann->activation_function_hidden){
			case FANN_SIGMOID:
			case FANN_SIGMOID_STEPWISE:
				ann->activation_hidden_values[i] = (fann_type)((((log(ann->multiplier/(float)ann->activation_hidden_results[i] -1)*(float)ann->multiplier) / -2.0)*(float)ann->multiplier) / ann->activation_hidden_steepness);
				break;
			case FANN_SIGMOID_SYMMETRIC:
			case FANN_SIGMOID_SYMMETRIC_STEPWISE:
				ann->activation_hidden_values[i] = (fann_type)((((log((ann->multiplier - (float)ann->activation_hidden_results[i])/((float)ann->activation_hidden_results[i] + ann->multiplier))*(float)ann->multiplier) / -2.0)*(float)ann->multiplier) / ann->activation_hidden_steepness);
				break;
		}
#endif
	}
}

/* INTERNAL FUNCTION
   Adjust the steepwise functions (if used)
*/
void fann_update_stepwise_output(struct fann *ann)
{
	unsigned int i = 0;
#ifndef FIXEDFANN
	/* For use in stepwise linear activation function.
	   results 0.005, 0.05, 0.25, 0.75, 0.95, 0.995
	*/
	switch(ann->activation_function_output){
		case FANN_SIGMOID:
		case FANN_SIGMOID_STEPWISE:
			ann->activation_output_results[0] = (fann_type)0.005;
			ann->activation_output_results[1] = (fann_type)0.05;
			ann->activation_output_results[2] = (fann_type)0.25;
			ann->activation_output_results[3] = (fann_type)0.75;
			ann->activation_output_results[4] = (fann_type)0.95;
			ann->activation_output_results[5] = (fann_type)0.995;	
			break;
		case FANN_SIGMOID_SYMMETRIC:
		case FANN_SIGMOID_SYMMETRIC_STEPWISE:
			ann->activation_output_results[0] = (fann_type)-0.99;
			ann->activation_output_results[1] = (fann_type)-0.9;
			ann->activation_output_results[2] = (fann_type)-0.5;
			ann->activation_output_results[3] = (fann_type)0.5;
			ann->activation_output_results[4] = (fann_type)0.9;
			ann->activation_output_results[5] = (fann_type)0.99;
			break;
		default:
			/* the actiavation functions which do not have a stepwise function
			   should not have it calculated */
			return;
	}
#else
	/* Calculate the parameters for the stepwise linear
	   sigmoid function fixed point.
	   Using a rewritten sigmoid function.
	   results 0.005, 0.05, 0.25, 0.75, 0.95, 0.995
	*/
	switch(ann->activation_function_output){
		case FANN_SIGMOID:
		case FANN_SIGMOID_STEPWISE:
			ann->activation_output_results[0] = (fann_type)(ann->multiplier/200.0+0.5);
			ann->activation_output_results[1] = (fann_type)(ann->multiplier/20.0+0.5);
			ann->activation_output_results[2] = (fann_type)(ann->multiplier/4.0+0.5);
			ann->activation_output_results[3] = ann->multiplier - (fann_type)(ann->multiplier/4.0+0.5);
			ann->activation_output_results[4] = ann->multiplier - (fann_type)(ann->multiplier/20.0+0.5);
			ann->activation_output_results[5] = ann->multiplier - (fann_type)(ann->multiplier/200.0+0.5);
			break;
		case FANN_SIGMOID_SYMMETRIC:
		case FANN_SIGMOID_SYMMETRIC_STEPWISE:
			ann->activation_output_results[0] = (fann_type)((ann->multiplier/100.0) - ann->multiplier + 0.5);
			ann->activation_output_results[1] = (fann_type)((ann->multiplier/10.0) - ann->multiplier + 0.5);
			ann->activation_output_results[2] = (fann_type)((ann->multiplier/2.0) - ann->multiplier + 0.5);
			ann->activation_output_results[3] = ann->multiplier - (fann_type)(ann->multiplier/2.0+0.5);
			ann->activation_output_results[4] = ann->multiplier - (fann_type)(ann->multiplier/10.0+0.5);
			ann->activation_output_results[5] = ann->multiplier - (fann_type)(ann->multiplier/100.0+0.5);
			break;
		default:
			/* the actiavation functions which do not have a stepwise function
			   should not have it calculated */
			return;
	}			
#endif

	for(i = 0; i < 6; i++){
#ifndef FIXEDFANN
		switch(ann->activation_function_output){
			case FANN_SIGMOID:
				break;
			case FANN_SIGMOID_STEPWISE:
				ann->activation_output_values[i] = (fann_type)((log(1.0/ann->activation_output_results[i] -1.0) * 1.0/-2.0) * 1.0/ann->activation_output_steepness);
				break;
			case FANN_SIGMOID_SYMMETRIC:
			case FANN_SIGMOID_SYMMETRIC_STEPWISE:
				ann->activation_output_values[i] = (fann_type)((log((1.0-ann->activation_output_results[i]) / (ann->activation_output_results[i]+1.0)) * 1.0/-2.0) * 1.0/ann->activation_output_steepness);
				break;
		}
#else
		switch(ann->activation_function_output){
			case FANN_SIGMOID:
			case FANN_SIGMOID_STEPWISE:
				ann->activation_output_values[i] = (fann_type)((((log(ann->multiplier/(float)ann->activation_output_results[i] -1)*(float)ann->multiplier) / -2.0)*(float)ann->multiplier) / ann->activation_output_steepness);
				break;
			case FANN_SIGMOID_SYMMETRIC:
			case FANN_SIGMOID_SYMMETRIC_STEPWISE:
				ann->activation_output_values[i] = (fann_type)((((log((ann->multiplier - (float)ann->activation_output_results[i])/((float)ann->activation_output_results[i] + ann->multiplier))*(float)ann->multiplier) / -2.0)*(float)ann->multiplier) / ann->activation_output_steepness);
				break;
		}
#endif
	}
}
