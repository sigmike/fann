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

#ifndef __fann_activation_h__
#define __fann_activation_h__
/* internal include file, not to be included directly
 */

/* The possible activation functions.
   They are described with functions,
   where x is the input to the activation function,
   y is the output,
   s is the steepness and
   d is the derivation.
 */

/* Linear activation function.
   span: -inf < y < inf
   y = x*s, d = 1*s
   Can NOT be used in fixed point.
   NOT implemented yet.
*/
#define FANN_LINEAR 4

/* Threshold activation function.
   x < 0 -> y = 0, x >= 0 -> y = 1
   Can NOT be used during training.
*/
#define FANN_THRESHOLD 2

/* Sigmoid activation function.
   One of the most used activation functions.
   span: 0 < y < 1
   y = 1/(1 + exp(-2*s*x)), d = 2*s*y*(1 - y)
*/
#define FANN_SIGMOID 1

/* Stepwise linear approximation to sigmoid.
   Faster than sigmoid but a bit less precise.
*/
#define FANN_SIGMOID_STEPWISE 3 /* (default) */


/* Symmetric sigmoid activation function, aka. tanh.
   One of the most used activation functions.
   span: -1 < y < 1
   y = tanh(s*x) = 2/(1 + exp(-2*s*x)) - 1, d = s*(1-(y*y))
   NOT implemented yet.
*/
#define FANN_SIGMOID_SYMMETRIC 5
	
/* Stepwise linear approximation to symmetric sigmoid.
   Faster than symmetric sigmoid but a bit less precise.
   NOT implemented yet.
*/
#define FANN_SIGMOID_SYMMETRIC_STEPWISE 6

/* Gausian activation function.
   0 when x = -inf, 1 when x = 0 and 0 when x = inf
   span: 0 < y < 1
   y = exp(-x*s*x*s), d = -2*x*y*s
   NOT implemented yet.
*/
#define FANN_GAUSSIAN 7

/* Stepwise linear approximation to gaussian.
   Faster than gaussian but a bit less precise.
   NOT implemented yet.
*/
#define FANN_GAUSSIAN_STEPWISE 8 /* not implemented yet. */

/* Fast (sigmoid like) activation function defined by David Elliott
   span: 0 < y < 1
   y = ((x*s) / 2) / (1 + |x*s|) + 0.5, d = s*1/(2*(1+|x|)*(1+|x|))
   NOT implemented yet.
*/
#define FANN_ELLIOT 9

/* Fast (symmetric sigmoid like) activation function defined by David Elliott
   span: -1 < y < 1   
   y = (x*s) / (1 + |x*s|), d = s*1/((1+|x|)*(1+|x|))
   NOT implemented yet.
*/
#define FANN_ELLIOT_SYMMETRIC 10

/* Implementation of the activation functions
 */

/* stepwise linear functions used for some of the activation functions */
#define fann_linear_func(v1, r1, v2, r2, value) ((((r2-r1) * (value-v1))/(v2-v1)) + r1)
#define fann_stepwise(v1, v2, v3, v4, v5, v6, r1, r2, r3, r4, r5, r6, value, multiplier) (value < v5 ? (value < v3 ? (value < v2 ? (value < v1 ? 0 : fann_linear_func(v1, r1, v2, r2, value)) : fann_linear_func(v2, r2, v3, r3, value)) : (value < v4 ? fann_linear_func(v3, r3, v4, r4, value) : fann_linear_func(v4, r4, v5, r5, value))) : (value < v6 ? fann_linear_func(v5, r5, v6, r6, value) : multiplier))

#ifdef FIXEDFANN
#define fann_sigmoid(steepness, value) ((fann_type)(0.5+((1.0/(1.0 + exp(-2.0 * ((float)steepness/multiplier) * ((float)value/multiplier))))*multiplier)))

#else

#define fann_sigmoid(steepness, value) (1.0/(1.0 + exp(-2.0 * steepness * value)))
#define fann_sigmoid_derive(steepness, value) ((2.0 * steepness * value * (1.0 - value)) + 0.01) /* the plus is a trick to the derived function, to avoid getting stuck on flat spots */
#endif


#endif
