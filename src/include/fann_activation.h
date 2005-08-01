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

enum
{
	/* Linear activation function.
	 * span: -inf < y < inf
	 * y = x*s, d = 1*s
	 * Can NOT be used in fixed point.
	 */
	FANN_LINEAR = 0,

	/* Threshold activation function.
	 * x < 0 -> y = 0, x >= 0 -> y = 1
	 * Can NOT be used during training.
	 */
	FANN_THRESHOLD,

	/* Threshold activation function.
	 * x < 0 -> y = 0, x >= 0 -> y = 1
	 * Can NOT be used during training.
	 */
	FANN_THRESHOLD_SYMMETRIC,

	/* Sigmoid activation function.
	 * One of the most used activation functions.
	 * span: 0 < y < 1
	 * y = 1/(1 + exp(-2*s*x))
	 * d = 2*s*y*(1 - y)
	 */
	FANN_SIGMOID,

	/* Stepwise linear approximation to sigmoid.
	 * Faster than sigmoid but a bit less precise.
	 */
	FANN_SIGMOID_STEPWISE,		/* (default) */


	/* Symmetric sigmoid activation function, aka. tanh.
	 * One of the most used activation functions.
	 * span: -1 < y < 1
	 * y = tanh(s*x) = 2/(1 + exp(-2*s*x)) - 1
	 * d = s*(1-(y*y))
	 */
	FANN_SIGMOID_SYMMETRIC,

	/* Stepwise linear approximation to symmetric sigmoid.
	 * Faster than symmetric sigmoid but a bit less precise.
	 */
	FANN_SIGMOID_SYMMETRIC_STEPWISE,

	/* Gaussian activation function.
	 * 0 when x = -inf, 1 when x = 0 and 0 when x = inf
	 * span: 0 < y < 1
	 * y = exp(-x*s*x*s)
	 * d = -2*x*s*y*s
	 */
	FANN_GAUSSIAN,

	/* Symmetric gaussian activation function.
	 * -1 when x = -inf, 1 when x = 0 and 0 when x = inf
	 * span: -1 < y < 1
	 * y = exp(-x*s*x*s)*2-1
	 * d = -2*x*s*(y+1)*s
	 */
	FANN_GAUSSIAN_SYMMETRIC,

	/* Stepwise linear approximation to gaussian.
	 * Faster than gaussian but a bit less precise.
	 * NOT implemented yet.
	 */
	FANN_GAUSSIAN_STEPWISE,

	/* Fast (sigmoid like) activation function defined by David Elliott
	 * span: 0 < y < 1
	 * y = ((x*s) / 2) / (1 + |x*s|) + 0.5
	 * d = s*1/(2*(1+|x*s|)*(1+|x*s|))
	 */
	FANN_ELLIOT,

	/* Fast (symmetric sigmoid like) activation function defined by David Elliott
	 * span: -1 < y < 1   
	 * y = (x*s) / (1 + |x*s|)
	 * d = s*1/((1+|x*s|)*(1+|x*s|))
	 */
	FANN_ELLIOT_SYMMETRIC
};

static char const *const FANN_ACTIVATION_NAMES[] = {
	"FANN_LINEAR",
	"FANN_THRESHOLD",
	"FANN_THRESHOLD_SYMMETRIC",
	"FANN_SIGMOID",
	"FANN_SIGMOID_STEPWISE",
	"FANN_SIGMOID_SYMMETRIC",
	"FANN_SIGMOID_SYMMETRIC_STEPWISE",
	"FANN_GAUSSIAN",
	"FANN_GAUSSIAN_SYMMETRIC",
	"FANN_GAUSSIAN_STEPWISE",
	"FANN_ELLIOT",
	"FANN_ELLIOT_SYMMETRIC"
};

/* Implementation of the activation functions
 */

/* stepwise linear functions used for some of the activation functions */

/* defines used for the stepwise linear functions */

#define fann_linear_func(v1, r1, v2, r2, sum) ((((r2-r1) * (sum-v1))/(v2-v1)) + r1)
#define fann_stepwise(v1, v2, v3, v4, v5, v6, r1, r2, r3, r4, r5, r6, min, max, sum) (sum < v5 ? (sum < v3 ? (sum < v2 ? (sum < v1 ? min : fann_linear_func(v1, r1, v2, r2, sum)) : fann_linear_func(v2, r2, v3, r3, sum)) : (sum < v4 ? fann_linear_func(v3, r3, v4, r4, sum) : fann_linear_func(v4, r4, v5, r5, sum))) : (sum < v6 ? fann_linear_func(v5, r5, v6, r6, sum) : max))

/* FANN_LINEAR */
#define fann_linear(steepness, sum) fann_mult(steepness, sum)
#define fann_linear_derive(steepness, value) (steepness)

/* FANN_SIGMOID */
#define fann_sigmoid(steepness, sum) (1.0f/(1.0f + exp(-2.0f * steepness * sum)))
#define fann_sigmoid_real(sum) (1.0f/(1.0f + exp(-2.0f * sum)))
#define fann_sigmoid_derive(steepness, value) (2.0f * steepness * value * (1.0f - value))

/* FANN_SIGMOID_SYMMETRIC */
#define fann_sigmoid_symmetric(steepness, sum) (2.0f/(1.0f + exp(-2.0f * steepness * sum)) - 1.0f)
#define fann_sigmoid_symmetric_real(sum) (2.0f/(1.0f + exp(-2.0f * sum)) - 1.0f)
#define fann_sigmoid_symmetric_derive(steepness, value) steepness * (1.0f - (value*value))

/* FANN_GAUSSIAN */
#define fann_gaussian(steepness, sum) (exp(-sum * steepness * sum * steepness))
#define fann_gaussian_real(sum) (exp(-sum * sum))
#define fann_gaussian_derive(steepness, value, sum) (-2.0f * sum * value * steepness)

/* FANN_GAUSSIAN_SYMMETRIC */
#define fann_gaussian_symmetric(steepness, sum) ((exp(-sum * steepness * sum * steepness)*2.0)-1.0)
#define fann_gaussian_symmetric_real(sum) ((exp(-sum * sum)*2.0)-1.0)
#define fann_gaussian_symmetric_derive(steepness, value, sum) (-2.0f * sum * (value+1.0f) * steepness)

/* FANN_ELLIOT */
#define fann_elliot(steepness, sum) (((sum * steepness) / 2.0f) / (1.0f + abs(sum * steepness)) + 0.5f)
#define fann_elliot_real(sum) (((sum) / 2.0f) / (1.0f + abs(sum)) + 0.5f)
#define fann_elliot_derive(steepness, value, sum) (steepness * 1.0f / (2.0f * (1.0f + abs(sum)) * (1.0f + abs(sum))))

/* FANN_ELLIOT_SYMMETRIC */
#define fann_elliot_symmetric(steepness, sum) ((sum * steepness) / (1.0f + abs(sum * steepness)))
#define fann_elliot_symmetric_real(sum) ((sum) / (1.0f + abs(sum)))
#define fann_elliot_symmetric_derive(steepness, value, sum) (steepness * 1.0f / ((1.0f + abs(sum)) * (1.0f + abs(sum))))

#define fann_activation_switch(ann, activation_function, value, result) \
switch(activation_function) \
{ \
	case FANN_LINEAR: \
		result = value; \
        break; \
	case FANN_SIGMOID: \
		result = (fann_type)fann_sigmoid_real(value); \
        break; \
	case FANN_SIGMOID_SYMMETRIC: \
		result = (fann_type)fann_sigmoid_symmetric_real(value); \
        break; \
	case FANN_SIGMOID_SYMMETRIC_STEPWISE: \
		result = (fann_type)fann_stepwise(-2.64665293693542480469e+00, -1.47221934795379638672e+00, -5.49306154251098632812e-01, 5.49306154251098632812e-01, 1.47221934795379638672e+00, 2.64665293693542480469e+00, -9.90000009536743164062e-01, -8.99999976158142089844e-01, -5.00000000000000000000e-01, 5.00000000000000000000e-01, 8.99999976158142089844e-01, 9.90000009536743164062e-01, -1, 1, value); \
        break; \
	case FANN_SIGMOID_STEPWISE: \
		result = (fann_type)fann_stepwise(-2.64665246009826660156e+00, -1.47221946716308593750e+00, -5.49306154251098632812e-01, 5.49306154251098632812e-01, 1.47221934795379638672e+00, 2.64665293693542480469e+00, 4.99999988824129104614e-03, 5.00000007450580596924e-02, 2.50000000000000000000e-01, 7.50000000000000000000e-01, 9.49999988079071044922e-01, 9.95000004768371582031e-01, 0, 1, value); \
        break; \
	case FANN_THRESHOLD: \
		result = (fann_type)((value < 0) ? 0 : 1); \
        break; \
	case FANN_THRESHOLD_SYMMETRIC: \
		result = (fann_type)((value < 0) ? -1 : 1); \
        break; \
	case FANN_GAUSSIAN: \
		result = (fann_type)fann_gaussian_real(value); \
        break; \
	case FANN_GAUSSIAN_SYMMETRIC: \
		result = (fann_type)fann_gaussian_symmetric_real(value); \
        break; \
	case FANN_ELLIOT: \
		result = (fann_type)fann_elliot_real(value); \
        break; \
	case FANN_ELLIOT_SYMMETRIC: \
		result = (fann_type)fann_elliot_symmetric_real(value); \
        break; \
	default: \
		fann_error((struct fann_error *)ann, FANN_E_CANT_USE_ACTIVATION); \
		result = 0; \
        break; \
}

#endif
