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
   NOT be included directly. If included directly it will react as if
   floatfann.h was included.
*/ 

/* Package: FANN Create/Destroy */
	
#ifndef FANN_INCLUDE
/* just to allow for inclusion of fann.h in normal stuations where only floats are needed */ 
#ifdef FIXEDFANN
#include "fixedfann.h"
#else
#include "floatfann.h"
#endif	/* FIXEDFANN  */
	
#else
	
/* COMPAT_TIME REPLACEMENT */ 
#ifndef _WIN32
#include <sys/time.h>
#else	/* _WIN32 */
#if !defined(_MSC_EXTENSIONS) &&
	 !defined(_INC_WINDOWS)  extern unsigned long __stdcall GetTickCount(void);


#else	/* _MSC_EXTENSIONS */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif	/* _MSC_EXTENSIONS */
#endif	/* _WIN32 */
	
#include "fann_data.h"
#include "fann_internal.h"
#include "fann_activation.h"
#include "fann_errno.h"
	
#ifndef __fann_h__
#define __fann_h__
	
#ifdef __cplusplus
extern "C"
{
	
#ifndef __cplusplus
} /* to fool automatic indention engines */ 
#endif
#endif	/* __cplusplus */
 
#ifndef NULL
#define NULL 0
#endif	/* NULL */
 
/* ----- Macros used to define DLL external entrypoints ----- */ 
/*
 DLL Export, import and calling convention for Windows.
 Only defined for Microsoft VC++ FANN_EXTERNAL indicates
 that a function will be exported/imported from a dll
 FANN_API ensures that the DLL calling convention
 will be used for  a function regardless of the calling convention
 used when compiling.

 For a function to be exported from a DLL its prototype and
 declaration must be like this:
    FANN_EXTERNAL void FANN_API function(char *argument)

 The following ifdef block is a way of creating macros which
 make exporting from a DLL simple. All files within a DLL are
 compiled with the FANN_DLL_EXPORTS symbol defined on the
 command line. This symbol should not be defined on any project
 that uses this DLL. This way any other project whose source
 files include this file see FANN_EXTERNAL functions as being imported
 from a DLL, whereas a DLL sees symbols defined with this
 macro as being exported which makes calls more efficient.
 The __stdcall calling convention is used for functions in a
 windows DLL.

 The callback functions for fann_train_on_data_callback and
 fann_train_on_file_callback must be declared as FANN_API
 so the DLL and the application program both use the same
 calling convention. The callback functions must of this form:
     int FANN_API user_callback(unsigned int epochs, float error)
*/ 
 
/*
 The following sets the default for MSVC++ 2003 or later to use
 the fann dll's. To use a lib or fixedfann.c, floatfann.c or doublefann.c
 with those compilers FANN_NO_DLL has to be defined before
 including the fann headers.
 The default for previous MSVC compilers such as VC++ 6 is not
 to use dll's. To use dll's FANN_USE_DLL has to be defined before
 including the fann headers.
*/ 
#if (_MSC_VER > 1300)
#ifndef FANN_NO_DLL
#define FANN_USE_DLL
#endif	/* FANN_USE_LIB */
#endif	/* _MSC_VER */
#if defined(_MSC_VER) && (defined(FANN_USE_DLL) || defined(FANN_DLL_EXPORTS))
#ifdef FANN_DLL_EXPORTS
#define FANN_EXTERNAL __declspec(dllexport)
#else							/*  */
#define FANN_EXTERNAL __declspec(dllimport)
#endif	/* FANN_DLL_EXPORTS*/
#define FANN_API __stdcall
#else							/*  */
#define FANN_EXTERNAL
#define FANN_API
#endif	/* _MSC_VER */
/* ----- End of macros used to define DLL external entrypoints ----- */ 

#include "fann_train.h"
#include "fann_train_data.h"
#include "fann_cascade.h"
#include "fann_error.h"
#include "fann_io.h"

/* ----- Implemented in fann.c Creation, running and destruction of ANNs ----- */ 
 
/* Function: fann_create
   Constructs a backpropagation neural network, from an connection rate,
   a learning rate, the number of layers and the number of neurons in each
   of the layers.

   The connection rate controls how many connections there will be in the
   network. If the connection rate is set to 1, the network will be fully
   connected, but if it is set to 0.5 only half of the connections will be set.

   There will be a bias neuron in each layer (except the output layer),
   and this bias neuron will be connected to all neurons in the next layer.
   When running the network, the bias nodes always emits 1
*/ 
FANN_EXTERNAL struct fann *FANN_API fann_create(float connection_rate, float learning_rate, 
												  /* the number of layers, including the input and output layer */ 
												  unsigned int num_layers, 
												  /* the number of neurons in each of the layers, starting with
												   * the input layer and ending with the output layer */ 
												  ...);

/* Function: fann_create_array
   Just like fann_create, but with an array of layer sizes
   instead of individual parameters.
*/ 
FANN_EXTERNAL struct fann *FANN_API fann_create_array(float connection_rate, float learning_rate,
													  unsigned int num_layers,
													  unsigned int *layers);

/* Function: fann_create_shortcut

   create a fully connected neural network with shortcut connections.
 */ 
FANN_EXTERNAL struct fann *FANN_API fann_create_shortcut(float learning_rate,
														 unsigned int num_layers, 
														 ...);

/* the number of neurons in each of the layers, starting with the input layer and ending with the output layer */
	
/* Function: fann_create_shortcut_array
   create a neural network with shortcut connections.
 */ 
FANN_EXTERNAL struct fann *FANN_API fann_create_shortcut_array(float learning_rate,
															   unsigned int num_layers,
															   unsigned int *layers);

/* Function: fann_run
   Runs a input through the network, and returns the output.
 */ 
FANN_EXTERNAL fann_type * FANN_API fann_run(struct fann *ann, fann_type * input);

/* Function: fann_destroy
   Destructs the entire network.
   Be sure to call this function after finished using the network.
 */ 
FANN_EXTERNAL void FANN_API fann_destroy(struct fann *ann);

/* Function: fann_randomize_weights
   Randomize weights (from the beginning the weights are random between -0.1 and 0.1)
 */ 
FANN_EXTERNAL void FANN_API fann_randomize_weights(struct fann *ann, fann_type min_weight,
												   fann_type max_weight);

/* Function: fann_init_weights
   Initialize the weights using Widrow + Nguyen's algorithm.
*/ 
FANN_EXTERNAL void FANN_API fann_init_weights(struct fann *ann, struct fann_train_data *train_data);

/* Function: fann_print_connections
   print out which connections there are in the ann */ 
FANN_EXTERNAL void FANN_API fann_print_connections(struct fann *ann);

/* Group: Parameters */
/* Function: fann_print_parameters

   Prints all of the parameters and options of the ANN */ 
FANN_EXTERNAL void FANN_API fann_print_parameters(struct fann *ann);


/* Function: fann_get_num_input

   Get the number of input neurons.
 */ 
FANN_EXTERNAL unsigned int FANN_API fann_get_num_input(struct fann *ann);


/* Function: fann_get_num_output

   Get the number of output neurons.
 */ 
FANN_EXTERNAL unsigned int FANN_API fann_get_num_output(struct fann *ann);


/* Function: fann_get_total_neurons

   Get the total number of neurons in the entire network.
 */ 
FANN_EXTERNAL unsigned int FANN_API fann_get_total_neurons(struct fann *ann);


/* Function: fann_get_total_connections

   Get the total number of connections in the entire network.
 */ 
FANN_EXTERNAL unsigned int FANN_API fann_get_total_connections(struct fann *ann);

#ifdef FIXEDFANN
	
/* Function: fann_get_decimal_point

   returns the position of the decimal point.
 */ 
FANN_EXTERNAL unsigned int FANN_API fann_get_decimal_point(struct fann *ann);


/* Function: fann_get_multiplier

   returns the multiplier that fix point data is multiplied with.
 */ 
FANN_EXTERNAL unsigned int FANN_API fann_get_multiplier(struct fann *ann);

#endif	/* FIXEDFANN */

#ifdef __cplusplus
#ifndef __cplusplus
/* to fool automatic indention engines */ 
{
	
#endif
} 
#endif	/* __cplusplus */
	
#endif	/* __fann_h__ */
	
#endif /* NOT FANN_INCLUDE */
