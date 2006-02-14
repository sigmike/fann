#ifndef FANN_EXTENSIONS_H_INCLUDED
#define FANN_EXTENSIONS_H_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/*
 *  Fast Artificial Neural Network (fann) C Extensions
 *  Copyright (C) 2004-2006 created by freegoldbar (at) yahoo dot com
 *
 *  This wrapper is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU Lesser General Public
 *  License as published by the Free Software Foundation; either
 *  version 2.1 of the License, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *  See the GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public
 *  License along with this library; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 */

/*
 *  Title:  Fann C Extensions 2.0.0
 *
 *  Important information:
 *  The Fann C Extensions are subject to change at any time and only work
 *  with Fann version 2.0.0. The extensions will change as similar
 *  functionality becomes available in the Fann Library. The changes may
 *  require code changes.
 *
 *  Overview:
 *  Fann C Extensions gives access to neural network layout, connections,
 *  and weights.
 *  The save_ex functions are no longer available as similar functionality is
 *  now present in the Fann C Library - remove the _ex postfix to use them.
 *
 *  Download:
 *  The Fann C Extensions header and documentation is included in the Fann C++ Wrapper.
 *  Download it here: http://www.sourceforge.net/projects/fann
 *
 *  Example
 *  See the end of the fann_extensions.h file for a small example.
 *  The C++ Wrapper and fannKernel uses the Fann C Extensions.
 */

/*****************************************************************************/
/* Section: C Extension Data and Types */


/* Enum: fann_network_types

    Definition of network types used by <fann_get_network_type>

    FANN_LAYER - Each layer only has connections to the next layer
    FANN_SHORTCUT - Each layer has connections to all following layers

   See Also:
      <fann_get_network_type>
*/
enum fann_network_types
{
    FANN_LAYER = 0, /* Each layer only has connections to the next layer */
    FANN_SHORTCUT /* Each layer has connections to all following layers */
};

/* Constant: FANN_NETWORK_TYPE_NAMES
   
   Constant array consisting of the names for the network types, so that the name of an
   network type can be received by:
   (code)
   char *network_type_name = FANN_NETWORK_TYPE_NAMES[fann_get_network_type(ann)];
   (end)

   See Also:
      <fann_get_network_type>
*/
static char const *const FANN_NETWORK_TYPE_NAMES[] = {
	"FANN_LAYER",
	"FANN_SHORTCUT"
};

/* Type: fann_connection

    Describes a connection between two neurons and its weight

    from_neuron - Unique number used to identify source neuron
    to_neuron - Unique number used to identify destination neuron
    weight - The numerical value of the weight

    See Also:
        <fann_get_connection_array>, <fann_set_weight_array>
*/
struct fann_connection
{
    /* Unique number used to identify source neuron */
    unsigned int from_neuron;
    /* Unique number used to identify destination neuron */
    unsigned int to_neuron;
    /* The numerical value of the weight */
    fann_type weight;
};

/*****************************************************************************/
/* Section: C Extension Functions */


/* Function: fann_get_network_type

    Get the type of neural network it was created as.

    Parameters:
		ann - A previously created neural network structure of
            type <struct fann> pointer.

	Returns:
        The neural network type from enum <fann_network_types>

    See Also:
        <fann_network_types>
*/
enum fann_network_types fann_get_network_type(struct fann *ann);

/* Function: fann_get_connection_rate

    Get the connection rate used when the network was created

    Parameters:
		ann - A previously created neural network structure of
            type <struct fann> pointer.

	Returns:
        The connection rate
*/
float fann_get_connection_rate(struct fann *ann);

/*****************************************************************************/

/* Function: fann_get_num_layers

    Get the number of layers in the network

    Parameters:
		ann - A previously created neural network structure of
            type <struct fann> pointer.
			
	Returns:
		The number of layers in the neural network
			
	Example:
		> // Obtain the number of layers in a neural network
		> struct fann *ann = fann_create_standard(4, 2, 8, 9, 1);
        > unsigned int num_layers = fann_get_num_layers(ann);
*/
unsigned int fann_get_num_layers(struct fann *ann);

/*Function: fann_get_layer_array

    Get the number of neurons in each layer in the network.

    Bias is not included so the layers match the fann_create functions.

    Parameters:
		ann - A previously created neural network structure of
            type <struct fann> pointer.

    The layers array must be preallocated to at least
    sizeof(unsigned int) * fann_num_layers() long.
*/
void fann_get_layer_array(struct fann *ann, unsigned int *layers);

/* Function: fann_get_bias_array

    Get the number of bias in each layer in the network.

    Parameters:
		ann - A previously created neural network structure of
            type <struct fann> pointer.

    The bias array must be preallocated to at least
    sizeof(unsigned int) * fann_num_layers() long.
*/
void fann_get_bias_array(struct fann *ann, unsigned int *bias);

/* Function: fann_get_connection_array

    Get the connections in the network.

    Parameters:
		ann - A previously created neural network structure of
            type <struct fann> pointer.

    The connections array must be preallocated to at least
    sizeof(struct fann_connection) * fann_get_total_connections() long.
*/
void fann_get_connection_array(struct fann *ann, struct fann_connection *connections);

/* Function: fann_set_weight_array

    Set connections in the network.

    Parameters:
		ann - A previously created neural network structure of
            type <struct fann> pointer.

    Only the weights can be changed, connections and weights are ignored
    if they do not already exist in the network.

    The array must have sizeof(struct fann_connection) * num_connections size.
*/
void fann_set_weight_array(struct fann *ann,
    struct fann_connection *connections, unsigned int num_connections);

/* Function: fann_set_weight

    Set a connection in the network.

    Parameters:
		ann - A previously created neural network structure of
            type <struct fann> pointer.

    Only the weights can be changed. The connection/weight is
    ignored if it does not already exist in the network.
*/
void fann_set_weight(struct fann *ann,
    unsigned int from_neuron, unsigned int to_neuron, fann_type weight);

/* Function: fann_get_activation_function

   Get the activation function for neuron number *neuron* in layer number *layer*, 
   counting the input layer as layer 0. 
   
   It is not possible to get activation functions for the neurons in the input layer.
   
   Information about the individual activation functions is available at <fann_activationfunc_enum>.

   Returns:
    The activation function for the neuron or -1 if the neuron is not defined in the neural network.
   
   See also:
   	<fann_set_activation_function_layer>, <fann_set_activation_function_hidden>,
   	<fann_set_activation_function_output>, <fann_set_activation_steepness>

   This function appears in FANN >= 2.0.1.
 */ 
FANN_EXTERNAL enum fann_activationfunc_enum FANN_API fann_get_activation_function(struct fann *ann,
																int layer,
																int neuron);

/* Function: fann_get_activation_steepness

   Get the activation steepness for neuron number *neuron* in layer number *layer*, 
   counting the input layer as layer 0. 
   
   It is not possible to get activation steepness for the neurons in the input layer.
   
   The steepness of an activation function says something about how fast the activation function 
   goes from the minimum to the maximum. A high value for the activation function will also
   give a more agressive training.
   
   When training neural networks where the output values should be at the extremes (usually 0 and 1, 
   depending on the activation function), a steep activation function can be used (e.g. 1.0).
   
   The default activation steepness is 0.5.
   
   Returns:
    The activation steepness for the neuron or -1 if the neuron is not defined in the neural network.
   
   See also:
   	<fann_set_activation_steepness_layer>, <fann_set_activation_steepness_hidden>,
   	<fann_set_activation_steepness_output>, <fann_set_activation_function>

   This function appears in FANN >= 2.0.1.
 */ 
FANN_EXTERNAL fann_type FANN_API fann_get_activation_steepness(struct fann *ann,
																int layer,
																int neuron);

/*****************************************************************************/

/* #define SIMPLE_TEST_CASE */
#ifdef SIMPLE_TEST_CASE
/**  @example sample.c */
/**
 *  Simple test case showing sample usage of how to get
 *  layer/bias arrays and how to get/set weights in the
 *  connection array. @see Examples
 */
void simple_test_extensions(struct fann *ann)
{
    unsigned int num_layers;
    unsigned int *layers;
    unsigned int num_connections;
    struct fann_connection *connections;

    num_layers = fann_get_num_layers(ann);
    layers = (unsigned int *)malloc(sizeof(unsigned int) * num_layers);
    bias = (unsigned int *)malloc(sizeof(unsigned int) * num_layers);
    if (layers != NULL)
    {
        fann_get_layer_array(ann, layers);
        fann_get_bias_array(ann, bias);
        free(bias);
        free(layers);
    }

    num_connections = fann_get_total_connections(ann);
    connections = (struct fann_connection *)
        malloc(sizeof(struct fann_connection) * num_connections);
    if (connections != NULL)
    {
        fann_get_connection_array(ann, connections);
        fann_set_connection_array(ann, connections, num_connections);
        free(connections);
    }
}
#endif /* SIMPLE_TEST_CASE */

/*****************************************************************************/

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* FANN_EXTENSIONS_H_INCLUDED */

/*****************************************************************************/
