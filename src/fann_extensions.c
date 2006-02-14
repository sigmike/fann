/**
 *
 *  @file   fann_extensions.c
 *
 *  @brief  Fast Artificial Neural Network (fann) C Extensions 2.0.0
 *  Copyright (C) 2004 created by freegoldbar (at) yahoo dot com
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

#include "fann_extensions.h"

/**
 *  @mainpage  Fann C Extensions 2.0.0.
 *
 *  @section alert  Important information
 *
 *  The Fann C Extensions are subject to change at any time and only work
 *  with Fann version 2.0.0. The extensions will change as similar
 *  functionality becomes available in the Fann Library. The changes may
 *  require code changes.
 *
 *  @section overview  Overview
 *  Fann C Extensions gives access to neural network layout, connections,
 *  and weights.
 *  The save_ex functions are no longer available as similar functionality is
 *  now present in the Fann C Library - remove the _ex postfix to use them.
 *
 *  @section download  Download
 *
 *  The Fann C Extensions header and documentation is included in the Fann C++ Wrapper.
 *
 *  Download it here: http://www.sourceforge.net/projects/fann
 *
 *  @section usage  Usage
 *  See the Examples section for a small example. The fannKernel and C++ Wrapper
 *  uses the Fann C Extensions.
 *  Source is available at http://www.sourceforge.net/projects/fann
 *
 */


/** Get the type of network as defined in fann_network_types */
FANN_EXTERNAL unsigned int FANN_API fann_get_network_type(struct fann *ann)
{
    /* Currently two types: LAYER = 0, SHORTCUT = 1 */
    /* Enum network_types must be set to match the return values  */
    return ann->shortcut_connections;
}

/** Get the connection rate used when the network was created */
FANN_EXTERNAL float FANN_API fann_get_connection_rate(struct fann *ann)
{
    return ann->connection_rate;
}

/*****************************************************************************/

/** Get the number of layers in the network */
FANN_EXTERNAL unsigned int FANN_API fann_get_num_layers(struct fann *ann)
{
    return ann->last_layer - ann->first_layer;
}

/** Get the number of neurons in each layer in the network.
    Bias is not included so the layers match the fann_create functions
    The layers array must be preallocated to at least
    sizeof(unsigned int) * fann_num_layers() long. */
FANN_EXTERNAL void FANN_API fann_get_layer_array(struct fann *ann, unsigned int *layers)
{
    struct fann_layer *layer_it;

    for (layer_it = ann->first_layer; layer_it != ann->last_layer; layer_it++) {
        unsigned int count = layer_it->last_neuron - layer_it->first_neuron;
        /* Remove the bias from the count of neurons. */
        switch (fann_get_network_type(ann)) {
            case FANN_LAYER: {
                --count;
                break;
            }
            case FANN_SHORTCUT: {
                --count;
                break;
            }
            default: {
                /* Unknown network type, assume no bias present  */
                break;
            }
        }
        *layers++ = count;
    }
}

/** Get the number of bias in each layer in the network.
    The bias array must be preallocated to at least
    sizeof(unsigned int) * fann_num_layers() long. */
FANN_EXTERNAL void FANN_API fann_get_bias_array(struct fann *ann, unsigned int *bias)
{
    struct fann_layer *layer_it;

    for (layer_it = ann->first_layer; layer_it != ann->last_layer; ++layer_it, ++bias) {
        switch (fann_get_network_type(ann)) {
            case FANN_LAYER: {
                /* Report one bias in each layer except the last */
                if (layer_it != ann->last_layer-1)
                    *bias = 1;
                else
                    *bias = 0;
                break;
            }
            case FANN_SHORTCUT: {
                /* Bias for current shortcut net is the same as for a layered net */
                /* TODO When shortcut has one bias in first layer change to:
                    if (layer_it == ann->first_layer) */
                if (layer_it != ann->last_layer-1)
                    *bias = 1;
                else
                    *bias = 0;
                break;
            }
            default: {
                /* Unknown network type, assume no bias present  */
                *bias = 0;
                break;
            }
        }
    }
}

/** Get the connections in the network.
    The connections array must be preallocated to at least
    sizeof(struct fann_connection) * fann_get_total_connections() long. */
FANN_EXTERNAL void FANN_API fann_get_connection_array(struct fann *ann, struct fann_connection *connections)
{
    struct fann_neuron *first_neuron;
    struct fann_layer *layer_it;
    struct fann_neuron *neuron_it;
    unsigned int index;
    unsigned int source_index;
    unsigned int destination_index;

    first_neuron = ann->first_layer->first_neuron;

    source_index = 0;
    destination_index = 0;
    
    /* The following assumes that the last unused bias has no connections */

    /* for each layer */
    for(layer_it = ann->first_layer; layer_it != ann->last_layer; layer_it++){
        /* for each neuron */
        for(neuron_it = layer_it->first_neuron; neuron_it != layer_it->last_neuron; neuron_it++){
            /* for each connection */
            for (index = neuron_it->first_con; index < neuron_it->last_con; index++){
                /* Assign the source, destination and weight */
                connections->from_neuron = ann->connections[source_index] - first_neuron;
                connections->to_neuron = destination_index;
                connections->weight = ann->weights[source_index];

                connections++;
                source_index++;
            }
            destination_index++;
        }
    }
}

/** Set weights in the network.
    Only the weights can be changed, connections and weights are ignored
    if they do not already exist in the network.
    The array must have sizeof(struct fann_connection) * num_connections size. */
FANN_EXTERNAL void FANN_API fann_set_weight_array(struct fann *ann,
    struct fann_connection *connections, unsigned int num_connections)
{
    unsigned int index;

    for (index = 0; index < num_connections; index++) {
        fann_set_weight(ann, connections[index].from_neuron,
            connections[index].to_neuron, connections[index].weight);
    }
}

/** Set a weight in the network.
    Only the weights can be changed. The connection/weight is
    ignored if it does not already exist in the network. */
FANN_EXTERNAL void FANN_API fann_set_weight(struct fann *ann,
    unsigned int from_neuron, unsigned int to_neuron, fann_type weight)
{
    struct fann_neuron *first_neuron;
    struct fann_layer *layer_it;
    struct fann_neuron *neuron_it;
    unsigned int index;
    unsigned int source_index;
    unsigned int destination_index;

    first_neuron = ann->first_layer->first_neuron;

    source_index = 0;
    destination_index = 0;

    /* Find the connection, simple brute force search through the network
       for one or more connections that match to minimize datastructure dependencies.
       Nothing is done if the connection does not already exist in the network. */

    /* for each layer */
    for(layer_it = ann->first_layer; layer_it != ann->last_layer; layer_it++){
        /* for each neuron */
        for(neuron_it = layer_it->first_neuron; neuron_it != layer_it->last_neuron; neuron_it++){
            /* for each connection */
            for (index = neuron_it->first_con; index < neuron_it->last_con; index++){
                /* If the source and destination neurons match, assign the weight */
                if (((int)from_neuron == ann->connections[source_index] - first_neuron) &&
                    (to_neuron == destination_index))
                {
                    ann->weights[source_index] = weight;
                }
                source_index++;
            }
            destination_index++;
        }
    }
}

/*****************************************************************************/

/* TODO: Remove the extern declaration if the following functions are included in fann_train.c */
extern FANN_EXTERNAL struct fann_neuron* FANN_API fann_get_neuron(struct fann *ann, unsigned int layer, int neuron);

/** Get the activation function for a neuron in a layer, counting the input layer as layer 0.
    @return The activation function for the neuron or -1 if the neuron is not defined in the neural network. */   
FANN_EXTERNAL enum fann_activationfunc_enum FANN_API
    fann_get_activation_function(struct fann *ann, int layer, int neuron)
{
	struct fann_neuron* neuron_it = fann_get_neuron(ann, layer, neuron);
	if (neuron_it == NULL)
		return -1;
    else
	    return neuron_it->activation_function;
}

/** Get the activation steepness for a neuron in a layer, counting the input layer as layer 0.
    @return The activation function for the neuron or -1 if the neuron is not defined in the neural network. */   
FANN_EXTERNAL fann_type FANN_API
    fann_get_activation_steepness(struct fann *ann, int layer, int neuron)
{
	struct fann_neuron* neuron_it = fann_get_neuron(ann, layer, neuron);
	if(neuron_it == NULL)
		return -1;
    else
        return neuron_it->activation_steepness;
}

/*****************************************************************************/
