#ifndef FANN_CPP_H_INCLUDED
#define FANN_CPP_H_INCLUDED

/**
 *
 *  @file   fann_cpp.h
 *
 *  @brief  Fast Artificial Neural Network (fann) C++ Wrapper
 *  Copyright (C) 2004-2006 created by freegoldbar (at) yahoo dot com
 *
 *  This wrapper is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU Lesser General Public
 *  License as published by the Free Software Foundation; either
 *  version 2.1 of the License, or (at your option) any later version.
 *
 *  This wrapper is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public
 *  License along with this library; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 */

/**
 *  @mainpage  Fann C++ Wrapper 2.0.0
 *
 *  @section overview  Overview
 *
 *  The C++ Fann Wrapper provides two classes: neural_net,
 *  encapsulates the struct fann pointer with the corresponding C
 *  API functions, and training_data, groups the struct
 *  fann_train_data pointer and associated C API functions.
 *
 *  @section download  Download
 *
 *  The Fann C++ Wrapper includes documentation, a sample and the
 *  Fann C Extensions header.
 *
 *  Download it here: http://www.sourceforge.net/projects/fann
 *
 *  @section notes  Notes and differences from C API
 *
 *  -#   To use the C++ wrapper include doublefann.h, floatfann.h or
 *       fixedfann.h first, then include the header file. To use the optional
 *       extensions include the fann_extensions.h before the fann_cpp.h
 *       see the xor_sample.cpp for an example (get_network_type).
 *  -#   It is a minimal wrapper without templates or exception handling.
 *       Benefits include stricter type checking, simpler memory
 *       management and possibly code completion in program editor.
 *  -#   Method names are the same as the function names in the C
 *       API except the fann_ prefix has been removed. Enums in the
 *       namespace are similarly defined without the FANN_ prefix.
 *  -#   The arguments to the methods are the same as the C API
 *       except that the struct fann *ann/struct fann_train_data *data
 *       arguments are encapsulated so they are not present in the
 *       method signatures.
 *  -#   The various create methods return a boolean set to true to
 *       indicate that the neural network was created, false otherwise.
 *       The same goes for the read_train_from_file method.
 *  -#   The neural network and training data is automatically cleaned
 *       up in the destructors and create/read methods.
 *  -#   To make the destructors virtual define USE_VIRTUAL_DESTRUCTOR
 *       before including the header file.
 *  -#   Additional methods are available on the training_data class to
 *       give access to the underlying training data. They are get_num_data,
 *       get_num_input, get_num_output, get_input, get_output and
 *       set_train_data. Finally fann_duplicate_train_data has been
 *       replaced by a copy constructor.
 *  -#   For further usage documentation on the methods refer to the
 *       C API documentation. Functions marked in the C API as obsolete
 *       are not included.
 *
 *  @section changes  Changes
 *
 *  - Version 2.0.0
 *     - General update to fann C library 2.0.0
 *     - Due to changes in the C API the C++ API is not backward compatible:
 *     - The callback function has changed to include more parameters.
 *     - The save_ex functions are no longer available as similar functionality is
 *     - now present in the Fann C Library - remove the _ex postfix to use them.
 *
 *  - Version 1.2.0
 *     - Changed char pointers to const std::string references
 *     - Added const_casts where the C API required it
 *     - Initialized enums from the C enums instead of numeric constants
 *     - Added a method set_train_data that copies and allocates training
 *     - data in a way that is compatible with the way the C API deallocates
 *     - the data thus making it possible to change training data.
 *     - The get_rprop_increase_factor method did not return its value
 *
 *  - Version 1.0.0
 *     - Initial version
 *
 */

/**
 *  @example xor_sample.cpp
 *  The example illustrates the XOR sample similar to xor_train.c
 */

#include <stdarg.h>
#include <string>

/** The FANN namespace groups C++ Wrapper definitions together */
namespace FANN
{
    /*************************************************************************/

    /* Enum: error_function_enum
	    Error function used during training.
    	
	    ERRORFUNC_LINEAR - Standard linear error function.
	    ERRORFUNC_TANH - Tanh error function, usually better 
		    but can require a lower learning rate. This error function agressively targets outputs that
		    differ much from the desired, while not targetting outputs that only differ a little that much.
		    This activation function is not recommended for cascade training and incremental training.

	    See also:
		    <set_train_error_function>, <get_train_error_function>
    */
    enum error_function_enum {
        /* Standard linear error function */
        ERRORFUNC_LINEAR = FANN_ERRORFUNC_LINEAR,
        /* Tanh error function, usually better but can require a lower learning rate */
        ERRORFUNC_TANH
    };

    /* Enum: stop_function_enum
	    Stop criteria used during training.

	    STOPFUNC_MSE - Stop criteria is Mean Square Error (MSE) value.
	    STOPFUNC_BIT - Stop criteria is number of bits that fail. The number of bits; means the
		    number of output neurons which differ more than the bit fail limit 
		    (see <get_bit_fail_limit>, <set_bit_fail_limit>). 
		    The bits are counted in all of the training data, so this number can be higher than
		    the number of training data.

	    See also:
		    <set_train_stop_function>, <get_train_stop_function>
    */
    enum stop_function_enum
    {
	    STOPFUNC_MSE = FANN_STOPFUNC_MSE,
	    STOPFUNC_BIT
    };

    /* Enum: training_algorithm_enum
	    The Training algorithms used when training on <struct fann_train_data> with functions like
	    <train_on_data> or <train_on_file>. The incremental training looks alters the weights
	    after each time it is presented an input pattern, while batch only alters the weights once after
	    it has been presented to all the patterns.

	    TRAIN_INCREMENTAL -  Standard backpropagation algorithm, where the weights are 
		    updated after each training pattern. This means that the weights are updated many 
		    times during a single epoch. For this reason some problems, will train very fast with 
		    this algorithm, while other more advanced problems will not train very well.
	    TRAIN_BATCH -  Standard backpropagation algorithm, where the weights are updated after 
		    calculating the mean square error for the whole training set. This means that the weights 
		    are only updated once during a epoch. For this reason some problems, will train slower with 
		    this algorithm. But since the mean square error is calculated more correctly than in 
		    incremental training, some problems will reach a better solutions with this algorithm.
	    TRAIN_RPROP - A more advanced batch training algorithm which achieves good results 
		    for many problems. The RPROP training algorithm is adaptive, and does therefore not 
		    use the learning_rate. Some other parameters can however be set to change the way the 
		    RPROP algorithm works, but it is only recommended for users with insight in how the RPROP 
		    training algorithm works. The RPROP training algorithm is described by 
		    [Riedmiller and Braun, 1993], but the actual learning algorithm used here is the 
		    iRPROP- training algorithm which is described by [Igel and Husken, 2000] which 
    	    is an variety of the standard RPROP training algorithm.
	    TRAIN_QUICKPROP - A more advanced batch training algorithm which achieves good results 
		    for many problems. The quickprop training algorithm uses the learning_rate parameter 
		    along with other more advanced parameters, but it is only recommended to change these 
		    advanced parameters, for users with insight in how the quickprop training algorithm works.
		    The quickprop training algorithm is described by [Fahlman, 1988].
    	
	    See also:
		    <set_training_algorithm>, <get_training_algorithm>
    */
    enum training_algorithm_enum {
        /* Standard backpropagation incremental or online training */
        TRAIN_INCREMENTAL = FANN_TRAIN_INCREMENTAL,
        /* Standard backpropagation batch training */
        TRAIN_BATCH,
        /* The iRprop- training algorithm */
        TRAIN_RPROP,
        /* The quickprop training algorithm */
        TRAIN_QUICKPROP
    };

    /* Enums: activation_function_enum
       
	    The activation functions used for the neurons during training. The activation functions
	    can either be defined for a group of neurons by <set_activation_function_hidden> and
	    <set_activation_function_output> or it can be defined for a single neuron by <set_activation_function>.

	    The steepness of an activation function is defined in the same way by 
	    <set_activation_steepness_hidden>, <set_activation_steepness_output> and <set_activation_steepness>.
       
       The functions are described with functions where:
       * x is the input to the activation function,
       * y is the output,
       * s is the steepness and
       * d is the derivation.

       FANN_LINEAR - Linear activation function. 
         * span: -inf < y < inf
	     * y = x*s, d = 1*s
	     * Can NOT be used in fixed point.

       FANN_THRESHOLD - Threshold activation function.
	     * x < 0 -> y = 0, x >= 0 -> y = 1
	     * Can NOT be used during training.

       FANN_THRESHOLD_SYMMETRIC - Threshold activation function.
	     * x < 0 -> y = 0, x >= 0 -> y = 1
	     * Can NOT be used during training.

       FANN_SIGMOID - Sigmoid activation function.
	     * One of the most used activation functions.
	     * span: 0 < y < 1
	     * y = 1/(1 + exp(-2*s*x))
	     * d = 2*s*y*(1 - y)

       FANN_SIGMOID_STEPWISE - Stepwise linear approximation to sigmoid.
	     * Faster than sigmoid but a bit less precise.

       FANN_SIGMOID_SYMMETRIC - Symmetric sigmoid activation function, aka. tanh.
	     * One of the most used activation functions.
	     * span: -1 < y < 1
	     * y = tanh(s*x) = 2/(1 + exp(-2*s*x)) - 1
	     * d = s*(1-(y*y))

       FANN_SIGMOID_SYMMETRIC - Stepwise linear approximation to symmetric sigmoid.
	     * Faster than symmetric sigmoid but a bit less precise.

       FANN_GAUSSIAN - Gaussian activation function.
	     * 0 when x = -inf, 1 when x = 0 and 0 when x = inf
	     * span: 0 < y < 1
	     * y = exp(-x*s*x*s)
	     * d = -2*x*s*y*s

       FANN_GAUSSIAN_SYMMETRIC - Symmetric gaussian activation function.
	     * -1 when x = -inf, 1 when x = 0 and 0 when x = inf
	     * span: -1 < y < 1
	     * y = exp(-x*s*x*s)*2-1
	     * d = -2*x*s*(y+1)*s
    	 
       FANN_ELLIOT - Fast (sigmoid like) activation function defined by David Elliott
	     * span: 0 < y < 1
	     * y = ((x*s) / 2) / (1 + |x*s|) + 0.5
	     * d = s*1/(2*(1+|x*s|)*(1+|x*s|))
    	 
       FANN_ELLIOT_SYMMETRIC - Fast (symmetric sigmoid like) activation function defined by David Elliott
	     * span: -1 < y < 1   
	     * y = (x*s) / (1 + |x*s|)
	     * d = s*1/((1+|x*s|)*(1+|x*s|))

	    FANN_LINEAR_PIECE - Bounded linear activation function.
	     * span: 0 < y < 1
	     * y = x*s, d = 1*s
    	 
	    FANN_LINEAR_PIECE_SYMMETRIC - Bounded Linear activation function.
	     * span: -1 < y < 1
	     * y = x*s, d = 1*s
	
        FANN_SIN_SYMMETRIC - Periodical sinus activation function.
         * span: -1 <= y <= 1
         * y = sin(x*s)
         * d = s*cos(x*s)
         
        FANN_COS_SYMMETRIC - Periodical cosinus activation function.
         * span: -1 <= y <= 1
         * y = cos(x*s)
         * d = s*-sin(x*s)
    	 
	    See also:
		    <set_activation_function_hidden>,
		    <set_activation_function_output>
    */
    enum activation_function_enum {
        LINEAR = FANN_LINEAR,
        THRESHOLD,
        THRESHOLD_SYMMETRIC,
        SIGMOID,
        SIGMOID_STEPWISE,
        SIGMOID_SYMMETRIC,
        SIGMOID_SYMMETRIC_STEPWISE,
        GAUSSIAN,
        GAUSSIAN_SYMMETRIC,
        GAUSSIAN_STEPWISE,
        ELLIOT,
        ELLIOT_SYMMETRIC,
        LINEAR_PIECE,
        LINEAR_PIECE_SYMMETRIC,
	    SIN_SYMMETRIC,
	    COS_SYMMETRIC
    };

    /* Enum: network_type_enum

        Definition of network types used by <get_network_type>

        FANN_NETTYPE_LAYER - Each layer only has connections to the next layer
        FANN_NETTYPE_SHORTCUT - Each layer has connections to all following layers

       See Also:
          <get_network_type>, <fann_get_network_type>

       This enumeration appears in FANN >= 2.1.0
    */
    enum network_type_enum
    {
        LAYER = FANN_NETTYPE_LAYER,
        SHORTCUT
    };

    /* Type: connection

        Describes a connection between two neurons and its weight

        from_neuron - Unique number used to identify source neuron
        to_neuron - Unique number used to identify destination neuron
        weight - The numerical value of the weight

        See Also:
            <get_connection_array>, <set_weight_array>

       This structure appears in FANN >= 2.1.0
    */
    typedef struct fann_connection connection;

    /* Type: callback_type
       This callback function can be called during training when using <train_on_data>, 
       <train_on_file> or <cascadetrain_on_data>.
    	
	    >typedef int (FANN_API * fann_callback_type) (struct fann *ann, struct fann_train_data *train, 
	    >											  unsigned int max_epochs, 
	    >                                             unsigned int epochs_between_reports, 
	    >                                             float desired_error, unsigned int epochs);
    	
	    The callback can be set by using <set_callback> and is very usefull for doing custom 
	    things during training. It is recommended to use this function when implementing custom 
	    training procedures, or when visualizing the training in a GUI etc. The parameters which the
	    callback function takes is the parameters given to the <train_on_data>, plus an epochs
	    parameter which tells how many epochs the training have taken so far.
    	
	    The callback function should return an integer, if the callback function returns -1, the training
	    will terminate.
    	
	    Example of a callback function:
		    >int FANN_API test_callback(struct fann *ann, struct fann_train_data *train,
		    >				            unsigned int max_epochs, unsigned int epochs_between_reports, 
		    >				            float desired_error, unsigned int epochs)
		    >{
		    >	printf("Epochs     %8d. MSE: %.5f. Desired-MSE: %.5f\n", epochs, fann_get_MSE(ann), desired_error);
		    >	return 0;
		    >}
    	
	    See also:
		    <set_callback>, <fann_callback_type>
     */ 
    typedef fann_callback_type callback_type;

    /*************************************************************************/

    /** Encapsulation of training data and related functions */
    class training_data
    {
    public:
        /** Default constructor. Use read_train_from_file to initialize. */
        training_data() : train_data(NULL)
        {
        }

#ifndef FIXEDFANN
        /** Copy constructor. Constructs a copy of the training data.
            Corresponds to the C API duplicate_train_data function. */
        training_data(const training_data &data)
        {
            destroy_train();
            if (data.train_data != NULL)
            {
                train_data = fann_duplicate_train_data(data.train_data);
            }
        }
#endif /* NOT FIXEDFANN */

        /** Reads a file that stores training data, in the format:
        num_train_data num_input num_output\n
        inputdata seperated by space\n
        outputdata seperated by space\n
            ...
        inputdata seperated by space\n
        outputdata seperated by space\n  */
        bool read_train_from_file(const std::string &filename)
        {
            destroy_train();
            train_data = fann_read_train_from_file(filename.c_str());
            return (train_data != NULL);
        }

        /** Destructor. Automatic cleanup. */
#ifdef USE_VIRTUAL_DESTRUCTOR
        virtual
#endif
        ~training_data()
        {
            destroy_train();
        }

        /** Destructs the training data. Called automatically by the destructor. */
        void destroy_train()
        {
            if (train_data != NULL)
            {
                fann_destroy_train(train_data);
                train_data = NULL;
            }
        }

        /** Save the training structure to a file. Returns false if save failed. */
        bool save_train(const std::string &filename)
        {
            if (train_data == NULL)
            {
                return false;
            }
            if (fann_save_train(train_data, filename.c_str()) == -1)
            {
                return false;
            }
            return true;
        }

        /** Saves the training structure to a fixed point data file. Returns false if save failed. */
        bool save_train_to_fixed(const std::string &filename, unsigned int decimal_point)
        {
            if (train_data == NULL)
            {
                return false;
            }
            if (fann_save_train_to_fixed(train_data, filename.c_str(), decimal_point) == -1)
            {
                return false;
            }
            return true;
        }

        /** Shuffles training data, randomizing the order */
        void shuffle_train_data()
        {
            if (train_data != NULL)
            {
                fann_shuffle_train_data(train_data);
            }
        }

        /** Merges training data. */
        void merge_train_data(const training_data &data)
        {
            fann_train_data *new_data = fann_merge_train_data(train_data, data.train_data);
            if (new_data != NULL)
            {
                destroy_train();
                train_data = new_data;
            }
        }

        /*********************************************************************/

        /* Function: length_train_data
           
           Returns the number of training patterns in the <training_data>.

           See also:
           <num_input_train_data>, <num_output_train_data>, <fann_length_train_data>

           This function appears in FANN >= 2.0.0.
         */ 
        unsigned int length_train_data()
        {
            if (train_data == NULL)
            {
                return 0;
            }
            else
            {
                return fann_length_train_data(train_data);
            }
        }

        /* Function: num_input_train_data

           Returns the number of inputs in each of the training patterns in the <training_data>.
           
           See also:
           <num_output_train_data>, <length_train_data>, <fann_num_input_train_data>

           This function appears in FANN >= 2.0.0.
         */ 
        unsigned int num_input_train_data()
        {
            if (train_data == NULL)
            {
                return 0;
            }
            else
            {
                return fann_num_input_train_data(train_data);
            }
        }

        /* Function: num_output_train_data
           
           Returns the number of outputs in each of the training patterns in the <struct fann_train_data>.
           
           See also:
           <num_input_train_data>, <length_train_data>, <fann_num_output_train_data>

           This function appears in FANN >= 2.0.0.
         */ 
        unsigned int num_output_train_data()
        {
            if (train_data == NULL)
            {
                return 0;
            }
            else
            {
                return fann_num_output_train_data(train_data);
            }
        }

        /* Grant access to the encapsulated data since many situations
            and applications creates the data from sources other than files
            or uses the training data for testing and related functions */

        /** Get a pointer to the array of input training data */
        fann_type **get_input()
        {
            if (train_data == NULL)
            {
                return NULL;
            }
            else
            {
                return train_data->input;
            }
        }

        /** Get a pointer to the array of output training data */
        fann_type **get_output()
        {
            if (train_data == NULL)
            {
                return NULL;
            }
            else
            {
                return train_data->output;
            }
        }

        /** Set the training data to the input and output data provided.
            A copy of the data is made so there are no restrictions on the
            allocation of the input/output data and the caller is responsible
            for the deallocation of the data pointed to by input and output. */
        void set_train_data(unsigned int num_data,
            unsigned int num_input, fann_type **input,
            unsigned int num_output, fann_type **output)
        {
            struct fann_train_data *data =
                (struct fann_train_data *)malloc(sizeof(struct fann_train_data));
            data->input = (fann_type **)calloc(num_data, sizeof(fann_type *));
            data->output = (fann_type **)calloc(num_data, sizeof(fann_type *));

            data->num_data = num_data;
            data->num_input = num_input;
            data->num_output = num_output;

        	fann_type *data_input = (fann_type *)calloc(num_input*num_data, sizeof(fann_type));
        	fann_type *data_output = (fann_type *)calloc(num_output*num_data, sizeof(fann_type));

            for (unsigned int i = 0; i < num_data; ++i)
            {
                data->input[i] = data_input;
                data_input += num_input;
                for (unsigned int j = 0; j < num_input; ++j)
                {
                    data->input[i][j] = input[i][j];
                }
                data->output[i] = data_output;
		        data_output += num_output;
                for (unsigned int j = 0; j < num_output; ++j)
                {
                    data->output[i][j] = output[i][j];
                }
            }
            set_train_data(data);
        }

private:
        /** Set the training data to the struct fann_training_data pointer.
            The struct has to be allocated with malloc to be compatible
            with fann_destroy. */
        void set_train_data(struct fann_train_data *data)
        {
            destroy_train();
            train_data = data;
        }

public:
        /*********************************************************************/

        /* Function: create_train_from_callback
           Creates the training data struct from a user supplied function.
           As the training data are numerable (data 1, data 2...), the user must write
           a function that receives the number of the training data set (input,output)
           and returns the set.

           Parameters:
             num_data      - The number of training data
             num_input     - The number of inputs per training data
             num_output    - The number of ouputs per training data
             user_function - The user suplied function

           Parameters for the user function:
             num        - The number of the training data set
             num_input  - The number of inputs per training data
             num_output - The number of ouputs per training data
             input      - The set of inputs
             output     - The set of desired outputs
          
           See also:
             <read_train_from_file>, <train_on_data>, <fann_create_train_from_callback>

            This function appears in FANN >= 2.1.0
        */ 
        void create_train_from_callback(unsigned int num_data,
                                                  unsigned int num_input,
                                                  unsigned int num_output,
                                                  FANN_EXTERNAL void (FANN_API *user_function)( unsigned int,
                                                                         unsigned int,
                                                                         unsigned int,
                                                                         fann_type * ,
                                                                         fann_type * ))
        {
            destroy_train();
            train_data = fann_create_train_from_callback(num_data, num_input, num_output, user_function);
        }

        /* Function: scale_input_train_data
           
           Scales the inputs in the training data to the specified range.

           See also:
   	        <scale_output_train_data>, <scale_train_data>, <fann_scale_input_train_data>

           This function appears in FANN >= 2.0.0.
         */ 
        void scale_input_train_data(fann_type new_min, fann_type new_max)
        {
            if (train_data != NULL)
            {
                fann_scale_input_train_data(train_data, new_min, new_max);
            }
        }

        /* Function: scale_output_train_data
           
           Scales the outputs in the training data to the specified range.

           See also:
   	        <scale_input_train_data>, <scale_train_data>, <fann_scale_output_train_data>

           This function appears in FANN >= 2.0.0.
         */ 
        void scale_output_train_data(fann_type new_min, fann_type new_max)
        {
            if (train_data != NULL)
            {
                fann_scale_output_train_data(train_data, new_min, new_max);
            }
        }

        /* Function: scale_train_data
           
           Scales the inputs and outputs in the training data to the specified range.
           
           See also:
   	        <scale_output_train_data>, <scale_input_train_data>, <fann_scale_train_data>

           This function appears in FANN >= 2.0.0.
         */ 
        void scale_train_data(fann_type new_min, fann_type new_max)
        {
            if (train_data != NULL)
            {
                fann_scale_train_data(train_data, new_min, new_max);
            }
        }

        /* Function: subset_train_data
           
           Changes the training data to a subset, starting at position *pos* 
           and *length* elements forward. Use the copy constructor to work
           on a new copy of the training data.
           
           >FANN::training_data small_data_set = new FANN::training_data(full_data_set);
           >small_data_set.subset_train_data(0, 2); // Only use first two
           
           See also:
   	        <fann_subset_train_data>

           This function appears in FANN >= 2.0.0.
         */
        void subset_train_data(unsigned int pos, unsigned int length)
        {
            if (train_data != NULL)
            {
                struct fann_train_data *temp = fann_subset_train_data(train_data, pos, length);
                destroy_train();
                train_data = temp;
            }
        }

        /*********************************************************************/

    private:
        /** The neural_net class has direct access to the training data */
        friend class neural_net;
        /** Pointer to the encapsulated training data */
        struct fann_train_data* train_data;
    };

    /*************************************************************************/

    /** Encapsulation of a neural network and related functions */
    class neural_net
    {
    public:
        /** Constructor. Use one of the create functions to create a neural network. */
        neural_net() : ann(NULL)
        {
        }

        /** Destructor. Automatic cleanup. */
#ifdef USE_VIRTUAL_DESTRUCTOR
        virtual
#endif
        ~neural_net()
        {
            destroy();
        }

        /** Destructs the entire network. Called automatically by the destructor. */
        void destroy()
        {
            if (ann != NULL)
            {
                fann_destroy(ann);
                ann = NULL;
            }
        }

        /* Function: create_standard
        	
	        Creates a standard fully connected backpropagation neural network.

	        There will be a bias neuron in each layer (except the output layer),
	        and this bias neuron will be connected to all neurons in the next layer.
	        When running the network, the bias nodes always emits 1.
        	
	        Parameters:
		        num_layers - The total number of layers including the input and the output layer.
		        ... - Integer values determining the number of neurons in each layer starting with the 
			        input layer and ending with the output layer.
        			
	        Returns:
		        Boolean true if the network was created, false otherwise.
        			
	        See also:
		        <create_standard_array>, <create_sparse>, <create_shortcut>,
		        <fann_create_standard_array>

	        This function appears in FANN >= 2.0.0.
        */ 
        bool create_standard(unsigned int num_layers, ...)
        {
            va_list layers;
            va_start(layers, num_layers);
            bool status = create_standard_array(num_layers,
                reinterpret_cast<const unsigned int *>(layers));
            va_end(layers);
            return status;
        }

        /* Function: create_standard_array

           Just like <create_standard>, but with an array of layer sizes
           instead of individual parameters.

	        See also:
		        <create_standard>, <create_sparse>, <create_shortcut>,
		        <fann_create_standard>

	        This function appears in FANN >= 2.0.0.
        */ 
        bool create_standard_array(unsigned int num_layers, const unsigned int * layers)
        {
            destroy();
            ann = fann_create_standard_array(num_layers, layers);
            return (ann != NULL);
        }

        /* Function: create_sparse

	        Creates a standard backpropagation neural network, which is not fully connected.

	        Parameters:
		        connection_rate - The connection rate controls how many connections there will be in the
   			        network. If the connection rate is set to 1, the network will be fully
   			        connected, but if it is set to 0.5 only half of the connections will be set.
			        A connection rate of 1 will yield the same result as <fann_create_standard>
		        num_layers - The total number of layers including the input and the output layer.
		        ... - Integer values determining the number of neurons in each layer starting with the 
			        input layer and ending with the output layer.
        			
	        Returns:
		        Boolean true if the network was created, false otherwise.

	        See also:
		        <create_standard>, <create_sparse_array>, <create_shortcut>,
		        <fann_create_sparse>

	        This function appears in FANN >= 2.0.0.
        */
        bool create_sparse(float connection_rate, unsigned int num_layers, ...)
        {
            va_list layers;
            va_start(layers, num_layers);
            bool status = create_sparse_array(connection_rate, num_layers,
                reinterpret_cast<const unsigned int *>(layers));
            va_end(layers);
            return status;
        }

        /* Function: create_sparse_array
           Just like <create_sparse>, but with an array of layer sizes
           instead of individual parameters.

           See <create_sparse> for a description of the parameters.

	        See also:
		        <create_standard>, <create_sparse>, <create_shortcut>,
		        <fann_create_sparse_array>

	        This function appears in FANN >= 2.0.0.
        */
        bool create_sparse_array(float connection_rate,
            unsigned int num_layers, const unsigned int * layers)
        {
            destroy();
            ann = fann_create_sparse_array(connection_rate, num_layers, layers);
            return (ann != NULL);
        }

        /* Function: create_shortcut

	        Creates a standard backpropagation neural network, which is not fully connected and which
	        also has shortcut connections.

 	        Shortcut connections are connections that skip layers. A fully connected network with shortcut 
	        connections, is a network where all neurons are connected to all neurons in later layers. 
	        Including direct connections from the input layer to the output layer.

	        See <create_standard> for a description of the parameters.

	        See also:
		        <create_standard>, <create_sparse>, <create_shortcut_array>,
		        <fann_create_shortcut>

	        This function appears in FANN >= 2.0.0.
        */ 
        bool create_shortcut(unsigned int num_layers, ...)
        {
            va_list layers;
            va_start(layers, num_layers);
            bool status = create_shortcut_array(num_layers,
                reinterpret_cast<const unsigned int *>(layers));
            va_end(layers);
            return status;
        }

        /* Function: create_shortcut_array

           Just like <create_shortcut>, but with an array of layer sizes
           instead of individual parameters.

	        See <create_standard_array> for a description of the parameters.

	        See also:
		        <create_standard>, <create_sparse>, <create_shortcut>,
		        <fann_create_shortcut_array>

	        This function appears in FANN >= 2.0.0.
        */
        bool create_shortcut_array(unsigned int num_layers,
            const unsigned int * layers)
        {
            destroy();
            ann = fann_create_shortcut_array(num_layers, layers);
            return (ann != NULL);
        }

        /** Runs an input through the network, and returns the output.    */
        fann_type* run(fann_type *input)
        {
            if (ann == NULL)
            {
                return NULL;
            }
            return fann_run(ann, input);
        }

        /** Randomize weights (from the beginning the weights are random between -0.1 and 0.1) */
        void randomize_weights(fann_type min_weight, fann_type max_weight)
        {
            if (ann != NULL)
            {
                fann_randomize_weights(ann, min_weight, max_weight);
            }
        }

        /** Initialize the weights using Widrow + Nguyen's algorithm.*/
        void init_weights(const training_data &data)
        {
            if ((ann != NULL) && (data.train_data != NULL))
            {
                fann_init_weights(ann, data.train_data);
            }
        }

        /** Print out which connections there are in the ann */
        void print_connections()
        {
            if (ann != NULL)
            {
                fann_print_connections(ann);
            }
        }

        /** Constructs a backpropagation neural network from a configuration file. */
        bool create_from_file(const std::string &configuration_file)
        {
            destroy();
            ann = fann_create_from_file(configuration_file.c_str());
            return (ann != NULL);
        }

        /** Save the entire network to a configuration file. Return false if save failed. */
        bool save(const std::string &configuration_file)
        {
            if (ann == NULL)
            {
                return false;
            }
            if (fann_save(ann, configuration_file.c_str()) == -1)
            {
                return false;
            }
            return true;
        }

        /** Saves the entire network to a configuration file.
            But it is saved in fixed point format no matter which
            format it is currently in.

            This is usefull for training a network in floating points,
            and then later executing it in fixed point.

            The function returns the bit position of the fix point, which
            can be used to find out how accurate the fixed point network will be.
            A high value indicates high precision, and a low value indicates low
            precision.

            A negative value indicates very low precision, and a very
            strong possibility for overflow.
            (the actual fix point will be set to 0, since a negative
            fix point does not make sence).

            Generally, a fix point lower than 6 is bad, and should be avoided.
            The best way to avoid this, is to have less connections to each neuron,
            or just less neurons in each layer.

            The fixed point use of this network is only intended for use on machines that
            have no floating point processor, like an iPAQ. On normal computers the floating
            point version is actually faster. */
        int save_to_fixed(const std::string &configuration_file)
        {
            int fixpoint = 0;
            if (ann != NULL)
            {
                fixpoint = fann_save_to_fixed(ann, configuration_file.c_str());
            }
            return fixpoint;
        }

#ifndef FIXEDFANN
        /** Train one iteration with a set of inputs, and a set of desired outputs. */
        void train(fann_type *input, fann_type *desired_output)
        {
            if (ann != NULL)
            {
                fann_train(ann, input, desired_output);
            }
        }
#endif /* NOT FIXEDFANN */

        /** Test with a set of inputs, and a set of desired outputs.
        This operation updates the mean square error, but does not
        change the network in any way. */
        fann_type * test(fann_type *input, fann_type *desired_output)
        {
            fann_type * output = NULL;
            if (ann != NULL)
            {
                output = fann_test(ann, input, desired_output);
            }
            return output;
        }

        /** Reads the mean square error from the network. */
        float get_MSE()
        {
            float mse = 0.0f;
            if (ann != NULL)
            {
                mse = fann_get_MSE(ann);
            }
            return mse;
        }

        /** Resets the mean square error from the network. */
        void reset_MSE()
        {
            if (ann != NULL)
            {
                fann_reset_MSE(ann);
            }
        }

#ifndef FIXEDFANN
        /** Train one epoch with a set of training data. */
        float train_epoch(const training_data &data)
        {
            float mse = 0.0f;
            if ((ann != NULL) && (data.train_data != NULL))
            {
                mse = fann_train_epoch(ann, data.train_data);
            }
            return mse;
        }

        /** Test a set of training data and calculate the MSE */
        float test_data(const training_data &data)
        {
            float mse = 0.0f;
            if ((ann != NULL) && (data.train_data != NULL))
            {
                mse = fann_test_data(ann, data.train_data);
            }
            return mse;
        }

        /** Trains on an entire dataset, for a maximum of max_epochs
        epochs or until mean square error is lower than desired_error.
        Reports about the progress is given every
        epochs_between_reports epochs.
        If epochs_between_reports is zero, no reports are given. */
        void train_on_data(const training_data &data, unsigned int max_epochs,
            unsigned int epochs_between_reports, float desired_error)
        {
            if ((ann != NULL) && (data.train_data != NULL))
            {
                fann_train_on_data(ann, data.train_data, max_epochs,
                    epochs_between_reports, desired_error);
            }
        }

        /** Does the same as train_on_data, but reads the data directly from a file. */
        void train_on_file(const std::string &filename, unsigned int max_epochs,
            unsigned int epochs_between_reports, float desired_error)
        {
            if (ann != NULL)
            {
                fann_train_on_file(ann, filename.c_str(),
                    max_epochs, epochs_between_reports, desired_error);
            }
        }
#endif /* NOT FIXEDFANN */

        /* Function: set_callback
           
           Sets the callback function for use during training.
         	
           See <callback_type> for more information about the callback function.
           
           The default callback function simply prints out some status information.

           This function appears in FANN >= 2.0.0.
         */
        void set_callback(callback_type callback)
        {
            if (ann != NULL)
            {
                fann_set_callback(ann, callback);
            }
        }

        /** Prints all of the parameters and options of the ANN */
        void print_parameters()
        {
            if (ann != NULL)
            {
                fann_print_parameters(ann);
            }
        }

        /** Get the training algorithm. */
        training_algorithm_enum get_training_algorithm()
        {
            fann_train_enum training_algorithm = FANN_TRAIN_INCREMENTAL;
            if (ann != NULL)
            {
                training_algorithm = fann_get_training_algorithm(ann);
            }
            return static_cast<training_algorithm_enum>(training_algorithm);
        }

        /** Set the training algorithm. */
        void set_training_algorithm(training_algorithm_enum training_algorithm)
        {
            if (ann != NULL)
            {
                fann_set_training_algorithm(ann,
					static_cast<fann_train_enum>(training_algorithm));
            }
        }

        /** Get the learning rate. */
        float get_learning_rate()
        {
            float learning_rate = 0.0f;
            if (ann != NULL)
            {
                learning_rate = fann_get_learning_rate(ann);
            }
            return learning_rate;
        }

        /** Set the learning rate. */
        void set_learning_rate(float learning_rate)
        {
            if (ann != NULL)
            {
                fann_set_learning_rate(ann, learning_rate);
            }
        }

        /** Get the activation function used in the hidden layers. */
        activation_function_enum get_activation_function_hidden()
        {
            unsigned int activation_function = 0;
            if (ann != NULL)
            {
#if 0 // TODO Reimplement
                activation_function = fann_get_activation_function_hidden(ann);
#endif
            }
            return (activation_function_enum)activation_function;
        }

        /** Set the activation function for the hidden layers. */
        void set_activation_function_hidden(activation_function_enum activation_function)
        {
            if (ann != NULL)
            {
                fann_set_activation_function_hidden(ann,
					static_cast<fann_activationfunc_enum>(activation_function));
            }
        }

        /** Get the activation function used in the output layer. */
        activation_function_enum get_activation_function_output()
        {
            unsigned int activation_function = 0;
            if (ann != NULL)
            {
#if 0 // TODO Reimplement
                activation_function = fann_get_activation_function_output(ann);
#endif
            }
            return (activation_function_enum)activation_function;
        }

        /** Set the activation function for the output layer. */
        void set_activation_function_output(activation_function_enum activation_function)
        {
            if (ann != NULL)
            {
                fann_set_activation_function_output(ann,
					static_cast<fann_activationfunc_enum>(activation_function));
            }
        }

        /** Get the steepness parameter for the sigmoid function used in the hidden layers. */
        fann_type get_activation_steepness_hidden()
        {
            fann_type activation_steepness = 0;
            if (ann != NULL)
            {
#if 0 // TODO Reimplement
                activation_steepness = fann_get_activation_steepness_hidden(ann);
#endif
            }
            return activation_steepness;
        }

        /** Set the steepness of the sigmoid function used in the hidden layers.
        Only usefull if sigmoid function is used in the hidden layers (default 0.5). */
        void set_activation_steepness_hidden(fann_type steepness)
        {
            if (ann != NULL)
            {
                fann_set_activation_steepness_hidden(ann, steepness);
            }
        }

        /** Get the steepness parameter for the sigmoid function used in the output layer. */
        fann_type get_activation_steepness_output()
        {
            fann_type activation_steepness = 0;
            if (ann != NULL)
            {
#if 0 // TODO Reimplement
                activation_steepness = fann_get_activation_steepness_output(ann);
#endif
            }
            return activation_steepness;
        }

        /** Set the steepness of the sigmoid function used in the output layer.
        Only useful if sigmoid function is used in the output layer (default 0.5). */
        void set_activation_steepness_output(fann_type steepness)
        {
            if (ann != NULL)
            {
                fann_set_activation_steepness_output(ann, steepness);
            }
        }

#if 0 //TODO Reimplement
/* Function: fann_get_activation_function

   Get the activation function for neuron number *neuron* in layer number *layer*, 
   counting the input layer as layer 0. 
   
   It is not possible to get activation functions for the neurons in the input layer.
   
   Information about the individual activation functions is available at <fann_activationfunc_enum>.

   Returns:
    The activation function for the neuron or -1 if the neuron is not defined in the neural network.
   
   See also:
   	<fann_set_activation_function_layer>, <fann_set_activation_function_hidden>,
   	<fann_set_activation_function_output>, <fann_set_activation_steepness>,
    <fann_set_activation_function>

   This function appears in FANN >= 2.1.0
 */ 
FANN_EXTERNAL enum fann_activationfunc_enum FANN_API fann_get_activation_function(struct fann *ann,
																int layer,
																int neuron);

/* Function: fann_set_activation_function

   Set the activation function for neuron number *neuron* in layer number *layer*, 
   counting the input layer as layer 0. 
   
   It is not possible to set activation functions for the neurons in the input layer.
   
   When choosing an activation function it is important to note that the activation 
   functions have different range. FANN_SIGMOID is e.g. in the 0 - 1 range while 
   FANN_SIGMOID_SYMMETRIC is in the -1 - 1 range and FANN_LINEAR is unbound.
   
   Information about the individual activation functions is available at <fann_activationfunc_enum>.
   
   The default activation function is FANN_SIGMOID_STEPWISE.
   
   See also:
   	<fann_set_activation_function_layer>, <fann_set_activation_function_hidden>,
   	<fann_set_activation_function_output>, <fann_set_activation_steepness>,
    <fann_get_activation_function>

   This function appears in FANN >= 2.0.0.
 */ 
FANN_EXTERNAL void FANN_API fann_set_activation_function(struct fann *ann,
																enum fann_activationfunc_enum
																activation_function,
																int layer,
																int neuron);

/* Function: fann_set_activation_function_layer

   Set the activation function for all the neurons in the layer number *layer*, 
   counting the input layer as layer 0. 
   
   It is not possible to set activation functions for the neurons in the input layer.

   See also:
   	<fann_set_activation_function>, <fann_set_activation_function_hidden>,
   	<fann_set_activation_function_output>, <fann_set_activation_steepness_layer>

   This function appears in FANN >= 2.0.0.
 */ 
FANN_EXTERNAL void FANN_API fann_set_activation_function_layer(struct fann *ann,
																enum fann_activationfunc_enum
																activation_function,
																int layer);

/* Function: fann_set_activation_function_hidden

   Set the activation function for all of the hidden layers.

   See also:
   	<fann_set_activation_function>, <fann_set_activation_function_layer>,
   	<fann_set_activation_function_output>, <fann_set_activation_steepness_hidden>

   This function appears in FANN >= 1.0.0.
 */ 
FANN_EXTERNAL void FANN_API fann_set_activation_function_hidden(struct fann *ann,
																enum fann_activationfunc_enum
																activation_function);


/* Function: fann_set_activation_function_output

   Set the activation function for the output layer.

   See also:
   	<fann_set_activation_function>, <fann_set_activation_function_layer>,
   	<fann_set_activation_function_hidden>, <fann_set_activation_steepness_output>

   This function appears in FANN >= 1.0.0.
 */ 
FANN_EXTERNAL void FANN_API fann_set_activation_function_output(struct fann *ann,
																enum fann_activationfunc_enum
																activation_function);

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
   	<fann_set_activation_steepness_output>, <fann_set_activation_function>,
    <fann_set_activation_steepness>

   This function appears in FANN >= 2.1.0
 */ 
FANN_EXTERNAL fann_type FANN_API fann_get_activation_steepness(struct fann *ann,
																int layer,
																int neuron);

/* Function: fann_set_activation_steepness

   Set the activation steepness for neuron number *neuron* in layer number *layer*, 
   counting the input layer as layer 0. 
   
   It is not possible to set activation steepness for the neurons in the input layer.
   
   The steepness of an activation function says something about how fast the activation function 
   goes from the minimum to the maximum. A high value for the activation function will also
   give a more agressive training.
   
   When training neural networks where the output values should be at the extremes (usually 0 and 1, 
   depending on the activation function), a steep activation function can be used (e.g. 1.0).
   
   The default activation steepness is 0.5.
   
   See also:
   	<fann_set_activation_steepness_layer>, <fann_set_activation_steepness_hidden>,
   	<fann_set_activation_steepness_output>, <fann_set_activation_function>,
    <fann_get_activation_steepness>

   This function appears in FANN >= 2.0.0.
 */ 
FANN_EXTERNAL void FANN_API fann_set_activation_steepness(struct fann *ann,
																fann_type steepness,
																int layer,
																int neuron);

/* Function: fann_set_activation_steepness_layer

   Set the activation steepness all of the neurons in layer number *layer*, 
   counting the input layer as layer 0. 
   
   It is not possible to set activation steepness for the neurons in the input layer.
   
   See also:
   	<fann_set_activation_steepness>, <fann_set_activation_steepness_hidden>,
   	<fann_set_activation_steepness_output>, <fann_set_activation_function_layer>

   This function appears in FANN >= 2.0.0.
 */ 
FANN_EXTERNAL void FANN_API fann_set_activation_steepness_layer(struct fann *ann,
																fann_type steepness,
																int layer);

/* Function: fann_set_activation_steepness_hidden

   Set the steepness of the activation steepness in all of the hidden layers.

   See also:
   	<fann_set_activation_steepness>, <fann_set_activation_steepness_layer>,
   	<fann_set_activation_steepness_output>, <fann_set_activation_function_hidden>

   This function appears in FANN >= 1.2.0.
 */ 
FANN_EXTERNAL void FANN_API fann_set_activation_steepness_hidden(struct fann *ann,
																 fann_type steepness);


/* Function: fann_set_activation_steepness_output

   Set the steepness of the activation steepness in the output layer.

   See also:
   	<fann_set_activation_steepness>, <fann_set_activation_steepness_layer>,
   	<fann_set_activation_steepness_hidden>, <fann_set_activation_function_output>

   This function appears in FANN >= 1.2.0.
 */ 
FANN_EXTERNAL void FANN_API fann_set_activation_steepness_output(struct fann *ann,
																 fann_type steepness);

#endif

        /** Get the error function used during training. */
        error_function_enum get_train_error_function()
        {
            fann_errorfunc_enum train_error_function = FANN_ERRORFUNC_LINEAR;
            if (ann != NULL)
            {
                train_error_function = fann_get_train_error_function(ann);
            }
            return static_cast<error_function_enum>(train_error_function);
        }

        /** Get the error function used during training. (default FANN::ERRORFUNC_TANH) */
        void set_train_error_function(error_function_enum train_error_function)
        {
            if (ann != NULL)
            {
                fann_set_train_error_function(ann,
					static_cast<fann_errorfunc_enum>(train_error_function));
            }
        }

        /** Decay is used to make the weights do not go so high (default -0.0001). */
        float get_quickprop_decay()
        {
            float quickprop_decay = 0.0f;
            if (ann != NULL)
            {
                quickprop_decay = fann_get_quickprop_decay(ann);
            }
            return quickprop_decay;
        }

        /** Decay is used to make the weights do not go so high (default -0.0001). */
        void set_quickprop_decay(float quickprop_decay)
        {
            if (ann != NULL)
            {
                fann_set_quickprop_decay(ann, quickprop_decay);
            }
        }

        /** Mu is a factor used to increase and decrease the stepsize (default 1.75). */
        float get_quickprop_mu()
        {
            float quickprop_mu = 0.0f;
            if (ann != NULL)
            {
                quickprop_mu = fann_get_quickprop_mu(ann);
            }
            return quickprop_mu;
        }

        /** Mu is a factor used to increase and decrease the stepsize (default 1.75). */
        void set_quickprop_mu(float quickprop_mu)
        {
            if (ann != NULL)
            {
                fann_set_quickprop_mu(ann, quickprop_mu);
            }
        }

        /** Tells how much the stepsize should increase during learning (default 1.2). */
        float get_rprop_increase_factor()
        {
            float factor = 0.0f;
            if (ann != NULL)
            {
                factor = fann_get_rprop_increase_factor(ann);
            }
            return factor;
        }

        /** Tells how much the stepsize should increase during learning (default 1.2). */
        void set_rprop_increase_factor(float rprop_increase_factor)
        {
            if (ann != NULL)
            {
                fann_set_rprop_increase_factor(ann, rprop_increase_factor);
            }
        }

        /** Tells how much the stepsize should decrease during learning (default 0.5). */
        float get_rprop_decrease_factor()
        {
            float factor = 0.0f;
            if (ann != NULL)
            {
                factor = fann_get_rprop_decrease_factor(ann);
            }
            return factor;
        }

        /** Tells how much the stepsize should decrease during learning (default 0.5). */
        void set_rprop_decrease_factor(float rprop_decrease_factor)
        {
            if (ann != NULL)
            {
                fann_set_rprop_decrease_factor(ann, rprop_decrease_factor);
            }
        }

        /** The minimum stepsize (default 0.0). */
        float get_rprop_delta_min()
        {
            float delta = 0.0f;
            if (ann != NULL)
            {
                delta = fann_get_rprop_delta_min(ann);
            }
            return delta;
        }

        /** The minimum stepsize (default 0.0). */
        void set_rprop_delta_min(float rprop_delta_min)
        {
            if (ann != NULL)
            {
                fann_set_rprop_delta_min(ann, rprop_delta_min);
            }
        }

        /** The maximum stepsize (default 50.0). */
        float get_rprop_delta_max()
        {
            float delta = 0.0f;
            if (ann != NULL)
            {
                delta = fann_get_rprop_delta_max(ann);
            }
            return delta;
        }

        /** The maximum stepsize (default 50.0). */
        void set_rprop_delta_max(float rprop_delta_max)
        {
            if (ann != NULL)
            {
                fann_set_rprop_delta_max(ann, rprop_delta_max);
            }
        }

        /** Get the number of input neurons. */
        unsigned int get_num_input()
        {
            unsigned int num_input = 0;
            if (ann != NULL)
            {
                num_input = fann_get_num_input(ann);
            }
            return num_input;
        }

        /** Get the number of output neurons. */
        unsigned int get_num_output()
        {
            unsigned int num_output = 0;
            if (ann != NULL)
            {
                num_output = fann_get_num_output(ann);
            }
            return num_output;
        }

        /** Get the total number of neurons in the entire network.    */
        unsigned int get_total_neurons()
        {
            if (ann == NULL)
            {
                return 0;
            }
            return fann_get_total_neurons(ann);
        }

        /** Get the total number of connections in the entire network. */
        unsigned int get_total_connections()
        {
            if (ann == NULL)
            {
                return 0;
            }
            return fann_get_total_connections(ann);
        }

#ifdef FIXEDFANN
        /** Returns the position of the decimal point. */
        unsigned int get_decimal_point()
        {
            if (ann == NULL)
            {
                return 0;
            }
            return fann_get_decimal_point(ann);
        }

        /** Returns the multiplier that fix point data is multiplied with. */
        unsigned int get_multiplier()
        {
            if (ann == NULL)
            {
                return 0;
            }
            return fann_get_multiplier(ann);
        }
#endif /* FIXEDFANN */

        /*********************************************************************/

        /** Change where errors are logged to */
        static void set_error_log(struct fann_error *errdat, FILE *log_file)
        {
            fann_set_error_log(errdat, log_file);
        }

        /** Returns the last error number */
        static unsigned int get_errno(struct fann_error *errdat)
        {
            return fann_get_errno(errdat);
        }

        /** Resets the last error number */
        static void reset_errno(struct fann_error *errdat)
        {
            fann_reset_errno(errdat);
        }

        /** Resets the last error string */
        static void reset_errstr(struct fann_error *errdat)
        {
            fann_reset_errstr(errdat);
        }

        /** Returns the last errstr.
        This function calls fann_reset_errno and fann_reset_errstr */
        static std::string get_errstr(struct fann_error *errdat)
        {
            return std::string(fann_get_errstr(errdat));
        }

        /** Prints the last error to stderr */
        static void print_error(struct fann_error *errdat)
        {
            fann_print_error(errdat);
        }

        /*********************************************************************/

        /** Get the type of network as defined in fann_network_types */
        network_type_enum get_network_type()
        {
            fann_nettype_enum network_type = FANN_NETTYPE_LAYER;
            if (ann != NULL)
            {
                network_type = fann_get_network_type(ann);
            }
            return static_cast<network_type_enum>(network_type);
        }

        /** Get the connection rate used when the network was created */
        float get_connection_rate()
        {
            if (ann == NULL)
            {
                return 0;
            }
            return fann_get_connection_rate(ann);
        }

        /** Get the number of layers in the network */
        unsigned int get_num_layers()
        {
            if (ann == NULL)
            {
                return 0;
            }
            return fann_get_num_layers(ann);
        }

        /** Get the number of neurons in each layer in the network.
            Bias is not included so the layers match the fann_create functions
            The layers array must be preallocated to at least
            sizeof(unsigned int) * fann_num_layers() long. */
        void get_layer_array(unsigned int *layers)
        {
            if (ann != NULL)
            {
                fann_get_layer_array(ann, layers);
            }
        }

        /** Get the number of bias in each layer in the network.
            The bias array must be preallocated to at least
            sizeof(unsigned int) * fann_num_layers() long. */
        void get_bias_array(unsigned int *bias)
        {
            if (ann != NULL)
            {
                fann_get_bias_array(ann, bias);
            }
        }

        /** Get the connections in the network.
            The connections array must be preallocated to at least
            sizeof(connection) * fann_get_total_connections() long. */
        void get_connection_array(connection *connections)
        {
            if (ann != NULL)
            {
                fann_get_connection_array(ann, connections);
            }
        }

        /** Set connections in the network.
            Only the weights can be changed, connections and weights are ignored
            if they do not already exist in the network.
            The array must have sizeof(connection) * num_connections size. */
        void set_weight_array(connection *connections, unsigned int num_connections)
        {
            if (ann != NULL)
            {
                fann_set_weight_array(ann, connections, num_connections);
            }
        }

        /** Set a connection in the network.
            Only the weight can be changed. The connection/weight is
            ignored if it does not already exist in the network. */
        void set_weight(unsigned int from_neuron, unsigned int to_neuron, fann_type weight)
        {
            if (ann != NULL)
            {
                fann_set_weight(ann, from_neuron, to_neuron, weight);
            }
        }

        /*********************************************************************/

        /* Function: get_learning_momentum

           Get the learning momentum.
           
           The learning momentum can be used to speed up FANN::TRAIN_INCREMENTAL training.
           A too high momentum will however not benefit training. Setting momentum to 0 will
           be the same as not using the momentum parameter. The recommended value of this parameter
           is between 0.0 and 1.0.

           The default momentum is 0.
           
           See also:
           <set_learning_momentum>, <set_training_algorithm>

           This function appears in FANN >= 2.0.0.   	
         */ 
        float get_learning_momentum()
        {
            float learning_momentum = 0.0f;
            if (ann != NULL)
            {
                learning_momentum = fann_get_learning_momentum(ann);
            }
            return learning_momentum;
        }

        /* Function: set_learning_momentum

           Set the learning momentum.

           More info available in <get_learning_momentum>

           This function appears in FANN >= 2.0.0.   	
         */ 
        void set_learning_momentum(float learning_momentum)
        {
            if (ann != NULL)
            {
                fann_set_learning_momentum(ann, learning_momentum);
            }
        }

        /* Function: get_train_stop_function

           Returns the the stop function used during training.
           
           The stop function is described further in <stop_function_enum>
           
           The default stop function is FANN::STOPFUNC_MSE
           
           See also:
   	        <get_train_stop_function>, <get_bit_fail_limit>
              
           This function appears in FANN >= 2.0.0.
         */ 
        stop_function_enum get_train_stop_function()
        {
            enum fann_stopfunc_enum stopfunc = FANN_STOPFUNC_MSE;
            if (ann != NULL)
            {
                stopfunc = fann_get_train_stop_function(ann);
            }
            return static_cast<stop_function_enum>(stopfunc);
        }

        /* Function: set_train_stop_function

           Set the stop function used during training.

           The stop function is described further in <stop_function_enum>
           
           See also:
   	        <get_train_stop_function>
              
           This function appears in FANN >= 2.0.0.
         */ 
        void set_train_stop_function(stop_function_enum train_stop_function)
        {
            if (ann != NULL)
            {
                fann_set_train_stop_function(ann,
                    static_cast<enum fann_stopfunc_enum>(train_stop_function));
            }
        }

        /* Function: get_bit_fail_limit

           Returns the bit fail limit used during training.
           
           The bit fail limit is used during training when the <stop_function_enum> is set to FANN_STOPFUNC_BIT.

           The limit is the maximum accepted difference between the desired output and the actual output during
           training. Each output that diverges more than this limit is counted as an error bit.
           This difference is divided by two when dealing with symmetric activation functions,
           so that symmetric and not symmetric activation functions can use the same limit.
           
           The default bit fail limit is 0.35.
           
           See also:
   	        <set_bit_fail_limit>
           
           This function appears in FANN >= 2.0.0.
         */ 
        fann_type get_bit_fail_limit()
        {
            fann_type bit_fail_limit = 0.0f;

            if (ann != NULL)
            {
                bit_fail_limit = fann_get_bit_fail_limit(ann);
            }
            return bit_fail_limit;
        }

        /* Function: set_bit_fail_limit

           Set the bit fail limit used during training.
          
           See also:
   	        <get_bit_fail_limit>
           
           This function appears in FANN >= 2.0.0.
         */
        void set_bit_fail_limit(fann_type bit_fail_limit)
        {
            if (ann != NULL)
            {
                fann_set_bit_fail_limit(ann, bit_fail_limit);
            }
        }

        /* Function: get_bit_fail
        	
	        The number of fail bits; means the number of output neurons which differ more 
	        than the bit fail limit (see <get_bit_fail_limit>, <set_bit_fail_limit>). 
	        The bits are counted in all of the training data, so this number can be higher than
	        the number of training data.
        	
	        This value is reset by <reset_MSE> and updated by all the same functions which also
	        updates the MSE value (e.g. <test_data>, <train_epoch>)
        	
	        See also:
		        <stop_function_enum>, <get_MSE>

	        This function appears in FANN >= 2.0.0
        */
        unsigned int get_bit_fail()
        {
            unsigned int bit_fail = 0;
            if (ann != NULL)
            {
                bit_fail = fann_get_bit_fail(ann);
            }
            return bit_fail;
        }

        /*********************************************************************/

        /* Function: cascadetrain_on_data

           Trains on an entire dataset, for a period of time using the Cascade2 training algorithm.
           This algorithm adds neurons to the neural network while training, which means that it
           needs to start with an ANN without any hidden layers. The neural network should also use
           shortcut connections, so <create_shortcut> should be used to create the ANN like this:
           >net.create_shortcut(2, train_data.num_input_train_data(), train_data.num_input_train_data());
           
           This training uses the parameters set using the set_cascade_..., but it also uses another
           training algorithm as it's internal training algorithm. This algorithm can be set to either
           FANN::TRAIN_RPROP or FANN::TRAIN_QUICKPROP by <set_training_algorithm>, and the parameters 
           set for these training algorithms will also affect the cascade training.
           
           Parameters:
   		        data - The data, which should be used during training
   		        max_neuron - The maximum number of neurons to be added to neural network
   		        neurons_between_reports - The number of neurons between printing a status report to stdout.
   			        A value of zero means no reports should be printed.
   		        desired_error - The desired <fann_get_MSE> or <fann_get_bit_fail>, depending on which stop function
   			        is chosen by <fann_set_train_stop_function>.

	        Instead of printing out reports every neurons_between_reports, a callback function can be called 
	        (see <set_callback>).
        	
	        See also:
		        <train_on_data>, <cascadetrain_on_file>, <fann_cascadetrain_on_data>

	        This function appears in FANN >= 2.0.0. 
        */
        void cascadetrain_on_data(const training_data &data, unsigned int max_neurons,
            unsigned int neurons_between_reports, float desired_error)
        {
            if ((ann != NULL) && (data.train_data != NULL))
            {
                fann_cascadetrain_on_data(ann, data.train_data, max_neurons,
                    neurons_between_reports, desired_error);
            }
        }

        /* Function: cascadetrain_on_file
           
           Does the same as <cascadetrain_on_data>, but reads the training data directly from a file.
           
           See also:
   		        <fann_cascadetrain_on_data>, <fann_cascadetrain_on_file>

	        This function appears in FANN >= 2.0.0.
        */ 
        void cascadetrain_on_file(const std::string &filename, unsigned int max_neurons,
            unsigned int neurons_between_reports, float desired_error)
        {
            if (ann != NULL)
            {
                fann_cascadetrain_on_file(ann, filename.c_str(),
                    max_neurons, neurons_between_reports, desired_error);
            }
        }

        /* Function: get_cascade_output_change_fraction

           The cascade output change fraction is a number between 0 and 1 determining how large a fraction
           the <get_MSE> value should change within <get_cascade_output_stagnation_epochs> during
           training of the output connections, in order for the training not to stagnate. If the training 
           stagnates, the training of the output connections will be ended and new candidates will be prepared.
           
           This means:
           If the MSE does not change by a fraction of <get_cascade_output_change_fraction> during a 
           period of <get_cascade_output_stagnation_epochs>, the training of the output connections
           is stopped because the training has stagnated.

           If the cascade output change fraction is low, the output connections will be trained more and if the
           fraction is high they will be trained less.
           
           The default cascade output change fraction is 0.01, which is equalent to a 1% change in MSE.

           See also:
   		        <set_cascade_output_change_fraction>, <get_MSE>,
                <get_cascade_output_stagnation_epochs>, <fann_get_cascade_output_change_fraction>

	        This function appears in FANN >= 2.0.0.
         */
        float get_cascade_output_change_fraction()
        {
            float change_fraction = 0.0f;
            if (ann != NULL)
            {
                change_fraction = fann_get_cascade_output_change_fraction(ann);
            }
            return change_fraction;
        }

        /* Function: set_cascade_output_change_fraction

           Sets the cascade output change fraction.
           
           See also:
   		        <get_cascade_output_change_fraction>, <fann_set_cascade_output_change_fraction>

	        This function appears in FANN >= 2.0.0.
         */
        void set_cascade_output_change_fraction(float cascade_output_change_fraction)
        {
            if (ann != NULL)
            {
                fann_set_cascade_output_change_fraction(ann, cascade_output_change_fraction);
            }
        }

        /* Function: get_cascade_output_stagnation_epochs

           The number of cascade output stagnation epochs determines the number of epochs training is allowed to
           continue without changing the MSE by a fraction of <get_cascade_output_change_fraction>.
           
           See more info about this parameter in <get_cascade_output_change_fraction>.
           
           The default number of cascade output stagnation epochs is 12.

           See also:
   		        <set_cascade_output_stagnation_epochs>, <get_cascade_output_change_fraction>,
                <fann_get_cascade_output_stagnation_epochs>

	        This function appears in FANN >= 2.0.0.
         */
        unsigned int get_cascade_output_stagnation_epochs()
        {
            unsigned int stagnation_epochs = 0;
            if (ann != NULL)
            {
                stagnation_epochs = fann_get_cascade_output_stagnation_epochs(ann);
            }
            return stagnation_epochs;
        }

        /* Function: set_cascade_output_stagnation_epochs

           Sets the number of cascade output stagnation epochs.
           
           See also:
   		        <get_cascade_output_stagnation_epochs>, <fann_set_cascade_output_stagnation_epochs>

	        This function appears in FANN >= 2.0.0.
         */
        void set_cascade_output_stagnation_epochs(unsigned int cascade_output_stagnation_epochs)
        {
            if (ann != NULL)
            {
                fann_set_cascade_output_stagnation_epochs(ann, cascade_output_stagnation_epochs);
            }
        }

        /* Function: get_cascade_candidate_change_fraction

           The cascade candidate change fraction is a number between 0 and 1 determining how large a fraction
           the <get_MSE> value should change within <get_cascade_candidate_stagnation_epochs> during
           training of the candidate neurons, in order for the training not to stagnate. If the training 
           stagnates, the training of the candidate neurons will be ended and the best candidate will be selected.
           
           This means:
           If the MSE does not change by a fraction of <get_cascade_candidate_change_fraction> during a 
           period of <get_cascade_candidate_stagnation_epochs>, the training of the candidate neurons
           is stopped because the training has stagnated.

           If the cascade candidate change fraction is low, the candidate neurons will be trained more and if the
           fraction is high they will be trained less.
           
           The default cascade candidate change fraction is 0.01, which is equalent to a 1% change in MSE.

           See also:
   		        <set_cascade_candidate_change_fraction>, <get_MSE>,
                <get_cascade_candidate_stagnation_epochs>, <fann_get_cascade_candidate_change_fraction>

	        This function appears in FANN >= 2.0.0.
         */
        float get_cascade_candidate_change_fraction()
        {
            float change_fraction = 0.0f;
            if (ann != NULL)
            {
                change_fraction = fann_get_cascade_candidate_change_fraction(ann);
            }
            return change_fraction;
        }

        /* Function: set_cascade_candidate_change_fraction

           Sets the cascade candidate change fraction.
           
           See also:
   		        <get_cascade_candidate_change_fraction>,
                <fann_set_cascade_candidate_change_fraction>

	        This function appears in FANN >= 2.0.0.
         */
        void set_cascade_candidate_change_fraction(float cascade_candidate_change_fraction)
        {
            if (ann != NULL)
            {
                fann_set_cascade_candidate_change_fraction(ann, cascade_candidate_change_fraction);
            }
        }

        /* Function: get_cascade_candidate_stagnation_epochs

           The number of cascade candidate stagnation epochs determines the number of epochs training is allowed to
           continue without changing the MSE by a fraction of <get_cascade_candidate_change_fraction>.
           
           See more info about this parameter in <get_cascade_candidate_change_fraction>.

           The default number of cascade candidate stagnation epochs is 12.

           See also:
   		        <set_cascade_candidate_stagnation_epochs>, <get_cascade_candidate_change_fraction>,
                <fann_get_cascade_candidate_stagnation_epochs>

	        This function appears in FANN >= 2.0.0.
         */
        unsigned int get_cascade_candidate_stagnation_epochs()
        {
            unsigned int stagnation_epochs = 0;
            if (ann != NULL)
            {
                stagnation_epochs = fann_get_cascade_candidate_stagnation_epochs(ann);
            }
            return stagnation_epochs;
        }

        /* Function: set_cascade_candidate_stagnation_epochs

           Sets the number of cascade candidate stagnation epochs.
           
           See also:
   		        <get_cascade_candidate_stagnation_epochs>,
                <fann_set_cascade_candidate_stagnation_epochs>

	        This function appears in FANN >= 2.0.0.
         */
        void set_cascade_candidate_stagnation_epochs(unsigned int cascade_candidate_stagnation_epochs)
        {
            if (ann != NULL)
            {
                fann_set_cascade_candidate_stagnation_epochs(ann, cascade_candidate_stagnation_epochs);
            }
        }

        /* Function: get_cascade_weight_multiplier

           The weight multiplier is a parameter which is used to multiply the weights from the candidate neuron
           before adding the neuron to the neural network. This parameter is usually between 0 and 1, and is used
           to make the training a bit less aggressive.

           The default weight multiplier is 0.4

           See also:
   		        <set_cascade_weight_multiplier>, <fann_get_cascade_weight_multiplier>

	        This function appears in FANN >= 2.0.0.
         */
        fann_type get_cascade_weight_multiplier()
        {
            fann_type weight_multiplier = 0;
            if (ann != NULL)
            {
                weight_multiplier = fann_get_cascade_weight_multiplier(ann);
            }
            return weight_multiplier;
        }

        /* Function: set_cascade_weight_multiplier
           
           Sets the weight multiplier.
           
           See also:
   		        <get_cascade_weight_multiplier>, <fann_set_cascade_weight_multiplier>

	        This function appears in FANN >= 2.0.0.
         */
        void set_cascade_weight_multiplier(fann_type cascade_weight_multiplier)
        {
            if (ann != NULL)
            {
                fann_set_cascade_weight_multiplier(ann, cascade_weight_multiplier);
            }
        }

        /* Function: get_cascade_candidate_limit

           The candidate limit is a limit for how much the candidate neuron may be trained.
           The limit is a limit on the proportion between the MSE and candidate score.
           
           Set this to a lower value to avoid overfitting and to a higher if overfitting is
           not a problem.
           
           The default candidate limit is 1000.0

           See also:
   		        <set_cascade_candidate_limit>, <fann_get_cascade_candidate_limit>

	        This function appears in FANN >= 2.0.0.
         */
        fann_type get_cascade_candidate_limit()
        {
            fann_type candidate_limit = 0;
            if (ann != NULL)
            {
                candidate_limit = fann_get_cascade_candidate_limit(ann);
            }
            return candidate_limit;
        }

        /* Function: set_cascade_candidate_limit

           Sets the candidate limit.
          
           See also:
   		        <get_cascade_candidate_limit>, <fann_set_cascade_candidate_limit>

	        This function appears in FANN >= 2.0.0.
         */
        void set_cascade_candidate_limit(fann_type cascade_candidate_limit)
        {
            if (ann != NULL)
            {
                fann_set_cascade_candidate_limit(ann, cascade_candidate_limit);
            }
        }

        /* Function: get_cascade_max_out_epochs

           The maximum out epochs determines the maximum number of epochs the output connections
           may be trained after adding a new candidate neuron.
           
           The default max out epochs is 150

           See also:
   		        <set_cascade_max_out_epochs>, <fann_get_cascade_max_out_epochs>

	        This function appears in FANN >= 2.0.0.
         */
        unsigned int get_cascade_max_out_epochs()
        {
            unsigned int max_out_epochs = 0;
            if (ann != NULL)
            {
                max_out_epochs = fann_get_cascade_max_out_epochs(ann);
            }
            return max_out_epochs;
        }

        /* Function: set_cascade_max_out_epochs

           Sets the maximum out epochs.

           See also:
   		        <get_cascade_max_out_epochs>, <fann_set_cascade_max_out_epochs>

	        This function appears in FANN >= 2.0.0.
         */
        void set_cascade_max_out_epochs(unsigned int cascade_max_out_epochs)
        {
            if (ann != NULL)
            {
                fann_set_cascade_max_out_epochs(ann, cascade_max_out_epochs);
            }
        }

        /* Function: get_cascade_max_cand_epochs

           The maximum candidate epochs determines the maximum number of epochs the input 
           connections to the candidates may be trained before adding a new candidate neuron.
           
           The default max candidate epochs is 150

           See also:
   		        <set_cascade_max_cand_epochs>, <fann_get_cascade_max_cand_epochs>

	        This function appears in FANN >= 2.0.0.
         */
        unsigned int get_cascade_max_cand_epochs()
        {
            unsigned int max_cand_epochs = 0;
            if (ann != NULL)
            {
                max_cand_epochs = fann_get_cascade_max_cand_epochs(ann);
            }
            return max_cand_epochs;
        }

        /* Function: set_cascade_max_cand_epochs

           Sets the max candidate epochs.
          
           See also:
   		        <get_cascade_max_cand_epochs>, <fann_set_cascade_max_cand_epochs>

	        This function appears in FANN >= 2.0.0.
         */
        void set_cascade_max_cand_epochs(unsigned int cascade_max_cand_epochs)
        {
            if (ann != NULL)
            {
                fann_set_cascade_max_cand_epochs(ann, cascade_max_cand_epochs);
            }
        }

        /* Function: get_cascade_num_candidates

           The number of candidates used during training (calculated by multiplying <get_cascade_activation_functions_count>,
           <get_cascade_activation_steepnesses_count> and <get_cascade_num_candidate_groups>). 

           The actual candidates is defined by the <get_cascade_activation_functions> and 
           <get_cascade_activation_steepnesses> arrays. These arrays define the activation functions 
           and activation steepnesses used for the candidate neurons. If there are 2 activation functions
           in the activation function array and 3 steepnesses in the steepness array, then there will be 
           2x3=6 different candidates which will be trained. These 6 different candidates can be copied into
           several candidate groups, where the only difference between these groups is the initial weights.
           If the number of groups is set to 2, then the number of candidate neurons will be 2x3x2=12. The 
           number of candidate groups is defined by <set_cascade_num_candidate_groups>.

           The default number of candidates is 6x4x2 = 48

           See also:
   		        <get_cascade_activation_functions>, <get_cascade_activation_functions_count>, 
   		        <get_cascade_activation_steepnesses>, <get_cascade_activation_steepnesses_count>,
   		        <get_cascade_num_candidate_groups>, <fann_get_cascade_num_candidates>

	        This function appears in FANN >= 2.0.0.
         */ 
        unsigned int get_cascade_num_candidates()
        {
            unsigned int num_candidates = 0;
            if (ann != NULL)
            {
                num_candidates = fann_get_cascade_num_candidates(ann);
            }
            return num_candidates;
        }

        /* Function: get_cascade_activation_functions_count

           The number of activation functions in the <get_cascade_activation_functions> array.

           The default number of activation functions is 6.

           See also:
   		        <get_cascade_activation_functions>, <set_cascade_activation_functions>,
                <fann_get_cascade_activation_functions_count>

	        This function appears in FANN >= 2.0.0.
         */
        unsigned int get_cascade_activation_functions_count()
        {
            unsigned int activation_functions_count = 0;
            if (ann != NULL)
            {
                activation_functions_count = fann_get_cascade_activation_functions_count(ann);
            }
            return activation_functions_count;
        }

        /* Function: get_cascade_activation_functions

           The cascade activation functions array is an array of the different activation functions used by
           the candidates. 
           
           See <get_cascade_num_candidates> for a description of which candidate neurons will be 
           generated by this array.
           
           See also:
   		        <get_cascade_activation_functions_count>, <set_cascade_activation_functions>,
   		        <activationfunc_enum>

	        This function appears in FANN >= 2.0.0.
         */
        activation_function_enum * get_cascade_activation_functions()
        {
            enum fann_activationfunc_enum *activation_functions = NULL;
            if (ann != NULL)
            {
                activation_functions = fann_get_cascade_activation_functions(ann);
            }
            return reinterpret_cast<activation_function_enum *>(activation_functions);
        }

        /* Function: set_cascade_activation_functions

           Sets the array of cascade candidate activation functions. The array must be just as long
           as defined by the count.

           See <get_cascade_num_candidates> for a description of which candidate neurons will be 
           generated by this array.

           See also:
   		        <get_cascade_activation_steepnesses_count>, <get_cascade_activation_steepnesses>,
                <fann_set_cascade_activation_functions>

	        This function appears in FANN >= 2.0.0.
         */
        void set_cascade_activation_functions(activation_function_enum *cascade_activation_functions,
            unsigned int cascade_activation_functions_count)
        {
            if (ann != NULL)
            {
                fann_set_cascade_activation_functions(ann,
                    reinterpret_cast<enum fann_activationfunc_enum *>(cascade_activation_functions),
                    cascade_activation_functions_count);
            }
        }

        /* Function: get_cascade_activation_steepnesses_count

           The number of activation steepnesses in the <get_cascade_activation_functions> array.

           The default number of activation steepnesses is 4.

           See also:
   		        <get_cascade_activation_steepnesses>, <set_cascade_activation_functions>,
                <fann_get_cascade_activation_steepnesses_count>

	        This function appears in FANN >= 2.0.0.
         */
        unsigned int get_cascade_activation_steepnesses_count()
        {
            unsigned int activation_steepness_count = 0;
            if (ann != NULL)
            {
                activation_steepness_count = fann_get_cascade_activation_steepnesses_count(ann);
            }
            return activation_steepness_count;
        }

        /* Function: get_cascade_activation_steepnesses

           The cascade activation steepnesses array is an array of the different activation functions used by
           the candidates.

           See <get_cascade_num_candidates> for a description of which candidate neurons will be 
           generated by this array.

           The default activation steepnesses is {0.25, 0.50, 0.75, 1.00}

           See also:
   		        <set_cascade_activation_steepnesses>, <get_cascade_activation_steepnesses_count>,
                <fann_get_cascade_activation_steepnesses>

	        This function appears in FANN >= 2.0.0.
         */
        fann_type *get_cascade_activation_steepnesses()
        {
            fann_type *activation_steepnesses = NULL;
            if (ann != NULL)
            {
                activation_steepnesses = fann_get_cascade_activation_steepnesses(ann);
            }
            return activation_steepnesses;
        }																

        /* Function: set_cascade_activation_steepnesses

           Sets the array of cascade candidate activation steepnesses. The array must be just as long
           as defined by the count.

           See <get_cascade_num_candidates> for a description of which candidate neurons will be 
           generated by this array.

           See also:
   		        <get_cascade_activation_steepnesses>, <get_cascade_activation_steepnesses_count>,
                <fann_set_cascade_activation_steepnesses>

	        This function appears in FANN >= 2.0.0.
         */
        void set_cascade_activation_steepnesses(fann_type *cascade_activation_steepnesses,
            unsigned int cascade_activation_steepnesses_count)
        {
            if (ann != NULL)
            {
                fann_set_cascade_activation_steepnesses(ann,
                    cascade_activation_steepnesses, cascade_activation_steepnesses_count);
            }
        }

        /* Function: get_cascade_num_candidate_groups

           The number of candidate groups is the number of groups of identical candidates which will be used
           during training.
           
           This number can be used to have more candidates without having to define new parameters for the candidates.
           
           See <get_cascade_num_candidates> for a description of which candidate neurons will be 
           generated by this parameter.
           
           The default number of candidate groups is 2

           See also:
   		        <set_cascade_num_candidate_groups>, <fann_get_cascade_num_candidate_groups>

	        This function appears in FANN >= 2.0.0.
         */
        unsigned int get_cascade_num_candidate_groups()
        {
            unsigned int num_candidate_groups = 0;
            if (ann != NULL)
            {
                num_candidate_groups = fann_get_cascade_num_candidate_groups(ann);
            }
            return num_candidate_groups;
        }

        /* Function: set_cascade_num_candidate_groups

           Sets the number of candidate groups.

           See also:
   		        <get_cascade_num_candidate_groups>, <fann_set_cascade_num_candidate_groups>

	        This function appears in FANN >= 2.0.0.
         */
        void set_cascade_num_candidate_groups(unsigned int cascade_num_candidate_groups)
        {
            if (ann != NULL)
            {
                fann_set_cascade_num_candidate_groups(ann, cascade_num_candidate_groups);
            }
        }

        /*********************************************************************/

    private:
        /** Pointer the encapsulated fann neural net structure */
        struct fann *ann;
    };

    /*************************************************************************/
};

#endif /* FANN_CPP_H_INCLUDED */
