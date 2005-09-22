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

#ifndef __fann_io_h__
#define __fann_io_h__
	
/* Section: FANN File Input/Output */	

/* Group: File Input and Output */	
	
/* Function: fann_create_from_file
   Constructs a backpropagation neural network from a configuration file.
 */ 
FANN_EXTERNAL struct fann *FANN_API fann_create_from_file(const char *configuration_file);


/* Function: fann_save
   Save the entire network to a configuration file.
 */ 
FANN_EXTERNAL void FANN_API fann_save(struct fann *ann, const char *configuration_file);


/* Function: fann_save_to_fixed
   Saves the entire network to a configuration file.
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
   point version is actually faster.
*/ 
FANN_EXTERNAL int FANN_API fann_save_to_fixed(struct fann *ann, const char *configuration_file);
	
#endif
