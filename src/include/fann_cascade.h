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

#ifndef __fann_cascade_h__
#define __fann_cascade_h__

/* Section: FANN Cascade Training
   test info about cascade training
*/

/* Group: Cascade Training */

/* Function: fann_cascadetrain_on_data
*/
FANN_EXTERNAL void fann_cascadetrain_on_data(struct fann *ann,
													  struct fann_train_data *data,
													  unsigned int max_out_epochs,
													  unsigned int neurons_between_reports,
													  float desired_error);

/* Group: Parameters */
													  
/* Function: fann_get_cascade_num_candidates

   The number of candidates (calculated from cascade_activation_functions_count,
   cascade_activation_steepnesses_count and cascade_num_candidate_groups). 
 */ 
FANN_EXTERNAL unsigned int FANN_API fann_get_cascade_num_candidates(struct fann *ann);

/* Function: fann_get_cascade_change_fraction
 */
FANN_EXTERNAL float FANN_API fann_get_cascade_change_fraction(struct fann *ann);


/* Function: fann_set_cascade_change_fraction
 */
FANN_EXTERNAL void FANN_API fann_set_cascade_change_fraction(struct fann *ann, 
															 float cascade_change_fraction);

/* Function: fann_get_cascade_stagnation_epochs
 */
FANN_EXTERNAL unsigned int FANN_API fann_get_cascade_stagnation_epochs(struct fann *ann);


/* Function: fann_set_cascade_stagnation_epochs
 */
FANN_EXTERNAL void FANN_API fann_set_cascade_stagnation_epochs(struct fann *ann, 
															 unsigned int cascade_stagnation_epochs);


/* Function: fann_get_cascade_num_candidate_groups
 */
FANN_EXTERNAL unsigned int FANN_API fann_get_cascade_num_candidate_groups(struct fann *ann);


/* Function: fann_set_cascade_num_candidate_groups
 */
FANN_EXTERNAL void FANN_API fann_set_cascade_num_candidate_groups(struct fann *ann, 
															 unsigned int cascade_num_candidate_groups);


/* Function: fann_get_cascade_weight_multiplier
 */
FANN_EXTERNAL fann_type FANN_API fann_get_cascade_weight_multiplier(struct fann *ann);


/* Function: fann_set_cascade_weight_multiplier
 */
FANN_EXTERNAL void FANN_API fann_set_cascade_weight_multiplier(struct fann *ann, 
															 fann_type cascade_weight_multiplier);


/* Function: fann_get_cascade_candidate_limit
 */
FANN_EXTERNAL fann_type FANN_API fann_get_cascade_candidate_limit(struct fann *ann);


/* Function: fann_set_cascade_candidate_limit
 */
FANN_EXTERNAL void FANN_API fann_set_cascade_candidate_limit(struct fann *ann, 
															 fann_type cascade_candidate_limit);


/* Function: fann_get_cascade_max_out_epochs
 */
FANN_EXTERNAL unsigned int FANN_API fann_get_cascade_max_out_epochs(struct fann *ann);


/* Function: fann_set_cascade_max_out_epochs
 */
FANN_EXTERNAL void FANN_API fann_set_cascade_max_out_epochs(struct fann *ann, 
															 unsigned int cascade_max_out_epochs);


/* Function: fann_get_cascade_max_cand_epochs
 */
FANN_EXTERNAL unsigned int FANN_API fann_get_cascade_max_cand_epochs(struct fann *ann);


/* Function: fann_set_cascade_max_cand_epochs
 */
FANN_EXTERNAL void FANN_API fann_set_cascade_max_cand_epochs(struct fann *ann, 
															 unsigned int cascade_max_cand_epochs);


/* Function: fann_get_cascade_activation_functions_count
 */
FANN_EXTERNAL unsigned int FANN_API fann_get_cascade_activation_functions_count(struct fann *ann);


/* Function: fann_get_cascade_activation_functions
 */
FANN_EXTERNAL enum fann_activationfunc_enum * FANN_API fann_get_cascade_activation_functions(
															struct fann *ann);


/* Function: fann_set_cascade_activation_functions
 */
FANN_EXTERNAL void fann_set_cascade_activation_functions(struct fann *ann,
														 enum fann_activationfunc_enum *
														 cascade_activation_functions,
														 unsigned int 
														 cascade_activation_functions_count);


/* Function: fann_get_cascade_activation_steepnesses_count
 */
FANN_EXTERNAL unsigned int FANN_API fann_get_cascade_activation_steepnesses_count(struct fann *ann);


/* Function: fann_get_cascade_activation_steepnesses
 */
FANN_EXTERNAL fann_type * FANN_API fann_get_cascade_activation_steepnesses(struct fann *ann);
																

/* Function: fann_set_cascade_activation_steepnesses
 */
FANN_EXTERNAL void fann_set_cascade_activation_steepnesses(struct fann *ann,
														   fann_type *
														   cascade_activation_steepnesses,
														   unsigned int 
														   cascade_activation_steepnesses_count);

#endif
