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

/* Reads training data from a file.
 */
struct fann_train_data* fann_read_train_from_file(char *configuration_file)
{
	struct fann_train_data* data;
	FILE *file = fopen(configuration_file, "r");

	if(!file){
		fann_error(NULL, FANN_E_CANT_OPEN_CONFIG_R, configuration_file);
		return NULL;
	}

	data = fann_read_train_from_fd(file, configuration_file);
	fclose(file);
	return data;
}

/* Save training data to a file
 */
void fann_save_train(struct fann_train_data* data, char *filename)
{
	fann_save_train_internal(data, filename, 0, 0);
}

/* Save training data to a file in fixed point algebra.
   (Good for testing a network in fixed point)
*/
void fann_save_train_to_fixed(struct fann_train_data* data, char *filename, unsigned int decimal_point)
{
	fann_save_train_internal(data, filename, 1, decimal_point);
}

/* deallocate the train data structure.
 */
void fann_destroy_train(struct fann_train_data *data)
{
	unsigned int i;
	if(data->input){
		for(i = 0; i != data->num_data; i++){
			fann_safe_free(data->input[i]);
		}
	}

	if(data->output){
		for(i = 0; i != data->num_data; i++){
			fann_safe_free(data->output[i]);
		}
	}
	
	fann_safe_free(data->input);
	fann_safe_free(data->output);
	fann_safe_free(data);
}

#ifndef FIXEDFANN

/* Train directly on the training data.
 */
void fann_train_on_data_callback(struct fann *ann, struct fann_train_data *data, unsigned int max_epochs, unsigned int epochs_between_reports, float desired_error, int (*callback)(unsigned int epochs, float error))
{
	float error;
	unsigned int i, j;
	
	if(epochs_between_reports && callback == NULL){
		printf("Max epochs %8d. Desired error: %.10f\n", max_epochs, desired_error);
	}
	
	for(i = 1; i <= max_epochs; i++){
		/* train */
		fann_reset_MSE(ann);
		
		for(j = 0; j != data->num_data; j++){
			fann_train(ann, data->input[j], data->output[j]);
		}
		
		error = fann_get_MSE(ann);
		
		/* print current output */
		if(epochs_between_reports &&
			(i % epochs_between_reports == 0
				|| i == max_epochs
				|| i == 1
				|| error < desired_error)){
			if (callback == NULL) {
				printf("Epochs     %8d. Current error: %.10f\n", i, error);
			} else if((*callback)(i, error) == -1){
				/* you can break the training by returning -1 */
				break;
			}
		}
		
		if(error < desired_error){
			break;
		}
	}
}

void fann_train_on_data(struct fann *ann, struct fann_train_data *data, unsigned int max_epochs, unsigned int epochs_between_reports, float desired_error)
{
	fann_train_on_data_callback(ann, data, max_epochs, epochs_between_reports, desired_error, NULL);
}


/* Wrapper to make it easy to train directly on a training data file.
 */
void fann_train_on_file_callback(struct fann *ann, char *filename, unsigned int max_epochs, unsigned int epochs_between_reports, float desired_error, int (*callback)(unsigned int epochs, float error))
{
	struct fann_train_data *data = fann_read_train_from_file(filename);
	if(data == NULL){
		return;
	}
	fann_train_on_data_callback(ann, data, max_epochs, epochs_between_reports, desired_error, callback);
	fann_destroy_train(data);
}

void fann_train_on_file(struct fann *ann, char *filename, unsigned int max_epochs, unsigned int epochs_between_reports, float desired_error)
{
	fann_train_on_file_callback(ann, filename, max_epochs, epochs_between_reports, desired_error, NULL);
}

#endif

/* shuffles training data, randomizing the order
 */
void fann_shuffle_train_data(struct fann_train_data *train_data) {
	unsigned int dat = 0, elem, swap;
	fann_type temp;

	for ( ; dat < train_data->num_data ; dat++ ) {
		swap = (unsigned int)(rand() % train_data->num_data);
		if ( swap != dat ) {
			for ( elem = 0 ; elem < train_data->num_input ; elem++ ) {
				temp = train_data->input[dat][elem];
				train_data->input[dat][elem] = train_data->input[swap][elem];
				train_data->input[swap][elem] = temp;
			}
			for ( elem = 0 ; elem < train_data->num_output ; elem++ ) {
				temp = train_data->output[dat][elem];
				train_data->output[dat][elem] = train_data->output[swap][elem];
				train_data->output[swap][elem] = temp;
			}
		}
	}
}

/* merges training data into a single struct.
 */
struct fann_train_data * fann_merge_train_data(struct fann_train_data *data1, struct fann_train_data *data2) {
	struct fann_train_data * train_data;
	unsigned int x;

	if ( (data1->num_input  != data2->num_input) ||
	     (data1->num_output != data2->num_output) ) {
		fann_error(NULL, FANN_E_TRAIN_DATA_MISMATCH);
		return NULL;
	}

	train_data = (struct fann_train_data *)malloc(sizeof(struct fann_train_data));

	fann_init_error_data((struct fann_error *)train_data);

	train_data->num_data = data1->num_data + data2->num_data;
	train_data->num_input = data1->num_input;
	train_data->num_output = data1->num_output;

	if ( ((train_data->input  = (fann_type **)calloc(train_data->num_data, sizeof(fann_type *))) == NULL) ||
	     ((train_data->output = (fann_type **)calloc(train_data->num_data, sizeof(fann_type *))) == NULL) ) {
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		fann_destroy_train(train_data);
		return NULL;
	}
	for ( x = 0 ; x < train_data->num_data ; x++ ) {
		if ( ((train_data->input[x]  = (fann_type *)calloc(train_data->num_input,  sizeof(fann_type))) == NULL) ||
		     ((train_data->output[x] = (fann_type *)calloc(train_data->num_output, sizeof(fann_type))) == NULL) ) {
			fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
			fann_destroy_train(train_data);
			return NULL;
		}
		memcpy(train_data->input[x],
		       ( x < data1->num_data ) ? data1->input[x]  : data2->input[x - data1->num_data],
		       train_data->num_input  * sizeof(fann_type));
		memcpy(train_data->output[x],
		       ( x < data1->num_data ) ? data1->output[x] : data2->output[x - data1->num_data],
		       train_data->num_output * sizeof(fann_type));
	}

	return train_data;
}

/* return a copy of a fann_train_data struct
 */
struct fann_train_data * fann_duplicate_train_data(struct fann_train_data *data) {
	struct fann_train_data * dest;
	unsigned int x;

	if ( (dest = (struct fann_train_data *)malloc(sizeof(struct fann_train_data))) == NULL ) {
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		return NULL;
	}

	fann_init_error_data((struct fann_error *)dest);

	dest->num_data = data->num_data;
	dest->num_input = data->num_input;
	dest->num_output = data->num_output;

	if ( ((dest->input  = (fann_type **)calloc(dest->num_data, sizeof(fann_type *))) == NULL) ||
	     ((dest->output = (fann_type **)calloc(dest->num_data, sizeof(fann_type *))) == NULL) ) {
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		fann_destroy_train(dest);
		return NULL;
	}

	for ( x = 0 ; x < dest->num_data ; x++ ) {
		if ( ((dest->input[x]  = (fann_type *)calloc(dest->num_input,  sizeof(fann_type))) == NULL) ||
		     ((dest->output[x] = (fann_type *)calloc(dest->num_output, sizeof(fann_type))) == NULL) ) {
			fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
			fann_destroy_train(dest);
			return NULL;
		}
		memcpy(dest->input[x],  data->input[x],  dest->num_input  * sizeof(fann_type));
		memcpy(dest->output[x], data->output[x], dest->num_output * sizeof(fann_type));
	}
	return dest;
}

/* INTERNAL FUNCTION
   Reads training data from a file descriptor.
 */
struct fann_train_data* fann_read_train_from_fd(FILE *file, char *filename)
{
	unsigned int num_input, num_output, num_data, i, j;
	unsigned int line = 1;
	struct fann_train_data* data = (struct fann_train_data *)malloc(sizeof(struct fann_train_data));

	if(data == NULL){
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		return NULL;
	}
	
	if(fscanf(file, "%u %u %u\n", &num_data, &num_input, &num_output) != 3){
		fann_error(NULL, FANN_E_CANT_READ_TD, filename, line);
		fann_destroy_train(data);
		return NULL;
	}
	line++;

	fann_init_error_data((struct fann_error *)data);

	data->num_data = num_data;
	data->num_input = num_input;
	data->num_output = num_output;
	data->input = (fann_type **)calloc(num_data, sizeof(fann_type *));
	if(data->input == NULL){
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		fann_destroy_train(data);
		return NULL;
	}
	
	data->output = (fann_type **)calloc(num_data, sizeof(fann_type *));
	if(data->output == NULL){
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		fann_destroy_train(data);
		return NULL;
	}
	
	for(i = 0; i != num_data; i++){
		data->input[i] = (fann_type *)calloc(num_input, sizeof(fann_type));
		if(data->input[i] == NULL){
			fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
			fann_destroy_train(data);
			return NULL;
		}
		
		for(j = 0; j != num_input; j++){
			if(fscanf(file, FANNSCANF" ", &data->input[i][j]) != 1){
				fann_error(NULL, FANN_E_CANT_READ_TD, filename, line);
				fann_destroy_train(data);
				return NULL;
			}
		}
		line++;
		
		data->output[i] = (fann_type *)calloc(num_output, sizeof(fann_type));
		if(data->output[i] == NULL){
			fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
			fann_destroy_train(data);
			return NULL;
		}

		for(j = 0; j != num_output; j++){
			if(fscanf(file, FANNSCANF" ", &data->output[i][j]) != 1){
				fann_error(NULL, FANN_E_CANT_READ_TD, filename, line);
				fann_destroy_train(data);
				return NULL;
			}
		}
		line++;
	}
	return data;
}
