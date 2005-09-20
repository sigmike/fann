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

#ifndef __fann_error_h__
#define __fann_error_h__

/* Package: FANN Error Handling */
	
/* Function: fann_set_error_log

   change where errors are logged to
 */ 
FANN_EXTERNAL void FANN_API fann_set_error_log(struct fann_error *errdat, FILE * log_file);


/* Function: fann_get_errno

   returns the last error number
 */ 
FANN_EXTERNAL enum fann_errno_enum FANN_API fann_get_errno(struct fann_error *errdat);


/* Function: fann_reset_errno

   resets the last error number
 */ 
FANN_EXTERNAL void FANN_API fann_reset_errno(struct fann_error *errdat);


/* Function: fann_reset_errstr

   resets the last error string
 */ 
FANN_EXTERNAL void FANN_API fann_reset_errstr(struct fann_error *errdat);


/* Function: fann_get_errstr

   returns the last errstr.
 * This function calls fann_reset_errno and fann_reset_errstr
 */ 
FANN_EXTERNAL char *FANN_API fann_get_errstr(struct fann_error *errdat);


/* Function: fann_print_error

   prints the last error to stderr
 */ 
FANN_EXTERNAL void FANN_API fann_print_error(struct fann_error *errdat);

#endif
