/* File : fann.i */
%module fann

%include "typemaps.i"

%{
#include "../src/include/fann.h"
%}

/* Let's just grab the original header file here */
#define FANN_INCLUDE
%varargs(10,int n = 0) fann_create;
%include "../src/include/fann.h"

