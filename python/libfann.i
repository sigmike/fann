/* File : fann.i */
%module libfann

%include "typemaps.i"

%{
#include "../src/include/fann.h"
%}

%typemap(in) fann_type[ANY] {
  int i;
  if (!PySequence_Check($input)) {
    PyErr_SetString(PyExc_ValueError,"Expected a sequence");
    return NULL;
  }
  if (PySequence_Length($input) == 0) {
    PyErr_SetString(PyExc_ValueError,"Size mismatch. Expected some elements");
    return NULL;
  }
  $1 = (float *) malloc(PySequence_Length($input)*sizeof(float));
  for (i = 0; i < PySequence_Length($input); i++) {
    PyObject *o = PySequence_GetItem($input,i);
    if (PyNumber_Check(o)) {
      $1[i] = (float) PyFloat_AsDouble(o);
    } else {
      PyErr_SetString(PyExc_ValueError,"Sequence elements must be numbers");      
      return NULL;
    }
  }
}

%typemap(in) int[ANY] {
  int i;
  if (!PySequence_Check($input)) {
    PyErr_SetString(PyExc_ValueError,"Expected a sequence");
    return NULL;
  }
  if (PySequence_Length($input) == 0) {
    PyErr_SetString(PyExc_ValueError,"Size mismatch. Expected some elements");
    return NULL;
  }
  $1 = (unsigned int *) malloc(PySequence_Length($input)*sizeof(unsigned int));
  for (i = 0; i < PySequence_Length($input); i++) {
    PyObject *o = PySequence_GetItem($input,i);
    if (PyNumber_Check(o)) {
      $1[i] = (int) PyInt_AsLong(o);
    } else {
      PyErr_SetString(PyExc_ValueError,"Sequence elements must be numbers");      
      return NULL;
    }
  }
}

%typemap(freearg) fann_type* {
   if ($1) free($1);
}

%typemap(out) PyObject* {
  $result = $1;
}

%apply fann_type[ANY] {fann_type *};
%apply int[ANY] {int *, unsigned int*};

#define FANN_INCLUDE
%varargs(10,int n = 0) fann_create;
%rename(fann_run_old) fann_run;
%rename(fann_run) fann_run2;

%rename(fann_test_old) fann_test;
%rename(fann_test) fann_test2;

%include "../src/include/fann.h"
%include "../src/include/fann_data.h"

// Helper functions
PyObject* fann_run2(struct fann *ann, fann_type *input);
PyObject* fann_test2(struct fann *ann, fann_type *input, fann_type *desired_output);
PyObject* get_train_data_input(struct fann_train_data *ann, int row);
PyObject* get_train_data_output(struct fann_train_data *ann, int row);


