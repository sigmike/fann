#include <Python.h>
#include <fann.h>

PyObject* fann_type_to_PyList(fann_type *array,int n)
{
  int i;
  PyObject* res = PyList_New(n);
  for (i = 0; i < n; i++) {
    PyObject *o = PyFloat_FromDouble((double) array[i]);
    PyList_SetItem(res,i,o);
  }
  return res;
}

PyObject* fann_run2(struct fann *ann, fann_type *input)
{
  if (!ann) return NULL;
  return fann_type_to_PyList(fann_run(ann,input),ann->num_output);
}


PyObject* fann_test2(struct fann *ann, fann_type *input, fann_type *desired_output)
{
  if (!ann) return NULL;
  return fann_type_to_PyList(fann_test(ann,input,desired_output),ann->num_output);
}
