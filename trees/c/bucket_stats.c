#include "Python.h"
#include "math.h"
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "arrayobject.h"

static PyObject* bucket_stats(PyObject *self, PyObject *args);


static PyMethodDef Methods[] = {
    {"bucket_stats", bucket_stats, METH_VARARGS, "aggregate y statistics for each feature and value"},
    {NULL, NULL, 0, NULL}
};


static PyObject* bucket_stats(PyObject *dummy, PyObject *args)
{
    PyObject *X_arg=NULL, *y_arg=NULL, *count_arg=NULL;
    PyObject *X=NULL, *y=NULL, *count_out=NULL;

    if (!PyArg_ParseTuple(args, "O!O!O!",
        &PyArray_Type, &X_arg,
        &PyArray_Type, &y_arg,
        &PyArray_Type, &count_arg)) return NULL;

    X = PyArray_FROM_OTF(X_arg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (X == NULL) return NULL;
    y = PyArray_FROM_OTF(y_arg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (y == NULL) goto fail;
    count_out = PyArray_FROM_OTF(count_arg, NPY_UINT32, NPY_ARRAY_INOUT_ARRAY2);
    if (count_out == NULL) goto fail;

    npy_intp count_dims[1];
    npy_uint32 *count_ptr;
    PyArray_AsCArray(&count_out, &count_ptr, dims, 1, PyArray_DescrFromType(NPY_UINT32));
    int n = (int) dims[0];
    printf("n = %d\n", n);


    for (int i = 0; i < n; i++) {
       count_ptr[i] = i+1;
    }

    Py_DECREF(X);
    Py_DECREF(y);
    PyArray_ResolveWritebackIfCopy((PyArrayObject *) count_out);
    Py_DECREF(count_out);
    Py_INCREF(Py_None);
    return Py_None;

 fail:
    Py_XDECREF(X);
    Py_XDECREF(y);
    PyArray_DiscardWritebackIfCopy((PyArrayObject *) count_out);
    Py_XDECREF(count_out);
    return NULL;
}

static struct PyModuleDef mod_def =
{
    PyModuleDef_HEAD_INIT,
    "bucket_stats", /* name of module */
    "",          /* module documentation, may be NULL */
    -1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    Methods
};

PyMODINIT_FUNC
PyInit_bucket_stats(void)
{
    import_array();
    return PyModule_Create(&mod_def);
}
