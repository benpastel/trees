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
    PyObject *X_arg=NULL, *y_arg=NULL, *count_arg=NULL, *sum_arg=NULL, *sum_sqs_arg=NULL;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!",
        &PyArray_Type, &X_arg,
        &PyArray_Type, &y_arg,
        &PyArray_Type, &count_arg,
        &PyArray_Type, &sum_arg,
        &PyArray_Type, &sum_sqs_arg)) return NULL;

    PyObject *X = PyArray_FROM_OTF(X_arg, NPY_UINT8, NPY_ARRAY_IN_ARRAY);
    PyObject *y = PyArray_FROM_OTF(y_arg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyObject *count_out = PyArray_FROM_OTF(count_arg, NPY_UINT32, NPY_ARRAY_OUT_ARRAY);
    PyObject *sum_out = PyArray_FROM_OTF(sum_arg, NPY_DOUBLE, NPY_ARRAY_OUT_ARRAY);
    PyObject *sum_sqs_out = PyArray_FROM_OTF(sum_sqs_arg, NPY_DOUBLE, NPY_ARRAY_OUT_ARRAY);
    if (X == NULL || y == NULL || count_out == NULL || sum_out == NULL || sum_sqs_out == NULL) {
        Py_XDECREF(X);
        Py_XDECREF(y);
        Py_XDECREF(count_out);
        Py_XDECREF(sum_out);
        Py_XDECREF(sum_sqs_out);
        return NULL;
    }
    npy_uint8 *X_ptr = (npy_uint8 *) PyArray_DATA((PyArrayObject *) X);
    npy_float64 *y_ptr = (npy_float64 *) PyArray_DATA((PyArrayObject *) y);
    npy_uint32 *count_ptr = (npy_uint32 *) PyArray_DATA((PyArrayObject *) count_out);
    npy_float64 *sum_ptr = (npy_float64 *) PyArray_DATA((PyArrayObject *) sum_out);
    npy_float64 *sum_sqs_ptr = (npy_float64 *) PyArray_DATA((PyArrayObject *) sum_sqs_out);

    int rows = (int) PyArray_DIM((PyArrayObject *) X, 0);
    int cols = (int) PyArray_DIM((PyArrayObject *) X, 1);
    int vals = (int) PyArray_DIM((PyArrayObject *) count_out, 1);
    for (int r = 0; r < rows; r++) {
        npy_float64 y_val = y_ptr[r];
        npy_float64 y_square = y_val * y_val;
        for (int c = 0; c < cols; c++) {
            int idx = c * vals + X_ptr[r * cols + c];
            count_ptr[idx]++;
            sum_ptr[idx] += y_val;
            sum_sqs_ptr[idx] += y_square;
        }
    }
    Py_DECREF(X);
    Py_DECREF(y);
    Py_DECREF(count_out);
    Py_DECREF(sum_out);
    Py_DECREF(sum_sqs_out);
    Py_INCREF(Py_None);
    return Py_None;
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
