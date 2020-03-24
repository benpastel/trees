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
    PyObject *X=NULL, *y=NULL, *count_out=NULL, *sum_out=NULL, *sum_sqs_out=NULL;

    if (!PyArg_ParseTuple(args, "O!O!O!O!O!",
        &PyArray_Type, &X_arg,
        &PyArray_Type, &y_arg,
        &PyArray_Type, &count_arg,
        &PyArray_Type, &sum_arg,
        &PyArray_Type, &sum_sqs_arg)) return NULL;

    X = PyArray_FROM_OTF(X_arg, NPY_UINT8, NPY_ARRAY_IN_ARRAY);
    if (X == NULL) return NULL;
    y = PyArray_FROM_OTF(y_arg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (y == NULL) goto fail;
    count_out = PyArray_FROM_OTF(count_arg, NPY_UINT32, NPY_ARRAY_INOUT_ARRAY2);
    if (count_out == NULL) goto fail;
    sum_out = PyArray_FROM_OTF(sum_arg, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    if (sum_out == NULL) goto fail;
    sum_sqs_out = PyArray_FROM_OTF(sum_sqs_arg, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
    if (sum_sqs_out == NULL) goto fail;

    npy_intp X_dims[2];
    npy_intp y_dims[1];
    npy_intp count_dims[2];
    npy_intp sum_dims[2];
    npy_intp sum_sqs_dims[2];

    npy_uint8 **X_ptr;
    npy_float64 *y_ptr;
    npy_uint32 **count_ptr;
    npy_float64 **sum_ptr;
    npy_float64 **sum_sqs_ptr;
    PyArray_AsCArray(&X, &X_ptr, X_dims, 2, PyArray_DescrFromType(NPY_UINT8));
    PyArray_AsCArray(&y, &y_ptr, y_dims, 1, PyArray_DescrFromType(NPY_FLOAT64));
    PyArray_AsCArray(&count_out, &count_ptr, count_dims, 2, PyArray_DescrFromType(NPY_UINT32));
    PyArray_AsCArray(&sum_out, &sum_ptr, sum_dims, 2, PyArray_DescrFromType(NPY_FLOAT64));
    PyArray_AsCArray(&sum_sqs_out, &sum_sqs_ptr, sum_sqs_dims, 2, PyArray_DescrFromType(NPY_FLOAT64));

    int rows = (int) X_dims[0];
    int cols = (int) X_dims[1];

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            npy_uint8 val = X_ptr[r][c];
            count_ptr[c][val] = count_ptr[c][val] + 1;
            sum_ptr[c][val] = sum_ptr[c][val] + y_ptr[r];
            sum_sqs_ptr[c][val] = sum_sqs_ptr[c][val] + (y_ptr[r] * y_ptr[r]);
        }
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