#include <Python.h>
#include <math.h>
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include <arrayobject.h>

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
    uint8_t * restrict X_ptr = (uint8_t *) PyArray_DATA((PyArrayObject *) X);
    double * restrict y_ptr = (double *) PyArray_DATA((PyArrayObject *) y);
    uint32_t * restrict count_ptr = (uint32_t *) PyArray_DATA((PyArrayObject *) count_out);
    double * restrict sum_ptr = (double *) PyArray_DATA((PyArrayObject *) sum_out);
    double * restrict sum_sqs_ptr = (double *) PyArray_DATA((PyArrayObject *) sum_sqs_out);

    const int rows = (int) PyArray_DIM((PyArrayObject *) X, 0);
    const int cols = (int) PyArray_DIM((PyArrayObject *) X, 1);
    const int vals = (int) PyArray_DIM((PyArrayObject *) count_out, 1);

    // parallelize over the features
    // so that each thread writes to distinct memory
    #pragma omp parallel for
    for (int c = 0; c < cols; c++) {
        for (int r = 0; r < rows; r++) {
            int idx = c * vals + X_ptr[r * cols + c];
            count_ptr[idx]++;
            sum_ptr[idx] += y_ptr[r];
            sum_sqs_ptr[idx] += y_ptr[r] * y_ptr[r];
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
