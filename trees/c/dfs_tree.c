#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include <arrayobject.h>
#include <stdbool.h>
#include <float.h>
#include <sys/time.h>
#include <omp.h>

static PyObject* update_histograms(PyObject *self, PyObject *args);

static PyMethodDef Methods[] = {
    {"update_histograms", update_histograms, METH_VARARGS, ""},
    // {"eval_tree", eval_tree, METH_VARARGS, ""},
    // {"apply_bins", apply_bins, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}
};

#define VERBOSE 0


static PyObject* update_histograms(PyObject *dummy, PyObject *args)
{
    PyObject *memberships_arg;
    PyObject *X_arg, *y_arg;
    PyObject *hist_counts_arg, *hist_sums_arg, *hist_sum_sqs_arg;
    int node_arg;

    // parse input arguments
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!i",
        &PyArray_Type, &memberships_arg,
        &PyArray_Type, &X_arg,
        &PyArray_Type, &y_arg,
        &PyArray_Type, &hist_counts_arg,
        &PyArray_Type, &hist_sums_arg,
        &PyArray_Type, &hist_sum_sqs_arg,
        &node_arg)) return NULL;

    PyObject *memberships_obj = PyArray_FROM_OTF(memberships_arg, NPY_UINT16, NPY_ARRAY_IN_ARRAY);
    PyObject *X_obj = PyArray_FROM_OTF(X_arg, NPY_UINT8, NPY_ARRAY_IN_ARRAY);
    PyObject *y_obj = PyArray_FROM_OTF(y_arg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    PyObject *hist_counts_obj = PyArray_FROM_OTF(hist_counts_arg, NPY_UINT32, NPY_ARRAY_OUT_ARRAY);
    PyObject *hist_sums_obj = PyArray_FROM_OTF(hist_sums_arg, NPY_DOUBLE, NPY_ARRAY_OUT_ARRAY);
    PyObject *hist_sum_sqs_obj = PyArray_FROM_OTF(hist_sum_sqs_arg, NPY_DOUBLE, NPY_ARRAY_OUT_ARRAY);

    const int node = node_arg;
    const uint64_t rows = (uint64_t) PyArray_DIM((PyArrayObject *) X_obj, 0);
    const uint64_t cols = (uint64_t) PyArray_DIM((PyArrayObject *) X_obj, 1);
    const uint64_t vals = (uint64_t) PyArray_DIM((PyArrayObject *) hist_counts_obj, 2);

    // cast data sections of numpy arrays to plain C pointers
    // this assumes the arrays are C-order, aligned, non-strided
    uint8_t *  __restrict X           = PyArray_DATA((PyArrayObject *) X_obj);
    double *   __restrict y           = PyArray_DATA((PyArrayObject *) y_obj);
    uint16_t * __restrict memberships = PyArray_DATA((PyArrayObject *) memberships_obj);

    // the histograms are indexed [node, column, bucket] => [column, bucket]
    uint32_t * __restrict counts = PyArray_DATA((PyArrayObject *) hist_counts_obj);
    double * __restrict sums = PyArray_DATA((PyArrayObject *) hist_sums_obj);
    double * __restrict sum_sqs = PyArray_DATA((PyArrayObject *) hist_sum_sqs_obj);

    for (uint64_t r = 0; r < rows; r++) {
        if (memberships[r] != node) continue;

        for (uint64_t c = 0; c < cols; c++) {
            uint8_t v = X[r*cols + c];
            uint64_t idx = node*cols*vals + c*vals + v;
            counts[idx]++;
            sums[idx] += y[r];
            sum_sqs[idx] += y[r] * y[r];
        }
    }

    Py_RETURN_NONE;
}


static struct PyModuleDef mod_def =
{
    PyModuleDef_HEAD_INIT,
    "dfs_tree",
    "",
    -1,
    Methods
};

PyMODINIT_FUNC
PyInit_dfs_tree(void)
{
    import_array();
    return PyModule_Create(&mod_def);
}