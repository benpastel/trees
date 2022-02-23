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

static PyObject* update_memberships_and_counts(PyObject *self, PyObject *args);

static PyMethodDef Methods[] = {
    {"update_histograms", update_histograms, METH_VARARGS, ""},
    // {"update_node_splits", update_node_splits, METH_VARARGS, ""},
    {"update_memberships_and_counts", update_memberships_and_counts, METH_VARARGS, ""},
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

    PyObject *memberships_obj = PyArray_FROM_OTF(memberships_arg, NPY_UINT64, NPY_ARRAY_IN_ARRAY);
    PyObject *X_obj = PyArray_FROM_OTF(X_arg, NPY_UINT8, NPY_ARRAY_IN_ARRAY);
    PyObject *y_obj = PyArray_FROM_OTF(y_arg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    PyObject *hist_counts_obj = PyArray_FROM_OTF(hist_counts_arg, NPY_UINT32, NPY_ARRAY_OUT_ARRAY);
    PyObject *hist_sums_obj = PyArray_FROM_OTF(hist_sums_arg, NPY_DOUBLE, NPY_ARRAY_OUT_ARRAY);
    PyObject *hist_sum_sqs_obj = PyArray_FROM_OTF(hist_sum_sqs_arg, NPY_DOUBLE, NPY_ARRAY_OUT_ARRAY);

    const int node = node_arg;
    const uint64_t rows_in_node = (uint64_t) PyArray_DIM((PyArrayObject *) memberships_obj, 0);
    const uint64_t cols = (uint64_t) PyArray_DIM((PyArrayObject *) X_obj, 1);
    const uint64_t vals = (uint64_t) PyArray_DIM((PyArrayObject *) hist_counts_obj, 2);

    // cast data sections of numpy arrays to plain C pointers
    // this assumes the arrays are C-order, aligned, non-strided
    uint8_t *  __restrict X           = PyArray_DATA((PyArrayObject *) X_obj);
    double *   __restrict y           = PyArray_DATA((PyArrayObject *) y_obj);
    uint64_t * __restrict memberships = PyArray_DATA((PyArrayObject *) memberships_obj);

    // the histograms are indexed [node, column, bucket] => [column, bucket]
    uint32_t * __restrict counts = PyArray_DATA((PyArrayObject *) hist_counts_obj);
    double * __restrict sums = PyArray_DATA((PyArrayObject *) hist_sums_obj);
    double * __restrict sum_sqs = PyArray_DATA((PyArrayObject *) hist_sum_sqs_obj);

    // iterate over the rows in this node
    for (uint64_t i = 0; i < rows_in_node; i++) {
        uint64_t r = memberships[i];

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

static PyObject* update_memberships_and_counts(PyObject *dummy, PyObject *args)
{
    // X: (rows, cols) array of uint8s,
    // memberships: (rows,)  array of uint16s,
    // node_counts: (nodes,) array of uint64s
    // col: uint64 column we are splitting on
    // parent: uint16 parent node,
    // left_child: uint16 left child node (right child is +1)
    // split_val: uint8 (all values <= split_val go to the left child)
    //
    // change memberships from parent to left or right child
    // set the counts of left & right children
    //
    // TODO: change memberships to arrays; then this will become harder
    PyObject *X_arg, *memberships_arg, *node_counts_arg;
    int col_arg;
    int parent_arg;
    int left_child_arg;
    int split_val_arg;

    // parse input arguments
    if (!PyArg_ParseTuple(args, "O!O!O!iiii",
        &PyArray_Type, &X_arg,
        &PyArray_Type, &memberships_arg,
        &PyArray_Type, &node_counts_arg,
        &col_arg,
        &parent_arg,
        &left_child_arg,
        &split_val_arg
    )) return NULL;

    PyObject *X_obj = PyArray_FROM_OTF(X_arg, NPY_UINT8, NPY_ARRAY_IN_ARRAY);
    PyObject *memberships_obj = PyArray_FROM_OTF(memberships_arg, NPY_UINT16, NPY_ARRAY_OUT_ARRAY);
    PyObject *node_counts_obj = PyArray_FROM_OTF(node_counts_arg, NPY_UINT64, NPY_ARRAY_OUT_ARRAY);

    const uint64_t col = col_arg;
    const uint16_t parent = parent_arg;
    const uint16_t left_child = left_child_arg;
    const uint16_t right_child = left_child + 1;
    const uint8_t split_val = split_val_arg;
    const uint64_t rows = (uint64_t) PyArray_DIM((PyArrayObject *) X_obj, 0);
    const uint64_t cols = (uint64_t) PyArray_DIM((PyArrayObject *) X_obj, 1);

    // cast data sections of numpy arrays to plain C pointers
    // this assumes the arrays are C-order, aligned, non-strided
    uint8_t *  __restrict X           = PyArray_DATA((PyArrayObject *) X_obj);
    uint16_t * __restrict memberships = PyArray_DATA((PyArrayObject *) memberships_obj);
    double *   __restrict node_counts = PyArray_DATA((PyArrayObject *) node_counts_obj);

    uint64_t left_count = 0;

    for (uint64_t r = 0; r < rows; r++) {
        if (memberships[r] != parent) continue;

        if (X[r * cols + col] <= split_val) {
            // assign to left child
            memberships[r] = left_child;
            left_count++;
        } else {
            // assign to right child
            memberships[r] = right_child;
        }
    }
    node_counts[left_child] = left_count;

    // everything else is right count
    node_counts[right_child] = node_counts[parent] - node_counts[left_child];

    Py_RETURN_NONE;
}

// static PyObject* update_node_splits(PyObject *dummy, PyObject *args)
// {
//     PyObject *hist_counts_arg, *hist_sums_arg, *hist_sum_sqs_arg;
//     PyObject *node_gains_arg, *split_cols_arg, *split_bins_arg;
//     int node_arg;

//     // parse input arguments
//     if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!i",
//         &PyArray_Type, &hist_counts_arg,
//         &PyArray_Type, &hist_sums_arg,
//         &PyArray_Type, &hist_sum_sqs_arg,
//         &PyArray_Type, &node_gains_arg,
//         &PyArray_Type, &split_cols_arg,
//         &PyArray_Type, &split_bins_arg,
//         &node_arg)) return NULL;

//     PyObject *hist_counts_obj = PyArray_FROM_OTF(hist_counts_arg, NPY_UINT32, NPY_ARRAY_IN_ARRAY);
//     PyObject *hist_sums_obj = PyArray_FROM_OTF(hist_sums_arg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
//     PyObject *hist_sum_sqs_obj = PyArray_FROM_OTF(hist_sum_sqs_arg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

//     PyObject *node_gains_obj = PyArray_FROM_OTF(node_gains_arg, NPY_DOUBLE, NPY_ARRAY_OUT_ARRAY);
//     PyObject *split_cols_obj = PyArray_FROM_OTF(split_cols_arg, NPY_UINT64, NPY_ARRAY_OUT_ARRAY);
//     PyObject *split_bins_obj = PyArray_FROM_OTF(split_bins_arg, NPY_UINT8, NPY_ARRAY_OUT_ARRAY);

//     // cast data sections of numpy arrays to plain C pointers
//     // this assumes the arrays are C-order, aligned, non-strided
//     double   * __restrict node_gains = PyArray_DATA((PyArrayObject *) node_gains_obj);
//     uint64_t * __restrict split_cols = PyArray_DATA((PyArrayObject *) split_cols_obj);
//     uint8_t  * __restrict split_bins = PyArray_DATA((PyArrayObject *) split_bins_obj);

//     // the histograms are indexed [node, column, bucket] => [column, bucket]
//     uint32_t * __restrict counts = PyArray_DATA((PyArrayObject *) hist_counts_obj);
//     double * __restrict sums = PyArray_DATA((PyArrayObject *) hist_sums_obj);
//     double * __restrict sum_sqs = PyArray_DATA((PyArrayObject *) hist_sum_sqs_obj);

//     const int node = node_arg;
//     const uint64_t cols = (uint64_t) PyArray_DIM((PyArrayObject *) hist_counts_obj, 1);
//     const uint64_t vals = (uint64_t) PyArray_DIM((PyArrayObject *) hist_counts_obj, 2);

//     double best_gain =
//     for (uint64_t c = 0; c < cols; c++) {



//     }

//     Py_RETURN_NONE;
// }


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