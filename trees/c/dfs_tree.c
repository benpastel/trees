#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include <arrayobject.h>
#include <stdbool.h>
#include <float.h>
#include <sys/time.h>
#include <omp.h>
#include <assert.h>

static PyObject* update_histograms(PyObject *self, PyObject *args);
static PyObject* update_node_splits(PyObject *self, PyObject *args);
static PyObject* update_memberships(PyObject *self, PyObject *args);
static PyObject* eval_tree(PyObject *self, PyObject *args);

static PyMethodDef Methods[] = {
    {"update_histograms", update_histograms, METH_VARARGS, ""},
    {"update_node_splits", update_node_splits, METH_VARARGS, ""},
    {"update_memberships", update_memberships, METH_VARARGS, ""},
    {"eval_tree", eval_tree, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}
};

#define VERBOSE 0
#define MIN_LEAF_SIZE 1 // TODO parameter

// TODO: py deref objects everywhere

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

    // the histograms are indexed [node, column, bucket]
    uint32_t * __restrict counts = PyArray_DATA((PyArrayObject *) hist_counts_obj);
    double * __restrict sums = PyArray_DATA((PyArrayObject *) hist_sums_obj);
    double * __restrict sum_sqs = PyArray_DATA((PyArrayObject *) hist_sum_sqs_obj);

    // => [column, bucket]
    counts += node*cols*vals;
    sums += node*cols*vals;
    sum_sqs += node*cols*vals;

    // iterate over the rows in this node
    for (uint64_t i = 0; i < rows_in_node; i++) {
        uint64_t r = memberships[i];

        for (uint64_t c = 0; c < cols; c++) {
            uint8_t v = X[r*cols + c];
            uint64_t idx = c*vals + v;
            counts[idx]++;
            sums[idx] += y[r];
            sum_sqs[idx] += y[r] * y[r];
        }
    }

    Py_RETURN_NONE;
}

static PyObject* update_memberships(PyObject *dummy, PyObject *args)
{
    // For each row in parent, add to left or right child
    PyObject *X_arg, *parent_members_arg, *left_members_arg, *right_members_arg;
    int col_arg;
    int val_arg;

    // parse input arguments
    if (!PyArg_ParseTuple(args, "O!O!O!O!ii",
        &PyArray_Type, &X_arg,
        &PyArray_Type, &parent_members_arg,
        &PyArray_Type, &left_members_arg,
        &PyArray_Type, &right_members_arg,
        &col_arg,
        &val_arg
    )) return NULL;

    PyObject *X_obj = PyArray_FROM_OTF(X_arg, NPY_UINT8, NPY_ARRAY_IN_ARRAY);
    PyObject *parent_members_obj = PyArray_FROM_OTF(parent_members_arg, NPY_UINT64, NPY_ARRAY_IN_ARRAY);
    PyObject *left_members_obj = PyArray_FROM_OTF(left_members_arg, NPY_UINT64, NPY_ARRAY_OUT_ARRAY);
    PyObject *right_members_obj = PyArray_FROM_OTF(right_members_arg, NPY_UINT64, NPY_ARRAY_OUT_ARRAY);

    const uint64_t col = col_arg;
    const uint8_t val = val_arg;
    const uint64_t rows_in_parent = (uint64_t) PyArray_DIM((PyArrayObject *) parent_members_obj, 0);
    const uint64_t cols = (uint64_t) PyArray_DIM((PyArrayObject *) X_obj, 1);

    // cast data sections of numpy arrays to plain C pointers
    // this assumes the arrays are C-order, aligned, non-strided
    uint8_t *  __restrict X = PyArray_DATA((PyArrayObject *) X_obj);
    uint64_t * __restrict parent_members = PyArray_DATA((PyArrayObject *) parent_members_obj);
    uint64_t * __restrict left_members = PyArray_DATA((PyArrayObject *) left_members_obj);
    uint64_t * __restrict right_members = PyArray_DATA((PyArrayObject *) right_members_obj);

    uint64_t left = 0;
    uint64_t right = 0;
    for (uint64_t p = 0; p < rows_in_parent; p++) {
        uint64_t r = parent_members[p];
        if (X[r * cols + col] <= val) {
            // assign to left child
            left_members[left++] = r;
        } else {
            // assign to right child
            right_members[right++] = r;
        }
    }
    Py_RETURN_NONE;
}

static inline double _variance(const double sum_sqs, const double sum, const uint32_t count) {
    assert(count > 0);
    return (sum_sqs - (sum * sum) / count) / count;
}

static inline double _gain(
    const double left_var,
    const double right_var,
    const double parent_var,
    const uint32_t left_count,
    const uint32_t right_count,
    const uint32_t parent_count
) {
    // how good is a potential split?
    //
    // TODO add penalty back in
    //
    // we'll start with the reduction in train MSE:
    //   higher is better
    //   above 0 means we made an improvement
    //   below 0 means we got worse
    //
    // predictions at leaves are E(X), so train MSE is sum((X - E(X))^2)
    //  = (variance at each leaf) * (number of examples in that leaf)
    //
    // variance is E(X - E(X))^2
    // E(X) will be the
    // i.e. (change in variance) * (number of rows that change applies to)
    assert(left_count + right_count == parent_count);

    const double old_mse = parent_var * parent_count;

    const double new_mse = left_var * left_count + right_var * right_count;

    return old_mse - new_mse;
}

static PyObject* update_node_splits(PyObject *dummy, PyObject *args)
{
    // set:
    //  split_cols[node] to the best column to split this node on
    //  split_bins[node] to the value to split that column
    //  node_gains[node] to the gain from taking the split
    PyObject *hist_counts_arg, *hist_sums_arg, *hist_sum_sqs_arg;
    PyObject *node_gains_arg, *split_cols_arg, *split_bins_arg;
    int node_arg;

    // parse input arguments
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!i",
        &PyArray_Type, &hist_counts_arg,
        &PyArray_Type, &hist_sums_arg,
        &PyArray_Type, &hist_sum_sqs_arg,
        &PyArray_Type, &node_gains_arg,
        &PyArray_Type, &split_cols_arg,
        &PyArray_Type, &split_bins_arg,
        &node_arg)) return NULL;

    PyObject *hist_counts_obj = PyArray_FROM_OTF(hist_counts_arg, NPY_UINT32, NPY_ARRAY_IN_ARRAY);
    PyObject *hist_sums_obj = PyArray_FROM_OTF(hist_sums_arg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyObject *hist_sum_sqs_obj = PyArray_FROM_OTF(hist_sum_sqs_arg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    PyObject *node_gains_obj = PyArray_FROM_OTF(node_gains_arg, NPY_DOUBLE, NPY_ARRAY_OUT_ARRAY);
    PyObject *split_cols_obj = PyArray_FROM_OTF(split_cols_arg, NPY_UINT64, NPY_ARRAY_OUT_ARRAY);
    PyObject *split_bins_obj = PyArray_FROM_OTF(split_bins_arg, NPY_UINT8, NPY_ARRAY_OUT_ARRAY);

    // cast data sections of numpy arrays to plain C pointers
    // this assumes the arrays are C-order, aligned, non-strided
    double   * __restrict node_gains = PyArray_DATA((PyArrayObject *) node_gains_obj);
    uint64_t * __restrict split_cols = PyArray_DATA((PyArrayObject *) split_cols_obj);
    uint8_t  * __restrict split_bins = PyArray_DATA((PyArrayObject *) split_bins_obj);

    // the histograms are indexed [node, column, bucket]
    uint32_t * __restrict counts = PyArray_DATA((PyArrayObject *) hist_counts_obj);
    double * __restrict sums = PyArray_DATA((PyArrayObject *) hist_sums_obj);
    double * __restrict sum_sqs = PyArray_DATA((PyArrayObject *) hist_sum_sqs_obj);

    const int node = node_arg;
    const uint64_t cols = (uint64_t) PyArray_DIM((PyArrayObject *) hist_counts_obj, 1);
    const uint64_t vals = (uint64_t) PyArray_DIM((PyArrayObject *) hist_counts_obj, 2);

    // higher gain is better; above 0 means the split improved MSE
    double best_gain = 0;
    uint64_t best_col = 0;
    uint8_t best_v = 0;

    for (uint64_t c = 0; c < cols; c++) {

        // find the histogram totals for this column
        uint32_t total_count = 0;
        double total_sum = 0;
        double total_sum_sqs = 0;
        for (uint64_t v = 0; v < vals; v++) {
            uint64_t idx = node * cols * vals + c * vals + v;
            total_count += counts[idx];
            total_sum += sums[idx];
            total_sum_sqs += sum_sqs[idx];
        }
        double parent_var = _variance(total_sum_sqs, total_sum, total_count);

        uint32_t left_count = 0;
        double left_sum = 0;
        double left_sum_sqs = 0;

        // v is a proposed split value; x <= v will go left
        // max value is vals - 2 so that x = (vals - 1) will go right
        for (uint64_t v = 0; v < vals - 1; v++) {
            uint64_t idx = node * cols * vals + c * vals + v;
            left_count += counts[idx];
            left_sum += sums[idx];
            left_sum_sqs += sum_sqs[idx];

            if (left_count < MIN_LEAF_SIZE) continue;

            uint32_t right_count = total_count - left_count;
            double right_sum = total_sum - left_sum;
            double right_sum_sqs = total_sum_sqs - left_sum_sqs;

            if (right_count < MIN_LEAF_SIZE) continue;

            double left_var = _variance(left_sum_sqs, left_sum, left_count);
            double right_var = _variance(right_sum_sqs, right_sum, right_count);
            double gain = _gain(left_var, right_var, parent_var, left_count, right_count, total_count);

            if (gain > best_gain) {
                best_gain = gain;
                best_col = c;
                best_v = v;
            }
        }
    }
    node_gains[node] = best_gain;
    split_cols[node] = best_col;
    split_bins[node] = best_v;

    Py_RETURN_NONE;
}


static PyObject* eval_tree(PyObject *dummy, PyObject *args)
{
    PyObject *X_arg;
    PyObject *split_cols_arg, *split_vals_arg;
    PyObject *left_children_arg, *node_mean_arg;
    PyObject *out_arg;

    // parse input arguments
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!",
        &PyArray_Type, &X_arg,
        &PyArray_Type, &split_cols_arg,
        &PyArray_Type, &split_vals_arg,
        &PyArray_Type, &left_children_arg,
        &PyArray_Type, &node_mean_arg,
        &PyArray_Type, &out_arg)) return NULL;

    PyObject *X_obj = PyArray_FROM_OTF(X_arg, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyObject *split_cols_obj = PyArray_FROM_OTF(split_cols_arg, NPY_UINT64, NPY_ARRAY_IN_ARRAY);
    PyObject *split_vals_obj = PyArray_FROM_OTF(split_vals_arg, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyObject *left_children_obj = PyArray_FROM_OTF(left_children_arg, NPY_UINT16, NPY_ARRAY_IN_ARRAY);
    PyObject *node_mean_obj = PyArray_FROM_OTF(node_mean_arg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyObject *out_obj = PyArray_FROM_OTF(out_arg, NPY_DOUBLE, NPY_ARRAY_OUT_ARRAY);

    if (X_obj == NULL ||
        split_cols_obj == NULL ||
        split_vals_obj == NULL ||
        left_children_obj == NULL ||
        node_mean_obj == NULL ||
        out_obj == NULL)
    {
        Py_XDECREF(X_obj);
        Py_XDECREF(split_cols_obj);
        Py_XDECREF(split_vals_obj);
        Py_XDECREF(left_children_obj);
        Py_XDECREF(node_mean_obj);
        Py_XDECREF(out_obj);
        return NULL;
    }
    // cast data sections of numpy arrays to plain C pointers
    // this assumes the arrays are C-order, aligned, non-strided
    float *    __restrict X            = PyArray_DATA((PyArrayObject *) X_obj);
    uint64_t * __restrict split_cols    = PyArray_DATA((PyArrayObject *) split_cols_obj);
    float *    __restrict split_vals    = PyArray_DATA((PyArrayObject *) split_vals_obj);
    uint16_t * __restrict left_children  = PyArray_DATA((PyArrayObject *) left_children_obj);
    double *   __restrict node_means   = PyArray_DATA((PyArrayObject *) node_mean_obj);
    double *   __restrict out          = PyArray_DATA((PyArrayObject *) out_obj);

    const uint64_t rows = (uint64_t) PyArray_DIM((PyArrayObject *) X_obj, 0);
    const uint64_t cols = (uint64_t) PyArray_DIM((PyArrayObject *) X_obj, 1);

    #pragma omp parallel for
    for (uint64_t r = 0; r < rows; r++) {
        uint16_t n = 0;
        uint16_t left;
        while ((left = left_children[n])) {
            float val = X[r*cols + split_cols[n]];
            // right child is left + 1
            n = left + (val > split_vals[n]);
        }
        out[r] = node_means[n];
    }

    Py_DECREF(X_obj);
    Py_DECREF(split_cols_obj);
    Py_DECREF(split_vals_obj);
    Py_DECREF(left_children_obj);
    Py_DECREF(node_mean_obj);
    Py_DECREF(out_obj);
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