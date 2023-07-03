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
#define MIN_PARALLEL_SPLIT 1024
#define SPLIT_BUF_SIZE 1024

static PyObject* update_histograms(PyObject *dummy, PyObject *args)
{
    PyObject *memberships_arg;
    PyObject *X_arg, *y_arg;
    PyObject *hist_counts_arg, *hist_sums_arg, *hist_sum_sqs_arg;

    // parse input arguments
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!",
        &PyArray_Type, &memberships_arg,
        &PyArray_Type, &X_arg,
        &PyArray_Type, &y_arg,
        &PyArray_Type, &hist_counts_arg,
        &PyArray_Type, &hist_sums_arg,
        &PyArray_Type, &hist_sum_sqs_arg)) return NULL;

    PyObject *memberships_obj = PyArray_FROM_OTF(memberships_arg, NPY_UINT32, NPY_ARRAY_IN_ARRAY);
    PyObject *X_obj = PyArray_FROM_OTF(X_arg, NPY_UINT8, NPY_ARRAY_IN_ARRAY);
    PyObject *y_obj = PyArray_FROM_OTF(y_arg, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);

    PyObject *hist_counts_obj = PyArray_FROM_OTF(hist_counts_arg, NPY_UINT32, NPY_ARRAY_OUT_ARRAY);
    PyObject *hist_sums_obj = PyArray_FROM_OTF(hist_sums_arg, NPY_FLOAT32, NPY_ARRAY_OUT_ARRAY);
    PyObject *hist_sum_sqs_obj = PyArray_FROM_OTF(hist_sum_sqs_arg, NPY_FLOAT32, NPY_ARRAY_OUT_ARRAY);

    const uint32_t rows_in_node = (uint32_t) PyArray_DIM((PyArrayObject *) memberships_obj, 0);
    const uint32_t rows = (uint32_t) PyArray_DIM((PyArrayObject *) X_obj, 0);
    const uint32_t cols = (uint32_t) PyArray_DIM((PyArrayObject *) X_obj, 1);
    const uint32_t cols_over_16 = cols / 16;
    const uint32_t vals = (uint32_t) PyArray_DIM((PyArrayObject *) hist_counts_obj, 1);

    uint8_t  * __restrict X = PyArray_DATA((PyArrayObject *) X_obj);
    float    * __restrict y = PyArray_DATA((PyArrayObject *) y_obj);
    uint32_t * __restrict memberships = PyArray_DATA((PyArrayObject *) memberships_obj);
    uint32_t * __restrict counts = PyArray_DATA((PyArrayObject *) hist_counts_obj);
    float    * __restrict sums = PyArray_DATA((PyArrayObject *) hist_sums_obj);
    float    * __restrict sum_sqs = PyArray_DATA((PyArrayObject *) hist_sum_sqs_obj);

    if (rows_in_node < MIN_PARALLEL_SPLIT) {
        // build the histogram single-threaded
        for (uint32_t i = 0; i < rows_in_node; i++) {
            uint32_t r = memberships[i];

            for (uint32_t c = 0; c < cols_over_16 * 16; c++) {
                uint8_t v = X[r*cols + c];
                uint32_t idx = c*vals + v;
                counts[idx]++;
                sums[idx] += y[r];
                sum_sqs[idx] += y[r]*y[r];
            }
        }
    } else {
        #pragma omp parallel
        {
            // accumulate a separate histogram locally in each thread
            uint32_t * __restrict local_counts = calloc(cols*vals, sizeof(uint32_t));
            float * __restrict local_sums = calloc(cols*vals, sizeof(float));
            float * __restrict local_sum_sqs = calloc(cols*vals, sizeof(float));

            if (rows == rows_in_node) {
                // the root node histogram accounts for a lot of the final runtime
                // as a slight optimization, we can skip the memberships lookup
                // because all rows belong to the root
                #pragma omp for nowait
                for (uint32_t r = 0; r < rows_in_node; r++) {
                    for (uint32_t c = 0; c < cols_over_16 * 16; c++) {
                        uint8_t v = X[r*cols + c];
                        uint32_t idx = c*vals + v;
                        local_counts[idx]++;
                        local_sums[idx] += y[r];
                        local_sum_sqs[idx] += y[r]*y[r];
                    }
                }
            } else {
                // the general case: we need to look up which rows belong to the node
                #pragma omp for nowait
                for (uint32_t i = 0; i < rows_in_node; i++) {
                    uint32_t r = memberships[i];

                    for (uint32_t c = 0; c < cols_over_16 * 16; c++) {
                        uint8_t v = X[r*cols + c];
                        uint32_t idx = c*vals + v;
                        local_counts[idx]++;
                        local_sums[idx] += y[r];
                        local_sum_sqs[idx] += y[r]*y[r];
                    }
                }
            }

            // add the histograms together
            #pragma omp critical
            {
                for (uint32_t c = 0; c < cols_over_16 * 16; c++) {
                    for (uint v = 0; v < vals; v++) {
                        uint32_t i = c*vals + v;
                        counts[i] += local_counts[i];
                        sums[i] += local_sums[i];
                        sum_sqs[i] += local_sum_sqs[i];
                    }
                }
            }
            free(local_counts);
            free(local_sums);
            free(local_sum_sqs);
        }
    }
    Py_DECREF(memberships_obj);
    Py_DECREF(X_obj);
    Py_DECREF(y_obj);
    Py_DECREF(hist_counts_obj);
    Py_DECREF(hist_sums_obj);
    Py_DECREF(hist_sum_sqs_obj);
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
    PyObject *parent_members_obj = PyArray_FROM_OTF(parent_members_arg, NPY_UINT32, NPY_ARRAY_IN_ARRAY);
    PyObject *left_members_obj = PyArray_FROM_OTF(left_members_arg, NPY_UINT32, NPY_ARRAY_OUT_ARRAY);
    PyObject *right_members_obj = PyArray_FROM_OTF(right_members_arg, NPY_UINT32, NPY_ARRAY_OUT_ARRAY);

    const uint32_t col = col_arg;
    const uint8_t val = val_arg;
    const uint32_t rows_in_parent = (uint32_t) PyArray_DIM((PyArrayObject *) parent_members_obj, 0);
    const uint32_t rows = (uint32_t) PyArray_DIM((PyArrayObject *) X_obj, 0);
    const uint32_t cols = (uint32_t) PyArray_DIM((PyArrayObject *) X_obj, 1);

    // cast data sections of numpy arrays to plain C pointers
    // this assumes the arrays are C-order, aligned, non-strided
    uint8_t *  __restrict X = PyArray_DATA((PyArrayObject *) X_obj);
    uint32_t * __restrict parent_members = PyArray_DATA((PyArrayObject *) parent_members_obj);
    uint32_t * __restrict left_members = PyArray_DATA((PyArrayObject *) left_members_obj);
    uint32_t * __restrict right_members = PyArray_DATA((PyArrayObject *) right_members_obj);

    uint32_t left_i = 0;
    uint32_t right_i = 0;
    if (rows_in_parent < MIN_PARALLEL_SPLIT) {
        for (uint32_t p = 0; p < rows_in_parent; p++) {
            uint32_t r = parent_members[p];
            if (X[r * cols + col] <= val) {
                // assign to left child
                left_members[left_i++] = r;
            } else {
                // assign to right child
                right_members[right_i++] = r;
            }
        }
        Py_DECREF(X_obj);
        Py_DECREF(parent_members_obj);
        Py_DECREF(left_members_obj);
        Py_DECREF(right_members_obj );
        Py_RETURN_NONE;
    }
    // otherwise, multi-threaded
    //
    // the order of rows doesn't matter
    //
    // aggregate a buffer within each thread
    // once full, copy it into memberships
    #pragma omp parallel
    {
        uint32_t left_buf [SPLIT_BUF_SIZE];
        uint32_t right_buf [SPLIT_BUF_SIZE];

        uint32_t local_left_i = 0;
        uint32_t local_right_i = 0;

        uint32_t copy_start;

        if (rows == rows_in_parent) {
            // special case when splitting the root
            // we don't need to lookup parent members
            #pragma omp for nowait
            for (uint32_t r = 0; r < rows; r++) {
                if (X[r * cols + col] <= val) {
                    left_buf[local_left_i++] = r;
                    if (local_left_i == SPLIT_BUF_SIZE) {
                        #pragma omp critical
                        {
                            copy_start = left_i;
                            left_i = copy_start + SPLIT_BUF_SIZE;
                        }
                        memcpy(left_members + copy_start, left_buf, SPLIT_BUF_SIZE * sizeof(uint32_t));
                        local_left_i = 0;
                    }
                } else {
                    right_buf[local_right_i++] = r;
                    if (local_right_i == SPLIT_BUF_SIZE) {
                        #pragma omp critical
                        {
                            copy_start = right_i;
                            right_i = copy_start + SPLIT_BUF_SIZE;
                        }
                        memcpy(right_members + copy_start, right_buf, SPLIT_BUF_SIZE * sizeof(uint32_t));
                        local_right_i = 0;
                    }
                }
            }
        } else {
            // general case

            #pragma omp for nowait
            for (uint32_t p = 0; p < rows_in_parent; p++) {
                uint32_t r = parent_members[p];

                if (X[r * cols + col] <= val) {
                    left_buf[local_left_i++] = r;

                    if (local_left_i == SPLIT_BUF_SIZE) {
                        #pragma omp critical
                        {
                            copy_start = left_i;
                            left_i = copy_start + SPLIT_BUF_SIZE;
                        }
                        memcpy(left_members + copy_start, left_buf, SPLIT_BUF_SIZE * sizeof(uint32_t));
                        local_left_i = 0;
                    }
                } else {
                    right_buf[local_right_i++] = r;

                    if (local_right_i == SPLIT_BUF_SIZE) {
                        #pragma omp critical
                        {
                            copy_start = right_i;
                            right_i = copy_start + SPLIT_BUF_SIZE;
                        }
                        memcpy(right_members + copy_start, right_buf, SPLIT_BUF_SIZE * sizeof(uint32_t));
                        local_right_i = 0;
                    }
                }
            }
        }
        // copy anything leftover in the buffers
        if (local_left_i > 0) {
            #pragma omp critical
            {
                copy_start = left_i;
                left_i = copy_start + local_left_i;
            }
            memcpy(left_members + copy_start, left_buf, local_left_i * sizeof(uint32_t));
        }
        if (local_right_i > 0) {
            #pragma omp critical
            {
                copy_start = right_i;
                right_i = copy_start + local_right_i;
            }
            memcpy(right_members + copy_start, right_buf, local_right_i * sizeof(uint32_t));
        }
    }
    Py_DECREF(X_obj);
    Py_DECREF(parent_members_obj);
    Py_DECREF(left_members_obj);
    Py_DECREF(right_members_obj );
    Py_RETURN_NONE;
}

static inline float _variance(const float sum_sqs, const float sum, const uint32_t count) {
    assert(count > 0);
    return (sum_sqs - (sum * sum) / count) / count;
}

static inline float _gain(
    const float left_var,
    const float right_var,
    const float parent_var,
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

    const float old_mse = parent_var * parent_count;

    const float new_mse = left_var * left_count + right_var * right_count;

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
    PyObject *hist_sums_obj = PyArray_FROM_OTF(hist_sums_arg, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyObject *hist_sum_sqs_obj = PyArray_FROM_OTF(hist_sum_sqs_arg, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);

    PyObject *node_gains_obj = PyArray_FROM_OTF(node_gains_arg, NPY_FLOAT32, NPY_ARRAY_OUT_ARRAY);
    PyObject *split_cols_obj = PyArray_FROM_OTF(split_cols_arg, NPY_UINT32, NPY_ARRAY_OUT_ARRAY);
    PyObject *split_bins_obj = PyArray_FROM_OTF(split_bins_arg, NPY_UINT8, NPY_ARRAY_OUT_ARRAY);

    // cast data sections of numpy arrays to plain C pointers
    // this assumes the arrays are C-order, aligned, non-strided
    float    * __restrict node_gains = PyArray_DATA((PyArrayObject *) node_gains_obj);
    uint32_t * __restrict split_cols = PyArray_DATA((PyArrayObject *) split_cols_obj);
    uint8_t  * __restrict split_bins = PyArray_DATA((PyArrayObject *) split_bins_obj);

    // the histograms are indexed [node, column, bucket]
    uint32_t * __restrict counts = PyArray_DATA((PyArrayObject *) hist_counts_obj);
    float * __restrict sums = PyArray_DATA((PyArrayObject *) hist_sums_obj);
    float * __restrict sum_sqs = PyArray_DATA((PyArrayObject *) hist_sum_sqs_obj);

    const int node = node_arg;
    const uint32_t cols = (uint32_t) PyArray_DIM((PyArrayObject *) hist_counts_obj, 0);
    const uint32_t cols_over_16 = cols / 16;
    // const uint32_t vals = (uint32_t) PyArray_DIM((PyArrayObject *) hist_counts_obj, 1);
    const uint32_t vals = 64;

    // higher gain is better; above 0 means the split improved MSE
    float best_gain = 0;
    uint32_t best_col = 0;
    uint8_t best_v = 0;

    // parallelizing this for loop doesn't seem to gain anything, so single-threaded for now
    for (uint32_t c = 0; c < cols_over_16 * 16; c++) {

        // find the histogram totals for this column
        uint32_t total_count = 0;
        float total_sum = 0;
        float total_sum_sqs = 0;
        for (uint32_t v = 0; v < vals; v++) {
            uint32_t idx = c * vals + v;
            total_count += counts[idx];
            total_sum += sums[idx];
            total_sum_sqs += sum_sqs[idx];
        }
        float parent_var = _variance(total_sum_sqs, total_sum, total_count);

        uint32_t left_count = 0;
        float left_sum = 0;
        float left_sum_sqs = 0;

        // v is a proposed split value; x <= v will go left
        // max value is vals - 2 so that x = (vals - 1) will go right
        for (uint32_t v = 0; v < vals - 1; v++) {
            uint32_t idx = c * vals + v;
            left_count += counts[idx];
            left_sum += sums[idx];
            left_sum_sqs += sum_sqs[idx];

            if (left_count < MIN_LEAF_SIZE) continue;

            uint32_t right_count = total_count - left_count;
            float right_sum = total_sum - left_sum;
            float right_sum_sqs = total_sum_sqs - left_sum_sqs;

            if (right_count < MIN_LEAF_SIZE) continue;

            float left_var = _variance(left_sum_sqs, left_sum, left_count);
            float right_var = _variance(right_sum_sqs, right_sum, right_count);
            float gain = _gain(left_var, right_var, parent_var, left_count, right_count, total_count);

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

    Py_DECREF(hist_counts_arg);
    Py_DECREF(hist_sums_arg);
    Py_DECREF(hist_sum_sqs_arg);
    Py_DECREF(node_gains_arg);
    Py_DECREF(split_cols_arg);
    Py_DECREF(split_bins_arg);
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
    PyObject *split_cols_obj = PyArray_FROM_OTF(split_cols_arg, NPY_UINT32, NPY_ARRAY_IN_ARRAY);
    PyObject *split_vals_obj = PyArray_FROM_OTF(split_vals_arg, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyObject *left_children_obj = PyArray_FROM_OTF(left_children_arg, NPY_UINT16, NPY_ARRAY_IN_ARRAY);
    PyObject *node_mean_obj = PyArray_FROM_OTF(node_mean_arg, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyObject *out_obj = PyArray_FROM_OTF(out_arg, NPY_FLOAT32, NPY_ARRAY_OUT_ARRAY);

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
    float    * __restrict X             = PyArray_DATA((PyArrayObject *) X_obj);
    uint32_t * __restrict split_cols    = PyArray_DATA((PyArrayObject *) split_cols_obj);
    float    * __restrict split_vals    = PyArray_DATA((PyArrayObject *) split_vals_obj);
    uint16_t * __restrict left_children = PyArray_DATA((PyArrayObject *) left_children_obj);
    float    * __restrict node_means    = PyArray_DATA((PyArrayObject *) node_mean_obj);
    float    * __restrict out           = PyArray_DATA((PyArrayObject *) out_obj);

    const uint32_t rows = (uint32_t) PyArray_DIM((PyArrayObject *) X_obj, 0);
    const uint32_t cols = (uint32_t) PyArray_DIM((PyArrayObject *) X_obj, 1);

    #pragma omp parallel for
    for (uint32_t r = 0; r < rows; r++) {
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