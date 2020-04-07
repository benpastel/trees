#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include <arrayobject.h>
#include <stdbool.h>
#include <float.h>

static PyObject* build_tree(PyObject *self, PyObject *args);
static PyObject* eval_tree(PyObject *self, PyObject *args);
static PyObject* apply_bins(PyObject *self, PyObject *args);

static PyMethodDef Methods[] = {
    {"build_tree", build_tree, METH_VARARGS, ""},
    {"eval_tree", eval_tree, METH_VARARGS, ""},
    {"apply_bins", apply_bins, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}
};


static PyObject* build_tree(PyObject *dummy, PyObject *args)
{
    PyObject *X_arg, *y_arg;
    PyObject *split_col_arg, *split_val_arg, *left_children_arg, *right_children_arg, *node_mean_arg;
    double smooth_factor_arg;

    // parse input arguments
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!d",
        &PyArray_Type, &X_arg,
        &PyArray_Type, &y_arg,
        &PyArray_Type, &split_col_arg,
        &PyArray_Type, &split_val_arg,
        &PyArray_Type, &left_children_arg,
        &PyArray_Type, &right_children_arg,
        &PyArray_Type, &node_mean_arg,
        &smooth_factor_arg)) return NULL;
    const double smooth_factor = smooth_factor_arg;

    PyObject *X_obj = PyArray_FROM_OTF(X_arg, NPY_UINT8, NPY_ARRAY_IN_ARRAY);
    PyObject *y_obj = PyArray_FROM_OTF(y_arg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyObject *split_col_obj = PyArray_FROM_OTF(split_col_arg, NPY_UINT64, NPY_ARRAY_OUT_ARRAY);
    PyObject *split_val_obj = PyArray_FROM_OTF(split_val_arg, NPY_UINT8, NPY_ARRAY_OUT_ARRAY);
    PyObject *left_children_obj = PyArray_FROM_OTF(left_children_arg, NPY_UINT16, NPY_ARRAY_OUT_ARRAY);
    PyObject *right_children_obj = PyArray_FROM_OTF(right_children_arg, NPY_UINT16, NPY_ARRAY_OUT_ARRAY);
    PyObject *node_mean_obj = PyArray_FROM_OTF(node_mean_arg, NPY_DOUBLE, NPY_ARRAY_OUT_ARRAY);

    if (X_obj == NULL ||
        y_obj == NULL ||
        split_col_obj == NULL ||
        split_val_obj == NULL ||
        left_children_obj == NULL ||
        right_children_obj == NULL ||
        node_mean_obj == NULL)
    {
        Py_XDECREF(X_obj);
        Py_XDECREF(y_obj);
        Py_XDECREF(split_col_obj);
        Py_XDECREF(split_val_obj);
        Py_XDECREF(left_children_obj);
        Py_XDECREF(right_children_obj);
        Py_XDECREF(node_mean_obj);
        return NULL;
    }
    // cast data sections of numpy arrays to plain C pointers
    // this assumes the arrays are C-order, aligned, non-strided
    uint8_t *  restrict X              = PyArray_DATA((PyArrayObject *) X_obj);
    double *   restrict y              = PyArray_DATA((PyArrayObject *) y_obj);
    uint64_t * restrict split_col      = PyArray_DATA((PyArrayObject *) split_col_obj);
    uint8_t *  restrict split_val      = PyArray_DATA((PyArrayObject *) split_val_obj);
    uint16_t * restrict left_children  = PyArray_DATA((PyArrayObject *) left_children_obj);
    uint16_t * restrict right_children = PyArray_DATA((PyArrayObject *) right_children_obj);
    double *   restrict node_means     = PyArray_DATA((PyArrayObject *) node_mean_obj);

    const uint64_t rows = (uint64_t) PyArray_DIM((PyArrayObject *) X_obj, 0);
    const uint64_t cols = (uint64_t) PyArray_DIM((PyArrayObject *) X_obj, 1);
    const uint16_t max_nodes = (uint16_t) PyArray_DIM((PyArrayObject *) left_children_obj, 0);
    const int vals = 256;
    const uint64_t block_size = 16384;
    const uint64_t blocks = rows / block_size;

    // the node index each row is assigned to
    uint16_t * restrict memberships = calloc(rows, sizeof(uint16_t));

    uint64_t * restrict counts = calloc(max_nodes * cols * vals, sizeof(uint64_t));
    double * restrict sums = calloc(max_nodes * cols * vals, sizeof(double));
    double * restrict sum_sqs = calloc(max_nodes * cols * vals, sizeof(double));

    if (memberships == NULL || counts == NULL || sums == NULL || sum_sqs == NULL) {
        if (memberships != NULL) free(memberships);
        if (counts != NULL) free(counts);
        if (sums != NULL) free(sums);
        if (sum_sqs != NULL) free(sum_sqs);
        Py_DECREF(X_obj);
        Py_DECREF(y_obj);
        Py_DECREF(split_col_obj);
        Py_DECREF(split_val_obj);
        Py_DECREF(left_children_obj);
        Py_DECREF(right_children_obj);
        Py_DECREF(node_mean_obj);
        return NULL;
    }

    uint16_t node_count = 1;
    uint16_t done_count = 0;

    double   node_scores  [max_nodes];
    uint64_t node_counts  [max_nodes];
    double   node_sums    [max_nodes];
    double   node_sum_sqs [max_nodes];
    bool     should_split [max_nodes];

    for (uint16_t n = 0; n < max_nodes; n++) {
        node_scores[n] = DBL_MAX;
        node_counts[n] = 0;
        node_sums[n] = 0.0;
        node_sum_sqs[n] = 0.0;
        should_split[n] = false;
    }

    // find the baseline of the root
    // accumulate in local variables so clang vectorizes
    double root_sums = 0.0;
    double root_sum_sqs = 0.0;
    for (uint64_t r = 0; r < rows; r++) {
        root_sums += y[r];
        root_sum_sqs += y[r] * y[r];
    }
    const double root_var = (root_sum_sqs / rows) - (root_sums / rows) * (root_sums / rows);
    const double penalty = root_var * smooth_factor;
    node_counts[0] = rows;
    node_sums[0] = root_sums;
    node_sum_sqs[0] = root_sum_sqs;
    node_scores[0] = root_var;
    // printf("root_var = %f, penalty = %f\n", root_var, penalty);

    // TODO worry about casting

    while (node_count < max_nodes - 1 && done_count < node_count) {
        // build stats
        #pragma omp parallel for collapse(3)
        for (uint64_t b = 0; b < blocks; b++) {
            for (uint64_t c = 0; c < cols; c++) {
                for (uint64_t n = done_count; n < node_count; n++) {
                    uint64_t block_counts [vals] = {0};
                    double block_sums     [vals] = {0.0};
                    double block_sum_sqs  [vals] = {0.0};

                    for (uint64_t r = b * block_size; r < (b + 1) * block_size; r++) {
                        if (memberships[r] == n) {
                            uint8_t v = X[r * cols + c];
                            block_counts[v]++;
                            block_sums[v] += y[r];
                            block_sum_sqs[v] += y[r] * r[y];
                        }
                    }
                    for (int v = 0; v < vals; v++) {
                        uint64_t global_idx = n*cols*vals + c*vals + v;
                        counts[global_idx] += block_counts[v];
                        sums[global_idx] += block_sums[v];
                        sum_sqs[global_idx] += block_sum_sqs[v];
                    }
                }
            }
        }
        // count the remainders
        #pragma omp parallel for
        for (uint64_t c = 0; c < cols; c++) {
            for (uint64_t r = blocks * block_size; r < rows; r++) {
                uint16_t n = memberships[r];
                uint8_t v = X[r * cols + c];
                uint64_t idx = n*cols*vals + c*vals + v;
                counts[idx]++;
                sums[idx] += y[r];
                sum_sqs[idx] += y[r] * y[r];
            }
        }

        // find splits
        #pragma omp parallel for
        for (uint64_t c = 0; c < cols; c++) {
            for (uint16_t n = done_count; n < node_count; n++) {
                // running sums from the left side
                uint64_t left_count = 0;
                double left_sum = 0.0;
                double left_sum_sqs = 0.0;

                uint64_t right_count = node_counts[n];
                double right_sum = node_sums[n];
                double right_sum_sqs = node_sum_sqs[n];

                // track the best split in this column separately
                // so we don't need to sync threads until the end
                uint8_t col_split_val = 0;
                double col_split_score = DBL_MAX;

                // evaluate each possible splitting point
                for (int v = 0; v < vals; v++) {
                    int idx = n*cols*vals + c*vals + v;
                    if (counts[idx] == 0) {
                        continue;
                    }

                    left_count += counts[idx];
                    left_sum += sums[idx];
                    left_sum_sqs += sum_sqs[idx];
                    right_count -= counts[idx];
                    right_sum -= sums[idx];
                    right_sum_sqs -= sum_sqs[idx];

                    if (right_count == 0) {
                        break;
                    }
                    // "smooth" each variance by adding smooth_factor datapoints from root
                    // and take weight average of left & right
                    // but we can cancel some counts out, and drop some constants
                    double left_var = left_sum_sqs - (left_sum * left_sum / left_count);
                    double right_var = right_sum_sqs - (right_sum * right_sum / right_count);
                    double score = (left_var + right_var + penalty) / node_counts[n];

                    // printf("val = %d left_var = %f right_var = %f score = %f\n", v, left_var, right_var, score);

                    if (score < col_split_score) {
                        col_split_score = score;
                        col_split_val = v;
                    }
                }

                #pragma omp critical
                {
                    if (col_split_score < node_scores[n]) {
                        node_scores[n] = col_split_score;
                        split_col[n] = c;
                        split_val[n] = col_split_val;
                        should_split[n] = true;
                    }

                }
            }
        }

        // finished choosing splits
        // update node metadata for new splits
        int new_node_count = node_count;
        for (uint16_t n = 0; n < node_count; n++) {
            if (should_split[n] && new_node_count <= max_nodes - 2) {
                // make the split
                left_children[n] = new_node_count;
                right_children[n] = new_node_count + 1;
                node_scores[left_children[n]] = node_scores[n];
                node_scores[right_children[n]] = node_scores[n];
                new_node_count += 2;
            } else if (should_split[n]) {
                // no room; abort the split
                should_split[n] = false;
            }
        }


        for (uint64_t r = 0; r < rows; r++) {
            uint16_t old_n = memberships[r];
            if (should_split[old_n]) {
                uint8_t val = X[r * cols + split_col[old_n]];
                uint16_t n = (val <= split_val[old_n]) ? left_children[old_n] : right_children[old_n];
                memberships[r] = n;
                node_counts[n]++;
                node_sums[n] += y[r];
                node_sum_sqs[n] += y[r] * y[r];
            }
        }

        done_count = node_count;
        node_count = new_node_count;
        for (uint16_t n = 0; n < node_count; n++) {;
            should_split[n] = false;
        }
    }

    // finally, calculate the mean at each leaf node
    for (uint16_t n = 0; n < node_count; n++) {
        if (node_counts[n] > 0) {
            node_means[n] = node_sums[n] / node_counts[n];
        }
    }

    free(memberships);
    free(counts);
    free(sums);
    free(sum_sqs);
    Py_DECREF(X_obj);
    Py_DECREF(y_obj);
    Py_DECREF(split_col_obj);
    Py_DECREF(split_val_obj);
    Py_DECREF(left_children_obj);
    Py_DECREF(right_children_obj);
    Py_DECREF(node_mean_obj);
    return Py_BuildValue("i", node_count);
}


static PyObject* eval_tree(PyObject *dummy, PyObject *args)
{
    PyObject *X_arg, *split_col_arg, *split_val_arg, *left_children_arg, *right_children_arg, *node_mean_arg;
    PyObject *out_arg;

    // parse input arguments
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!",
        &PyArray_Type, &X_arg,
        &PyArray_Type, &split_col_arg,
        &PyArray_Type, &split_val_arg,
        &PyArray_Type, &left_children_arg,
        &PyArray_Type, &right_children_arg,
        &PyArray_Type, &node_mean_arg,
        &PyArray_Type, &out_arg)) return NULL;

    PyObject *X_obj = PyArray_FROM_OTF(X_arg, NPY_UINT8, NPY_ARRAY_IN_ARRAY);
    PyObject *split_col_obj = PyArray_FROM_OTF(split_col_arg, NPY_UINT64, NPY_ARRAY_IN_ARRAY);
    PyObject *split_val_obj = PyArray_FROM_OTF(split_val_arg, NPY_UINT8, NPY_ARRAY_IN_ARRAY);
    PyObject *left_children_obj = PyArray_FROM_OTF(left_children_arg, NPY_UINT16, NPY_ARRAY_IN_ARRAY);
    PyObject *right_children_obj = PyArray_FROM_OTF(right_children_arg, NPY_UINT16, NPY_ARRAY_IN_ARRAY);
    PyObject *node_mean_obj = PyArray_FROM_OTF(node_mean_arg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyObject *out_obj = PyArray_FROM_OTF(out_arg, NPY_DOUBLE, NPY_ARRAY_OUT_ARRAY);

    if (X_obj == NULL ||
        split_col_obj == NULL ||
        split_val_obj == NULL ||
        left_children_obj == NULL ||
        right_children_obj == NULL ||
        node_mean_obj == NULL ||
        out_obj == NULL)
    {
        Py_XDECREF(X_obj);
        Py_XDECREF(split_col_obj);
        Py_XDECREF(split_val_obj);
        Py_XDECREF(left_children_obj);
        Py_XDECREF(right_children_obj);
        Py_XDECREF(node_mean_obj);
        Py_XDECREF(out_obj);
        return NULL;
    }
    // cast data sections of numpy arrays to plain C pointers
    // this assumes the arrays are C-order, aligned, non-strided
    uint8_t *  restrict X              = PyArray_DATA((PyArrayObject *) X_obj);
    uint64_t * restrict split_col      = PyArray_DATA((PyArrayObject *) split_col_obj);
    uint8_t *  restrict split_val      = PyArray_DATA((PyArrayObject *) split_val_obj);
    uint16_t * restrict left_children  = PyArray_DATA((PyArrayObject *) left_children_obj);
    uint16_t * restrict right_children = PyArray_DATA((PyArrayObject *) right_children_obj);
    double *   restrict node_means     = PyArray_DATA((PyArrayObject *) node_mean_obj);
    double *   restrict out            = PyArray_DATA((PyArrayObject *) out_obj);

    const uint64_t rows = (uint64_t) PyArray_DIM((PyArrayObject *) X_obj, 0);
    const uint64_t cols = (uint64_t) PyArray_DIM((PyArrayObject *) X_obj, 1);

    #pragma omp parallel for
    for (uint64_t r = 0; r < rows; r++) {
        uint16_t n = 0;
        uint16_t left;
        while ((left = left_children[n])) {
            uint8_t val = X[r*cols + split_col[n]];
            n = (val <= split_val[n]) ? left : right_children[n];
        }
        out[r] = node_means[n];
    }
    Py_DECREF(X_obj);
    Py_DECREF(split_col_obj);
    Py_DECREF(split_val_obj);
    Py_DECREF(left_children_obj);
    Py_DECREF(right_children_obj);
    Py_DECREF(node_mean_obj);
    Py_DECREF(out_obj);
    Py_RETURN_NONE;
}


static PyObject* apply_bins(PyObject *dummy, PyObject *args)
{
    PyObject *X_arg, *bins_arg, *out_arg;

    // parse input arguments
    if (!PyArg_ParseTuple(args, "O!O!O!",
        &PyArray_Type, &X_arg,
        &PyArray_Type, &bins_arg,
        &PyArray_Type, &out_arg)) return NULL;

    PyObject *X_obj    = PyArray_FROM_OTF(X_arg,    NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyObject *bins_obj = PyArray_FROM_OTF(bins_arg, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyObject *out_obj  = PyArray_FROM_OTF(out_arg,  NPY_UINT8,   NPY_ARRAY_OUT_ARRAY);

    if (X_obj == NULL || bins_obj == NULL || out_obj == NULL) {
        Py_XDECREF(X_obj);
        Py_XDECREF(bins_obj);
        Py_XDECREF(out_obj);
        return NULL;
    }
    float *   restrict X    = PyArray_DATA((PyArrayObject *) X_obj);
    float *   restrict bins = PyArray_DATA((PyArrayObject *) bins_obj);
    uint8_t * restrict out  = PyArray_DATA((PyArrayObject *) out_obj);

    const uint64_t rows = PyArray_DIM((PyArrayObject *) X_obj, 0);
    const uint64_t cols = PyArray_DIM((PyArrayObject *) X_obj, 1);
    const uint64_t seps = 255;
    const uint8_t max_val = 255;

    // bins is a (cols, 255) array separating X into 256 values
    // for binning the data in X from float => uint8
    //
    // such that (floats in bucket 0) <= bins[c, 0] < (floats in bucket 1) <= bins[c, 1] ...
    //
    // instead of searching for the first bin that a value falls into
    // we count 255 - (the number of seps the is less than);
    // this is easier to vectorize
    //
    #pragma omp parallel for
    for (uint64_t c = 0; c < cols; c++) {
        for (uint64_t r = 0; r < rows; r++) {
            float val = X[r*cols + c];
            uint8_t sum = 0; // simple accumulator so clang can vectorize
            for (uint64_t v = 0; v < seps; v++) {
                sum += (val <= bins[c*seps + v]);
            }
            out[r*cols + c] = max_val - sum;
        }
    }
    Py_DECREF(X_obj);
    Py_DECREF(bins_obj);
    Py_DECREF(out_obj);
    Py_RETURN_NONE;
}

static struct PyModuleDef mod_def =
{
    PyModuleDef_HEAD_INIT,
    "tree",
    "",
    -1,
    Methods
};

PyMODINIT_FUNC
PyInit_tree(void)
{
    import_array();
    return PyModule_Create(&mod_def);
}
