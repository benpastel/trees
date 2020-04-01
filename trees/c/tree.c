#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include <arrayobject.h>
#include <stdbool.h>
#include <float.h>

static PyObject* build_tree(PyObject *self, PyObject *args);
static PyObject* eval_tree(PyObject *self, PyObject *args);

static PyMethodDef Methods[] = {
    {"build_tree", build_tree, METH_VARARGS, ""},
    {"eval_tree", eval_tree, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}
};


static PyObject* build_tree(PyObject *dummy, PyObject *args)
{
    PyObject *X_arg, *y_arg;
    PyObject *split_col_arg, *split_val_arg, *left_children_arg, *right_children_arg, *node_mean_arg;
    double split_penalty;
    int int_min_leaf_size;

    // parse input arguments
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!di",
        &PyArray_Type, &X_arg,
        &PyArray_Type, &y_arg,
        &PyArray_Type, &split_col_arg,
        &PyArray_Type, &split_val_arg,
        &PyArray_Type, &left_children_arg,
        &PyArray_Type, &right_children_arg,
        &PyArray_Type, &node_mean_arg,
        &split_penalty,
        &int_min_leaf_size)) return NULL;

    // row count needs to be uint64_t for large datasets,
    // so also use it for anything that gets compared to rows
    const uint64_t min_leaf_size = (uint64_t) int_min_leaf_size;

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

    // the node index each row is assigned to
    uint16_t * restrict memberships = calloc(rows, sizeof(uint16_t));
    if (memberships == NULL) {
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

    double node_scores   [max_nodes];
    uint64_t node_counts [max_nodes];
    double node_sums     [max_nodes];
    double node_sum_sqs  [max_nodes];
    bool should_split    [max_nodes];

    for (uint16_t n = 0; n < max_nodes; n++) {
        node_scores[n] = DBL_MAX;
        node_counts[n] = 0;
        node_sums[n] = 0.0;
        node_sum_sqs[n] = 0.0;
        should_split[n] = false;
    }

    // find the baseline of the root
    // accumulate in local variables so clang understands how to vectorize the loop
    double root_sums = 0.0;
    double root_sum_sqs = 0.0;
    for (uint64_t r = 0; r < rows; r++) {
        root_sums += y[r];
        root_sum_sqs += y[r] * y[r];
    }
    node_counts[0] = rows;
    node_sums[0] = root_sums;
    node_sum_sqs[0] = root_sum_sqs;
    node_scores[0] = (root_sum_sqs / rows) - (root_sums / rows) * (root_sums / rows);

    while (node_count < max_nodes - 1 && done_count < node_count) {

        // build stats for all nodes, parellized over columns
        #pragma omp parallel for
        for (uint64_t c = 0; c < cols; c++) {
            // for each node & each unique X value, aggregate stats about y
            uint64_t counts  [node_count * vals];
            double   sums    [node_count * vals];
            double   sum_sqs [node_count * vals];

            memset(counts,  0, sizeof counts);
            memset(sums,    0, sizeof sums);
            memset(sum_sqs, 0, sizeof sum_sqs);

            for (uint64_t r = 0; r < rows; r++) {
                int idx = memberships[r] * vals + X[r * cols + c];
                counts [idx]++;
                sums   [idx] += y[r];
                sum_sqs[idx] += y[r] * y[r];
            }

            // for each node, decide if this column is worth splitting
            for (uint16_t n = done_count; n < node_count; n++) {
                // running sums from the left side
                uint64_t left_count = 0;
                double left_sum = 0.0;
                double left_sum_sqs = 0.0;

                // track the best split in this column separately
                // so we don't need to sync threads until the end
                uint8_t col_split_val = 0;
                double col_split_score = DBL_MAX;

                // evaluate each possible splitting point
                // splits are <= v, so the last val is invalid
                for (int v = 0; v < vals - 1; v++) {
                    int idx = n*vals + v;
                    left_count += counts[idx];
                    left_sum += sums[idx];
                    left_sum_sqs += sum_sqs[idx];

                    uint64_t right_count = node_counts[n] - left_count;

                    if (counts[idx] == 0 || left_count < min_leaf_size || right_count < min_leaf_size) {
                        // not a valid splitting point
                        continue;
                    }

                    double right_sum = node_sums[n] - left_sum;
                    double right_sum_sqs = node_sum_sqs[n] - left_sum_sqs;

                    double left_mean = left_sum / left_count;
                    double right_mean = right_sum / right_count;

                    double left_var = left_sum_sqs / left_count - left_mean * left_mean;
                    double right_var = right_sum_sqs / right_count - right_mean * right_mean;

                    double score = (left_var * left_count + right_var * right_count) / rows;

                    if (score < col_split_score) {
                        col_split_score = score;
                        col_split_val = v;
                    }
                }

                // TODO: also try letting each thread keep a local copy to avoid the sync
                // (although the sync might be good because it forces everyone to iterate X at the same time?)
                #pragma omp critical
                {
                    if (col_split_score + split_penalty < node_scores[n]) {
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

        // update row membership & stats in the nodes that split
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
