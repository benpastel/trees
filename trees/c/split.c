#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include <arrayobject.h>

static PyObject* split(PyObject *self, PyObject *args);

static PyMethodDef Methods[] = {
    {"split", split, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}
};

/*
 * outputs None or (split_col, split_val, split_score)
 */
static PyObject* split(PyObject *dummy, PyObject *args)
{
    PyObject *X_arg=NULL, *y_arg=NULL;
    double max_split_score;
    int int_min_leaf_size;

    if (!PyArg_ParseTuple(args, "O!O!di",
        &PyArray_Type, &X_arg,
        &PyArray_Type, &y_arg,
        &max_split_score,
        &int_min_leaf_size)) return NULL;

    uint64_t min_leaf_size = (uint64_t) int_min_leaf_size;

    PyObject *X = PyArray_FROM_OTF(X_arg, NPY_UINT8, NPY_ARRAY_IN_ARRAY);
    PyObject *y = PyArray_FROM_OTF(y_arg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (X == NULL || y == NULL) {
        Py_XDECREF(X);
        Py_XDECREF(y);
        return NULL;
    }
    uint8_t * restrict X_ptr = (uint8_t *) PyArray_DATA((PyArrayObject *) X);
    double * restrict y_ptr = (double *) PyArray_DATA((PyArrayObject *) y);

    const uint64_t rows = (uint64_t) PyArray_DIM((PyArrayObject *) X, 0);
    const uint64_t cols = (uint64_t) PyArray_DIM((PyArrayObject *) X, 1);
    const int vals = 256;

    int best_split_col = -1;
    uint8_t best_split_val = -1;
    double best_split_score = max_split_score;

    // parallelize over the features
    // so that each thread writes to distinct memory
    #pragma omp parallel for
    for (uint64_t c = 0; c < cols; c++) {

        // for each unique X value, aggregate stats about y
        uint64_t counts [vals];
        double sums [vals];
        double sum_sqs [vals];

        memset(counts, 0, sizeof(counts));
        memset(sums, 0, sizeof(sums));
        memset(sum_sqs, 0, sizeof(sum_sqs));

        for (uint64_t r = 0; r < rows; r++) {
            uint8_t v = X_ptr[r * cols + c];
            counts[v]++;
            sums[v] += y_ptr[r];
            sum_sqs[v] += y_ptr[r] * y_ptr[r];
        }

        // totals
        uint64_t total_count = 0;
        double total_sum = 0;
        double total_sum_sqs = 0;
        for (int v = 0; v < vals; v++) {
            total_count += counts[v];
            total_sum += sums[v];
            total_sum_sqs += sum_sqs[v];
        }

        // running sums from the left side
        uint64_t left_count = 0;
        double left_sum = 0;
        double left_sum_sqs = 0;

        // track the best split in this column separately
        // so we don't need to sync threads until the end
        uint8_t col_split_val = -1;
        double col_split_score = max_split_score;

        // evaluate each possible splitting point
        // splits are <= v, so the last vals is invalid
        for (int v = 0; v < vals - 1; v++) {
            left_count += counts[v];
            left_sum += sums[v];
            left_sum_sqs += sum_sqs[v];

            uint64_t right_count = total_count - left_count;

            if (counts[v] == 0 || left_count < min_leaf_size || right_count < min_leaf_size) {
                // not a valid splitting point
                continue;
            }

            double right_sum = total_sum - left_sum;
            double right_sum_sqs = total_sum_sqs - left_sum_sqs;

            double left_mean = left_sum / left_count;
            double right_mean = right_sum / right_count;

            double left_var = left_sum_sqs / left_count - left_mean * left_mean;
            double right_var = right_sum_sqs / right_count - right_mean * right_mean;

            double score = (left_var * left_count + right_var * right_count) / (2.0 * rows);

            if (score < col_split_score) {
                col_split_score = score;
                col_split_val = v;
            }
        }

        // TODO: also try letting each thread keep a local copy to avoid the sync
        // (although the sync might be good because it forces everyone to iterate X at the same time)
        #pragma omp critical
        {
            if (col_split_score < best_split_score) {
                best_split_col = c;
                best_split_score = col_split_score;
                best_split_val = col_split_val;
            }

        }
    }
    Py_DECREF(X);
    Py_DECREF(y);

    if (best_split_score < max_split_score) {
        return Py_BuildValue("(iid)", best_split_col, best_split_val, best_split_score);
    } else {
        // unable to improve score by splitting
        Py_RETURN_NONE;
    }
}

static struct PyModuleDef mod_def =
{
    PyModuleDef_HEAD_INIT,
    "split",
    "",
    -1,
    Methods
};

PyMODINIT_FUNC
PyInit_split(void)
{
    import_array();
    return PyModule_Create(&mod_def);
}
