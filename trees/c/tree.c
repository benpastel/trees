#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include <arrayobject.h>
#include <stdbool.h>
#include <float.h>
// #include <time.h>
#include <sys/time.h>
#include <omp.h>

static PyObject* build_tree(PyObject *self, PyObject *args);
static PyObject* eval_tree(PyObject *self, PyObject *args);
static PyObject* apply_bins(PyObject *self, PyObject *args);

static PyMethodDef Methods[] = {
    {"build_tree", build_tree, METH_VARARGS, ""},
    {"eval_tree", eval_tree, METH_VARARGS, ""},
    {"apply_bins", apply_bins, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}
};

static float msec(struct timeval t0, struct timeval t1)
{
    return (t1.tv_sec - t0.tv_sec) * 1000.0f + (t1.tv_usec - t0.tv_usec) / 1000.0f;
}

#define VERBOSE 0

static PyObject* build_tree(PyObject *dummy, PyObject *args)
{
    PyObject *XT_bin_arg, *XT_reg_arg, *y_arg;
    PyObject *split_col_arg, *split_lo_arg, *split_hi_arg, *coefs_arg;
    PyObject *left_childs_arg, *mid_childs_arg, *right_childs_arg, *node_mean_arg;
    PyObject *preds_arg;
    double smooth_factor_arg;
    int max_depth_arg;

    struct timeval total_start;
    struct timeval init_end;
    struct timeval stat_start;
    struct timeval split_start;
    struct timeval split_end;
    struct timeval post_start;
    struct timeval total_end;
    long stat_ms = 0;
    long split_ms = 0;
    gettimeofday(&total_start, NULL);

    // parse input arguments
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!O!O!O!O!O!di",
        &PyArray_Type, &XT_bin_arg,
        &PyArray_Type, &XT_reg_arg,
        &PyArray_Type, &y_arg,
        &PyArray_Type, &split_col_arg,
        &PyArray_Type, &split_lo_arg,
        &PyArray_Type, &split_hi_arg,
        &PyArray_Type, &coefs_arg,
        &PyArray_Type, &left_childs_arg,
        &PyArray_Type, &mid_childs_arg,
        &PyArray_Type, &right_childs_arg,
        &PyArray_Type, &node_mean_arg,
        &PyArray_Type, &preds_arg,
        &smooth_factor_arg,
        &max_depth_arg)) return NULL;
    const double smooth_factor = smooth_factor_arg;
    const int max_depth = max_depth_arg;

    PyObject *XT_bin_obj = PyArray_FROM_OTF(XT_bin_arg, NPY_UINT8, NPY_ARRAY_IN_ARRAY);
    PyObject *XT_reg_obj = PyArray_FROM_OTF(XT_reg_arg, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyObject *y_obj = PyArray_FROM_OTF(y_arg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    PyObject *split_col_obj = PyArray_FROM_OTF(split_col_arg, NPY_UINT64, NPY_ARRAY_OUT_ARRAY);
    PyObject *split_lo_obj = PyArray_FROM_OTF(split_lo_arg, NPY_UINT8, NPY_ARRAY_OUT_ARRAY);
    PyObject *split_hi_obj = PyArray_FROM_OTF(split_hi_arg, NPY_UINT8, NPY_ARRAY_OUT_ARRAY);
    PyObject *coefs_obj = PyArray_FROM_OTF(coefs_arg, NPY_FLOAT32, NPY_ARRAY_OUT_ARRAY);

    PyObject *left_childs_obj = PyArray_FROM_OTF(left_childs_arg, NPY_UINT16, NPY_ARRAY_OUT_ARRAY);
    PyObject *mid_childs_obj = PyArray_FROM_OTF(mid_childs_arg, NPY_UINT16, NPY_ARRAY_OUT_ARRAY);
    PyObject *right_childs_obj = PyArray_FROM_OTF(right_childs_arg, NPY_UINT16, NPY_ARRAY_OUT_ARRAY);
    PyObject *node_mean_obj = PyArray_FROM_OTF(node_mean_arg, NPY_DOUBLE, NPY_ARRAY_OUT_ARRAY);
    PyObject *preds_obj = PyArray_FROM_OTF(preds_arg, NPY_DOUBLE, NPY_ARRAY_OUT_ARRAY);

    if (XT_bin_obj == NULL ||
        XT_reg_obj == NULL ||
        y_obj == NULL ||
        split_col_obj == NULL ||
        split_lo_obj == NULL ||
        split_hi_obj == NULL ||
        coefs_obj == NULL ||
        left_childs_obj == NULL ||
        mid_childs_obj == NULL ||
        right_childs_obj == NULL ||
        node_mean_obj == NULL ||
        preds_obj == NULL)
    {
        Py_XDECREF(XT_bin_obj);
        Py_XDECREF(XT_reg_obj);
        Py_XDECREF(y_obj);
        Py_XDECREF(split_col_obj);
        Py_XDECREF(split_lo_obj);
        Py_XDECREF(split_hi_obj);
        Py_XDECREF(coefs_obj);
        Py_XDECREF(left_childs_obj);
        Py_XDECREF(mid_childs_obj);
        Py_XDECREF(right_childs_obj);
        Py_XDECREF(node_mean_obj);
        Py_XDECREF(preds_obj);
        return NULL;
    }

    // convert data sections of numpy arrays to plain C pointers
    // the arrays should already be C-order, aligned, non-strided
    // otherwise PyArray_DATA will copy
    uint8_t *  restrict XT_bin       = PyArray_DATA((PyArrayObject *) XT_bin_obj);
    float *    restrict XT_reg       = PyArray_DATA((PyArrayObject *) XT_reg_obj);
    double *   restrict y            = PyArray_DATA((PyArrayObject *) y_obj);
    uint64_t * restrict split_col    = PyArray_DATA((PyArrayObject *) split_col_obj);
    uint8_t *  restrict split_lo     = PyArray_DATA((PyArrayObject *) split_lo_obj);
    uint8_t *  restrict split_hi     = PyArray_DATA((PyArrayObject *) split_hi_obj);
    float *    restrict coefs        = PyArray_DATA((PyArrayObject *) coefs_obj);
    uint16_t * restrict left_childs  = PyArray_DATA((PyArrayObject *) left_childs_obj);
    uint16_t * restrict mid_childs   = PyArray_DATA((PyArrayObject *) mid_childs_obj);
    uint16_t * restrict right_childs = PyArray_DATA((PyArrayObject *) right_childs_obj);
    double *   restrict node_means   = PyArray_DATA((PyArrayObject *) node_mean_obj);
    double *   restrict preds        = PyArray_DATA((PyArrayObject *) preds_obj);

    const uint64_t rows = (uint64_t) PyArray_DIM((PyArrayObject *) XT_bin_obj, 1);
    const uint64_t cols = (uint64_t) PyArray_DIM((PyArrayObject *) XT_bin_obj, 0);
    const uint16_t max_nodes = (uint16_t) PyArray_DIM((PyArrayObject *) left_childs_obj, 0);
    const uint64_t vals = 256;

    uint16_t * restrict memberships = calloc(rows, sizeof(uint16_t));

    uint16_t node_count = 1;
    uint16_t done_count = 0;

    double   node_scores  [max_nodes];
    uint64_t node_counts  [max_nodes];
    double   node_sums    [max_nodes];
    double   node_sum_sqs [max_nodes];
    bool     should_split [max_nodes];
    omp_lock_t node_locks [max_nodes];

    uint64_t left_counts [max_nodes];
    uint64_t mid_counts  [max_nodes];
    uint64_t right_counts[max_nodes];

    double left_sums [max_nodes];
    double mid_sums  [max_nodes];
    double right_sums [max_nodes];

    double left_sum_sqs[max_nodes];
    double mid_sum_sqs[max_nodes];
    double right_sum_sqs[max_nodes];

    double left_vars  [max_nodes];
    double mid_vars   [max_nodes];
    double right_vars [max_nodes];

    for (uint16_t n = 0; n < max_nodes; n++) {
        node_scores[n] = DBL_MAX;
        node_counts[n] = 0;
        node_sums[n] = 0.0;
        node_sum_sqs[n] = 0.0;
        should_split[n] = false;
        left_counts[n] = 0;
        mid_counts[n] = 0;
        right_counts[n] = 0;
        omp_init_lock(&node_locks[n]);
    }

    // find the baseline of the root
    // accumulate in local variables so clang vectorizes
    double root_sum = 0.0;
    double root_sum_sq = 0.0;
    for (uint64_t r = 0; r < rows; r++) {
        root_sum += y[r];
        root_sum_sq += y[r] * y[r];
    }
    const double root_var = (root_sum_sq / rows) - (root_sum / rows) * (root_sum / rows);
    const double penalty = root_var * smooth_factor;
    node_counts[0] = rows;
    node_sums[0] = root_sum;
    node_sum_sqs[0] = root_sum_sq;
    node_scores[0] = root_var;

    gettimeofday(&init_end, NULL);

    int depth = 0;
    while (depth++ < max_depth && node_count < max_nodes - 2 && done_count < node_count) {

        gettimeofday(&stat_start, NULL);

        #pragma omp parallel for
        for (uint64_t c = 0; c < cols; c++) {
            uint64_t * restrict counts = calloc(node_count * vals, sizeof(uint64_t));
            double * restrict sums = calloc(node_count * vals, sizeof(double));
            double * restrict sum_sqs = calloc(node_count * vals, sizeof(double));

            // stats
            for (uint64_t r = 0; r < rows; r++) {
                uint32_t v = XT_bin[c*rows + r];
                uint32_t n = memberships[r];
                uint32_t idx = n*vals + v;
                counts[idx]++;
                sums[idx] += y[r];
                sum_sqs[idx] += y[r] * y[r];
            }

            // splits
            for (uint16_t n = done_count; n < node_count; n++) {
                // min leaf size is 1 for left & right, 0 for mid
                if (node_counts[n] < 2) continue;

                // running sums from the left side
                uint64_t left_count = 0;
                double left_sum = 0.0;
                double left_sum_sq = 0.0;

                // evaluate each possible splitting point
                for (uint64_t lo = 0; lo < vals - 2; lo++) {
                    uint64_t lo_i = n*vals + lo;

                    // force non-empty left split
                    if (counts[lo_i] == 0) continue;

                    left_count += counts[lo_i];
                    left_sum += sums[lo_i];
                    left_sum_sq += sum_sqs[lo_i];
                    double left_var = left_sum_sq - (left_sum * left_sum / left_count);

                    uint64_t mid_count = 0;
                    double mid_sum = 0.0;
                    double mid_sum_sq = 0.0;

                    // allow mid split to be empty
                    // so start at hi = lo and allow mid_count = 0
                    for (uint64_t hi = lo; hi < vals - 1; hi++) {
                        uint64_t hi_i = n*vals + hi;

                        if (hi > lo) {
                            mid_count += counts[hi_i];
                            mid_sum += sums[hi_i];
                            mid_sum_sq += sum_sqs[hi_i];
                        }

                        uint64_t right_count = node_counts[n] - left_count - mid_count;
                        double right_sum = node_sums[n] - left_sum - mid_sum;
                        double right_sum_sq = node_sum_sqs[n] - left_sum_sq - mid_sum_sq;

                        // force non-empty right split
                        if (right_count == 0) break;

                        // weighted average of splits' variance
                        // TODO maybe now we can encourage empty splits by penalty * nonempties?
                        // TODO try k splits for different k?
                        double mid_var = (mid_count == 0) ? 0 : mid_sum_sq - (mid_sum * mid_sum / mid_count);
                        double right_var = right_sum_sq - (right_sum * right_sum / right_count);
                        double score = (left_var + mid_var + right_var + penalty) / node_counts[n];

                        // node_scores[n] may be stale, but it only decreases
                        // first check without the lock for efficiency
                        if (score < node_scores[n]) {
                            // now check with the lock for correctness
                            omp_set_lock(&node_locks[n]);
                            if (score < node_scores[n]) {
                                node_scores[n] = score;
                                split_col[n] = c;
                                split_lo[n] = lo;
                                split_hi[n] = hi;

                                left_counts[n] = left_count;
                                mid_counts[n] = mid_count;
                                right_counts[n] = right_count;

                                left_sums[n] = left_sum;
                                mid_sums[n] = mid_sum;
                                right_sums[n] = right_sum;

                                left_sum_sqs[n] = left_sum_sq;
                                mid_sum_sqs[n] = mid_sum_sq;
                                right_sum_sqs[n] = right_sum_sq;

                                left_vars[n] = left_var;
                                mid_vars[n] = mid_var;
                                right_vars[n] = right_var;

                                should_split[n] = true;
                            }
                            omp_unset_lock(&node_locks[n]);
                        }
                        // if (hi < 5)
                        //     printf("    %llu split=(%llu,%llu) var=(%f,%f,%f) score=%f\n", c, lo, hi, left_var, mid_var, right_var, score);
                    }
                }
            }
            free(counts);
            free(sums);
            free(sum_sqs);
        }
        gettimeofday(&split_start, NULL);
        stat_ms += msec(stat_start, split_start);

        // we've finised choosing the splits

        // update node metadata for the splits
        int new_node_count = node_count;
        for (uint16_t n = 0; n < node_count; n++) {
            if (should_split[n] && new_node_count <= max_nodes - 3) {
                left_childs[n] = new_node_count;
                mid_childs[n] = new_node_count + 1;
                right_childs[n] = new_node_count + 2;

                node_scores[left_childs[n]] = left_vars[n] / left_counts[n];
                node_scores[mid_childs[n]] = (mid_counts[n] == 0) ? 0 : mid_vars[n] / mid_counts[n];
                node_scores[right_childs[n]] = right_vars[n] / right_counts[n];

                node_counts[left_childs[n]] = left_counts[n];
                node_counts[mid_childs[n]] = mid_counts[n];
                node_counts[right_childs[n]] = right_counts[n];

                node_sums[left_childs[n]] = left_sums[n];
                node_sums[mid_childs[n]] = mid_sums[n];
                node_sums[right_childs[n]] = right_sums[n];

                node_sum_sqs[left_childs[n]] = left_sum_sqs[n];
                node_sum_sqs[mid_childs[n]] = mid_sum_sqs[n];
                node_sum_sqs[right_childs[n]] = right_sum_sqs[n];

                new_node_count += 3;
            } else if (should_split[n]) {
                // no room; abort the split
                should_split[n] = false;
            }
        }

        // update memberships
        #pragma omp parallel for
        for (uint64_t r = 0; r < rows; r++) {
            uint16_t n = memberships[r];
            if (!should_split[n]) continue;

            uint8_t v = XT_bin[split_col[n]*rows + r];

            uint16_t child = (
                v <= split_lo[n] ? left_childs[n] :
                v <= split_hi[n] ? mid_childs[n] :
                right_childs[n]);

            memberships[r] = child;
        }
        done_count = node_count;
        node_count = new_node_count;
        for (uint16_t n = 0; n < node_count; n++) {;
            should_split[n] = false;
        }

        gettimeofday(&split_end, NULL);
        split_ms += msec(split_start, split_end);
    }
    gettimeofday(&post_start, NULL);

    // calculate the mean at each leaf node
    for (uint16_t n = 0; n < node_count; n++) {
        if (node_counts[n] > 0) {
            node_means[n] = node_sums[n] / node_counts[n];
        }
    }
    // find the prediction for each leaf
    #pragma omp parallel for
    for (uint64_t r = 0; r < rows; r++) {
        preds[r] = node_means[memberships[r]];
    }

    for (uint16_t n = 0; n < max_nodes; n++) {
        omp_destroy_lock(&node_locks[n]);
    }

    free(memberships);
    Py_DECREF(XT_bin_obj);
    Py_DECREF(XT_reg_obj);
    Py_DECREF(y_obj);
    Py_DECREF(split_col_obj);
    Py_DECREF(split_lo_obj);
    Py_DECREF(split_hi_obj);
    Py_DECREF(coefs_obj);
    Py_DECREF(left_childs_obj);
    Py_DECREF(mid_childs_obj);
    Py_DECREF(right_childs_obj);
    Py_DECREF(node_mean_obj);
    Py_DECREF(preds_obj);

    gettimeofday(&total_end, NULL);
#if VERBOSE
    printf("Fit depth %d / %d nodes: %.1f total, %.1f init, %.1f stats, %.1f splits, %.1f post\n",
        depth,
        node_count,
        ((float) msec(total_start, total_end)) / 1000.0,
        ((float) msec(total_start, init_end)) / 1000.0,
        ((float) stat_ms) / 1000.0,
        ((float) split_ms) / 1000.0,
        ((float) msec(post_start, total_end)) / 1000.0);
#endif

    return Py_BuildValue("i", node_count);
}

static PyObject* eval_tree(PyObject *dummy, PyObject *args)
{
    PyObject *X_arg;
    PyObject *split_col_arg, *split_lo_arg, *split_hi_arg, *coefs_arg;
    PyObject *left_childs_arg, *mid_childs_arg, *right_childs_arg, *node_mean_arg;
    PyObject *out_arg;

    struct timeval total_start;
    struct timeval loop_start;
    struct timeval loop_stop;
    struct timeval total_stop;
    gettimeofday(&total_start, NULL);

    // parse input arguments
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!O!O!O!",
        &PyArray_Type, &X_arg,
        &PyArray_Type, &split_col_arg,
        &PyArray_Type, &split_lo_arg,
        &PyArray_Type, &split_hi_arg,
        &PyArray_Type, &coefs_arg,
        &PyArray_Type, &left_childs_arg,
        &PyArray_Type, &mid_childs_arg,
        &PyArray_Type, &right_childs_arg,
        &PyArray_Type, &node_mean_arg,
        &PyArray_Type, &out_arg)) return NULL;

    PyObject *X_obj = PyArray_FROM_OTF(X_arg, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);

    PyObject *split_col_obj = PyArray_FROM_OTF(split_col_arg, NPY_UINT64, NPY_ARRAY_IN_ARRAY);
    PyObject *split_lo_obj = PyArray_FROM_OTF(split_lo_arg, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyObject *split_hi_obj = PyArray_FROM_OTF(split_hi_arg, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyObject *coefs_obj = PyArray_FROM_OTF(coefs_arg, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);

    PyObject *left_childs_obj = PyArray_FROM_OTF(left_childs_arg, NPY_UINT16, NPY_ARRAY_IN_ARRAY);
    PyObject *mid_childs_obj = PyArray_FROM_OTF(mid_childs_arg, NPY_UINT16, NPY_ARRAY_IN_ARRAY);
    PyObject *right_childs_obj = PyArray_FROM_OTF(right_childs_arg, NPY_UINT16, NPY_ARRAY_IN_ARRAY);
    PyObject *node_mean_obj = PyArray_FROM_OTF(node_mean_arg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    PyObject *out_obj = PyArray_FROM_OTF(out_arg, NPY_DOUBLE, NPY_ARRAY_OUT_ARRAY);

    if (X_obj == NULL ||
        split_col_obj == NULL ||
        split_lo_obj == NULL ||
        split_hi_obj == NULL ||
        coefs_obj == NULL |
        left_childs_obj == NULL ||
        mid_childs_obj == NULL ||
        right_childs_obj == NULL ||
        node_mean_obj == NULL ||
        out_obj == NULL)
    {
        Py_XDECREF(X_obj);
        Py_XDECREF(split_col_obj);
        Py_XDECREF(split_lo_obj);
        Py_XDECREF(split_hi_obj);
        Py_XDECREF(coefs_obj);
        Py_XDECREF(left_childs_obj);
        Py_XDECREF(mid_childs_obj);
        Py_XDECREF(right_childs_obj);
        Py_XDECREF(node_mean_obj);
        Py_XDECREF(out_obj);
        return NULL;
    }
    // cast data sections of numpy arrays to plain C pointers
    // this assumes the arrays are C-order, aligned, non-strided
    float *    restrict X            = PyArray_DATA((PyArrayObject *) X_obj);
    uint64_t * restrict split_col    = PyArray_DATA((PyArrayObject *) split_col_obj);
    float *    restrict split_lo     = PyArray_DATA((PyArrayObject *) split_lo_obj);
    float *    restrict split_hi     = PyArray_DATA((PyArrayObject *) split_hi_obj);
    float *    restrict coefs        = PyArray_DATA((PyArrayObject *) coefs_obj);
    uint16_t * restrict left_childs  = PyArray_DATA((PyArrayObject *) left_childs_obj);
    uint16_t * restrict mid_childs   = PyArray_DATA((PyArrayObject *) mid_childs_obj);
    uint16_t * restrict right_childs = PyArray_DATA((PyArrayObject *) right_childs_obj);
    double *   restrict node_means   = PyArray_DATA((PyArrayObject *) node_mean_obj);
    double *   restrict out          = PyArray_DATA((PyArrayObject *) out_obj);

    const uint64_t rows = (uint64_t) PyArray_DIM((PyArrayObject *) X_obj, 0);
    const uint64_t cols = (uint64_t) PyArray_DIM((PyArrayObject *) X_obj, 1);

    gettimeofday(&loop_start, NULL);
    #pragma omp parallel for
    for (uint64_t r = 0; r < rows; r++) {
        double row_val = 0.0;
        uint16_t n = 0;
        uint16_t left;
        while ((left = left_childs[n])) {
            float split_val = X[r*cols + split_col[n]];
            row_val += coefs[n] * split_val;

            n = (split_val <= split_lo[n] || coefs[n]) ? left :
                (split_val <= split_hi[n]) ? mid_childs[n] :
                right_childs[n];
        }
        out[r] = row_val + node_means[n];

    }
    gettimeofday(&loop_stop, NULL);

    Py_DECREF(X_obj);
    Py_DECREF(split_col_obj);
    Py_DECREF(split_lo_obj);
    Py_DECREF(split_hi_obj);
    Py_DECREF(coefs_obj);
    Py_DECREF(left_childs_obj);
    Py_DECREF(mid_childs_obj);
    Py_DECREF(right_childs_obj);
    Py_DECREF(node_mean_obj);
    Py_DECREF(out_obj);

    gettimeofday(&total_stop, NULL);
#if VERBOSE
    printf("  eval: %.1f (%.1f loop)\n",
        ((float) msec(total_start, total_stop)) / 1000.0,
        ((float) msec(loop_start, loop_stop)) / 1000.0);
#endif

    Py_RETURN_NONE;
}


static PyObject* apply_bins(PyObject *dummy, PyObject *args)
{
    PyObject *X_arg, *bins_arg, *out_arg;

    struct timeval total_start;
    struct timeval loop_start;
    struct timeval loop_stop;
    struct timeval total_stop;
    gettimeofday(&total_start, NULL);

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
    // we also transpose (rows, cols) => (cols, rows) when we write to out
    //
    gettimeofday(&loop_start, NULL);
    #pragma omp parallel for
    for (uint64_t r = 0; r < rows; r++) {
        for (uint64_t c = 0; c < cols; c++) {
            float val = X[r*cols + c];
            uint8_t sum = 0; // simple accumulator so clang can vectorize
            for (uint64_t v = 0; v < seps; v++) {
                sum += (val <= bins[c*seps + v]);
            }
            out[c*rows + r] = max_val - sum;
        }
    }
    gettimeofday(&loop_stop, NULL);
    Py_DECREF(X_obj);
    Py_DECREF(bins_obj);
    Py_DECREF(out_obj);

    gettimeofday(&total_stop, NULL);
#if VERBOSE
    printf("apply bins: %.1f (%.1f loop)\n",
        ((float) msec(total_start, total_stop)) / 1000.0,
        ((float) msec(loop_start, loop_stop)) / 1000.0);
#endif

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
