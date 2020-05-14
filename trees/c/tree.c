#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include <arrayobject.h>
#include <stdbool.h>
#include <float.h>
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

#define SPLIT_BUF_SIZE 1024
#define MIN_PARALLEL_SPLIT 1024

static PyObject* build_tree(PyObject *dummy, PyObject *args)
{
    PyObject *X_arg, *y_arg;
    PyObject *split_col_arg, *split_lo_arg, *split_hi_arg;
    PyObject *left_childs_arg, *mid_childs_arg, *right_childs_arg, *node_mean_arg;
    PyObject *preds_arg;
    double smooth_factor_arg;
    int max_depth_arg;
    double third_split_penalty_arg;
    int vals_arg;

    struct timeval total_start;
    struct timeval init_end;
    struct timeval stat_start;
    struct timeval choose_split_start;
    struct timeval make_split_start;
    struct timeval split_end;
    struct timeval post_start;
    struct timeval total_end;
    long stat_ms = 0;
    long choose_split_ms = 0;
    long make_split_ms = 0;
    gettimeofday(&total_start, NULL);

    // parse input arguments
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!O!O!O!didi",
        &PyArray_Type, &X_arg,
        &PyArray_Type, &y_arg,
        &PyArray_Type, &split_col_arg,
        &PyArray_Type, &split_lo_arg,
        &PyArray_Type, &split_hi_arg,
        &PyArray_Type, &left_childs_arg,
        &PyArray_Type, &mid_childs_arg,
        &PyArray_Type, &right_childs_arg,
        &PyArray_Type, &node_mean_arg,
        &PyArray_Type, &preds_arg,
        &smooth_factor_arg,
        &max_depth_arg,
        &third_split_penalty_arg,
        &vals_arg)) return NULL;
    const double smooth_factor = smooth_factor_arg;
    const int max_depth = max_depth_arg;
    const double third_split_penalty = third_split_penalty_arg;
    const uint vals = vals_arg;

    PyObject *X_obj = PyArray_FROM_OTF(X_arg, NPY_UINT8, NPY_ARRAY_IN_ARRAY);
    PyObject *y_obj = PyArray_FROM_OTF(y_arg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyObject *split_col_obj = PyArray_FROM_OTF(split_col_arg, NPY_UINT64, NPY_ARRAY_OUT_ARRAY);
    PyObject *split_lo_obj = PyArray_FROM_OTF(split_lo_arg, NPY_UINT8, NPY_ARRAY_OUT_ARRAY);
    PyObject *split_hi_obj = PyArray_FROM_OTF(split_hi_arg, NPY_UINT8, NPY_ARRAY_OUT_ARRAY);
    PyObject *left_childs_obj = PyArray_FROM_OTF(left_childs_arg, NPY_UINT16, NPY_ARRAY_OUT_ARRAY);
    PyObject *mid_childs_obj = PyArray_FROM_OTF(mid_childs_arg, NPY_UINT16, NPY_ARRAY_OUT_ARRAY);
    PyObject *right_childs_obj = PyArray_FROM_OTF(right_childs_arg, NPY_UINT16, NPY_ARRAY_OUT_ARRAY);
    PyObject *node_mean_obj = PyArray_FROM_OTF(node_mean_arg, NPY_DOUBLE, NPY_ARRAY_OUT_ARRAY);
    PyObject *preds_obj = PyArray_FROM_OTF(preds_arg, NPY_DOUBLE, NPY_ARRAY_OUT_ARRAY);

    if (X_obj == NULL ||
        y_obj == NULL ||
        split_col_obj == NULL ||
        split_lo_obj == NULL ||
        split_hi_obj == NULL ||
        left_childs_obj == NULL ||
        mid_childs_obj == NULL ||
        right_childs_obj == NULL ||
        node_mean_obj == NULL ||
        preds_obj == NULL)
    {
        Py_XDECREF(X_obj);
        Py_XDECREF(y_obj);
        Py_XDECREF(split_col_obj);
        Py_XDECREF(split_lo_obj);
        Py_XDECREF(split_hi_obj);
        Py_XDECREF(left_childs_obj);
        Py_XDECREF(mid_childs_obj);
        Py_XDECREF(right_childs_obj);
        Py_XDECREF(node_mean_obj);
        Py_XDECREF(preds_obj);
        return NULL;
    }

    // cast data sections of numpy arrays to plain C pointers
    // this assumes the arrays are C-order, aligned, non-strided
    uint8_t *  __restrict X            = PyArray_DATA((PyArrayObject *) X_obj);
    double *   __restrict y            = PyArray_DATA((PyArrayObject *) y_obj);
    uint64_t * __restrict split_col    = PyArray_DATA((PyArrayObject *) split_col_obj);
    uint8_t *  __restrict split_lo     = PyArray_DATA((PyArrayObject *) split_lo_obj);
    uint8_t *  __restrict split_hi     = PyArray_DATA((PyArrayObject *) split_hi_obj);
    uint16_t * __restrict left_childs  = PyArray_DATA((PyArrayObject *) left_childs_obj);
    uint16_t * __restrict mid_childs   = PyArray_DATA((PyArrayObject *) mid_childs_obj);
    uint16_t * __restrict right_childs = PyArray_DATA((PyArrayObject *) right_childs_obj);
    double *   __restrict node_means   = PyArray_DATA((PyArrayObject *) node_mean_obj);
    double *   __restrict preds        = PyArray_DATA((PyArrayObject *) preds_obj);

    const uint64_t rows = (uint64_t) PyArray_DIM((PyArrayObject *) X_obj, 0);
    const uint64_t cols = (uint64_t) PyArray_DIM((PyArrayObject *) X_obj, 1);
    const uint16_t max_nodes = (uint16_t) PyArray_DIM((PyArrayObject *) left_childs_obj, 0);

    // for each node n, an array of length node_counts[n] containing the rows in n
    uint32_t * __restrict memberships [max_nodes];

    // [col, node, v] => stat
    uint32_t * __restrict counts = calloc(cols * max_nodes * vals, sizeof(uint32_t));
    double * __restrict sums = calloc(cols * max_nodes * vals, sizeof(double));
    double * __restrict sum_sqs = calloc(cols * max_nodes * vals, sizeof(double));

    uint16_t node_count = 1;
    uint16_t done_count = 0;

    double   node_scores  [max_nodes];
    uint32_t node_counts  [max_nodes];
    double   node_sums    [max_nodes];
    double   node_sum_sqs [max_nodes];
    uint16_t node_parents [max_nodes];
    bool     should_split [max_nodes];
    bool     should_subtract [max_nodes];

    uint32_t left_counts [max_nodes];
    uint32_t mid_counts  [max_nodes];
    uint32_t right_counts[max_nodes];

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
        node_parents[n] = 0;
        should_split[n] = false;
        should_subtract[n] = false;
        left_counts[n] = 0;
        mid_counts[n] = 0;
        right_counts[n] = 0;
        memberships[n] = NULL;
    }

    memberships[0] = calloc(rows, sizeof(uint32_t));

    // find the baseline of the root
    // accumulate in local variables so clang vectorizes
    double root_sum = 0.0;
    double root_sum_sq = 0.0;
    for (uint32_t r = 0; r < rows; r++) {
        root_sum += y[r];
        root_sum_sq += y[r] * y[r];
        memberships[0][r] = r;
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

        // build histograms
        for (uint16_t n = done_count; n < node_count; n++) {
            if (node_counts[n] == 0 || should_subtract[n]) continue;

            if (node_counts[n] < MIN_PARALLEL_SPLIT) {
               for (uint32_t i = 0; i < node_counts[n]; i++) {
                    uint32_t r = memberships[n][i];

                    for (uint32_t c = 0; c < cols; c++) {
                        uint8_t v = X[r*cols + c];
                        uint64_t idx = c*max_nodes*vals + n*vals + v;
                        counts[idx]++;
                        sums[idx] += y[r];
                        sum_sqs[idx] += y[r]*y[r];
                    }
                }
            } else {
                #pragma omp parallel
                {
                    uint32_t * __restrict local_counts = calloc(cols*vals, sizeof(uint32_t));
                    double * __restrict local_sums = calloc(cols*vals, sizeof(double));
                    double * __restrict local_sum_sqs = calloc(cols*vals, sizeof(double));

                    #pragma omp for
                    for (uint32_t i = 0; i < node_counts[n]; i++) {
                        uint32_t r = memberships[n][i];

                        for (uint32_t c = 0; c < cols; c++) {
                            uint8_t v = X[r*cols + c];
                            uint64_t idx = c*vals + v;
                            local_counts[idx]++;
                            local_sums[idx] += y[r];
                            local_sum_sqs[idx] += y[r]*y[r];
                        }
                    }

                    // TODO try atomic
                    //  OR back to synchronizing on the nodes...
                    //  and/or fix the values so we can vectorize
                    #pragma omp critical
                    {
                        for (uint32_t c = 0; c < cols; c++) {
                            for (uint v = 0; v < vals; v++) {
                                uint64_t local_i = c*vals + v;
                                uint64_t global_i = c*max_nodes*vals + n*vals + v;
                                counts[global_i] += local_counts[local_i];
                                sums[global_i] += local_sums[local_i];
                                sum_sqs[global_i] += local_sum_sqs[local_i];
                            }
                        }
                    }
                    free(local_counts);
                    free(local_sums);
                    free(local_sum_sqs);
                }
            }
        }

        // instead of summing stats from data,
        // derive from (parents - siblings)
        // this requires that we've already calculated the siblings
        // so the siblings must be at n-1 and n-2
        // TODO assumption no longer required
        #pragma omp parallel for
        for (uint16_t n = done_count; n < node_count; n++) {
            if (node_counts[n] == 0 || !should_subtract[n]) continue;
            for (uint32_t c = 0; c < cols; c++) {
                for (uint v = 0; v < vals; v++) {
                    uint64_t i = c*max_nodes*vals + n*vals + v;
                    uint64_t parent_i = c*max_nodes*vals + node_parents[n]*vals + v;
                    uint64_t bro_i = c*max_nodes*vals + (n-1)*vals + v;
                    uint64_t sis_i = c*max_nodes*vals + (n-2)*vals + v;

                    counts[i]  = counts[parent_i]  - counts[bro_i]  - counts[sis_i];
                    sums[i]    = sums[parent_i]    - sums[bro_i]    - sums[sis_i];
                    sum_sqs[i] = sum_sqs[parent_i] - sum_sqs[bro_i] - sum_sqs[sis_i];
                }
            }
        }

        gettimeofday(&choose_split_start, NULL);
        stat_ms += msec(stat_start, choose_split_start);

        // choose splits for all nodes on this level, node-parallel
        #pragma omp parallel for
        for (uint16_t n = done_count; n < node_count; n++) {
            for (uint32_t c = 0; c < cols; c++) {
                // min leaf size is 1 for left & right, 0 for mid
                if (node_counts[n] < 2) continue;

                // evaluate each possible splitting point
                // running sums from the left side
                uint32_t left_count = 0;
                double left_sum = 0.0;
                double left_sum_sq = 0.0;
                for (uint lo = 0; lo < vals - 2; lo++) {
                    uint64_t lo_i = c*max_nodes*vals + n*vals + lo;

                    // force non-empty left split
                    if (counts[lo_i] == 0) continue;

                    left_count += counts[lo_i];
                    left_sum += sums[lo_i];
                    left_sum_sq += sum_sqs[lo_i];
                    double left_var = left_sum_sq - (left_sum * left_sum / left_count);

                    uint64_t mid_count = 0;
                    double mid_sum = 0.0;
                    double mid_sum_sq = 0.0;

                    // allow mid split to be empty in the hi == lo case ONLY
                    for (uint hi = lo; hi < vals - 1; hi++) {
                        uint64_t hi_i = c*max_nodes*vals + n*vals + hi;
                        double split_penalty;

                        if (hi > lo && counts[hi_i] == 0) {
                            // this value doesn't change the split stats
                            continue;
                        } else if (hi > lo) {
                            // middle split is nonempty
                            // penalize it by a factor
                            split_penalty = penalty + third_split_penalty * penalty;

                            mid_count += counts[hi_i];
                            mid_sum += sums[hi_i];
                            mid_sum_sq += sum_sqs[hi_i];
                        } else {
                            // middle split is empty
                            split_penalty = penalty;
                        }

                        uint32_t right_count = node_counts[n] - left_count - mid_count;
                        double right_sum = node_sums[n] - left_sum - mid_sum;
                        double right_sum_sq = node_sum_sqs[n] - left_sum_sq - mid_sum_sq;

                        // force non-empty right split
                        if (right_count == 0) break;

                        // weighted average of splits' variance
                        double mid_var = (mid_count == 0) ? 0 : mid_sum_sq - (mid_sum * mid_sum / mid_count);
                        double right_var = right_sum_sq - (right_sum * right_sum / right_count);
                        double score = (left_var + mid_var + right_var + split_penalty) / node_counts[n];

                        // TODO something to make sure node_scores etc are on different cache lines?
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
                    }
                }
            }
        }

        gettimeofday(&make_split_start, NULL);
        choose_split_ms += msec(choose_split_start, make_split_start);

        // we've finised choosing the splits

        // update node metadata for the splits
        int new_node_count = node_count;
        for (uint16_t n = 0; n < node_count; n++) {
            if (should_split[n] && new_node_count <= max_nodes - 3) {

                // child with the most data is going to derive stats by subtraction
                // so it must go last
                if (left_counts[n] > mid_counts[n] && left_counts[n] > right_counts[n]) {
                    // left last
                    mid_childs[n] = new_node_count + 0;
                    right_childs[n] = new_node_count + 1;
                    left_childs[n] = new_node_count + 2;
                    should_subtract[left_childs[n]] = true;
                } else if (mid_counts[n] > right_counts[n]) {
                    // mid last
                    left_childs[n] = new_node_count + 0;
                    right_childs[n] = new_node_count + 1;
                    mid_childs[n] = new_node_count + 2;
                    should_subtract[mid_childs[n]] = true;
                } else {
                    // right last
                    left_childs[n] = new_node_count + 0;
                    mid_childs[n] = new_node_count + 1;
                    right_childs[n] = new_node_count + 2;
                    should_subtract[right_childs[n]] = true;
                }
                node_parents[left_childs[n]] = n;
                node_parents[mid_childs[n]] = n;
                node_parents[right_childs[n]] = n;

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
        for (uint16_t n = done_count; n < node_count; n++) {
            if (!should_split[n]) continue;

            // update memberships
            uint32_t left_i = 0;
            uint32_t mid_i = 0;
            uint32_t right_i = 0;

            uint16_t left = left_childs[n];
            uint16_t mid = mid_childs[n];
            uint16_t right = right_childs[n];

            memberships[left] = calloc(node_counts[left], sizeof(uint32_t));
            if (node_counts[mid] > 0) {
                memberships[mid] = calloc(node_counts[mid], sizeof(uint32_t));
            }
            memberships[right] = calloc(node_counts[right], sizeof(uint32_t));

            if (node_counts[n] < MIN_PARALLEL_SPLIT) {
                // single-threaded
                for (uint32_t i = 0; i < node_counts[n]; i++) {
                    uint32_t r = memberships[n][i];
                    uint8_t v = X[r*cols + split_col[n]];

                    if (v <= split_lo[n]) {
                        memberships[left][left_i++] = r;
                    } else if (v <= split_hi[n]) {
                        memberships[mid][mid_i++] = r;
                    } else {
                        memberships[right][right_i++] = r;
                    }
                }
            } else {
                // multi-threaded
                //
                // the order of rows doesn't matter
                //
                // aggregate a buffer within each thread
                // once full, copy it into memberships
                #pragma omp parallel
                {
                    uint32_t left_buf [SPLIT_BUF_SIZE];
                    uint32_t mid_buf  [SPLIT_BUF_SIZE];
                    uint32_t right_buf [SPLIT_BUF_SIZE];

                    uint32_t local_left_i = 0;
                    uint32_t local_mid_i = 0;
                    uint32_t local_right_i = 0;

                    uint32_t copy_start;

                    #pragma omp for
                    for (uint32_t i = 0; i < node_counts[n]; i++) {
                        uint32_t r = memberships[n][i];
                        uint8_t v = X[r*cols + split_col[n]];

                        if (v <= split_lo[n]) {
                            left_buf[local_left_i++] = r;
                        } else if (v <= split_hi[n]) {
                            mid_buf[local_mid_i++] = r;
                        } else {
                            right_buf[local_right_i++] = r;
                        }

                        if (local_left_i == SPLIT_BUF_SIZE) {
                            #pragma omp critical
                            {
                                copy_start = left_i;
                                left_i = copy_start + SPLIT_BUF_SIZE;
                            }
                            memcpy(memberships[left] + copy_start, left_buf, SPLIT_BUF_SIZE * sizeof(uint32_t));
                            local_left_i = 0;
                        }
                        if (local_mid_i == SPLIT_BUF_SIZE) {
                            #pragma omp critical
                            {
                                copy_start = mid_i;
                                mid_i = copy_start + SPLIT_BUF_SIZE;
                            }
                            memcpy(memberships[mid] + copy_start, mid_buf, SPLIT_BUF_SIZE * sizeof(uint32_t));
                            local_mid_i = 0;
                        }
                        if (local_right_i == SPLIT_BUF_SIZE) {
                            #pragma omp critical
                            {
                                copy_start = right_i;
                                right_i = copy_start + SPLIT_BUF_SIZE;
                            }
                            memcpy(memberships[right] + copy_start, right_buf, SPLIT_BUF_SIZE * sizeof(uint32_t));
                            local_right_i = 0;
                        }
                    }
                    // copy anything leftover in the buffers
                    if (local_left_i > 0) {
                        #pragma omp critical
                        {
                            copy_start = left_i;
                            left_i = copy_start + local_left_i;
                        }
                        memcpy(memberships[left] + copy_start, left_buf, local_left_i * sizeof(uint32_t));
                    }
                    if (local_mid_i > 0) {
                        #pragma omp critical
                        {
                            copy_start = mid_i;
                            mid_i = copy_start + local_mid_i;
                        }
                        memcpy(memberships[mid] + copy_start, mid_buf, local_mid_i * sizeof(uint32_t));

                    }
                    if (local_right_i > 0) {
                        #pragma omp critical
                        {
                            copy_start = right_i;
                            right_i = copy_start + local_right_i;
                        }
                        memcpy(memberships[right] + copy_start, right_buf, local_right_i * sizeof(uint32_t));
                    }
                }
            }
            // done with the parent now
            free(memberships[n]);
            memberships[n] = NULL;
        }

        done_count = node_count;
        node_count = new_node_count;
        for (uint16_t n = 0; n < node_count; n++) {;
            should_split[n] = false;
        }

        gettimeofday(&split_end, NULL);
        make_split_ms += msec(make_split_start, split_end);
    }

    gettimeofday(&post_start, NULL);

    // calculate the mean at each leaf node & predictions
    #pragma omp parallel for
    for (uint16_t n = 0; n < node_count; n++) {
        // write predictions for non-empty leaves only
        if (!node_counts[n] || left_childs[n]) continue;

        double mean = node_sums[n] / node_counts[n];
        for (uint32_t i = 0; i < node_counts[n]; i++) {
            uint32_t r = memberships[n][i];
            preds[r] = mean;
        }
        node_means[n] = mean;

        free(memberships[n]);
        memberships[n] = NULL;
    }

    free(counts);
    free(sums);
    free(sum_sqs);
    Py_DECREF(X_obj);
    Py_DECREF(y_obj);
    Py_DECREF(split_col_obj);
    Py_DECREF(split_lo_obj);
    Py_DECREF(split_hi_obj);
    Py_DECREF(left_childs_obj);
    Py_DECREF(mid_childs_obj);
    Py_DECREF(right_childs_obj);
    Py_DECREF(node_mean_obj);
    Py_DECREF(preds_obj);

    gettimeofday(&total_end, NULL);
#if VERBOSE
    printf("Fit depth %d / %d nodes: %.1f total, %.1f init, %.1f stats, %.1f choose splits, %.1f make splits, %.1f post\n",
        depth,
        node_count,
        ((float) msec(total_start, total_end)) / 1000.0,
        ((float) msec(total_start, init_end)) / 1000.0,
        ((float) stat_ms) / 1000.0,
        ((float) choose_split_ms) / 1000.0,
        ((float) make_split_ms) / 1000.0,
        ((float) msec(post_start, total_end)) / 1000.0);
#endif

    return Py_BuildValue("i", node_count);
}

static PyObject* eval_tree(PyObject *dummy, PyObject *args)
{
    PyObject *X_arg;
    PyObject *split_col_arg, *split_lo_arg, *split_hi_arg;
    PyObject *left_childs_arg, *mid_childs_arg, *right_childs_arg, *node_mean_arg;
    PyObject *out_arg;

    struct timeval total_start;
    struct timeval loop_start;
    struct timeval loop_stop;
    struct timeval total_stop;
    gettimeofday(&total_start, NULL);

    // parse input arguments
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!O!O!",
        &PyArray_Type, &X_arg,
        &PyArray_Type, &split_col_arg,
        &PyArray_Type, &split_lo_arg,
        &PyArray_Type, &split_hi_arg,
        &PyArray_Type, &left_childs_arg,
        &PyArray_Type, &mid_childs_arg,
        &PyArray_Type, &right_childs_arg,
        &PyArray_Type, &node_mean_arg,
        &PyArray_Type, &out_arg)) return NULL;

    PyObject *X_obj = PyArray_FROM_OTF(X_arg, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyObject *split_col_obj = PyArray_FROM_OTF(split_col_arg, NPY_UINT64, NPY_ARRAY_IN_ARRAY);
    PyObject *split_lo_obj = PyArray_FROM_OTF(split_lo_arg, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyObject *split_hi_obj = PyArray_FROM_OTF(split_hi_arg, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyObject *left_childs_obj = PyArray_FROM_OTF(left_childs_arg, NPY_UINT16, NPY_ARRAY_IN_ARRAY);
    PyObject *mid_childs_obj = PyArray_FROM_OTF(mid_childs_arg, NPY_UINT16, NPY_ARRAY_IN_ARRAY);
    PyObject *right_childs_obj = PyArray_FROM_OTF(right_childs_arg, NPY_UINT16, NPY_ARRAY_IN_ARRAY);
    PyObject *node_mean_obj = PyArray_FROM_OTF(node_mean_arg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyObject *out_obj = PyArray_FROM_OTF(out_arg, NPY_DOUBLE, NPY_ARRAY_OUT_ARRAY);

    if (X_obj == NULL ||
        split_col_obj == NULL ||
        split_lo_obj == NULL ||
        split_hi_obj == NULL ||
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
        Py_XDECREF(left_childs_obj);
        Py_XDECREF(mid_childs_obj);
        Py_XDECREF(right_childs_obj);
        Py_XDECREF(node_mean_obj);
        Py_XDECREF(out_obj);
        return NULL;
    }
    // cast data sections of numpy arrays to plain C pointers
    // this assumes the arrays are C-order, aligned, non-strided
    float *    __restrict X            = PyArray_DATA((PyArrayObject *) X_obj);
    uint64_t * __restrict split_col    = PyArray_DATA((PyArrayObject *) split_col_obj);
    float *    __restrict split_lo     = PyArray_DATA((PyArrayObject *) split_lo_obj);
    float *    __restrict split_hi     = PyArray_DATA((PyArrayObject *) split_hi_obj);
    uint16_t * __restrict left_childs  = PyArray_DATA((PyArrayObject *) left_childs_obj);
    uint16_t * __restrict mid_childs   = PyArray_DATA((PyArrayObject *) mid_childs_obj);
    uint16_t * __restrict right_childs = PyArray_DATA((PyArrayObject *) right_childs_obj);
    double *   __restrict node_means   = PyArray_DATA((PyArrayObject *) node_mean_obj);
    double *   __restrict out          = PyArray_DATA((PyArrayObject *) out_obj);

    const uint64_t rows = (uint64_t) PyArray_DIM((PyArrayObject *) X_obj, 0);
    const uint64_t cols = (uint64_t) PyArray_DIM((PyArrayObject *) X_obj, 1);

    gettimeofday(&loop_start, NULL);
    #pragma omp parallel for
    for (uint64_t r = 0; r < rows; r++) {
        uint16_t n = 0;
        uint16_t left;
        while ((left = left_childs[n])) {
            float val = X[r*cols + split_col[n]];
            n = (val <= split_lo[n]) ? left :
                (val <= split_hi[n]) ? mid_childs[n] :
                right_childs[n];
        }
        out[r] = node_means[n];
    }
    gettimeofday(&loop_stop, NULL);

    Py_DECREF(X_obj);
    Py_DECREF(split_col_obj);
    Py_DECREF(split_lo_obj);
    Py_DECREF(split_hi_obj);
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
    float *   __restrict X    = PyArray_DATA((PyArrayObject *) X_obj);
    float *   __restrict bins = PyArray_DATA((PyArrayObject *) bins_obj);
    uint8_t * __restrict out  = PyArray_DATA((PyArrayObject *) out_obj);

    const uint64_t rows = PyArray_DIM((PyArrayObject *) X_obj, 0);
    const uint64_t cols = PyArray_DIM((PyArrayObject *) X_obj, 1);
    const uint8_t splits = PyArray_DIM((PyArrayObject *) bins_obj, 1);
    const uint vals = splits + 1; // may be 256, overflowing uint8_t

    if (vals > 256) {
        printf("Bad vals: %d\n", vals);
        return NULL;
    }

    // bins is a (cols, splits) array separating X into splits+1 values
    // for binning the data in X from float => uint8
    //
    // such that (floats in bucket 0) <= bins[c, 0] < (floats in bucket 1) <= bins[c, 1] ...
    //
    gettimeofday(&loop_start, NULL);
    #pragma omp parallel for
    for (uint64_t r = 0; r < rows; r++) {
        for (uint64_t c = 0; c < cols; c++) {
            uint64_t idx = r*cols + c;
            float val = X[idx];

            // shortcut the 0 case because it's common (val <= bins[c*splits)
            // since out was 0-initialized, we can just skip
            if (val <= bins[c*splits]) continue;

            // single round of binary search
            uint8_t b = val <= bins[c*splits + splits/2] ? 1 : splits/2 + 1;

            // now linear search
            while (b < vals - 1 && val > bins[c*splits + b]) b++;

            out[idx] = b;
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
