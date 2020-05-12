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

#define VERBOSE 1

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
    struct timeval split_start;
    struct timeval split_end;
    struct timeval post_start;
    struct timeval total_end;
    long stat_ms = 0;
    long split_ms = 0;
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

    const uint64_t rows = (uint64_t) PyArray_DIM((PyArrayObject *) X_obj, 1);
    const uint64_t cols = (uint64_t) PyArray_DIM((PyArrayObject *) X_obj, 0);
    const uint16_t max_nodes = (uint16_t) PyArray_DIM((PyArrayObject *) left_childs_obj, 0);

    // for each node n, an array of length node_counts[n] containing the rows in n
    uint64_t * __restrict memberships [max_nodes];

    // [col, node, v] => stat
    uint64_t * __restrict saved_counts = calloc(cols * max_nodes * vals, sizeof(uint64_t));
    double * __restrict saved_sums = calloc(cols * max_nodes * vals, sizeof(double));
    double * __restrict saved_sum_sqs = calloc(cols * max_nodes * vals, sizeof(double));

    uint16_t node_count = 1;
    uint16_t done_count = 0;

    double   node_scores  [max_nodes];
    uint64_t node_counts  [max_nodes];
    double   node_sums    [max_nodes];
    double   node_sum_sqs [max_nodes];
    bool     should_split [max_nodes];
    omp_lock_t node_locks [max_nodes];

    // parent or -1
    // siblings are always n-1 and n-2
    // TODO this is hackyyyyy
    int derive_stats_from [max_nodes];

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
        memberships[n] = NULL;

        derive_stats_from[n] = -1;
    }

    memberships[0] = calloc(rows, sizeof(uint64_t));

    // find the baseline of the root
    // accumulate in local variables so clang vectorizes
    double root_sum = 0.0;
    double root_sum_sq = 0.0;
    for (uint64_t r = 0; r < rows; r++) {
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

        #pragma omp parallel for
        for (uint64_t c = 0; c < cols; c++) {

            for (uint16_t n = done_count; n < node_count; n++) {

                if (node_counts[n] == 0) continue;

                // min leaf size is 1 for left & right, 0 for mid
                // but in those cases we still need to update stats in case our siblings depend on them
                uint64_t counts [vals];
                double sums [vals];
                double sum_sqs [vals];

                if (derive_stats_from[n] == -1) {
                    // calc stats
                    for (uint v = 0; v < vals; v++) {
                        counts[v] = 0;
                        sums[v] = 0.0;
                        sum_sqs[v] = 0.0;
                    }
                    for (uint64_t i = 0; i < node_counts[n]; i++) {
                        uint64_t r = memberships[n][i];
                        uint32_t v = X[c*rows + r];
                        counts[v]++;
                        sums[v] += y[r];
                        sum_sqs[v] += y[r] * y[r];
                    }
                } else {
                    // derive stats
                    uint16_t parent = (uint16_t) derive_stats_from[n];
                    uint16_t bro = n-1;
                    uint16_t sis = n-2;
                    // if (parent < 0 || bro < 0 || sis < 0) {
                    //     printf("bad node idx\n");
                    //     fflush(stdout);
                    //     return NULL;
                    // }

                    for (uint v = 0; v < vals; v++) {

                        counts[v] = saved_counts[c*max_nodes*vals + parent*vals + v]
                                  - saved_counts[c*max_nodes*vals + bro*vals + v]
                                  - saved_counts[c*max_nodes*vals + sis*vals + v];

                        // if (counts[v]) {
                        //     printf("n=%d c=%d v=%d: parent %llu - bro %llu - sis %llu = counts[v] %llu\n",
                        //         n, c, v,
                        //         saved_counts[c*max_nodes*vals + parent*vals + v],
                        //         saved_counts[c*max_nodes*vals + bro*vals + v],
                        //         saved_counts[c*max_nodes*vals + sis*vals + v],
                        //         counts[v]);
                        //     printf("node_counts[n]=%llu, node_counts[parent]=%llu\n", node_counts[n], node_counts[parent]);
                        // }

                        sums[v] = saved_sums[c*max_nodes*vals + parent*vals + v]
                                  - saved_sums[c*max_nodes*vals + bro*vals + v]
                                  - saved_sums[c*max_nodes*vals + sis*vals + v];
                        sum_sqs[v] = saved_sum_sqs[c*max_nodes*vals + parent*vals + v]
                                  - saved_sum_sqs[c*max_nodes*vals + bro*vals + v]
                                  - saved_sum_sqs[c*max_nodes*vals + sis*vals + v];
                    }
                }
                // either way, save these so we can derive future stats
                for (uint v = 0; v < vals; v++) {
                    saved_counts[c*max_nodes*vals + n*vals + v] = counts[v];
                    saved_sums[c*max_nodes*vals + n*vals + v] = sums[v];
                    saved_sum_sqs[c*max_nodes*vals + n*vals + v] = sum_sqs[v];
                }

                // min leaf size is 1 for left & right, 0 for mid
                if (node_counts[n] < 2) continue;


                // evaluate each possible splitting point
                // running sums from the left side
                uint64_t left_count = 0;
                double left_sum = 0.0;
                double left_sum_sq = 0.0;
                for (uint64_t lo = 0; lo < vals - 2; lo++) {
                    uint64_t lo_i = lo; // TODO

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
                    for (uint64_t hi = lo; hi < vals - 1; hi++) {
                        uint64_t hi_i = hi; // TODO

                        double split_penalty;

                        if (hi > lo && !counts[hi_i]) {
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

                        uint64_t right_count = node_counts[n] - left_count - mid_count;
                        double right_sum = node_sums[n] - left_sum - mid_sum;
                        double right_sum_sq = node_sum_sqs[n] - left_sum_sq - mid_sum_sq;

                        // force non-empty right split
                        if (right_count == 0) break;

                        // weighted average of splits' variance
                        double mid_var = (mid_count == 0) ? 0 : mid_sum_sq - (mid_sum * mid_sum / mid_count);
                        double right_var = right_sum_sq - (right_sum * right_sum / right_count);
                        double score = (left_var + mid_var + right_var + split_penalty) / node_counts[n];

                        // node_scores[n] may be stale, but it only decreases
                        // first check without the lock for efficiency
                        if (score < node_scores[n]) {
                            // now check with the lock for correctness
                            omp_set_lock(&node_locks[n]);
                            if (score < node_scores[n]) {
                                // printf("  n=%d c=%d score %f -> %f counts (%llu, %llu, %llu)\n",
                                //     n, c, node_scores[n], score, left_count, mid_count, right_count);

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
                    }
                }
            }
        }

        gettimeofday(&split_start, NULL);
        stat_ms += msec(stat_start, split_start);

        // we've finised choosing the splits

        // update node metadata for the splits
        int new_node_count = node_count;
        for (uint16_t n = 0; n < node_count; n++) {
            if (should_split[n] && new_node_count <= max_nodes - 3) {

                // largest sibling is going to have derived stats
                // so it must have the highest ID to be processed last
                // (the order of the other 2 doesn't matter)
                // TODO less hacky
                if (left_counts[n] > mid_counts[n] && left_counts[n] > right_counts[n]) {
                    mid_childs[n] = new_node_count + 0;
                    right_childs[n] = new_node_count + 1;
                    left_childs[n] = new_node_count + 2;
                    derive_stats_from[left_childs[n]] = n;
                } else if (mid_counts[n] > right_counts[n]) {
                    left_childs[n] = new_node_count + 0;
                    right_childs[n] = new_node_count + 1;
                    mid_childs[n] = new_node_count + 2;
                    derive_stats_from[mid_childs[n]] = n;
                } else {
                    left_childs[n] = new_node_count + 0;
                    mid_childs[n] = new_node_count + 1;
                    right_childs[n] = new_node_count + 2;
                    derive_stats_from[right_childs[n]] = n;
                }

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

                // printf("  %d [%llu] => (%d [%llu], %d [%llu], %d [%llu])\n",
                //     n, node_counts[n],
                //     left_childs[n], node_counts[left_childs[n]],
                //     mid_childs[n], node_counts[mid_childs[n]],
                //     right_childs[n], node_counts[right_childs[n]]);
                // fflush(stdout);

                new_node_count += 3;
            } else if (should_split[n]) {
                // no room; abort the split
                should_split[n] = false;
            }
        }

        // update memberships
        #pragma omp parallel for
        for (uint16_t n = done_count; n < node_count; n++) {
            if (!should_split[n]) continue;

            uint64_t left_i = 0;
            uint64_t mid_i = 0;
            uint64_t right_i = 0;

            uint16_t left = left_childs[n];
            uint16_t mid = mid_childs[n];
            uint16_t right = right_childs[n];

            memberships[left] = calloc(node_counts[left], sizeof(uint64_t));
            if (node_counts[mid] > 0) {
                memberships[mid] = calloc(node_counts[mid], sizeof(uint64_t));
            }
            memberships[right] = calloc(node_counts[right], sizeof(uint64_t));

            for (uint64_t i = 0; i < node_counts[n]; i++) {
                uint64_t r = memberships[n][i];
                uint8_t v = X[split_col[n]*rows + r];

                if (v <= split_lo[n]) {
                    memberships[left][left_i++] = r;
                } else if (v <= split_hi[n]) {
                    memberships[mid][mid_i++] = r;
                } else {
                    memberships[right][right_i++] = r;
                }
            }
            // TODO can update node_means and de-alloc parent now?
            //
            // if (left_i != node_counts[left] || mid_i != node_counts[mid] || right_i != node_counts[right]) {
            //     printf("bad memberships: %llu, %llu, %llu\n", left_i, mid_i, right_i);
            //     return NULL;
            // }
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

    // calculate the mean at each leaf node & predictions
    // TODO this is only accidentally correct; overwriting internal preds with leaf preds
    // only keep the leaf memberships around
    //
    // #pragma omp parallel for
    for (uint16_t n = 0; n < node_count; n++) {
        if (node_counts[n] == 0) continue;

        double mean = node_sums[n] / node_counts[n];
        for (uint64_t i = 0; i < node_counts[n]; i++) {
            uint64_t r = memberships[n][i];
            preds[r] = mean;
        }
        node_means[n] = mean;
    }

    for (uint16_t n = 0; n < max_nodes; n++) {
        omp_destroy_lock(&node_locks[n]);
        free(memberships[n]);
    }

    free(saved_counts);
    free(saved_sums);
    free(saved_sum_sqs);
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

    const uint64_t rows = PyArray_DIM((PyArrayObject *) X_obj, 1);
    const uint64_t cols = PyArray_DIM((PyArrayObject *) X_obj, 0);
    const uint8_t splits = PyArray_DIM((PyArrayObject *) bins_obj, 1);
    const int vals = splits + 1; // may be 256, overflowing uint8_t

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
    for (uint64_t c = 0; c < cols; c++) {

        uint8_t b = 0;

        for (uint64_t r = 0; r < rows; r++) {
            uint64_t idx = c*rows + r;
            float val = X[idx];

            // shortcut the 0 case because it's common (val <= bins[c*splits)
            // since out was 0-initialized, we can just skip
            if (val <= bins[c*splits]) continue;

            // start at either the value of the previous iteration, or 1
            // this is kind of like a single round of binary search
            // but if there are multiple of the same value in a row, we'll shortcut nicely
            if (val <= bins[c*splits + b - 1]) b = 1;

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
