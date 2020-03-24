#include <Python.h>
#include <math.h>


static PyObject* bucket_stats(PyObject *self, PyObject *args);


static PyMethodDef Methods[] = {
    {"bucket_stats",
        bucket_stats,
        METH_VARARGS, "aggregate y statistics for each feature and value"},
    {NULL, NULL, 0, NULL}
};


static PyObject* bucket_stats(PyObject *self, PyObject *args)
{
    double p;

    /* This parses the Python argument into a double */
    if(!PyArg_ParseTuple(args, "d", &p)) {
        return NULL;
    }

    /* THE ACTUAL LOGIT FUNCTION */
    p = p/(1-p);
    p = log(p);

    /*This builds the answer back into a python object */
    return Py_BuildValue("d", p);
}

#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "bucket_stats",
    NULL,
    -1,
    Methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_spam(void)
{
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }
    return m;
}
#else
PyMODINIT_FUNC initspam(void)
{
    PyObject *m;

    m = Py_InitModule("bucket_stats", Methods);
    if (m == NULL) {
        return;
    }
}
#endif