#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <limits.h>

/* floor(sqrt(LLONG_MAX)) = 3037000499, valid for signed 64-bit long long. */
#define MAX_SQUARE_ROOT 3037000499LL

/*
 * Optimized, deliberately narrow kernel.
 * Returns NotImplemented for all unsupported inputs so the Python wrapper
 * can preserve exact Python semantics, including arbitrary-precision ints.
 * Never release the GIL: PyList/PyTuple and PyLong objects are being read.
 */
static PyObject *sum_squares_ints(PyObject *self, PyObject *seq)
{
    (void)self;
    const int is_list = PyList_CheckExact(seq);
    const int is_tuple = PyTuple_CheckExact(seq);
    if (!is_list && !is_tuple) {
        Py_RETURN_NOTIMPLEMENTED;
    }

    Py_ssize_t size = is_list ? PyList_GET_SIZE(seq) : PyTuple_GET_SIZE(seq);
    long long result = 0;
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = is_list ? PyList_GET_ITEM(seq, i) : PyTuple_GET_ITEM(seq, i);
        if (!PyLong_CheckExact(item)) {
            Py_RETURN_NOTIMPLEMENTED;
        }
        int overflow = 0;
        long long number = PyLong_AsLongLongAndOverflow(item, &overflow);
        if (overflow != 0 || (number == -1 && PyErr_Occurred())) {
            if (PyErr_Occurred()) {
                return NULL;
            }
            Py_RETURN_NOTIMPLEMENTED;
        }
        if (number > MAX_SQUARE_ROOT || number < -MAX_SQUARE_ROOT) {
            Py_RETURN_NOTIMPLEMENTED;
        }
        long long square = number * number;  /* proven not to overflow */
        if (result > LLONG_MAX - square) {
            Py_RETURN_NOTIMPLEMENTED;
        }
        result += square;
    }
    return PyLong_FromLongLong(result);
}

/*
 * Candidate A: use fast signed 64-bit accumulation until it cannot represent
 * the exact result, then promote once to PyLong. Pure exact integers are the
 * only inputs admitted; any subclass or non-int falls back to the public
 * expression, so no user-defined __mul__ is skipped or invoked twice.
 *
 * Separate entrypoint for A/B tests. NOT activated by sum_squares().
 */
static PyObject *sum_squares_bigints(PyObject *self, PyObject *seq)
{
    (void)self;
    const int is_list = PyList_CheckExact(seq);
    const int is_tuple = PyTuple_CheckExact(seq);
    if (!is_list && !is_tuple) {
        Py_RETURN_NOTIMPLEMENTED;
    }

    Py_ssize_t size = is_list ? PyList_GET_SIZE(seq) : PyTuple_GET_SIZE(seq);
    long long fast_total = 0;
    PyObject *big_total = NULL;

    for (Py_ssize_t i = 0; i < size; ++i) {
        PyObject *item = is_list ? PyList_GET_ITEM(seq, i) : PyTuple_GET_ITEM(seq, i);
        if (!PyLong_CheckExact(item)) {
            Py_XDECREF(big_total);
            Py_RETURN_NOTIMPLEMENTED;
        }
        if (big_total == NULL) {
            int overflow = 0;
            long long value = PyLong_AsLongLongAndOverflow(item, &overflow);
            if (value == -1 && PyErr_Occurred()) {
                return NULL;
            }
            if (overflow == 0 && value <= MAX_SQUARE_ROOT &&
                    value >= -MAX_SQUARE_ROOT) {
                long long square = value * value;
                if (fast_total <= LLONG_MAX - square) {
                    fast_total += square;
                    continue;
                }
            }
            big_total = PyLong_FromLongLong(fast_total);
            if (big_total == NULL) {
                return NULL;
            }
        }
        PyObject *square = PyNumber_Multiply(item, item);
        if (square == NULL) {
            Py_DECREF(big_total);
            return NULL;
        }
        PyObject *updated = PyNumber_Add(big_total, square);
        Py_DECREF(square);
        Py_DECREF(big_total);
        if (updated == NULL) {
            return NULL;
        }
        big_total = updated;
    }
    if (big_total != NULL) {
        return big_total;
    }
    return PyLong_FromLongLong(fast_total);
}

static PyMethodDef AlexPyMethods[] = {
    {"sum_squares_bigints", (PyCFunction)sum_squares_bigints, METH_O,
     "Experimental mixed-precision exact-int kernel (not default)."},
    {"sum_squares_ints", (PyCFunction)sum_squares_ints, METH_O,
     "Accelerated, exact 64-bit sum of squares for plain list/tuple[int]."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef alexpy_module = {
    PyModuleDef_HEAD_INIT,
    "_native",
    "Conservative CPython C extension for fast numerical loops",
    -1,
    AlexPyMethods,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit__native(void)
{
    return PyModule_Create(&alexpy_module);
}
