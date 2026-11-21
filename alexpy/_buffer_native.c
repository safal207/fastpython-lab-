#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <limits.h>
#include <string.h>

/*
 * Opt-in kernels for one-dimensional, C-contiguous, native signed 64-bit
 * buffers. Python owns the backing memory. Keep the GIL while reading.
 * This module does not modify the existing AlexPy public dispatcher.
 */
#define MAX_SQUARE_ROOT 3037000499LL
#define INTERLEAVE 64

static int add_exact_square(long long value, long long *fast, PyObject **big)
{
    if (*big == NULL && value >= -MAX_SQUARE_ROOT &&
            value <= MAX_SQUARE_ROOT) {
        const long long square = value * value;
        if (*fast <= LLONG_MAX - square) {
            *fast += square;
            return 0;
        }
    }
    if (*big == NULL) {
        *big = PyLong_FromLongLong(*fast);
        if (*big == NULL) return -1;
    }
    PyObject *operand = PyLong_FromLongLong(value);
    if (operand == NULL) return -1;
    PyObject *square = PyNumber_Multiply(operand, operand);
    Py_DECREF(operand);
    if (square == NULL) return -1;
    PyObject *updated = PyNumber_Add(*big, square);
    Py_DECREF(square);
    if (updated == NULL) return -1;
    Py_SETREF(*big, updated);
    return 0;
}

static PyObject *accumulate(PyObject *source, int interleaved)
{
    Py_buffer view;
    if (PyObject_GetBuffer(source, &view,
            PyBUF_FORMAT | PyBUF_ND | PyBUF_STRIDES) < 0) {
        if (PyErr_ExceptionMatches(PyExc_TypeError) ||
                PyErr_ExceptionMatches(PyExc_BufferError)) {
            PyErr_Clear();
            Py_RETURN_NOTIMPLEMENTED;
        }
        return NULL;
    }
    /* Reject endian-qualified and non-integer formats, non-native widths,
       multidimensional/strided views. No byte swapping is performed. */
    const int supported = view.ndim == 1 && view.format != NULL &&
        strcmp(view.format, "q") == 0 &&
        view.itemsize == (Py_ssize_t)sizeof(long long) &&
        view.len >= 0 && view.len % (Py_ssize_t)sizeof(long long) == 0 &&
        PyBuffer_IsContiguous(&view, 'C');
    if (!supported) {
        PyBuffer_Release(&view);
        Py_RETURN_NOTIMPLEMENTED;
    }
    const Py_ssize_t count = view.len / (Py_ssize_t)sizeof(long long);
    const char *data = (const char *)view.buf;
    long long fast = 0;
    PyObject *big = NULL;
    if (interleaved) {
        for (Py_ssize_t offset = 0; offset < INTERLEAVE && offset < count; ++offset) {
            for (Py_ssize_t i = offset; i < count; i += INTERLEAVE) {
                long long value;
                memcpy(&value, data + i * sizeof(long long), sizeof(value));
                if (add_exact_square(value, &fast, &big) < 0) goto fail;
            }
        }
    } else {
        for (Py_ssize_t i = 0; i < count; ++i) {
            long long value;
            memcpy(&value, data + i * sizeof(long long), sizeof(value));
            if (add_exact_square(value, &fast, &big) < 0) goto fail;
        }
    }
    PyBuffer_Release(&view);
    if (big != NULL) return big;
    return PyLong_FromLongLong(fast);
fail:
    Py_XDECREF(big);
    PyBuffer_Release(&view);
    return NULL;
}

static PyObject *linear(PyObject *self, PyObject *source)
{
    (void)self;
    return accumulate(source, 0);
}
static PyObject *interleaved(PyObject *self, PyObject *source)
{
    (void)self;
    return accumulate(source, 1);
}
static PyMethodDef methods[] = {
    {"sum_squares_linear", (PyCFunction)linear, METH_O,
     "Experimental zero-copy signed q-buffer sequential sum of squares."},
    {"sum_squares_interleaved", (PyCFunction)interleaved, METH_O,
     "Experimental zero-copy signed q-buffer 64-way interleaved traversal."},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_buffer_native", "Opt-in memory-layout experiment", -1,
    methods, NULL, NULL, NULL, NULL
};
PyMODINIT_FUNC PyInit__buffer_native(void) { return PyModule_Create(&module); }
