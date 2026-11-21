#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <stdlib.h>

/* Bounded, CPU-native memory probes. These are NOT an interpreter speedup. */
typedef struct {
    Py_ssize_t count;
    uint32_t *successor;
} Ring;

#define RING_NAME "alexpy._memprobe.Ring"
#define MAX_NODES 16777216 /* 64 MiB of indexed links */
#define MAX_ACCESSES 200000000

static void dispose_ring(PyObject *capsule) {
    Ring *ring = (Ring *)PyCapsule_GetPointer(capsule, RING_NAME);
    if (ring == NULL) { PyErr_Clear(); return; }
    free(ring->successor);
    free(ring);
}

static uint32_t xorshift32(uint32_t *state) {
    uint32_t v = *state;
    v ^= v << 13;
    v ^= v >> 17;
    v ^= v << 5;
    *state = v;
    return v;
}

static PyObject *make_ring(PyObject *self, PyObject *args) {
    Py_ssize_t nodes;
    unsigned int seed = 0x1234abcdU;
    if (!PyArg_ParseTuple(args, "n|I:make_ring", &nodes, &seed)) return NULL;
    if (nodes < 2 || nodes > MAX_NODES) {
        PyErr_SetString(PyExc_ValueError, "nodes must be in [2, 16777216]");
        return NULL;
    }
    Ring *ring = (Ring *)calloc(1, sizeof(Ring));
    if (ring == NULL) return PyErr_NoMemory();
    ring->successor = (uint32_t *)malloc((size_t)nodes * sizeof(uint32_t));
    if (ring->successor == NULL) { free(ring); return PyErr_NoMemory(); }
    uint32_t *perm = (uint32_t *)malloc((size_t)nodes * sizeof(uint32_t));
    if (perm == NULL) { free(ring->successor); free(ring); return PyErr_NoMemory(); }
    ring->count = nodes;
    for (Py_ssize_t i=0; i<nodes; i++) perm[i]=(uint32_t)i;
    uint32_t state = seed ? seed : 0x9e3779b9U;
    for (Py_ssize_t i=nodes-1; i>0; i--) {
        Py_ssize_t j = (Py_ssize_t)(xorshift32(&state) % (uint32_t)(i+1));
        uint32_t tmp=perm[i]; perm[i]=perm[j]; perm[j]=tmp;
    }
    for (Py_ssize_t i=0; i<nodes; i++) {
        ring->successor[perm[i]] = perm[(i+1)%nodes];
    }
    free(perm);
    return PyCapsule_New(ring, RING_NAME, dispose_ring);
}

static PyObject *walk(PyObject *self, PyObject *args) {
    PyObject *capsule;
    Py_ssize_t steps;
    if (!PyArg_ParseTuple(args, "On:walk", &capsule, &steps)) return NULL;
    Ring *ring = (Ring *)PyCapsule_GetPointer(capsule, RING_NAME);
    if (ring == NULL) return NULL;
    if (steps < 0 || steps > MAX_ACCESSES) {
        PyErr_SetString(PyExc_ValueError, "invalid number of steps"); return NULL;
    }
    uint32_t cursor=0;
    for (Py_ssize_t i=0; i<steps; i++) cursor=ring->successor[cursor];
    return PyLong_FromUnsignedLong((unsigned long)cursor);
}

static PyObject *scan(PyObject *self, PyObject *args) {
    PyObject *capsule;
    Py_ssize_t rounds;
    if (!PyArg_ParseTuple(args, "On:scan", &capsule, &rounds)) return NULL;
    Ring *ring = (Ring *)PyCapsule_GetPointer(capsule, RING_NAME);
    if (ring == NULL) return NULL;
    if (rounds < 1 || rounds > MAX_ACCESSES/ring->count) {
        PyErr_SetString(PyExc_ValueError, "invalid number of rounds"); return NULL;
    }
    /* Volatile loads ensure the compiler cannot hoist or erase memory reads.
       Report *effective indexed-read rate*, not main-memory bandwidth. */
    volatile const uint32_t *links = ring->successor;
    uint64_t result=0;
    for (Py_ssize_t j=0; j<rounds; j++)
        for (Py_ssize_t i=0; i<ring->count; i++) result += links[i];
    return PyLong_FromUnsignedLongLong((unsigned long long)result);
}

static PyObject *nodes(PyObject *self, PyObject *args) {
    PyObject *capsule;
    if (!PyArg_ParseTuple(args, "O:nodes", &capsule)) return NULL;
    Ring *ring = (Ring *)PyCapsule_GetPointer(capsule, RING_NAME);
    if (ring == NULL) return NULL;
    return PyLong_FromSsize_t(ring->count);
}

static PyMethodDef methods[] = {
    {"make_ring", make_ring, METH_VARARGS, "Build a deterministic cycle over shuffled indices."},
    {"walk", walk, METH_VARARGS, "Dependent random pointer chasing (start node 0)."},
    {"scan", scan, METH_VARARGS, "Sequential volatile reads of the same buffer."},
    {"nodes", nodes, METH_VARARGS, "Return number of indexed nodes."},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_memprobe", "Memory microbenchmark primitives", -1, methods};
PyMODINIT_FUNC PyInit__memprobe(void) {return PyModule_Create(&module);}
