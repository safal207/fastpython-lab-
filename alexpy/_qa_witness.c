/* Research-only compact exact witness segment tree.
 * All batch validation/staging happens before the authoritative QA commit.
 * commit() performs no Python allocation, so once authoritative commit
 * succeeds, the witness mirror cannot fail for memory-allocation reasons.
 * No score, digest or duration aggregates are approximated here.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <string.h>
#include <limits.h>

typedef struct {
    PyObject_HEAD
    Py_ssize_t n, base, limit;
    unsigned char *failed;
    uint32_t *tree;
} WitnessTree;

typedef struct { Py_ssize_t index; unsigned char fail; } Update;

static void witness_dealloc(WitnessTree *s) {
    PyMem_Free(s->failed);
    PyMem_Free(s->tree);
    Py_TYPE(s)->tp_free((PyObject*)s);
}

static int witness_init(WitnessTree *s, PyObject *args, PyObject *kwargs) {
    PyObject *n_obj=NULL,*limit_obj=NULL;
    static char *names[]={"n","limit",NULL};
    if (s->tree || s->failed) {
        PyErr_SetString(PyExc_RuntimeError,"cannot reinitialize witness tree");return -1;
    }
    if (!PyArg_ParseTupleAndKeywords(args,kwargs,"O|O",names,&n_obj,&limit_obj))return -1;
    if (Py_TYPE(n_obj)!=&PyLong_Type || (limit_obj && Py_TYPE(limit_obj)!=&PyLong_Type)) {
        PyErr_SetString(PyExc_ValueError,"n and limit must be exact integers");return -1;
    }
    s->n=PyLong_AsSsize_t(n_obj);
    if (s->n==-1 && PyErr_Occurred())return -1;
    s->limit=limit_obj?PyLong_AsSsize_t(limit_obj):4;
    if (s->limit==-1 && PyErr_Occurred())return -1;
    if (s->n<=0 || s->limit<=0 || (uint64_t)s->n>UINT32_MAX || s->n>PY_SSIZE_T_MAX/4) {
        PyErr_SetString(PyExc_ValueError,"invalid witness tree dimensions");return -1;
    }
    s->base=1;
    while(s->base<s->n)s->base*=2;
    s->tree=PyMem_Calloc((size_t)s->base*2,sizeof(uint32_t));
    s->failed=PyMem_Calloc((size_t)s->n,sizeof(unsigned char));
    if (!s->tree || !s->failed) {PyErr_NoMemory();return -1;}
    return 0;
}

/* stage() validates a private dict of latest per-test (status, duration)
 * and packs it in one immutable bytes object.  No input is retained. */
static PyObject *witness_stage(WitnessTree *s, PyObject *latest) {
    if (!s->tree) {PyErr_SetString(PyExc_RuntimeError,"uninitialized witness tree");return NULL;}
    if (Py_TYPE(latest)!=&PyDict_Type) {
        PyErr_SetString(PyExc_TypeError,"expected an exact dict");return NULL;
    }
    Py_ssize_t n=PyDict_Size(latest);
    if (n<0)return NULL;
    if ((size_t)n>PY_SSIZE_T_MAX/sizeof(Update)) {PyErr_NoMemory();return NULL;}
    PyObject *packed=PyBytes_FromStringAndSize(NULL,n*(Py_ssize_t)sizeof(Update));
    if (!packed)return NULL;
    Py_ssize_t cursor=0, pos=0;
    PyObject *key,*value;
    char *bytes=PyBytes_AS_STRING(packed);
    while(PyDict_Next(latest,&pos,&key,&value)) {
        if(Py_TYPE(key)!=&PyLong_Type || Py_TYPE(value)!=&PyTuple_Type || PyTuple_GET_SIZE(value)!=2 ||
           Py_TYPE(PyTuple_GET_ITEM(value,0))!=&PyLong_Type ||
           Py_TYPE(PyTuple_GET_ITEM(value,1))!=&PyLong_Type) {
            PyErr_SetString(PyExc_TypeError,"invalid pending update shape");goto fail;
        }
        Py_ssize_t i=PyLong_AsSsize_t(key);
        if((i==-1 && PyErr_Occurred()) || i<0 || i>=s->n) {
            if(!PyErr_Occurred())PyErr_SetString(PyExc_IndexError,"invalid pending index");goto fail;
        }
        long status=PyLong_AsLong(PyTuple_GET_ITEM(value,0));
        if((status==-1 && PyErr_Occurred()) || status<1 || status>3) {
            if(!PyErr_Occurred())PyErr_SetString(PyExc_ValueError,"invalid pending status");goto fail;
        }
        Update u={0};u.index=i;u.fail=(status==2);
        memcpy(bytes+cursor*sizeof(Update),&u,sizeof(Update));
        cursor++;
    }
    if(cursor!=n) {PyErr_SetString(PyExc_RuntimeError,"pending dictionary changed");goto fail;}
    return packed;
fail:
    Py_DECREF(packed);return NULL;
}

/* Full preflight before any update; second loop cannot raise or allocate. */
static PyObject *witness_commit(WitnessTree *s, PyObject *packed) {
    if (!s->tree) {PyErr_SetString(PyExc_RuntimeError,"uninitialized witness tree");return NULL;}
    if(Py_TYPE(packed)!=&PyBytes_Type || PyBytes_GET_SIZE(packed)%(Py_ssize_t)sizeof(Update)) {
        PyErr_SetString(PyExc_TypeError,"expected staged update bytes");return NULL;
    }
    const char *buf=PyBytes_AS_STRING(packed);
    Py_ssize_t n=PyBytes_GET_SIZE(packed)/(Py_ssize_t)sizeof(Update);
    for(Py_ssize_t j=0;j<n;j++) {
        Update u;memcpy(&u,buf+j*sizeof(Update),sizeof(Update));
        if(u.index<0 || u.index>=s->n || u.fail>1) {
            PyErr_SetString(PyExc_ValueError,"invalid staged update");return NULL;
        }
    }
    for(Py_ssize_t j=0;j<n;j++) {
        Update u;memcpy(&u,buf+j*sizeof(Update),sizeof(Update));
        if(s->failed[u.index]==u.fail)continue;
        int delta=u.fail?1:-1;
        s->failed[u.index]=u.fail;
        for(Py_ssize_t node=s->base+u.index;node;node>>=1) {
            s->tree[node]=(uint32_t)((int64_t)s->tree[node]+delta);
        }
    }
    Py_RETURN_NONE;
}

/* Query exactly the first min(K, failures) leaf IDs, in index order. */
static int gather(WitnessTree *s,Py_ssize_t node,PyObject *out,Py_ssize_t *cursor,Py_ssize_t cap) {
    if(s->tree[node]==0 || *cursor>=cap)return 0;
    if(node>=s->base) {
        Py_ssize_t i=node-s->base;
        if(i>=s->n) {PyErr_SetString(PyExc_AssertionError,"invalid tree leaf");return -1;}
        PyObject *item=PyLong_FromSsize_t(i);
        if(!item)return -1;
        PyTuple_SET_ITEM(out,(*cursor)++,item);
    } else {
        if(gather(s,node*2,out,cursor,cap)<0)return -1;
        if(gather(s,node*2+1,out,cursor,cap)<0)return -1;
    }
    return 0;
}
static PyObject *witness_first(WitnessTree *s,PyObject *unused){
    if (!s->tree) {PyErr_SetString(PyExc_RuntimeError,"uninitialized witness tree");return NULL;}
    Py_ssize_t cap=s->tree[1]<s->limit?s->tree[1]:s->limit;
    PyObject *out=PyTuple_New(cap);if(!out)return NULL;
    Py_ssize_t i=0;
    if(gather(s,1,out,&i,cap)<0){Py_DECREF(out);return NULL;}
    return out;
}
static PyObject *witness_dump(WitnessTree *s,PyObject *unused){
    if (!s->tree) {PyErr_SetString(PyExc_RuntimeError,"uninitialized witness tree");return NULL;}
    return PyBytes_FromStringAndSize((const char*)s->failed,s->n);
}
static PyObject *witness_verify(WitnessTree *s,PyObject *unused){
    if (!s->tree) {PyErr_SetString(PyExc_RuntimeError,"uninitialized witness tree");return NULL;}
    for(Py_ssize_t i=0;i<s->n;i++){
        if(s->failed[i]>1 || s->tree[s->base+i]!=s->failed[i])Py_RETURN_FALSE;
    }
    for(Py_ssize_t i=s->n;i<s->base;i++)if(s->tree[s->base+i]!=0)Py_RETURN_FALSE;
    for(Py_ssize_t i=s->base-1;i>0;i--)
        if(s->tree[i]!=s->tree[i*2]+s->tree[i*2+1])Py_RETURN_FALSE;
    Py_RETURN_TRUE;
}
static PyObject *witness_count(WitnessTree *s,void *unused){
    if (!s->tree) {PyErr_SetString(PyExc_RuntimeError,"uninitialized witness tree");return NULL;}
    return PyLong_FromUnsignedLong(s->tree[1]);
}
static PyMethodDef methods[]={
    {"stage",(PyCFunction)witness_stage,METH_O,"Validate and serialize pending latest updates before native QA commit"},
    {"commit",(PyCFunction)witness_commit,METH_O,"Apply staged updates without allocation"},
    {"witnesses",(PyCFunction)witness_first,METH_NOARGS,"Return exact first K failed test IDs"},
    {"dump",(PyCFunction)witness_dump,METH_NOARGS,"Copy failed flags"},
    {"verify",(PyCFunction)witness_verify,METH_NOARGS,"Independent structural invariants"},
    {NULL}
};
static PyGetSetDef attrs[]={{"failed_count",(getter)witness_count,NULL,"Exact failed count",NULL},{NULL}};
static PyTypeObject T={PyVarObject_HEAD_INIT(NULL,0) .tp_name="alexpy._qa_witness.WitnessTree",
    .tp_basicsize=sizeof(WitnessTree),.tp_flags=Py_TPFLAGS_DEFAULT,
    .tp_doc="Compact exact failure segment tree (experimental)",
    .tp_new=PyType_GenericNew,.tp_init=(initproc)witness_init,.tp_dealloc=(destructor)witness_dealloc,
    .tp_methods=methods,.tp_getset=attrs};
static struct PyModuleDef module={PyModuleDef_HEAD_INIT,"_qa_witness",NULL,-1,NULL};
PyMODINIT_FUNC PyInit__qa_witness(void){
    if(PyType_Ready(&T)<0)return NULL;
    PyObject *m=PyModule_Create(&module);if(!m)return NULL;
    if(PyModule_AddObjectRef(m,"WitnessTree",(PyObject*)&T)<0){Py_DECREF(m);return NULL;}
    return m;
}
