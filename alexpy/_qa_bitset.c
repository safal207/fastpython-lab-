/* Experimental exact failure bitmap with one-level nonempty-word directory.
 * All staging allocations and validation precede NativeBatchQA commit.
 * The update path after native commit is allocation-free and exception-free.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <string.h>
#include <limits.h>

typedef struct {
    PyObject_HEAD
    Py_ssize_t n, limit, nwords, nsummary;
    uint64_t *words, *summary;
    Py_ssize_t count;
} WitnessBits;
typedef struct { Py_ssize_t index; unsigned char fail; } Update;

static void bits_dealloc(WitnessBits *s) {
    PyMem_Free(s->words); PyMem_Free(s->summary);
    Py_TYPE(s)->tp_free((PyObject *)s);
}
static int bits_init(WitnessBits *s, PyObject *args, PyObject *kw) {
    PyObject *n_obj=NULL, *limit_obj=NULL;
    static char *names[]={"n","limit",NULL};
    if(s->words || s->summary){PyErr_SetString(PyExc_RuntimeError,"already initialized");return -1;}
    if(!PyArg_ParseTupleAndKeywords(args,kw,"O|O",names,&n_obj,&limit_obj))return -1;
    if(Py_TYPE(n_obj)!=&PyLong_Type || (limit_obj&&Py_TYPE(limit_obj)!=&PyLong_Type)){
        PyErr_SetString(PyExc_ValueError,"dimensions must be exact integers");return -1;
    }
    s->n=PyLong_AsSsize_t(n_obj); if(s->n==-1&&PyErr_Occurred())return -1;
    s->limit=limit_obj?PyLong_AsSsize_t(limit_obj):4;
    if(s->limit==-1&&PyErr_Occurred())return -1;
    if(s->n<=0 || s->limit<=0 || s->n>PY_SSIZE_T_MAX-63){
        PyErr_SetString(PyExc_ValueError,"invalid dimensions");return -1;
    }
    s->nwords=(s->n+63)/64;
    s->nsummary=(s->nwords+63)/64;
    if((size_t)s->nwords>SIZE_MAX/sizeof(uint64_t) ||
       (size_t)s->nsummary>SIZE_MAX/sizeof(uint64_t)){
        PyErr_NoMemory();return -1;
    }
    s->words=PyMem_Calloc((size_t)s->nwords,sizeof(uint64_t));
    s->summary=PyMem_Calloc((size_t)s->nsummary,sizeof(uint64_t));
    if(!s->words || !s->summary){PyErr_NoMemory();return -1;}
    return 0;
}
static PyObject *bits_stage(WitnessBits *s,PyObject *latest) {
    if(!s->words){PyErr_SetString(PyExc_RuntimeError,"uninitialized witness bits");return NULL;}
    if(Py_TYPE(latest)!=&PyDict_Type){PyErr_SetString(PyExc_TypeError,"expected exact dict");return NULL;}
    Py_ssize_t n=PyDict_Size(latest); if(n<0)return NULL;
    if((size_t)n>PY_SSIZE_T_MAX/sizeof(Update)){PyErr_NoMemory();return NULL;}
    PyObject *staged=PyBytes_FromStringAndSize(NULL,n*(Py_ssize_t)sizeof(Update));
    if(!staged)return NULL;
    char *data=PyBytes_AS_STRING(staged);
    Py_ssize_t pos=0,k=0;
    PyObject *key,*val;
    while(PyDict_Next(latest,&pos,&key,&val)) {
        if(Py_TYPE(key)!=&PyLong_Type || Py_TYPE(val)!=&PyTuple_Type || PyTuple_GET_SIZE(val)!=2 ||
           Py_TYPE(PyTuple_GET_ITEM(val,0))!=&PyLong_Type ||
           Py_TYPE(PyTuple_GET_ITEM(val,1))!=&PyLong_Type) {
            PyErr_SetString(PyExc_TypeError,"invalid pending update shape");goto fail;
        }
        Py_ssize_t idx=PyLong_AsSsize_t(key);
        if((idx==-1&&PyErr_Occurred())||idx<0||idx>=s->n){
            if(!PyErr_Occurred())PyErr_SetString(PyExc_IndexError,"bad test id");goto fail;
        }
        long status=PyLong_AsLong(PyTuple_GET_ITEM(val,0));
        if((status==-1&&PyErr_Occurred())||status<1||status>3){
            if(!PyErr_Occurred())PyErr_SetString(PyExc_ValueError,"bad status");goto fail;
        }
        Update u={0};u.index=idx;u.fail=(status==2);
        memcpy(data+k*sizeof(Update),&u,sizeof(Update));k++;
    }
    if(k!=n){PyErr_SetString(PyExc_RuntimeError,"pending dict changed");goto fail;}
    return staged;
fail:
    Py_DECREF(staged);return NULL;
}
static PyObject *bits_commit(WitnessBits *s,PyObject *staged) {
    if(!s->words){PyErr_SetString(PyExc_RuntimeError,"uninitialized witness bits");return NULL;}
    if(Py_TYPE(staged)!=&PyBytes_Type || PyBytes_GET_SIZE(staged)%(Py_ssize_t)sizeof(Update)) {
        PyErr_SetString(PyExc_TypeError,"expected staged bytes");return NULL;
    }
    const char *data=PyBytes_AS_STRING(staged);
    Py_ssize_t k=PyBytes_GET_SIZE(staged)/(Py_ssize_t)sizeof(Update);
    for(Py_ssize_t j=0;j<k;j++) {
        Update u;memcpy(&u,data+j*sizeof(Update),sizeof(Update));
        if(u.index<0 || u.index>=s->n || u.fail>1){
            PyErr_SetString(PyExc_ValueError,"invalid staged update");return NULL;
        }
    }
    for(Py_ssize_t j=0;j<k;j++) {
        Update u;memcpy(&u,data+j*sizeof(Update),sizeof(Update));
        Py_ssize_t wi=u.index>>6;
        uint64_t mask=UINT64_C(1)<<(u.index&63);
        uint64_t old=s->words[wi];
        int was=(old&mask)!=0;
        if(was==u.fail)continue;
        uint64_t now=u.fail?(old|mask):(old&~mask);
        s->words[wi]=now;
        s->count+=u.fail?1:-1;
        uint64_t bit=UINT64_C(1)<<(wi&63);
        if(!old && now)s->summary[wi>>6]|=bit;
        else if(old && !now)s->summary[wi>>6]&=~bit;
    }
    Py_RETURN_NONE;
}
static unsigned ctz64(uint64_t val) {
#if defined(__GNUC__) || defined(__clang__)
    return (unsigned)__builtin_ctzll((unsigned long long)val);
#else
    unsigned i=0;while(!(val&1u)){val>>=1;i++;}return i;
#endif
}
static PyObject *bits_witnesses(WitnessBits *s,PyObject *unused) {
    if(!s->words){PyErr_SetString(PyExc_RuntimeError,"uninitialized witness bits");return NULL;}
    Py_ssize_t cap=s->count<s->limit?s->count:s->limit;
    PyObject *out=PyTuple_New(cap);if(!out)return NULL;
    Py_ssize_t cursor=0;
    for(Py_ssize_t si=0;si<s->nsummary&&cursor<cap;si++) {
        uint64_t sw=s->summary[si];
        while(sw && cursor<cap) {
            unsigned inner=ctz64(sw);sw&=sw-1;
            Py_ssize_t wi=si*64+inner;
            if(wi>=s->nwords)continue;
            uint64_t bits=s->words[wi];
            while(bits && cursor<cap) {
                unsigned offset=ctz64(bits);bits&=bits-1;
                Py_ssize_t id=wi*64+offset;
                if(id>=s->n){PyErr_SetString(PyExc_AssertionError,"overflow bit");goto fail;}
                PyObject *item=PyLong_FromSsize_t(id);
                if(!item)goto fail;
                PyTuple_SET_ITEM(out,cursor++,item);
            }
        }
    }
    if(cursor!=cap){PyErr_SetString(PyExc_AssertionError,"missing witnesses");goto fail;}
    return out;
fail:
    /* Avoid exposing uninitialized tuple slots when an exception occurs. */
    for(Py_ssize_t i=cursor;i<cap;i++){Py_INCREF(Py_None);PyTuple_SET_ITEM(out,i,Py_None);}
    Py_DECREF(out);return NULL;
}
static PyObject *bits_dump(WitnessBits *s,PyObject *unused){
    if(!s->words){PyErr_SetString(PyExc_RuntimeError,"uninitialized witness bits");return NULL;}
    PyObject *out=PyBytes_FromStringAndSize(NULL,s->n);if(!out)return NULL;
    char *dst=PyBytes_AS_STRING(out);
    for(Py_ssize_t i=0;i<s->n;i++)dst[i]=(s->words[i>>6]>>(i&63))&1u;
    return out;
}
static PyObject *bits_verify(WitnessBits *s,PyObject *unused) {
    if(!s->words){PyErr_SetString(PyExc_RuntimeError,"uninitialized witness bits");return NULL;}
    Py_ssize_t count=0;
    for(Py_ssize_t i=0;i<s->nwords;i++) {
        uint64_t v=s->words[i];
        if(i==s->nwords-1 && (s->n&63) && (v>>(s->n&63)))Py_RETURN_FALSE;
        if(!!(s->summary[i>>6]&(UINT64_C(1)<<(i&63)))!=!!v)Py_RETURN_FALSE;
        while(v){v&=v-1;count++;}
    }
    for(Py_ssize_t i=s->nwords;i<s->nsummary*64;i++)
        if(s->summary[i>>6]&(UINT64_C(1)<<(i&63)))Py_RETURN_FALSE;
    if(count!=s->count)Py_RETURN_FALSE;
    Py_RETURN_TRUE;
}
static PyObject *bits_count(WitnessBits *s,void *unused) {
    if(!s->words){PyErr_SetString(PyExc_RuntimeError,"uninitialized witness bits");return NULL;}
    return PyLong_FromSsize_t(s->count);
}
static PyMethodDef methods[]={
    {"stage",(PyCFunction)bits_stage,METH_O,"Validate latest updates and stage immutable patch"},
    {"commit",(PyCFunction)bits_commit,METH_O,"Apply staged patch without allocation"},
    {"witnesses",(PyCFunction)bits_witnesses,METH_NOARGS,"First K failed test IDs"},
    {"dump",(PyCFunction)bits_dump,METH_NOARGS,"Copy failure flags"},
    {"verify",(PyCFunction)bits_verify,METH_NOARGS,"Check summary and count invariants"},
    {NULL}
};
static PyGetSetDef attrs[]={{"failed_count",(getter)bits_count,NULL,"Failed count",NULL},{NULL}};
static PyTypeObject T={PyVarObject_HEAD_INIT(NULL,0)
    .tp_name="alexpy._qa_bitset.WitnessBits",.tp_basicsize=sizeof(WitnessBits),
    .tp_flags=Py_TPFLAGS_DEFAULT,.tp_doc="Experimental native failure bitmap",
    .tp_new=PyType_GenericNew,.tp_init=(initproc)bits_init,
    .tp_dealloc=(destructor)bits_dealloc,.tp_methods=methods,.tp_getset=attrs};
static struct PyModuleDef module={PyModuleDef_HEAD_INIT,"_qa_bitset",NULL,-1,NULL};
PyMODINIT_FUNC PyInit__qa_bitset(void){
    if(PyType_Ready(&T)<0)return NULL;
    PyObject *m=PyModule_Create(&module);if(!m)return NULL;
    if(PyModule_AddObjectRef(m,"WitnessBits",(PyObject *)&T)<0){Py_DECREF(m);return NULL;}
    return m;
}
