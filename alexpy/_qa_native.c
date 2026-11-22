/* Experimental exact QA batch accumulator: all validation and allocations
 * precede any observable mutation. CPython extension, no external libs. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <limits.h>
#include <structmember.h>
#include <string.h>
#include <stdlib.h>

typedef struct {
    PyObject_HEAD
    Py_ssize_t n, ns, tick, version;
    Py_ssize_t *suites, *counts, *seen, *indices;
    unsigned char *status, *pending_status;
    int64_t *duration, *pending_duration;
    Py_ssize_t global[4];
    PyObject **sums;
    PyObject *total_sum;
    Py_ssize_t epoch;
} QAState;

static PyObject *event_type = NULL;
static PyObject *transition_type = NULL;
static int same_exact_long(PyObject *o) { return Py_TYPE(o) == &PyLong_Type; }
static void fail_type(const char *message) { PyErr_SetString(PyExc_TypeError, message); }

static void state_dealloc(QAState *s) {
    if (s->sums) {for (Py_ssize_t k=0;k<s->ns;k++) Py_XDECREF(s->sums[k]);}
    Py_XDECREF(s->total_sum);
    PyMem_Free(s->sums); PyMem_Free(s->suites); PyMem_Free(s->counts);
    PyMem_Free(s->status); PyMem_Free(s->duration); PyMem_Free(s->seen);
    PyMem_Free(s->indices); PyMem_Free(s->pending_status); PyMem_Free(s->pending_duration);
    Py_TYPE(s)->tp_free((PyObject *)s);
}

static int state_init(QAState *s, PyObject *args, PyObject *kw) {
    PyObject *suites; PyObject *count_obj; static char *names[]={"suites","suite_count",NULL};
    if (s->suites) {PyErr_SetString(PyExc_RuntimeError,"QAState cannot be reinitialized");return -1;}
    if (!PyArg_ParseTupleAndKeywords(args,kw,"OO",names,&suites,&count_obj)) return -1;
    if (!same_exact_long(count_obj)) {PyErr_SetString(PyExc_ValueError,"suite_count must be a positive exact integer");return -1;}
    s->ns=PyLong_AsSsize_t(count_obj);
    if ((s->ns==-1 && PyErr_Occurred()) || s->ns<1) {if(!PyErr_Occurred())PyErr_SetString(PyExc_ValueError,"suite_count must be positive");return -1;}
    if (Py_TYPE(suites)!=&PyList_Type && Py_TYPE(suites)!=&PyTuple_Type) {PyErr_SetString(PyExc_ValueError,"non-empty exact list/tuple of suite IDs required");return -1;}
    s->n=PySequence_Size(suites);
    if(s->n<1) {PyErr_SetString(PyExc_ValueError,"non-empty suites required");return -1;}
    if (s->ns > PY_SSIZE_T_MAX / 4) {PyErr_SetString(PyExc_OverflowError,"too many suites");return -1;}
    s->suites=PyMem_Calloc((size_t)s->n,sizeof(Py_ssize_t));
    s->counts=PyMem_Calloc((size_t)s->ns*4,sizeof(Py_ssize_t));
    s->status=PyMem_Calloc((size_t)s->n,sizeof(unsigned char));
    s->duration=PyMem_Calloc((size_t)s->n,sizeof(int64_t));
    s->seen=PyMem_Calloc((size_t)s->n,sizeof(Py_ssize_t));
    s->indices=PyMem_Calloc((size_t)s->n,sizeof(Py_ssize_t));
    s->pending_status=PyMem_Calloc((size_t)s->n,sizeof(unsigned char));
    s->pending_duration=PyMem_Calloc((size_t)s->n,sizeof(int64_t));
    s->sums=PyMem_Calloc((size_t)s->ns,sizeof(PyObject*));
    if(!s->suites||!s->counts||!s->status||!s->duration||!s->seen||!s->indices||!s->pending_status||!s->pending_duration||!s->sums){PyErr_NoMemory();return -1;}
    for(Py_ssize_t i=0;i<s->n;i++) {
        PyObject *o=PySequence_Fast_GET_ITEM(suites,i);
        if(!same_exact_long(o)){PyErr_SetString(PyExc_ValueError,"invalid suite ID");return -1;}
        Py_ssize_t v=PyLong_AsSsize_t(o);
        if((v==-1 && PyErr_Occurred())||v<0||v>=s->ns){if(!PyErr_Occurred())PyErr_SetString(PyExc_ValueError,"invalid suite ID");return -1;}
        s->suites[i]=v; s->counts[4*v]++;
    }
    s->global[0]=s->n;
    for(Py_ssize_t i=0;i<s->ns;i++){s->sums[i]=PyLong_FromLong(0);if(!s->sums[i])return -1;}
    s->total_sum=PyLong_FromLong(0);
    if(!s->total_sum)return -1;
    return 0;
}

static PyObject *add_durations(PyObject *base,int64_t old,int64_t new){
    PyObject *a=PyLong_FromLongLong(new),*b=PyLong_FromLongLong(old);
    if(!a||!b){Py_XDECREF(a);Py_XDECREF(b);return NULL;}
    PyObject *delta=PyNumber_Subtract(a,b);
    Py_DECREF(a);Py_DECREF(b);
    if(!delta)return NULL;
    PyObject *out=PyNumber_Add(base,delta);
    Py_DECREF(delta);
    return out;
}
static int index_cmp(const void *a,const void *b){
    Py_ssize_t x=*(const Py_ssize_t*)a,y=*(const Py_ssize_t*)b;
    return (x>y)-(x<y);
}

static PyObject *state_apply_batch(QAState *s,PyObject *batch){
    PyObject *result=NULL,*changed_tuple=NULL,*suites_tuple=NULL,*seq=NULL,*new_total=NULL;
    PyObject **new_sums=NULL;
    unsigned char *affected=NULL;
    Py_ssize_t *changed=NULL,unique=0,n_changed=0,n_affected=0;
    if(Py_TYPE(batch)!=&PyList_Type && Py_TYPE(batch)!=&PyTuple_Type){fail_type("events must be an exact list/tuple of QAEvent");return NULL;}
    seq=PySequence_Fast(batch,"expected list/tuple");if(!seq)return NULL;
    Py_ssize_t ne=PySequence_Fast_GET_SIZE(seq);
    affected=PyMem_Calloc((size_t)s->ns,sizeof(unsigned char));
    new_sums=PyMem_Calloc((size_t)s->ns,sizeof(PyObject*));
    changed=PyMem_Calloc((size_t)(ne?s->n<ne?s->n:ne:1),sizeof(Py_ssize_t));
    if(!affected||!new_sums||!changed){PyErr_NoMemory();goto done;}
    if(s->epoch==PY_SSIZE_T_MAX){memset(s->seen,0,sizeof(Py_ssize_t)*(size_t)s->n);s->epoch=0;}
    s->epoch++;
    for(Py_ssize_t i=0;i<ne;i++) {
        PyObject *e=PySequence_Fast_GET_ITEM(seq,i);
        if(Py_TYPE(e)!=(PyTypeObject*)event_type){fail_type("event must be an exact QAEvent");goto done;}
        PyObject *i_obj=PyObject_GetAttrString(e,"test_id");
        PyObject *st_obj=PyObject_GetAttrString(e,"status");
        PyObject *dur_obj=PyObject_GetAttrString(e,"duration_ms");
        if(!i_obj||!st_obj||!dur_obj){Py_XDECREF(i_obj);Py_XDECREF(st_obj);Py_XDECREF(dur_obj);goto done;}
        if(!same_exact_long(i_obj)){PyErr_SetString(PyExc_IndexError,"invalid test ID");goto event_error;}
        Py_ssize_t idx=PyLong_AsSsize_t(i_obj);
        if((idx==-1&&PyErr_Occurred())||idx<0||idx>=s->n){if(!PyErr_Occurred())PyErr_SetString(PyExc_IndexError,"invalid test ID");goto event_error;}
        if(!same_exact_long(st_obj)){PyErr_SetString(PyExc_ValueError,"status must be PASS, FAIL or SKIP");goto event_error;}
        long st=PyLong_AsLong(st_obj);
        if((st==-1 && PyErr_Occurred()) || st<1||st>3){if(!PyErr_Occurred())PyErr_SetString(PyExc_ValueError,"status must be PASS, FAIL or SKIP");goto event_error;}
        if(!same_exact_long(dur_obj)){PyErr_SetString(PyExc_ValueError,"duration must be an exact nonnegative signed-q integer");goto event_error;}
        long long duration=PyLong_AsLongLong(dur_obj);
        if((duration==-1&&PyErr_Occurred())||duration<0){if(!PyErr_Occurred())PyErr_SetString(PyExc_ValueError,"invalid duration");goto event_error;}
        if(st==3&&duration!=0){PyErr_SetString(PyExc_ValueError,"skipped tests must have zero duration");goto event_error;}
        if(s->seen[idx]!=s->epoch){s->indices[unique++]=idx;s->seen[idx]=s->epoch;}
        s->pending_status[idx]=(unsigned char)st;
        s->pending_duration[idx]=(int64_t)duration;
        Py_DECREF(i_obj);Py_DECREF(st_obj);Py_DECREF(dur_obj);
        continue;
        event_error: Py_DECREF(i_obj);Py_DECREF(st_obj);Py_DECREF(dur_obj);goto done;
    }
    new_total=Py_NewRef(s->total_sum);
    for(Py_ssize_t k=0;k<unique;k++){
        Py_ssize_t idx=s->indices[k];
        if(s->status[idx]==s->pending_status[idx]&&s->duration[idx]==s->pending_duration[idx])continue;
        changed[n_changed++]=idx;
        Py_ssize_t suite=s->suites[idx];
        if(!affected[suite]){affected[suite]=1;n_affected++;}
        if(s->duration[idx]!=s->pending_duration[idx]){
            PyObject *base=new_sums[suite]?new_sums[suite]:s->sums[suite];
            PyObject *n_sum=add_durations(base,s->duration[idx],s->pending_duration[idx]);
            if(!n_sum)goto done;
            Py_XDECREF(new_sums[suite]);new_sums[suite]=n_sum;
            PyObject *n_total=add_durations(new_total,s->duration[idx],s->pending_duration[idx]);
            if(!n_total)goto done;
            Py_SETREF(new_total,n_total);
        }
    }
    qsort(changed,(size_t)n_changed,sizeof(Py_ssize_t),index_cmp);
    changed_tuple=PyTuple_New(n_changed);suites_tuple=PyTuple_New(n_affected);
    if(!changed_tuple||!suites_tuple)goto done;
    for(Py_ssize_t j=0;j<n_changed;j++){
        PyObject *n=PyLong_FromSsize_t(changed[j]);if(!n)goto done;
        PyTuple_SET_ITEM(changed_tuple,j,n);
    }
    Py_ssize_t si=0;
    for(Py_ssize_t j=0;j<s->ns;j++)if(affected[j]){
        PyObject *n=PyLong_FromSsize_t(j);if(!n)goto done;
        PyTuple_SET_ITEM(suites_tuple,si++,n);
    }
    PyObject *transition_args=Py_BuildValue("nnnnOO",s->tick+1,s->version+(n_changed!=0),ne,unique,changed_tuple,suites_tuple);
    if(!transition_args)goto done;
    result=PyObject_CallObject(transition_type,transition_args);
    Py_DECREF(transition_args);
    if(!result)goto done;
    /* Commit: all possible Python allocation failures have been handled. */
    for(Py_ssize_t i=0;i<n_changed;i++){
        Py_ssize_t idx=changed[i],suite=s->suites[idx];
        unsigned char old=s->status[idx],next=s->pending_status[idx];
        s->counts[suite*4+old]--;s->counts[suite*4+next]++;
        s->global[old]--;s->global[next]++;
        s->status[idx]=next;s->duration[idx]=s->pending_duration[idx];
    }
    for(Py_ssize_t k=0;k<s->ns;k++)if(new_sums[k]){
        Py_SETREF(s->sums[k],new_sums[k]);new_sums[k]=NULL;
    }
    Py_SETREF(s->total_sum,new_total);new_total=NULL;
    s->tick++;if(n_changed)s->version++;
    done:
    Py_XDECREF(seq);Py_XDECREF(changed_tuple);Py_XDECREF(suites_tuple);Py_XDECREF(new_total);
    if(new_sums){for(Py_ssize_t k=0;k<s->ns;k++)Py_XDECREF(new_sums[k]);}
    PyMem_Free(new_sums);PyMem_Free(changed);PyMem_Free(affected);
    return result;
}

static PyObject *counts_tuple(QAState *s,Py_ssize_t suite){
    PyObject *out=PyTuple_New(5);if(!out)return NULL;
    for(int k=0;k<4;k++){
        PyObject *v=PyLong_FromSsize_t(suite<0?s->global[k]:s->counts[suite*4+k]);
        if(!v){Py_DECREF(out);return NULL;}
        PyTuple_SET_ITEM(out,k,v);
    }
    PyTuple_SET_ITEM(out,4,Py_NewRef(suite<0?s->total_sum:s->sums[suite]));
    return out;
}
static PyObject *state_snapshot_raw(QAState *s,PyObject *ignored){
    PyObject *tot=counts_tuple(s,-1),*parts=PyTuple_New(s->ns);
    if(!tot||!parts){Py_XDECREF(tot);Py_XDECREF(parts);return NULL;}
    for(Py_ssize_t i=0;i<s->ns;i++){
        PyObject *part=counts_tuple(s,i);if(!part){Py_DECREF(parts);Py_DECREF(tot);return NULL;}
        PyTuple_SET_ITEM(parts,i,part);
    }
    PyObject *out=PyTuple_Pack(2,tot,parts);
    Py_DECREF(tot);Py_DECREF(parts);return out;
}
static PyObject *state_dump(QAState *s,PyObject *ignored){
    PyObject *st=PyList_New(s->n),*du=PyList_New(s->n);
    if(!st||!du){Py_XDECREF(st);Py_XDECREF(du);return NULL;}
    for(Py_ssize_t i=0;i<s->n;i++){
        PyObject *a=PyLong_FromLong(s->status[i]),*b=PyLong_FromLongLong(s->duration[i]);
        if(!a||!b){Py_XDECREF(a);Py_XDECREF(b);Py_DECREF(st);Py_DECREF(du);return NULL;}
        PyList_SET_ITEM(st,i,a);PyList_SET_ITEM(du,i,b);
    }
    PyObject *out=PyTuple_Pack(2,st,du);Py_DECREF(st);Py_DECREF(du);return out;
}
static PyMemberDef state_members[]={
    {"tick",Py_T_PYSSIZET,offsetof(QAState,tick),READONLY,NULL},
    {"version",Py_T_PYSSIZET,offsetof(QAState,version),READONLY,NULL},
    {NULL}
};
static PyMethodDef state_methods[]={
    {"apply_batch",(PyCFunction)state_apply_batch,METH_O,"Atomic validated batch."},
    {"snapshot_raw",(PyCFunction)state_snapshot_raw,METH_NOARGS,"Return (global,suites) tuples."},
    {"dump",(PyCFunction)state_dump,METH_NOARGS,"Return copied status and duration arrays."},
    {NULL}
};
static PyTypeObject StateType={
    PyVarObject_HEAD_INIT(NULL,0)
    .tp_name="alexpy._qa_native.QAState",.tp_basicsize=sizeof(QAState),
    .tp_flags=Py_TPFLAGS_DEFAULT,.tp_doc="Exact signed-q QA native accumulator",
    .tp_new=PyType_GenericNew,.tp_init=(initproc)state_init,
    .tp_dealloc=(destructor)state_dealloc,.tp_methods=state_methods,.tp_members=state_members,
};
static struct PyModuleDef module={PyModuleDef_HEAD_INIT,"_qa_native",NULL,-1,NULL};
PyMODINIT_FUNC PyInit__qa_native(void){
    if(PyType_Ready(&StateType)<0)return NULL;
    PyObject *m=PyModule_Create(&module);if(!m)return NULL;
    PyObject *qa=PyImport_ImportModule("alexpy.qa_lab");
    if(!qa){Py_DECREF(m);return NULL;}
    event_type=PyObject_GetAttrString(qa,"QAEvent");Py_DECREF(qa);
    if(!event_type){Py_DECREF(m);return NULL;}
    PyObject *batch_mod=PyImport_ImportModule("alexpy.qa_batch");
    if(!batch_mod){Py_DECREF(m);return NULL;}
    transition_type=PyObject_GetAttrString(batch_mod,"QABatchTransition");
    Py_DECREF(batch_mod);
    if(!transition_type){Py_DECREF(m);return NULL;}
    if(PyModule_AddObjectRef(m,"QAState",(PyObject*)&StateType)<0){Py_DECREF(m);return NULL;}
    return m;
}
