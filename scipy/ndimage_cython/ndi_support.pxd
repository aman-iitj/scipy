cimport cython

import numpy as np
cimport numpy as np

np.import_array()

cdef extern from "numpy/arrayobject.h" nogil:
    ctypedef struct PyArrayIterObject:
    	pass

    void PyArray_ITER_NEXT(PyArrayIterObject *it)
    int PyArray_ITER_NOTDONE(PyArrayIterObject *it)
    void PyArray_ITER_RESET(PyArrayIterObject *it)
    void *PyArray_ITER_DATA(PyArrayIterObject *it)

    void *PyDataMem_NEW(size_t)
    void PyDataMem_FREE(void *)
    void *PyDataMem_RENEW(void *, size_t)

cdef:
	np.flatiter _iti, _ito
    PyArrayIterObject *iti
    PyArrayIterObject *ito

_ito = np.PyArray_IterNew(output, &axis)
_iti = np.PyArray_IterNew(input, &axis)

ito = <PyArrayIterObject *> _ito
iti = <PyArrayIterObject *> _iti

 
cdef inline int NI_NormalizeType(int type_num)
{
	if NPY_SIZEOF_INT == NPY_SIZEOF_LONG:
    	if (type_num == NPY_INT):
    		type_num = NPY_LONG
    	if (type_num == NPY_UINT):
        	type_num = NPY_ULONG
        
    return type_num;
}

