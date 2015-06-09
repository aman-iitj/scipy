######################################################################
# Yet to do:
# 1. Release GIL wherever possible
# 2. Add cases for exception
# 3. Add comments
# 4. Write test cases
######################################################################

cimport cython
from cython cimport sizeof
import numpy as np
cimport numpy as np

np.import_array()

cdef extern from *:
   ctypedef int Py_intptr_t

ctypedef void (*PyArray_CopySwapFunc)(void *, void *, int, void *)

cdef extern from "numpy/arrayobject.h" nogil:
    ctypedef struct PyArrayIterObject:
        np.npy_intp *coordinates
        char *dataptr
        np.npy_bool contiguous

    ctypedef struct PyArray_ArrFuncs:
        PyArray_CopySwapFunc *copyswap
    
    ctypedef struct PyArray_Descr:
        PyArray_ArrFuncs *f

    ctypedef struct PyArrayObject:
        PyArray_Descr *descr
        int nd

    void copyswap(void *dest, void *src, int swap, void *arr)

    # void PyArray_ITER_NEXT(PyArrayIterObject *it)
    # int PyArray_ITER_NOTDONE(PyArrayIterObject *it)
    void *PyArray_ITER_DATA(PyArrayIterObject *it)
    PyArray_Descr *PyArray_DESCR(PyArrayObject* arr)
    # int PyArray_ISBYTESWAPPED(PyArrayObject* arr)

    void *PyDataMem_NEW(size_t)
    void PyDataMem_FREE(void *)


######################################################################
# Use Cython's type templates for type specialization
######################################################################



cpdef fun(np.ndarray arr):
    cdef:
        np.flatiter _iti
        PyArrayIterObject *iti

    _iti = np.PyArray_IterNew(arr)
    iti = <PyArrayIterObject *> _iti

    iti.contiguous = 0

    while np.PyArray_ITER_NOTDONE(_iti):
        print iti.coordinates[0], iti.coordinates[1]
        np.PyArray_ITER_NEXT(_iti) 