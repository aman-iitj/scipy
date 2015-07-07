######################################################################
# Cython version of ndimage.find_objects() function.
######################################################################

cimport cython
from cython cimport sizeof
import numpy as np
cimport numpy as np

np.import_array()

cdef extern from *:
   ctypedef int Py_intptr_t

ctypedef void (*PyArray_CopySwapFunc)(void *, void *, int, void *)

cdef extern from "Python.h" nogil:
    np.PyObject* PyTuple_New(int len)
    np.PyObject* PyLong_FromSsize_t(int v)
    np.PyObject* PySlice_New(np.PyObject *start, np.PyObject *stop, np.PyObject *step)
    int PyList_SetItem(np.PyObject *list, int index, np.PyObject *item)


cdef extern from "numpy/arrayobject.h" nogil:
    ctypedef struct PyArrayIterObject:
        np.intp_t *coordinates
        char *dataptr
        np.npy_bool contiguous

    ctypedef struct PyArray_ArrFuncs:
        PyArray_CopySwapFunc *copyswap
    
    ctypedef struct PyArray_Descr:
        PyArray_ArrFuncs *f

    ctypedef struct PyArrayObject:
        PyArray_Descr *descr
        int nd

    void *PyArray_ITER_DATA(PyArrayIterObject *it)
    PyArray_Descr *PyArray_DESCR(PyArrayObject* arr)

    void *PyDataMem_NEW(size_t)
    void PyDataMem_FREE(void *)


######################################################################
# Use Cython's type templates for type speecialization
######################################################################

# Only integer values are allowed.
ctypedef fused data_t:
    np.int8_t
    np.int16_t
    np.int32_t
    np.int64_t
    np.uint8_t
    np.uint16_t
    np.uint32_t
    np.uint64_t

#####################################################################
# Function Specializers and asociate function for using fused type
#####################################################################

ctypedef void (*func_p)(void *data, np.flatiter iti, PyArrayIterObject *iti, 
                        np.ndarray input, int* regions, 
                        int rank) nogil

def get_funcs(np.ndarray[data_t] input):
    return (<Py_intptr_t> findObjectsPoint[data_t])


######################################################################
# Dereferncing pointer and Dealing with Misalligned pointers
######################################################################

ctypedef data_t (* func2_p)(data_t *, np.flatiter, np.ndarray)

cdef data_t get_from_iter(data_t *data, np.flatiter iter, np.ndarray arr):
    return (<data_t *>np.PyArray_ITER_DATA(iter))[0]

cdef data_t get_misaligned_from_iter(data_t *data, np.flatiter iter, np.ndarray arr):

    cdef data_t ret = 0
    cdef PyArray_CopySwapFunc copyswap = <PyArray_CopySwapFunc> <void *> PyArray_DESCR(<PyArrayObject*> arr).f.copyswap

    copyswap(&ret, np.PyArray_ITER_DATA(iter), 1,<void *> arr)

    return ret


######################################################################
# Update Regions According to Input Data Type
######################################################################

cdef int findObjectsPoint(data_t *data, np.flatiter _iti, PyArrayIterObject *iti, 
                                np.ndarray input, int* regions,
                                int rank):
    cdef int kk =0
    cdef np.intp_t cc
    
    cdef func2_p deref_p
    if np.PyArray_ISBYTESWAPPED(input) == True:
        deref_p = get_misaligned_from_iter

    else:
        deref_p = get_from_iter

    # only integer or boolean values are allowed, since s_index is being used in indexing
    cdef np.intp_t s_index = deref_p(data, _iti, input)

  
    regions[rank] = s_index

    return 1


######################################################################
# Implementaion of find_Objects function:-
######################################################################


cpdef _findObjects(np.ndarray input, np.intp_t max_label):
    cdef funcs = get_funcs(input.take([0]))
    
    # cdef func2_p deref_p 

    cdef:
        int ii, rank, size_regions
        int jj, idx
        # np.intp_t start, end
        int *regions

        # Array Iterator defining and Initialization:
        np.flatiter _iti
        PyArrayIterObject *iti

    cdef:
        func_p findObjectsPoint = <func_p> <void *> <Py_intptr_t> funcs

    rank = input.ndim
        
    regions = <int *> PyDataMem_NEW(input.size * sizeof(int))




    _iti = np.PyArray_IterNew(input)
    iti = <PyArrayIterObject *> _iti

    # if iterator is contiguos, PyArray_ITER_NEXT will treat it as 1D Array
    iti.contiguous = 0
    idx = 0
    #Iteration over all points:
    while np.PyArray_ITER_NOTDONE(_iti):
        findObjectsPoint(np.PyArray_ITER_DATA(_iti), _iti, iti, input, 
                         regions, idx)
        idx += 1
        np.PyArray_ITER_NEXT(_iti)

    result = []
    for ii in range(input.size):
        result.append(regions[ii])
    return result
