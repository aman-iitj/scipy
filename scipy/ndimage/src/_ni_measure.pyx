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
                        np.ndarray input, np.intp_t max_label, np.intp_t* regions, 
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
                                np.ndarray input, np.intp_t max_label, np.intp_t* regions,
                                int rank):
    cdef int kk =0
    cdef np.intp_t cc
    
    cdef func2_p deref_p
    if np.PyArray_ISBYTESWAPPED(input) == True:
        deref_p = get_misaligned_from_iter

    else:
        deref_p = get_from_iter

    # only integer or boolean values are allowed, since s_index is being used in indexing
    cdef np.uintp_t s_index = deref_p(data, _iti, input) - 1

    if s_index >=0  and s_index < max_label:
        if rank > 0:
            s_index *= 2 * rank
            if regions[s_index] < 0:
                for kk in range(rank):
                    cc = iti.coordinates[kk]
                    regions[s_index + kk] = cc
                    regions[s_index + kk + rank] = cc + 1

            else:
                for kk in range(rank):
                    cc = iti.coordinates[kk]
                    if cc < regions[s_index + kk]:
                        regions[s_index + kk] = cc
                    if cc +1 > regions[s_index + kk + rank]:
                        regions[s_index + kk + rank] = cc + 1
        else:
            regions[s_index] = 1

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
        np.intp_t *regions

        # Array Iterator defining and Initialization:
        np.flatiter _iti
        PyArrayIterObject *iti

    cdef:
        func_p findObjectsPoint = <func_p> <void *> <Py_intptr_t> funcs

    rank = input.ndim
    
    if max_label < 0:
        max_label = 0
    
    # Declaring output array
    size_regions = 0
    if max_label >0:
        if rank > 0:
            size_regions = 2 * max_label * rank

        else:
            size_regions = max_label
        
        regions = <np.intp_t *> PyDataMem_NEW(size_regions * sizeof(np.intp_t))

    else:
        regions = NULL

    if rank > 0:
        for jj in range(size_regions):
            regions[jj] = -1


    _iti = np.PyArray_IterNew(input)
    iti = <PyArrayIterObject *> _iti

    # if iterator is contiguos, PyArray_ITER_NEXT will treat it as 1D Array
    iti.contiguous = 0

    #Iteration over all points:
    while np.PyArray_ITER_NOTDONE(_iti):
        findObjectsPoint(np.PyArray_ITER_DATA(_iti), _iti, iti, input, 
                        max_label, regions, rank)
        np.PyArray_ITER_NEXT(_iti)

    result = []
    for ii in range(size_regions):
        result.append(regions[ii])
    return result

    # result = []

    # for ii in range(max_label):
    #     if rank > 0:
    #         idx = 2 * rank * ii

    #     else:
    #         idx = ii

    #     if regions[idx] >= 0:
    #         slc = ()
    #         for jj in range(rank):
    #             start = regions[idx + jj]
    #             end = regions[idx + jj + rank]

    #             slc += (slice(start, end),)
    #         result.append(slc)

    #     else:
    #         result.append(None)

    # PyDataMem_FREE(regions)

    # return result






    # cdef:
    #     np.PyObject *result = NULL, *ttuple = NULL, *start = NULL, *end = NULL
    #     np.PyObject *slc = NULL

    # idx = size_regions
    # for ii in range(max_label):
    #     if regions[idx] >= 0:
    #         ttuple = <np.PyObject *> < Py_intptr_t>PyTuple_New(rank)

    #         for jj in range(rank):
    #             start = <np.PyObject *> < Py_intptr_t>PyLong_FromSsize_t(regions[idx + jj])
    #             end = <np.PyObject *> < Py_intptr_t>PyLong_FromSsize_t(regions[idx + jj + rank])

    #             slc =  <np.PyObject *> < Py_intptr_t>PySlice_New(start, end,NULL)

    #             start = NULL
    #             end = NULL
    #         PyList_SetItem(result, ii, ttuple)
    #         ttuple = NULL

    #         PyList_SetItem(result, ii, <np.PyObject *> <int>np.Py_None)

    # return 1
