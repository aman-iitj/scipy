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
from libc.stdlib cimport malloc

np.import_array()

cdef extern from *:
   ctypedef int Py_intptr_t

cdef extern from "numpy/arrayobject.h" nogil:
    ctypedef struct PyArrayIterObject:
        np.npy_intp *coordinates
        char *dataptr
        np.npy_bool contiguous

    ctypedef struct PyArray_ArrFuncs:
        PyArray_CopySwapFunc *copyswap
    
    ctypedef struct PyArray_Descr:
        int type_num
        PyArray_ArrFuncs *f

    ctypedef struct PyArrayObject:
        PyArray_Descr *descr
        int nd

    void copyswap(void *dest, void *src, int swap, void *arr)

    void PyArray_ITER_NEXT(PyArrayIterObject *it)
    int PyArray_ITER_NOTDONE(PyArrayIterObject *it)
    void PyArray_ITER_RESET(PyArrayIterObject *it)
    void *PyArray_ITER_DATA(PyArrayIterObject *it)

    void *PyDataMem_NEW(size_t)
    void PyDataMem_FREE(void *)
    int PyArray_ISBYTESWAPPED(arr)


######################################################################
# Use Cython's type templates for type specialization
######################################################################

ctypedef fused data_t:
    np.int8_t
    np.int16_t
    np.int32_t
    np.int64_t
    np.uint8_t
    np.uint16_t

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

ctypedef data_t (* func2_p)(data_t, PyArrayIterObject, np.ndarray)

# change PyarrayIterObject to np.flatiter
cdef data_t get_from_iter(data_t *data, PyArrayIterObject *iter, np.ndarray arr):
    return (<data_t *>PyArray_ITER_DATA(iter))[0]

# change PyarrayIterObject to np.flatiter
cdef data_t get_misaligned_from_iter(data_t *data, PyArrayIterObject *iter, np.ndarray arr):

    cdef data_t ret;
    cdef PyArray_CopySwapFunc *copyswap

    copyswap = PyArray_DESCR(arr).f.copyswap

    copyswap(&ret, PyArray_ITER_DATA(iter), PyArray_ISBYTESWAPPED(arr), arr)

    return ret


######################################################################
# Update Regions According to Input Data Type
######################################################################

cdef int findObjectsPoint(data_t *data, np.flatiter _iti, PyArrayIterObject *iti, 
                                np.ndarray input, np.intp_t max_label, np.intp_t* regions,
                                int rank):
    cdef int kk =0
    cdef np.intp_t cc

    # only integer or boolean values are allowed, since s_index is being used in indexing
    # cdef np.uintp_t s_index =  <np.uintp_t> ((<data_t *> iti.dataptr)[0])-1
    cdef np.uintp_t s_index =  <np.uintp_t> deref_p(iti.dataptr, iti, input) - 1

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


cpdef NI_FindObjects(np.ndarray input, np.intp_t max_label):
    ##### Assertions left
    funcs = get_funcs(input.take([0]))

    cdef:
            func2_p deref_p 

    cdef:
        int ii, rank, size_regions
        int start, end
        np.intp_t jj, idx, *regions

        # Array Iterator defining and Initialization:
        np.flatiter _iti
        PyArrayIterObject *iti

    cdef:
        func_p findObjectsPoint = <func_p> <void *> <Py_intptr_t> funcs

    # Array Declaration for returning values:

    deref_p = get_misaligned_from_iter if PyArray_ISBYTESWAPPED(input) == True else deref_p = get_from_iter

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
        # error in allocation

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
    while PyArray_ITER_NOTDONE(iti):
        # Function Implementaton cross check
        findObjectsPoint(PyArray_ITER_DATA(iti), _iti, iti, input, max_label, regions, rank)
        PyArray_ITER_NEXT(iti)

    result = []

    for ii in range(max_label):
        if rank > 0:
            idx = 2 * rank * ii

        else:
            idx = ii

        if regions[idx] >= 0:
            slc = ()
            for jj in range(rank):
                start = regions[idx + jj]
                end = regions[idx + jj + rank]

                slc += (slice(start, end),)
            result.append(slc)

        else:
            result.append(None)

    PyDataMem_FREE(regions)

    return result
