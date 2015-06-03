######################################################################
# Yet to do:
# 1. Release GIL wherever possible
# 2. Add cases for exception
# 3. Add comments
# 4. Write test cases
# 5. Testing
######################################################################
# 
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
        np.npy_intp *dims_m1
        char *dataptr
        np.npy_bool contiguous

cdef extern from "numpy/numpyconfig.h":
    cdef: 
        int NPY_SIZEOF_LONG
        int NPY_SIZEOF_INT
        int NPY_INT
        int NPY_LONG
        int NPY_ULONG
        int NPY_UINT


    void PyArray_ITER_NEXT(PyArrayIterObject *it)
    int PyArray_ITER_NOTDONE(PyArrayIterObject *it)
    void PyArray_ITER_RESET(PyArrayIterObject *it)
    void *PyArray_ITER_DATA(PyArrayIterObject *it)

    void *PyDataMem_NEW(size_t)
    void PyDataMem_FREE(void *)
    void *PyDataMem_RENEW(void *, size_t)


#cdef extern from ndi_support:
#    int NI_NormalizeType(int type_num)

cdef inline int NI_NormalizeType(int type_num):
    if NPY_SIZEOF_LONG == NPY_SIZEOF_INT:
        if (type_num == NPY_INT):
            type_num = NPY_LONG
        if (type_num == NPY_UINT):
            type_num = NPY_ULONG
    return type_num


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
    np.uint32_t
    np.uint64_t
    np.float32_t
    np.float64_t


######################################################################
# Update Regions According to Input Data Type
######################################################################

cdef int findObjectsPoint(PyArrayIterObject *iti, 
                                np.intp_t max_label, np.intp_t* regions, int rank):
    cdef int kk =0
    cdef np.intp_t cc
    cdef np.intp_t s_index =  (<np.intp_t *> iti.dataptr)[0]-1
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
    cdef:
        int ii, rank, size_regions
        int start, end
        np.intp_t jj, idx, *regions

        # Array Iterator defining and Initialization:
        np.flatiter _iti
        PyArrayIterObject *iti

    # Array Declaration for returning values:
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
            # regions = <np.intp_t *> malloc(max_label * sizeof(np.intp_t)) 
        regions = <np.intp_t *> malloc(size_regions * sizeof(np.intp_t))
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
        NI_NormalizeType((<PyArrayObject *> input).descr.type_num)
        # Function Implementaton cross check
        findObjectsPoint(iti, max_label, regions, rank)
        PyArray_ITER_NEXT(iti)

    
    result = []

    for ii in range(max_label):
        if rank > 0:
            idx = 2 * rank * ii

        else:
            idx = ii

        slc = ()
        for jj in range(rank):
            start = regions[idx + jj]
            end = regions[idx + jj + rank]

            slc += (slice(start, end),)
        result.append(slc)

    return result

