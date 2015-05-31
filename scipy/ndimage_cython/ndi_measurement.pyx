


## Have messed with indentation  at some places. Use space in place of tabs.

######################################################################
# Yet to do:
# 1. Release GIL wherever possible
# 2. Add cases for exception
# 3. Add comments
# 4. Write test cases
######################################################################

cimport cython

import numpy as np
cimport numpy as np
cimport ndi_support

np.import_array()

######################################################################
# Declaring Array Iterators and Macros for Iteration 
######################################################################

cdef extern from "numpy/arrayobject.h" nogil:
    ctypedef struct PyArrayIterObject:
        np.npy_intp *coordinates
        pass

    void PyArray_ITER_NEXT(PyArrayIterObject *it)
    int PyArray_ITER_NOTDONE(PyArrayIterObject *it)
    void PyArray_ITER_RESET(PyArrayIterObject *it)
    void *PyArray_ITER_DATA(PyArrayIterObject *it)

    void *PyDataMem_NEW(size_t)
    void PyDataMem_FREE(void *)
    void *PyDataMem_RENEW(void *, size_t)


cdef extern from ndi_support:
    int NI_NormalizeType(int type_num)



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

cdef int findObjectsPoint(np.ndarray input[data_t],PyArrayIterObject iti, 
                                np.intp_t max_label, np.intp_t* regions )
{
    cdef int ii
    cdef np.intp_t cc
    cdef int rank = input -> nd
    np.intp_t s_index = *(data_t *) input.data -1
    if s_index >=0  and s_index < max_label:
        if rank > 0:
            s_index = 2 * rank
            if regions[s_index] < 0:
                for ii in range(rank):
                    cc = iti.coordinates[kk]
                    regions[s_index + ii] = cc
                    regions[s_index + ii + rank] = cc + 1

            else:
                for ii in range(rank):
                    cc = iti.coordinates[kk]

                    if cc < regions[s_index + ii]:
                        regions[s_index + ii]
                    if cc +1 > regions[s_index + ii + rank]:
                        regions[s_index + ii + rank] = cc + 1
        else:
            regions[s_index] = 1
}


######################################################################
# Implementaion of find_Objects function:-
######################################################################


cpdef int _NI_FindObjects(np.ndarray input, np.intp_t max_label,
                                     np.intp_t* regions)
{
##### Assertions left

    cdef:
        int kk, ii
        np.intp_t size, jj

# Array Iterator defining and Initialization:

    cdef:
        np.flatiter _iti, _ito
        PyArrayIterObject *iti
        PyArrayIterObject *ito

    _iti = np.PyArray_IterNew(input, &axis)

    iti = <PyArrayIterObject *> _iti
        
    size = 1

    #This line should be implemented using factors...
    for ii in range(input->nd):
        size *= input->dimensions[kk];

#Iteration over all points:
    for ii in range(size):
        input->descr->type_num = NI_NormalizeType(input->descr->type_num)

#Function Implementaton cross check
        findObjectsPoint(input, iti, max_label, regions)

        PyArray_ITER_NEXT(iti)

    return 1

}
