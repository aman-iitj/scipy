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


cpdef NI_FindObjects(np.ndarray input, np.intp_t max_label):
    funcs = get_funcs(input.take([0]))
    
    if max_label < 1:
        max_label = input.max()
    # cdef func2_p deref_p 

    cdef:
        int ii, rank, size_regions
        int start, jj, idx, end
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



##############################################################################
##############################################################################
#       NI_CentreOfMass

# Points to check:
#  dereferencing again
#  NI get value function
##############################################################################
# Typedefs and declarations related to Implementation of fused type
##############################################################################

ctypedef fused data_com:
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

ctypedef data_com (* func_com_deref)(data_com *, np.flatiter, np.ndarray)

ctypedef void (*funcP_com)(void *data, np.flatiter _ii, np.intp_t *label, np.ndarray labels)

def getFuncs_com(np.ndarray[data_com] labels):
    return (<Py_intptr_t> get_value[data_com])

################################################################################
# Delcaration of functions for using fused data type conversions
################################################################################

cdef int get_value(data_com *data, np.flatiter iter, np.intp_t *value, np.ndarray array):
    
    cdef func_com_deref deref_p # from NI_find Objects funtions

    if np.PyArray_ITER_DATA(iter) != NULL:
        
        if np.PyArray_ISBYTESWAPPED(array) == True:
            deref_p = get_misaligned_from_iter

        else:
            deref_p = get_from_iter

        value[0] = <np.intp_t> deref_p(data, iter, array)

    return 1

################################################################################
# Implementation of cenre of mass functtion
################################################################################

cpdef NI_CentreOfMass(np.ndarray input, np.ndarray labels,
                      np.intp_t min_label, np.intp_t max_label, np.intp_t *indices,
                      np.intp_t n_results, np.float_t *center_of_mass):
    
    funcs = getFuncs_com(labels.take([0]))

    cdef:

        np.flatiter _ii, _mi
        PyArrayIterObject *ii, *mi
        np.uintp_t jj, kk, qq
        np.intp_t size, rank, idx, doit
        np.intp_t *label
        double *sum = NULL, *val = NULL
        
        funcP_com get_value = <funcP_com> <void *> <Py_intptr_t> funcs

    label = <np.intp_t *>PyDataMem_NEW(sizeof(np.intp_t))
    val = <double *>PyDataMem_NEW(sizeof(double))

# Initialization of values:
    idx = 0
    label[0] = 1
    doit = 1

# Initialization of iterator
    _ii = np.PyArray_IterNew(input)
    _mi = np.PyArray_IterNew(labels)

    ii = <PyArrayIterObject *> _ii
    mi = <PyArrayIterObject *> _mi

    size = input.size
    rank = input.ndimage

    sum = <double *>PyDataMem_NEW(n_results * sizeof(double))
    #error if memory Not assigned

    for jj in range(n_results):
        sum[jj] = 0.0
        for kk in range(rank):
            center_of_mass[jj * rank + kk] = 0.0

    # Iterate Over Array
    for jj in size:
        get_value(np.PyArray_ITER_DATA(_mi), _ii, label, labels)
        if min_label >=0 :
            if label[0] >= min_label and label[0] <= max_label:
                idx = indices[label[0] - min_label]
                doit = idx >= 0

            else:
                doit = 0

        else:
            doit = label[0] != 0

        if doit is True:
            # get_value(np.PyArray_ITER_DATA(_ii), _ii, val, input)
            sum[idx] += val[0]
            for kk in range(rank):
                center_of_mass[idx * rank + kk] += val[0] * ii.coordinates[kk]

        if label[0] is True:
            np.PyArray_ITER_NEXT(_ii)
            np.PyArray_ITER_NEXT(_mi)

        else:
            np.PyArray_ITER_NEXT(_ii)

    for jj in range(n_results):
        for kk in range(rank):
            center_of_mass[jj * rank + kk] /= sum[jj]

    PyDataMem_FREE(sum)
    PyDataMem_FREE(val)
    PyDataMem_FREE(label)


#EOF