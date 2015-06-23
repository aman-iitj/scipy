######################################################################
# Cython version of ndimage.morphology file
######################################################################

cimport cython
from cython cimport sizeof
import numpy as np
cimport numpy as np
from libc.math cimport sqrt

np.import_array()

cdef extern from *:
   ctypedef int Py_intptr_t

ctypedef void (*PyArray_CopySwapFunc)(void *, void *, int, void *)

cdef extern from "numpy/arrayobject.h" nogil:
    ctypedef struct PyArrayIterObject:
        np.intp_t *coordinates
        char *dataptr
        np.npy_bool contiguous


    void *PyArray_ITER_DATA(PyArrayIterObject *it)
    # PyArray_Descr *PyArray_DESCR(PyArrayObject* arr)

    void *PyDataMem_NEW(size_t)
    void PyDataMem_FREE(void *)

##########################################################################
# Cython version of DistanceTransformBruteForce function
##########################################################################

ctypedef struct border_element:
    np.intp_t *coordinates
    np.intp_t index
    void *next

cpdef DistanceTransformBruteForce(np.ndarray input, int metric, np.ndarray sampling_arr, 
                                  np.ndarray distances, np.ndarray features):
    cdef:
        np.intp_t size, jj, min_index = 0
        int kk
        border_element *border_elements = NULL, *temp
        np.flatiter _ii, _di, _fi
        PyArrayIterObject *ii, *di, *fi
        np.float64_t *sampling = <np.float64_t *>np.PyArray_ITER_DATA(sampling_arr) if sampling_arr else NULL

    #Error Condition
    size = input.size

    # Iterator Intialization
    _di = np.PyArray_IterNew(distances)
    _fi = np.PyArray_IterNew(features)
    _ii = np.PyArray_IterNew(input)

    di = <PyArrayIterObject *> _di
    fi = <PyArrayIterObject *> _fi
    ii = <PyArrayIterObject *> _ii

    di.contiguous = 0
    fi.contiguous = 0
    ii.contiguous = 0

    for jj in range(size):
        if (<np.int8_t *> np.PyArray_ITER_DATA(_ii))[0] < 0:
            temp = <border_element *> PyDataMem_NEW(sizeof(border_element))
            # Unlikely wali Condition
            temp.next = border_elements
            border_elements = temp
            temp.index = jj
            temp.coordinates = <np.intp_t *> PyDataMem_NEW(input.ndim * sizeof(np.intp_t))
            for kk in range(input.ndim):
                temp.coordinates[kk] = ii.coordinates[kk]
        np.PyArray_ITER_NEXT(_ii)

    np.NI_ITERATOR_RESET(_ii)
    
    cdef double distance, d, t
    #applying all the if metric ==s 
    if metric == NI_DISTANCE_EUCLIDIAN:
        for jj in range(size):
            if (<np.int8_t *> np.PyArray_ITER_DATA(_ii))[0] > 0:
                distance = DBL_MAX
                temp = border_elements
                while temp:
                    d = 0.0
                    for kk in range(input.ndim):
                        #error prone don't know line 591
                        t = ii.coordinates[kk] - temp.coordinates[kk]
                        if sampling:
                            t *= sampling[kk]
                        d += t * t

                    if d < distance:
                        distance = d
                        if features:
                            min_index = temp.index

                    temp = <border_element *>temp.next

                if distances:
                    (<np.float64_t *>np.PyArray_ITER_DATA(_di) )[0] = sqrt(distance)
                if features:
                    (<np.intp_t *>np.PyArray_ITER_DATA(_fi))[0]= min_index
            else:
                if distances:
                    (<np.float64_t*>np.PyArray_ITER_DATA(_di) )[0] = 0.0
                if features:
                    (<np.int32_t *>np.PyArray_ITER_DATA(_fi))[0] = jj
            
            if features and distances:
                # NI_ITERATOR_NEXT3ii, di, fi, pi, np.PyArray_ITER_DATA(_di), pf
                np.PyArray_ITER_NEXT(_ii)
                np.PyArray_ITER_NEXT(_di)
                np.PyArray_ITER_NEXT(_fi)
            elif distances:
                # NI_ITERATOR_NEXT2ii, di, pi, pd
                np.PyArray_ITER_NEXT(_ii)
                np.PyArray_ITER_NEXT(_di)
            else:
                # NI_ITERATOR_NEXT2ii, fi, pi, pf
                np.PyArray_ITER_NEXT(_ii)
                np.PyArray_ITER_NEXT(_fi)
                
        break

    # if metric == NI_DISTANCE_CITY_BLOCK:
    cdef int distance_n, d_n
    cdef np.intp_t t_n
    if metric == NI_DISTANCE_CHESSBOARD:
        for jj in range(size):
            if (<np.int8_t *> np.PyArray_ITER_DATA(_ii))[0] > 0:
                distance_n = UINT_MAX
                temp = border_elements
                while temp:
                    d_n = 0
                    for kk in range(input.ndim):
                        t_n = ii.coordinates[kk] - temp.coordinates[kk]
                        if t_n < 0:
                            t_n = -t_n
                        if metric == NI_DISTANCE_CITY_BLOCK:
                            d_n += t_n
                        else:
                            if <np.uintp_t>t > d_n:
                                d_n = t_n

                    if d_n < distance_n:
                        distance_n = d_n
                        if features:
                            min_index = temp.index

                    temp = <border_element *> temp.next

                if distances:
                   (<np.uint32_t *> np.PyArray_ITER_DATA(_di))[0] = distance_n
                if features:
                    (<np.uint32_t *> np.PyArray_ITER_DATA(_fi))[0] = min_index
            else:
                if distances:
                    (<np.uint32_t *> np.PyArray_ITER_DATA(_di))[0] = 0
                if features:
                    (<np.uint32_t *> np.PyArray_ITER_DATA(_fi))[0] = jj

            if features and distances:
                # NI_ITERATOR_NEXT3ii, di, fi, pi, pd, pf
                np.PyArray_ITER_NEXT(_ii)
                np.PyArray_ITER_NEXT(_di)
                np.PyArray_ITER_NEXT(_fi)
            elif distances:
                # NI_ITERATOR_NEXT2ii, di, pi, pd
                np.PyArray_ITER_NEXT(_ii)
                np.PyArray_ITER_NEXT(_di)
            
            else:
                # NI_ITERATOR_NEXT2ii, fi, pi, pf
                np.PyArray_ITER_NEXT(_ii)
                np.PyArray_ITER_NEXT(_fi)

    # else:
    #     PyErr_SetString(PyExc_RuntimeError,  "distance metric not supported")

    while border_elements:
        temp = border_elements
        border_elements = <border_element *> border_elements.next
        PyDataMem_FREE(temp.coordinates)
        PyDataMem_FREE(temp)


