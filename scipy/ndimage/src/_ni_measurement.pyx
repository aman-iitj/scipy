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


cpdef _find_objects(np.ndarray input, np.intp_t max_label):
    cdef funcs = get_funcs(input.take([0]))
    
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
##############################################################################
##############################################################################


# No wrapper function is used. This function is not being used bt scipy.ndimage
# module to do anything.

'''cpdef NI_CentreOfMass(np.ndarray input, np.ndarray labels,
                      np.intp_t min_label, np.intp_t max_label,
                      np.intp_t n_results):
    cdef:
        np.flatiter _ii, _mi
        PyArrayIterObject *ii, *mi
        np.uintp_t jj, kk, qq
        np.intp_t size, rank, idx, label, doit
        np.intp_t *indices           ################# Note
        np.float_t *center_of_mass   ################# Note
        double *sum = NULL, val


# Initialization of values:
    idx = 0
    label = 1
    doit = 1

# Initialization of iterator
    _ii = np.PyArray_IterNew(input)
    _mi = np.PyArray_IterNew(labels)

    ii = <PyArrayIterObject *> _ii
    mi = <PyArrayIterObject *> _mi

    size = input.size
    rank = input.ndim

    sum = <double *>PyDataMem_NEW(n_results * sizeof(double))
    #error if memory Not assigned

    for jj in range(n_results):
        sum[jj] = 0.0
        for kk in range(rank):
            center_of_mass[jj * rank + kk] = 0.0

    # Iterate Over Array
    for jj in size:
        # NI_GET_LABEL:
        if min_label >=0 :
            if label >= min_label and label <= max_label:
                    idx = indices[label - min_label]
                    doit = idx >= 0

            else:
                doit = 0

        else:
            doit = label != 0

        if doit is True:
            # val = NI_GET_VALUE()
            sum[idx] += val
            for kk in range(rank):
                center_of_mass[idx * rank + kk] += val * ii.coordinates[kk]

        if label is True:
            np.PyArray_ITER_NEXT(_ii)
            np.PyArray_ITER_NEXT(_mi)

        else:
            np.PyArray_ITER_NEXT(_ii)

    for jj in range(n_results):
        for kk in range(rank):
            center_of_mass[jj * rank + kk] /= sum[jj]

    PyDataMem_FREE(sum)
'''


##############################################################################
##############################################################################
#   Py_WatershedIFT function in cython
##############################################################################
##############################################################################


DEF WS_MAXDIM = 7
# DEF DONE_TYPE = np.uint8_t      Getting error: Np not definedat compile time
# DEF COST_TYPE = np.uint16_t

#############################################################################
# Fused type delarations
#############################################################################
ctypedef fused data_watershed:
    np.int8_t
    np.int16_t

ctypedef fused data_markers:
    np.int8_t
    np.int16_t
    np.int32_t
    np.int64_t
    np.uint8_t
    np.uint16_t
    np.uint32_t
    np.uint64_t


ctypedef fused data_output:
    np.int8_t
    np.int16_t
    np.int32_t
    np.int64_t
    np.uint8_t
    np.uint16_t
    np.uint32_t
    np.uint64_t

def get_funcp_watershed(np.ndarray[data_watershed] input, np.ndarray[data_markers] markers,
                       np.ndarray[data_output] output):
    return (<Py_intptr_t> get_value[data_watershed],
            <Py_intptr_t> markers_to_output[data_markers, data_output] )

ctypedef np.intp_t (*funcp_watershed)(void *data, np.flatiter x,
                    PyArrayIterObject *y, np.ndarray input) nogil

ctypedef np.intp_t (*funcp_markers_to_output)(void * data_m, void * data_o, np.flatiter _mi, 
                    np.flatiter _li)

cdef np.intp_t get_value(data_t *data, np.flatiter _iti, PyArrayObject *iti, 
                         np.ndarray array):
    #use byteswapped technique
    return <np.intp_t>((<data_t *> np.PyArray_ITER_DATA(_iti))[0])

cdef np.intp_t markers_to_output(data_markers *data_m, data_output *data_p, np.flatiter _mi, 
                                np.flatiter _li):
    #use byteswapping technique
    cdef np.intp_t temp = 4
    temp = (<data_markers *> np.PyArray_ITER_DATA(_mi))[0]
    (<data_output *> np.PyArray_ITER_DATA(_mi))[0] = < data_output > temp
    return temp

#############################################################################
# Basic function
#############################################################################

cdef struct WatershedElement:
    np.uintp_t index
    np.uint16_t cost
    void *next, *prev
    np.uint8_t done

cpdef int watershed_ift(np.ndarray input, np.ndarray markers, np.ndarray structure, 
                        np.ndarray output):
    cdef:
        funcs = get_funcp_watershed(input.take([0]), markers.take([0]), output.take([0]))
        int ll, jj, hh, kk, i_contiguous, o_contiguous, label
        np.intp_t size, maxval, nneigh, ssize, ival
        np.intp_t strides[WS_MAXDIM], coordinates[WS_MAXDIM]
        np.intp_t *nstrides = NULL
        bint *ps = NULL
        np.flatiter _ii, _li, _mi
        PyArrayIterObject *ii, *li, *mi
        WatershedElement *temp = NULL, **first = NULL, **last = NULL
    
    
    # if input.ndim > WS_MAXDIM:
    #     Raise RuntimeError("Too many dimensions")

    ssize = structure.ndim
    size = input.ndim

    temp = <WatershedElement *> PyDataMem_NEW(size * sizeof(WatershedElement))
    # Error condition
    

    # Iterator inititalization
    _ii = np.PyArray_IterNew(input)
    _li = np.PyArray_IterNew(output)
    _mi = np.PyArray_IterNew(markers)

    ii = <PyArrayIterObject *> _ii
    li = <PyArrayIterObject *> _li
    mi = <PyArrayIterObject *> _mi

    cdef funcp_watershed get_value = <funcp_watershed> <void *><Py_intptr_t> funcs[0]
    cdef funcp_markers_to_output markers_to_output = <funcp_markers_to_output> <void *> <Py_intptr_t> funcs[1]

    
    for jj in range(size):
        # Need value in function in ival from pi using fused_type
        ival = get_value(np.PyArray_ITER_DATA(_ii), _ii, ii, input)

        temp[jj].index = jj
        temp[jj].done = 0
        if ival > maxval:
            maxval = ival

        np.PyArray_ITER_NEXT(_ii)
    
    # Allocate and initialize the storage for the queue
    first = <WatershedElement ** >  PyDataMem_NEW((maxval + 1) * sizeof(WatershedElement *))
    last =  <WatershedElement ** > PyDataMem_NEW((maxval + 1) * sizeof(WatershedElement *))
    # error in allocations
    

    for hh in range(maxval):
        first[hh] = last[hh] = NULL

    for ll in range(input.ndim):
        coordinates[ll] = 0

    for jj in range(size):
        label = markers_to_output(np.PyArray_ITER_DATA(_mi), np.PyArray_ITER_DATA(_li), _mi, _li)
        np.PyArray_ITER_NEXT(_mi)
        np.PyArray_ITER_NEXT(_li)
        if label != 0:
            temp[jj].cost = 0
            if first[0] is NULL:
                first[0] = &(temp[jj])
                # beware here.. could get erreors
                first[0].prev = NULL
                first[0].next = NULL
                last[0] = first[0]

            else:
                if label > 0:
                    temp[jj].next = first[0]
                    temp[jj].prev = NULL
                    first[0].prev = &(temp[jj])
                    first[0] = &(temp[jj])

                else:
                    temp[jj].next = NULL
                    temp[jj].prev = last[0]
                    last[0].next = &(temp[jj])
                    last[0] = &(temp[jj])

        else:
            temp[jj].cost = maxval + 1
            temp[jj].next = NULL
            temp[jj].prev = NULL

        ll = input.ndim - 1
        while ll >=0:
            if coordinates[ll] < input.dimensions[ll] - 1:
                coordinates[ll] += 1
                break

            else:
                coordinates[ll] = 0
            ll -= 1

    nneigh = 0
    for kk in range(ssize):
        if ps[kk] and kk != ssize/2:
            nneigh += 1

    nstrides = <np.intp_t *> PyDataMem_NEW(nneigh * sizeof(np.intp_t))

    ##
    ## NI_UNLIKELY
    ## 
    
    strides[input.ndim -1] = 1

    for ll in range(input.ndim -1) :
        strides[ll] = input.dimensions[ll + 1] * strides[ll + 1]

    for ll in range(input.ndim):
        coordinates[ll] = -1

    for kk in range(nneigh):
        nstrides[kk] = 0

    jj = 0
    cdef int offset = 0

    for kk in range(ssize):
        if ps[kk] is not 0:
            for ll in range(input.ndim):
                offset += coordinates[ll] * strides[ll]
            if offset is not 0:
                nstrides[jj] += offset
                jj += 1


        ll = input.ndim -1
        while ll >= 0:
            if coordinates[ll] < 1:
                coordinates[ll] += 1

            else:
                coordinates[ll] = -1


    # Propogation Phase
    cdef:
        WatershedElement *v, *p, *prev, *next
        np.intp_t v_index, p_index, idx, cc
        int qq, outside
        int max, pval, vval, wvp, pcost, p_idx, v_idx
    
    for jj in range(maxval +1):
        while first[jj] is not NULL:
            v = first[jj]
            first[jj] = <WatershedElement *>first[jj].next
            if first[jj] is not NULL:
                first[jj].prev = NULL

            v.prev = NULL
            v.next = NULL

            v.done = 1

            for hh in range(nneigh):
                v_index = v.index
                p_index = v.index
                outside = 0
                p_index += nstrides[hh]
                # Check if the neighbour is within the extenet of the array
                idx = p_index
                for qq in range(input.ndim):
                    cc = idx / strides[qq]
                    if cc < 0 or cc >=input.dimensions[qq]:
                        outside = 1
                        break

                if outside is not 0:
                    p = &(temp[p_index])
                    # If the neighbour is not processed Yet
                    if p.done is 0:
                        # CASE_Windex
                        # Case_windex
                        # case_Windex
                        
                        # Calculate Cost
                        wvp = pval - vval
                        if wvp < 0:
                            wvp = -wvp
                        # Find the maximum of this cost and the current element cost
                        pcost = p.cost
                        if v.cost > wvp:
                            max = v.cost

                        else:
                            max = wvp

                        if max < pcost:
                            # If this maximum is less than the neighbors cost,
                            # adapt the cost and the label of the neighbor: 
                            p.cost = max
                            # CASE_WINDEX2
                            # CASE_WINDEX2
                            # CASE_WINDEX2
                            # CASE_WINDEX2
                            # CASE_WINDEX2
                            # CASE_WINDEX3
                            # CASE_WINDEX3
                            # CASE_WINDEX3
                            # CASE_WINDEX3
                            # CASE_WINDEX3
                            
                            # If the neighbor is in a queue, remove it:
                            if p.next or p.prev is not NULL:
                                prev = <WatershedElement *> p.prev
                                next = <WatershedElement *> p.next
                                if first[pcost] == p:
                                    first[pcost] = next

                                if last[pcost] == p:
                                    last[pcost] = p

                                if prev is not NULL:
                                    prev.next = next

                                if next is not NULL:
                                    next.prev = prev

                            # Insert the neighbor in the appropiate queue:
                            if label < 0:
                                p.prev = last[max]
                                p.next = NULL
                                if last[max] is not NULL:
                                    last[max].next = p
                                last[max] = p
                                if first[max] is NULL:
                                    first[max] = p

                            else:
                                p.next = first[max]
                                p.prev = NULL
                                if first[max] is not NULL:
                                    first[max].prev = p
                                first[max] = p
                                if last[max] is NULL:
                                    last[max] = p
    
    PyDataMem_FREE(temp)
    PyDataMem_FREE(first)
    PyDataMem_FREE(last)
    PyDataMem_FREE(nstrides)

    return 1
