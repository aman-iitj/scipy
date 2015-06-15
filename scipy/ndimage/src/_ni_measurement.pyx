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


cpdef _find_objects(np.ndarray input, np.intp_t max_label):
    funcs = get_funcs(input.take([0]))
    
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

cpdef NI_CentreOfMass(np.ndarray input, np.ndarray labels,
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



##############################################################################
##############################################################################
#   Py_WatershedIFT function in cython
##############################################################################
##############################################################################

DEF WS_MAXDIM 7
DEF DONE_TYPE UInt8
DEF COST_TYPE UInt16

struct WatershedElement:
    np.intp_t index
    COST_TYPE cost
    void *next, *prev
    DONE_TYPE done

cpdef int watershed_ift(np.ndarray input, np.ndarray markers, np.ndarray structure, 
                        np.ndarray output):
    cdef:
        int ll, jj, hh, kk, i_contiguous, o_contiguous
        np.intp_t size, maxval, nneigh, ssize
        np.intp_t strides[WS_MAXDIM], coordinates[WS_MAXDIM]
        np.intp_t *nstrides = NULL
        Bool *ps = NULL
        np.flatiter _mi, _ii, _li
        WatershedElement *temp = NULL, **first = NULL, **last = NULL





    













char *pl, *pm, *pi;
    int ll;
    npy_intp size, jj, hh, kk, maxval;
    npy_intp strides[WS_MAXDIM], coordinates[WS_MAXDIM];
    npy_intp *nstrides = NULL, nneigh, ssize;
    int i_contiguous, o_contiguous;
    NI_WatershedElement *temp = NULL, **first = NULL, **last = NULL;
    Bool *ps = NULL;
    NI_Iterator mi, ii, li;
    NPY_BEGIN_THREADS_DEF;

    i_contiguous = PyArray_ISCONTIGUOUS(input);
    o_contiguous = PyArray_ISCONTIGUOUS(output);
    ssize = 1;
    for(ll = 0; ll < strct->nd; ll++)
        ssize *= strct->dimensions[ll];
    if (input->nd > WS_MAXDIM) {
        PyErr_SetString(PyExc_RuntimeError, "too many dimensions");
        goto exit;
    }
    size = 1;
    for(ll = 0; ll < input->nd; ll++)
        size *= input->dimensions[ll];
    /* Storage for the temporary queue data. */
    temp = malloc(size * sizeof(NI_WatershedElement));
    if (!temp) {
        PyErr_NoMemory();
        goto exit;
    }

    NPY_BEGIN_THREADS;

    pi = (void *)PyArray_DATA(input);
    if (!NI_InitPointIterator(input, &ii))
        goto exit;
    /* Initialization and find the maximum of the input. */
    maxval = 0;
    for(jj = 0; jj < size; jj++) {
        npy_intp ival = 0;
        switch(NI_NormalizeType(input->descr->type_num)) {
        CASE_GET_INPUT(ival, pi, UInt8);
        CASE_GET_INPUT(ival, pi, UInt16);
        default:
            NPY_END_THREADS;
            PyErr_SetString(PyExc_RuntimeError, "data type not supported");
            goto exit;
        }
        temp[jj].index = jj;
        temp[jj].done = 0;
        if (ival > maxval)
            maxval = ival;
        NI_ITERATOR_NEXT(ii, pi);
    }
    pi = (void *)PyArray_DATA(input);
    /* Allocate and initialize the storage for the queue. */
    first = malloc((maxval + 1) * sizeof(NI_WatershedElement*));
    last = malloc((maxval + 1) * sizeof(NI_WatershedElement*));
    if (NI_UNLIKELY(!first || !last)) {
        NPY_END_THREADS;
        PyErr_NoMemory();
        goto exit;
    }
    for(hh = 0; hh <= maxval; hh++) {
        first[hh] = NULL;
        last[hh] = NULL;
    }
    if (!NI_InitPointIterator(markers, &mi))
        goto exit;
    if (!NI_InitPointIterator(output, &li))
        goto exit;
    pm = (void *)PyArray_DATA(markers);
    pl = (void *)PyArray_DATA(output);
    /* initialize all nodes */
    for(ll = 0; ll < input->nd; ll++)
        coordinates[ll] = 0;
    for(jj = 0; jj < size; jj++) {
        /* get marker */
        int label = 0;
        switch(NI_NormalizeType(markers->descr->type_num)) {
        CASE_GET_LABEL(label, pm, UInt8);
        CASE_GET_LABEL(label, pm, UInt16);
        CASE_GET_LABEL(label, pm, UInt32);
#if HAS_UINT64
        CASE_GET_LABEL(label, pm, UInt64);
#endif
        CASE_GET_LABEL(label, pm, Int8);
        CASE_GET_LABEL(label, pm, Int16);
        CASE_GET_LABEL(label, pm, Int32);
        CASE_GET_LABEL(label, pm, Int64);
        default:
            NPY_END_THREADS;
            PyErr_SetString(PyExc_RuntimeError, "data type not supported");
            goto exit;
        }
        switch(NI_NormalizeType(output->descr->type_num)) {
        CASE_PUT_LABEL(label, pl, UInt8);
        CASE_PUT_LABEL(label, pl, UInt16);
        CASE_PUT_LABEL(label, pl, UInt32);
#if HAS_UINT64
        CASE_PUT_LABEL(label, pl, UInt64);
#endif
        CASE_PUT_LABEL(label, pl, Int8);
        CASE_PUT_LABEL(label, pl, Int16);
        CASE_PUT_LABEL(label, pl, Int32);
        CASE_PUT_LABEL(label, pl, Int64);
        default:
            NPY_END_THREADS;
            PyErr_SetString(PyExc_RuntimeError, "data type not supported");
            goto exit;
        }
        NI_ITERATOR_NEXT2(mi, li, pm, pl);
        if (label != 0) {
            /* This node is a marker */
            temp[jj].cost = 0;
            if (!first[0]) {
                first[0] = &(temp[jj]);
                first[0]->next = NULL;
                first[0]->prev = NULL;
                last[0] = first[0];
            } else {
                if (label > 0) {
                    /* object markers are enqueued at the beginning, so they
                       are processed first. */
                    temp[jj].next = first[0];
                    temp[jj].prev = NULL;
                    first[0]->prev = &(temp[jj]);
                    first[0] = &(temp[jj]);
                } else {
                    /* background markers are enqueued at the end, so they are
                         processed after the object markers. */
                    temp[jj].next = NULL;
                    temp[jj].prev = last[0];
                    last[0]->next = &(temp[jj]);
                    last[0] = &(temp[jj]);
                }
            }
        } else {
            /* This node is not a marker */
            temp[jj].cost = maxval + 1;
            temp[jj].next = NULL;
            temp[jj].prev = NULL;
        }
        for(ll = input->nd - 1; ll >= 0; ll--)
            if (coordinates[ll] < input->dimensions[ll] - 1) {
                coordinates[ll]++;
                break;
            } else {
                coordinates[ll] = 0;
            }
    }

    pl = (void *)PyArray_DATA(output);
    ps = (Bool*)PyArray_DATA(strct);
    nneigh = 0;
    for (kk = 0; kk < ssize; kk++)
        if (ps[kk] && kk != (ssize / 2))
            ++nneigh;
    nstrides = malloc(nneigh * sizeof(npy_intp));
    if (NI_UNLIKELY(!nstrides)) {
        NPY_END_THREADS;
        PyErr_NoMemory();
        goto exit;
    }
    strides[input->nd - 1] = 1;
    for(ll = input->nd - 2; ll >= 0; ll--)
        strides[ll] = input->dimensions[ll + 1] * strides[ll + 1];
    for(ll = 0; ll < input->nd; ll++)
        coordinates[ll] = -1;
    for(kk = 0; kk < nneigh; kk++)
        nstrides[kk] = 0;
    jj = 0;
    for(kk = 0; kk < ssize; kk++) {
        if (ps[kk]) {
            int offset = 0;
            for(ll = 0; ll < input->nd; ll++)
                offset += coordinates[ll] * strides[ll];
            if (offset != 0)
                nstrides[jj++] += offset;
        }
        for(ll = input->nd - 1; ll >= 0; ll--)
            if (coordinates[ll] < 1) {
                coordinates[ll]++;
                break;
            } else {
                coordinates[ll] = -1;
            }
    }
    /* Propagation phase: */
    for(jj = 0; jj <= maxval; jj++) {
        while (first[jj]) {
            /* dequeue first element: */
            NI_WatershedElement *v = first[jj];
            first[jj] = first[jj]->next;
            if (first[jj])
                first[jj]->prev = NULL;
            v->prev = NULL;
            v->next = NULL;
            /* Mark element as done: */
            v->done = 1;
            /* Iterate over the neighbors of the element: */
            for(hh = 0; hh < nneigh; hh++) {
                npy_intp v_index = v->index, p_index = v->index, idx, cc;
                int qq, outside = 0;
                p_index += nstrides[hh];
                /* check if the neighbor is within the extent of the array: */
                idx = p_index;
                for (qq = 0; qq < input->nd; qq++) {
                    cc = idx / strides[qq];
                    if (cc < 0 || cc >= input->dimensions[qq]) {
                        outside = 1;
                        break;
                    }
                    idx -= cc * strides[qq];
                }
                if (!outside) {
                    NI_WatershedElement *p = &(temp[p_index]);
                    if (!(p->done)) {
                        /* If the neighbor was not processed yet: */
                        int max, pval, vval, wvp, pcost, label, p_idx, v_idx;
                        switch(NI_NormalizeType(input->descr->type_num)) {
                        CASE_WINDEX1(v_index, p_index, strides, input->strides,
                                                 input->nd, i_contiguous, p_idx, v_idx, pi,
                                                 vval, pval, UInt8);
                        CASE_WINDEX1(v_index, p_index, strides, input->strides,
                                                 input->nd, i_contiguous, p_idx, v_idx, pi,
                                                 vval, pval, UInt16);
                        default:
                            NPY_END_THREADS;
                            PyErr_SetString(PyExc_RuntimeError,
                                                            "data type not supported");
                            goto exit;
                        }
                        /* Calculate cost: */
                        wvp = pval - vval;
                        if (wvp < 0)
                            wvp = -wvp;
                        /* Find the maximum of this cost and the current
                             element cost: */
                        pcost = p->cost;
                        max = v->cost > wvp ? v->cost : wvp;
                        if (max < pcost) {
                            /* If this maximum is less than the neighbors cost,
                                 adapt the cost and the label of the neighbor: */
                            int idx;
                            p->cost = max;
                            switch(NI_NormalizeType(output->descr->type_num)) {
                            CASE_WINDEX2(v_index, strides, output->strides, input->nd,
                                                     idx, o_contiguous, label, pl, UInt8);
                            CASE_WINDEX2(v_index, strides, output->strides, input->nd,
                                                     idx, o_contiguous, label, pl, UInt16);
                            CASE_WINDEX2(v_index, strides, output->strides, input->nd,
                                                     idx, o_contiguous, label, pl, UInt32);
#if HAS_UINT64
                            CASE_WINDEX2(v_index, strides, output->strides, input->nd,
                                                     idx, o_contiguous, label, pl, UInt64);
#endif
                            CASE_WINDEX2(v_index, strides, output->strides, input->nd,
                                                     idx, o_contiguous, label, pl, Int8);
                            CASE_WINDEX2(v_index, strides, output->strides, input->nd,
                                                     idx, o_contiguous, label, pl, Int16);
                            CASE_WINDEX2(v_index, strides, output->strides, input->nd,
                                                     idx, o_contiguous, label, pl, Int32);
                            CASE_WINDEX2(v_index, strides, output->strides, input->nd,
                                                     idx, o_contiguous, label, pl, Int64);
                            default:
                                NPY_END_THREADS;
                                PyErr_SetString(PyExc_RuntimeError,
                                                                "data type not supported");
                                goto exit;
                            }
                            switch(NI_NormalizeType(output->descr->type_num)) {
                            CASE_WINDEX3(p_index, strides, output->strides, input->nd,
                                                     idx, o_contiguous, label, pl, UInt8);
                            CASE_WINDEX3(p_index, strides, output->strides, input->nd,
                                                     idx, o_contiguous, label, pl, UInt16);
                            CASE_WINDEX3(p_index, strides, output->strides, input->nd,
                                                     idx, o_contiguous, label, pl, UInt32);
#if HAS_UINT64
                            CASE_WINDEX3(p_index, strides, output->strides, input->nd,
                                                     idx, o_contiguous, label, pl, UInt64);
#endif
                            CASE_WINDEX3(p_index, strides, output->strides, input->nd,
                                                     idx, o_contiguous, label, pl, Int8);
                            CASE_WINDEX3(p_index, strides, output->strides, input->nd,
                                                     idx, o_contiguous, label, pl, Int16);
                            CASE_WINDEX3(p_index, strides, output->strides, input->nd,
                                                     idx, o_contiguous, label, pl, Int32);
                            CASE_WINDEX3(p_index, strides, output->strides, input->nd,
                                                     idx, o_contiguous, label, pl, Int64);
                            default:
                                NPY_END_THREADS;
                                PyErr_SetString(PyExc_RuntimeError,
                                                                "data type not supported");
                                goto exit;
                            }
                            /* If the neighbor is in a queue, remove it: */
                            if (p->next || p->prev) {
                                NI_WatershedElement *prev = p->prev, *next = p->next;
                                if (first[pcost] == p)
                                    first[pcost] = next;
                                if (last[pcost] == p)
                                    last[pcost] = prev;
                                if (prev)
                                    prev->next = next;
                                if (next)
                                    next->prev = prev;
                            }
                            /* Insert the neighbor in the appropiate queue: */
                            if (label < 0) {
                                p->prev = last[max];
                                p->next = NULL;
                                if (last[max])
                                    last[max]->next = p;
                                last[max] = p;
                                if (!first[max])
                                    first[max] = p;
                            } else {
                                p->next = first[max];
                                p->prev = NULL;
                                if (first[max])
                                    first[max]->prev = p;
                                first[max] = p;
                                if (!last[max])
                                    last[max] = p;
                            }
                        }
                    }
                }
            }
        }
    }
 exit:
    NPY_END_THREADS;
    free(temp);
    free(first);
    free(last);
    free(nstrides);
    return PyErr_Occurred() ? 0 : 1;
}
