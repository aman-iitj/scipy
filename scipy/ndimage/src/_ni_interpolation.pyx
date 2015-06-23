# Cython Version of ni_interpolation.pyx

cimport cython
from cython cimport sizeof
import numpy as np
cimport numpy as np
from libc.math cimport floor, fabs

np.import_array()

cdef extern from *:
   ctypedef int Py_intptr_t

cdef enum NI_ExtendMode:
    NI_EXTEND_FIRST = 0,
    NI_EXTEND_NEAREST = 0,
    NI_EXTEND_WRAP = 1,
    NI_EXTEND_REFLECT = 2,
    NI_EXTEND_MIRROR = 3,
    NI_EXTEND_CONSTANT = 4,
    NI_EXTEND_LAST = NI_EXTEND_CONSTANT,
    NI_EXTEND_DEFAULT = NI_EXTEND_MIRROR

ctypedef void (*PyArray_CopySwapFunc)(void *, void *, int, void *)

cdef extern from "numpy/arrayobject.h" nogil:
    cdef enum:
        NPY_MAXDIMS
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


# Define all the required files:

cdef int spline_coefficients(double x, int order, double *result):
    cdef:
        int hh
        double y, start, f

    if order & 1:
        start = <int> floor(x) - order / 2
    else:
        start = <int> floor(x + 0.5) - order / 2

    for hh in range(order + 1):
        y = fabs(start - x + hh)
        

        #No switch statements in cython :(
        if order == 1:
            if y > 1.0:
                result[hh] = 0.0 - y
            else:
                result[hh] = 1.0- y

        elif order == 2:
            if y < 0.5:
                result[hh] = 0.75 - y * y

            elif y < 1.5:
                y = 1.5 - y
            else:
                result[hh] = 0.0

        elif order == 3:
            if y < 1.0:
                result[hh] = (y * y * (y - 2.0) * 3.0 + 4.0) / 6.0
            elif y < 2.0:
                y = 2.0 -y
                result[hh] = y * y * y / 6.0
            else:
                result[hh] = 0.0

        elif order == 4:
            if y < 0.5:
                y *= y
                result[hh] = y * (y * 0.25 - 0.625) + 115.0 / 192.0
            elif y < 1.5:
                result[hh] = y * (y * (y * (5.0 / 6.0 - y / 6.0) - 1.25) + 5.0 / 24.0) + 55.0 / 96.0
            elif y <  2.5:
                y -= 2.5
                y *= y
                result[hh] = y * y / 24.0
            else:
                result[hh] = 0.0

        elif order == 5:
            if y < 1.0:
                f = y * y
                result[hh] = f * (f * (0.25 - y / 12.0) - 0.5) + 0.55
            elif y < 2.0:
                result[hh] = y * (y * (y * (y * (y / 24.0 - 0.375) + 1.25) - 1.75) + 0.625) + 0.425
            elif y < 3.0:
                f = 3.0 - y
                y = f * f
                result[hh] = f * y * y / 120.0
            else:
                result[hh] = 0.0

        # else:
        #     error raise
    return 0

cdef double map_coordinate(double cc, np.intp_t len, int mode):
    cdef np.intp_t sz, sz2
    if cc < 0:
        if mode == NI_EXTEND_MIRROR:
            if len <= 1:
                cc = 0
            else:
                sz2 = 2 * len - 2
                cc = sz2 * <np.intp_t> (-cc / sz2) + cc
                if cc <= 1 - len:
                    cc += sz2
                else:
                    cc = -cc

        elif mode == NI_EXTEND_REFLECT:
            if len <= 1:
                cc = 0
            else:
                sz2 = 2 * len
            if cc < -sz2:
                cc = sz2 * <np.intp_t> (-cc / sz2) + cc
            if cc < -len:
                cc +=sz2
            else:
                cc = -cc - 1

        elif mode == NI_EXTEND_WRAP:
            if len <= 1:
                cc = 0
            else:
                sz = len -1
                # Integer division of -cc/sz gives (-cc mod sz)
                # Note that 'cc' is negative
                cc += sz * (<np.intp_t> (-cc / sz) + 1)

        elif mode == NI_EXTEND_NEAREST:
            cc = 0

        elif mode == NI_EXTEND_CONSTANT:
            cc = -1

    elif cc > len - 1:
        if mode == NI_EXTEND_MIRROR:
            if len <= 1:
                cc = 0
            else:
                sz2 = 2 * len - 2
                cc -= sz2 * <np.intp_t> (cc / sz2)
                if cc >= len:
                    cc = sz2 - cc

        if mode == NI_EXTEND_REFLECT:
            if len <= 1:
                cc = 0
            else:
                sz2 = 2 * len
                cc -= sz2 * <np.intp_t>(cc / sz2)
                if cc >= len:
                    cc = sz2 - cc - 1

        if mode == NI_EXTEND_WRAP:
            if len <= 1:
                cc = 0
            else:
                sz = len - 1
                cc -= sz * <np.intp_t> (cc / sz)

        if mode == NI_EXTEND_NEAREST:
            cc = len - 1

        if mode == NI_EXTEND_CONSTANT:
            cc = -1
`
    return cc


####################################################################################
# Implmenetations of function zoom_shift
#################################################################################

# Have to leave it since this ucntion uses line_iterator

cpdef int _geometric_Transform(np.ndarray input, ........................):
    cdef:
        np.intp_t **edge_offsets = NULL, **data_offsets = NULL, filter_size
        np.intp_t ftmp[MAXDIM], *fcoordinates = NULL, *foffsets = NULL
        np.intp_t cstride = 0, kk, hh, ll, jj
        np.intp_t size
        double **splvals = NULL, icoor[MAXDIM]
        np.intp_t idimensions[MAXDIM], istrides[MAXDIM]
        np.flatiter _io, _ic
        PyArrayIterObject *io, *ic
        np.float64_t  *matrix #= matrix_ar ? (Float64*)PyArray_DATA(matrix_ar) : NULL;
        np.float64_t *shift #= shift_ar ? (Float64*)PyArray_DATA(shift_ar) : NULL;
        int irank = 0, orank, qq
    
    irank = input.ndim
    orank = output.ndim
    # Keep an eye whether you can complete the function without this
    for kk in range(irank):
        idimensions[kk] = input.dimensions[kk]
        istrides[kk] = input.strides[kk]

    # if the mapping is from array coordinates:
    if coordinates:
        # use of line_itrator


    






####################################################################################
# Implmenetations of function zoom_shift
#################################################################################

cpdef int _zoom_shift(np.ndarray input, np.ndarray zoom_ar, np.ndarray shift_ar, np.ndarray output, int order, int mode, double cval):
    cdef:
        np.intp_t **zeroes = NULL, **offsets = NULL, ***edge_offsets = NULL
        np.intp_t ftmp[NPY_MAXDIMS], *fcoordinates = NULL, *foffsets = NULL
        np.intp_t filter_size, odimensions[NPY_MAXDIMS]
        np.intp_t idimensions[NPY_MAXDIMS], istrides[NPY_MAXDIMS]
        np.intp_t size
        double ***splvals = NULL
        np.flatiter _io
        PyArrayIterObject *io 
        np.float64_t *zooms# = zoom_ar ? (Float64*)PyArray_DATA(zoom_ar) : NULL
        np.float64_t *shifts # = shift_ar ? (Float64*)PyArray_DATA(shift_ar) : NULL
        int rank = 0, jj, hh, kk, qq


    for kk in range(input.ndim):
        idimensions[kk] = input.dimensions[kk]
        istrides[kk] = input.strides[kk]
        odimensions[kk] = output.dimensions[kk]

    rank = input.ndim

      # If the mode is 'constant' we need some temps later:
    if 1:# if mode == NI_EXTEND_CONSTANT:
        zeros = <np.intp_t **> PyDataMem_NEW(rank * sizeof(np.intp_t* ))
        
        ############################################################### NI_unlikely.. wali condition

        for jj in range(rank):
            zeros[jj] = NULL
        for jj in range(rank):
            zeros[jj] = <np.intp_t *> PyDataMem_NEW(odimensions[jj] * sizeof(np.intp_t))
            # if (NI_UNLIKELY(!zeros[jj]))  

    # For scoring offsets along each axis
    offsets = <np.intp_t **> PyDataMem_NEW(rank * sizeof(np.intp_t *))
    # Store spine cooefficient, along each exis
    splvals = <double ***> PyDataMem_NEW(rank * sizeof(double **))
    # Store offsets at all edge_offsets
    edge_offsets = <np.intp_t ***> PyDataMem_NEW(rank * sizeof(np.intp_t **))

    ##NI_unlikely() for all 3 line 721

    for jj in range(rank):
        offsets[jj] = NULL
        splvals[jj] = NULL
        edge_offsets[jj] = NULL

    for jj in range(rank):
        offsets[jj] = <np.intp_t *> PyDataMem_NEW(odimensions[jj] * sizeof(np.intp_t))
        splvals[jj] = <double **>PyDataMem_NEW(odimensions[jj] * sizeof(double*))
        edge_offsets[jj] = <np.intp_t **>PyDataMem_NEW(odimensions[jj] * sizeof(np.intp_t*))
        # NI_unlikely on all 3

        for hh in range(odimensions[jj]):
            splvals[jj][hh] = NULL
            edge_offsets[jj][hh] = NULL

    
    cdef double shift, zoom, cc
    cdef np.intp_t start, s2
    cdef int idx, len
    # Precalculate offsets, and offsets at the edge_offsets
    for jj in range(rank):
        shift = 0.0
        zoom = 0.0
        if shifts is not NULL:
            shift = shifts[jj]

        if zooms is not NULL:
            zoom = zooms[jj]

        for kk in range(odimensions[jj]):
            cc = <double> kk
            if shifts is not NULL:
                cc += shift

            if zooms is not NULL:
                cc *= zoom

        # cc = map_coordinate(cc, idimensions[jj], mode)
        if cc > -1.0:
            if zeros is not NULL and zeros[jj] is not NULL:
                zeros[jj][kk] = 0
            if order & 1:
                start = <np.intp_t>floor(cc) - order / 2
            else:
                start = <np.intp_t>floor(cc + 0.5) - order / 2

            offsets[jj][kk] = istrides[jj] * start

            if start < 0 and start + order >= idimensions[jj]:
                edge_offsets[jj][kk] = <np.intp_t *> PyDataMem_NEW((order + 1) * sizeof(np.intp_t))
                # NI_unlikely line 772
                for hh in range(order + 1):
                    idx = start + hh
                    len = idimensions[jj]
                    if len <= 1:
                        idx = 0

                    else:
                        s2 = 2 * len -2
                        if idx < 0:
                            idx = s2 * <np.intp_t> (-idx / s2) +idx
                            # idx =   ############Apply if coniiotn line  786

                        elif idx >= len:
                            idx -= s2 * <np.intp_t> (idx / s2)
                            if idx >= len:
                                idx = s2 - idx

                        edge_offsets[jj][kk][hh] = istrides[jj] * (idx - start)

            if order > 0:
                splvals[jj][kk] = <double *> PyDataMem_NEW((order + 1) * sizeof(double))
                # NI_unlikely line 798

                spline_coefficients(cc, order, splvals[jj][kk]) #######################Functions  

            else:
                zeros[jj][kk] = 1

    filter_size = 1
    for jj in range(rank):
        filter_size *= order + 1
    
    # Iterator initialzation
    _io = np.Pyarray_Iter_New(output)
    io = <PyArrayIterObject *> _io

    fcoordinates = <np.intp_t *> PyDataMem_NEW(rank * filter_size * sizeof(np.intp_t))
    foffsets = <np.intp_t *> PyDataMem_NEW(filter_size * sizeof(np.intp_t))
    # NI_UNLIKELY ########### line 824

    for jj in range(rank):
        ftmp[jj] = 0

    kk = 0
    for hh in range(filter_size):
        for jj in range(rank):
            fcoordinates[jj + hh * rank] = ftmp[jj]
        foffsets[hh] = kk
        jj = rank - 1
        while jj >= 0:
            if ftmp[jj] < order:
                ftmp[jj] += 1
                kk += istrides[jj]
                break
            else:
                ftmp[jj] = 0
                kk -= istrides[jj] * order
                jj -= 1

    cdef double t
    cdef np.intp_t edge, oo
    cdef np.intp_t *ff
    cdef double coeff
    idx = 0
    size = output.size
    for kk in range(size):
        t = 0.0
        edge = 0
        oo = 0
        zero = 0

        for hh in range(rank):
            if zeros and zeros[hh][io.coordinates[hh]]:
                 # we use constant border condition
                zero = 1
                break
            oo += offsets[hh][io.coordinates[hh]]
            if edge_offsets[hh][io.coordinates[hh]]:
                edge = 1

        if not zero:
            ff = fcoordinates
            # cdef int type_num = NO Use.... Normalization is being done
            t = 0.0
            for hh in range(filter_size):
                idx = 0
                coeff = 0.0

                if 1:# if(NI_unlikely(edge)):
                    # Use precalculated edge offsets
                    for jj in range(rank):
                        if (edge_offsets[jj][io.coordinates[jj]][ff[jj]]):
                            idx += edge_offsets[jj][io.coordinates[jj]][ff[jj]]
                        else:
                            idx += ff[jj] * istrides[jj]


                    idx += oo

            else:
                # use normal offsets:
                idx += oo + foffsets[hh]

            # use fucntion for getting values
            # func get value
            # calculate inerpolated values:

            for jj in range(rank):
                if order > 0:
                    coeff *= splvals[jj][io.coordinates[jj]][ff[jj]]
            t += coeff
            ff += rank

    else:
        t = cval

    # Store output
    # Again some kind of macros from line 919

    if zeros:
        for jj in range(rank):
            PyDataMem_FREE(zeros[jj])
        PyDataMem_FREE(zeros)

    if offsets:
        for jj in range(rank):
            PyDataMem_FREE(offsets[jj])
        PyDataMem_FREE(offsets)

    if splvals:
        for jj in range(rank):
            if splvals[jj]:
                for hh in range(odimensions[jj]):
                    PyDataMem_FREE(edge_offsets[jj][hh])
                PyDataMem_FREE(edge_offsets[jj])
     
        PyDataMem_FREE(edge_offsets)

    PyDataMem_FREE(foffsets)
    PyDataMem_FREE(fcoordinates)

