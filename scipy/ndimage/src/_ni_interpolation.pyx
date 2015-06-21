# Cython Version of ni_interpolation.pyx

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


# Define all the required files:

cpdef int _zoom_shift(np.ndarray input, np.ndarray zoom, np.ndarray shift, np.ndarray output, int integer, int mode, double cval):
    cdef:
    	np.intp_t **zeroes = NULL, **offsets = NULL, ***edge_offsets = NULL
    	np.intp_t ftmp[MAXDIM], *fcoordinates = NULL, *foffsets = NULL
    	np.intp_t filter_size, odimensions[MAXDIM]
    	np.intp_t idimensions[MAXDIM], istrides[MAXDIM]
    	np.intp_t size
    	double ***splvals = NULL
    	np.flatiter _io
    	PyArrayIterObject *io 
    	np.float *zooms# = zoom_ar ? (Float64*)PyArray_DATA(zoom_ar) : NULL
    	Float64 *shifts# = shift_ar ? (Float64*)PyArray_DATA(shift_ar) : NULL
    	int rank = 0, jj, hh, kk, qq


    for kk in range(input.ndim):
    	idimensions[kk] = input.dimensions[kk]
    	istrides[kk] = input.strides[kk]
    	odimensions[kk] = output.dimensions[kk]

    rank = input.ndim

      # If the mode is 'constant' we need some temps later:
    if mode == NI_EXTEND_CONSTANT:
        zeros = <np.intp_t **> PyDataMem_NEW(rank * sizeof(npy_intp*))
        
        ############################################################### NI_unlikely.. wali condition

        for jj in range(rank):
            zeros[jj] = NULL
        for jj in range(rank):
            zeros[jj] = <np.intp_t *> PyDataMem_NEW(odimensions[jj] * sizeof(npy_intp))
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
        splvals[jj] = PyDataMem_NEW(odimensions[jj] * sizeof(double*))
        edge_offsets[jj] = PyDataMem_NEW(odimensions[jj] * sizeof(npy_intp*))
        # NI_unlikely on all 3

        for hh in range(odimensions[jj]):
        	splvals[jj][hh] = NULL
        	edge_offsets[jj][hh] = NULL

    
    cdef double shift, zoom, cc
    cdef np.intp_t start
    # Precalculate offsets, and offsets at the edge_offsets
    for jj in range(rank):
    	shift = 0.0
    	zoom = 0.0
    	if shifts is not NULL:
    		shift = shifts[jj]

    	if zoooms is not NULL:
    		zoom = zoooms[jj]

    	for kk in range(odimensions[jj]):
            cc = <double> kk
            if shifts is not NULL:
            	cc += shift

            if zooms is not NULL:
            	cc *= zoom

        cc = map_coordinate(cc, idimensions[jj], mode)
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
                	cdef int idx, len
                	idx = start + hh
                	len = idimensions[jj]
                	if len <= 1:
                		idx = 0

                	else:
                		np.intp_t s2 = 2 * len -2
                		if idx < 0:
                			idx = s2 * <np.intp_t> (-idx / s2) +idx
                			idx = 	############Apply if coniiotn line  786

                		else: if idx >= len:
                			idx -= s2 * <np.intp_t> (idx / s2)
                			if idx >= len:
                				idx = s2 - idx

                		edge_offsets[jj][kk][hh] = istrides[jj] * (idx - start)

            if order > 0:
            	splvals[jj][kk] = <double *> PyDataMem_NEW((order + ) * sizeof(double))
                # NI_unlikely line 798

                spline_coefficients(cc, order, splvals[jj][kk])

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
    	ftemp[jj] = 0

    kk = 0
    for hh in range(filter_size):
    	for jj in range(rank):
    		fcoordinates[jj + hh * rank] = ftmp[jj]
    	offsets[hh] = kk
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
    cdef np.intp_t edge, oo, edge
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
    		cdef np.intp_t *ff = fcoordinates
    		# cdef int type_num = NO Use.... Normalization is being done
    		t = 0.0
    		for hh in range(filter_size):
    			cdef np.intp_t idx = 0
    			cdef double coeff = 0.0

    			if(NI_unlikely(edge)):
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
    	for jj in range(offsets):
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

