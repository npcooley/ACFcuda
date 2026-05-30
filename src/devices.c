/* ============================================================================
 * devices.c
 * author: nicholas cooley
 * maintainer: nicholas cooley
 *
 * CUDA Runtime API device interrogation functions
 *
 * CUDA does not really have a concept of a 'default' device the way that
 * METAL does therefore device selection will 'default' to the first enumerated
 * device. The ability to select will need to be exposed eventually
 * 
 * 
 * DESIGN NOTE vs ACFmetal/devices.m:
 *   Metal requires ObjC bridge casting (__bridge_retained / CFRelease) to
 *   move device handles across the ARC boundary into C.  CUDA device handles
 *   are plain integers -- there is no object to retain, release, or bridge.
 *   All functions here take or return int / size_t / const char*, never
 *   opaque pointers to GPU-managed objects.
 *
 * STATIC BUFFER WARNING:
 *   cuda_device_name() returns a pointer to a static char buffer.  The
 *   caller (devices.c) must copy the string (e.g. via mkString()) before
 *   this function is called again.  This matches the pattern used in
 *   ACFmetal's metal_device_name() which returns [NSString UTF8String].
 * ========================================================================= */

#include <cuda_runtime.h>
#include <cuda.h>
#include <Rinternals.h>
#include <stdio.h>
#include <string.h>
#include "ACFcuda.h"

/* ============================================================================
 * SECTION: hooks and things
 * ========================================================================= */

// hook function, return TRUE if at least 1 CUDA device is present
SEXP c_cuda_device_hook(void) {
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess) {
    return ScalarLogical(FALSE);
  }
  return ScalarLogical(count > 0);
}

SEXP c_cuda_device_count(void) {
  int count = 0;
  cudaGetDeviceCount(&count);
  return ScalarInteger(count);
}

/* ============================================================================
 * SECTION: device information
 * ========================================================================= */

// get relevant device information
SEXP c_cuda_device_information(SEXP device_index_sexp) {
  
  // input validation
  if (TYPEOF(device_index_sexp) != INTSXP &&
      TYPEOF(device_index_sexp) != REALSXP) {
    Rf_error("device index must be an integer or numeric");
  }
  
  int device_index;
  if (TYPEOF(device_index_sexp) == INTSXP) {
    device_index = INTEGER(device_index_sexp)[0];
  } else {
    device_index = (int)REAL(device_index_sexp)[0];
  }
  
  // index bound checks
  int total_devices = 0;
  cudaGetDeviceCount(&total_devices);
  
  if (device_index < 0 || device_index >= total_devices) {
    Rf_error("device_index %d is out of range; %d device(s) present",
             device_index,
             total_devices);
  }
  
  // get cuda's built in properties structure
  struct cudaDeviceProp props;
  cudaError_t err = cudaGetDeviceProperties(&props, device_index);
  if (err != cudaSuccess) {
    Rf_error("cudaGetDeviceProperties failed for device %d", device_index);
  }
  
  // get information from the properties structure
  int n_fields = 16;
  SEXP result = PROTECT(allocVector(VECSXP, n_fields));
  SEXP names  = PROTECT(allocVector(STRSXP, n_fields));
  
  int i = 0;
  
  // device id and general info
  SET_STRING_ELT(names, i, mkChar("name"));
  SET_VECTOR_ELT(result, i, mkString(props.name));
  i++;
  
  SET_STRING_ELT(names, i, mkChar("device_index"));
  SET_VECTOR_ELT(result, i, ScalarInteger(device_index));
  i++;
  
  SET_STRING_ELT(names, i, mkChar("compute_capability_major"));
  SET_VECTOR_ELT(result, i, ScalarInteger(props.major));
  i++;
  
  SET_STRING_ELT(names, i, mkChar("compute_capability_minor"));
  SET_VECTOR_ELT(result, i, ScalarInteger(props.minor));
  i++;
  
  // device memory
  SET_STRING_ELT(names, i, mkChar("total_memory_bytes"));
  SET_VECTOR_ELT(result, i, ScalarReal((double)props.totalGlobalMem));
  i++;
  
  SET_STRING_ELT(names, i, mkChar("shared_mem_per_block_bytes"));
  SET_VECTOR_ELT(result, i, ScalarReal((double)props.sharedMemPerBlock));
  i++;
  
  SET_STRING_ELT(names, i, mkChar("can_map_host_memory"));
  SET_VECTOR_ELT(result, i, ScalarLogical(props.canMapHostMemory));
  i++;
  
  SET_STRING_ELT(names, i, mkChar("is_integrated"));
  SET_VECTOR_ELT(result, i, ScalarLogical(props.integrated));
  i++;
  
  // threads and things
  SET_STRING_ELT(names, i, mkChar("multiprocessor_count"));
  SET_VECTOR_ELT(result, i, ScalarInteger(props.multiProcessorCount));
  i++;
  
  SET_STRING_ELT(names, i, mkChar("warp_size"));
  SET_VECTOR_ELT(result, i, ScalarInteger(props.warpSize));
  i++;
  
  SET_STRING_ELT(names, i, mkChar("max_threads_per_block"));
  SET_VECTOR_ELT(result, i, ScalarInteger(props.maxThreadsPerBlock));
  i++;
  
  SEXP block_dims = PROTECT(allocVector(INTSXP, 3));
  INTEGER(block_dims)[0] = props.maxThreadsDim[0];
  INTEGER(block_dims)[1] = props.maxThreadsDim[1];
  INTEGER(block_dims)[2] = props.maxThreadsDim[2];
  SET_STRING_ELT(names, i, mkChar("max_block_dims"));
  SET_VECTOR_ELT(result, i, block_dims);
  UNPROTECT(1);
  i++;
  
  SEXP grid_dims = PROTECT(allocVector(INTSXP, 3));
  INTEGER(grid_dims)[0] = props.maxGridSize[0];
  INTEGER(grid_dims)[1] = props.maxGridSize[1];
  INTEGER(grid_dims)[2] = props.maxGridSize[2];
  SET_STRING_ELT(names, i, mkChar("max_grid_dims"));
  SET_VECTOR_ELT(result, i, grid_dims);
  UNPROTECT(1);
  i++;
  
  SET_STRING_ELT(names, i, mkChar("max_threads_per_multiprocessor"));
  SET_VECTOR_ELT(result, i, ScalarInteger(props.maxThreadsPerMultiProcessor));
  i++;
  
  SET_STRING_ELT(names, i, mkChar("memory_clock_rate_khz"));
  SET_VECTOR_ELT(result, i, ScalarInteger(props.memoryClockRate));
  i++;
  
  SET_STRING_ELT(names, i, mkChar("memory_bus_width_bits"));
  SET_VECTOR_ELT(result, i, ScalarInteger(props.memoryBusWidth));
  i++;
  
  setAttrib(result, R_NamesSymbol, names);
  UNPROTECT(2);
  return result;
}

