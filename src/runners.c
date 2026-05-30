/* ============================================================================
 * runners.c
 * author: nicholas cooley
 * maintainer: nicholas cooley
 *
 * R-callable entry point for simple CUDA kernel dispatch.
 * currently supports ptx dispatch ... honestly this was just a 'winging it'
 * choice, it felt the most thematically similar to what i chose to do with
 * metal, but supporting other runners isn't out of the question
 * ========================================================================= */

#include <Rinternals.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdlib.h>
#include <string.h>
#include "ACFcuda.h"

/* ============================================================================
 * cuda_simple_runner
 * 
 * only supports integer and numeric results for now...
 * 
 * context_ptr == ptr container for device index
 * kernel_ptr == ptr container for the ptx function representation
 * arg_types == character vector
 * args_list == a list of R vectors, currently only supports integer and
 *  numeric inputs
 * work_dims == NULL, or a vector of length 3
 * block_dims == NULL, or a vector of length 3
 * ========================================================================= */

SEXP cuda_simple_runner(SEXP context_ptr,
                        SEXP kernel_ptr,
                        SEXP arg_types,
                        SEXP args_list,
                        SEXP work_dims,
                        SEXP block_dims) {
  
  // validating inputs
  // context and device
  if (TYPEOF(context_ptr) != EXTPTRSXP) {
    Rf_error("'cuda_context' must be an external pointer from cuda_make_context()");
  }
  CudaContext *ctx = (CudaContext *)R_ExternalPtrAddr(context_ptr);
  if (ctx == NULL) {
    Rf_error("'cuda_context' pointer is NULL or has been released");
  }
  
  cudaError_t set_err = cudaSetDevice(ctx->device_index);
  if (set_err != cudaSuccess) {
    Rf_error("cudaSetDevice() failed for device %d: %s",
             ctx->device_index,
             cudaGetErrorString(set_err));
  }
  
  // kernel function
  if (TYPEOF(kernel_ptr) != EXTPTRSXP) {
    Rf_error("'kernel' must be an external pointer from cuda_kernel_from_ptx()");
  }
  CudaKernel *kernel = (CudaKernel *)R_ExternalPtrAddr(kernel_ptr);
  if (kernel == NULL) {
    Rf_error("'kernel' pointer is NULL or has been released");
  }
  
  // arg types and arg list
  if (TYPEOF(arg_types) != STRSXP || LENGTH(arg_types) < 1) {
    Rf_error("'arg_types' must be a non-empty character vector");
  }
  if (TYPEOF(arg_list) != VECSXP) {
    Rf_error("'args_list' must be an R list");
  }
  if (LENGTH(arg_types) != LENGTH(arg_list)) {
    Rf_error("length of 'arg_types' must equal length of 'args_list'");
  }
  
  int total_args = LENGTH(arg_types);
  int total_types = LENGTH(arg_types);
  if (total_args != total_types) {
    Rf_error("'arg_types' and 'arg_list' must be of equal length.")
  }
  
  // output template
  SEXP output_template = VECTOR_ELT(arg_list, 0);
  if (TYPEOF(output_template) != REALSXP && TYPEOF(output_template) != INTSXP) {
    Rf_error("first element of 'args_list' must be a numeric or integer vector representing the output template");
  }
  
  const char *first_type_str = CHAR(STRING_ELT(arg_types, 0));
  CudaType output_type = cuda_parse_type(first_type_str);
  int output_length = LENGTH(output_template);
  
  // resolve work dims if supplied, else just default to the output length
  if (work_dims != R_NilValue) {
    if ((TYPEOF(work_dims) != REALSXP &&
        TYPEOF(work_dims) != INTSXP) ||
        LENGTH(work_dims) != 3) {
      Rf_error("'work_dims' must be a numeric or integer vector of length 3");
    }
    int work_dims[3] = {output_length, 1, 1};
    for (int d = 0; d < 3; d++) {
      work_dims[d] = (TYPEOF(work_dims) == REALSXP)
      ? (int)REAL(work_dims)[d]
      : INTEGER(work_dims)[d];
    }
  } else {
    int work_dims[3] = {output_length, 1, 1};
  }
  
  // resolve block dimensions if supplied
  // default block dims are a little more complicated and are based on the work
  // dims
  
  if (block_dims != R_NilValue) {
    if ((TYPEOF(block_dims) != REALSXP &&
        TYPEOF(block_dims) != INTSXP) ||
        LENGTH(block_dims) != 3) {
      Rf_error("'block_dims' must be a numeric or integer vector of length 3");
    }
    int block_dims[3] = {256, 1, 1};
    for (int d = 0; d < 3; d++) {
      block_dims[d] = (TYPEOF(block_dims) == REALSXP)
      ? (int)REAL(block_dims)[d]
      : INTEGER(block_dims)[d];
    }
  } else {
    int block_dims[3] = {256, 1, 1};
    if (work_dims[2] > 1) {
      block_dims[0] = 8;  block_dims[1] = 8;  block_dims[2] = 4;
    } else if (work_dims[1] > 1) {
      block_dims[0] = 16; block_dims[1] = 16; block_dims[2] = 1;
    }
    // ending else condition already covered...
  }
  
  // compute grid dimensions
  int grid_dims[3];
  for (int d = 0; d < 3; d++) {
    grid_dims[d] = (work_dims[d] + block_dims[d] - 1) / block_dims[d];
    if (grid_dims[d] < 1) { grid_dims[d] = 1; }
  }
  
  // count up buffers and scalars
  int buffer_count = 1;  /* index 0 reserved for output */
  int scalar_count = 0;
  
  for (int i = 1; i < total_args; i++) {
    SEXP arg = VECTOR_ELT(arg_list, i);
    if (LENGTH(arg) == 1) {
      scalar_count++;
    } else {
      buffer_count++;
    }
  }
  
  int num_input_args = buffer_count - 1 + scalar_count;
  
  // argument tracking arrays
  void **device_buffers = (void **)malloc(buffer_count * sizeof(void *));
  void **host_staging = (void **)malloc(buffer_count * sizeof(void *));
  void **scalar_values = (void **)malloc((scalar_count == 0 ? 1 : scalar_count) * sizeof(void *));
  size_t *scalar_sizes = (size_t *)malloc((scalar_count == 0 ? 1 : scalar_count) * sizeof(size_t));
  int *is_scalar = (int *)malloc((num_input_args == 0 ? 1 : num_input_args) * sizeof(int));
  
  // exit early if something is a problem
  if (device_buffers == NULL ||
      host_staging == NULL ||
      scalar_values  == NULL ||
      scalar_sizes == NULL ||
      is_scalar == NULL) {
    if (device_buffers) {
      free(device_buffers);
    }
    if (host_staging) {
      free(host_staging);
    }
    if (scalar_values) {
      free(scalar_values);
    }
    if (scalar_sizes) {
      free(scalar_sizes);
    }
    if (is_scalar) {
      free(is_scalar);
    }
    Rf_error("failed to allocate argument tracking arrays");
  }
  
  for (int i = 0; i < buffer_count; i++) {
    device_buffers[i] = NULL;
    host_staging[i]   = NULL;
  }
  
  // allocate output buffer on the device
  size_t output_elem_size = cuda_get_element_size(output_type);
  size_t output_byte_size = (size_t)output_length * output_elem_size;
  
  device_buffers[0] = cuda_device_alloc(output_byte_size);
  if (device_buffers[0] == NULL) {
    free(device_buffers);
    free(host_staging);
    free(scalar_values);
    free(scalar_sizes);
    free(is_scalar);
    Rf_error("failed to allocate output buffer on device");
  }
  
  // input args
  int buf_idx = 1;
  int scl_idx = 0;
  int arg_pos = 0;
  
  for (int i = 1; i < total_args; i++) {
    const char *type_str = CHAR(STRING_ELT(arg_types, i));
    SEXP arg = VECTOR_ELT(arg_list, i);
    CudaType arg_type = cuda_parse_type(type_str);
    size_t arg_elem_size = cuda_get_element_size(arg_type);
    size_t arg_length = (size_t)LENGTH(arg);
    
    if (TYPEOF(arg) != REALSXP &&
        TYPEOF(arg) != INTSXP) {
      for (int j = 0; j < buf_idx; j++) {
        cuda_device_free(device_buffers[j]);
        if (host_staging[j] != NULL) {
          free(host_staging[j]);
        }
      }
      for (int j = 0; j < scl_idx; j++) {
        free(scalar_values[j]);
      }
      free(device_buffers);
      free(host_staging);
      free(scalar_values);
      free(scalar_sizes);
      free(is_scalar);
      Rf_error("argument %d must be a numeric or integer vector", i);
    }
    
    // length check is the consistent check for whether or not is a scalar
    // set up device conversions
    if (arg_length == 1) {
      // is a scalar
      is_scalar[arg_pos++] = 1;
      scalar_sizes[scl_idx] = arg_elem_size;
      scalar_values[scl_idx] = malloc(arg_elem_size);
      if (scalar_values[scl_idx] == NULL) {
        for (int j = 0; j < buf_idx; j++) {
          cuda_device_free(device_buffers[j]);
          if (host_staging[j] != NULL) {
            free(host_staging[j]);
          }
        }
        for (int j = 0; j < scl_idx; j++) {
          free(scalar_values[j]);
        }
        free(device_buffers);
        free(host_staging);
        free(scalar_values);
        free(scalar_sizes);
        free(is_scalar);
        Rf_error("failed to allocate scalar staging buffer for argument %d", i);
      }
      if (TYPEOF(arg) == REALSXP) {
        cuda_convert_r_numeric_to_host(REAL(arg),
                                       scalar_values[scl_idx],
                                       1,
                                       arg_type);
      } else {
        cuda_convert_r_int_to_host(INTEGER(arg),
                                   scalar_values[scl_idx],
                                   1,
                                   arg_type);
      }
      scl_idx++;
      
    } else {
      // is not a scalar
      is_scalar[arg_pos++] = 0;
      size_t arg_byte_size = arg_length * arg_elem_size;
      
      host_staging[buf_idx] = malloc(arg_byte_size);
      if (host_staging[buf_idx] == NULL) {
        for (int j = 0; j < buf_idx; j++) {
          cuda_device_free(device_buffers[j]);
          if (host_staging[j] != NULL) {
            free(host_staging[j]);
          }
        }
        for (int j = 0; j < scl_idx; j++) {
          free(scalar_values[j]);
        }
        free(device_buffers);
        free(host_staging);
        free(scalar_values);
        free(scalar_sizes);
        free(is_scalar);
        Rf_error("failed to allocate host staging buffer for argument %d", i);
      }
      
      if (TYPEOF(arg) == REALSXP) {
        cuda_convert_r_numeric_to_host(REAL(arg),
                                       host_staging[buf_idx],
                                       arg_length,
                                       arg_type);
      } else {
        cuda_convert_r_int_to_host(INTEGER(arg),
                                   host_staging[buf_idx],
                                   arg_length,
                                   arg_type);
      }
      
      device_buffers[buf_idx] = cuda_device_alloc(arg_byte_size);
      if (device_buffers[buf_idx] == NULL) {
        free(host_staging[buf_idx]);
        for (int j = 0; j < buf_idx; j++) {
          cuda_device_free(device_buffers[j]);
          if (host_staging[j] != NULL) {
            free(host_staging[j]);
          }
        }
        for (int j = 0; j < scl_idx; j++) {
          free(scalar_values[j]);
        }
        free(device_buffers);
        free(host_staging);
        free(scalar_values);
        free(scalar_sizes);
        free(is_scalar);
        Rf_error("failed to allocate device buffer for argument %d", i);
      }
      
      cuda_memcpy_to_device(device_buffers[buf_idx],
                            host_staging[buf_idx],
                            arg_byte_size);
      buf_idx++;
    }
  }
  
  /* ---- 11. build kernel_params array for cuLaunchKernel ------------------ */
  /* cuLaunchKernel expects void** where each element points to a kernel
   * argument -- for buffer args, a pointer to the device pointer;
   * for scalar args, a pointer to the host scalar value               */
  
  int total_params = 1 + num_input_args;
  void **kernel_params = (void **)malloc(total_params * sizeof(void *));
  if (kernel_params == NULL) {
    for (int i = 0; i < buffer_count; i++) {
      cuda_device_free(device_buffers[i]);
      if (host_staging[i] != NULL) {
        free(host_staging[i]);
      }
    }
    for (int i = 0; i < scalar_count; i++) {
      free(scalar_values[i]);
    }
    free(device_buffers);
    free(host_staging);
    free(scalar_values);
    free(scalar_sizes);
    free(is_scalar);
    Rf_error("failed to allocate kernel_params array");
  }
  
  // index zero is the output / result buffer
  kernel_params[0] = &device_buffers[0];
  
  // ensure that original order is maintained for submission and arg passing
  int kp_buf_idx = 1;
  int kp_scl_idx = 0;
  for (int i = 0; i < num_input_args; i++) {
    if (is_scalar[i]) {
      kernel_params[1 + i] = scalar_values[kp_scl_idx++];
    } else {
      kernel_params[1 + i] = &device_buffers[kp_buf_idx++];
    }
  }
  
  // function dispatch
  CUresult launch_res = cuLaunchKernel(kernel->function,
                                       (unsigned int)grid_dims[0],
                                       (unsigned int)grid_dims[1],
                                       (unsigned int)grid_dims[2],
                                       (unsigned int)block_dims[0],
                                       (unsigned int)block_dims[1],
                                       (unsigned int)block_dims[2],
                                       0,
                                       (CUstream)ctx->stream,
                                       kernel_params,
                                       NULL);
  free(kernel_params);
  
  if (launch_res != CUDA_SUCCESS) {
    for (int i = 0; i < buffer_count; i++) {
      cuda_device_free(device_buffers[i]);
      if (host_staging[i] != NULL) {
        free(host_staging[i]);
      }
    }
    for (int i = 0; i < scalar_count; i++) {
      free(scalar_values[i]);
    }
    free(device_buffers);
    free(host_staging);
    free(scalar_values);
    free(scalar_sizes);
    free(is_scalar);
    Rf_error("cuLaunchKernel() failed for kernel '%s': CUresult %d",
             kernel->kernel_name,
             (int)launch_res);
  }
  
  // device synchronization
  cuda_synchronize(ctx);
  
  // return output back to R
  void *output_host = malloc(output_byte_size);
  if (output_host == NULL) {
    for (int i = 0; i < buffer_count; i++) {
      cuda_device_free(device_buffers[i]);
      if (host_staging[i] != NULL) {
        free(host_staging[i]);
      }
    }
    for (int i = 0; i < scalar_count; i++) {
      free(scalar_values[i]);
    }
    free(device_buffers);
    free(host_staging);
    free(scalar_values);
    free(scalar_sizes);
    free(is_scalar);
    Rf_error("failed to allocate host buffer for output readback");
  }
  
  cuda_memcpy_to_host(output_host,
                      device_buffers[0],
                      output_byte_size);
  
  SEXP result = PROTECT(allocVector(REALSXP, output_length));
  cuda_convert_host_to_r(output_host,
                         REAL(result),
                         (size_t)output_length,
                         output_type);
  free(output_host);
  
  // release device and host buffers
  for (int i = 0; i < buffer_count; i++) {
    cuda_device_free(device_buffers[i]);
    if (host_staging[i] != NULL) {
      free(host_staging[i]);
    }
  }
  for (int i = 0; i < scalar_count; i++) {
    free(scalar_values[i]);
  }
  free(device_buffers);
  free(host_staging);
  free(scalar_values);
  free(scalar_sizes);
  free(is_scalar);
  
  UNPROTECT(1);
  return result;
}


