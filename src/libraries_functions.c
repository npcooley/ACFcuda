/* ============================================================================
 * libraries_functions.c
 * author: nicholas cooley
 * maintainer: nicholas cooley
 *
 * we need to drive the function that exposes kernels to be run
 * ========================================================================= */

#include <Rinternals.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "ACFcuda.h"

/* ============================================================================
 * SECTION: internal helpers
 * eventually move these to utils as globals...
 * ========================================================================= */

// create a human readable string for a CUresult value
// cuGetErrorString writes a pointer into *out_str; the string is owned by the
// driver and must not be freed
// falls back to numeric representation if the driver does not recognize the
// code ...
static void cuda_driver_error_string(CUresult res,
                                     char *buf,
                                     size_t buf_size) {
  const char *str = NULL;
  CUresult query = cuGetErrorString(res, &str);
  if (query == CUDA_SUCCESS && str != NULL) {
    strncpy(buf, str, buf_size - 1);
    buf[buf_size - 1] = '\0';
  } else {
    snprintf(buf, buf_size, "CUresult %d", (int)res);
  }
}

/* ============================================================================
 * SECTION: kernel handlers
 * ========================================================================= */

// extract a function representation from a ptx intermediate file
SEXP c_cuda_kernel_from_ptx(SEXP cuda_context_sexp,
                            SEXP ptx_path_sexp,
                            SEXP kernel_name_sexp) {
  
  // validate inputs, 
  // this needs to be an r external pointer
  if (TYPEOF(cuda_context_sexp) != EXTPTRSXP) {
    Rf_error("'cuda_context' must be an external pointer from cuda_make_context()");
  }
  
  // cast the pointer from R into C
  // R_ExternalPtrAddr returns a void* (pointer)
  // we cast that in and assure C that it's of type 'CudaContext'
  CudaContext *ctx = (CudaContext *)R_ExternalPtrAddr(cuda_context_sexp);
  if (ctx == NULL) {
    Rf_error("'cuda_context' pointer is NULL or has been released");
  }
  
  // further validation and import
  if (TYPEOF(ptx_path_sexp) != STRSXP || LENGTH(ptx_path_sexp) != 1) {
    Rf_error("'ptx_file' must be a character scalar");
  }
  if (TYPEOF(kernel_name_sexp) != STRSXP || LENGTH(kernel_name_sexp) != 1) {
    Rf_error("'kernel_name' must be a character scalar");
  }
  
  const char *ptx_path = CHAR(STRING_ELT(ptx_path_sexp,    0));
  const char *kernel_name = CHAR(STRING_ELT(kernel_name_sexp, 0));
  
  // wrapper for cuInit(0) call, this is a check more than anything, because
  // it should have already been called, we check it to be defensive
  if (cuda_driver_init() != 0) {
    Rf_error("cuInit() failed");
  }
  
  // set the device from the context
  // thing1->thing2
  // reads the 'thing2' field of the 'thing1' struct
  // equivalent to (*thing1).thing2
  cudaError_t err = cudaSetDevice(ctx->device_index);
  if (err != cudaSuccess) {
    Rf_error("cudaSetDevice() failed for device %d: %s",
             ctx->device_index,
             cudaGetErrorString(err));
  }
  
  // this is the cuda concept of context not *ours*, it is the handle that
  // the Driver API uses to interact with what the Runtime API creates with
  // cudaSetDevice
  // ...
  // cuCtxGetCurrent is going to overwrite the address for cu_ctx, and res
  // is a success / failure flag that we can check
  CUcontext cu_ctx = NULL;
  CUresult res = cuCtxGetCurrent(&cu_ctx);
  
  if (res != CUDA_SUCCESS || cu_ctx == NULL) {
    Rf_error("no current CUDA context; cuda_make_context() must be called first");
  }
  
  // cuModuleLoadData expects the ptx file as a null terminated string, not a 
  // filepath, we need to convert our file to the thing we want with the 
  // read_file_to_buffer function...
  char  read_err[512];
  char *ptx_buf = read_file_to_buffer(ptx_path, read_err, sizeof(read_err));
  
  if (ptx_buf == NULL) {
    Rf_error("%s", read_err);
  }
  
  // finally read in the module
  CUmodule module = NULL;
  res = cuModuleLoadData(&module, ptx_buf);
  free(ptx_buf);
  
  if (res != CUDA_SUCCESS) {
    char err_str[256];
    cuda_driver_error_string(res, err_str, sizeof(err_str));
    Rf_error("cuModuleLoadData() failed: %s", err_str);
  }
  
  // retrieve the function handle
  CUfunction function = NULL;
  res = cuModuleGetFunction(&function, module, kernel_name);
  
  if (res != CUDA_SUCCESS) {
    char err_str[256];
    cuda_driver_error_string(res, err_str, sizeof(err_str));
    cuModuleUnload(module);
    Rf_error("cuModuleGetFunction() failed for kernel '%s': %s",
             kernel_name, err_str);
  }
  
  // populate our struct for R
  CudaKernel *dk = (CudaKernel *)malloc(sizeof(CudaKernel));
  if (dk == NULL) {
    cuModuleUnload(module);
    Rf_error("failed to allocate CudaKernel struct");
  }
  
  dk->module = module;
  dk->function = function;
  
  strncpy(dk->kernel_name, kernel_name, sizeof(dk->kernel_name) - 1);
  dk->kernel_name[sizeof(dk->kernel_name) - 1] = '\0';
  
  // wrap up our struct in a pointer and a finalizer
  SEXP result = PROTECT(R_MakeExternalPtr(dk, R_NilValue, R_NilValue));
  R_RegisterCFinalizer(result, cuda_kernel_finalizer);
  setAttrib(result, R_ClassSymbol, mkString("cudaKernel"));
  
  UNPROTECT(1);
  return result;
}
