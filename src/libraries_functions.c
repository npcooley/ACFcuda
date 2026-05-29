/* ============================================================================
 * libraries_functions.c
 * author: nicholas cooley
 * maintainer: nicholas cooley
 *
 * compiling and loading a new function written from scratch feels conceptually
 * different from how Metal does it, but is hypothetically analogous, with
 * some diverging programming patterns
 * shell calls to nvcc perform compilation, and we need the driver api
 * to dynamically load these
 * Driver API functions for loading PTX binaries at runtime and retrieving
 * CUfunction handles from them.
 *
 * this file is the dynamic-loading counterpart to the static launcher
 * registry in runners.c.  where runners.c dispatches kernels that were
 * compiled into the package binary at build time, the functions here load
 * PTX that was compiled at runtime (by compile_and_manage() in R) and
 * return an opaque handle the dynamic runner can dispatch through.
 *
 * Claude note -- check this as development progresses: Driver API vs Runtime API
 *   the Runtime API (cuda_runtime.h, cudaMalloc, cudaMemcpy, <<<>>>)
 *   has no mechanism for loading new kernel code into a running process.
 *   kernel addresses are resolved by the linker at build time and the
 *   <<<>>> dispatch syntax is a compiler extension, not a runtime call.
 *
 *   the Driver API (cuda.h, CUmodule, CUfunction, cuLaunchKernel) exists
 *   precisely to support runtime loading.  both APIs can coexist in the
 *   same process; they share the same underlying CUDA context.
 *
 *   the only extra requirement is:
 *     - link against libcuda in addition to libcudart
 *     - call cuInit(0) once before any Driver API function
 *     - retrieve the Runtime API's implicit context with cuCtxGetCurrent()
 *       rather than creating a second context
 *
 * STRUCT LAYOUT:
 *   CudaDynamicKernel holds the CUmodule (the loaded PTX binary) and the
 *   CUfunction (the handle to the specific kernel within that module).
 *   both are stored so the finalizer can unload the module when R's garbage
 *   collector collects the external pointer.
 *   kernel_name is stored for error messages in the runner.
 * ========================================================================= */

#include <Rinternals.h>
#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "ACFcuda.h"

/* ============================================================================
 * SECTION: internal helpers
 * ========================================================================= */


/* --------------------------------------------------------------------------
 * cuda_driver_error_string
 * retrieve a human-readable string for a CUresult value.
 * cuGetErrorString writes a pointer into *out_str; the string is owned by
 * the driver and must not be freed.
 * falls back to a numeric representation if the driver does not recognise
 * the code (should not happen in practice).
 * -------------------------------------------------------------------------- */
static void cuda_driver_error_string(CUresult res,
                                     char     *buf,
                                     size_t    buf_size) {
  const char *str = NULL;
  CUresult    query = cuGetErrorString(res, &str);
  if (query == CUDA_SUCCESS && str != NULL) {
    strncpy(buf, str, buf_size - 1);
    buf[buf_size - 1] = '\0';
  } else {
    snprintf(buf, buf_size, "CUresult %d", (int)res);
  }
}

/* --------------------------------------------------------------------------
 * read_file_to_buffer
 * read the entire contents of a file into a newly malloc'd buffer.
 * caller is responsible for free()'ing the returned pointer.
 * returns NULL and writes an error message into err_buf on failure.
 * -------------------------------------------------------------------------- */
static char *read_file_to_buffer(const char *path,
                                 char       *err_buf,
                                 size_t      err_buf_size) {
  FILE *fh = fopen(path, "rb");
  if (fh == NULL) {
    snprintf(err_buf, err_buf_size,
             "could not open PTX file for reading: %s", path);
    return NULL;
  }
  
  /* determine file size */
  if (fseek(fh, 0L, SEEK_END) != 0) {
    snprintf(err_buf, err_buf_size,
             "fseek failed on PTX file: %s", path);
    fclose(fh);
    return NULL;
  }
  
  long file_size = ftell(fh);
  if (file_size < 0L) {
    snprintf(err_buf, err_buf_size,
             "ftell failed on PTX file: %s", path);
    fclose(fh);
    return NULL;
  }
  if (file_size == 0L) {
    snprintf(err_buf, err_buf_size,
             "PTX file is empty: %s", path);
    fclose(fh);
    return NULL;
  }
  
  rewind(fh);
  
  /* allocate buffer -- +1 for null terminator; cuModuleLoadData expects a
   * null-terminated PTX string */
  char *buf = (char *)malloc((size_t)file_size + 1);
  if (buf == NULL) {
    snprintf(err_buf, err_buf_size,
             "failed to allocate %ld bytes for PTX buffer", file_size + 1);
    fclose(fh);
    return NULL;
  }
  
  size_t bytes_read = fread(buf, 1, (size_t)file_size, fh);
  fclose(fh);
  
  if ((long)bytes_read != file_size) {
    snprintf(err_buf, err_buf_size,
             "fread returned %zu bytes, expected %ld, for file: %s",
             bytes_read, file_size, path);
    free(buf);
    return NULL;
  }
  
  buf[file_size] = '\0';
  return buf;
}

/* ============================================================================
 * SECTION: finalizer
 * ========================================================================= */

/* --------------------------------------------------------------------------
 * cuda_dynamic_kernel_finalizer
 * called by R's garbage collector when a cudaDynamicKernel external pointer
 * is collected.  unloads the CUmodule (which also invalidates all CUfunction
 * handles derived from it) and frees the struct.
 *
 * note: CUfunction handles do not need to be explicitly released -- they are
 * owned by the CUmodule and are invalidated when the module is unloaded.
 * -------------------------------------------------------------------------- */
void cuda_dynamic_kernel_finalizer(SEXP ptr) {
  CudaDynamicKernel *dk = (CudaDynamicKernel *)R_ExternalPtrAddr(ptr);
  if (dk == NULL) {
    return;
  }
  if (dk->module != NULL) {
    /* ignore return value -- nothing useful to do if unload fails at GC time */
    cuModuleUnload(dk->module);
    dk->module   = NULL;
    dk->function = NULL;
  }
  free(dk);
  R_ClearExternalPtr(ptr);
}

/* ============================================================================
 * SECTION: c_cuda_load_kernel
 * ========================================================================= */

/* --------------------------------------------------------------------------
 * c_cuda_load_kernel
 * R-callable entry point (.Call interface, 2 arguments).
 *
 * arguments:
 *   ptx_path_sexp    -- character scalar: path to the compiled PTX file
 *   kernel_name_sexp -- character scalar: name of the kernel function within
 *                       the PTX module (must match the __global__ function
 *                       name in the .cu source exactly)
 *
 * returns:
 *   an R external pointer of class "cudaDynamicKernel" wrapping a heap-
 *   allocated CudaDynamicKernel struct.  the struct holds the CUmodule and
 *   the CUfunction; a registered finalizer unloads the module when the
 *   pointer is garbage collected.
 * -------------------------------------------------------------------------- */
SEXP c_cuda_load_kernel(SEXP ptx_path_sexp,
                        SEXP kernel_name_sexp) {
  /* ---- 1. validate R arguments ----------------------------------------- */
  
  if (TYPEOF(ptx_path_sexp) != STRSXP || LENGTH(ptx_path_sexp) != 1) {
    Rf_error("'ptx_path' must be a character scalar");
  }
  if (TYPEOF(kernel_name_sexp) != STRSXP || LENGTH(kernel_name_sexp) != 1) {
    Rf_error("'kernel_name' must be a character scalar");
  }
  
  const char *ptx_path    = CHAR(STRING_ELT(ptx_path_sexp,    0));
  const char *kernel_name = CHAR(STRING_ELT(kernel_name_sexp, 0));
  
  if (strlen(ptx_path) == 0) {
    Rf_error("'ptx_path' must not be an empty string");
  }
  if (strlen(kernel_name) == 0) {
    Rf_error("'kernel_name' must not be an empty string");
  }
  
  /* ---- 2. initialise the Driver API ------------------------------------ */
  
  if (cuda_driver_init() != 0) {
    Rf_error("cuInit() failed -- driver may not be installed or accessible");
  }
  
  /* ---- 3. retrieve the current CUDA context ----------------------------
   * the Runtime API creates an implicit context on first use; we retrieve
   * it here so the Driver API shares the same context rather than creating
   * a second one.  if no Runtime API call has been made yet there may be
   * no current context, in which case we error with a clear message. */
  
  CUcontext ctx = NULL;
  CUresult  res = cuCtxGetCurrent(&ctx);
  
  if (res != CUDA_SUCCESS) {
    char err_str[256];
    cuda_driver_error_string(res, err_str, sizeof(err_str));
    Rf_error("cuCtxGetCurrent() failed: %s", err_str);
  }
  
  if (ctx == NULL) {
    Rf_error("no current CUDA context found; "
               "call cuda_make_context() before cuda_load_kernel()");
  }
  
  /* ---- 4. read PTX file into host buffer ------------------------------- */
  
  char  read_err[512];
  char *ptx_buf = read_file_to_buffer(ptx_path, read_err, sizeof(read_err));
  
  if (ptx_buf == NULL) {
    Rf_error("%s", read_err);
  }
  
  /* ---- 5. load PTX buffer into a CUmodule ------------------------------ */
  
  CUmodule module = NULL;
  res = cuModuleLoadData(&module, ptx_buf);
  free(ptx_buf);  /* driver has consumed the buffer; safe to free now */
  
  if (res != CUDA_SUCCESS) {
    char err_str[256];
    cuda_driver_error_string(res, err_str, sizeof(err_str));
    Rf_error("cuModuleLoadData() failed: %s -- "
               "check that the PTX was compiled for the correct architecture",
               err_str);
  }
  
  /* ---- 6. retrieve CUfunction handle by kernel name -------------------- */
  
  CUfunction function = NULL;
  res = cuModuleGetFunction(&function, module, kernel_name);
  
  if (res != CUDA_SUCCESS) {
    char err_str[256];
    cuda_driver_error_string(res, err_str, sizeof(err_str));
    /* unload the module before erroring -- we own it and have no other
     * path to release it if we do not wrap it in a finalizer first */
    cuModuleUnload(module);
    Rf_error("cuModuleGetFunction() failed for kernel '%s': %s -- "
               "check that the kernel name matches the __global__ declaration "
               "in the .cu source exactly",
               kernel_name, err_str);
  }
  
  /* ---- 7. allocate and populate CudaDynamicKernel struct --------------- */
  
  CudaDynamicKernel *dk = (CudaDynamicKernel *)malloc(sizeof(CudaDynamicKernel));
  if (dk == NULL) {
    cuModuleUnload(module);
    Rf_error("failed to allocate CudaDynamicKernel struct");
  }
  
  dk->module   = module;
  dk->function = function;
  
  /* store the kernel name for use in runner error messages;
   * strncpy + explicit null terminator avoids truncation surprises */
  strncpy(dk->kernel_name, kernel_name, sizeof(dk->kernel_name) - 1);
  dk->kernel_name[sizeof(dk->kernel_name) - 1] = '\0';
  
  /* ---- 8. wrap in R external pointer with finalizer -------------------- */
  
  SEXP result = PROTECT(R_MakeExternalPtr(dk, R_NilValue, R_NilValue));
  R_RegisterCFinalizer(result, cuda_dynamic_kernel_finalizer);
  setAttrib(result, R_ClassSymbol, mkString("cudaDynamicKernel"));
  
  UNPROTECT(1);
  return result;
}


