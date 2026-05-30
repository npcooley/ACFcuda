/* ============================================================================
 * utils.c
 * author: nicholas cooley
 * maintainer: nicholas cooley
 *
 * utility C functions: finalizers, type parsing, element size, read in
 *
 * DESIGN NOTE vs ACFmetal/utils.c and utils.m:
 *   ACFmetal needs both a utils.c (R finalizer registration) and a utils.m
 *   (ObjC CFRelease wrappers) because Metal objects live under ARC and must
 *   be released with CFRelease(), not free().
 *
 *   ACFcuda has no ObjC layer.  The only GPU resource that requires explicit
 *   teardown is cudaStream_t (stored in CudaContext.stream).  Device memory
 *   pointers (void* from cudaMalloc) are freed inside runners.c immediately
 *   after the result is copied back to host; they do not live long enough to
 *   need a finalizer.
 *
 *   Consequently utils.c here replaces both utils.c and utils.m from
 *   ACFmetal, and the per-type release functions
 *   (metal_release_buffer, metal_release_pipeline, metal_release_library,
 *    metal_release_function, metal_release_command_queue, ...) are absent
 *   because there are no corresponding persistent GPU objects to release.
 * ========================================================================= */

#include <Rinternals.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <string.h>
#include <stdlib.h>
#include "ACFcuda.h"

/* ============================================================================
 * SECTION: general utilities
 * ========================================================================= */

// driver handshake helper
// int is fine here for success failure
CUresult cuda_driver_init(void) {
  CUresult res = cuInit(0);
  if (res != CUDA_SUCCESS) {
    return res;
  }
  return CUDA_SUCCESS;
}

/* ============================================================================
 * SECTION: finalizers
 * ========================================================================= */

/* --------------------------------------------------------------------------
 * cuda_context_finalizer
 * called by R's garbage collector when a cudaContext external pointer is
 * collected.  destroys the stream if non-NULL, then frees the struct.
 * -------------------------------------------------------------------------- */
void cuda_context_finalizer(SEXP ctx_exp) {
  CudaContext *ctx = (CudaContext *)R_ExternalPtrAddr(ctx_exp);
  if (ctx == NULL) {
    return;
  }
  if (ctx->stream != NULL) {
    cuda_stream_destroy(ctx->stream);
    ctx->stream = NULL;
  }
  free(ctx);
  R_ClearExternalPtr(ctx_exp);
}

void cuda_kernel_finalizer(SEXP ptr) {
  CudaKernel *dk = (CudaKernel *)R_ExternalPtrAddr(ptr);
  if (dk == NULL) {
    return;
  }
  if (dk->module != NULL) {
    cuModuleUnload(dk->module);
    dk->module   = NULL;
    dk->function = NULL;
  }
  free(dk);
  R_ClearExternalPtr(ptr);
}

/* --------------------------------------------------------------------------
 * cuda_device_ptr_finalizer
 * generic finalizer for any void* device pointer wrapped in an external ptr.
 * currently not used by runners.c (which frees device memory inline), but
 * provided for callers that wrap device allocations as persistent R objects.
 * -------------------------------------------------------------------------- */
void cuda_device_ptr_finalizer(SEXP ptr_exp) {
  void *ptr = R_ExternalPtrAddr(ptr_exp);
  if (ptr == NULL) {
    return;
  }
  cuda_device_free(ptr);
  R_ClearExternalPtr(ptr_exp);
}

/* ============================================================================
 * SECTION: type parsing and element size
 * mirrors ACFmetal/utils.c -- same type string vocabulary, same logic
 * ========================================================================= */

CudaType cuda_parse_type(const char *type_str) {
  if (type_str == NULL) {
    Rf_warning("NULL type string, defaulting to float");
    return CUDA_TYPE_FLOAT;
  }
  if (strcmp(type_str, "float")  == 0) return CUDA_TYPE_FLOAT;
  if (strcmp(type_str, "double") == 0) return CUDA_TYPE_DOUBLE;
  if (strcmp(type_str, "char")   == 0) return CUDA_TYPE_INT8;
  if (strcmp(type_str, "short")  == 0) return CUDA_TYPE_INT16;
  if (strcmp(type_str, "int")    == 0) return CUDA_TYPE_INT;
  if (strcmp(type_str, "long")   == 0) return CUDA_TYPE_INT64;
  if (strcmp(type_str, "uchar")  == 0) return CUDA_TYPE_UINT8;
  if (strcmp(type_str, "ushort") == 0) return CUDA_TYPE_UINT16;
  if (strcmp(type_str, "uint")   == 0) return CUDA_TYPE_UINT;
  if (strcmp(type_str, "ulong")  == 0) return CUDA_TYPE_UINT64;
  /* half is not implemented; warn and fall through to float */
  if (strcmp(type_str, "half")   == 0) {
    Rf_warning("type 'half' is not implemented, defaulting to float");
    return CUDA_TYPE_FLOAT;
  }
  Rf_warning("unknown type string '%s', defaulting to float", type_str);
  return CUDA_TYPE_FLOAT;
}

size_t cuda_get_element_size(CudaType type) {
  switch (type) {
  case CUDA_TYPE_FLOAT:  return sizeof(float);
  case CUDA_TYPE_DOUBLE: return sizeof(double);
  case CUDA_TYPE_INT8:
  case CUDA_TYPE_UINT8:  return 1;
  case CUDA_TYPE_INT16:
  case CUDA_TYPE_UINT16: return 2;
  case CUDA_TYPE_INT:
  case CUDA_TYPE_UINT:   return 4;
  case CUDA_TYPE_INT64:
  case CUDA_TYPE_UINT64: return 8;
  default:               return 0;
  }
}

const char *cuda_type_name(CudaType type) {
  switch (type) {
  case CUDA_TYPE_FLOAT:  return "float";
  case CUDA_TYPE_DOUBLE: return "double";
  case CUDA_TYPE_INT8:   return "int8";
  case CUDA_TYPE_INT16:  return "int16";
  case CUDA_TYPE_INT:    return "int32";
  case CUDA_TYPE_INT64:  return "int64";
  case CUDA_TYPE_UINT8:  return "uint8";
  case CUDA_TYPE_UINT16: return "uint16";
  case CUDA_TYPE_UINT:   return "uint32";
  case CUDA_TYPE_UINT64: return "uint64";
  default:               return "unknown";
  }
}

// read_file_to_buffer is our helper to read ptx files in as character strings
char *read_file_to_buffer(const char *path,
                          char *err_buf,
                          size_t err_buf_size) {
  FILE *fh = fopen(path, "rb");
  if (fh == NULL) {
    snprintf(err_buf, err_buf_size,
             "could not open PTX file for reading: %s", path);
    return NULL;
  }
  
  // get the file size
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
  
  // cuModuleLoadData expects a null terminator, allocate a buffer with space
  // for that terminator
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

