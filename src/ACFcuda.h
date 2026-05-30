/* ============================================================================
 * ACFcuda.h
 *
 * Public interface for ACFcuda C functions.
 * author: nicholas cooley
 * maintainer: nicholas cooley
 *
 * naming convention (inherited from ACFmetal):
 *    no prefix = function has no user facing R analog
 *    c_ prefix = function has a user facing R analog
 *    cuda_prefix = no distinct use case yet
 * ========================================================================= */

#ifndef ACFCUDA_H
#define ACFCUDA_H

#include <Rinternals.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <R_ext/Visibility.h>
#include <stdint.h>
#include <stddef.h>

/* ============================================================================
 * types, constants, structs, and their helpers
 * ========================================================================= */

// cuda kernel type!
#define CUDA_KERNEL_NAME_MAX 256
typedef struct {
  CUmodule   module;
  CUfunction function;
  char       kernel_name[CUDA_KERNEL_NAME_MAX];
} CudaKernel;

// typedefs to standardize handoffs between R data types and CUDA data types 
typedef enum {
  CUDA_TYPE_FLOAT = 1, // "float" = 32-bit float 
  CUDA_TYPE_DOUBLE = 2, // "double" = 64-bit float
  CUDA_TYPE_INT8 = 3, // "char" = 8-bit signed integer
  CUDA_TYPE_INT16 = 4, // "short" = 16-bit signed integer
  CUDA_TYPE_INT = 5, // "int" = 32-bit signed integer
  CUDA_TYPE_INT64 = 6, // "long" = 64-bit signed integer
  CUDA_TYPE_UINT8 = 7, // "uchar" = 8-bit unsigned integer
  CUDA_TYPE_UINT16 = 8, // "ushort" = 16-bit unsigned integer
  CUDA_TYPE_UINT = 9, // "uint" = 32-bit unsigned integer
  CUDA_TYPE_UINT64 = 10 // "ulong" = 64-bit unsigned integer
} CudaType;

// struct for managing data and device identification and interface to R
// the stream pointer needs a teardown call to cudaStreamDestroy
// this mirrors the setup for ACFmetal, but may ultimately need to be different
// or may be unnecessary?
typedef struct {
  int device_index; /* integer index passed to cudaSetDevice()        */
  void *stream;       /* cudaStream_t cast to void*, or NULL for default */
} CudaContext;

/* CudaLauncherFn is the function pointer type that all registered kernel
 * launcher shims must match.  Each .cu file that provides a user kernel
 * also provides one launcher function of this type and registers it via
 * cuda_register_kernel().
 *
 * Parameters:
 *   device_buffers  - void* array; index 0 is the output buffer,
 *                     indices 1..n are input buffers
 *   scalar_values   - void* array of host-side scalar values
 *   scalar_sizes    - byte width of each scalar
 *   is_scalar       - 1 if the i-th non-output argument is a scalar, else 0
 *   num_args        - total number of non-output arguments
 *   grid_x/y/z      - CUDA grid dimensions
 *   block_x/y/z     - CUDA block (thread) dimensions
 *   stream          - cudaStream_t as void*, NULL for default stream      */
typedef void (*CudaLauncherFn)(void **device_buffers,
                               void **scalar_values,
                               size_t *scalar_sizes,
                               int *is_scalar,
                               int num_args,
                               int grid_x,
                               int grid_y,
                               int grid_z,
                               int block_x,
                               int block_y,
                               int block_z,
                               void *stream);

/* ============================================================================
 * buffers.c
 * host-side type conversion and device memory helpers
 * ========================================================================= */

void cuda_convert_r_numeric_to_host(const double *r_data,
                                    void *host_buf,
                                    size_t length,
                                    CudaType type);

void cuda_convert_r_int_to_host(const int *r_data,
                                void *host_buf,
                                size_t length,
                                CudaType type);

void cuda_convert_host_to_r(const void *host_buf,
                            double *r_data,
                            size_t length,
                            CudaType type);

void *cuda_device_alloc(size_t byte_count);
void cuda_device_free(void *device_ptr);

void cuda_memcpy_to_device(void *device_dst,
                           const void *host_src,
                           size_t byte_count);

void cuda_memcpy_to_host(void *host_dst,
                         const void *device_src,
                         size_t byte_count);

/* ============================================================================
 * command.c
 * context creation, stream management, and synchronization
 * ========================================================================= */

attribute_visible SEXP c_cuda_make_context(SEXP device_index_sexp,
                                           SEXP use_default_stream_sexp);

void cuda_synchronize(CudaContext *ctx);
void *cuda_stream_create(void);
void cuda_stream_destroy(void *stream);
void cuda_stream_synchronize(void *stream);
void cuda_device_synchronize(void);

// wrappers for device interaction ... may drop these eventually
void cuda_set_device(int device_index);
int cuda_get_device_count(void);

/* ============================================================================
 * devices.c
 * device enumeration and property queries
 * ========================================================================= */

attribute_visible SEXP c_cuda_device_hook(void);
attribute_visible SEXP c_cuda_device_count(void);
attribute_visible SEXP c_cuda_device_information(SEXP device_index_sexp);

/* ============================================================================
 * libraries_functions.c
 * ========================================================================= */
void cuda_kernel_finalizer(SEXP ptr);
attribute_visible SEXP c_cuda_kernel_from_ptx(SEXP cuda_context_sexp,
                                              SEXP ptx_path_sexp,
                                              SEXP kernel_name_sexp);


/* ============================================================================
 * runners.c
 * kernel registry and .External dispatch entry point
 * ========================================================================= */

void cuda_register_kernel(const char *name,
                          CudaLauncherFn launcher,
                          int max_threads_per_block);

attribute_visible SEXP cuda_simple_runner(SEXP args);

/* ============================================================================
 * utils.c
 * finalizers, type utilities, and general helpers
 * ========================================================================= */

CUresult cuda_driver_init(void);

void cuda_context_finalizer(SEXP ctx_exp);
void cuda_device_ptr_finalizer(SEXP ptr_exp);

CudaType cuda_parse_type(const char *type_str);
size_t cuda_get_element_size(CudaType type);
const char *cuda_type_name(CudaType type);

char *read_file_to_buffer(const char *path,
                          char *err_buf,
                          size_t err_buf_size);

/* end header guard */
#endif
