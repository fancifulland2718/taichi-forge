#pragma once

#include <cstddef>
#include <cstdint>

#if defined(TI_WITH_CUDA_TOOLKIT_HEADERS)

#include <cublas_v2.h>
#include <cuda.h>
#include <cusolverSp.h>
#include <cusparse.h>

// CUDA 13 removed some legacy cuSOLVER low-level preview declarations that
// Taichi still keeps in the dynamic-loader signature list. Keep them opaque so
// toolkit builds do not depend on those removed preview headers.
struct csrcholInfoHost;
typedef struct csrcholInfoHost *csrcholInfoHost_t;
struct csrcholInfo;
typedef struct csrcholInfo *csrcholInfo_t;
struct csrluInfoHost;
typedef struct csrluInfoHost *csrluInfoHost_t;

#else

using CUexternalMemory = void *;
using CUexternalSemaphore = void *;
typedef struct CUuuid_st {
  char bytes[16];
} CUuuid;
using CUsurfObject = uint64_t;
using CUtexObject = uint64_t;
using CUstream = void *;
using CUgraph = void *;
using CUgraphExec = void *;
using CUdeviceptr = void *;
using CUmipmappedArray = void *;
using CUarray = void *;

// copied from <cuda.h>

/**
 * Resource types
 */
typedef enum CUresourcetype_enum {
  CU_RESOURCE_TYPE_ARRAY = 0x00,           /**< Array resource */
  CU_RESOURCE_TYPE_MIPMAPPED_ARRAY = 0x01, /**< Mipmapped array resource */
  CU_RESOURCE_TYPE_LINEAR = 0x02,          /**< Linear resource */
  CU_RESOURCE_TYPE_PITCH2D = 0x03          /**< Pitch 2D resource */
} CUresourcetype;

/**
 * Array formats
 */
typedef enum CUarray_format_enum {
  CU_AD_FORMAT_UNSIGNED_INT8 = 0x01,  /**< Unsigned 8-bit integers */
  CU_AD_FORMAT_UNSIGNED_INT16 = 0x02, /**< Unsigned 16-bit integers */
  CU_AD_FORMAT_UNSIGNED_INT32 = 0x03, /**< Unsigned 32-bit integers */
  CU_AD_FORMAT_SIGNED_INT8 = 0x08,    /**< Signed 8-bit integers */
  CU_AD_FORMAT_SIGNED_INT16 = 0x09,   /**< Signed 16-bit integers */
  CU_AD_FORMAT_SIGNED_INT32 = 0x0a,   /**< Signed 32-bit integers */
  CU_AD_FORMAT_HALF = 0x10,           /**< 16-bit floating point */
  CU_AD_FORMAT_FLOAT = 0x20           /**< 32-bit floating point */
} CUarray_format;

typedef enum CUmemorytype_enum {
  CU_MEMORYTYPE_HOST = 0x01,
  CU_MEMORYTYPE_DEVICE = 0x02,
  CU_MEMORYTYPE_ARRAY = 0x03,
  CU_MEMORYTYPE_UNIFIED = 0x04
} CUmemorytype;

typedef enum CUaddress_mode_enum {
  CU_TR_ADDRESS_MODE_WRAP = 0,
  CU_TR_ADDRESS_MODE_CLAMP = 1,
  CU_TR_ADDRESS_MODE_MIRROR = 2,
  CU_TR_ADDRESS_MODE_BORDER = 3
} CUaddress_mode;

typedef enum CUfilter_mode_enum {
  CU_TR_FILTER_MODE_POINT = 0,
  CU_TR_FILTER_MODE_LINEAR = 1
} CUfilter_mode;

typedef enum CUfunction_attribute_enum {
  CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0,
  CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1,
  CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2,
  CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3,
  CU_FUNC_ATTRIBUTE_NUM_REGS = 4,
  CU_FUNC_ATTRIBUTE_PTX_VERSION = 5,
  CU_FUNC_ATTRIBUTE_BINARY_VERSION = 6,
  CU_FUNC_ATTRIBUTE_CACHE_MODE_CA = 7,
  CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8,
  CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9,
  CU_FUNC_ATTRIBUTE_MAX
} CUfunction_attribute;

typedef enum CUstreamCaptureMode_enum {
  CU_STREAM_CAPTURE_MODE_GLOBAL = 0,
  CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = 1,
  CU_STREAM_CAPTURE_MODE_RELAXED = 2
} CUstreamCaptureMode;

/**
 * 3D array descriptor
 */
typedef struct CUDA_ARRAY3D_DESCRIPTOR_st {
  size_t Width;  /**< Width of 3D array */
  size_t Height; /**< Height of 3D array */
  size_t Depth;  /**< Depth of 3D array */

  CUarray_format Format;    /**< Array format */
  unsigned int NumChannels; /**< Channels per array element */
  unsigned int Flags;       /**< Flags */
} CUDA_ARRAY3D_DESCRIPTOR;

typedef struct CUDA_MEMCPY3D_st {
  size_t srcXInBytes;
  size_t srcY;
  size_t srcZ;
  size_t srcLOD;
  CUmemorytype srcMemoryType;
  const void *srcHost;
  CUdeviceptr srcDevice;
  CUarray srcArray;
  void *reserved0;
  size_t srcPitch;
  size_t srcHeight;

  size_t dstXInBytes;
  size_t dstY;
  size_t dstZ;
  size_t dstLOD;
  CUmemorytype dstMemoryType;
  void *dstHost;
  CUdeviceptr dstDevice;
  CUarray dstArray;
  void *reserved1;
  size_t dstPitch;
  size_t dstHeight;

  size_t WidthInBytes;
  size_t Height;
  size_t Depth;
} CUDA_MEMCPY3D;

/**
 * CUDA Resource descriptor
 */
typedef struct CUDA_RESOURCE_DESC_st {
  CUresourcetype resType; /**< Resource type */

  union {
    struct {
      CUarray hArray; /**< CUDA array */
    } array;
    struct {
      CUmipmappedArray hMipmappedArray; /**< CUDA mipmapped array */
    } mipmap;
    struct {
      CUdeviceptr devPtr;       /**< Device pointer */
      CUarray_format format;    /**< Array format */
      unsigned int numChannels; /**< Channels per array element */
      size_t sizeInBytes;       /**< Size in bytes */
    } linear;
    struct {
      CUdeviceptr devPtr;       /**< Device pointer */
      CUarray_format format;    /**< Array format */
      unsigned int numChannels; /**< Channels per array element */
      size_t width;             /**< Width of the array in elements */
      size_t height;            /**< Height of the array in elements */
      size_t pitchInBytes;      /**< Pitch between two rows in bytes */
    } pitch2D;
    struct {
      int reserved[32];
    } reserved;
  } res;

  unsigned int flags; /**< Flags (must be zero) */
} CUDA_RESOURCE_DESC;

typedef struct CUDA_TEXTURE_DESC_st {
  CUaddress_mode addressMode[3];
  CUfilter_mode filterMode;
  unsigned int flags;
  unsigned int maxAnisotropy;
  CUfilter_mode mipmapFilterMode;
  float mipmapLevelBias;
  float minMipmapLevelClamp;
  float maxMipmapLevelClamp;
  float borderColor[4];
  int reserved[12];
} CUDA_TEXTURE_DESC;

typedef enum CUexternalMemoryHandleType_enum {
  /**
   * Handle is an opaque file descriptor
   */
  CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD = 1,
  /**
   * Handle is an opaque shared NT handle
   */
  CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32 = 2,
  /**
   * Handle is an opaque, globally shared handle
   */
  CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT = 3,
  /**
   * Handle is a D3D12 heap object
   */
  CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP = 4,
  /**
   * Handle is a D3D12 committed resource
   */
  CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE = 5,
  /**
   * Handle is a shared NT handle to a D3D11 resource
   */
  CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE = 6,
  /**
   * Handle is a globally shared handle to a D3D11 resource
   */
  CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT = 7,
  /**
   * Handle is an NvSciBuf object
   */
  CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF = 8
} CUexternalMemoryHandleType;

typedef struct CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st {
  /**
   * Type of the handle
   */
  CUexternalMemoryHandleType type;
  union {
    /**
     * File descriptor referencing the memory object. Valid
     * when type is
     * ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD
     */
    int fd;
    /**
     * Win32 handle referencing the semaphore object. Valid when
     * type is one of the following:
     * - ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32
     * - ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT
     * - ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP
     * - ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE
     * - ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE
     * - ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT
     * Exactly one of 'handle' and 'name' must be non-NULL. If
     * type is one of the following:
     * ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT
     * ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT
     * then 'name' must be NULL.
     */
    struct {
      /**
       * Valid NT handle. Must be NULL if 'name' is non-NULL
       */
      void *handle;
      /**
       * Name of a valid memory object.
       * Must be NULL if 'handle' is non-NULL.
       */
      const void *name;
    } win32;
    /**
     * A handle representing an NvSciBuf Object. Valid when type
     * is ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF
     */
    const void *nvSciBufObject;
  } handle;
  /**
   * Size of the memory allocation
   */
  unsigned long long size;
  /**
   * Flags must either be zero or ::CUDA_EXTERNAL_MEMORY_DEDICATED
   */
  unsigned int flags;
  unsigned int reserved[16];
} CUDA_EXTERNAL_MEMORY_HANDLE_DESC;

typedef struct CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st {
  /**
   * Offset into the memory object where the buffer's base is
   */
  unsigned long long offset;
  /**
   * Size of the buffer
   */
  unsigned long long size;
  /**
   * Flags reserved for future use. Must be zero.
   */
  unsigned int flags;
  unsigned int reserved[16];
} CUDA_EXTERNAL_MEMORY_BUFFER_DESC;

/**
 * External memory mipmap descriptor
 */
typedef struct CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st {
  /**
   * Offset into the memory object where the base level of the
   * mipmap chain is.
   */
  unsigned long long offset;
  /**
   * Format, dimension and type of base level of the mipmap chain
   */
  CUDA_ARRAY3D_DESCRIPTOR arrayDesc;
  /**
   * Total number of levels in the mipmap chain
   */
  unsigned int numLevels;
  unsigned int reserved[16];
} CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC;

/**
 * External semaphore handle types
 */
typedef enum CUexternalSemaphoreHandleType_enum {
  /**
   * Handle is an opaque file descriptor
   */
  CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD = 1,
  /**
   * Handle is an opaque shared NT handle
   */
  CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32 = 2,
  /**
   * Handle is an opaque, globally shared handle
   */
  CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT = 3,
  /**
   * Handle is a shared NT handle referencing a D3D12 fence object
   */
  CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE = 4,
  /**
   * Handle is a shared NT handle referencing a D3D11 fence object
   */
  CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE = 5,
  /**
   * Opaque handle to NvSciSync Object
   */
  CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC = 6,
  /**
   * Handle is a shared NT handle referencing a D3D11 keyed mutex object
   */
  CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX = 7,
  /**
   * Handle is a globally shared handle referencing a D3D11 keyed mutex object
   */
  CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX_KMT = 8
} CUexternalSemaphoreHandleType;

/**
 * External semaphore handle descriptor
 */
typedef struct CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st {
  /**
   * Type of the handle
   */
  CUexternalSemaphoreHandleType type;
  union {
    /**
     * File descriptor referencing the semaphore object. Valid
     * when type is
     * ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD
     */
    int fd;
    /**
     * Win32 handle referencing the semaphore object. Valid when
     * type is one of the following:
     * - ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32
     * - ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT
     * - ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE
     * - ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE
     * - ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX
     * Exactly one of 'handle' and 'name' must be non-NULL. If
     * type is one of the following:
     * ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT
     * ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX_KMT
     * then 'name' must be NULL.
     */
    struct {
      /**
       * Valid NT handle. Must be NULL if 'name' is non-NULL
       */
      void *handle;
      /**
       * Name of a valid synchronization primitive.
       * Must be NULL if 'handle' is non-NULL.
       */
      const void *name;
    } win32;
    /**
     * Valid NvSciSyncObj. Must be non NULL
     */
    const void *nvSciSyncObj;
  } handle;
  /**
   * Flags reserved for the future. Must be zero.
   */
  unsigned int flags;
  unsigned int reserved[16];
} CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC;

/**
 * External semaphore signal parameters
 */
typedef struct CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st {
  struct {
    /**
     * Parameters for fence objects
     */
    struct {
      /**
       * Value of fence to be signaled
       */
      unsigned long long value;
    } fence;
    union {
      /**
       * Pointer to NvSciSyncFence. Valid if ::CUexternalSemaphoreHandleType
       * is of type ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC.
       */
      void *fence;
      unsigned long long reserved;
    } nvSciSync;
    /**
     * Parameters for keyed mutex objects
     */
    struct {
      /**
       * Value of key to release the mutex with
       */
      unsigned long long key;
    } keyedMutex;
    unsigned int reserved[12];
  } params;
  /**
   * Only when ::CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS is used to
   * signal a ::CUexternalSemaphore of type
   * ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC, the valid flag is
   * ::CUDA_EXTERNAL_SEMAPHORE_SIGNAL_SKIP_NVSCIBUF_MEMSYNC which indicates
   * that while signaling the ::CUexternalSemaphore, no memory synchronization
   * operations should be performed for any external memory object imported
   * as ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF.
   * For all other types of ::CUexternalSemaphore, flags must be zero.
   */
  unsigned int flags;
  unsigned int reserved[16];
} CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS;

/**
 * External semaphore wait parameters
 */
typedef struct CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st {
  struct {
    /**
     * Parameters for fence objects
     */
    struct {
      /**
       * Value of fence to be waited on
       */
      unsigned long long value;
    } fence;
    /**
     * Pointer to NvSciSyncFence. Valid if CUexternalSemaphoreHandleType
     * is of type CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC.
     */
    union {
      void *fence;
      unsigned long long reserved;
    } nvSciSync;
    /**
     * Parameters for keyed mutex objects
     */
    struct {
      /**
       * Value of key to acquire the mutex with
       */
      unsigned long long key;
      /**
       * Timeout in milliseconds to wait to acquire the mutex
       */
      unsigned int timeoutMs;
    } keyedMutex;
    unsigned int reserved[10];
  } params;
  /**
   * Only when ::CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS is used to wait on
   * a ::CUexternalSemaphore of type
   * ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC, the valid flag is
   * ::CUDA_EXTERNAL_SEMAPHORE_WAIT_SKIP_NVSCIBUF_MEMSYNC which indicates that
   * while waiting for the ::CUexternalSemaphore, no memory synchronization
   * operations should be performed for any external memory object imported as
   * ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF. For all other types of
   * ::CUexternalSemaphore, flags must be zero.
   */
  unsigned int flags;
  unsigned int reserved[16];
} CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS;

/**
 * Indicates that the external memory object is a dedicated resource
 */
#define CUDA_EXTERNAL_MEMORY_DEDICATED 0x1

/**
 * This flag must be set in order to bind a surface reference
 * to the CUDA array
 */
#define CUDA_ARRAY3D_SURFACE_LDST 0x02

/**
 * This flag indicates that the CUDA array may be bound as a color target
 * in an external graphics API
 */
#define CUDA_ARRAY3D_COLOR_ATTACHMENT 0x20

// copy from cusparse.h
struct cusparseContext;
typedef struct cusparseContext *cusparseHandle_t;

struct cusparseMatDescr;
typedef struct cusparseMatDescr *cusparseMatDescr_t;

struct cusparseSpVecDescr;
struct cusparseDnVecDescr;
struct cusparseSpMatDescr;
typedef struct cusparseSpVecDescr *cusparseSpVecDescr_t;
typedef struct cusparseDnVecDescr *cusparseDnVecDescr_t;
typedef struct cusparseSpMatDescr *cusparseSpMatDescr_t;

struct cusparseSpGEMMDescr;
typedef struct cusparseSpGEMMDescr *cusparseSpGEMMDescr_t;

typedef enum {
  CUSPARSE_INDEX_16U = 1,  ///< 16-bit unsigned integer for matrix/vector
                           ///< indices
  CUSPARSE_INDEX_32I = 2,  ///< 32-bit signed integer for matrix/vector indices
  CUSPARSE_INDEX_64I = 3   ///< 64-bit signed integer for matrix/vector indices
} cusparseIndexType_t;

typedef enum { CUSPARSE_ORDER_COL = 1, CUSPARSE_ORDER_ROW = 2 } cusparseOrder_t;

typedef enum {
  CUSPARSE_INDEX_BASE_ZERO = 0,
  CUSPARSE_INDEX_BASE_ONE = 1
} cusparseIndexBase_t;

typedef enum cudaDataType_t {
  CUDA_R_16F = 2,   /* real as a half */
  CUDA_C_16F = 6,   /* complex as a pair of half numbers */
  CUDA_R_16BF = 14, /* real as a nv_bfloat16 */
  CUDA_C_16BF = 15, /* complex as a pair of nv_bfloat16 numbers */
  CUDA_R_32F = 0,   /* real as a float */
  CUDA_C_32F = 4,   /* complex as a pair of float numbers */
  CUDA_R_64F = 1,   /* real as a double */
  CUDA_C_64F = 5,   /* complex as a pair of double numbers */
  CUDA_R_4I = 16,   /* real as a signed 4-bit int */
  CUDA_C_4I = 17,   /* complex as a pair of signed 4-bit int numbers */
  CUDA_R_4U = 18,   /* real as a unsigned 4-bit int */
  CUDA_C_4U = 19,   /* complex as a pair of unsigned 4-bit int numbers */
  CUDA_R_8I = 3,    /* real as a signed 8-bit int */
  CUDA_C_8I = 7,    /* complex as a pair of signed 8-bit int numbers */
  CUDA_R_8U = 8,    /* real as a unsigned 8-bit int */
  CUDA_C_8U = 9,    /* complex as a pair of unsigned 8-bit int numbers */
  CUDA_R_16I = 20,  /* real as a signed 16-bit int */
  CUDA_C_16I = 21,  /* complex as a pair of signed 16-bit int numbers */
  CUDA_R_16U = 22,  /* real as a unsigned 16-bit int */
  CUDA_C_16U = 23,  /* complex as a pair of unsigned 16-bit int numbers */
  CUDA_R_32I = 10,  /* real as a signed 32-bit int */
  CUDA_C_32I = 11,  /* complex as a pair of signed 32-bit int numbers */
  CUDA_R_32U = 12,  /* real as a unsigned 32-bit int */
  CUDA_C_32U = 13,  /* complex as a pair of unsigned 32-bit int numbers */
  CUDA_R_64I = 24,  /* real as a signed 64-bit int */
  CUDA_C_64I = 25,  /* complex as a pair of signed 64-bit int numbers */
  CUDA_R_64U = 26,  /* real as a unsigned 64-bit int */
  CUDA_C_64U = 27   /* complex as a pair of unsigned 64-bit int numbers */
} cudaDataType;

typedef enum {
  CUSPARSE_OPERATION_NON_TRANSPOSE = 0,
  CUSPARSE_OPERATION_TRANSPOSE = 1,
  CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE = 2
} cusparseOperation_t;

typedef enum {
  CUSPARSE_SPMV_ALG_DEFAULT = 0,
  CUSPARSE_SPMV_COO_ALG1 = 1,
  CUSPARSE_SPMV_CSR_ALG1 = 2,
  CUSPARSE_SPMV_CSR_ALG2 = 3,
  CUSPARSE_SPMV_COO_ALG2 = 4,
  CUSPARSE_SPMV_BSR_ALG1 = 6
} cusparseSpMVAlg_t;

typedef enum {
  CUSPARSE_SPGEMM_DEFAULT = 0,
  CUSPARSE_SPGEMM_CSR_ALG_DETERMINITIC = 1,
  CUSPARSE_SPGEMM_CSR_ALG_NONDETERMINITIC = 2
} cusparseSpGEMMAlg_t;

typedef enum {
  CUSPARSE_POINTER_MODE_HOST = 0,
  CUSPARSE_POINTER_MODE_DEVICE = 1
} cusparsePointerMode_t;

typedef enum {
  CUSPARSE_ACTION_SYMBOLIC = 0,
  CUSPARSE_ACTION_NUMERIC = 1
} cusparseAction_t;

typedef enum {
  CUSPARSE_CSR2CSC_ALG1 = 1,  // faster than V2 (in general), deterministc
  CUSPARSE_CSR2CSC_ALG2 = 2   // low memory requirement, non-deterministc
} cusparseCsr2CscAlg_t;

typedef enum {
  CUSPARSE_MATRIX_TYPE_GENERAL = 0,
  CUSPARSE_MATRIX_TYPE_SYMMETRIC = 1,
  CUSPARSE_MATRIX_TYPE_HERMITIAN = 2,
  CUSPARSE_MATRIX_TYPE_TRIANGULAR = 3
} cusparseMatrixType_t;

typedef enum {
  CUSPARSE_FILL_MODE_LOWER = 0,
  CUSPARSE_FILL_MODE_UPPER = 1
} cusparseFillMode_t;

typedef enum {
  CUSPARSE_DIAG_TYPE_NON_UNIT = 0,
  CUSPARSE_DIAG_TYPE_UNIT = 1
} cusparseDiagType_t;

// copy from cusolver.h
typedef enum libraryPropertyType_t {
  MAJOR_VERSION,
  MINOR_VERSION,
  PATCH_LEVEL
} libraryPropertyType;

struct cusolverSpContext;
typedef struct cusolverSpContext *cusolverSpHandle_t;

// copy from cusolverSp_LOWLEVEL_PREVIEW.h
struct csrcholInfoHost;
typedef struct csrcholInfoHost *csrcholInfoHost_t;
struct csrcholInfo;
typedef struct csrcholInfo *csrcholInfo_t;
struct csrluInfoHost;
typedef struct csrluInfoHost *csrluInfoHost_t;

// copy from cublas_api.h
/* CUBLAS status type returns */
typedef enum {
  CUBLAS_STATUS_SUCCESS = 0,
  CUBLAS_STATUS_NOT_INITIALIZED = 1,
  CUBLAS_STATUS_ALLOC_FAILED = 3,
  CUBLAS_STATUS_INVALID_VALUE = 7,
  CUBLAS_STATUS_ARCH_MISMATCH = 8,
  CUBLAS_STATUS_MAPPING_ERROR = 11,
  CUBLAS_STATUS_EXECUTION_FAILED = 13,
  CUBLAS_STATUS_INTERNAL_ERROR = 14,
  CUBLAS_STATUS_NOT_SUPPORTED = 15,
  CUBLAS_STATUS_LICENSE_ERROR = 16
} cublasStatus_t;

typedef enum {
  CUBLAS_POINTER_MODE_HOST = 0,
  CUBLAS_POINTER_MODE_DEVICE = 1
} cublasPointerMode_t;

typedef enum {
  CUBLAS_OP_N = 0,
  CUBLAS_OP_T = 1,
  CUBLAS_OP_C = 2,
  CUBLAS_OP_HERMITAN = 2,
  CUBLAS_OP_CONJG = 3
} cublasOperation_t;

/* Opaque structure holding CUBLAS library context */
struct cublasContext;
typedef struct cublasContext *cublasHandle_t;

#endif

// Stable ABI shims for CUDA conditional graph nodes. They intentionally use
// only fixed-width and opaque types so driver-only builds do not require a
// CUDA toolkit header. CUDA 12.8 defines CUgraphNodeParams as 256 bytes with
// a 232-byte parameter union.
struct TaichiCudaConditionalNodeParams {
  std::uint64_t handle;
  std::uint32_t type;
  std::uint32_t size;
  CUgraph *ph_graph_out;
  void *context;
};

struct TaichiCudaGraphNodeParams {
  std::uint32_t type;
  std::int32_t reserved0[3];
  union {
    std::int64_t reserved1[29];
    TaichiCudaConditionalNodeParams conditional;
  } parameters;
  std::int64_t reserved2;
};

static_assert(sizeof(TaichiCudaConditionalNodeParams) == 32);
static_assert(offsetof(TaichiCudaConditionalNodeParams, handle) == 0);
static_assert(offsetof(TaichiCudaConditionalNodeParams, type) == 8);
static_assert(offsetof(TaichiCudaConditionalNodeParams, size) == 12);
static_assert(offsetof(TaichiCudaConditionalNodeParams, ph_graph_out) == 16);
static_assert(offsetof(TaichiCudaConditionalNodeParams, context) == 24);
static_assert(sizeof(TaichiCudaGraphNodeParams) == 256);
static_assert(offsetof(TaichiCudaGraphNodeParams, parameters) == 16);
static_assert(offsetof(TaichiCudaGraphNodeParams, reserved2) == 248);

// Stable CUDA 12.4 extensible-launch ABI shims. Keeping these fixed-width and
// opaque lets the ordinary runtime discover cuLaunchKernelEx dynamically
// without taking a CUDA Toolkit dependency. CUDA 12.4 and later define the
// launch attribute value as a 64-byte union and CUlaunchConfig as 56 bytes on
// the supported 64-bit platforms.
struct alignas(8) TaichiCudaDeviceUpdatableKernelNode {
  std::int32_t device_updatable;
  std::int32_t reserved;
  void *device_node;
};

struct alignas(8) TaichiCudaLaunchAttributeValue {
  union {
    std::uint8_t pad[64];
    TaichiCudaDeviceUpdatableKernelNode device_updatable_kernel_node;
  };
};

struct alignas(8) TaichiCudaLaunchAttribute {
  std::int32_t id;
  std::int32_t reserved;
  TaichiCudaLaunchAttributeValue value;
};

struct alignas(8) TaichiCudaLaunchConfig {
  std::uint32_t grid_dim_x;
  std::uint32_t grid_dim_y;
  std::uint32_t grid_dim_z;
  std::uint32_t block_dim_x;
  std::uint32_t block_dim_y;
  std::uint32_t block_dim_z;
  std::uint32_t shared_mem_bytes;
  std::uint32_t reserved;
  void *stream;
  TaichiCudaLaunchAttribute *attributes;
  std::uint32_t num_attributes;
  std::uint32_t reserved2;
};

constexpr std::int32_t TAICHI_CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE =
    13;

static_assert(sizeof(TaichiCudaLaunchAttributeValue) == 64);
static_assert(sizeof(TaichiCudaDeviceUpdatableKernelNode) == 16);
static_assert(offsetof(TaichiCudaDeviceUpdatableKernelNode, device_updatable) ==
              0);
static_assert(offsetof(TaichiCudaDeviceUpdatableKernelNode, device_node) == 8);
static_assert(sizeof(TaichiCudaLaunchAttribute) == 72);
static_assert(offsetof(TaichiCudaLaunchAttribute, value) == 8);
static_assert(sizeof(TaichiCudaLaunchConfig) == 56);
static_assert(offsetof(TaichiCudaLaunchConfig, stream) == 32);
static_assert(offsetof(TaichiCudaLaunchConfig, attributes) == 40);
static_assert(offsetof(TaichiCudaLaunchConfig, num_attributes) == 48);
