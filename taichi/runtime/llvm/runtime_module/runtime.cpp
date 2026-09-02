// This file will only be compiled into llvm bitcode by clang.
// The generated bitcode will likely get inlined for performance.

#if !defined(TI_INCLUDED) || !defined(_WIN32)
// The latest MSVC(Visual Studio 2019 version 16.10.1, MSVC 14.29.30037)
// uses llvm-11 as requirements. Check this link for details:
// https://github.com/microsoft/STL/blob/1866b848f0175c3361a916680a4318e7f0cc5482/stl/inc/yvals_core.h#L550-L561.
// However, we use llvm-10 for now and building will fail due to clang version
// mismatch. Therefore, we workaround this problem by define such flag to skip
// the version check.
// NOTE(#2428)
#if defined(_WIN32) || defined(_WIN64)
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <atomic>
#include <cstdint>
#include <cmath>
#include <cstdarg>
#include <cstdlib>
#include <algorithm>
#include <type_traits>
#include <cstring>

#include "taichi/inc/constants.h"
#include "taichi/inc/cuda_kernel_utils.inc.h"
#include "taichi/math/arithmetic.h"
#include "taichi/runtime/llvm/list_manager_constants.h"
#include "taichi/runtime/llvm/snode_runtime_state.h"
#include "taichi/runtime/llvm/sparse_tree_statistics.h"

struct RuntimeContext;
using assert_failed_type = void (*)(const char *);
using host_printf_type = void (*)(const char *, ...);
// In llvm 15, host_printf_type will be saved as ptr instead of ptr of
// FunctionType.
// Add dummy function to save function type for host_printf_type.
extern "C" void get_func_type_host_printf(const char *, ...) {
}

using host_vsnprintf_type = int (*)(char *,
                                    std::size_t,
                                    const char *,
                                    std::va_list);
using host_allocator_type = void *(*)(void *, std::size_t, std::size_t);
using host_releaser_type = void (*)(void *, std::size_t, void *);
using RangeForTaskFunc = void(RuntimeContext *, const char *tls, int i);
using MeshForTaskFunc = void(RuntimeContext *, const char *tls, uint32_t i);
using parallel_for_type = void (*)(void *thread_pool,
                                   int splits,
                                   int num_desired_threads,
                                   void *context,
                                   void (*func)(void *, int thread_id, int i));

#if defined(__linux__) && !ARCH_cuda && defined(TI_ARCH_x64)
__asm__(".symver logf,logf@GLIBC_2.2.5");
__asm__(".symver powf,powf@GLIBC_2.2.5");
__asm__(".symver expf,expf@GLIBC_2.2.5");
#endif

// For accessing struct fields
#define STRUCT_FIELD(S, F)                              \
  extern "C" decltype(S::F) S##_get_##F(S *s) {         \
    return s->F;                                        \
  }                                                     \
  extern "C" decltype(S::F) *S##_get_ptr_##F(S *s) {    \
    return &(s->F);                                     \
  }                                                     \
  extern "C" void S##_set_##F(S *s, decltype(S::F) f) { \
    s->F = f;                                           \
  }

#define STRUCT_FIELD_ARRAY(S, F)                                             \
  extern "C" std::remove_all_extents_t<decltype(S::F)> S##_get_##F(S *s,     \
                                                                   int i) {  \
    return s->F[i];                                                          \
  }                                                                          \
  extern "C" void S##_set_##F(S *s, int i,                                   \
                              std::remove_all_extents_t<decltype(S::F)> f) { \
    s->F[i] = f;                                                             \
  };

// For fetching struct fields from device to host
#define RUNTIME_STRUCT_FIELD(S, F)                                    \
  extern "C" void runtime_##S##_get_##F(LLVMRuntime *runtime, S *s) { \
    runtime->set_result(taichi_result_buffer_runtime_query_id, s->F); \
  }

#define RUNTIME_STRUCT_FIELD_ARRAY(S, F)                                     \
  extern "C" void runtime_##S##_get_##F(LLVMRuntime *runtime, S *s, int i) { \
    runtime->set_result(taichi_result_buffer_runtime_query_id, s->F[i]);     \
  }

using int8 = int8_t;
using int16 = int16_t;
using int32 = int32_t;
using int64 = int64_t;
using uint1 = bool;
using uint8 = uint8_t;
using uint16 = uint16_t;
using uint32 = uint32_t;
using uint64 = uint64_t;
using float32 = float;
using float64 = double;

using i8 = int8;
using i16 = int16;
using i32 = int32;
using i64 = int64;
using u1 = uint1;
using u8 = uint8;
using u16 = uint16;
using u32 = uint32;
using u64 = uint64;
using f32 = float32;
using f64 = float64;

using uint8 = uint8_t;
using Ptr = uint8 *;

using RuntimeContextArgType = long long;

#if ARCH_cuda || ARCH_amdgpu
extern "C" {

void __assertfail(const char *message,
                  const char *file,
                  i32 line,
                  const char *function,
                  std::size_t charSize);
};
#endif

template <typename T>
void locked_task(void *lock, const T &func);

template <typename T, typename G>
void locked_task(void *lock, const T &func, const G &test);

struct LLVMRuntime;
template <typename... Args>
void taichi_printf(LLVMRuntime *runtime, const char *format, Args &&...args);

extern "C" {

// This is not really a runtime function. Include this in a function body to
// mark it as force no inline. Helpful when preventing inlining huge function
// bodies.
void mark_force_no_inline() {
}

i64 cuda_clock_i64() {
  return 0;
}

void system_memfence() {
}

#if ARCH_cuda
void cuda_vprintf(Ptr format, Ptr arg);
#endif

// Note that strlen is undefined on the CUDA backend, so we manually
// implement it here.
std::size_t taichi_strlen(const char *str) {
  std::size_t len = 0;
  for (auto p = str; *p; p++)
    len++;
  return len;
}

#define DEFINE_UNARY_REAL_FUNC(F) \
  f32 F##_f32(f32 x) {            \
    return std::F(x);             \
  }                               \
  f64 F##_f64(f64 x) {            \
    return std::F(x);             \
  }

DEFINE_UNARY_REAL_FUNC(exp)
DEFINE_UNARY_REAL_FUNC(log)
DEFINE_UNARY_REAL_FUNC(tan)
DEFINE_UNARY_REAL_FUNC(tanh)
DEFINE_UNARY_REAL_FUNC(abs)
DEFINE_UNARY_REAL_FUNC(acos)
DEFINE_UNARY_REAL_FUNC(asin)
DEFINE_UNARY_REAL_FUNC(cos)
DEFINE_UNARY_REAL_FUNC(sin)

i32 abs_i32(i32 a) {
  return a >= 0 ? a : -a;
}

i64 abs_i64(i64 a) {
  return a >= 0 ? a : -a;
}

u16 min_u16(u16 a, u16 b) {
  return a < b ? a : b;
}

i16 min_i16(i16 a, i16 b) {
  return a < b ? a : b;
}

u32 min_u32(u32 a, u32 b) {
  return a < b ? a : b;
}

int min_i32(i32 a, i32 b) {
  return a < b ? a : b;
}

u64 min_u64(u64 a, u64 b) {
  return a < b ? a : b;
}

i64 min_i64(i64 a, i64 b) {
  return a < b ? a : b;
}

u16 max_u16(u16 a, u16 b) {
  return a > b ? a : b;
}

i16 max_i16(i16 a, i16 b) {
  return a > b ? a : b;
}

u32 max_u32(u32 a, u32 b) {
  return a > b ? a : b;
}

int max_i32(i32 a, i32 b) {
  return a > b ? a : b;
}

u64 max_u64(u64 a, u64 b) {
  return a > b ? a : b;
}

i64 max_i64(i64 a, i64 b) {
  return a > b ? a : b;
}

float32 sgn_f32(float32 a) {
  float32 b;
  if (a > 0)
    b = 1;
  else if (a < 0)
    b = -1;
  else
    b = 0;
  return b;
}

float64 sgn_f64(float64 a) {
  float32 b;
  if (a > 0)
    b = 1;
  else if (a < 0)
    b = -1;
  else
    b = 0;
  return b;
}

f32 atan2_f32(f32 a, f32 b) {
  return std::atan2(a, b);
}

f64 atan2_f64(f64 a, f64 b) {
  return std::atan2(a, b);
}

f32 pow_f32(f32 a, f32 b) {
  return std::pow(a, b);
}

f64 pow_f64(f64 a, f64 b) {
  return std::pow(a, b);
}

f32 __nv_sgnf(f32 x) {
  return sgn_f32(x);
}

f64 __nv_sgn(f64 x) {
  return sgn_f64(x);
}

struct PhysicalCoordinates {
  i32 val[taichi_max_num_indices];
};

STRUCT_FIELD_ARRAY(PhysicalCoordinates, val);

#include "taichi/program/context.h"

STRUCT_FIELD(RuntimeContext, runtime);
STRUCT_FIELD(RuntimeContext, result_buffer)

#include "taichi/runtime/llvm/runtime_module/atomic.h"

// These structures are accessible by both the LLVM backend and this C++ runtime
// file here (for building complex runtime functions in C++)

// These structs contain some "template parameters"

// Common Attributes
struct StructMeta {
  i32 snode_tree_id;
  i32 runtime_local_id;
  i32 snode_id;
  std::size_t element_size;
  i64 max_num_elements;
  u1 listgen_reuse;

  Ptr (*lookup_element)(Ptr, Ptr, int i);

  Ptr (*from_parent_element)(Ptr);

  u1 (*is_active)(Ptr, Ptr, int i);

  i32 (*get_num_elements)(Ptr, Ptr);

  void (*refine_coordinates)(PhysicalCoordinates *inp_coord,
                             PhysicalCoordinates *refined_coord,
                             int index);

  RuntimeContext *context;
};

STRUCT_FIELD(StructMeta, snode_tree_id)
STRUCT_FIELD(StructMeta, runtime_local_id)
STRUCT_FIELD(StructMeta, snode_id)
STRUCT_FIELD(StructMeta, element_size)
STRUCT_FIELD(StructMeta, max_num_elements)
STRUCT_FIELD(StructMeta, listgen_reuse)
STRUCT_FIELD(StructMeta, get_num_elements);
STRUCT_FIELD(StructMeta, lookup_element);
STRUCT_FIELD(StructMeta, from_parent_element);
STRUCT_FIELD(StructMeta, refine_coordinates);
STRUCT_FIELD(StructMeta, is_active);
STRUCT_FIELD(StructMeta, context);

struct LLVMRuntime;

constexpr bool enable_assert = true;

void taichi_assert(RuntimeContext *context, u1 test, const char *msg);
void taichi_assert_runtime(LLVMRuntime *runtime, u1 test, const char *msg);
#define TI_ASSERT_INFO(x, msg) taichi_assert(context, (u1)(x), msg)
#define TI_ASSERT(x) TI_ASSERT_INFO(x, #x)

void ___stubs___() {
#if ARCH_cuda
  cuda_vprintf(nullptr, nullptr);
  cuda_clock_i64();
#endif
}
}

#if defined(__clang__) || defined(__GNUC__)
template <typename T>
T debug_add(RuntimeContext *ctx, T a, T b, const char *tb) {
  T c;
  if (__builtin_add_overflow(a, b, &c)) {
    taichi_printf(ctx->runtime, "Addition overflow detected in %s\n", tb);
  }
  return c;
}

template <typename T>
T debug_sub(RuntimeContext *ctx, T a, T b, const char *tb) {
  T c;
  if (__builtin_sub_overflow(a, b, &c)) {
    taichi_printf(ctx->runtime, "Subtraction overflow detected in %s\n", tb);
  }
  return c;
}

template <typename T>
T debug_mul(RuntimeContext *ctx, T a, T b, const char *tb) {
  T c;
  if (__builtin_mul_overflow(a, b, &c)) {
    taichi_printf(ctx->runtime, "Multiplication overflow detected in %s\n", tb);
  }
  return c;
}

template <typename T>
T debug_shl(RuntimeContext *ctx, T a, i32 b, const char *tb) {
  T c = a << b;
  if (c >> b != a) {
    taichi_printf(ctx->runtime, "Shift left overflow detected in %s\n", tb);
  }
  return c;
}

extern "C" {

#define DEFINE_DEBUG_BIN_OP_TY(op, ty)                                    \
  ty debug_##op##_##ty(RuntimeContext *ctx, ty a, ty b, const char *tb) { \
    return debug_##op(ctx, a, b, tb);                                     \
  }

#define DEFINE_DEBUG_BIN_OP(op)   \
  DEFINE_DEBUG_BIN_OP_TY(op, i8)  \
  DEFINE_DEBUG_BIN_OP_TY(op, u8)  \
  DEFINE_DEBUG_BIN_OP_TY(op, i16) \
  DEFINE_DEBUG_BIN_OP_TY(op, u16) \
  DEFINE_DEBUG_BIN_OP_TY(op, i32) \
  DEFINE_DEBUG_BIN_OP_TY(op, u32) \
  DEFINE_DEBUG_BIN_OP_TY(op, i64) \
  DEFINE_DEBUG_BIN_OP_TY(op, u64)

DEFINE_DEBUG_BIN_OP(add)
DEFINE_DEBUG_BIN_OP(sub)
DEFINE_DEBUG_BIN_OP(mul)
DEFINE_DEBUG_BIN_OP(shl)
}
#endif

bool is_power_of_two(uint32 x) {
  return x != 0 && (x & (x - 1)) == 0;
}

/*
A simple list data structure that is infinitely long.
Data are organized in chunks, where each chunk is allocated on demand.
*/

// Forward decl for Phase 1 per-SNode dedicated pool pointer in ListManager.
struct PreallocatedMemoryChunk;

// TODO: there are many i32 types in this class, which may be an issue if there
// are >= 2 ** 31 elements.
struct ListManager {
  static constexpr std::size_t max_num_chunks =
      kLlvmListManagerMaxNumChunks;
  static constexpr i32 inline_num_chunks =
      (i32)kLlvmListManagerInlineChunks;
  static constexpr i32 chunks_per_directory =
      (i32)kLlvmListManagerChunksPerDirectory;
  static constexpr i32 num_chunk_directories =
      (i32)kLlvmListManagerDirectoryCount;
  Ptr inline_chunks[kLlvmListManagerInlineChunks]{};
  Ptr chunk_directories[kLlvmListManagerDirectoryCount]{};
  std::size_t element_size{0};
  std::size_t max_num_elements_per_chunk;
  i32 log2chunk_num_elements;
  i32 lock;
  i32 num_elements;
  LLVMRuntime *runtime;
  // Phase 1 (2026-05): when non-null, touch_chunk allocates from this
  // dedicated PreallocatedMemoryChunk instead of runtime_memory_chunk.
  // NodeManager sets this for its data_list/free_list/recycled_list.
  PreallocatedMemoryChunk *backing_chunk{nullptr};

  ListManager(LLVMRuntime *runtime,
              std::size_t element_size,
              std::size_t num_elements_per_chunk,
              PreallocatedMemoryChunk *backing = nullptr)
      : element_size(element_size),
        max_num_elements_per_chunk(num_elements_per_chunk),
        runtime(runtime),
        backing_chunk(backing) {
    taichi_assert_runtime(runtime, is_power_of_two(max_num_elements_per_chunk),
                          "max_num_elements_per_chunk must be POT.");
    lock = 0;
    num_elements = 0;
    log2chunk_num_elements = taichi::log2int(num_elements_per_chunk);
  }

  void append(void *data_ptr);

  i32 reserve_new_element() {
    auto i = atomic_add_i32(&num_elements, 1);
    auto chunk_id = i >> log2chunk_num_elements;
    touch_chunk(chunk_id);
    return i;
  }

  template <typename T>
  void push_back(const T &t) {
    this->append((void *)&t);
  }

  Ptr allocate();

  void touch_chunk(int chunk_id);

  Ptr get_chunk_ptr(i32 chunk_id) {
    if (chunk_id < inline_num_chunks) {
      return inline_chunks[chunk_id];
    }
    const i32 relative = chunk_id - inline_num_chunks;
    const i32 directory_id = relative / chunks_per_directory;
    const i32 directory_offset = relative % chunks_per_directory;
    auto directory = (Ptr *)chunk_directories[directory_id];
    return directory == nullptr ? nullptr : directory[directory_offset];
  }

  i32 get_num_active_chunks() {
    i32 counter = 0;
    for (int i = 0; i < inline_num_chunks; i++) {
      counter += (inline_chunks[i] != nullptr);
    }
    for (int i = 0; i < num_chunk_directories; i++) {
      auto directory = (Ptr *)chunk_directories[i];
      if (directory == nullptr) {
        continue;
      }
      for (int j = 0; j < chunks_per_directory; j++) {
        counter += (directory[j] != nullptr);
      }
    }
    return counter;
  }

  i32 get_num_active_chunk_directories() {
    i32 counter = 0;
    for (int i = 0; i < num_chunk_directories; i++) {
      counter += (chunk_directories[i] != nullptr);
    }
    return counter;
  }

  void clear() {
    num_elements = 0;
  }

  void resize(i32 n) {
    num_elements = n;
  }

  Ptr get_element_ptr(i32 i) {
    return get_chunk_ptr(i >> log2chunk_num_elements) +
           element_size * (i & ((1 << log2chunk_num_elements) - 1));
  }

  template <typename T>
  T &get(i32 i) {
    return *(T *)get_element_ptr(i);
  }

  Ptr touch_and_get(i32 i) {
    touch_chunk(i >> log2chunk_num_elements);
    return get_element_ptr(i);
  }

  i32 size() {
    return num_elements;
  }

  i32 ptr2index(Ptr ptr) {
    auto chunk_size = max_num_elements_per_chunk * element_size;
    for (int i = 0; i < inline_num_chunks; i++) {
      auto chunk = inline_chunks[i];
      taichi_assert_runtime(runtime, chunk != nullptr, "ptr not found.");
      if (chunk <= ptr && ptr < chunk + chunk_size) {
        return (i << log2chunk_num_elements) +
               i32((ptr - chunk) / element_size);
      }
    }
    for (int directory_id = 0; directory_id < num_chunk_directories;
         directory_id++) {
      auto directory = (Ptr *)chunk_directories[directory_id];
      taichi_assert_runtime(runtime, directory != nullptr, "ptr not found.");
      for (int directory_offset = 0;
           directory_offset < chunks_per_directory; directory_offset++) {
        auto chunk = directory[directory_offset];
        taichi_assert_runtime(runtime, chunk != nullptr, "ptr not found.");
        if (chunk <= ptr && ptr < chunk + chunk_size) {
          const i32 chunk_id = inline_num_chunks +
                               directory_id * chunks_per_directory +
                               directory_offset;
          return (chunk_id << log2chunk_num_elements) +
                 i32((ptr - chunk) / element_size);
        }
      }
    }
    return -1;
  }
};

extern "C" {

struct Element {
  Ptr element;
  int loop_bounds[2];
  PhysicalCoordinates pcoord;
};

STRUCT_FIELD(Element, element);
STRUCT_FIELD(Element, pcoord);
STRUCT_FIELD_ARRAY(Element, loop_bounds);

struct RandState {
  u32 x;
  u32 y;
  u32 z;
  u32 w;
  i32 lock;
};

void initialize_rand_state(RandState *state, u32 i) {
  state->x = 123456789 * i * 1000000007;
  state->y = 362436069;
  state->z = 521288629;
  state->w = 88675123;
  state->lock = 0;
}
}

struct NodeManager;

#if !ARCH_cuda && !ARCH_amdgpu
struct RecycledDirectAmbient {
  Ptr ptr;
  std::size_t size;
};
#endif

struct PreallocatedMemoryChunk {
  Ptr preallocated_head = nullptr;
  Ptr preallocated_tail = nullptr;
  std::size_t preallocated_size = 0;
};

struct LLVMRuntime {
  PreallocatedMemoryChunk runtime_objects_chunk;
  PreallocatedMemoryChunk runtime_memory_chunk;

  host_allocator_type host_allocator;
  assert_failed_type assert_failed;
  host_printf_type host_printf;
  host_vsnprintf_type host_vsnprintf;
  Ptr memory_pool;

  LlvmSNodeTreeRuntimeState **snode_tree_states;
  i32 snode_tree_state_capacity;

  Ptr thread_pool;
  parallel_for_type parallel_for;
#if !ARCH_cuda && !ARCH_amdgpu
  // These arrays are a bounded CPU performance cache, not an address space or
  // correctness limit. Overflowing entries are released instead of rejected.
  static constexpr i32 kRecycledSNodeCacheEntries = 4096;
  ListManager *recycled_element_lists[kRecycledSNodeCacheEntries *
                                      kLlvmElementListChunkSizeClasses];
  i32 recycled_element_list_count;
  NodeManager *recycled_node_allocators[kRecycledSNodeCacheEntries];
  i32 recycled_node_allocator_count;
  RecycledDirectAmbient
      recycled_direct_ambients[kRecycledSNodeCacheEntries];
  i32 recycled_direct_ambient_count;
#endif
  Ptr temporaries;
  RandState *rand_states;

  // Cross backend (CPU, CUDA, AMDGPU) runtime memory allocation
  Ptr allocate_aligned(PreallocatedMemoryChunk &memory_chunk,
                       std::size_t size,
                       std::size_t alignment,
                       bool request = false);

  // Allocate from preallocated memory (CUDA, AMDGPU)
  Ptr allocate_from_reserved_memory(PreallocatedMemoryChunk &memory_chunk,
                                    std::size_t size,
                                    std::size_t alignment);
  Ptr profiler;
  void (*profiler_start)(Ptr, Ptr);
  void (*profiler_stop)(Ptr);

  char error_message_template[taichi_error_message_max_length];
  uint64 error_message_arguments[taichi_error_message_max_num_arguments];
  i32 error_message_lock = 0;
  i64 error_code = 0;

  Ptr result_buffer;
  i32 allocator_lock;

  i32 num_rand_states;

  i64 total_requested_memory;

  i32 hash_insert_probe_count;
  i32 hash_insert_probe_total;
  i32 hash_insert_probe_max;
  i32 hash_lookup_probe_count;
  i32 hash_lookup_probe_total;
  i32 hash_lookup_probe_max;

#if !ARCH_cuda && !ARCH_amdgpu
  // Appended so existing LLVMRuntime field offsets stay stable. CPU SNodeTree
  // teardown uses these to bound process-wide sparse payload retained for
  // reconstruction without changing CUDA/AMDGPU allocation behavior.
  host_releaser_type host_releaser;
  std::size_t recycled_sparse_payload_bytes;
  std::size_t destroying_tree_sparse_payload_bytes;
  bool release_current_tree_sparse_payload;
#endif
  PreallocatedMemoryChunk *materializing_element_list_backing_chunk;
#if !ARCH_cuda && !ARCH_amdgpu
  // CPU listgen telemetry is debug-only. Keep just one task-local sample in
  // the runtime: KernelLauncher serializes offloaded tasks and accumulates the
  // sample immediately after each listgen task. This avoids a fixed per-SNode
  // counter array and any atomics in the listgen scan loops.
  bool sparse_listgen_work_recording;
  bool sparse_listgen_work_available;
  uint64 sparse_listgen_scanned_elements;
  uint64 sparse_listgen_emitted_elements;
  i32 sparse_listgen_execution_strategy;
  ListManager *cpu_parallel_listgen_offsets;
  bool sparse_listgen_reused;
#endif

  template <typename T>
  void set_result(std::size_t i, T t) {
    static_assert(sizeof(T) <= sizeof(uint64));
    ((u64 *)result_buffer)[i] =
        taichi_union_cast_with_different_sizes<uint64>(t);
  }

  template <typename T, typename... Args>
  T *create(Args &&...args) {
    auto ptr = (T *)allocate_aligned(runtime_memory_chunk, sizeof(T), 4096,
                                     true /*request*/);
    new (ptr) T(std::forward<Args>(args)...);
    return ptr;
  }
};

LlvmSNodeTreeRuntimeState *snode_tree_runtime_state(LLVMRuntime *runtime,
                                                   int tree_id) {
  taichi_assert_runtime(runtime, tree_id >= 0,
                        "Negative LLVM SNodeTree runtime id.");
  taichi_assert_runtime(runtime,
                        tree_id < runtime->snode_tree_state_capacity,
                        "LLVM SNodeTree runtime directory is too small.");
  auto *tree = runtime->snode_tree_states[tree_id];
  taichi_assert_runtime(runtime, tree != nullptr,
                        "LLVM SNodeTree runtime state is unavailable.");
  return tree;
}

LlvmSNodeRuntimeState *snode_runtime_state(LLVMRuntime *runtime,
                                           int tree_id,
                                           int local_id) {
  auto *tree = snode_tree_runtime_state(runtime, tree_id);
  taichi_assert_runtime(runtime, local_id >= 0 && local_id < tree->node_count,
                        "LLVM tree-local SNode id is out of bounds.");
  return &tree->nodes[local_id];
}

LlvmSNodeRuntimeState *snode_runtime_state(LLVMRuntime *runtime,
                                           StructMeta *meta) {
  return snode_runtime_state(runtime, meta->snode_tree_id,
                             meta->runtime_local_id);
}

LlvmSNodeRuntimeState *snode_runtime_state(LLVMRuntime *runtime,
                                           uint64 runtime_key) {
  const int tree_id = static_cast<int>(runtime_key >> 32);
  const int local_id = static_cast<int>(runtime_key & 0xffffffffu);
  return snode_runtime_state(runtime, tree_id, local_id);
}

u1 runtime_has_error(LLVMRuntime *runtime) {
  return __atomic_load_n(&runtime->error_code,
                         std::memory_order::memory_order_acquire) != 0;
}

// TODO: are these necessary?
STRUCT_FIELD(LLVMRuntime, temporaries);
STRUCT_FIELD(LLVMRuntime, assert_failed);
STRUCT_FIELD(LLVMRuntime, host_printf);
STRUCT_FIELD(LLVMRuntime, host_vsnprintf);
STRUCT_FIELD(LLVMRuntime, profiler);
STRUCT_FIELD(LLVMRuntime, profiler_start);
STRUCT_FIELD(LLVMRuntime, profiler_stop);
#if !ARCH_cuda && !ARCH_amdgpu
STRUCT_FIELD(LLVMRuntime, host_releaser);
#endif

// NodeManager of node S (hash, pointer) managers the memory allocation of S_ch
// It makes use of three ListManagers.
struct NodeManager {
  LLVMRuntime *runtime;
  i32 lock;

  i32 element_size;
  i32 chunk_num_elements;
  i32 free_list_used;
  i32 deterministic_capacity;
  i32 deterministic_active;
  i32 deterministic_peak;

  ListManager *free_list, *recycled_list, *data_list;
  i32 recycle_list_size_backup;

  // Phase 1 (2026-05): per-SNode dedicated bump region carved from the
  // global pool buffer. When has_dedicated is true, all 3 ListManagers
  // route data chunk allocations to dedicated_chunk instead of
  // runtime->runtime_memory_chunk. Zero-init = legacy path.
  PreallocatedMemoryChunk dedicated_chunk;
  bool has_dedicated{false};

  using list_data_type = i32;

  NodeManager(LLVMRuntime *runtime,
              i32 element_size,
              i32 chunk_num_elements = -1)
      : runtime(runtime), element_size(element_size) {
    // 128K elements per chunk, by default
    if (chunk_num_elements == -1) {
      chunk_num_elements = 128 * 1024;
    }
    // Maximum chunk size = 128 MB
    while (chunk_num_elements > 1 &&
           (uint64)chunk_num_elements * element_size > 128UL * 1024 * 1024) {
      chunk_num_elements /= 2;
    }
    this->chunk_num_elements = chunk_num_elements;
    free_list_used = 0;
    deterministic_capacity = 0;
    deterministic_active = 0;
    deterministic_peak = 0;
    free_list = runtime->create<ListManager>(runtime, sizeof(list_data_type),
                                             chunk_num_elements);
    recycled_list = runtime->create<ListManager>(
        runtime, sizeof(list_data_type), chunk_num_elements);
    data_list =
        runtime->create<ListManager>(runtime, element_size, chunk_num_elements);
  }

  // Phase 1 (2026-05): assign a dedicated bump region to this NodeManager
  // AFTER construction. ptr+size define the PreallocatedMemoryChunk sub-range
  // carved from the global pool buffer. All 3 ListManagers are retroactively
  // pointed to this chunk for data allocations.
  void set_dedicated_pool(Ptr ptr, std::size_t size) {
    dedicated_chunk.preallocated_head = ptr;
    dedicated_chunk.preallocated_tail = ptr + size;
    dedicated_chunk.preallocated_size = size;
    has_dedicated = true;
    free_list->backing_chunk = &dedicated_chunk;
    recycled_list->backing_chunk = &dedicated_chunk;
    data_list->backing_chunk = &dedicated_chunk;
  }

  Ptr allocate() {
    int old_cursor = atomic_add_i32(&free_list_used, 1);
    i32 l;
    if (old_cursor >= free_list->size()) {
      // running out of free list. allocate new.
      l = data_list->reserve_new_element();
    } else {
      // reuse
      l = free_list->get<list_data_type>(old_cursor);
    }
    return data_list->get_element_ptr(l);
  }

  i32 locate(Ptr ptr) {
    return data_list->ptr2index(ptr);
  }

  void recycle(Ptr ptr) {
    auto index = locate(ptr);
    recycled_list->append(&index);
  }

  void gc_serial() {
    // compact free list
    for (int i = free_list_used; i < free_list->size(); i++) {
      free_list->get<list_data_type>(i - free_list_used) =
          free_list->get<list_data_type>(i);
    }
    const i32 num_unused = max_i32(free_list->size() - free_list_used, 0);
    free_list_used = 0;
    free_list->resize(num_unused);

    // zero-fill recycled and push to free list
    for (int i = 0; i < recycled_list->size(); i++) {
      auto idx = recycled_list->get<list_data_type>(i);
      auto ptr = data_list->get_element_ptr(idx);
      std::memset(ptr, 0, element_size);
      free_list->push_back(idx);
    }
    recycled_list->clear();
  }
};

#if !ARCH_cuda && !ARCH_amdgpu
std::size_t list_manager_dynamic_storage_bytes(ListManager *list) {
  if (list == nullptr) {
    return 0;
  }
  const std::size_t chunk_bytes =
      list->max_num_elements_per_chunk * list->element_size;
  return std::size_t(list->get_num_active_chunks()) * chunk_bytes +
         std::size_t(list->get_num_active_chunk_directories()) *
             kLlvmListManagerDirectoryPageBytes;
}

std::size_t node_manager_dynamic_storage_bytes(NodeManager *manager) {
  if (manager == nullptr) {
    return 0;
  }
  return list_manager_dynamic_storage_bytes(manager->free_list) +
         list_manager_dynamic_storage_bytes(manager->recycled_list) +
         list_manager_dynamic_storage_bytes(manager->data_list);
}

void subtract_recycled_sparse_payload(LLVMRuntime *runtime,
                                        std::size_t bytes) {
  taichi_assert_runtime(
      runtime, runtime->recycled_sparse_payload_bytes >= bytes,
      "Recycled sparse payload accounting underflow.");
  runtime->recycled_sparse_payload_bytes -= bytes;
}

void release_list_manager_dynamic_storage(LLVMRuntime *runtime,
                                          ListManager *list) {
  taichi_assert_runtime(runtime, runtime->host_releaser != nullptr,
                        "Host sparse payload releaser is not initialized.");
  const std::size_t chunk_bytes =
      list->max_num_elements_per_chunk * list->element_size;
  for (int i = 0; i < ListManager::inline_num_chunks; ++i) {
    if (list->inline_chunks[i] != nullptr) {
      runtime->host_releaser(runtime->memory_pool, chunk_bytes,
                             list->inline_chunks[i]);
      list->inline_chunks[i] = nullptr;
    }
  }
  for (int i = 0; i < ListManager::num_chunk_directories; ++i) {
    auto directory = (Ptr *)list->chunk_directories[i];
    if (directory == nullptr) {
      continue;
    }
    for (int j = 0; j < ListManager::chunks_per_directory; ++j) {
      if (directory[j] != nullptr) {
        runtime->host_releaser(runtime->memory_pool, chunk_bytes,
                               directory[j]);
        directory[j] = nullptr;
      }
    }
    runtime->host_releaser(runtime->memory_pool,
                           kLlvmListManagerDirectoryPageBytes, directory);
    list->chunk_directories[i] = nullptr;
  }
}

void reset_list_manager_for_reuse(ListManager *list, bool zero_chunks) {
  list->lock = 0;
  list->num_elements = 0;
  list->backing_chunk = nullptr;
  if (!zero_chunks) {
    return;
  }
  const std::size_t chunk_bytes =
      list->max_num_elements_per_chunk * list->element_size;
  for (int i = 0; i < ListManager::inline_num_chunks; ++i) {
    if (list->inline_chunks[i] != nullptr) {
      std::memset(list->inline_chunks[i], 0, chunk_bytes);
    }
  }
  for (int i = 0; i < ListManager::num_chunk_directories; ++i) {
    auto directory = (Ptr *)list->chunk_directories[i];
    if (directory == nullptr) {
      continue;
    }
    for (int j = 0; j < ListManager::chunks_per_directory; ++j) {
      if (directory[j] != nullptr) {
        std::memset(directory[j], 0, chunk_bytes);
      }
    }
  }
}

ListManager *acquire_element_list(LLVMRuntime *runtime,
                                  std::size_t chunk_num_elements) {
  for (int i = runtime->recycled_element_list_count - 1; i >= 0; --i) {
    ListManager *list = runtime->recycled_element_lists[i];
    if (list->max_num_elements_per_chunk != chunk_num_elements) {
      continue;
    }
    runtime->recycled_element_lists[i] =
        runtime->recycled_element_lists[
            --runtime->recycled_element_list_count];
    subtract_recycled_sparse_payload(
        runtime, list_manager_dynamic_storage_bytes(list));
    reset_list_manager_for_reuse(list, false);
    return list;
  }
  return runtime->create<ListManager>(runtime, sizeof(Element),
                                      chunk_num_elements);
}

NodeManager *acquire_node_manager(LLVMRuntime *runtime,
                                  std::size_t node_size,
                                  int chunk_num_elements) {
  for (int i = runtime->recycled_node_allocator_count - 1; i >= 0; --i) {
    NodeManager *manager = runtime->recycled_node_allocators[i];
    if (manager->element_size != node_size ||
        manager->chunk_num_elements != chunk_num_elements) {
      continue;
    }
    runtime->recycled_node_allocators[i] =
        runtime->recycled_node_allocators[
            --runtime->recycled_node_allocator_count];
    subtract_recycled_sparse_payload(
        runtime, node_manager_dynamic_storage_bytes(manager));
    manager->lock = 0;
    manager->free_list_used = 0;
    manager->deterministic_capacity = 0;
    manager->deterministic_active = 0;
    manager->deterministic_peak = 0;
    manager->recycle_list_size_backup = 0;
    manager->dedicated_chunk = {};
    manager->has_dedicated = false;
    reset_list_manager_for_reuse(manager->free_list, false);
    reset_list_manager_for_reuse(manager->recycled_list, false);
    reset_list_manager_for_reuse(manager->data_list, true);
    return manager;
  }
  return runtime->create<NodeManager>(runtime, node_size, chunk_num_elements);
}

Ptr acquire_direct_ambient(LLVMRuntime *runtime, std::size_t node_size) {
  for (int i = runtime->recycled_direct_ambient_count - 1; i >= 0; --i) {
    const auto ambient = runtime->recycled_direct_ambients[i];
    if (ambient.size != node_size) {
      continue;
    }
    runtime->recycled_direct_ambients[i] =
        runtime->recycled_direct_ambients[
            --runtime->recycled_direct_ambient_count];
    subtract_recycled_sparse_payload(runtime, ambient.size);
    std::memset(ambient.ptr, 0, node_size);
    return ambient.ptr;
  }
  return runtime->allocate_aligned(runtime->runtime_memory_chunk, node_size, 8,
                                   true /*request*/);
}
#endif

extern "C" void runtime_prepare_snode_tree_destroy(
    LLVMRuntime *runtime,
    int tree_id,
    int local_id,
    std::size_t direct_ambient_bytes,
    int first_snode,
    int last_snode) {
#if !ARCH_cuda && !ARCH_amdgpu
  auto *state = snode_runtime_state(runtime, tree_id, local_id);
  if (first_snode != 0) {
    runtime->destroying_tree_sparse_payload_bytes = 0;
  }
  runtime->destroying_tree_sparse_payload_bytes += direct_ambient_bytes;
  runtime->destroying_tree_sparse_payload_bytes +=
      list_manager_dynamic_storage_bytes(state->element_list);
  runtime->destroying_tree_sparse_payload_bytes +=
      node_manager_dynamic_storage_bytes(state->node_allocator);

  if (last_snode != 0) {
    const std::size_t tree_payload_bytes =
        runtime->destroying_tree_sparse_payload_bytes;
    const std::size_t budget = kLlvmHostSparseRecycledPayloadBudgetBytes;
    runtime->release_current_tree_sparse_payload =
        runtime->host_releaser != nullptr &&
        (tree_payload_bytes > budget ||
         runtime->recycled_sparse_payload_bytes >
             budget - tree_payload_bytes);
  }
#else
  (void)runtime;
  (void)tree_id;
  (void)local_id;
  (void)direct_ambient_bytes;
  (void)first_snode;
  (void)last_snode;
#endif
}

extern "C" void runtime_destroy_snode_resources(
    LLVMRuntime *runtime,
    int tree_id,
    int local_id,
    int has_element_list,
    int has_node_allocator,
    int has_direct_ambient,
    std::size_t direct_ambient_size) {
#if !ARCH_cuda && !ARCH_amdgpu
  auto *state = snode_runtime_state(runtime, tree_id, local_id);
  if (has_element_list) {
    ListManager *list = state->element_list;
    const std::size_t list_bytes =
        list_manager_dynamic_storage_bytes(list);
    const bool cache_available =
        runtime->recycled_element_list_count <
        LLVMRuntime::kRecycledSNodeCacheEntries *
            kLlvmElementListChunkSizeClasses;
    if (runtime->release_current_tree_sparse_payload || !cache_available) {
      release_list_manager_dynamic_storage(runtime, list);
    } else {
      runtime->recycled_sparse_payload_bytes += list_bytes;
    }
    if (cache_available) {
      reset_list_manager_for_reuse(list, false);
      runtime
          ->recycled_element_lists[runtime->recycled_element_list_count++] =
          list;
    } else {
      runtime->host_releaser(runtime->memory_pool, sizeof(ListManager), list);
    }
    state->element_list = nullptr;
  }

  if (has_node_allocator) {
    NodeManager *manager = state->node_allocator;
    const std::size_t manager_bytes =
        node_manager_dynamic_storage_bytes(manager);
    const bool cache_available =
        runtime->recycled_node_allocator_count <
        LLVMRuntime::kRecycledSNodeCacheEntries;
    if (runtime->release_current_tree_sparse_payload || !cache_available) {
      release_list_manager_dynamic_storage(runtime, manager->free_list);
      release_list_manager_dynamic_storage(runtime, manager->recycled_list);
      release_list_manager_dynamic_storage(runtime, manager->data_list);
    } else {
      runtime->recycled_sparse_payload_bytes += manager_bytes;
    }
    if (cache_available) {
      runtime->recycled_node_allocators[
          runtime->recycled_node_allocator_count++] = manager;
    } else {
      runtime->host_releaser(runtime->memory_pool, sizeof(ListManager),
                             manager->free_list);
      runtime->host_releaser(runtime->memory_pool, sizeof(ListManager),
                             manager->recycled_list);
      runtime->host_releaser(runtime->memory_pool, sizeof(ListManager),
                             manager->data_list);
      runtime->host_releaser(runtime->memory_pool, sizeof(NodeManager),
                             manager);
    }
    state->node_allocator = nullptr;
  }

  if (has_direct_ambient && direct_ambient_size > 0) {
    if (runtime->release_current_tree_sparse_payload) {
      runtime->host_releaser(runtime->memory_pool, direct_ambient_size,
                             state->ambient_element);
    } else {
      if (runtime->recycled_direct_ambient_count <
          LLVMRuntime::kRecycledSNodeCacheEntries) {
        runtime->recycled_direct_ambients[
            runtime->recycled_direct_ambient_count++] = {
            state->ambient_element, direct_ambient_size};
        runtime->recycled_sparse_payload_bytes += direct_ambient_size;
      } else {
        runtime->host_releaser(runtime->memory_pool, direct_ambient_size,
                               state->ambient_element);
      }
    }
  }

  state->ambient_element = nullptr;
  state->element_list_dirty_epoch = 1;
  state->element_list_dirty_flag = 1;
  state->element_list_version = 0;
  state->element_list_clean_epoch = 0;
  state->element_list_clean_parent_version = 0;
#else
  (void)runtime;
  (void)tree_id;
  (void)local_id;
  (void)has_element_list;
  (void)has_node_allocator;
  (void)has_direct_ambient;
  (void)direct_ambient_size;
#endif
}

extern "C" {

void RuntimeContext_store_result(RuntimeContext *ctx, u64 ret, u32 idx) {
  ctx->result_buffer[taichi_result_buffer_ret_value_id + idx] = ret;
}

void LLVMRuntime_profiler_start(LLVMRuntime *runtime, Ptr kernel_name) {
  runtime->profiler_start(runtime->profiler, kernel_name);
}

void LLVMRuntime_profiler_stop(LLVMRuntime *runtime) {
  runtime->profiler_stop(runtime->profiler);
}

Ptr get_temporary_pointer(LLVMRuntime *runtime, u64 offset) {
  return runtime->temporaries + offset;
}

void runtime_retrieve_and_reset_error_code(LLVMRuntime *runtime) {
  runtime->set_result(taichi_result_buffer_error_id,
                      atomic_exchange_i64(&runtime->error_code, 0));
}

u1 LLVMRuntime_has_error(LLVMRuntime *runtime) {
  return runtime_has_error(runtime);
}

void runtime_retrieve_error_message(LLVMRuntime *runtime, int i) {
  runtime->set_result(taichi_result_buffer_error_id,
                      runtime->error_message_template[i]);
}

void runtime_retrieve_error_message_argument(LLVMRuntime *runtime,
                                             int argument_id) {
  runtime->set_result(taichi_result_buffer_error_id,
                      runtime->error_message_arguments[argument_id]);
}

void runtime_ListManager_get_num_active_chunks(LLVMRuntime *runtime,
                                               ListManager *list_manager) {
  runtime->set_result(taichi_result_buffer_runtime_query_id,
                      list_manager->get_num_active_chunks());
}

void runtime_sparse_tree_statistics_reset(uint64 *result) {
  for (int i = 0; i < kLlvmSparseTreeStatisticCount; ++i) {
    result[i] = 0;
  }
}

void runtime_sparse_snode_statistics_collect(LLVMRuntime *runtime,
                                             int tree_id,
                                             int local_id,
                                             int has_element_lists,
                                             uint64 *result) {
  auto *state = snode_runtime_state(runtime, tree_id, local_id);
  if (has_element_lists != 0) {
    ListManager *active_list = state->element_list;
    if (active_list != nullptr) {
      const uint64 chunk_bytes =
          uint64(active_list->max_num_elements_per_chunk) *
          uint64(active_list->element_size);
      result[kLlvmSparseRuntimeMetadataRequestedBytes] +=
          sizeof(ListManager) +
          uint64(active_list->get_num_active_chunk_directories()) *
              kLlvmListManagerDirectoryPageBytes;
      result[kLlvmSparseActiveListReservedBytes] +=
          uint64(active_list->get_num_active_chunks()) * chunk_bytes;
      result[kLlvmSparseActiveListUsedBytes] +=
          uint64(active_list->size()) * uint64(active_list->element_size);
    }
  }

  NodeManager *allocator = state->node_allocator;
  if (allocator == nullptr) {
    return;
  }
  ListManager *data = allocator->data_list;
  ListManager *free = allocator->free_list;
  ListManager *recycled = allocator->recycled_list;
  result[kLlvmSparseRuntimeMetadataRequestedBytes] +=
      sizeof(NodeManager) + 3 * sizeof(ListManager) +
      uint64(data->get_num_active_chunk_directories() +
             free->get_num_active_chunk_directories() +
             recycled->get_num_active_chunk_directories()) *
          kLlvmListManagerDirectoryPageBytes;

  if (allocator->deterministic_capacity > 0) {
    const i64 active = max_i32(allocator->deterministic_active, 0);
    const i64 peak = max_i32(allocator->deterministic_peak, active);
    result[kLlvmSparseAllocatorPayloadReservedBytes] +=
        uint64(allocator->deterministic_capacity) *
        uint64(allocator->element_size);
    result[kLlvmSparseAllocatorPayloadUsedBytes] +=
        uint64(active) * uint64(allocator->element_size);
    result[kLlvmSparseAllocatorInUseElements] += uint64(active);
    result[kLlvmSparseAllocatorFreeElements] += uint64(peak - active);
    return;
  }

  const uint64 data_chunk_bytes =
      uint64(data->max_num_elements_per_chunk) * uint64(data->element_size);
  const uint64 free_chunk_bytes =
      uint64(free->max_num_elements_per_chunk) * uint64(free->element_size);
  const uint64 recycled_chunk_bytes =
      uint64(recycled->max_num_elements_per_chunk) *
      uint64(recycled->element_size);
  result[kLlvmSparseAllocatorPayloadReservedBytes] +=
      uint64(data->get_num_active_chunks()) * data_chunk_bytes;
  result[kLlvmSparseAllocatorBookkeepingReservedBytes] +=
      uint64(free->get_num_active_chunks()) * free_chunk_bytes +
      uint64(recycled->get_num_active_chunks()) * recycled_chunk_bytes;

  const i64 free_elements =
      max_i32(free->size() - allocator->free_list_used, 0);
  const i64 recycled_elements = recycled->size();
  const i64 in_use_elements =
      std::max(i64(data->size()) - free_elements - recycled_elements, i64(0));
  result[kLlvmSparseAllocatorInUseElements] += uint64(in_use_elements);
  result[kLlvmSparseAllocatorFreeElements] += uint64(free_elements);
  result[kLlvmSparseAllocatorRecycledElements] += uint64(recycled_elements);
  result[kLlvmSparseAllocatorPayloadUsedBytes] +=
      uint64(in_use_elements) * uint64(data->element_size);
}

RUNTIME_STRUCT_FIELD(LLVMRuntime, total_requested_memory);

Ptr LLVMRuntime_get_snode_tree_root(LLVMRuntime *runtime, int tree_id) {
  return snode_tree_runtime_state(runtime, tree_id)->root;
}

// Kernel entry points are registered and launched while Program owns the
// SNodeTree lifecycle read transaction. Graph entry points additionally
// validate the captured tree generation before reaching the backend. Keep the
// checked accessor above for runtime queries and diagnostics, but do not
// repeat directory bounds/null assertions at every statically bound field
// access in a hot kernel.
Ptr LLVMRuntime_get_snode_tree_root_unchecked(LLVMRuntime *runtime,
                                              int tree_id) {
  return runtime->snode_tree_states[tree_id]->root;
}

void runtime_get_snode_node_allocator(LLVMRuntime *runtime,
                                      int tree_id,
                                      int local_id) {
  runtime->set_result(taichi_result_buffer_runtime_query_id,
                      snode_runtime_state(runtime, tree_id, local_id)
                          ->node_allocator);
}

void runtime_get_snode_element_list(LLVMRuntime *runtime,
                                    int tree_id,
                                    int local_id) {
  runtime->set_result(taichi_result_buffer_runtime_query_id,
                      snode_runtime_state(runtime, tree_id, local_id)
                          ->element_list);
}

void runtime_hash_probe_stats_reset(LLVMRuntime *runtime) {
  runtime->hash_insert_probe_count = 0;
  runtime->hash_insert_probe_total = 0;
  runtime->hash_insert_probe_max = 0;
  runtime->hash_lookup_probe_count = 0;
  runtime->hash_lookup_probe_total = 0;
  runtime->hash_lookup_probe_max = 0;
}

void runtime_hash_probe_stats_get(LLVMRuntime *runtime, int index) {
  i32 value = 0;
  if (index == 0) {
    value = runtime->hash_insert_probe_count;
  } else if (index == 1) {
    value = runtime->hash_insert_probe_total;
  } else if (index == 2) {
    value = runtime->hash_insert_probe_max;
  } else if (index == 3) {
    value = runtime->hash_lookup_probe_count;
  } else if (index == 4) {
    value = runtime->hash_lookup_probe_total;
  } else if (index == 5) {
    value = runtime->hash_lookup_probe_max;
  }
  runtime->set_result(taichi_result_buffer_runtime_query_id, value);
}

RUNTIME_STRUCT_FIELD(NodeManager, free_list);
RUNTIME_STRUCT_FIELD(NodeManager, recycled_list);
RUNTIME_STRUCT_FIELD(NodeManager, data_list);
RUNTIME_STRUCT_FIELD(NodeManager, free_list_used);
RUNTIME_STRUCT_FIELD(NodeManager, deterministic_capacity);
RUNTIME_STRUCT_FIELD(NodeManager, deterministic_peak);

RUNTIME_STRUCT_FIELD(ListManager, num_elements);
RUNTIME_STRUCT_FIELD(ListManager, max_num_elements_per_chunk);
RUNTIME_STRUCT_FIELD(ListManager, element_size);

void mark_element_lists_dirty_if_reuse(StructMeta *meta) {
  if (!meta->listgen_reuse) {
    return;
  }
  auto runtime = meta->context->runtime;
  auto *state = snode_runtime_state(runtime, meta);
  if (state->element_list_dirty_flag != 0) {
    return;
  }
  if (atomic_exchange_i32(&state->element_list_dirty_flag, 1) == 0) {
    atomic_add_i32(&state->element_list_dirty_epoch, 1);
  }
}

u1 element_list_is_current(LLVMRuntime *runtime,
                           StructMeta *parent,
                           StructMeta *child) {
  auto *child_state = snode_runtime_state(runtime, child);
  auto *parent_state = snode_runtime_state(runtime, parent);
  return child_state->element_list_clean_epoch ==
             child_state->element_list_dirty_epoch &&
         child_state->element_list_clean_parent_version ==
             parent_state->element_list_version;
}

void mark_element_list_current(LLVMRuntime *runtime,
                               StructMeta *parent,
                               StructMeta *child) {
  auto *child_state = snode_runtime_state(runtime, child);
  auto *parent_state = snode_runtime_state(runtime, parent);
  if (block_idx() == 0 && thread_idx() == 0) {
    child_state->element_list_clean_epoch =
        child_state->element_list_dirty_epoch;
    child_state->element_list_clean_parent_version =
        parent_state->element_list_version;
    child_state->element_list_dirty_flag = 0;
    atomic_add_i32(&child_state->element_list_version, 1);
  }
}

void taichi_assert(RuntimeContext *context, u1 test, const char *msg) {
  taichi_assert_runtime(context->runtime, test, msg);
}

void taichi_assert_format(LLVMRuntime *runtime,
                          u1 test,
                          const char *format,
                          int num_arguments,
                          uint64 *arguments) {
#ifdef ARCH_amdgpu
  // TODO: find out why error with mark_force_no_inline
  //  llvm::SDValue llvm::SelectionDAG::getNode(unsigned int, const llvm::SDLoc
  //  &, llvm::EVT, llvm::SDValue, const llvm::SDNodeFlags): Assertion
  //  `VT.getSizeInBits() == Operand.getValueSizeInBits() && "Cannot BITCAST
  //  between types of different sizes!"' failed.
#else
  mark_force_no_inline();
#endif
  if (!enable_assert || test != 0)
    return;
  if (!runtime_has_error(runtime)) {
    locked_task(
        &runtime->error_message_lock,
        [&] {
          memset(runtime->error_message_template, 0,
                 taichi_error_message_max_length);
          memcpy(runtime->error_message_template, format,
                 std::min(taichi_strlen(format),
                          taichi_error_message_max_length - 1));
          for (int i = 0; i < num_arguments; i++) {
            runtime->error_message_arguments[i] = arguments[i];
          }
          // Publish the fault only after its message is complete. The atomic
          // flag is also the cooperative-cancellation signal on CPU.
          atomic_exchange_i64(&runtime->error_code, 1);
        },
        [&] { return !runtime_has_error(runtime); });
  }
#if ARCH_cuda
  // Kill this CUDA thread.
  asm("exit;");
#elif ARCH_amdgpu
  asm("S_ENDPGM");
#endif
}

void taichi_assert_runtime(LLVMRuntime *runtime, u1 test, const char *msg) {
  taichi_assert_format(runtime, test, msg, 0, nullptr);
}

// [ON HOST] CPU backend
// [ON DEVICE] CUDA/AMDGPU backend
Ptr LLVMRuntime::allocate_aligned(PreallocatedMemoryChunk &memory_chunk,
                                  std::size_t size,
                                  std::size_t alignment,
                                  bool request) {
  if (request)
    atomic_add_i64(&total_requested_memory, size);

  if (memory_chunk.preallocated_size > 0) {
    return allocate_from_reserved_memory(memory_chunk, size, alignment);
  }

  return (Ptr)host_allocator(memory_pool, size, alignment);
}

// [ONLY ON DEVICE] CUDA/AMDGPU backend
Ptr LLVMRuntime::allocate_from_reserved_memory(
    PreallocatedMemoryChunk &memory_chunk,
    std::size_t size,
    std::size_t alignment) {
  Ptr ret = nullptr;
  bool success = false;
  locked_task(&allocator_lock, [&] {
    std::size_t preallocated_head = (std::size_t)memory_chunk.preallocated_head;
    std::size_t preallocated_tail = (std::size_t)memory_chunk.preallocated_tail;

    auto alignment_bytes =
        alignment - 1 - (preallocated_head + alignment - 1) % alignment;
    size += alignment_bytes;
    if (preallocated_head + size <= preallocated_tail) {
      ret = (Ptr)(preallocated_head + alignment_bytes);
      memory_chunk.preallocated_head += size;
      success = true;
    } else {
      success = false;
    }
  });
  if (!success) {
#if ARCH_cuda
    // Here unfortunately we have to rely on a native CUDA assert failure to
    // halt the whole grid. Using a taichi_assert_runtime will not finish the
    // whole kernel execution immediately.
    __assertfail(
        "Out of CUDA pre-allocated memory.\n"
        "Consider using ti.init(device_memory_fraction=0.9) or "
        "ti.init(device_memory_GB=4) to allocate more"
        " GPU memory",
        "Taichi JIT", 0, "allocate_from_reserved_memory", 1);
#endif
  }
  taichi_assert_runtime(this, success, "Out of pre-allocated memory");
  return ret;
}

// External API
// [ON HOST] CPU backend
// [ON DEVICE] CUDA/AMDGPU backend
void runtime_memory_allocate_aligned(LLVMRuntime *runtime,
                                     std::size_t size,
                                     std::size_t alignment,
                                     uint64 *result) {
  *result =
      taichi_union_cast_with_different_sizes<uint64>(runtime->allocate_aligned(
          runtime->runtime_memory_chunk, size, alignment));
}

// External API
// [ON HOST] CPU backend
// [ON DEVICE] CUDA/AMDGPU backend
void runtime_get_memory_requirements(Ptr result_buffer,
                                     i32 num_rand_states,
                                     i32 use_preallocated_buffer) {
  i64 size = 0;

  if (use_preallocated_buffer) {
    size += taichi::iroundup(i64(sizeof(LLVMRuntime)), taichi_page_size);
  }

  size +=
      taichi::iroundup(i64(taichi_global_tmp_buffer_size), taichi_page_size);
  size += taichi::iroundup(i64(sizeof(RandState)) * num_rand_states,
                           taichi_page_size);

  reinterpret_cast<i64 *>(result_buffer)[0] = size;
}

// External API
// [ON HOST] CPU backend
// [ON DEVICE] CUDA/AMDGPU backend
void runtime_initialize(
    Ptr result_buffer,
    Ptr memory_pool,
    std::size_t
        preallocated_size,  // Non-zero means use the preallocated buffer
    Ptr preallocated_buffer,
    i32 num_rand_states,
    void *_host_allocator,
    void *_host_printf,
    void *_host_vsnprintf) {
  // bootstrap
  auto host_allocator = (host_allocator_type)_host_allocator;
  auto host_printf = (host_printf_type)_host_printf;
  auto host_vsnprintf = (host_vsnprintf_type)_host_vsnprintf;
  LLVMRuntime *runtime = nullptr;
  Ptr preallocated_tail = preallocated_buffer + preallocated_size;
  if (preallocated_size) {
    runtime = (LLVMRuntime *)preallocated_buffer;
    preallocated_buffer +=
        taichi::iroundup(sizeof(LLVMRuntime), taichi_page_size);
  } else {
    runtime =
        (LLVMRuntime *)host_allocator(memory_pool, sizeof(LLVMRuntime), 128);
  }

  PreallocatedMemoryChunk runtime_objects_chunk;
  runtime_objects_chunk.preallocated_size = preallocated_size;
  runtime_objects_chunk.preallocated_head = preallocated_buffer;
  runtime_objects_chunk.preallocated_tail = preallocated_tail;

  runtime->runtime_objects_chunk = std::move(runtime_objects_chunk);

  runtime->result_buffer = result_buffer;
  runtime->set_result(taichi_result_buffer_ret_value_id, runtime);
  runtime->host_allocator = host_allocator;
  runtime->host_printf = host_printf;
  runtime->host_vsnprintf = host_vsnprintf;
  runtime->memory_pool = memory_pool;
  runtime->snode_tree_states = nullptr;
  runtime->snode_tree_state_capacity = 0;

  runtime->total_requested_memory = 0;
  runtime->materializing_element_list_backing_chunk = nullptr;
  runtime_hash_probe_stats_reset(runtime);
#if !ARCH_cuda && !ARCH_amdgpu
  runtime->recycled_element_list_count = 0;
  runtime->recycled_node_allocator_count = 0;
  runtime->recycled_direct_ambient_count = 0;
  runtime->host_releaser = nullptr;
  runtime->recycled_sparse_payload_bytes = 0;
  runtime->destroying_tree_sparse_payload_bytes = 0;
  runtime->release_current_tree_sparse_payload = false;
  runtime->sparse_listgen_work_recording = false;
  runtime->sparse_listgen_work_available = false;
  runtime->sparse_listgen_scanned_elements = 0;
  runtime->sparse_listgen_emitted_elements = 0;
  runtime->sparse_listgen_execution_strategy = 0;
  runtime->cpu_parallel_listgen_offsets = nullptr;
  runtime->sparse_listgen_reused = false;
#endif

  runtime->temporaries = (Ptr)runtime->allocate_aligned(
      runtime->runtime_objects_chunk, taichi_global_tmp_buffer_size,
      taichi_page_size);

  runtime->num_rand_states = num_rand_states;
  runtime->rand_states = (RandState *)runtime->allocate_aligned(
      runtime->runtime_objects_chunk,
      sizeof(RandState) * runtime->num_rand_states, taichi_page_size);
}

void runtime_initialize_memory(LLVMRuntime *runtime,
                               std::size_t preallocated_size,
                               Ptr preallocated_buffer) {
  if (preallocated_size) {
    runtime->runtime_memory_chunk.preallocated_size = preallocated_size;
    runtime->runtime_memory_chunk.preallocated_head = preallocated_buffer;
    runtime->runtime_memory_chunk.preallocated_tail =
        preallocated_buffer + preallocated_size;
  }
}

void runtime_initialize_rand_states_cuda(LLVMRuntime *runtime,
                                         int starting_rand_state) {
  int i = block_dim() * block_idx() + thread_idx();
  initialize_rand_state(&runtime->rand_states[i], starting_rand_state + i);
}

void runtime_initialize_rand_states_serial(LLVMRuntime *runtime,
                                           int starting_rand_state) {
  for (int i = 0; i < runtime->num_rand_states; i++) {
    initialize_rand_state(&runtime->rand_states[i], starting_rand_state + i);
  }
}

#if !ARCH_cuda && !ARCH_amdgpu
void runtime_sparse_listgen_work_begin(LLVMRuntime *runtime) {
  runtime->sparse_listgen_work_recording = true;
  runtime->sparse_listgen_work_available = false;
  runtime->sparse_listgen_scanned_elements = 0;
  runtime->sparse_listgen_emitted_elements = 0;
  runtime->sparse_listgen_execution_strategy = 0;
  runtime->sparse_listgen_reused = false;
}

void runtime_sparse_listgen_work_read(LLVMRuntime *runtime, uint64 *result) {
  result[0] = runtime->sparse_listgen_work_available ? 1 : 0;
  result[1] = runtime->sparse_listgen_scanned_elements;
  result[2] = runtime->sparse_listgen_emitted_elements;
  result[3] = static_cast<uint64>(runtime->sparse_listgen_execution_strategy);
  result[4] = runtime->sparse_listgen_reused ? 1 : 0;
}

void runtime_cpu_parallel_listgen_workspace_statistics(LLVMRuntime *runtime,
                                                       uint64 *result) {
  auto workspace = runtime->cpu_parallel_listgen_offsets;
  if (workspace == nullptr) {
    result[0] = 0;
    return;
  }
  result[0] = sizeof(ListManager) +
              static_cast<uint64>(workspace->get_num_active_chunks()) *
                  workspace->max_num_elements_per_chunk *
                  workspace->element_size +
              static_cast<uint64>(
                  workspace->get_num_active_chunk_directories()) *
                  kLlvmListManagerDirectoryPageBytes;
}
#endif

void runtime_set_snode_tree_directory(LLVMRuntime *runtime,
                                      Ptr directory,
                                      int capacity) {
  taichi_assert_runtime(runtime, capacity >= 0,
                        "Negative LLVM SNodeTree directory capacity.");
  runtime->snode_tree_states =
      reinterpret_cast<LlvmSNodeTreeRuntimeState **>(directory);
  runtime->snode_tree_state_capacity = capacity;
}

void runtime_register_snode_tree_state(LLVMRuntime *runtime,
                                       int tree_id,
                                       Ptr state_ptr,
                                       uint64 generation) {
  taichi_assert_runtime(runtime, tree_id >= 0 &&
                                     tree_id < runtime->snode_tree_state_capacity,
                        "LLVM SNodeTree runtime directory is too small.");
  auto *tree = reinterpret_cast<LlvmSNodeTreeRuntimeState *>(state_ptr);
  taichi_assert_runtime(runtime, tree != nullptr &&
                                     tree->generation == generation,
                        "LLVM SNodeTree runtime generation mismatch.");
  runtime->snode_tree_states[tree_id] = tree;
}

void runtime_unregister_snode_tree_state(LLVMRuntime *runtime,
                                         int tree_id,
                                         uint64 generation) {
  auto *tree = snode_tree_runtime_state(runtime, tree_id);
  taichi_assert_runtime(runtime, tree->generation == generation,
                        "Stale LLVM SNodeTree runtime destroy request.");
  runtime->snode_tree_states[tree_id] = nullptr;
}

void runtime_initialize_snode_element_list(LLVMRuntime *runtime,
                                           int tree_id,
                                           int local_id,
                                           std::size_t chunk_num_elements) {
  auto *state = snode_runtime_state(runtime, tree_id, local_id);
  taichi_assert_runtime(runtime, state->element_list == nullptr,
                        "SNode element list initialized twice.");
#if !ARCH_cuda && !ARCH_amdgpu
  state->element_list = acquire_element_list(runtime, chunk_num_elements);
#else
  state->element_list = runtime->create<ListManager>(
      runtime, sizeof(Element), chunk_num_elements);
#endif
}

void runtime_initialize_snodes(LLVMRuntime *runtime,
                               std::size_t root_size,
                               int root_local_id,
                               int num_snodes,
                               int snode_tree_id,
                               uint64 generation,
                               std::size_t rounded_size,
                               Ptr ptr,
                               Ptr state_ptr,
                               std::size_t state_bytes,
                               std::size_t root_list_chunk_num_elements,
                               bool all_dense) {
  (void)root_size;
  std::size_t required_state_bytes = 0;
  taichi_assert_runtime(
      runtime,
      llvm_snode_tree_runtime_state_bytes(num_snodes, &required_state_bytes) &&
          required_state_bytes <= state_bytes,
      "LLVM SNodeTree runtime state allocation is too small.");
  auto *tree = reinterpret_cast<LlvmSNodeTreeRuntimeState *>(state_ptr);
  tree->root = ptr;
  tree->root_mem_size = rounded_size;
  tree->nodes = reinterpret_cast<LlvmSNodeRuntimeState *>(
      state_ptr + llvm_snode_tree_runtime_nodes_offset());
  tree->node_count = num_snodes;
  tree->reserved = 0;
  tree->generation = generation;
  for (int i = 0; i < num_snodes; ++i) {
    tree->nodes[i] = LlvmSNodeRuntimeState{};
  }
  runtime_register_snode_tree_state(runtime, snode_tree_id, state_ptr,
                                    generation);

  // runtime->request_allocate_aligned ready to use
  // initialize the root node element list
  if (all_dense) {
    return;
  }
  runtime_initialize_snode_element_list(runtime, snode_tree_id, root_local_id,
                                        root_list_chunk_num_elements);
  auto *root_state = snode_runtime_state(runtime, snode_tree_id, root_local_id);
  Element elem;
  elem.loop_bounds[0] = 0;
  elem.loop_bounds[1] = 1;
  elem.element = tree->root;
  for (int i = 0; i < taichi_max_num_indices; i++) {
    elem.pcoord.val[i] = 0;
  }

  root_state->element_list->append(&elem);
  root_state->element_list_clean_epoch = root_state->element_list_dirty_epoch;
  root_state->element_list_dirty_flag = 0;
  root_state->element_list_clean_parent_version = 0;
  root_state->element_list_version = 1;
}

void LLVMRuntime_initialize_thread_pool(LLVMRuntime *runtime,
                                        void *thread_pool,
                                        void *parallel_for) {
  runtime->thread_pool = (Ptr)thread_pool;
  runtime->parallel_for = (parallel_for_type)parallel_for;
}

void runtime_NodeAllocator_initialize(LLVMRuntime *runtime,
                                      int tree_id,
                                      int local_id,
                                      std::size_t node_size) {
  auto *state = snode_runtime_state(runtime, tree_id, local_id);
#if !ARCH_cuda && !ARCH_amdgpu
  state->node_allocator = acquire_node_manager(runtime, node_size, 1024 * 16);
#else
  state->node_allocator =
      runtime->create<NodeManager>(runtime, node_size, 1024 * 16);
#endif
}

// Phase 1-D (2026-05): initialize a NodeAllocator with a custom
// chunk_num_elements. When num_cells_per_container is small, a
// reduced chunk size dramatically cuts per-SNode pool VRAM (e.g.
// 112.5 MiB → 14.7 MiB for MPM pointer). The NodeManager ctor
// still applies its 128 MiB ceiling halving as a safety clamp.
void runtime_NodeAllocator_initialize_ex(LLVMRuntime *runtime,
                                          int tree_id,
                                          int local_id,
                                          std::size_t node_size,
                                          int chunk_num_elements) {
  auto *state = snode_runtime_state(runtime, tree_id, local_id);
#if !ARCH_cuda && !ARCH_amdgpu
  state->node_allocator =
      acquire_node_manager(runtime, node_size, chunk_num_elements);
#else
  state->node_allocator =
      runtime->create<NodeManager>(runtime, node_size, chunk_num_elements);
#endif
}

void runtime_allocate_ambient_direct(LLVMRuntime *runtime,
                                     int tree_id,
                                     int local_id,
                                     std::size_t node_size) {
  auto *state = snode_runtime_state(runtime, tree_id, local_id);
#if !ARCH_cuda && !ARCH_amdgpu
  auto ambient = acquire_direct_ambient(runtime, node_size);
#else
  auto ambient = runtime->allocate_aligned(runtime->runtime_memory_chunk,
                                          node_size, 8, true /*request*/);
#endif
  std::memset(ambient, 0, node_size);
  state->ambient_element = ambient;
}

// Phase 1 (2026-05): assign a dedicated bump region (carved from the global
// pool buffer) to an already-initialized NodeManager. ptr+size define the
// PreallocatedMemoryChunk sub-range. All 3 ListManagers (free/recycled/data)
// will route their data chunk allocations to this region instead of
// runtime->runtime_memory_chunk.
void runtime_NodeAllocator_set_dedicated_pool(LLVMRuntime *runtime,
                                              int tree_id,
                                              int local_id,
                                              Ptr ptr,
                                              std::size_t size) {
  snode_runtime_state(runtime, tree_id, local_id)
      ->node_allocator->set_dedicated_pool(ptr, size);
}

void runtime_NodeAllocator_set_deterministic_capacity(LLVMRuntime *runtime,
                                                      int tree_id,
                                                      int local_id,
                                                      int capacity) {
  auto allocator =
      snode_runtime_state(runtime, tree_id, local_id)->node_allocator;
  allocator->deterministic_capacity = capacity;
  allocator->deterministic_active = 0;
  allocator->deterministic_peak = 0;
}

// CUDA auto-sized per-SNode pools are allocated per materialized sparse
// SNodeTree. `runtime_memory_chunk` is rebound to the next tree's global
// region when another sparse tree is materialized, so element lists from the
// current tree must not keep allocating future chunks from that mutable global
// runtime pointer. Snapshot the remaining global region into a stable chunk and
// route this tree's element-list chunk allocations there.
void runtime_element_lists_prepare_backing_pool(LLVMRuntime *runtime) {
  if (runtime->runtime_memory_chunk.preallocated_size == 0) {
    runtime->materializing_element_list_backing_chunk = nullptr;
    return;
  }

  auto backing = runtime->create<PreallocatedMemoryChunk>();
  *backing = runtime->runtime_memory_chunk;
  runtime->materializing_element_list_backing_chunk = backing;
}

void runtime_element_list_set_backing_pool(LLVMRuntime *runtime,
                                           int tree_id,
                                           int local_id) {
  auto *state = snode_runtime_state(runtime, tree_id, local_id);
  if (runtime->materializing_element_list_backing_chunk != nullptr &&
      state->element_list != nullptr) {
    state->element_list->backing_chunk =
        runtime->materializing_element_list_backing_chunk;
  }
}

void runtime_element_lists_finalize_backing_pool(LLVMRuntime *runtime) {
  if (runtime->materializing_element_list_backing_chunk == nullptr) {
    return;
  }
  // Prevent accidental fallback allocations from overlapping with the
  // tree-owned element-list region. The host will rebind runtime_memory_chunk
  // before materializing the next CUDA sparse tree.
  runtime->runtime_memory_chunk.preallocated_head =
      runtime->runtime_memory_chunk.preallocated_tail;
  runtime->materializing_element_list_backing_chunk = nullptr;
}

void mutex_lock_i32(Ptr mutex) {
  while (atomic_exchange_i32((i32 *)mutex, 1) == 1)
    ;
}

void mutex_unlock_i32(Ptr mutex) {
  atomic_exchange_i32((i32 *)mutex, 0);
}

int32 ctlz_i32(i32 val) {
  return 0;
}

int32 cttz_i32(i32 val) {
  return 0;
}

int32 cuda_compute_capability() {
  return 0;
}

int32 cuda_ballot(bool bit) {
  return 0;
}

i32 cuda_shfl_down_sync_i32(u32 mask, i32 val, i32 delta, int width) {
  return 0;
}

i32 cuda_shfl_down_i32(i32 delta, i32 val, int width) {
  return 0;
}

f32 cuda_shfl_down_sync_f32(u32 mask, f32 val, i32 delta, int width) {
  return 0;
}

f32 cuda_shfl_down_f32(i32 delta, f32 val, int width) {
  return 0;
}

i32 cuda_shfl_xor_sync_i32(u32 mask, i32 val, i32 delta, int width) {
  return 0;
}

i32 cuda_shfl_up_sync_i32(u32 mask, i32 val, i32 delta, int width) {
  return 0;
}

f32 cuda_shfl_up_sync_f32(u32 mask, f32 val, i32 delta, int width) {
  return 0;
}

i32 cuda_shfl_sync_i32(u32 mask, i32 val, i32 delta, int width) {
  return 0;
}

f32 cuda_shfl_sync_f32(u32 mask, f32 val, i32 delta, int width) {
  return 0;
}

bool cuda_all_sync(u32 mask, bool bit) {
  return false;
}

int32 cuda_all_sync_i32(u32 mask, int32 predicate) {
  return (int32)cuda_all_sync(mask, (bool)predicate);
}

bool cuda_any_sync(u32 mask, bool bit) {
  return false;
}

int32 cuda_any_sync_i32(u32 mask, int32 predicate) {
  return (int32)cuda_any_sync(mask, (bool)predicate);
}

bool cuda_uni_sync(u32 mask, bool bit) {
  return false;
}

int32 cuda_uni_sync_i32(u32 mask, int32 predicate) {
  return (int32)cuda_uni_sync(mask, (bool)predicate);
}

int32 cuda_ballot_sync(int32 mask, bool bit) {
  return 0;
}

int32 cuda_ballot_i32(int32 predicate) {
  return cuda_ballot_sync(UINT32_MAX, (bool)predicate);
}

int32 cuda_ballot_sync_i32(u32 mask, int32 predicate) {
  return cuda_ballot_sync(mask, (bool)predicate);
}

uint32 cuda_match_any_sync_i32(u32 mask, i32 value) {
  return 0;
}

u32 cuda_match_all_sync_i32(u32 mask, i32 value) {
#if ARCH_cuda
  u32 ret;
  asm volatile("match.all.sync.b32  %0, %1, %2;"
               : "=r"(ret)
               : "r"(value), "r"(mask));
  return ret;
#else
  return 0;
#endif
}

uint32 cuda_match_any_sync_i64(u32 mask, i64 value) {
#if ARCH_cuda
  u32 ret;
  asm volatile("match.any.sync.b64  %0, %1, %2;"
               : "=r"(ret)
               : "l"(value), "r"(mask));
  return ret;
#else
  return 0;
#endif
}

#if ARCH_cuda
uint32 cuda_active_mask() {
  unsigned int mask;
  asm volatile("activemask.b32 %0;" : "=r"(mask));
  return mask;
}
#else
uint32 cuda_active_mask() {
  return 0;
}
#endif

void block_barrier() {
}

int32 block_barrier_and_i32(int32 predicate) {
  return 0;
}

int32 block_barrier_or_i32(int32 predicate) {
  return 0;
}

int32 block_barrier_count_i32(int32 predicate) {
  return 0;
}

void warp_barrier(uint32 mask) {
}

void block_memfence() {
}

void grid_memfence() {
}

// CUDA 12.4 device-updatable Graph nodes expose device-callable setters. The
// host uploads only handles belonging to this Graph executable, and Forge's
// producer-owned control record supplies a capacity-clamped grid. Updating
// every node on every launch is required because these attributes persist
// across Graph launches rather than returning to their captured baseline.
#if ARCH_cuda
struct CudaGraphDim3 {
  u32 x;
  u32 y;
  u32 z;
};

struct CudaGraphKernelNodeUpdate {
  void *node;
  u32 field;
  union {
    CudaGraphDim3 grid_dim;
    struct {
      const void *value;
      u64 offset;
      u64 size;
    } parameter;
    u32 enabled;
  } data;
};

static_assert(sizeof(CudaGraphKernelNodeUpdate) == 40);
static_assert(offsetof(CudaGraphKernelNodeUpdate, node) == 0);
static_assert(offsetof(CudaGraphKernelNodeUpdate, field) == 8);
static_assert(offsetof(CudaGraphKernelNodeUpdate, data) == 16);

extern "C" i32 cudaGraphKernelNodeUpdatesApply(
    const CudaGraphKernelNodeUpdate *updates,
    u64 update_count);
#endif

i32 cuda_graph_update_bounded_group(u64 nodes_address,
                                    u32 node_count,
                                    u32 grid_x,
                                    u32 enabled) {
#if ARCH_cuda
  auto *nodes = reinterpret_cast<u64 *>(nodes_address);
  for (u32 index = 0; index < node_count; ++index) {
    void *node = reinterpret_cast<void *>(nodes[index]);
    CudaGraphKernelNodeUpdate updates[2]{};
    updates[0].node = node;
    updates[0].field = 3;  // cudaGraphKernelNodeFieldEnabled
    updates[0].data.enabled = enabled != 0 ? 1 : 0;
    u64 update_count = 1;
    if (enabled != 0) {
      updates[1].node = node;
      updates[1].field = 1;  // cudaGraphKernelNodeFieldGridDim
      updates[1].data.grid_dim = {grid_x, 1, 1};
      update_count = 2;
    }
    const i32 status =
        cudaGraphKernelNodeUpdatesApply(updates, update_count);
    if (status != 0) {
      return status;
    }
  }
  return 0;
#else
  (void)nodes_address;
  (void)node_count;
  (void)grid_x;
  (void)enabled;
  return -1;
#endif
}

// these trivial functions are needed by the DEFINE_REDUCTION macro
i32 op_add_i32(i32 a, i32 b) {
  return a + b;
}
f32 op_add_f32(f32 a, f32 b) {
  return a + b;
}

i32 op_min_i32(i32 a, i32 b) {
  return std::min(a, b);
}
f32 op_min_f32(f32 a, f32 b) {
  return std::min(a, b);
}

i32 op_max_i32(i32 a, i32 b) {
  return std::max(a, b);
}
f32 op_max_f32(f32 a, f32 b) {
  return std::max(a, b);
}

i32 op_and_i32(i32 a, i32 b) {
  return a & b;
}
i32 op_or_i32(i32 a, i32 b) {
  return a | b;
}
i32 op_xor_i32(i32 a, i32 b) {
  return a ^ b;
}

#define DEFINE_REDUCTION(op, dtype)                                    \
  dtype warp_reduce_##op##_##dtype(uint32_t mask, dtype val) {         \
    for (int offset = 16; offset > 0; offset /= 2)                     \
      val = op_##op##_##dtype(                                         \
          val, cuda_shfl_down_sync_##dtype(mask, val, offset, 31));    \
    return val;                                                        \
  }                                                                    \
  dtype reduce_##op##_##dtype(dtype *result, dtype val) {              \
    uint32_t mask = cuda_active_mask();                                \
    if (mask != 0xFFFFFFFF) {                                          \
      atomic_##op##_##dtype(result, val);                              \
    } else {                                                           \
      dtype warp_result = warp_reduce_##op##_##dtype(0xFFFFFFFF, val); \
      if ((thread_idx() & (warp_size() - 1)) == 0) {                   \
        atomic_##op##_##dtype(result, warp_result);                    \
      }                                                                \
    }                                                                  \
    return val;                                                        \
  }

DEFINE_REDUCTION(add, i32);
DEFINE_REDUCTION(add, f32);

DEFINE_REDUCTION(min, i32);
DEFINE_REDUCTION(min, f32);

DEFINE_REDUCTION(max, i32);
DEFINE_REDUCTION(max, f32);

DEFINE_REDUCTION(and, i32);
DEFINE_REDUCTION(or, i32);
DEFINE_REDUCTION(xor, i32);

// "Element", "component" are different concepts

void clear_list(LLVMRuntime *runtime, StructMeta *parent, StructMeta *child) {
  if (child->listgen_reuse && element_list_is_current(runtime, parent, child)) {
    return;
  }
  auto child_list = snode_runtime_state(runtime, child)->element_list;
  child_list->clear();
}

/*
 * The element list of a SNode, maintains pointers to its instances, and
 * instances' parents' coordinates
 */

// For the root node there is only one container,
// therefore we use a special kernel for more parallelism.
extern "C++" {
template <bool RecordWork>
void record_sparse_listgen_work(LLVMRuntime *runtime,
                                uint64 scanned_elements,
                                uint64 emitted_elements,
                                bool reused = false) {
#if !ARCH_cuda && !ARCH_amdgpu
  if constexpr (RecordWork) {
    runtime->sparse_listgen_work_available = true;
    runtime->sparse_listgen_scanned_elements += scanned_elements;
    runtime->sparse_listgen_emitted_elements += emitted_elements;
    runtime->sparse_listgen_execution_strategy = reused ? 0 : 1;
    runtime->sparse_listgen_reused = reused;
  }
#else
  (void)runtime;
  (void)scanned_elements;
  (void)emitted_elements;
  (void)reused;
#endif
}

template <bool RecordWork>
void element_listgen_root_impl(LLVMRuntime *runtime,
                               StructMeta *parent,
                               StructMeta *child) {
  if (child->listgen_reuse && element_list_is_current(runtime, parent, child)) {
    record_sparse_listgen_work<RecordWork>(runtime, 0, 0, true);
    return;
  }
  // If there's just one element in the parent list, we need to use the blocks
  // (instead of threads) to split the parent container
  auto parent_list = snode_runtime_state(runtime, parent)->element_list;
  auto child_list = snode_runtime_state(runtime, child)->element_list;
  int child_list_size_before = 0;
  if constexpr (RecordWork) {
    child_list_size_before = child_list->size();
  }
  // Cache the func pointers here for better compiler optimization
  auto parent_lookup_element = parent->lookup_element;
  auto child_get_num_elements = child->get_num_elements;
  auto child_from_parent_element = child->from_parent_element;
#if ARCH_cuda || ARCH_amdgpu
  // All blocks share the only root container, which has only one child
  // container.
  // Each thread processes a subset of the child container for more parallelism.
  int c_start = block_dim() * block_idx() + thread_idx();
  int c_step = grid_dim() * block_dim();
#else
  int c_start = 0;
  int c_step = 1;
#endif
  // Note that the root node has only one container, and the `element`
  // representing that single container has only one 'child':
  // element.loop_bounds[0] = 0 and element.loop_bounds[1] = 1
  // Therefore, compared with element_listgen_nonroot,
  // we need neither `i` to loop over the `elements`, nor `j` to
  // loop over the children.

  auto element = parent_list->get<Element>(0);

  auto ch_element = parent_lookup_element((Ptr)parent, element.element, 0);
  ch_element = child_from_parent_element((Ptr)ch_element);
  auto ch_num_elements = child_get_num_elements((Ptr)child, ch_element);
  auto ch_element_size =
      std::min(ch_num_elements, taichi_listgen_max_element_size);

  // Here is a grid-stride loop.
  for (int c = c_start; c * ch_element_size < ch_num_elements; c += c_step) {
    Element elem;
    elem.element = ch_element;
    elem.loop_bounds[0] = c * ch_element_size;
    elem.loop_bounds[1] = std::min((c + 1) * ch_element_size, ch_num_elements);
    // There is no need to refine coordinates for root listgen, since its
    // num_bits is always zero
    elem.pcoord = element.pcoord;
    child_list->append(&elem);
  }
  if (child->listgen_reuse) {
    mark_element_list_current(runtime, parent, child);
  }
  if constexpr (RecordWork) {
    record_sparse_listgen_work<RecordWork>(
        runtime, static_cast<uint64>(ch_num_elements),
        static_cast<uint64>(child_list->size() - child_list_size_before));
  }
}
}  // extern "C++"

void element_listgen_root(LLVMRuntime *runtime,
                          StructMeta *parent,
                          StructMeta *child) {
#if !ARCH_cuda && !ARCH_amdgpu
  if (runtime->sparse_listgen_work_recording) {
    element_listgen_root_impl<true>(runtime, parent, child);
    return;
  }
#endif
  element_listgen_root_impl<false>(runtime, parent, child);
}

#if !ARCH_cuda && !ARCH_amdgpu
extern "C++" {
constexpr i32 kCpuParallelListgenMinParentElements = 64;
constexpr uint64 kCpuParallelListgenMinCandidateSlots = 65536;
constexpr i32 kCpuParallelListgenOffsetChunkElements = 1024;
constexpr std::size_t kCpuParallelListgenMaxWorkspaceBytes = 64 * 1024 * 1024;

struct CpuParallelListgenContext {
  StructMeta *parent;
  StructMeta *child;
  ListManager *parent_list;
  ListManager *child_list;
  ListManager *output_offsets;
};

void cpu_parallel_listgen_count(void *context_, int thread_id, int i) {
  (void)thread_id;
  auto context = (CpuParallelListgenContext *)context_;
  auto element = context->parent_list->get<Element>(i);
  i32 output_count = 0;
  for (int j = element.loop_bounds[0]; j < element.loop_bounds[1]; ++j) {
    if (!context->parent->is_active((Ptr)context->parent, element.element, j)) {
      continue;
    }
    auto ch_element = context->parent->lookup_element(
        (Ptr)context->parent, element.element, j);
    ch_element = context->child->from_parent_element((Ptr)ch_element);
    const i32 ch_num_elements =
        context->child->get_num_elements((Ptr)context->child, ch_element);
    if (ch_num_elements > 0) {
      output_count +=
          1 + (ch_num_elements - 1) / taichi_listgen_max_element_size;
    }
  }
  context->output_offsets->get<i32>(i + 1) = output_count;
}

void cpu_parallel_listgen_fill(void *context_, int thread_id, int i) {
  (void)thread_id;
  auto context = (CpuParallelListgenContext *)context_;
  auto element = context->parent_list->get<Element>(i);
  i32 output_index = context->output_offsets->get<i32>(i);
  const i32 output_end = context->output_offsets->get<i32>(i + 1);
  for (int j = element.loop_bounds[0]; j < element.loop_bounds[1]; ++j) {
    PhysicalCoordinates refined_coord;
    context->parent->refine_coordinates(&element.pcoord, &refined_coord, j);
    if (!context->parent->is_active((Ptr)context->parent, element.element, j)) {
      continue;
    }
    auto ch_element = context->parent->lookup_element(
        (Ptr)context->parent, element.element, j);
    ch_element = context->child->from_parent_element((Ptr)ch_element);
    const i32 ch_num_elements =
        context->child->get_num_elements((Ptr)context->child, ch_element);
    const i32 ch_element_size =
        std::min(ch_num_elements, taichi_listgen_max_element_size);
    for (int ch_lower = 0; ch_lower < ch_num_elements;
         ch_lower += ch_element_size) {
      auto &output = context->child_list->get<Element>(output_index++);
      output.element = ch_element;
      output.loop_bounds[0] = ch_lower;
      output.loop_bounds[1] =
          std::min(ch_lower + ch_element_size, ch_num_elements);
      output.pcoord = refined_coord;
    }
  }
  taichi_assert_runtime(context->child_list->runtime,
                        output_index == output_end,
                        "Parallel listgen count/fill mismatch.");
}

bool element_listgen_nonroot_parallel(LLVMRuntime *runtime,
                                      StructMeta *parent,
                                      StructMeta *child,
                                      uint64 *scanned_elements,
                                      uint64 *emitted_elements) {
  auto parent_list = snode_runtime_state(runtime, parent)->element_list;
  const i32 num_parent_elements = parent_list->size();
  if (runtime->parallel_for == nullptr || runtime->thread_pool == nullptr ||
      runtime->num_rand_states <= 1 ||
      num_parent_elements < kCpuParallelListgenMinParentElements) {
    return false;
  }
  const std::size_t num_offsets =
      static_cast<std::size_t>(num_parent_elements) + 1;
  if (num_offsets >
      kCpuParallelListgenMaxWorkspaceBytes / sizeof(i32)) {
    return false;
  }

  uint64 candidates = 0;
  for (int i = 0; i < num_parent_elements; ++i) {
    const auto &element = parent_list->get<Element>(i);
    candidates +=
        static_cast<uint64>(element.loop_bounds[1] - element.loop_bounds[0]);
  }
  if (candidates < kCpuParallelListgenMinCandidateSlots) {
    return false;
  }

  if (runtime->cpu_parallel_listgen_offsets == nullptr) {
    runtime->cpu_parallel_listgen_offsets = runtime->create<ListManager>(
        runtime, sizeof(i32), kCpuParallelListgenOffsetChunkElements);
  }
  auto output_offsets = runtime->cpu_parallel_listgen_offsets;
  const i32 last_offset_chunk =
      (static_cast<i32>(num_offsets) - 1) >>
      output_offsets->log2chunk_num_elements;
  for (int chunk_id = 0; chunk_id <= last_offset_chunk; ++chunk_id) {
    output_offsets->touch_chunk(chunk_id);
  }
  output_offsets->resize(static_cast<i32>(num_offsets));
  output_offsets->get<i32>(0) =
      snode_runtime_state(runtime, child)->element_list->size();

  CpuParallelListgenContext context{
      parent,
      child,
      parent_list,
      snode_runtime_state(runtime, child)->element_list,
      output_offsets,
  };
  runtime->parallel_for(runtime->thread_pool, num_parent_elements,
                        runtime->num_rand_states, &context,
                        cpu_parallel_listgen_count);

  bool overflow = false;
  for (int i = 0; i < num_parent_elements; ++i) {
    const uint64 next =
        static_cast<uint64>(output_offsets->get<i32>(i)) +
        static_cast<uint64>(output_offsets->get<i32>(i + 1));
    if (next > 0x7fffffffULL) {
      overflow = true;
      break;
    }
    output_offsets->get<i32>(i + 1) = static_cast<i32>(next);
  }
  if (overflow) {
    taichi_assert_runtime(runtime, false,
                          "Parallel listgen output exceeds i32 capacity.");
    return true;
  }

  auto child_list = context.child_list;
  const i32 output_begin = output_offsets->get<i32>(0);
  const i32 output_end = output_offsets->get<i32>(num_parent_elements);
  if (output_end > output_begin) {
    const i32 first_chunk = output_begin >> child_list->log2chunk_num_elements;
    const i32 last_chunk =
        (output_end - 1) >> child_list->log2chunk_num_elements;
    for (int chunk_id = first_chunk; chunk_id <= last_chunk; ++chunk_id) {
      child_list->touch_chunk(chunk_id);
    }
  }
  child_list->resize(output_end);
  runtime->parallel_for(runtime->thread_pool, num_parent_elements,
                        runtime->num_rand_states, &context,
                        cpu_parallel_listgen_fill);

  *scanned_elements = candidates;
  *emitted_elements = static_cast<uint64>(output_end - output_begin);
  return true;
}
}  // extern "C++"
#endif

extern "C++" {
template <bool RecordWork>
void element_listgen_nonroot_impl(LLVMRuntime *runtime,
                                  StructMeta *parent,
                                  StructMeta *child) {
  if (child->listgen_reuse && element_list_is_current(runtime, parent, child)) {
    record_sparse_listgen_work<RecordWork>(runtime, 0, 0, true);
    return;
  }
  auto parent_list = snode_runtime_state(runtime, parent)->element_list;
  int num_parent_elements = parent_list->size();
  auto child_list = snode_runtime_state(runtime, child)->element_list;
  int child_list_size_before = 0;
  uint64 scanned_elements = 0;
  if constexpr (RecordWork) {
    child_list_size_before = child_list->size();
  }
  // Cache the func pointers here for better compiler optimization
  auto parent_refine_coordinates = parent->refine_coordinates;
  auto parent_is_active = parent->is_active;
  auto parent_lookup_element = parent->lookup_element;
  auto child_get_num_elements = child->get_num_elements;
  auto child_from_parent_element = child->from_parent_element;
#if ARCH_cuda || ARCH_amdgpu
  // Each block processes a slice of a parent container
  int i_start = block_idx();
  int i_step = grid_dim();
  // Each thread processes an element of the parent container
  int j_start = thread_idx();
  int j_step = block_dim();
#else
  int i_start = 0;
  int i_step = 1;
  int j_start = 0;
  int j_step = 1;
#endif
  for (int i = i_start; i < num_parent_elements; i += i_step) {
    auto element = parent_list->get<Element>(i);
    int j_lower = element.loop_bounds[0] + j_start;
    int j_higher = element.loop_bounds[1];
    if constexpr (RecordWork) {
      scanned_elements += static_cast<uint64>(j_higher - j_lower);
    }
    for (int j = j_lower; j < j_higher; j += j_step) {
      PhysicalCoordinates refined_coord;
      parent_refine_coordinates(&element.pcoord, &refined_coord, j);
      if (parent_is_active((Ptr)parent, element.element, j)) {
        auto ch_element =
            parent_lookup_element((Ptr)parent, element.element, j);
        ch_element = child_from_parent_element((Ptr)ch_element);
        auto ch_num_elements = child_get_num_elements((Ptr)child, ch_element);
        auto ch_element_size =
            std::min(ch_num_elements, taichi_listgen_max_element_size);
        for (int ch_lower = 0; ch_lower < ch_num_elements;
             ch_lower += ch_element_size) {
          Element elem;
          elem.element = ch_element;
          elem.loop_bounds[0] = ch_lower;
          elem.loop_bounds[1] =
              std::min(ch_lower + ch_element_size, ch_num_elements);
          elem.pcoord = refined_coord;
          child_list->append(&elem);
        }
      }
    }
  }
  if (child->listgen_reuse) {
    mark_element_list_current(runtime, parent, child);
  }
  if constexpr (RecordWork) {
    record_sparse_listgen_work<RecordWork>(
        runtime, scanned_elements,
        static_cast<uint64>(child_list->size() - child_list_size_before));
  }
}
}  // extern "C++"

void element_listgen_nonroot(LLVMRuntime *runtime,
                             StructMeta *parent,
                             StructMeta *child) {
#if !ARCH_cuda && !ARCH_amdgpu
  if (child->listgen_reuse && element_list_is_current(runtime, parent, child)) {
    if (runtime->sparse_listgen_work_recording) {
      record_sparse_listgen_work<true>(runtime, 0, 0, true);
    }
    return;
  }
  uint64 scanned_elements = 0;
  uint64 emitted_elements = 0;
  if (element_listgen_nonroot_parallel(runtime, parent, child,
                                       &scanned_elements,
                                       &emitted_elements)) {
    if (child->listgen_reuse) {
      mark_element_list_current(runtime, parent, child);
    }
    if (runtime->sparse_listgen_work_recording) {
      runtime->sparse_listgen_work_available = true;
      runtime->sparse_listgen_scanned_elements += scanned_elements;
      runtime->sparse_listgen_emitted_elements += emitted_elements;
      runtime->sparse_listgen_execution_strategy = 2;
    }
    return;
  }
  if (runtime->sparse_listgen_work_recording) {
    element_listgen_nonroot_impl<true>(runtime, parent, child);
    return;
  }
#endif
  element_listgen_nonroot_impl<false>(runtime, parent, child);
}

using BlockTask = void(RuntimeContext *, char *, Element *, int, int);

struct cpu_block_task_helper_context {
  RuntimeContext *context;
  BlockTask *task;
  ListManager *list;
  int element_size;
  int element_split;
  int element_batch_size;
  std::size_t tls_buffer_size;
};

// TODO: To enforce inlining, we need to create in LLVM a new function that
// calls block_helper and the BLS xlogues, and pass that function to the
// scheduler.

// TODO: TLS should be directly passed to the scheduler, so that it lives
// with the threads (instead of blocks).

void cpu_struct_for_block_helper(void *ctx_, int thread_id, int i) {
  auto ctx = (cpu_block_task_helper_context *)(ctx_);
  if (ctx->element_batch_size > 1) {
    int element_begin = i * ctx->element_batch_size;
    int element_end = std::min(element_begin + ctx->element_batch_size,
                               ctx->list->size());
    alignas(8) char tls_buffer[ctx->tls_buffer_size];
    RuntimeContext this_thread_context = *ctx->context;
    this_thread_context.cpu_thread_id = thread_id;
    for (int element_id = element_begin; element_id < element_end;
         element_id++) {
      auto &e = ctx->list->get<Element>(element_id);
      int lower = e.loop_bounds[0];
      int upper = e.loop_bounds[1];
      if (lower < upper) {
        (*ctx->task)(&this_thread_context, tls_buffer, &e, lower, upper);
      }
    }
    return;
  }
  int element_id = i / ctx->element_split;
  int part_size = ctx->element_size / ctx->element_split;
  int part_id = i % ctx->element_split;
  auto &e = ctx->list->get<Element>(element_id);
  int lower = e.loop_bounds[0] + part_id * part_size;
  int upper = e.loop_bounds[0] + (part_id + 1) * part_size;
  upper = std::min(upper, e.loop_bounds[1]);
  alignas(8) char tls_buffer[ctx->tls_buffer_size];

  RuntimeContext this_thread_context = *ctx->context;
  this_thread_context.cpu_thread_id = thread_id;
  if (lower < upper) {
    (*ctx->task)(&this_thread_context, tls_buffer,
                 &ctx->list->get<Element>(element_id), lower, upper);
  }
}

void parallel_struct_for(RuntimeContext *context,
                         uint64 snode_runtime_key,
                         int element_size,
                         int element_split,
                         BlockTask *task,
                         std::size_t tls_buffer_size,
                         int num_threads) {
  auto list =
      snode_runtime_state(context->runtime, snode_runtime_key)->element_list;
  auto list_tail = list->size();
#if ARCH_cuda || ARCH_amdgpu
  int i = block_idx();
  // Note: CUDA requires compile-time constant local array sizes.
  // We use "1" here and modify it during codegen to tls_buffer_size.
  alignas(8) char tls_buffer[1];
  // TODO: refactor element_split more systematically.
  element_split = 1;
  const auto part_size = element_size / element_split;
  while (true) {
    int element_id = i / element_split;
    if (element_id >= list_tail)
      break;
    auto part_id = i % element_split;
    auto &e = list->get<Element>(element_id);
    int lower = e.loop_bounds[0] + part_id * part_size;
    int upper = e.loop_bounds[0] + (part_id + 1) * part_size;
    upper = std::min(upper, e.loop_bounds[1]);
    if (lower < upper)
      task(context, tls_buffer, &list->get<Element>(element_id), lower, upper);
    i += grid_dim();
  }
#else
  cpu_block_task_helper_context ctx;
  ctx.context = context;
  ctx.task = task;
  ctx.list = list;
  ctx.element_size = element_size;
  ctx.element_split = element_split;
  ctx.tls_buffer_size = tls_buffer_size;
  ctx.element_batch_size = 1;
  int task_count = list_tail * element_split;
  if (element_size == 1 && element_split == 1) {
    const int effective_num_threads = std::max(1, num_threads);
    constexpr int kSingletonElementBatchSize = 32;
    if (list_tail > effective_num_threads * kSingletonElementBatchSize) {
      ctx.element_batch_size = kSingletonElementBatchSize;
      task_count =
          (list_tail + ctx.element_batch_size - 1) / ctx.element_batch_size;
    }
  }
  auto runtime = context->runtime;
  runtime->parallel_for(runtime->thread_pool, task_count, num_threads, &ctx,
                        cpu_struct_for_block_helper);
#endif
}

using range_for_xlogue = void (*)(RuntimeContext *, /*TLS*/ char *tls_base);
using mesh_for_xlogue = void (*)(RuntimeContext *,
                                 /*TLS*/ char *tls_base,
                                 uint32_t patch_idx);

struct range_task_helper_context {
  RuntimeContext *context;
  range_for_xlogue prologue{nullptr};
  RangeForTaskFunc *body{nullptr};
  range_for_xlogue epilogue{nullptr};
  std::size_t tls_size{1};
  int begin;
  int end;
  int block_size;
  int step;
};

void cpu_parallel_range_for_task(void *range_context,
                                 int thread_id,
                                 int task_id) {
  auto ctx = *(range_task_helper_context *)range_context;
  alignas(8) char tls_buffer[ctx.tls_size];
  auto tls_ptr = &tls_buffer[0];
  if (ctx.prologue)
    ctx.prologue(ctx.context, tls_ptr);

  RuntimeContext this_thread_context = *ctx.context;
  this_thread_context.cpu_thread_id = thread_id;
  if (ctx.step == 1) {
    int block_start = ctx.begin + task_id * ctx.block_size;
    int block_end = std::min(block_start + ctx.block_size, ctx.end);
    for (int i = block_start; i < block_end; i++) {
      ctx.body(&this_thread_context, tls_ptr, i);
    }
  } else if (ctx.step == -1) {
    int block_start = ctx.end - task_id * ctx.block_size;
    int block_end = std::max(ctx.begin, block_start * ctx.block_size);
    for (int i = block_start - 1; i >= block_end; i--) {
      ctx.body(&this_thread_context, tls_ptr, i);
    }
  }
  if (ctx.epilogue)
    ctx.epilogue(ctx.context, tls_ptr);
}

void cpu_parallel_range_for_cancellable_task(void *range_context,
                                             int thread_id,
                                             int task_id) {
  auto ctx = *(range_task_helper_context *)range_context;
  if (runtime_has_error(ctx.context->runtime)) {
    return;
  }

  alignas(8) char tls_buffer[ctx.tls_size];
  auto tls_ptr = &tls_buffer[0];
  if (ctx.prologue) {
    ctx.prologue(ctx.context, tls_ptr);
  }

  RuntimeContext this_thread_context = *ctx.context;
  this_thread_context.cpu_thread_id = thread_id;
  if (ctx.step == 1) {
    int block_start = ctx.begin + task_id * ctx.block_size;
    int block_end = std::min(block_start + ctx.block_size, ctx.end);
    for (int i = block_start;
         i < block_end && !runtime_has_error(ctx.context->runtime); i++) {
      ctx.body(&this_thread_context, tls_ptr, i);
    }
  } else if (ctx.step == -1) {
    int block_start = ctx.end - task_id * ctx.block_size;
    int block_end = std::max(ctx.begin, block_start * ctx.block_size);
    for (int i = block_start - 1;
         i >= block_end && !runtime_has_error(ctx.context->runtime); i--) {
      ctx.body(&this_thread_context, tls_ptr, i);
    }
  }
  if (ctx.epilogue && !runtime_has_error(ctx.context->runtime)) {
    ctx.epilogue(ctx.context, tls_ptr);
  }
}

void cpu_parallel_range_for(RuntimeContext *context,
                            int num_threads,
                            int begin,
                            int end,
                            int step,
                            int block_dim,
                            range_for_xlogue prologue,
                            RangeForTaskFunc *body,
                            range_for_xlogue epilogue,
                            std::size_t tls_size) {
  range_task_helper_context ctx;
  ctx.context = context;
  ctx.prologue = prologue;
  ctx.tls_size = tls_size;
  ctx.body = body;
  ctx.epilogue = epilogue;
  ctx.begin = begin;
  ctx.end = end;
  ctx.step = step;
  if (step != 1 && step != -1) {
    taichi_printf(context->runtime, "step must not be %d\n", step);
    exit(-1);
  }
  ctx.block_size = block_dim;
  auto runtime = context->runtime;
  runtime->parallel_for(runtime->thread_pool,
                        (end - begin + block_dim - 1) / block_dim, num_threads,
                        &ctx, cpu_parallel_range_for_task);
}

void cpu_parallel_range_for_cancellable(RuntimeContext *context,
                                        int num_threads,
                                        int begin,
                                        int end,
                                        int step,
                                        int block_dim,
                                        range_for_xlogue prologue,
                                        RangeForTaskFunc *body,
                                        range_for_xlogue epilogue,
                                        std::size_t tls_size) {
  range_task_helper_context ctx;
  ctx.context = context;
  ctx.prologue = prologue;
  ctx.tls_size = tls_size;
  ctx.body = body;
  ctx.epilogue = epilogue;
  ctx.begin = begin;
  ctx.end = end;
  ctx.step = step;
  if (step != 1 && step != -1) {
    taichi_printf(context->runtime, "step must not be %d\n", step);
    exit(-1);
  }
  ctx.block_size = block_dim;
  auto runtime = context->runtime;
  runtime->parallel_for(runtime->thread_pool,
                        (end - begin + block_dim - 1) / block_dim, num_threads,
                        &ctx, cpu_parallel_range_for_cancellable_task);
}

int cpu_bounded_range_end(RuntimeContext *context, int begin, int end) {
  auto binding = (CpuBoundedRangeBinding *)(
      (char *)context->arg_buffer - sizeof(CpuBoundedRangeBinding));
  auto extent = (i32 *)binding->extent;
  i32 count = extent[0];
  if (count < 0) {
    count = 0;
    extent[1] = 1;
  } else if (count > binding->capacity) {
    count = binding->capacity;
    extent[1] = 1;
  }
  extent[0] = count;
  const std::int64_t bounded_end =
      static_cast<std::int64_t>(begin) + static_cast<std::int64_t>(count);
  return bounded_end < static_cast<std::int64_t>(end)
             ? static_cast<int>(bounded_end)
             : end;
}

void cpu_parallel_range_for_bounded(RuntimeContext *context,
                                    int num_threads,
                                    int begin,
                                    int end,
                                    int step,
                                    int block_dim,
                                    range_for_xlogue prologue,
                                    RangeForTaskFunc *body,
                                    range_for_xlogue epilogue,
                                    std::size_t tls_size) {
  cpu_parallel_range_for(context, num_threads, begin,
                         cpu_bounded_range_end(context, begin, end), step,
                         block_dim, prologue, body, epilogue, tls_size);
}

void cpu_parallel_range_for_bounded_cancellable(
    RuntimeContext *context,
    int num_threads,
    int begin,
    int end,
    int step,
    int block_dim,
    range_for_xlogue prologue,
    RangeForTaskFunc *body,
    range_for_xlogue epilogue,
    std::size_t tls_size) {
  cpu_parallel_range_for_cancellable(
      context, num_threads, begin,
      cpu_bounded_range_end(context, begin, end), step, block_dim, prologue,
      body, epilogue, tls_size);
}

void gpu_parallel_range_for(RuntimeContext *context,
                            int begin,
                            int end,
                            range_for_xlogue prologue,
                            RangeForTaskFunc *func,
                            range_for_xlogue epilogue,
                            const std::size_t tls_size) {
  int idx = thread_idx() + block_dim() * block_idx() + begin;
#ifdef ARCH_amdgpu
  // AMDGPU doesn't support dynamic array
  // TODO: find a better way to set the tls_size (maybe like struct_for
  alignas(8) char tls_buffer[64];
#else
  alignas(8) char tls_buffer[tls_size];
#endif
  auto tls_ptr = &tls_buffer[0];
  if (prologue)
    prologue(context, tls_ptr);
  while (idx < end) {
    func(context, tls_ptr, idx);
    idx += block_dim() * grid_dim();
  }
  if (epilogue)
    epilogue(context, tls_ptr);
}

void gpu_parallel_range_for_one_to_one(RuntimeContext *context,
                                       int begin,
                                       int end,
                                       range_for_xlogue prologue,
                                       RangeForTaskFunc *func,
                                       range_for_xlogue epilogue,
                                       const std::size_t tls_size) {
  int idx = thread_idx() + block_dim() * block_idx() + begin;
  alignas(8) char tls_buffer[tls_size];
  auto tls_ptr = &tls_buffer[0];
  if (prologue)
    prologue(context, tls_ptr);
  if (idx < end)
    func(context, tls_ptr, idx);
  if (epilogue)
    epilogue(context, tls_ptr);
}

void gpu_parallel_range_for_shared_staged(RuntimeContext *context,
                                          int begin,
                                          int end,
                                          range_for_xlogue prologue,
                                          range_for_xlogue bls_prologue,
                                          RangeForTaskFunc *func,
                                          range_for_xlogue epilogue,
                                          const std::size_t tls_size) {
  int idx = thread_idx() + block_dim() * block_idx() + begin;
  alignas(8) char tls_buffer[tls_size];
  auto tls_ptr = &tls_buffer[0];
  if (prologue)
    prologue(context, tls_ptr);
  bls_prologue(context, tls_ptr);
  block_barrier();
  if (idx < end)
    func(context, tls_ptr, idx);
  if (epilogue)
    epilogue(context, tls_ptr);
}

void gpu_parallel_range_for_shared_staged_2d(RuntimeContext *context,
                                             int begin,
                                             int end,
                                             int logical_height,
                                             int logical_width,
                                             int tile_height,
                                             int tile_width,
                                             range_for_xlogue prologue,
                                             range_for_xlogue bls_prologue,
                                             RangeForTaskFunc *func,
                                             range_for_xlogue epilogue,
                                             const std::size_t tls_size) {
  int tiles_per_row = (logical_width + tile_width - 1) / tile_width;
  int tile_row = block_idx() / tiles_per_row;
  int tile_column = block_idx() % tiles_per_row;
  int local_row = thread_idx() / tile_width;
  int local_column = thread_idx() % tile_width;
  int row = tile_row * tile_height + local_row;
  int column = tile_column * tile_width + local_column;
  int idx = begin + row * logical_width + column;
  alignas(8) char tls_buffer[tls_size];
  auto tls_ptr = &tls_buffer[0];
  if (prologue)
    prologue(context, tls_ptr);
  bls_prologue(context, tls_ptr);
  block_barrier();
  if (row < logical_height && column < logical_width && idx < end)
    func(context, tls_ptr, idx);
  if (epilogue)
    epilogue(context, tls_ptr);
}

struct mesh_task_helper_context {
  RuntimeContext *context;
  mesh_for_xlogue prologue{nullptr};
  RangeForTaskFunc *body{nullptr};
  mesh_for_xlogue epilogue{nullptr};
  std::size_t tls_size{1};
  int num_patches;
  int block_size;
};

void cpu_parallel_mesh_for_task(void *range_context,
                                int thread_id,
                                int task_id) {
  auto ctx = *(mesh_task_helper_context *)range_context;
  alignas(8) char tls_buffer[ctx.tls_size];
  auto tls_ptr = &tls_buffer[0];

  RuntimeContext this_thread_context = *ctx.context;
  this_thread_context.cpu_thread_id = thread_id;

  int block_start = task_id * ctx.block_size;
  int block_end = std::min(block_start + ctx.block_size, ctx.num_patches);

  for (int idx = block_start; idx < block_end; idx++) {
    if (ctx.prologue)
      ctx.prologue(ctx.context, tls_ptr, idx);
    ctx.body(&this_thread_context, tls_ptr, idx);
    if (ctx.epilogue)
      ctx.epilogue(ctx.context, tls_ptr, idx);
  }
}

void cpu_parallel_mesh_for(RuntimeContext *context,
                           int num_threads,
                           int num_patches,
                           int block_dim,
                           mesh_for_xlogue prologue,
                           RangeForTaskFunc *body,
                           mesh_for_xlogue epilogue,
                           std::size_t tls_size) {
  mesh_task_helper_context ctx;
  ctx.context = context;
  ctx.prologue = prologue;
  ctx.tls_size = tls_size;
  ctx.body = body;
  ctx.epilogue = epilogue;
  ctx.num_patches = num_patches;
  if (block_dim == 0) {
    // adaptive block dim
    // ensure each thread has at least ~32 tasks for load balancing
    // and each task has at least 512 items to amortize scheduler overhead
    block_dim = std::min(512, std::max(1, num_patches / (num_threads * 32)));
  }
  ctx.block_size = block_dim;
  auto runtime = context->runtime;
  runtime->parallel_for(runtime->thread_pool,
                        (num_patches + block_dim - 1) / block_dim, num_threads,
                        &ctx, cpu_parallel_mesh_for_task);
}

void gpu_parallel_mesh_for(RuntimeContext *context,
                           int num_patches,
                           mesh_for_xlogue prologue,
                           MeshForTaskFunc *func,
                           mesh_for_xlogue epilogue,
                           const std::size_t tls_size) {
#ifdef ARCH_amdgpu
  // AMDGPU doesn't support dynamic array
  // TODO: find a better way to set the tls_size (maybe like struct_for
  alignas(8) char tls_buffer[64];
#else
  alignas(8) char tls_buffer[tls_size];
#endif
  auto tls_ptr = &tls_buffer[0];
  for (int idx = block_idx(); idx < num_patches; idx += grid_dim()) {
    if (prologue)
      prologue(context, tls_ptr, idx);
    func(context, tls_ptr, idx);
    if (epilogue)
      epilogue(context, tls_ptr, idx);
  }
}

i32 linear_thread_idx(RuntimeContext *context) {
#if ARCH_cuda || ARCH_amdgpu
  return block_idx() * block_dim() + thread_idx();
#else
  return context->cpu_thread_id;
#endif
}

#include "node_dense.h"
#include "node_dynamic.h"
#include "node_hash.h"
#include "node_pointer.h"
#include "node_root.h"
#include "node_bitmasked.h"

void ListManager::touch_chunk(int chunk_id) {
  taichi_assert_runtime(runtime,
                        chunk_id >= 0 &&
                            (std::size_t)chunk_id < max_num_chunks,
                        "List manager out of chunks.");
  if (!get_chunk_ptr(chunk_id)) {
    locked_task(&lock, [&] {
      // may have been allocated during lock contention
      if (!get_chunk_ptr(chunk_id)) {
        Ptr *chunk_slot;
        PreallocatedMemoryChunk &mc =
            backing_chunk ? *backing_chunk : runtime->runtime_memory_chunk;
        if (chunk_id < inline_num_chunks) {
          chunk_slot = &inline_chunks[chunk_id];
        } else {
          const i32 relative = chunk_id - inline_num_chunks;
          const i32 directory_id = relative / chunks_per_directory;
          const i32 directory_offset = relative % chunks_per_directory;
          if (chunk_directories[directory_id] == nullptr) {
            auto directory = runtime->allocate_aligned(
                mc, kLlvmListManagerDirectoryPageBytes,
                kLlvmListManagerAllocationAlignment, true /*request*/);
            std::memset(directory, 0, kLlvmListManagerDirectoryPageBytes);
            grid_memfence();
            atomic_exchange_u64((u64 *)&chunk_directories[directory_id],
                                (u64)directory);
          }
          chunk_slot =
              &((Ptr *)chunk_directories[directory_id])[directory_offset];
        }
        grid_memfence();
        // Phase 1 (2026-05): route data allocations to per-SNode dedicated
        // chunk when set, otherwise fall back to global runtime_memory_chunk.
        auto chunk_ptr = runtime->allocate_aligned(
            mc, max_num_elements_per_chunk * element_size, 4096,
            true /*request*/);
        atomic_exchange_u64((u64 *)chunk_slot, (u64)chunk_ptr);
      }
    });
  }
}

void ListManager::append(void *data_ptr) {
  auto ptr = allocate();
  std::memcpy(ptr, data_ptr, element_size);
}

Ptr ListManager::allocate() {
  auto i = reserve_new_element();
  return get_element_ptr(i);
}

void node_gc(LLVMRuntime *runtime, uint64 snode_runtime_key) {
  snode_runtime_state(runtime, snode_runtime_key)->node_allocator->gc_serial();
}

void gc_parallel_impl_0(RuntimeContext *context, NodeManager *allocator) {
  auto free_list = allocator->free_list;
  auto free_list_size = free_list->size();
  auto free_list_used = allocator->free_list_used;
  using T = NodeManager::list_data_type;

  // Move unused elements to the beginning of the free_list
  int i = linear_thread_idx(context);
  if (free_list_used * 2 > free_list_size) {
    // Directly copy. Dst and src does not overlap
    auto items_to_copy = free_list_size - free_list_used;
    while (i < items_to_copy) {
      free_list->get<T>(i) = free_list->get<T>(free_list_used + i);
      i += grid_dim() * block_dim();
    }
  } else {
    // Move only non-overlapping parts
    auto items_to_copy = free_list_used;
    while (i < items_to_copy) {
      free_list->get<T>(i) =
          free_list->get<T>(free_list_size - items_to_copy + i);
      i += grid_dim() * block_dim();
    }
  }
}

void gc_parallel_0(RuntimeContext *context, uint64 snode_runtime_key) {
  LLVMRuntime *runtime = context->runtime;
  gc_parallel_impl_0(
      context,
      snode_runtime_state(runtime, snode_runtime_key)->node_allocator);
}

void gc_parallel_impl_1(NodeManager *allocator) {
  auto free_list = allocator->free_list;

  const i32 num_unused =
      max_i32(free_list->size() - allocator->free_list_used, 0);
  free_list->resize(num_unused);

  allocator->free_list_used = 0;
  allocator->recycle_list_size_backup = allocator->recycled_list->size();
  allocator->recycled_list->clear();
}

void gc_parallel_1(RuntimeContext *context, uint64 snode_runtime_key) {
  LLVMRuntime *runtime = context->runtime;
  gc_parallel_impl_1(
      snode_runtime_state(runtime, snode_runtime_key)->node_allocator);
}

void gc_parallel_impl_2(NodeManager *allocator) {
  auto elements = allocator->recycle_list_size_backup;
  auto free_list = allocator->free_list;
  auto recycled_list = allocator->recycled_list;
  auto data_list = allocator->data_list;
  auto element_size = allocator->element_size;
  using T = NodeManager::list_data_type;
  auto i = block_idx();
  while (i < elements) {
    auto idx = recycled_list->get<T>(i);
    auto ptr = data_list->get_element_ptr(idx);
    if (thread_idx() == 0) {
      free_list->push_back(idx);
    }
    // memset
    auto ptr_stop = ptr + element_size;
    if ((uint64)ptr % 4 != 0) {
      auto new_ptr = ptr + 4 - (uint64)ptr % 4;
      if (thread_idx() == 0) {
        for (uint8 *p = ptr; p < new_ptr; p++) {
          *p = 0;
        }
      }
      ptr = new_ptr;
    }
    // now ptr is a multiple of 4
    ptr += thread_idx() * sizeof(uint32);
    while (ptr + sizeof(uint32) <= ptr_stop) {
      *(uint32 *)ptr = 0;
      ptr += sizeof(uint32) * block_dim();
    }
    while (ptr < ptr_stop) {
      *ptr = 0;
      ptr++;
    }
    i += grid_dim();
  }
}

void gc_parallel_2(RuntimeContext *context, uint64 snode_runtime_key) {
  LLVMRuntime *runtime = context->runtime;
  gc_parallel_impl_2(
      snode_runtime_state(runtime, snode_runtime_key)->node_allocator);
}
}

extern "C" {

u32 rand_u32(RuntimeContext *context) {
  auto state = &((LLVMRuntime *)context->runtime)
                    ->rand_states[linear_thread_idx(context)];

  auto &x = state->x;
  auto &y = state->y;
  auto &z = state->z;
  auto &w = state->w;
  auto t = x ^ (x << 11);

  x = y;
  y = z;
  z = w;
  w = (w ^ (w >> 19)) ^ (t ^ (t >> 8));

  return w * 1000000007;  // multiply a prime number here is very necessary -
                          // it decorrelates streams of PRNGs.
}

uint64 rand_u64(RuntimeContext *context) {
  return ((u64)rand_u32(context) << 32) + rand_u32(context);
}

f32 rand_f32(RuntimeContext *context) {
  return (rand_u32(context) >> 8) * (1.0f / 16777216.0f);
}

f64 rand_f64(RuntimeContext *context) {
  return (rand_u64(context) >> 11) * (1.0 / 9007199254740992.0);
}

i32 rand_i32(RuntimeContext *context) {
  return rand_u32(context);
}

i64 rand_i64(RuntimeContext *context) {
  return rand_u64(context);
}
};

struct printf_helper {
  char buffer[1024];
  int tail;

  printf_helper() {
    std::memset(buffer, 0, sizeof(buffer));
    tail = 0;
  }

  void push_back() {
  }

  template <typename... Args, typename T>
  void push_back(T t, Args &&...args) {
    *(T *)&buffer[tail] = t;
    if (tail % sizeof(T) != 0)
      tail += sizeof(T) - tail % sizeof(T);
    // align
    tail += sizeof(T);
    if constexpr ((sizeof...(args)) != 0) {
      push_back(std::forward<Args>(args)...);
    }
  }

  Ptr ptr() {
    return (Ptr) & (buffer[0]);
  }
};

template <typename... Args>
void taichi_printf(LLVMRuntime *runtime, const char *format, Args &&...args) {
#if ARCH_cuda
  printf_helper helper;
  helper.push_back(std::forward<Args>(args)...);
  cuda_vprintf((Ptr)format, helper.ptr());
#elif ARCH_amdgpu
// TODO: add printf for amdgpu backend
#else
  runtime->host_printf(format, args...);
#endif
}

#include "locked_task.h"

extern "C" {  // local stack operations

Ptr stack_top_primal(Ptr stack, std::size_t element_size) {
  auto n = *(u64 *)stack;
  return stack + sizeof(u64) + (n - 1) * 2 * element_size;
}

Ptr stack_top_adjoint(Ptr stack, std::size_t element_size) {
  return stack_top_primal(stack, element_size) + element_size;
}

void stack_init(Ptr stack) {
  *(u64 *)stack = 0;
}

void stack_pop(Ptr stack) {
  auto &n = *(u64 *)stack;
  n--;
}

void stack_push(Ptr stack, size_t max_num_elements, std::size_t element_size) {
  u64 &n = *(u64 *)stack;
  n += 1;
  // TODO: assert n <= max_elements
  std::memset(stack_top_primal(stack, element_size), 0, element_size * 2);
}

#include "internal_functions.h"

// TODO: make here less repetitious.
// Original implementation is
// u##N mask = ((((u##N)1 << bits) - 1) << offset);
// When N equals bits equals 32, 32 times of left shifting will be carried on
// which is an undefined behavior.
// see #2096 for more details
#define DEFINE_SET_PARTIAL_BITS(N)                                            \
  void set_mask_b##N(u##N *ptr, u64 mask, u##N value) {                       \
    u##N mask_N = (u##N)mask;                                                 \
    *ptr = (*ptr & (~mask_N)) | (value & mask);                               \
  }                                                                           \
                                                                              \
  void atomic_set_mask_b##N(u##N *ptr, u64 mask, u##N value) {                \
    u##N mask_N = (u##N)mask;                                                 \
    u##N new_value = 0;                                                       \
    u##N old_value = *ptr;                                                    \
    do {                                                                      \
      old_value = *ptr;                                                       \
      new_value = (old_value & (~mask_N)) | (value & mask);                   \
    } while (                                                                 \
        !__atomic_compare_exchange(ptr, &old_value, &new_value, true,         \
                                   std::memory_order::memory_order_seq_cst,   \
                                   std::memory_order::memory_order_seq_cst)); \
  }                                                                           \
                                                                              \
  void set_partial_bits_b##N(u##N *ptr, u32 offset, u32 bits, u##N value) {   \
    u##N mask = ((~(u##N)0) << (N - bits)) >> (N - offset - bits);            \
    set_mask_b##N(ptr, mask, value << offset);                                \
  }                                                                           \
                                                                              \
  void atomic_set_partial_bits_b##N(u##N *ptr, u32 offset, u32 bits,          \
                                    u##N value) {                             \
    u##N mask = ((~(u##N)0) << (N - bits)) >> (N - offset - bits);            \
    atomic_set_mask_b##N(ptr, mask, value << offset);                         \
  }                                                                           \
                                                                              \
  u##N atomic_add_partial_bits_b##N(u##N *ptr, u32 offset, u32 bits,          \
                                    u##N value) {                             \
    u##N mask = ((~(u##N)0) << (N - bits)) >> (N - offset - bits);            \
    u##N new_value = 0;                                                       \
    u##N old_value = *ptr;                                                    \
    do {                                                                      \
      old_value = *ptr;                                                       \
      new_value = old_value + (value << offset);                              \
      new_value = (old_value & (~mask)) | (new_value & mask);                 \
    } while (                                                                 \
        !__atomic_compare_exchange(ptr, &old_value, &new_value, true,         \
                                   std::memory_order::memory_order_seq_cst,   \
                                   std::memory_order::memory_order_seq_cst)); \
    return old_value;                                                         \
  }

DEFINE_SET_PARTIAL_BITS(8);
DEFINE_SET_PARTIAL_BITS(16);
DEFINE_SET_PARTIAL_BITS(32);
DEFINE_SET_PARTIAL_BITS(64);

f32 rounding_prepare_f32(f32 f) {
  /* slower (but clearer) version with branching:
  if (f > 0)
    return f + 0.5;
  else
    return f - 0.5;
  */

  // Branch-free implementation: copy the sign bit of "f" to "0.5"
  i32 delta_bits =
      (taichi_union_cast<i32>(f) & 0x80000000) | taichi_union_cast<i32>(0.5f);
  f32 delta = taichi_union_cast<f32>(delta_bits);
  return f + delta;
}

f64 rounding_prepare_f64(f64 f) {
  // Same as above
  i64 delta_bits = (taichi_union_cast<i64>(f) & 0x8000000000000000LL) |
                   taichi_union_cast<i64>(0.5);
  f64 delta = taichi_union_cast<f64>(delta_bits);
  return f + delta;
}
}

#endif
