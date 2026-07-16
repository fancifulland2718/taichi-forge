// Source of the low-PTX CUDA scan/reduce/histogram provider.
//
// This file is not part of the normal CMake target.  It is compiled with the
// repository-provisioned Clang in CUDA device-only mode, with -nocudainc and
// -nocudalib, and the resulting PTX is embedded in hierarchical_ptx.inc.h.
// Keep this source free of CUDA Toolkit headers and runtime calls.

#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __host__ __attribute__((host))
#define __shared__ __attribute__((shared))

using u8 = unsigned char;
using i32 = int;
using u32 = unsigned int;
using i64 = long long;
using u64 = unsigned long long;

#include <__clang_cuda_builtin_vars.h>

template <typename To, typename From>
__device__ To bit_cast(From value) {
  static_assert(sizeof(To) == sizeof(From), "bit_cast size mismatch");
  union {
    From from;
    To to;
  } bits;
  bits.from = value;
  return bits.to;
}

__device__ void block_barrier() {
  asm volatile("bar.sync 0;" : : : "memory");
}

template <typename T>
__device__ T shfl_up(T value, u32 delta);

template <>
__device__ i32 shfl_up(i32 value, u32 delta) {
  i32 result;
  asm("shfl.up.b32 %0, %1, %2, 0x0;" : "=r"(result) : "r"(value), "r"(delta));
  return result;
}

template <>
__device__ u32 shfl_up(u32 value, u32 delta) {
  return static_cast<u32>(shfl_up(static_cast<i32>(value), delta));
}

template <>
__device__ float shfl_up(float value, u32 delta) {
  return bit_cast<float>(shfl_up(bit_cast<i32>(value), delta));
}

template <typename T>
__device__ T shfl_up_64(T value, u32 delta) {
  u64 bits = bit_cast<u64>(value);
  u32 lo = static_cast<u32>(bits);
  u32 hi = static_cast<u32>(bits >> 32);
  lo = shfl_up(lo, delta);
  hi = shfl_up(hi, delta);
  return bit_cast<T>((static_cast<u64>(hi) << 32) | lo);
}

template <>
__device__ i64 shfl_up(i64 value, u32 delta) {
  return shfl_up_64(value, delta);
}

template <>
__device__ u64 shfl_up(u64 value, u32 delta) {
  return shfl_up_64(value, delta);
}

template <>
__device__ double shfl_up(double value, u32 delta) {
  return shfl_up_64(value, delta);
}

template <typename T>
__device__ T shfl_down(T value, u32 delta);

template <>
__device__ i32 shfl_down(i32 value, u32 delta) {
  i32 result;
  asm("shfl.down.b32 %0, %1, %2, 0x1f;"
      : "=r"(result)
      : "r"(value), "r"(delta));
  return result;
}

template <>
__device__ u32 shfl_down(u32 value, u32 delta) {
  return static_cast<u32>(shfl_down(static_cast<i32>(value), delta));
}

template <>
__device__ float shfl_down(float value, u32 delta) {
  return bit_cast<float>(shfl_down(bit_cast<i32>(value), delta));
}

template <typename T>
__device__ T shfl_down_64(T value, u32 delta) {
  u64 bits = bit_cast<u64>(value);
  u32 lo = static_cast<u32>(bits);
  u32 hi = static_cast<u32>(bits >> 32);
  lo = shfl_down(lo, delta);
  hi = shfl_down(hi, delta);
  return bit_cast<T>((static_cast<u64>(hi) << 32) | lo);
}

template <>
__device__ i64 shfl_down(i64 value, u32 delta) {
  return shfl_down_64(value, delta);
}

template <>
__device__ u64 shfl_down(u64 value, u32 delta) {
  return shfl_down_64(value, delta);
}

template <>
__device__ double shfl_down(double value, u32 delta) {
  return shfl_down_64(value, delta);
}

template <typename T>
__device__ bool is_nan(T) {
  return false;
}

template <>
__device__ bool is_nan(float value) {
  u32 bits = bit_cast<u32>(value);
  return (bits & 0x7fffffffu) > 0x7f800000u;
}

template <>
__device__ bool is_nan(double value) {
  u64 bits = bit_cast<u64>(value);
  return (bits & 0x7fffffffffffffffull) > 0x7ff0000000000000ull;
}

template <typename T>
__device__ T add_values(T lhs, T rhs) {
  return lhs + rhs;
}

template <>
__device__ i32 add_values(i32 lhs, i32 rhs) {
  return static_cast<i32>(static_cast<u32>(lhs) + static_cast<u32>(rhs));
}

template <>
__device__ i64 add_values(i64 lhs, i64 rhs) {
  return static_cast<i64>(static_cast<u64>(lhs) + static_cast<u64>(rhs));
}

template <typename T>
__device__ T combine(T lhs, T rhs, i32 op) {
  if (op == 0) {
    return add_values(lhs, rhs);
  }
  if (is_nan(lhs)) {
    return lhs;
  }
  if (is_nan(rhs)) {
    return rhs;
  }
  if (op == 1) {
    return rhs < lhs ? rhs : lhs;
  }
  return lhs < rhs ? rhs : lhs;
}

template <typename T>
__device__ T identity(i32 op);

template <>
__device__ i32 identity(i32 op) {
  if (op == 1) {
    return 0x7fffffff;
  }
  if (op == 2) {
    return static_cast<i32>(0x80000000u);
  }
  return 0;
}

template <>
__device__ u32 identity(i32 op) {
  if (op == 1) {
    return 0xffffffffu;
  }
  return 0u;
}

template <>
__device__ float identity(i32 op) {
  if (op == 1) {
    return bit_cast<float>(0x7f800000u);
  }
  if (op == 2) {
    return bit_cast<float>(0xff800000u);
  }
  return 0.0f;
}

template <>
__device__ i64 identity(i32 op) {
  if (op == 1) {
    return 0x7fffffffffffffffll;
  }
  if (op == 2) {
    return static_cast<i64>(0x8000000000000000ull);
  }
  return 0ll;
}

template <>
__device__ u64 identity(i32 op) {
  if (op == 1) {
    return 0xffffffffffffffffull;
  }
  return 0ull;
}

template <>
__device__ double identity(i32 op) {
  if (op == 1) {
    return bit_cast<double>(0x7ff0000000000000ull);
  }
  if (op == 2) {
    return bit_cast<double>(0xfff0000000000000ull);
  }
  return 0.0;
}

template <typename T>
__device__ T load_strided(const u8 *base, u64 offset, u64 stride, u32 index) {
  return *reinterpret_cast<const T *>(base + offset +
                                      static_cast<u64>(index) * stride);
}

template <typename T>
__device__ void store_strided(u8 *base,
                              u64 offset,
                              u64 stride,
                              u32 index,
                              T value) {
  *reinterpret_cast<T *>(base + offset + static_cast<u64>(index) * stride) =
      value;
}

template <typename T>
__device__ T warp_inclusive_sum(T value) {
  const u32 lane = threadIdx.x & 31u;
  for (u32 delta = 1; delta < 32; delta <<= 1) {
    T other = shfl_up(value, delta);
    if (lane >= delta) {
      value = combine(value, other, 0);
    }
  }
  return value;
}

template <typename T>
__device__ T warp_reduce(T value, i32 op) {
  const u32 lane = threadIdx.x & 31u;
  for (u32 delta = 16; delta > 0; delta >>= 1) {
    T other = shfl_down(value, delta);
    if (lane + delta < 32) {
      value = combine(value, other, op);
    }
  }
  return value;
}

template <typename T>
__device__ void scan_blocks_impl(u8 *values,
                                 u64 offset,
                                 u64 stride,
                                 u32 n,
                                 u8 *block_sums,
                                 u64 sums_offset,
                                 i32 reverse,
                                 T *warp_sums) {
  const u32 tid = threadIdx.x;
  const u32 lane = tid & 31u;
  const u32 warp = tid >> 5;
  const u32 block = blockIdx.x;
  const u32 index = block * 256u + tid;

  const u32 physical_index = reverse != 0 ? n - 1u - index : index;
  T value = index < n ? load_strided<T>(values, offset, stride, physical_index)
                      : identity<T>(0);
  value = warp_inclusive_sum(value);
  if (lane == 31u) {
    warp_sums[warp] = value;
  }
  block_barrier();

  if (warp == 0u) {
    T warp_value = lane < 8u ? warp_sums[lane] : identity<T>(0);
    warp_value = warp_inclusive_sum(warp_value);
    if (lane < 8u) {
      warp_sums[lane] = warp_value;
    }
  }
  block_barrier();

  if (warp > 0u) {
    value = combine(value, warp_sums[warp - 1u], 0);
  }
  if (index < n) {
    store_strided<T>(values, offset, stride, physical_index, value);
  }
  if (tid == 255u && block_sums != nullptr) {
    store_strided<T>(block_sums, sums_offset, sizeof(T), block, warp_sums[7]);
  }
}

template <typename T>
__device__ void uniform_add_impl(u8 *values,
                                 u64 offset,
                                 u64 stride,
                                 u32 n,
                                 const u8 *block_sums,
                                 u64 sums_offset,
                                 i32 reverse) {
  const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) {
    return;
  }
  const u32 block = index >> 8;
  if (block == 0u) {
    return;
  }
  const u32 physical_index = reverse != 0 ? n - 1u - index : index;
  T value = load_strided<T>(values, offset, stride, physical_index);
  T base = load_strided<T>(block_sums, sums_offset, sizeof(T), block - 1u);
  store_strided<T>(values, offset, stride, physical_index,
                   combine(value, base, 0));
}

template <typename T>
__device__ void reduce_blocks_impl(const u8 *values,
                                   u64 offset,
                                   u64 stride,
                                   u32 n,
                                   u8 *output,
                                   u64 output_offset,
                                   u64 output_stride,
                                   i32 op,
                                   T *warp_sums) {
  const u32 tid = threadIdx.x;
  const u32 lane = tid & 31u;
  const u32 warp = tid >> 5;
  const u32 block = blockIdx.x;
  const u32 tile_begin = block * 1024u;

  T value = identity<T>(op);
  for (u32 item = 0; item < 4u; ++item) {
    const u32 index = tile_begin + tid + item * 256u;
    if (index < n) {
      value =
          combine(value, load_strided<T>(values, offset, stride, index), op);
    }
  }
  value = warp_reduce(value, op);
  if (lane == 0u) {
    warp_sums[warp] = value;
  }
  block_barrier();

  if (warp == 0u) {
    T warp_value = lane < 8u ? warp_sums[lane] : identity<T>(op);
    warp_value = warp_reduce(warp_value, op);
    if (lane == 0u) {
      store_strided<T>(output, output_offset, output_stride, block, warp_value);
    }
  }
}

#define DEFINE_SCAN_KERNEL(NAME, TYPE)                                         \
  extern "C" __global__ void NAME(u8 *values, u64 offset, u64 stride, u32 n,   \
                                  u8 *block_sums, u64 sums_offset,             \
                                  i32 reverse) {                               \
    __shared__ TYPE warp_sums[8];                                              \
    scan_blocks_impl<TYPE>(values, offset, stride, n, block_sums, sums_offset, \
                           reverse, warp_sums);                                \
  }

#define DEFINE_UNIFORM_KERNEL(NAME, TYPE)                                      \
  extern "C" __global__ void NAME(u8 *values, u64 offset, u64 stride, u32 n,   \
                                  const u8 *block_sums, u64 sums_offset,       \
                                  i32 reverse) {                               \
    uniform_add_impl<TYPE>(values, offset, stride, n, block_sums, sums_offset, \
                           reverse);                                           \
  }

#define DEFINE_REDUCE_KERNEL(NAME, TYPE)                                       \
  extern "C" __global__ void NAME(const u8 *values, u64 offset, u64 stride,    \
                                  u32 n, u8 *output, u64 output_offset,        \
                                  u64 output_stride, i32 op) {                 \
    __shared__ TYPE warp_sums[8];                                              \
    reduce_blocks_impl<TYPE>(values, offset, stride, n, output, output_offset, \
                             output_stride, op, warp_sums);                    \
  }

DEFINE_SCAN_KERNEL(scan_blocks_i32, i32)
DEFINE_SCAN_KERNEL(scan_blocks_u32, u32)
DEFINE_SCAN_KERNEL(scan_blocks_f32, float)
DEFINE_SCAN_KERNEL(scan_blocks_i64, i64)
DEFINE_SCAN_KERNEL(scan_blocks_u64, u64)
DEFINE_SCAN_KERNEL(scan_blocks_f64, double)

DEFINE_UNIFORM_KERNEL(uniform_add_i32, i32)
DEFINE_UNIFORM_KERNEL(uniform_add_u32, u32)
DEFINE_UNIFORM_KERNEL(uniform_add_f32, float)
DEFINE_UNIFORM_KERNEL(uniform_add_i64, i64)
DEFINE_UNIFORM_KERNEL(uniform_add_u64, u64)
DEFINE_UNIFORM_KERNEL(uniform_add_f64, double)

DEFINE_REDUCE_KERNEL(reduce_blocks_i32, i32)
DEFINE_REDUCE_KERNEL(reduce_blocks_u32, u32)
DEFINE_REDUCE_KERNEL(reduce_blocks_f32, float)
DEFINE_REDUCE_KERNEL(reduce_blocks_i64, i64)
DEFINE_REDUCE_KERNEL(reduce_blocks_u64, u64)
DEFINE_REDUCE_KERNEL(reduce_blocks_f64, double)

__device__ i32 atomic_add_i32(i32 *address, i32 value) {
  i32 old;
  asm volatile("atom.global.add.u32 %0, [%1], %2;"
               : "=r"(old)
               : "l"(address), "r"(value));
  return old;
}

__device__ i64 atomic_add_i64(i64 *address, i64 value) {
  i64 old;
  asm volatile("atom.global.add.u64 %0, [%1], %2;"
               : "=l"(old)
               : "l"(address), "l"(value));
  return old;
}

template <typename Counter>
__device__ void atomic_increment(Counter *address);

template <>
__device__ void atomic_increment(i32 *address) {
  atomic_add_i32(address, 1);
}

template <>
__device__ void atomic_increment(i64 *address) {
  atomic_add_i64(address, 1);
}

template <typename Value, typename Counter>
__device__ void histogram_impl(const u8 *values,
                               u64 values_offset,
                               u64 values_stride,
                               u32 n,
                               u8 *bins,
                               u64 bins_offset,
                               u64 bins_stride,
                               u32 num_bins) {
  const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) {
    return;
  }
  Value sample =
      load_strided<Value>(values, values_offset, values_stride, index);
  i64 bin = static_cast<i64>(sample);
  if (bin >= 0 && static_cast<u64>(bin) < num_bins) {
    Counter *counter = reinterpret_cast<Counter *>(
        bins + bins_offset + static_cast<u64>(bin) * bins_stride);
    atomic_increment(counter);
  }
}

template <typename Counter>
__device__ void zero_bins_impl(u8 *bins,
                               u64 bins_offset,
                               u64 bins_stride,
                               u32 num_bins) {
  const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < num_bins) {
    store_strided<Counter>(bins, bins_offset, bins_stride, index,
                           static_cast<Counter>(0));
  }
}

#define DEFINE_HISTOGRAM_KERNEL(NAME, VALUE, COUNTER)                          \
  extern "C" __global__ void NAME(                                             \
      const u8 *values, u64 values_offset, u64 values_stride, u32 n, u8 *bins, \
      u64 bins_offset, u64 bins_stride, u32 num_bins) {                        \
    histogram_impl<VALUE, COUNTER>(values, values_offset, values_stride, n,    \
                                   bins, bins_offset, bins_stride, num_bins);  \
  }

extern "C" __global__ void zero_bins_i32(u8 *bins,
                                         u64 bins_offset,
                                         u64 bins_stride,
                                         u32 num_bins) {
  zero_bins_impl<i32>(bins, bins_offset, bins_stride, num_bins);
}

extern "C" __global__ void zero_bins_i64(u8 *bins,
                                         u64 bins_offset,
                                         u64 bins_stride,
                                         u32 num_bins) {
  zero_bins_impl<i64>(bins, bins_offset, bins_stride, num_bins);
}

DEFINE_HISTOGRAM_KERNEL(histogram_i32_i32, i32, i32)
DEFINE_HISTOGRAM_KERNEL(histogram_u32_i32, u32, i32)
DEFINE_HISTOGRAM_KERNEL(histogram_i32_i64, i32, i64)
DEFINE_HISTOGRAM_KERNEL(histogram_u32_i64, u32, i64)

__device__ u32 atomic_add_u32(u32 *address, u32 value) {
  u32 old;
  asm volatile("atom.global.add.u32 %0, [%1], %2;"
               : "=r"(old)
               : "l"(address), "r"(value));
  return old;
}

__device__ u64 atomic_add_u64(u64 *address, u64 value) {
  u64 old;
  asm volatile("atom.global.add.u64 %0, [%1], %2;"
               : "=l"(old)
               : "l"(address), "l"(value));
  return old;
}

__device__ float atomic_add_f32(float *address, float value) {
  float old;
  asm volatile("atom.global.add.f32 %0, [%1], %2;"
               : "=f"(old)
               : "l"(address), "f"(value));
  return old;
}

__device__ u64 atomic_cas_u64(u64 *address, u64 compare, u64 value) {
  u64 old;
  asm volatile("atom.global.cas.b64 %0, [%1], %2, %3;"
               : "=l"(old)
               : "l"(address), "l"(compare), "l"(value));
  return old;
}

__device__ double atomic_add_f64(double *address, double value) {
  u64 *bits = reinterpret_cast<u64 *>(address);
  u64 observed = *bits;
  while (true) {
    const u64 expected = observed;
    const double updated = bit_cast<double>(expected) + value;
    observed = atomic_cas_u64(bits, expected, bit_cast<u64>(updated));
    if (observed == expected) {
      return bit_cast<double>(expected);
    }
  }
}

template <typename T>
__device__ void atomic_add_value(T *address, T value);

template <>
__device__ void atomic_add_value(i32 *address, i32 value) {
  atomic_add_i32(address, value);
}
template <>
__device__ void atomic_add_value(u32 *address, u32 value) {
  atomic_add_u32(address, value);
}
template <>
__device__ void atomic_add_value(float *address, float value) {
  atomic_add_f32(address, value);
}
template <>
__device__ void atomic_add_value(i64 *address, i64 value) {
  atomic_add_i64(address, value);
}
template <>
__device__ void atomic_add_value(u64 *address, u64 value) {
  atomic_add_u64(address, value);
}
template <>
__device__ void atomic_add_value(double *address, double value) {
  atomic_add_f64(address, value);
}

template <typename T>
__device__ void add_strided_impl(const u8 *src,
                                 u64 src_offset,
                                 u64 src_stride,
                                 u8 *dst,
                                 u64 dst_offset,
                                 u64 dst_stride,
                                 u32 n) {
  const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    const T lhs = load_strided<T>(dst, dst_offset, dst_stride, index);
    const T rhs = load_strided<T>(src, src_offset, src_stride, index);
    store_strided<T>(dst, dst_offset, dst_stride, index, add_values(lhs, rhs));
  }
}

template <typename T>
__device__ void add_scaled_strided_impl(const u8 *src,
                                        u64 src_offset,
                                        u64 src_stride,
                                        u8 *dst,
                                        u64 dst_offset,
                                        u64 dst_stride,
                                        u32 n,
                                        double scale) {
  const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    const T lhs = load_strided<T>(dst, dst_offset, dst_stride, index);
    const T rhs = load_strided<T>(src, src_offset, src_stride, index);
    store_strided<T>(dst, dst_offset, dst_stride, index,
                     lhs + rhs * static_cast<T>(scale));
  }
}

template <typename T>
__device__ void scatter_add_strided_impl(const u8 *src,
                                         u64 src_offset,
                                         u64 src_stride,
                                         const u8 *indices,
                                         u64 indices_offset,
                                         u64 indices_stride,
                                         u8 *dst,
                                         u64 dst_offset,
                                         u64 dst_stride,
                                         u32 n,
                                         u32 index_bound) {
  const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) {
    return;
  }
  const i32 key =
      load_strided<i32>(indices, indices_offset, indices_stride, index);
  if (key >= 0 && static_cast<u32>(key) < index_bound) {
    T *target = reinterpret_cast<T *>(dst + dst_offset +
                                      static_cast<u64>(key) * dst_stride);
    atomic_add_value(target,
                     load_strided<T>(src, src_offset, src_stride, index));
  }
}

template <typename T>
__device__ void zero_strided_impl(u8 *dst,
                                  u64 dst_offset,
                                  u64 dst_stride,
                                  u32 n) {
  const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    store_strided<T>(dst, dst_offset, dst_stride, index, static_cast<T>(0));
  }
}

template <typename T>
__device__ void gather_add_strided_impl(const u8 *src,
                                        u64 src_offset,
                                        u64 src_stride,
                                        const u8 *indices,
                                        u64 indices_offset,
                                        u64 indices_stride,
                                        u8 *dst,
                                        u64 dst_offset,
                                        u64 dst_stride,
                                        u32 n,
                                        u32 index_bound) {
  const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) {
    return;
  }
  const i32 key =
      load_strided<i32>(indices, indices_offset, indices_stride, index);
  if (key >= 0 && static_cast<u32>(key) < index_bound) {
    const T lhs = load_strided<T>(dst, dst_offset, dst_stride, index);
    const T rhs =
        load_strided<T>(src, src_offset, src_stride, static_cast<u32>(key));
    store_strided<T>(dst, dst_offset, dst_stride, index, lhs + rhs);
  }
}

#define DEFINE_ADD_KERNEL(NAME, TYPE)                                      \
  extern "C" __global__ void NAME(const u8 *src, u64 src_offset,           \
                                  u64 src_stride, u8 *dst, u64 dst_offset, \
                                  u64 dst_stride, u32 n) {                 \
    add_strided_impl<TYPE>(src, src_offset, src_stride, dst, dst_offset,   \
                           dst_stride, n);                                 \
  }

#define DEFINE_SCATTER_ADD_KERNEL(NAME, TYPE)                               \
  extern "C" __global__ void NAME(                                          \
      const u8 *src, u64 src_offset, u64 src_stride, const u8 *indices,     \
      u64 indices_offset, u64 indices_stride, u8 *dst, u64 dst_offset,      \
      u64 dst_stride, u32 n, u32 index_bound) {                             \
    scatter_add_strided_impl<TYPE>(src, src_offset, src_stride, indices,    \
                                   indices_offset, indices_stride, dst,     \
                                   dst_offset, dst_stride, n, index_bound); \
  }

#define DEFINE_ADD_SCALED_KERNEL(NAME, TYPE)                               \
  extern "C" __global__ void NAME(const u8 *src, u64 src_offset,           \
                                  u64 src_stride, u8 *dst, u64 dst_offset, \
                                  u64 dst_stride, u32 n, double scale) {   \
    add_scaled_strided_impl<TYPE>(src, src_offset, src_stride, dst,        \
                                  dst_offset, dst_stride, n, scale);       \
  }

#define DEFINE_ZERO_KERNEL(NAME, TYPE)                                     \
  extern "C" __global__ void NAME(u8 *dst, u64 dst_offset, u64 dst_stride, \
                                  u32 n) {                                 \
    zero_strided_impl<TYPE>(dst, dst_offset, dst_stride, n);               \
  }

#define DEFINE_GATHER_ADD_KERNEL(NAME, TYPE)                               \
  extern "C" __global__ void NAME(                                         \
      const u8 *src, u64 src_offset, u64 src_stride, const u8 *indices,    \
      u64 indices_offset, u64 indices_stride, u8 *dst, u64 dst_offset,     \
      u64 dst_stride, u32 n, u32 index_bound) {                            \
    gather_add_strided_impl<TYPE>(src, src_offset, src_stride, indices,    \
                                  indices_offset, indices_stride, dst,     \
                                  dst_offset, dst_stride, n, index_bound); \
  }

#define DEFINE_TYPED_LINEAR_KERNELS(SUFFIX, TYPE)               \
  DEFINE_ADD_KERNEL(add_strided_##SUFFIX, TYPE)                 \
  DEFINE_SCATTER_ADD_KERNEL(scatter_add_strided_##SUFFIX, TYPE) \
  DEFINE_ZERO_KERNEL(zero_strided_##SUFFIX, TYPE)

DEFINE_TYPED_LINEAR_KERNELS(i32, i32)
DEFINE_TYPED_LINEAR_KERNELS(u32, u32)
DEFINE_TYPED_LINEAR_KERNELS(f32, float)
DEFINE_TYPED_LINEAR_KERNELS(i64, i64)
DEFINE_TYPED_LINEAR_KERNELS(u64, u64)
DEFINE_TYPED_LINEAR_KERNELS(f64, double)
DEFINE_ADD_SCALED_KERNEL(add_scaled_strided_f32, float)
DEFINE_ADD_SCALED_KERNEL(add_scaled_strided_f64, double)
DEFINE_GATHER_ADD_KERNEL(gather_add_strided_f32, float)
DEFINE_GATHER_ADD_KERNEL(gather_add_strided_f64, double)

extern "C" __global__ void normalize_flags_i32(const u8 *flags,
                                               u64 flags_offset,
                                               u64 flags_stride,
                                               i32 *prefix,
                                               u32 n) {
  const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    prefix[index] =
        load_strided<i32>(flags, flags_offset, flags_stride, index) != 0 ? 1
                                                                         : 0;
  }
}

extern "C" __global__ void compact_scatter_words(const u8 *values,
                                                 u64 values_offset,
                                                 u64 values_stride,
                                                 const i32 *prefix,
                                                 u8 *output,
                                                 u64 output_offset,
                                                 u64 output_stride,
                                                 i32 *count,
                                                 u64 count_offset,
                                                 u32 n,
                                                 u32 item_words) {
  const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n && prefix[index] != (index == 0 ? 0 : prefix[index - 1])) {
    const u32 output_index = static_cast<u32>(prefix[index] - 1);
    const u32 *src = reinterpret_cast<const u32 *>(
        values + values_offset + static_cast<u64>(index) * values_stride);
    u32 *dst =
        reinterpret_cast<u32 *>(output + output_offset +
                                static_cast<u64>(output_index) * output_stride);
    for (u32 word = 0; word < item_words; ++word) {
      dst[word] = src[word];
    }
  }
  if (index == n - 1u) {
    *reinterpret_cast<i32 *>(reinterpret_cast<u8 *>(count) + count_offset) =
        prefix[index];
  }
}

extern "C" __global__ void copy_i32_strided(const u8 *src,
                                            u64 src_offset,
                                            u64 src_stride,
                                            u8 *dst,
                                            u64 dst_offset,
                                            u64 dst_stride,
                                            u32 n) {
  const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    store_strided<i32>(dst, dst_offset, dst_stride, index,
                       load_strided<i32>(src, src_offset, src_stride, index));
  }
}

extern "C" __global__ void bucket_scatter_words(const u8 *keys,
                                                u64 keys_offset,
                                                u64 keys_stride,
                                                const u8 *values,
                                                u64 values_offset,
                                                u64 values_stride,
                                                u8 *output,
                                                u64 output_offset,
                                                u64 output_stride,
                                                i32 *cursor,
                                                u32 n,
                                                u32 num_bins,
                                                u32 item_words) {
  const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) {
    return;
  }
  const i32 key = load_strided<i32>(keys, keys_offset, keys_stride, index);
  if (key < 0 || static_cast<u32>(key) >= num_bins) {
    return;
  }
  const i32 output_index = atomic_add_i32(cursor + key, 1);
  const u32 *src = reinterpret_cast<const u32 *>(
      values + values_offset + static_cast<u64>(index) * values_stride);
  u32 *dst = reinterpret_cast<u32 *>(
      output + output_offset +
      static_cast<u64>(static_cast<u32>(output_index)) * output_stride);
  for (u32 word = 0; word < item_words; ++word) {
    dst[word] = src[word];
  }
}

template <typename T>
__device__ u32 sortable_key32(T value, i32 nan_policy);

template <>
__device__ u32 sortable_key32<u32>(u32 value, i32) {
  return value;
}

template <>
__device__ u32 sortable_key32<i32>(i32 value, i32) {
  return static_cast<u32>(value) ^ 0x80000000u;
}

template <>
__device__ u32 sortable_key32<float>(float value, i32 nan_policy) {
  const u32 bits = bit_cast<u32>(value);
  constexpr u32 sign = 0x80000000u;
  const u32 magnitude = bits & 0x7fffffffu;
  if (nan_policy == 0 && magnitude > 0x7f800000u) {
    return 0xffffffffu;
  }
  if (nan_policy == 0 && magnitude == 0u) {
    return sign;
  }
  return (bits & sign) != 0u ? ~bits : bits ^ sign;
}

template <typename T>
__device__ u64 sortable_key64(T value, i32 nan_policy);

template <>
__device__ u64 sortable_key64<u64>(u64 value, i32) {
  return value;
}

template <>
__device__ u64 sortable_key64<i64>(i64 value, i32) {
  return static_cast<u64>(value) ^ 0x8000000000000000ull;
}

template <>
__device__ u64 sortable_key64<double>(double value, i32 nan_policy) {
  const u64 bits = bit_cast<u64>(value);
  constexpr u64 sign = 0x8000000000000000ull;
  const u64 magnitude = bits & 0x7fffffffffffffffull;
  if (nan_policy == 0 && magnitude > 0x7ff0000000000000ull) {
    return 0xffffffffffffffffull;
  }
  if (nan_policy == 0 && magnitude == 0ull) {
    return sign;
  }
  return (bits & sign) != 0ull ? ~bits : bits ^ sign;
}

template <typename T, typename SortableT>
__device__ void radix_init_impl(const u8 *keys,
                                u64 keys_offset,
                                u64 keys_stride,
                                SortableT *sortable,
                                u32 *indices,
                                u32 n,
                                i32 nan_policy) {
  const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) {
    return;
  }
  const T key = load_strided<T>(keys, keys_offset, keys_stride, index);
  if constexpr (sizeof(SortableT) == sizeof(u32)) {
    sortable[index] = sortable_key32<T>(key, nan_policy);
  } else {
    sortable[index] = sortable_key64<T>(key, nan_policy);
  }
  indices[index] = index;
}

#define DEFINE_RADIX_INIT_KERNEL(NAME, KEY_TYPE, SORTABLE_TYPE)             \
  extern "C" __global__ void NAME(const u8 *keys, u64 keys_offset,          \
                                  u64 keys_stride, SORTABLE_TYPE *sortable, \
                                  u32 *indices, u32 n, i32 nan_policy) {    \
    radix_init_impl<KEY_TYPE, SORTABLE_TYPE>(                               \
        keys, keys_offset, keys_stride, sortable, indices, n, nan_policy);  \
  }

DEFINE_RADIX_INIT_KERNEL(radix_init_i32, i32, u32)
DEFINE_RADIX_INIT_KERNEL(radix_init_u32, u32, u32)
DEFINE_RADIX_INIT_KERNEL(radix_init_f32, float, u32)
DEFINE_RADIX_INIT_KERNEL(radix_init_i64, i64, u64)
DEFINE_RADIX_INIT_KERNEL(radix_init_u64, u64, u64)
DEFINE_RADIX_INIT_KERNEL(radix_init_f64, double, u64)

template <typename T>
__device__ void radix_zero_flags_impl(const T *keys,
                                      i32 *prefix,
                                      u32 n,
                                      u32 bit) {
  const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    prefix[index] = ((keys[index] >> bit) & static_cast<T>(1)) == 0 ? 1 : 0;
  }
}

template <typename T>
__device__ void radix_scatter_impl(const T *keys_in,
                                   const u32 *indices_in,
                                   const i32 *prefix,
                                   T *keys_out,
                                   u32 *indices_out,
                                   u32 n,
                                   u32 bit) {
  const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) {
    return;
  }
  const u32 zero_count = static_cast<u32>(prefix[n - 1u]);
  const u32 zeros_through = static_cast<u32>(prefix[index]);
  const bool is_zero = ((keys_in[index] >> bit) & static_cast<T>(1)) == 0;
  const u32 output_index =
      is_zero ? zeros_through - 1u : zero_count + index - zeros_through;
  keys_out[output_index] = keys_in[index];
  indices_out[output_index] = indices_in[index];
}

extern "C" __global__ void radix_zero_flags_u32(const u32 *keys,
                                                i32 *prefix,
                                                u32 n,
                                                u32 bit) {
  radix_zero_flags_impl(keys, prefix, n, bit);
}

extern "C" __global__ void radix_zero_flags_u64(const u64 *keys,
                                                i32 *prefix,
                                                u32 n,
                                                u32 bit) {
  radix_zero_flags_impl(keys, prefix, n, bit);
}

extern "C" __global__ void radix_scatter_u32(const u32 *keys_in,
                                             const u32 *indices_in,
                                             const i32 *prefix,
                                             u32 *keys_out,
                                             u32 *indices_out,
                                             u32 n,
                                             u32 bit) {
  radix_scatter_impl(keys_in, indices_in, prefix, keys_out, indices_out, n,
                     bit);
}

extern "C" __global__ void radix_scatter_u64(const u64 *keys_in,
                                             const u32 *indices_in,
                                             const i32 *prefix,
                                             u64 *keys_out,
                                             u32 *indices_out,
                                             u32 n,
                                             u32 bit) {
  radix_scatter_impl(keys_in, indices_in, prefix, keys_out, indices_out, n,
                     bit);
}

extern "C" __global__ void radix_gather_words(const u8 *src,
                                              u64 src_offset,
                                              u64 src_stride,
                                              const u32 *indices,
                                              u32 *dst,
                                              u32 n,
                                              u32 item_words) {
  const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) {
    return;
  }
  const u32 source_index = indices[index];
  const u32 *source = reinterpret_cast<const u32 *>(
      src + src_offset + static_cast<u64>(source_index) * src_stride);
  u32 *output = dst + static_cast<u64>(index) * item_words;
  for (u32 word = 0; word < item_words; ++word) {
    output[word] = source[word];
  }
}

extern "C" __global__ void radix_copy_words(const u32 *src,
                                            u8 *dst,
                                            u64 dst_offset,
                                            u64 dst_stride,
                                            u32 n,
                                            u32 item_words) {
  const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) {
    return;
  }
  const u32 *source = src + static_cast<u64>(index) * item_words;
  u32 *output = reinterpret_cast<u32 *>(dst + dst_offset +
                                        static_cast<u64>(index) * dst_stride);
  for (u32 word = 0; word < item_words; ++word) {
    output[word] = source[word];
  }
}
