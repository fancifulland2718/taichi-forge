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

__device__ u32 warp_ballot(i32 predicate) {
  u32 result;
  asm("{ .reg .pred p; setp.ne.s32 p, %1, 0; vote.ballot.b32 %0, p; }"
      : "=r"(result)
      : "r"(predicate));
  return result;
}

__device__ u32 popc(u32 value) {
  u32 result;
  asm("popc.b32 %0, %1;" : "=r"(result) : "r"(value));
  return result;
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
__device__ void scan_tiles_impl(u8 *values,
                                u64 offset,
                                u64 stride,
                                u32 n,
                                u8 *block_sums,
                                u64 sums_offset,
                                i32 reverse,
                                T *warp_sums,
                                T *tile_base) {
  const u32 tid = threadIdx.x;
  const u32 lane = tid & 31u;
  const u32 warp = tid >> 5;
  const u32 block = blockIdx.x;
  const u32 tile_begin = block * 1024u;
  if (tid == 0u) {
    *tile_base = identity<T>(0);
  }
  block_barrier();

  for (u32 item = 0; item < 4u; ++item) {
    const T chunk_base = *tile_base;
    // Every warp must snapshot the previous chunk total before warp 7 can
    // publish the current chunk total.
    block_barrier();
    const u32 index = tile_begin + item * 256u + tid;
    const u32 physical_index = reverse != 0 ? n - 1u - index : index;
    T value = index < n
                  ? load_strided<T>(values, offset, stride, physical_index)
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
    value = combine(chunk_base, value, 0);
    if (index < n) {
      store_strided<T>(values, offset, stride, physical_index, value);
    }
    if (tid == 255u) {
      *tile_base = value;
    }
    block_barrier();
  }
  if (tid == 255u && block_sums != nullptr) {
    store_strided<T>(block_sums, sums_offset, sizeof(T), block, *tile_base);
  }
}

template <typename T>
__device__ void uniform_add_impl(u8 *values,
                                 u64 offset,
                                 u64 stride,
                                 u32 n,
                                 const u8 *block_sums,
                                 u64 sums_offset,
                                 i32 reverse,
                                 u32 tile_shift) {
  const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) {
    return;
  }
  const u32 block = index >> tile_shift;
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

#define DEFINE_SCAN_TILE_KERNEL(NAME, TYPE)                                   \
  extern "C" __global__ void NAME(u8 *values, u64 offset, u64 stride, u32 n,  \
                                  u8 *block_sums, u64 sums_offset,            \
                                  i32 reverse) {                              \
    __shared__ TYPE warp_sums[8];                                             \
    __shared__ TYPE tile_base;                                                \
    scan_tiles_impl<TYPE>(values, offset, stride, n, block_sums, sums_offset, \
                          reverse, warp_sums, &tile_base);                    \
  }

#define DEFINE_UNIFORM_KERNEL(NAME, TYPE)                                      \
  extern "C" __global__ void NAME(u8 *values, u64 offset, u64 stride, u32 n,   \
                                  const u8 *block_sums, u64 sums_offset,       \
                                  i32 reverse, u32 tile_shift) {               \
    uniform_add_impl<TYPE>(values, offset, stride, n, block_sums, sums_offset, \
                           reverse, tile_shift);                               \
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

DEFINE_SCAN_TILE_KERNEL(scan_tiles_i32, i32)
DEFINE_SCAN_TILE_KERNEL(scan_tiles_u32, u32)
DEFINE_SCAN_TILE_KERNEL(scan_tiles_f32, float)
DEFINE_SCAN_TILE_KERNEL(scan_tiles_i64, i64)
DEFINE_SCAN_TILE_KERNEL(scan_tiles_u64, u64)
DEFINE_SCAN_TILE_KERNEL(scan_tiles_f64, double)

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

__device__ void sparse_refresh_atomic_min_u32(u32 *address, u32 value) {
  u32 previous;
  asm volatile("atom.global.min.u32 %0, [%1], %2;"
               : "=r"(previous)
               : "l"(address), "r"(value)
               : "memory");
}

__device__ bool sparse_refresh_finite_f32(float value) {
  return (bit_cast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}

__device__ float sparse_refresh_abs_f32(float value) {
  return bit_cast<float>(bit_cast<u32>(value) & 0x7fffffffu);
}

__device__ float sparse_refresh_sqrt_f32(float value) {
  float result;
  asm("sqrt.rn.f32 %0, %1;" : "=f"(result) : "f"(value));
  return result;
}

namespace sparse_minres {

constexpr u32 kDot = 0u;
constexpr u32 kTrueResidualSquared = 1u;
constexpr u32 kRhsSquared = 2u;
constexpr u32 kBeta = 3u;
constexpr u32 kOldBeta = 4u;
constexpr u32 kAlpha = 5u;
constexpr u32 kCosine = 6u;
constexpr u32 kSine = 7u;
constexpr u32 kDbar = 8u;
constexpr u32 kEpsln = 9u;
constexpr u32 kPhibar = 10u;
constexpr u32 kInverseBeta = 11u;
constexpr u32 kOldResidualScale = 12u;
constexpr u32 kAlphaResidualScale = 13u;
constexpr u32 kOldEpsln = 14u;
constexpr u32 kDelta = 15u;
constexpr u32 kInverseGamma = 16u;
constexpr u32 kPhi = 17u;
constexpr u32 kEstimatedResidual = 18u;
constexpr u32 kAbsoluteTolerance = 19u;
constexpr u32 kRelativeTolerance = 20u;
constexpr u32 kRelativeReferenceNorm = 21u;
constexpr u32 kEffectiveTolerance = 22u;
constexpr u32 kToleranceSquared = 23u;
constexpr u32 kInitialResidualSquared = 24u;
constexpr u32 kFloatCount = 25u;

constexpr u32 kStatus = kFloatCount + 0u;
constexpr u32 kCompletedIterations = kFloatCount + 1u;
constexpr u32 kActive = kFloatCount + 2u;
constexpr u32 kUpdateEnabled = kFloatCount + 3u;
constexpr u32 kKrylovClosed = kFloatCount + 4u;
constexpr u32 kHasPreconditioner = kFloatCount + 5u;
constexpr u32 kStopOnEstimate = kFloatCount + 6u;
constexpr u32 kReserved = kFloatCount + 7u;

constexpr i32 kNotRun = -1;
constexpr i32 kMaxIterations = 0;
constexpr i32 kBreakdown = 1;
constexpr i32 kConverged = 2;

__device__ float load_float(const u32 *state, u32 index) {
  return bit_cast<float>(state[index]);
}

__device__ void store_float(u32 *state, u32 index, float value) {
  state[index] = bit_cast<u32>(value);
}

__device__ i32 load_int(const u32 *state, u32 index) {
  return bit_cast<i32>(state[index]);
}

__device__ void store_int(u32 *state, u32 index, i32 value) {
  state[index] = bit_cast<u32>(value);
}

__device__ bool finite(float value) {
  return sparse_refresh_finite_f32(value);
}

__device__ float abs(float value) {
  return sparse_refresh_abs_f32(value);
}

__device__ void fail(u32 *state) {
  store_int(state, kStatus, kBreakdown);
  store_int(state, kActive, 0);
  store_int(state, kUpdateEnabled, 0);
}

}  // namespace sparse_minres

extern "C" __global__ void sparse_minres_scalar_f32(
    const float *initial_residual_squared,
    const float *rhs_squared,
    const float *dot,
    u32 *state,
    float absolute_tolerance,
    float relative_tolerance,
    u32 stage,
    u32 limit_reached,
    u32 has_preconditioner,
    u32 stop_on_estimate) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) {
    return;
  }
  using namespace sparse_minres;
  if (stage == 0u) {
    for (u32 index = 0; index < kFloatCount + 8u; ++index) {
      state[index] = 0u;
    }
    store_float(state, kDot, 1.0f);
    store_int(state, kStatus, kNotRun);
    store_int(state, kActive, 1);
    store_int(state, kHasPreconditioner,
              has_preconditioner != 0u ? 1 : 0);
    store_int(state, kStopOnEstimate, stop_on_estimate != 0u ? 1 : 0);
    const float rr = initial_residual_squared[0];
    const float rhs2 = rhs_squared[0];
    const float beta2 = dot[0];
    store_float(state, kTrueResidualSquared, rr);
    store_float(state, kInitialResidualSquared, rr);
    store_float(state, kRhsSquared, rhs2);
    store_float(state, kAbsoluteTolerance, absolute_tolerance);
    store_float(state, kRelativeTolerance, relative_tolerance);
    if (!finite(rr) || rr < 0.0f || !finite(rhs2) || rhs2 < 0.0f ||
        !finite(beta2) || beta2 < 0.0f) {
      fail(state);
      return;
    }
    const float reference = sparse_refresh_sqrt_f32(rhs2);
    const float relative = relative_tolerance * reference;
    const float tolerance =
        absolute_tolerance > relative ? absolute_tolerance : relative;
    const float tolerance_squared = tolerance * tolerance;
    if (!finite(reference) || !finite(tolerance) ||
        !finite(tolerance_squared)) {
      fail(state);
      return;
    }
    store_float(state, kRelativeReferenceNorm, reference);
    store_float(state, kEffectiveTolerance, tolerance);
    store_float(state, kToleranceSquared, tolerance_squared);
    if (rr <= tolerance_squared) {
      store_int(state, kStatus, kConverged);
      store_int(state, kActive, 0);
      return;
    }
    if (beta2 <= 0.0f) {
      fail(state);
      return;
    }
    const float beta = sparse_refresh_sqrt_f32(beta2);
    if (!finite(beta) || beta <= 0.0f) {
      fail(state);
      return;
    }
    store_float(state, kBeta, beta);
    store_float(state, kCosine, -1.0f);
    store_float(state, kPhibar, beta);
    store_float(state, kEstimatedResidual, beta);
    if (limit_reached != 0u) {
      store_int(state, kStatus, kMaxIterations);
      store_int(state, kActive, 0);
    }
    return;
  }

  store_int(state, kUpdateEnabled, 0);
  if (stage == 4u) {
    if (load_int(state, kStatus) == kBreakdown) {
      return;
    }
    const float rr = dot[0];
    store_float(state, kTrueResidualSquared, rr);
    if (!finite(rr) || rr < 0.0f) {
      fail(state);
      return;
    }
    if (rr <= load_float(state, kToleranceSquared)) {
      store_int(state, kStatus, kConverged);
      store_int(state, kActive, 0);
    } else if (load_int(state, kKrylovClosed) != 0) {
      fail(state);
    } else if (limit_reached != 0u) {
      store_int(state, kStatus, kMaxIterations);
      store_int(state, kActive, 0);
    } else {
      store_int(state, kStatus, kNotRun);
      store_int(state, kActive, 1);
    }
    return;
  }

  if (load_int(state, kActive) == 0) {
    return;
  }
  const float beta = load_float(state, kBeta);
  if (!finite(beta) || beta <= 0.0f) {
    fail(state);
    return;
  }
  if (stage == 1u) {
    store_float(state, kInverseBeta, 1.0f / beta);
    const i32 completed = load_int(state, kCompletedIterations);
    float old_residual_scale = 0.0f;
    if (completed > 0) {
      const float old_beta = load_float(state, kOldBeta);
      if (!finite(old_beta) || old_beta <= 0.0f) {
        fail(state);
        return;
      }
      old_residual_scale = -beta / old_beta;
    }
    store_float(state, kOldResidualScale, old_residual_scale);
    return;
  }
  if (stage == 2u) {
    const float alpha = dot[0];
    if (!finite(alpha)) {
      fail(state);
      return;
    }
    store_float(state, kAlpha, alpha);
    store_float(state, kAlphaResidualScale, -alpha / beta);
    return;
  }
  if (stage != 3u) {
    fail(state);
    return;
  }

  const float beta2 = dot[0];
  if (!finite(beta2) || beta2 < 0.0f) {
    fail(state);
    return;
  }
  const float beta_new = sparse_refresh_sqrt_f32(beta2);
  const float alpha = load_float(state, kAlpha);
  const float cs = load_float(state, kCosine);
  const float sn = load_float(state, kSine);
  const float dbar = load_float(state, kDbar);
  const float epsln = load_float(state, kEpsln);
  const float phibar = load_float(state, kPhibar);
  const float oldeps = epsln;
  const float delta = cs * dbar + sn * alpha;
  const float gbar = sn * dbar - cs * alpha;
  const float next_epsln = sn * beta_new;
  const float next_dbar = -cs * beta_new;
  const float gamma =
      sparse_refresh_sqrt_f32(gbar * gbar + beta_new * beta_new);
  if (!finite(beta_new) || !finite(delta) || !finite(gamma) ||
      gamma <= 0.0f) {
    fail(state);
    return;
  }
  const float next_cs = gbar / gamma;
  const float next_sn = beta_new / gamma;
  const float phi = next_cs * phibar;
  const float next_phibar = next_sn * phibar;
  if (!finite(next_cs) || !finite(next_sn) || !finite(phi) ||
      !finite(next_phibar)) {
    fail(state);
    return;
  }
  store_float(state, kOldEpsln, oldeps);
  store_float(state, kDelta, delta);
  store_float(state, kInverseGamma, 1.0f / gamma);
  store_float(state, kPhi, phi);
  store_float(state, kOldBeta, beta);
  store_float(state, kBeta, beta_new);
  store_float(state, kCosine, next_cs);
  store_float(state, kSine, next_sn);
  store_float(state, kDbar, next_dbar);
  store_float(state, kEpsln, next_epsln);
  store_float(state, kPhibar, next_phibar);
  store_float(state, kEstimatedResidual, abs(next_phibar));
  store_int(state, kCompletedIterations,
            load_int(state, kCompletedIterations) + 1);
  store_int(state, kUpdateEnabled, 1);
  const bool closed = beta_new == 0.0f;
  store_int(state, kKrylovClosed, closed ? 1 : 0);
  const bool provisional_stop =
      load_int(state, kStopOnEstimate) != 0 &&
      abs(next_phibar) <= load_float(state, kEffectiveTolerance);
  store_int(state, kActive, closed || provisional_stop ? 0 : 1);
}

extern "C" __global__ void sparse_minres_vector_state_f32(
    const float *source,
    float *destination,
    const u32 *state,
    u32 n,
    u32 coefficient_index,
    u32 add) {
  const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n ||
      sparse_minres::load_int(state, sparse_minres::kActive) == 0) {
    return;
  }
  const float coefficient =
      sparse_minres::load_float(state, coefficient_index);
  const float value = coefficient * source[index];
  destination[index] = add != 0u ? destination[index] + value : value;
}

extern "C" __global__ void sparse_minres_commit_f32(
    const float *v,
    float *r1,
    float *r2,
    const float *lanczos_residual,
    float *w_older,
    float *w_old,
    float *w,
    float *solution,
    const u32 *state,
    u32 n) {
  const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n ||
      sparse_minres::load_int(state, sparse_minres::kUpdateEnabled) == 0) {
    return;
  }
  const float old_w_old = w_old[index];
  const float old_w = w[index];
  const float new_w =
      (v[index] -
       sparse_minres::load_float(state, sparse_minres::kOldEpsln) *
           old_w_old -
       sparse_minres::load_float(state, sparse_minres::kDelta) * old_w) *
      sparse_minres::load_float(state, sparse_minres::kInverseGamma);
  w_older[index] = old_w_old;
  w_old[index] = old_w;
  w[index] = new_w;
  solution[index] +=
      sparse_minres::load_float(state, sparse_minres::kPhi) * new_w;
  r1[index] = r2[index];
  r2[index] = lanczos_residual[index];
}

namespace sparse_bicgstab {

constexpr u32 kRhoOld = 0u;
constexpr u32 kAlpha = 1u;
constexpr u32 kOmega = 2u;
constexpr u32 kRho = 3u;
constexpr u32 kBeta = 4u;
constexpr u32 kAlphaDenominator = 5u;
constexpr u32 kOmegaNumerator = 6u;
constexpr u32 kOmegaDenominator = 7u;
constexpr u32 kIntermediateResidualSquared = 8u;
constexpr u32 kResidualSquared = 9u;
constexpr u32 kTrueResidualSquared = 10u;
constexpr u32 kInitialResidualSquared = 11u;
constexpr u32 kRhsSquared = 12u;
constexpr u32 kRelativeReferenceNorm = 13u;
constexpr u32 kEffectiveTolerance = 14u;
constexpr u32 kToleranceSquared = 15u;
constexpr u32 kFloatCount = 16u;

constexpr u32 kStatus = kFloatCount + 0u;
constexpr u32 kCompletedIterations = kFloatCount + 1u;
constexpr u32 kActive = kFloatCount + 2u;
constexpr u32 kCommitEnabled = kFloatCount + 3u;
constexpr u32 kFreshDirection = kFloatCount + 4u;
constexpr u32 kStopKind = kFloatCount + 5u;
constexpr u32 kBreakdownReason = kFloatCount + 6u;
constexpr u32 kReconcileMode = kFloatCount + 7u;
constexpr u32 kStateWords = kFloatCount + 8u;

constexpr i32 kNotRun = -1;
constexpr i32 kMaxIterations = 0;
constexpr i32 kBreakdown = 1;
constexpr i32 kConverged = 2;

constexpr i32 kStopNone = 0;
constexpr i32 kStopRestart = 1;
constexpr i32 kStopBreakdown = 2;

constexpr i32 kReasonNone = 0;
constexpr i32 kReasonNonfinite = 1;
constexpr i32 kReasonRho = 2;
constexpr i32 kReasonAlphaDenominator = 3;
constexpr i32 kReasonOmegaDenominator = 4;
constexpr i32 kReasonOmega = 5;

__device__ float load_float(const u32 *state, u32 index) {
  return bit_cast<float>(state[index]);
}

__device__ void store_float(u32 *state, u32 index, float value) {
  state[index] = bit_cast<u32>(value);
}

__device__ i32 load_int(const u32 *state, u32 index) {
  return bit_cast<i32>(state[index]);
}

__device__ void store_int(u32 *state, u32 index, i32 value) {
  state[index] = bit_cast<u32>(value);
}

__device__ bool finite(float value) {
  return sparse_refresh_finite_f32(value);
}

__device__ void fail(u32 *state, i32 reason) {
  store_int(state, kStatus, kBreakdown);
  store_int(state, kActive, 0);
  store_int(state, kCommitEnabled, 0);
  store_int(state, kStopKind, kStopBreakdown);
  store_int(state, kBreakdownReason, reason);
}

}  // namespace sparse_bicgstab

extern "C" __global__ void sparse_bicgstab_scalar_f32(
    const float *initial_residual_squared,
    const float *rhs_squared,
    const float *dot0,
    const float *dot1,
    u32 *state,
    float absolute_tolerance,
    float relative_tolerance,
    u32 stage,
    u32 limit_reached) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) {
    return;
  }
  using namespace sparse_bicgstab;
  if (stage == 0u) {
    for (u32 index = 0; index < kStateWords; ++index) {
      state[index] = 0u;
    }
    store_float(state, kRhoOld, 1.0f);
    store_float(state, kAlpha, 1.0f);
    store_float(state, kOmega, 1.0f);
    store_int(state, kStatus, kNotRun);
    store_int(state, kActive, 1);
    store_int(state, kFreshDirection, 1);
    store_int(state, kReconcileMode, 1);
    const float rr = initial_residual_squared[0];
    const float rhs2 = rhs_squared[0];
    store_float(state, kTrueResidualSquared, rr);
    store_float(state, kInitialResidualSquared, rr);
    store_float(state, kRhsSquared, rhs2);
    if (!finite(rr) || rr < 0.0f || !finite(rhs2) || rhs2 < 0.0f) {
      fail(state, kReasonNonfinite);
      store_int(state, kReconcileMode, 0);
      return;
    }
    const float reference = sparse_refresh_sqrt_f32(rhs2);
    const float relative = relative_tolerance * reference;
    const float tolerance =
        absolute_tolerance > relative ? absolute_tolerance : relative;
    const float tolerance_squared = tolerance * tolerance;
    if (!finite(reference) || !finite(tolerance) ||
        !finite(tolerance_squared)) {
      fail(state, kReasonNonfinite);
      store_int(state, kReconcileMode, 0);
      return;
    }
    store_float(state, kRelativeReferenceNorm, reference);
    store_float(state, kEffectiveTolerance, tolerance);
    store_float(state, kToleranceSquared, tolerance_squared);
    if (rr <= tolerance_squared) {
      store_int(state, kStatus, kConverged);
      store_int(state, kActive, 0);
      store_int(state, kReconcileMode, 0);
    } else if (limit_reached != 0u) {
      store_int(state, kStatus, kMaxIterations);
      store_int(state, kActive, 0);
      store_int(state, kReconcileMode, 0);
    } else if (rhs2 == 0.0f) {
      store_float(state, kTrueResidualSquared, 0.0f);
      store_int(state, kStatus, kConverged);
      store_int(state, kActive, 0);
      store_int(state, kReconcileMode, 2);
    }
    return;
  }

  if (stage == 6u) {
    const float rr = dot0[0];
    store_float(state, kTrueResidualSquared, rr);
    store_int(state, kReconcileMode, 0);
    if (!finite(rr) || rr < 0.0f) {
      fail(state, kReasonNonfinite);
      return;
    }
    if (rr <= load_float(state, kToleranceSquared)) {
      store_int(state, kStatus, kConverged);
      store_int(state, kActive, 0);
      store_int(state, kBreakdownReason, kReasonNone);
      return;
    }
    const i32 stop_kind = load_int(state, kStopKind);
    if (stop_kind == kStopRestart) {
      store_float(state, kRhoOld, 1.0f);
      store_float(state, kAlpha, 1.0f);
      store_float(state, kOmega, 1.0f);
      store_int(state, kStatus, kNotRun);
      store_int(state, kActive, 1);
      store_int(state, kFreshDirection, 1);
      store_int(state, kStopKind, kStopNone);
      store_int(state, kBreakdownReason, kReasonNone);
      store_int(state, kReconcileMode, 1);
      return;
    }
    if (stop_kind == kStopBreakdown ||
        load_int(state, kStatus) == kBreakdown) {
      store_int(state, kStatus, kBreakdown);
      store_int(state, kActive, 0);
      return;
    }
    if (limit_reached != 0u) {
      store_int(state, kStatus, kMaxIterations);
      store_int(state, kActive, 0);
    } else {
      store_int(state, kStatus, kNotRun);
      store_int(state, kActive, 1);
    }
    return;
  }

  if (load_int(state, kActive) == 0) {
    return;
  }
  if (stage == 1u) {
    const float rho = dot0[0];
    store_float(state, kRho, rho);
    if (!finite(rho)) {
      fail(state, kReasonNonfinite);
      return;
    }
    if (rho == 0.0f) {
      store_int(state, kActive, 0);
      store_int(state, kStopKind, kStopRestart);
      store_int(state, kBreakdownReason, kReasonRho);
      return;
    }
    if (load_int(state, kFreshDirection) != 0) {
      store_float(state, kBeta, 0.0f);
      return;
    }
    const float rho_old = load_float(state, kRhoOld);
    const float alpha = load_float(state, kAlpha);
    const float omega = load_float(state, kOmega);
    if (rho_old == 0.0f) {
      fail(state, kReasonRho);
      return;
    }
    if (omega == 0.0f) {
      fail(state, kReasonOmega);
      return;
    }
    const float beta = (rho / rho_old) * (alpha / omega);
    if (!finite(beta)) {
      fail(state, kReasonNonfinite);
      return;
    }
    store_float(state, kBeta, beta);
    return;
  }
  if (stage == 2u) {
    const float denominator = dot0[0];
    store_float(state, kAlphaDenominator, denominator);
    if (!finite(denominator)) {
      fail(state, kReasonNonfinite);
      return;
    }
    if (denominator == 0.0f) {
      fail(state, kReasonAlphaDenominator);
      return;
    }
    const float alpha = load_float(state, kRho) / denominator;
    if (!finite(alpha)) {
      fail(state, kReasonNonfinite);
      return;
    }
    store_float(state, kAlpha, alpha);
    return;
  }
  if (stage == 3u) {
    const float rr = dot0[0];
    store_float(state, kIntermediateResidualSquared, rr);
    if (!finite(rr) || rr < 0.0f) {
      fail(state, kReasonNonfinite);
      return;
    }
    if (rr <= load_float(state, kToleranceSquared)) {
      store_int(state, kStopKind, kStopRestart);
      store_int(state, kBreakdownReason, kReasonNone);
    }
    return;
  }
  if (stage == 4u) {
    float omega = 0.0f;
    if (load_int(state, kStopKind) == kStopNone) {
      const float numerator = dot0[0];
      const float denominator = dot1[0];
      store_float(state, kOmegaNumerator, numerator);
      store_float(state, kOmegaDenominator, denominator);
      if (!finite(numerator) || !finite(denominator)) {
        fail(state, kReasonNonfinite);
        return;
      }
      if (denominator == 0.0f) {
        store_int(state, kStopKind, kStopBreakdown);
        store_int(state, kBreakdownReason, kReasonOmegaDenominator);
      } else {
        omega = numerator / denominator;
        if (!finite(omega)) {
          fail(state, kReasonNonfinite);
          return;
        }
        if (omega == 0.0f) {
          store_int(state, kStopKind, kStopBreakdown);
          store_int(state, kBreakdownReason, kReasonOmega);
        }
      }
    }
    store_float(state, kOmega, omega);
    store_int(state, kCommitEnabled, 1);
    store_int(state, kFreshDirection, 0);
    store_int(state, kCompletedIterations,
              load_int(state, kCompletedIterations) + 1);
    return;
  }
  if (stage == 5u) {
    store_int(state, kCommitEnabled, 0);
    const float rr = dot0[0];
    store_float(state, kResidualSquared, rr);
    if (!finite(rr) || rr < 0.0f) {
      fail(state, kReasonNonfinite);
      return;
    }
    if (load_int(state, kStopKind) != kStopNone) {
      store_int(state, kActive, 0);
      return;
    }
    if (rr <= load_float(state, kToleranceSquared)) {
      store_int(state, kStopKind, kStopRestart);
      store_int(state, kActive, 0);
      return;
    }
    store_float(state, kRhoOld, load_float(state, kRho));
    return;
  }
  fail(state, kReasonNonfinite);
}

extern "C" __global__ void sparse_bicgstab_direction_f32(
    const float *residual,
    float *direction,
    const float *operator_direction,
    const u32 *state,
    u32 n) {
  const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n || sparse_bicgstab::load_int(
                        state, sparse_bicgstab::kActive) == 0) {
    return;
  }
  if (sparse_bicgstab::load_int(
          state, sparse_bicgstab::kFreshDirection) != 0) {
    direction[index] = residual[index];
    return;
  }
  const float beta = sparse_bicgstab::load_float(
      state, sparse_bicgstab::kBeta);
  const float omega = sparse_bicgstab::load_float(
      state, sparse_bicgstab::kOmega);
  direction[index] = residual[index] +
                     beta * (direction[index] -
                             omega * operator_direction[index]);
}

extern "C" __global__ void sparse_bicgstab_intermediate_f32(
    const float *residual,
    const float *operator_direction,
    float *intermediate,
    const u32 *state,
    u32 n) {
  const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n || sparse_bicgstab::load_int(
                        state, sparse_bicgstab::kActive) == 0) {
    return;
  }
  intermediate[index] = residual[index] -
      sparse_bicgstab::load_float(state, sparse_bicgstab::kAlpha) *
          operator_direction[index];
}

extern "C" __global__ void sparse_bicgstab_commit_f32(
    const float *solution_direction,
    const float *solution_intermediate,
    const float *intermediate,
    const float *operator_intermediate,
    float *solution,
    float *residual,
    const u32 *state,
    u32 n) {
  const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n || sparse_bicgstab::load_int(
                        state, sparse_bicgstab::kCommitEnabled) == 0) {
    return;
  }
  const float alpha = sparse_bicgstab::load_float(
      state, sparse_bicgstab::kAlpha);
  const float omega = sparse_bicgstab::load_float(
      state, sparse_bicgstab::kOmega);
  solution[index] += alpha * solution_direction[index] +
                     omega * solution_intermediate[index];
  residual[index] = intermediate[index] -
                    omega * operator_intermediate[index];
}

extern "C" __global__ void sparse_bicgstab_reconcile_f32(
    const float *true_residual,
    float *residual,
    float *shadow_residual,
    float *direction,
    float *operator_direction,
    float *solution,
    const u32 *state,
    u32 n) {
  const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
  const i32 mode = sparse_bicgstab::load_int(
      state, sparse_bicgstab::kReconcileMode);
  if (index >= n || mode == 0) {
    return;
  }
  if (mode == 2) {
    solution[index] = 0.0f;
    residual[index] = 0.0f;
    shadow_residual[index] = 0.0f;
  } else {
    residual[index] = true_residual[index];
    shadow_residual[index] = true_residual[index];
  }
  direction[index] = 0.0f;
  operator_direction[index] = 0.0f;
}

namespace sparse_gmres {

constexpr u32 kInitialResidualSquared = 0u;
constexpr u32 kTrueResidualSquared = 1u;
constexpr u32 kRhsSquared = 2u;
constexpr u32 kRelativeReferenceNorm = 3u;
constexpr u32 kEffectiveTolerance = 4u;
constexpr u32 kToleranceSquared = 5u;
constexpr u32 kBeta = 6u;
constexpr u32 kEstimatedResidual = 7u;
constexpr u32 kPreorthogonalNorm = 8u;
constexpr u32 kNextNorm = 9u;
constexpr u32 kFloatCount = 16u;
constexpr u32 kStatus = kFloatCount + 0u;
constexpr u32 kCompletedIterations = kFloatCount + 1u;
constexpr u32 kSolveActive = kFloatCount + 2u;
constexpr u32 kCycleActive = kFloatCount + 3u;
constexpr u32 kCycleCompleted = kFloatCount + 4u;
constexpr u32 kHappy = kFloatCount + 5u;
constexpr u32 kBreakdownReason = kFloatCount + 6u;
constexpr u32 kCommitEnabled = kFloatCount + 7u;
constexpr u32 kRestartCycles = kFloatCount + 8u;
constexpr u32 kHappyBreakdowns = kFloatCount + 9u;
constexpr u32 kStateWords = 32u;

constexpr i32 kNotRun = -1;
constexpr i32 kMaxIterations = 0;
constexpr i32 kBreakdown = 1;
constexpr i32 kConverged = 2;
constexpr i32 kReasonNone = 0;
constexpr i32 kReasonNonfinite = 1;
constexpr i32 kReasonArnoldi = 6;
constexpr i32 kReasonOrthogonalization = 7;
constexpr i32 kReasonHessenberg = 8;

__device__ float load_float(const u32 *state, u32 index) {
  return bit_cast<float>(state[index]);
}

__device__ void store_float(u32 *state, u32 index, float value) {
  state[index] = bit_cast<u32>(value);
}

__device__ i32 load_int(const u32 *state, u32 index) {
  return bit_cast<i32>(state[index]);
}

__device__ void store_int(u32 *state, u32 index, i32 value) {
  state[index] = bit_cast<u32>(value);
}

__device__ bool finite(float value) {
  return sparse_refresh_finite_f32(value);
}

__device__ float abs(float value) {
  return value < 0.0f ? -value : value;
}

__device__ float max(float lhs, float rhs) {
  return lhs > rhs ? lhs : rhs;
}

__device__ void fail(u32 *state, i32 reason) {
  store_int(state, kStatus, kBreakdown);
  store_int(state, kSolveActive, 0);
  store_int(state, kCycleActive, 0);
  store_int(state, kCommitEnabled, 0);
  store_int(state, kBreakdownReason, reason);
}

}  // namespace sparse_gmres

// Computes every Arnoldi inner product while loading each work element once.
// The group-major partials are finalized without floating-point atomics so the
// primitive has the same numerical contract on CUDA and Vulkan.
extern "C" __global__ void sparse_gmres_multi_dot_partial_f32(
    const float *basis,
    const float *work,
    float *partials,
    const u32 *state,
    u32 n,
    u32 basis_stride,
    u32 basis_count,
    u32 group_count) {
  __shared__ float partial_sum[256];
  const u32 local = threadIdx.x;
  const u32 group = blockIdx.x;
  if (group >= group_count || basis_count == 0u || basis_count > 32u) {
    return;
  }
  float sums[32];
  for (u32 row = 0u; row < basis_count; ++row) {
    sums[row] = 0.0f;
  }
  if (sparse_gmres::load_int(state, sparse_gmres::kCycleActive) != 0) {
    u32 block_begin = group * 1024u;
    const u32 block_stride = group_count * 1024u;
    while (block_begin < n) {
      const u32 remaining = n - block_begin;
      const u32 block_size = remaining < 1024u ? remaining : 1024u;
      const u32 block_end = block_begin + block_size;
      for (u32 index = block_begin + local; index < block_end;
           index += 256u) {
        const float value = work[index];
        for (u32 row = 0u; row < basis_count; ++row) {
          sums[row] += basis[row * basis_stride + index] * value;
        }
      }
      if (remaining <= block_stride) {
        break;
      }
      block_begin += block_stride;
    }
  }
  for (u32 row = 0u; row < basis_count; ++row) {
    partial_sum[local] = sums[row];
    block_barrier();
    for (u32 stride = 128u; stride > 0u; stride >>= 1u) {
      if (local < stride) {
        partial_sum[local] += partial_sum[local + stride];
      }
      block_barrier();
    }
    if (local == 0u) {
      partials[row * group_count + group] = partial_sum[0];
    }
    block_barrier();
  }
}

extern "C" __global__ void sparse_gmres_multi_dot_final_f32(
    const float *partials,
    float *projection,
    const u32 *state,
    u32 group_count,
    u32 basis_count) {
  __shared__ float partial_sum[256];
  const u32 row = blockIdx.x;
  const u32 local = threadIdx.x;
  if (row >= basis_count) {
    return;
  }
  float sum = 0.0f;
  if (sparse_gmres::load_int(state, sparse_gmres::kCycleActive) != 0) {
    for (u32 group = local; group < group_count; group += 256u) {
      sum += partials[row * group_count + group];
    }
  }
  partial_sum[local] = sum;
  block_barrier();
  for (u32 stride = 128u; stride > 0u; stride >>= 1u) {
    if (local < stride) {
      partial_sum[local] += partial_sum[local + stride];
    }
    block_barrier();
  }
  if (local == 0u) {
    projection[row] = partial_sum[0];
  }
}

extern "C" __global__ void sparse_gmres_projection_f32(
    const float *basis,
    float *work,
    const float *projection,
    float *hessenberg,
    const u32 *state,
    u32 n,
    u32 basis_stride,
    u32 restart,
    u32 step,
    u32 pass) {
  const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index == 0u &&
      sparse_gmres::load_int(state, sparse_gmres::kCycleActive) != 0) {
    for (u32 row = 0u; row <= step; ++row) {
      float &entry = hessenberg[row * restart + step];
      entry = pass == 0u ? projection[row] : entry + projection[row];
    }
  }
  if (index >= n ||
      sparse_gmres::load_int(state, sparse_gmres::kCycleActive) == 0) {
    return;
  }
  float value = work[index];
  for (u32 row = 0u; row <= step; ++row) {
    value -= projection[row] * basis[row * basis_stride + index];
  }
  work[index] = value;
}

extern "C" __global__ void sparse_gmres_basis_f32(
    const float *source,
    float *basis,
    float *current,
    const u32 *state,
    u32 n,
    u32 basis_stride,
    u32 row,
    u32 mode) {
  const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) {
    return;
  }
  const bool active = mode == 0u
      ? sparse_gmres::load_int(state, sparse_gmres::kSolveActive) != 0
      : sparse_gmres::load_int(state, sparse_gmres::kCycleActive) != 0;
  const float denominator = sparse_gmres::load_float(
      state, mode == 0u ? sparse_gmres::kBeta : sparse_gmres::kNextNorm);
  const float value =
      active && denominator != 0.0f ? source[index] / denominator : 0.0f;
  basis[row * basis_stride + index] = value;
  current[index] = value;
}

extern "C" __global__ void sparse_gmres_combine_f32(
    const float *basis,
    const float *coefficients,
    float *update,
    const u32 *state,
    u32 n,
    u32 basis_stride) {
  const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) {
    return;
  }
  float value = 0.0f;
  if (sparse_gmres::load_int(state, sparse_gmres::kCommitEnabled) != 0) {
    const u32 used = static_cast<u32>(sparse_gmres::load_int(
        state, sparse_gmres::kCycleCompleted));
    for (u32 row = 0u; row < used; ++row) {
      value += coefficients[row] * basis[row * basis_stride + index];
    }
  }
  update[index] = value;
}

extern "C" __global__ void sparse_gmres_scalar_f32(
    const float *initial_residual_squared,
    const float *rhs_squared,
    const float *dot0,
    const float *dot1,
    float *hessenberg,
    float *cosines,
    float *sines,
    float *g,
    float *coefficients,
    u32 *state,
    float absolute_tolerance,
    float relative_tolerance,
    u32 restart,
    u32 max_iterations,
    u32 stage,
    u32 step,
    u32 limit_reached) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) {
    return;
  }
  using namespace sparse_gmres;
  if (stage == 0u) {
    for (u32 index = 0u; index < kStateWords; ++index) {
      state[index] = 0u;
    }
    store_int(state, kStatus, kNotRun);
    const float rr = initial_residual_squared[0];
    const float rhs2 = rhs_squared[0];
    store_float(state, kInitialResidualSquared, rr);
    store_float(state, kTrueResidualSquared, rr);
    store_float(state, kRhsSquared, rhs2);
    if (!finite(rr) || rr < 0.0f || !finite(rhs2) || rhs2 < 0.0f) {
      fail(state, kReasonNonfinite);
      return;
    }
    const float reference = sparse_refresh_sqrt_f32(rhs2);
    const float relative = relative_tolerance * reference;
    const float tolerance =
        absolute_tolerance > relative ? absolute_tolerance : relative;
    const float tolerance_squared = tolerance * tolerance;
    store_float(state, kRelativeReferenceNorm, reference);
    store_float(state, kEffectiveTolerance, tolerance);
    store_float(state, kToleranceSquared, tolerance_squared);
    store_float(state, kEstimatedResidual, sparse_refresh_sqrt_f32(rr));
    if (!finite(reference) || !finite(tolerance) ||
        !finite(tolerance_squared)) {
      fail(state, kReasonNonfinite);
    } else if (rr <= tolerance_squared) {
      store_int(state, kStatus, kConverged);
    } else if (max_iterations == 0u || limit_reached != 0u) {
      store_int(state, kStatus, kMaxIterations);
    } else {
      store_int(state, kSolveActive, 1);
    }
    return;
  }
  if (stage == 1u) {
    store_int(state, kCycleCompleted, 0);
    store_int(state, kHappy, 0);
    store_int(state, kCommitEnabled, 0);
    for (u32 index = 0u; index < restart * (restart + 1u); ++index) {
      hessenberg[index] = 0.0f;
    }
    for (u32 index = 0u; index < restart; ++index) {
      cosines[index] = 0.0f;
      sines[index] = 0.0f;
      coefficients[index] = 0.0f;
      g[index] = 0.0f;
    }
    g[restart] = 0.0f;
    if (load_int(state, kSolveActive) == 0) {
      store_int(state, kCycleActive, 0);
      return;
    }
    const float beta = sparse_refresh_sqrt_f32(
        load_float(state, kTrueResidualSquared));
    if (!finite(beta) || beta == 0.0f) {
      fail(state, kReasonOrthogonalization);
      return;
    }
    store_float(state, kBeta, beta);
    store_float(state, kEstimatedResidual, beta);
    g[0] = beta;
    store_int(state, kCycleActive, 1);
    return;
  }
  if (stage == 2u) {
    if (load_int(state, kCycleActive) == 0) {
      return;
    }
    const float pre_squared = dot0[0];
    const float next_squared = dot1[0];
    if (!finite(pre_squared) || pre_squared < 0.0f ||
        !finite(next_squared) || next_squared < 0.0f) {
      fail(state, kReasonOrthogonalization);
      return;
    }
    const float pre_norm = sparse_refresh_sqrt_f32(pre_squared);
    const float next_norm = sparse_refresh_sqrt_f32(next_squared);
    if (!finite(pre_norm) || !finite(next_norm)) {
      fail(state, kReasonOrthogonalization);
      return;
    }
    store_float(state, kPreorthogonalNorm, pre_norm);
    store_float(state, kNextNorm, next_norm);
    hessenberg[(step + 1u) * restart + step] = next_norm;
    for (u32 row = 0u; row < step; ++row) {
      const float upper = hessenberg[row * restart + step];
      const float lower = hessenberg[(row + 1u) * restart + step];
      hessenberg[row * restart + step] =
          cosines[row] * upper + sines[row] * lower;
      hessenberg[(row + 1u) * restart + step] =
          -sines[row] * upper + cosines[row] * lower;
    }
    const float diagonal = hessenberg[step * restart + step];
    const float magnitude = sparse_refresh_sqrt_f32(
        diagonal * diagonal + next_norm * next_norm);
    if (!finite(diagonal) || !finite(magnitude)) {
      fail(state, kReasonOrthogonalization);
      return;
    }
    const bool happy = next_norm <=
        7.62939453125e-6f * max(pre_norm, 1.1754943508222875e-38f);
    float cosine = 1.0f;
    float sine = 0.0f;
    if (magnitude != 0.0f) {
      cosine = diagonal / magnitude;
      sine = next_norm / magnitude;
    }
    cosines[step] = cosine;
    sines[step] = sine;
    hessenberg[step * restart + step] = magnitude;
    hessenberg[(step + 1u) * restart + step] = 0.0f;
    const float previous_g = g[step];
    g[step] = cosine * previous_g;
    g[step + 1u] = -sine * previous_g;
    const float estimate = abs(g[step + 1u]);
    store_float(state, kEstimatedResidual, estimate);
    store_int(state, kCompletedIterations,
              load_int(state, kCompletedIterations) + 1);
    store_int(state, kCycleCompleted,
              load_int(state, kCycleCompleted) + 1);
    store_int(state, kHappy, happy ? 1 : 0);
    const bool converged =
        estimate <= load_float(state, kEffectiveTolerance);
    const bool at_limit = static_cast<u32>(
        load_int(state, kCompletedIterations)) >= max_iterations;
    if (happy || converged || at_limit || step + 1u >= restart) {
      store_int(state, kCycleActive, 0);
    }
    return;
  }
  if (stage == 3u) {
    store_int(state, kCommitEnabled, 0);
    if (load_int(state, kSolveActive) == 0) {
      return;
    }
    const i32 signed_used = load_int(state, kCycleCompleted);
    if (signed_used <= 0 || static_cast<u32>(signed_used) > restart) {
      fail(state, kReasonHessenberg);
      return;
    }
    const u32 used = static_cast<u32>(signed_used);
    for (i32 row = signed_used - 1; row >= 0; --row) {
      float value = g[static_cast<u32>(row)];
      float row_scale = 1.0f;
      for (u32 column = static_cast<u32>(row) + 1u; column < used;
           ++column) {
        const float entry =
            hessenberg[static_cast<u32>(row) * restart + column];
        value -= entry * coefficients[column];
        row_scale = max(row_scale, abs(entry));
      }
      const float pivot =
          hessenberg[static_cast<u32>(row) * restart +
                     static_cast<u32>(row)];
      if (!finite(value) || !finite(pivot) ||
          abs(pivot) <= 3.814697265625e-6f * row_scale) {
        fail(state, kReasonHessenberg);
        return;
      }
      coefficients[static_cast<u32>(row)] = value / pivot;
      if (!finite(coefficients[static_cast<u32>(row)])) {
        fail(state, kReasonHessenberg);
        return;
      }
    }
    store_int(state, kCommitEnabled, 1);
    return;
  }
  if (stage == 4u) {
    store_int(state, kCommitEnabled, 0);
    store_int(state, kRestartCycles,
              load_int(state, kRestartCycles) + 1);
    const float rr = dot0[0];
    store_float(state, kTrueResidualSquared, rr);
    if (!finite(rr) || rr < 0.0f) {
      fail(state, kReasonNonfinite);
      return;
    }
    if (rr <= load_float(state, kToleranceSquared)) {
      if (load_int(state, kHappy) != 0) {
        store_int(state, kHappyBreakdowns,
                  load_int(state, kHappyBreakdowns) + 1);
      }
      store_int(state, kStatus, kConverged);
      store_int(state, kSolveActive, 0);
      store_int(state, kCycleActive, 0);
      store_int(state, kBreakdownReason, kReasonNone);
      return;
    }
    if (load_int(state, kStatus) == kBreakdown) {
      store_int(state, kSolveActive, 0);
      return;
    }
    if (load_int(state, kHappy) != 0) {
      fail(state, kReasonArnoldi);
      return;
    }
    if (static_cast<u32>(load_int(state, kCompletedIterations)) >=
            max_iterations ||
        limit_reached != 0u) {
      store_int(state, kStatus, kMaxIterations);
      store_int(state, kSolveActive, 0);
      store_int(state, kCycleActive, 0);
      return;
    }
    store_int(state, kStatus, kNotRun);
    store_int(state, kSolveActive, 1);
    store_int(state, kCycleActive, 0);
    store_int(state, kBreakdownReason, kReasonNone);
    return;
  }
  fail(state, kReasonNonfinite);
}

extern "C" __global__ void sparse_diagonal_refresh_f32(
    const float *values,
    const i32 *diagonal_offsets,
    float *staging_inverse,
    u32 *status,
    u32 rows,
    u32 nnz) {
  const u32 row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }
  constexpr u32 kReasonDiagonalNotFinite = 1u;
  constexpr u32 kReasonDiagonalZero = 2u;
  constexpr u32 kReasonInverseNotFinite = 3u;
  constexpr u32 kReasonOffsetInvalid = 4u;
  const i32 signed_offset = diagonal_offsets[row];
  if (signed_offset < 0 || static_cast<u32>(signed_offset) >= nnz) {
    sparse_refresh_atomic_min_u32(
        status, (row << 3u) | kReasonOffsetInvalid);
    return;
  }
  const float diagonal = values[static_cast<u32>(signed_offset)];
  if (!sparse_refresh_finite_f32(diagonal)) {
    sparse_refresh_atomic_min_u32(
        status, (row << 3u) | kReasonDiagonalNotFinite);
    return;
  }
  if ((bit_cast<u32>(diagonal) & 0x7fffffffu) == 0u) {
    sparse_refresh_atomic_min_u32(
        status, (row << 3u) | kReasonDiagonalZero);
    return;
  }
  const float inverse = 1.0f / diagonal;
  if (!sparse_refresh_finite_f32(inverse)) {
    sparse_refresh_atomic_min_u32(
        status, (row << 3u) | kReasonInverseNotFinite);
    return;
  }
  staging_inverse[row] = inverse;
}

extern "C" __global__ void sparse_diagonal_apply_f32(
    const float *inverse_diagonal,
    const float *input,
    float *output,
    u32 n) {
  const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    output[index] = inverse_diagonal[index] * input[index];
  }
}

extern "C" __global__ void sparse_block_cholesky_refresh_f32(
    const float *values,
    const i32 *diagonal_block_offsets,
    float *staging_factors,
    u32 *status,
    u32 block_rows,
    u32 block_nnz,
    u32 block_size) {
  const u32 block_row = blockIdx.x * blockDim.x + threadIdx.x;
  if (block_row >= block_rows) {
    return;
  }
  constexpr u32 kReasonValueNotFinite = 1u;
  constexpr u32 kReasonNotSymmetric = 2u;
  constexpr u32 kReasonNotPositiveDefinite = 3u;
  constexpr u32 kReasonFactorNotFinite = 4u;
  constexpr u32 kReasonOffsetInvalid = 5u;
  constexpr float kSymmetryEpsilon = 3.814697265625e-6f;
  const i32 signed_offset = diagonal_block_offsets[block_row];
  if (signed_offset < 0 || static_cast<u32>(signed_offset) >= block_nnz) {
    sparse_refresh_atomic_min_u32(
        status, (block_row << 3u) | kReasonOffsetInvalid);
    return;
  }
  float factor[144];
  const u32 block_width = block_size * block_size;
  const u32 source_base = static_cast<u32>(signed_offset) * block_width;
  for (u32 index = 0; index < block_width; ++index) {
    const float value = values[source_base + index];
    if (!sparse_refresh_finite_f32(value)) {
      sparse_refresh_atomic_min_u32(
          status, (block_row << 3u) | kReasonValueNotFinite);
      return;
    }
    factor[index] = value;
  }
  for (u32 row = 1; row < block_size; ++row) {
    for (u32 column = 0; column < row; ++column) {
      const float lower = factor[row * block_size + column];
      const float upper = factor[column * block_size + row];
      const float lower_abs = sparse_refresh_abs_f32(lower);
      const float upper_abs = sparse_refresh_abs_f32(upper);
      float scale = lower_abs > upper_abs ? lower_abs : upper_abs;
      scale = scale > 1.0f ? scale : 1.0f;
      if (sparse_refresh_abs_f32(lower - upper) >
          kSymmetryEpsilon * scale) {
        sparse_refresh_atomic_min_u32(
            status, (block_row << 3u) | kReasonNotSymmetric);
        return;
      }
    }
  }
  for (u32 row = 0; row < block_size; ++row) {
    for (u32 column = 0; column <= row; ++column) {
      float value = factor[row * block_size + column];
      for (u32 k = 0; k < column; ++k) {
        value -= factor[row * block_size + k] *
                 factor[column * block_size + k];
      }
      if (row == column) {
        if (!sparse_refresh_finite_f32(value) || value <= 0.0f) {
          sparse_refresh_atomic_min_u32(
              status,
              (block_row << 3u) | kReasonNotPositiveDefinite);
          return;
        }
        value = sparse_refresh_sqrt_f32(value);
      } else {
        value /= factor[column * block_size + column];
      }
      if (!sparse_refresh_finite_f32(value)) {
        sparse_refresh_atomic_min_u32(
            status, (block_row << 3u) | kReasonFactorNotFinite);
        return;
      }
      factor[row * block_size + column] = value;
    }
  }
  const u32 target_base = block_row * block_width;
  for (u32 row = 0; row < block_size; ++row) {
    for (u32 column = 0; column < block_size; ++column) {
      staging_factors[target_base + row * block_size + column] =
          column <= row ? factor[row * block_size + column] : 0.0f;
    }
  }
}

extern "C" __global__ void sparse_block_diagonal_apply_f32(
    const float *factor_blocks,
    const float *input,
    float *output,
    u32 block_rows,
    u32 block_size) {
  const u32 block_row = blockIdx.x * blockDim.x + threadIdx.x;
  if (block_row >= block_rows) {
    return;
  }
  float local_solution[12] = {0.0f};
  const u32 vector_base = block_row * block_size;
  const u32 block_base = block_row * block_size * block_size;
  // Forward solve: L y = b.
  for (u32 row = 0; row < block_size; ++row) {
    float value = input[vector_base + row];
    for (u32 column = 0; column < row; ++column) {
      value -= factor_blocks[block_base + row * block_size + column] *
               local_solution[column];
    }
    local_solution[row] =
        value / factor_blocks[block_base + row * block_size + row];
  }
  // Backward solve: L^T x = y.
  for (i32 row = static_cast<i32>(block_size) - 1; row >= 0; --row) {
    float value = local_solution[row];
    for (u32 column = static_cast<u32>(row) + 1; column < block_size;
         ++column) {
      value -= factor_blocks[block_base + column * block_size + row] *
               local_solution[column];
    }
    local_solution[row] =
        value / factor_blocks[block_base + row * block_size + row];
  }
  for (u32 row = 0; row < block_size; ++row) {
    output[vector_base + row] = local_solution[row];
  }
}

extern "C" __global__ void compact_rank_tiles_i32(const u8 *flags,
                                                  u64 flags_offset,
                                                  u64 flags_stride,
                                                  i32 *local_prefix,
                                                  i32 *tile_counts,
                                                  u32 n) {
  __shared__ i32 warp_sums[8];
  __shared__ i32 tile_base;
  const u32 tid = threadIdx.x;
  const u32 lane = tid & 31u;
  const u32 warp = tid >> 5;
  const u32 tile = blockIdx.x;
  const u32 tile_begin = tile * 1024u;
  if (tid == 0u) {
    tile_base = 0;
  }
  block_barrier();

  for (u32 item = 0; item < 4u; ++item) {
    const i32 chunk_base = tile_base;
    block_barrier();
    const u32 index = tile_begin + item * 256u + tid;
    i32 value = index < n &&
                        load_strided<i32>(flags, flags_offset, flags_stride,
                                          index) != 0
                    ? 1
                    : 0;
    value = warp_inclusive_sum(value);
    if (lane == 31u) {
      warp_sums[warp] = value;
    }
    block_barrier();
    if (warp == 0u) {
      i32 warp_value = lane < 8u ? warp_sums[lane] : 0;
      warp_value = warp_inclusive_sum(warp_value);
      if (lane < 8u) {
        warp_sums[lane] = warp_value;
      }
    }
    block_barrier();
    if (warp > 0u) {
      value += warp_sums[warp - 1u];
    }
    value += chunk_base;
    if (index < n) {
      local_prefix[index] = value;
    }
    if (tid == 255u) {
      tile_base = value;
    }
    block_barrier();
  }
  if (tid == 255u) {
    tile_counts[tile] = tile_base;
  }
}

extern "C" __global__ void compact_scatter_tiled_words(
    const u8 *values,
    u64 values_offset,
    u64 values_stride,
    const i32 *local_prefix,
    const i32 *tile_counts,
    u8 *output,
    u64 output_offset,
    u64 output_stride,
    i32 *count,
    u64 count_offset,
    u32 n,
    u32 item_words) {
  const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) {
    return;
  }
  const u32 tile = index >> 10;
  const u32 tile_begin = tile << 10;
  const i32 local_previous =
      index == tile_begin ? 0 : local_prefix[index - 1u];
  const i32 local = local_prefix[index];
  const i32 tile_base = tile == 0u ? 0 : tile_counts[tile - 1u];
  if (local != local_previous) {
    const u32 output_index = static_cast<u32>(tile_base + local - 1);
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
        tile_base + local;
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
__device__ void radix_rank4_impl(const T *keys,
                                 u32 *local_ranks,
                                 u32 *block_histogram,
                                 u32 n,
                                 u32 block_count,
                                 u32 shift,
                                 u32 *warp_counts,
                                 u32 *digit_chunk_bases) {
  const u32 tid = threadIdx.x;
  const u32 lane = tid & 31u;
  const u32 warp = tid >> 5;
  const u32 block = blockIdx.x;
  const u32 block_begin = block * 1024u;
  if (tid < 16u) {
    digit_chunk_bases[tid] = 0u;
  }
  block_barrier();

  for (u32 item = 0; item < 4u; ++item) {
    const u32 index = block_begin + item * 256u + tid;
    const bool valid = index < n;
    const u32 digit =
        valid ? static_cast<u32>((keys[index] >> shift) & static_cast<T>(15))
              : 0u;
    u32 peer_mask = 0u;
    for (u32 candidate = 0; candidate < 16u; ++candidate) {
      const u32 mask =
          warp_ballot(valid && digit == candidate ? 1 : 0);
      if (lane == 31u) {
        warp_counts[warp * 16u + candidate] = popc(mask);
      }
      if (digit == candidate) {
        peer_mask = mask;
      }
    }
    block_barrier();

    if (valid) {
      u32 rank = digit_chunk_bases[digit] + 1u;
      for (u32 previous_warp = 0; previous_warp < warp; ++previous_warp) {
        rank += warp_counts[previous_warp * 16u + digit];
      }
      const u32 lane_mask = lane == 0u ? 0u : (1u << lane) - 1u;
      rank += popc(peer_mask & lane_mask);
      local_ranks[index] = rank;
    }
    block_barrier();

    if (tid < 16u) {
      u32 chunk_count = 0u;
      for (u32 source_warp = 0; source_warp < 8u; ++source_warp) {
        chunk_count += warp_counts[source_warp * 16u + tid];
      }
      digit_chunk_bases[tid] += chunk_count;
      if (item == 3u) {
        block_histogram[tid * block_count + block] =
            digit_chunk_bases[tid];
      }
    }
    block_barrier();
  }
}

#define DEFINE_RADIX_RANK4_KERNEL(NAME, TYPE)                              \
  extern "C" __global__ void NAME(const TYPE *keys, u32 *local_ranks,     \
                                   u32 *block_histogram, u32 n,            \
                                   u32 block_count, u32 shift) {           \
    __shared__ u32 warp_counts[8 * 16];                                    \
    __shared__ u32 digit_chunk_bases[16];                                  \
    radix_rank4_impl(keys, local_ranks, block_histogram, n, block_count,   \
                     shift, warp_counts, digit_chunk_bases);               \
  }

DEFINE_RADIX_RANK4_KERNEL(radix_rank4_u32, u32)
DEFINE_RADIX_RANK4_KERNEL(radix_rank4_u64, u64)

extern "C" __global__ void radix_hist_scan(u32 *histogram,
                                           u32 count_per_digit,
                                           u32 *tile_sums,
                                           u32 tile_count) {
  __shared__ u32 warp_sums[8];
  __shared__ u32 tile_base;
  const u32 tid = threadIdx.x;
  const u32 lane = tid & 31u;
  const u32 warp = tid >> 5;
  const u32 digit = blockIdx.x / tile_count;
  const u32 tile = blockIdx.x - digit * tile_count;
  const u32 tile_begin = digit * count_per_digit + tile * 1024u;
  if (tid == 0u) {
    tile_base = 0u;
  }
  block_barrier();

  for (u32 item = 0; item < 4u; ++item) {
    const u32 chunk_base = tile_base;
    block_barrier();
    const u32 item_in_digit = tile * 1024u + item * 256u + tid;
    const u32 index = tile_begin + item * 256u + tid;
    u32 value =
        item_in_digit < count_per_digit ? histogram[index] : 0u;
    value = warp_inclusive_sum(value);
    if (lane == 31u) {
      warp_sums[warp] = value;
    }
    block_barrier();
    if (warp == 0u) {
      u32 warp_value = lane < 8u ? warp_sums[lane] : 0u;
      warp_value = warp_inclusive_sum(warp_value);
      if (lane < 8u) {
        warp_sums[lane] = warp_value;
      }
    }
    block_barrier();
    if (warp > 0u) {
      value += warp_sums[warp - 1u];
    }
    value += chunk_base;
    if (item_in_digit < count_per_digit) {
      histogram[index] = value;
    }
    if (tid == 255u) {
      tile_base = value;
    }
    block_barrier();
  }
  if (tid == 255u && tile_sums != nullptr) {
    tile_sums[digit * tile_count + tile] = tile_base;
  }
}

extern "C" __global__ void radix_hist_uniform(u32 *histogram,
                                              u32 count_per_digit,
                                              const u32 *tile_prefix,
                                              u32 tile_count) {
  const u32 blocks_per_digit = (count_per_digit + 255u) / 256u;
  const u32 digit = blockIdx.x / blocks_per_digit;
  const u32 chunk = blockIdx.x - digit * blocks_per_digit;
  const u32 item = chunk * 256u + threadIdx.x;
  if (item >= count_per_digit) {
    return;
  }
  const u32 tile = item >> 10;
  if (tile != 0u) {
    histogram[digit * count_per_digit + item] +=
        tile_prefix[digit * tile_count + tile - 1u];
  }
}

extern "C" __global__ void radix_digit_bases(const u32 *block_histogram,
                                             u32 block_count,
                                             u32 *digit_bases) {
  __shared__ u32 totals[16];
  const u32 digit = threadIdx.x;
  if (digit < 16u) {
    totals[digit] =
        block_histogram[digit * block_count + block_count - 1u];
  }
  block_barrier();
  if (digit < 16u) {
    u32 base = 0u;
    for (u32 previous = 0; previous < digit; ++previous) {
      base += totals[previous];
    }
    digit_bases[digit] = base;
  }
}

template <typename T>
__device__ void radix_scatter4_impl(const T *keys_in,
                                    const u32 *indices_in,
                                    const u32 *local_ranks,
                                    const u32 *block_histogram,
                                    const u32 *digit_bases,
                                    T *keys_out,
                                    u32 *indices_out,
                                    u32 n,
                                    u32 block_count,
                                    u32 shift) {
  const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) {
    return;
  }
  const T key = keys_in[index];
  const u32 digit =
      static_cast<u32>((key >> shift) & static_cast<T>(15));
  const u32 block = index >> 10;
  const u32 block_base =
      block == 0u ? 0u
                  : block_histogram[digit * block_count + block - 1u];
  const u32 output_index =
      digit_bases[digit] + block_base + local_ranks[index] - 1u;
  keys_out[output_index] = key;
  indices_out[output_index] = indices_in[index];
}

#define DEFINE_RADIX_SCATTER4_KERNEL(NAME, TYPE)                            \
  extern "C" __global__ void NAME(                                         \
      const TYPE *keys_in, const u32 *indices_in, const u32 *local_ranks,   \
      const u32 *block_histogram, const u32 *digit_bases, TYPE *keys_out,   \
      u32 *indices_out, u32 n, u32 block_count, u32 shift) {                \
    radix_scatter4_impl(keys_in, indices_in, local_ranks, block_histogram,  \
                        digit_bases, keys_out, indices_out, n, block_count, \
                        shift);                                             \
  }

DEFINE_RADIX_SCATTER4_KERNEL(radix_scatter4_u32, u32)
DEFINE_RADIX_SCATTER4_KERNEL(radix_scatter4_u64, u64)

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

__device__ bool sparse_assembly_finite(float value) {
  return (bit_cast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}

__device__ void sparse_assembly_set_first_status(i32 *status, i32 code) {
  u32 previous;
  asm volatile("atom.global.cas.b32 %0, [%1], %2, %3;"
               : "=r"(previous)
               : "l"(status), "r"(0u),
                 "r"(static_cast<u32>(-code))
               : "memory");
}

__device__ void sparse_assembly_atomic_add(u32 *address, u32 value) {
  u32 previous;
  asm volatile("atom.global.add.u32 %0, [%1], %2;"
               : "=r"(previous)
               : "l"(address), "r"(value)
               : "memory");
}

extern "C" __global__ void sparse_assembly_pack_validate(
    const i32 *triplet_rows,
    const i32 *triplet_columns,
    const float *triplet_values,
    u64 *sorted_keys,
    float *sorted_values,
    i32 *active_count,
    i32 *control,
    u32 n,
    u32 rows,
    u32 columns) {
  const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) {
    return;
  }
  if (index == 0u) {
    active_count[0] = static_cast<i32>(n);
  }
  i32 row = triplet_rows[index];
  i32 column = triplet_columns[index];
  float value = triplet_values[index];
  const bool valid_index = row >= 0 && column >= 0 &&
                           static_cast<u32>(row) < rows &&
                           static_cast<u32>(column) < columns;
  if (!valid_index) {
    sparse_assembly_set_first_status(control, 1);
    row = 0;
    column = 0;
  }
  if (!sparse_assembly_finite(value)) {
    sparse_assembly_set_first_status(control, 2);
    value = 0.0f;
  }
  sorted_keys[index] =
      (static_cast<u64>(static_cast<u32>(row)) << 32) |
      static_cast<u32>(column);
  sorted_values[index] = value;
}

extern "C" __global__ void sparse_assembly_pack_packed_validate(
    const u32 *packed_triplets,
    u64 *sorted_keys,
    float *sorted_values,
    i32 *active_count,
    i32 *control,
    u32 capacity,
    u32 rows,
    u32 columns) {
  const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= capacity) {
    return;
  }
  const i32 raw_count = static_cast<i32>(packed_triplets[0]);
  const u32 header_capacity = packed_triplets[1];
  const bool valid_count =
      raw_count >= 0 && static_cast<u32>(raw_count) <= capacity &&
      header_capacity == capacity;
  const u32 count = valid_count ? static_cast<u32>(raw_count) : 0u;
  if (index == 0u) {
    active_count[0] = static_cast<i32>(count);
    if (!valid_count) {
      control[1] = raw_count;
      sparse_assembly_set_first_status(control, 5);
    }
  }
  if (index >= count) {
    sorted_keys[index] = ~static_cast<u64>(0);
    sorted_values[index] = 0.0f;
    return;
  }
  const u32 *triplet = packed_triplets + 2u + index * 3u;
  i32 row = static_cast<i32>(triplet[0]);
  i32 column = static_cast<i32>(triplet[1]);
  float value = bit_cast<float>(triplet[2]);
  const bool valid_index = row >= 0 && column >= 0 &&
                           static_cast<u32>(row) < rows &&
                           static_cast<u32>(column) < columns;
  if (!valid_index) {
    sparse_assembly_set_first_status(control, 1);
    row = 0;
    column = 0;
  }
  if (!sparse_assembly_finite(value)) {
    sparse_assembly_set_first_status(control, 2);
    value = 0.0f;
  }
  sorted_keys[index] =
      (static_cast<u64>(static_cast<u32>(row)) << 32) |
      static_cast<u32>(column);
  sorted_values[index] = value;
}

extern "C" __global__ void sparse_assembly_mark_segments(
    const u64 *sorted_keys,
    i32 *segment_ids,
    const i32 *active_count,
    i32 *control,
    u32 capacity) {
  const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= capacity) {
    return;
  }
  const i32 active = active_count[0];
  if (active < 0 || static_cast<u32>(active) > capacity) {
    if (index == 0u) {
      sparse_assembly_set_first_status(control, 4);
    }
    segment_ids[index] = 0;
    return;
  }
  segment_ids[index] =
      index < static_cast<u32>(active) &&
              (index == 0u ||
               sorted_keys[index] != sorted_keys[index - 1u])
          ? 1
          : 0;
}

extern "C" __global__ void sparse_assembly_scatter_segments(
    const u64 *sorted_keys,
    const i32 *segment_ids,
    u64 *unique_keys,
    i32 *segment_offsets,
    const i32 *active_count,
    i32 *control,
    u32 capacity) {
  const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= capacity) {
    return;
  }
  const i32 active = active_count[0];
  if (active < 0 || static_cast<u32>(active) > capacity) {
    if (index == 0u) {
      sparse_assembly_set_first_status(control, 4);
      control[1] = 0;
    }
    return;
  }
  if (active == 0) {
    if (index == 0u) {
      segment_offsets[0] = 0;
      if (control[0] == 0) {
        control[1] = 0;
      }
    }
    return;
  }
  if (index >= static_cast<u32>(active)) {
    return;
  }
  const bool starts_segment =
      index == 0u || sorted_keys[index] != sorted_keys[index - 1u];
  const i32 segment_id = segment_ids[index];
  if (starts_segment) {
    if (segment_id <= 0 ||
        static_cast<u32>(segment_id) > static_cast<u32>(active)) {
      sparse_assembly_set_first_status(control, 4);
    } else {
      const u32 segment = static_cast<u32>(segment_id - 1);
      unique_keys[segment] = sorted_keys[index];
      segment_offsets[segment] = static_cast<i32>(index);
    }
  }
  if (index + 1u == static_cast<u32>(active)) {
    if (segment_id <= 0 ||
        static_cast<u32>(segment_id) > static_cast<u32>(active)) {
      sparse_assembly_set_first_status(control, 4);
      control[1] = 0;
    } else {
      segment_offsets[segment_id] = active;
      control[1] = segment_id;
    }
  }
}

extern "C" __global__ void sparse_assembly_reduce_segments(
    const float *sorted_values,
    const i32 *segment_offsets,
    float *unique_values,
    const i32 *active_count,
    i32 *control,
    u32 capacity) {
  const u32 segment = blockIdx.x * blockDim.x + threadIdx.x;
  const i32 active = active_count[0];
  const i32 unique_count = control[1];
  if (active < 0 || static_cast<u32>(active) > capacity ||
      unique_count < 0 || unique_count > active) {
    if (segment == 0u) {
      sparse_assembly_set_first_status(control, 4);
    }
    return;
  }
  if (active == 0) {
    return;
  }
  if (segment >= static_cast<u32>(unique_count)) {
    return;
  }
  const i32 begin = segment_offsets[segment];
  const i32 end = segment_offsets[segment + 1u];
  if (begin < 0 || end <= begin || end > active) {
    sparse_assembly_set_first_status(control, 4);
    unique_values[segment] = 0.0f;
    return;
  }
  float sum = 0.0f;
  for (i32 index = begin; index < end; ++index) {
    sum += sorted_values[index];
  }
  if (!sparse_assembly_finite(sum)) {
    sparse_assembly_set_first_status(control, 3);
    sum = 0.0f;
  }
  unique_values[segment] = sum;
}

extern "C" __global__ void sparse_assembly_emit_csr(
    const u64 *unique_keys,
    i32 *row_offsets,
    i32 *column_indices,
    const i32 *active_count,
    i32 *control,
    u32 capacity,
    u32 rows,
    u32 columns) {
  const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
  const i32 active = active_count[0];
  const i32 unique_count = control[1];
  if (active < 0 || static_cast<u32>(active) > capacity ||
      unique_count < 0 || unique_count > active) {
    if (index == 0u) {
      sparse_assembly_set_first_status(control, 4);
    }
    return;
  }
  if (active == 0) {
    return;
  }
  if (index >= static_cast<u32>(unique_count)) {
    return;
  }
  const u64 key = unique_keys[index];
  const u32 row = static_cast<u32>(key >> 32);
  const u32 column = static_cast<u32>(key);
  if (row >= rows || column >= columns) {
    sparse_assembly_set_first_status(control, 4);
    return;
  }
  column_indices[index] = static_cast<i32>(column);
  sparse_assembly_atomic_add(
      reinterpret_cast<u32 *>(row_offsets + row + 1u), 1u);
}

extern "C" __global__ void sparse_assembly_finalize_control(
    const i32 *active_count,
    i32 *control,
    u32 capacity) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) {
    return;
  }
  const i32 active = active_count[0];
  if (active < 0 || static_cast<u32>(active) > capacity) {
    sparse_assembly_set_first_status(control, 4);
    return;
  }
  if (control[0] == 0) {
    control[0] = active;
  }
}
