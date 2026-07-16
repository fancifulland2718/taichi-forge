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
