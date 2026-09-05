// Graph-owned segmented carry propagation using CUB's device-wide scan.
// The head bitset belongs to one frozen segment layout, not to each replay.
#pragma once

#include <cub/device/device_scan.cuh>
#include <cuda/iterator>
#include <cstddef>
#include <cstdint>
#include <iterator>

namespace forge_cub {

struct ResetValue {
  std::uint32_t value;
  std::uint32_t head;
};

struct ResetSum {
  __host__ __device__ ResetValue operator()(ResetValue left,
                                           ResetValue right) const {
    // Associative, deliberately non-commutative, exact modulo 2^32.
    return {right.head ? right.value : left.value + right.value,
            left.head | right.head};
  }
};

struct HeadInput {
  const std::uint32_t *values;
  const std::uint32_t *heads;

  __host__ __device__ ResetValue operator()(int index) const {
    return {values[index], (heads[index >> 5] >> (index & 31)) & 1u};
  }
};

template <bool Exclusive>
struct ProjectedOutput {
  std::uint32_t *values;
  const std::uint32_t *heads;
  std::ptrdiff_t index;

  struct Reference {
    std::uint32_t *value;
    const std::uint32_t *heads;
    std::ptrdiff_t index;
    __device__ void operator=(ResetValue aggregate) const {
      // ExclusiveScan visits the prior prefix at a segment's first item;
      // discard that prior segment's carry, without a correction kernel.
      const bool reset = Exclusive && ((heads[index >> 5] >> (index & 31)) & 1u);
      *value = reset ? 0u : aggregate.value;
    }
  };

  using value_type = ResetValue;
  using difference_type = std::ptrdiff_t;
  using pointer = void;
  using reference = Reference;
  using iterator_category = std::random_access_iterator_tag;

  __host__ __device__ Reference operator*() const {
    return {values + index, heads, index};
  }
  __host__ __device__ Reference operator[](difference_type offset) const {
    return *(*this + offset);
  }
  __host__ __device__ ProjectedOutput operator+(difference_type offset) const {
    return {values, heads, index + offset};
  }
  __host__ __device__ ProjectedOutput &operator+=(difference_type offset) {
    index += offset;
    return *this;
  }
  __host__ __device__ ProjectedOutput operator-(difference_type offset) const {
    return *this + -offset;
  }
  __host__ __device__ difference_type operator-(ProjectedOutput other) const {
    return index - other.index;
  }
  __host__ __device__ ProjectedOutput &operator-=(difference_type offset) {
    return *this += -offset;
  }
  __host__ __device__ ProjectedOutput &operator++() {
    return *this += 1;
  }
  __host__ __device__ ProjectedOutput operator++(int) {
    auto previous = *this;
    ++*this;
    return previous;
  }
  __host__ __device__ ProjectedOutput &operator--() {
    return *this -= 1;
  }
  __host__ __device__ ProjectedOutput operator--(int) {
    auto previous = *this;
    --*this;
    return previous;
  }
  __host__ __device__ bool operator==(ProjectedOutput other) const {
    return values == other.values && heads == other.heads && index == other.index;
  }
  __host__ __device__ bool operator!=(ProjectedOutput other) const {
    return !(*this == other);
  }
  __host__ __device__ bool operator<(ProjectedOutput other) const {
    return index < other.index;
  }
  __host__ __device__ bool operator>(ProjectedOutput other) const {
    return other < *this;
  }
  __host__ __device__ bool operator<=(ProjectedOutput other) const {
    return !(other < *this);
  }
  __host__ __device__ bool operator>=(ProjectedOutput other) const {
    return !(*this < other);
  }
  friend __host__ __device__ ProjectedOutput operator+(difference_type offset,
                                                      ProjectedOutput item) {
    return item + offset;
  }
};

template <bool Exclusive>
cudaError_t segmented_scan(void *workspace,
                           std::size_t &bytes,
                           const std::uint32_t *values,
                           const std::uint32_t *heads,
                           std::uint32_t *output,
                           int count,
                           cudaStream_t stream) {
  auto input = cuda::make_transform_iterator(cuda::counting_iterator<int>(0),
                                              HeadInput{values, heads});
  ProjectedOutput<Exclusive> projected{output, heads, 0};
  if constexpr (Exclusive) {
    return cub::DeviceScan::ExclusiveScan(workspace, bytes, input, projected,
                                         ResetSum{}, ResetValue{0, 0}, count,
                                         stream);
  } else {
    return cub::DeviceScan::InclusiveScan(workspace, bytes, input, projected,
                                         ResetSum{}, count, stream);
  }
}

}  // namespace forge_cub
