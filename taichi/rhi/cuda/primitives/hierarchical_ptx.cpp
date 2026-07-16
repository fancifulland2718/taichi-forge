#include "taichi/rhi/cuda/primitives/hierarchical_ptx.h"

#include "taichi/common/core.h"
#include "taichi/program/primitive_workspace.h"
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_driver.h"
#include "taichi/rhi/cuda/primitives/ptx/hierarchical_ptx.inc.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace taichi::lang::cuda {
namespace {

constexpr std::uint32_t kBlockDim = 256;
constexpr std::uint32_t kReduceItemsPerThread = 4;
constexpr std::uint32_t kReduceTileItems = kBlockDim * kReduceItemsPerThread;
constexpr std::uint64_t kWorkspaceVariant = 0x4452565054580001ull;

struct KernelSet {
  void *module{nullptr};
  std::array<void *, 6> scan{};
  std::array<void *, 6> uniform_add{};
  std::array<void *, 6> reduce{};
  std::array<void *, 2> zero_bins{};
  std::array<std::array<void *, 2>, 2> histogram{};
};

std::once_flag kernel_set_once;
KernelSet kernel_set;

std::size_t value_type_index(CudaTransformValueType value_type) {
  const int index = static_cast<int>(value_type);
  TI_ERROR_IF(index < 0 || index >= 6,
              "CUDA Driver hierarchical primitive received an unsupported "
              "value type.");
  return static_cast<std::size_t>(index);
}

std::size_t value_type_size(CudaTransformValueType value_type) {
  switch (value_type) {
    case CudaTransformValueType::i32:
    case CudaTransformValueType::f32:
    case CudaTransformValueType::u32:
      return 4;
    case CudaTransformValueType::u64:
    case CudaTransformValueType::i64:
    case CudaTransformValueType::f64:
      return 8;
  }
  TI_ERROR("Unsupported CUDA hierarchical primitive value type.");
  return 0;
}

void load_kernel_set_once() {
  auto &ctx = CUDAContext::get_instance();
  auto context_guard = ctx.get_guard();
  auto &driver = CUDADriver::get_instance();
  driver.module_load_data_ex(&kernel_set.module, kCudaHierarchicalPtx, 0,
                             nullptr, nullptr);

  constexpr std::array<const char *, 6> suffixes{"i32", "f32", "u32",
                                                 "u64", "i64", "f64"};
  for (std::size_t i = 0; i < suffixes.size(); ++i) {
    const std::string scan_name = std::string("scan_blocks_") + suffixes[i];
    const std::string uniform_name = std::string("uniform_add_") + suffixes[i];
    const std::string reduce_name = std::string("reduce_blocks_") + suffixes[i];
    driver.module_get_function(&kernel_set.scan[i], kernel_set.module,
                               scan_name.c_str());
    driver.module_get_function(&kernel_set.uniform_add[i], kernel_set.module,
                               uniform_name.c_str());
    driver.module_get_function(&kernel_set.reduce[i], kernel_set.module,
                               reduce_name.c_str());
  }
  driver.module_get_function(&kernel_set.zero_bins[0], kernel_set.module,
                             "zero_bins_i32");
  driver.module_get_function(&kernel_set.zero_bins[1], kernel_set.module,
                             "zero_bins_i64");
  driver.module_get_function(&kernel_set.histogram[0][0], kernel_set.module,
                             "histogram_i32_i32");
  driver.module_get_function(&kernel_set.histogram[1][0], kernel_set.module,
                             "histogram_u32_i32");
  driver.module_get_function(&kernel_set.histogram[0][1], kernel_set.module,
                             "histogram_i32_i64");
  driver.module_get_function(&kernel_set.histogram[1][1], kernel_set.module,
                             "histogram_u32_i64");
}

KernelSet &kernels() {
  std::call_once(kernel_set_once, load_kernel_set_once);
  return kernel_set;
}

class DriverWorkspace final {
 public:
  DriverWorkspace() = default;
  DriverWorkspace(const DriverWorkspace &) = delete;
  DriverWorkspace &operator=(const DriverWorkspace &) = delete;

  ~DriverWorkspace() {
    release_noexcept();
  }

  void ensure(std::size_t required_bytes) {
    if (required_bytes <= capacity_) {
      return;
    }
    std::size_t new_capacity = std::max<std::size_t>(required_bytes, 4096);
    if (capacity_ != 0 &&
        capacity_ <= std::numeric_limits<std::size_t>::max() / 2) {
      new_capacity = std::max(new_capacity, capacity_ * 2);
    }
    new_capacity = (new_capacity + 255u) & ~std::size_t{255u};

    auto &ctx = CUDAContext::get_instance();
    auto context_guard = ctx.get_guard();
    void *new_block = nullptr;
    CUDADriver::get_instance().malloc(&new_block, new_capacity);
    try {
      allocations_.push_back({new_block, new_capacity});
    } catch (...) {
      CUDADriver::get_instance().mem_free(new_block);
      throw;
    }
    current_ = new_block;
    capacity_ = new_capacity;
    allocated_bytes_ += new_capacity;
  }

  void *data() const noexcept {
    return current_;
  }

  std::size_t allocated_bytes() const noexcept {
    return allocated_bytes_;
  }

 private:
  void release_noexcept() noexcept {
    try {
      auto &ctx = CUDAContext::get_instance();
      auto context_guard = ctx.get_guard();
      auto &driver = CUDADriver::get_instance();
      for (auto it = allocations_.rbegin(); it != allocations_.rend(); ++it) {
        driver.mem_free(it->first);
      }
    } catch (...) {
      // Program teardown already established a synchronization boundary.  A
      // destructor must not mask the original teardown exception.
    }
    allocations_.clear();
    current_ = nullptr;
    capacity_ = 0;
    allocated_bytes_ = 0;
  }

  std::vector<std::pair<void *, std::size_t>> allocations_;
  void *current_{nullptr};
  std::size_t capacity_{0};
  std::size_t allocated_bytes_{0};
};

PrimitiveWorkspaceArena::Lease<DriverWorkspace> acquire_workspace(
    PrimitiveWorkspaceArena *arena,
    PrimitiveWorkspaceFamily family,
    void *stream) {
  TI_ERROR_IF(!arena,
              "CUDA Driver hierarchical primitive requires a Program-owned "
              "workspace arena.");
  PrimitiveWorkspaceKey key{
      PrimitiveWorkspaceBackend::cuda, family,
      static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(stream)),
      kWorkspaceVariant};
  return arena->acquire<DriverWorkspace>(
      key, [] { return std::make_shared<DriverWorkspace>(); });
}

struct LevelLayout {
  std::array<std::uint32_t, 8> counts{};
  std::array<std::size_t, 8> offsets{};
  std::size_t level_count{0};
  std::size_t bytes{0};
};

LevelLayout scan_layout(std::uint32_t num_items, std::size_t value_size) {
  LevelLayout layout;
  std::uint32_t count =
      (num_items + kBlockDim - 1u) / static_cast<std::uint32_t>(kBlockDim);
  while (count > 1u) {
    TI_ASSERT(layout.level_count < layout.counts.size());
    layout.bytes = (layout.bytes + 255u) & ~std::size_t{255u};
    layout.counts[layout.level_count] = count;
    layout.offsets[layout.level_count] = layout.bytes;
    layout.bytes += static_cast<std::size_t>(count) * value_size;
    ++layout.level_count;
    count = (count + kBlockDim - 1u) / kBlockDim;
  }
  return layout;
}

LevelLayout reduce_layout(std::uint32_t num_items, std::size_t value_size) {
  LevelLayout layout;
  std::uint32_t count = (num_items + kReduceTileItems - 1u) / kReduceTileItems;
  while (count > 1u) {
    TI_ASSERT(layout.level_count < layout.counts.size());
    layout.bytes = (layout.bytes + 255u) & ~std::size_t{255u};
    layout.counts[layout.level_count] = count;
    layout.offsets[layout.level_count] = layout.bytes;
    layout.bytes += static_cast<std::size_t>(count) * value_size;
    ++layout.level_count;
    if (count <= kReduceTileItems) {
      break;
    }
    count = (count + kReduceTileItems - 1u) / kReduceTileItems;
  }
  return layout;
}

void *byte_offset(void *base, std::size_t offset) {
  return reinterpret_cast<std::uint8_t *>(base) + offset;
}

}  // namespace

bool driver_hierarchical_available() {
  return CUDADriver::get_instance_without_context().detected();
}

std::size_t driver_inclusive_scan_strided(
    void *data,
    int num_items,
    CudaTransformValueType value_type,
    std::size_t offset,
    std::size_t stride,
    bool reverse,
    void *stream,
    PrimitiveWorkspaceArena *workspace_arena) {
  TI_ERROR_IF(num_items < 0,
              "CUDA Driver scan expects non-negative num_items.");
  TI_ERROR_IF(!data, "CUDA Driver scan received a null data pointer.");
  const std::size_t value_size = value_type_size(value_type);
  TI_ERROR_IF(stride < value_size,
              "CUDA Driver scan received a stride smaller than its value.");
  if (num_items <= 1) {
    return 0;
  }

  const std::uint32_t n = static_cast<std::uint32_t>(num_items);
  const auto layout = scan_layout(n, value_size);
  std::optional<PrimitiveWorkspaceArena::Lease<DriverWorkspace>> workspace;
  std::array<void *, 8> level_ptrs{};
  if (layout.level_count != 0) {
    workspace.emplace(acquire_workspace(
        workspace_arena, PrimitiveWorkspaceFamily::scan, stream));
    (*workspace)->ensure(layout.bytes);
    for (std::size_t i = 0; i < layout.level_count; ++i) {
      level_ptrs[i] = byte_offset((*workspace)->data(), layout.offsets[i]);
    }
  }

  const auto type_index = value_type_index(value_type);
  auto &kernel = kernels();
  auto launch_scan = [&](void *values, std::uint64_t values_offset,
                         std::uint64_t values_stride, std::uint32_t count,
                         void *sums, std::int32_t reverse_order) {
    void *values_arg = values;
    void *sums_arg = sums;
    std::uint64_t sums_offset = 0;
    std::vector<void *> args{&values_arg,   &values_offset, &values_stride,
                             &count,        &sums_arg,      &sums_offset,
                             &reverse_order};
    const unsigned grid = (count + kBlockDim - 1u) / kBlockDim;
    CUDAContext::get_instance().launch(kernel.scan[type_index],
                                       "cuda_driver_scan_blocks", args, {},
                                       grid, kBlockDim, 0, stream);
  };
  auto launch_uniform = [&](void *values, std::uint64_t values_offset,
                            std::uint64_t values_stride, std::uint32_t count,
                            void *sums, std::int32_t reverse_order) {
    void *values_arg = values;
    void *sums_arg = sums;
    std::uint64_t sums_offset = 0;
    std::vector<void *> args{&values_arg,   &values_offset, &values_stride,
                             &count,        &sums_arg,      &sums_offset,
                             &reverse_order};
    const unsigned grid = (count + kBlockDim - 1u) / kBlockDim;
    CUDAContext::get_instance().launch(kernel.uniform_add[type_index],
                                       "cuda_driver_scan_uniform_add", args, {},
                                       grid, kBlockDim, 0, stream);
  };

  launch_scan(data, static_cast<std::uint64_t>(offset),
              static_cast<std::uint64_t>(stride), n, level_ptrs[0],
              reverse ? 1 : 0);
  for (std::size_t i = 0; i < layout.level_count; ++i) {
    void *next = i + 1 < layout.level_count ? level_ptrs[i + 1] : nullptr;
    launch_scan(level_ptrs[i], 0, value_size, layout.counts[i], next, 0);
  }
  for (std::size_t i = layout.level_count; i > 1; --i) {
    const std::size_t level = i - 2;
    launch_uniform(level_ptrs[level], 0, value_size, layout.counts[level],
                   level_ptrs[level + 1], 0);
  }
  if (layout.level_count != 0) {
    launch_uniform(data, static_cast<std::uint64_t>(offset),
                   static_cast<std::uint64_t>(stride), n, level_ptrs[0],
                   reverse ? 1 : 0);
    return (*workspace)->allocated_bytes();
  }
  return 0;
}

std::size_t driver_reduce_strided(void *values,
                                  void *output,
                                  int num_items,
                                  CudaTransformValueType value_type,
                                  std::size_t values_offset,
                                  std::size_t values_stride,
                                  std::size_t output_offset,
                                  std::size_t output_stride,
                                  CudaHierarchicalReduceOp op,
                                  void *stream,
                                  PrimitiveWorkspaceArena *workspace_arena) {
  TI_ERROR_IF(num_items <= 0,
              "CUDA Driver reduce expects at least one input item.");
  TI_ERROR_IF(!values || !output,
              "CUDA Driver reduce received a null pointer.");
  const std::size_t value_size = value_type_size(value_type);
  TI_ERROR_IF(values_stride < value_size || output_stride < value_size,
              "CUDA Driver reduce received a stride smaller than its value.");
  const int op_value = static_cast<int>(op);
  TI_ERROR_IF(op_value < 0 || op_value > 2,
              "CUDA Driver reduce received an unsupported operation.");

  const std::uint32_t n = static_cast<std::uint32_t>(num_items);
  const auto layout = reduce_layout(n, value_size);
  std::optional<PrimitiveWorkspaceArena::Lease<DriverWorkspace>> workspace;
  std::array<void *, 8> level_ptrs{};
  if (layout.level_count != 0) {
    workspace.emplace(acquire_workspace(
        workspace_arena, PrimitiveWorkspaceFamily::reduce, stream));
    (*workspace)->ensure(layout.bytes);
    for (std::size_t i = 0; i < layout.level_count; ++i) {
      level_ptrs[i] = byte_offset((*workspace)->data(), layout.offsets[i]);
    }
  }

  const auto type_index = value_type_index(value_type);
  auto &kernel = kernels();
  auto launch_reduce = [&](void *input, std::uint64_t input_offset,
                           std::uint64_t input_stride, std::uint32_t count,
                           void *result, std::uint64_t result_offset,
                           std::uint64_t result_stride) {
    void *input_arg = input;
    void *result_arg = result;
    std::int32_t operation = op_value;
    std::vector<void *> args{&input_arg,     &input_offset, &input_stride,
                             &count,         &result_arg,   &result_offset,
                             &result_stride, &operation};
    const unsigned grid = (count + kBlockDim - 1u) / kBlockDim;
    CUDAContext::get_instance().launch(kernel.reduce[type_index],
                                       "cuda_driver_reduce_blocks", args, {},
                                       grid, kBlockDim, 0, stream);
  };

  if (layout.level_count == 0) {
    launch_reduce(values, static_cast<std::uint64_t>(values_offset),
                  static_cast<std::uint64_t>(values_stride), n, output,
                  static_cast<std::uint64_t>(output_offset),
                  static_cast<std::uint64_t>(output_stride));
    return 0;
  }

  launch_reduce(values, static_cast<std::uint64_t>(values_offset),
                static_cast<std::uint64_t>(values_stride), n, level_ptrs[0], 0,
                value_size);
  for (std::size_t i = 0; i + 1 < layout.level_count; ++i) {
    launch_reduce(level_ptrs[i], 0, value_size, layout.counts[i],
                  level_ptrs[i + 1], 0, value_size);
  }
  const std::size_t final_level = layout.level_count - 1;
  launch_reduce(level_ptrs[final_level], 0, value_size,
                layout.counts[final_level], output,
                static_cast<std::uint64_t>(output_offset),
                static_cast<std::uint64_t>(output_stride));
  return (*workspace)->allocated_bytes();
}

std::size_t driver_histogram_strided(void *values,
                                     void *bins,
                                     int num_items,
                                     int num_bins,
                                     CudaTransformValueType value_type,
                                     CudaTransformValueType bin_type,
                                     std::size_t values_offset,
                                     std::size_t values_stride,
                                     std::size_t bins_offset,
                                     std::size_t bins_stride,
                                     void *stream) {
  TI_ERROR_IF(num_items < 0,
              "CUDA Driver histogram expects non-negative num_items.");
  TI_ERROR_IF(num_bins <= 0, "CUDA Driver histogram expects at least one bin.");
  TI_ERROR_IF((num_items > 0 && !values) || !bins,
              "CUDA Driver histogram received a null pointer.");
  const int value_index = value_type == CudaTransformValueType::i32   ? 0
                          : value_type == CudaTransformValueType::u32 ? 1
                                                                      : -1;
  const int bin_index = bin_type == CudaTransformValueType::i32   ? 0
                        : bin_type == CudaTransformValueType::i64 ? 1
                                                                  : -1;
  TI_ERROR_IF(value_index < 0,
              "CUDA Driver histogram supports only i32/u32 bin ids.");
  TI_ERROR_IF(bin_index < 0,
              "CUDA Driver histogram supports only i32/i64 counters.");
  TI_ERROR_IF(values_stride < value_type_size(value_type) ||
                  bins_stride < value_type_size(bin_type),
              "CUDA Driver histogram received an invalid stride.");

  auto &kernel = kernels();
  void *bins_arg = bins;
  std::uint64_t bins_offset_arg = bins_offset;
  std::uint64_t bins_stride_arg = bins_stride;
  std::uint32_t bins_count_arg = static_cast<std::uint32_t>(num_bins);
  std::vector<void *> zero_args{&bins_arg, &bins_offset_arg, &bins_stride_arg,
                                &bins_count_arg};
  const unsigned bins_grid = (bins_count_arg + kBlockDim - 1u) / kBlockDim;
  CUDAContext::get_instance().launch(kernel.zero_bins[bin_index],
                                     "cuda_driver_histogram_zero", zero_args,
                                     {}, bins_grid, kBlockDim, 0, stream);

  if (num_items == 0) {
    return 0;
  }
  void *values_arg = values;
  std::uint64_t values_offset_arg = values_offset;
  std::uint64_t values_stride_arg = values_stride;
  std::uint32_t item_count_arg = static_cast<std::uint32_t>(num_items);
  std::vector<void *> histogram_args{
      &values_arg, &values_offset_arg, &values_stride_arg, &item_count_arg,
      &bins_arg,   &bins_offset_arg,   &bins_stride_arg,   &bins_count_arg};
  const unsigned item_grid = (item_count_arg + kBlockDim - 1u) / kBlockDim;
  CUDAContext::get_instance().launch(kernel.histogram[value_index][bin_index],
                                     "cuda_driver_histogram", histogram_args,
                                     {}, item_grid, kBlockDim, 0, stream);
  return 0;
}

}  // namespace taichi::lang::cuda
