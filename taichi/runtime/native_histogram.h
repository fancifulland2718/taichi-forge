#pragma once

#include "taichi/runtime/native_operation_description.h"

namespace taichi::lang {

// Cold algorithm preferences, not admission limits. Zero disables single
// sharing, or removes the corresponding private-selection threshold.
struct VulkanHistogramPolicy {
  uint64_t private_min_n{65536};
  uint64_t private_max_bins{512};
  uint64_t single_shared_max_n{4096};
};

struct HistogramOperation {
  static constexpr const char *kind = "histogram";
  static constexpr uint32_t version = 1;
  size_t size{0}, num_bins{0};
  int value_type{0}, bin_type{0}, cpu_max_threads{0};
  VulkanHistogramPolicy vulkan_policy;
  NativeArrayBinding values, bins;
};

void validate_histogram_storage(Arch arch, const HistogramOperation &operation);
class NativeOperation;
namespace gfx {
std::shared_ptr<NativeOperation> prepare_vulkan_histogram_operation(
    Device &device,
    const HistogramOperation &operation);
}
}  // namespace taichi::lang
