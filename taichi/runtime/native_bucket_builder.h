#pragma once

#include "taichi/runtime/native_operation_description.h"
#include "taichi/runtime/vulkan_bucket_builder.h"

namespace taichi::lang {

struct BucketBuilderOperation {
  static constexpr const char *kind = "bucket_builder";
  static constexpr uint32_t version = 1;
  size_t size{0}, num_bins{0};
  int value_type{0}, cpu_max_threads{0};
  std::vector<uint64_t> element_shape;
  gfx::VulkanBucketBuilderPolicy vulkan_policy;
  NativeArrayBinding keys, values, offsets, output;
};

}  // namespace taichi::lang
