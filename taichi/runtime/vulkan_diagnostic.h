#pragma once

#include <memory>
#include "taichi/rhi/device.h"

namespace taichi::lang {
class NativeOperation;
namespace gfx {

// Cold logical view. The enclosing executor retains allocation ownership;
// prepared operations own pipelines/descriptors and never copy source data.
struct VulkanDiagnosticInput {
  DevicePtr pointer{kDeviceNullPtr};
  size_t stride{0};
};

std::shared_ptr<NativeOperation> prepare_vulkan_check_operation(
    Device &device,
    VulkanDiagnosticInput values,
    DevicePtr output,
    size_t count,
    int value_type,
    int check_op,
    int32_t lower,
    int32_t upper,
    uint64_t items_per_group = 16384);

std::shared_ptr<NativeOperation> prepare_vulkan_metric_operation(
    Device &device,
    VulkanDiagnosticInput values,
    VulkanDiagnosticInput other,
    DevicePtr output,
    size_t count,
    int value_type,
    int metric_op,
    uint64_t items_per_group = 16384);

}  // namespace gfx
}  // namespace taichi::lang
