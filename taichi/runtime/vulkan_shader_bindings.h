#pragma once

#ifdef TI_WITH_VULKAN
#include <map>
#include "taichi/rhi/device.h"

namespace taichi::lang {
namespace vulkan {
class VulkanPipeline;
}
using VulkanShaderResourceSets =
    std::map<uint32_t, std::unique_ptr<ShaderResourceSet>>;
using PreparedVulkanShaderBindings =
    std::vector<std::pair<uint32_t, std::unique_ptr<ShaderResourceSet>>>;

// Common cold reflection/descriptor finalization for raster and compute.
// The caller validates physical views, usage and ownership before binding;
// the returned sets must be retained through completion by the executor.
PreparedVulkanShaderBindings prepare_vulkan_shader_bindings(
    vulkan::VulkanPipeline &pipeline,
    VulkanShaderResourceSets resources);
}  // namespace taichi::lang
#endif
