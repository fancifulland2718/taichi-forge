#pragma once

#include "taichi/rhi/device.h"

#ifdef TI_WITH_VULKAN
namespace taichi::lang {
namespace vulkan {
class VulkanDevice;
}

// Cold pipeline construction shared by JIT and native resource factories.
// Sources are consumed synchronously. The caller owns queue ordering, leases
// and retirement; this function never creates or submits through a Program.
std::unique_ptr<Pipeline> prepare_vulkan_graphics_pipeline(
    vulkan::VulkanDevice &device,
    const std::vector<PipelineSourceDesc> &sources,
    const RasterParams &raster,
    const std::vector<VertexInputBinding> &bindings,
    const std::vector<VertexInputAttribute> &attributes,
    const std::string &name);
}  // namespace taichi::lang
#endif
