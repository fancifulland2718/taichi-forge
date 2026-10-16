#include "taichi/runtime/vulkan_parallel_sort_plan.h"

#include <limits>
#include <stdexcept>

namespace taichi::lang {
std::uint64_t vulkan_parallel_sort_storage_bytes(std::uint64_t key_count) {
  if (!key_count || key_count > std::numeric_limits<uint32_t>::max() ||
      key_count > std::numeric_limits<size_t>::max() / sizeof(uint32_t))
    throw std::invalid_argument(
        "Parallel Sort key count exceeds its shader/platform ABI");
  return key_count * sizeof(uint32_t);
}
}  // namespace taichi::lang

#ifdef TI_WITH_VULKAN
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/runtime/native_array_contract.h"

namespace taichi::lang {
namespace {
struct Parameters {
  uint32_t keys;
  int32_t blocks_per_group;
  uint32_t groups, extra_groups, reduced_per_bin, scan_values, shift;
};
static_assert(sizeof(Parameters) == 28);

// Only this provider interprets its shaders' bins, passes and scratch layout.
struct SortResources final : vkapi::DeviceObj {
  struct Dispatch {
    int pipeline;
    Parameters parameters;
    uint32_t groups;
  };
  std::shared_ptr<void> storage_lease;
  std::vector<DeviceAllocationUnique> workspace;
  std::vector<std::unique_ptr<Pipeline>> pipelines;
  std::vector<std::unique_ptr<ShaderResourceSet>> bindings;
  std::vector<Dispatch> dispatches;

  void record(vulkan::VulkanCommandList &list) {
    list.memory_barrier();
    for (size_t i = 0; i < dispatches.size(); ++i) {
      const auto &dispatch = dispatches[i];
      list.bind_pipeline(pipelines[dispatch.pipeline].get());
      TI_ERROR_IF(
          list.bind_shader_resources(bindings[i].get()) != RhiResult::success,
          "Parallel Sort shader binding failed.");
      list.push_constants(&dispatch.parameters, sizeof(Parameters));
      TI_ERROR_IF(list.dispatch(dispatch.groups) != RhiResult::success,
                  "Parallel Sort dispatch recording failed.");
      list.memory_barrier();
    }
  }
};

class SortPlan final : public VulkanParallelSortPlanCore {
 public:
  std::shared_ptr<SortResources> resources;
  std::function<void(CommandList *)> replay;
  std::unordered_map<std::string, uint64_t> facts;

  void record(CommandList *commands, bool inline_recording) override {
    auto *list = dynamic_cast<vulkan::VulkanCommandList *>(commands);
    TI_ERROR_IF(!list, "Parallel Sort requires a Vulkan command list.");
    if (inline_recording) {
      list->begin_external_compute(resources);
      resources->record(*list);
    } else {
      TI_ERROR_IF(!replay,
                  "Parallel Sort plan has no retained secondary recording.");
      replay(commands);
    }
  }
  std::unordered_map<std::string, uint64_t> statistics() const override {
    return facts;
  }
};
}  // namespace

std::shared_ptr<VulkanParallelSortPlanCore> make_vulkan_parallel_sort_plan(
    vulkan::VulkanDevice &device,
    DevicePtr keys,
    std::optional<DevicePtr> values,
    std::shared_ptr<void> storage_lease,
    uint64_t key_count,
    const std::vector<std::vector<uint32_t>> &shaders,
    bool retain_secondary) {
  const auto bytes = vulkan_parallel_sort_storage_bytes(key_count);
  const auto &limits = device.get_vk_physical_device_props().limits;
  if (!storage_lease)
    throw std::invalid_argument(
        "Parallel Sort requires storage lifetime leases");
  auto range = [&](DevicePtr ptr) {
    if (ptr.device != &device ||
        ptr.offset %
            std::max<uint64_t>(4, limits.minStorageBufferOffsetAlignment) ||
        bytes > limits.maxStorageBufferRange)
      throw std::invalid_argument(
          "Parallel Sort storage violates device/alignment/range requirements");
    return resolve_native_storage_range(Arch::vulkan, ptr, bytes);
  };
  const auto key_range = range(keys);
  if (values && !key_range.proven_disjoint(range(*values)))
    throw std::invalid_argument(
        "Parallel Sort keys and payload must not overlap");
  if (shaders.size() != 3 && shaders.size() != 5)
    throw std::invalid_argument(
        "Parallel Sort requires three fused or five original shaders");
  for (const auto &shader : shaders)
    if (shader.size() < 5 || shader.front() != 0x07230203u ||
        shader.size() > std::numeric_limits<size_t>::max() / sizeof(uint32_t))
      throw std::invalid_argument(
          "Parallel Sort requires valid SPIR-V binary headers");
  const auto &caps = device.get_caps();
  TI_ERROR_IF(!caps.get(DeviceCapability::spirv_has_subgroup_basic) ||
                  !caps.get(DeviceCapability::spirv_has_subgroup_arithmetic) ||
                  !caps.get(DeviceCapability::spirv_has_subgroup_ballot),
              "Parallel Sort requires compute subgroup basic/arithmetic/ballot "
              "support.");
  VkPhysicalDeviceSubgroupProperties subgroup{};
  subgroup.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
  VkPhysicalDeviceProperties2 properties2{};
  properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  properties2.pNext = &subgroup;
  TI_ERROR_IF(
      !device.vk_device().instance_functions().vkGetPhysicalDeviceProperties2,
      "Parallel Sort requires Vulkan 1.1 subgroup properties.");
  device.vk_device().instance_functions().vkGetPhysicalDeviceProperties2(
      device.vk_physical_device(), &properties2);
  TI_ERROR_IF(
      !(subgroup.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) ||
          !(subgroup.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT),
      "Parallel Sort requires compute subgroup shuffle support.");

  const bool fused = shaders.size() == 3;
  const int steps = fused ? 3 : 5;
  auto resources = std::make_shared<SortResources>();
  resources->device = device.vk_device();
  resources->storage_lease = std::move(storage_lease);
  uint64_t workspace_bytes = 0;
  auto allocate = [&](uint64_t size) {
    if (size > limits.maxStorageBufferRange)
      throw std::invalid_argument(
          "Parallel Sort workspace exceeds storage-buffer device limits");
    Device::AllocParams params;
    params.size = size;
    params.usage = AllocUsage::Storage;
    auto [allocation, result] = device.allocate_memory_unique(params);
    TI_ERROR_IF(result != RhiResult::success,
                "Parallel Sort workspace allocation failed: {}", int(result));
    const DevicePtr ptr = allocation->get_ptr();
    resources->workspace.push_back(std::move(allocation));
    workspace_bytes += size;
    return ptr;
  };
  const uint32_t n = uint32_t(key_count);
  const uint32_t blocks = (uint64_t(n) + 511) / 512;
  // Existing scheduling choice, not a key-count limit: each group processes
  // additional blocks. Keep the original prefix workload and shader sequence.
  const uint32_t groups = std::min(blocks, 1024u);
  const uint32_t reduced_per_bin = (groups + 511) / 512;
  Parameters parameters{n,
                        int32_t(blocks / groups),
                        groups,
                        blocks % groups,
                        reduced_per_bin,
                        16 * reduced_per_bin,
                        0};
  const auto scratch_key = allocate(bytes);
  const auto scratch_value = values ? allocate(bytes) : keys;
  const uint64_t sums_bytes = uint64_t(groups) * 16 * 4;
  const uint64_t reduced_bytes =
      uint64_t(fused ? groups * 16 : parameters.scan_values) * 4;
  const auto sums = allocate(sums_bytes);
  const auto reduced = allocate(reduced_bytes);
  const auto value = values.value_or(keys);
  constexpr const char *names[] = {"count", "reduce", "scan", "scan_add",
                                   "scatter"};
  for (int step = 0; step < steps; ++step) {
    const auto &shader = shaders[step];
    PipelineSourceDesc source{PipelineSourceType::spirv_binary, shader.data(),
                              shader.size() * 4, PipelineStageType::compute};
    auto [pipeline, result] = device.create_pipeline_unique(
        source, std::string("fidelityfx_parallel_sort_") +
                    (fused ? (step == 0   ? "count"
                              : step == 1 ? "prefix"
                                          : "scatter")
                           : names[step]));
    TI_ERROR_IF(result != RhiResult::success,
                "Parallel Sort pipeline creation failed: {}", int(result));
    resources->pipelines.push_back(std::move(pipeline));
  }
  const uint32_t dispatches[] = {groups, parameters.scan_values, 1,
                                 parameters.scan_values, groups};
  for (uint32_t pass = 0; pass < 8; ++pass) {
    parameters.shift = pass * 4;
    for (int index = 0; index < steps; ++index) {
      const int step = fused && index == 2 ? 4 : index;
      auto binding =
          std::unique_ptr<ShaderResourceSet>(device.create_resource_set());
      if (step == 0 || step == 4) {
        binding->rw_buffer(0, pass % 2 ? scratch_key : keys, bytes);
        binding->rw_buffer(4, fused && step == 4 ? reduced : sums,
                           fused && step == 4 ? reduced_bytes : sums_bytes);
      }
      if (step == 4) {
        binding->rw_buffer(1, pass % 2 ? keys : scratch_key, bytes);
        if (values) {
          binding->rw_buffer(2, pass % 2 ? scratch_value : value, bytes);
          binding->rw_buffer(3, pass % 2 ? value : scratch_value, bytes);
        }
      } else if (step == 1) {
        binding->rw_buffer(4, sums, sums_bytes);
        binding->rw_buffer(5, reduced, reduced_bytes);
      } else if (step == 2 || step == 3) {
        binding->rw_buffer(6, step == 2 ? reduced : sums,
                           step == 2 ? reduced_bytes : sums_bytes);
        binding->rw_buffer(7, step == 2 ? reduced : sums,
                           step == 2 ? reduced_bytes : sums_bytes);
        if (step == 3)
          binding->rw_buffer(8, reduced, reduced_bytes);
      }
      resources->bindings.push_back(std::move(binding));
      resources->dispatches.push_back(
          {index, parameters, fused && step == 1 ? 1 : dispatches[step]});
    }
  }
  auto plan = std::make_shared<SortPlan>();
  plan->resources = resources;
  if (retain_secondary) {
    auto commands = device.get_compute_stream()->new_secondary_command_list();
    TI_ERROR_IF(!commands,
                "Parallel Sort requires secondary compute recording.");
    auto *list = static_cast<vulkan::VulkanCommandList *>(commands.get());
    resources->record(*list);
    plan->replay = list->finalize_secondary(resources);
    TI_ERROR_IF(!plan->replay,
                "Parallel Sort requires secondary compute recording.");
  }
  const auto &properties = device.get_vk_physical_device_props();
  plan->facts = {{"key_count", n},
                 {"payload", values ? 1 : 0},
                 {"dispatch_count", uint64_t(steps * 8)},
                 {"barrier_count", uint64_t(steps * 8 + 1)},
                 {"fused_prefix", fused ? 1 : 0},
                 {"threadgroups", groups},
                 {"workspace_bytes", workspace_bytes},
                 {"device_copy_count", 0},
                 {"device_vendor_id", properties.vendorID},
                 {"device_id", properties.deviceID},
                 {"vulkan_driver_version", properties.driverVersion}};
  return plan;
}
}  // namespace taichi::lang
#endif
