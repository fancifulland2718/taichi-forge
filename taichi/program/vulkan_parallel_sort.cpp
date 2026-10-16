#include "taichi/program/program.h"
#include "taichi/program/ndarray.h"

#ifdef TI_WITH_VULKAN
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/runtime/vulkan_parallel_sort_plan.h"

namespace taichi::lang {
class VulkanParallelSortPlan {
 public:
  std::shared_ptr<VulkanParallelSortPlanCore> core;
};

std::uint64_t Program::create_vulkan_parallel_sort_plan(
    Ndarray *keys,
    Ndarray *values,
    const std::vector<std::vector<std::uint32_t>> &shaders) {
  auto guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Parallel Sort requires the Vulkan backend.");
  std::vector<const Ndarray *> arrays{keys};
  if (values)
    arrays.push_back(values);
  auto leases =
      std::make_shared<NdarrayLaunchLeases>(acquire_ndarray_leases(arrays));
  for (const auto *array : arrays)
    TI_ERROR_IF(!array || array->owning_program() != this ||
                    array->dtype != PrimitiveType::u32 ||
                    array->shape.size() != 1 ||
                    !array->get_element_shape().empty() || array->shape[0] <= 0,
                "Parallel Sort requires local, nonempty, scalar u32 ndarrays.");
  TI_ERROR_IF(values && values->shape != keys->shape,
              "Parallel Sort payload must have the keys' shape.");
  auto &device = *static_cast<vulkan::VulkanDevice *>(get_graphics_device());
  auto plan = std::make_shared<VulkanParallelSortPlan>();
  plan->core = make_vulkan_parallel_sort_plan(
      device, keys->get_device_allocation().get_ptr(),
      values
          ? std::optional<DevicePtr>(values->get_device_allocation().get_ptr())
          : std::nullopt,
      std::move(leases), keys->shape[0], shaders, true);
  const auto handle = next_vulkan_parallel_sort_plan_handle_++;
  vulkan_parallel_sort_plans_.emplace(handle, std::move(plan));
  return handle;
}

void Program::vulkan_parallel_sort_execute(std::uint64_t handle) {
  auto guard = acquire_runtime_resource_submission_guard();
  const auto found = vulkan_parallel_sort_plans_.find(handle);
  TI_ERROR_IF(found == vulkan_parallel_sort_plans_.end(),
              "Parallel Sort plan is closed.");
  auto plan = found->second;
  enqueue_compute_op_lambda(
      [plan](Device *, CommandList *commands) {
        plan->core->record(commands, false);
      },
      {});
  mark_runtime_submission_pending();
}

std::unordered_map<std::string, std::uint64_t>
Program::vulkan_parallel_sort_plan_statistics(std::uint64_t handle) {
  auto guard = acquire_runtime_resource_submission_guard();
  const auto found = vulkan_parallel_sort_plans_.find(handle);
  TI_ERROR_IF(found == vulkan_parallel_sort_plans_.end(),
              "Parallel Sort plan is closed.");
  return found->second->core->statistics();
}

void Program::destroy_vulkan_parallel_sort_plan(std::uint64_t handle) {
  std::lock_guard<std::recursive_mutex> lock(
      runtime_resource_submission_mutex_);
  vulkan_parallel_sort_plans_.erase(handle);
}
void Program::vulkan_clear_parallel_sort_plans() {
  // Teardown runs after submissions close, but before destroying the device.
  // Do not use the live-submission guard on this retirement boundary.
  std::lock_guard<std::recursive_mutex> lock(
      runtime_resource_submission_mutex_);
  vulkan_parallel_sort_plans_.clear();
}
}  // namespace taichi::lang
#else
namespace taichi::lang {
std::uint64_t Program::create_vulkan_parallel_sort_plan(
    Ndarray *,
    Ndarray *,
    const std::vector<std::vector<std::uint32_t>> &) {
  TI_ERROR("Parallel Sort is unavailable in this build.");
}
void Program::vulkan_parallel_sort_execute(std::uint64_t) {
  TI_ERROR("Parallel Sort is unavailable in this build.");
}
std::unordered_map<std::string, std::uint64_t>
Program::vulkan_parallel_sort_plan_statistics(std::uint64_t) {
  TI_ERROR("Parallel Sort is unavailable in this build.");
}
void Program::destroy_vulkan_parallel_sort_plan(std::uint64_t) {
}
void Program::vulkan_clear_parallel_sort_plans() {
}
}  // namespace taichi::lang
#endif
