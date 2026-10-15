#include "taichi/program/program.h"

#include "taichi/program/ndarray.h"
#include "taichi/runtime/gfx/graph_recording.h"

#ifdef TI_WITH_VULKAN
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/program/storage_view.h"
#include "taichi/runtime/vulkan_fft_plan.h"

namespace taichi::lang {

// Program-local source identity. Recorded commands retain the shared core,
// while source commands only weakly reference this public handle generation.
class VulkanFftPlan {
 public:
  std::shared_ptr<VulkanFftPlanCore> core;
  DeviceAllocation allocation;
  Program *program{nullptr};
  std::vector<int> shape;
  bool public_open{true};
};

std::uint64_t Program::create_vulkan_fft_plan(
    const std::string &adapter_path,
    Ndarray *data,
    const std::vector<int> &dimensions,
    int batches,
    int direction,
    bool normalize_inverse) {
  return create_vulkan_fft_recipe_plan(adapter_path, data, dimensions, batches,
                                       direction, normalize_inverse, 0);
}

std::uint64_t Program::create_vulkan_fft_recipe_plan(
    const std::string &adapter_path,
    Ndarray *data,
    const std::vector<int> &dimensions,
    int batches,
    int direction,
    bool normalize_inverse,
    int batch_tile) {
  std::lock_guard<std::recursive_mutex> lock(
      runtime_resource_submission_mutex_);
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan FFT requires the Vulkan backend.");
  TI_ERROR_IF(!data || data->owning_program() != this ||
                  data->dtype != PrimitiveType::f32 || dimensions.empty() ||
                  dimensions.size() > 3 || batches <= 0,
              "Vulkan FFT requires a local compact scalar-f32 ndarray and "
              "one to three positive transform dimensions.");
  TI_ERROR_IF(
      batch_tile < 0 || batch_tile > batches,
      "Vulkan FFT batch tile must be zero (legacy) or in [1, batches].");
  std::vector<int> shape;
  if (batches > 1) {
    shape.push_back(batches);
  }
  for (int dimension : dimensions) {
    TI_ERROR_IF(dimension <= 0, "Vulkan FFT dimensions must be positive.");
    shape.push_back(dimension);
  }
  shape.push_back(2);
  TI_ERROR_IF(data->shape != shape,
              "Vulkan FFT ndarray shape must match compact complex batches.");
  auto leases =
      std::make_shared<NdarrayLaunchLeases>(acquire_ndarray_leases({data}));
  auto *device = static_cast<vulkan::VulkanDevice *>(get_graphics_device());
  const auto allocation = data->get_device_allocation();
  TI_ERROR_IF(allocation.device != device,
              "Vulkan FFT storage belongs to another device.");
  VulkanFftPlanConfig config;
  config.dimensions.assign(dimensions.begin(), dimensions.end());
  config.batches = batches;
  config.direction = direction;
  config.normalize_inverse = normalize_inverse;
  config.batch_tile = batch_tile;
  auto plan = std::make_shared<VulkanFftPlan>();
  plan->core = make_vulkan_fft_plan(
      *device, allocation, data->get_nelement() * data->get_element_size(),
      std::move(leases), config, adapter_path);
  plan->allocation = allocation;
  plan->program = this;
  plan->shape = shape;
  const auto handle = next_vulkan_fft_plan_handle_++;
  vulkan_fft_plans_.emplace(handle, std::move(plan));
  return handle;
}

void Program::vulkan_fft_execute(std::uint64_t handle) {
  std::lock_guard<std::recursive_mutex> lock(
      runtime_resource_submission_mutex_);
  auto entry = vulkan_fft_plans_.find(handle);
  TI_ERROR_IF(entry == vulkan_fft_plans_.end(), "Vulkan FFT plan is closed.");
  auto plan = entry->second;
  enqueue_compute_op_lambda(
      [plan](Device *, CommandList *commands) {
        auto *list = static_cast<vulkan::VulkanCommandList *>(commands);
        list->memory_barrier();
        plan->core->record(commands, false);
        list->memory_barrier();
      },
      {});
  // VkFFT's secondary command sequence is already recorded. The enclosing
  // runtime-ordered Graph call currently retains its ordinary host dispatch;
  // it is not a claim of one enclosing native Graph replay region.
  mark_runtime_submission_pending();
}

namespace {
class VulkanFftGraphCommand final : public gfx::ExternalGraphCommand {
 public:
  VulkanFftGraphCommand(std::shared_ptr<VulkanFftPlan> plan, std::string name)
      : plan_(plan), name_(std::move(name)), rank_(plan->shape.size()) {
  }
  std::vector<aot::Arg> arguments() const override {
    return {aot::Arg(aot::ArgKind::kNdarray, name_, PrimitiveType::f32, rank_)};
  }
  void validate(
      Program &program,
      const std::unordered_map<std::string, aot::IValue> &args) const override {
    const auto plan = plan_.lock();
    TI_ERROR_IF(!plan || !plan->public_open,
                "Vulkan FFT source command is closed");
    TI_ERROR_IF(plan->program != &program,
                "Vulkan FFT command belongs to another runtime");
    auto found = args.find(name_);
    TI_ERROR_IF(found == args.end() ||
                    found->second.tag != aot::ArgKind::kNdarray ||
                    !found->second.val,
                "Vulkan FFT command requires its original ndarray binding");
    const auto &value = found->second;
    const auto *array = reinterpret_cast<const Ndarray *>(value.val);
    program.validate_ndarrays_for_external_submission(&array, 1);
    TI_ERROR_IF(array->get_device_allocation() != plan->allocation ||
                    array->shape != plan->shape ||
                    array->dtype != PrimitiveType::f32,
                "Vulkan FFT command requires its original compact storage");
    if (value.runtime_storage) {
      const auto resolved =
          program.resolve_runtime_storage_argument_under_graph_guard(
              *value.runtime_storage);
      TI_ERROR_IF(!resolved.valid || resolved.allocation != plan->allocation ||
                      resolved.byte_offset != 0 ||
                      resolved.byte_size !=
                          array->get_nelement() * array->get_element_size(),
                  "Vulkan FFT command cannot bind a sliced storage view");
    }
  }
  void record(Device *, CommandList *commands) const override {
    // Source tokens do not extend a Vulkan device lifetime. Only the recorded
    // command-buffer refs acquire ownership, and the runtime retires those
    // before device destruction. Validation and this append share its guard.
    const auto plan = plan_.lock();
    TI_ASSERT(plan && plan->public_open);
    plan->core->record(commands);
  }
  bool supports_inline_recording() const override {
    const auto plan = plan_.lock();
    return plan && plan->core->supports_inline_recording();
  }

 private:
  std::weak_ptr<VulkanFftPlan> plan_;
  std::string name_;
  std::size_t rank_;
};
}  // namespace

std::shared_ptr<gfx::ExternalGraphCommand> Program::vulkan_fft_graph_command(
    std::uint64_t handle,
    const std::string &binding_name) {
  std::lock_guard<std::recursive_mutex> lock(
      runtime_resource_submission_mutex_);
  auto found = vulkan_fft_plans_.find(handle);
  TI_ERROR_IF(found == vulkan_fft_plans_.end() || binding_name.empty(),
              "Vulkan FFT command requires an open plan and a binding name");
  return std::make_shared<VulkanFftGraphCommand>(found->second, binding_name);
}

std::unordered_map<std::string, std::uint64_t>
Program::vulkan_fft_plan_statistics(std::uint64_t handle) {
  std::lock_guard<std::recursive_mutex> lock(
      runtime_resource_submission_mutex_);
  auto entry = vulkan_fft_plans_.find(handle);
  TI_ERROR_IF(entry == vulkan_fft_plans_.end(), "Vulkan FFT plan is closed.");
  return entry->second->core->statistics();
}

void Program::destroy_vulkan_fft_plan(std::uint64_t handle) {
  std::lock_guard<std::recursive_mutex> lock(
      runtime_resource_submission_mutex_);
  // In-flight and cached command buffers retain their own plan/storage lease.
  const auto found = vulkan_fft_plans_.find(handle);
  if (found != vulkan_fft_plans_.end()) {
    found->second->public_open = false;
  }
  vulkan_fft_plans_.erase(handle);
}

void Program::vulkan_clear_fft_plans() {
  std::lock_guard<std::recursive_mutex> lock(
      runtime_resource_submission_mutex_);
  for (const auto &[handle, plan] : vulkan_fft_plans_) {
    plan->public_open = false;
  }
  vulkan_fft_plans_.clear();
}
}  // namespace taichi::lang

#else
namespace taichi::lang {
std::uint64_t Program::create_vulkan_fft_plan(const std::string &,
                                              Ndarray *,
                                              const std::vector<int> &,
                                              int,
                                              int,
                                              bool) {
  TI_ERROR("Vulkan FFT is unavailable in this build.");
}
std::uint64_t Program::create_vulkan_fft_recipe_plan(const std::string &,
                                                     Ndarray *,
                                                     const std::vector<int> &,
                                                     int,
                                                     int,
                                                     bool,
                                                     int) {
  TI_ERROR("Vulkan FFT is unavailable in this build.");
}
void Program::vulkan_fft_execute(std::uint64_t) {
  TI_ERROR("Vulkan FFT is unavailable in this build.");
}
std::unordered_map<std::string, std::uint64_t>
Program::vulkan_fft_plan_statistics(std::uint64_t) {
  TI_ERROR("Vulkan FFT is unavailable in this build.");
}
void Program::destroy_vulkan_fft_plan(std::uint64_t) {
}
void Program::vulkan_clear_fft_plans() {
}
std::shared_ptr<gfx::ExternalGraphCommand> Program::vulkan_fft_graph_command(
    std::uint64_t,
    const std::string &) {
  TI_ERROR("Vulkan FFT is unavailable in this build.");
}
}  // namespace taichi::lang
#endif
