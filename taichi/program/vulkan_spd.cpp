#include "taichi/program/program.h"
#include "taichi/program/texture.h"
#include "taichi/runtime/gfx/graph_recording.h"

#ifdef TI_WITH_VULKAN
#include "taichi/rhi/vulkan/vulkan_device.h"

namespace taichi::lang {
namespace {
struct SpdResources final : vkapi::DeviceObj {
  std::shared_ptr<void> textures;
  DeviceAllocationUnique counter;
  std::unique_ptr<Pipeline> pipeline;
  std::unique_ptr<ShaderResourceSet> bindings;
};
}  // namespace

class VulkanSpdPlan {
 public:
  Program *program{nullptr};
  bool public_open{true};
  std::function<void(CommandList *)> replay;
  std::function<void(CommandList *)> record_inline;
  std::vector<ComputeOpImageRef> images;
  std::unordered_map<std::string, std::uint64_t> statistics;
};

std::uint64_t Program::create_vulkan_spd_plan(
    Texture *source,
    Texture *output,
    const std::vector<std::uint32_t> &shader) {
  auto guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "SPD requires the Vulkan backend.");
  TI_ERROR_IF(!source || !output || source == output ||
                  source->owning_program() != this ||
                  output->owning_program() != this ||
                  source->get_dimension() != ImageDimension::d2D ||
                  output->get_dimension() != ImageDimension::d2D,
              "SPD requires distinct local 2D textures.");
  const auto size = source->get_size();
  const auto dest_size = output->get_size();
  const auto levels = output->get_mip_levels();
  const auto src_format = source->get_buffer_format();
  const auto dst_format = output->get_buffer_format();
  TI_ERROR_IF(
      size[0] < 2 || size[1] < 2 || size[0] > 4096 || size[1] > 4096 ||
          dest_size[0] != size[0] / 2 || dest_size[1] != size[1] / 2 ||
          levels < 1 || levels > 12 || (size[0] >> levels) < 1 ||
          (size[1] >> levels) < 1,
      "SPD requires half-size output and 2D mip levels within a 2..4096 source.");
  TI_ERROR_IF(
      !((src_format == BufferFormat::r32f && dst_format == BufferFormat::r32f) ||
        ((src_format == BufferFormat::rgba8 ||
          src_format == BufferFormat::rgba32f) &&
         dst_format == BufferFormat::rgba32f)),
      "SPD requires R32F to R32F or RGBA8/RGBA32F to RGBA32F.");
  TI_ERROR_IF(shader.empty() || shader.front() != 0x07230203u ||
                  shader.size() > (4u << 20),
              "SPD requires a bounded SPIR-V binary.");
  auto *dev = static_cast<vulkan::VulkanDevice *>(get_graphics_device());
  const auto &props = dev->get_vk_physical_device_props();
  TI_ERROR_IF(props.limits.maxComputeWorkGroupInvocations < 256 ||
                  props.limits.maxComputeWorkGroupSize[0] < 256 ||
                  props.limits.maxComputeSharedMemorySize < 4100 ||
                  props.limits.maxPerStageDescriptorStorageImages < levels + 1,
              "SPD compute or storage-image limits are unavailable.");
  auto resources = std::make_shared<SpdResources>();
  resources->device = dev->vk_device();
  resources->textures = std::make_shared<TextureLaunchLeases>(
      acquire_texture_leases({source, output}));
  const auto src = source->get_device_allocation();
  const auto dst = output->get_device_allocation();
  TI_ERROR_IF(src.device != dev || dst.device != dev || src == dst,
              "SPD texture allocations must be distinct on the active device.");
  auto [counter, alloc_result] =
      dev->allocate_memory_unique({4, false, false, false, AllocUsage::Storage});
  TI_ERROR_IF(alloc_result != RhiResult::success, "SPD counter allocation failed.");
  resources->counter = std::move(counter);
  PipelineSourceDesc desc{PipelineSourceType::spirv_binary, shader.data(),
                          shader.size() * 4, PipelineStageType::compute};
  auto [pipeline, result] = dev->create_pipeline_unique(desc, "fidelityfx_spd");
  TI_ERROR_IF(result != RhiResult::success, "SPD pipeline creation failed.");
  resources->pipeline = std::move(pipeline);
  resources->bindings.reset(dev->create_resource_set());
  resources->bindings->rw_image(0, src, 0);
  resources->bindings->rw_buffer(1, *resources->counter);
  for (int mip = 0; mip < levels; ++mip) {
    resources->bindings->rw_image(mip + 2, dst, mip);
  }
  const uint32_t groups_x = (size[0] + 63) / 64;
  const uint32_t groups_y = (size[1] + 63) / 64;
  auto plan = std::make_shared<VulkanSpdPlan>();
  plan->program = this;
  plan->record_inline = [resources, groups_x, groups_y,
                         levels](CommandList *commands) {
    auto *list = static_cast<vulkan::VulkanCommandList *>(commands);
    // Attach the cold pipeline/texture/counter owner to the final command
    // buffer, without retaining a public plan token across runtime reset.
    list->begin_external_compute(resources);
    list->bind_pipeline(resources->pipeline.get());
    TI_ERROR_IF(list->bind_shader_resources(resources->bindings.get()) !=
                    RhiResult::success,
                "SPD inline binding failed.");
    const uint32_t constants[]{uint32_t(levels), groups_x * groups_y};
    list->push_constants(constants, sizeof(constants));
    TI_ERROR_IF(list->dispatch(groups_x, groups_y) != RhiResult::success,
                "SPD inline dispatch failed.");
  };
  auto commands = dev->get_compute_stream()->new_secondary_command_list();
  TI_ERROR_IF(!commands, "SPD requires secondary compute recording.");
  auto *list = static_cast<vulkan::VulkanCommandList *>(commands.get());
  list->memory_barrier();
  plan->record_inline(list);
  list->memory_barrier();
  plan->replay = list->finalize_secondary(resources);
  TI_ERROR_IF(!plan->replay, "SPD secondary command finalization failed.");
  plan->images = {{src, ImageLayout::shader_read_write, ImageLayout::shader_read},
                  {dst, ImageLayout::shader_read_write, ImageLayout::shader_read}};
  plan->statistics = {{"width", uint64_t(size[0])},
                      {"height", uint64_t(size[1])},
                      {"levels", uint64_t(levels)},
                      {"dispatch_count", 1},
                      {"barrier_count", 2},
                      {"workspace_bytes", 4},
                      {"inline_recording_available", 1},
                      {"initial_counter_clear_count", 1},
                      {"per_run_counter_clear_count", 0},
                      {"threadgroups", groups_x * groups_y},
                      {"device_copy_count", 0},
                      {"device_vendor_id", props.vendorID},
                      {"device_id", props.deviceID},
                      {"vulkan_driver_version", props.driverVersion}};
  // Initialize once on the same stream, without host synchronization. Secondary
  // ownership retains counter/textures even if a later publication step fails.
  auto initialize = dev->get_compute_stream()->new_secondary_command_list();
  TI_ERROR_IF(!initialize, "SPD counter initialization recording failed.");
  initialize->buffer_fill(resources->counter->get_ptr(0), 4, 0);
  initialize->memory_barrier();
  auto initialize_replay =
      static_cast<vulkan::VulkanCommandList *>(initialize.get())
          ->finalize_secondary(resources);
  TI_ERROR_IF(!initialize_replay,
              "SPD counter initialization finalization failed.");
  const auto handle = next_vulkan_spd_plan_handle_++;
  vulkan_spd_plans_.emplace(handle, plan);
  try {
    enqueue_compute_op_lambda(
        [initialize_replay](Device *, CommandList *cmd) {
          initialize_replay(cmd);
        },
        {});
    mark_runtime_submission_pending();
  } catch (...) {
    vulkan_spd_plans_.erase(handle);
    throw;
  }
  return handle;
}

void Program::vulkan_spd_execute(std::uint64_t handle) {
  auto guard = acquire_runtime_resource_submission_guard();
  const auto found = vulkan_spd_plans_.find(handle);
  TI_ERROR_IF(found == vulkan_spd_plans_.end(), "SPD plan is closed.");
  auto plan = found->second;
  enqueue_compute_op_lambda(
      [plan](Device *, CommandList *commands) { plan->replay(commands); },
      plan->images);
  mark_runtime_submission_pending();
}

std::unordered_map<std::string, std::uint64_t>
Program::vulkan_spd_plan_statistics(std::uint64_t handle) {
  auto guard = acquire_runtime_resource_submission_guard();
  const auto found = vulkan_spd_plans_.find(handle);
  TI_ERROR_IF(found == vulkan_spd_plans_.end(), "SPD plan is closed.");
  return found->second->statistics;
}
void Program::destroy_vulkan_spd_plan(std::uint64_t handle) {
  std::lock_guard<std::recursive_mutex> lock(runtime_resource_submission_mutex_);
  const auto found = vulkan_spd_plans_.find(handle);
  if (found != vulkan_spd_plans_.end()) {
    found->second->public_open = false;
  }
  vulkan_spd_plans_.erase(handle);
}
void Program::vulkan_clear_spd_plans() {
  std::lock_guard<std::recursive_mutex> lock(runtime_resource_submission_mutex_);
  for (const auto &[handle, plan] : vulkan_spd_plans_) {
    plan->public_open = false;
  }
  vulkan_spd_plans_.clear();
}

namespace {
class VulkanSpdGraphCommand final : public gfx::ExternalGraphCommand {
 public:
  VulkanSpdGraphCommand(const std::shared_ptr<VulkanSpdPlan> &plan,
                       std::string source,
                       std::string output)
      : plan_(plan), names_{std::move(source), std::move(output)} {
  }
  std::vector<aot::Arg> arguments() const override {
    return {aot::Arg(aot::ArgKind::kTexture, names_[0], PrimitiveType::f32, 4, {2, 2}),
            aot::Arg(aot::ArgKind::kTexture, names_[1], PrimitiveType::f32, 4, {2, 2})};
  }
  void validate(
      Program &program,
      const std::unordered_map<std::string, aot::IValue> &args) const override {
    const auto plan = plan_.lock();
    TI_ERROR_IF(!plan || !plan->public_open || plan->program != &program,
                "SPD source command is closed or belongs to another runtime");
    for (int i = 0; i < 2; ++i) {
      const auto found = args.find(names_[i]);
      TI_ERROR_IF(found == args.end() ||
                      found->second.tag != aot::ArgKind::kTexture ||
                      !found->second.val,
                  "SPD command requires its original texture bindings");
      const auto *texture = reinterpret_cast<Texture *>(found->second.val);
      TI_ERROR_IF(texture->owning_program() != &program ||
                      texture->get_device_allocation() != plan->images[i].image,
                  "SPD command requires its original texture bindings");
    }
  }
  void record(Device *, CommandList *commands) const override {
    const auto plan = plan_.lock();
    TI_ASSERT(plan && plan->public_open);
    plan->record_inline(commands);
  }
  bool supports_inline_recording() const override {
    return true;
  }
  std::vector<std::pair<DeviceAllocation, ImageLayout>> image_uses()
      const override {
    const auto plan = plan_.lock();
    TI_ASSERT(plan && plan->public_open);
    return {{plan->images[0].image, ImageLayout::shader_read_write},
            {plan->images[1].image, ImageLayout::shader_read_write}};
  }

 private:
  std::weak_ptr<VulkanSpdPlan> plan_;
  std::array<std::string, 2> names_;
};
}  // namespace
std::shared_ptr<gfx::ExternalGraphCommand> Program::vulkan_spd_graph_command(
    std::uint64_t handle,
    const std::string &source,
    const std::string &output) {
  auto guard = acquire_runtime_resource_submission_guard();
  const auto found = vulkan_spd_plans_.find(handle);
  TI_ERROR_IF(found == vulkan_spd_plans_.end() || source.empty() ||
                  output.empty() || source == output,
              "SPD command requires an open plan and distinct binding names");
  return std::make_shared<VulkanSpdGraphCommand>(found->second, source, output);
}
}  // namespace taichi::lang
#else
namespace taichi::lang {
std::uint64_t Program::create_vulkan_spd_plan(
    Texture *, Texture *, const std::vector<std::uint32_t> &) {
  TI_ERROR("SPD is unavailable in this build.");
}
void Program::vulkan_spd_execute(std::uint64_t) {
  TI_ERROR("SPD is unavailable in this build.");
}
std::unordered_map<std::string, std::uint64_t>
Program::vulkan_spd_plan_statistics(std::uint64_t) {
  TI_ERROR("SPD is unavailable in this build.");
}
void Program::destroy_vulkan_spd_plan(std::uint64_t) {
}
void Program::vulkan_clear_spd_plans() {
}
std::shared_ptr<gfx::ExternalGraphCommand> Program::vulkan_spd_graph_command(
    std::uint64_t, const std::string &, const std::string &) {
  TI_ERROR("SPD is unavailable in this build.");
}
}  // namespace taichi::lang
#endif
