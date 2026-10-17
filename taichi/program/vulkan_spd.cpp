#include "taichi/program/program.h"
#include "taichi/program/texture.h"
#include "taichi/runtime/gfx/graph_recording.h"
#include "taichi/runtime/vulkan_spd.h"

#ifdef TI_WITH_VULKAN
#include "taichi/rhi/vulkan/vulkan_device.h"

namespace taichi::lang {
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
  auto *device = static_cast<vulkan::VulkanDevice *>(get_graphics_device());
  const auto parameters = [](Texture *texture) {
    ImageParams p{};
    const auto size = texture->get_size();
    p.dimension = texture->get_dimension();
    p.format = texture->get_buffer_format();
    p.x = size[0];
    p.y = size[1];
    p.z = size[2];
    p.mip_levels = texture->get_mip_levels();
    return p;
  };
  const auto src = source->get_device_allocation();
  const auto dst = output->get_device_allocation();
  auto prepared = prepare_vulkan_spd(
      *device, parameters(source), src, parameters(output), dst, shader,
      std::make_shared<TextureLaunchLeases>(acquire_texture_leases({source, output})));
  auto plan = std::make_shared<VulkanSpdPlan>();
  plan->program = this;
  plan->record_inline = std::move(prepared.record_inline);
  plan->replay = std::move(prepared.replay);
  plan->statistics = std::move(prepared.statistics);
  plan->images = {{src, ImageLayout::shader_read_write, ImageLayout::shader_read},
                  {dst, ImageLayout::shader_read_write, ImageLayout::shader_read}};
  // Keep initialization ordered on this Program's existing submission path.
  auto initialize_replay = std::move(prepared.initialize);
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
