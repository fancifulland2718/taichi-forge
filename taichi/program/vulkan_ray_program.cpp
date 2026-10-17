#include "taichi/program/program.h"
#include "taichi/program/vulkan_ray_program.h"
#include "taichi/program/storage_view.h"
#include "taichi/program/texture.h"
#include "taichi/runtime/gfx/graph_recording.h"

#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>

#ifdef TI_WITH_VULKAN
#include "taichi/runtime/vulkan_ray_program.h"
#include "taichi/runtime/gfx/kernel_launcher.h"

namespace taichi::lang {
namespace {
using vulkan::VulkanPipeline;
using vulkan::VulkanResourceSet;
using vulkan::VulkanShaderBindingTable;
using ResourceSets = VulkanRayProgramResourceSets;

struct RayLaunchPayload {
  std::shared_ptr<VulkanRayProgramResource> program;
  std::unique_ptr<VulkanShaderBindingTable> sbt;
  ResourceSets sets;
  std::vector<std::function<void(CommandList *)>> scene_reads;
};

class RayLaunchLease final : public PreparedResourceLease {
 public:
  explicit RayLaunchLease(std::shared_ptr<RayLaunchPayload> payload)
      : payload_(std::move(payload)) {
  }
  void clear() noexcept override {
    payload_.reset();
  }
  std::shared_ptr<RayLaunchPayload> acquire() const {
    return payload_;
  }

 private:
  // Program's submission guard serializes prepare/close/record publication.
  std::shared_ptr<RayLaunchPayload> payload_;
};

}  // namespace

class VulkanRayProgramResource {
 public:
  std::unique_ptr<VulkanPipeline> pipeline;
  std::vector<std::weak_ptr<PreparedResourceLease>> launches;
  void close() noexcept {
    for (const auto &entry : launches) {
      if (auto lease = entry.lock())
        lease->clear();
    }
    launches.clear();
  }
};

struct PreparedVulkanRayLaunch::State {
  Program *owner{nullptr};
  std::uint64_t generation{0};
  std::uint64_t dependency{0};
  std::shared_ptr<PreparedNativeStorage> storage;
  std::vector<RuntimeResourceHandle> textures;
  std::vector<ComputeOpImageRef> images;
  std::vector<DeviceAllocation> buffers;
  std::vector<SNodeTreeDependency> trees;
  std::vector<std::uint64_t> dependencies;
  std::array<std::uint32_t, 3> dimensions{};
  std::vector<std::uint8_t> push_constants;
  std::shared_ptr<RayLaunchLease> resources;
  bool initialized{false};

  std::shared_ptr<RayLaunchPayload> payload(Program &program,
                                            bool require_ready = true) const {
    TI_ERROR_IF(
        owner != &program || generation != program.runtime_program_generation(),
        "Vulkan ray launch belongs to another or retired runtime");
    auto result = resources ? resources->acquire() : nullptr;
    TI_ERROR_IF(!result,
                "Vulkan ray launch is closed or its resources were retired");
    TI_ERROR_IF(require_ready && !initialized,
                "Vulkan ray launch requires explicit initialization");
    return result;
  }

  void record(CommandList *commands) const {
    auto payload = resources->acquire();
    TI_ERROR_IF(!payload, "Vulkan ray launch was closed before recording");
    record_vulkan_ray_program(commands, *payload->program->pipeline,
                              *payload->sbt, payload->sets, payload->scene_reads,
                              dimensions, push_constants);
  }
};

bool Program::vulkan_ray_program_available() const {
  if (compile_config().arch != Arch::vulkan || !program_impl_)
    return false;
  auto *device = static_cast<vulkan::VulkanDevice *>(
      const_cast<Program *>(this)->get_compute_device());
  return device && device->vk_caps().ray_tracing_pipeline;
}

std::uint64_t Program::create_vulkan_ray_program(
    const std::vector<VulkanRayProgramShader> &shaders,
    const std::vector<VulkanRayProgramGroup> &groups,
    std::uint32_t recursion_depth,
    bool opacity_micromap) {
  auto guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(!vulkan_ray_program_available(),
              "Vulkan RT pipelines are unavailable");
  auto resource = std::make_shared<VulkanRayProgramResource>();
  resource->pipeline = prepare_vulkan_ray_program_pipeline(
      *static_cast<vulkan::VulkanDevice *>(get_compute_device()), shaders,
      groups, recursion_depth, opacity_micromap);
  std::lock_guard<std::mutex> lock(vulkan_ray_scene_mutex_);
  TI_ERROR_IF(!next_vulkan_ray_resource_handle_,
              "Vulkan ray dependency ID space exhausted");
  const auto handle = next_vulkan_ray_resource_handle_++;
  vulkan_ray_programs_.emplace(handle, std::move(resource));
  return handle;
}

std::shared_ptr<PreparedVulkanRayLaunch> Program::prepare_vulkan_ray_launch(
    std::uint64_t handle,
    const VulkanRayProgramLaunch &launch) {
  auto lifecycle_guard = acquire_snode_tree_lifecycle_read_guard();
  auto guard = acquire_runtime_resource_submission_guard();
  std::shared_ptr<VulkanRayProgramResource> program;
  {
    std::lock_guard<std::mutex> lock(vulkan_ray_scene_mutex_);
    const auto found = vulkan_ray_programs_.find(handle);
    TI_ERROR_IF(found == vulkan_ray_programs_.end(),
                "Vulkan ray program is closed or stale");
    program = found->second;
  }
  auto *device = static_cast<vulkan::VulkanDevice *>(get_compute_device());
  validate_vulkan_ray_program_launch(*device, *program->pipeline,
                                     launch.dimensions, launch.push_constants);
  auto packet = std::make_shared<PreparedVulkanRayLaunch>();
  packet->state = std::make_shared<PreparedVulkanRayLaunch::State>();
  auto &state = *packet->state;
  state.owner = this;
  state.generation = runtime_program_generation();
  state.dimensions = launch.dimensions;
  state.push_constants = launch.push_constants;
  state.dependencies.push_back(handle);
  auto payload = std::make_shared<RayLaunchPayload>();
  payload->program = program;
  state.resources = std::make_shared<RayLaunchLease>(payload);
  std::set<std::pair<std::uint32_t, std::uint32_t>> claimed;
  const auto resource_set = [&](std::uint32_t set,
                                std::uint32_t binding) -> VulkanResourceSet & {
    TI_ERROR_IF(!claimed.emplace(set, binding).second,
                "Duplicate ray descriptor binding");
    auto &result = payload->sets[set];
    if (!result)
      result = std::make_unique<VulkanResourceSet>(device);
    return *result;
  };
  std::vector<const storage::DenseStorageDescriptor *> descriptors;
  std::vector<bool> writable;
  for (const auto &buffer : launch.buffers) {
    TI_ERROR_IF(buffer.uniform && buffer.writable,
                "Uniform ray buffers cannot be writable");
    descriptors.push_back(buffer.storage);
    writable.push_back(buffer.writable);
  }
  state.storage = prepare_native_storage(descriptors, writable);
  state.trees = state.storage->trees_;
  const auto &limits = device->get_vk_physical_device_props().limits;
  for (std::size_t i = 0; i < launch.buffers.size(); ++i) {
    const auto &input = launch.buffers[i];
    const auto &value = state.storage->binding(i);
    const DeviceAllocation allocation{value.pointer.device,
                                      value.pointer.alloc_id};
    const auto alignment = input.uniform
                               ? limits.minUniformBufferOffsetAlignment
                               : limits.minStorageBufferOffsetAlignment;
    const auto maximum = input.uniform ? limits.maxUniformBufferRange
                                       : limits.maxStorageBufferRange;
    const auto usage =
        input.uniform ? AllocUsage::Uniform : AllocUsage::Storage;
    TI_ERROR_IF(allocation.device != device || value.bytes == 0 ||
                    value.bytes > maximum || value.pointer.offset % alignment ||
                    !int(device->allocation_usage(allocation) & usage),
                "Ray buffer range, alignment, usage or device is incompatible "
                "with its descriptor");
    auto &set = resource_set(input.set, input.binding);
    if (input.uniform)
      set.buffer(input.binding, value.pointer, value.bytes);
    else
      set.rw_buffer(input.binding, value.pointer, value.bytes);
    if (std::find(state.buffers.begin(), state.buffers.end(), allocation) ==
        state.buffers.end())
      state.buffers.push_back(allocation);
  }
  for (const auto &input : launch.images) {
    auto *image = input.texture;
    TI_ERROR_IF(
        !image || image->owning_program() != this || image->is_cuda_texture(),
        "Ray images must be managed Vulkan Textures on the active runtime");
    TI_ERROR_IF(input.mip_level >= image->get_mip_levels() ||
                    (!input.storage && input.mip_level),
                "Ray storage mip is out of range or a sampled image requests a "
                "storage-only mip view");
    const auto allocation = image->get_device_allocation();
    const auto layout = input.storage ? ImageLayout::shader_read_write
                                      : ImageLayout::shader_read;
    auto &set = resource_set(input.set, input.binding);
    if (input.storage) {
      VkFormatProperties properties{};
      device->vk_device()
          .instance_functions()
          .vkGetPhysicalDeviceFormatProperties(
              device->vk_physical_device(),
              std::get<2>(device->get_vk_image(allocation)), &properties);
      TI_ERROR_IF(!(properties.optimalTilingFeatures &
                    VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT),
                  "Ray storage Texture format lacks storage-image support");
      set.rw_image(input.binding, allocation, input.mip_level);
    } else {
      program->pipeline->validate_sampled_texture(
          input.set, input.binding, image->sampler_config_.compare_op >= 0);
      set.image(input.binding, allocation, image->sampler_config_);
    }
    auto prior =
        std::find_if(state.images.begin(), state.images.end(),
                     [&](const auto &ref) { return ref.image == allocation; });
    TI_ERROR_IF(prior != state.images.end() && prior->initial_layout != layout,
                "One ray launch cannot bind the same Texture as both sampled "
                "and storage");
    if (prior == state.images.end()) {
      state.images.push_back({allocation, layout, layout});
      state.textures.push_back(image->runtime_resource_handle());
    }
  }
  for (const auto &input : launch.scenes) {
    const auto properties = vulkan_ray_kernel_resource_properties(input.handle);
    TI_ERROR_IF(!launch.hit.empty() &&
                    properties.at("max_sbt_record_offset") >= launch.hit.size(),
                "Ray hit records do not cover the bound instance SBT offsets");
    auto &set = resource_set(input.set, input.binding);
    payload->scene_reads.push_back(prepare_vulkan_ray_shader_binding(
        input.handle, &set, input.binding, state.resources,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, false));
    if (std::find(state.dependencies.begin(), state.dependencies.end(),
                  input.handle) == state.dependencies.end())
      state.dependencies.push_back(input.handle);
  }
  prepare_vulkan_ray_program_bindings(*program->pipeline, payload->sets);
  payload->sbt = prepare_vulkan_ray_program_sbt(
      *device, *program->pipeline, launch.raygen, launch.miss, launch.hit);
  {
    std::lock_guard<std::mutex> lock(vulkan_ray_scene_mutex_);
    TI_ERROR_IF(!next_vulkan_ray_resource_handle_,
                "Vulkan ray dependency ID space exhausted");
    state.dependency = next_vulkan_ray_resource_handle_++;
    state.dependencies.push_back(state.dependency);
    program->launches.erase(
        std::remove_if(program->launches.begin(), program->launches.end(),
                       [](const auto &lease) { return lease.expired(); }),
        program->launches.end());
    program->launches.push_back(state.resources);
  }
  return packet;
}

void Program::initialize_vulkan_ray_launch(
    const std::shared_ptr<PreparedVulkanRayLaunch> &launch) {
  auto guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(!launch || !launch->state, "Vulkan ray launch is missing");
  auto state = launch->state;
  auto payload = state->payload(*this, false);
  if (state->initialized)
    return;
  // Readiness is published only after enqueue succeeds. Failure retires this
  // packet; neither a retry nor Graph binding may use half-initialized state.
  try {
    mark_runtime_submission_pending();
    // Initialization is explicitly submitted, not intercepted as a replay body.
    program_impl_->enqueue_compute_op_lambda(
        [payload](Device *, CommandList *commands) {
          payload->sbt->record_initialization(
              *static_cast<vulkan::VulkanCommandList *>(commands));
        },
        {});
    state->initialized = true;
  } catch (...) {
    state->resources->clear();
    throw;
  }
}

void Program::execute_vulkan_ray_launch(
    const std::shared_ptr<PreparedVulkanRayLaunch> &launch) {
  TI_ERROR_IF(!launch || !launch->state, "Vulkan ray launch is missing");
  const auto state = launch->state;
  TI_ERROR_IF(!state->storage, "Vulkan ray launch is closed");
  with_prepared_native_storage(*state->storage, [&] {
    state->payload(*this);
    TextureLaunchLeases textures;
    for (const auto handle : state->textures) {
      TI_ERROR_IF(handle.index >= texture_view_slots_.size(),
                  "Prepared ray Texture was retired");
      const auto &slot = texture_view_slots_[handle.index];
      TI_ERROR_IF(slot.handle != handle || !slot.view,
                  "Prepared ray Texture was retired");
      if (texture_inflight_leases_.find(texture_lease_key(handle)) ==
          texture_inflight_leases_.end()) {
        auto acquired = texture_resources_.acquire(handle);
        TI_ERROR_IF(acquired.first != TextureResourceRegistry::Result::kSuccess,
                    "Cannot retain ray Texture");
        textures.add(std::move(acquired.second));
      }
    }
    pin_texture_launch_leases(textures);
    mark_runtime_submission_pending();
    enqueue_compute_op_lambda(
        [state](Device *, CommandList *commands) { state->record(commands); },
        state->images);
  });
}

void Program::close_vulkan_ray_launch(
    const std::shared_ptr<PreparedVulkanRayLaunch> &launch) {
  // Closing must work after a backend fault; no new GPU operation or wait.
  std::lock_guard<std::recursive_mutex> guard(
      runtime_resource_submission_mutex_);
  if (!launch || !launch->state)
    return;
  auto &state = *launch->state;
  TI_ERROR_IF(
      state.owner != this || state.generation != runtime_program_generation(),
      "Cannot close a Vulkan ray launch through another runtime");
  if (!state.resources || !state.resources->acquire())
    return;
  if (auto *launcher =
          dynamic_cast<gfx::KernelLauncher *>(&get_kernel_launcher()))
    launcher->runtime()->retire_ray_resource_recordings(state.dependency);
  state.resources->clear();
  state.initialized = false;
  state.storage.reset();
}

void Program::destroy_vulkan_ray_program(std::uint64_t handle) {
  std::lock_guard<std::recursive_mutex> guard(
      runtime_resource_submission_mutex_);
  std::shared_ptr<VulkanRayProgramResource> resource;
  {
    std::lock_guard<std::mutex> lock(vulkan_ray_scene_mutex_);
    const auto found = vulkan_ray_programs_.find(handle);
    if (found == vulkan_ray_programs_.end())
      return;
    resource = std::move(found->second);
    vulkan_ray_programs_.erase(found);
  }
  if (auto *launcher =
          dynamic_cast<gfx::KernelLauncher *>(&get_kernel_launcher()))
    launcher->runtime()->retire_ray_resource_recordings(handle);
  resource->close();
  // Pipeline/layout/SBT/descriptor Vk objects are retained by recorded command
  // buffers. Their existing stream/completion lifetime, not a new registry,
  // keeps in-flight work alive after this logical owner is closed.
}

void Program::vulkan_clear_ray_programs() {
  std::lock_guard<std::mutex> lock(vulkan_ray_scene_mutex_);
  for (const auto &[handle, resource] : vulkan_ray_programs_)
    resource->close();
  vulkan_ray_programs_.clear();
}

std::unordered_map<std::string, std::uint64_t> Program::vulkan_ray_launch_info(
    const std::shared_ptr<PreparedVulkanRayLaunch> &launch) {
  std::lock_guard<std::recursive_mutex> guard(
      runtime_resource_submission_mutex_);
  TI_ERROR_IF(!launch || !launch->state, "Vulkan ray launch is missing");
  const auto &state = *launch->state;
  const auto payload = state.resources ? state.resources->acquire() : nullptr;
  const auto bytes = payload ? payload->sbt->allocated_bytes() : 0;
  return {{"closed", !payload},
          {"initialized", payload && state.initialized},
          {"sbt_requested_bytes", bytes},
          {"upload_requested_bytes",
           payload && !payload->sbt->initialization_recorded() ? bytes : 0},
          {"descriptor_sets", payload ? payload->sets.size() : 0},
          {"width", state.dimensions[0]},
          {"height", state.dimensions[1]},
          {"depth", state.dimensions[2]}};
}

namespace {
class RayProgramGraphCommand final : public gfx::ExternalGraphCommand {
 public:
  RayProgramGraphCommand(std::shared_ptr<PreparedVulkanRayLaunch::State> state,
                         std::shared_ptr<void> arrays,
                         std::shared_ptr<void> textures)
      : state_(std::move(state)),
        arrays_(std::move(arrays)),
        textures_(std::move(textures)) {
  }
  std::vector<aot::Arg> arguments() const override {
    return {};
  }
  void validate(
      Program &program,
      const std::unordered_map<std::string, aot::IValue> &) const override {
    state_->payload(program);
  }
  void record(Device *, CommandList *commands) const override {
    state_->record(commands);
  }
  bool supports_inline_recording() const override {
    return true;
  }
  std::vector<std::pair<DeviceAllocation, ImageLayout>> image_uses()
      const override {
    std::vector<std::pair<DeviceAllocation, ImageLayout>> result;
    for (const auto &image : state_->images)
      result.emplace_back(image.image, image.initial_layout);
    return result;
  }
  std::optional<std::vector<DeviceAllocation>> buffer_uses() const override {
    return state_->buffers;
  }
  std::vector<SNodeTreeDependency> snode_tree_dependencies() const override {
    return state_->trees;
  }
  std::vector<std::uint64_t> ray_resource_dependencies() const override {
    return state_->dependencies;
  }

 private:
  std::shared_ptr<PreparedVulkanRayLaunch::State> state_;
  std::shared_ptr<void> arrays_, textures_;
};
}  // namespace

std::shared_ptr<gfx::ExternalGraphCommand>
Program::vulkan_ray_program_graph_command(
    const std::shared_ptr<PreparedVulkanRayLaunch> &launch) {
  auto lifecycle_guard = acquire_snode_tree_lifecycle_read_guard();
  auto guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(!launch || !launch->state, "Vulkan ray launch is missing");
  const auto state = launch->state;
  state->payload(*this);
  validate_snode_tree_dependencies(state->trees, "Vulkan ray Graph");
  NdarrayLaunchLeases arrays;
  for (const auto handle : state->storage->ndarray_handles_) {
    TI_ERROR_IF(handle.index >= ndarray_view_slots_.size(),
                "Prepared ray Graph ndarray was retired");
    const auto &slot = ndarray_view_slots_[handle.index];
    TI_ERROR_IF(slot.handle != handle || !slot.view || !slot.resource,
                "Prepared ray Graph ndarray was retired");
    auto lease = slot.resource->lease.clone();
    TI_ERROR_IF(!lease, "Cannot retain ray Graph ndarray");
    arrays.add(std::move(lease));
  }
  TextureLaunchLeases textures;
  for (const auto handle : state->textures) {
    auto acquired = texture_resources_.acquire(handle);
    TI_ERROR_IF(acquired.first != TextureResourceRegistry::Result::kSuccess,
                "Cannot retain ray Graph Texture");
    textures.add(std::move(acquired.second));
  }
  return std::make_shared<RayProgramGraphCommand>(
      state, std::make_shared<NdarrayLaunchLeases>(std::move(arrays)),
      std::make_shared<TextureLaunchLeases>(std::move(textures)));
}
}  // namespace taichi::lang

#else
namespace taichi::lang {
bool Program::vulkan_ray_program_available() const {
  return false;
}
std::uint64_t Program::create_vulkan_ray_program(
    const std::vector<VulkanRayProgramShader> &,
    const std::vector<VulkanRayProgramGroup> &,
    std::uint32_t,
    bool) {
  TI_ERROR("Vulkan ray programs require TI_WITH_VULKAN=ON");
}
void Program::destroy_vulkan_ray_program(std::uint64_t) {
}
void Program::vulkan_clear_ray_programs() {
}
std::shared_ptr<PreparedVulkanRayLaunch> Program::prepare_vulkan_ray_launch(
    std::uint64_t,
    const VulkanRayProgramLaunch &) {
  TI_ERROR("Vulkan ray programs are unavailable");
}
void Program::initialize_vulkan_ray_launch(
    const std::shared_ptr<PreparedVulkanRayLaunch> &) {
  TI_ERROR("Vulkan ray programs are unavailable");
}
void Program::execute_vulkan_ray_launch(
    const std::shared_ptr<PreparedVulkanRayLaunch> &) {
  TI_ERROR("Vulkan ray programs are unavailable");
}
void Program::close_vulkan_ray_launch(
    const std::shared_ptr<PreparedVulkanRayLaunch> &) {
}
std::unordered_map<std::string, std::uint64_t> Program::vulkan_ray_launch_info(
    const std::shared_ptr<PreparedVulkanRayLaunch> &) {
  return {{"closed", 1}};
}
std::shared_ptr<gfx::ExternalGraphCommand>
Program::vulkan_ray_program_graph_command(
    const std::shared_ptr<PreparedVulkanRayLaunch> &) {
  TI_ERROR("Vulkan ray programs are unavailable");
}
}  // namespace taichi::lang
#endif
