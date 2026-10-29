#include "taichi_gfx_impl.h"
#include "taichi/runtime/gfx/aot_module_loader_impl.h"
#include "taichi/codegen/spirv/snode_struct_compiler.h"

GfxRuntime::GfxRuntime(taichi::Arch arch) : Runtime(arch) {
}

std::size_t GfxRuntime::compile_dense_snode_layout(taichi::lang::SNode &root) {
  if (arch != taichi::Arch::vulkan)
    throw std::invalid_argument("Dense SNode groups require CPU/CUDA/Vulkan");
  // Use the same layout pass as SNodeTreeManager, including its placement
  // ordering and scalar alignment. No LLVM layout or producer offsets apply.
  return taichi::lang::spirv::compile_snode_structs(root).root_size;
}

std::pair<taichi::lang::DeviceAllocation, std::shared_ptr<void>>
GfxRuntime::allocate_dense_snode_storage(std::size_t size) {
  using namespace taichi::lang;
  auto [buffer, result] = get().allocate_memory_unique(
      {size, false, false, false, AllocUsage::Storage});
  if (result != RhiResult::success)
    throw std::bad_alloc();
  auto *stream = get().get_compute_stream();
  auto [commands, status] = stream->new_command_list_unique();
  if (status != RhiResult::success)
    throw std::runtime_error("Dense SNode group initialization failed");
  commands->buffer_fill(buffer->get_ptr(), kBufferSizeEntireSize, 0);
  stream->submit_synced(commands.get());
  const DeviceAllocation allocation = *buffer;
  return {allocation,
          std::shared_ptr<DeviceAllocationGuard>(std::move(buffer))};
}

Error GfxRuntime::create_aot_module(
    std::shared_ptr<const taichi::io::VirtualDir> dir,
    TiAotModule &out) {
  taichi::lang::gfx::AotModuleParams params{};
  params.source = std::move(dir);
  params.runtime = &get_gfx_runtime();
  std::unique_ptr<taichi::lang::aot::Module> aot_module =
      taichi::lang::aot::Module::load(arch, params);
  if (aot_module->is_corrupted()) {
    return Error(TI_ERROR_CORRUPTED_DATA, "aot_module");
  }

  const taichi::lang::DeviceCapabilityConfig &current_devcaps =
      params.runtime->get_ti_device()->get_caps();
  const taichi::lang::DeviceCapabilityConfig &required_devcaps =
      aot_module->get_required_caps();
  for (const auto &pair : required_devcaps.devcaps) {
    uint32_t current_version = current_devcaps.get(pair.first);
    uint32_t required_version = pair.second;
    if (current_version < required_version) {
      return Error(TI_ERROR_INCOMPATIBLE_MODULE,
                   taichi::lang::to_string(pair.first).c_str());
    }
  }

  taichi::lang::gfx::allocate_aot_root_buffers(aot_module.get());
  out = (TiAotModule) new AotModule(*this, std::move(aot_module));
  return Error();
}
void GfxRuntime::buffer_copy(const taichi::lang::DevicePtr &dst,
                             const taichi::lang::DevicePtr &src,
                             size_t size) {
  get_gfx_runtime().buffer_copy(dst, src, size);
}
void GfxRuntime::copy_image(const taichi::lang::DeviceAllocation &dst,
                            const taichi::lang::DeviceAllocation &src,
                            const taichi::lang::ImageCopyParams &params) {
  get_gfx_runtime().copy_image(dst, src, params);
}
void GfxRuntime::track_image(const taichi::lang::DeviceAllocation &image,
                             taichi::lang::ImageLayout layout) {
  get_gfx_runtime().track_image(image, layout);
}
void GfxRuntime::untrack_image(const taichi::lang::DeviceAllocation &image) {
  get_gfx_runtime().untrack_image(image);
}
void GfxRuntime::transition_image(const taichi::lang::DeviceAllocation &image,
                                  taichi::lang::ImageLayout layout) {
  get_gfx_runtime().transition_image(image, layout);
}
void GfxRuntime::flush() {
  get_gfx_runtime().flush();
}
void GfxRuntime::wait() {
  // (penguinliong) It's currently waiting for the entire runtime to stop.
  // Should be simply waiting for its fence to finish.
  synchronize_execution_domains();
  get_gfx_runtime().synchronize();
}
