#pragma once

#include "taichi_core_impl.h"
#include "taichi/runtime/gfx/runtime.h"
#include "taichi/common/virtual_dir.h"

class GfxRuntime;

class GfxRuntime : public Runtime {
 public:
  GfxRuntime(taichi::Arch arch);
  virtual taichi::lang::gfx::GfxRuntime &get_gfx_runtime() = 0;

  virtual Error create_aot_module(std::shared_ptr<const taichi::io::VirtualDir> dir,
                                  TiAotModule &out) override final;
  virtual void buffer_copy(const taichi::lang::DevicePtr &dst,
                           const taichi::lang::DevicePtr &src,
                           size_t size) override final;
  virtual void copy_image(
      const taichi::lang::DeviceAllocation &dst,
      const taichi::lang::DeviceAllocation &src,
      const taichi::lang::ImageCopyParams &params) override final;
  virtual void track_image(const taichi::lang::DeviceAllocation &image,
                           taichi::lang::ImageLayout layout) override final;
  virtual void untrack_image(
      const taichi::lang::DeviceAllocation &image) override final;
  virtual void transition_image(
      const taichi::lang::DeviceAllocation &image,
      taichi::lang::ImageLayout layout) override final;
  virtual void flush() override final;
  virtual void wait() override final;

 protected:
  std::size_t compile_dense_snode_layout(taichi::lang::SNode &root) override;
  std::pair<taichi::lang::DeviceAllocation, std::shared_ptr<void>>
  allocate_dense_snode_storage(std::size_t size) override;
};
