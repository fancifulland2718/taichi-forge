#pragma once

#include "taichi/codegen/spirv/kernel_utils.h"
#include "taichi/rhi/device.h"

namespace taichi::lang::gfx {

// Cold-owned controller for one compiled task and one resolved workspace.
// Payload code is unchanged. The recording slot owns this object until its
// complete command stream retires; there is no allocation or host readback in
// replay. Distinct instances never share the mutable indirect packet.
class RangeDispatch {
 public:
  static constexpr std::size_t kPacketBytes = 3 * sizeof(uint32_t);
  static void validate(const spirv::TaskAttributes &task,
                       std::size_t global_tmp_bytes);
  RangeDispatch(Device &device,
                const spirv::TaskAttributes &task,
                DeviceAllocation global_tmps,
                std::size_t global_tmp_bytes);
  void record_controller(CommandList &commands) const;
  DevicePtr packet() const {
    return packet_->get_ptr();
  }

 private:
  DeviceAllocation global_tmps_;
  std::size_t global_tmp_bytes_;
  std::unique_ptr<DeviceAllocationGuard> packet_;
  std::unique_ptr<Pipeline> pipeline_;
  std::unique_ptr<ShaderResourceSet> resources_;
};

}  // namespace taichi::lang::gfx
