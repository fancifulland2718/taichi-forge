#pragma once

#include "taichi/runtime/native_resource.h"
#include "taichi/ir/type.h"

namespace taichi::lang {
class CallableBase;

// One fixed binding owns one result frame. The enclosing executor must retain
// it through completion and serialize reads against subsequent writes. This
// object adds neither a submission nor an implicit per-kernel readback.
class NativeKernelResultStorage final : public NativeResource {
 public:
  NativeKernelResultStorage(Arch arch, Device &device, const CallableBase &kernel);
  Arch arch() const override { return arch_; }
  Device &device() const override { return device_; }
  std::size_t workspace_bytes() const override { return 0; }
  std::size_t device_payload_bytes() const override { return bytes_; }
  DeviceAllocation allocation() const {
    return allocation_ ? *allocation_ : kDeviceNullAllocation;
  }
  std::uint64_t address() const { return address_; }
  bool matches(Arch arch, Device &device, const CallableBase &kernel) const;
  // The caller has waited for the exact producer and owns its submission lock.
  // Copying never changes ownership or releases the reusable frame.
  void read_completed(void *destination, std::size_t bytes) const;

 private:
  Arch arch_;
  Device &device_;
  const StructType *type_;
  std::size_t bytes_;
  std::uint64_t address_{0};
  DeviceAllocationUnique allocation_;
};
}  // namespace taichi::lang
