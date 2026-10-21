#pragma once

#include "taichi/runtime/native_resource.h"
#include "taichi/ir/type.h"
#include <unordered_map>
#include <vector>

namespace taichi::lang {
class CallableBase;

// One fixed binding owns one result frame. The enclosing executor must retain
// it through completion and serialize reads against subsequent writes. This
// object adds neither a submission nor an implicit per-kernel readback.
class NativeKernelResultStorage final : public NativeResource {
 public:
  enum class Use { host_result, discarded };
  NativeKernelResultStorage(Arch arch, Device &device, const CallableBase &kernel);
  NativeKernelResultStorage(Arch arch, Device &device, const CallableBase &kernel,
                            Use use);
  Arch arch() const override { return arch_; }
  Device &device() const override { return device_; }
  std::size_t workspace_bytes() const override { return 0; }
  std::size_t device_payload_bytes() const override { return bytes_; }
  DeviceAllocation allocation() const {
    return allocation_ ? *allocation_ : kDeviceNullAllocation;
  }
  std::uint64_t address() const { return address_; }
  bool observable() const { return use_ == Use::host_result; }
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
  Use use_;
};

// Cold scope for a proven sequential region. Concurrent branches must use
// distinct pools; returned owners outlive the pool through their recordings.
class NativeKernelResultPool {
 public:
  NativeKernelResultPool(Arch arch, Device &device) : arch_(arch), device_(device) {}
  std::shared_ptr<NativeKernelResultStorage> acquire(const CallableBase &kernel);
  const std::vector<std::shared_ptr<NativeKernelResultStorage>> &resources() const {
    return resources_;
  }

 private:
  Arch arch_;
  Device &device_;
  std::unordered_map<const StructType *,
                     std::unordered_map<std::size_t,
                                        std::shared_ptr<NativeKernelResultStorage>>> frames_;
  std::vector<std::shared_ptr<NativeKernelResultStorage>> resources_;
};
}  // namespace taichi::lang
