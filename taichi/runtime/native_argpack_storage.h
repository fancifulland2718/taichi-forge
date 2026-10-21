#pragma once

#include <map>
#include <memory>
#include "taichi/aot/argpack_layout.h"
#include "taichi/runtime/native_resource.h"

namespace taichi::lang {

// Immutable, owner-local storage. Construction completes uploads before
// publishing; replay only consumes allocations/addresses. Enclosing bindings
// must retain this resource and its device through their completion fence.
// Lifted arrays/images keep their existing binding/retirement contracts.
class NativeArgPackStorage final : public NativeResource {
 public:
  using Values = std::map<std::vector<int>, std::vector<std::uint8_t>>;
  NativeArgPackStorage(Arch arch,
                      Device &device,
                      std::shared_ptr<const aot::ArgPackLayout> layout,
                      const Values &values);
  Arch arch() const override {
    return arch_;
  }
  Device &device() const override {
    return device_;
  }
  std::size_t workspace_bytes() const override {
    return 0;
  }
  std::size_t device_payload_bytes() const override {
    return bytes_;
  }
  const aot::ArgPackLayout &layout() const {
    return *layout_;
  }
  DeviceAllocation allocation(std::size_t index) const {
    return *allocations_.at(index);
  }
  std::uint64_t address(std::size_t index) const {
    return addresses_.at(index);
  }

 private:
  Arch arch_;
  Device &device_;
  const std::shared_ptr<const aot::ArgPackLayout> layout_;
  std::vector<DeviceAllocationUnique> allocations_;
  std::vector<std::uint64_t> addresses_;
  std::size_t bytes_{0};
};

}  // namespace taichi::lang
