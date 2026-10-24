#pragma once

#include "taichi/program/prepared_resource_lease.h"
#include "taichi/runtime/native_ray_admission.h"
#include "taichi/runtime/native_resource.h"

namespace taichi::lang {

// A Program packet may outlive its device. Its ordinary access permit must be
// invalidated with the existing prepared leases, rather than retain AS owners
// until the last Python packet disappears. Submission keeps a local permit
// through check + append under the runtime's existing admission gate.
class PreparedVulkanRayAccess final : public PreparedResourceLease {
 public:
  explicit PreparedVulkanRayAccess(
      std::shared_ptr<NativeRayAdmission::Permit> permit,
      std::vector<NativeResourceUse> uses)
      : permit_(std::move(permit)), uses_(std::move(uses)) {
  }

  // Cold projection for the enclosing fixed recording's aggregated permit.
  // Keep these owners in the clearable lease, never in the packet facade.
  std::vector<NativeResourceUse> resource_uses() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!permit_)
      throw std::runtime_error("Prepared Vulkan AS access is closed or retired");
    return uses_;
  }

  template <class Submit>
  decltype(auto) submit(Submit &&submit) const {
    std::shared_ptr<NativeRayAdmission::Permit> permit;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      permit = permit_;
    }
    if (!permit)
      throw std::runtime_error("Prepared Vulkan AS access is closed or retired");
    return permit->submit(std::forward<Submit>(submit));
  }

  void clear() noexcept override {
    std::shared_ptr<NativeRayAdmission::Permit> retired;
    std::vector<NativeResourceUse> uses;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      retired.swap(permit_);
      uses.swap(uses_);
    }
    // Releasing the final owner can unregister AS identities. Do that outside
    // the lease mutex, consistently with the existing admission gate order.
  }

 private:
  mutable std::mutex mutex_;
  std::shared_ptr<NativeRayAdmission::Permit> permit_;
  std::vector<NativeResourceUse> uses_;
};

}  // namespace taichi::lang
