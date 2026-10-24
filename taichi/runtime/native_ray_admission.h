#pragma once

#include "taichi/program/runtime_completion.h"
#include "taichi/runtime/shared_resource_admission.h"

namespace taichi::lang {

struct NativeRayAdmissionName {
  static constexpr const char *value = "AS";
};
using NativeRayAdmission =
    SharedResourceAdmission<RuntimeCompletion, NativeRayAdmissionName>;

// Only metadata is registered here. The original AS owner retains this token;
// imports share it, and the admission table never owns the AS or its device.
// The existing registry supplies reusable slots and generation-qualified IDs.
class NativeRayResourceRegistration {
 public:
  struct Identity {};
  using Registry = RuntimeResourceRegistry<Identity>;

  static std::shared_ptr<NativeRayResourceRegistration> create(
      std::shared_ptr<Registry> registry,
      std::shared_ptr<NativeRayAdmission> admission) {
    auto result = std::shared_ptr<NativeRayResourceRegistration>(
        new NativeRayResourceRegistration(std::move(registry),
                                           std::move(admission)));
    const auto [status, handle] = result->registry_->emplace(1);
    if (status != Registry::Result::kSuccess)
      throw std::runtime_error("Cannot register AS access identity");
    try {
      result->admission_->register_resource(handle);
    } catch (...) {
      result->registry_->retire(handle);
      throw;
    }
    result->handle_ = handle;
    return result;
  }
  ~NativeRayResourceRegistration() {
    if (handle_) {
      // Active loans/permits retain the AS owner, so this is necessarily the
      // last physical use. Removal remains legal after the runtime closes.
      admission_->remove_resource(handle_);
      registry_->retire(handle_);
    }
  }
  RuntimeResourceHandle handle() const { return handle_; }
  bool belongs_to(const NativeRayAdmission &admission) const {
    return admission_.get() == &admission;
  }
  void require_use(SharedResourceUse use) const {
    admission_->require_use({handle_, use});
  }

 private:
  NativeRayResourceRegistration(std::shared_ptr<Registry> registry,
                                std::shared_ptr<NativeRayAdmission> admission)
      : registry_(std::move(registry)), admission_(std::move(admission)) {}
  std::shared_ptr<Registry> registry_;
  std::shared_ptr<NativeRayAdmission> admission_;
  RuntimeResourceHandle handle_;
};

}  // namespace taichi::lang
