#pragma once

#include "taichi/runtime/shared_resource_admission.h"

namespace taichi::lang {

// Generation-qualified admission metadata, owned by the physical resource.
// Loans and permits retain that owner, never the admission table itself.
template <class Admission>
class SharedResourceRegistration {
 public:
  struct Identity {};
  using Registry = RuntimeResourceRegistry<Identity>;

  static std::shared_ptr<SharedResourceRegistration> create(
      std::shared_ptr<Registry> registry,
      std::shared_ptr<Admission> admission) {
    auto result = std::shared_ptr<SharedResourceRegistration>(
        new SharedResourceRegistration(std::move(registry),
                                       std::move(admission)));
    const auto [status, handle] = result->registry_->emplace(1);
    if (status != Registry::Result::kSuccess)
      throw std::runtime_error(
          "Cannot register shared resource access identity");
    try {
      result->admission_->register_resource(handle);
    } catch (...) {
      result->registry_->retire(handle);
      throw;
    }
    result->handle_ = handle;
    result->registered_ = true;
    return result;
  }

  ~SharedResourceRegistration() {
    retire();
  }

  // The owner serializes explicit retirement with its host API gate. Existing
  // permits keep storage alive, but can no longer submit after this boundary.
  // Active loans reject retirement before either registry is changed.
  void retire() {
    if (registered_) {
      admission_->remove_resource(handle_);
      registry_->retire(handle_);
      registered_ = false;
    }
  }

  RuntimeResourceHandle handle() const {
    return handle_;
  }
  bool belongs_to(const Admission &admission) const {
    return admission_.get() == &admission;
  }
  void require_use(SharedResourceUse use) const {
    admission_->require_use({handle_, use});
  }

 private:
  SharedResourceRegistration(std::shared_ptr<Registry> registry,
                             std::shared_ptr<Admission> admission)
      : registry_(std::move(registry)), admission_(std::move(admission)) {
  }
  std::shared_ptr<Registry> registry_;
  std::shared_ptr<Admission> admission_;
  RuntimeResourceHandle handle_;
  bool registered_{false};
};

}  // namespace taichi::lang
