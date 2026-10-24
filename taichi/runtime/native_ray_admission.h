#pragma once

#include "taichi/program/runtime_completion.h"
#include "taichi/runtime/shared_resource_registration.h"

namespace taichi::lang {

struct NativeRayAdmissionName {
  static constexpr const char *value = "AS";
};
using NativeRayAdmission =
    SharedResourceAdmission<RuntimeCompletion, NativeRayAdmissionName>;
using NativeRayResourceRegistration =
    SharedResourceRegistration<NativeRayAdmission>;

}  // namespace taichi::lang
