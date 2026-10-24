#pragma once

#include "taichi/runtime/shared_resource_admission.h"

namespace taichi::lang::gfx {
// Layout changes are exclusive even when they only read pixels. The GFX
// adapter still publishes/restores shader_read and proves storage aliasing.
using SharedImageUse = SharedResourceUse;
struct SharedImageAdmissionName {
  static constexpr const char *value = "image";
};
template <class Boundary>
using SharedImageAdmission =
    SharedResourceAdmission<Boundary, SharedImageAdmissionName>;
}  // namespace taichi::lang::gfx
