#pragma once

#include "taichi/program/runtime_completion.h"
#include "taichi/runtime/shared_resource_registration.h"

namespace taichi::lang {

struct SparseTreeAdmissionName {
  static constexpr const char *value = "SNode tree";
};
using SparseTreeAdmission =
    SharedResourceAdmission<RuntimeCompletion, SparseTreeAdmissionName>;
using SparseTreeRegistration = SharedResourceRegistration<SparseTreeAdmission>;

}  // namespace taichi::lang
