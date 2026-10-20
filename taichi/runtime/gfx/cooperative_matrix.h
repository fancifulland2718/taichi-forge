#pragma once

#include "taichi/codegen/spirv/kernel_utils.h"

namespace taichi::lang::gfx {

void validate_cooperative_matrix_metadata(
    const spirv::TaskAttributes &task,
    const std::vector<std::uint32_t> &code,
    std::uint32_t metadata_version);

void validate_cooperative_matrix_device(
    Device *device,
    const spirv::TaskAttributes &task,
    const std::vector<std::uint32_t> &code);

}  // namespace taichi::lang::gfx
