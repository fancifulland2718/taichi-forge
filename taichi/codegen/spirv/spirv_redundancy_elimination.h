#pragma once

#include "spirv-tools/optimizer.hpp"

namespace taichi::lang::spirv {

// Total redundancy elimination with storage proportional to the active
// dominator path, rather than a copy of all visible values for every block.
spvtools::Optimizer::PassToken create_scoped_redundancy_elimination_pass();

}  // namespace taichi::lang::spirv
