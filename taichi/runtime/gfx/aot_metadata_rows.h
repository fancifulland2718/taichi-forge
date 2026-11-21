#pragma once

#include <ostream>

#include "taichi/runtime/gfx/aot_utils.h"

namespace taichi::lang::gfx {

// This envelope preserves the source metadata version so existing shader,
// layout and binding validation still uses the producer's actual contracts.
constexpr int kRowStorageVersion = 25;
void write_kernel_row_metadata(std::ostream &out, TaichiAotData &&artifact);
void read_kernel_metadata(const char *begin,
                          const char *end,
                          TaichiAotData &out,
                          bool allow_row_storage);

}  // namespace taichi::lang::gfx
