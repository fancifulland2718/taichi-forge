#pragma once

#include "taichi/program/callable.h"

namespace taichi::lang::aot {

// Current JIT Graph bindings publish primal storage only. AOT's explicit
// gradient binding extension is separate: never capture an adjoint with a null
// gradient pointer while admitting a different JIT execution strategy.
inline void validate_jit_graph_gradients(const CallableBase &callable) {
  for (const auto &entry : callable.nested_parameters) {
    TI_ERROR_IF(entry.second.needs_grad,
                "JIT Graph ndarray gradients require explicit AOT C API "
                "bindings; use direct kernel calls for Python autodiff");
  }
}

}  // namespace taichi::lang::aot
