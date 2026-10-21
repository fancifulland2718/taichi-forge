#include "taichi/ir/ir.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/frontend_ir.h"
#include "taichi/program/callable.h"
#include "taichi/system/profiler.h"

#include <algorithm>

namespace taichi::lang {

namespace irpass {

void reverse_segments(IRNode *root) {
  TI_AUTO_PROF;
  auto *block = root->as<Block>();
  const auto *callable = block->parent_callable();
  const size_t prefix = callable ? callable->argument_initialization_size : 0;
  TI_ASSERT(prefix <= block->size());

  // Parameter unpacking creates local copies before the user's body. Keep
  // their initialization in place while reversing the user's independent
  // loops. Only the frontend can mark this boundary; user statements still
  // obey the existing mixed-loop restriction.
  bool has_for = false;
  bool has_non_for = false;
  for (size_t i = prefix; i < block->size(); ++i) {
    if (block->statements[i]->is<FrontendForStmt>()) {
      has_for = true;
    } else {
      has_non_for = true;
    }
  }
  if (has_for && has_non_for) {
    TI_ERROR(
        "Invalid program input for autodiff: "
        "Mixed usage of for-loops and statements without looping. \n"
        "Please split them into two kernels "
        "and check the documentation for more details:\n"
        "https://docs.taichi-lang.org/docs/"
        "differentiable_programming");
  }
  if (has_for) {
    std::reverse(block->statements.begin() + prefix, block->statements.end());
  }
}

}  // namespace irpass

}  // namespace taichi::lang
