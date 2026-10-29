#pragma once

#include <memory>
#include <vector>
#include "taichi/runtime/llvm/sparse_runtime_view.h"

namespace taichi::lang {
class LlvmRuntimeExecutor;
struct LLVMRuntime;
namespace cpu {

// Immutable per-preparation directory. CPU calls complete synchronously under
// the existing launcher/submission locks; owners retire calls before this view.
class SparseRuntimeView {
 public:
  SparseRuntimeView(LlvmRuntimeExecutor &executor,
                    const std::vector<LlvmSparseRuntimeViewBinding> &bindings);
  ~SparseRuntimeView();
  LLVMRuntime *bind(LLVMRuntime *source, const std::vector<int> &trees) const;
  void check_error() const;

 private:
  struct State;
  std::unique_ptr<State> state_;
};
}  // namespace cpu
}  // namespace taichi::lang
