#pragma once

#include <memory>
#include <vector>

#include "taichi/program/runtime_fault.h"
#include "taichi/program/runtime_completion.h"
#include "taichi/runtime/runtime_client.h"
#include "taichi/runtime/llvm/sparse_runtime_view.h"

namespace taichi::lang {
struct LLVMRuntime;
class LlvmRuntimeExecutor;
namespace cuda {

// Cold owner for one generation-qualified sparse closure. The enclosing
// prepared instance retains the executor/code owners and retires recordings
// before this object. It must prove the last replay complete before retire().
// Creation and device initialization are separate so that a domain can own
// the failure transaction before any manager is rebound.
class SparseRuntimeView final : public RuntimeClient {
 public:
  SparseRuntimeView();
  ~SparseRuntimeView() override;
  void initialize(LlvmRuntimeExecutor &executor,
                  const std::vector<LlvmSparseRuntimeViewTree> &trees,
                  std::shared_ptr<RuntimeFaultDomain> owner);
  void retire() override;
  void retire_after(const RuntimeCompletion &prefix) override;

  // Cold packet binding verifies the source and complete selected closure.
  // It never changes the source LaunchContextBuilder or registered code.
  LLVMRuntime *bind(LLVMRuntime *source, const std::vector<int> &trees) const;
  void *temporaries() const;
  std::size_t bytes() const;

  class DiagnosticSnapshot : public RuntimeCompletionObserver {
   public:
    ~DiagnosticSnapshot();
    // Only after the exact completion covering enqueue_diagnostics().
    void observe() const override;

   private:
    friend class SparseRuntimeView;
    struct State;
    explicit DiagnosticSnapshot(std::unique_ptr<State> state);
    std::unique_ptr<State> state_;
  };
  std::shared_ptr<DiagnosticSnapshot> prepare_diagnostic_snapshot();
  // Enqueue after execution on its selected stream. Writes mapped host memory
  // and resets this view's error slot without a shared-copy-engine dependency;
  // observing an old ticket never reads live runtime state.
  void enqueue_diagnostics(const DiagnosticSnapshot &snapshot, void *stream);

 private:
  struct State;
  std::unique_ptr<State> state_;
};

}  // namespace cuda
}  // namespace taichi::lang
