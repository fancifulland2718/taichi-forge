#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "taichi/runtime/cuda/capture_resources.h"
#include "taichi/runtime/cuda/provider_completion_resource.h"

namespace taichi::lang {
class RuntimeFaultDomain;

// Validated compact f32/i32 CSR in the current CUDA context. The caller owns
// storage and serializes preparation/submission. Live callers supply their
// matrix/pattern/numeric identities; graph owners freeze a private CSR
// snapshot.
struct CudaCudssMatrixView {
  std::size_t rows{0}, columns{0}, nonzeros{0};
  void *row_offsets{nullptr}, *column_indices{nullptr}, *values{nullptr};
  std::uint64_t matrix_id{0}, pattern_version{0}, numeric_version{0};
  std::uint64_t shared_pattern_id{0};
};

// Internal shared solver implementation, independent of Program handles,
// Ndarray, Python or a persistent schema. Calls are cold construction or
// ordinary host submissions; captured graph replay never invokes this API.
class CudaCudssPlan : public CudaProviderCompletionResource {
 public:
  virtual std::size_t rows() const noexcept = 0;
  virtual std::size_t nonzeros() const noexcept = 0;
  virtual void analyze(const CudaCudssMatrixView &matrix) = 0;
  virtual void factorize(const CudaCudssMatrixView &matrix,
                         bool refactorize) = 0;
  virtual void solve(const CudaCudssMatrixView &matrix,
                     void *rhs,
                     void *solution) = 0;
  virtual void reserve_refactor_solve() = 0;
  virtual void cancel_unsubmitted_refactor_solve() noexcept = 0;
  virtual void execute_reserved_refactor_solve(void *values,
                                               void *rhs,
                                               void *solution) = 0;
  virtual void debug_fail_next_refactor_solve() = 0;
  virtual std::unordered_map<std::string, std::uint64_t> statistics() const = 0;
  virtual std::unordered_map<std::string, std::int64_t> configuration()
      const = 0;
  virtual void claim_graph_phase(int phase) = 0;
  // The out-parameter retains snapshots even if a subsequent recording fails.
  // The caller hands it to the graph's finalize/retirement owner after capture.
  virtual void record_capture(
      int phase,
      void *values,
      const std::vector<std::pair<void *, void *>> &rhs_and_outputs,
      void *stream,
      std::shared_ptr<CudaGraphCaptureResources> &resources) = 0;
  virtual void destroy(bool provider_calls_safe) noexcept = 0;
};

std::shared_ptr<CudaCudssPlan> make_cuda_cudss_plan(
    const CudaCudssMatrixView &matrix,
    int matrix_type,
    int matrix_view,
    const std::string &adapter_path,
    const std::string &runtime_library_path,
    std::shared_ptr<RuntimeFaultDomain> fault_domain,
    const std::vector<int> &configuration,
    bool graph_owned);

}  // namespace taichi::lang
