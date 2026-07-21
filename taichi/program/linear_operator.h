#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "taichi/ir/type.h"

namespace taichi::lang {

class Ndarray;
class Program;
class SparseMatrix;
class CpuSparseCsrMatrix;
class CpuSparseBsrMatrix;
class CompiledKernelLinearOperator;

enum class OperatorInnerProductKind : std::uint8_t {
  euclidean,
};

enum class OperatorApplyMode : std::uint8_t {
  forward,
  adjoint,
};

struct OperatorSpaceDesc {
  DataType scalar_type{PrimitiveType::unknown};
  std::size_t scalar_extent{0};
  std::vector<int> entry_shape;
  OperatorInnerProductKind inner_product_kind{
      OperatorInnerProductKind::euclidean};

  bool operator==(const OperatorSpaceDesc &other) const;
  bool operator!=(const OperatorSpaceDesc &other) const {
    return !(*this == other);
  }
};

struct OperatorDescriptor {
  OperatorSpaceDesc domain;
  OperatorSpaceDesc range;
};

struct OperatorCapabilities {
  bool forward_apply{true};
  bool adjoint_apply{false};
  bool native_generalized_apply{false};
  bool asynchronous_submit{false};
};

struct OperatorResourceStamp {
  std::uintptr_t program_identity{0};
  std::uint64_t schema_revision{1};
  std::uint64_t topology_revision{1};
  std::uint64_t numeric_revision{1};
  std::uint64_t binding_revision{1};
};

struct OperatorVectorView {
  OperatorSpaceDesc space;
  std::uintptr_t data{0};
  std::uintptr_t allocation_identity{0};
  const Ndarray *ndarray{nullptr};
  Program *program{nullptr};
  bool writable{false};

  static OperatorVectorView from_const_host(const void *data,
                                            const OperatorSpaceDesc &space);
  static OperatorVectorView from_mutable_host(void *data,
                                              const OperatorSpaceDesc &space);
  static OperatorVectorView from_ndarray(Program *program,
                                         const Ndarray &array,
                                         const OperatorSpaceDesc &space,
                                         bool writable);
};

struct OperatorApplyRequest {
  OperatorApplyMode mode{OperatorApplyMode::forward};
  OperatorVectorView input;
  const OperatorVectorView *addend{nullptr};
  OperatorVectorView output;
  double alpha{1.0};
  double beta{0.0};
};

struct OperatorSubmission {
  OperatorResourceStamp resource_stamp;
  bool completed_synchronously{true};
};

// Type-erased ownership for one provider transaction. Bindings use this to
// keep provider-specific locks and snapshots alive without exposing their
// concrete types to plans or solvers.
class OperatorResourceLease {
 public:
  OperatorResourceLease() = default;

  template <typename Lease>
  static OperatorResourceLease hold(Lease lease) {
    return OperatorResourceLease(
        std::make_shared<Lease>(std::move(lease)));
  }

  explicit operator bool() const {
    return state_ != nullptr;
  }

 private:
  explicit OperatorResourceLease(std::shared_ptr<void> state);

  std::shared_ptr<void> state_;
};

class OperatorAction {
 public:
  using ResourceStampFn = std::function<OperatorResourceStamp()>;
  using OverwriteApplyFn = std::function<void(OperatorApplyMode,
                                              const OperatorVectorView &,
                                              const OperatorVectorView &)>;

  OperatorAction(OperatorDescriptor descriptor,
                 OperatorCapabilities capabilities,
                 std::string provider_name,
                 ResourceStampFn resource_stamp,
                 OverwriteApplyFn overwrite_apply);

  const OperatorDescriptor &descriptor() const;
  const OperatorCapabilities &capabilities() const;
  const std::string &provider_name() const;
  OperatorResourceStamp resource_stamp() const;
  void apply_overwrite(OperatorApplyMode mode,
                       const OperatorVectorView &input,
                       const OperatorVectorView &output) const;

 private:
  struct State;
  std::shared_ptr<const State> state_;
};

class OperatorBinding {
 public:
  using AcquireResourceLeaseFn =
      std::function<OperatorResourceLease()>;

  explicit OperatorBinding(
      OperatorAction action,
      AcquireResourceLeaseFn acquire_resource_lease = {});

  const OperatorAction &action() const;
  OperatorResourceLease acquire_resource_lease() const;

 private:
  OperatorAction action_;
  AcquireResourceLeaseFn acquire_resource_lease_;
};

struct OperatorPlanRuntimeStatistics {
  std::uint64_t submissions{0};
  std::uint64_t primitive_apply_calls{0};
  std::uint64_t generalized_lowerings{0};
  std::uint64_t scratch_builds{0};
  std::uint64_t scratch_reuses{0};
  std::uint64_t scratch_reserved_bytes{0};
};

class OperatorPlan {
 public:
  OperatorPlan(Program *program, OperatorAction action);
  OperatorPlan(Program *program, OperatorBinding binding);
  OperatorPlan(const OperatorPlan &) = delete;
  OperatorPlan &operator=(const OperatorPlan &) = delete;
  ~OperatorPlan();

  const OperatorDescriptor &descriptor() const;
  const OperatorCapabilities &capabilities() const;
  const std::string &provider_name() const;
  OperatorResourceStamp resource_stamp() const;
  OperatorResourceLease acquire_resource_lease() const;
  OperatorSubmission submit(const OperatorApplyRequest &request);
  OperatorPlanRuntimeStatistics debug_runtime_statistics() const;

 private:
  struct Scratch;
  OperatorVectorView scratch_for(const OperatorSpaceDesc &space,
                                 OperatorApplyMode mode);
  void release_scratch(Scratch &scratch);

  Program *program_{nullptr};
  OperatorBinding binding_;
  std::unique_ptr<Scratch> forward_scratch_;
  std::unique_ptr<Scratch> adjoint_scratch_;
  OperatorPlanRuntimeStatistics statistics_;
};

OperatorAction make_dense_reference_operator_action(
    OperatorDescriptor descriptor,
    std::vector<double> row_major_values);

OperatorBinding make_cpu_csr_operator_binding(Program *program,
                                              CpuSparseCsrMatrix &matrix);
OperatorBinding make_cpu_bsr_operator_binding(Program *program,
                                              CpuSparseBsrMatrix &matrix);
OperatorBinding make_cpu_program_kernel_operator_binding(
    Program *program,
    CompiledKernelLinearOperator &matrix);

// Internal compatibility adapter used while stored and compiled providers
// still expose their apply primitive through SparseMatrix::nd_spmv().
OperatorAction make_cpu_sparse_matrix_operator_action(Program *program,
                                                      SparseMatrix &matrix);

}  // namespace taichi::lang
