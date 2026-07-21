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
  std::uint64_t program_generation{0};
  std::uint64_t schema_revision{1};
  std::uint64_t topology_revision{1};
  std::uint64_t numeric_revision{1};
  std::uint64_t binding_revision{1};
};

enum class OperatorResourceDependency : std::uint32_t {
  program = 1u << 0,
  schema = 1u << 1,
  topology = 1u << 2,
  numeric = 1u << 3,
  binding = 1u << 4,
};

using OperatorDependencyMask = std::uint32_t;

constexpr OperatorDependencyMask operator_dependency(
    OperatorResourceDependency dependency) {
  return static_cast<OperatorDependencyMask>(dependency);
}

constexpr OperatorDependencyMask operator_plan_schema_dependencies() {
  return operator_dependency(OperatorResourceDependency::program) |
         operator_dependency(OperatorResourceDependency::schema);
}

enum class OperatorPlanInvalidationKind : std::uint8_t {
  current,
  refresh_binding,
  rebuild,
  program_invalid,
};

struct OperatorPlanInvalidation {
  OperatorDependencyMask changes{0};
  OperatorDependencyMask relevant_changes{0};
  OperatorPlanInvalidationKind kind{OperatorPlanInvalidationKind::current};
};

OperatorDependencyMask operator_resource_changes(
    const OperatorResourceStamp &planned,
    const OperatorResourceStamp &current);
OperatorPlanInvalidation evaluate_operator_plan_invalidation(
    const OperatorResourceStamp &planned,
    const OperatorResourceStamp &current,
    OperatorDependencyMask dependencies);

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

// One immutable action/resource generation. The action and its retained
// resources always originate from the same atomic publisher snapshot.
class OperatorPinnedAction {
 public:
  OperatorPinnedAction() = default;

  explicit operator bool() const;
  const OperatorDescriptor &descriptor() const;
  const OperatorCapabilities &capabilities() const;
  const std::string &provider_name() const;
  OperatorResourceStamp resource_stamp() const;
  void apply_overwrite(OperatorApplyMode mode,
                       const OperatorVectorView &input,
                       const OperatorVectorView &output) const;

 private:
  friend class OperatorBinding;
  friend class OperatorResourceGenerationPublisher;

  OperatorPinnedAction(OperatorAction action,
                       OperatorResourceStamp stamp,
                       OperatorResourceLease resource_lease);

  std::shared_ptr<OperatorAction> action_;
  OperatorResourceStamp stamp_;
  OperatorResourceLease resource_lease_;
};

struct OperatorResourceGenerationStatistics {
  std::uint64_t published{0};
  std::uint64_t retired{0};
  std::uint64_t released{0};
  std::uint64_t active_leases{0};
  bool has_current{false};
};

// Linearizable publish/acquire/retire state machine for immutable operator
// generations. Retiring a generation rejects new acquisition while existing
// pins remain usable until their last lease is released.
class OperatorResourceGenerationPublisher {
 public:
  OperatorResourceGenerationPublisher();
  OperatorResourceGenerationPublisher(
      const OperatorResourceGenerationPublisher &) = delete;
  OperatorResourceGenerationPublisher &operator=(
      const OperatorResourceGenerationPublisher &) = delete;
  ~OperatorResourceGenerationPublisher();

  void publish(OperatorAction action,
               OperatorResourceLease resources = {});
  OperatorPinnedAction acquire() const;
  void retire_current();
  OperatorResourceGenerationStatistics debug_statistics() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class OperatorBinding {
 public:
  using AcquireResourceLeaseFn =
      std::function<OperatorResourceLease()>;
  using AcquirePinnedActionFn = std::function<OperatorPinnedAction()>;

  explicit OperatorBinding(
      OperatorAction action,
      AcquireResourceLeaseFn acquire_resource_lease = {});

  static OperatorBinding from_generation_publisher(
      OperatorAction metadata_action,
      AcquirePinnedActionFn acquire_pinned_action);

  const OperatorAction &action() const;
  OperatorResourceLease acquire_resource_lease() const;
  OperatorPinnedAction pin() const;

 private:
  OperatorBinding(OperatorAction metadata_action,
                  AcquirePinnedActionFn acquire_pinned_action,
                  bool generation_bound);

  OperatorAction action_;
  AcquireResourceLeaseFn acquire_resource_lease_;
  AcquirePinnedActionFn acquire_pinned_action_;
};

struct OperatorPlanRuntimeStatistics {
  std::uint64_t submissions{0};
  std::uint64_t primitive_apply_calls{0};
  std::uint64_t generalized_lowerings{0};
  std::uint64_t scratch_builds{0};
  std::uint64_t scratch_reuses{0};
  std::uint64_t scratch_reserved_bytes{0};
  std::uint64_t generation_pins{0};
  std::uint64_t generation_changes{0};
  std::uint64_t numeric_generation_changes{0};
  std::uint64_t binding_generation_changes{0};
  std::uint64_t invalidations{0};
};

class OperatorPlan {
 public:
  OperatorPlan(
      Program *program,
      OperatorAction action,
      OperatorDependencyMask dependencies =
          operator_plan_schema_dependencies());
  OperatorPlan(
      Program *program,
      OperatorBinding binding,
      OperatorDependencyMask dependencies =
          operator_plan_schema_dependencies());
  OperatorPlan(const OperatorPlan &) = delete;
  OperatorPlan &operator=(const OperatorPlan &) = delete;
  ~OperatorPlan();

  const OperatorDescriptor &descriptor() const;
  const OperatorCapabilities &capabilities() const;
  const std::string &provider_name() const;
  OperatorDependencyMask dependencies() const;
  OperatorResourceStamp resource_stamp() const;
  OperatorResourceLease acquire_resource_lease() const;
  OperatorPinnedAction pin();
  OperatorSubmission submit(const OperatorApplyRequest &request);
  OperatorSubmission submit(const OperatorPinnedAction &pinned,
                            const OperatorApplyRequest &request);
  OperatorPlanRuntimeStatistics debug_runtime_statistics() const;

 private:
  struct Scratch;
  OperatorVectorView scratch_for(const OperatorSpaceDesc &space,
                                 OperatorApplyMode mode);
  void release_scratch(Scratch &scratch);

  Program *program_{nullptr};
  OperatorBinding binding_;
  OperatorDependencyMask dependencies_{operator_plan_schema_dependencies()};
  OperatorResourceStamp planned_stamp_;
  OperatorResourceStamp last_pinned_stamp_;
  bool has_pinned_generation_{false};
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

}  // namespace taichi::lang
