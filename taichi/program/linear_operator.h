#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "taichi/ir/type.h"
#include "taichi/program/runtime_completion.h"
#include "taichi/struct/snode_tree.h"

namespace taichi::lang {

namespace aot {
struct CompiledGraph;
}

class Ndarray;
class Kernel;
class Program;
class SparseMatrix;
class CpuSparseCsrMatrix;
class CpuSparseBsrMatrix;
class CuSparseMatrix;
class CuSparseBsrMatrix;
class VulkanSparseMatrix;
class VulkanSparseBsrMatrix;
class CompiledKernelLinearOperator;
class CompiledGraphLinearOperator;
namespace storage {
class DenseStorageDescriptor;
class RuntimeStorageArgument;
struct ResolvedDenseBinding;
}  // namespace storage

enum class OperatorInnerProductKind : std::uint8_t {
  euclidean,
};

enum class OperatorApplyMode : std::uint8_t {
  forward,
  adjoint,
};

enum class OperatorExecutionKind : std::uint8_t {
  direct,
  explicit_sequence,
  compiled_graph,
  runtime_capture,
};

const char *operator_execution_kind_name(OperatorExecutionKind kind);

enum class OperatorBackendExecutionPath : std::uint8_t {
  unavailable,
  direct,
  explicit_sequence,
  ordinary_graph_fallback,
  cuda_capture,
  cuda_exact_replay,
  cuda_patched_replay,
  vulkan_record,
  vulkan_replay,
};

const char *operator_backend_execution_path_name(
    OperatorBackendExecutionPath path);

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
  bool explicit_sequence{false};
  bool compiled_graph{false};
  bool runtime_capture{false};
  bool binding_rebind{false};
  bool persistent_workspace{false};
  bool dense_storage_operands{false};
  bool dense_storage_affine_operands{false};
};

using OperatorDependencyMask = std::uint32_t;

enum class OperatorTraitProvenance : std::uint8_t {
  unspecified,
  asserted_by_user,
  derived_structurally,
  constructed_by_framework,
  empirically_checked,
};

struct OperatorTraitClaim {
  bool value{false};
  OperatorTraitProvenance provenance{
      OperatorTraitProvenance::unspecified};
  OperatorDependencyMask validity_scope{0};

  bool known() const {
    return provenance != OperatorTraitProvenance::unspecified;
  }
};

struct OperatorMathematicalTraits {
  OperatorTraitClaim self_adjoint;
  OperatorTraitClaim positive_definite;
  OperatorTraitClaim positive_semidefinite;
  OperatorTraitClaim singular;
};

OperatorMathematicalTraits make_spd_operator_traits(
    OperatorTraitProvenance provenance,
    OperatorDependencyMask validity_scope);

enum class OperatorSolverFamily : std::uint8_t {
  cg,
  pcg,
  minres,
  bicgstab,
  gmres,
};

struct OperatorResourceStamp {
  std::uintptr_t program_identity{0};
  std::uint64_t program_generation{0};
  std::uint64_t schema_revision{1};
  std::uint64_t topology_revision{1};
  std::uint64_t numeric_revision{1};
  std::uint64_t binding_revision{1};
};

class LinearOperatorRecordableKernel {
 public:
  using FixedI32Arguments =
      std::unordered_map<std::string, std::int32_t>;
  using FixedNdarrayArguments =
      std::vector<std::pair<std::string, Ndarray *>>;

  LinearOperatorRecordableKernel(
      Program *program,
      Kernel *kernel,
      std::int32_t active_size,
      Ndarray *topology,
      Ndarray *numeric,
      OperatorResourceStamp stamp,
      std::shared_ptr<void> generation_owner);
  LinearOperatorRecordableKernel(
      Program *program,
      const aot::CompiledGraph *graph,
      FixedI32Arguments fixed_i32,
      FixedNdarrayArguments fixed_ndarrays,
      std::vector<SNodeTreeDependency> state_dependencies,
      OperatorResourceStamp stamp,
      std::shared_ptr<void> generation_owner);

  Program *program() const;
  Kernel *kernel() const;
  const aot::CompiledGraph *graph() const;
  std::int32_t active_size() const;
  Ndarray *topology() const;
  Ndarray *numeric() const;
  const FixedI32Arguments &fixed_i32() const;
  const FixedNdarrayArguments &fixed_ndarrays() const;
  const std::vector<SNodeTreeDependency> &state_dependencies() const;
  OperatorResourceStamp resource_stamp() const;

 private:
  Program *program_{nullptr};
  Kernel *kernel_{nullptr};
  const aot::CompiledGraph *graph_{nullptr};
  std::int32_t active_size_{0};
  Ndarray *topology_{nullptr};
  Ndarray *numeric_{nullptr};
  FixedI32Arguments fixed_i32_;
  FixedNdarrayArguments fixed_ndarrays_;
  std::vector<SNodeTreeDependency> state_dependencies_;
  OperatorResourceStamp stamp_;
  std::shared_ptr<void> generation_owner_;
};

enum class OperatorResourceDependency : std::uint32_t {
  program = 1u << 0,
  schema = 1u << 1,
  topology = 1u << 2,
  numeric = 1u << 3,
  binding = 1u << 4,
};

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
  const storage::DenseStorageDescriptor *dense_storage{nullptr};
  const storage::RuntimeStorageArgument *runtime_storage{nullptr};
  const storage::ResolvedDenseBinding *resolved_dense_storage{nullptr};
  const void *allocation_device_identity{nullptr};
  std::uint64_t allocation_id{0};
  std::uint64_t byte_begin{0};
  std::uint64_t byte_end{0};

  static OperatorVectorView from_const_host(const void *data,
                                            const OperatorSpaceDesc &space);
  static OperatorVectorView from_mutable_host(void *data,
                                              const OperatorSpaceDesc &space);
  static OperatorVectorView from_ndarray(Program *program,
                                         const Ndarray &array,
                                         const OperatorSpaceDesc &space,
                                         bool writable);
  static OperatorVectorView from_device_pointer(Program *program,
                                                std::uintptr_t data,
                                                const OperatorSpaceDesc &space,
                                                bool writable);
  static OperatorVectorView from_dense_storage(
      Program *program,
      const storage::RuntimeStorageArgument &argument,
      const storage::ResolvedDenseBinding &binding,
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

// Type-erased ownership for one provider transaction. Bindings use this to
// keep provider-specific locks and snapshots alive without exposing their
// concrete types to plans or solvers.
class OperatorResourceLease {
 public:
  OperatorResourceLease() = default;

  template <typename Lease>
  static OperatorResourceLease hold(Lease lease) {
    return OperatorResourceLease(std::make_shared<Lease>(std::move(lease)));
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
  OperatorAction(OperatorDescriptor descriptor,
                 OperatorMathematicalTraits mathematical_traits,
                 OperatorCapabilities capabilities,
                 std::string provider_name,
                 ResourceStampFn resource_stamp,
                 OverwriteApplyFn overwrite_apply);

  const OperatorDescriptor &descriptor() const;
  const OperatorMathematicalTraits &mathematical_traits() const;
  const OperatorCapabilities &capabilities() const;
  const std::string &provider_name() const;
  OperatorResourceStamp resource_stamp() const;
  void apply_overwrite(OperatorApplyMode mode,
                       const OperatorVectorView &input,
                       const OperatorVectorView &output) const;
  OperatorAction with_mathematical_traits(
      OperatorMathematicalTraits mathematical_traits) const;

 private:
  struct State;
  std::shared_ptr<const State> state_;
};

// One immutable action/resource generation. The action and its retained
// resources always originate from the same atomic publisher snapshot.
class OperatorPinnedAction {
 public:
  OperatorPinnedAction() = default;
  static OperatorPinnedAction from_retained_action(
      OperatorAction action,
      OperatorResourceStamp stamp,
      OperatorResourceLease resource_lease = {});

  explicit operator bool() const;
  const OperatorDescriptor &descriptor() const;
  const OperatorMathematicalTraits &mathematical_traits() const;
  const OperatorCapabilities &capabilities() const;
  const std::string &provider_name() const;
  OperatorResourceStamp resource_stamp() const;
  void apply_overwrite(OperatorApplyMode mode,
                       const OperatorVectorView &input,
                       const OperatorVectorView &output) const;
  OperatorPinnedAction with_mathematical_traits(
      OperatorMathematicalTraits mathematical_traits) const;

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

// Move-only submission ticket. An asynchronous ticket retains the exact
// action/resource generation until its backend completion is observed. If a
// caller discards a pending ticket, destruction waits before releasing the
// generation; internal solve plans instead retain one explicit pin across all
// of their submissions and synchronize at the solve boundary.
class OperatorSubmission {
 public:
  OperatorSubmission() = default;
  OperatorSubmission(const OperatorSubmission &) = delete;
  OperatorSubmission &operator=(const OperatorSubmission &) = delete;
  OperatorSubmission(OperatorSubmission &&other) noexcept;
  OperatorSubmission &operator=(OperatorSubmission &&) = delete;
  ~OperatorSubmission();

  bool done() const;
  void wait() const;

  OperatorResourceStamp resource_stamp;
  bool completed_synchronously{true};

 private:
  friend class OperatorPlan;
  OperatorSubmission(OperatorPinnedAction generation,
                     RuntimeCompletion completion,
                     bool completed_synchronously);

  OperatorPinnedAction generation_;
  RuntimeCompletion completion_;
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

  void publish(OperatorAction action, OperatorResourceLease resources = {});
  OperatorPinnedAction acquire() const;
  void retire_current();
  OperatorResourceGenerationStatistics debug_statistics() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class OperatorBinding {
 public:
  using AcquireResourceLeaseFn = std::function<OperatorResourceLease()>;
  using AcquirePinnedActionFn = std::function<OperatorPinnedAction()>;
  struct ExecutionRuntimeStatistics {
    OperatorBackendExecutionPath last_backend_path{
        OperatorBackendExecutionPath::unavailable};
    std::uint64_t sequence_submissions{0};
    std::uint64_t compiled_graph_submissions{0};
    std::uint64_t runtime_capture_submissions{0};
    std::uint64_t backend_captures{0};
    std::uint64_t backend_replays{0};
    std::uint64_t ordinary_fallbacks{0};
    std::uint64_t cache_invalidations{0};
  };
  using ExecutionRuntimeStatisticsFn =
      std::function<ExecutionRuntimeStatistics()>;

  explicit OperatorBinding(OperatorAction action,
                           AcquireResourceLeaseFn acquire_resource_lease = {});

  static OperatorBinding from_generation_publisher(
      OperatorAction metadata_action,
      AcquirePinnedActionFn acquire_pinned_action);

  const OperatorAction &action() const;
  OperatorBinding with_mathematical_traits(
      OperatorMathematicalTraits mathematical_traits) const;
  OperatorBinding with_execution_lowering(
      OperatorExecutionKind execution_kind,
      ExecutionRuntimeStatisticsFn execution_statistics = {}) const;
  OperatorExecutionKind execution_kind() const;
  ExecutionRuntimeStatistics execution_runtime_statistics() const;
  OperatorResourceLease acquire_resource_lease() const;
  OperatorPinnedAction pin() const;

 private:
  OperatorBinding(OperatorAction metadata_action,
                  AcquirePinnedActionFn acquire_pinned_action,
                  bool generation_bound);

  OperatorAction action_;
  AcquireResourceLeaseFn acquire_resource_lease_;
  AcquirePinnedActionFn acquire_pinned_action_;
  OperatorExecutionKind execution_kind_{OperatorExecutionKind::direct};
  ExecutionRuntimeStatisticsFn execution_statistics_;
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
  std::uint64_t execution_plan_builds{0};
  std::uint64_t execution_plan_reuses{0};
  std::uint64_t binding_rebinds{0};
  OperatorExecutionKind execution_kind{OperatorExecutionKind::direct};
  OperatorBackendExecutionPath last_backend_path{
      OperatorBackendExecutionPath::unavailable};
  std::uint64_t sequence_submissions{0};
  std::uint64_t compiled_graph_submissions{0};
  std::uint64_t runtime_capture_submissions{0};
  std::uint64_t backend_captures{0};
  std::uint64_t backend_replays{0};
  std::uint64_t ordinary_fallbacks{0};
  std::uint64_t cache_invalidations{0};
};

class OperatorPlan {
 public:
  OperatorPlan(Program *program,
               OperatorAction action,
               OperatorDependencyMask dependencies =
                   operator_plan_schema_dependencies());
  OperatorPlan(Program *program,
               OperatorBinding binding,
               OperatorDependencyMask dependencies =
                   operator_plan_schema_dependencies());
  OperatorPlan(const OperatorPlan &) = delete;
  OperatorPlan &operator=(const OperatorPlan &) = delete;
  ~OperatorPlan();

  const OperatorDescriptor &descriptor() const;
  const OperatorMathematicalTraits &mathematical_traits() const;
  const OperatorCapabilities &capabilities() const;
  const std::string &provider_name() const;
  OperatorExecutionKind execution_kind() const;
  OperatorDependencyMask dependencies() const;
  OperatorResourceStamp resource_stamp() const;
  OperatorResourceLease acquire_resource_lease() const;
  OperatorPinnedAction pin();
  // Owns a completion ticket for one standalone submission.
  OperatorSubmission submit(const OperatorApplyRequest &request);
  // Hot-loop form: the caller retains |pinned| and owns the backend
  // synchronization/completion boundary for the whole operation sequence.
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
  bool has_vector_binding_{false};
  std::uintptr_t last_input_binding_{0};
  std::uintptr_t last_output_binding_{0};
  std::unique_ptr<Scratch> forward_scratch_;
  std::unique_ptr<Scratch> adjoint_scratch_;
  OperatorPlanRuntimeStatistics statistics_;
};

// Opaque native state behind the stable Python LinearOperator API.
// It deliberately exposes neither provider resources nor submission tickets:
// the public synchronous boundary owns one reusable OperatorPlan and keeps all
// provider-specific execution/lifetime rules in this layer.
class LinearOperatorHandle {
 public:
  using NumericUpdateArguments =
      std::unordered_map<std::string, const Ndarray *>;
  using NumericUpdateFn = std::function<void(
      Program *,
      const NumericUpdateArguments &,
      std::uint64_t,
      std::uint64_t)>;
  using AffineParameterUpdateFn = std::function<std::uint64_t(
      double, double, std::uint64_t, std::uint64_t)>;
  using RecordableKernelFn = std::function<
      std::shared_ptr<LinearOperatorRecordableKernel>(OperatorApplyMode)>;

  LinearOperatorHandle(Program *program,
                       OperatorBinding binding,
                       std::shared_ptr<void> provider_owner = {},
                       NumericUpdateFn numeric_update = {},
                       RecordableKernelFn recordable_kernel = {},
                       AffineParameterUpdateFn affine_parameter_update = {});
  LinearOperatorHandle(
      const LinearOperatorHandle &) = delete;
  LinearOperatorHandle &operator=(
      const LinearOperatorHandle &) = delete;
  ~LinearOperatorHandle();

  Program *program() const;
  const OperatorDescriptor &descriptor() const;
  const OperatorMathematicalTraits &mathematical_traits() const;
  const OperatorCapabilities &capabilities() const;
  const std::string &provider_name() const;
  OperatorExecutionKind execution_kind() const;
  OperatorResourceStamp resource_stamp() const;
  OperatorPlanRuntimeStatistics debug_runtime_statistics() const;
  OperatorBinding binding() const;

  std::unique_ptr<class LinearOperatorSession> begin_session();

  void apply(Program *program,
             const Ndarray &input,
             const Ndarray &output);
  void apply_generalized(Program *program,
                         const Ndarray &input,
                         const Ndarray *addend,
                         const Ndarray &output,
                         double alpha,
                         double beta);
  void apply_dense_storage(
      Program *program,
      const storage::RuntimeStorageArgument &input,
      const storage::RuntimeStorageArgument &output);
  void update_numeric(Program *program,
                      const NumericUpdateArguments &arguments,
                      std::uint64_t expected_topology_version,
                      std::uint64_t expected_numeric_version);
  bool supports_numeric_update() const;
  std::uint64_t update_affine_parameters(double alpha,
                                         double beta,
                                         std::uint64_t expected_version,
                                         std::uint64_t next_version);
  bool supports_affine_parameter_update() const;
  std::shared_ptr<LinearOperatorRecordableKernel> recordable_kernel(
      OperatorApplyMode mode);
  bool supports_recordable_kernel() const;

 private:
  Program *program_{nullptr};
  std::shared_ptr<void> provider_owner_;
  NumericUpdateFn numeric_update_;
  RecordableKernelFn recordable_kernel_;
  AffineParameterUpdateFn affine_parameter_update_;
  OperatorBinding binding_;
  std::unique_ptr<OperatorPlan> plan_;
};

// A solve-scoped pinned generation used by iterative plans that interleave
// provider submissions with backend-native recurrence kernels. It is private
// to the Python experimental API: ordinary LinearOperator.apply() remains a
// synchronous boundary.
class LinearOperatorSession {
 public:
  LinearOperatorSession(Program *program,
                                    OperatorPlan *plan,
                                    OperatorPinnedAction generation);
  LinearOperatorSession(
      const LinearOperatorSession &) = delete;
  LinearOperatorSession &operator=(
      const LinearOperatorSession &) = delete;
  ~LinearOperatorSession();

  void submit(Program *program,
              const Ndarray &input,
              const Ndarray &output);
  void wait();
  void mark_synchronized();

 private:
  Program *program_{nullptr};
  OperatorPlan *plan_{nullptr};
  OperatorPinnedAction generation_;
  bool submitted_{false};
};

struct ExperimentalPreconditionerPlanRuntimeStatistics {
  std::uint64_t setup_calls{0};
  std::uint64_t update_calls{0};
  std::uint64_t update_successes{0};
  std::uint64_t update_noops{0};
  std::uint64_t update_failures{0};
  std::uint64_t target_generation_changes{0};
  std::uint64_t action_generation_changes{0};
  std::uint64_t rebuild_attestations{0};
  std::uint64_t reuse_attestations{0};
  std::uint64_t pins{0};
  std::uint64_t apply_calls{0};
  std::uint64_t stale_rejections{0};
  std::uint64_t approved_generations_published{0};
  std::uint64_t approved_generations_retired{0};
  std::uint64_t approved_generations_released{0};
  std::uint64_t approved_generation_active_leases{0};
  bool has_current_approved_generation{false};
};

// One immutable, explicitly approved target/action generation pair. The
// target pin is retained even though only the approximate-inverse action is
// submitted, preventing either side of the consumer scope from being retired
// independently.
class ExperimentalPreconditionerSession {
 public:
  ExperimentalPreconditionerSession(
      Program *program,
      OperatorPlan *action_plan,
      OperatorPinnedAction target_generation,
      OperatorPinnedAction action_generation,
      std::shared_ptr<std::atomic<std::uint64_t>> apply_counter);
  ExperimentalPreconditionerSession(
      const ExperimentalPreconditionerSession &) = delete;
  ExperimentalPreconditionerSession &operator=(
      const ExperimentalPreconditionerSession &) = delete;
  ~ExperimentalPreconditionerSession();

  void apply(Program *program,
             const Ndarray &input,
             const Ndarray &output);
  OperatorResourceStamp target_stamp() const;
  OperatorResourceStamp action_stamp() const;

 private:
  Program *program_{nullptr};
  OperatorPlan *action_plan_{nullptr};
  OperatorPinnedAction target_generation_;
  OperatorPinnedAction action_generation_;
  std::shared_ptr<std::atomic<std::uint64_t>> apply_counter_;
};

// Public fixed-linear lifecycle state. External code updates the target and
// action providers at a host boundary, then explicitly attests either a
// rebuild or lagged reuse. apply()/pin() never invoke a Python callback.
class ExperimentalPreconditionerPlanHandle {
 public:
  ExperimentalPreconditionerPlanHandle(
      Program *program,
      LinearOperatorHandle &target,
      LinearOperatorHandle &action,
      std::string method);
  ExperimentalPreconditionerPlanHandle(
      const ExperimentalPreconditionerPlanHandle &) = delete;
  ExperimentalPreconditionerPlanHandle &operator=(
      const ExperimentalPreconditionerPlanHandle &) = delete;
  ~ExperimentalPreconditionerPlanHandle();

  void setup(Program *program);
  void validate_update(Program *program, bool accept_reuse);
  void update(Program *program, bool accept_reuse);
  std::unique_ptr<ExperimentalPreconditionerSession> pin(Program *program);
  bool is_setup() const;
  const std::string &method() const;
  OperatorResourceStamp built_from_operator_stamp() const;
  OperatorResourceStamp accepted_target_stamp() const;
  OperatorResourceStamp accepted_action_stamp() const;
  OperatorBinding consumer_binding();
  bool supports_recordable_action() const;
  std::shared_ptr<LinearOperatorRecordableKernel> recordable_kernel(
      OperatorApplyMode mode);
  ExperimentalPreconditionerPlanRuntimeStatistics
  debug_runtime_statistics() const;

 private:
  std::unique_ptr<ExperimentalPreconditionerSession> pin_locked();
  void publish_approved_generation(
      const OperatorPinnedAction &target_generation,
      const OperatorPinnedAction &action_generation);
  void validate_program(Program *program) const;

  Program *program_{nullptr};
  OperatorDescriptor target_descriptor_;
  OperatorBinding target_binding_;
  LinearOperatorHandle *action_handle_{nullptr};
  std::unique_ptr<OperatorPlan> action_plan_;
  std::unique_ptr<OperatorResourceGenerationPublisher>
      approved_generations_;
  std::string method_;
  bool is_setup_{false};
  OperatorResourceStamp built_from_operator_stamp_;
  OperatorResourceStamp accepted_target_stamp_;
  OperatorResourceStamp accepted_action_stamp_;
  ExperimentalPreconditionerPlanRuntimeStatistics statistics_;
  std::shared_ptr<std::atomic<std::uint64_t>> apply_counter_;
  mutable std::mutex mutex_;
};

enum class PreconditionerBehavior : std::uint8_t {
  fixed_linear,
  variable_linear,
  nonlinear,
};

void validate_operator_solver_compatibility(
    const OperatorDescriptor &descriptor,
    const OperatorMathematicalTraits &traits,
    OperatorSolverFamily family,
    PreconditionerBehavior preconditioner_behavior =
        PreconditionerBehavior::fixed_linear);

struct PreconditionerPlanRuntimeStatistics {
  std::uint64_t setup_calls{0};
  std::uint64_t update_calls{0};
  std::uint64_t update_successes{0};
  std::uint64_t update_noops{0};
  std::uint64_t update_failures{0};
  std::uint64_t target_generation_changes{0};
};

// Internal lifecycle wrapper for an approximate-inverse OperatorAction. The
// target generation remains pinned while the provider updates and pins its
// corresponding action generation, so a solver can consume one immutable
// target/preconditioner pair for the whole solve.
class PreconditionerPlan {
 public:
  using UpdateFn =
      std::function<void(const OperatorResourceStamp &, bool target_changed)>;

  PreconditionerPlan(Program *program,
                     OperatorDescriptor target_descriptor,
                     OperatorBinding action_binding,
                     PreconditionerBehavior behavior,
                     std::string method,
                     UpdateFn update);
  PreconditionerPlan(const PreconditionerPlan &) = delete;
  PreconditionerPlan &operator=(const PreconditionerPlan &) = delete;
  ~PreconditionerPlan();

  void setup(const OperatorPinnedAction &target_generation);
  OperatorPinnedAction update_and_pin(
      const OperatorPinnedAction &target_generation);
  const OperatorPlan &action() const;
  OperatorPlan &action();
  PreconditionerBehavior behavior() const;
  const std::string &method() const;
  PreconditionerPlanRuntimeStatistics debug_runtime_statistics() const;

 private:
  OperatorPinnedAction update_and_pin_impl(
      const OperatorPinnedAction &target_generation,
      bool setup);

  Program *program_{nullptr};
  OperatorDescriptor target_descriptor_;
  std::unique_ptr<OperatorPlan> action_plan_;
  PreconditionerBehavior behavior_{PreconditionerBehavior::fixed_linear};
  std::string method_;
  UpdateFn update_;
  bool is_setup_{false};
  OperatorResourceStamp target_stamp_;
  PreconditionerPlanRuntimeStatistics statistics_;
};

OperatorAction make_dense_reference_operator_action(
    OperatorDescriptor descriptor,
    std::vector<double> row_major_values);

OperatorBinding make_adjoint_operator_binding(OperatorBinding operand);
OperatorBinding make_identity_operator_binding(OperatorSpaceDesc space,
                                                Program *program = nullptr);
OperatorBinding make_scaled_operator_binding(double scale,
                                              OperatorBinding operand,
                                              Program *program = nullptr);
OperatorBinding make_sum_operator_binding(OperatorBinding left,
                                           OperatorBinding right,
                                           Program *program = nullptr);
OperatorBinding make_composed_operator_binding(OperatorBinding outer,
                                                OperatorBinding inner,
                                                Program *program = nullptr);
OperatorBinding make_block_diagonal_operator_binding(
    std::vector<OperatorBinding> blocks,
    Program *program = nullptr);

OperatorBinding make_cpu_csr_operator_binding(Program *program,
                                              CpuSparseCsrMatrix &matrix);
OperatorBinding make_cpu_bsr_operator_binding(Program *program,
                                              CpuSparseBsrMatrix &matrix);
OperatorBinding make_cpu_fixed_sparse_operator_binding(
    Program *program,
    SparseMatrix &matrix);
OperatorBinding make_cpu_program_kernel_operator_binding(
    Program *program,
    CompiledKernelLinearOperator &matrix);
OperatorBinding make_cpu_program_graph_operator_binding(
    Program *program,
    CompiledGraphLinearOperator &matrix);
OperatorBinding make_cuda_csr_operator_binding(Program *program,
                                               CuSparseMatrix &matrix);
OperatorBinding make_cuda_bsr_operator_binding(Program *program,
                                               CuSparseBsrMatrix &matrix);
OperatorBinding make_cuda_program_kernel_operator_binding(
    Program *program,
    CompiledKernelLinearOperator &matrix);
OperatorBinding make_cuda_program_graph_operator_binding(
    Program *program,
    CompiledGraphLinearOperator &matrix,
    OperatorExecutionKind execution_kind =
        OperatorExecutionKind::compiled_graph);
OperatorBinding make_vulkan_csr_operator_binding(Program *program,
                                                 VulkanSparseMatrix &matrix);
OperatorBinding make_vulkan_bsr_operator_binding(Program *program,
                                                 VulkanSparseBsrMatrix &matrix);
OperatorBinding make_vulkan_program_kernel_operator_binding(
    Program *program,
    CompiledKernelLinearOperator &matrix);
OperatorBinding make_vulkan_program_graph_operator_binding(
    Program *program,
    CompiledGraphLinearOperator &matrix);

// Compatibility selection for the public experimental provider factory. The
// concrete provider remains fixed: unsupported backend/storage combinations
// fail instead of materializing or copying through a different provider.
OperatorBinding make_program_sparse_operator_binding(Program *program,
                                                      SparseMatrix &matrix);

OperatorMathematicalTraits make_asserted_operator_traits(
    int self_adjoint,
    int positive_definite,
    int positive_semidefinite,
    int singular);

std::unique_ptr<LinearOperatorHandle>
make_linear_operator_handle(
    Program *program,
    SparseMatrix &matrix,
    OperatorMathematicalTraits mathematical_traits);
std::unique_ptr<LinearOperatorHandle>
make_compiled_kernel_operator_handle(
    Program *program,
    Kernel &forward_kernel,
    Kernel *adjoint_kernel,
    std::size_t range_extent,
    std::size_t domain_extent,
    std::uint64_t topology_version,
    std::uint64_t numeric_version,
    const Ndarray &topology_data,
    const Ndarray *numeric_data,
    OperatorMathematicalTraits mathematical_traits);
std::unique_ptr<LinearOperatorHandle>
make_compiled_graph_operator_handle(
    Program *program,
    const aot::CompiledGraph &forward_graph,
    const aot::CompiledGraph *adjoint_graph,
    std::size_t range_extent,
    std::size_t domain_extent,
    std::uint64_t topology_version,
    std::uint64_t numeric_version,
    std::unordered_map<std::string, std::int32_t> fixed_i32_arguments,
    std::unordered_map<std::string, const Ndarray *> topology_arguments,
    std::unordered_map<std::string, const Ndarray *> numeric_arguments,
    std::unordered_map<std::string, const Ndarray *> workspace_arguments,
    std::vector<SNodeTreeDependency> state_dependencies,
    OperatorMathematicalTraits mathematical_traits);
std::unique_ptr<LinearOperatorHandle>
make_identity_operator_handle(Program *program,
                                           OperatorSpaceDesc space);
std::unique_ptr<LinearOperatorHandle>
make_adjoint_operator_handle(
    LinearOperatorHandle &operand);
std::unique_ptr<LinearOperatorHandle>
make_scaled_operator_handle(
    double scale,
    LinearOperatorHandle &operand);
std::unique_ptr<LinearOperatorHandle>
make_sum_operator_handle(
    LinearOperatorHandle &left,
    LinearOperatorHandle &right);
std::unique_ptr<LinearOperatorHandle>
make_composed_operator_handle(
    LinearOperatorHandle &outer,
    LinearOperatorHandle &inner);
std::unique_ptr<LinearOperatorHandle>
make_parameterized_affine_operator_handle(
    LinearOperatorHandle &left,
    LinearOperatorHandle &right,
    double alpha,
    double beta,
    double alpha_min,
    double alpha_max,
    double beta_min,
    double beta_max);
std::unique_ptr<LinearOperatorHandle>
make_block_diagonal_operator_handle(
    const std::vector<LinearOperatorHandle *> &blocks);
std::unique_ptr<ExperimentalPreconditionerPlanHandle>
make_experimental_preconditioner_plan_handle(
    Program *program,
    LinearOperatorHandle &target,
    LinearOperatorHandle &action,
    std::string method);
std::unique_ptr<LinearOperatorHandle>
make_experimental_preconditioner_action_handle(
    Program *program,
    ExperimentalPreconditionerPlanHandle &plan);

}  // namespace taichi::lang
