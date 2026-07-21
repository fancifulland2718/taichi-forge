#include "taichi/program/linear_operator.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <limits>
#include <mutex>
#include <utility>

#include "taichi/common/core.h"
#include "taichi/program/ndarray.h"
#include "taichi/program/program.h"
#include "taichi/program/runtime_resource_registry.h"
#include "taichi/program/sparse_matrix.h"

namespace taichi::lang {
namespace {

std::atomic<std::uint64_t> next_operator_generation_domain{1};

std::uint64_t allocate_operator_generation_domain() {
  const auto domain =
      next_operator_generation_domain.fetch_add(1, std::memory_order_relaxed);
  TI_ASSERT(domain != 0 &&
            domain != (std::numeric_limits<std::uint64_t>::max)());
  return domain;
}

bool supported_scalar_type(DataType type) {
  return type == PrimitiveType::f32 || type == PrimitiveType::f64;
}

void validate_space(const OperatorSpaceDesc &space, const char *role) {
  TI_ERROR_IF(!supported_scalar_type(space.scalar_type) ||
                  space.scalar_extent == 0 || !space.entry_shape.empty(),
              "M1 operator {} space must be a non-empty scalar f32/f64 "
              "space.",
              role);
}

std::size_t space_bytes(const OperatorSpaceDesc &space) {
  return space.scalar_extent * data_type_size(space.scalar_type);
}

const OperatorSpaceDesc &input_space(const OperatorDescriptor &descriptor,
                                     OperatorApplyMode mode) {
  return mode == OperatorApplyMode::forward ? descriptor.domain
                                            : descriptor.range;
}

const OperatorSpaceDesc &output_space(const OperatorDescriptor &descriptor,
                                      OperatorApplyMode mode) {
  return mode == OperatorApplyMode::forward ? descriptor.range
                                            : descriptor.domain;
}

void validate_view(const OperatorVectorView &view,
                   const OperatorSpaceDesc &expected,
                   Program *program,
                   const char *role,
                   bool require_writable) {
  TI_ERROR_IF(view.space != expected || (view.data == 0 && !view.ndarray) ||
                  view.allocation_identity == 0 ||
                  (require_writable && !view.writable),
              "Operator {} view does not match its declared space or "
              "access mode.",
              role);
  if (program) {
    TI_ERROR_IF(view.program != program,
                "Program-bound operator {} view must belong to the plan "
                "Program.",
                role);
  } else {
    TI_ERROR_IF(view.program || view.ndarray,
                "Host-reference operator {} view must not carry Program "
                "state.",
                role);
  }
}

template <typename T>
void axpby(const OperatorVectorView *applied,
           const OperatorVectorView *addend,
           const OperatorVectorView &output,
           double alpha,
           double beta) {
  const auto count = output.space.scalar_extent;
  const T *applied_data =
      applied ? reinterpret_cast<const T *>(applied->data) : nullptr;
  const T *addend_data =
      addend ? reinterpret_cast<const T *>(addend->data) : nullptr;
  auto *output_data = reinterpret_cast<T *>(output.data);
  for (std::size_t index = 0; index < count; ++index) {
    const double applied_value =
        applied_data ? static_cast<double>(applied_data[index]) : 0.0;
    const double addend_value =
        addend_data ? static_cast<double>(addend_data[index]) : 0.0;
    output_data[index] =
        static_cast<T>(alpha * applied_value + beta * addend_value);
  }
}

}  // namespace

OperatorDependencyMask operator_resource_changes(
    const OperatorResourceStamp &planned,
    const OperatorResourceStamp &current) {
  OperatorDependencyMask result = 0;
  if (planned.program_identity != current.program_identity ||
      planned.program_generation != current.program_generation) {
    result |= operator_dependency(OperatorResourceDependency::program);
  }
  if (planned.schema_revision != current.schema_revision) {
    result |= operator_dependency(OperatorResourceDependency::schema);
  }
  if (planned.topology_revision != current.topology_revision) {
    result |= operator_dependency(OperatorResourceDependency::topology);
  }
  if (planned.numeric_revision != current.numeric_revision) {
    result |= operator_dependency(OperatorResourceDependency::numeric);
  }
  if (planned.binding_revision != current.binding_revision) {
    result |= operator_dependency(OperatorResourceDependency::binding);
  }
  return result;
}

OperatorPlanInvalidation evaluate_operator_plan_invalidation(
    const OperatorResourceStamp &planned,
    const OperatorResourceStamp &current,
    OperatorDependencyMask dependencies) {
  OperatorPlanInvalidation result;
  result.changes = operator_resource_changes(planned, current);
  result.relevant_changes = result.changes & dependencies;
  if (result.relevant_changes &
      operator_dependency(OperatorResourceDependency::program)) {
    result.kind = OperatorPlanInvalidationKind::program_invalid;
  } else if (result.relevant_changes &
             (operator_dependency(OperatorResourceDependency::schema) |
              operator_dependency(OperatorResourceDependency::topology) |
              operator_dependency(OperatorResourceDependency::numeric))) {
    result.kind = OperatorPlanInvalidationKind::rebuild;
  } else if (result.relevant_changes &
             operator_dependency(OperatorResourceDependency::binding)) {
    result.kind = OperatorPlanInvalidationKind::refresh_binding;
  }
  return result;
}

bool OperatorSpaceDesc::operator==(const OperatorSpaceDesc &other) const {
  return scalar_type == other.scalar_type &&
         scalar_extent == other.scalar_extent &&
         entry_shape == other.entry_shape &&
         inner_product_kind == other.inner_product_kind;
}

const char *operator_execution_kind_name(OperatorExecutionKind kind) {
  switch (kind) {
    case OperatorExecutionKind::direct:
      return "direct";
    case OperatorExecutionKind::explicit_sequence:
      return "explicit_sequence";
    case OperatorExecutionKind::compiled_graph:
      return "compiled_graph";
    case OperatorExecutionKind::runtime_capture:
      return "runtime_capture";
  }
  return "unknown";
}

const char *operator_backend_execution_path_name(
    OperatorBackendExecutionPath path) {
  switch (path) {
    case OperatorBackendExecutionPath::unavailable:
      return "unavailable";
    case OperatorBackendExecutionPath::direct:
      return "direct";
    case OperatorBackendExecutionPath::explicit_sequence:
      return "explicit_sequence";
    case OperatorBackendExecutionPath::ordinary_graph_fallback:
      return "ordinary_graph_fallback";
    case OperatorBackendExecutionPath::cuda_capture:
      return "cuda_capture";
    case OperatorBackendExecutionPath::cuda_exact_replay:
      return "cuda_exact_replay";
    case OperatorBackendExecutionPath::cuda_patched_replay:
      return "cuda_patched_replay";
    case OperatorBackendExecutionPath::vulkan_record:
      return "vulkan_record";
    case OperatorBackendExecutionPath::vulkan_replay:
      return "vulkan_replay";
  }
  return "unknown";
}

OperatorMathematicalTraits make_spd_operator_traits(
    OperatorTraitProvenance provenance,
    OperatorDependencyMask validity_scope) {
  TI_ERROR_IF(provenance == OperatorTraitProvenance::unspecified ||
                  validity_scope == 0,
              "SPD traits require explicit provenance and validity scope.");
  OperatorMathematicalTraits traits;
  traits.self_adjoint = {true, provenance, validity_scope};
  traits.positive_definite = {true, provenance, validity_scope};
  traits.positive_semidefinite = {true, provenance, validity_scope};
  traits.singular = {false, provenance, validity_scope};
  return traits;
}

namespace {

bool trait_is_trusted_true(const OperatorTraitClaim &claim) {
  return claim.known() && claim.value &&
         claim.provenance != OperatorTraitProvenance::empirically_checked;
}

void validate_trait_claim(const OperatorTraitClaim &claim,
                          const char *name) {
  TI_ERROR_IF(claim.known() && claim.validity_scope == 0,
              "Operator trait '{}' requires a non-empty validity scope.",
              name);
  TI_ERROR_IF(!claim.known() && claim.validity_scope != 0,
              "Unknown operator trait '{}' must not carry validity scope.",
              name);
}

bool same_trait_claim(const OperatorTraitClaim &left,
                      const OperatorTraitClaim &right) {
  return left.value == right.value &&
         left.provenance == right.provenance &&
         left.validity_scope == right.validity_scope;
}

bool same_mathematical_traits(const OperatorMathematicalTraits &left,
                              const OperatorMathematicalTraits &right) {
  return same_trait_claim(left.self_adjoint, right.self_adjoint) &&
         same_trait_claim(left.positive_definite,
                          right.positive_definite) &&
         same_trait_claim(left.positive_semidefinite,
                          right.positive_semidefinite) &&
         same_trait_claim(left.singular, right.singular);
}

void validate_operator_execution_kind(
    const OperatorCapabilities &capabilities,
    OperatorExecutionKind kind) {
  bool supported = kind == OperatorExecutionKind::direct;
  if (kind == OperatorExecutionKind::explicit_sequence) {
    supported = capabilities.explicit_sequence;
  } else if (kind == OperatorExecutionKind::compiled_graph) {
    supported = capabilities.compiled_graph;
  } else if (kind == OperatorExecutionKind::runtime_capture) {
    supported = capabilities.runtime_capture;
  }
  TI_ERROR_IF(!supported,
              "Operator execution lowering '{}' is unsupported by this "
              "binding; no fallback was performed.",
              operator_execution_kind_name(kind));
}

}  // namespace

void validate_operator_solver_compatibility(
    const OperatorDescriptor &descriptor,
    const OperatorMathematicalTraits &traits,
    OperatorSolverFamily family,
    PreconditionerBehavior preconditioner_behavior) {
  TI_ERROR_IF(descriptor.domain != descriptor.range,
              "Krylov solver requires a square operator descriptor.");
  if (family == OperatorSolverFamily::bicgstab) {
    return;
  }
  TI_ERROR_IF(family == OperatorSolverFamily::pcg &&
                  preconditioner_behavior !=
                      PreconditionerBehavior::fixed_linear,
              "Ordinary PCG requires a fixed-linear preconditioner; "
              "flexible or nonlinear behavior needs a compatible solver.");
  TI_ERROR_IF(!trait_is_trusted_true(traits.self_adjoint),
              "CG/PCG requires a trusted self-adjoint trait; "
              "unknown or empirically-checked claims are insufficient.");
  TI_ERROR_IF(!trait_is_trusted_true(traits.positive_definite),
              "CG/PCG requires a trusted positive-definite trait; "
              "unknown or empirically-checked claims are insufficient.");
  TI_ERROR_IF(traits.singular.known() && traits.singular.value,
              "CG/PCG rejects operators declared singular.");
}

OperatorVectorView OperatorVectorView::from_const_host(
    const void *data,
    const OperatorSpaceDesc &space) {
  validate_space(space, "host input");
  const auto address = reinterpret_cast<std::uintptr_t>(data);
  return {space, address, address, nullptr, nullptr, false};
}

OperatorVectorView OperatorVectorView::from_mutable_host(
    void *data,
    const OperatorSpaceDesc &space) {
  validate_space(space, "host output");
  const auto address = reinterpret_cast<std::uintptr_t>(data);
  return {space, address, address, nullptr, nullptr, true};
}

OperatorVectorView OperatorVectorView::from_ndarray(
    Program *program,
    const Ndarray &array,
    const OperatorSpaceDesc &space,
    bool writable) {
  validate_space(space, "ndarray");
  TI_ERROR_IF(!program || array.owning_program() != program ||
                  array.get_element_data_type() != space.scalar_type ||
                  array.get_element_shape() != space.entry_shape ||
                  array.shape.size() != 1 ||
                  array.get_nelement() != space.scalar_extent,
              "Operator ndarray view must match Program, dtype, entry "
              "shape, rank, and scalar extent.");
  return {
      space,
      static_cast<std::uintptr_t>(program->get_ndarray_data_ptr_as_int(&array)),
      static_cast<std::uintptr_t>(array.get_device_allocation_ptr_as_int()),
      &array,
      program,
      writable};
}

OperatorVectorView OperatorVectorView::from_device_pointer(
    Program *program,
    std::uintptr_t data,
    const OperatorSpaceDesc &space,
    bool writable) {
  validate_space(space, "device pointer");
  TI_ERROR_IF(!program || data == 0,
              "Operator device pointer view requires an active Program and "
              "non-null device address.");
  return {space, data, data, nullptr, program, writable};
}

struct OperatorAction::State {
  OperatorDescriptor descriptor;
  OperatorMathematicalTraits mathematical_traits;
  OperatorCapabilities capabilities;
  std::string provider_name;
  ResourceStampFn resource_stamp;
  OverwriteApplyFn overwrite_apply;
};

OperatorAction::OperatorAction(OperatorDescriptor descriptor,
                               OperatorCapabilities capabilities,
                               std::string provider_name,
                               ResourceStampFn resource_stamp,
                               OverwriteApplyFn overwrite_apply)
    : OperatorAction(std::move(descriptor), OperatorMathematicalTraits{},
                     capabilities, std::move(provider_name),
                     std::move(resource_stamp), std::move(overwrite_apply)) {
}

OperatorAction::OperatorAction(
    OperatorDescriptor descriptor,
    OperatorMathematicalTraits mathematical_traits,
    OperatorCapabilities capabilities,
    std::string provider_name,
    ResourceStampFn resource_stamp,
    OverwriteApplyFn overwrite_apply) {
  validate_space(descriptor.domain, "domain");
  validate_space(descriptor.range, "range");
  validate_trait_claim(mathematical_traits.self_adjoint, "self_adjoint");
  validate_trait_claim(mathematical_traits.positive_definite,
                       "positive_definite");
  validate_trait_claim(mathematical_traits.positive_semidefinite,
                       "positive_semidefinite");
  validate_trait_claim(mathematical_traits.singular, "singular");
  TI_ERROR_IF(mathematical_traits.positive_definite.known() &&
                  mathematical_traits.positive_definite.value &&
                  (!mathematical_traits.self_adjoint.known() ||
                   !mathematical_traits.self_adjoint.value),
              "A positive-definite trait requires an explicit true "
              "self-adjoint trait.");
  TI_ERROR_IF(provider_name.empty() || !resource_stamp || !overwrite_apply ||
                  !capabilities.forward_apply,
              "OperatorAction requires a named forward provider, resource "
              "stamp, and overwrite apply function.");
  state_ = std::make_shared<State>(State{
      std::move(descriptor), std::move(mathematical_traits), capabilities,
      std::move(provider_name), std::move(resource_stamp),
      std::move(overwrite_apply)});
}

const OperatorDescriptor &OperatorAction::descriptor() const {
  return state_->descriptor;
}

const OperatorMathematicalTraits &OperatorAction::mathematical_traits() const {
  return state_->mathematical_traits;
}

const OperatorCapabilities &OperatorAction::capabilities() const {
  return state_->capabilities;
}

const std::string &OperatorAction::provider_name() const {
  return state_->provider_name;
}

OperatorResourceStamp OperatorAction::resource_stamp() const {
  return state_->resource_stamp();
}

void OperatorAction::apply_overwrite(OperatorApplyMode mode,
                                     const OperatorVectorView &input,
                                     const OperatorVectorView &output) const {
  TI_ERROR_IF(
      mode == OperatorApplyMode::adjoint && !state_->capabilities.adjoint_apply,
      "Operator provider '{}' does not support adjoint apply; no "
      "fallback was performed.",
      state_->provider_name);
  state_->overwrite_apply(mode, input, output);
}

OperatorAction OperatorAction::with_mathematical_traits(
    OperatorMathematicalTraits mathematical_traits) const {
  return OperatorAction(state_->descriptor, std::move(mathematical_traits),
                        state_->capabilities, state_->provider_name,
                        state_->resource_stamp, state_->overwrite_apply);
}

OperatorResourceLease::OperatorResourceLease(std::shared_ptr<void> state)
    : state_(std::move(state)) {
}

OperatorPinnedAction::OperatorPinnedAction(OperatorAction action,
                                           OperatorResourceStamp stamp,
                                           OperatorResourceLease resource_lease)
    : action_(std::make_shared<OperatorAction>(std::move(action))),
      stamp_(stamp),
      resource_lease_(std::move(resource_lease)) {
}

OperatorPinnedAction OperatorPinnedAction::from_retained_action(
    OperatorAction action,
    OperatorResourceStamp stamp,
    OperatorResourceLease resource_lease) {
  TI_ERROR_IF(operator_resource_changes(action.resource_stamp(), stamp) != 0,
              "Retained operator action and resource stamp must describe "
              "the same immutable generation.");
  return OperatorPinnedAction(std::move(action), stamp,
                              std::move(resource_lease));
}

OperatorPinnedAction::operator bool() const {
  return action_ != nullptr;
}

const OperatorDescriptor &OperatorPinnedAction::descriptor() const {
  TI_ERROR_IF(!action_, "Operator generation pin is empty.");
  return action_->descriptor();
}

const OperatorMathematicalTraits &
OperatorPinnedAction::mathematical_traits() const {
  TI_ERROR_IF(!action_, "Operator generation pin is empty.");
  return action_->mathematical_traits();
}

const OperatorCapabilities &OperatorPinnedAction::capabilities() const {
  TI_ERROR_IF(!action_, "Operator generation pin is empty.");
  return action_->capabilities();
}

const std::string &OperatorPinnedAction::provider_name() const {
  TI_ERROR_IF(!action_, "Operator generation pin is empty.");
  return action_->provider_name();
}

OperatorResourceStamp OperatorPinnedAction::resource_stamp() const {
  TI_ERROR_IF(!action_, "Operator generation pin is empty.");
  return stamp_;
}

void OperatorPinnedAction::apply_overwrite(
    OperatorApplyMode mode,
    const OperatorVectorView &input,
    const OperatorVectorView &output) const {
  TI_ERROR_IF(!action_, "Operator generation pin is empty.");
  action_->apply_overwrite(mode, input, output);
}

OperatorPinnedAction OperatorPinnedAction::with_mathematical_traits(
    OperatorMathematicalTraits mathematical_traits) const {
  TI_ERROR_IF(!action_, "Operator generation pin is empty.");
  return OperatorPinnedAction(
      action_->with_mathematical_traits(std::move(mathematical_traits)),
      stamp_, resource_lease_);
}

OperatorSubmission::OperatorSubmission(OperatorPinnedAction generation,
                                       RuntimeCompletion completion,
                                       bool synchronous)
    : resource_stamp(generation.resource_stamp()),
      completed_synchronously(synchronous),
      generation_(std::move(generation)),
      completion_(std::move(completion)) {
}

OperatorSubmission::OperatorSubmission(OperatorSubmission &&other) noexcept
    : resource_stamp(other.resource_stamp),
      completed_synchronously(other.completed_synchronously),
      generation_(std::move(other.generation_)),
      completion_(std::move(other.completion_)) {
  other.completed_synchronously = true;
}

OperatorSubmission::~OperatorSubmission() {
  if (!completed_synchronously && completion_.valid()) {
    try {
      completion_.wait();
    } catch (...) {
      // Destruction cannot report an asynchronous backend error. The shared
      // RuntimeFaultDomain keeps it observable through Program diagnostics.
    }
  }
}

bool OperatorSubmission::done() const {
  if (completed_synchronously) {
    return true;
  }
  TI_ERROR_IF(!completion_.valid(),
              "Pinned OperatorPlan submission uses an externally managed "
              "completion boundary.");
  return completion_.done();
}

void OperatorSubmission::wait() const {
  if (completed_synchronously) {
    return;
  }
  TI_ERROR_IF(!completion_.valid(),
              "Pinned OperatorPlan submission uses an externally managed "
              "completion boundary.");
  completion_.wait();
}

namespace {

struct PublishedOperatorGeneration {
  PublishedOperatorGeneration(OperatorAction action,
                              OperatorResourceStamp stamp,
                              OperatorResourceLease resources)
      : action(std::move(action)),
        stamp(stamp),
        resources(std::move(resources)) {
  }

  OperatorAction action;
  OperatorResourceStamp stamp;
  OperatorResourceLease resources;
};

using OperatorGenerationRegistry =
    RuntimeResourceRegistry<PublishedOperatorGeneration>;
constexpr OperatorGenerationRegistry::Kind kOperatorGenerationKind = 1;

}  // namespace

struct OperatorResourceGenerationPublisher::Impl {
  explicit Impl(std::uint64_t domain) : registry(domain) {
  }

  mutable std::mutex mutex;
  OperatorGenerationRegistry registry;
  OperatorGenerationRegistry::Handle current;
};

OperatorResourceGenerationPublisher::OperatorResourceGenerationPublisher()
    : impl_(std::make_unique<Impl>(allocate_operator_generation_domain())) {
}

OperatorResourceGenerationPublisher::~OperatorResourceGenerationPublisher() {
  try {
    retire_current();
  } catch (...) {
  }
}

void OperatorResourceGenerationPublisher::publish(
    OperatorAction action,
    OperatorResourceLease resources) {
  const auto stamp = action.resource_stamp();
  TI_ERROR_IF((stamp.program_identity == 0) != (stamp.program_generation == 0),
              "Operator generation Program identity and generation must "
              "either both be present or both be absent.");
  std::lock_guard<std::mutex> lock(impl_->mutex);
  auto [result, next] = impl_->registry.emplace(
      kOperatorGenerationKind, std::move(action), stamp, std::move(resources));
  TI_ERROR_IF(result != OperatorGenerationRegistry::Result::kSuccess,
              "Unable to publish immutable operator resource generation.");
  const auto previous = impl_->current;
  impl_->current = next;
  if (previous) {
    const auto retire_result = impl_->registry.retire(previous);
    TI_ERROR_IF(retire_result != OperatorGenerationRegistry::Result::kSuccess,
                "Unable to retire replaced operator resource generation.");
  }
}

OperatorPinnedAction OperatorResourceGenerationPublisher::acquire() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  TI_ERROR_IF(!impl_->current,
              "Operator resource generation publisher has no current "
              "generation.");
  auto [result, lease] = impl_->registry.acquire(impl_->current);
  TI_ERROR_IF(result != OperatorGenerationRegistry::Result::kSuccess || !lease,
              "Unable to pin current operator resource generation.");
  auto action = lease->action;
  const auto stamp = lease->stamp;
  return OperatorPinnedAction(std::move(action), stamp,
                              OperatorResourceLease::hold(std::move(lease)));
}

void OperatorResourceGenerationPublisher::retire_current() {
  if (!impl_) {
    return;
  }
  std::lock_guard<std::mutex> lock(impl_->mutex);
  const auto current =
      std::exchange(impl_->current, OperatorGenerationRegistry::Handle{});
  if (!current) {
    return;
  }
  const auto result = impl_->registry.retire(current);
  TI_ERROR_IF(result != OperatorGenerationRegistry::Result::kSuccess,
              "Unable to retire current operator resource generation.");
}

OperatorResourceGenerationStatistics
OperatorResourceGenerationPublisher::debug_statistics() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  const auto registry_stats = impl_->registry.stats();
  return {registry_stats.created_total, registry_stats.retired_total,
          registry_stats.released_total, registry_stats.leases,
          static_cast<bool>(impl_->current)};
}

OperatorBinding::OperatorBinding(OperatorAction action,
                                 AcquireResourceLeaseFn acquire_resource_lease)
    : action_(std::move(action)),
      acquire_resource_lease_(std::move(acquire_resource_lease)) {
}

OperatorBinding::OperatorBinding(OperatorAction metadata_action,
                                 AcquirePinnedActionFn acquire_pinned_action,
                                 bool generation_bound)
    : action_(std::move(metadata_action)),
      acquire_pinned_action_(std::move(acquire_pinned_action)) {
  TI_ERROR_IF(!generation_bound || !acquire_pinned_action_,
              "Generation-bound operator binding requires a pin callback.");
}

OperatorBinding OperatorBinding::from_generation_publisher(
    OperatorAction metadata_action,
    AcquirePinnedActionFn acquire_pinned_action) {
  return OperatorBinding(std::move(metadata_action),
                         std::move(acquire_pinned_action), true);
}

const OperatorAction &OperatorBinding::action() const {
  return action_;
}

OperatorBinding OperatorBinding::with_mathematical_traits(
    OperatorMathematicalTraits mathematical_traits) const {
  auto source = *this;
  auto metadata = action_.with_mathematical_traits(mathematical_traits);
  auto result = OperatorBinding::from_generation_publisher(
      std::move(metadata),
      [source = std::move(source),
       mathematical_traits = std::move(mathematical_traits)] {
        return source.pin().with_mathematical_traits(mathematical_traits);
      });
  result.execution_kind_ = execution_kind_;
  result.execution_statistics_ = execution_statistics_;
  return result;
}

OperatorBinding OperatorBinding::with_execution_lowering(
    OperatorExecutionKind execution_kind,
    ExecutionRuntimeStatisticsFn execution_statistics) const {
  auto result = *this;
  result.execution_kind_ = execution_kind;
  result.execution_statistics_ = std::move(execution_statistics);
  return result;
}

OperatorExecutionKind OperatorBinding::execution_kind() const {
  return execution_kind_;
}

OperatorBinding::ExecutionRuntimeStatistics
OperatorBinding::execution_runtime_statistics() const {
  if (execution_statistics_) {
    return execution_statistics_();
  }
  ExecutionRuntimeStatistics result;
  if (execution_kind_ == OperatorExecutionKind::direct) {
    result.last_backend_path = OperatorBackendExecutionPath::direct;
  } else if (execution_kind_ == OperatorExecutionKind::explicit_sequence) {
    result.last_backend_path =
        OperatorBackendExecutionPath::explicit_sequence;
  }
  return result;
}

OperatorResourceLease OperatorBinding::acquire_resource_lease() const {
  if (acquire_pinned_action_) {
    return OperatorResourceLease::hold(acquire_pinned_action_());
  }
  return acquire_resource_lease_ ? acquire_resource_lease_()
                                 : OperatorResourceLease{};
}

OperatorPinnedAction OperatorBinding::pin() const {
  if (acquire_pinned_action_) {
    return acquire_pinned_action_();
  }
  // Acquire the provider transaction before reading its stamp. Function
  // argument evaluation order must not decide whether the stamp and retained
  // resources describe the same numeric generation.
  auto resource_lease = acquire_resource_lease();
  const auto stamp = action_.resource_stamp();
  return OperatorPinnedAction(action_, stamp, std::move(resource_lease));
}

struct OperatorPlan::Scratch {
  OperatorSpaceDesc space;
  Ndarray *array{nullptr};
  std::vector<std::uint64_t> host_words;
  OperatorVectorView view;
};

OperatorPlan::OperatorPlan(Program *program,
                           OperatorAction action,
                           OperatorDependencyMask dependencies)
    : OperatorPlan(program, OperatorBinding(std::move(action)), dependencies) {
}

OperatorPlan::OperatorPlan(Program *program,
                           OperatorBinding binding,
                           OperatorDependencyMask dependencies)
    : program_(program),
      binding_(std::move(binding)),
      dependencies_(dependencies),
      planned_stamp_(binding_.action().resource_stamp()) {
  validate_operator_execution_kind(binding_.action().capabilities(),
                                   binding_.execution_kind());
  statistics_.execution_plan_builds = 1;
  statistics_.execution_kind = binding_.execution_kind();
  TI_ERROR_IF(program_ && !arch_is_cpu(program_->compile_config().arch) &&
                  !arch_is_cuda(program_->compile_config().arch) &&
                  program_->compile_config().arch != Arch::vulkan,
              "OperatorPlan supports CPU, CUDA, and Vulkan Programs only; "
              "no fallback was performed.");
  if (program_) {
    TI_ERROR_IF(
        planned_stamp_.program_identity !=
                reinterpret_cast<std::uintptr_t>(program_) ||
            planned_stamp_.program_generation !=
                program_->runtime_program_generation(),
        "OperatorPlan binding belongs to a different Program generation.");
  }
}

OperatorPlan::~OperatorPlan() {
  if (forward_scratch_) {
    release_scratch(*forward_scratch_);
  }
  if (adjoint_scratch_) {
    release_scratch(*adjoint_scratch_);
  }
}

const OperatorDescriptor &OperatorPlan::descriptor() const {
  return binding_.action().descriptor();
}

const OperatorMathematicalTraits &OperatorPlan::mathematical_traits() const {
  return binding_.action().mathematical_traits();
}

const OperatorCapabilities &OperatorPlan::capabilities() const {
  return binding_.action().capabilities();
}

const std::string &OperatorPlan::provider_name() const {
  return binding_.action().provider_name();
}

OperatorExecutionKind OperatorPlan::execution_kind() const {
  return binding_.execution_kind();
}

OperatorDependencyMask OperatorPlan::dependencies() const {
  return dependencies_;
}

OperatorResourceStamp OperatorPlan::resource_stamp() const {
  auto pinned = binding_.pin();
  return pinned.resource_stamp();
}

OperatorResourceLease OperatorPlan::acquire_resource_lease() const {
  return binding_.acquire_resource_lease();
}

OperatorPinnedAction OperatorPlan::pin() {
  auto pinned = binding_.pin();
  TI_ERROR_IF(pinned.descriptor().domain != descriptor().domain ||
                  pinned.descriptor().range != descriptor().range ||
                  pinned.provider_name() != provider_name() ||
                  !same_mathematical_traits(
                      pinned.mathematical_traits(), mathematical_traits()),
              "Operator generation changed descriptor, mathematical traits, "
              "or provider identity without rebuilding its binding.");
  const auto stamp = pinned.resource_stamp();
  const auto invalidation =
      evaluate_operator_plan_invalidation(planned_stamp_, stamp, dependencies_);
  statistics_.generation_pins++;
  if (has_pinned_generation_) {
    const auto changes = operator_resource_changes(last_pinned_stamp_, stamp);
    if (changes != 0) {
      statistics_.generation_changes++;
    }
    if (changes & operator_dependency(OperatorResourceDependency::numeric)) {
      statistics_.numeric_generation_changes++;
    }
    if (changes & operator_dependency(OperatorResourceDependency::binding)) {
      statistics_.binding_generation_changes++;
    }
  }
  last_pinned_stamp_ = stamp;
  has_pinned_generation_ = true;
  if (invalidation.kind != OperatorPlanInvalidationKind::current) {
    statistics_.invalidations++;
  }
  TI_ERROR_IF(
      invalidation.kind == OperatorPlanInvalidationKind::program_invalid,
      "OperatorPlan belongs to a stale Program generation.");
  TI_ERROR_IF(invalidation.kind == OperatorPlanInvalidationKind::rebuild,
              "OperatorPlan dependencies changed and require plan rebuild.");
  TI_ERROR_IF(
      invalidation.kind == OperatorPlanInvalidationKind::refresh_binding,
      "OperatorPlan binding identity changed and requires refresh.");
  return pinned;
}

namespace {

void validate_preconditioner_descriptor(
    const OperatorDescriptor &target,
    const OperatorDescriptor &preconditioner) {
  TI_ERROR_IF(target.domain != target.range,
              "PreconditionerPlan requires a square target operator.");
  TI_ERROR_IF(preconditioner.domain != target.range ||
                  preconditioner.range != target.domain,
              "PreconditionerPlan action must map the target range back to "
              "its domain.");
}

void validate_preconditioner_generation_pair(
    const OperatorPinnedAction &target,
    const OperatorPinnedAction &preconditioner) {
  const auto target_stamp = target.resource_stamp();
  const auto preconditioner_stamp = preconditioner.resource_stamp();
  TI_ERROR_IF(
      target_stamp.program_identity != preconditioner_stamp.program_identity ||
          target_stamp.program_generation !=
              preconditioner_stamp.program_generation ||
          target_stamp.schema_revision !=
              preconditioner_stamp.schema_revision ||
          target_stamp.topology_revision !=
              preconditioner_stamp.topology_revision ||
          target_stamp.numeric_revision !=
              preconditioner_stamp.numeric_revision,
      "PreconditionerPlan action generation does not match the pinned target "
      "operator generation.");
}

}  // namespace

PreconditionerPlan::PreconditionerPlan(Program *program,
                                       OperatorDescriptor target_descriptor,
                                       OperatorBinding action_binding,
                                       PreconditionerBehavior behavior,
                                       std::string method,
                                       UpdateFn update)
    : program_(program),
      target_descriptor_(std::move(target_descriptor)),
      action_plan_(
          std::make_unique<OperatorPlan>(program, std::move(action_binding))),
      behavior_(behavior),
      method_(std::move(method)),
      update_(std::move(update)) {
  validate_space(target_descriptor_.domain, "preconditioner target domain");
  validate_space(target_descriptor_.range, "preconditioner target range");
  validate_preconditioner_descriptor(target_descriptor_,
                                     action_plan_->descriptor());
  TI_ERROR_IF(behavior_ != PreconditionerBehavior::fixed_linear,
              "M3 PreconditionerPlan implements fixed-linear behavior only.");
  TI_ERROR_IF(method_.empty() || !update_,
              "PreconditionerPlan requires a method name and update "
              "callback.");
}

PreconditionerPlan::~PreconditionerPlan() = default;

void PreconditionerPlan::setup(const OperatorPinnedAction &target_generation) {
  TI_ERROR_IF(is_setup_, "PreconditionerPlan setup may only run once.");
  (void)update_and_pin_impl(target_generation, true);
}

OperatorPinnedAction PreconditionerPlan::update_and_pin(
    const OperatorPinnedAction &target_generation) {
  TI_ERROR_IF(!is_setup_,
              "PreconditionerPlan must be setup before update/action use.");
  return update_and_pin_impl(target_generation, false);
}

OperatorPinnedAction PreconditionerPlan::update_and_pin_impl(
    const OperatorPinnedAction &target_generation,
    bool setup) {
  TI_ERROR_IF(!target_generation,
              "PreconditionerPlan requires a pinned target generation.");
  TI_ERROR_IF(
      target_generation.descriptor().domain != target_descriptor_.domain ||
          target_generation.descriptor().range != target_descriptor_.range,
      "PreconditionerPlan target descriptor changed after plan "
      "construction.");
  const auto target_stamp = target_generation.resource_stamp();
  const bool target_changed =
      !setup && operator_resource_changes(target_stamp_, target_stamp) != 0;
  if (setup) {
    statistics_.setup_calls++;
  } else {
    statistics_.update_calls++;
    if (target_changed) {
      statistics_.target_generation_changes++;
    }
  }
  try {
    update_(target_stamp, setup || target_changed);
    auto preconditioner_generation = action_plan_->pin();
    validate_preconditioner_generation_pair(target_generation,
                                            preconditioner_generation);
    target_stamp_ = target_stamp;
    is_setup_ = true;
    if (!setup) {
      if (target_changed) {
        statistics_.update_successes++;
      } else {
        statistics_.update_noops++;
      }
    }
    return preconditioner_generation;
  } catch (...) {
    if (!setup) {
      statistics_.update_failures++;
    }
    throw;
  }
}

const OperatorPlan &PreconditionerPlan::action() const {
  return *action_plan_;
}

OperatorPlan &PreconditionerPlan::action() {
  return *action_plan_;
}

PreconditionerBehavior PreconditionerPlan::behavior() const {
  return behavior_;
}

const std::string &PreconditionerPlan::method() const {
  return method_;
}

PreconditionerPlanRuntimeStatistics
PreconditionerPlan::debug_runtime_statistics() const {
  return statistics_;
}

OperatorVectorView OperatorPlan::scratch_for(const OperatorSpaceDesc &space,
                                             OperatorApplyMode mode) {
  auto &slot =
      mode == OperatorApplyMode::forward ? forward_scratch_ : adjoint_scratch_;
  if (slot) {
    TI_ERROR_IF(slot->space != space,
                "OperatorPlan scratch space changed without rebuilding the "
                "plan.");
    statistics_.scratch_reuses++;
    return slot->view;
  }

  auto scratch = std::make_unique<Scratch>();
  scratch->space = space;
  const std::size_t bytes = space_bytes(space);
  if (program_) {
    TI_ERROR_IF(space.scalar_extent >
                    static_cast<std::size_t>(std::numeric_limits<int>::max()),
                "OperatorPlan scratch extent exceeds ndarray limits.");
    scratch->array = program_->create_ndarray(
        space.scalar_type, {static_cast<int>(space.scalar_extent)},
        ExternalArrayLayout::kNull, false);
    scratch->view = OperatorVectorView::from_ndarray(program_, *scratch->array,
                                                     space, true);
  } else {
    scratch->host_words.resize((bytes + sizeof(std::uint64_t) - 1) /
                               sizeof(std::uint64_t));
    scratch->view = OperatorVectorView::from_mutable_host(
        scratch->host_words.data(), space);
  }
  statistics_.scratch_builds++;
  statistics_.scratch_reserved_bytes += bytes;
  slot = std::move(scratch);
  return slot->view;
}

void OperatorPlan::release_scratch(Scratch &scratch) {
  if (scratch.array && program_) {
    program_->delete_ndarray(scratch.array);
    scratch.array = nullptr;
  }
}

OperatorSubmission OperatorPlan::submit(const OperatorApplyRequest &request) {
  auto pinned = pin();
  if (!program_ || !capabilities().asynchronous_submit) {
    return submit(pinned, request);
  }

  auto transaction = program_->begin_runtime_submission_transaction();
  try {
    (void)submit(pinned, request);
  } catch (...) {
    const auto submission_error = std::current_exception();
    transaction->mark_submission();
    try {
      auto completion = transaction->finish();
      completion.wait();
    } catch (...) {
    }
    std::rethrow_exception(submission_error);
  }
  transaction->mark_submission();
  auto completion = transaction->finish();
  const bool synchronous = !completion.has_backend_work();
  return OperatorSubmission(std::move(pinned), std::move(completion),
                            synchronous);
}

OperatorSubmission OperatorPlan::submit(const OperatorPinnedAction &pinned,
                                        const OperatorApplyRequest &request) {
  TI_ERROR_IF(!std::isfinite(request.alpha) || !std::isfinite(request.beta),
              "Operator generalized apply coefficients must be finite.");
  const auto &expected_input = input_space(descriptor(), request.mode);
  const auto &expected_output = output_space(descriptor(), request.mode);
  validate_view(request.input, expected_input, program_, "input", false);
  validate_view(request.output, expected_output, program_, "output", true);
  TI_ERROR_IF(
      request.input.allocation_identity == request.output.allocation_identity,
      "Operator input and output must not alias.");
  if (request.beta != 0.0) {
    TI_ERROR_IF(!request.addend,
                "Operator generalized apply with nonzero beta requires an "
                "addend.");
    validate_view(*request.addend, expected_output, program_, "addend", false);
  }

  if (has_vector_binding_ &&
      (last_input_binding_ != request.input.allocation_identity ||
       last_output_binding_ != request.output.allocation_identity)) {
    statistics_.binding_rebinds++;
  }
  has_vector_binding_ = true;
  last_input_binding_ = request.input.allocation_identity;
  last_output_binding_ = request.output.allocation_identity;
  statistics_.execution_plan_reuses++;
  statistics_.submissions++;
  const auto &action = pinned;
  if (request.alpha == 1.0 && request.beta == 0.0) {
    action.apply_overwrite(request.mode, request.input, request.output);
    statistics_.primitive_apply_calls++;
    return OperatorSubmission(pinned, RuntimeCompletion{},
                              !capabilities().asynchronous_submit);
  }

  TI_ERROR_IF(program_ && !arch_is_cpu(program_->compile_config().arch),
              "Generalized operator lowering is unavailable on this GPU "
              "backend; only overwrite apply is supported and no host "
              "fallback was performed.");
  statistics_.generalized_lowerings++;
  OperatorVectorView applied;
  OperatorVectorView *applied_ptr = nullptr;
  if (request.alpha != 0.0) {
    applied = scratch_for(expected_output, request.mode);
    action.apply_overwrite(request.mode, request.input, applied);
    statistics_.primitive_apply_calls++;
    applied_ptr = &applied;
  }
  const OperatorVectorView *addend =
      request.beta == 0.0 ? nullptr : request.addend;
  if (expected_output.scalar_type == PrimitiveType::f32) {
    axpby<float32>(applied_ptr, addend, request.output, request.alpha,
                   request.beta);
  } else {
    axpby<float64>(applied_ptr, addend, request.output, request.alpha,
                   request.beta);
  }
  return OperatorSubmission(pinned, RuntimeCompletion{}, true);
}

OperatorPlanRuntimeStatistics OperatorPlan::debug_runtime_statistics() const {
  auto result = statistics_;
  const auto execution = binding_.execution_runtime_statistics();
  result.last_backend_path = execution.last_backend_path;
  result.sequence_submissions = execution.sequence_submissions;
  result.compiled_graph_submissions =
      execution.compiled_graph_submissions;
  result.runtime_capture_submissions =
      execution.runtime_capture_submissions;
  result.backend_captures = execution.backend_captures;
  result.backend_replays = execution.backend_replays;
  result.ordinary_fallbacks = execution.ordinary_fallbacks;
  result.cache_invalidations = execution.cache_invalidations;
  return result;
}

OperatorAction make_dense_reference_operator_action(
    OperatorDescriptor descriptor,
    std::vector<double> row_major_values) {
  validate_space(descriptor.domain, "dense domain");
  validate_space(descriptor.range, "dense range");
  TI_ERROR_IF(descriptor.domain.scalar_type != descriptor.range.scalar_type ||
                  row_major_values.size() != descriptor.domain.scalar_extent *
                                                 descriptor.range.scalar_extent,
              "Dense reference operator requires matching scalar types and "
              "exact row-major storage.");
  auto values =
      std::make_shared<const std::vector<double>>(std::move(row_major_values));
  const auto apply_descriptor = descriptor;
  const auto stamp = OperatorResourceStamp{};
  OperatorCapabilities capabilities;
  capabilities.adjoint_apply = true;
  return OperatorAction(
      descriptor, capabilities, "dense_reference", [stamp] { return stamp; },
      [descriptor = apply_descriptor, values](
          OperatorApplyMode mode, const OperatorVectorView &input,
          const OperatorVectorView &output) {
        const std::size_t rows = descriptor.range.scalar_extent;
        const std::size_t cols = descriptor.domain.scalar_extent;
        auto apply = [&](auto scalar_tag) {
          using T = decltype(scalar_tag);
          const auto *input_data = reinterpret_cast<const T *>(input.data);
          auto *output_data = reinterpret_cast<T *>(output.data);
          if (mode == OperatorApplyMode::forward) {
            for (std::size_t row = 0; row < rows; ++row) {
              double sum = 0.0;
              for (std::size_t col = 0; col < cols; ++col) {
                sum += (*values)[row * cols + col] *
                       static_cast<double>(input_data[col]);
              }
              output_data[row] = static_cast<T>(sum);
            }
          } else {
            for (std::size_t col = 0; col < cols; ++col) {
              double sum = 0.0;
              for (std::size_t row = 0; row < rows; ++row) {
                sum += (*values)[row * cols + col] *
                       static_cast<double>(input_data[row]);
              }
              output_data[col] = static_cast<T>(sum);
            }
          }
        };
        if (descriptor.domain.scalar_type == PrimitiveType::f32) {
          apply(float32{});
        } else {
          apply(float64{});
        }
      });
}

OperatorBinding make_adjoint_operator_binding(OperatorBinding operand) {
  const auto &source = operand.action();
  TI_ERROR_IF(!source.capabilities().adjoint_apply,
              "Operator provider '{}' cannot form an adjoint binding "
              "because explicit adjoint apply is unavailable; no "
              "materialization or symmetry fallback was performed.",
              source.provider_name());

  const OperatorDescriptor descriptor{source.descriptor().range,
                                      source.descriptor().domain};
  OperatorMathematicalTraits traits;
  if (source.descriptor().domain == source.descriptor().range) {
    const auto derive = [](const OperatorTraitClaim &claim) {
      if (!claim.known()) {
        return OperatorTraitClaim{};
      }
      const auto provenance =
          claim.provenance == OperatorTraitProvenance::empirically_checked
              ? OperatorTraitProvenance::empirically_checked
              : OperatorTraitProvenance::derived_structurally;
      return OperatorTraitClaim{claim.value, provenance,
                                claim.validity_scope};
    };
    const auto &source_traits = source.mathematical_traits();
    traits.self_adjoint = derive(source_traits.self_adjoint);
    traits.positive_definite = derive(source_traits.positive_definite);
    traits.positive_semidefinite =
        derive(source_traits.positive_semidefinite);
    traits.singular = derive(source_traits.singular);
  }

  auto capabilities = source.capabilities();
  capabilities.forward_apply = true;
  capabilities.adjoint_apply = true;
  capabilities.native_generalized_apply = false;
  const std::string provider_name =
      "adjoint(" + source.provider_name() + ")";
  auto metadata = OperatorAction(
      descriptor, traits, capabilities, provider_name,
      [operand] { return operand.pin().resource_stamp(); },
      [](OperatorApplyMode, const OperatorVectorView &,
         const OperatorVectorView &) {
        TI_ERROR("Adjoint operator metadata action cannot be submitted "
                 "without pinning its operand generation.");
      });
  return OperatorBinding::from_generation_publisher(
      std::move(metadata),
      [operand = std::move(operand), descriptor, traits, capabilities,
       provider_name] {
        auto source_generation = operand.pin();
        const auto stamp = source_generation.resource_stamp();
        auto action = OperatorAction(
            descriptor, traits, capabilities, provider_name,
            [stamp] { return stamp; },
            [source_generation = std::move(source_generation)](
                OperatorApplyMode mode, const OperatorVectorView &input,
                const OperatorVectorView &output) {
              source_generation.apply_overwrite(
                  mode == OperatorApplyMode::forward
                      ? OperatorApplyMode::adjoint
                      : OperatorApplyMode::forward,
                  input, output);
            });
        return OperatorPinnedAction::from_retained_action(std::move(action),
                                                          stamp);
      });
}

namespace {

void validate_synchronous_composite_operand(const OperatorAction &operand,
                                            const char *role) {
  TI_ERROR_IF(operand.capabilities().asynchronous_submit,
              "M3 {} operator composition requires a synchronous operand; "
              "GPU/Graph composite lowering is deferred.",
              role);
}

void validate_host_composite_views(const OperatorVectorView &input,
                                   const OperatorVectorView &output) {
  TI_ERROR_IF(input.program != output.program,
              "Composite operator input and output must belong to the same "
              "Program.");
  TI_ERROR_IF(input.program &&
                  !arch_is_cpu(input.program->compile_config().arch),
              "M3 composite operators support host and CPU views only; "
              "GPU/Graph lowering is deferred.");
}

OperatorVectorView as_raw_composite_view(const OperatorVectorView &view,
                                         bool writable) {
  auto result = view;
  result.allocation_identity = view.data;
  result.ndarray = nullptr;
  result.writable = writable;
  return result;
}

template <typename T>
void copy_composite_vector(const OperatorVectorView &input,
                           const OperatorVectorView &output) {
  const auto *source = reinterpret_cast<const T *>(input.data);
  auto *target = reinterpret_cast<T *>(output.data);
  std::copy(source, source + output.space.scalar_extent, target);
}

template <typename T>
void scale_composite_vector(double scale,
                            const OperatorVectorView &output) {
  auto *target = reinterpret_cast<T *>(output.data);
  for (std::size_t i = 0; i < output.space.scalar_extent; ++i) {
    target[i] = static_cast<T>(scale * static_cast<double>(target[i]));
  }
}

template <typename T>
void add_composite_vector(const OperatorVectorView &addend,
                          const OperatorVectorView &output) {
  const auto *source = reinterpret_cast<const T *>(addend.data);
  auto *target = reinterpret_cast<T *>(output.data);
  for (std::size_t i = 0; i < output.space.scalar_extent; ++i) {
    target[i] += source[i];
  }
}

void copy_composite_vector(const OperatorVectorView &input,
                           const OperatorVectorView &output) {
  if (output.space.scalar_type == PrimitiveType::f32) {
    copy_composite_vector<float32>(input, output);
  } else {
    copy_composite_vector<float64>(input, output);
  }
}

void scale_composite_vector(double scale,
                            const OperatorVectorView &output) {
  if (output.space.scalar_type == PrimitiveType::f32) {
    scale_composite_vector<float32>(scale, output);
  } else {
    scale_composite_vector<float64>(scale, output);
  }
}

void add_composite_vector(const OperatorVectorView &addend,
                          const OperatorVectorView &output) {
  if (output.space.scalar_type == PrimitiveType::f32) {
    add_composite_vector<float32>(addend, output);
  } else {
    add_composite_vector<float64>(addend, output);
  }
}

struct HostCompositeScratch {
  explicit HostCompositeScratch(OperatorSpaceDesc space)
      : space(std::move(space)),
        words((space_bytes(this->space) + sizeof(std::uint64_t) - 1) /
              sizeof(std::uint64_t)) {
  }

  OperatorVectorView view(Program *program) {
    if (program) {
      return OperatorVectorView::from_device_pointer(
          program, reinterpret_cast<std::uintptr_t>(words.data()), space,
          true);
    }
    return OperatorVectorView::from_mutable_host(words.data(), space);
  }

  OperatorSpaceDesc space;
  std::vector<std::uint64_t> words;
};

struct SumCompositeScratch {
  explicit SumCompositeScratch(const OperatorDescriptor &descriptor)
      : forward(descriptor.range), adjoint(descriptor.domain) {
  }

  std::mutex mutex;
  HostCompositeScratch forward;
  HostCompositeScratch adjoint;
};

struct ProductCompositeScratch {
  explicit ProductCompositeScratch(OperatorSpaceDesc intermediate)
      : intermediate(std::move(intermediate)) {
  }

  std::mutex mutex;
  HostCompositeScratch intermediate;
};

OperatorTraitClaim structurally_derived_claim(
    const OperatorTraitClaim &source) {
  if (!source.known()) {
    return {};
  }
  const auto provenance =
      source.provenance == OperatorTraitProvenance::empirically_checked
          ? OperatorTraitProvenance::empirically_checked
          : OperatorTraitProvenance::derived_structurally;
  return {source.value, provenance, source.validity_scope};
}

OperatorTraitClaim structurally_derived_true_claim(
    const OperatorTraitClaim &left,
    const OperatorTraitClaim &right) {
  if (!left.known() || !left.value || !right.known() || !right.value) {
    return {};
  }
  const auto provenance =
      left.provenance == OperatorTraitProvenance::empirically_checked ||
              right.provenance ==
                  OperatorTraitProvenance::empirically_checked
          ? OperatorTraitProvenance::empirically_checked
          : OperatorTraitProvenance::derived_structurally;
  return {true, provenance, left.validity_scope | right.validity_scope};
}

std::uint64_t combine_operator_revision(std::uint64_t seed,
                                        std::uint64_t revision) {
  seed ^= revision + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2);
  return seed == 0 ? 1 : seed;
}

OperatorResourceStamp combine_operator_generations(
    const std::vector<OperatorPinnedAction> &operands) {
  TI_ERROR_IF(operands.empty(),
              "Composite operator requires at least one operand.");
  const auto first = operands.front().resource_stamp();
  OperatorResourceStamp result{first.program_identity,
                               first.program_generation,
                               0xcbf29ce484222325ull,
                               0xcbf29ce484222325ull,
                               0xcbf29ce484222325ull,
                               0xcbf29ce484222325ull};
  for (const auto &operand : operands) {
    const auto stamp = operand.resource_stamp();
    TI_ERROR_IF(stamp.program_identity != result.program_identity ||
                    stamp.program_generation != result.program_generation,
                "Composite operator operands must belong to the same "
                "Program generation.");
    result.schema_revision =
        combine_operator_revision(result.schema_revision,
                                  stamp.schema_revision);
    result.topology_revision =
        combine_operator_revision(result.topology_revision,
                                  stamp.topology_revision);
    result.numeric_revision =
        combine_operator_revision(result.numeric_revision,
                                  stamp.numeric_revision);
    result.binding_revision =
        combine_operator_revision(result.binding_revision,
                                  stamp.binding_revision);
  }
  return result;
}

std::vector<OperatorPinnedAction> pin_composite_operands(
    const std::vector<OperatorBinding> &operands) {
  std::vector<OperatorPinnedAction> result;
  result.reserve(operands.size());
  for (const auto &operand : operands) {
    result.push_back(operand.pin());
  }
  return result;
}

OperatorAction make_composite_metadata_action(
    OperatorDescriptor descriptor,
    OperatorMathematicalTraits traits,
    OperatorCapabilities capabilities,
    std::string provider_name,
    OperatorAction::ResourceStampFn resource_stamp) {
  auto error_provider_name = provider_name;
  return OperatorAction(
      std::move(descriptor), std::move(traits), capabilities,
      std::move(provider_name), std::move(resource_stamp),
      [error_provider_name = std::move(error_provider_name)](
          OperatorApplyMode, const OperatorVectorView &,
          const OperatorVectorView &) {
        TI_ERROR("Composite operator metadata '{}' cannot be submitted "
                 "without pinning all operand generations.",
                 error_provider_name);
      });
}

OperatorMathematicalTraits scaled_operator_traits(
    double scale,
    const OperatorAction &operand) {
  OperatorMathematicalTraits traits;
  if (operand.descriptor().domain != operand.descriptor().range) {
    return traits;
  }
  if (scale == 0.0) {
    const auto scope =
        operator_dependency(OperatorResourceDependency::schema);
    traits.self_adjoint = {
        true, OperatorTraitProvenance::constructed_by_framework, scope};
    traits.positive_definite = {
        false, OperatorTraitProvenance::constructed_by_framework, scope};
    traits.positive_semidefinite = {
        true, OperatorTraitProvenance::constructed_by_framework, scope};
    traits.singular = {
        true, OperatorTraitProvenance::constructed_by_framework, scope};
    return traits;
  }
  const auto &source = operand.mathematical_traits();
  traits.self_adjoint = structurally_derived_claim(source.self_adjoint);
  traits.singular = structurally_derived_claim(source.singular);
  if (scale > 0.0) {
    traits.positive_definite =
        structurally_derived_claim(source.positive_definite);
    traits.positive_semidefinite =
        structurally_derived_claim(source.positive_semidefinite);
  }
  return traits;
}

OperatorMathematicalTraits sum_operator_traits(const OperatorAction &left,
                                               const OperatorAction &right) {
  OperatorMathematicalTraits traits;
  if (left.descriptor().domain != left.descriptor().range) {
    return traits;
  }
  const auto &left_traits = left.mathematical_traits();
  const auto &right_traits = right.mathematical_traits();
  traits.self_adjoint = structurally_derived_true_claim(
      left_traits.self_adjoint, right_traits.self_adjoint);
  traits.positive_semidefinite = structurally_derived_true_claim(
      left_traits.positive_semidefinite,
      right_traits.positive_semidefinite);
  auto positive_definite = structurally_derived_true_claim(
      left_traits.positive_definite,
      right_traits.positive_semidefinite);
  if (!positive_definite.known()) {
    positive_definite = structurally_derived_true_claim(
        left_traits.positive_semidefinite,
        right_traits.positive_definite);
  }
  if (!traits.self_adjoint.known() || !traits.self_adjoint.value) {
    positive_definite = {};
  }
  traits.positive_definite = positive_definite;
  if (positive_definite.known() && positive_definite.value) {
    traits.singular = {false, positive_definite.provenance,
                       positive_definite.validity_scope};
  }
  return traits;
}

}  // namespace

OperatorBinding make_identity_operator_binding(OperatorSpaceDesc space,
                                                Program *program) {
  validate_space(space, "identity");
  TI_ERROR_IF(program && !arch_is_cpu(program->compile_config().arch),
              "M3 identity composition supports host and CPU Programs only.");
  const OperatorDescriptor descriptor{space, space};
  const auto scope =
      operator_dependency(OperatorResourceDependency::schema);
  auto traits = make_spd_operator_traits(
      OperatorTraitProvenance::constructed_by_framework, scope);
  OperatorCapabilities capabilities;
  capabilities.adjoint_apply = true;
  auto action = OperatorAction(
      descriptor, traits, capabilities, "identity",
      [program] {
        return program
                   ? OperatorResourceStamp{
                         reinterpret_cast<std::uintptr_t>(program),
                         program->runtime_program_generation(), 1, 1, 1, 1}
                   : OperatorResourceStamp{};
      },
      [](OperatorApplyMode, const OperatorVectorView &input,
         const OperatorVectorView &output) {
        validate_host_composite_views(input, output);
        copy_composite_vector(input, output);
      });
  return OperatorBinding(std::move(action));
}

OperatorBinding make_scaled_operator_binding(double scale,
                                              OperatorBinding operand) {
  TI_ERROR_IF(!std::isfinite(scale),
              "Scaled operator requires a finite scalar.");
  const auto &source = operand.action();
  validate_synchronous_composite_operand(source, "scaled");
  const auto descriptor = source.descriptor();
  const auto traits = scaled_operator_traits(scale, source);
  auto capabilities = source.capabilities();
  capabilities.native_generalized_apply = false;
  capabilities.asynchronous_submit = false;
  const std::string provider_name =
      "scale(" + source.provider_name() + ")";
  auto metadata = make_composite_metadata_action(
      descriptor, traits, capabilities, provider_name,
      [operand] { return operand.pin().resource_stamp(); });
  return OperatorBinding::from_generation_publisher(
      std::move(metadata),
      [operand = std::move(operand), descriptor, traits, capabilities,
       provider_name, scale] {
        auto source_generation = operand.pin();
        const auto stamp = source_generation.resource_stamp();
        auto action = OperatorAction(
            descriptor, traits, capabilities, provider_name,
            [stamp] { return stamp; },
            [source_generation = std::move(source_generation), scale](
                OperatorApplyMode mode, const OperatorVectorView &input,
                const OperatorVectorView &output) {
              validate_host_composite_views(input, output);
              source_generation.apply_overwrite(mode, input, output);
              scale_composite_vector(scale, output);
            });
        return OperatorPinnedAction::from_retained_action(std::move(action),
                                                          stamp);
      });
}

namespace {

OperatorTraitClaim structurally_derived_all_true_claim(
    const std::vector<OperatorTraitClaim> &claims) {
  TI_ERROR_IF(claims.empty(),
              "Composite trait derivation requires at least one operand.");
  auto result = structurally_derived_claim(claims.front());
  if (!result.known() || !result.value) {
    return {};
  }
  for (std::size_t i = 1; i < claims.size(); ++i) {
    result = structurally_derived_true_claim(result, claims[i]);
    if (!result.known()) {
      return {};
    }
  }
  return result;
}

OperatorMathematicalTraits block_diagonal_operator_traits(
    const std::vector<OperatorBinding> &blocks,
    const OperatorDescriptor &descriptor) {
  OperatorMathematicalTraits traits;
  if (descriptor.domain != descriptor.range) {
    return traits;
  }
  std::vector<OperatorTraitClaim> self_adjoint;
  std::vector<OperatorTraitClaim> positive_definite;
  std::vector<OperatorTraitClaim> positive_semidefinite;
  self_adjoint.reserve(blocks.size());
  positive_definite.reserve(blocks.size());
  positive_semidefinite.reserve(blocks.size());
  for (const auto &block : blocks) {
    const auto &block_traits = block.action().mathematical_traits();
    self_adjoint.push_back(block_traits.self_adjoint);
    positive_definite.push_back(block_traits.positive_definite);
    positive_semidefinite.push_back(
        block_traits.positive_semidefinite);
  }
  traits.self_adjoint =
      structurally_derived_all_true_claim(self_adjoint);
  traits.positive_definite =
      structurally_derived_all_true_claim(positive_definite);
  traits.positive_semidefinite =
      structurally_derived_all_true_claim(positive_semidefinite);
  if (traits.positive_definite.known() &&
      traits.positive_definite.value) {
    traits.singular = {false, traits.positive_definite.provenance,
                       traits.positive_definite.validity_scope};
  }
  return traits;
}

bool same_composite_space_kind(const OperatorSpaceDesc &left,
                               const OperatorSpaceDesc &right) {
  return left.scalar_type == right.scalar_type &&
         left.entry_shape == right.entry_shape &&
         left.inner_product_kind == right.inner_product_kind;
}

OperatorVectorView composite_subview(const OperatorVectorView &view,
                                     const OperatorSpaceDesc &space,
                                     std::size_t scalar_offset) {
  auto result = as_raw_composite_view(view, view.writable);
  result.space = space;
  result.data += scalar_offset * data_type_size(space.scalar_type);
  result.allocation_identity = result.data;
  return result;
}

}  // namespace

OperatorBinding make_sum_operator_binding(OperatorBinding left,
                                           OperatorBinding right) {
  const auto &left_action = left.action();
  const auto &right_action = right.action();
  TI_ERROR_IF(left_action.descriptor().domain !=
                      right_action.descriptor().domain ||
                  left_action.descriptor().range !=
                      right_action.descriptor().range,
              "Operator sum requires identical operand descriptors.");
  validate_synchronous_composite_operand(left_action, "sum");
  validate_synchronous_composite_operand(right_action, "sum");

  const auto descriptor = left_action.descriptor();
  const auto traits = sum_operator_traits(left_action, right_action);
  OperatorCapabilities capabilities;
  capabilities.adjoint_apply =
      left_action.capabilities().adjoint_apply &&
      right_action.capabilities().adjoint_apply;
  const std::string provider_name =
      "sum(" + left_action.provider_name() + "," +
      right_action.provider_name() + ")";
  std::vector<OperatorBinding> operands;
  operands.push_back(std::move(left));
  operands.push_back(std::move(right));
  auto scratch = std::make_shared<SumCompositeScratch>(descriptor);
  auto metadata = make_composite_metadata_action(
      descriptor, traits, capabilities, provider_name, [operands] {
        return combine_operator_generations(
            pin_composite_operands(operands));
      });
  return OperatorBinding::from_generation_publisher(
      std::move(metadata),
      [operands = std::move(operands), scratch, descriptor, traits,
       capabilities, provider_name] {
        auto pins = pin_composite_operands(operands);
        const auto stamp = combine_operator_generations(pins);
        auto action = OperatorAction(
            descriptor, traits, capabilities, provider_name,
            [stamp] { return stamp; },
            [pins = std::move(pins), scratch](
                OperatorApplyMode mode, const OperatorVectorView &input,
                const OperatorVectorView &output) {
              validate_host_composite_views(input, output);
              std::lock_guard<std::mutex> lock(scratch->mutex);
              auto raw_input = as_raw_composite_view(input, false);
              auto raw_output = as_raw_composite_view(output, true);
              auto temporary =
                  (mode == OperatorApplyMode::forward
                       ? scratch->forward
                       : scratch->adjoint)
                      .view(input.program);
              pins[0].apply_overwrite(mode, raw_input, raw_output);
              pins[1].apply_overwrite(mode, raw_input, temporary);
              add_composite_vector(temporary, raw_output);
            });
        return OperatorPinnedAction::from_retained_action(std::move(action),
                                                          stamp);
      });
}

OperatorBinding make_composed_operator_binding(OperatorBinding outer,
                                                OperatorBinding inner) {
  const auto &outer_action = outer.action();
  const auto &inner_action = inner.action();
  TI_ERROR_IF(inner_action.descriptor().range !=
                  outer_action.descriptor().domain,
              "Operator composition requires the inner range to equal the "
              "outer domain.");
  validate_synchronous_composite_operand(outer_action, "product");
  validate_synchronous_composite_operand(inner_action, "product");

  const OperatorDescriptor descriptor{inner_action.descriptor().domain,
                                      outer_action.descriptor().range};
  const auto intermediate_space = inner_action.descriptor().range;
  const OperatorMathematicalTraits traits;
  OperatorCapabilities capabilities;
  capabilities.adjoint_apply =
      outer_action.capabilities().adjoint_apply &&
      inner_action.capabilities().adjoint_apply;
  const std::string provider_name =
      "compose(" + outer_action.provider_name() + "," +
      inner_action.provider_name() + ")";
  std::vector<OperatorBinding> operands;
  operands.push_back(std::move(outer));
  operands.push_back(std::move(inner));
  auto scratch = std::make_shared<ProductCompositeScratch>(
      intermediate_space);
  auto metadata = make_composite_metadata_action(
      descriptor, traits, capabilities, provider_name, [operands] {
        return combine_operator_generations(
            pin_composite_operands(operands));
      });
  return OperatorBinding::from_generation_publisher(
      std::move(metadata),
      [operands = std::move(operands), scratch, descriptor, traits,
       capabilities, provider_name] {
        auto pins = pin_composite_operands(operands);
        const auto stamp = combine_operator_generations(pins);
        auto action = OperatorAction(
            descriptor, traits, capabilities, provider_name,
            [stamp] { return stamp; },
            [pins = std::move(pins), scratch](
                OperatorApplyMode mode, const OperatorVectorView &input,
                const OperatorVectorView &output) {
              validate_host_composite_views(input, output);
              std::lock_guard<std::mutex> lock(scratch->mutex);
              auto raw_input = as_raw_composite_view(input, false);
              auto raw_output = as_raw_composite_view(output, true);
              auto temporary = scratch->intermediate.view(input.program);
              if (mode == OperatorApplyMode::forward) {
                pins[1].apply_overwrite(mode, raw_input, temporary);
                pins[0].apply_overwrite(mode, temporary, raw_output);
              } else {
                pins[0].apply_overwrite(mode, raw_input, temporary);
                pins[1].apply_overwrite(mode, temporary, raw_output);
              }
            });
        return OperatorPinnedAction::from_retained_action(std::move(action),
                                                          stamp);
      });
}

OperatorBinding make_block_diagonal_operator_binding(
    std::vector<OperatorBinding> blocks) {
  TI_ERROR_IF(blocks.empty(),
              "Block-diagonal operator requires at least one block.");
  const auto first_descriptor = blocks.front().action().descriptor();
  OperatorDescriptor descriptor = first_descriptor;
  descriptor.domain.scalar_extent = 0;
  descriptor.range.scalar_extent = 0;
  bool adjoint_apply = true;
  for (const auto &block : blocks) {
    const auto &action = block.action();
    validate_synchronous_composite_operand(action, "block-diagonal");
    TI_ERROR_IF(!same_composite_space_kind(
                    action.descriptor().domain, first_descriptor.domain) ||
                    !same_composite_space_kind(
                        action.descriptor().range, first_descriptor.range),
                "Block-diagonal operator requires compatible scalar spaces.");
    TI_ERROR_IF(
        descriptor.domain.scalar_extent >
                (std::numeric_limits<std::size_t>::max)() -
                    action.descriptor().domain.scalar_extent ||
            descriptor.range.scalar_extent >
                (std::numeric_limits<std::size_t>::max)() -
                    action.descriptor().range.scalar_extent,
        "Block-diagonal operator scalar extent overflow.");
    descriptor.domain.scalar_extent +=
        action.descriptor().domain.scalar_extent;
    descriptor.range.scalar_extent +=
        action.descriptor().range.scalar_extent;
    adjoint_apply =
        adjoint_apply && action.capabilities().adjoint_apply;
  }
  const auto traits =
      block_diagonal_operator_traits(blocks, descriptor);
  OperatorCapabilities capabilities;
  capabilities.adjoint_apply = adjoint_apply;
  const std::string provider_name = "block_diagonal";
  auto metadata = make_composite_metadata_action(
      descriptor, traits, capabilities, provider_name, [blocks] {
        return combine_operator_generations(
            pin_composite_operands(blocks));
      });
  return OperatorBinding::from_generation_publisher(
      std::move(metadata),
      [blocks = std::move(blocks), descriptor, traits, capabilities,
       provider_name] {
        auto pins = pin_composite_operands(blocks);
        const auto stamp = combine_operator_generations(pins);
        auto action = OperatorAction(
            descriptor, traits, capabilities, provider_name,
            [stamp] { return stamp; },
            [pins = std::move(pins)](
                OperatorApplyMode mode, const OperatorVectorView &input,
                const OperatorVectorView &output) {
              validate_host_composite_views(input, output);
              std::size_t input_offset = 0;
              std::size_t output_offset = 0;
              for (const auto &pin : pins) {
                const auto &block_descriptor = pin.descriptor();
                const auto &block_input =
                    input_space(block_descriptor, mode);
                const auto &block_output =
                    output_space(block_descriptor, mode);
                auto input_view =
                    composite_subview(input, block_input, input_offset);
                auto output_view =
                    composite_subview(output, block_output, output_offset);
                pin.apply_overwrite(mode, input_view, output_view);
                input_offset += block_input.scalar_extent;
                output_offset += block_output.scalar_extent;
              }
            });
        return OperatorPinnedAction::from_retained_action(std::move(action),
                                                          stamp);
      });
}

namespace {

template <typename Provider>
OperatorBinding make_cpu_typed_operator_binding(Program *program,
                                                Provider &provider,
                                                const char *expected_provider,
                                                const char *expected_storage) {
  TI_ERROR_IF(!program || !arch_is_cpu(program->compile_config().arch),
              "CPU operator bindings require an active CPU Program.");
  const auto initial = provider.debug_runtime_statistics();
  TI_ERROR_IF(initial.backend_family != "cpu" ||
                  initial.provider_name != expected_provider ||
                  initial.storage_format != expected_storage,
              "CPU operator binding expected provider '{}' with storage "
              "'{}', got provider '{}' on backend '{}' with storage '{}'; "
              "no fallback was performed.",
              expected_provider, expected_storage, initial.provider_name,
              initial.backend_family, initial.storage_format);
  OperatorDescriptor descriptor;
  descriptor.domain = {provider.get_data_type(),
                       static_cast<std::size_t>(provider.num_cols())};
  descriptor.range = {provider.get_data_type(),
                      static_cast<std::size_t>(provider.num_rows())};
  auto action = OperatorAction(
      descriptor, OperatorCapabilities{}, expected_provider,
      [program, &provider] {
        const auto statistics = provider.debug_runtime_statistics();
        return OperatorResourceStamp{reinterpret_cast<std::uintptr_t>(program),
                                     program->runtime_program_generation(),
                                     1,
                                     statistics.pattern_version,
                                     statistics.numeric_version,
                                     provider.matrix_id()};
      },
      [program, &provider](OperatorApplyMode mode,
                           const OperatorVectorView &input,
                           const OperatorVectorView &output) {
        TI_ERROR_IF(mode != OperatorApplyMode::forward,
                    "CPU sparse operator bindings support forward apply "
                    "only.");
        if (input.ndarray && output.ndarray) {
          provider.nd_spmv(program, *input.ndarray, *output.ndarray);
          return;
        }
        TI_ERROR_IF(input.ndarray || output.ndarray,
                    "CPU sparse operator input/output views must both use "
                    "ndarrays or both use raw Program pointers.");
        provider.spmv_cpu_raw(program, input.data, output.data);
      });
  return OperatorBinding(std::move(action), [&provider] {
    return OperatorResourceLease::hold(provider.acquire_numeric_access_guard());
  });
}

template <typename Provider, typename Apply>
OperatorBinding make_gpu_typed_operator_binding(Program *program,
                                                Provider &provider,
                                                Arch expected_arch,
                                                const char *expected_backend,
                                                const char *expected_provider,
                                                const char *expected_storage,
                                                Apply apply) {
  TI_ERROR_IF(!program || program->compile_config().arch != expected_arch,
              "{} operator bindings require their owning {} Program.",
              expected_backend, expected_backend);
  const auto initial = provider.debug_runtime_statistics();
  TI_ERROR_IF(initial.backend_family != expected_backend ||
                  initial.provider_name != expected_provider ||
                  initial.storage_format != expected_storage,
              "{} operator binding expected provider '{}' with storage "
              "'{}', got provider '{}' on backend '{}' with storage '{}'; "
              "no fallback was performed.",
              expected_backend, expected_provider, expected_storage,
              initial.provider_name, initial.backend_family,
              initial.storage_format);
  OperatorDescriptor descriptor;
  descriptor.domain = {provider.get_data_type(),
                       static_cast<std::size_t>(provider.num_cols())};
  descriptor.range = {provider.get_data_type(),
                      static_cast<std::size_t>(provider.num_rows())};
  OperatorCapabilities capabilities;
  capabilities.asynchronous_submit = true;
  auto action = OperatorAction(
      descriptor, capabilities, expected_provider,
      [program, &provider] {
        const auto statistics = provider.debug_runtime_statistics();
        return OperatorResourceStamp{reinterpret_cast<std::uintptr_t>(program),
                                     program->runtime_program_generation(),
                                     1,
                                     statistics.pattern_version,
                                     statistics.numeric_version,
                                     provider.matrix_id()};
      },
      [apply = std::move(apply)](OperatorApplyMode mode,
                                 const OperatorVectorView &input,
                                 const OperatorVectorView &output) {
        TI_ERROR_IF(mode != OperatorApplyMode::forward,
                    "GPU sparse operator bindings support forward apply "
                    "only.");
        apply(input, output);
      });
  return OperatorBinding(std::move(action), [&provider] {
    return OperatorResourceLease::hold(provider.acquire_numeric_access_guard());
  });
}

}  // namespace

OperatorBinding make_cpu_csr_operator_binding(Program *program,
                                              CpuSparseCsrMatrix &matrix) {
  return make_cpu_typed_operator_binding(program, matrix, "forge_cpu_native",
                                         "csr");
}

OperatorBinding make_cpu_bsr_operator_binding(Program *program,
                                              CpuSparseBsrMatrix &matrix) {
  return make_cpu_typed_operator_binding(program, matrix, "forge_cpu_native",
                                         "bsr");
}

OperatorBinding make_cpu_fixed_sparse_operator_binding(
    Program *program,
    SparseMatrix &matrix) {
  if (auto *csr = dynamic_cast<CpuSparseCsrMatrix *>(&matrix)) {
    return make_cpu_csr_operator_binding(program, *csr);
  }
  if (auto *bsr = dynamic_cast<CpuSparseBsrMatrix *>(&matrix)) {
    return make_cpu_bsr_operator_binding(program, *bsr);
  }
  const auto statistics = matrix.debug_runtime_statistics();
  TI_ERROR(
      "CPU fixed sparse operator binding does not support backend '{}' with "
      "storage '{}' (provider '{}'); no fallback was performed.",
      statistics.backend_family, statistics.storage_format,
      statistics.provider_name);
}

OperatorBinding make_cpu_program_kernel_operator_binding(
    Program *program,
    CompiledKernelLinearOperator &matrix) {
  TI_ERROR_IF(matrix.owning_program() != program,
              "CPU program-kernel operator binding requires its owning "
              "Program; no fallback was performed.");
  return matrix.make_operator_binding();
}

OperatorBinding make_cpu_program_graph_operator_binding(
    Program *program,
    CompiledGraphLinearOperator &matrix) {
  TI_ERROR_IF(!program || !arch_is_cpu(program->compile_config().arch) ||
                  matrix.owning_program() != program,
              "CPU program-graph binding requires its owning CPU Program.");
  return matrix.make_operator_binding(
      OperatorExecutionKind::explicit_sequence);
}

OperatorBinding make_cuda_csr_operator_binding(Program *program,
                                               CuSparseMatrix &matrix) {
  return make_gpu_typed_operator_binding(
      program, matrix, Arch::cuda, "cuda", "cusparse", "csr",
      [&matrix](const OperatorVectorView &input,
                const OperatorVectorView &output) {
        matrix.spmv(input.data, output.data);
      });
}

OperatorBinding make_cuda_bsr_operator_binding(Program *program,
                                               CuSparseBsrMatrix &matrix) {
  return make_gpu_typed_operator_binding(
      program, matrix, Arch::cuda, "cuda", "cusparse", "bsr",
      [&matrix](const OperatorVectorView &input,
                const OperatorVectorView &output) {
        matrix.spmv(input.data, output.data);
      });
}

OperatorBinding make_cuda_program_kernel_operator_binding(
    Program *program,
    CompiledKernelLinearOperator &matrix) {
  TI_ERROR_IF(!program || program->compile_config().arch != Arch::cuda ||
                  matrix.owning_program() != program,
              "CUDA program-kernel binding requires its owning CUDA "
              "Program.");
  return matrix.make_operator_binding();
}

OperatorBinding make_cuda_program_graph_operator_binding(
    Program *program,
    CompiledGraphLinearOperator &matrix,
    OperatorExecutionKind execution_kind) {
  TI_ERROR_IF(!program || program->compile_config().arch != Arch::cuda ||
                  matrix.owning_program() != program,
              "CUDA program-graph binding requires its owning CUDA Program.");
  TI_ERROR_IF(execution_kind != OperatorExecutionKind::compiled_graph &&
                  execution_kind != OperatorExecutionKind::runtime_capture,
              "CUDA program-graph bindings require compiled_graph or "
              "runtime_capture execution; got '{}'.",
              operator_execution_kind_name(execution_kind));
  return matrix.make_operator_binding(execution_kind);
}

OperatorBinding make_vulkan_csr_operator_binding(Program *program,
                                                 VulkanSparseMatrix &matrix) {
  return make_gpu_typed_operator_binding(
      program, matrix, Arch::vulkan, "vulkan", "forge_vulkan_native", "csr",
      [program, &matrix](const OperatorVectorView &input,
                         const OperatorVectorView &output) {
        TI_ERROR_IF(!input.ndarray || !output.ndarray,
                    "Vulkan CSR operator binding requires ndarray views.");
        matrix.nd_spmv(program, *input.ndarray, *output.ndarray);
      });
}

OperatorBinding make_vulkan_bsr_operator_binding(
    Program *program,
    VulkanSparseBsrMatrix &matrix) {
  return make_gpu_typed_operator_binding(
      program, matrix, Arch::vulkan, "vulkan", "forge_vulkan_native", "bsr",
      [program, &matrix](const OperatorVectorView &input,
                         const OperatorVectorView &output) {
        TI_ERROR_IF(!input.ndarray || !output.ndarray,
                    "Vulkan BSR operator binding requires ndarray views.");
        matrix.nd_spmv(program, *input.ndarray, *output.ndarray);
      });
}

OperatorBinding make_vulkan_program_kernel_operator_binding(
    Program *program,
    CompiledKernelLinearOperator &matrix) {
  TI_ERROR_IF(!program || program->compile_config().arch != Arch::vulkan ||
                  matrix.owning_program() != program,
              "Vulkan program-kernel binding requires its owning Vulkan "
              "Program.");
  return matrix.make_operator_binding();
}

OperatorBinding make_vulkan_program_graph_operator_binding(
    Program *program,
    CompiledGraphLinearOperator &matrix) {
  TI_ERROR_IF(!program || program->compile_config().arch != Arch::vulkan ||
                  matrix.owning_program() != program,
              "Vulkan program-graph binding requires its owning Vulkan "
              "Program.");
  return matrix.make_operator_binding(
      OperatorExecutionKind::compiled_graph);
}

ExperimentalLinearOperatorHandle::ExperimentalLinearOperatorHandle(
    Program *program,
    OperatorBinding binding)
    : program_(program),
      binding_(std::move(binding)),
      plan_(std::make_unique<OperatorPlan>(program_, binding_)) {
  TI_ERROR_IF(!program_,
              "Experimental LinearOperator requires an active Program.");
  const auto stamp = plan_->resource_stamp();
  TI_ERROR_IF(stamp.program_identity !=
                  reinterpret_cast<std::uintptr_t>(program_),
              "LinearOperator provider belongs to a different Program; "
              "no cross-runtime binding was performed.");
}

ExperimentalLinearOperatorHandle::~ExperimentalLinearOperatorHandle() =
    default;

Program *ExperimentalLinearOperatorHandle::program() const {
  return program_;
}

const OperatorDescriptor &
ExperimentalLinearOperatorHandle::descriptor() const {
  return plan_->descriptor();
}

const OperatorMathematicalTraits &
ExperimentalLinearOperatorHandle::mathematical_traits() const {
  return plan_->mathematical_traits();
}

const OperatorCapabilities &
ExperimentalLinearOperatorHandle::capabilities() const {
  return plan_->capabilities();
}

const std::string &ExperimentalLinearOperatorHandle::provider_name() const {
  return plan_->provider_name();
}

OperatorExecutionKind
ExperimentalLinearOperatorHandle::execution_kind() const {
  return plan_->execution_kind();
}

OperatorResourceStamp
ExperimentalLinearOperatorHandle::resource_stamp() const {
  return plan_->resource_stamp();
}

OperatorPlanRuntimeStatistics
ExperimentalLinearOperatorHandle::debug_runtime_statistics() const {
  return plan_->debug_runtime_statistics();
}

OperatorBinding ExperimentalLinearOperatorHandle::binding() const {
  return binding_;
}

void ExperimentalLinearOperatorHandle::apply(Program *program,
                                             const Ndarray &input,
                                             const Ndarray &output) {
  TI_ERROR_IF(program != program_,
              "LinearOperator apply must use its construction Program.");
  const auto &descriptor = plan_->descriptor();
  auto submission = plan_->submit(
      {OperatorApplyMode::forward,
       OperatorVectorView::from_ndarray(program_, input,
                                        descriptor.domain, false),
       nullptr,
       OperatorVectorView::from_ndarray(program_, output,
                                        descriptor.range, true)});
  submission.wait();
}

OperatorBinding make_program_sparse_operator_binding(Program *program,
                                                      SparseMatrix &matrix) {
  TI_ERROR_IF(!program,
              "LinearOperator provider binding requires an active Program.");
  if (auto *kernel =
          dynamic_cast<CompiledKernelLinearOperator *>(&matrix)) {
    if (arch_is_cpu(program->compile_config().arch)) {
      return make_cpu_program_kernel_operator_binding(program, *kernel);
    }
    if (program->compile_config().arch == Arch::cuda) {
      return make_cuda_program_kernel_operator_binding(program, *kernel);
    }
    if (program->compile_config().arch == Arch::vulkan) {
      return make_vulkan_program_kernel_operator_binding(program, *kernel);
    }
  }
  if (auto *graph =
          dynamic_cast<CompiledGraphLinearOperator *>(&matrix)) {
    if (arch_is_cpu(program->compile_config().arch)) {
      return make_cpu_program_graph_operator_binding(program, *graph);
    }
    if (program->compile_config().arch == Arch::cuda) {
      return make_cuda_program_graph_operator_binding(program, *graph);
    }
    if (program->compile_config().arch == Arch::vulkan) {
      return make_vulkan_program_graph_operator_binding(program, *graph);
    }
  }
  if (arch_is_cpu(program->compile_config().arch)) {
    return make_cpu_fixed_sparse_operator_binding(program, matrix);
  }
  if (program->compile_config().arch == Arch::cuda) {
    if (auto *csr = dynamic_cast<CuSparseMatrix *>(&matrix)) {
      return make_cuda_csr_operator_binding(program, *csr);
    }
    if (auto *bsr = dynamic_cast<CuSparseBsrMatrix *>(&matrix)) {
      return make_cuda_bsr_operator_binding(program, *bsr);
    }
  }
  if (program->compile_config().arch == Arch::vulkan) {
    if (auto *csr = dynamic_cast<VulkanSparseMatrix *>(&matrix)) {
      return make_vulkan_csr_operator_binding(program, *csr);
    }
    if (auto *bsr = dynamic_cast<VulkanSparseBsrMatrix *>(&matrix)) {
      return make_vulkan_bsr_operator_binding(program, *bsr);
    }
  }
  const auto statistics = matrix.debug_runtime_statistics();
  TI_ERROR(
      "LinearOperator does not support backend '{}' with storage '{}' "
      "(provider '{}'); no fallback or materialization was performed.",
      statistics.backend_family, statistics.storage_format,
      statistics.provider_name);
}

OperatorMathematicalTraits make_asserted_operator_traits(
    int self_adjoint,
    int positive_definite,
    int positive_semidefinite,
    int singular) {
  const auto scope =
      operator_dependency(OperatorResourceDependency::program) |
      operator_dependency(OperatorResourceDependency::schema) |
      operator_dependency(OperatorResourceDependency::topology) |
      operator_dependency(OperatorResourceDependency::numeric) |
      operator_dependency(OperatorResourceDependency::binding);
  const auto claim = [scope](int value, const char *name) {
    TI_ERROR_IF(value < -1 || value > 1,
                "LinearOperator trait '{}' must be -1, 0, or 1.", name);
    return value < 0
               ? OperatorTraitClaim{}
               : OperatorTraitClaim{
                     value != 0,
                     OperatorTraitProvenance::asserted_by_user, scope};
  };
  OperatorMathematicalTraits result;
  result.self_adjoint = claim(self_adjoint, "self_adjoint");
  result.positive_definite =
      claim(positive_definite, "positive_definite");
  result.positive_semidefinite =
      claim(positive_semidefinite, "positive_semidefinite");
  result.singular = claim(singular, "singular");
  TI_ERROR_IF(result.positive_definite.known() &&
                  result.positive_definite.value &&
                  result.self_adjoint.known() &&
                  !result.self_adjoint.value,
              "A positive-definite LinearOperator cannot be declared "
              "non-self-adjoint.");
  TI_ERROR_IF(result.positive_definite.known() &&
                  result.positive_definite.value &&
                  result.singular.known() && result.singular.value,
              "A positive-definite LinearOperator cannot be declared "
              "singular.");
  return result;
}

std::unique_ptr<ExperimentalLinearOperatorHandle>
make_experimental_linear_operator_handle(
    Program *program,
    SparseMatrix &matrix,
    OperatorMathematicalTraits mathematical_traits) {
  auto binding = make_program_sparse_operator_binding(program, matrix)
                     .with_mathematical_traits(
                         std::move(mathematical_traits));
  return std::make_unique<ExperimentalLinearOperatorHandle>(
      program, std::move(binding));
}

std::unique_ptr<ExperimentalLinearOperatorHandle>
make_experimental_identity_operator_handle(Program *program,
                                           OperatorSpaceDesc space) {
  return std::make_unique<ExperimentalLinearOperatorHandle>(
      program, make_identity_operator_binding(std::move(space), program));
}

std::unique_ptr<ExperimentalLinearOperatorHandle>
make_experimental_adjoint_operator_handle(
    ExperimentalLinearOperatorHandle &operand) {
  return std::make_unique<ExperimentalLinearOperatorHandle>(
      operand.program(), make_adjoint_operator_binding(operand.binding()));
}

std::unique_ptr<ExperimentalLinearOperatorHandle>
make_experimental_scaled_operator_handle(
    double scale,
    ExperimentalLinearOperatorHandle &operand) {
  return std::make_unique<ExperimentalLinearOperatorHandle>(
      operand.program(),
      make_scaled_operator_binding(scale, operand.binding()));
}

namespace {

void validate_same_public_operator_program(
    const ExperimentalLinearOperatorHandle &left,
    const ExperimentalLinearOperatorHandle &right,
    const char *operation) {
  TI_ERROR_IF(left.program() != right.program(),
              "LinearOperator {} operands must belong to the same Program.",
              operation);
}

}  // namespace

std::unique_ptr<ExperimentalLinearOperatorHandle>
make_experimental_sum_operator_handle(
    ExperimentalLinearOperatorHandle &left,
    ExperimentalLinearOperatorHandle &right) {
  validate_same_public_operator_program(left, right, "sum");
  return std::make_unique<ExperimentalLinearOperatorHandle>(
      left.program(),
      make_sum_operator_binding(left.binding(), right.binding()));
}

std::unique_ptr<ExperimentalLinearOperatorHandle>
make_experimental_composed_operator_handle(
    ExperimentalLinearOperatorHandle &outer,
    ExperimentalLinearOperatorHandle &inner) {
  validate_same_public_operator_program(outer, inner, "composition");
  return std::make_unique<ExperimentalLinearOperatorHandle>(
      outer.program(),
      make_composed_operator_binding(outer.binding(), inner.binding()));
}

std::unique_ptr<ExperimentalLinearOperatorHandle>
make_experimental_block_diagonal_operator_handle(
    const std::vector<ExperimentalLinearOperatorHandle *> &blocks) {
  TI_ERROR_IF(blocks.empty(),
              "LinearOperator block_diagonal requires at least one block.");
  TI_ERROR_IF(!blocks.front(),
              "LinearOperator block_diagonal received a null block.");
  Program *program = blocks.front()->program();
  std::vector<OperatorBinding> bindings;
  bindings.reserve(blocks.size());
  for (const auto *block : blocks) {
    TI_ERROR_IF(!block || block->program() != program,
                "LinearOperator block_diagonal operands must belong to the "
                "same Program.");
    bindings.push_back(block->binding());
  }
  return std::make_unique<ExperimentalLinearOperatorHandle>(
      program, make_block_diagonal_operator_binding(std::move(bindings)));
}

}  // namespace taichi::lang
