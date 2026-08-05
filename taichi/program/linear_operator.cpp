#include "taichi/program/linear_operator.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <limits>
#include <mutex>
#include <utility>

#include "taichi/analysis/gather_snode_tree_dependencies.h"
#include "taichi/aot/graph_data.h"
#include "taichi/common/core.h"
#include "taichi/ir/type_factory.h"
#include "taichi/program/kernel.h"
#include "taichi/program/ndarray.h"
#include "taichi/program/program.h"
#include "taichi/program/runtime_resource_registry.h"
#include "taichi/program/sparse_matrix.h"
#include "taichi/program/storage_view.h"

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
  TI_ERROR_IF(view.space != expected ||
                  (view.data == 0 && !view.ndarray && !view.dense_storage) ||
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
    TI_ERROR_IF(view.program || view.ndarray || view.dense_storage,
                "Host-reference operator {} view must not carry Program "
                "state.",
                role);
  }
}

bool operator_views_overlap(const OperatorVectorView &lhs,
                            const OperatorVectorView &rhs) {
  if (lhs.resolved_dense_storage && rhs.resolved_dense_storage) {
    const auto &left = *lhs.resolved_dense_storage;
    const auto &right = *rhs.resolved_dense_storage;
    if (left.allocation != right.allocation) {
      return false;
    }
    const std::uint64_t left_end = left.byte_offset + left.byte_size;
    const std::uint64_t right_end = right.byte_offset + right.byte_size;
    return left.byte_offset < right_end && right.byte_offset < left_end;
  }
  if (lhs.dense_storage && rhs.dense_storage) {
    return storage::analyze_logical_storage_alias(*lhs.dense_storage,
                                                  *rhs.dense_storage) !=
           storage::StorageAliasRelation::kProvenDisjoint;
  }
  return lhs.allocation_identity == rhs.allocation_identity;
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
  if (family == OperatorSolverFamily::bicgstab ||
      family == OperatorSolverFamily::gmres) {
    return;
  }
  if (family == OperatorSolverFamily::minres) {
    TI_ERROR_IF(!trait_is_trusted_true(traits.self_adjoint),
                "MINRES requires a trusted self-adjoint trait; unknown or "
                "empirically-checked claims are insufficient.");
    TI_ERROR_IF(traits.singular.known() && traits.singular.value,
                "MINRES does not provide singular minimum-length semantics "
                "and rejects operators declared singular.");
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

OperatorVectorView OperatorVectorView::from_dense_storage(
    Program *program,
    const storage::RuntimeStorageArgument &argument,
    const storage::ResolvedDenseBinding &binding,
    const OperatorSpaceDesc &space,
    bool writable) {
  const auto &descriptor = argument.descriptor();
  validate_space(space, "dense storage");
  TI_ERROR_IF(!program || !binding.valid ||
                  descriptor.scalar_type() != space.scalar_type ||
                  descriptor.properties().scalar_count !=
                      space.scalar_extent,
              "Operator dense storage must match Program, dtype, and scalar "
              "extent.");
  const auto address = static_cast<std::uintptr_t>(
      program->get_dense_storage_data_ptr_as_int(binding));
  std::uintptr_t identity = address;
  if (identity == 0) {
    identity = reinterpret_cast<std::uintptr_t>(binding.allocation.device) ^
               static_cast<std::uintptr_t>(binding.allocation.alloc_id) ^
               static_cast<std::uintptr_t>(binding.byte_offset) ^
               static_cast<std::uintptr_t>(descriptor.fingerprint());
    if (identity == 0) {
      identity = 1;
    }
  }
  OperatorVectorView result{
      space, address, identity, nullptr, program, writable};
  result.dense_storage = &descriptor;
  result.runtime_storage = &argument;
  result.resolved_dense_storage = &binding;
  result.allocation_device_identity = binding.allocation.device;
  result.allocation_id = binding.allocation.alloc_id;
  result.byte_begin = binding.byte_offset;
  result.byte_end = binding.byte_offset + binding.byte_size;
  return result;
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
  TI_ERROR_IF(behavior_ == PreconditionerBehavior::nonlinear,
              "PreconditionerPlan does not execute nonlinear actions.");
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
      operator_views_overlap(request.input, request.output),
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

void validate_composite_operand(const OperatorAction &operand,
                                const char *role,
                                Program *program) {
  const bool gpu_composition =
      program && !arch_is_cpu(program->compile_config().arch);
  TI_ERROR_IF(!gpu_composition &&
                  operand.capabilities().asynchronous_submit,
              "{} operator composition requires a synchronous operand on "
              "host and CPU.",
              role);
  TI_ERROR_IF(gpu_composition &&
                  program->compile_config().arch != Arch::cuda &&
                  program->compile_config().arch != Arch::vulkan,
              "{} operator composition supports only CPU, CUDA, and Vulkan.",
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

void validate_device_composite_views(const OperatorVectorView &input,
                                     const OperatorVectorView &output,
                                     Program *program) {
  TI_ERROR_IF(!program || input.program != program || output.program != program,
              "GPU composite operator views must belong to their Program.");
  TI_ERROR_IF(!input.ndarray || !output.ndarray,
              "GPU composite operator lowering requires scalar ndarrays; "
              "qualified Field operands are staged on device.");
  TI_ERROR_IF(input.space.scalar_type != PrimitiveType::f32 ||
                  output.space.scalar_type != PrimitiveType::f32,
              "GPU composite operator lowering currently requires f32.");
}

void transform_composite_ndarray(Program *program,
                                 Ndarray *values,
                                 double scale) {
  TI_ASSERT(program && values);
  const Arch arch = program->compile_config().arch;
  if (arch == Arch::cuda) {
    TI_ERROR_IF(!program->cuda_device_transform_available(),
                "CUDA composite scaling requires native device transform.");
    program->cuda_device_transform_affine_ndarray(values, values, 1, scale,
                                                   0.0);
    return;
  }
  TI_ERROR_IF(arch != Arch::vulkan || !program->vulkan_transform_available() ||
                  !program->vulkan_transform_value_type_available(1),
              "Vulkan composite scaling requires native f32 transform.");
  program->vulkan_transform_affine_ndarray_trusted(values, values, 1, scale,
                                                    0.0);
}

void add_composite_ndarray(Program *program,
                           Ndarray *addend,
                           Ndarray *output) {
  TI_ASSERT(program && addend && output);
  const Arch arch = program->compile_config().arch;
  if (arch == Arch::cuda) {
    TI_ERROR_IF(!program->cuda_device_add_merge_available(),
                "CUDA operator sum requires native device add-merge.");
    program->cuda_device_add_merge_ndarray(addend, output, 1);
    return;
  }
  TI_ERROR_IF(arch != Arch::vulkan || !program->vulkan_add_merge_available() ||
                  !program->vulkan_add_merge_value_type_available(1),
              "Vulkan operator sum requires native f32 add-merge.");
  program->vulkan_add_merge_ndarray(addend, output, 1);
}

OperatorVectorView as_raw_composite_view(const OperatorVectorView &view,
                                         bool writable) {
  auto result = view;
  result.allocation_identity = view.data;
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

struct CompositeScratch {
  CompositeScratch(OperatorSpaceDesc space, Program *program)
      : host(std::move(space)),
        program(program),
        program_lifetime(program ? program->weak_resource_lifetime_token()
                                 : std::weak_ptr<ProgramLifetimeToken>{}) {
    if (program) {
      TI_ERROR_IF(
          (!arch_is_cpu(program->compile_config().arch) &&
           host.space.scalar_type != PrimitiveType::f32) ||
              host.space.scalar_extent > static_cast<std::size_t>(
                                             (std::numeric_limits<int>::max)()),
          "Program-backed composite scratch requires an extent no larger "
          "than INT_MAX and f32 storage on GPU.");
      array = program->create_ndarray(
          host.space.scalar_type,
          {static_cast<int>(host.space.scalar_extent)},
          ExternalArrayLayout::kNull, false);
    }
  }

  ~CompositeScratch() {
    Program::delete_ndarray_if_alive(program, program_lifetime, array);
  }

  OperatorVectorView view(Program *active_program) {
    if (array) {
      TI_ERROR_IF(active_program != program,
                  "Composite scratch belongs to another Program.");
      return OperatorVectorView::from_ndarray(program, *array, host.space,
                                              true);
    }
    TI_ERROR_IF(active_program,
                "Program-backed composite scratch requires an ndarray "
                "allocation.");
    return host.view(active_program);
  }

  Ndarray *ndarray() const {
    return array;
  }

  HostCompositeScratch host;
  Program *program{nullptr};
  std::weak_ptr<ProgramLifetimeToken> program_lifetime;
  Ndarray *array{nullptr};
};

struct SumCompositeScratch {
  SumCompositeScratch(const OperatorDescriptor &descriptor, Program *program)
      : forward(descriptor.range, program), adjoint(descriptor.domain, program) {
  }

  std::mutex mutex;
  CompositeScratch forward;
  CompositeScratch adjoint;
};

struct ProductCompositeScratch {
  ProductCompositeScratch(OperatorSpaceDesc intermediate, Program *program)
      : intermediate(std::move(intermediate), program) {
  }

  std::mutex mutex;
  CompositeScratch intermediate;
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
  capabilities.dense_storage_operands = program != nullptr;
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
                                              OperatorBinding operand,
                                              Program *program) {
  TI_ERROR_IF(!std::isfinite(scale),
              "Scaled operator requires a finite scalar.");
  const auto &source = operand.action();
  validate_composite_operand(source, "scaled", program);
  const auto descriptor = source.descriptor();
  const auto traits = scaled_operator_traits(scale, source);
  auto capabilities = source.capabilities();
  capabilities.native_generalized_apply = false;
  capabilities.asynchronous_submit =
      program && !arch_is_cpu(program->compile_config().arch);
  capabilities.explicit_sequence = capabilities.asynchronous_submit;
  capabilities.dense_storage_operands = false;
  capabilities.dense_storage_affine_operands = false;
  const std::string provider_name =
      "scale(" + source.provider_name() + ")";
  auto metadata = make_composite_metadata_action(
      descriptor, traits, capabilities, provider_name,
      [operand] { return operand.pin().resource_stamp(); });
  return OperatorBinding::from_generation_publisher(
      std::move(metadata),
      [operand = std::move(operand), descriptor, traits, capabilities,
       provider_name, scale, program] {
        auto source_generation = operand.pin();
        const auto stamp = source_generation.resource_stamp();
        auto action = OperatorAction(
            descriptor, traits, capabilities, provider_name,
            [stamp] { return stamp; },
            [source_generation = std::move(source_generation), scale,
             program](
                OperatorApplyMode mode, const OperatorVectorView &input,
                const OperatorVectorView &output) {
              if (program && !arch_is_cpu(program->compile_config().arch)) {
                validate_device_composite_views(input, output, program);
              } else {
                validate_host_composite_views(input, output);
              }
              source_generation.apply_overwrite(mode, input, output);
              if (program && !arch_is_cpu(program->compile_config().arch)) {
                transform_composite_ndarray(
                    program, const_cast<Ndarray *>(output.ndarray), scale);
              } else {
                scale_composite_vector(scale, output);
              }
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
  // An offset view no longer describes the original ndarray or dense-storage
  // binding. Block-diagonal composition is host-only until subrange bindings
  // have a first-class runtime representation.
  result.ndarray = nullptr;
  result.dense_storage = nullptr;
  result.runtime_storage = nullptr;
  result.resolved_dense_storage = nullptr;
  return result;
}

}  // namespace

OperatorBinding make_sum_operator_binding(OperatorBinding left,
                                           OperatorBinding right,
                                           Program *program) {
  const auto &left_action = left.action();
  const auto &right_action = right.action();
  TI_ERROR_IF(left_action.descriptor().domain !=
                      right_action.descriptor().domain ||
                  left_action.descriptor().range !=
                      right_action.descriptor().range,
              "Operator sum requires identical operand descriptors.");
  validate_composite_operand(left_action, "sum", program);
  validate_composite_operand(right_action, "sum", program);

  const auto descriptor = left_action.descriptor();
  const auto traits = sum_operator_traits(left_action, right_action);
  OperatorCapabilities capabilities;
  capabilities.adjoint_apply =
      left_action.capabilities().adjoint_apply &&
      right_action.capabilities().adjoint_apply;
  capabilities.asynchronous_submit =
      program && !arch_is_cpu(program->compile_config().arch);
  capabilities.explicit_sequence = capabilities.asynchronous_submit;
  capabilities.persistent_workspace = program != nullptr;
  const std::string provider_name =
      "sum(" + left_action.provider_name() + "," +
      right_action.provider_name() + ")";
  std::vector<OperatorBinding> operands;
  operands.push_back(std::move(left));
  operands.push_back(std::move(right));
  auto scratch = std::make_shared<SumCompositeScratch>(descriptor, program);
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
              const bool device = input.program &&
                                  !arch_is_cpu(
                                      input.program->compile_config().arch);
              if (device) {
                validate_device_composite_views(input, output, input.program);
              } else {
                validate_host_composite_views(input, output);
              }
              std::lock_guard<std::mutex> lock(scratch->mutex);
              auto raw_input = as_raw_composite_view(input, false);
              auto raw_output = as_raw_composite_view(output, true);
              auto temporary =
                  (mode == OperatorApplyMode::forward
                       ? scratch->forward
                       : scratch->adjoint)
                      .view(input.program);
              TI_ERROR_IF(input.program &&
                              (!raw_input.ndarray ||
                               !raw_output.ndarray || !temporary.ndarray),
                          "Program-backed operator sum requires ndarray "
                          "views for input, output, and scratch.");
              pins[0].apply_overwrite(mode, raw_input, raw_output);
              pins[1].apply_overwrite(mode, raw_input, temporary);
              if (device) {
                add_composite_ndarray(
                    input.program,
                    const_cast<Ndarray *>(temporary.ndarray),
                    const_cast<Ndarray *>(raw_output.ndarray));
              } else {
                add_composite_vector(temporary, raw_output);
              }
            });
        return OperatorPinnedAction::from_retained_action(std::move(action),
                                                          stamp);
      });
}

OperatorBinding make_composed_operator_binding(OperatorBinding outer,
                                                OperatorBinding inner,
                                                Program *program) {
  const auto &outer_action = outer.action();
  const auto &inner_action = inner.action();
  TI_ERROR_IF(inner_action.descriptor().range !=
                  outer_action.descriptor().domain,
              "Operator composition requires the inner range to equal the "
              "outer domain.");
  validate_composite_operand(outer_action, "product", program);
  validate_composite_operand(inner_action, "product", program);

  const OperatorDescriptor descriptor{inner_action.descriptor().domain,
                                      outer_action.descriptor().range};
  const auto intermediate_space = inner_action.descriptor().range;
  const OperatorMathematicalTraits traits;
  OperatorCapabilities capabilities;
  capabilities.adjoint_apply =
      outer_action.capabilities().adjoint_apply &&
      inner_action.capabilities().adjoint_apply;
  capabilities.asynchronous_submit =
      program && !arch_is_cpu(program->compile_config().arch);
  capabilities.explicit_sequence = capabilities.asynchronous_submit;
  capabilities.persistent_workspace = program != nullptr;
  const std::string provider_name =
      "compose(" + outer_action.provider_name() + "," +
      inner_action.provider_name() + ")";
  std::vector<OperatorBinding> operands;
  operands.push_back(std::move(outer));
  operands.push_back(std::move(inner));
  auto scratch = std::make_shared<ProductCompositeScratch>(
      intermediate_space, program);
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
              if (input.program &&
                  !arch_is_cpu(input.program->compile_config().arch)) {
                validate_device_composite_views(input, output, input.program);
              } else {
                validate_host_composite_views(input, output);
              }
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
    validate_composite_operand(action, "block-diagonal", nullptr);
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
  OperatorCapabilities capabilities;
  capabilities.dense_storage_operands = true;
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
  capabilities.dense_storage_operands = expected_arch == Arch::cuda;
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

LinearOperatorRecordableKernel::LinearOperatorRecordableKernel(
    Program *program,
    Kernel *kernel,
    std::int32_t active_size,
    Ndarray *topology,
    Ndarray *numeric,
    OperatorResourceStamp stamp,
    std::shared_ptr<void> generation_owner)
    : program_(program),
      kernel_(kernel),
      active_size_(active_size),
      topology_(topology),
      numeric_(numeric),
      stamp_(stamp),
      generation_owner_(std::move(generation_owner)) {
  TI_ERROR_IF(!program_ || !kernel_ || active_size_ <= 0 || !topology_ ||
                  !generation_owner_,
              "Recordable LinearOperator kernels require a live Program, "
              "kernel, positive active size, topology, and generation.");
}

LinearOperatorRecordableKernel::LinearOperatorRecordableKernel(
    Program *program,
    const aot::CompiledGraph *graph,
    FixedI32Arguments fixed_i32,
    FixedNdarrayArguments fixed_ndarrays,
    std::vector<SNodeTreeDependency> state_dependencies,
    OperatorResourceStamp stamp,
    std::shared_ptr<void> generation_owner)
    : program_(program),
      graph_(graph),
      fixed_i32_(std::move(fixed_i32)),
      fixed_ndarrays_(std::move(fixed_ndarrays)),
      state_dependencies_(std::move(state_dependencies)),
      stamp_(stamp),
      generation_owner_(std::move(generation_owner)) {
  TI_ERROR_IF(!program_ || !graph_ || graph_->dispatches.empty() ||
                  !generation_owner_,
              "Recordable LinearOperator Graph actions require a live "
              "Program, non-empty Graph, and immutable generation.");
  for (const auto &dispatch : graph_->dispatches) {
    TI_ERROR_IF(!dispatch.ti_kernel,
                "Recordable LinearOperator Graph actions require JIT "
                "dispatch kernels.");
  }
}

Program *LinearOperatorRecordableKernel::program() const {
  return program_;
}

Kernel *LinearOperatorRecordableKernel::kernel() const {
  return kernel_;
}

const aot::CompiledGraph *LinearOperatorRecordableKernel::graph() const {
  return graph_;
}

std::int32_t LinearOperatorRecordableKernel::active_size() const {
  return active_size_;
}

Ndarray *LinearOperatorRecordableKernel::topology() const {
  return topology_;
}

Ndarray *LinearOperatorRecordableKernel::numeric() const {
  return numeric_;
}

const LinearOperatorRecordableKernel::FixedI32Arguments &
LinearOperatorRecordableKernel::fixed_i32() const {
  return fixed_i32_;
}

const LinearOperatorRecordableKernel::FixedNdarrayArguments &
LinearOperatorRecordableKernel::fixed_ndarrays() const {
  return fixed_ndarrays_;
}

const std::vector<SNodeTreeDependency> &
LinearOperatorRecordableKernel::state_dependencies() const {
  return state_dependencies_;
}

OperatorResourceStamp LinearOperatorRecordableKernel::resource_stamp() const {
  return stamp_;
}

LinearOperatorHandle::LinearOperatorHandle(
    Program *program,
    OperatorBinding binding,
    std::shared_ptr<void> provider_owner,
    NumericUpdateFn numeric_update,
    RecordableKernelFn recordable_kernel)
    : program_(program),
      provider_owner_(std::move(provider_owner)),
      numeric_update_(std::move(numeric_update)),
      recordable_kernel_(std::move(recordable_kernel)),
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

LinearOperatorHandle::~LinearOperatorHandle() =
    default;

Program *LinearOperatorHandle::program() const {
  return program_;
}

const OperatorDescriptor &
LinearOperatorHandle::descriptor() const {
  return plan_->descriptor();
}

const OperatorMathematicalTraits &
LinearOperatorHandle::mathematical_traits() const {
  return plan_->mathematical_traits();
}

const OperatorCapabilities &
LinearOperatorHandle::capabilities() const {
  return plan_->capabilities();
}

const std::string &LinearOperatorHandle::provider_name() const {
  return plan_->provider_name();
}

OperatorExecutionKind
LinearOperatorHandle::execution_kind() const {
  return plan_->execution_kind();
}

OperatorResourceStamp
LinearOperatorHandle::resource_stamp() const {
  return plan_->resource_stamp();
}

OperatorPlanRuntimeStatistics
LinearOperatorHandle::debug_runtime_statistics() const {
  return plan_->debug_runtime_statistics();
}

OperatorBinding LinearOperatorHandle::binding() const {
  return binding_;
}

std::unique_ptr<LinearOperatorSession>
LinearOperatorHandle::begin_session() {
  return std::make_unique<LinearOperatorSession>(
      program_, plan_.get(), plan_->pin());
}

void LinearOperatorHandle::apply(Program *program,
                                             const Ndarray &input,
                                             const Ndarray &output) {
  apply_generalized(program, input, nullptr, output, 1.0, 0.0);
}

void LinearOperatorHandle::apply_generalized(
    Program *program,
    const Ndarray &input,
    const Ndarray *addend,
    const Ndarray &output,
    double alpha,
    double beta) {
  TI_ERROR_IF(program != program_,
              "LinearOperator apply must use its construction Program.");
  const auto &descriptor = plan_->descriptor();
  OperatorVectorView addend_view;
  const OperatorVectorView *addend_view_ptr = nullptr;
  if (beta != 0.0) {
    TI_ERROR_IF(!addend,
                "LinearOperator generalized apply with nonzero beta "
                "requires an addend.");
    addend_view = OperatorVectorView::from_ndarray(
        program_, *addend, descriptor.range, false);
    addend_view_ptr = &addend_view;
  }
  auto submission = plan_->submit(
      {OperatorApplyMode::forward,
       OperatorVectorView::from_ndarray(program_, input,
                                        descriptor.domain, false),
       addend_view_ptr,
       OperatorVectorView::from_ndarray(program_, output,
                                        descriptor.range, true),
       alpha,
       beta});
  submission.wait();
}

void LinearOperatorHandle::apply_dense_storage(
    Program *program,
    const storage::RuntimeStorageArgument &input,
    const storage::RuntimeStorageArgument &output) {
  TI_ERROR_IF(program != program_,
              "LinearOperator dense storage apply must use its construction "
              "Program.");
  const auto &capabilities = plan_->capabilities();
  TI_ERROR_IF(!capabilities.dense_storage_operands,
              "LinearOperator provider does not accept direct dense storage "
              "operands.");
  const auto validate_argument = [&](const storage::RuntimeStorageArgument &arg,
                                     const char *role) {
    const auto &qualification = arg.qualification();
    TI_ERROR_IF(!qualification.capabilities.bindable ||
                    !qualification.capabilities.zero_copy_qualified,
                "LinearOperator {} runtime storage is not directly bindable: "
                "{}.",
                role, storage::to_string(qualification.reason));
    TI_ERROR_IF(qualification.dense.execution_mode ==
                        storage::StorageExecutionMode::kDirectAffine &&
                    !capabilities.dense_storage_affine_operands,
                "LinearOperator provider does not accept affine dense storage "
                "{} operands.",
                role);
  };
  validate_argument(input, "input");
  validate_argument(output, "output");
  const std::vector<const storage::RuntimeStorageArgument *> arguments{
      &input, &output};
  program_->with_resolved_runtime_storage_arguments(
      arguments,
      [&](const storage::ResolvedDenseBinding *bindings, std::size_t count) {
        TI_ASSERT(count == 2);
        const auto &operator_descriptor = plan_->descriptor();
        auto submission = plan_->submit(
            {OperatorApplyMode::forward,
             OperatorVectorView::from_dense_storage(
                 program_, input, bindings[0], operator_descriptor.domain,
                 false),
             nullptr,
             OperatorVectorView::from_dense_storage(
                 program_, output, bindings[1], operator_descriptor.range,
                 true)});
        submission.wait();
      });
}

void LinearOperatorHandle::update_numeric(
    Program *program,
    const NumericUpdateArguments &arguments,
    std::uint64_t expected_topology_version,
    std::uint64_t expected_numeric_version) {
  TI_ERROR_IF(program != program_,
              "LinearOperator numeric update must use its construction "
              "Program.");
  TI_ERROR_IF(!numeric_update_,
              "LinearOperator provider does not support numeric updates.");
  numeric_update_(program, arguments, expected_topology_version,
                  expected_numeric_version);
}

bool LinearOperatorHandle::supports_numeric_update() const {
  return static_cast<bool>(numeric_update_);
}

std::shared_ptr<LinearOperatorRecordableKernel>
LinearOperatorHandle::recordable_kernel(OperatorApplyMode mode) {
  TI_ERROR_IF(!recordable_kernel_,
              "LinearOperator provider does not expose a recordable kernel.");
  return recordable_kernel_(mode);
}

bool LinearOperatorHandle::supports_recordable_kernel() const {
  return static_cast<bool>(recordable_kernel_);
}

LinearOperatorSession::LinearOperatorSession(
    Program *program,
    OperatorPlan *plan,
    OperatorPinnedAction generation)
    : program_(program), plan_(plan), generation_(std::move(generation)) {
  TI_ERROR_IF(!program_ || !plan_,
              "LinearOperator session requires a live operator plan.");
}

LinearOperatorSession::~LinearOperatorSession() {
  if (submitted_ && program_) {
    try {
      program_->synchronize();
    } catch (...) {
      // Destruction cannot surface a backend error. Program diagnostics keep
      // the fault observable, matching OperatorSubmission destruction.
    }
  }
}

void LinearOperatorSession::submit(Program *program,
                                               const Ndarray &input,
                                               const Ndarray &output) {
  TI_ERROR_IF(program != program_,
              "LinearOperator session must use its construction Program.");
  const auto &descriptor = plan_->descriptor();
  (void)plan_->submit(
      generation_,
      {OperatorApplyMode::forward,
       OperatorVectorView::from_ndarray(program_, input, descriptor.domain,
                                        false),
       nullptr,
       OperatorVectorView::from_ndarray(program_, output, descriptor.range,
                                        true)});
  submitted_ = true;
}

void LinearOperatorSession::wait() {
  if (submitted_) {
    program_->synchronize();
    submitted_ = false;
  }
}

void LinearOperatorSession::mark_synchronized() {
  submitted_ = false;
}

ExperimentalPreconditionerSession::ExperimentalPreconditionerSession(
    Program *program,
    OperatorPlan *action_plan,
    OperatorPinnedAction target_generation,
    OperatorPinnedAction action_generation,
    std::shared_ptr<std::atomic<std::uint64_t>> apply_counter)
    : program_(program),
      action_plan_(action_plan),
      target_generation_(std::move(target_generation)),
      action_generation_(std::move(action_generation)),
      apply_counter_(std::move(apply_counter)) {
  TI_ERROR_IF(!program_ || !action_plan_ || !target_generation_ ||
                  !action_generation_ || !apply_counter_,
              "Preconditioner session requires a live Program, plan, "
              "target generation, and action generation.");
}

ExperimentalPreconditionerSession::~ExperimentalPreconditionerSession() =
    default;

void ExperimentalPreconditionerSession::apply(Program *program,
                                              const Ndarray &input,
                                              const Ndarray &output) {
  TI_ERROR_IF(program != program_,
              "Preconditioner session must use its construction Program.");
  const auto &descriptor = action_plan_->descriptor();
  (void)action_plan_->submit(
      action_generation_,
      {OperatorApplyMode::forward,
       OperatorVectorView::from_ndarray(program_, input, descriptor.domain,
                                        false),
       nullptr,
       OperatorVectorView::from_ndarray(program_, output, descriptor.range,
                                        true)});
  // Pinned hot-loop submissions deliberately use an externally managed
  // completion boundary. This public convenience method is synchronous, so
  // it owns that boundary explicitly instead of waiting on a ticket that has
  // no per-submit backend completion object.
  program_->synchronize();
  apply_counter_->fetch_add(1, std::memory_order_relaxed);
}

OperatorResourceStamp ExperimentalPreconditionerSession::target_stamp()
    const {
  return target_generation_.resource_stamp();
}

OperatorResourceStamp ExperimentalPreconditionerSession::action_stamp()
    const {
  return action_generation_.resource_stamp();
}

ExperimentalPreconditionerPlanHandle::ExperimentalPreconditionerPlanHandle(
    Program *program,
    LinearOperatorHandle &target,
    LinearOperatorHandle &action,
    std::string method)
    : program_(program),
      target_descriptor_(target.descriptor()),
      target_binding_(target.binding()),
      action_plan_(
          std::make_unique<OperatorPlan>(program, action.binding())),
      approved_generations_(
          std::make_unique<OperatorResourceGenerationPublisher>()),
      method_(std::move(method)),
      apply_counter_(
          std::make_shared<std::atomic<std::uint64_t>>(0)) {
  TI_ERROR_IF(!program_ || target.program() != program_ ||
                  action.program() != program_,
              "PreconditionerPlan target and action must belong to its "
              "construction Program.");
  validate_preconditioner_descriptor(target_descriptor_,
                                     action_plan_->descriptor());
  TI_ERROR_IF(method_.empty(),
              "PreconditionerPlan method must be non-empty.");
}

ExperimentalPreconditionerPlanHandle::~ExperimentalPreconditionerPlanHandle() {
  try {
    if (approved_generations_) {
      approved_generations_->retire_current();
      approved_generations_.reset();
    }
    action_plan_.reset();
  } catch (...) {
  }
}

void ExperimentalPreconditionerPlanHandle::validate_program(
    Program *program) const {
  TI_ERROR_IF(program != program_,
              "PreconditionerPlan must use its construction Program.");
}

void ExperimentalPreconditionerPlanHandle::setup(Program *program) {
  validate_program(program);
  std::lock_guard<std::mutex> lock(mutex_);
  TI_ERROR_IF(is_setup_, "PreconditionerPlan setup may only run once.");
  statistics_.setup_calls++;
  auto target_generation = target_binding_.pin();
  auto action_generation = action_plan_->pin();
  publish_approved_generation(target_generation, action_generation);
  built_from_operator_stamp_ = target_generation.resource_stamp();
  accepted_target_stamp_ = built_from_operator_stamp_;
  accepted_action_stamp_ = action_generation.resource_stamp();
  is_setup_ = true;
  statistics_.rebuild_attestations++;
}

void ExperimentalPreconditionerPlanHandle::validate_update(
    Program *program,
    bool accept_reuse) {
  validate_program(program);
  std::lock_guard<std::mutex> lock(mutex_);
  TI_ERROR_IF(!is_setup_,
              "PreconditionerPlan must be setup before update.");
  const auto target_stamp =
      target_binding_.action().resource_stamp();
  const auto action_stamp = action_plan_->resource_stamp();
  const bool target_changed =
      operator_resource_changes(accepted_target_stamp_, target_stamp) != 0;
  const bool action_changed =
      operator_resource_changes(accepted_action_stamp_, action_stamp) != 0;
  TI_ERROR_IF(
      target_changed && !action_changed && !accept_reuse,
      "PreconditionerPlan target changed while its action did not; publish "
      "a rebuilt action or explicitly set accept_reuse=True.");
  TI_ERROR_IF(
      accept_reuse && action_changed,
      "PreconditionerPlan accept_reuse=True requires the previously "
      "approved action generation; use a rebuild update for a new action.");
}

void ExperimentalPreconditionerPlanHandle::update(Program *program,
                                                  bool accept_reuse) {
  validate_program(program);
  std::lock_guard<std::mutex> lock(mutex_);
  TI_ERROR_IF(!is_setup_,
              "PreconditionerPlan must be setup before update.");
  statistics_.update_calls++;
  try {
    auto target_generation = target_binding_.pin();
    auto action_generation = action_plan_->pin();
    const auto target_stamp = target_generation.resource_stamp();
    const auto action_stamp = action_generation.resource_stamp();
    const bool target_changed =
        operator_resource_changes(accepted_target_stamp_, target_stamp) != 0;
    const bool action_changed =
        operator_resource_changes(accepted_action_stamp_, action_stamp) != 0;
    if (target_changed) {
      statistics_.target_generation_changes++;
    }
    if (action_changed) {
      statistics_.action_generation_changes++;
    }
    if (!target_changed && !action_changed) {
      statistics_.update_noops++;
      return;
    }
    TI_ERROR_IF(
        target_changed && !action_changed && !accept_reuse,
        "PreconditionerPlan target changed while its action did not; publish "
        "a rebuilt action or explicitly set accept_reuse=True.");
    TI_ERROR_IF(
        accept_reuse && action_changed,
        "PreconditionerPlan accept_reuse=True requires the previously "
        "approved action generation; use a rebuild update for a new action.");
    if (accept_reuse) {
      publish_approved_generation(target_generation, action_generation);
      accepted_target_stamp_ = target_stamp;
      statistics_.reuse_attestations++;
    } else {
      publish_approved_generation(target_generation, action_generation);
      built_from_operator_stamp_ = target_stamp;
      accepted_target_stamp_ = target_stamp;
      accepted_action_stamp_ = action_stamp;
      statistics_.rebuild_attestations++;
    }
    statistics_.update_successes++;
  } catch (...) {
    statistics_.update_failures++;
    throw;
  }
}

std::unique_ptr<ExperimentalPreconditionerSession>
ExperimentalPreconditionerPlanHandle::pin_locked() {
  TI_ERROR_IF(!is_setup_,
              "PreconditionerPlan must be setup before pin/apply.");
  auto target_generation = target_binding_.pin();
  auto action_generation = action_plan_->pin();
  const bool target_stale =
      operator_resource_changes(accepted_target_stamp_,
                                target_generation.resource_stamp()) != 0;
  const bool action_stale =
      operator_resource_changes(accepted_action_stamp_,
                                action_generation.resource_stamp()) != 0;
  if (target_stale || action_stale) {
    statistics_.stale_rejections++;
    TI_ERROR(
        "PreconditionerPlan is stale: {}{} generation changed without an "
        "explicit update.",
        target_stale ? "target" : "",
        target_stale && action_stale
            ? " and action"
            : (action_stale ? "action" : ""));
  }
  const auto target_stamp = target_generation.resource_stamp();
  const auto action_stamp = action_generation.resource_stamp();
  TI_ERROR_IF(
      target_stamp.program_identity != action_stamp.program_identity ||
          target_stamp.program_generation != action_stamp.program_generation,
      "PreconditionerPlan target and action belong to different Program "
      "generations.");
  statistics_.pins++;
  return std::make_unique<ExperimentalPreconditionerSession>(
      program_, action_plan_.get(), std::move(target_generation),
      std::move(action_generation), apply_counter_);
}

void ExperimentalPreconditionerPlanHandle::publish_approved_generation(
    const OperatorPinnedAction &target_generation,
    const OperatorPinnedAction &action_generation) {
  const auto accepted_stamp = target_generation.resource_stamp();
  auto action = OperatorAction(
      action_generation.descriptor(),
      action_generation.mathematical_traits(),
      action_generation.capabilities(), "forge_preconditioner_plan_action",
      [accepted_stamp] { return accepted_stamp; },
      [target_generation, action_generation](
          OperatorApplyMode mode, const OperatorVectorView &input,
          const OperatorVectorView &output) {
        (void)target_generation;
        action_generation.apply_overwrite(mode, input, output);
      });
  approved_generations_->publish(std::move(action));
}

OperatorBinding ExperimentalPreconditionerPlanHandle::consumer_binding() {
  auto metadata_action = OperatorAction(
      action_plan_->descriptor(), action_plan_->mathematical_traits(),
      action_plan_->capabilities(), "forge_preconditioner_plan_action",
      [this] {
        std::lock_guard<std::mutex> lock(mutex_);
        return is_setup_ ? accepted_target_stamp_
                         : target_binding_.action().resource_stamp();
      },
      [this](OperatorApplyMode mode, const OperatorVectorView &input,
             const OperatorVectorView &output) {
        auto approved = approved_generations_->acquire();
        approved.apply_overwrite(mode, input, output);
      });
  auto binding = OperatorBinding::from_generation_publisher(
      std::move(metadata_action),
      [this] { return approved_generations_->acquire(); });
  return binding.with_execution_lowering(action_plan_->execution_kind());
}

std::unique_ptr<ExperimentalPreconditionerSession>
ExperimentalPreconditionerPlanHandle::pin(Program *program) {
  validate_program(program);
  std::lock_guard<std::mutex> lock(mutex_);
  return pin_locked();
}

bool ExperimentalPreconditionerPlanHandle::is_setup() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return is_setup_;
}

const std::string &ExperimentalPreconditionerPlanHandle::method() const {
  return method_;
}

OperatorResourceStamp
ExperimentalPreconditionerPlanHandle::built_from_operator_stamp() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return built_from_operator_stamp_;
}

OperatorResourceStamp
ExperimentalPreconditionerPlanHandle::accepted_target_stamp() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return accepted_target_stamp_;
}

OperatorResourceStamp
ExperimentalPreconditionerPlanHandle::accepted_action_stamp() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return accepted_action_stamp_;
}

ExperimentalPreconditionerPlanRuntimeStatistics
ExperimentalPreconditionerPlanHandle::debug_runtime_statistics() const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto result = statistics_;
  result.apply_calls =
      apply_counter_->load(std::memory_order_relaxed);
  const auto generations = approved_generations_->debug_statistics();
  result.approved_generations_published = generations.published;
  result.approved_generations_retired = generations.retired;
  result.approved_generations_released = generations.released;
  result.approved_generation_active_leases = generations.active_leases;
  result.has_current_approved_generation = generations.has_current;
  return result;
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

namespace {

bool is_action_scalar_kernel_parameter(
    const CallableBase::Parameter &parameter,
    DataType dtype) {
  return parameter.ptype == ParameterType::kScalar && !parameter.is_array &&
         parameter.get_dtype() == dtype;
}

bool is_action_scalar_ndarray_kernel_parameter(
    const CallableBase::Parameter &parameter,
    DataType dtype,
    std::size_t dimensions) {
  const DataType expected_type(
      TypeFactory::get_instance().get_ndarray_struct_type(
          dtype, static_cast<int>(dimensions), false));
  return parameter.ptype == ParameterType::kNdarray && parameter.is_array &&
         !parameter.needs_grad && parameter.get_dtype() == expected_type &&
         parameter.get_element_shape().empty() &&
         parameter.total_dim == dimensions;
}

void validate_action_resource(Program *program,
                              const Ndarray &data,
                              const char *role) {
  TI_ERROR_IF(data.owning_program() != program ||
                  !data.get_element_shape().empty() || data.shape.empty() ||
                  data.get_nelement() == 0,
              "Compiled-kernel action {} must be a non-empty scalar ndarray "
              "owned by the same Program.",
              role);
}

class ActionNdarrayOwner {
 public:
  ActionNdarrayOwner(Program *program, Ndarray *array)
      : program_(program), array_(array) {
  }
  ~ActionNdarrayOwner() {
    if (program_ && array_) {
      program_->delete_ndarray(array_);
    }
  }
  ActionNdarrayOwner(const ActionNdarrayOwner &) = delete;
  ActionNdarrayOwner &operator=(const ActionNdarrayOwner &) = delete;

  Ndarray *get() const {
    return array_;
  }
  Ndarray *release() {
    return std::exchange(array_, nullptr);
  }

 private:
  Program *program_{nullptr};
  Ndarray *array_{nullptr};
};

struct CompiledKernelActionTopology {
  CompiledKernelActionTopology(Program *program, Ndarray *data)
      : program(program),
        program_lifetime(program ? program->weak_resource_lifetime_token()
                                 : std::weak_ptr<ProgramLifetimeToken>{}),
        data(data) {
  }
  ~CompiledKernelActionTopology() {
    Program::delete_ndarray_if_alive(program, program_lifetime, data);
  }

  Program *program{nullptr};
  std::weak_ptr<ProgramLifetimeToken> program_lifetime;
  Ndarray *data{nullptr};
};

class CompiledKernelActionLaunch {
 public:
  CompiledKernelActionLaunch(Program *program,
                             Kernel *kernel,
                             const CompiledKernelData *compiled_kernel,
                             int active_size,
                             std::shared_ptr<CompiledKernelActionTopology>
                                 topology,
                             Ndarray *numeric_data,
                             std::size_t input_arg_index,
                             std::size_t output_arg_index)
      : program_(program),
        compiled_kernel_(compiled_kernel),
        topology_(std::move(topology)),
        numeric_data_(numeric_data),
        input_arg_index_(input_arg_index),
        output_arg_index_(output_arg_index),
        launch_context_(std::make_unique<LaunchContextBuilder>(kernel)) {
    launch_context_->set_arg_int({0}, active_size);
    restore_fixed_arguments();
  }

  void apply(const OperatorVectorView &input,
             const OperatorVectorView &output) {
    auto valid_operand = [](const OperatorVectorView &view) {
      return view.ndarray ||
             (view.dense_storage && view.resolved_dense_storage);
    };
    TI_ERROR_IF(!valid_operand(input) || !valid_operand(output),
                "Compiled-kernel actions require ndarray or resolved dense "
                "storage views.");
    std::lock_guard<std::mutex> lock(launch_mutex_);
    // CPU launchers lower ndarray placeholders to raw pointers in place.
    // Restore fixed resources so this context remains generation-stable.
    restore_fixed_arguments();
    auto bind_operand = [&](std::size_t argument,
                            const OperatorVectorView &view) {
      const std::vector<int> arg_id{static_cast<int>(argument)};
      if (view.dense_storage) {
        launch_context_->set_arg_resolved_dense_storage(
            arg_id, *view.dense_storage, *view.resolved_dense_storage);
      } else {
        launch_context_->set_arg_ndarray(arg_id, *view.ndarray);
      }
    };
    bind_operand(input_arg_index_, input);
    bind_operand(output_arg_index_, output);
    program_->launch_kernel(*compiled_kernel_, *launch_context_);
  }

 private:
  void restore_fixed_arguments() {
    launch_context_->set_arg_ndarray({1}, *topology_->data);
    if (numeric_data_) {
      launch_context_->set_arg_ndarray({2}, *numeric_data_);
    }
  }

  Program *program_{nullptr};
  const CompiledKernelData *compiled_kernel_{nullptr};
  std::shared_ptr<CompiledKernelActionTopology> topology_;
  Ndarray *numeric_data_{nullptr};
  std::size_t input_arg_index_{0};
  std::size_t output_arg_index_{0};
  std::unique_ptr<LaunchContextBuilder> launch_context_;
  std::mutex launch_mutex_;
};

struct CompiledKernelActionGeneration {
  CompiledKernelActionGeneration(
      Program *program,
      Kernel *forward_kernel,
      const CompiledKernelData *forward_compiled,
      Kernel *adjoint_kernel,
      const CompiledKernelData *adjoint_compiled,
      const OperatorDescriptor &descriptor,
      std::shared_ptr<CompiledKernelActionTopology> topology,
      Ndarray *numeric_data,
      std::size_t input_arg_index,
      std::size_t output_arg_index)
      : program(program),
        program_lifetime(program ? program->weak_resource_lifetime_token()
                                 : std::weak_ptr<ProgramLifetimeToken>{}),
        topology(std::move(topology)),
        numeric_data(numeric_data) {
    forward = std::make_unique<CompiledKernelActionLaunch>(
        program, forward_kernel, forward_compiled,
        static_cast<int>(descriptor.range.scalar_extent), this->topology,
        numeric_data, input_arg_index, output_arg_index);
    if (adjoint_kernel) {
      adjoint = std::make_unique<CompiledKernelActionLaunch>(
          program, adjoint_kernel, adjoint_compiled,
          static_cast<int>(descriptor.domain.scalar_extent), this->topology,
          numeric_data, input_arg_index, output_arg_index);
    }
  }

  ~CompiledKernelActionGeneration() {
    // Launch contexts only borrow the numeric ndarray. Destroy them first.
    adjoint.reset();
    forward.reset();
    Program::delete_ndarray_if_alive(program, program_lifetime, numeric_data);
  }

  void apply(OperatorApplyMode mode,
             const OperatorVectorView &input,
             const OperatorVectorView &output) {
    if (mode == OperatorApplyMode::forward) {
      forward->apply(input, output);
      return;
    }
    TI_ERROR_IF(!adjoint,
                "Compiled-kernel action has no explicit adjoint provider.");
    adjoint->apply(input, output);
  }

  Program *program{nullptr};
  std::weak_ptr<ProgramLifetimeToken> program_lifetime;
  std::shared_ptr<CompiledKernelActionTopology> topology;
  Ndarray *numeric_data{nullptr};
  OperatorResourceStamp stamp;
  std::unique_ptr<CompiledKernelActionLaunch> forward;
  std::unique_ptr<CompiledKernelActionLaunch> adjoint;
};

class CompiledKernelActionProvider {
 public:
  CompiledKernelActionProvider(Program *program,
                               Kernel &forward_kernel,
                               Kernel *adjoint_kernel,
                               OperatorDescriptor descriptor,
                               std::uint64_t topology_version,
                               std::uint64_t numeric_version,
                               const Ndarray &topology_data,
                               const Ndarray *numeric_data)
      : program_(program),
        forward_kernel_(&forward_kernel),
        adjoint_kernel_(adjoint_kernel),
        descriptor_(std::move(descriptor)),
        topology_version_(topology_version),
        numeric_version_(numeric_version) {
    TI_ERROR_IF(!program_ || descriptor_.domain.scalar_extent == 0 ||
                    descriptor_.range.scalar_extent == 0 ||
                    descriptor_.domain.scalar_extent >
                        static_cast<std::size_t>(
                            (std::numeric_limits<int>::max)()) ||
                    descriptor_.range.scalar_extent >
                        static_cast<std::size_t>(
                            (std::numeric_limits<int>::max)()) ||
                    topology_version_ == 0 || numeric_version_ == 0,
                "Compiled-kernel actions require an owning Program, positive "
                "f32 domain/range extents, and positive resource versions.");
    TI_ERROR_IF(descriptor_.domain.scalar_type != PrimitiveType::f32 ||
                    descriptor_.range.scalar_type != PrimitiveType::f32 ||
                    !descriptor_.domain.entry_shape.empty() ||
                    !descriptor_.range.entry_shape.empty(),
                "Compiled-kernel actions currently require scalar f32 "
                "domain and range spaces.");
    const Arch arch = program_->compile_config().arch;
    TI_ERROR_IF(!arch_is_cpu(arch) && !arch_is_cuda(arch) &&
                    arch != Arch::vulkan,
                "Compiled-kernel actions support CPU, CUDA, and Vulkan only; "
                "got {}. No fallback was performed.",
                arch_name(arch));
    validate_action_resource(program_, topology_data, "topology data");
    if (numeric_data) {
      validate_action_resource(program_, *numeric_data, "numeric data");
    }
    has_numeric_data_ = numeric_data != nullptr;
    if (numeric_data) {
      numeric_type_ = numeric_data->get_element_data_type();
      numeric_shape_ = numeric_data->shape;
      numeric_layout_ = numeric_data->layout;
    }
    input_arg_index_ = has_numeric_data_ ? 3 : 2;
    output_arg_index_ = input_arg_index_ + 1;

    Ndarray *owned_topology = nullptr;
    Ndarray *owned_numeric = nullptr;
    try {
      owned_topology = program_->create_ndarray(
          topology_data.get_element_data_type(), topology_data.shape,
          topology_data.layout, false);
      program_->copy_ndarray_fast(
          owned_topology, const_cast<Ndarray *>(&topology_data));
      if (numeric_data) {
        owned_numeric = program_->create_ndarray(
            numeric_data->get_element_data_type(), numeric_data->shape,
            numeric_data->layout, false);
        program_->copy_ndarray_fast(
            owned_numeric, const_cast<Ndarray *>(numeric_data));
      }
    } catch (...) {
      if (owned_numeric) {
        program_->delete_ndarray(owned_numeric);
      }
      if (owned_topology) {
        program_->delete_ndarray(owned_topology);
      }
      throw;
    }
    ActionNdarrayOwner topology_owner(program_, owned_topology);
    ActionNdarrayOwner numeric_owner(program_, owned_numeric);
    topology_ = std::make_shared<CompiledKernelActionTopology>(
        program_, topology_owner.get());
    topology_owner.release();
    forward_compiled_ = validate_and_compile(forward_kernel, "forward");
    if (adjoint_kernel_) {
      adjoint_compiled_ =
          validate_and_compile(*adjoint_kernel_, "adjoint");
    }
    generations_ =
        std::make_unique<OperatorResourceGenerationPublisher>();
    publish(numeric_owner.release(), numeric_version_, binding_revision_);
  }

  ~CompiledKernelActionProvider() {
    try {
      if (generations_) {
        generations_->retire_current();
        generations_.reset();
      }
      topology_.reset();
    } catch (...) {
    }
  }

  OperatorBinding binding() {
    const auto capabilities = make_capabilities();
    auto metadata_action = OperatorAction(
        descriptor_, capabilities, "forge_compiled_kernel_action",
        [this] { return current_stamp(); },
        [this](OperatorApplyMode mode, const OperatorVectorView &input,
               const OperatorVectorView &output) {
          auto generation = generations_->acquire();
          generation.apply_overwrite(mode, input, output);
        });
    return OperatorBinding::from_generation_publisher(
        std::move(metadata_action),
        [this] { return generations_->acquire(); });
  }

  std::shared_ptr<LinearOperatorRecordableKernel> recordable_kernel(
      OperatorApplyMode mode) {
    std::lock_guard<std::mutex> lock(update_mutex_);
    TI_ERROR_IF(!current_generation_,
                "Compiled-kernel action has no published generation.");
    Kernel *kernel = mode == OperatorApplyMode::forward ? forward_kernel_
                                                        : adjoint_kernel_;
    TI_ERROR_IF(!kernel,
                "Compiled-kernel action has no explicit adjoint provider.");
    const auto extent = mode == OperatorApplyMode::forward
                            ? descriptor_.range.scalar_extent
                            : descriptor_.domain.scalar_extent;
    return std::make_shared<LinearOperatorRecordableKernel>(
        program_, kernel, static_cast<std::int32_t>(extent),
        current_generation_->topology->data,
        current_generation_->numeric_data, current_generation_->stamp,
        current_generation_);
  }

  void update_numeric(
      Program *program,
      const LinearOperatorHandle::NumericUpdateArguments
          &arguments,
      std::uint64_t expected_topology_version,
      std::uint64_t expected_numeric_version) {
    TI_ERROR_IF(program != program_ || !has_numeric_data_,
                "Compiled-kernel action numeric update requires its owning "
                "Program and a numeric resource.");
    TI_ERROR_IF(arguments.size() != 1 ||
                    arguments.find("numeric") == arguments.end() ||
                    !arguments.at("numeric"),
                "Compiled-kernel action numeric update requires exactly the "
                "'numeric' ndarray.");
    const Ndarray &numeric_data = *arguments.at("numeric");
    validate_action_resource(program_, numeric_data, "numeric update");
    std::lock_guard<std::mutex> lock(update_mutex_);
    TI_ERROR_IF(expected_topology_version != topology_version_ ||
                    expected_numeric_version != numeric_version_,
                "Compiled-kernel action numeric update version mismatch: "
                "expected topology/numeric ({}, {}), current ({}, {}).",
                expected_topology_version, expected_numeric_version,
                topology_version_, numeric_version_);
    TI_ERROR_IF(numeric_version_ ==
                        (std::numeric_limits<std::uint64_t>::max)() ||
                    binding_revision_ ==
                        (std::numeric_limits<std::uint64_t>::max)(),
                "Compiled-kernel action resource version overflow.");
    TI_ERROR_IF(numeric_data.get_element_data_type() != numeric_type_ ||
                    numeric_data.shape != numeric_shape_ ||
                    numeric_data.layout != numeric_layout_,
                "Compiled-kernel action numeric update must preserve dtype, "
                "shape, and layout.");
    Ndarray *replacement = program_->create_ndarray(
        numeric_type_, numeric_shape_, numeric_layout_, false);
    try {
      program_->copy_ndarray_fast(replacement,
                                  const_cast<Ndarray *>(&numeric_data));
    } catch (...) {
      program_->delete_ndarray(replacement);
      throw;
    }
    const auto next_numeric = numeric_version_ + 1;
    const auto next_binding = binding_revision_ + 1;
    publish(replacement, next_numeric, next_binding);
    numeric_version_ = next_numeric;
    binding_revision_ = next_binding;
  }

 private:
  const CompiledKernelData *validate_and_compile(Kernel &kernel,
                                                  const char *role) {
    TI_ERROR_IF(kernel.program != program_ ||
                    kernel.arch != program_->compile_config().arch,
                "Compiled-kernel action {} kernel must belong to the same "
                "Program and backend.",
                role);
    const auto &parameters = kernel.parameter_list;
    const std::size_t expected_count = output_arg_index_ + 1;
    bool valid = parameters.size() == expected_count && kernel.rets.empty() &&
                 is_action_scalar_kernel_parameter(
                     parameters[0], PrimitiveType::i32) &&
                 is_action_scalar_ndarray_kernel_parameter(
                     parameters[1], topology_->data->get_element_data_type(),
                     topology_->data->shape.size());
    if (has_numeric_data_) {
      valid = valid && is_action_scalar_ndarray_kernel_parameter(
                           parameters[2], numeric_type_,
                           numeric_shape_.size());
    }
    valid = valid && is_action_scalar_ndarray_kernel_parameter(
                         parameters[input_arg_index_], PrimitiveType::f32, 1) &&
            is_action_scalar_ndarray_kernel_parameter(
                parameters[output_arg_index_], PrimitiveType::f32, 1);
    TI_ERROR_IF(!valid,
                "Compiled-kernel action {} ABI must be exactly active_size, "
                "topology, optional numeric, f32[1D] input, f32[1D] output "
                "with no return values.",
                role);
    const auto &compiled = program_->compile_kernel(
        program_->compile_config(), program_->get_device_caps(), kernel);
    TI_ERROR_IF(!compiled.snode_tree_ids().empty(),
                "Compiled-kernel actions must not depend on an SNodeTree; "
                "use explicit topology/numeric ndarray arguments.");
    return &compiled;
  }

  OperatorCapabilities make_capabilities() const {
    OperatorCapabilities capabilities;
    capabilities.adjoint_apply = adjoint_kernel_ != nullptr;
    capabilities.asynchronous_submit =
        !arch_is_cpu(program_->compile_config().arch);
    capabilities.binding_rebind = true;
    capabilities.dense_storage_operands = true;
    capabilities.dense_storage_affine_operands = true;
    return capabilities;
  }

  OperatorResourceStamp current_stamp() const {
    return generations_->acquire().resource_stamp();
  }

  void publish(Ndarray *numeric_data,
               std::uint64_t numeric_version,
               std::uint64_t binding_revision) {
    ActionNdarrayOwner numeric_owner(program_, numeric_data);
    auto generation = std::make_shared<CompiledKernelActionGeneration>(
        program_, forward_kernel_, forward_compiled_, adjoint_kernel_,
        adjoint_compiled_, descriptor_, topology_, numeric_owner.get(),
        input_arg_index_, output_arg_index_);
    numeric_owner.release();
    const OperatorResourceStamp stamp{
        reinterpret_cast<std::uintptr_t>(program_),
        program_->runtime_program_generation(), 1, topology_version_,
        numeric_version, binding_revision};
    generation->stamp = stamp;
    current_generation_ = generation;
    const auto capabilities = make_capabilities();
    auto action = OperatorAction(
        descriptor_, capabilities, "forge_compiled_kernel_action",
        [stamp] { return stamp; },
        [generation = std::move(generation)](
            OperatorApplyMode mode, const OperatorVectorView &input,
            const OperatorVectorView &output) {
          generation->apply(mode, input, output);
        });
    generations_->publish(std::move(action));
  }

  Program *program_{nullptr};
  Kernel *forward_kernel_{nullptr};
  Kernel *adjoint_kernel_{nullptr};
  const CompiledKernelData *forward_compiled_{nullptr};
  const CompiledKernelData *adjoint_compiled_{nullptr};
  OperatorDescriptor descriptor_;
  std::shared_ptr<CompiledKernelActionTopology> topology_;
  std::shared_ptr<CompiledKernelActionGeneration> current_generation_;
  std::unique_ptr<OperatorResourceGenerationPublisher> generations_;
  DataType numeric_type_{PrimitiveType::unknown};
  std::vector<int> numeric_shape_;
  ExternalArrayLayout numeric_layout_{ExternalArrayLayout::kNull};
  bool has_numeric_data_{false};
  std::size_t input_arg_index_{2};
  std::size_t output_arg_index_{3};
  std::uint64_t topology_version_{0};
  std::uint64_t numeric_version_{0};
  std::uint64_t binding_revision_{1};
  std::mutex update_mutex_;
};

}  // namespace

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
    OperatorMathematicalTraits mathematical_traits) {
  const OperatorDescriptor descriptor{
      OperatorSpaceDesc{PrimitiveType::f32, domain_extent},
      OperatorSpaceDesc{PrimitiveType::f32, range_extent}};
  auto provider = std::make_shared<CompiledKernelActionProvider>(
      program, forward_kernel, adjoint_kernel, descriptor, topology_version,
      numeric_version, topology_data, numeric_data);
  auto binding = provider->binding().with_mathematical_traits(
      std::move(mathematical_traits));
  LinearOperatorHandle::NumericUpdateFn update;
  if (numeric_data) {
    update = [provider](
                 Program *update_program,
                 const LinearOperatorHandle::NumericUpdateArguments
                     &arguments,
                 std::uint64_t expected_topology_version,
                 std::uint64_t expected_numeric_version) {
      provider->update_numeric(update_program, arguments,
                               expected_topology_version,
                               expected_numeric_version);
    };
  }
  LinearOperatorHandle::RecordableKernelFn recordable =
      [provider](OperatorApplyMode mode) {
        return provider->recordable_kernel(mode);
      };
  return std::make_unique<LinearOperatorHandle>(
      program, std::move(binding), provider, std::move(update),
      std::move(recordable));
}

namespace {

using GraphFixedI32Arguments =
    std::unordered_map<std::string, std::int32_t>;
using GraphNdarrayArguments =
    std::unordered_map<std::string, const Ndarray *>;

struct GraphActionOwnedResource {
  std::string name;
  Ndarray *value{nullptr};
};

struct GraphActionNumericSpec {
  std::string name;
  DataType dtype{PrimitiveType::unknown};
  std::vector<int> shape;
  ExternalArrayLayout layout{ExternalArrayLayout::kNull};
};

struct GraphActionDefinition {
  GraphActionDefinition(Program *program,
                        const aot::CompiledGraph &forward_graph,
                        const aot::CompiledGraph *adjoint_graph)
      : program(program),
        program_lifetime(program ? program->weak_resource_lifetime_token()
                                 : std::weak_ptr<ProgramLifetimeToken>{}),
        forward(std::make_unique<aot::CompiledGraph>(forward_graph)) {
    if (adjoint_graph) {
      adjoint = std::make_unique<aot::CompiledGraph>(*adjoint_graph);
    }
  }

  ~GraphActionDefinition() {
    adjoint.reset();
    forward.reset();
    for (auto &resource : fixed_ndarrays) {
      Program::delete_ndarray_if_alive(program, program_lifetime,
                                       resource.value);
    }
  }

  aot::CompiledGraph &graph(OperatorApplyMode mode) const {
    if (mode == OperatorApplyMode::forward) {
      return *forward;
    }
    TI_ERROR_IF(!adjoint,
                "Compiled-Graph action has no explicit adjoint provider.");
    return *adjoint;
  }

  Program *program{nullptr};
  std::weak_ptr<ProgramLifetimeToken> program_lifetime;
  std::unique_ptr<aot::CompiledGraph> forward;
  std::unique_ptr<aot::CompiledGraph> adjoint;
  GraphFixedI32Arguments fixed_i32;
  std::vector<GraphActionOwnedResource> fixed_ndarrays;
  std::vector<GraphActionNumericSpec> numeric_specs;
  mutable std::mutex launch_mutex;
};

void validate_compiled_graph_action(
    Program *program,
    const aot::CompiledGraph &graph,
    const char *role,
    const GraphFixedI32Arguments &fixed_i32,
    const GraphNdarrayArguments &topology,
    const GraphNdarrayArguments &numeric,
    const GraphNdarrayArguments &workspace,
    const std::vector<SNodeTreeDependency> &state_dependencies) {
  const Arch arch = program->compile_config().arch;
  TI_ERROR_IF(graph.dispatches.empty(),
              "Compiled-Graph action {} requires at least one dispatch.",
              role);
  auto normalized_state = state_dependencies;
  auto graph_state = graph.snode_tree_dependencies;
  auto normalize_dependencies = [](auto *dependencies) {
    std::sort(dependencies->begin(), dependencies->end());
    dependencies->erase(
        std::unique(dependencies->begin(), dependencies->end()),
        dependencies->end());
  };
  normalize_dependencies(&normalized_state);
  normalize_dependencies(&graph_state);
  const bool state_matches =
      normalized_state.size() == graph_state.size() &&
      std::equal(
          normalized_state.begin(), normalized_state.end(),
          graph_state.begin(),
          [](const SNodeTreeDependency &declared,
             const SNodeTreeDependency &compiled) {
            return declared.tree_id == compiled.tree_id &&
                   declared.generation == compiled.generation &&
                   declared.layout_fingerprint ==
                       compiled.layout_fingerprint;
          });
  TI_ERROR_IF(
      !state_matches,
      "Compiled-Graph action {} SNode dependencies must exactly match "
      "the explicitly declared root-dense fixed Field state.",
      role);
  TI_ERROR_IF(
      irpass::analysis::has_non_dense_snode_tree_dependency(
          *program, graph.snode_tree_dependencies),
      "Compiled-Graph action {} fixed Field state requires purely dense "
      "SNodeTrees; a dependent tree contains sparse or dynamic SNodes.",
      role);
  for (const auto &dispatch : graph.dispatches) {
    TI_ERROR_IF(!dispatch.ti_kernel ||
                    dispatch.ti_kernel->program != program ||
                    dispatch.ti_kernel->arch != arch,
                "Compiled-Graph action {} dispatches must be JIT kernels "
                "owned by the same Program/backend.",
                role);
  }

  enum class ArgumentRole {
    input,
    output,
    fixed_i32,
    topology,
    numeric,
    workspace,
  };
  std::unordered_map<std::string, ArgumentRole> roles;
  auto register_role = [&](const std::string &name, ArgumentRole argument_role) {
    TI_ERROR_IF(name.empty() || !roles.emplace(name, argument_role).second,
                "Compiled-Graph action arguments must be non-empty and have "
                "one role; duplicate '{}'.",
                name);
  };
  register_role("input", ArgumentRole::input);
  register_role("output", ArgumentRole::output);
  for (const auto &[name, value] : fixed_i32) {
    (void)value;
    register_role(name, ArgumentRole::fixed_i32);
  }
  auto register_ndarrays = [&](const GraphNdarrayArguments &arguments,
                               ArgumentRole argument_role) {
    for (const auto &[name, source] : arguments) {
      TI_ERROR_IF(!source,
                  "Compiled-Graph action fixed ndarray '{}' is null.", name);
      validate_action_resource(program, *source, name.c_str());
      register_role(name, argument_role);
    }
  };
  register_ndarrays(topology, ArgumentRole::topology);
  register_ndarrays(numeric, ArgumentRole::numeric);
  register_ndarrays(workspace, ArgumentRole::workspace);
  TI_ERROR_IF(roles.size() != graph.args.size(),
              "Compiled-Graph action {} has {} arguments but {} explicit "
              "roles were supplied.",
              role, graph.args.size(), roles.size());

  auto source_for_role = [&](const std::string &name,
                             ArgumentRole argument_role) -> const Ndarray * {
    const GraphNdarrayArguments *arguments = nullptr;
    if (argument_role == ArgumentRole::topology) {
      arguments = &topology;
    } else if (argument_role == ArgumentRole::numeric) {
      arguments = &numeric;
    } else if (argument_role == ArgumentRole::workspace) {
      arguments = &workspace;
    }
    if (!arguments) {
      return nullptr;
    }
    const auto found = arguments->find(name);
    return found == arguments->end() ? nullptr : found->second;
  };
  for (const auto &[name, argument] : graph.args) {
    const auto role_it = roles.find(name);
    TI_ERROR_IF(role_it == roles.end(),
                "Compiled-Graph action argument '{}' has no explicit role.",
                name);
    const auto argument_role = role_it->second;
    if (argument_role == ArgumentRole::input ||
        argument_role == ArgumentRole::output) {
      TI_ERROR_IF(argument.tag != aot::ArgKind::kNdarray ||
                      argument.dtype() != PrimitiveType::f32 ||
                      !argument.element_shape.empty() ||
                      argument.field_dim != 1,
                  "Compiled-Graph action reserved '{}' argument must be a "
                  "scalar f32 one-dimensional ndarray.",
                  name);
    } else if (argument_role == ArgumentRole::fixed_i32) {
      TI_ERROR_IF(argument.tag != aot::ArgKind::kScalar ||
                      argument.dtype() != PrimitiveType::i32,
                  "Compiled-Graph action fixed scalar '{}' must be i32.",
                  name);
    } else {
      const Ndarray *source = source_for_role(name, argument_role);
      TI_ASSERT(source != nullptr);
      TI_ERROR_IF(argument.tag != aot::ArgKind::kNdarray ||
                      argument.dtype() != source->get_element_data_type() ||
                      !argument.element_shape.empty() ||
                      argument.field_dim != source->shape.size(),
                  "Compiled-Graph action fixed ndarray '{}' does not match "
                  "the graph dtype/rank declaration.",
                  name);
    }
  }
}

struct GraphActionExecutionState {
  explicit GraphActionExecutionState(OperatorExecutionKind kind,
                                     bool has_adjoint)
      : kind(kind) {
    if (kind == OperatorExecutionKind::compiled_graph ||
        kind == OperatorExecutionKind::runtime_capture) {
      forward_cache = std::make_unique<aot::CompiledGraphJITCache>();
      forward_cache->debug_graph_stats();
      if (has_adjoint) {
        adjoint_cache = std::make_unique<aot::CompiledGraphJITCache>();
        adjoint_cache->debug_graph_stats();
      }
    }
  }

  ~GraphActionExecutionState() {
    if (adjoint_cache) {
      adjoint_cache->clear_runtime_state();
    }
    if (forward_cache) {
      forward_cache->clear_runtime_state();
    }
  }

  aot::CompiledGraphJITCache *cache(OperatorApplyMode mode) const {
    return mode == OperatorApplyMode::forward ? forward_cache.get()
                                               : adjoint_cache.get();
  }

  OperatorExecutionKind kind{OperatorExecutionKind::direct};
  std::unique_ptr<aot::CompiledGraphJITCache> forward_cache;
  std::unique_ptr<aot::CompiledGraphJITCache> adjoint_cache;
  std::atomic<std::uint64_t> sequence_submissions{0};
  std::atomic<std::uint64_t> graph_submissions{0};
  std::atomic<std::uint64_t> capture_submissions{0};
};

OperatorBackendExecutionPath graph_action_backend_path(
    aot::CompiledGraphExecutionPath path) {
  switch (path) {
    case aot::CompiledGraphExecutionPath::ordinary_fallback:
      return OperatorBackendExecutionPath::ordinary_graph_fallback;
    case aot::CompiledGraphExecutionPath::cuda_capture:
      return OperatorBackendExecutionPath::cuda_capture;
    case aot::CompiledGraphExecutionPath::cuda_exact_replay:
      return OperatorBackendExecutionPath::cuda_exact_replay;
    case aot::CompiledGraphExecutionPath::cuda_patched_replay:
      return OperatorBackendExecutionPath::cuda_patched_replay;
    case aot::CompiledGraphExecutionPath::cuda_masked_capture:
      return OperatorBackendExecutionPath::cuda_capture;
    case aot::CompiledGraphExecutionPath::cuda_masked_replay:
      return OperatorBackendExecutionPath::cuda_exact_replay;
    case aot::CompiledGraphExecutionPath::cuda_masked_patched_replay:
      return OperatorBackendExecutionPath::cuda_patched_replay;
    case aot::CompiledGraphExecutionPath::cuda_device_update_nested_capture:
      return OperatorBackendExecutionPath::cuda_capture;
    case aot::CompiledGraphExecutionPath::cuda_device_update_nested_replay:
      return OperatorBackendExecutionPath::cuda_exact_replay;
    case aot::CompiledGraphExecutionPath::
        cuda_device_update_nested_patched_replay:
      return OperatorBackendExecutionPath::cuda_patched_replay;
    case aot::CompiledGraphExecutionPath::vulkan_record:
      return OperatorBackendExecutionPath::vulkan_record;
    case aot::CompiledGraphExecutionPath::vulkan_replay:
      return OperatorBackendExecutionPath::vulkan_replay;
    case aot::CompiledGraphExecutionPath::vulkan_patched_replay:
      return OperatorBackendExecutionPath::vulkan_replay;
    case aot::CompiledGraphExecutionPath::none:
      return OperatorBackendExecutionPath::unavailable;
  }
  TI_UNREACHABLE;
}

struct GraphActionGeneration {
  GraphActionGeneration(
      Program *program,
      std::shared_ptr<GraphActionDefinition> definition,
      std::vector<GraphActionOwnedResource> numeric_ndarrays)
      : program(program),
        program_lifetime(program ? program->weak_resource_lifetime_token()
                                 : std::weak_ptr<ProgramLifetimeToken>{}),
        definition(std::move(definition)),
        numeric_ndarrays(std::move(numeric_ndarrays)) {
  }

  ~GraphActionGeneration() {
    for (auto &resource : numeric_ndarrays) {
      Program::delete_ndarray_if_alive(program, program_lifetime,
                                       resource.value);
    }
  }

  void apply(OperatorApplyMode mode,
             const OperatorVectorView &input,
             const OperatorVectorView &output,
             const std::shared_ptr<GraphActionExecutionState> &state) {
    const auto valid_operand = [](const OperatorVectorView &view) {
      return view.ndarray != nullptr || view.runtime_storage != nullptr;
    };
    TI_ERROR_IF(!valid_operand(input) || !valid_operand(output),
                "Compiled-Graph actions require ndarray or runtime-storage "
                "views.");
    std::lock_guard<std::mutex> lock(definition->launch_mutex);
    std::unordered_map<std::string, aot::IValue> arguments;
    arguments.reserve(2 + definition->fixed_i32.size() +
                      definition->fixed_ndarrays.size() +
                      numeric_ndarrays.size());
    const auto graph_value = [](const OperatorVectorView &view) {
      if (view.runtime_storage) {
        return aot::IValue::create(*view.runtime_storage);
      }
      TI_ASSERT(view.ndarray != nullptr);
      return aot::IValue::create(*view.ndarray);
    };
    arguments.emplace("input", graph_value(input));
    arguments.emplace("output", graph_value(output));
    for (const auto &[name, value] : definition->fixed_i32) {
      arguments.emplace(name, aot::IValue::create(value));
    }
    for (const auto &resource : definition->fixed_ndarrays) {
      arguments.emplace(resource.name,
                        aot::IValue::create(*resource.value));
    }
    for (const auto &resource : numeric_ndarrays) {
      arguments.emplace(resource.name,
                        aot::IValue::create(*resource.value));
    }
    auto &graph = definition->graph(mode);
    if (state->kind == OperatorExecutionKind::direct ||
        state->kind == OperatorExecutionKind::explicit_sequence) {
      state->sequence_submissions.fetch_add(1, std::memory_order_relaxed);
      graph.jit_run(program->compile_config(), arguments);
      return;
    }
    auto *cache = state->cache(mode);
    TI_ERROR_IF(!cache,
                "Compiled-Graph action execution cache is unavailable.");
    if (state->kind == OperatorExecutionKind::compiled_graph) {
      state->graph_submissions.fetch_add(1, std::memory_order_relaxed);
    } else {
      state->capture_submissions.fetch_add(1, std::memory_order_relaxed);
    }
    graph.jit_run_cached(program->compile_config(), arguments, *cache);
  }

  Program *program{nullptr};
  std::weak_ptr<ProgramLifetimeToken> program_lifetime;
  std::shared_ptr<GraphActionDefinition> definition;
  std::vector<GraphActionOwnedResource> numeric_ndarrays;
  OperatorResourceStamp stamp;
};

class CompiledGraphActionProvider {
 public:
  CompiledGraphActionProvider(
      Program *program,
      const aot::CompiledGraph &forward_graph,
      const aot::CompiledGraph *adjoint_graph,
      OperatorDescriptor descriptor,
      std::uint64_t topology_version,
      std::uint64_t numeric_version,
      GraphFixedI32Arguments fixed_i32,
      GraphNdarrayArguments topology,
      GraphNdarrayArguments numeric,
      GraphNdarrayArguments workspace,
      std::vector<SNodeTreeDependency> state_dependencies)
      : program_(program),
        descriptor_(std::move(descriptor)),
        topology_version_(topology_version),
        numeric_version_(numeric_version) {
    TI_ERROR_IF(!program_ || topology.empty() ||
                    descriptor_.domain.scalar_extent == 0 ||
                    descriptor_.range.scalar_extent == 0 ||
                    topology_version_ == 0 || numeric_version_ == 0,
                "Compiled-Graph actions require an owning Program, positive "
                "domain/range extents and versions, and topology resources.");
    TI_ERROR_IF(descriptor_.domain.scalar_type != PrimitiveType::f32 ||
                    descriptor_.range.scalar_type != PrimitiveType::f32 ||
                    !descriptor_.domain.entry_shape.empty() ||
                    !descriptor_.range.entry_shape.empty(),
                "Compiled-Graph actions currently require scalar f32 domain "
                "and range spaces.");
    const Arch arch = program_->compile_config().arch;
    TI_ERROR_IF(!arch_is_cpu(arch) && !arch_is_cuda(arch) &&
                    arch != Arch::vulkan,
                "Compiled-Graph actions support CPU, CUDA, and Vulkan only; "
                "got {}. No fallback was performed.",
                arch_name(arch));
    validate_compiled_graph_action(program_, forward_graph, "forward",
                                   fixed_i32, topology, numeric, workspace,
                                   state_dependencies);
    if (adjoint_graph) {
      validate_compiled_graph_action(program_, *adjoint_graph, "adjoint",
                                     fixed_i32, topology, numeric, workspace,
                                     state_dependencies);
    }
    definition_ = std::make_shared<GraphActionDefinition>(
        program_, forward_graph, adjoint_graph);
    definition_->fixed_i32 = std::move(fixed_i32);
    std::sort(state_dependencies.begin(), state_dependencies.end());
    state_dependencies.erase(
        std::unique(state_dependencies.begin(), state_dependencies.end()),
        state_dependencies.end());
    state_dependencies_ = std::move(state_dependencies);
    has_adjoint_ = adjoint_graph != nullptr;
    execution_state_ = std::make_shared<GraphActionExecutionState>(
        arch_is_cpu(arch) ? OperatorExecutionKind::explicit_sequence
                          : OperatorExecutionKind::compiled_graph,
        has_adjoint_);

    auto snapshot_fixed = [&](const GraphNdarrayArguments &arguments) {
      for (const auto &[name, source] : arguments) {
        Ndarray *owned = program_->create_ndarray(
            source->get_element_data_type(), source->shape, source->layout,
            false);
        try {
          program_->copy_ndarray_fast(owned, const_cast<Ndarray *>(source));
          definition_->fixed_ndarrays.push_back({name, owned});
        } catch (...) {
          program_->delete_ndarray(owned);
          throw;
        }
      }
    };
    snapshot_fixed(topology);
    snapshot_fixed(workspace);
    for (const auto &[name, source] : numeric) {
      definition_->numeric_specs.push_back(
          {name, source->get_element_data_type(), source->shape,
           source->layout});
    }
    generations_ =
        std::make_unique<OperatorResourceGenerationPublisher>();
    publish(snapshot_numeric(numeric), numeric_version_, binding_revision_);
  }

  ~CompiledGraphActionProvider() {
    try {
      if (generations_) {
        generations_->retire_current();
        generations_.reset();
      }
      execution_state_.reset();
      definition_.reset();
    } catch (...) {
    }
  }

  OperatorBinding binding() {
    const Arch arch = program_->compile_config().arch;
    const OperatorExecutionKind execution_kind =
        arch_is_cpu(arch) ? OperatorExecutionKind::explicit_sequence
                          : OperatorExecutionKind::compiled_graph;
    const auto capabilities = make_capabilities();
    auto state = execution_state_;
    TI_ASSERT(state && state->kind == execution_kind);
    auto metadata_action = OperatorAction(
        descriptor_, capabilities, "forge_compiled_graph_action",
        [this] { return current_stamp(); },
        [this, state](OperatorApplyMode mode,
                      const OperatorVectorView &input,
                      const OperatorVectorView &output) {
          auto generation = generations_->acquire();
          generation.apply_overwrite(mode, input, output);
        });
    auto binding = OperatorBinding::from_generation_publisher(
        std::move(metadata_action),
        [this] { return generations_->acquire(); });
    return binding.with_execution_lowering(
        execution_kind, [state] { return execution_statistics(state); });
  }

  bool has_numeric_resources() const {
    return !definition_->numeric_specs.empty();
  }

  std::shared_ptr<LinearOperatorRecordableKernel> recordable_kernel(
      OperatorApplyMode mode) {
    std::lock_guard<std::mutex> lock(update_mutex_);
    TI_ERROR_IF(!current_generation_,
                "Compiled-Graph action has no published generation.");
    auto &graph = definition_->graph(mode);
    TI_ERROR_IF(
        graph.has_indirect_dispatches(),
        "Compiled-Graph actions with indirect dispatch cannot be inlined "
        "into an outer Graph yet; rebuild the provider with direct "
        "dispatches.");
    LinearOperatorRecordableKernel::FixedNdarrayArguments fixed_ndarrays;
    fixed_ndarrays.reserve(definition_->fixed_ndarrays.size() +
                           current_generation_->numeric_ndarrays.size());
    for (const auto &resource : definition_->fixed_ndarrays) {
      fixed_ndarrays.emplace_back(resource.name, resource.value);
    }
    for (const auto &resource : current_generation_->numeric_ndarrays) {
      fixed_ndarrays.emplace_back(resource.name, resource.value);
    }
    return std::make_shared<LinearOperatorRecordableKernel>(
        program_, &graph, definition_->fixed_i32, std::move(fixed_ndarrays),
        state_dependencies_, current_generation_->stamp,
        current_generation_);
  }

  void update_numeric(
      Program *program,
      const LinearOperatorHandle::NumericUpdateArguments
          &arguments,
      std::uint64_t expected_topology_version,
      std::uint64_t expected_numeric_version) {
    TI_ERROR_IF(program != program_ || !has_numeric_resources(),
                "Compiled-Graph action numeric update requires its owning "
                "Program and numeric resources.");
    std::lock_guard<std::mutex> lock(update_mutex_);
    TI_ERROR_IF(expected_topology_version != topology_version_ ||
                    expected_numeric_version != numeric_version_,
                "Compiled-Graph action numeric update version mismatch: "
                "expected topology/numeric ({}, {}), current ({}, {}).",
                expected_topology_version, expected_numeric_version,
                topology_version_, numeric_version_);
    TI_ERROR_IF(numeric_version_ ==
                        (std::numeric_limits<std::uint64_t>::max)() ||
                    binding_revision_ ==
                        (std::numeric_limits<std::uint64_t>::max)(),
                "Compiled-Graph action resource version overflow.");
    GraphNdarrayArguments native_arguments;
    native_arguments.reserve(arguments.size());
    for (const auto &[name, value] : arguments) {
      native_arguments.emplace(name, value);
    }
    const auto next_numeric = numeric_version_ + 1;
    const auto next_binding = binding_revision_ + 1;
    publish(snapshot_numeric(native_arguments), next_numeric, next_binding);
    numeric_version_ = next_numeric;
    binding_revision_ = next_binding;
  }

 private:
  static OperatorBinding::ExecutionRuntimeStatistics execution_statistics(
      const std::shared_ptr<GraphActionExecutionState> &state) {
    OperatorBinding::ExecutionRuntimeStatistics result;
    result.sequence_submissions =
        state->sequence_submissions.load(std::memory_order_relaxed);
    result.compiled_graph_submissions =
        state->graph_submissions.load(std::memory_order_relaxed);
    result.runtime_capture_submissions =
        state->capture_submissions.load(std::memory_order_relaxed);
    if (state->kind == OperatorExecutionKind::explicit_sequence) {
      result.last_backend_path =
          OperatorBackendExecutionPath::explicit_sequence;
      return result;
    }
    auto accumulate_cache = [&](aot::CompiledGraphJITCache *cache) {
      if (!cache) {
        return;
      }
      const auto snapshot = cache->debug_graph_stats();
      if (snapshot.stats.last_path != aot::CompiledGraphExecutionPath::none) {
        result.last_backend_path =
            graph_action_backend_path(snapshot.stats.last_path);
      }
      result.backend_captures +=
          snapshot.stats.captures + snapshot.stats.records;
      result.backend_replays += snapshot.stats.exact_replays +
                                snapshot.stats.patched_replays +
                                snapshot.stats.replays;
      result.ordinary_fallbacks += snapshot.stats.ordinary_fallbacks;
    };
    accumulate_cache(state->forward_cache.get());
    accumulate_cache(state->adjoint_cache.get());
    return result;
  }

  OperatorCapabilities make_capabilities() const {
    const Arch arch = program_->compile_config().arch;
    OperatorCapabilities capabilities;
    capabilities.adjoint_apply = has_adjoint_;
    // The public handle is synchronous. Async reuse of mutable Graph
    // workspace has not been qualified for this new provider.
    capabilities.asynchronous_submit = false;
    capabilities.explicit_sequence = true;
    capabilities.compiled_graph = arch_is_cuda(arch) || arch == Arch::vulkan;
    capabilities.runtime_capture = arch_is_cuda(arch);
    capabilities.binding_rebind = true;
    capabilities.persistent_workspace = true;
    capabilities.dense_storage_operands = true;
    capabilities.dense_storage_affine_operands = true;
    return capabilities;
  }

  std::vector<GraphActionOwnedResource> snapshot_numeric(
      const GraphNdarrayArguments &arguments) const {
    TI_ERROR_IF(arguments.size() != definition_->numeric_specs.size(),
                "Compiled-Graph action numeric update must provide exactly "
                "the declared resource names.");
    std::vector<GraphActionOwnedResource> owned_resources;
    try {
      for (const auto &spec : definition_->numeric_specs) {
        const auto found = arguments.find(spec.name);
        TI_ERROR_IF(found == arguments.end() || !found->second,
                    "Compiled-Graph action numeric resource '{}' is missing.",
                    spec.name);
        const Ndarray &source = *found->second;
        validate_action_resource(program_, source, spec.name.c_str());
        TI_ERROR_IF(source.get_element_data_type() != spec.dtype ||
                        source.shape != spec.shape ||
                        source.layout != spec.layout,
                    "Compiled-Graph action numeric resource '{}' must "
                    "preserve dtype, shape, and layout.",
                    spec.name);
        Ndarray *owned = program_->create_ndarray(
            spec.dtype, spec.shape, spec.layout, false);
        try {
          program_->copy_ndarray_fast(owned,
                                      const_cast<Ndarray *>(&source));
          owned_resources.push_back({spec.name, owned});
        } catch (...) {
          program_->delete_ndarray(owned);
          throw;
        }
      }
    } catch (...) {
      for (auto &resource : owned_resources) {
        program_->delete_ndarray(resource.value);
      }
      throw;
    }
    return owned_resources;
  }

  OperatorResourceStamp current_stamp() const {
    return generations_->acquire().resource_stamp();
  }

  void publish(std::vector<GraphActionOwnedResource> numeric_resources,
               std::uint64_t numeric_version,
               std::uint64_t binding_revision) {
    std::shared_ptr<GraphActionGeneration> generation;
    try {
      generation = std::make_shared<GraphActionGeneration>(
          program_, definition_, std::move(numeric_resources));
    } catch (...) {
      for (auto &resource : numeric_resources) {
        if (resource.value) {
          program_->delete_ndarray(resource.value);
        }
      }
      throw;
    }
    const OperatorResourceStamp stamp{
        reinterpret_cast<std::uintptr_t>(program_),
        program_->runtime_program_generation(), 1, topology_version_,
        numeric_version, binding_revision};
    generation->stamp = stamp;
    current_generation_ = generation;
    const auto capabilities = make_capabilities();
    auto state = execution_state_;
    TI_ASSERT(state != nullptr);
    auto action = OperatorAction(
        descriptor_, capabilities, "forge_compiled_graph_action",
        [stamp] { return stamp; },
        [generation, state](
            OperatorApplyMode mode, const OperatorVectorView &input,
            const OperatorVectorView &output) {
          generation->apply(mode, input, output, state);
        });
    generations_->publish(std::move(action));
  }

  Program *program_{nullptr};
  OperatorDescriptor descriptor_;
  std::shared_ptr<GraphActionDefinition> definition_;
  std::unique_ptr<OperatorResourceGenerationPublisher> generations_;
  std::shared_ptr<GraphActionExecutionState> execution_state_;
  std::shared_ptr<GraphActionGeneration> current_generation_;
  std::vector<SNodeTreeDependency> state_dependencies_;
  bool has_adjoint_{false};
  std::uint64_t topology_version_{0};
  std::uint64_t numeric_version_{0};
  std::uint64_t binding_revision_{1};
  std::mutex update_mutex_;
};

}  // namespace

std::unique_ptr<LinearOperatorHandle>
make_compiled_graph_operator_handle(
    Program *program,
    const aot::CompiledGraph &forward_graph,
    const aot::CompiledGraph *adjoint_graph,
    std::size_t range_extent,
    std::size_t domain_extent,
    std::uint64_t topology_version,
    std::uint64_t numeric_version,
    GraphFixedI32Arguments fixed_i32_arguments,
    GraphNdarrayArguments topology_arguments,
    GraphNdarrayArguments numeric_arguments,
    GraphNdarrayArguments workspace_arguments,
    std::vector<SNodeTreeDependency> state_dependencies,
    OperatorMathematicalTraits mathematical_traits) {
  const OperatorDescriptor descriptor{
      OperatorSpaceDesc{PrimitiveType::f32, domain_extent},
      OperatorSpaceDesc{PrimitiveType::f32, range_extent}};
  auto provider = std::make_shared<CompiledGraphActionProvider>(
      program, forward_graph, adjoint_graph, descriptor, topology_version,
      numeric_version, std::move(fixed_i32_arguments),
      std::move(topology_arguments), std::move(numeric_arguments),
      std::move(workspace_arguments), std::move(state_dependencies));
  auto binding = provider->binding().with_mathematical_traits(
      std::move(mathematical_traits));
  LinearOperatorHandle::NumericUpdateFn update;
  if (provider->has_numeric_resources()) {
    update = [provider](
                 Program *update_program,
                 const LinearOperatorHandle::NumericUpdateArguments
                     &arguments,
                 std::uint64_t expected_topology_version,
                 std::uint64_t expected_numeric_version) {
      provider->update_numeric(update_program, arguments,
                               expected_topology_version,
                               expected_numeric_version);
    };
  }
  LinearOperatorHandle::RecordableKernelFn recordable =
      [provider](OperatorApplyMode mode) {
        return provider->recordable_kernel(mode);
      };
  return std::make_unique<LinearOperatorHandle>(
      program, std::move(binding), provider, std::move(update),
      std::move(recordable));
}

std::unique_ptr<LinearOperatorHandle>
make_linear_operator_handle(
    Program *program,
    SparseMatrix &matrix,
    OperatorMathematicalTraits mathematical_traits) {
  auto binding = make_program_sparse_operator_binding(program, matrix)
                     .with_mathematical_traits(
                         std::move(mathematical_traits));
  LinearOperatorHandle::RecordableKernelFn recordable;
  if (auto *compiled =
          dynamic_cast<CompiledKernelLinearOperator *>(&matrix)) {
    recordable = [compiled](OperatorApplyMode mode) {
      TI_ERROR_IF(mode != OperatorApplyMode::forward,
                  "Square compiled-kernel operators do not expose an "
                  "implicit adjoint recordable action.");
      return compiled->recordable_kernel();
    };
  } else if (auto *compiled_graph =
                 dynamic_cast<CompiledGraphLinearOperator *>(&matrix)) {
    recordable = [compiled_graph](OperatorApplyMode mode) {
      TI_ERROR_IF(mode != OperatorApplyMode::forward,
                  "Square compiled-Graph operators do not expose an "
                  "implicit adjoint recordable action.");
      return compiled_graph->recordable_kernel();
    };
  }
  return std::make_unique<LinearOperatorHandle>(
      program, std::move(binding), std::shared_ptr<void>{},
      LinearOperatorHandle::NumericUpdateFn{}, std::move(recordable));
}

std::unique_ptr<LinearOperatorHandle>
make_identity_operator_handle(Program *program,
                                           OperatorSpaceDesc space) {
  return std::make_unique<LinearOperatorHandle>(
      program, make_identity_operator_binding(std::move(space), program));
}

std::unique_ptr<LinearOperatorHandle>
make_adjoint_operator_handle(
    LinearOperatorHandle &operand) {
  return std::make_unique<LinearOperatorHandle>(
      operand.program(), make_adjoint_operator_binding(operand.binding()));
}

std::unique_ptr<LinearOperatorHandle>
make_scaled_operator_handle(
    double scale,
    LinearOperatorHandle &operand) {
  return std::make_unique<LinearOperatorHandle>(
      operand.program(),
      make_scaled_operator_binding(scale, operand.binding(),
                                   operand.program()));
}

namespace {

void validate_same_public_operator_program(
    const LinearOperatorHandle &left,
    const LinearOperatorHandle &right,
    const char *operation) {
  TI_ERROR_IF(left.program() != right.program(),
              "LinearOperator {} operands must belong to the same Program.",
              operation);
}

}  // namespace

std::unique_ptr<LinearOperatorHandle>
make_sum_operator_handle(
    LinearOperatorHandle &left,
    LinearOperatorHandle &right) {
  validate_same_public_operator_program(left, right, "sum");
  return std::make_unique<LinearOperatorHandle>(
      left.program(),
      make_sum_operator_binding(left.binding(), right.binding(),
                                left.program()));
}

std::unique_ptr<LinearOperatorHandle>
make_composed_operator_handle(
    LinearOperatorHandle &outer,
    LinearOperatorHandle &inner) {
  validate_same_public_operator_program(outer, inner, "composition");
  return std::make_unique<LinearOperatorHandle>(
      outer.program(),
      make_composed_operator_binding(outer.binding(), inner.binding(),
                                     outer.program()));
}

std::unique_ptr<LinearOperatorHandle>
make_block_diagonal_operator_handle(
    const std::vector<LinearOperatorHandle *> &blocks) {
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
  return std::make_unique<LinearOperatorHandle>(
      program, make_block_diagonal_operator_binding(std::move(bindings)));
}

std::unique_ptr<ExperimentalPreconditionerPlanHandle>
make_experimental_preconditioner_plan_handle(
    Program *program,
    LinearOperatorHandle &target,
    LinearOperatorHandle &action,
    std::string method) {
  return std::make_unique<ExperimentalPreconditionerPlanHandle>(
      program, target, action, std::move(method));
}

std::unique_ptr<LinearOperatorHandle>
make_experimental_preconditioner_action_handle(
    Program *program,
    ExperimentalPreconditionerPlanHandle &plan) {
  return std::make_unique<LinearOperatorHandle>(
      program, plan.consumer_binding());
}

}  // namespace taichi::lang
