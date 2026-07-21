#include "taichi/program/linear_operator.h"

#include <algorithm>
#include <atomic>
#include <cmath>
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
  TI_ERROR_IF(view.space != expected ||
                  (view.data == 0 && !view.ndarray) ||
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
  OperatorCapabilities capabilities;
  std::string provider_name;
  ResourceStampFn resource_stamp;
  OverwriteApplyFn overwrite_apply;
};

OperatorAction::OperatorAction(OperatorDescriptor descriptor,
                               OperatorCapabilities capabilities,
                               std::string provider_name,
                               ResourceStampFn resource_stamp,
                               OverwriteApplyFn overwrite_apply) {
  validate_space(descriptor.domain, "domain");
  validate_space(descriptor.range, "range");
  TI_ERROR_IF(provider_name.empty() || !resource_stamp || !overwrite_apply ||
                  !capabilities.forward_apply,
              "OperatorAction requires a named forward provider, resource "
              "stamp, and overwrite apply function.");
  state_ = std::make_shared<State>(
      State{std::move(descriptor), capabilities, std::move(provider_name),
            std::move(resource_stamp), std::move(overwrite_apply)});
}

const OperatorDescriptor &OperatorAction::descriptor() const {
  return state_->descriptor;
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

OperatorResourceLease::OperatorResourceLease(std::shared_ptr<void> state)
    : state_(std::move(state)) {
}

OperatorPinnedAction::OperatorPinnedAction(
    OperatorAction action,
    OperatorResourceStamp stamp,
    OperatorResourceLease resource_lease)
    : action_(std::make_shared<OperatorAction>(std::move(action))),
      stamp_(stamp),
      resource_lease_(std::move(resource_lease)) {
}

OperatorPinnedAction::operator bool() const {
  return action_ != nullptr;
}

const OperatorDescriptor &OperatorPinnedAction::descriptor() const {
  TI_ERROR_IF(!action_, "Operator generation pin is empty.");
  return action_->descriptor();
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
    : impl_(std::make_unique<Impl>(
          allocate_operator_generation_domain())) {
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
  TI_ERROR_IF((stamp.program_identity == 0) !=
                  (stamp.program_generation == 0),
              "Operator generation Program identity and generation must "
              "either both be present or both be absent.");
  std::lock_guard<std::mutex> lock(impl_->mutex);
  auto [result, next] = impl_->registry.emplace(
      kOperatorGenerationKind, std::move(action), stamp,
      std::move(resources));
  TI_ERROR_IF(result != OperatorGenerationRegistry::Result::kSuccess,
              "Unable to publish immutable operator resource generation.");
  const auto previous = impl_->current;
  impl_->current = next;
  if (previous) {
    const auto retire_result = impl_->registry.retire(previous);
    TI_ERROR_IF(retire_result !=
                    OperatorGenerationRegistry::Result::kSuccess,
                "Unable to retire replaced operator resource generation.");
  }
}

OperatorPinnedAction OperatorResourceGenerationPublisher::acquire() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  TI_ERROR_IF(!impl_->current,
              "Operator resource generation publisher has no current "
              "generation.");
  auto [result, lease] = impl_->registry.acquire(impl_->current);
  TI_ERROR_IF(result != OperatorGenerationRegistry::Result::kSuccess ||
                  !lease,
              "Unable to pin current operator resource generation.");
  auto action = lease->action;
  const auto stamp = lease->stamp;
  return OperatorPinnedAction(
      std::move(action), stamp,
      OperatorResourceLease::hold(std::move(lease)));
}

void OperatorResourceGenerationPublisher::retire_current() {
  if (!impl_) {
    return;
  }
  std::lock_guard<std::mutex> lock(impl_->mutex);
  const auto current = std::exchange(
      impl_->current, OperatorGenerationRegistry::Handle{});
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

OperatorBinding::OperatorBinding(
    OperatorAction action,
    AcquireResourceLeaseFn acquire_resource_lease)
    : action_(std::move(action)),
      acquire_resource_lease_(std::move(acquire_resource_lease)) {
}

OperatorBinding::OperatorBinding(
    OperatorAction metadata_action,
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
  return OperatorPinnedAction(action_, action_.resource_stamp(),
                              acquire_resource_lease());
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
    : OperatorPlan(program, OperatorBinding(std::move(action)),
                   dependencies) {
}

OperatorPlan::OperatorPlan(Program *program,
                           OperatorBinding binding,
                           OperatorDependencyMask dependencies)
    : program_(program),
      binding_(std::move(binding)),
      dependencies_(dependencies),
      planned_stamp_(binding_.action().resource_stamp()) {
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

const OperatorCapabilities &OperatorPlan::capabilities() const {
  return binding_.action().capabilities();
}

const std::string &OperatorPlan::provider_name() const {
  return binding_.action().provider_name();
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
                  pinned.provider_name() != provider_name(),
              "Operator generation changed descriptor or provider identity "
              "without rebuilding its binding.");
  const auto stamp = pinned.resource_stamp();
  const auto invalidation = evaluate_operator_plan_invalidation(
      planned_stamp_, stamp, dependencies_);
  statistics_.generation_pins++;
  if (has_pinned_generation_) {
    const auto changes =
        operator_resource_changes(last_pinned_stamp_, stamp);
    if (changes != 0) {
      statistics_.generation_changes++;
    }
    if (changes & operator_dependency(
                      OperatorResourceDependency::numeric)) {
      statistics_.numeric_generation_changes++;
    }
    if (changes & operator_dependency(
                      OperatorResourceDependency::binding)) {
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
  TI_ERROR_IF(
      invalidation.kind == OperatorPlanInvalidationKind::rebuild,
      "OperatorPlan dependencies changed and require plan rebuild.");
  TI_ERROR_IF(
      invalidation.kind == OperatorPlanInvalidationKind::refresh_binding,
      "OperatorPlan binding identity changed and requires refresh.");
  return pinned;
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
  return submit(pinned, request);
}

OperatorSubmission OperatorPlan::submit(
    const OperatorPinnedAction &pinned,
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

  statistics_.submissions++;
  const auto &action = pinned;
  if (request.alpha == 1.0 && request.beta == 0.0) {
    action.apply_overwrite(request.mode, request.input, request.output);
    statistics_.primitive_apply_calls++;
    return {pinned.resource_stamp(), true};
  }

  TI_ERROR_IF(program_ &&
                  !arch_is_cpu(program_->compile_config().arch),
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
  return {pinned.resource_stamp(), true};
}

OperatorPlanRuntimeStatistics OperatorPlan::debug_runtime_statistics() const {
  return statistics_;
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

namespace {

template <typename Provider>
OperatorBinding make_cpu_typed_operator_binding(
    Program *program,
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
  descriptor.domain = {
      provider.get_data_type(),
      static_cast<std::size_t>(provider.num_cols())};
  descriptor.range = {
      provider.get_data_type(),
      static_cast<std::size_t>(provider.num_rows())};
  auto action = OperatorAction(
      descriptor, OperatorCapabilities{}, expected_provider,
      [program, &provider] {
        const auto statistics = provider.debug_runtime_statistics();
        return OperatorResourceStamp{
            reinterpret_cast<std::uintptr_t>(program),
            program->runtime_program_generation(),
            1,
            statistics.pattern_version,
            statistics.numeric_version,
            provider.matrix_id()};
      },
      [program, &provider](OperatorApplyMode mode,
                           const OperatorVectorView &input,
                           const OperatorVectorView &output) {
        TI_ERROR_IF(mode != OperatorApplyMode::forward || !input.ndarray ||
                        !output.ndarray,
                    "CPU operator binding requires forward ndarray views; "
                    "no fallback was performed.");
        provider.nd_spmv(program, *input.ndarray, *output.ndarray);
      });
  return OperatorBinding(
      std::move(action), [&provider] {
        return OperatorResourceLease::hold(
            provider.acquire_numeric_access_guard());
      });
}

template <typename Provider, typename Apply>
OperatorBinding make_gpu_typed_operator_binding(
    Program *program,
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
  descriptor.domain = {
      provider.get_data_type(),
      static_cast<std::size_t>(provider.num_cols())};
  descriptor.range = {
      provider.get_data_type(),
      static_cast<std::size_t>(provider.num_rows())};
  OperatorCapabilities capabilities;
  capabilities.asynchronous_submit = true;
  auto action = OperatorAction(
      descriptor, capabilities, expected_provider,
      [program, &provider] {
        const auto statistics = provider.debug_runtime_statistics();
        return OperatorResourceStamp{
            reinterpret_cast<std::uintptr_t>(program),
            program->runtime_program_generation(),
            1,
            statistics.pattern_version,
            statistics.numeric_version,
            provider.matrix_id()};
      },
      [apply = std::move(apply)](
          OperatorApplyMode mode, const OperatorVectorView &input,
          const OperatorVectorView &output) {
        TI_ERROR_IF(mode != OperatorApplyMode::forward,
                    "GPU sparse operator bindings support forward apply "
                    "only.");
        apply(input, output);
      });
  return OperatorBinding(
      std::move(action), [&provider] {
        return OperatorResourceLease::hold(
            provider.acquire_numeric_access_guard());
      });
}

}  // namespace

OperatorBinding make_cpu_csr_operator_binding(Program *program,
                                              CpuSparseCsrMatrix &matrix) {
  return make_cpu_typed_operator_binding(
      program, matrix, "forge_cpu_native", "csr");
}

OperatorBinding make_cpu_bsr_operator_binding(Program *program,
                                              CpuSparseBsrMatrix &matrix) {
  return make_cpu_typed_operator_binding(
      program, matrix, "forge_cpu_native", "bsr");
}

OperatorBinding make_cpu_program_kernel_operator_binding(
    Program *program,
    CompiledKernelLinearOperator &matrix) {
  TI_ERROR_IF(matrix.owning_program() != program,
              "CPU program-kernel operator binding requires its owning "
              "Program; no fallback was performed.");
  return matrix.make_operator_binding();
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

OperatorBinding make_vulkan_csr_operator_binding(
    Program *program,
    VulkanSparseMatrix &matrix) {
  return make_gpu_typed_operator_binding(
      program, matrix, Arch::vulkan, "vulkan", "forge_vulkan_native",
      "csr", [program, &matrix](const OperatorVectorView &input,
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
      program, matrix, Arch::vulkan, "vulkan", "forge_vulkan_native",
      "bsr", [program, &matrix](const OperatorVectorView &input,
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

}  // namespace taichi::lang
