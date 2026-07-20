#include "taichi/program/linear_operator.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "taichi/common/core.h"
#include "taichi/program/ndarray.h"
#include "taichi/program/program.h"

namespace taichi::lang {
namespace {

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
  TI_ERROR_IF(view.space != expected || view.data == 0 ||
                  view.allocation_identity == 0 ||
                  (require_writable && !view.writable),
              "Operator {} view does not match its declared space or "
              "access mode.",
              role);
  if (program) {
    TI_ERROR_IF(view.program != program || !view.ndarray,
                "Program-bound operator {} view must retain an ndarray "
                "owned by the plan Program.",
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

struct OperatorPlan::Scratch {
  OperatorSpaceDesc space;
  Ndarray *array{nullptr};
  std::vector<std::uint64_t> host_words;
  OperatorVectorView view;
};

OperatorPlan::OperatorPlan(Program *program, OperatorAction action)
    : program_(program), action_(std::move(action)) {
  TI_ERROR_IF(program_ && !arch_is_cpu(program_->compile_config().arch),
              "M1 generalized operator lowering currently requires a CPU "
              "Program; no host fallback was performed.");
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
  return action_.descriptor();
}

const OperatorCapabilities &OperatorPlan::capabilities() const {
  return action_.capabilities();
}

const std::string &OperatorPlan::provider_name() const {
  return action_.provider_name();
}

OperatorResourceStamp OperatorPlan::resource_stamp() const {
  return action_.resource_stamp();
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
  if (request.alpha == 1.0 && request.beta == 0.0) {
    action_.apply_overwrite(request.mode, request.input, request.output);
    statistics_.primitive_apply_calls++;
    return {action_.resource_stamp(), true};
  }

  statistics_.generalized_lowerings++;
  OperatorVectorView applied;
  OperatorVectorView *applied_ptr = nullptr;
  if (request.alpha != 0.0) {
    applied = scratch_for(expected_output, request.mode);
    action_.apply_overwrite(request.mode, request.input, applied);
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
  return {action_.resource_stamp(), true};
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

}  // namespace taichi::lang
