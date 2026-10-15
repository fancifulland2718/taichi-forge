#include "taichi/program/program.h"
#include "taichi/program/cuda_cudss_capture.h"
#include "taichi/runtime/cuda/cudss_plan.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(TI_WITH_CUDA)
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_driver.h"

namespace taichi::lang {
namespace {

const CuSparseMatrix &require_cudss_matrix(SparseMatrix *matrix,
                                           Program *program) {
  TI_ERROR_IF(!matrix, "CUDA cuDSS received a null sparse matrix.");
  const auto *csr = dynamic_cast<const CuSparseMatrix *>(matrix);
  TI_ERROR_IF(!csr,
              "CUDA cuDSS requires a CUDA scalar CSR SparseMatrix; no "
              "format conversion or host fallback was performed.");
  TI_ERROR_IF(csr->num_rows() <= 0 || csr->num_rows() != csr->num_cols(),
              "CUDA cuDSS requires a non-empty square sparse matrix.");
  TI_ERROR_IF(csr->get_data_type() != PrimitiveType::f32,
              "The first CUDA cuDSS provider slice supports f32 only.");
  TI_ERROR_IF(csr->get_nnz() <= 0 || !csr->get_row_ptr() ||
                  !csr->get_col_ind() || !csr->get_val_ptr(),
              "CUDA cuDSS requires a materialized non-empty CSR matrix.");
  (void)program;
  return *csr;
}

CudaCudssMatrixView cudss_matrix_view(const CuSparseMatrix &matrix,
                                      bool pattern_facts = false) {
  return {
      static_cast<std::size_t>(matrix.num_rows()),
      static_cast<std::size_t>(matrix.num_cols()),
      static_cast<std::size_t>(matrix.get_nnz()),
      matrix.get_row_ptr(),
      matrix.get_col_ind(),
      matrix.get_val_ptr(),
      matrix.matrix_id(),
      matrix.pattern_version(),
      matrix.numeric_version(),
      pattern_facts ? matrix.debug_runtime_statistics().shared_pattern_id : 0};
}

void validate_cudss_vector(Ndarray *array,
                           std::size_t expected_elements,
                           const char *name,
                           Program *program) {
  TI_ERROR_IF(!array, "CUDA cuDSS {} received a null ndarray.", name);
  TI_ERROR_IF(!array->get_element_shape().empty() ||
                  array->get_element_data_type() != PrimitiveType::f32 ||
                  array->get_nelement() != expected_elements ||
                  array->get_element_size() != sizeof(float32),
              "CUDA cuDSS {} must be a compact scalar f32 ndarray with {} "
              "entries.",
              name, expected_elements);
  TI_ERROR_IF(array->owning_program() != program,
              "CUDA cuDSS {} must belong to the active runtime.", name);
}

void validate_cudss_values(Ndarray *array,
                           std::size_t expected_elements,
                           Program *program) {
  TI_ERROR_IF(!array, "CUDA cuDSS matrix values received a null ndarray.");
  TI_ERROR_IF(!array->get_element_shape().empty() ||
                  array->get_element_data_type() != PrimitiveType::f32 ||
                  array->get_nelement() != expected_elements ||
                  array->get_element_size() != sizeof(float32),
              "CUDA cuDSS matrix values must be a compact scalar f32 ndarray "
              "with {} entries.",
              expected_elements);
  TI_ERROR_IF(array->owning_program() != program,
              "CUDA cuDSS matrix values must belong to the active runtime.");
}

}  // namespace

namespace {
class CudaCudssCaptureCommand final : public aot::CudaGraphCaptureCommand {
 public:
  CudaCudssCaptureCommand(Program *program,
                          std::uint64_t handle,
                          int phase,
                          const std::vector<aot::Arg> &arguments)
      : program_(program), phase_(phase), arguments_(arguments) {
    TI_ERROR_IF(!program || program->compile_config().arch != Arch::cuda,
                "cuDSS capture requires a CUDA Program.");
    const std::size_t offset = phase ? 1 : 0;
    TI_ERROR_IF(
        arguments.size() < offset + 2 || (arguments.size() - offset) % 2,
        "cuDSS capture bindings do not match the numerical phase.");
    for (std::size_t i = 0; i < arguments.size(); ++i) {
      const auto &arg = arguments[i];
      TI_ERROR_IF(arg.tag != aot::ArgKind::kNdarray ||
                      arg.dtype_id != PrimitiveTypeID::f32 ||
                      arg.field_dim != 1 || !arg.element_shape.empty() ||
                      arg.name.empty(),
                  "cuDSS capture requires named scalar f32 vectors.");
      for (std::size_t j = 0; j < i; ++j) {
        TI_ERROR_IF(arg.name == arguments[j].name,
                    "cuDSS capture binding names must be distinct.");
      }
    }
    owner_ = program->retain_cuda_cudss_capture_plan(handle);
    owner_->claim_graph_phase(phase);
  }

  const char *kind() const override {
    return "cudss_retained_solve_f32";
  }
  Program *program() const override {
    return program_;
  }
  bool supports_binding_frames() const override {
    return true;
  }
  bool supports_conditional_body() const override {
    // Cold seed factors are ready, and one frozen numerical policy covers
    // the complete update + RHS group on every entered iteration.
    return true;
  }
  std::shared_ptr<void> retain_binding_frame_plan(Program &) override {
    return owner_;
  }
  std::shared_ptr<aot::CudaGraphCaptureResources> take_capture_resources()
      override {
    return std::exchange(capture_inputs_, {});
  }

  bool supports(const std::unordered_map<std::string, aot::IValue> &args,
                Program &program) const override {
    if (&program != program_)
      return false;
    for (std::size_t i = 0; i < arguments_.size(); ++i) {
      auto *value = array(i, args);
      if (!value)
        return false;
      for (std::size_t j = 0; j < i; ++j) {
        if (value->get_device_allocation() ==
            array(j, args)->get_device_allocation())
          return false;
      }
    }
    return true;
  }

  void prepare(const std::unordered_map<std::string, aot::IValue> &args,
               Program &program) override {
    TI_ERROR_IF(!supports(args, program),
                "cuDSS capture bindings are incompatible.");
    // Owner creation already warmed a private numerical snapshot. Never run
    // mathematical work with caller bindings before the first Graph submission.
  }

  void record(const std::unordered_map<std::string, aot::IValue> &args,
              Program &program,
              void *stream) override {
    TI_ERROR_IF(!supports(args, program),
                "cuDSS capture bindings are incompatible.");
    const auto pointer = [&](std::size_t i) {
      return reinterpret_cast<void *>(
          program.get_ndarray_data_ptr_as_int(array(i, args)));
    };
    const std::size_t offset = phase_ ? 1 : 0;
    std::vector<std::pair<void *, void *>> rhs_and_outputs;
    rhs_and_outputs.reserve((arguments_.size() - offset) / 2);
    for (std::size_t i = offset; i < arguments_.size(); i += 2)
      rhs_and_outputs.emplace_back(pointer(i), pointer(i + 1));
    owner_->record_capture(phase_, phase_ ? pointer(0) : nullptr,
                           rhs_and_outputs, stream, capture_inputs_);
  }

 private:
  Ndarray *array(
      std::size_t index,
      const std::unordered_map<std::string, aot::IValue> &args) const {
    const auto found = args.find(arguments_[index].name);
    if (found == args.end() || found->second.tag != aot::ArgKind::kNdarray)
      return nullptr;
    auto *array = reinterpret_cast<Ndarray *>(found->second.val);
    if (!array || array->owning_program() != program_ ||
        array->get_element_data_type() != PrimitiveType::f32 ||
        !array->get_element_shape().empty() || array->shape.size() != 1 ||
        array->get_nelement() !=
            (phase_ && index == 0 ? owner_->nonzeros() : owner_->rows()))
      return nullptr;
    return array;
  }
  Program *program_;
  const int phase_;
  const std::vector<aot::Arg> arguments_;
  std::shared_ptr<CudaCudssPlan> owner_;
  std::shared_ptr<CudaGraphCaptureResources> capture_inputs_;
};
}  // namespace

std::shared_ptr<aot::CudaGraphCaptureCommand> make_cuda_cudss_capture_command(
    Program *program,
    std::uint64_t handle,
    int numeric_phase,
    const std::vector<aot::Arg> &arguments) {
  return std::make_shared<CudaCudssCaptureCommand>(program, handle,
                                                   numeric_phase, arguments);
}

std::shared_ptr<CudaCudssPlan> Program::retain_cuda_cudss_capture_plan(
    std::uint64_t handle) {
  std::lock_guard<std::mutex> lock(cuda_cudss_plan_mutex_);
  const auto found = cuda_cudss_graph_plans_.find(handle);
  TI_ERROR_IF(found == cuda_cudss_graph_plans_.end(),
              "CUDA cuDSS Graph owner is stale, closed, or not Graph-owned.");
  return found->second;
}

CudaCudssPlanSnapshot snapshot_cuda_cudss_graph_seed(Program &program,
                                                     std::uint64_t handle) {
  auto submission = program.acquire_runtime_resource_submission_guard();
  auto plan = program.retain_cuda_cudss_capture_plan(handle);
  auto &cuda = CUDAContext::get_instance();
  auto lock = cuda.get_submission_lock_guard();
  auto guard = cuda.get_guard();
  return plan->snapshot_graph_seed();
}

std::uint64_t Program::create_cuda_cudss_plan(
    SparseMatrix *matrix,
    int matrix_type,
    int matrix_view,
    const std::string &adapter_path,
    const std::string &runtime_library_path) {
  return create_cuda_cudss_configured_plan(
      matrix, matrix_type, matrix_view, adapter_path, runtime_library_path, {});
}

std::uint64_t Program::create_cuda_cudss_configured_plan(
    SparseMatrix *matrix,
    int matrix_type,
    int matrix_view,
    const std::string &adapter_path,
    const std::string &runtime_library_path,
    const std::vector<int> &configuration,
    bool graph_owned) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA cuDSS plans require the CUDA backend.");
  TI_ERROR_IF(
      !CUDADriver::get_instance_without_context().nvidia_extensions_available(),
      "CUDA cuDSS requires the NVIDIA CUDA provider.");
  const auto &csr = require_cudss_matrix(matrix, this);
  auto cuda_submission_guard =
      CUDAContext::get_instance().get_submission_lock_guard();
  auto context_guard = CUDAContext::get_instance().get_guard();
  auto plan = make_cuda_cudss_plan(
      cudss_matrix_view(csr), matrix_type, matrix_view, adapter_path,
      runtime_library_path, runtime_fault_domain_, configuration, graph_owned);
  std::lock_guard<std::mutex> lock(cuda_cudss_plan_mutex_);
  TI_ERROR_IF(next_cuda_cudss_plan_handle_ == 0,
              "CUDA cuDSS plan handle space exhausted.");
  const auto handle = next_cuda_cudss_plan_handle_++;
  auto &owners = graph_owned ? cuda_cudss_graph_plans_ : cuda_cudss_plans_;
  owners.emplace(handle, std::move(plan));
  return handle;
}

void Program::cuda_cudss_analyze(std::uint64_t handle, SparseMatrix *matrix) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  std::shared_ptr<CudaCudssPlan> plan;
  {
    std::lock_guard<std::mutex> lock(cuda_cudss_plan_mutex_);
    const auto found = cuda_cudss_plans_.find(handle);
    TI_ERROR_IF(found == cuda_cudss_plans_.end(),
                "CUDA cuDSS plan handle is stale or closed.");
    plan = found->second;
  }
  auto cuda_submission_guard =
      CUDAContext::get_instance().get_submission_lock_guard();
  auto context_guard = CUDAContext::get_instance().get_guard();
  const auto &csr = require_cudss_matrix(matrix, this);
  plan->analyze(cudss_matrix_view(csr, true));
  pin_cuda_provider_plan(plan);
  mark_runtime_submission_pending();
}

void Program::cuda_cudss_factorize(std::uint64_t handle,
                                   SparseMatrix *matrix,
                                   bool refactorize) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  std::shared_ptr<CudaCudssPlan> plan;
  {
    std::lock_guard<std::mutex> lock(cuda_cudss_plan_mutex_);
    const auto found = cuda_cudss_plans_.find(handle);
    TI_ERROR_IF(found == cuda_cudss_plans_.end(),
                "CUDA cuDSS plan handle is stale or closed.");
    plan = found->second;
  }
  auto cuda_submission_guard =
      CUDAContext::get_instance().get_submission_lock_guard();
  auto context_guard = CUDAContext::get_instance().get_guard();
  const auto &csr = require_cudss_matrix(matrix, this);
  plan->factorize(cudss_matrix_view(csr, true), refactorize);
  pin_cuda_provider_plan(plan);
  mark_runtime_submission_pending();
}

std::size_t Program::cuda_cudss_solve(std::uint64_t handle,
                                      SparseMatrix *matrix,
                                      Ndarray *rhs,
                                      Ndarray *solution) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA cuDSS solve requires the CUDA backend.");
  std::shared_ptr<CudaCudssPlan> plan;
  {
    std::lock_guard<std::mutex> lock(cuda_cudss_plan_mutex_);
    const auto found = cuda_cudss_plans_.find(handle);
    TI_ERROR_IF(found == cuda_cudss_plans_.end(),
                "CUDA cuDSS plan handle is stale or closed.");
    plan = found->second;
  }
  const auto stats = plan->statistics();
  const auto rows = static_cast<std::size_t>(stats.at("rows"));
  validate_cudss_vector(rhs, rows, "right-hand side", this);
  validate_cudss_vector(solution, rows, "solution", this);
  TI_ERROR_IF(rhs->get_device_allocation() == solution->get_device_allocation(),
              "The first CUDA cuDSS slice requires distinct right-hand-side "
              "and solution allocations.");
  auto cuda_submission_guard =
      CUDAContext::get_instance().get_submission_lock_guard();
  auto context_guard = CUDAContext::get_instance().get_guard();
  const auto &csr = require_cudss_matrix(matrix, this);
  auto *rhs_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(rhs));
  auto *solution_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(solution));
  TI_ERROR_IF(!rhs_ptr || !solution_ptr,
              "CUDA cuDSS received a null dense-vector device pointer.");
  plan->solve(cudss_matrix_view(csr), rhs_ptr, solution_ptr);
  pin_cuda_provider_plan(plan);
  auto leases = acquire_ndarray_leases({rhs, solution});
  pin_ndarray_launch_leases(leases);
  mark_runtime_submission_pending();
  return 0;
}

std::size_t Program::cuda_cudss_refactor_solve(std::uint64_t handle,
                                               Ndarray *values,
                                               Ndarray *rhs,
                                               Ndarray *solution) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA cuDSS refactorize+solve requires the CUDA backend.");
  std::shared_ptr<CudaCudssPlan> plan;
  {
    std::lock_guard<std::mutex> lock(cuda_cudss_plan_mutex_);
    const auto found = cuda_cudss_plans_.find(handle);
    TI_ERROR_IF(found == cuda_cudss_plans_.end(),
                "CUDA cuDSS plan handle is stale or closed.");
    plan = found->second;
  }
  validate_cudss_values(values, plan->nonzeros(), this);
  validate_cudss_vector(rhs, plan->rows(), "right-hand side", this);
  validate_cudss_vector(solution, plan->rows(), "solution", this);
  const auto values_allocation = values->get_device_allocation();
  const auto rhs_allocation = rhs->get_device_allocation();
  const auto solution_allocation = solution->get_device_allocation();
  TI_ERROR_IF(values_allocation == rhs_allocation ||
                  values_allocation == solution_allocation ||
                  rhs_allocation == solution_allocation,
              "CUDA cuDSS refactorize+solve values, right-hand side, and "
              "solution allocations must be distinct.");
  auto leases = acquire_ndarray_leases({values, rhs, solution});
  auto cuda_submission_guard =
      CUDAContext::get_instance().get_submission_lock_guard();
  auto context_guard = CUDAContext::get_instance().get_guard();
  auto *values_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(values));
  auto *rhs_ptr = reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(rhs));
  auto *solution_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(solution));
  TI_ERROR_IF(!values_ptr || !rhs_ptr || !solution_ptr,
              "CUDA cuDSS refactorize+solve received a null device pointer.");

  plan->reserve_refactor_solve();
  try {
    pin_cuda_provider_plan(plan);
    pin_ndarray_launch_leases(leases);
    mark_runtime_submission_pending();
  } catch (...) {
    plan->cancel_unsubmitted_refactor_solve();
    throw;
  }
  plan->execute_reserved_refactor_solve(values_ptr, rhs_ptr, solution_ptr);
  return 0;
}

std::unordered_map<std::string, std::uint64_t>
Program::cuda_cudss_plan_statistics(std::uint64_t handle) {
  std::lock_guard<std::mutex> lock(cuda_cudss_plan_mutex_);
  const auto found = cuda_cudss_plans_.find(handle);
  TI_ERROR_IF(found == cuda_cudss_plans_.end(),
              "CUDA cuDSS plan handle is stale or closed.");
  return found->second->statistics();
}

std::unordered_map<std::string, std::int64_t>
Program::cuda_cudss_plan_configuration(std::uint64_t handle) {
  std::lock_guard<std::mutex> lock(cuda_cudss_plan_mutex_);
  const auto graph = cuda_cudss_graph_plans_.find(handle);
  if (graph != cuda_cudss_graph_plans_.end()) {
    return graph->second->configuration();
  }
  const auto found = cuda_cudss_plans_.find(handle);
  TI_ERROR_IF(found == cuda_cudss_plans_.end(),
              "CUDA cuDSS plan handle is stale or closed.");
  return found->second->configuration();
}

void Program::debug_cuda_cudss_fail_next_refactor_solve(std::uint64_t handle) {
  std::shared_ptr<CudaCudssPlan> plan;
  {
    std::lock_guard<std::mutex> lock(cuda_cudss_plan_mutex_);
    const auto found = cuda_cudss_plans_.find(handle);
    TI_ERROR_IF(found == cuda_cudss_plans_.end(),
                "CUDA cuDSS plan handle is stale or closed.");
    plan = found->second;
  }
  plan->debug_fail_next_refactor_solve();
}

void Program::destroy_cuda_cudss_plan(std::uint64_t handle) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  std::shared_ptr<CudaCudssPlan> plan;
  {
    std::lock_guard<std::mutex> lock(cuda_cudss_plan_mutex_);
    auto *owners = cuda_cudss_graph_plans_.count(handle)
                       ? &cuda_cudss_graph_plans_
                       : &cuda_cudss_plans_;
    const auto found = owners->find(handle);
    if (found == owners->end()) {
      return;
    }
    plan = std::move(found->second);
    owners->erase(found);
  }
  // RuntimeCompletion owns any in-flight reference. Destruction therefore
  // occurs immediately only when no submitted phase still uses this plan.
  plan.reset();
}

void Program::cuda_clear_cudss_plans() {
  std::vector<std::shared_ptr<CudaCudssPlan>> plans;
  {
    std::lock_guard<std::mutex> lock(cuda_cudss_plan_mutex_);
    plans.reserve(cuda_cudss_plans_.size() + cuda_cudss_graph_plans_.size());
    for (auto &[handle, plan] : cuda_cudss_plans_) {
      plans.push_back(std::move(plan));
    }
    cuda_cudss_plans_.clear();
    for (auto &[handle, plan] : cuda_cudss_graph_plans_) {
      plans.push_back(std::move(plan));
    }
    cuda_cudss_graph_plans_.clear();
  }
  const bool provider_calls_safe = !runtime_has_fatal_fault();
  if (provider_calls_safe && !plans.empty()) {
    auto cuda_submission_guard =
        CUDAContext::get_instance().get_submission_lock_guard();
    auto context_guard = CUDAContext::get_instance().get_guard();
    for (auto &plan : plans) {
      plan->destroy(true);
    }
  } else {
    for (auto &plan : plans) {
      plan->destroy(false);
    }
  }
}

}  // namespace taichi::lang

#else

namespace taichi::lang {

CudaCudssPlanSnapshot snapshot_cuda_cudss_graph_seed(Program &, std::uint64_t) {
  TI_ERROR("CUDA cuDSS requires TI_WITH_CUDA=ON.");
}

std::shared_ptr<aot::CudaGraphCaptureCommand> make_cuda_cudss_capture_command(
    Program *,
    std::uint64_t,
    int,
    const std::vector<aot::Arg> &) {
  TI_ERROR("CUDA cuDSS requires TI_WITH_CUDA=ON.");
}

std::shared_ptr<CudaCudssPlan> Program::retain_cuda_cudss_capture_plan(
    std::uint64_t) {
  TI_ERROR("CUDA cuDSS requires TI_WITH_CUDA=ON.");
}

std::uint64_t Program::create_cuda_cudss_configured_plan(
    SparseMatrix *,
    int,
    int,
    const std::string &,
    const std::string &,
    const std::vector<int> &,
    bool) {
  TI_ERROR("CUDA cuDSS requires TI_WITH_CUDA=ON.");
}

std::unordered_map<std::string, std::int64_t>
Program::cuda_cudss_plan_configuration(std::uint64_t) {
  TI_ERROR("CUDA cuDSS requires TI_WITH_CUDA=ON.");
}

std::uint64_t Program::create_cuda_cudss_plan(SparseMatrix *,
                                              int,
                                              int,
                                              const std::string &,
                                              const std::string &) {
  TI_ERROR("CUDA cuDSS requires TI_WITH_CUDA=ON.");
}

void Program::cuda_cudss_analyze(std::uint64_t, SparseMatrix *) {
  TI_ERROR("CUDA cuDSS requires TI_WITH_CUDA=ON.");
}

void Program::cuda_cudss_factorize(std::uint64_t, SparseMatrix *, bool) {
  TI_ERROR("CUDA cuDSS requires TI_WITH_CUDA=ON.");
}

std::size_t Program::cuda_cudss_solve(std::uint64_t,
                                      SparseMatrix *,
                                      Ndarray *,
                                      Ndarray *) {
  TI_ERROR("CUDA cuDSS requires TI_WITH_CUDA=ON.");
}

std::size_t Program::cuda_cudss_refactor_solve(std::uint64_t,
                                               Ndarray *,
                                               Ndarray *,
                                               Ndarray *) {
  TI_ERROR("CUDA cuDSS requires TI_WITH_CUDA=ON.");
}

std::unordered_map<std::string, std::uint64_t>
Program::cuda_cudss_plan_statistics(std::uint64_t) {
  TI_ERROR("CUDA cuDSS requires TI_WITH_CUDA=ON.");
}

void Program::debug_cuda_cudss_fail_next_refactor_solve(std::uint64_t) {
  TI_ERROR("CUDA cuDSS requires TI_WITH_CUDA=ON.");
}

void Program::destroy_cuda_cudss_plan(std::uint64_t) {
}

void Program::cuda_clear_cudss_plans() {
}

}  // namespace taichi::lang

#endif
