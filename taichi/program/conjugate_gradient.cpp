#include "conjugate_gradient.h"
#include "linear_operator.h"
#include "sparse_preconditioner.h"
#include "taichi/rhi/cuda/primitives/solver_ptx.h"
#include "taichi/util/environ_config.h"

#if defined(TI_WITH_CUDA)
#include "taichi/rhi/cuda/cuda_context.h"
#endif

#if defined(TI_WITH_VULKAN)
#include "taichi/program/vulkan_command_replay.h"
#endif

#include <algorithm>
#include <functional>
#include <limits>
#include <optional>
#include <vector>

namespace taichi::lang {

#if defined(TI_WITH_CUDA)

struct CudaSolverChunkReplayKey {
  RuntimeResourceHandle solution;
  std::uint64_t program_generation{0};
  std::uint64_t schema_revision{0};
  std::uint64_t topology_revision{0};
  std::uint64_t binding_revision{0};
  std::uint64_t preconditioner_schema_revision{0};
  std::uint64_t preconditioner_topology_revision{0};
  std::uint64_t preconditioner_binding_revision{0};
  std::uintptr_t x{0};
  std::uintptr_t ax{0};
  std::uintptr_t residual{0};
  std::uintptr_t direction{0};
  std::uintptr_t preconditioned_residual{0};
  std::uintptr_t scalars{0};
  std::uintptr_t provider{0};
  std::uintptr_t preconditioner{0};
  int rows{0};
  int iterations{0};

  bool operator==(const CudaSolverChunkReplayKey &other) const {
    return solution == other.solution &&
           program_generation == other.program_generation &&
           schema_revision == other.schema_revision &&
           topology_revision == other.topology_revision &&
           binding_revision == other.binding_revision &&
           preconditioner_schema_revision ==
               other.preconditioner_schema_revision &&
           preconditioner_topology_revision ==
               other.preconditioner_topology_revision &&
           preconditioner_binding_revision ==
               other.preconditioner_binding_revision &&
           x == other.x && ax == other.ax && residual == other.residual &&
           direction == other.direction &&
           preconditioned_residual == other.preconditioned_residual &&
           scalars == other.scalars && provider == other.provider &&
           preconditioner == other.preconditioner && rows == other.rows &&
           iterations == other.iterations;
  }
};

struct CudaSolverChunkReplayEntry {
  CUgraphExec executable{nullptr};
  CudaSolverChunkReplayKey key;
  bool key_valid{false};
  std::uint64_t operator_numeric_revision{0};
  std::uint64_t preconditioner_numeric_revision{0};
  std::optional<Program::NdarrayResourceLease> solution_lease;

  void reset() {
    if (executable != nullptr) {
      CUDADriver::get_instance().graph_exec_destroy(executable);
      executable = nullptr;
    }
    key_valid = false;
    solution_lease.reset();
  }
};

struct CudaSolverChunkReplayState {
  CUstream capture_stream{nullptr};
  std::array<CudaSolverChunkReplayEntry, 9> entries;
  bool disabled{false};
  std::string unavailable_reason{"not_built"};

  ~CudaSolverChunkReplayState() {
    CUDAContext::get_instance().make_current();
    CUDADriver::get_instance().stream_synchronize(nullptr);
    for (auto &entry : entries) {
      entry.reset();
    }
    if (capture_stream != nullptr) {
      CUDADriver::get_instance().stream_destroy(capture_stream);
      capture_stream = nullptr;
    }
  }

  CUstream ensure_capture_stream() {
    if (capture_stream == nullptr) {
      CUDAContext::get_instance().make_current();
      CUDADriver::get_instance().stream_create(
          reinterpret_cast<void **>(&capture_stream), CU_STREAM_NON_BLOCKING);
    }
    return capture_stream;
  }

  void reset_entries() {
    CUDAContext::get_instance().make_current();
    CUDADriver::get_instance().stream_synchronize(nullptr);
    for (auto &entry : entries) {
      entry.reset();
    }
  }
};

#else

struct CudaSolverChunkReplayState {};

#endif

#if defined(TI_WITH_VULKAN)

struct VulkanSolverChunkReplayState {
  struct Slot {
    VulkanCommandReplayCache cache;
    RuntimeResourceHandle solution_handle;
    std::optional<Program::NdarrayResourceLease> solution_lease;
    std::uint64_t operator_numeric_revision{0};
    std::uint64_t preconditioner_numeric_revision{0};
  };

  std::vector<std::unique_ptr<Slot>> slots;
  std::string unavailable_reason{"not_built"};

  Slot &slot(std::size_t index) {
    if (slots.size() <= index) {
      slots.resize(index + 1);
    }
    if (!slots[index]) {
      slots[index] = std::make_unique<Slot>();
    }
    return *slots[index];
  }

  void reset() {
    slots.clear();
  }
};

#else

struct VulkanSolverChunkReplayState {};

#endif

namespace {

OperatorMathematicalTraits legacy_cg_traits() {
  const auto scope =
      operator_dependency(OperatorResourceDependency::program) |
      operator_dependency(OperatorResourceDependency::schema) |
      operator_dependency(OperatorResourceDependency::topology) |
      operator_dependency(OperatorResourceDependency::numeric);
  return make_spd_operator_traits(
      OperatorTraitProvenance::asserted_by_user, scope);
}

OperatorBinding with_legacy_cg_traits(OperatorBinding binding) {
  return binding.with_mathematical_traits(legacy_cg_traits());
}

#if defined(TI_WITH_VULKAN)
void push_solver_chunk_resource(VulkanCommandReplayKey &key,
                                const Ndarray *array) {
  if (array == nullptr) {
    key.push(0);
    key.push(0);
    key.push(0);
    key.push(0);
    return;
  }
  const auto handle = array->runtime_resource_handle();
  key.push(handle.domain);
  key.push(handle.kind);
  key.push(handle.index);
  key.push(handle.generation);
}

void push_solver_chunk_stamp(VulkanCommandReplayKey &key,
                             const OperatorResourceStamp &stamp) {
  key.push(stamp.program_generation);
  key.push(stamp.schema_revision);
  key.push(stamp.topology_revision);
  key.push(stamp.binding_revision);
}
#endif

void validate_cg_plan(const OperatorPlan &operator_plan,
                      const PreconditionerPlan *preconditioner_plan) {
  validate_operator_solver_compatibility(
      operator_plan.descriptor(), operator_plan.mathematical_traits(),
      preconditioner_plan ? OperatorSolverFamily::pcg
                          : OperatorSolverFamily::cg,
      preconditioner_plan ? preconditioner_plan->behavior()
                          : PreconditionerBehavior::fixed_linear);
}

OperatorBinding bind_preconditioner_action(
    Program *program,
    SparseMatrix &matrix,
    SparseJacobiPreconditionerPlan &preconditioner,
    std::string provider_name);
OperatorBinding bind_preconditioner_action(
    Program *program,
    SparseMatrix &matrix,
    SparseBlockJacobiPreconditionerPlan &preconditioner,
    std::string provider_name);
OperatorBinding bind_preconditioner_action(
    Program *program,
    CompiledKernelLinearOperator &matrix,
    CompiledKernelPreconditionerPlan &preconditioner,
    std::string provider_name);
std::unique_ptr<PreconditionerPlan> make_fixed_preconditioner_plan(
    Program *program,
    OperatorPlan &target_plan,
    SparseMatrix &matrix,
    SparseJacobiPreconditionerPlan &preconditioner,
    std::string provider_name,
    std::string method);
std::unique_ptr<PreconditionerPlan> make_fixed_preconditioner_plan(
    Program *program,
    OperatorPlan &target_plan,
    SparseMatrix &matrix,
    SparseBlockJacobiPreconditionerPlan &preconditioner,
    std::string provider_name,
    std::string method);
std::unique_ptr<PreconditionerPlan> make_fixed_preconditioner_plan(
    Program *program,
    OperatorPlan &target_plan,
    CompiledKernelLinearOperator &matrix,
    CompiledKernelPreconditionerPlan &preconditioner,
    std::string provider_name,
    std::string method);
std::unique_ptr<PreconditionerPlan> make_fixed_preconditioner_plan(
    Program *program,
    OperatorPlan &target_plan,
    ExperimentalLinearOperatorHandle &preconditioner,
    std::string method);
void append_operator_plan_statistics(
    const OperatorPlan &plan,
    bool preconditioner,
    SparseSolvePlanRuntimeStatistics &statistics);
void append_preconditioner_plan_statistics(
    const PreconditionerPlan &plan,
    SparseSolvePlanRuntimeStatistics &statistics);

}  // namespace

const char *sparse_solve_execution_policy_name(
    SparseSolveExecutionPolicy policy) {
  switch (policy) {
    case SparseSolveExecutionPolicy::host_each_iteration:
      return "host_each_iteration";
    case SparseSolveExecutionPolicy::host_check_every_k:
      return "host_check_every_k";
    case SparseSolveExecutionPolicy::fixed_budget_masked:
      return "fixed_budget_masked";
    case SparseSolveExecutionPolicy::device_convergent:
      return "device_convergent";
  }
  return "unknown";
}

SparseSolveExecutionCapabilities sparse_solve_execution_capabilities(
    Arch arch) {
  SparseSolveExecutionCapabilities result;
  if (arch_is_cpu(arch)) {
    result.host_each_iteration = true;
  } else if (arch == Arch::cuda) {
    result.host_each_iteration = true;
    result.host_check_every_k = true;
  } else if (arch == Arch::vulkan) {
    result.host_check_every_k = true;
    result.fixed_budget_masked = true;
  }
  return result;
}

void validate_sparse_solve_execution_policy(
    Arch arch,
    SparseSolveExecutionPolicy policy,
    int host_check_interval) {
  TI_ERROR_IF(host_check_interval <= 0,
              "Solver host-check interval must be positive.");
  TI_ERROR_IF(policy !=
                      SparseSolveExecutionPolicy::host_check_every_k &&
                  host_check_interval != 1,
              "Only host_check_every_k accepts a host-check interval other "
              "than one.");
  const auto capabilities = sparse_solve_execution_capabilities(arch);
  bool supported = false;
  switch (policy) {
    case SparseSolveExecutionPolicy::host_each_iteration:
      supported = capabilities.host_each_iteration;
      break;
    case SparseSolveExecutionPolicy::host_check_every_k:
      supported = capabilities.host_check_every_k;
      break;
    case SparseSolveExecutionPolicy::fixed_budget_masked:
      supported = capabilities.fixed_budget_masked;
      break;
    case SparseSolveExecutionPolicy::device_convergent:
      supported = capabilities.device_convergent;
      break;
  }
  TI_ERROR_IF(!supported,
              "Solver execution policy '{}' is unsupported on backend '{}'; "
              "no fallback was performed.",
              sparse_solve_execution_policy_name(policy), arch_name(arch));
}

CUCG::CUCG(SparseMatrix &A,
           int max_iters,
           float absolute_tolerance,
           bool verbose,
           float relative_tolerance)
    : A_(A),
      cuda_csr_operator_(dynamic_cast<CuSparseMatrix *>(&A)),
      max_iters_(max_iters),
      absolute_tolerance_(absolute_tolerance),
      relative_tolerance_(relative_tolerance),
      verbose_(verbose) {
  TI_ERROR_IF(!cuda_csr_operator_,
              "CUDA conjugate gradient currently requires a CSR matrix.");
  validate_controls();
  init_solver();
}

CUCG::CUCG(Program *program,
           SparseMatrix &A,
           SparseJacobiPreconditionerPlan &preconditioner,
           int max_iters,
           float absolute_tolerance,
           bool verbose,
           float relative_tolerance)
    : program_(program),
      A_(A),
      cuda_csr_operator_(dynamic_cast<CuSparseMatrix *>(&A)),
      preconditioner_(&preconditioner),
      max_iters_(max_iters),
      absolute_tolerance_(absolute_tolerance),
      relative_tolerance_(relative_tolerance),
      verbose_(verbose) {
  TI_ERROR_IF(!cuda_csr_operator_,
              "CUDA Jacobi-PCG currently requires a CSR matrix.");
  operator_plan_ = std::make_unique<OperatorPlan>(
      program_, with_legacy_cg_traits(make_cuda_csr_operator_binding(
                    program_, *cuda_csr_operator_)));
  validate_controls();
  preconditioner_plan_ = make_fixed_preconditioner_plan(
      program_, *operator_plan_, A_, preconditioner, "cuda_jacobi",
      "jacobi");
  validate_cg_plan(*operator_plan_, preconditioner_plan_.get());
  init_solver();
}

CUCG::CUCG(Program *program,
           SparseMatrix &A,
           SparseBlockJacobiPreconditionerPlan &preconditioner,
           int max_iters,
           float absolute_tolerance,
           bool verbose,
           float relative_tolerance)
    : program_(program),
      A_(A),
      cuda_bsr_operator_(dynamic_cast<CuSparseBsrMatrix *>(&A)),
      block_preconditioner_(&preconditioner),
      max_iters_(max_iters),
      absolute_tolerance_(absolute_tolerance),
      relative_tolerance_(relative_tolerance),
      verbose_(verbose) {
  TI_ERROR_IF(!cuda_bsr_operator_,
              "CUDA block-Jacobi PCG requires an internal BSR matrix.");
  operator_plan_ = std::make_unique<OperatorPlan>(
      program_, with_legacy_cg_traits(make_cuda_bsr_operator_binding(
                    program_, *cuda_bsr_operator_)));
  validate_controls();
  preconditioner_plan_ = make_fixed_preconditioner_plan(
      program_, *operator_plan_, A_, preconditioner, "cuda_block_jacobi",
      "block_jacobi");
  validate_cg_plan(*operator_plan_, preconditioner_plan_.get());
  init_solver();
}

CUCG::CUCG(Program *program,
           CompiledKernelLinearOperator &A,
           CompiledKernelPreconditionerPlan *preconditioner,
           int max_iters,
           float absolute_tolerance,
           bool verbose,
           float relative_tolerance)
    : program_(program),
      A_(A),
      compiled_kernel_operator_(&A),
      compiled_kernel_preconditioner_(preconditioner),
      max_iters_(max_iters),
      absolute_tolerance_(absolute_tolerance),
      relative_tolerance_(relative_tolerance),
      verbose_(verbose) {
  TI_ERROR_IF(!program_ || program_->compile_config().arch != Arch::cuda ||
                  A.owning_program() != program_,
              "CUDA compiled-kernel CG requires its owning CUDA Program.");
  validate_controls();
  operator_plan_ = std::make_unique<OperatorPlan>(
      program_, with_legacy_cg_traits(
                    make_cuda_program_kernel_operator_binding(program_, A)));
  if (compiled_kernel_preconditioner_) {
    preconditioner_plan_ = make_fixed_preconditioner_plan(
        program_, *operator_plan_, A, *compiled_kernel_preconditioner_,
        "cuda_compiled_inverse_apply", "compiled_kernel_inverse_apply");
  }
  validate_cg_plan(*operator_plan_, preconditioner_plan_.get());
  init_solver();
}

CUCG::CUCG(Program *program,
           CompiledKernelLinearOperator &A,
           ExperimentalLinearOperatorHandle &preconditioner,
           int max_iters,
           float absolute_tolerance,
           bool verbose,
           float relative_tolerance)
    : program_(program),
      A_(A),
      compiled_kernel_operator_(&A),
      operator_preconditioner_(&preconditioner),
      max_iters_(max_iters),
      absolute_tolerance_(absolute_tolerance),
      relative_tolerance_(relative_tolerance),
      verbose_(verbose) {
  TI_ERROR_IF(!program_ || program_->compile_config().arch != Arch::cuda ||
                  A.owning_program() != program_ ||
                  preconditioner.program() != program_,
              "CUDA LinearOperator PCG requires A and M owned by the same "
              "CUDA Program.");
  validate_controls();
  operator_plan_ = std::make_unique<OperatorPlan>(
      program_, with_legacy_cg_traits(
                    make_cuda_program_kernel_operator_binding(program_, A)));
  preconditioner_plan_ = make_fixed_preconditioner_plan(
      program_, *operator_plan_, preconditioner, "linear_operator");
  validate_cg_plan(*operator_plan_, preconditioner_plan_.get());
  init_solver();
}

CUCG::CUCG(Program *program,
           CompiledGraphLinearOperator &A,
           int max_iters,
           float absolute_tolerance,
           bool verbose,
           float relative_tolerance)
    : program_(program),
      A_(A),
      compiled_graph_operator_(&A),
      max_iters_(max_iters),
      absolute_tolerance_(absolute_tolerance),
      relative_tolerance_(relative_tolerance),
      verbose_(verbose) {
  TI_ERROR_IF(!program_ || program_->compile_config().arch != Arch::cuda ||
                  A.owning_program() != program_,
              "CUDA compiled-graph CG requires its owning CUDA Program.");
  validate_controls();
  operator_plan_ = std::make_unique<OperatorPlan>(
      program_, with_legacy_cg_traits(
                    make_cuda_program_graph_operator_binding(program_, A)));
  validate_cg_plan(*operator_plan_, nullptr);
  init_solver();
}

bool CUCG::has_preconditioner() const {
  return preconditioner_plan_ != nullptr;
}

void CUCG::configure_execution_policy(
    SparseSolveExecutionPolicy policy,
    int host_check_interval) {
  std::lock_guard<std::mutex> lock(solve_mutex_);
  TI_ERROR_IF(solve_calls_ != 0,
              "CUDA CG execution policy must be configured before solve.");
  validate_sparse_solve_execution_policy(Arch::cuda, policy,
                                         host_check_interval);
  TI_ERROR_IF(policy == SparseSolveExecutionPolicy::host_check_every_k &&
                  host_check_interval != 4 && host_check_interval != 8,
              "CUDA host_check_every_k currently supports K=4 or K=8.");
  execution_policy_ = policy;
  host_check_interval_ = host_check_interval;
}

void CUCG::validate_controls() const {
  validate_sparse_solve_execution_policy(
      Arch::cuda, SparseSolveExecutionPolicy::host_each_iteration);
  TI_ERROR_IF(max_iters_ < 0,
              "CUDA SparseCG requires non-negative max iterations.");
  TI_ERROR_IF(!std::isfinite(absolute_tolerance_) ||
                  !std::isfinite(relative_tolerance_) ||
                  absolute_tolerance_ < 0.0f ||
                  relative_tolerance_ < 0.0f ||
                  (absolute_tolerance_ == 0.0f &&
                   relative_tolerance_ == 0.0f),
              "CUDA SparseCG requires finite non-negative atol and rtol "
              "with at least one positive tolerance.");
}

void CUCG::apply_preconditioner(Program *program,
                                const OperatorPinnedAction &generation,
                                float *input,
                                float *output,
                                const Ndarray *input_array,
                                const Ndarray *output_array) {
  TI_ERROR_IF(!preconditioner_plan_,
              "CUDA CG preconditioner plan is not initialized.");
  auto &action = preconditioner_plan_->action();
  const auto &descriptor = action.descriptor();
  const auto input_view =
      input_array
          ? OperatorVectorView::from_ndarray(
                program, *input_array, descriptor.domain, false)
          : OperatorVectorView::from_device_pointer(
                program, reinterpret_cast<std::uintptr_t>(input),
                descriptor.domain, false);
  const auto output_view =
      output_array
          ? OperatorVectorView::from_ndarray(
                program, *output_array, descriptor.range, true)
          : OperatorVectorView::from_device_pointer(
                program, reinterpret_cast<std::uintptr_t>(output),
                descriptor.range, true);
  action.submit(
      generation,
      {OperatorApplyMode::forward, input_view, nullptr, output_view});
}

void CUCG::apply_operator(Program *program,
                          const OperatorPinnedAction &generation,
                          std::uintptr_t input,
                          std::uintptr_t output,
                          const Ndarray *input_array,
                          const Ndarray *output_array) {
  TI_ERROR_IF(!operator_plan_,
              "CUDA CG operator plan is not initialized.");
  const auto &descriptor = operator_plan_->descriptor();
  const auto input_view =
      input_array
          ? OperatorVectorView::from_ndarray(
                program, *input_array, descriptor.domain, false)
          : OperatorVectorView::from_device_pointer(
                program, input, descriptor.domain, false);
  const auto output_view =
      output_array
          ? OperatorVectorView::from_ndarray(
                program, *output_array, descriptor.range, true)
          : OperatorVectorView::from_device_pointer(
                program, output, descriptor.range, true);
  operator_plan_->submit(
      generation,
      {OperatorApplyMode::forward, input_view, nullptr, output_view});
}

void CUCG::ensure_operator_plan(Program *program) {
  if (operator_plan_) {
    TI_ERROR_IF(program_ && program != program_,
                "CUDA CG must keep using its construction Program.");
    return;
  }
  TI_ERROR_IF(!program || program->compile_config().arch != Arch::cuda ||
                  !cuda_csr_operator_,
              "CUDA CG compatibility binding requires an active CUDA "
              "Program and CSR operator.");
  program_ = program;
  operator_plan_ = std::make_unique<OperatorPlan>(
      program_, with_legacy_cg_traits(make_cuda_csr_operator_binding(
                    program_, *cuda_csr_operator_)));
  validate_cg_plan(*operator_plan_, nullptr);
}

bool CUCG::native_solver_chunk_eligible() const {
  return native_solver_chunk_unavailable_reason() == "none";
}

std::string CUCG::native_solver_chunk_unavailable_reason() const {
#if defined(TI_WITH_CUDA)
  if (get_environ_config("TI_CUDA_SOLVER_CHUNK_REPLAY", 1) == 0) {
    return "native_solver_chunk_replay_disabled";
  }
  if (!program_) {
    return "program_unavailable";
  }
  if (program_->compile_config().debug) {
    return "debug_mode_enabled";
  }
  if ((!cuda_csr_operator_ && !cuda_bsr_operator_) ||
      (has_preconditioner() && !preconditioner_ && !block_preconditioner_)) {
    return "provider_not_capture_composable";
  }
  const bool stream_binding =
      cuda_csr_operator_
          ? cuda_csr_operator_->supports_spmv_stream_binding()
          : cuda_bsr_operator_->supports_spmv_stream_binding();
  if (!stream_binding) {
    return "spmv_stream_binding_unavailable";
  }
  auto &driver = CUDADriver::get_instance();
  if (!driver.stream_begin_capture.available() ||
      !driver.stream_end_capture.available() ||
      !driver.graph_instantiate_with_flags.available() ||
      !driver.graph_launch.available() || !driver.graph_destroy.available() ||
      !driver.graph_exec_destroy.available()) {
    return "cuda_graph_capture_unavailable";
  }
  return "none";
#else
  return "cuda_backend_unavailable";
#endif
}

void CUCG::issue_native_solver_iteration(
    Program *program,
    CUstream stream,
    float *d_x,
    float *d_ax,
    float *d_r,
    float *d_p,
    float *d_z,
    cuda::CudaCGScalarState *state) {
#if defined(TI_WITH_CUDA)
  if (cuda_csr_operator_) {
    cuda_csr_operator_->spmv(reinterpret_cast<std::uintptr_t>(d_p),
                             reinterpret_cast<std::uintptr_t>(d_ax), stream);
  } else {
    TI_ASSERT(cuda_bsr_operator_);
    cuda_bsr_operator_->spmv(reinterpret_cast<std::uintptr_t>(d_p),
                             reinterpret_cast<std::uintptr_t>(d_ax), stream);
  }
  auto &cublas = CUBLASDriver::get_instance();
  const int rows = A_.num_rows();
  cublas.cubSdot(handle_, rows, d_p, 1, d_ax, 1, &state->p_ap);
  cuda::driver_cg_prepare_alpha(state, stream);
  cublas.cubSaxpy(handle_, rows, &state->alpha, d_p, 1, d_x, 1);
  cublas.cubSaxpy(handle_, rows, &state->negative_alpha, d_ax, 1, d_r, 1);
  cublas.cubSdot(handle_, rows, d_r, 1, d_r, 1, &state->rr_next);
  cuda::driver_cg_finish_iteration(state, stream);
  if (preconditioner_) {
    preconditioner_->apply_cuda_raw(
        program, reinterpret_cast<std::uintptr_t>(d_r),
        reinterpret_cast<std::uintptr_t>(d_z), stream);
    cublas.cubSdot(handle_, rows, d_r, 1, d_z, 1, &state->rho_next);
  } else if (block_preconditioner_) {
    block_preconditioner_->apply_cuda_raw(
        program, reinterpret_cast<std::uintptr_t>(d_r),
        reinterpret_cast<std::uintptr_t>(d_z), stream);
    cublas.cubSdot(handle_, rows, d_r, 1, d_z, 1, &state->rho_next);
  }
  cuda::driver_cg_prepare_direction(state, stream);
  cublas.cubSscal(handle_, rows, &state->beta, d_p, 1);
  cublas.cubSaxpy(handle_, rows, &state->source_scale,
                  has_preconditioner() ? d_z : d_r, 1, d_p, 1);
#else
  TI_NOT_IMPLEMENTED;
#endif
}

bool CUCG::try_submit_solver_chunk(
    Program *program,
    const Ndarray &x,
    const OperatorPinnedAction &operator_generation,
    const OperatorPinnedAction &preconditioner_generation,
    int chunk_iterations,
    float *d_x,
    float *d_ax,
    float *d_r,
    float *d_p,
    float *d_z,
    cuda::CudaCGScalarState *state) {
#if defined(TI_WITH_CUDA)
  if (!native_solver_chunk_eligible() || chunk_iterations <= 0 ||
      chunk_iterations > 8) {
    return false;
  }
  if (!solver_chunk_replay_state_) {
    solver_chunk_replay_state_ =
        std::make_unique<CudaSolverChunkReplayState>();
  }
  auto &replay = *solver_chunk_replay_state_;
  if (replay.disabled) {
    return false;
  }
  const auto operator_stamp = operator_generation.resource_stamp();
  const auto preconditioner_stamp =
      preconditioner_generation
          ? preconditioner_generation.resource_stamp()
          : OperatorResourceStamp{};
  CudaSolverChunkReplayKey key;
  key.solution = x.runtime_resource_handle();
  key.program_generation = operator_stamp.program_generation;
  key.schema_revision = operator_stamp.schema_revision;
  key.topology_revision = operator_stamp.topology_revision;
  key.binding_revision = operator_stamp.binding_revision;
  key.preconditioner_schema_revision = preconditioner_stamp.schema_revision;
  key.preconditioner_topology_revision =
      preconditioner_stamp.topology_revision;
  key.preconditioner_binding_revision =
      preconditioner_stamp.binding_revision;
  key.x = reinterpret_cast<std::uintptr_t>(d_x);
  key.ax = reinterpret_cast<std::uintptr_t>(d_ax);
  key.residual = reinterpret_cast<std::uintptr_t>(d_r);
  key.direction = reinterpret_cast<std::uintptr_t>(d_p);
  key.preconditioned_residual = reinterpret_cast<std::uintptr_t>(d_z);
  key.scalars = reinterpret_cast<std::uintptr_t>(state);
  key.provider = reinterpret_cast<std::uintptr_t>(
      cuda_csr_operator_ ? static_cast<void *>(cuda_csr_operator_)
                         : static_cast<void *>(cuda_bsr_operator_));
  key.preconditioner = reinterpret_cast<std::uintptr_t>(
      preconditioner_ ? static_cast<void *>(preconditioner_)
                      : static_cast<void *>(block_preconditioner_));
  key.rows = A_.num_rows();
  key.iterations = chunk_iterations;

  auto &entry = replay.entries[static_cast<std::size_t>(chunk_iterations)];
  if (entry.executable != nullptr && entry.key_valid && entry.key == key) {
    if (entry.operator_numeric_revision != operator_stamp.numeric_revision ||
        entry.preconditioner_numeric_revision !=
            preconditioner_stamp.numeric_revision) {
      entry.operator_numeric_revision = operator_stamp.numeric_revision;
      entry.preconditioner_numeric_revision =
          preconditioner_stamp.numeric_revision;
      ++solver_chunk_rebinds_;
    }
    CUDADriver::get_instance().graph_launch(entry.executable, nullptr);
    if (preconditioner_) {
      preconditioner_->record_replayed_apply_calls(chunk_iterations);
    } else if (block_preconditioner_) {
      block_preconditioner_->record_replayed_apply_calls(chunk_iterations);
    }
    ++solver_chunk_reuses_;
    ++solver_chunk_replays_;
    replay.unavailable_reason = "none";
    return true;
  }

  if (entry.executable != nullptr) {
    if (entry.key_valid && entry.key.solution != key.solution) {
      ++solver_chunk_rebinds_;
    }
    CUDADriver::get_instance().stream_synchronize(nullptr);
    entry.reset();
    ++solver_chunk_invalidations_;
  }

  // Warm cuSPARSE descriptors and external storage before capture. The extra
  // SpMV is a cold-build cost only; the captured iteration overwrites d_ax
  // before any solver scalar consumes it.
  if (cuda_csr_operator_) {
    cuda_csr_operator_->spmv(reinterpret_cast<std::uintptr_t>(d_p),
                             reinterpret_cast<std::uintptr_t>(d_ax), nullptr);
  } else {
    cuda_bsr_operator_->spmv(reinterpret_cast<std::uintptr_t>(d_p),
                             reinterpret_cast<std::uintptr_t>(d_ax), nullptr);
  }

  auto &driver = CUDADriver::get_instance();
  auto &cublas = CUBLASDriver::get_instance();
  const CUstream capture_stream = replay.ensure_capture_stream();
  driver.stream_synchronize(capture_stream);
  cublas.cubSetStream(handle_, capture_stream);
  CUgraph graph = nullptr;
  auto capture_lock = CUDAContext::get_instance().get_graph_capture_lock_guard();
  const auto begin_error = driver.stream_begin_capture.call(
      capture_stream, CU_STREAM_CAPTURE_MODE_RELAXED);
  if (begin_error != CUDA_SUCCESS) {
    cublas.cubSetStream(handle_, solver_stream_);
    replay.disabled = true;
    replay.unavailable_reason = "stream_capture_begin_failed";
    return false;
  }
  try {
    for (int i = 0; i < chunk_iterations; ++i) {
      issue_native_solver_iteration(program, capture_stream, d_x, d_ax, d_r,
                                    d_p, d_z, state);
    }
  } catch (...) {
    (void)driver.stream_end_capture.call(capture_stream, &graph);
    if (graph != nullptr) {
      driver.graph_destroy.call(graph);
    }
    cublas.cubSetStream(handle_, solver_stream_);
    throw;
  }
  const auto end_error =
      driver.stream_end_capture.call(capture_stream, &graph);
  cublas.cubSetStream(handle_, solver_stream_);
  if (end_error != CUDA_SUCCESS || graph == nullptr) {
    if (graph != nullptr) {
      driver.graph_destroy.call(graph);
    }
    replay.disabled = true;
    replay.unavailable_reason = "stream_capture_end_failed";
    return false;
  }
  CUgraphExec executable = nullptr;
  const auto instantiate_error = driver.graph_instantiate_with_flags.call(
      &executable, graph, 0);
  driver.graph_destroy.call(graph);
  if (instantiate_error != CUDA_SUCCESS || executable == nullptr) {
    replay.disabled = true;
    replay.unavailable_reason = "graph_instantiate_failed";
    return false;
  }
  entry.executable = executable;
  entry.key = key;
  entry.key_valid = true;
  entry.operator_numeric_revision = operator_stamp.numeric_revision;
  entry.preconditioner_numeric_revision =
      preconditioner_stamp.numeric_revision;
  entry.solution_lease.emplace(
      program->acquire_ndarray_external_lease(key.solution));
  ++solver_chunk_builds_;
  replay.unavailable_reason = "none";
  driver.graph_launch(entry.executable, nullptr);
  return true;
#else
  return false;
#endif
}

void CUCG::init_solver() {
#if defined(TI_WITH_CUDA)
  if (!CUBLASDriver::get_instance().is_loaded()) {
    bool load_success = CUBLASDriver::get_instance().load_cublas();
    if (!load_success) {
      TI_ERROR("Failed to load cublas library!");
    }
  }
  auto &cublas = CUBLASDriver::get_instance();
  TI_ERROR_IF(!cublas.cubSetPointerMode.available() ||
                  !cublas.cubGetPointerMode.available() ||
                  !cublas.cubSetStream.available() ||
                  !cublas.cubGetStream.available(),
              "CUDA CG requires cuBLAS pointer-mode and stream binding "
              "symbols.");
  cublas.cubCreate(&handle_);
  TI_ERROR_IF(!handle_,
              "CUDA CG failed to create its dedicated cuBLAS handle.");
  // Program CUDA work currently uses the legacy default stream. Keep the
  // solver stream behind one explicit binding point so a future RHI stream
  // handoff cannot silently diverge from operator actions.
  solver_stream_ = nullptr;
  cublas.cubSetStream(handle_, solver_stream_);
  CUstream bound_stream = reinterpret_cast<CUstream>(1);
  cublas.cubGetStream(handle_, &bound_stream);
  TI_ERROR_IF(bound_stream != solver_stream_,
              "CUDA CG cuBLAS stream binding could not be verified.");
  cublas_stream_bound_ = true;
  cublas.cubSetPointerMode(handle_, CUBLAS_POINTER_MODE_HOST);
  cublasPointerMode_t pointer_mode = CUBLAS_POINTER_MODE_DEVICE;
  cublas.cubGetPointerMode(handle_, &pointer_mode);
  TI_ERROR_IF(pointer_mode != CUBLAS_POINTER_MODE_HOST,
              "CUDA CG cuBLAS pointer mode could not be verified.");
  cublas_device_pointer_mode_ = false;
  int version;
  cublas.cubGetVersion(handle_, &version);
  TI_TRACE("CUBLAS version: {}\n", version);
#endif
}

CUCG::~CUCG() {
#if defined(TI_WITH_CUDA)
  release_workspace();
  if (handle_) {
    CUBLASDriver::get_instance().cubDestroy(handle_);
  }
#endif
}

void CUCG::ensure_workspace(Program *program, int size) {
#if defined(TI_WITH_CUDA)
  const bool needs_scalars =
      execution_policy_ == SparseSolveExecutionPolicy::host_check_every_k;
  if (workspace_size_ == size && workspace_ax_ && workspace_r_ &&
      workspace_p_ && (!has_preconditioner() || workspace_z_) &&
      (!needs_scalars || workspace_scalars_)) {
    workspace_reuses_++;
    return;
  }
  release_workspace();
  if (size <= 0) {
    return;
  }
  if (compiled_kernel_operator_ || compiled_graph_operator_) {
    TI_ERROR_IF(program != program_,
                "CUDA program-bound CG workspace requires its owning "
                "Program.");
    auto create_vector = [&]() {
      return program->create_ndarray(PrimitiveType::f32, {size},
                                     ExternalArrayLayout::kNull, false);
    };
    try {
      workspace_ax_ndarray_ = create_vector();
      workspace_r_ndarray_ = create_vector();
      workspace_p_ndarray_ = create_vector();
      if (has_preconditioner()) {
        workspace_z_ndarray_ = create_vector();
      }
      workspace_ax_ = reinterpret_cast<float *>(
          program->get_ndarray_data_ptr_as_int(workspace_ax_ndarray_));
      workspace_r_ = reinterpret_cast<float *>(
          program->get_ndarray_data_ptr_as_int(workspace_r_ndarray_));
      workspace_p_ = reinterpret_cast<float *>(
          program->get_ndarray_data_ptr_as_int(workspace_p_ndarray_));
      if (workspace_z_ndarray_) {
        workspace_z_ = reinterpret_cast<float *>(
            program->get_ndarray_data_ptr_as_int(workspace_z_ndarray_));
      }
      if (needs_scalars) {
        CUDADriver::get_instance().malloc(
            &workspace_scalars_, sizeof(cuda::CudaCGScalarState));
      }
    } catch (...) {
      release_workspace();
      throw;
    }
    workspace_size_ = size;
    workspace_builds_++;
    return;
  }
  CUDADriver::get_instance().malloc((void **)&workspace_ax_,
                                    sizeof(float) * size);
  CUDADriver::get_instance().malloc((void **)&workspace_r_,
                                    sizeof(float) * size);
  CUDADriver::get_instance().malloc((void **)&workspace_p_,
                                    sizeof(float) * size);
  if (has_preconditioner()) {
    CUDADriver::get_instance().malloc((void **)&workspace_z_,
                                      sizeof(float) * size);
  }
  if (needs_scalars) {
    CUDADriver::get_instance().malloc(
        &workspace_scalars_, sizeof(cuda::CudaCGScalarState));
  }
  workspace_size_ = size;
  workspace_builds_++;
#endif
}

void CUCG::release_workspace() {
#if defined(TI_WITH_CUDA)
  solver_chunk_replay_state_.reset();
  if (workspace_ax_ndarray_ && program_)
    program_->delete_ndarray(workspace_ax_ndarray_);
  else if (workspace_ax_)
    CUDADriver::get_instance().mem_free(workspace_ax_);
  if (workspace_r_ndarray_ && program_)
    program_->delete_ndarray(workspace_r_ndarray_);
  else if (workspace_r_)
    CUDADriver::get_instance().mem_free(workspace_r_);
  if (workspace_p_ndarray_ && program_)
    program_->delete_ndarray(workspace_p_ndarray_);
  else if (workspace_p_)
    CUDADriver::get_instance().mem_free(workspace_p_);
  if (workspace_z_ndarray_ && program_)
    program_->delete_ndarray(workspace_z_ndarray_);
  else if (workspace_z_)
    CUDADriver::get_instance().mem_free(workspace_z_);
  if (workspace_scalars_)
    CUDADriver::get_instance().mem_free(workspace_scalars_);
  workspace_ax_ndarray_ = nullptr;
  workspace_r_ndarray_ = nullptr;
  workspace_p_ndarray_ = nullptr;
  workspace_z_ndarray_ = nullptr;
  workspace_ax_ = nullptr;
  workspace_r_ = nullptr;
  workspace_p_ = nullptr;
  workspace_z_ = nullptr;
  workspace_scalars_ = nullptr;
  workspace_size_ = 0;
#endif
}

void CUCG::solve_device_scalar(
    Program *prog,
    const Ndarray &x,
    const Ndarray &b,
    const OperatorPinnedAction &operator_generation,
    const OperatorPinnedAction &preconditioner_generation) {
#if defined(TI_WITH_CUDA)
  TI_ERROR_IF(!workspace_scalars_ || !cuda::driver_cg_scalar_available(),
              "CUDA host-check CG requires the device scalar primitive.");
  auto *state =
      static_cast<cuda::CudaCGScalarState *>(workspace_scalars_);
  auto &driver = CUDADriver::get_instance();
  auto &cublas = CUBLASDriver::get_instance();
  cublas.cubSetPointerMode(handle_, CUBLAS_POINTER_MODE_DEVICE);
  cublasPointerMode_t pointer_mode = CUBLAS_POINTER_MODE_HOST;
  cublas.cubGetPointerMode(handle_, &pointer_mode);
  TI_ERROR_IF(pointer_mode != CUBLAS_POINTER_MODE_DEVICE,
              "CUDA host-check CG could not enable device pointer mode.");
  cublas_device_pointer_mode_ = true;

  cuda::CudaCGScalarState host_state;
  host_state.absolute_tolerance = absolute_tolerance_;
  host_state.relative_tolerance = relative_tolerance_;
  host_state.has_preconditioner = has_preconditioner() ? 1 : 0;
  driver.memcpy_host_to_device(state, &host_state, sizeof(host_state));
  host_to_device_bytes_ += sizeof(host_state);

  const auto d_x = prog->get_ndarray_data_ptr_as_int(&x);
  const auto d_b = prog->get_ndarray_data_ptr_as_int(&b);
  const int rows = A_.num_rows();
  auto *d_ax = workspace_ax_;
  auto *d_r = workspace_r_;
  auto *d_p = workspace_p_;
  auto *d_z = workspace_z_;
  auto read_state = [&]() {
    driver.memcpy_device_to_host(&host_state, state, sizeof(host_state));
    host_scalar_readbacks_++;
    host_synchronizations_++;
    device_to_host_bytes_ += sizeof(host_state);
  };

  driver.memcpy_device_to_device(d_r, reinterpret_cast<void *>(d_b),
                                 sizeof(float) * rows);
  device_to_device_bytes_ += sizeof(float) * rows;
  apply_operator(prog, operator_generation, d_x,
                 reinterpret_cast<std::uintptr_t>(d_ax), &x,
                 workspace_ax_ndarray_);
  operator_apply_calls_++;
  cublas.cubSaxpy(handle_, rows, &state->negative_one, d_ax, 1, d_r, 1);
  cublas.cubSdot(handle_, rows, d_r, 1, d_r, 1, &state->rr_current);
  device_scalar_operations_++;
  if (relative_tolerance_ > 0.0f) {
    const auto *rhs = reinterpret_cast<const float *>(d_b);
    cublas.cubSdot(handle_, rows, rhs, 1, rhs, 1, &state->rhs_squared);
    device_scalar_operations_++;
  }
  cuda::driver_cg_initialize(state, solver_stream_);
  device_scalar_operations_++;
  read_state();

  if (host_state.active != 0 && max_iters_ > 0) {
    if (has_preconditioner()) {
      apply_preconditioner(prog, preconditioner_generation, d_r, d_z,
                           workspace_r_ndarray_, workspace_z_ndarray_);
      preconditioner_apply_calls_++;
      cublas.cubSdot(handle_, rows, d_r, 1, d_z, 1,
                     &state->rho_current);
      cuda::driver_cg_validate_rho(state, solver_stream_);
      device_scalar_operations_ += 2;
    }
    driver.memcpy_device_to_device(
        d_p, has_preconditioner() ? d_z : d_r, sizeof(float) * rows);
    device_to_device_bytes_ += sizeof(float) * rows;
  }

  int issued_iterations = 0;
  while (host_state.active != 0 && issued_iterations < max_iters_) {
    const int chunk_iterations =
        std::min(host_check_interval_, max_iters_ - issued_iterations);
    const bool submitted = try_submit_solver_chunk(
        prog, x, operator_generation, preconditioner_generation,
        chunk_iterations, reinterpret_cast<float *>(d_x), d_ax, d_r, d_p,
        d_z, state);
    if (!submitted) {
      ++solver_chunk_direct_submissions_;
      for (int chunk_index = 0; chunk_index < chunk_iterations;
           ++chunk_index) {
        apply_operator(prog, operator_generation,
                       reinterpret_cast<std::uintptr_t>(d_p),
                       reinterpret_cast<std::uintptr_t>(d_ax),
                       workspace_p_ndarray_, workspace_ax_ndarray_);
        cublas.cubSdot(handle_, rows, d_p, 1, d_ax, 1, &state->p_ap);
        cuda::driver_cg_prepare_alpha(state, solver_stream_);
        cublas.cubSaxpy(handle_, rows, &state->alpha, d_p, 1,
                        reinterpret_cast<float *>(d_x), 1);
        cublas.cubSaxpy(handle_, rows, &state->negative_alpha, d_ax, 1, d_r,
                        1);
        cublas.cubSdot(handle_, rows, d_r, 1, d_r, 1, &state->rr_next);
        cuda::driver_cg_finish_iteration(state, solver_stream_);
        if (has_preconditioner()) {
          apply_preconditioner(prog, preconditioner_generation, d_r, d_z,
                               workspace_r_ndarray_, workspace_z_ndarray_);
          cublas.cubSdot(handle_, rows, d_r, 1, d_z, 1,
                         &state->rho_next);
        }
        cuda::driver_cg_prepare_direction(state, solver_stream_);
        cublas.cubSscal(handle_, rows, &state->beta, d_p, 1);
        cublas.cubSaxpy(handle_, rows, &state->source_scale,
                        has_preconditioner() ? d_z : d_r, 1, d_p, 1);
      }
    }
    issued_iterations += chunk_iterations;
    executed_iterations_ += static_cast<std::uint64_t>(chunk_iterations);
    operator_apply_calls_ += static_cast<std::uint64_t>(chunk_iterations);
    device_scalar_operations_ += static_cast<std::uint64_t>(
        chunk_iterations * (has_preconditioner() ? 6 : 5));
    if (has_preconditioner()) {
      preconditioner_apply_calls_ +=
          static_cast<std::uint64_t>(chunk_iterations);
    }
    read_state();
    if (verbose_) {
      fmt::print("chunk: {}, completed: {}, rr: {}\n",
                 solver_chunk_direct_submissions_,
                 host_state.completed_iterations, host_state.rr_current);
    }
  }

  iterations_ = host_state.completed_iterations;
  initial_residual_norm_ =
      std::isfinite(host_state.initial_rr) && host_state.initial_rr >= 0.0f
          ? std::sqrt(static_cast<double>(host_state.initial_rr))
          : std::numeric_limits<double>::quiet_NaN();
  residual_norm_ =
      std::isfinite(host_state.rr_current) && host_state.rr_current >= 0.0f
          ? std::sqrt(static_cast<double>(host_state.rr_current))
          : std::numeric_limits<double>::quiet_NaN();
  relative_reference_norm_ = host_state.relative_reference_norm;
  effective_tolerance_ = host_state.effective_tolerance;
  status_ = static_cast<SparseSolveStatus>(host_state.status);
  total_iterations_ += static_cast<std::uint64_t>(iterations_);
#else
  TI_NOT_IMPLEMENTED;
#endif
}

void CUCG::solve(Program *prog, const Ndarray &x, const Ndarray &b) {
#if defined(TI_WITH_CUDA)
  std::lock_guard<std::mutex> lock(solve_mutex_);
  TI_ERROR_IF(
      (compiled_kernel_operator_ || compiled_graph_operator_) &&
          (prog != program_ || x.owning_program() != program_ ||
           b.owning_program() != program_),
      "CUDA program-bound CG requires solution and RHS ndarrays owned by "
      "its construction Program.");
  ensure_operator_plan(prog);
  auto operator_generation = operator_plan_->pin();
  auto preconditioner_generation =
      preconditioner_plan_
          ? preconditioner_plan_->update_and_pin(operator_generation)
          : OperatorPinnedAction{};
  if (has_preconditioner()) {
    TI_ERROR_IF(prog != program_,
                "CUDA preconditioned CG must be solved by its construction "
                "Program.");
  }
  const auto operator_stamp = operator_generation.resource_stamp();
  solve_calls_++;
  last_solve_pattern_version_ = operator_stamp.topology_revision;
  last_solve_numeric_version_ = operator_stamp.numeric_revision;
  status_ = SparseSolveStatus::kNotRun;
  iterations_ = 0;
  initial_residual_norm_ = 0.0;
  residual_norm_ = 0.0;
  relative_reference_norm_ = 0.0;
  effective_tolerance_ =
      static_cast<double>(absolute_tolerance_);

  size_t dX = prog->get_ndarray_data_ptr_as_int(&x);
  size_t db = prog->get_ndarray_data_ptr_as_int(&b);
  int m = A_.num_rows();

  ensure_workspace(prog, m);
  if (execution_policy_ ==
      SparseSolveExecutionPolicy::host_check_every_k) {
    solve_device_scalar(prog, x, b, operator_generation,
                        preconditioner_generation);
    return;
  }
  float *d_Ax = workspace_ax_;
  float *d_r = workspace_r_;
  float *d_p = workspace_p_;
  float *d_z = workspace_z_;

  // r = b
  CUDADriver::get_instance().memcpy_device_to_device((void *)d_r, (void *)db,
                                                     sizeof(float) * m);
  device_to_device_bytes_ += sizeof(float) * m;

  // Ax = A @ x
  apply_operator(prog, operator_generation, dX, size_t(d_Ax), &x,
                 workspace_ax_ndarray_);
  operator_apply_calls_++;

  // r = r - Ax = b - Ax
  float alpham1 = -1.0f;
  CUBLASDriver::get_instance().cubSaxpy(handle_, m, &alpham1, d_Ax, 1, d_r, 1);

  float r1 = 0.0f;
  CUBLASDriver::get_instance().cubSdot(handle_, m, d_r, 1, d_r, 1, &r1);
  host_scalar_reductions_++;
  initial_residual_norm_ = std::sqrt(std::max(r1, 0.0f));

  bool breakdown = !std::isfinite(r1) || r1 < 0.0f ||
                   !std::isfinite(initial_residual_norm_);
  if (!breakdown && relative_tolerance_ > 0.0f) {
    float rhs_squared_norm = 0.0f;
    const auto *rhs = reinterpret_cast<const float *>(db);
    CUBLASDriver::get_instance().cubSdot(
        handle_, m, rhs, 1, rhs, 1, &rhs_squared_norm);
    host_scalar_reductions_++;
    if (!std::isfinite(rhs_squared_norm) || rhs_squared_norm < 0.0f) {
      breakdown = true;
    } else {
      relative_reference_norm_ =
          std::sqrt(static_cast<double>(rhs_squared_norm));
      effective_tolerance_ = std::max(
          static_cast<double>(absolute_tolerance_),
          static_cast<double>(relative_tolerance_) *
              relative_reference_norm_);
      breakdown = !std::isfinite(effective_tolerance_);
    }
  }

  float alpha = 1.0f;
  float beta = 0.0f;
  float r0 = 0.0f;
  float rho = r1;
  float rho0 = 0.0f;
  float dot = 0.0f;
  const double tolerance_squared =
      effective_tolerance_ * effective_tolerance_;
  if (!breakdown && has_preconditioner() && r1 > tolerance_squared &&
      max_iters_ > 0) {
    apply_preconditioner(prog, preconditioner_generation, d_r, d_z,
                         workspace_r_ndarray_, workspace_z_ndarray_);
    preconditioner_apply_calls_++;
    CUBLASDriver::get_instance().cubSdot(handle_, m, d_r, 1, d_z, 1,
                                         &rho);
    host_scalar_reductions_++;
    breakdown = !std::isfinite(rho) || rho <= 0.0f;
  }

  while (!breakdown && r1 > tolerance_squared &&
         iterations_ < max_iters_) {
    if (iterations_ > 0) {
      beta = has_preconditioner() ? rho / rho0 : r1 / r0;
      if (has_preconditioner() && !std::isfinite(beta)) {
        breakdown = true;
        break;
      }
      // p = z + beta * p for PCG, or r + beta * p for identity CG.
      CUBLASDriver::get_instance().cubSscal(handle_, m, &beta, d_p, 1);
      CUBLASDriver::get_instance().cubSaxpy(
          handle_, m, &alpha, has_preconditioner() ? d_z : d_r, 1, d_p, 1);
    } else {
      // p = z for PCG, or r for identity CG.
      CUDADriver::get_instance().memcpy_device_to_device(
          (void *)d_p, (void *)(has_preconditioner() ? d_z : d_r),
          sizeof(float) * m);
      device_to_device_bytes_ += sizeof(float) * m;
    }

    // Ap = A @ p
    apply_operator(prog, operator_generation, size_t(d_p), size_t(d_Ax),
                   workspace_p_ndarray_, workspace_ax_ndarray_);
    operator_apply_calls_++;
    // dot = p @ Ap
    CUBLASDriver::get_instance().cubSdot(handle_, m, d_p, 1, d_Ax, 1, &dot);
    host_scalar_reductions_++;
    if (!std::isfinite(dot) || dot <= 0.0f) {
      breakdown = true;
      break;
    }
    const float numerator = has_preconditioner() ? rho : r1;
    float a = numerator / dot;
    if (has_preconditioner() && !std::isfinite(a)) {
      breakdown = true;
      break;
    }
    // x = x + a * p
    CUBLASDriver::get_instance().cubSaxpy(handle_, m, &a, d_p, 1, (float *)dX,
                                          1);
    // r = r - a * Ap
    float na = -a;
    CUBLASDriver::get_instance().cubSaxpy(handle_, m, &na, d_Ax, 1, d_r, 1);
    r0 = r1;
    if (has_preconditioner()) {
      rho0 = rho;
    }
    // r1 = r @ r
    CUBLASDriver::get_instance().cubSdot(handle_, m, d_r, 1, d_r, 1, &r1);
    host_scalar_reductions_++;
    iterations_++;
    if (has_preconditioner() && std::isfinite(r1) &&
        r1 > tolerance_squared && iterations_ < max_iters_) {
      apply_preconditioner(prog, preconditioner_generation, d_r, d_z,
                           workspace_r_ndarray_, workspace_z_ndarray_);
      preconditioner_apply_calls_++;
      CUBLASDriver::get_instance().cubSdot(handle_, m, d_r, 1, d_z, 1,
                                           &rho);
      host_scalar_reductions_++;
      if (!std::isfinite(rho) || rho <= 0.0f) {
        breakdown = true;
      }
    }
    if (verbose_)
      fmt::print("iter: {}, r1: {}\n", iterations_, r1);
  }
  residual_norm_ = std::sqrt(std::max(r1, 0.0f));
  total_iterations_ += static_cast<std::uint64_t>(iterations_);
  if (breakdown || !std::isfinite(r1) ||
      !std::isfinite(initial_residual_norm_) ||
      !std::isfinite(residual_norm_)) {
    status_ = SparseSolveStatus::kBreakdown;
  } else if (residual_norm_ <= effective_tolerance_) {
    status_ = SparseSolveStatus::kConverged;
  } else {
    status_ = SparseSolveStatus::kMaxIterations;
  }

#endif
}

SparseSolvePlanRuntimeStatistics CUCG::debug_runtime_statistics() const {
  std::lock_guard<std::mutex> lock(solve_mutex_);
  const auto operator_stats = A_.debug_runtime_statistics();
  SparseSolvePlanRuntimeStatistics result;
  result.backend_family = "cuda";
  if (compiled_kernel_operator_ || compiled_graph_operator_) {
    result.method = has_preconditioner() ? "pcg_compiled_kernel"
                                         : (compiled_graph_operator_
                                                ? "cg_compiled_graph"
                                                : "cg_compiled_kernel");
    result.preconditioner_method =
        preconditioner_plan_ ? preconditioner_plan_->method() : "identity";
    result.external_preconditioner = has_preconditioner();
    result.preconditioner_ownership_scope =
        has_preconditioner() ? "external_plan" : "none";
  } else if (has_preconditioner()) {
    result.method =
        block_preconditioner_ ? "pcg_block_jacobi" : "pcg_jacobi";
    result.preconditioner_method =
        block_preconditioner_ ? "block_jacobi" : "jacobi";
    result.external_preconditioner = true;
    result.preconditioner_ownership_scope = "external_plan";
  }
  result.dtype = data_type_name(A_.get_data_type());
  result.rows = A_.num_rows();
  result.cols = A_.num_cols();
  result.max_iterations = max_iters_;
  result.absolute_tolerance =
      static_cast<double>(absolute_tolerance_);
  result.relative_tolerance =
      static_cast<double>(relative_tolerance_);
  result.last_relative_reference_norm = relative_reference_norm_;
  result.last_effective_tolerance = effective_tolerance_;
  result.operator_pattern_version = operator_stats.pattern_version;
  result.operator_numeric_version = operator_stats.numeric_version;
  result.last_solve_pattern_version = last_solve_pattern_version_;
  result.last_solve_numeric_version = last_solve_numeric_version_;
  result.operator_pattern_changed_since_last_solve =
      solve_calls_ > 0 && result.operator_pattern_version !=
                              result.last_solve_pattern_version;
  result.operator_numeric_changed_since_last_solve =
      solve_calls_ > 0 && result.operator_numeric_version !=
                              result.last_solve_numeric_version;
  result.solve_calls = solve_calls_;
  result.total_iterations = total_iterations_;
  result.logical_iterations = total_iterations_;
  result.executed_iterations =
      execution_policy_ == SparseSolveExecutionPolicy::host_check_every_k
          ? executed_iterations_
          : total_iterations_;
  result.wasted_iterations =
      result.executed_iterations - result.logical_iterations;
  result.workspace_builds = workspace_builds_;
  result.workspace_reuses = workspace_reuses_;
  result.operator_apply_calls = operator_apply_calls_;
  result.operator_apply_calls_available = true;
  result.preconditioner_apply_calls = preconditioner_apply_calls_;
  result.host_scalar_reductions = host_scalar_reductions_;
  result.device_scalar_operations = device_scalar_operations_;
  result.host_scalar_readbacks = host_scalar_readbacks_;
  result.host_synchronizations = host_synchronizations_;
  result.requested_solver_execution_policy =
      sparse_solve_execution_policy_name(execution_policy_);
  result.solver_execution_policy =
      sparse_solve_execution_policy_name(execution_policy_);
  result.host_check_interval = host_check_interval_;
  result.solver_chunk_builds = solver_chunk_builds_;
  result.solver_chunk_reuses = solver_chunk_reuses_;
  result.solver_chunk_direct_submissions =
      solver_chunk_direct_submissions_;
  result.solver_chunk_replays = solver_chunk_replays_;
  result.solver_chunk_rebinds = solver_chunk_rebinds_;
  result.solver_chunk_invalidations = solver_chunk_invalidations_;
  result.solver_graph_enabled = solver_chunk_builds_ > 0;
  result.solver_replay_unavailable_reason =
      execution_policy_ == SparseSolveExecutionPolicy::host_check_every_k
          ? (result.solver_graph_enabled
                 ? "none"
                 : (solver_chunk_replay_state_
                        ? solver_chunk_replay_state_->unavailable_reason
                        : (native_solver_chunk_eligible()
                               ? "not_built"
                               : native_solver_chunk_unavailable_reason())))
          : "not_requested";
  result.persistent_vector_count =
      workspace_ax_ != nullptr && workspace_r_ != nullptr &&
              workspace_p_ != nullptr &&
              (!has_preconditioner() || workspace_z_ != nullptr)
          ? (has_preconditioner() ? 4 : 3)
          : 0;
  result.persistent_vector_reserved_bytes =
      result.persistent_vector_count == 0
          ? 0
          : result.persistent_vector_count *
                static_cast<std::uint64_t>(workspace_size_) * sizeof(float);
  result.cublas_handle_count = handle_ != nullptr ? 1 : 0;
  result.cublas_stream_bound = cublas_stream_bound_;
  result.cublas_device_pointer_mode = cublas_device_pointer_mode_;
  result.solver_scalar_location =
      execution_policy_ == SparseSolveExecutionPolicy::host_check_every_k
          ? "device"
          : "host";
  result.solver_stream_policy = result.solver_graph_enabled
                                    ? "capture_stream_default_replay"
                                    : "legacy_default_stream";
  result.persistent_scalar_count = workspace_scalars_ ? 23 : 0;
  result.persistent_scalar_reserved_bytes =
      workspace_scalars_ ? sizeof(cuda::CudaCGScalarState) : 0;
  result.device_to_device_bytes = device_to_device_bytes_;
  result.device_to_host_bytes = device_to_host_bytes_;
  result.host_to_device_bytes = host_to_device_bytes_;
  if (operator_plan_) {
    append_operator_plan_statistics(*operator_plan_, false, result);
  }
  if (preconditioner_plan_) {
    append_preconditioner_plan_statistics(*preconditioner_plan_, result);
  }
  return result;
}

std::unique_ptr<CUCG> make_cucg_solver(SparseMatrix &A,
                                       int max_iters,
                                       float absolute_tolerance,
                                       bool verbose,
                                       float relative_tolerance) {
  return std::make_unique<CUCG>(A, max_iters, absolute_tolerance,
                                verbose, relative_tolerance);
}

std::unique_ptr<CUCG> make_cuda_jacobi_pcg_solver(
    Program *program,
    SparseMatrix &A,
    SparseJacobiPreconditionerPlan &preconditioner,
    int max_iters,
    float absolute_tolerance,
    bool verbose,
    float relative_tolerance) {
  return std::make_unique<CUCG>(program, A, preconditioner, max_iters,
                                absolute_tolerance, verbose,
                                relative_tolerance);
}

std::unique_ptr<CUCG> make_cuda_block_jacobi_pcg_solver(
    Program *program,
    SparseMatrix &A,
    SparseBlockJacobiPreconditionerPlan &preconditioner,
    int max_iters,
    float absolute_tolerance,
    bool verbose,
    float relative_tolerance) {
  return std::make_unique<CUCG>(program, A, preconditioner, max_iters,
                                absolute_tolerance, verbose,
                                relative_tolerance);
}

std::unique_ptr<CUCG> make_cuda_compiled_kernel_cg_solver(
    Program *program,
    CompiledKernelLinearOperator &A,
    int max_iters,
    float absolute_tolerance,
    bool verbose,
    float relative_tolerance) {
  return std::make_unique<CUCG>(program, A, nullptr, max_iters,
                                absolute_tolerance, verbose,
                                relative_tolerance);
}

std::unique_ptr<CUCG> make_cuda_compiled_graph_cg_solver(
    Program *program,
    CompiledGraphLinearOperator &A,
    int max_iters,
    float absolute_tolerance,
    bool verbose,
    float relative_tolerance) {
  return std::make_unique<CUCG>(program, A, max_iters, absolute_tolerance,
                                verbose, relative_tolerance);
}

std::unique_ptr<CUCG> make_cuda_compiled_kernel_pcg_solver(
    Program *program,
    CompiledKernelLinearOperator &A,
    CompiledKernelPreconditionerPlan &preconditioner,
    int max_iters,
    float absolute_tolerance,
    bool verbose,
    float relative_tolerance) {
  return std::make_unique<CUCG>(program, A, &preconditioner, max_iters,
                                absolute_tolerance, verbose,
                                relative_tolerance);
}

std::unique_ptr<CUCG>
make_cuda_experimental_linear_operator_pcg_solver(
    Program *program,
    CompiledKernelLinearOperator &A,
    ExperimentalLinearOperatorHandle &preconditioner,
    int max_iters,
    float absolute_tolerance,
    bool verbose,
    float relative_tolerance) {
  return std::make_unique<CUCG>(program, A, preconditioner, max_iters,
                                absolute_tolerance, verbose,
                                relative_tolerance);
}

struct CpuSparseCGPlan::PreconditionerBinding {
  OperatorBinding binding;
  PreconditionerPlan::UpdateFn update;
  std::string method;
};

namespace {

OperatorDescriptor square_operator_descriptor(const SparseMatrix &matrix) {
  const OperatorSpaceDesc space{matrix.get_data_type(),
                                static_cast<std::size_t>(matrix.num_rows())};
  return {space, space};
}

OperatorBinding bind_cpu_operator_compatibility(Program *program,
                                                SparseMatrix &matrix) {
  if (auto *csr = dynamic_cast<CpuSparseCsrMatrix *>(&matrix)) {
    return make_cpu_csr_operator_binding(program, *csr);
  }
  if (auto *bsr = dynamic_cast<CpuSparseBsrMatrix *>(&matrix)) {
    return make_cpu_bsr_operator_binding(program, *bsr);
  }
  if (auto *kernel =
          dynamic_cast<CompiledKernelLinearOperator *>(&matrix)) {
    return make_cpu_program_kernel_operator_binding(program, *kernel);
  }
  if (auto *graph =
          dynamic_cast<CompiledGraphLinearOperator *>(&matrix)) {
    return make_cpu_program_graph_operator_binding(program, *graph);
  }
  const auto statistics = matrix.debug_runtime_statistics();
  TI_ERROR(
      "CPU operator CG compatibility factory does not support backend '{}' "
      "with storage format '{}' (provider '{}'); no fallback was performed.",
      statistics.backend_family, statistics.storage_format,
      statistics.provider_name);
}

OperatorResourceStamp preconditioner_stamp(
    Program *program,
    const SparsePreconditionerPlanRuntimeStatistics &statistics,
    const void *identity) {
  return {
      reinterpret_cast<std::uintptr_t>(program),
      program->runtime_program_generation(),
      1,
      statistics.operator_pattern_version_current,
      statistics.operator_numeric_version_current,
      reinterpret_cast<std::uintptr_t>(identity),
  };
}

OperatorBinding bind_preconditioner_action(
    Program *program,
    SparseMatrix &matrix,
    SparseJacobiPreconditionerPlan &preconditioner,
    std::string provider_name) {
  OperatorCapabilities capabilities;
  capabilities.asynchronous_submit =
      !arch_is_cpu(program->compile_config().arch);
  auto action = OperatorAction(
      square_operator_descriptor(matrix), capabilities,
      std::move(provider_name),
      [program, &preconditioner] {
        return preconditioner_stamp(program,
                                    preconditioner.debug_runtime_statistics(),
                                    &preconditioner);
      },
      [program, &preconditioner](OperatorApplyMode mode,
                                 const OperatorVectorView &input,
                                 const OperatorVectorView &output) {
        TI_ERROR_IF(mode != OperatorApplyMode::forward,
                    "Jacobi preconditioner action supports forward apply "
                    "only.");
        const auto arch = program->compile_config().arch;
        if (arch_is_cuda(arch)) {
          preconditioner.apply_cuda_raw(program, input.data, output.data);
          return;
        }
        if (arch == Arch::vulkan) {
          TI_ERROR_IF(!input.ndarray || !output.ndarray,
                      "Vulkan Jacobi action requires ndarray views.");
          preconditioner.apply(program, *input.ndarray, *output.ndarray);
          return;
        }
        TI_ERROR_IF(!arch_is_cpu(arch),
                    "Jacobi action supports CPU, CUDA, and Vulkan only.");
        preconditioner.apply_cpu_raw(program, input.data, output.data);
      });
  return OperatorBinding(
      std::move(action),
      [&preconditioner] { return preconditioner.acquire_resource_lease(); });
}

OperatorBinding bind_preconditioner_action(
    Program *program,
    SparseMatrix &matrix,
    SparseBlockJacobiPreconditionerPlan &preconditioner,
    std::string provider_name) {
  OperatorCapabilities capabilities;
  capabilities.asynchronous_submit =
      !arch_is_cpu(program->compile_config().arch);
  auto action = OperatorAction(
      square_operator_descriptor(matrix), capabilities,
      std::move(provider_name),
      [program, &preconditioner] {
        return preconditioner_stamp(program,
                                    preconditioner.debug_runtime_statistics(),
                                    &preconditioner);
      },
      [program, &preconditioner](OperatorApplyMode mode,
                                 const OperatorVectorView &input,
                                 const OperatorVectorView &output) {
        TI_ERROR_IF(mode != OperatorApplyMode::forward,
                    "Block-Jacobi preconditioner action supports forward "
                    "apply only.");
        const auto arch = program->compile_config().arch;
        if (arch_is_cuda(arch)) {
          preconditioner.apply_cuda_raw(program, input.data, output.data);
          return;
        }
        if (arch == Arch::vulkan) {
          TI_ERROR_IF(!input.ndarray || !output.ndarray,
                      "Vulkan block-Jacobi action requires ndarray views.");
          preconditioner.apply(program, *input.ndarray, *output.ndarray);
          return;
        }
        TI_ERROR_IF(
            !arch_is_cpu(arch),
            "Block-Jacobi action supports CPU, CUDA, and Vulkan only.");
        preconditioner.apply_cpu_raw(program, input.data, output.data);
      });
  return OperatorBinding(
      std::move(action),
      [&preconditioner] { return preconditioner.acquire_resource_lease(); });
}

OperatorBinding bind_preconditioner_action(
    Program *program,
    CompiledKernelLinearOperator &matrix,
    CompiledKernelPreconditionerPlan &preconditioner,
    std::string provider_name) {
  OperatorCapabilities capabilities;
  capabilities.asynchronous_submit =
      !arch_is_cpu(program->compile_config().arch);
  auto action = OperatorAction(
      square_operator_descriptor(matrix), capabilities,
      std::move(provider_name),
      [program, &preconditioner] {
        return preconditioner_stamp(program,
                                    preconditioner.debug_runtime_statistics(),
                                    &preconditioner);
      },
      [program, &matrix, &preconditioner](
          OperatorApplyMode mode, const OperatorVectorView &input,
          const OperatorVectorView &output) {
        TI_ERROR_IF(mode != OperatorApplyMode::forward || !input.ndarray ||
                        !output.ndarray,
                    "Compiled inverse action requires forward ndarray "
                    "views.");
        preconditioner.apply(program, matrix, *input.ndarray,
                             *output.ndarray);
      });
  return OperatorBinding(
      std::move(action),
      [&preconditioner] { return preconditioner.acquire_resource_lease(); });
}

template <typename SparsePreconditioner>
void update_sparse_preconditioner(
    Program *program,
    SparseMatrix &matrix,
    SparsePreconditioner &preconditioner,
    const OperatorResourceStamp &) {
  preconditioner.validate_compatible(program, matrix);
}

template <typename SparsePreconditioner>
std::unique_ptr<PreconditionerPlan> make_sparse_fixed_preconditioner_plan(
    Program *program,
    OperatorPlan &target_plan,
    SparseMatrix &matrix,
    SparsePreconditioner &preconditioner,
    std::string provider_name,
    std::string method) {
  auto plan = std::make_unique<PreconditionerPlan>(
      program, target_plan.descriptor(),
      bind_preconditioner_action(program, matrix, preconditioner,
                                 std::move(provider_name)),
      PreconditionerBehavior::fixed_linear, std::move(method),
      [program, &matrix, &preconditioner](
          const OperatorResourceStamp &target_stamp, bool) {
        update_sparse_preconditioner(program, matrix, preconditioner,
                                     target_stamp);
      });
  auto target_generation = target_plan.pin();
  plan->setup(target_generation);
  return plan;
}

std::unique_ptr<PreconditionerPlan> make_fixed_preconditioner_plan(
    Program *program,
    OperatorPlan &target_plan,
    SparseMatrix &matrix,
    SparseJacobiPreconditionerPlan &preconditioner,
    std::string provider_name,
    std::string method) {
  return make_sparse_fixed_preconditioner_plan(
      program, target_plan, matrix, preconditioner,
      std::move(provider_name), std::move(method));
}

std::unique_ptr<PreconditionerPlan> make_fixed_preconditioner_plan(
    Program *program,
    OperatorPlan &target_plan,
    SparseMatrix &matrix,
    SparseBlockJacobiPreconditionerPlan &preconditioner,
    std::string provider_name,
    std::string method) {
  return make_sparse_fixed_preconditioner_plan(
      program, target_plan, matrix, preconditioner,
      std::move(provider_name), std::move(method));
}

std::unique_ptr<PreconditionerPlan> make_fixed_preconditioner_plan(
    Program *program,
    OperatorPlan &target_plan,
    CompiledKernelLinearOperator &matrix,
    CompiledKernelPreconditionerPlan &preconditioner,
    std::string provider_name,
    std::string method) {
  auto plan = std::make_unique<PreconditionerPlan>(
      program, target_plan.descriptor(),
      bind_preconditioner_action(program, matrix, preconditioner,
                                 std::move(provider_name)),
      PreconditionerBehavior::fixed_linear, std::move(method),
      [program, &matrix, &preconditioner](
          const OperatorResourceStamp &, bool) {
        preconditioner.validate_compatible(program, matrix);
      });
  auto target_generation = target_plan.pin();
  plan->setup(target_generation);
  return plan;
}

void validate_fixed_linear_operator_preconditioner(
    Program *program,
    const OperatorDescriptor &target_descriptor,
    ExperimentalLinearOperatorHandle &preconditioner) {
  TI_ERROR_IF(!program || preconditioner.program() != program,
              "LinearOperator preconditioner must belong to the target "
              "Program generation.");
  const auto &descriptor = preconditioner.descriptor();
  TI_ERROR_IF(descriptor.domain != target_descriptor.range ||
                  descriptor.range != target_descriptor.domain,
              "LinearOperator preconditioner must map the target range "
              "back to its domain.");
  const auto &traits = preconditioner.mathematical_traits();
  TI_ERROR_IF(!traits.self_adjoint.known() ||
                  !traits.self_adjoint.value ||
                  !traits.positive_definite.known() ||
                  !traits.positive_definite.value ||
                  (traits.singular.known() && traits.singular.value),
              "A fixed-linear Krylov preconditioner must "
              "have trusted self_adjoint=True, positive_definite=True, "
              "and non-singular traits.");
}

std::unique_ptr<PreconditionerPlan> make_fixed_preconditioner_plan(
    Program *program,
    OperatorPlan &target_plan,
    ExperimentalLinearOperatorHandle &preconditioner,
    std::string method) {
  validate_fixed_linear_operator_preconditioner(
      program, target_plan.descriptor(), preconditioner);
  auto plan = std::make_unique<PreconditionerPlan>(
      program, target_plan.descriptor(), preconditioner.binding(),
      PreconditionerBehavior::fixed_linear, std::move(method),
      [program, &preconditioner](const OperatorResourceStamp &, bool) {
        TI_ERROR_IF(preconditioner.program() != program,
                    "LinearOperator preconditioner changed Program "
                    "generation.");
      });
  auto target_generation = target_plan.pin();
  plan->setup(target_generation);
  return plan;
}

void append_operator_plan_statistics(
    const OperatorPlan &plan,
    bool preconditioner,
    SparseSolvePlanRuntimeStatistics &statistics) {
  const auto plan_statistics = plan.debug_runtime_statistics();
  if (preconditioner) {
    statistics.preconditioner_action_provider = plan.provider_name();
    statistics.preconditioner_asynchronous_submit =
        plan.capabilities().asynchronous_submit;
    statistics.preconditioner_generation_pins =
        plan_statistics.generation_pins;
    statistics.preconditioner_generation_changes =
        plan_statistics.generation_changes;
    statistics.preconditioner_numeric_generation_changes =
        plan_statistics.numeric_generation_changes;
    statistics.preconditioner_binding_generation_changes =
        plan_statistics.binding_generation_changes;
    statistics.preconditioner_plan_invalidations =
        plan_statistics.invalidations;
    return;
  }
  statistics.operator_action_provider = plan.provider_name();
  statistics.operator_asynchronous_submit =
      plan.capabilities().asynchronous_submit;
  statistics.operator_execution_kind =
      operator_execution_kind_name(plan.execution_kind());
  statistics.operator_backend_execution_path =
      operator_backend_execution_path_name(
          plan_statistics.last_backend_path);
  statistics.operator_execution_plan_builds =
      plan_statistics.execution_plan_builds;
  statistics.operator_execution_plan_reuses =
      plan_statistics.execution_plan_reuses;
  statistics.operator_binding_rebinds =
      plan_statistics.binding_rebinds;
  statistics.operator_sequence_submissions =
      plan_statistics.sequence_submissions;
  statistics.operator_compiled_graph_submissions =
      plan_statistics.compiled_graph_submissions;
  statistics.operator_runtime_capture_submissions =
      plan_statistics.runtime_capture_submissions;
  statistics.operator_backend_captures =
      plan_statistics.backend_captures;
  statistics.operator_backend_replays =
      plan_statistics.backend_replays;
  statistics.operator_ordinary_fallbacks =
      plan_statistics.ordinary_fallbacks;
  statistics.operator_cache_invalidations =
      plan_statistics.cache_invalidations;
  statistics.operator_generation_pins = plan_statistics.generation_pins;
  statistics.operator_generation_changes =
      plan_statistics.generation_changes;
  statistics.operator_numeric_generation_changes =
      plan_statistics.numeric_generation_changes;
  statistics.operator_binding_generation_changes =
      plan_statistics.binding_generation_changes;
  statistics.operator_plan_invalidations = plan_statistics.invalidations;
}

void append_preconditioner_plan_statistics(
    const PreconditionerPlan &plan,
    SparseSolvePlanRuntimeStatistics &statistics) {
  append_operator_plan_statistics(plan.action(), true, statistics);
  const auto lifecycle = plan.debug_runtime_statistics();
  statistics.preconditioner_behavior =
      plan.behavior() == PreconditionerBehavior::fixed_linear
          ? "fixed_linear"
          : "unsupported";
  statistics.preconditioner_setup_calls = lifecycle.setup_calls;
  statistics.preconditioner_update_calls = lifecycle.update_calls;
  statistics.preconditioner_update_successes = lifecycle.update_successes;
  statistics.preconditioner_update_noops = lifecycle.update_noops;
  statistics.preconditioner_update_failures = lifecycle.update_failures;
}

}  // namespace

std::unique_ptr<PreconditionerPlan> make_solver_preconditioner_plan(
    Program *program,
    OperatorPlan &target_plan,
    SparseMatrix &matrix,
    SparseJacobiPreconditionerPlan &preconditioner,
    std::string method) {
  std::string provider =
      arch_name(program->compile_config().arch) + "_" + method;
  return make_fixed_preconditioner_plan(
      program, target_plan, matrix, preconditioner, std::move(provider),
      std::move(method));
}

std::unique_ptr<PreconditionerPlan> make_solver_preconditioner_plan(
    Program *program,
    OperatorPlan &target_plan,
    SparseMatrix &matrix,
    SparseBlockJacobiPreconditionerPlan &preconditioner,
    std::string method) {
  std::string provider =
      arch_name(program->compile_config().arch) + "_" + method;
  return make_fixed_preconditioner_plan(
      program, target_plan, matrix, preconditioner, std::move(provider),
      std::move(method));
}

std::unique_ptr<PreconditionerPlan> make_solver_preconditioner_plan(
    Program *program,
    OperatorPlan &target_plan,
    ExperimentalLinearOperatorHandle &preconditioner,
    std::string method) {
  return make_fixed_preconditioner_plan(program, target_plan,
                                        preconditioner, std::move(method));
}

std::unique_ptr<CpuSparseCGPlan::PreconditionerBinding>
CpuSparseCGPlan::bind_preconditioner(
    Program *program,
    SparseMatrix &matrix,
    SparseJacobiPreconditionerPlan &preconditioner) {
  return std::make_unique<PreconditionerBinding>(PreconditionerBinding{
      bind_preconditioner_action(program, matrix, preconditioner,
                                 "cpu_jacobi"),
      [program, &matrix, &preconditioner](
          const OperatorResourceStamp &target_stamp, bool) {
        update_sparse_preconditioner(program, matrix, preconditioner,
                                     target_stamp);
      },
      "jacobi"});
}

std::unique_ptr<CpuSparseCGPlan::PreconditionerBinding>
CpuSparseCGPlan::bind_preconditioner(
    Program *program,
    SparseMatrix &matrix,
    SparseBlockJacobiPreconditionerPlan &preconditioner) {
  return std::make_unique<PreconditionerBinding>(PreconditionerBinding{
      bind_preconditioner_action(program, matrix, preconditioner,
                                 "cpu_block_jacobi"),
      [program, &matrix, &preconditioner](
          const OperatorResourceStamp &target_stamp, bool) {
        update_sparse_preconditioner(program, matrix, preconditioner,
                                     target_stamp);
      },
      "block_jacobi"});
}

std::unique_ptr<CpuSparseCGPlan::PreconditionerBinding>
CpuSparseCGPlan::bind_preconditioner(
    Program *program,
    CompiledKernelLinearOperator &matrix,
    CompiledKernelPreconditionerPlan &preconditioner) {
  return std::make_unique<PreconditionerBinding>(PreconditionerBinding{
      bind_preconditioner_action(program, matrix, preconditioner,
                                 "compiled_kernel_inverse_apply"),
      [program, &matrix, &preconditioner](
          const OperatorResourceStamp &, bool) {
        preconditioner.validate_compatible(program, matrix);
      },
      "compiled_kernel_inverse_apply"});
}

std::unique_ptr<CpuSparseCGPlan::PreconditionerBinding>
CpuSparseCGPlan::bind_preconditioner(
    Program *program,
    ExperimentalLinearOperatorHandle &preconditioner) {
  TI_ERROR_IF(preconditioner.program() != program,
              "CPU LinearOperator preconditioner must belong to the "
              "construction Program.");
  const auto &traits = preconditioner.mathematical_traits();
  TI_ERROR_IF(!traits.self_adjoint.known() ||
                  !traits.self_adjoint.value ||
                  !traits.positive_definite.known() ||
                  !traits.positive_definite.value ||
                  (traits.singular.known() && traits.singular.value),
              "Fixed-linear PCG requires the preconditioner operator to "
              "have trusted self_adjoint=True, positive_definite=True, "
              "and non-singular traits.");
  return std::make_unique<PreconditionerBinding>(PreconditionerBinding{
      preconditioner.binding(),
      [program, &preconditioner](const OperatorResourceStamp &, bool) {
        TI_ERROR_IF(preconditioner.program() != program,
                    "LinearOperator preconditioner changed Program "
                    "generation.");
      },
      "linear_operator"});
}

CpuSparseCGPlan::CpuSparseCGPlan(Program *program,
                                 SparseMatrix &matrix,
                                 int max_iterations,
                                 double absolute_tolerance,
                                 double relative_tolerance)
    : CpuSparseCGPlan(program,
                      bind_cpu_operator_compatibility(program, matrix),
                      nullptr,
                      max_iterations,
                      absolute_tolerance,
                      relative_tolerance,
                      true) {
}

CpuSparseCGPlan::CpuSparseCGPlan(Program *program,
                                 SparseMatrix &matrix,
                                 SparseJacobiPreconditionerPlan &preconditioner,
                                 int max_iterations,
                                 double absolute_tolerance,
                                 double relative_tolerance)
    : CpuSparseCGPlan(program,
                      bind_cpu_operator_compatibility(program, matrix),
                      bind_preconditioner(program, matrix, preconditioner),
                      max_iterations,
                      absolute_tolerance,
                      relative_tolerance,
                      true) {
}

CpuSparseCGPlan::CpuSparseCGPlan(
    Program *program,
    SparseMatrix &matrix,
    SparseBlockJacobiPreconditionerPlan &preconditioner,
    int max_iterations,
    double absolute_tolerance,
    double relative_tolerance)
    : CpuSparseCGPlan(program,
                      bind_cpu_operator_compatibility(program, matrix),
                      bind_preconditioner(program, matrix, preconditioner),
                      max_iterations,
                      absolute_tolerance,
                      relative_tolerance,
                      true) {
}

CpuSparseCGPlan::CpuSparseCGPlan(
    Program *program,
    CompiledKernelLinearOperator &matrix,
    CompiledKernelPreconditionerPlan &preconditioner,
    int max_iterations,
    double absolute_tolerance,
    double relative_tolerance)
    : CpuSparseCGPlan(program,
                      bind_cpu_operator_compatibility(program, matrix),
                      bind_preconditioner(program, matrix, preconditioner),
                      max_iterations,
                      absolute_tolerance,
                      relative_tolerance,
                      true) {
}

CpuSparseCGPlan::CpuSparseCGPlan(
    Program *program,
    ExperimentalLinearOperatorHandle &operator_handle,
    int max_iterations,
    double absolute_tolerance,
    double relative_tolerance)
    : CpuSparseCGPlan(program,
                      operator_handle.binding(),
                      nullptr,
                      max_iterations,
                      absolute_tolerance,
                      relative_tolerance,
                      false) {
}

CpuSparseCGPlan::CpuSparseCGPlan(
    Program *program,
    ExperimentalLinearOperatorHandle &operator_handle,
    ExperimentalLinearOperatorHandle &preconditioner,
    int max_iterations,
    double absolute_tolerance,
    double relative_tolerance)
    : CpuSparseCGPlan(program,
                      operator_handle.binding(),
                      bind_preconditioner(program, preconditioner),
                      max_iterations,
                      absolute_tolerance,
                      relative_tolerance,
                      false) {
}

CpuSparseCGPlan::CpuSparseCGPlan(
    Program *program,
    OperatorBinding operator_binding,
    std::unique_ptr<PreconditionerBinding> preconditioner,
    int max_iterations,
    double absolute_tolerance,
    double relative_tolerance,
    bool assert_legacy_spd)
    : program_(program),
      preconditioner_binding_(std::move(preconditioner)),
      max_iterations_(max_iterations),
      absolute_tolerance_(absolute_tolerance),
      relative_tolerance_(relative_tolerance) {
  TI_ERROR_IF(!program_ || !arch_is_cpu(program_->compile_config().arch),
              "CPU operator CG/PCG requires an active CPU Program.");
  validate_sparse_solve_execution_policy(
      program_->compile_config().arch,
      SparseSolveExecutionPolicy::host_each_iteration);
  const auto &descriptor = operator_binding.action().descriptor();
  TI_ERROR_IF(descriptor.domain.scalar_type !=
                      descriptor.range.scalar_type ||
                  descriptor.domain.scalar_extent !=
                      descriptor.range.scalar_extent ||
                  descriptor.range.scalar_extent >
                      static_cast<std::size_t>(
                          std::numeric_limits<int>::max()),
              "CPU operator CG/PCG requires a non-empty square operator.");
  dtype_ = descriptor.range.scalar_type;
  rows_ = static_cast<int>(descriptor.range.scalar_extent);
  cols_ = static_cast<int>(descriptor.domain.scalar_extent);
  TI_ERROR_IF(dtype_ != PrimitiveType::f32 && dtype_ != PrimitiveType::f64,
              "CPU operator CG/PCG requires f32 or f64 values.");
  TI_ERROR_IF(max_iterations_ < 0,
              "CPU operator CG/PCG requires non-negative max iterations.");
  TI_ERROR_IF(!std::isfinite(absolute_tolerance_) ||
                  !std::isfinite(relative_tolerance_) ||
                  absolute_tolerance_ < 0.0 || relative_tolerance_ < 0.0 ||
                  (absolute_tolerance_ == 0.0 && relative_tolerance_ == 0.0),
              "CPU operator CG/PCG requires finite non-negative atol and rtol "
              "with at least one positive tolerance.");
  operator_plan_ = std::make_unique<OperatorPlan>(
      program_, assert_legacy_spd
                    ? with_legacy_cg_traits(std::move(operator_binding))
                    : std::move(operator_binding));
  auto initial_generation = operator_plan_->pin();
  const auto initial_stamp = initial_generation.resource_stamp();
  TI_ERROR_IF(
      initial_stamp.program_identity !=
          reinterpret_cast<std::uintptr_t>(program_),
      "CPU operator CG/PCG binding belongs to a different Program.");
  if (preconditioner_binding_) {
    preconditioner_plan_ = std::make_unique<PreconditionerPlan>(
        program_, operator_plan_->descriptor(),
        std::move(preconditioner_binding_->binding),
        PreconditionerBehavior::fixed_linear,
        preconditioner_binding_->method,
        std::move(preconditioner_binding_->update));
    preconditioner_plan_->setup(initial_generation);
  }
  validate_cg_plan(*operator_plan_, preconditioner_plan_.get());
  try {
    for (auto &vector : workspace_) {
      vector = program_->create_ndarray(dtype_, {rows_},
                                        ExternalArrayLayout::kNull, false);
    }
  } catch (...) {
    release_workspace();
    throw;
  }
}

CpuSparseCGPlan::~CpuSparseCGPlan() {
  release_workspace();
}

void CpuSparseCGPlan::release_workspace() {
  if (!program_) {
    return;
  }
  for (auto &vector : workspace_) {
    if (vector) {
      program_->delete_ndarray(vector);
      vector = nullptr;
    }
  }
}

void CpuSparseCGPlan::apply_operator(
    const OperatorPinnedAction &generation,
    const Ndarray &input,
    const Ndarray &output) {
  const auto &descriptor = operator_plan_->descriptor();
  operator_plan_->submit(
      generation,
      {OperatorApplyMode::forward,
       OperatorVectorView::from_ndarray(
           program_, input, descriptor.domain, false),
       nullptr,
       OperatorVectorView::from_ndarray(
           program_, output, descriptor.range, true)});
}

void CpuSparseCGPlan::apply_preconditioner(
    const OperatorPinnedAction &generation,
    const Ndarray &input,
    const Ndarray &output) {
  TI_ERROR_IF(!preconditioner_plan_,
              "Identity CG has no preconditioner action.");
  auto &action = preconditioner_plan_->action();
  const auto &descriptor = action.descriptor();
  action.submit(
      generation,
      {OperatorApplyMode::forward,
       OperatorVectorView::from_ndarray(
           program_, input, descriptor.domain, false),
       nullptr,
       OperatorVectorView::from_ndarray(
           program_, output, descriptor.range, true)});
}

template <typename T>
void CpuSparseCGPlan::solve_typed(T *x,
                                  const T *b,
                                  const std::array<T *, 4> &workspace,
                                  const Ndarray &solution_array,
                                  const OperatorPinnedAction
                                      &operator_generation,
                                  const OperatorPinnedAction
                                      *preconditioner_generation) {
  const int rows = rows_;
  T *ax = workspace[0];
  T *residual = workspace[1];
  T *direction = workspace[2];
  T *preconditioned_residual = workspace[3];
  auto dot = [&](const T *lhs, const T *rhs) {
    double result = 0.0;
    for (int index = 0; index < rows; ++index) {
      result +=
          static_cast<double>(lhs[index]) * static_cast<double>(rhs[index]);
    }
    host_scalar_reductions_++;
    return result;
  };

  relative_reference_norm_ = 0.0;
  effective_tolerance_ = absolute_tolerance_;
  if (relative_tolerance_ > 0.0) {
    const double rhs_squared_norm = dot(b, b);
    if (!std::isfinite(rhs_squared_norm) || rhs_squared_norm < 0.0) {
      status_ = SparseSolveStatus::kBreakdown;
      return;
    }
    relative_reference_norm_ = std::sqrt(rhs_squared_norm);
    effective_tolerance_ = std::max(
        absolute_tolerance_, relative_tolerance_ * relative_reference_norm_);
    if (!std::isfinite(effective_tolerance_)) {
      status_ = SparseSolveStatus::kBreakdown;
      return;
    }
  }

  apply_operator(operator_generation, solution_array, *workspace_[0]);
  operator_apply_calls_++;
  for (int index = 0; index < rows; ++index) {
    residual[index] = b[index] - ax[index];
  }
  double rr = dot(residual, residual);
  initial_residual_norm_ = std::isfinite(rr) && rr >= 0.0
                               ? std::sqrt(rr)
                               : std::numeric_limits<double>::quiet_NaN();
  residual_norm_ = initial_residual_norm_;
  const double tolerance_squared = effective_tolerance_ * effective_tolerance_;
  if (!std::isfinite(rr) || rr < 0.0) {
    status_ = SparseSolveStatus::kBreakdown;
    return;
  }
  if (rr <= tolerance_squared) {
    status_ = SparseSolveStatus::kConverged;
    return;
  }

  double rho = rr;
  if (preconditioner_plan_) {
    apply_preconditioner(*preconditioner_generation, *workspace_[1],
                         *workspace_[3]);
    preconditioner_apply_calls_++;
    rho = dot(residual, preconditioned_residual);
  }
  if (!std::isfinite(rho) || rho <= 0.0) {
    status_ = SparseSolveStatus::kBreakdown;
    return;
  }
  const T *initial_direction =
      preconditioner_plan_ ? preconditioned_residual : residual;
  std::copy(initial_direction, initial_direction + rows, direction);

  bool breakdown = false;
  while (iterations_ < max_iterations_ && rr > tolerance_squared) {
    apply_operator(operator_generation, *workspace_[2], *workspace_[0]);
    operator_apply_calls_++;
    const double p_ap = dot(direction, ax);
    if (!std::isfinite(p_ap) || p_ap <= 0.0) {
      breakdown = true;
      break;
    }
    const double alpha = rho / p_ap;
    if (!std::isfinite(alpha)) {
      breakdown = true;
      break;
    }
    for (int index = 0; index < rows; ++index) {
      x[index] = static_cast<T>(static_cast<double>(x[index]) +
                                alpha * static_cast<double>(direction[index]));
      residual[index] = static_cast<T>(static_cast<double>(residual[index]) -
                                       alpha * static_cast<double>(ax[index]));
    }
    rr = dot(residual, residual);
    iterations_++;
    if (!std::isfinite(rr) || rr < 0.0) {
      breakdown = true;
      break;
    }
    if (rr <= tolerance_squared || iterations_ >= max_iterations_) {
      break;
    }

    double next_rho = rr;
    if (preconditioner_plan_) {
      apply_preconditioner(*preconditioner_generation, *workspace_[1],
                           *workspace_[3]);
      preconditioner_apply_calls_++;
      next_rho = dot(residual, preconditioned_residual);
    }
    if (!std::isfinite(next_rho) || next_rho <= 0.0) {
      breakdown = true;
      break;
    }
    const double beta = next_rho / rho;
    if (!std::isfinite(beta)) {
      breakdown = true;
      break;
    }
    const T *updated_direction =
        preconditioner_plan_ ? preconditioned_residual : residual;
    for (int index = 0; index < rows; ++index) {
      direction[index] =
          static_cast<T>(static_cast<double>(updated_direction[index]) +
                         beta * static_cast<double>(direction[index]));
    }
    rho = next_rho;
  }

  residual_norm_ = std::isfinite(rr) && rr >= 0.0
                       ? std::sqrt(rr)
                       : std::numeric_limits<double>::quiet_NaN();
  if (breakdown || !std::isfinite(residual_norm_)) {
    status_ = SparseSolveStatus::kBreakdown;
  } else if (residual_norm_ <= effective_tolerance_) {
    status_ = SparseSolveStatus::kConverged;
  } else {
    status_ = SparseSolveStatus::kMaxIterations;
  }
}

void CpuSparseCGPlan::solve(Program *program,
                            const Ndarray &x,
                            const Ndarray &b) {
  TI_ERROR_IF(program != program_,
              "CPU operator CG/PCG must be solved by its construction "
              "Program.");
  const int rows = rows_;
  auto validate_vector = [&](const char *role, const Ndarray &array) {
    TI_ERROR_IF(array.get_element_data_type() != dtype_ ||
                    !array.get_element_shape().empty() ||
                    array.shape.size() != 1 ||
                    array.get_nelement() != static_cast<std::size_t>(rows) ||
                    array.get_element_size() != data_type_size(dtype_),
                "CPU operator CG/PCG {} must contain exactly {} scalar {} "
                "entries.",
                role, rows, data_type_name(dtype_));
  };
  validate_vector("solution", x);
  validate_vector("right-hand side", b);
  TI_ERROR_IF(x.owning_program() != program_ || b.owning_program() != program_,
              "CPU operator CG/PCG requires solution and RHS owned by "
              "its construction Program.");
  const auto solution = program_->get_ndarray_data_ptr_as_int(&x);
  const auto rhs = program_->get_ndarray_data_ptr_as_int(&b);
  TI_ERROR_IF(solution == rhs,
              "CPU operator CG/PCG solution and RHS must not alias.");

  std::lock_guard<std::mutex> lock(solve_mutex_);
  auto operator_generation = operator_plan_->pin();
  auto preconditioner_generation =
      preconditioner_plan_
          ? preconditioner_plan_->update_and_pin(operator_generation)
          : OperatorPinnedAction{};
  const auto operator_stamp = operator_generation.resource_stamp();
  if (has_solved_) {
    workspace_reuses_++;
  } else {
    has_solved_ = true;
  }
  solve_calls_++;
  last_solve_pattern_version_ = operator_stamp.topology_revision;
  last_solve_numeric_version_ = operator_stamp.numeric_revision;
  status_ = SparseSolveStatus::kNotRun;
  iterations_ = 0;
  initial_residual_norm_ = 0.0;
  residual_norm_ = 0.0;
  relative_reference_norm_ = 0.0;
  effective_tolerance_ = absolute_tolerance_;
  if (dtype_ == PrimitiveType::f32) {
    std::array<float32 *, 4> workspace{};
    for (int index = 0; index < 4; ++index) {
      workspace[index] = reinterpret_cast<float32 *>(
          program_->get_ndarray_data_ptr_as_int(workspace_[index]));
    }
    solve_typed(reinterpret_cast<float32 *>(solution),
                reinterpret_cast<const float32 *>(rhs), workspace, x,
                operator_generation,
                preconditioner_plan_ ? &preconditioner_generation : nullptr);
  } else {
    std::array<float64 *, 4> workspace{};
    for (int index = 0; index < 4; ++index) {
      workspace[index] = reinterpret_cast<float64 *>(
          program_->get_ndarray_data_ptr_as_int(workspace_[index]));
    }
    solve_typed(reinterpret_cast<float64 *>(solution),
                reinterpret_cast<const float64 *>(rhs), workspace, x,
                operator_generation,
                preconditioner_plan_ ? &preconditioner_generation : nullptr);
  }
  total_iterations_ += static_cast<std::uint64_t>(iterations_);
}

SparseSolvePlanRuntimeStatistics CpuSparseCGPlan::debug_runtime_statistics()
    const {
  std::lock_guard<std::mutex> lock(solve_mutex_);
  const auto operator_stamp = operator_plan_->resource_stamp();
  SparseSolvePlanRuntimeStatistics result;
  result.backend_family = "cpu";
  if (!preconditioner_binding_) {
    result.method = "cg_operator_action";
  } else if (preconditioner_binding_->method ==
             "compiled_kernel_inverse_apply") {
    result.method = "pcg_compiled_kernel";
  } else {
    result.method = "pcg_" + preconditioner_binding_->method;
  }
  result.dtype = data_type_name(dtype_);
  result.rows = rows_;
  result.cols = cols_;
  result.max_iterations = max_iterations_;
  result.absolute_tolerance = absolute_tolerance_;
  result.relative_tolerance = relative_tolerance_;
  result.last_relative_reference_norm = relative_reference_norm_;
  result.last_effective_tolerance = effective_tolerance_;
  result.operator_pattern_version = operator_stamp.topology_revision;
  result.operator_numeric_version = operator_stamp.numeric_revision;
  result.last_solve_pattern_version = last_solve_pattern_version_;
  result.last_solve_numeric_version = last_solve_numeric_version_;
  result.operator_pattern_changed_since_last_solve =
      solve_calls_ > 0 &&
      result.operator_pattern_version != result.last_solve_pattern_version;
  result.operator_numeric_changed_since_last_solve =
      solve_calls_ > 0 &&
      result.operator_numeric_version != result.last_solve_numeric_version;
  result.solve_calls = solve_calls_;
  result.total_iterations = total_iterations_;
  result.logical_iterations = total_iterations_;
  result.executed_iterations = total_iterations_;
  result.wasted_iterations = 0;
  result.workspace_builds = workspace_builds_;
  result.workspace_reuses = workspace_reuses_;
  result.operator_apply_calls = operator_apply_calls_;
  result.operator_apply_calls_available = true;
  result.preconditioner_method =
      preconditioner_binding_ ? preconditioner_binding_->method : "identity";
  result.preconditioner_apply_calls = preconditioner_apply_calls_;
  result.host_scalar_reductions = host_scalar_reductions_;
  result.fixed_iteration_only = false;
  result.bounded_masked_execution = false;
  result.persistent_vector_count = 4;
  result.persistent_vector_reserved_bytes =
      4 * static_cast<std::uint64_t>(rows_) *
      data_type_size(dtype_);
  result.external_preconditioner = preconditioner_binding_ != nullptr;
  result.preconditioner_ownership_scope =
      preconditioner_binding_ ? "external_plan" : "none";
  result.solver_state_rebuilt_each_solve = false;
  append_operator_plan_statistics(*operator_plan_, false, result);
  if (preconditioner_plan_) {
    append_preconditioner_plan_statistics(*preconditioner_plan_, result);
  }
  return result;
}

std::unique_ptr<CpuSparseCGPlan> make_cpu_operator_cg_solver(
    Program *program,
    SparseMatrix &matrix,
    int max_iterations,
    double absolute_tolerance,
    double relative_tolerance) {
  return std::make_unique<CpuSparseCGPlan>(
      program, matrix, max_iterations, absolute_tolerance, relative_tolerance);
}

std::unique_ptr<CpuSparseCGPlan>
make_cpu_experimental_linear_operator_cg_solver(
    Program *program,
    ExperimentalLinearOperatorHandle &operator_handle,
    int max_iterations,
    double absolute_tolerance,
    double relative_tolerance) {
  return std::make_unique<CpuSparseCGPlan>(
      program, operator_handle, max_iterations, absolute_tolerance,
      relative_tolerance);
}

std::unique_ptr<CpuSparseCGPlan>
make_cpu_experimental_linear_operator_pcg_solver(
    Program *program,
    ExperimentalLinearOperatorHandle &operator_handle,
    ExperimentalLinearOperatorHandle &preconditioner,
    int max_iterations,
    double absolute_tolerance,
    double relative_tolerance) {
  return std::make_unique<CpuSparseCGPlan>(
      program, operator_handle, preconditioner, max_iterations,
      absolute_tolerance, relative_tolerance);
}

std::unique_ptr<CpuSparseCGPlan> make_cpu_jacobi_pcg_solver(
    Program *program,
    SparseMatrix &matrix,
    SparseJacobiPreconditionerPlan &preconditioner,
    int max_iterations,
    double absolute_tolerance,
    double relative_tolerance) {
  return std::make_unique<CpuSparseCGPlan>(
      program, matrix, preconditioner, max_iterations, absolute_tolerance,
      relative_tolerance);
}

std::unique_ptr<CpuSparseCGPlan> make_cpu_block_jacobi_pcg_solver(
    Program *program,
    SparseMatrix &matrix,
    SparseBlockJacobiPreconditionerPlan &preconditioner,
    int max_iterations,
    double absolute_tolerance,
    double relative_tolerance) {
  return std::make_unique<CpuSparseCGPlan>(
      program, matrix, preconditioner, max_iterations,
      absolute_tolerance, relative_tolerance);
}

std::unique_ptr<CpuSparseCGPlan> make_cpu_compiled_kernel_pcg_solver(
    Program *program,
    CompiledKernelLinearOperator &matrix,
    CompiledKernelPreconditionerPlan &preconditioner,
    int max_iterations,
    double absolute_tolerance,
    double relative_tolerance) {
  return std::make_unique<CpuSparseCGPlan>(
      program, matrix, preconditioner, max_iterations,
      absolute_tolerance, relative_tolerance);
}

VulkanCGIterationPlan::VulkanCGIterationPlan(Program *program,
    SparseMatrix &matrix,
    int fixed_iterations)
    : VulkanCGIterationPlan(program, matrix, fixed_iterations, 0.0f, 0.0f,
                            false, false, false, nullptr, nullptr, nullptr,
                            nullptr) {
}

VulkanCGIterationPlan::VulkanCGIterationPlan(Program *program,
                                             SparseMatrix &matrix,
                                             int max_iterations,
                                             float absolute_tolerance,
                                             float relative_tolerance)
    : VulkanCGIterationPlan(program, matrix, max_iterations,
                            absolute_tolerance, relative_tolerance, true,
                            false, false, nullptr, nullptr, nullptr, nullptr) {
}

VulkanCGIterationPlan::VulkanCGIterationPlan(
    Program *program,
    SparseMatrix &matrix,
    SparseJacobiPreconditionerPlan &preconditioner,
    int max_iterations,
    float absolute_tolerance,
    float relative_tolerance)
    : VulkanCGIterationPlan(program, matrix, max_iterations,
                            absolute_tolerance, relative_tolerance, true,
                            false, false,
                            &preconditioner, nullptr, nullptr, nullptr) {
}

VulkanCGIterationPlan::VulkanCGIterationPlan(
    Program *program,
    SparseMatrix &matrix,
    SparseBlockJacobiPreconditionerPlan &preconditioner,
    int max_iterations,
    float absolute_tolerance,
    float relative_tolerance)
    : VulkanCGIterationPlan(program, matrix, max_iterations,
                            absolute_tolerance, relative_tolerance, true,
                            false, false, nullptr, &preconditioner, nullptr,
                            nullptr) {
}

VulkanCGIterationPlan::VulkanCGIterationPlan(
    Program *program,
    CompiledKernelLinearOperator &matrix,
    int max_iterations,
    float absolute_tolerance,
    float relative_tolerance)
    : VulkanCGIterationPlan(program, matrix, max_iterations,
                            absolute_tolerance, relative_tolerance, true,
                            true, false, nullptr, nullptr, nullptr, nullptr) {
}

VulkanCGIterationPlan::VulkanCGIterationPlan(
    Program *program,
    CompiledGraphLinearOperator &matrix,
    int max_iterations,
    float absolute_tolerance,
    float relative_tolerance)
    : VulkanCGIterationPlan(program, matrix, max_iterations,
                            absolute_tolerance, relative_tolerance, true,
                            false, true, nullptr, nullptr, nullptr, nullptr) {
}

VulkanCGIterationPlan::VulkanCGIterationPlan(
    Program *program,
    CompiledKernelLinearOperator &matrix,
    CompiledKernelPreconditionerPlan &preconditioner,
    int max_iterations,
    float absolute_tolerance,
    float relative_tolerance)
    : VulkanCGIterationPlan(program, matrix, max_iterations,
                            absolute_tolerance, relative_tolerance, true,
                            true, false, nullptr, nullptr, &preconditioner,
                            nullptr) {
}

VulkanCGIterationPlan::VulkanCGIterationPlan(
    Program *program,
    CompiledKernelLinearOperator &matrix,
    ExperimentalLinearOperatorHandle &preconditioner,
    int max_iterations,
    float absolute_tolerance,
    float relative_tolerance)
    : VulkanCGIterationPlan(program, matrix, max_iterations,
                            absolute_tolerance, relative_tolerance, true,
                            true, false, nullptr, nullptr, nullptr,
                            &preconditioner) {
}

VulkanCGIterationPlan::VulkanCGIterationPlan(Program *program,
                                             SparseMatrix &matrix,
                                             int max_iterations,
                                             float absolute_tolerance,
                                             float relative_tolerance,
                                             bool adaptive,
                                             bool allow_compiled_kernel_operator,
                                             bool allow_compiled_graph_operator,
                                             SparseJacobiPreconditionerPlan
                                                 *preconditioner,
                                             SparseBlockJacobiPreconditionerPlan
                                                 *block_preconditioner,
                                             CompiledKernelPreconditionerPlan
                                                 *compiled_kernel_preconditioner,
                                             ExperimentalLinearOperatorHandle
                                                 *operator_preconditioner) {
#if defined(TI_WITH_VULKAN)
  TI_ERROR_IF(!program || program->compile_config().arch != Arch::vulkan,
              "Vulkan CG iteration plans require an active Vulkan Program.");
  auto *vulkan_csr = dynamic_cast<VulkanSparseMatrix *>(&matrix);
  auto *vulkan_bsr = dynamic_cast<VulkanSparseBsrMatrix *>(&matrix);
  auto *compiled_kernel_operator =
      dynamic_cast<CompiledKernelLinearOperator *>(&matrix);
  auto *compiled_graph_operator =
      dynamic_cast<CompiledGraphLinearOperator *>(&matrix);
  const int preconditioner_count = (preconditioner ? 1 : 0) +
                                   (block_preconditioner ? 1 : 0) +
                                   (compiled_kernel_preconditioner ? 1 : 0) +
                                   (operator_preconditioner ? 1 : 0);
  TI_ERROR_IF(preconditioner_count > 1,
              "Vulkan CG iteration plans accept at most one "
              "preconditioner.");
  TI_ERROR_IF(allow_compiled_kernel_operator &&
                  allow_compiled_graph_operator,
              "Vulkan CG cannot select two program-bound operator kinds.");
  if (allow_compiled_kernel_operator) {
    TI_ERROR_IF(!compiled_kernel_operator || preconditioner ||
                    block_preconditioner ||
                    compiled_kernel_operator->owning_program() != program,
                "Private Vulkan compiled-kernel CG requires its owning "
                "compiled-kernel operator and either identity or a "
                "compiled-kernel preconditioner.");
  } else if (allow_compiled_graph_operator) {
    TI_ERROR_IF(!compiled_graph_operator || preconditioner ||
                    block_preconditioner ||
                    compiled_kernel_preconditioner ||
                    compiled_graph_operator->owning_program() != program,
                "Private Vulkan compiled-graph CG requires its owning "
                "compiled-graph operator and identity preconditioning.");
  } else if (block_preconditioner) {
    TI_ERROR_IF(!vulkan_bsr,
                "Vulkan block-Jacobi PCG requires an internal Vulkan BSR "
                "matrix.");
  } else {
    TI_ERROR_IF(!vulkan_csr,
                "Vulkan CG iteration plans require an internal Vulkan CSR "
                "matrix.");
  }
  TI_ERROR_IF(matrix.num_rows() <= 0 ||
                  matrix.num_rows() != matrix.num_cols(),
              "Vulkan CG iteration plans require a non-empty square matrix.");
  TI_ERROR_IF(matrix.get_data_type() != PrimitiveType::f32,
              "Vulkan CG iteration plans currently require f32 values.");
  TI_ERROR_IF(max_iterations <= 0,
              "Vulkan CG iteration plans require positive max iterations.");
  TI_ERROR_IF(adaptive &&
                  (!std::isfinite(absolute_tolerance) ||
                   !std::isfinite(relative_tolerance) ||
                   absolute_tolerance < 0.0f ||
                   relative_tolerance < 0.0f ||
                   (absolute_tolerance == 0.0f &&
                    relative_tolerance == 0.0f)),
              "Adaptive Vulkan CG plans require finite non-negative atol "
              "and rtol with at least one positive tolerance.");
  program_ = program;
  matrix_ = &matrix;
  csr_matrix_ = vulkan_csr;
  bsr_matrix_ = vulkan_bsr;
  preconditioner_ = preconditioner;
  block_preconditioner_ = block_preconditioner;
  compiled_kernel_preconditioner_ = compiled_kernel_preconditioner;
  operator_preconditioner_ = operator_preconditioner;
  compiled_kernel_operator_ =
      allow_compiled_kernel_operator ? compiled_kernel_operator : nullptr;
  compiled_graph_operator_ =
      allow_compiled_graph_operator ? compiled_graph_operator : nullptr;
  fixed_iterations_ = max_iterations;
  absolute_tolerance_ = absolute_tolerance;
  relative_tolerance_ = relative_tolerance;
  host_check_interval_ = max_iterations;
  adaptive_ = adaptive;
  if (compiled_graph_operator_) {
    operator_plan_ = std::make_unique<OperatorPlan>(
        program_, with_legacy_cg_traits(
                      make_vulkan_program_graph_operator_binding(
                          program_, *compiled_graph_operator_)));
  } else if (compiled_kernel_operator_) {
    operator_plan_ = std::make_unique<OperatorPlan>(
        program_, with_legacy_cg_traits(
                      make_vulkan_program_kernel_operator_binding(
                          program_, *compiled_kernel_operator_)));
  } else if (bsr_matrix_) {
    operator_plan_ = std::make_unique<OperatorPlan>(
        program_, with_legacy_cg_traits(
                      make_vulkan_bsr_operator_binding(program_,
                                                       *bsr_matrix_)));
  } else {
    operator_plan_ = std::make_unique<OperatorPlan>(
        program_, with_legacy_cg_traits(
                      make_vulkan_csr_operator_binding(program_,
                                                       *csr_matrix_)));
  }
  if (preconditioner_) {
    preconditioner_plan_ = make_fixed_preconditioner_plan(
        program_, *operator_plan_, matrix, *preconditioner_,
        "vulkan_jacobi", "jacobi");
  } else if (block_preconditioner_) {
    preconditioner_plan_ = make_fixed_preconditioner_plan(
        program_, *operator_plan_, matrix, *block_preconditioner_,
        "vulkan_block_jacobi", "block_jacobi");
  } else if (compiled_kernel_preconditioner_) {
    preconditioner_plan_ = make_fixed_preconditioner_plan(
        program_, *operator_plan_, *compiled_kernel_operator_,
        *compiled_kernel_preconditioner_,
        "vulkan_compiled_inverse_apply", "compiled_kernel_inverse_apply");
  } else if (operator_preconditioner_) {
    preconditioner_plan_ = make_fixed_preconditioner_plan(
        program_, *operator_plan_, *operator_preconditioner_,
        "linear_operator");
  }
  validate_cg_plan(*operator_plan_, preconditioner_plan_.get());
  const int n = matrix.num_rows();
  auto create_vector = [&]() {
    return program->create_ndarray(PrimitiveType::f32, {n},
                                   ExternalArrayLayout::kNull, false);
  };
  auto create_f32_scalar = [&]() {
    return program->create_ndarray(PrimitiveType::f32, {1},
                                   ExternalArrayLayout::kNull, false);
  };
  auto create_i32_scalar = [&]() {
    return program->create_ndarray(PrimitiveType::i32, {1},
                                   ExternalArrayLayout::kNull, false);
  };
  try {
    ap_ = create_vector();
    residual_ = create_vector();
    direction_ = create_vector();
    if (has_preconditioner()) {
      preconditioned_residual_ = create_vector();
    }
    initial_rr_ = create_f32_scalar();
    if (adaptive_) {
      rhs_squared_ = create_f32_scalar();
    }
    rr_a_ = create_f32_scalar();
    rr_b_ = create_f32_scalar();
    p_ap_ = create_f32_scalar();
    alpha_ = create_f32_scalar();
    beta_ = create_f32_scalar();
    residual_norm_scalar_ = create_f32_scalar();
    status_scalar_ = create_i32_scalar();
    zero_status_scalar_ = create_i32_scalar();
    if (adaptive_) {
      completed_iterations_scalar_ = create_i32_scalar();
    }
    const int32_t zero = 0;
    program->copy_ndarray_from_host(zero_status_scalar_, &zero,
                                    sizeof(zero));
  } catch (...) {
    release_workspace();
    throw;
  }
#else
  TI_NOT_IMPLEMENTED;
#endif
}

bool VulkanCGIterationPlan::has_preconditioner() const {
  return preconditioner_plan_ != nullptr;
}

void VulkanCGIterationPlan::configure_execution_policy(
    SparseSolveExecutionPolicy policy,
    int host_check_interval) {
  std::lock_guard<std::mutex> lock(solve_mutex_);
  TI_ERROR_IF(solve_calls_ != 0,
              "Vulkan CG execution policy must be configured before solve.");
  validate_sparse_solve_execution_policy(Arch::vulkan, policy,
                                         host_check_interval);
  TI_ERROR_IF(policy == SparseSolveExecutionPolicy::host_check_every_k &&
                  host_check_interval != 4 && host_check_interval != 8,
              "Vulkan host_check_every_k currently supports K=4 or K=8.");
  execution_policy_ = policy;
  host_check_interval_ =
      policy == SparseSolveExecutionPolicy::host_check_every_k
          ? host_check_interval
          : fixed_iterations_;
}

void VulkanCGIterationPlan::apply_preconditioner(
    Program *program,
    const OperatorPinnedAction &generation,
    const Ndarray &input,
    const Ndarray &output) {
  TI_ERROR_IF(!preconditioner_plan_,
              "Vulkan CG preconditioner plan is not initialized.");
  auto &action = preconditioner_plan_->action();
  const auto &descriptor = action.descriptor();
  action.submit(
      generation,
      {OperatorApplyMode::forward,
       OperatorVectorView::from_ndarray(
           program, input, descriptor.domain, false),
       nullptr,
       OperatorVectorView::from_ndarray(
           program, output, descriptor.range, true)});
}

void VulkanCGIterationPlan::apply_operator(Program *program,
                                           const OperatorPinnedAction
                                               &generation,
                                           const Ndarray &input,
                                           const Ndarray &output) {
  TI_ERROR_IF(!operator_plan_,
              "Vulkan CG operator plan is not initialized.");
  const auto &descriptor = operator_plan_->descriptor();
  operator_plan_->submit(
      generation,
      {OperatorApplyMode::forward,
       OperatorVectorView::from_ndarray(
           program, input, descriptor.domain, false),
       nullptr,
       OperatorVectorView::from_ndarray(
           program, output, descriptor.range, true)});
}

VulkanCGIterationPlan::~VulkanCGIterationPlan() {
  release_workspace();
}

void VulkanCGIterationPlan::solve(Program *program,
                                  const Ndarray &x,
                                  const Ndarray &b) {
#if defined(TI_WITH_VULKAN)
  TI_ERROR_IF(program != program_,
              "Vulkan CG iteration plan requires its owning Program.");
  const int n = matrix_->num_rows();
  auto check_vector = [&](const char *role, const Ndarray &array) {
    TI_ERROR_IF(array.shape.size() != 1 ||
                    array.get_element_data_type() != PrimitiveType::f32 ||
                    !array.get_element_shape().empty() ||
                    array.get_element_size() != sizeof(float32) ||
                    array.get_nelement() != static_cast<std::size_t>(n),
                "Vulkan CG iteration plan {} must contain exactly {} "
                "scalar f32 entries.",
                role, n);
  };
  check_vector("solution", x);
  check_vector("right-hand side", b);
  TI_ERROR_IF(x.get_device_allocation() == b.get_device_allocation(),
              "Vulkan CG iteration plan solution and RHS must not alias.");

  std::lock_guard<std::mutex> lock(solve_mutex_);
  auto operator_generation = operator_plan_->pin();
  auto preconditioner_generation =
      preconditioner_plan_
          ? preconditioner_plan_->update_and_pin(operator_generation)
          : OperatorPinnedAction{};
  auto submission_guard =
      program->acquire_runtime_resource_submission_guard();
  const Ndarray *resources[] = {
      &x,          &b,          ap_,          residual_,
      direction_,  preconditioned_residual_,  initial_rr_, rr_a_, rr_b_,
      p_ap_,       alpha_,      beta_,        residual_norm_scalar_,
      status_scalar_,           zero_status_scalar_,
      completed_iterations_scalar_, rhs_squared_};
  program->retain_ndarrays_for_external_submission(
      resources, std::size(resources));
  const auto operator_stamp = operator_generation.resource_stamp();
  if (has_solved_) {
    workspace_reuses_++;
  } else {
    has_solved_ = true;
  }
  solve_calls_++;
  last_solve_pattern_version_ = operator_stamp.topology_revision;
  last_solve_numeric_version_ = operator_stamp.numeric_revision;
  is_success_ = false;
  iterations_ = 0;
  status_ = static_cast<int>(SparseSolveStatus::kMaxIterations);
  initial_residual_norm_ = 0.0;
  residual_norm_ = 0.0;
  relative_reference_norm_ = 0.0;
  effective_tolerance_ = static_cast<double>(absolute_tolerance_);

  auto *mutable_x = const_cast<Ndarray *>(&x);
  auto *mutable_b = const_cast<Ndarray *>(&b);
  program->copy_ndarray_fast(status_scalar_, zero_status_scalar_);
  if (adaptive_) {
    program->copy_ndarray_fast(completed_iterations_scalar_,
                               zero_status_scalar_);
  }
  program->copy_ndarray_fast(residual_, mutable_b);
  apply_operator(program, operator_generation, x, *ap_);
  program->vulkan_sparse_axpy(ap_, residual_, n, -1.0f);
  program->vulkan_sparse_dot(residual_, residual_, initial_rr_, n);
  bool initial_terminal = false;
  std::uint64_t solve_host_readbacks = 0;
  std::uint64_t solve_host_synchronizations = 0;
  std::uint64_t solve_device_to_host_bytes = 0;
  const bool host_check_every_k =
      adaptive_ &&
      execution_policy_ == SparseSolveExecutionPolicy::host_check_every_k;
  if (adaptive_) {
    program->vulkan_sparse_dot(mutable_b, mutable_b, rhs_squared_, n);
    program->vulkan_sparse_convergence(
        initial_rr_, status_scalar_, completed_iterations_scalar_,
        rhs_squared_, absolute_tolerance_, relative_tolerance_, 0);
    if (host_check_every_k) {
      int32_t initial_status_host = 0;
      int32_t initial_completed_host = 0;
      program->synchronize();
      program->copy_ndarray_to_host(status_scalar_, &initial_status_host,
                                    sizeof(initial_status_host));
      program->copy_ndarray_to_host(completed_iterations_scalar_,
                                    &initial_completed_host,
                                    sizeof(initial_completed_host));
      solve_host_readbacks += 2;
      solve_host_synchronizations += 1;
      solve_device_to_host_bytes += 2 * sizeof(int32_t);
      initial_terminal = initial_status_host != 0;
    }
  }
  std::uint64_t preconditioner_applies_this_solve = 0;
  if (!initial_terminal) {
    if (has_preconditioner()) {
      apply_preconditioner(program, preconditioner_generation, *residual_,
                           *preconditioned_residual_);
      preconditioner_applies_this_solve++;
      program->vulkan_sparse_dot(residual_, preconditioned_residual_, rr_a_,
                                 n);
      program->copy_ndarray_fast(direction_, preconditioned_residual_);
    } else {
      program->copy_ndarray_fast(rr_a_, initial_rr_);
      program->copy_ndarray_fast(direction_, residual_);
    }
  }

  Ndarray *current_rr = rr_a_;
  Ndarray *next_rr = rr_b_;
  int executed_this_solve = 0;
  bool terminal = initial_terminal;
  const bool native_chunk_provider =
      (csr_matrix_ != nullptr || bsr_matrix_ != nullptr) &&
      (!has_preconditioner() || preconditioner_ != nullptr ||
       block_preconditioner_ != nullptr);
  const bool native_chunk_requested =
      get_environ_config("TI_VULKAN_SOLVER_CHUNK_REPLAY", 1) != 0 &&
      native_chunk_provider && program->profiler == nullptr;
  if (native_chunk_requested && !solver_chunk_replay_state_) {
    solver_chunk_replay_state_ =
        std::make_unique<VulkanSolverChunkReplayState>();
  }
  auto issue_iteration = [&](int iteration, Ndarray *iteration_rr,
                             Ndarray *iteration_next_rr) {
    apply_operator(program, operator_generation, *direction_, *ap_);
    program->vulkan_sparse_dot(direction_, ap_, p_ap_, n);
    program->vulkan_sparse_scalar_divide(iteration_rr, p_ap_, alpha_,
                                         status_scalar_);
    program->vulkan_sparse_cg_update(direction_, ap_, alpha_, mutable_x,
                                     residual_, n);
    if (has_preconditioner()) {
      program->vulkan_sparse_dot(residual_, residual_,
                                 residual_norm_scalar_, n);
      if (adaptive_) {
        program->vulkan_sparse_convergence(
            residual_norm_scalar_, status_scalar_,
            completed_iterations_scalar_, rhs_squared_, absolute_tolerance_,
            relative_tolerance_, static_cast<std::uint32_t>(iteration + 1));
      }
      apply_preconditioner(program, preconditioner_generation, *residual_,
                           *preconditioned_residual_);
      program->vulkan_sparse_dot(residual_, preconditioned_residual_,
                                 iteration_next_rr, n);
    } else {
      program->vulkan_sparse_dot(residual_, residual_, iteration_next_rr, n);
      if (adaptive_) {
        program->vulkan_sparse_convergence(
            iteration_next_rr, status_scalar_, completed_iterations_scalar_,
            rhs_squared_, absolute_tolerance_, relative_tolerance_,
            static_cast<std::uint32_t>(iteration + 1));
      }
    }
    if (iteration + 1 < fixed_iterations_) {
      program->vulkan_sparse_scalar_divide(iteration_next_rr, iteration_rr,
                                           beta_, status_scalar_);
      program->vulkan_sparse_cg_direction(
          has_preconditioner() ? preconditioned_residual_ : residual_, beta_,
          direction_, n);
    }
  };
  std::size_t chunk_slot_index = 0;
  while (!terminal && executed_this_solve < fixed_iterations_) {
    const int chunk_iterations =
        host_check_every_k
            ? std::min(host_check_interval_,
                       fixed_iterations_ - executed_this_solve)
            : (native_chunk_requested
                   ? std::min(8, fixed_iterations_ - executed_this_solve)
                   : fixed_iterations_ - executed_this_solve);
    const int chunk_start_iteration = executed_this_solve;
    bool submitted = false;
    if (native_chunk_requested) {
      auto &slot =
          solver_chunk_replay_state_->slot(chunk_slot_index);
      VulkanCommandReplayKey key;
      key.push(200);
      key.push(program->runtime_program_generation());
      key.push(program->vulkan_sparse_algebra_replay_generation());
      key.push(static_cast<std::uint64_t>(chunk_start_iteration));
      key.push(static_cast<std::uint64_t>(chunk_iterations));
      key.push(has_preconditioner() ? 1 : 0);
      key.push(adaptive_ ? 1 : 0);
      push_solver_chunk_stamp(key, operator_stamp);
      push_solver_chunk_stamp(
          key, preconditioner_generation
                   ? preconditioner_generation.resource_stamp()
                   : OperatorResourceStamp{});
      const Ndarray *chunk_resources[] = {
          &x,
          ap_,
          residual_,
          direction_,
          preconditioned_residual_,
          initial_rr_,
          rhs_squared_,
          rr_a_,
          rr_b_,
          p_ap_,
          alpha_,
          beta_,
          residual_norm_scalar_,
          status_scalar_,
          completed_iterations_scalar_};
      for (const auto *resource : chunk_resources) {
        push_solver_chunk_resource(key, resource);
      }

      const auto solution_handle = x.runtime_resource_handle();
      if (slot.solution_handle != solution_handle) {
        if (slot.solution_handle) {
          slot.cache.reset();
          slot.solution_lease.reset();
          ++solver_chunk_invalidations_;
          ++solver_chunk_rebinds_;
        }
        slot.solution_handle = solution_handle;
        slot.solution_lease.emplace(
            program->acquire_ndarray_external_lease(solution_handle));
      }
      const auto preconditioner_stamp =
          preconditioner_generation
              ? preconditioner_generation.resource_stamp()
              : OperatorResourceStamp{};
      if (slot.operator_numeric_revision != 0 &&
          (slot.operator_numeric_revision != operator_stamp.numeric_revision ||
           slot.preconditioner_numeric_revision !=
               preconditioner_stamp.numeric_revision)) {
        ++solver_chunk_rebinds_;
      }
      slot.operator_numeric_revision = operator_stamp.numeric_revision;
      slot.preconditioner_numeric_revision =
          preconditioner_stamp.numeric_revision;
      const bool replaces_recording =
          slot.cache.entry.cmdlist != nullptr &&
          slot.cache.entry.key != key;

      (void)program->flush_if_pending();
      auto record_chunk = [&](Device *device, CommandList *cmdlist) {
        VulkanNativeCommandRecordingScope recording(program, device, cmdlist);
        Ndarray *record_current_rr = current_rr;
        Ndarray *record_next_rr = next_rr;
        for (int local_iteration = 0; local_iteration < chunk_iterations;
             ++local_iteration) {
          const int iteration = chunk_start_iteration + local_iteration;
          issue_iteration(iteration, record_current_rr, record_next_rr);
          if (iteration + 1 < fixed_iterations_) {
            std::swap(record_current_rr, record_next_rr);
          }
        }
      };
      submitted = slot.cache.submit_or_record(
          program, program->get_compute_device(), key,
          program->profiler != nullptr, record_chunk);
      if (submitted) {
        if (slot.cache.last_path ==
            VulkanCommandReplayCache::LastPath::record) {
          ++solver_chunk_builds_;
          if (replaces_recording) {
            ++solver_chunk_invalidations_;
          }
        } else if (slot.cache.last_path ==
                   VulkanCommandReplayCache::LastPath::replay) {
          if (preconditioner_) {
            preconditioner_->record_replayed_apply_calls(chunk_iterations);
          } else if (block_preconditioner_) {
            block_preconditioner_->record_replayed_apply_calls(
                chunk_iterations);
          }
          ++solver_chunk_reuses_;
          ++solver_chunk_replays_;
        }
        solver_chunk_replay_state_->unavailable_reason = "none";
      } else {
        solver_chunk_replay_state_->unavailable_reason =
            "native_command_replay_fallback";
      }
    }
    if (!submitted) {
      if (host_check_every_k || native_chunk_requested) {
        ++solver_chunk_direct_submissions_;
      }
      for (int local_iteration = 0; local_iteration < chunk_iterations;
           ++local_iteration) {
        const int iteration = chunk_start_iteration + local_iteration;
        issue_iteration(iteration, current_rr, next_rr);
        if (iteration + 1 < fixed_iterations_) {
          std::swap(current_rr, next_rr);
        }
      }
    } else {
      int rr_swaps = 0;
      for (int local_iteration = 0; local_iteration < chunk_iterations;
           ++local_iteration) {
        if (chunk_start_iteration + local_iteration + 1 < fixed_iterations_) {
          ++rr_swaps;
        }
      }
      if ((rr_swaps & 1) != 0) {
        std::swap(current_rr, next_rr);
      }
    }
    executed_this_solve += chunk_iterations;
    if (has_preconditioner()) {
      preconditioner_applies_this_solve +=
          static_cast<std::uint64_t>(chunk_iterations);
    }
    ++chunk_slot_index;
    if (host_check_every_k) {
      int32_t chunk_status_host = 0;
      int32_t chunk_completed_host = 0;
      program->synchronize();
      program->copy_ndarray_to_host(status_scalar_, &chunk_status_host,
                                    sizeof(chunk_status_host));
      program->copy_ndarray_to_host(completed_iterations_scalar_,
                                    &chunk_completed_host,
                                    sizeof(chunk_completed_host));
      solve_host_readbacks += 2;
      solve_host_synchronizations += 1;
      solve_device_to_host_bytes += 2 * sizeof(int32_t);
      terminal = chunk_status_host != 0;
    }
  }
  program->vulkan_sparse_norm(residual_, residual_norm_scalar_, n);

  float initial_rr_host = 0.0f;
  float residual_norm_host = 0.0f;
  int32_t status_host = 0;
  int32_t completed_iterations_host = fixed_iterations_;
  float rhs_squared_host = 0.0f;
  program->synchronize();
  program->copy_ndarray_to_host(initial_rr_, &initial_rr_host,
                                sizeof(initial_rr_host));
  program->copy_ndarray_to_host(residual_norm_scalar_, &residual_norm_host,
                                sizeof(residual_norm_host));
  program->copy_ndarray_to_host(status_scalar_, &status_host,
                                sizeof(status_host));
  if (adaptive_) {
    program->copy_ndarray_to_host(completed_iterations_scalar_,
                                  &completed_iterations_host,
                                  sizeof(completed_iterations_host));
    if (relative_tolerance_ > 0.0f) {
      program->copy_ndarray_to_host(rhs_squared_, &rhs_squared_host,
                                    sizeof(rhs_squared_host));
    }
  }
  solve_host_readbacks +=
      adaptive_ ? (relative_tolerance_ > 0.0f ? 5 : 4) : 3;
  solve_host_synchronizations += 1;
  solve_device_to_host_bytes +=
      2 * sizeof(float32) +
      (adaptive_ ? (relative_tolerance_ > 0.0f ? sizeof(float32) : 0) +
                       2 * sizeof(int32_t)
                 : sizeof(int32_t));
  initial_residual_norm_ =
      std::isfinite(initial_rr_host) && initial_rr_host >= 0.0f
          ? std::sqrt(static_cast<double>(initial_rr_host))
          : std::numeric_limits<double>::quiet_NaN();
  residual_norm_ = static_cast<double>(residual_norm_host);
  if (adaptive_ && relative_tolerance_ > 0.0f &&
      std::isfinite(rhs_squared_host) && rhs_squared_host >= 0.0f) {
    relative_reference_norm_ =
        std::sqrt(static_cast<double>(rhs_squared_host));
  }
  effective_tolerance_ =
      std::max(static_cast<double>(absolute_tolerance_),
               static_cast<double>(relative_tolerance_) *
                   relative_reference_norm_);
  status_ = status_host;
  iterations_ = adaptive_ ? completed_iterations_host : fixed_iterations_;
  const bool finite_residuals = std::isfinite(initial_residual_norm_) &&
                                std::isfinite(residual_norm_);
  is_success_ =
      adaptive_
          ? status_ == static_cast<int>(SparseSolveStatus::kConverged) &&
                finite_residuals && residual_norm_ <= effective_tolerance_
          : status_ ==
                    static_cast<int>(SparseSolveStatus::kMaxIterations) &&
                finite_residuals;
  total_iterations_ += static_cast<std::uint64_t>(iterations_);
  executed_iterations_ += static_cast<std::uint64_t>(executed_this_solve);
  operator_apply_calls_ +=
      static_cast<std::uint64_t>(executed_this_solve + 1);
  if (has_preconditioner()) {
    preconditioner_apply_calls_ += preconditioner_applies_this_solve;
  }
  device_scalar_operations_ += static_cast<std::uint64_t>(
      (executed_this_solve > 0 ? 4 * executed_this_solve - 2 : 0) +
      (adaptive_ ? executed_this_solve + 2 : 0) +
      (has_preconditioner() ? preconditioner_applies_this_solve : 0));
  host_scalar_readbacks_ += solve_host_readbacks;
  host_synchronizations_ += solve_host_synchronizations;
  device_to_device_bytes_ +=
      2 * static_cast<std::uint64_t>(n) * sizeof(float32) +
      (adaptive_ ? 3 : 2) * sizeof(uint32_t);
  device_to_host_bytes_ += solve_device_to_host_bytes;
#else
  TI_NOT_IMPLEMENTED;
#endif
}

SparseSolvePlanRuntimeStatistics
VulkanCGIterationPlan::debug_runtime_statistics() const {
  std::lock_guard<std::mutex> lock(solve_mutex_);
  const auto operator_stats = matrix_->debug_runtime_statistics();
  SparseSolvePlanRuntimeStatistics result;
  result.backend_family = "vulkan";
  if (has_preconditioner()) {
    if (operator_preconditioner_) {
      result.method =
          adaptive_ ? "pcg_linear_operator_bounded_masked_probe"
                    : "pcg_linear_operator_fixed_iteration_probe";
      result.preconditioner_method = preconditioner_plan_->method();
    } else if (compiled_kernel_preconditioner_) {
      result.method =
          adaptive_ ? "pcg_compiled_kernel_bounded_masked_probe"
                    : "pcg_compiled_kernel_fixed_iteration_probe";
      result.preconditioner_method =
          compiled_kernel_preconditioner_->debug_runtime_statistics().method;
    } else if (block_preconditioner_) {
      result.method =
          adaptive_ ? "pcg_block_jacobi_bounded_masked_probe"
                    : "pcg_block_jacobi_fixed_iteration_probe";
      result.preconditioner_method = "block_jacobi";
    } else {
      result.method = adaptive_ ? "pcg_jacobi_bounded_masked_probe"
                                : "pcg_jacobi_fixed_iteration_probe";
      result.preconditioner_method = "jacobi";
    }
  } else {
    if (compiled_graph_operator_) {
      result.method = adaptive_
                          ? "cg_compiled_graph_bounded_masked_probe"
                          : "cg_compiled_graph_fixed_iteration_probe";
    } else if (compiled_kernel_operator_) {
      result.method = adaptive_
                          ? "cg_compiled_kernel_bounded_masked_probe"
                          : "cg_compiled_kernel_fixed_iteration_probe";
    } else {
      result.method = adaptive_ ? "cg_bounded_masked_probe"
                                : "cg_fixed_iteration_probe";
    }
  }
  result.dtype = "f32";
  result.rows = matrix_->num_rows();
  result.cols = matrix_->num_cols();
  result.max_iterations = fixed_iterations_;
  result.absolute_tolerance = static_cast<double>(absolute_tolerance_);
  result.relative_tolerance = static_cast<double>(relative_tolerance_);
  result.last_relative_reference_norm = relative_reference_norm_;
  result.last_effective_tolerance = effective_tolerance_;
  result.operator_pattern_version = operator_stats.pattern_version;
  result.operator_numeric_version = operator_stats.numeric_version;
  result.last_solve_pattern_version = last_solve_pattern_version_;
  result.last_solve_numeric_version = last_solve_numeric_version_;
  result.operator_pattern_changed_since_last_solve =
      solve_calls_ > 0 && result.operator_pattern_version !=
                              result.last_solve_pattern_version;
  result.operator_numeric_changed_since_last_solve =
      solve_calls_ > 0 && result.operator_numeric_version !=
                              result.last_solve_numeric_version;
  result.solve_calls = solve_calls_;
  result.total_iterations = total_iterations_;
  result.logical_iterations = total_iterations_;
  result.executed_iterations = executed_iterations_;
  result.wasted_iterations = executed_iterations_ - total_iterations_;
  result.workspace_builds = workspace_builds_;
  result.workspace_reuses = workspace_reuses_;
  result.operator_apply_calls = operator_apply_calls_;
  result.operator_apply_calls_available = true;
  result.preconditioner_apply_calls = preconditioner_apply_calls_;
  result.host_scalar_reductions = 0;
  result.device_scalar_operations = device_scalar_operations_;
  result.host_scalar_readbacks = host_scalar_readbacks_;
  result.host_synchronizations = host_synchronizations_;
  result.requested_solver_execution_policy =
      adaptive_ ? sparse_solve_execution_policy_name(execution_policy_)
                : "fixed_budget";
  result.solver_execution_policy =
      adaptive_ ? sparse_solve_execution_policy_name(execution_policy_)
                : "fixed_budget";
  result.host_check_interval = adaptive_ ? host_check_interval_ : 0;
  result.solver_chunk_builds = solver_chunk_builds_;
  result.solver_chunk_reuses = solver_chunk_reuses_;
  result.solver_chunk_direct_submissions =
      solver_chunk_direct_submissions_;
  result.solver_chunk_replays = solver_chunk_replays_;
  result.solver_chunk_rebinds = solver_chunk_rebinds_;
  result.solver_chunk_invalidations = solver_chunk_invalidations_;
  result.solver_graph_enabled = solver_chunk_builds_ > 0;
  result.solver_replay_unavailable_reason =
      result.solver_graph_enabled
          ? "none"
          : ((adaptive_ &&
              execution_policy_ ==
                  SparseSolveExecutionPolicy::host_check_every_k)
                 ? (solver_chunk_replay_state_
                        ? solver_chunk_replay_state_->unavailable_reason
                        : (((csr_matrix_ != nullptr || bsr_matrix_ != nullptr) &&
                            (!has_preconditioner() ||
                             preconditioner_ != nullptr ||
                             block_preconditioner_ != nullptr))
                               ? (program_->profiler != nullptr
                                      ? "runtime_profiler_scopes_enabled"
                                      : "native_command_replay_disabled")
                               : "provider_not_record_composable"))
                 : "not_requested");
  result.solver_scalar_location = "device";
  result.solver_stream_policy = result.solver_graph_enabled
                                    ? "recorded_compute_sequence"
                                    : "program_submission_order";
  result.fixed_iteration_only = !adaptive_;
  result.bounded_masked_execution = adaptive_;
  result.persistent_vector_count = has_preconditioner() ? 4 : 3;
  result.persistent_vector_reserved_bytes =
      result.persistent_vector_count *
      static_cast<std::uint64_t>(matrix_->num_rows()) * sizeof(float32);
  result.persistent_scalar_count = adaptive_ ? 11 : 9;
  result.persistent_scalar_reserved_bytes =
      (adaptive_ ? 8 : 7) * sizeof(float32) +
      (adaptive_ ? 3 : 2) * sizeof(int32_t);
  result.external_preconditioner = has_preconditioner();
  result.preconditioner_ownership_scope =
      has_preconditioner() ? "external_plan" : "none";
  result.solver_state_rebuilt_each_solve = false;
  result.device_to_device_bytes = device_to_device_bytes_;
  result.device_to_host_bytes = device_to_host_bytes_;
  result.host_to_device_bytes = host_to_device_bytes_;
  append_operator_plan_statistics(*operator_plan_, false, result);
  if (preconditioner_plan_) {
    append_preconditioner_plan_statistics(*preconditioner_plan_, result);
  }
  return result;
}

void VulkanCGIterationPlan::release_workspace() {
#if defined(TI_WITH_VULKAN)
  solver_chunk_replay_state_.reset();
  if (!program_) {
    return;
  }
  auto release = [&](Ndarray *&array) {
    if (array) {
      program_->delete_ndarray(array);
      array = nullptr;
    }
  };
  release(zero_status_scalar_);
  release(completed_iterations_scalar_);
  release(status_scalar_);
  release(residual_norm_scalar_);
  release(beta_);
  release(alpha_);
  release(p_ap_);
  release(rr_b_);
  release(rr_a_);
  release(initial_rr_);
  release(rhs_squared_);
  release(preconditioned_residual_);
  release(direction_);
  release(residual_);
  release(ap_);
#endif
}

std::unique_ptr<VulkanCGIterationPlan> make_vulkan_cg_iteration_plan(
    Program *program,
    SparseMatrix &matrix,
    int fixed_iterations) {
  return std::make_unique<VulkanCGIterationPlan>(program, matrix,
                                                 fixed_iterations);
}

std::unique_ptr<VulkanCGIterationPlan> make_vulkan_cg_convergence_plan(
    Program *program,
    SparseMatrix &matrix,
    int max_iterations,
    float absolute_tolerance,
    float relative_tolerance) {
  return std::make_unique<VulkanCGIterationPlan>(
      program, matrix, max_iterations, absolute_tolerance,
      relative_tolerance);
}

std::unique_ptr<VulkanCGIterationPlan>
make_vulkan_jacobi_pcg_convergence_plan(
    Program *program,
    SparseMatrix &matrix,
    SparseJacobiPreconditionerPlan &preconditioner,
    int max_iterations,
    float absolute_tolerance,
    float relative_tolerance) {
  return std::make_unique<VulkanCGIterationPlan>(
      program, matrix, preconditioner, max_iterations,
      absolute_tolerance, relative_tolerance);
}

std::unique_ptr<VulkanCGIterationPlan>
make_vulkan_block_jacobi_pcg_convergence_plan(
    Program *program,
    SparseMatrix &matrix,
    SparseBlockJacobiPreconditionerPlan &preconditioner,
    int max_iterations,
    float absolute_tolerance,
    float relative_tolerance) {
  return std::make_unique<VulkanCGIterationPlan>(
      program, matrix, preconditioner, max_iterations,
      absolute_tolerance, relative_tolerance);
}

std::unique_ptr<VulkanCGIterationPlan>
make_vulkan_compiled_kernel_cg_convergence_plan(
    Program *program,
    CompiledKernelLinearOperator &matrix,
    int max_iterations,
    float absolute_tolerance,
    float relative_tolerance) {
  return std::make_unique<VulkanCGIterationPlan>(
      program, matrix, max_iterations, absolute_tolerance,
      relative_tolerance);
}

std::unique_ptr<VulkanCGIterationPlan>
make_vulkan_compiled_graph_cg_convergence_plan(
    Program *program,
    CompiledGraphLinearOperator &matrix,
    int max_iterations,
    float absolute_tolerance,
    float relative_tolerance) {
  return std::make_unique<VulkanCGIterationPlan>(
      program, matrix, max_iterations, absolute_tolerance,
      relative_tolerance);
}

std::unique_ptr<VulkanCGIterationPlan>
make_vulkan_compiled_kernel_pcg_convergence_plan(
    Program *program,
    CompiledKernelLinearOperator &matrix,
    CompiledKernelPreconditionerPlan &preconditioner,
    int max_iterations,
    float absolute_tolerance,
    float relative_tolerance) {
  return std::make_unique<VulkanCGIterationPlan>(
      program, matrix, preconditioner, max_iterations, absolute_tolerance,
      relative_tolerance);
}

std::unique_ptr<VulkanCGIterationPlan>
make_vulkan_experimental_linear_operator_pcg_convergence_plan(
    Program *program,
    CompiledKernelLinearOperator &matrix,
    ExperimentalLinearOperatorHandle &preconditioner,
    int max_iterations,
    float absolute_tolerance,
    float relative_tolerance) {
  return std::make_unique<VulkanCGIterationPlan>(
      program, matrix, preconditioner, max_iterations, absolute_tolerance,
      relative_tolerance);
}
}  // namespace taichi::lang
