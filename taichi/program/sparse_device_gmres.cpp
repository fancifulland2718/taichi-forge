#include "taichi/program/sparse_device_gmres.h"

#include "taichi/program/linear_operator.h"
#include "taichi/rhi/cuda/primitives/hierarchical_ptx.h"
#include "taichi/util/environ_config.h"

#if defined(TI_WITH_CUDA)
#include "taichi/rhi/cuda/cuda_context.h"
#endif

#if defined(TI_WITH_VULKAN)
#include "taichi/program/vulkan_command_replay.h"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace taichi::lang {

namespace {

constexpr int kStateWords = 32;
constexpr int kStateBytes = kStateWords * sizeof(std::uint32_t);
constexpr int kInitialResidualSquared = 0;
constexpr int kTrueResidualSquared = 1;
constexpr int kRelativeReferenceNorm = 3;
constexpr int kEffectiveTolerance = 4;
constexpr int kStatus = 16;
constexpr int kCompletedIterations = 17;
constexpr int kSolveActive = 18;
constexpr int kBreakdownReason = 22;
constexpr int kRestartCycles = 24;
constexpr int kHappyBreakdowns = 25;

float state_float(const std::uint32_t *state, int index) {
  float value = 0.0f;
  std::memcpy(&value, state + index, sizeof(value));
  return value;
}

std::int32_t state_int(const std::uint32_t *state, int index) {
  std::int32_t value = 0;
  std::memcpy(&value, state + index, sizeof(value));
  return value;
}

bool is_cuda_or_vulkan(Arch arch) {
  return arch == Arch::cuda || arch == Arch::vulkan;
}

#if defined(TI_WITH_VULKAN)
void push_gmres_resource(VulkanCommandReplayKey &key,
                         const Ndarray *array) {
  if (!array) {
    for (int i = 0; i < 4; ++i) {
      key.push(0);
    }
    return;
  }
  const auto handle = array->runtime_resource_handle();
  key.push(handle.domain);
  key.push(handle.kind);
  key.push(handle.index);
  key.push(handle.generation);
}

void push_gmres_stamp(VulkanCommandReplayKey &key,
                      const OperatorResourceStamp &stamp) {
  key.push(stamp.program_generation);
  key.push(stamp.schema_revision);
  key.push(stamp.topology_revision);
  key.push(stamp.binding_revision);
}
#endif

SparseSolveBreakdownReason decode_breakdown_reason(std::int32_t value) {
  TI_ERROR_IF(value < static_cast<std::int32_t>(
                          SparseSolveBreakdownReason::none) ||
                  value > static_cast<std::int32_t>(
                              SparseSolveBreakdownReason::hessenberg_singular),
              "Device GMRES returned invalid breakdown reason {}.", value);
  return static_cast<SparseSolveBreakdownReason>(value);
}

}  // namespace

#if defined(TI_WITH_CUDA)

struct DeviceGMRESCudaReplayState {
  struct Key {
    RuntimeResourceHandle solution;
    RuntimeResourceHandle rhs;
    OperatorResourceStamp operator_stamp;
    std::array<std::uintptr_t, 19> resources{};
    std::uintptr_t provider{0};
    int rows{0};
    int restart{0};
    int cycle_steps{0};
    bool limit_reached{false};

    bool operator==(const Key &other) const {
      return solution == other.solution && rhs == other.rhs &&
             operator_stamp.program_generation ==
                 other.operator_stamp.program_generation &&
             operator_stamp.schema_revision ==
                 other.operator_stamp.schema_revision &&
             operator_stamp.topology_revision ==
                 other.operator_stamp.topology_revision &&
             operator_stamp.binding_revision ==
                 other.operator_stamp.binding_revision &&
             resources == other.resources && provider == other.provider &&
             rows == other.rows && restart == other.restart &&
             cycle_steps == other.cycle_steps &&
             limit_reached == other.limit_reached;
    }
  };

  struct Entry {
    CUgraphExec executable{nullptr};
    Key key;
    bool key_valid{false};
    std::uint64_t operator_numeric_revision{0};
    std::optional<Program::NdarrayResourceLease> solution_lease;
    std::optional<Program::NdarrayResourceLease> rhs_lease;

    void reset() {
      if (executable) {
        CUDADriver::get_instance().graph_exec_destroy(executable);
      }
      executable = nullptr;
      key_valid = false;
      solution_lease.reset();
      rhs_lease.reset();
    }
  };

  CUstream capture_stream{nullptr};
  std::array<Entry, 66> entries;
  bool disabled{false};
  std::string unavailable_reason{"not_built"};

  ~DeviceGMRESCudaReplayState() {
    CUDAContext::get_instance().make_current();
    CUDADriver::get_instance().stream_synchronize(nullptr);
    for (auto &entry : entries) {
      entry.reset();
    }
    if (capture_stream) {
      CUDADriver::get_instance().stream_destroy(capture_stream);
    }
  }

  CUstream ensure_capture_stream() {
    if (!capture_stream) {
      CUDAContext::get_instance().make_current();
      CUDADriver::get_instance().stream_create(
          reinterpret_cast<void **>(&capture_stream), CU_STREAM_NON_BLOCKING);
    }
    return capture_stream;
  }
};

#else

struct DeviceGMRESCudaReplayState {};

#endif

#if defined(TI_WITH_VULKAN)

struct DeviceGMRESVulkanReplayState {
  struct Slot {
    VulkanCommandReplayCache cache;
    RuntimeResourceHandle solution;
    RuntimeResourceHandle rhs;
    std::optional<Program::NdarrayResourceLease> solution_lease;
    std::optional<Program::NdarrayResourceLease> rhs_lease;
    std::uint64_t operator_numeric_revision{0};
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
};

#else

struct DeviceGMRESVulkanReplayState {};

#endif

DeviceGMRES::DeviceGMRES(
    Program *program,
    LinearOperatorHandle &operator_handle,
    SparseMatrix *stored_matrix,
    LinearOperatorHandle *preconditioner,
    int max_iterations,
    int restart,
    float absolute_tolerance,
    float relative_tolerance,
    std::vector<LinearOperatorHandle *>
        flexible_preconditioners)
    : program_(program),
      operator_handle_(&operator_handle),
      stored_matrix_(stored_matrix),
      operator_preconditioner_(preconditioner),
      operator_flexible_preconditioners_(
          std::move(flexible_preconditioners)),
      max_iterations_(max_iterations),
      restart_(restart),
      absolute_tolerance_(absolute_tolerance),
      relative_tolerance_(relative_tolerance),
      host_check_interval_(restart) {
  TI_ERROR_IF(!program_ || operator_handle.program() != program_ ||
                  !is_cuda_or_vulkan(program_->compile_config().arch),
              "Device GMRES requires a CUDA or Vulkan operator in its "
              "owning Program.");
  const auto &descriptor = operator_handle.descriptor();
  TI_ERROR_IF(descriptor.domain != descriptor.range ||
                  descriptor.domain.scalar_type != PrimitiveType::f32 ||
                  descriptor.domain.scalar_extent == 0 ||
                  descriptor.domain.scalar_extent >
                      static_cast<std::size_t>(
                          std::numeric_limits<int>::max()),
              "Device GMRES requires a non-empty square scalar f32 "
              "operator with int-sized extent.");
  rows_ = static_cast<int>(descriptor.domain.scalar_extent);
  multi_dot_groups_ = std::min(
      65535, (rows_ + 1024 - 1) / 1024);
  if (stored_matrix_) {
    TI_ERROR_IF(stored_matrix_->num_rows() != rows_ ||
                    stored_matrix_->num_cols() != rows_ ||
                    stored_matrix_->get_data_type() != PrimitiveType::f32,
                "Device GMRES stored provider does not match its operator "
                "descriptor.");
  }
  validate_controls();
  operator_plan_ =
      std::make_unique<OperatorPlan>(program_, operator_handle.binding());
  validate_operator_solver_compatibility(
      operator_plan_->descriptor(), operator_plan_->mathematical_traits(),
      OperatorSolverFamily::gmres);
  if (operator_preconditioner_) {
    preconditioner_plan_ = make_solver_right_preconditioner_plan(
        program_, *operator_plan_, *operator_preconditioner_,
        "linear_operator");
  }
  TI_ERROR_IF(operator_preconditioner_ &&
                  !operator_flexible_preconditioners_.empty(),
              "Device GMRES cannot combine fixed and variable right "
              "preconditioner bindings.");
  TI_ERROR_IF(operator_flexible_preconditioners_.size() > 32,
              "Device FGMRES supports at most 32 scheduled actions.");
  for (auto *action : operator_flexible_preconditioners_) {
    TI_ERROR_IF(!action,
                "Device FGMRES action table contains a null action.");
    flexible_preconditioner_plans_.push_back(
        make_solver_flexible_right_preconditioner_plan(
            program_, *operator_plan_, *action,
            "variable_linear_action"));
  }
  try {
    allocate_workspace();
    if (program_->compile_config().arch == Arch::cuda) {
      initialize_cuda();
    } else {
      execution_policy_ = SparseSolveExecutionPolicy::fixed_budget_masked;
      host_check_interval_ = max_iterations_;
    }
  } catch (...) {
    release_cuda();
    release_workspace();
    throw;
  }
}

DeviceGMRES::~DeviceGMRES() {
  cuda_replay_.reset();
  vulkan_replay_.reset();
  release_cuda();
  release_workspace();
}

void DeviceGMRES::validate_controls() const {
  TI_ERROR_IF((restart_ != 8 && restart_ != 16 && restart_ != 32) ||
                  max_iterations_ < 0 ||
                  !std::isfinite(absolute_tolerance_) ||
                  !std::isfinite(relative_tolerance_) ||
                  absolute_tolerance_ < 0.0f ||
                  relative_tolerance_ < 0.0f ||
                  (absolute_tolerance_ == 0.0f &&
                   relative_tolerance_ == 0.0f),
              "Device GMRES requires restart 8, 16, or 32, non-negative "
              "iterations, and finite non-negative atol/rtol with at least "
              "one positive tolerance.");
}

void DeviceGMRES::allocate_workspace() {
  auto f32 = [&](std::size_t count) {
    return program_->create_ndarray(
        PrimitiveType::f32, {static_cast<int>(count)},
        ExternalArrayLayout::kNull, false);
  };
  try {
    basis_ = f32(static_cast<std::size_t>(restart_ + 1) * rows_);
    if (flexible()) {
      preconditioned_basis_ =
          f32(static_cast<std::size_t>(restart_) * rows_);
    }
    residual_ = f32(rows_);
    current_ = f32(rows_);
    work_ = f32(rows_);
    update_ = f32(rows_);
    if (has_preconditioner()) {
      preconditioned_ = f32(rows_);
    }
    multi_dot_partials_ =
        f32(static_cast<std::size_t>(restart_) * multi_dot_groups_);
    projection_ = f32(restart_);
    hessenberg_ =
        f32(static_cast<std::size_t>(restart_) * (restart_ + 1));
    cosines_ = f32(restart_);
    sines_ = f32(restart_);
    least_squares_rhs_ = f32(restart_ + 1);
    coefficients_ = f32(restart_);
    initial_residual_squared_ = f32(1);
    rhs_squared_ = f32(1);
    dot0_ = f32(1);
    dot1_ = f32(1);
    state_ = program_->create_ndarray(
        PrimitiveType::i32, {kStateWords}, ExternalArrayLayout::kNull,
        false);
  } catch (...) {
    release_workspace();
    throw;
  }
}

void DeviceGMRES::release_workspace() {
  auto release = [&](Ndarray *&array) {
    if (array && program_) {
      program_->delete_ndarray(array);
    }
    array = nullptr;
  };
  release(state_);
  release(dot1_);
  release(dot0_);
  release(rhs_squared_);
  release(initial_residual_squared_);
  release(coefficients_);
  release(least_squares_rhs_);
  release(sines_);
  release(cosines_);
  release(hessenberg_);
  release(projection_);
  release(multi_dot_partials_);
  release(preconditioned_);
  release(update_);
  release(work_);
  release(current_);
  release(residual_);
  release(preconditioned_basis_);
  release(basis_);
}

void DeviceGMRES::initialize_cuda() {
#if defined(TI_WITH_CUDA)
  auto &cublas = CUBLASDriver::get_instance();
  TI_ERROR_IF(!cublas.is_loaded() && !cublas.load_cublas(),
              "Device GMRES failed to load cuBLAS.");
  cublasHandle_t handle = nullptr;
  cublas.cubCreate(&handle);
  TI_ERROR_IF(!handle, "Device GMRES failed to create a cuBLAS handle.");
  cublas_handle_ = handle;
  cublas.cubSetStream(handle, nullptr);
  CUstream observed = reinterpret_cast<CUstream>(1);
  cublas.cubGetStream(handle, &observed);
  TI_ERROR_IF(observed != nullptr,
              "Device GMRES could not bind cuBLAS to the solver stream.");
  cublas_stream_bound_ = true;
  cublas.cubSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
  cublasPointerMode_t mode = CUBLAS_POINTER_MODE_HOST;
  cublas.cubGetPointerMode(handle, &mode);
  TI_ERROR_IF(mode != CUBLAS_POINTER_MODE_DEVICE,
              "Device GMRES could not enable cuBLAS device scalar mode.");
  cublas_device_pointer_mode_ = true;
#else
  TI_NOT_IMPLEMENTED;
#endif
}

void DeviceGMRES::release_cuda() {
#if defined(TI_WITH_CUDA)
  if (cublas_handle_) {
    CUBLASDriver::get_instance().cubDestroy(
        static_cast<cublasHandle_t>(cublas_handle_));
  }
#endif
  cublas_handle_ = nullptr;
}

void DeviceGMRES::configure_execution_policy(
    SparseSolveExecutionPolicy policy,
    int host_check_interval) {
  std::lock_guard<std::mutex> lock(solve_mutex_);
  TI_ERROR_IF(solve_calls_ != 0,
              "Device GMRES execution policy must be configured before "
              "solve.");
  validate_sparse_solve_execution_policy(
      program_->compile_config().arch, policy, host_check_interval);
  if (program_->compile_config().arch == Arch::cuda) {
    TI_ERROR_IF(policy != SparseSolveExecutionPolicy::host_check_every_k,
                "CUDA GMRES supports host_check_every_k only.");
  } else {
    TI_ERROR_IF(policy != SparseSolveExecutionPolicy::host_check_every_k &&
                    policy != SparseSolveExecutionPolicy::fixed_budget_masked,
                "Vulkan GMRES supports host_check_every_k or "
                "fixed_budget_masked.");
  }
  TI_ERROR_IF(policy == SparseSolveExecutionPolicy::host_check_every_k &&
                  host_check_interval != restart_,
              "Device GMRES host_check_every_k requires "
              "check_interval == restart.");
  execution_policy_ = policy;
  host_check_interval_ =
      policy == SparseSolveExecutionPolicy::host_check_every_k
          ? restart_
          : max_iterations_;
}

bool DeviceGMRES::has_preconditioner() const {
  return preconditioner_plan_ != nullptr ||
         !flexible_preconditioner_plans_.empty();
}

bool DeviceGMRES::flexible() const {
  return !flexible_preconditioner_plans_.empty();
}

bool DeviceGMRES::native_stored_provider() const {
  if (!stored_matrix_ || has_preconditioner()) {
    return false;
  }
  const Arch arch = program_->compile_config().arch;
  if (arch == Arch::cuda) {
    return dynamic_cast<CuSparseMatrix *>(stored_matrix_) ||
           dynamic_cast<CuSparseBsrMatrix *>(stored_matrix_);
  }
  if (arch == Arch::vulkan) {
    return dynamic_cast<VulkanSparseMatrix *>(stored_matrix_) ||
           dynamic_cast<VulkanSparseBsrMatrix *>(stored_matrix_);
  }
  return false;
}

std::uintptr_t DeviceGMRES::address(const Ndarray *array) const {
  return program_->get_ndarray_data_ptr_as_int(array);
}

void DeviceGMRES::apply_operator(
    const OperatorPinnedAction &generation,
    const Ndarray &input,
    const Ndarray &output,
    void *stream,
    bool native_capture) {
#if defined(TI_WITH_CUDA)
  if (native_capture && program_->compile_config().arch == Arch::cuda) {
    if (auto *csr = dynamic_cast<CuSparseMatrix *>(stored_matrix_)) {
      csr->spmv(address(&input), address(&output),
                static_cast<CUstream>(stream));
      return;
    }
    auto *bsr = dynamic_cast<CuSparseBsrMatrix *>(stored_matrix_);
    TI_ASSERT(bsr);
    bsr->spmv(address(&input), address(&output),
              static_cast<CUstream>(stream));
    return;
  }
#endif
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

void DeviceGMRES::apply_preconditioner(
    const OperatorPinnedAction &generation,
    const Ndarray &input,
    const Ndarray &output) {
  TI_ERROR_IF(!preconditioner_plan_,
              "Device GMRES preconditioner plan is unavailable.");
  apply_preconditioner(*preconditioner_plan_, generation, input, output);
}

void DeviceGMRES::apply_preconditioner(
    PreconditionerPlan &preconditioner,
    const OperatorPinnedAction &generation,
    const Ndarray &input,
    const Ndarray &output) {
  auto &action = preconditioner.action();
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

void DeviceGMRES::backend_dot(const Ndarray &left,
                              const Ndarray &right,
                              const Ndarray &output,
                              void *stream) {
  if (program_->compile_config().arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    auto &cublas = CUBLASDriver::get_instance();
    cublas.cubSdot(
        static_cast<cublasHandle_t>(cublas_handle_), rows_,
        reinterpret_cast<const float *>(address(&left)), 1,
        reinterpret_cast<const float *>(address(&right)), 1,
        reinterpret_cast<float *>(address(&output)));
#else
    TI_NOT_IMPLEMENTED;
#endif
    return;
  }
  program_->vulkan_sparse_dot(
      const_cast<Ndarray *>(&left), const_cast<Ndarray *>(&right),
      const_cast<Ndarray *>(&output), rows_);
}

void DeviceGMRES::backend_true_residual(
    const OperatorPinnedAction &generation,
    const Ndarray &x,
    const Ndarray &b,
    void *stream,
    bool native_capture) {
  apply_operator(generation, x, *work_, stream, native_capture);
  if (program_->compile_config().arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    cuda::driver_zero_strided(
        reinterpret_cast<void *>(address(residual_)), rows_,
        cuda::CudaTransformValueType::f32, 0, sizeof(float), stream);
    cuda::driver_add_scaled_strided(
        reinterpret_cast<void *>(address(&b)),
        reinterpret_cast<void *>(address(residual_)), rows_,
        cuda::CudaTransformValueType::f32, 0, sizeof(float), 0,
        sizeof(float), 1.0, stream);
    cuda::driver_add_scaled_strided(
        reinterpret_cast<void *>(address(work_)),
        reinterpret_cast<void *>(address(residual_)), rows_,
        cuda::CudaTransformValueType::f32, 0, sizeof(float), 0,
        sizeof(float), -1.0, stream);
#else
    TI_NOT_IMPLEMENTED;
#endif
  } else {
    program_->copy_ndarray_fast(residual_, const_cast<Ndarray *>(&b));
    program_->vulkan_sparse_axpy(work_, residual_, rows_, -1.0f);
  }
  backend_dot(*residual_, *residual_, *dot0_, stream);
}

void DeviceGMRES::backend_scalar_stage(int stage,
                                       int step,
                                       bool limit_reached,
                                       void *stream) {
  if (program_->compile_config().arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    cuda::driver_sparse_gmres_scalar_f32(
        reinterpret_cast<void *>(address(initial_residual_squared_)),
        reinterpret_cast<void *>(address(rhs_squared_)),
        reinterpret_cast<void *>(address(dot0_)),
        reinterpret_cast<void *>(address(dot1_)),
        reinterpret_cast<void *>(address(hessenberg_)),
        reinterpret_cast<void *>(address(cosines_)),
        reinterpret_cast<void *>(address(sines_)),
        reinterpret_cast<void *>(address(least_squares_rhs_)),
        reinterpret_cast<void *>(address(coefficients_)),
        reinterpret_cast<void *>(address(state_)), absolute_tolerance_,
        relative_tolerance_, restart_, max_iterations_, stage, step,
        limit_reached, stream);
#else
    TI_NOT_IMPLEMENTED;
#endif
    return;
  }
  program_->vulkan_sparse_gmres_scalar(
      initial_residual_squared_, rhs_squared_, dot0_, dot1_, hessenberg_,
      cosines_, sines_, least_squares_rhs_, coefficients_, state_,
      absolute_tolerance_, relative_tolerance_, restart_, max_iterations_,
      stage, step, limit_reached);
}

void DeviceGMRES::backend_basis(const Ndarray &source,
                                int row,
                                int mode,
                                void *stream) {
  if (program_->compile_config().arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    cuda::driver_sparse_gmres_basis_f32(
        reinterpret_cast<void *>(address(&source)),
        reinterpret_cast<void *>(address(basis_)),
        reinterpret_cast<void *>(address(current_)),
        reinterpret_cast<void *>(address(state_)), rows_, rows_, row, mode,
        stream);
#else
    TI_NOT_IMPLEMENTED;
#endif
    return;
  }
  program_->vulkan_sparse_gmres_basis(
      const_cast<Ndarray *>(&source), basis_, current_, state_, rows_, rows_,
      row, mode);
}

void DeviceGMRES::backend_store_preconditioned_basis(int row,
                                                     void *stream) {
  TI_ASSERT(flexible() && preconditioned_basis_ && preconditioned_);
  if (program_->compile_config().arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    cuda::driver_sparse_gmres_basis_f32(
        reinterpret_cast<void *>(address(preconditioned_)),
        reinterpret_cast<void *>(address(preconditioned_basis_)),
        reinterpret_cast<void *>(address(update_)),
        reinterpret_cast<void *>(address(state_)), rows_, rows_, row, 2,
        stream);
#else
    TI_NOT_IMPLEMENTED;
#endif
    return;
  }
  program_->vulkan_sparse_gmres_basis(
      preconditioned_, preconditioned_basis_, update_, state_, rows_, rows_,
      row, 2);
}

void DeviceGMRES::backend_multi_dot(int basis_count, void *stream) {
  if (program_->compile_config().arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    cuda::driver_sparse_gmres_multi_dot_f32(
        reinterpret_cast<void *>(address(basis_)),
        reinterpret_cast<void *>(address(work_)),
        reinterpret_cast<void *>(address(multi_dot_partials_)),
        reinterpret_cast<void *>(address(projection_)),
        reinterpret_cast<void *>(address(state_)), rows_, rows_, basis_count,
        multi_dot_groups_, stream);
#else
    TI_NOT_IMPLEMENTED;
#endif
    return;
  }
  program_->vulkan_sparse_gmres_multi_dot(
      basis_, work_, multi_dot_partials_, projection_, state_, rows_, rows_,
      basis_count, multi_dot_groups_);
}

void DeviceGMRES::backend_projection(int step, int pass, void *stream) {
  if (program_->compile_config().arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    cuda::driver_sparse_gmres_projection_f32(
        reinterpret_cast<void *>(address(basis_)),
        reinterpret_cast<void *>(address(work_)),
        reinterpret_cast<void *>(address(projection_)),
        reinterpret_cast<void *>(address(hessenberg_)),
        reinterpret_cast<void *>(address(state_)), rows_, rows_, restart_,
        step, pass, stream);
#else
    TI_NOT_IMPLEMENTED;
#endif
    return;
  }
  program_->vulkan_sparse_gmres_projection(
      basis_, work_, projection_, hessenberg_, state_, rows_, rows_,
      restart_, step, pass);
}

void DeviceGMRES::backend_combine(void *stream) {
  Ndarray *combination_basis =
      flexible() ? preconditioned_basis_ : basis_;
  if (program_->compile_config().arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    cuda::driver_sparse_gmres_combine_f32(
        reinterpret_cast<void *>(address(combination_basis)),
        reinterpret_cast<void *>(address(coefficients_)),
        reinterpret_cast<void *>(address(update_)),
        reinterpret_cast<void *>(address(state_)), rows_, rows_, stream);
#else
    TI_NOT_IMPLEMENTED;
#endif
    return;
  }
  program_->vulkan_sparse_gmres_combine(
      combination_basis, coefficients_, update_, state_, rows_, rows_);
}

void DeviceGMRES::backend_add_update(const Ndarray &x,
                                     const Ndarray &update,
                                     void *stream) {
  if (program_->compile_config().arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    cuda::driver_add_scaled_strided(
        reinterpret_cast<void *>(address(&update)),
        reinterpret_cast<void *>(address(&x)), rows_,
        cuda::CudaTransformValueType::f32, 0, sizeof(float), 0,
        sizeof(float), 1.0, stream);
#else
    TI_NOT_IMPLEMENTED;
#endif
    return;
  }
  program_->vulkan_sparse_axpy(
      const_cast<Ndarray *>(&update), const_cast<Ndarray *>(&x), rows_,
      1.0f);
}

void DeviceGMRES::issue_cycle(
    const OperatorPinnedAction &operator_generation,
    const OperatorPinnedAction &preconditioner_generation,
    const std::vector<OperatorPinnedAction> &flexible_generations,
    const Ndarray &x,
    const Ndarray &b,
    int cycle_steps,
    int solve_iteration_offset,
    bool limit_reached,
    void *stream,
    bool native_capture) {
  backend_scalar_stage(1, 0, false, stream);
  backend_basis(*residual_, 0, 0, stream);
  for (int step = 0; step < cycle_steps; ++step) {
    if (flexible()) {
      const std::size_t action_index =
          static_cast<std::size_t>(solve_iteration_offset + step) %
          flexible_preconditioner_plans_.size();
      apply_preconditioner(
          *flexible_preconditioner_plans_[action_index],
          flexible_generations[action_index], *current_, *preconditioned_);
      backend_store_preconditioned_basis(step, stream);
      apply_operator(operator_generation, *preconditioned_, *work_, stream,
                     native_capture);
    } else if (has_preconditioner()) {
      apply_preconditioner(
          preconditioner_generation, *current_, *preconditioned_);
      apply_operator(operator_generation, *preconditioned_, *work_, stream,
                     native_capture);
    } else {
      apply_operator(operator_generation, *current_, *work_, stream,
                     native_capture);
    }
    backend_dot(*work_, *work_, *dot0_, stream);
    backend_multi_dot(step + 1, stream);
    backend_projection(step, 0, stream);
    backend_multi_dot(step + 1, stream);
    backend_projection(step, 1, stream);
    backend_dot(*work_, *work_, *dot1_, stream);
    backend_scalar_stage(2, step, false, stream);
    backend_basis(*work_, step + 1, 1, stream);
  }
  backend_scalar_stage(3, 0, false, stream);
  backend_combine(stream);
  if (preconditioner_plan_) {
    apply_preconditioner(
        preconditioner_generation, *update_, *preconditioned_);
    backend_add_update(x, *preconditioned_, stream);
  } else {
    backend_add_update(x, *update_, stream);
  }
  backend_true_residual(operator_generation, x, b, stream, native_capture);
  backend_scalar_stage(4, 0, limit_reached, stream);
}

bool DeviceGMRES::try_submit_cuda_cycle(
    const OperatorPinnedAction &operator_generation,
    const OperatorPinnedAction &preconditioner_generation,
    const std::vector<OperatorPinnedAction> &flexible_generations,
    const Ndarray &x,
    const Ndarray &b,
    int cycle_steps,
    int solve_iteration_offset,
    bool limit_reached) {
#if defined(TI_WITH_CUDA)
  if (program_->compile_config().arch != Arch::cuda ||
      !native_stored_provider() || cycle_steps <= 0 ||
      cycle_steps > restart_ ||
      get_environ_config("TI_CUDA_SOLVER_CHUNK_REPLAY", 1) == 0 ||
      program_->compile_config().debug) {
    return false;
  }
  auto *csr = dynamic_cast<CuSparseMatrix *>(stored_matrix_);
  auto *bsr = dynamic_cast<CuSparseBsrMatrix *>(stored_matrix_);
  if (!(csr ? csr->supports_spmv_stream_binding()
            : bsr && bsr->supports_spmv_stream_binding())) {
    return false;
  }
  auto &driver = CUDADriver::get_instance();
  if (!driver.stream_begin_capture.available() ||
      !driver.stream_end_capture.available() ||
      !driver.graph_instantiate_with_flags.available() ||
      !driver.graph_launch.available()) {
    return false;
  }
  if (!cuda_replay_) {
    cuda_replay_ = std::make_unique<DeviceGMRESCudaReplayState>();
  }
  if (cuda_replay_->disabled) {
    return false;
  }
  DeviceGMRESCudaReplayState::Key key;
  key.solution = x.runtime_resource_handle();
  key.rhs = b.runtime_resource_handle();
  key.operator_stamp = operator_generation.resource_stamp();
  const Ndarray *arrays[] = {
      basis_, preconditioned_basis_, residual_, current_, work_, update_,
      preconditioned_,
      multi_dot_partials_, projection_, hessenberg_, cosines_, sines_,
      least_squares_rhs_, coefficients_, initial_residual_squared_,
      rhs_squared_, dot0_, dot1_, state_};
  for (std::size_t index = 0; index < std::size(arrays); ++index) {
    key.resources[index] = arrays[index] ? address(arrays[index]) : 0;
  }
  key.provider = reinterpret_cast<std::uintptr_t>(stored_matrix_);
  key.rows = rows_;
  key.restart = restart_;
  key.cycle_steps = cycle_steps;
  key.limit_reached = limit_reached;
  const std::size_t entry_index =
      static_cast<std::size_t>(cycle_steps) +
      (limit_reached ? 33u : 0u);
  auto &entry = cuda_replay_->entries[entry_index];
  if (entry.executable && entry.key_valid && entry.key == key) {
    if (entry.operator_numeric_revision !=
        key.operator_stamp.numeric_revision) {
      ++solver_chunk_rebinds_;
      entry.operator_numeric_revision =
          key.operator_stamp.numeric_revision;
    }
    driver.graph_launch(entry.executable, nullptr);
    ++solver_chunk_reuses_;
    ++solver_chunk_replays_;
    cuda_replay_->unavailable_reason = "none";
    return true;
  }
  if (entry.executable) {
    driver.stream_synchronize(nullptr);
    entry.reset();
    ++solver_chunk_invalidations_;
  }
  if (csr) {
    csr->spmv(address(current_), address(work_), nullptr);
  } else {
    bsr->spmv(address(current_), address(work_), nullptr);
  }
  const CUstream capture_stream = cuda_replay_->ensure_capture_stream();
  driver.stream_synchronize(capture_stream);
  auto &cublas = CUBLASDriver::get_instance();
  cublas.cubSetStream(
      static_cast<cublasHandle_t>(cublas_handle_), capture_stream);
  CUgraph graph = nullptr;
  auto capture_lock =
      CUDAContext::get_instance().get_graph_capture_lock_guard();
  if (driver.stream_begin_capture.call(
          capture_stream, CU_STREAM_CAPTURE_MODE_RELAXED) != CUDA_SUCCESS) {
    cublas.cubSetStream(
        static_cast<cublasHandle_t>(cublas_handle_), nullptr);
    cuda_replay_->disabled = true;
    cuda_replay_->unavailable_reason = "stream_capture_begin_failed";
    return false;
  }
  try {
    issue_cycle(operator_generation, preconditioner_generation,
                flexible_generations, x, b, cycle_steps,
                solve_iteration_offset, limit_reached, capture_stream, true);
  } catch (...) {
    (void)driver.stream_end_capture.call(capture_stream, &graph);
    if (graph) {
      driver.graph_destroy.call(graph);
    }
    cublas.cubSetStream(
        static_cast<cublasHandle_t>(cublas_handle_), nullptr);
    throw;
  }
  const auto end_error =
      driver.stream_end_capture.call(capture_stream, &graph);
  cublas.cubSetStream(
      static_cast<cublasHandle_t>(cublas_handle_), nullptr);
  if (end_error != CUDA_SUCCESS || !graph) {
    if (graph) {
      driver.graph_destroy.call(graph);
    }
    cuda_replay_->disabled = true;
    cuda_replay_->unavailable_reason = "stream_capture_end_failed";
    return false;
  }
  CUgraphExec executable = nullptr;
  const auto instantiate =
      driver.graph_instantiate_with_flags.call(&executable, graph, 0);
  driver.graph_destroy.call(graph);
  if (instantiate != CUDA_SUCCESS || !executable) {
    cuda_replay_->disabled = true;
    cuda_replay_->unavailable_reason = "graph_instantiate_failed";
    return false;
  }
  entry.executable = executable;
  entry.key = key;
  entry.key_valid = true;
  entry.operator_numeric_revision = key.operator_stamp.numeric_revision;
  entry.solution_lease.emplace(
      program_->acquire_ndarray_external_lease(key.solution));
  entry.rhs_lease.emplace(
      program_->acquire_ndarray_external_lease(key.rhs));
  ++solver_chunk_builds_;
  cuda_replay_->unavailable_reason = "none";
  driver.graph_launch(entry.executable, nullptr);
  return true;
#else
  return false;
#endif
}

bool DeviceGMRES::try_submit_vulkan_cycle(
    const OperatorPinnedAction &operator_generation,
    const OperatorPinnedAction &preconditioner_generation,
    const std::vector<OperatorPinnedAction> &flexible_generations,
    const Ndarray &x,
    const Ndarray &b,
    int cycle_steps,
    int solve_iteration_offset,
    bool limit_reached,
    std::size_t slot_index) {
#if defined(TI_WITH_VULKAN)
  if (program_->compile_config().arch != Arch::vulkan ||
      !native_stored_provider() || program_->profiler != nullptr ||
      get_environ_config("TI_VULKAN_SOLVER_CHUNK_REPLAY", 1) == 0) {
    return false;
  }
  if (!vulkan_replay_) {
    vulkan_replay_ =
        std::make_unique<DeviceGMRESVulkanReplayState>();
  }
  auto &slot = vulkan_replay_->slot(slot_index);
  VulkanCommandReplayKey key;
  key.push(231);
  key.push(program_->runtime_program_generation());
  key.push(program_->vulkan_sparse_algebra_replay_generation());
  key.push(static_cast<std::uint64_t>(restart_));
  key.push(static_cast<std::uint64_t>(cycle_steps));
  key.push(limit_reached ? 1 : 0);
  push_gmres_stamp(key, operator_generation.resource_stamp());
  const Ndarray *resources[] = {
      &x, &b, basis_, preconditioned_basis_, residual_, current_, work_,
      update_,
      preconditioned_, multi_dot_partials_, projection_, hessenberg_,
      cosines_, sines_, least_squares_rhs_, coefficients_,
      initial_residual_squared_, rhs_squared_, dot0_, dot1_, state_};
  for (const auto *resource : resources) {
    push_gmres_resource(key, resource);
  }
  const auto solution_handle = x.runtime_resource_handle();
  const auto rhs_handle = b.runtime_resource_handle();
  if (slot.solution != solution_handle || slot.rhs != rhs_handle) {
    if (slot.solution || slot.rhs) {
      slot.cache.reset();
      slot.solution_lease.reset();
      slot.rhs_lease.reset();
      ++solver_chunk_invalidations_;
      ++solver_chunk_rebinds_;
    }
    slot.solution = solution_handle;
    slot.rhs = rhs_handle;
    slot.solution_lease.emplace(
        program_->acquire_ndarray_external_lease(solution_handle));
    slot.rhs_lease.emplace(
        program_->acquire_ndarray_external_lease(rhs_handle));
  }
  const auto operator_stamp = operator_generation.resource_stamp();
  if (slot.operator_numeric_revision != 0 &&
      slot.operator_numeric_revision != operator_stamp.numeric_revision) {
    ++solver_chunk_rebinds_;
  }
  slot.operator_numeric_revision = operator_stamp.numeric_revision;
  const bool replaces =
      slot.cache.entry.cmdlist && slot.cache.entry.key != key;
  (void)program_->flush_if_pending();
  auto record = [&](Device *device, CommandList *cmdlist) {
    VulkanNativeCommandRecordingScope scope(program_, device, cmdlist);
    issue_cycle(operator_generation, preconditioner_generation,
                flexible_generations, x, b, cycle_steps,
                solve_iteration_offset, limit_reached, nullptr, true);
  };
  const bool submitted = slot.cache.submit_or_record(
      program_, program_->get_compute_device(), key, false, record);
  if (!submitted) {
    vulkan_replay_->unavailable_reason =
        "native_command_replay_fallback";
    return false;
  }
  if (slot.cache.last_path ==
      VulkanCommandReplayCache::LastPath::record) {
    ++solver_chunk_builds_;
    if (replaces) {
      ++solver_chunk_invalidations_;
    }
  } else {
    ++solver_chunk_reuses_;
    ++solver_chunk_replays_;
  }
  vulkan_replay_->unavailable_reason = "none";
  return true;
#else
  return false;
#endif
}

void DeviceGMRES::read_state(bool synchronize) {
  if (program_->compile_config().arch == Arch::vulkan && synchronize) {
    program_->synchronize();
  }
  if (program_->compile_config().arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    CUDADriver::get_instance().memcpy_device_to_host(
        host_state_, reinterpret_cast<void *>(address(state_)), kStateBytes);
#endif
  } else {
    program_->copy_ndarray_to_host(state_, host_state_, kStateBytes);
  }
  ++host_scalar_readbacks_;
  ++host_synchronizations_;
  device_to_host_bytes_ += kStateBytes;
}

void DeviceGMRES::solve(Program *program,
                        const Ndarray &x,
                        const Ndarray &b) {
  TI_ERROR_IF(program != program_,
              "Device GMRES solve must use its construction Program.");
  auto check_vector = [&](const char *role, const Ndarray &array) {
    TI_ERROR_IF(array.shape.size() != 1 ||
                    array.get_element_data_type() != PrimitiveType::f32 ||
                    !array.get_element_shape().empty() ||
                    array.get_nelement() !=
                        static_cast<std::size_t>(rows_),
                "Device GMRES {} must contain exactly {} scalar f32 "
                "entries.",
                role, rows_);
  };
  check_vector("solution", x);
  check_vector("right-hand side", b);
  TI_ERROR_IF(x.get_device_allocation() == b.get_device_allocation(),
              "Device GMRES solution and RHS must not alias.");

  std::lock_guard<std::mutex> lock(solve_mutex_);
  auto operator_generation = operator_plan_->pin();
  auto preconditioner_generation =
      preconditioner_plan_
          ? preconditioner_plan_->update_and_pin(operator_generation)
          : OperatorPinnedAction{};
  std::vector<OperatorPinnedAction> flexible_generations;
  flexible_generations.reserve(flexible_preconditioner_plans_.size());
  for (auto &plan : flexible_preconditioner_plans_) {
    flexible_generations.push_back(
        plan->update_and_pin(operator_generation));
  }
  auto submission_guard =
      program_->acquire_runtime_resource_submission_guard();
  std::vector<const Ndarray *> resources = {
      &x,
      &b,
      basis_,
      residual_,
      current_,
      work_,
      update_,
      multi_dot_partials_,
      projection_,
      hessenberg_,
      cosines_,
      sines_,
      least_squares_rhs_,
      coefficients_,
      initial_residual_squared_,
      rhs_squared_,
      dot0_,
      dot1_,
      state_};
  if (preconditioned_) {
    resources.push_back(preconditioned_);
  }
  if (preconditioned_basis_) {
    resources.push_back(preconditioned_basis_);
  }
  program_->retain_ndarrays_for_external_submission(
      resources.data(), resources.size());
  const auto stamp = operator_generation.resource_stamp();
  if (solve_calls_ > 0) {
    ++workspace_reuses_;
  }
  ++solve_calls_;
  last_solve_pattern_version_ = stamp.topology_revision;
  last_solve_numeric_version_ = stamp.numeric_revision;
  status_ = SparseSolveStatus::kNotRun;
  breakdown_reason_ = SparseSolveBreakdownReason::none;
  iterations_ = 0;
  initial_residual_norm_ = 0.0;
  residual_norm_ = 0.0;
  relative_reference_norm_ = 0.0;
  effective_tolerance_ = absolute_tolerance_;

  backend_true_residual(operator_generation, x, b, nullptr, false);
  program_->copy_ndarray_fast(initial_residual_squared_, dot0_);
  backend_dot(b, b, *rhs_squared_, nullptr);
  backend_scalar_stage(0, 0, max_iterations_ == 0, nullptr);
  const bool host_observed =
      execution_policy_ != SparseSolveExecutionPolicy::fixed_budget_masked;
  bool terminal = false;
  if (host_observed) {
    read_state(true);
    terminal = state_int(host_state_, kSolveActive) == 0;
  }

  int executed = 0;
  std::uint64_t physical_cycles = 0;
  std::size_t slot_index = 0;
  while (!terminal && executed < max_iterations_) {
    const int cycle_steps = std::min(restart_, max_iterations_ - executed);
    const bool limit_reached =
        executed + cycle_steps >= max_iterations_;
    bool submitted = try_submit_cuda_cycle(
        operator_generation, preconditioner_generation,
        flexible_generations, x, b, cycle_steps, executed, limit_reached);
    if (!submitted) {
      submitted = try_submit_vulkan_cycle(
          operator_generation, preconditioner_generation,
          flexible_generations, x, b, cycle_steps, executed, limit_reached,
          slot_index);
    }
    if (!submitted) {
      ++solver_chunk_direct_submissions_;
      issue_cycle(operator_generation, preconditioner_generation,
                  flexible_generations, x, b, cycle_steps, executed,
                  limit_reached, nullptr, false);
    }
    executed += cycle_steps;
    ++physical_cycles;
    ++slot_index;
    if (host_observed) {
      read_state(true);
      terminal = state_int(host_state_, kSolveActive) == 0;
    }
  }
  if (!host_observed || !terminal) {
    read_state(true);
  }

  status_ = static_cast<SparseSolveStatus>(
      state_int(host_state_, kStatus));
  iterations_ = state_int(host_state_, kCompletedIterations);
  breakdown_reason_ = decode_breakdown_reason(
      state_int(host_state_, kBreakdownReason));
  const float initial_rr =
      state_float(host_state_, kInitialResidualSquared);
  const float true_rr = state_float(host_state_, kTrueResidualSquared);
  initial_residual_norm_ =
      std::isfinite(initial_rr) && initial_rr >= 0.0f
          ? std::sqrt(static_cast<double>(initial_rr))
          : std::numeric_limits<double>::quiet_NaN();
  residual_norm_ = std::isfinite(true_rr) && true_rr >= 0.0f
                       ? std::sqrt(static_cast<double>(true_rr))
                       : std::numeric_limits<double>::quiet_NaN();
  relative_reference_norm_ =
      state_float(host_state_, kRelativeReferenceNorm);
  effective_tolerance_ =
      state_float(host_state_, kEffectiveTolerance);
  if (status_ == SparseSolveStatus::kConverged &&
      (!std::isfinite(residual_norm_) ||
       residual_norm_ > effective_tolerance_)) {
    status_ = SparseSolveStatus::kBreakdown;
    breakdown_reason_ = SparseSolveBreakdownReason::nonfinite;
  }

  total_iterations_ += static_cast<std::uint64_t>(iterations_);
  executed_iterations_ += static_cast<std::uint64_t>(executed);
  restart_cycles_ +=
      static_cast<std::uint64_t>(state_int(host_state_, kRestartCycles));
  happy_breakdowns_ +=
      static_cast<std::uint64_t>(state_int(host_state_, kHappyBreakdowns));
  const std::uint64_t physical = static_cast<std::uint64_t>(executed);
  operator_apply_calls_ += 1u + physical + physical_cycles;
  preconditioner_apply_calls_ +=
      flexible() ? physical
                 : (has_preconditioner() ? physical + physical_cycles : 0u);
  if (flexible()) {
    preconditioner_action_selections_ += physical;
    preconditioner_schedule_wraps_ +=
        physical > 0
            ? (physical - 1u) / flexible_preconditioner_plans_.size()
            : 0u;
  }
  dot_product_calls_ += 2u + 2u * physical + physical_cycles;
  multi_dot_calls_ += 2u * physical;
  vector_update_calls_ +=
      1u + 3u * physical + 4u * physical_cycles +
      (flexible() ? physical : 0u);
  device_scalar_operations_ +=
      1u + physical + 3u * physical_cycles;
  const std::uint64_t vector_bytes =
      static_cast<std::uint64_t>(rows_) * sizeof(float);
  device_to_device_bytes_ += sizeof(float) + 2u * vector_bytes;
  if (program_->compile_config().arch == Arch::vulkan) {
    device_to_device_bytes_ +=
        (1u + physical_cycles) * vector_bytes;
  }
}

SparseSolveResult DeviceGMRES::get_last_result() const {
  return {status_, iterations_, initial_residual_norm_, residual_norm_,
          absolute_tolerance_, relative_tolerance_,
          relative_reference_norm_, effective_tolerance_,
          breakdown_reason_};
}

SparseSolvePlanRuntimeStatistics
DeviceGMRES::debug_runtime_statistics() const {
  std::lock_guard<std::mutex> lock(solve_mutex_);
  SparseSolvePlanRuntimeStatistics result;
  result.backend_family = arch_name(program_->compile_config().arch);
  result.method = flexible() ? "fgmres" : "gmres";
  result.dtype = "f32";
  result.rows = rows_;
  result.cols = rows_;
  result.max_iterations = max_iterations_;
  result.absolute_tolerance = absolute_tolerance_;
  result.relative_tolerance = relative_tolerance_;
  result.last_relative_reference_norm = relative_reference_norm_;
  result.last_effective_tolerance = effective_tolerance_;
  result.last_breakdown_reason =
      sparse_solve_breakdown_reason_name(breakdown_reason_);
  const auto stamp = operator_plan_->resource_stamp();
  result.operator_pattern_version = stamp.topology_revision;
  result.operator_numeric_version = stamp.numeric_revision;
  result.last_solve_pattern_version = last_solve_pattern_version_;
  result.last_solve_numeric_version = last_solve_numeric_version_;
  result.operator_pattern_changed_since_last_solve =
      solve_calls_ > 0 &&
      stamp.topology_revision != last_solve_pattern_version_;
  result.operator_numeric_changed_since_last_solve =
      solve_calls_ > 0 &&
      stamp.numeric_revision != last_solve_numeric_version_;
  result.operator_action_provider = operator_plan_->provider_name();
  result.operator_asynchronous_submit =
      operator_plan_->capabilities().asynchronous_submit;
  const auto operator_stats = operator_plan_->debug_runtime_statistics();
  result.operator_generation_pins = operator_stats.generation_pins;
  result.operator_generation_changes = operator_stats.generation_changes;
  result.operator_numeric_generation_changes =
      operator_stats.numeric_generation_changes;
  result.operator_binding_generation_changes =
      operator_stats.binding_generation_changes;
  result.operator_plan_invalidations = operator_stats.invalidations;
  result.solve_calls = solve_calls_;
  result.total_iterations = total_iterations_;
  result.workspace_builds = 1;
  result.workspace_reuses = workspace_reuses_;
  result.operator_apply_calls = operator_apply_calls_;
  result.operator_apply_calls_available = true;
  result.dot_product_calls = dot_product_calls_;
  result.dot_product_calls_available = true;
  result.multi_dot_calls = multi_dot_calls_;
  result.multi_dot_calls_available = true;
  result.vector_update_calls = vector_update_calls_;
  result.vector_update_calls_available = true;
  result.device_scalar_operations = device_scalar_operations_;
  result.host_scalar_readbacks = host_scalar_readbacks_;
  result.host_synchronizations = host_synchronizations_;
  result.logical_iterations = total_iterations_;
  result.executed_iterations = executed_iterations_;
  result.wasted_iterations = executed_iterations_ - total_iterations_;
  result.solver_chunk_direct_submissions =
      solver_chunk_direct_submissions_;
  result.solver_chunk_builds = solver_chunk_builds_;
  result.solver_chunk_reuses = solver_chunk_reuses_;
  result.solver_chunk_replays = solver_chunk_replays_;
  result.solver_chunk_rebinds = solver_chunk_rebinds_;
  result.solver_chunk_invalidations = solver_chunk_invalidations_;
  result.restart_cycles = restart_cycles_;
  result.happy_breakdowns = happy_breakdowns_;
  result.restart = restart_;
  result.orthogonalization_strategy = "cgs2";
  result.orthogonalization_passes = 2;
  result.requested_solver_execution_policy =
      sparse_solve_execution_policy_name(execution_policy_);
  result.solver_execution_policy =
      result.requested_solver_execution_policy;
  result.host_check_interval = host_check_interval_;
  result.solver_graph_enabled = solver_chunk_builds_ > 0;
  if (flexible()) {
    result.solver_replay_unavailable_reason =
        "variable_action_capture_contract_unavailable";
  } else if (!native_stored_provider()) {
    result.solver_replay_unavailable_reason =
        "provider_not_capture_composable";
  } else if (program_->compile_config().arch == Arch::cuda) {
    if (get_environ_config("TI_CUDA_SOLVER_CHUNK_REPLAY", 1) == 0) {
      result.solver_replay_unavailable_reason =
          "disabled_by_environment";
    } else if (program_->compile_config().debug) {
      result.solver_replay_unavailable_reason = "debug_mode";
    } else {
#if defined(TI_WITH_CUDA)
      auto *csr = dynamic_cast<CuSparseMatrix *>(stored_matrix_);
      auto *bsr = dynamic_cast<CuSparseBsrMatrix *>(stored_matrix_);
      if (!(csr ? csr->supports_spmv_stream_binding()
                : bsr && bsr->supports_spmv_stream_binding())) {
        result.solver_replay_unavailable_reason =
            "provider_stream_binding_unavailable";
      } else {
        auto &driver = CUDADriver::get_instance();
        const bool driver_ready =
            driver.stream_begin_capture.available() &&
            driver.stream_end_capture.available() &&
            driver.graph_instantiate_with_flags.available() &&
            driver.graph_launch.available();
        result.solver_replay_unavailable_reason =
            driver_ready
                ? (cuda_replay_ ? cuda_replay_->unavailable_reason
                                : "not_built")
                : "cuda_graph_driver_functions_unavailable";
      }
#else
      result.solver_replay_unavailable_reason =
          "cuda_backend_not_compiled";
#endif
    }
  } else if (program_->profiler != nullptr) {
    result.solver_replay_unavailable_reason = "profiler_active";
  } else if (get_environ_config(
                 "TI_VULKAN_SOLVER_CHUNK_REPLAY", 1) == 0) {
    result.solver_replay_unavailable_reason =
        "disabled_by_environment";
  } else {
    result.solver_replay_unavailable_reason =
        vulkan_replay_ ? vulkan_replay_->unavailable_reason : "not_built";
  }
  result.solver_scalar_location = "device";
  result.solver_stream_policy = "backend_default";
  result.preconditioning_side = has_preconditioner() ? "right" : "none";
  result.bounded_masked_execution =
      execution_policy_ ==
      SparseSolveExecutionPolicy::fixed_budget_masked;
  result.preconditioner_method =
      preconditioner_plan_
          ? preconditioner_plan_->method()
          : (flexible() ? "variable_linear_action_table" : "identity");
  result.preconditioner_behavior =
      preconditioner_plan_
          ? "fixed_linear"
          : (flexible() ? "variable_linear" : "identity");
  result.preconditioner_apply_calls = preconditioner_apply_calls_;
  result.preconditioner_action_selections =
      preconditioner_action_selections_;
  result.preconditioner_schedule_wraps =
      preconditioner_schedule_wraps_;
  result.preconditioner_apply_calls_available = true;
  result.external_preconditioner =
      operator_preconditioner_ != nullptr || flexible();
  result.preconditioner_ownership_scope =
      flexible() ? "solve_plan_action_table_snapshot"
                 : (preconditioner_plan_ ? "solve_plan" : "none");
  if (preconditioner_plan_) {
    const auto &action = preconditioner_plan_->action();
    result.preconditioner_action_provider = action.provider_name();
    result.preconditioner_asynchronous_submit =
        action.capabilities().asynchronous_submit;
    const auto action_stats = action.debug_runtime_statistics();
    result.preconditioner_generation_pins = action_stats.generation_pins;
    result.preconditioner_generation_changes =
        action_stats.generation_changes;
    result.preconditioner_numeric_generation_changes =
        action_stats.numeric_generation_changes;
    result.preconditioner_binding_generation_changes =
        action_stats.binding_generation_changes;
    result.preconditioner_plan_invalidations = action_stats.invalidations;
    const auto lifecycle =
        preconditioner_plan_->debug_runtime_statistics();
    result.preconditioner_setup_calls = lifecycle.setup_calls;
    result.preconditioner_update_calls = lifecycle.update_calls;
    result.preconditioner_update_successes = lifecycle.update_successes;
    result.preconditioner_update_noops = lifecycle.update_noops;
    result.preconditioner_update_failures = lifecycle.update_failures;
    result.preconditioner_action_count = 1;
    result.preconditioner_action_selection = "fixed";
  } else if (flexible()) {
    append_solver_flexible_preconditioner_plan_statistics(
        flexible_preconditioner_plans_, result);
  }
  const std::uint64_t basis_count =
      static_cast<std::uint64_t>(restart_ + 1);
  const std::uint64_t auxiliary_vectors = has_preconditioner() ? 5u : 4u;
  const std::uint64_t preconditioned_basis_count =
      flexible() ? static_cast<std::uint64_t>(restart_) : 0u;
  result.basis_vector_count = basis_count;
  result.basis_reserved_bytes =
      basis_count * static_cast<std::uint64_t>(rows_) * sizeof(float);
  result.preconditioned_basis_vector_count =
      preconditioned_basis_count;
  result.preconditioned_basis_reserved_bytes =
      preconditioned_basis_count * static_cast<std::uint64_t>(rows_) *
      sizeof(float);
  result.persistent_vector_count =
      basis_count + auxiliary_vectors + preconditioned_basis_count;
  result.persistent_vector_reserved_bytes =
      result.persistent_vector_count *
      static_cast<std::uint64_t>(rows_) * sizeof(float);
  const std::uint64_t scalar_count =
      kStateWords + 4u +
      static_cast<std::uint64_t>(restart_) * multi_dot_groups_ +
      restart_ + static_cast<std::uint64_t>(restart_) * (restart_ + 1) +
      2u * restart_ + (restart_ + 1u) + restart_;
  result.persistent_scalar_count = scalar_count;
  result.persistent_scalar_reserved_bytes =
      scalar_count * sizeof(float);
  result.cublas_handle_count = cublas_handle_ ? 1 : 0;
  result.cublas_stream_bound = cublas_stream_bound_;
  result.cublas_device_pointer_mode = cublas_device_pointer_mode_;
  result.solver_state_rebuilt_each_solve = false;
  result.transient_solver_workspace_bytes = 0;
  result.transient_solver_workspace_bytes_available = true;
  result.device_to_device_bytes = device_to_device_bytes_;
  result.device_to_host_bytes = device_to_host_bytes_;
  result.host_to_device_bytes = host_to_device_bytes_;
  append_solver_operator_plan_statistics(*operator_plan_, false, result);
  return result;
}

std::unique_ptr<DeviceGMRES> make_device_gmres_solver(
    Program *program,
    LinearOperatorHandle &operator_handle,
    SparseMatrix *stored_matrix,
    LinearOperatorHandle *preconditioner,
    int max_iterations,
    int restart,
    float absolute_tolerance,
    float relative_tolerance) {
  return std::make_unique<DeviceGMRES>(
      program, operator_handle, stored_matrix, preconditioner,
      max_iterations, restart, absolute_tolerance, relative_tolerance);
}

std::unique_ptr<DeviceGMRES> make_device_fgmres_solver(
    Program *program,
    LinearOperatorHandle &operator_handle,
    std::vector<LinearOperatorHandle *> preconditioners,
    int max_iterations,
    int restart,
    float absolute_tolerance,
    float relative_tolerance) {
  TI_ERROR_IF(preconditioners.empty(),
              "Device FGMRES requires a non-empty action table.");
  return std::make_unique<DeviceGMRES>(
      program, operator_handle, nullptr, nullptr, max_iterations, restart,
      absolute_tolerance, relative_tolerance,
      std::move(preconditioners));
}

}  // namespace taichi::lang
