#include "taichi/program/sparse_device_minres.h"

#include "taichi/program/linear_operator.h"
#include "taichi/program/sparse_preconditioner.h"
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

constexpr int kStateWords = 33;
constexpr int kStateBytes = kStateWords * sizeof(std::uint32_t);
constexpr int kOne = 0;
constexpr int kTrueResidualSquared = 1;
constexpr int kInverseBeta = 11;
constexpr int kOldResidualScale = 12;
constexpr int kAlphaResidualScale = 13;
constexpr int kEstimatedResidual = 18;
constexpr int kRelativeReferenceNorm = 21;
constexpr int kEffectiveTolerance = 22;
constexpr int kInitialResidualSquared = 24;
constexpr int kStatus = 25;
constexpr int kCompletedIterations = 26;
constexpr int kActive = 27;

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
void push_minres_resource(VulkanCommandReplayKey &key,
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

void push_minres_stamp(VulkanCommandReplayKey &key,
                       const OperatorResourceStamp &stamp) {
  key.push(stamp.program_generation);
  key.push(stamp.schema_revision);
  key.push(stamp.topology_revision);
  key.push(stamp.binding_revision);
}
#endif

}  // namespace

#if defined(TI_WITH_CUDA)

struct DeviceMINRESCudaReplayState {
  struct Key {
    RuntimeResourceHandle solution;
    RuntimeResourceHandle rhs;
    OperatorResourceStamp operator_stamp;
    OperatorResourceStamp preconditioner_stamp;
    std::array<std::uintptr_t, 13> resources{};
    std::uintptr_t provider{0};
    std::uintptr_t preconditioner{0};
    int rows{0};
    int iterations{0};
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
             preconditioner_stamp.schema_revision ==
                 other.preconditioner_stamp.schema_revision &&
             preconditioner_stamp.topology_revision ==
                 other.preconditioner_stamp.topology_revision &&
             preconditioner_stamp.binding_revision ==
                 other.preconditioner_stamp.binding_revision &&
             resources == other.resources && provider == other.provider &&
             preconditioner == other.preconditioner && rows == other.rows &&
             iterations == other.iterations &&
             limit_reached == other.limit_reached;
    }
  };

  struct Entry {
    CUgraphExec executable{nullptr};
    Key key;
    bool key_valid{false};
    std::uint64_t operator_numeric_revision{0};
    std::uint64_t preconditioner_numeric_revision{0};
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
  std::array<Entry, 18> entries;
  bool disabled{false};
  std::string unavailable_reason{"not_built"};

  ~DeviceMINRESCudaReplayState() {
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

struct DeviceMINRESCudaReplayState {};

#endif

#if defined(TI_WITH_VULKAN)

struct DeviceMINRESVulkanReplayState {
  struct Slot {
    VulkanCommandReplayCache cache;
    RuntimeResourceHandle solution;
    RuntimeResourceHandle rhs;
    std::optional<Program::NdarrayResourceLease> solution_lease;
    std::optional<Program::NdarrayResourceLease> rhs_lease;
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
};

#else

struct DeviceMINRESVulkanReplayState {};

#endif

DeviceMINRES::DeviceMINRES(
    Program *program,
    LinearOperatorHandle &operator_handle,
    SparseMatrix *stored_matrix,
    SparseJacobiPreconditionerPlan *jacobi,
    SparseBlockJacobiPreconditionerPlan *block_jacobi,
    LinearOperatorHandle *preconditioner,
    int max_iterations,
    float absolute_tolerance,
    float relative_tolerance)
    : program_(program),
      operator_handle_(&operator_handle),
      stored_matrix_(stored_matrix),
      jacobi_(jacobi),
      block_jacobi_(block_jacobi),
      operator_preconditioner_(preconditioner),
      max_iterations_(max_iterations),
      absolute_tolerance_(absolute_tolerance),
      relative_tolerance_(relative_tolerance) {
  TI_ERROR_IF(!program_ || operator_handle.program() != program_ ||
                  !is_cuda_or_vulkan(program_->compile_config().arch),
              "Device MINRES requires a CUDA or Vulkan operator in its "
              "owning Program.");
  const auto &descriptor = operator_handle.descriptor();
  TI_ERROR_IF(descriptor.domain != descriptor.range ||
                  descriptor.domain.scalar_type != PrimitiveType::f32 ||
                  descriptor.domain.scalar_extent == 0 ||
                  descriptor.domain.scalar_extent >
                      static_cast<std::size_t>(
                          std::numeric_limits<int>::max()),
              "Device MINRES requires a non-empty square scalar f32 "
              "operator with int-sized extent.");
  rows_ = static_cast<int>(descriptor.domain.scalar_extent);
  const int preconditioner_count = (jacobi_ ? 1 : 0) +
                                   (block_jacobi_ ? 1 : 0) +
                                   (operator_preconditioner_ ? 1 : 0);
  TI_ERROR_IF(preconditioner_count > 1,
              "Device MINRES accepts at most one preconditioner.");
  if (stored_matrix_) {
    TI_ERROR_IF(stored_matrix_->num_rows() != rows_ ||
                    stored_matrix_->num_cols() != rows_ ||
                    stored_matrix_->get_data_type() != PrimitiveType::f32,
                "Device MINRES stored provider does not match its operator "
                "descriptor.");
  }
  TI_ERROR_IF((jacobi_ || block_jacobi_) && !stored_matrix_,
              "Built-in MINRES preconditioners require their stored "
              "operator provider.");
  validate_controls();
  operator_plan_ =
      std::make_unique<OperatorPlan>(program_, operator_handle.binding());
  validate_operator_solver_compatibility(
      operator_plan_->descriptor(), operator_plan_->mathematical_traits(),
      OperatorSolverFamily::minres,
      PreconditionerBehavior::fixed_linear);
  if (jacobi_) {
    preconditioner_plan_ = make_solver_preconditioner_plan(
        program_, *operator_plan_, *stored_matrix_, *jacobi_, "jacobi");
  } else if (block_jacobi_) {
    preconditioner_plan_ = make_solver_preconditioner_plan(
        program_, *operator_plan_, *stored_matrix_, *block_jacobi_,
        "block_jacobi");
  } else if (operator_preconditioner_) {
    preconditioner_plan_ = make_solver_preconditioner_plan(
        program_, *operator_plan_, *operator_preconditioner_,
        "linear_operator");
  }
  try {
    allocate_workspace();
    if (program_->compile_config().arch == Arch::cuda) {
      initialize_cuda();
    } else {
      // Match SolvePlan's public Vulkan default. Fixed-budget execution still
      // records bounded eight-iteration command chunks, but observes state
      // only after the complete budget.
      execution_policy_ = SparseSolveExecutionPolicy::fixed_budget_masked;
      host_check_interval_ = max_iterations_;
    }
  } catch (...) {
    release_cuda();
    release_workspace();
    throw;
  }
}

DeviceMINRES::~DeviceMINRES() {
  cuda_replay_.reset();
  vulkan_replay_.reset();
  release_cuda();
  release_workspace();
}

void DeviceMINRES::validate_controls() const {
  TI_ERROR_IF(max_iterations_ < 0 || !std::isfinite(absolute_tolerance_) ||
                  !std::isfinite(relative_tolerance_) ||
                  absolute_tolerance_ < 0.0f ||
                  relative_tolerance_ < 0.0f ||
                  (absolute_tolerance_ == 0.0f &&
                   relative_tolerance_ == 0.0f),
              "Device MINRES requires non-negative iterations and finite "
              "non-negative atol/rtol with at least one positive "
              "tolerance.");
}

void DeviceMINRES::allocate_workspace() {
  auto vector = [&] {
    return program_->create_ndarray(PrimitiveType::f32, {rows_},
                                    ExternalArrayLayout::kNull, false);
  };
  auto scalar = [&] {
    return program_->create_ndarray(PrimitiveType::f32, {1},
                                    ExternalArrayLayout::kNull, false);
  };
  try {
    av_ = vector();
    v_ = vector();
    y_ = vector();
    r1_ = vector();
    r2_ = vector();
    w_older_ = vector();
    w_old_ = vector();
    w_ = vector();
    true_residual_ = vector();
    initial_residual_squared_ = scalar();
    rhs_squared_ = scalar();
    dot_ = scalar();
    state_ = program_->create_ndarray(PrimitiveType::i32, {kStateWords},
                                      ExternalArrayLayout::kNull, false);
  } catch (...) {
    release_workspace();
    throw;
  }
}

void DeviceMINRES::release_workspace() {
  auto release = [&](Ndarray *&array) {
    if (array && program_) {
      program_->delete_ndarray(array);
    }
    array = nullptr;
  };
  release(state_);
  release(dot_);
  release(rhs_squared_);
  release(initial_residual_squared_);
  release(true_residual_);
  release(w_);
  release(w_old_);
  release(w_older_);
  release(r2_);
  release(r1_);
  release(y_);
  release(v_);
  release(av_);
}

void DeviceMINRES::initialize_cuda() {
#if defined(TI_WITH_CUDA)
  auto &cublas = CUBLASDriver::get_instance();
  TI_ERROR_IF(!cublas.is_loaded() && !cublas.load_cublas(),
              "Device MINRES failed to load cuBLAS.");
  cublasHandle_t handle = nullptr;
  cublas.cubCreate(&handle);
  TI_ERROR_IF(!handle, "Device MINRES failed to create a cuBLAS handle.");
  cublas_handle_ = handle;
  solver_stream_ = nullptr;
  cublas.cubSetStream(handle, nullptr);
  CUstream observed = reinterpret_cast<CUstream>(1);
  cublas.cubGetStream(handle, &observed);
  TI_ERROR_IF(observed != nullptr,
              "Device MINRES could not bind cuBLAS to the solver stream.");
  cublas_stream_bound_ = true;
  cublas.cubSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
  cublasPointerMode_t mode = CUBLAS_POINTER_MODE_HOST;
  cublas.cubGetPointerMode(handle, &mode);
  TI_ERROR_IF(mode != CUBLAS_POINTER_MODE_DEVICE,
              "Device MINRES could not enable cuBLAS device scalar mode.");
  cublas_device_pointer_mode_ = true;
#else
  TI_NOT_IMPLEMENTED;
#endif
}

void DeviceMINRES::release_cuda() {
#if defined(TI_WITH_CUDA)
  if (cublas_handle_) {
    CUBLASDriver::get_instance().cubDestroy(
        static_cast<cublasHandle_t>(cublas_handle_));
  }
#endif
  cublas_handle_ = nullptr;
}

void DeviceMINRES::configure_execution_policy(
    SparseSolveExecutionPolicy policy,
    int host_check_interval) {
  std::lock_guard<std::mutex> lock(solve_mutex_);
  TI_ERROR_IF(solve_calls_ != 0,
              "Device MINRES execution policy must be configured before "
              "solve.");
  const Arch arch = program_->compile_config().arch;
  validate_sparse_solve_execution_policy(arch, policy, host_check_interval);
  if (arch == Arch::cuda) {
    TI_ERROR_IF(policy != SparseSolveExecutionPolicy::host_each_iteration &&
                    policy !=
                        SparseSolveExecutionPolicy::host_check_every_k,
                "CUDA MINRES supports host_each_iteration or "
                "host_check_every_k.");
  } else {
    TI_ERROR_IF(policy != SparseSolveExecutionPolicy::host_check_every_k &&
                    policy != SparseSolveExecutionPolicy::fixed_budget_masked,
                "Vulkan MINRES supports host_check_every_k or "
                "fixed_budget_masked.");
  }
  TI_ERROR_IF(policy == SparseSolveExecutionPolicy::host_check_every_k &&
                  host_check_interval != 4 && host_check_interval != 8,
              "Device MINRES host_check_every_k supports K=4 or K=8.");
  execution_policy_ = policy;
  host_check_interval_ =
      policy == SparseSolveExecutionPolicy::host_each_iteration
          ? 1
          : (policy == SparseSolveExecutionPolicy::host_check_every_k
                 ? host_check_interval
                 : max_iterations_);
}

bool DeviceMINRES::has_preconditioner() const {
  return preconditioner_plan_ != nullptr;
}

bool DeviceMINRES::native_stored_provider() const {
  if (!stored_matrix_ || operator_preconditioner_) {
    return false;
  }
  const Arch arch = program_->compile_config().arch;
  if (arch == Arch::cuda) {
    return (dynamic_cast<CuSparseMatrix *>(stored_matrix_) ||
            dynamic_cast<CuSparseBsrMatrix *>(stored_matrix_)) &&
           (!has_preconditioner() || jacobi_ || block_jacobi_);
  }
  if (arch == Arch::vulkan) {
    return (dynamic_cast<VulkanSparseMatrix *>(stored_matrix_) ||
            dynamic_cast<VulkanSparseBsrMatrix *>(stored_matrix_)) &&
           (!has_preconditioner() || jacobi_ || block_jacobi_);
  }
  return false;
}

std::uintptr_t DeviceMINRES::address(const Ndarray *array) const {
  return program_->get_ndarray_data_ptr_as_int(array);
}

void DeviceMINRES::apply_operator(
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
       OperatorVectorView::from_ndarray(program_, input, descriptor.domain,
                                        false),
       nullptr,
       OperatorVectorView::from_ndarray(program_, output, descriptor.range,
                                        true)});
}

void DeviceMINRES::apply_preconditioner(
    const OperatorPinnedAction &generation,
    const Ndarray &input,
    const Ndarray &output,
    void *stream,
    bool native_capture) {
  TI_ERROR_IF(!preconditioner_plan_,
              "Device MINRES preconditioner plan is unavailable.");
#if defined(TI_WITH_CUDA)
  if (native_capture && program_->compile_config().arch == Arch::cuda) {
    if (jacobi_) {
      jacobi_->apply_cuda_raw(program_, address(&input), address(&output),
                              static_cast<CUstream>(stream));
      return;
    }
    if (block_jacobi_) {
      block_jacobi_->apply_cuda_raw(
          program_, address(&input), address(&output),
          static_cast<CUstream>(stream));
      return;
    }
  }
#endif
  auto &action = preconditioner_plan_->action();
  const auto &descriptor = action.descriptor();
  action.submit(
      generation,
      {OperatorApplyMode::forward,
       OperatorVectorView::from_ndarray(program_, input, descriptor.domain,
                                        false),
       nullptr,
       OperatorVectorView::from_ndarray(program_, output, descriptor.range,
                                        true)});
}

void DeviceMINRES::backend_dot(const Ndarray &left,
                               const Ndarray &right,
                               const Ndarray &output,
                               void *stream) {
  if (program_->compile_config().arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    auto &cublas = CUBLASDriver::get_instance();
    cublas.cubSdot(static_cast<cublasHandle_t>(cublas_handle_), rows_,
                   reinterpret_cast<const float *>(address(&left)), 1,
                   reinterpret_cast<const float *>(address(&right)), 1,
                   reinterpret_cast<float *>(address(&output)));
#else
    TI_NOT_IMPLEMENTED;
#endif
    return;
  }
  program_->vulkan_sparse_dot(const_cast<Ndarray *>(&left),
                              const_cast<Ndarray *>(&right),
                              const_cast<Ndarray *>(&output), rows_);
}

void DeviceMINRES::backend_scalar_stage(int stage,
                                        bool limit_reached,
                                        bool stop_on_estimate,
                                        void *stream) {
  if (program_->compile_config().arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    cuda::driver_sparse_minres_scalar_f32(
        reinterpret_cast<void *>(address(initial_residual_squared_)),
        reinterpret_cast<void *>(address(rhs_squared_)),
        reinterpret_cast<void *>(address(dot_)),
        reinterpret_cast<void *>(address(state_)), absolute_tolerance_,
        relative_tolerance_, stage, limit_reached, has_preconditioner(),
        stop_on_estimate, stream);
#else
    TI_NOT_IMPLEMENTED;
#endif
    return;
  }
  program_->vulkan_sparse_minres_scalar(
      initial_residual_squared_, rhs_squared_, dot_, state_,
      absolute_tolerance_, relative_tolerance_,
      static_cast<std::uint32_t>(stage), limit_reached,
      has_preconditioner(), stop_on_estimate);
}

void DeviceMINRES::backend_vector_state(const Ndarray &source,
                                        const Ndarray &destination,
                                        int coefficient,
                                        bool add,
                                        void *stream) {
  if (program_->compile_config().arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    cuda::driver_sparse_minres_vector_state_f32(
        reinterpret_cast<void *>(address(&source)),
        reinterpret_cast<void *>(address(&destination)),
        reinterpret_cast<void *>(address(state_)), rows_, coefficient, add,
        stream);
#else
    TI_NOT_IMPLEMENTED;
#endif
    return;
  }
  program_->vulkan_sparse_minres_vector_state(
      const_cast<Ndarray *>(&source), const_cast<Ndarray *>(&destination),
      state_, rows_, static_cast<std::uint32_t>(coefficient), add);
}

void DeviceMINRES::backend_commit(const Ndarray &x, void *stream) {
  if (program_->compile_config().arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    cuda::driver_sparse_minres_commit_f32(
        reinterpret_cast<void *>(address(v_)),
        reinterpret_cast<void *>(address(r1_)),
        reinterpret_cast<void *>(address(r2_)),
        reinterpret_cast<void *>(address(av_)),
        reinterpret_cast<void *>(address(w_older_)),
        reinterpret_cast<void *>(address(w_old_)),
        reinterpret_cast<void *>(address(w_)),
        reinterpret_cast<void *>(address(&x)),
        reinterpret_cast<void *>(address(state_)), rows_, stream);
#else
    TI_NOT_IMPLEMENTED;
#endif
    return;
  }
  program_->vulkan_sparse_minres_commit(
      v_, r1_, r2_, av_, w_older_, w_old_, w_,
      const_cast<Ndarray *>(&x), state_, rows_);
}

void DeviceMINRES::backend_true_residual(
    const OperatorPinnedAction &generation,
    const Ndarray &x,
    const Ndarray &b,
    void *stream,
    bool native_capture) {
  apply_operator(generation, x, *av_, stream, native_capture);
  if (program_->compile_config().arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    cuda::driver_zero_strided(
        reinterpret_cast<void *>(address(true_residual_)), rows_,
        cuda::CudaTransformValueType::f32, 0, sizeof(float), stream);
    cuda::driver_add_scaled_strided(
        reinterpret_cast<void *>(address(&b)),
        reinterpret_cast<void *>(address(true_residual_)), rows_,
        cuda::CudaTransformValueType::f32, 0, sizeof(float), 0,
        sizeof(float), 1.0, stream);
    cuda::driver_add_scaled_strided(
        reinterpret_cast<void *>(address(av_)),
        reinterpret_cast<void *>(address(true_residual_)), rows_,
        cuda::CudaTransformValueType::f32, 0, sizeof(float), 0,
        sizeof(float), -1.0, stream);
#else
    TI_NOT_IMPLEMENTED;
#endif
  } else {
    program_->copy_ndarray_fast(true_residual_,
                                const_cast<Ndarray *>(&b));
    program_->vulkan_sparse_axpy(av_, true_residual_, rows_, -1.0f);
  }
  backend_dot(*true_residual_, *true_residual_, *dot_, stream);
}

void DeviceMINRES::issue_iteration(
    const OperatorPinnedAction &operator_generation,
    const OperatorPinnedAction &preconditioner_generation,
    const Ndarray &x,
    void *stream,
    bool native_capture) {
  backend_scalar_stage(1, false, false, stream);
  backend_vector_state(*y_, *v_, kInverseBeta, false, stream);
  apply_operator(operator_generation, *v_, *av_, stream, native_capture);
  backend_vector_state(*r1_, *av_, kOldResidualScale, true, stream);
  backend_dot(*v_, *av_, *dot_, stream);
  backend_scalar_stage(2, false, false, stream);
  backend_vector_state(*r2_, *av_, kAlphaResidualScale, true, stream);
  if (has_preconditioner()) {
    apply_preconditioner(preconditioner_generation, *av_, *y_, stream,
                         native_capture);
  } else {
    backend_vector_state(*av_, *y_, kOne, false, stream);
  }
  backend_dot(*av_, *y_, *dot_, stream);
  backend_scalar_stage(3, false, false, stream);
  backend_commit(x, stream);
}

void DeviceMINRES::issue_chunk(
    const OperatorPinnedAction &operator_generation,
    const OperatorPinnedAction &preconditioner_generation,
    const Ndarray &x,
    const Ndarray &b,
    int chunk_iterations,
    bool limit_reached,
    void *stream,
    bool native_capture) {
  for (int i = 0; i < chunk_iterations; ++i) {
    issue_iteration(operator_generation, preconditioner_generation, x,
                    stream, native_capture);
  }
  backend_true_residual(operator_generation, x, b, stream, native_capture);
  backend_scalar_stage(4, limit_reached, false, stream);
}

bool DeviceMINRES::try_submit_cuda_chunk(
    const OperatorPinnedAction &operator_generation,
    const OperatorPinnedAction &preconditioner_generation,
    const Ndarray &x,
    const Ndarray &b,
    int chunk_iterations,
    bool limit_reached) {
#if defined(TI_WITH_CUDA)
  if (program_->compile_config().arch != Arch::cuda ||
      !native_stored_provider() || chunk_iterations <= 0 ||
      chunk_iterations > 8 ||
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
    cuda_replay_ = std::make_unique<DeviceMINRESCudaReplayState>();
  }
  if (cuda_replay_->disabled) {
    return false;
  }
  DeviceMINRESCudaReplayState::Key key;
  key.solution = x.runtime_resource_handle();
  key.rhs = b.runtime_resource_handle();
  key.operator_stamp = operator_generation.resource_stamp();
  key.preconditioner_stamp = preconditioner_generation
                                 ? preconditioner_generation.resource_stamp()
                                 : OperatorResourceStamp{};
  const Ndarray *arrays[] = {av_, v_, y_, r1_, r2_, w_older_, w_old_,
                             w_, true_residual_, initial_residual_squared_,
                             rhs_squared_, dot_, state_};
  for (std::size_t i = 0; i < std::size(arrays); ++i) {
    key.resources[i] = address(arrays[i]);
  }
  key.provider = reinterpret_cast<std::uintptr_t>(stored_matrix_);
  key.preconditioner = reinterpret_cast<std::uintptr_t>(
      jacobi_ ? static_cast<void *>(jacobi_)
              : static_cast<void *>(block_jacobi_));
  key.rows = rows_;
  key.iterations = chunk_iterations;
  key.limit_reached = limit_reached;
  const std::size_t index = static_cast<std::size_t>(chunk_iterations) +
                            (limit_reached ? 9u : 0u);
  auto &entry = cuda_replay_->entries[index];
  if (entry.executable && entry.key_valid && entry.key == key) {
    if (entry.operator_numeric_revision !=
            key.operator_stamp.numeric_revision ||
        entry.preconditioner_numeric_revision !=
            key.preconditioner_stamp.numeric_revision) {
      ++solver_chunk_rebinds_;
      entry.operator_numeric_revision = key.operator_stamp.numeric_revision;
      entry.preconditioner_numeric_revision =
          key.preconditioner_stamp.numeric_revision;
    }
    driver.graph_launch(entry.executable, nullptr);
    if (jacobi_) {
      jacobi_->record_replayed_apply_calls(chunk_iterations);
    } else if (block_jacobi_) {
      block_jacobi_->record_replayed_apply_calls(chunk_iterations);
    }
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
    csr->spmv(address(v_), address(av_), nullptr);
  } else {
    bsr->spmv(address(v_), address(av_), nullptr);
  }
  const CUstream capture_stream = cuda_replay_->ensure_capture_stream();
  driver.stream_synchronize(capture_stream);
  auto &cublas = CUBLASDriver::get_instance();
  cublas.cubSetStream(static_cast<cublasHandle_t>(cublas_handle_),
                      capture_stream);
  CUgraph graph = nullptr;
  auto capture_lock = CUDAContext::get_instance().get_graph_capture_lock_guard();
  if (driver.stream_begin_capture.call(
          capture_stream, CU_STREAM_CAPTURE_MODE_RELAXED) != CUDA_SUCCESS) {
    cublas.cubSetStream(static_cast<cublasHandle_t>(cublas_handle_), nullptr);
    cuda_replay_->disabled = true;
    cuda_replay_->unavailable_reason = "stream_capture_begin_failed";
    return false;
  }
  try {
    issue_chunk(operator_generation, preconditioner_generation, x, b,
                chunk_iterations, limit_reached, capture_stream, true);
  } catch (...) {
    (void)driver.stream_end_capture.call(capture_stream, &graph);
    if (graph) {
      driver.graph_destroy.call(graph);
    }
    cublas.cubSetStream(static_cast<cublasHandle_t>(cublas_handle_), nullptr);
    throw;
  }
  const auto end_error =
      driver.stream_end_capture.call(capture_stream, &graph);
  cublas.cubSetStream(static_cast<cublasHandle_t>(cublas_handle_), nullptr);
  if (end_error != CUDA_SUCCESS || !graph) {
    if (graph) {
      driver.graph_destroy.call(graph);
    }
    cuda_replay_->disabled = true;
    cuda_replay_->unavailable_reason = "stream_capture_end_failed";
    return false;
  }
  CUgraphExec executable = nullptr;
  const auto instantiate = driver.graph_instantiate_with_flags.call(
      &executable, graph, 0);
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
  entry.preconditioner_numeric_revision =
      key.preconditioner_stamp.numeric_revision;
  entry.solution_lease.emplace(
      program_->acquire_ndarray_external_lease(key.solution));
  entry.rhs_lease.emplace(program_->acquire_ndarray_external_lease(key.rhs));
  ++solver_chunk_builds_;
  cuda_replay_->unavailable_reason = "none";
  driver.graph_launch(entry.executable, nullptr);
  return true;
#else
  return false;
#endif
}

bool DeviceMINRES::try_submit_vulkan_chunk(
    const OperatorPinnedAction &operator_generation,
    const OperatorPinnedAction &preconditioner_generation,
    const Ndarray &x,
    const Ndarray &b,
    int chunk_iterations,
    bool limit_reached,
    std::size_t slot_index) {
#if defined(TI_WITH_VULKAN)
  if (program_->compile_config().arch != Arch::vulkan ||
      !native_stored_provider() || program_->profiler != nullptr ||
      get_environ_config("TI_VULKAN_SOLVER_CHUNK_REPLAY", 1) == 0) {
    return false;
  }
  if (!vulkan_replay_) {
    vulkan_replay_ = std::make_unique<DeviceMINRESVulkanReplayState>();
  }
  auto &slot = vulkan_replay_->slot(slot_index);
  VulkanCommandReplayKey key;
  key.push(220);
  key.push(program_->runtime_program_generation());
  key.push(program_->vulkan_sparse_algebra_replay_generation());
  key.push(static_cast<std::uint64_t>(chunk_iterations));
  key.push(limit_reached ? 1 : 0);
  push_minres_stamp(key, operator_generation.resource_stamp());
  push_minres_stamp(key, preconditioner_generation
                              ? preconditioner_generation.resource_stamp()
                              : OperatorResourceStamp{});
  const Ndarray *resources[] = {
      &x, &b, av_, v_, y_, r1_, r2_, w_older_, w_old_, w_,
      true_residual_, initial_residual_squared_, rhs_squared_, dot_, state_};
  for (const auto *resource : resources) {
    push_minres_resource(key, resource);
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
  const auto pc_stamp = preconditioner_generation
                            ? preconditioner_generation.resource_stamp()
                            : OperatorResourceStamp{};
  if (slot.operator_numeric_revision != 0 &&
      (slot.operator_numeric_revision != operator_stamp.numeric_revision ||
       slot.preconditioner_numeric_revision != pc_stamp.numeric_revision)) {
    ++solver_chunk_rebinds_;
  }
  slot.operator_numeric_revision = operator_stamp.numeric_revision;
  slot.preconditioner_numeric_revision = pc_stamp.numeric_revision;
  const bool replaces =
      slot.cache.entry.cmdlist && slot.cache.entry.key != key;
  (void)program_->flush_if_pending();
  auto record = [&](Device *device, CommandList *cmdlist) {
    VulkanNativeCommandRecordingScope scope(program_, device, cmdlist);
    issue_chunk(operator_generation, preconditioner_generation, x, b,
                chunk_iterations, limit_reached, nullptr, true);
  };
  const bool submitted = slot.cache.submit_or_record(
      program_, program_->get_compute_device(), key, false, record);
  if (!submitted) {
    vulkan_replay_->unavailable_reason = "native_command_replay_fallback";
    return false;
  }
  if (slot.cache.last_path == VulkanCommandReplayCache::LastPath::record) {
    ++solver_chunk_builds_;
    if (replaces) {
      ++solver_chunk_invalidations_;
    }
  } else {
    if (jacobi_) {
      jacobi_->record_replayed_apply_calls(chunk_iterations);
    } else if (block_jacobi_) {
      block_jacobi_->record_replayed_apply_calls(chunk_iterations);
    }
    ++solver_chunk_reuses_;
    ++solver_chunk_replays_;
  }
  vulkan_replay_->unavailable_reason = "none";
  return true;
#else
  return false;
#endif
}

void DeviceMINRES::read_state(bool synchronize) {
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

void DeviceMINRES::solve(Program *program,
                         const Ndarray &x,
                         const Ndarray &b) {
  TI_ERROR_IF(program != program_,
              "Device MINRES solve must use its construction Program.");
  auto check_vector = [&](const char *role, const Ndarray &array) {
    TI_ERROR_IF(array.shape.size() != 1 ||
                    array.get_element_data_type() != PrimitiveType::f32 ||
                    !array.get_element_shape().empty() ||
                    array.get_nelement() != static_cast<std::size_t>(rows_),
                "Device MINRES {} must contain exactly {} scalar f32 "
                "entries.",
                role, rows_);
  };
  check_vector("solution", x);
  check_vector("right-hand side", b);
  TI_ERROR_IF(x.get_device_allocation() == b.get_device_allocation(),
              "Device MINRES solution and RHS must not alias.");

  std::lock_guard<std::mutex> lock(solve_mutex_);
  auto operator_generation = operator_plan_->pin();
  auto preconditioner_generation =
      preconditioner_plan_
          ? preconditioner_plan_->update_and_pin(operator_generation)
          : OperatorPinnedAction{};
  auto submission_guard =
      program_->acquire_runtime_resource_submission_guard();
  const Ndarray *resources[] = {
      &x, &b, av_, v_, y_, r1_, r2_, w_older_, w_old_, w_,
      true_residual_, initial_residual_squared_, rhs_squared_, dot_, state_};
  program_->retain_ndarrays_for_external_submission(resources,
                                                     std::size(resources));
  const auto stamp = operator_generation.resource_stamp();
  if (solve_calls_ > 0) {
    ++workspace_reuses_;
  }
  ++solve_calls_;
  last_solve_pattern_version_ = stamp.topology_revision;
  last_solve_numeric_version_ = stamp.numeric_revision;
  status_ = SparseSolveStatus::kNotRun;
  iterations_ = 0;
  initial_residual_norm_ = 0.0;
  residual_norm_ = 0.0;
  estimated_residual_norm_ = 0.0;
  relative_reference_norm_ = 0.0;
  effective_tolerance_ = absolute_tolerance_;

  for (Ndarray *array : {v_, w_older_, w_old_, w_}) {
    program_->fill_ndarray_fast_u32(array, 0);
  }
  backend_true_residual(operator_generation, x, b, nullptr, false);
  program_->copy_ndarray_fast(initial_residual_squared_, dot_);
  backend_dot(b, b, *rhs_squared_, nullptr);
  program_->copy_ndarray_fast(r1_, true_residual_);
  program_->copy_ndarray_fast(r2_, true_residual_);
  if (has_preconditioner()) {
    apply_preconditioner(preconditioner_generation, *r2_, *y_, nullptr,
                         false);
  } else {
    program_->copy_ndarray_fast(y_, r2_);
  }
  backend_dot(*r2_, *y_, *dot_, nullptr);
  const bool host_observed =
      execution_policy_ != SparseSolveExecutionPolicy::fixed_budget_masked;
  const bool stop_on_estimate =
      execution_policy_ == SparseSolveExecutionPolicy::host_each_iteration;
  backend_scalar_stage(0, max_iterations_ == 0, stop_on_estimate, nullptr);
  bool terminal = false;
  if (host_observed) {
    read_state(true);
    terminal = state_int(host_state_, kActive) == 0;
  }

  int executed = 0;
  std::size_t slot_index = 0;
  while (!terminal && executed < max_iterations_) {
    const int remaining = max_iterations_ - executed;
    const int chunk =
        execution_policy_ == SparseSolveExecutionPolicy::fixed_budget_masked
            ? std::min(8, remaining)
            : std::min(host_check_interval_, remaining);
    const bool limit_reached = executed + chunk >= max_iterations_;
    bool submitted = try_submit_cuda_chunk(
        operator_generation, preconditioner_generation, x, b, chunk,
        limit_reached);
    if (!submitted) {
      submitted = try_submit_vulkan_chunk(
          operator_generation, preconditioner_generation, x, b, chunk,
          limit_reached, slot_index);
    }
    if (!submitted) {
      ++solver_chunk_direct_submissions_;
      issue_chunk(operator_generation, preconditioner_generation, x, b,
                  chunk, limit_reached, nullptr, false);
    }
    executed += chunk;
    ++slot_index;
    if (host_observed) {
      read_state(true);
      terminal = state_int(host_state_, kActive) == 0;
    }
  }
  if (!host_observed || !terminal) {
    read_state(true);
  }

  status_ = static_cast<SparseSolveStatus>(
      state_int(host_state_, kStatus));
  iterations_ = state_int(host_state_, kCompletedIterations);
  const float initial_rr = state_float(host_state_, kInitialResidualSquared);
  const float true_rr = state_float(host_state_, kTrueResidualSquared);
  initial_residual_norm_ =
      std::isfinite(initial_rr) && initial_rr >= 0.0f
          ? std::sqrt(static_cast<double>(initial_rr))
          : std::numeric_limits<double>::quiet_NaN();
  residual_norm_ = std::isfinite(true_rr) && true_rr >= 0.0f
                       ? std::sqrt(static_cast<double>(true_rr))
                       : std::numeric_limits<double>::quiet_NaN();
  estimated_residual_norm_ = state_float(host_state_, kEstimatedResidual);
  relative_reference_norm_ =
      state_float(host_state_, kRelativeReferenceNorm);
  effective_tolerance_ = state_float(host_state_, kEffectiveTolerance);
  if (status_ == SparseSolveStatus::kConverged &&
      (!std::isfinite(residual_norm_) ||
       residual_norm_ > effective_tolerance_)) {
    status_ = SparseSolveStatus::kBreakdown;
  }
  total_iterations_ += static_cast<std::uint64_t>(iterations_);
  executed_iterations_ += static_cast<std::uint64_t>(executed);
  operator_apply_calls_ +=
      1u + static_cast<std::uint64_t>(executed + slot_index);
  if (has_preconditioner()) {
    preconditioner_apply_calls_ +=
        1u + static_cast<std::uint64_t>(executed);
  }
  device_scalar_operations_ +=
      4u + static_cast<std::uint64_t>(5 * executed + 2 * slot_index);
  const std::uint64_t vector_bytes =
      static_cast<std::uint64_t>(rows_) * sizeof(float);
  device_to_device_bytes_ += sizeof(float) + 2u * vector_bytes;
  if (!has_preconditioner()) {
    device_to_device_bytes_ += vector_bytes;
  }
  if (program_->compile_config().arch == Arch::vulkan) {
    device_to_device_bytes_ += (1u + slot_index) * vector_bytes;
  }
}

SparseSolveResult DeviceMINRES::get_last_result() const {
  return {status_, iterations_, initial_residual_norm_, residual_norm_,
          absolute_tolerance_, relative_tolerance_,
          relative_reference_norm_, effective_tolerance_};
}

SparseSolvePlanRuntimeStatistics
DeviceMINRES::debug_runtime_statistics() const {
  std::lock_guard<std::mutex> lock(solve_mutex_);
  SparseSolvePlanRuntimeStatistics result;
  result.backend_family = arch_name(program_->compile_config().arch);
  result.method = "minres";
  result.dtype = "f32";
  result.rows = rows_;
  result.cols = rows_;
  result.max_iterations = max_iterations_;
  result.absolute_tolerance = absolute_tolerance_;
  result.relative_tolerance = relative_tolerance_;
  result.last_relative_reference_norm = relative_reference_norm_;
  result.last_effective_tolerance = effective_tolerance_;
  const auto stamp = operator_plan_->resource_stamp();
  result.operator_pattern_version = stamp.topology_revision;
  result.operator_numeric_version = stamp.numeric_revision;
  result.last_solve_pattern_version = last_solve_pattern_version_;
  result.last_solve_numeric_version = last_solve_numeric_version_;
  result.operator_pattern_changed_since_last_solve =
      solve_calls_ > 0 && stamp.topology_revision !=
                              last_solve_pattern_version_;
  result.operator_numeric_changed_since_last_solve =
      solve_calls_ > 0 && stamp.numeric_revision !=
                              last_solve_numeric_version_;
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
  result.device_scalar_operations = device_scalar_operations_;
  result.host_scalar_readbacks = host_scalar_readbacks_;
  result.host_synchronizations = host_synchronizations_;
  result.logical_iterations = total_iterations_;
  result.executed_iterations = executed_iterations_;
  result.wasted_iterations = executed_iterations_ - total_iterations_;
  result.solver_chunk_builds = solver_chunk_builds_;
  result.solver_chunk_reuses = solver_chunk_reuses_;
  result.solver_chunk_direct_submissions =
      solver_chunk_direct_submissions_;
  result.solver_chunk_replays = solver_chunk_replays_;
  result.solver_chunk_rebinds = solver_chunk_rebinds_;
  result.solver_chunk_invalidations = solver_chunk_invalidations_;
  result.requested_solver_execution_policy =
      sparse_solve_execution_policy_name(execution_policy_);
  result.solver_execution_policy = result.requested_solver_execution_policy;
  result.host_check_interval = host_check_interval_;
  result.solver_graph_enabled = solver_chunk_builds_ > 0;
  if (!native_stored_provider()) {
    result.solver_replay_unavailable_reason =
        "provider_not_capture_composable";
  } else if (program_->compile_config().arch == Arch::cuda) {
    if (get_environ_config("TI_CUDA_SOLVER_CHUNK_REPLAY", 1) == 0) {
      result.solver_replay_unavailable_reason = "disabled_by_environment";
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
  } else if (get_environ_config("TI_VULKAN_SOLVER_CHUNK_REPLAY", 1) == 0) {
    result.solver_replay_unavailable_reason = "disabled_by_environment";
  } else {
    result.solver_replay_unavailable_reason =
        vulkan_replay_ ? vulkan_replay_->unavailable_reason : "not_built";
  }
  result.solver_scalar_location = "device";
  result.solver_stream_policy = "backend_default";
  result.bounded_masked_execution =
      execution_policy_ == SparseSolveExecutionPolicy::fixed_budget_masked;
  result.preconditioner_method =
      preconditioner_plan_ ? preconditioner_plan_->method() : "identity";
  result.preconditioner_behavior =
      preconditioner_plan_ ? "fixed_linear" : "identity";
  result.preconditioner_apply_calls = preconditioner_apply_calls_;
  result.preconditioner_apply_calls_available = true;
  result.external_preconditioner = operator_preconditioner_ != nullptr;
  result.preconditioner_ownership_scope =
      preconditioner_plan_ ? "solve_plan" : "none";
  if (preconditioner_plan_) {
    const auto &action = preconditioner_plan_->action();
    result.preconditioner_action_provider = action.provider_name();
    result.preconditioner_asynchronous_submit =
        action.capabilities().asynchronous_submit;
    const auto action_stats = action.debug_runtime_statistics();
    result.preconditioner_generation_pins = action_stats.generation_pins;
    result.preconditioner_generation_changes = action_stats.generation_changes;
    result.preconditioner_numeric_generation_changes =
        action_stats.numeric_generation_changes;
    result.preconditioner_binding_generation_changes =
        action_stats.binding_generation_changes;
    result.preconditioner_plan_invalidations = action_stats.invalidations;
    const auto lifecycle = preconditioner_plan_->debug_runtime_statistics();
    result.preconditioner_setup_calls = lifecycle.setup_calls;
    result.preconditioner_update_calls = lifecycle.update_calls;
    result.preconditioner_update_successes = lifecycle.update_successes;
    result.preconditioner_update_noops = lifecycle.update_noops;
    result.preconditioner_update_failures = lifecycle.update_failures;
  }
  result.persistent_vector_count = 9;
  result.persistent_vector_reserved_bytes =
      9ull * static_cast<std::uint64_t>(rows_) * sizeof(float);
  result.persistent_scalar_count = kStateWords + 3;
  result.persistent_scalar_reserved_bytes = kStateBytes + 3 * sizeof(float);
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

std::unique_ptr<DeviceMINRES> make_device_minres_solver(
    Program *program,
    LinearOperatorHandle &operator_handle,
    SparseMatrix *stored_matrix,
    SparseJacobiPreconditionerPlan *jacobi,
    SparseBlockJacobiPreconditionerPlan *block_jacobi,
    LinearOperatorHandle *preconditioner,
    int max_iterations,
    float absolute_tolerance,
    float relative_tolerance) {
  return std::make_unique<DeviceMINRES>(
      program, operator_handle, stored_matrix, jacobi, block_jacobi,
      preconditioner, max_iterations, absolute_tolerance,
      relative_tolerance);
}

}  // namespace taichi::lang
