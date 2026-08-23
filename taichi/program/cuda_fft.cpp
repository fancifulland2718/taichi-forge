#include "taichi/program/program.h"

#include <limits>
#include <memory>
#include <utility>
#include <vector>

#if defined(TI_WITH_CUDA)
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_driver.h"

namespace taichi::lang {
namespace {

constexpr int kCufftC2C = 0x29;
constexpr int kCufftR2C = 0x2a;
constexpr int kCufftC2R = 0x2c;
constexpr int kCufftForward = -1;
constexpr int kCufftInverse = 1;

enum class CufftTransformKind : int {
  c2c = 0,
  r2c = 1,
  c2r = 2,
};

struct CufftScalarCounts {
  std::size_t input{0};
  std::size_t output{0};
};

CufftTransformKind validate_transform_kind(int transform_kind) {
  TI_ERROR_IF(transform_kind < static_cast<int>(CufftTransformKind::c2c) ||
                  transform_kind > static_cast<int>(CufftTransformKind::c2r),
              "CUDA cuFFT transform_kind must be C2C, R2C, or C2R.");
  return static_cast<CufftTransformKind>(transform_kind);
}

CufftScalarCounts cufft_scalar_counts(std::size_t length,
                                      std::size_t batch_count,
                                      CufftTransformKind transform_kind) {
  TI_ERROR_IF(length == 0 ||
                  length > static_cast<std::size_t>(
                               (std::numeric_limits<int>::max)()),
              "CUDA cuFFT length must be in [1, INT_MAX].");
  TI_ERROR_IF(batch_count == 0 ||
                  batch_count > static_cast<std::size_t>(
                                    (std::numeric_limits<int>::max)()),
              "CUDA cuFFT batch_count must be in [1, INT_MAX].");
  const auto max_size = (std::numeric_limits<std::size_t>::max)();
  TI_ERROR_IF(length > max_size / batch_count,
              "CUDA cuFFT element count overflow.");
  const auto real_scalars = length * batch_count;
  const auto complex_length = length / 2 + 1;
  TI_ERROR_IF(complex_length > max_size / batch_count / 2,
              "CUDA cuFFT complex element count overflow.");
  const auto hermitian_scalars = complex_length * batch_count * 2;
  TI_ERROR_IF(length > max_size / batch_count / 2,
              "CUDA cuFFT complex element count overflow.");
  const auto complex_scalars = length * batch_count * 2;
  switch (transform_kind) {
    case CufftTransformKind::c2c:
      return {complex_scalars, complex_scalars};
    case CufftTransformKind::r2c:
      return {real_scalars, hermitian_scalars};
    case CufftTransformKind::c2r:
      return {hermitian_scalars, real_scalars};
  }
  TI_UNREACHABLE;
}

}  // namespace

class CudaFftPlan {
 public:
  CudaFftPlan(std::size_t length,
              std::size_t batch_count,
              CufftTransformKind transform_kind)
      : length_(length),
        batch_count_(batch_count),
        transform_kind_(transform_kind) {
    cufft_scalar_counts(length_, batch_count_, transform_kind_);
    auto &driver = CUFFTDriver::get_instance();
    TI_ERROR_IF(!driver.load_cufft(),
                "CUDA cuFFT could not load a compatible shared library and "
                "the required basic-plan symbols.");
    int handle = 0;
    const int transform_type =
        transform_kind_ == CufftTransformKind::c2c
            ? kCufftC2C
            : transform_kind_ == CufftTransformKind::r2c ? kCufftR2C
                                                         : kCufftC2R;
    const auto plan_status = driver.plan_1d.call(
        &handle, static_cast<int>(length_), transform_type,
        static_cast<int>(batch_count_));
    TI_ERROR_IF(plan_status != 0 || handle == 0,
                "CUDA cuFFT failed to create a 1D plan (status {}).", plan_status);
    handle_ = handle;
    const auto stream_status = driver.set_stream.call(handle_, nullptr);
    if (stream_status != 0) {
      const auto destroy_status = driver.destroy.call(handle_);
      handle_ = 0;
      TI_WARN_IF(destroy_status != 0,
                 "CUDA cuFFT cleanup after stream-binding failure returned "
                 "status {}.",
                 destroy_status);
      TI_ERROR("CUDA cuFFT failed to bind the runtime default stream "
               "(status {}).",
               stream_status);
    }
  }

  ~CudaFftPlan() {
    // Program owns the synchronization and provider-call decision. A nonzero
    // handle here is abandoned only after a fatal backend fault or an
    // exceptional construction path where calling into the provider is unsafe.
    handle_ = 0;
  }

  CudaFftPlan(const CudaFftPlan &) = delete;
  CudaFftPlan &operator=(const CudaFftPlan &) = delete;

  std::size_t length() const {
    return length_;
  }

  std::size_t batch_count() const {
    return batch_count_;
  }

  CufftTransformKind transform_kind() const {
    return transform_kind_;
  }

  void execute(void *input, void *output, int direction) {
    std::lock_guard<std::mutex> lock(mutex_);
    TI_ERROR_IF(handle_ == 0, "CUDA cuFFT plan is closed.");
    auto &driver = CUFFTDriver::get_instance();
    int status = 0;
    if (transform_kind_ == CufftTransformKind::c2c) {
      TI_ERROR_IF(direction != kCufftForward && direction != kCufftInverse,
                  "CUDA cuFFT C2C direction must be -1 (forward) or 1 "
                  "(inverse).");
      status = driver.exec_c2c.call(handle_, input, output, direction);
    } else if (transform_kind_ == CufftTransformKind::r2c) {
      TI_ERROR_IF(direction != kCufftForward,
                  "CUDA cuFFT R2C requires the forward direction.");
      status = driver.exec_r2c.call(handle_, input, output);
    } else {
      TI_ERROR_IF(direction != kCufftInverse,
                  "CUDA cuFFT C2R requires the inverse direction.");
      status = driver.exec_c2r.call(handle_, input, output);
    }
    TI_ERROR_IF(status != 0, "CUDA cuFFT execution failed (status {}).", status);
  }

  void destroy(bool provider_calls_safe) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (handle_ == 0) {
      return;
    }
    if (provider_calls_safe && CUFFTDriver::get_instance().is_loaded()) {
      const auto status = CUFFTDriver::get_instance().destroy.call(handle_);
      TI_WARN_IF(status != 0,
                 "CUDA cuFFT plan destruction returned status {}.", status);
    }
    handle_ = 0;
  }

 private:
  std::size_t length_{0};
  std::size_t batch_count_{0};
  CufftTransformKind transform_kind_{CufftTransformKind::c2c};
  int handle_{0};
  std::mutex mutex_;
};

std::uint64_t Program::create_cuda_cufft_plan_1d(std::size_t length,
                                                 std::size_t batch_count,
                                                 int transform_kind) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA cuFFT plans require the CUDA backend.");
  const auto validated_transform = validate_transform_kind(transform_kind);
  cufft_scalar_counts(length, batch_count, validated_transform);
  TI_ERROR_IF(!CUDADriver::get_instance_without_context()
                   .nvidia_extensions_available(),
              "CUDA cuFFT requires the NVIDIA CUDA provider.");
  auto cuda_submission_guard =
      CUDAContext::get_instance().get_submission_lock_guard();
  auto context_guard = CUDAContext::get_instance().get_guard();
  auto plan =
      std::make_shared<CudaFftPlan>(length, batch_count, validated_transform);

  std::lock_guard<std::mutex> lock(cuda_cufft_plan_mutex_);
  TI_ERROR_IF(next_cuda_cufft_plan_handle_ == 0,
              "CUDA cuFFT plan handle space exhausted.");
  const auto handle = next_cuda_cufft_plan_handle_++;
  cuda_cufft_plans_.emplace(handle, std::move(plan));
  return handle;
}

std::size_t Program::cuda_cufft_execute(std::uint64_t handle,
                                       Ndarray *input,
                                       Ndarray *output,
                                       int direction) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA cuFFT execution requires the CUDA backend.");
  TI_ERROR_IF(!input || !output,
              "CUDA cuFFT execution received a null ndarray.");

  std::shared_ptr<CudaFftPlan> plan;
  {
    std::lock_guard<std::mutex> lock(cuda_cufft_plan_mutex_);
    const auto found = cuda_cufft_plans_.find(handle);
    TI_ERROR_IF(found == cuda_cufft_plans_.end(),
                "CUDA cuFFT plan handle is stale or closed.");
    plan = found->second;
  }
  const auto expected = cufft_scalar_counts(
      plan->length(), plan->batch_count(), plan->transform_kind());
  const auto validate = [](const char *name,
                           Ndarray *array,
                           std::size_t expected_scalars) {
    TI_ERROR_IF(!array->get_element_shape().empty() ||
                    array->get_element_data_type() != PrimitiveType::f32 ||
                    array->get_nelement() != expected_scalars ||
                    array->get_element_size() != sizeof(float32),
                "CUDA cuFFT {} must be a compact scalar f32 ndarray with "
                "exactly the plan-declared scalar count.", name);
  };
  validate("input", input, expected.input);
  validate("output", output, expected.output);
  TI_ERROR_IF(input->owning_program() != this ||
                  output->owning_program() != this,
              "CUDA cuFFT arrays must belong to the active runtime.");
  TI_ERROR_IF(input->get_device_allocation() ==
                  output->get_device_allocation(),
              "The first CUDA cuFFT slice requires distinct input and output "
              "allocations.");

  auto cuda_submission_guard =
      CUDAContext::get_instance().get_submission_lock_guard();
  auto context_guard = CUDAContext::get_instance().get_guard();
  auto *input_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(input));
  auto *output_ptr =
      reinterpret_cast<void *>(get_ndarray_data_ptr_as_int(output));
  TI_ERROR_IF(!input_ptr || !output_ptr,
              "CUDA cuFFT received a null device pointer.");
  plan->execute(input_ptr, output_ptr, direction);
  mark_runtime_submission_pending();
  return 0;
}

std::size_t Program::cuda_cufft_execute_c2c(std::uint64_t handle,
                                            Ndarray *input,
                                            Ndarray *output,
                                            int direction) {
  {
    std::lock_guard<std::mutex> lock(cuda_cufft_plan_mutex_);
    const auto found = cuda_cufft_plans_.find(handle);
    TI_ERROR_IF(found == cuda_cufft_plans_.end(),
                "CUDA cuFFT plan handle is stale or closed.");
    TI_ERROR_IF(found->second->transform_kind() != CufftTransformKind::c2c,
                "CUDA cuFFT legacy C2C execution requires a C2C plan.");
  }
  return cuda_cufft_execute(handle, input, output, direction);
}

void Program::destroy_cuda_cufft_plan(std::uint64_t handle) {
  std::shared_ptr<CudaFftPlan> plan;
  {
    auto submission_guard = acquire_runtime_resource_submission_guard();
    std::lock_guard<std::mutex> lock(cuda_cufft_plan_mutex_);
    const auto found = cuda_cufft_plans_.find(handle);
    if (found == cuda_cufft_plans_.end()) {
      return;
    }
    plan = found->second;
    cuda_cufft_plans_.erase(found);
  }
  if (!runtime_has_fatal_fault()) {
    synchronize();
    auto cuda_submission_guard =
        CUDAContext::get_instance().get_submission_lock_guard();
    auto context_guard = CUDAContext::get_instance().get_guard();
    plan->destroy(true);
  } else {
    plan->destroy(false);
  }
}

void Program::cuda_clear_cufft_plans() {
  std::vector<std::shared_ptr<CudaFftPlan>> plans;
  {
    std::lock_guard<std::mutex> lock(cuda_cufft_plan_mutex_);
    plans.reserve(cuda_cufft_plans_.size());
    for (auto &[handle, plan] : cuda_cufft_plans_) {
      plans.push_back(std::move(plan));
    }
    cuda_cufft_plans_.clear();
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

std::uint64_t Program::create_cuda_cufft_plan_1d(std::size_t,
                                                 std::size_t,
                                                 int) {
  TI_ERROR("CUDA cuFFT requires TI_WITH_CUDA=ON.");
}

std::size_t Program::cuda_cufft_execute(std::uint64_t,
                                       Ndarray *,
                                       Ndarray *,
                                       int) {
  TI_ERROR("CUDA cuFFT requires TI_WITH_CUDA=ON.");
}

std::size_t Program::cuda_cufft_execute_c2c(std::uint64_t,
                                            Ndarray *,
                                            Ndarray *,
                                            int) {
  TI_ERROR("CUDA cuFFT requires TI_WITH_CUDA=ON.");
}

void Program::destroy_cuda_cufft_plan(std::uint64_t) {
}

void Program::cuda_clear_cufft_plans() {
}

}  // namespace taichi::lang

#endif
