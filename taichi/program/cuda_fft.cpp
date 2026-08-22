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
constexpr int kCufftForward = -1;
constexpr int kCufftInverse = 1;

std::size_t cufft_scalar_count(std::size_t length,
                               std::size_t batch_count) {
  TI_ERROR_IF(length == 0 ||
                  length > static_cast<std::size_t>(
                               (std::numeric_limits<int>::max)()),
              "CUDA cuFFT length must be in [1, INT_MAX].");
  TI_ERROR_IF(batch_count == 0 ||
                  batch_count > static_cast<std::size_t>(
                                    (std::numeric_limits<int>::max)()),
              "CUDA cuFFT batch_count must be in [1, INT_MAX].");
  TI_ERROR_IF(length >
                  (std::numeric_limits<std::size_t>::max)() / batch_count / 2,
              "CUDA cuFFT element count overflow.");
  return length * batch_count * 2;
}

}  // namespace

class CudaFftPlan {
 public:
  CudaFftPlan(std::size_t length, std::size_t batch_count)
      : length_(length), batch_count_(batch_count) {
    cufft_scalar_count(length_, batch_count_);
    auto &driver = CUFFTDriver::get_instance();
    TI_ERROR_IF(!driver.load_cufft(),
                "CUDA cuFFT could not load a compatible shared library and "
                "the required basic-plan symbols.");
    int handle = 0;
    const auto plan_status =
        driver.plan_1d.call(&handle, static_cast<int>(length_), kCufftC2C,
                            static_cast<int>(batch_count_));
    TI_ERROR_IF(plan_status != 0 || handle == 0,
                "CUDA cuFFT failed to create a 1D C2C plan (status {}).",
                plan_status);
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

  void execute(void *input, void *output, int direction) {
    std::lock_guard<std::mutex> lock(mutex_);
    TI_ERROR_IF(handle_ == 0, "CUDA cuFFT plan is closed.");
    TI_ERROR_IF(direction != kCufftForward && direction != kCufftInverse,
                "CUDA cuFFT direction must be -1 (forward) or 1 (inverse).");
    const auto status =
        CUFFTDriver::get_instance().exec_c2c.call(handle_, input, output,
                                                  direction);
    TI_ERROR_IF(status != 0, "CUDA cuFFT C2C execution failed (status {}).",
                status);
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
  int handle_{0};
  std::mutex mutex_;
};

std::uint64_t Program::create_cuda_cufft_plan_1d(std::size_t length,
                                                 std::size_t batch_count) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA cuFFT plans require the CUDA backend.");
  cufft_scalar_count(length, batch_count);
  TI_ERROR_IF(!CUDADriver::get_instance_without_context()
                   .nvidia_extensions_available(),
              "CUDA cuFFT requires the NVIDIA CUDA provider.");
  auto cuda_submission_guard =
      CUDAContext::get_instance().get_submission_lock_guard();
  auto context_guard = CUDAContext::get_instance().get_guard();
  auto plan = std::make_shared<CudaFftPlan>(length, batch_count);

  std::lock_guard<std::mutex> lock(cuda_cufft_plan_mutex_);
  TI_ERROR_IF(next_cuda_cufft_plan_handle_ == 0,
              "CUDA cuFFT plan handle space exhausted.");
  const auto handle = next_cuda_cufft_plan_handle_++;
  cuda_cufft_plans_.emplace(handle, std::move(plan));
  return handle;
}

std::size_t Program::cuda_cufft_execute_c2c(std::uint64_t handle,
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
  const std::size_t expected =
      cufft_scalar_count(plan->length(), plan->batch_count());
  const auto validate = [expected](const char *name, Ndarray *array) {
    TI_ERROR_IF(!array->get_element_shape().empty() ||
                    array->get_element_data_type() != PrimitiveType::f32 ||
                    array->get_nelement() != expected ||
                    array->get_element_size() != sizeof(float32),
                "CUDA cuFFT {} must be a compact scalar f32 ndarray with "
                "exactly batch_count * length * 2 entries.",
                name);
  };
  validate("input", input);
  validate("output", output);
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
                                                 std::size_t) {
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
