#include "taichi/program/runtime_completion.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>

#include "taichi/common/logging.h"
#include "taichi/program/runtime_fault.h"
#include "taichi/rhi/backend_error.h"

#ifdef TI_WITH_CUDA
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_driver.h"
#endif

namespace taichi::lang {
namespace {

enum class CompletionStatus : std::uint8_t {
  pending,
  completed,
  failed,
  invalidated,
};

class CompletionPrimitive {
 public:
  virtual ~CompletionPrimitive() = default;
  virtual bool is_ready() = 0;
  virtual void wait() = 0;
};

class StreamSemaphoreCompletion final : public CompletionPrimitive {
 public:
  explicit StreamSemaphoreCompletion(StreamSemaphore semaphore)
      : semaphore_(std::move(semaphore)) {
    TI_ASSERT(semaphore_ != nullptr);
  }

  bool is_ready() override {
    return semaphore_->is_ready();
  }

  void wait() override {
    TI_ERROR_IF(!semaphore_->wait(),
                "Backend semaphore does not expose host completion");
  }

 private:
  StreamSemaphore semaphore_;
};

#ifdef TI_WITH_CUDA
class CudaStreamGpuTimingObject final : public StreamGpuTimingObject {
 public:
  CudaStreamGpuTimingObject(
      void *stream,
      std::weak_ptr<RuntimeFaultDomain> fault_domain)
      : fault_domain_(std::move(fault_domain)) {
    auto context_guard = CUDAContext::get_instance().get_guard();
    auto &driver = CUDADriver::get_instance();
    std::uint32_t result =
        driver.event_create.call(&start_event_, CU_EVENT_DEFAULT);
    if (result != CUDA_SUCCESS) {
      throw BackendRuntimeError(
          Arch::cuda, result, "event_create",
          driver.event_create.get_error_message(result));
    }
    result = driver.event_create.call(&end_event_, CU_EVENT_DEFAULT);
    if (result != CUDA_SUCCESS) {
      driver.event_destroy.call_with_warning(start_event_);
      start_event_ = nullptr;
      throw BackendRuntimeError(
          Arch::cuda, result, "event_create",
          driver.event_create.get_error_message(result));
    }
    result = driver.event_record.call(start_event_, stream);
    if (result != CUDA_SUCCESS) {
      destroy_events(driver);
      throw BackendRuntimeError(
          Arch::cuda, result, "event_record",
          driver.event_record.get_error_message(result));
    }
  }

  ~CudaStreamGpuTimingObject() override {
    if (start_event_ == nullptr && end_event_ == nullptr) {
      return;
    }
    if (auto domain = fault_domain_.lock();
        domain && !domain->backend_calls_safe()) {
      start_event_ = nullptr;
      end_event_ = nullptr;
      return;
    }
    try {
      auto context_guard = CUDAContext::get_instance().get_guard();
      destroy_events(CUDADriver::get_instance());
    } catch (...) {
      start_event_ = nullptr;
      end_event_ = nullptr;
    }
  }

  void record_end(void *stream) {
    TI_ERROR_IF(ended_.load(std::memory_order_acquire),
                "CUDA GPU timing scope was already ended");
    auto context_guard = CUDAContext::get_instance().get_guard();
    auto &driver = CUDADriver::get_instance();
    const std::uint32_t result = driver.event_record.call(end_event_, stream);
    if (result != CUDA_SUCCESS) {
      throw BackendRuntimeError(
          Arch::cuda, result, "event_record",
          driver.event_record.get_error_message(result));
    }
    ended_.store(true, std::memory_order_release);
  }

  StreamGpuTimingSnapshot snapshot() const override {
    StreamGpuTimingSnapshot result;
    result.measurement_path_changed = true;
    // Taichi CUDA launches currently use the context's default stream.
    result.stream_id = 0;
    if (!ended_.load(std::memory_order_acquire)) {
      result.status = "not_ended";
      return result;
    }
    auto context_guard = CUDAContext::get_instance().get_guard();
    auto &driver = CUDADriver::get_instance();
    float elapsed_ms = 0.0f;
    const std::uint32_t elapsed_result = driver.event_elapsed_time.call(
        &elapsed_ms, start_event_, end_event_);
    if (elapsed_result == CUDA_ERROR_NOT_READY) {
      result.status = "not_ready";
      return result;
    }
    if (elapsed_result != CUDA_SUCCESS) {
      throw BackendRuntimeError(
          Arch::cuda, elapsed_result, "event_elapsed_time",
          driver.event_elapsed_time.get_error_message(elapsed_result));
    }
    const long double elapsed_ns =
        static_cast<long double>(elapsed_ms) * 1000000.0L;
    if (!std::isfinite(elapsed_ms) || elapsed_ns < 0.0L ||
        elapsed_ns >
            static_cast<long double>(
                (std::numeric_limits<std::uint64_t>::max)())) {
      result.status = "overflow";
      return result;
    }
    result.available = true;
    result.duration_ns = static_cast<std::uint64_t>(elapsed_ns + 0.5L);
    result.exact = true;
    result.status = "instrumented_exact";
    result.driver_owned_bytes_known = false;
    return result;
  }

 private:
  void destroy_events(CUDADriver &driver) noexcept {
    if (end_event_ != nullptr) {
      driver.event_destroy.call_with_warning(end_event_);
      end_event_ = nullptr;
    }
    if (start_event_ != nullptr) {
      driver.event_destroy.call_with_warning(start_event_);
      start_event_ = nullptr;
    }
  }

  void *start_event_{nullptr};
  void *end_event_{nullptr};
  std::weak_ptr<RuntimeFaultDomain> fault_domain_;
  std::atomic<bool> ended_{false};
};

class CudaEventCompletion final : public CompletionPrimitive {
 public:
  CudaEventCompletion(void *stream,
                      std::weak_ptr<RuntimeFaultDomain> fault_domain)
      : fault_domain_(std::move(fault_domain)) {
    auto context_guard = CUDAContext::get_instance().get_guard();
    auto &driver = CUDADriver::get_instance();
    const std::uint32_t create_result =
        driver.event_create.call(&event_, CU_EVENT_DISABLE_TIMING);
    if (create_result != CUDA_SUCCESS) {
      throw BackendRuntimeError(
          Arch::cuda, create_result, "event_create",
          driver.event_create.get_error_message(create_result));
    }
    const std::uint32_t record_result =
        driver.event_record.call(event_, stream);
    if (record_result != CUDA_SUCCESS) {
      if (classify_cuda_driver_error(record_result) !=
          BackendErrorClassification::kFatal) {
        driver.event_destroy.call_with_warning(event_);
      }
      // A fatal context error owns the unrecoverable handle until process
      // teardown; a nonfatal path destroyed it above.
      event_ = nullptr;
      throw BackendRuntimeError(
          Arch::cuda, record_result, "event_record",
          driver.event_record.get_error_message(record_result));
    }
  }

  ~CudaEventCompletion() override {
    if (event_ == nullptr) {
      return;
    }
    if (auto domain = fault_domain_.lock();
        domain && !domain->backend_calls_safe()) {
      // CUDA execution faults can make even event destruction fail. The
      // context owns the handle and will reclaim it at process teardown.
      event_ = nullptr;
      return;
    }
    try {
      auto context_guard = CUDAContext::get_instance().get_guard();
      CUDADriver::get_instance().event_destroy.call_with_warning(event_);
    } catch (...) {
      // Destructors cannot report a second backend failure. F3 records the
      // first fatal fault before Program teardown; this is only a last-resort
      // leak-safe path during process shutdown.
    }
    event_ = nullptr;
  }

  bool is_ready() override {
    auto context_guard = CUDAContext::get_instance().get_guard();
    auto &driver = CUDADriver::get_instance();
    const std::uint32_t result = driver.event_query.call(event_);
    if (result == CUDA_SUCCESS) {
      return true;
    }
    if (result == CUDA_ERROR_NOT_READY) {
      return false;
    }
    throw BackendRuntimeError(Arch::cuda, result, "event_query",
                              driver.event_query.get_error_message(result));
  }

  void wait() override {
    auto context_guard = CUDAContext::get_instance().get_guard();
    auto &driver = CUDADriver::get_instance();
    const std::uint32_t result = driver.event_synchronize.call(event_);
    if (result != CUDA_SUCCESS) {
      throw BackendRuntimeError(
          Arch::cuda, result, "event_synchronize",
          driver.event_synchronize.get_error_message(result));
    }
  }

 private:
  void *event_{nullptr};
  std::weak_ptr<RuntimeFaultDomain> fault_domain_;
};
#endif

std::string exception_message(const std::exception_ptr &error) {
  if (!error) {
    return {};
  }
  try {
    std::rethrow_exception(error);
  } catch (const std::exception &e) {
    return e.what();
  } catch (...) {
    return "Unknown runtime completion error";
  }
}

}  // namespace

struct RuntimeCompletion::State {
  State() : status(CompletionStatus::completed) {
  }

  State(std::unique_ptr<CompletionPrimitive> primitive,
        std::shared_ptr<RuntimeFaultDomain> fault_domain,
        std::uint64_t sequence,
        StreamGpuTiming gpu_timing = nullptr,
        std::vector<RuntimeGpuRegionTiming> gpu_region_timings = {})
      : status(CompletionStatus::pending),
        fault_domain(std::move(fault_domain)),
        sequence(sequence),
        primitive(std::move(primitive)),
        gpu_timing(std::move(gpu_timing)),
        gpu_region_timings(std::move(gpu_region_timings)) {
    TI_ASSERT(this->primitive != nullptr);
  }

  bool done() {
    CompletionStatus current = status.load(std::memory_order_acquire);
    if (current == CompletionStatus::completed) {
      return true;
    }
    if (current == CompletionStatus::failed ||
        current == CompletionStatus::invalidated) {
      rethrow_first_error();
    }

    std::unique_ptr<CompletionPrimitive> retired_primitive;
    std::shared_ptr<RuntimeCompletionResources> retired_resources;
    {
      std::lock_guard<std::mutex> lock(mutex);
      current = status.load(std::memory_order_relaxed);
      if (current == CompletionStatus::completed) {
        return true;
      }
      if (current == CompletionStatus::failed ||
          current == CompletionStatus::invalidated) {
        rethrow_first_error_locked();
      }
      try {
        if (!primitive->is_ready()) {
          return false;
        }
      } catch (const BackendRuntimeError &error) {
        report_backend_error_locked(error);
        record_first_error_locked(std::current_exception(),
                                  CompletionStatus::failed);
        throw;
      } catch (...) {
        record_first_error_locked(std::current_exception(),
                                  CompletionStatus::failed);
        throw;
      }
      status.store(CompletionStatus::completed, std::memory_order_release);
      retired_primitive = std::move(primitive);
      retired_resources = std::move(resources);
    }
    // Lease release and backend-handle destruction are outside the state lock.
    retired_resources.reset();
    retired_primitive.reset();
    return true;
  }

  void wait() {
    CompletionStatus current = status.load(std::memory_order_acquire);
    if (current == CompletionStatus::completed) {
      return;
    }
    if (current == CompletionStatus::failed ||
        current == CompletionStatus::invalidated) {
      rethrow_first_error();
    }

    std::unique_ptr<CompletionPrimitive> retired_primitive;
    std::shared_ptr<RuntimeCompletionResources> retired_resources;
    {
      std::lock_guard<std::mutex> lock(mutex);
      current = status.load(std::memory_order_relaxed);
      if (current == CompletionStatus::completed) {
        return;
      }
      if (current == CompletionStatus::failed ||
          current == CompletionStatus::invalidated) {
        rethrow_first_error_locked();
      }
      try {
        primitive->wait();
      } catch (const BackendRuntimeError &error) {
        report_backend_error_locked(error);
        record_first_error_locked(std::current_exception(),
                                  CompletionStatus::failed);
        throw;
      } catch (...) {
        record_first_error_locked(std::current_exception(),
                                  CompletionStatus::failed);
        throw;
      }
      status.store(CompletionStatus::completed, std::memory_order_release);
      retired_primitive = std::move(primitive);
      retired_resources = std::move(resources);
    }
    retired_resources.reset();
    retired_primitive.reset();
  }

  void attach(std::shared_ptr<RuntimeCompletionResources> value) {
    if (!value) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex);
    TI_ERROR_IF(status.load(std::memory_order_relaxed) !=
                    CompletionStatus::pending,
                "Cannot attach resources to a completed submission");
    TI_ERROR_IF(resources != nullptr,
                "Runtime completion resources were already attached");
    resources = std::move(value);
  }

  void mark_completed() noexcept {
    std::unique_ptr<CompletionPrimitive> retired_primitive;
    std::shared_ptr<RuntimeCompletionResources> retired_resources;
    {
      std::lock_guard<std::mutex> lock(mutex);
      const auto current = status.load(std::memory_order_relaxed);
      // A successful Program-wide synchronize proves that the backend no
      // longer dereferences resources retained by this completion, even when
      // an earlier query/wait already made the completion fault sticky. Keep
      // that first error observable, but do not retain its primitive and
      // resource batch until the last ticket reference is destroyed.
      if (current != CompletionStatus::failed &&
          current != CompletionStatus::invalidated) {
        status.store(CompletionStatus::completed, std::memory_order_release);
      }
      retired_primitive = std::move(primitive);
      retired_resources = std::move(resources);
    }
    retired_resources.reset();
    retired_primitive.reset();
  }

  void invalidate(const std::string &reason) noexcept {
    std::lock_guard<std::mutex> lock(mutex);
    if (status.load(std::memory_order_relaxed) ==
        CompletionStatus::completed) {
      return;
    }
    if (!first_error) {
      try {
        throw std::runtime_error(reason);
      } catch (...) {
        first_error = std::current_exception();
      }
      first_error_text = reason;
    }
    status.store(CompletionStatus::invalidated, std::memory_order_release);
  }

  std::size_t retained_count(std::uint32_t kind) const noexcept {
    std::lock_guard<std::mutex> lock(mutex);
    return resources ? resources->retained_resource_count(kind) : 0;
  }

  std::string error_message() const {
    std::lock_guard<std::mutex> lock(mutex);
    return first_error_text;
  }

  bool has_backend_work() const noexcept {
    return status.load(std::memory_order_acquire) == CompletionStatus::pending;
  }

  StreamGpuTimingSnapshot timing_snapshot() const {
    StreamGpuTiming timing;
    CompletionStatus current;
    {
      std::lock_guard<std::mutex> lock(mutex);
      current = status.load(std::memory_order_relaxed);
      timing = gpu_timing;
    }
    if (!timing) {
      return {};
    }
    if (current == CompletionStatus::pending) {
      StreamGpuTimingSnapshot result;
      result.status = "pending";
      result.measurement_path_changed = true;
      return result;
    }
    if (current == CompletionStatus::failed ||
        current == CompletionStatus::invalidated) {
      StreamGpuTimingSnapshot result;
      result.status = "failed";
      result.measurement_path_changed = true;
      return result;
    }
    return timing->snapshot();
  }

  std::vector<RuntimeGpuRegionTimingSnapshot> region_timing_snapshots() const {
    std::vector<RuntimeGpuRegionTiming> timings;
    CompletionStatus current;
    {
      std::lock_guard<std::mutex> lock(mutex);
      current = status.load(std::memory_order_relaxed);
      timings = gpu_region_timings;
    }
    std::vector<RuntimeGpuRegionTimingSnapshot> results;
    results.reserve(timings.size());
    for (const auto &region : timings) {
      StreamGpuTimingSnapshot snapshot;
      if (!region.timing) {
        snapshot.status = "unsupported";
        snapshot.measurement_path_changed = true;
      } else if (current == CompletionStatus::pending) {
        snapshot.status = "pending";
        snapshot.measurement_path_changed = true;
      } else if (current == CompletionStatus::failed ||
                 current == CompletionStatus::invalidated) {
        snapshot.status = "failed";
        snapshot.measurement_path_changed = true;
      } else {
        snapshot = region.timing->snapshot();
      }
      results.push_back({region.path_id, std::move(snapshot)});
    }
    return results;
  }

  void record_first_error_locked(std::exception_ptr error,
                                 CompletionStatus failure_status) {
    if (!first_error) {
      first_error = std::move(error);
      first_error_text = exception_message(first_error);
    }
    status.store(failure_status, std::memory_order_release);
  }

  void report_backend_error_locked(
      const BackendRuntimeError &error) noexcept {
    if (fault_domain) {
      fault_domain->report_backend_error(error, sequence);
    }
  }

  [[noreturn]] void rethrow_first_error_locked() const {
    if (first_error) {
      std::rethrow_exception(first_error);
    }
    throw std::runtime_error("Runtime completion failed without an error");
  }

  [[noreturn]] void rethrow_first_error() const {
    std::lock_guard<std::mutex> lock(mutex);
    rethrow_first_error_locked();
  }

  std::atomic<CompletionStatus> status;
  mutable std::mutex mutex;
  std::shared_ptr<RuntimeFaultDomain> fault_domain;
  std::uint64_t sequence{0};
  std::unique_ptr<CompletionPrimitive> primitive;
  StreamGpuTiming gpu_timing;
  std::vector<RuntimeGpuRegionTiming> gpu_region_timings;
  std::shared_ptr<RuntimeCompletionResources> resources;
  std::exception_ptr first_error;
  std::string first_error_text;
};

RuntimeCompletion::RuntimeCompletion(Arch backend,
                                     std::uint64_t program_domain,
                                     std::uint64_t sequence,
                                     std::shared_ptr<State> state,
                                     std::shared_ptr<RuntimeFaultDomain>
                                         fault_domain) noexcept
    : backend_(backend),
      program_domain_(program_domain),
      sequence_(sequence),
      state_(std::move(state)),
      fault_domain_(std::move(fault_domain)) {
}

RuntimeCompletion RuntimeCompletion::completed(
    Arch backend,
    std::uint64_t program_domain,
    std::uint64_t sequence,
    std::shared_ptr<RuntimeFaultDomain> fault_domain) noexcept {
  static std::shared_ptr<State> completed_state = std::make_shared<State>();
  return RuntimeCompletion(backend, program_domain, sequence,
                           completed_state, std::move(fault_domain));
}

RuntimeCompletion RuntimeCompletion::from_stream_semaphore(
    Arch backend,
    std::uint64_t program_domain,
    std::uint64_t sequence,
    StreamSemaphore semaphore,
    std::shared_ptr<RuntimeFaultDomain> fault_domain,
    StreamGpuTiming gpu_timing,
    std::vector<RuntimeGpuRegionTiming> gpu_region_timings) {
  if (!semaphore) {
    return completed(backend, program_domain, sequence,
                     std::move(fault_domain));
  }
  auto primitive =
      std::make_unique<StreamSemaphoreCompletion>(std::move(semaphore));
  auto state = std::make_shared<State>(std::move(primitive), fault_domain,
                                       sequence, std::move(gpu_timing),
                                       std::move(gpu_region_timings));
  return RuntimeCompletion(backend, program_domain, sequence,
                           std::move(state), std::move(fault_domain));
}

RuntimeCompletion RuntimeCompletion::from_cuda_stream(
    std::uint64_t program_domain,
    std::uint64_t sequence,
    void *stream,
    std::shared_ptr<RuntimeFaultDomain> fault_domain,
    StreamGpuTiming gpu_timing,
    std::vector<RuntimeGpuRegionTiming> gpu_region_timings) {
#ifdef TI_WITH_CUDA
  try {
    auto primitive =
        std::make_unique<CudaEventCompletion>(stream, fault_domain);
    auto state = std::make_shared<State>(std::move(primitive), fault_domain,
                                         sequence, std::move(gpu_timing),
                                         std::move(gpu_region_timings));
    return RuntimeCompletion(
        Arch::cuda, program_domain, sequence, std::move(state),
        std::move(fault_domain));
  } catch (const BackendRuntimeError &error) {
    if (fault_domain) {
      fault_domain->report_backend_error(error, sequence);
    }
    throw;
  }
#else
  (void)program_domain;
  (void)sequence;
  (void)stream;
  (void)fault_domain;
  (void)gpu_timing;
  (void)gpu_region_timings;
  TI_ERROR("CUDA runtime completion requested without CUDA support");
#endif
}

StreamGpuTiming RuntimeCompletion::begin_cuda_gpu_timing(
    void *stream,
    std::shared_ptr<RuntimeFaultDomain> fault_domain) {
#ifdef TI_WITH_CUDA
  try {
    return std::make_shared<CudaStreamGpuTimingObject>(stream, fault_domain);
  } catch (const BackendRuntimeError &error) {
    if (fault_domain) {
      fault_domain->report_backend_error(error, 0);
    }
    throw;
  }
#else
  (void)stream;
  (void)fault_domain;
  return nullptr;
#endif
}

void RuntimeCompletion::end_cuda_gpu_timing(
    const StreamGpuTiming &timing,
    void *stream,
    std::shared_ptr<RuntimeFaultDomain> fault_domain) {
#ifdef TI_WITH_CUDA
  if (!timing) {
    return;
  }
  auto cuda_timing =
      std::dynamic_pointer_cast<CudaStreamGpuTimingObject>(timing);
  TI_ERROR_IF(!cuda_timing,
              "CUDA stream received a timing object from another backend");
  try {
    cuda_timing->record_end(stream);
  } catch (const BackendRuntimeError &error) {
    if (fault_domain) {
      fault_domain->report_backend_error(error, 0);
    }
    throw;
  }
#else
  (void)timing;
  (void)stream;
  (void)fault_domain;
#endif
}

bool RuntimeCompletion::valid() const noexcept {
  return state_ != nullptr && program_domain_ != 0;
}

bool RuntimeCompletion::done() const {
  TI_ERROR_IF(!valid(), "Invalid runtime completion");
  if (fault_domain_) {
    fault_domain_->statistics().record_completion_poll();
  }
  return state_->done();
}

void RuntimeCompletion::wait() const {
  TI_ERROR_IF(!valid(), "Invalid runtime completion");
  if (!fault_domain_) {
    state_->wait();
    return;
  }
  const auto started = std::chrono::steady_clock::now();
  try {
    state_->wait();
  } catch (...) {
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started);
    fault_domain_->statistics().record_completion_wait(
        static_cast<std::uint64_t>(elapsed.count()));
    throw;
  }
  const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now() - started);
  fault_domain_->statistics().record_completion_wait(
      static_cast<std::uint64_t>(elapsed.count()));
}

bool RuntimeCompletion::has_backend_work() const noexcept {
  return state_ && state_->has_backend_work();
}

std::size_t RuntimeCompletion::retained_resource_count(
    std::uint32_t kind) const noexcept {
  return state_ ? state_->retained_count(kind) : 0;
}

std::string RuntimeCompletion::first_error_message() const {
  return state_ ? state_->error_message() : std::string{};
}

StreamGpuTimingSnapshot RuntimeCompletion::gpu_timing_snapshot() const {
  return state_ ? state_->timing_snapshot() : StreamGpuTimingSnapshot{};
}

std::vector<RuntimeGpuRegionTimingSnapshot>
RuntimeCompletion::gpu_region_timing_snapshots() const {
  return state_ ? state_->region_timing_snapshots()
                : std::vector<RuntimeGpuRegionTimingSnapshot>{};
}

void RuntimeCompletion::attach_resources(
    std::shared_ptr<RuntimeCompletionResources> resources) const {
  TI_ERROR_IF(!valid(), "Invalid runtime completion");
  state_->attach(std::move(resources));
}

void RuntimeCompletion::mark_completed() const noexcept {
  if (state_) {
    state_->mark_completed();
  }
}

void RuntimeCompletion::invalidate(const std::string &reason) const noexcept {
  if (state_) {
    state_->invalidate(reason);
  }
}

void RuntimeCompletion::invalidate_and_release(
    const std::string &reason) const noexcept {
  if (state_) {
    state_->invalidate(reason);
    state_->mark_completed();
  }
}

}  // namespace taichi::lang
