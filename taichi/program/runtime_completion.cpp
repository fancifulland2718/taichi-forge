#include "taichi/program/runtime_completion.h"

#include <atomic>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <utility>

#include "taichi/common/logging.h"

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
class CudaEventCompletion final : public CompletionPrimitive {
 public:
  explicit CudaEventCompletion(void *stream) {
    auto context_guard = CUDAContext::get_instance().get_guard();
    auto &driver = CUDADriver::get_instance();
    driver.event_create(&event_, CU_EVENT_DISABLE_TIMING);
    try {
      driver.event_record(event_, stream);
    } catch (...) {
      driver.event_destroy.call_with_warning(event_);
      event_ = nullptr;
      throw;
    }
  }

  ~CudaEventCompletion() override {
    if (event_ == nullptr) {
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
    TI_ERROR("{}", driver.event_query.get_error_message(result));
  }

  void wait() override {
    auto context_guard = CUDAContext::get_instance().get_guard();
    CUDADriver::get_instance().event_synchronize(event_);
  }

 private:
  void *event_{nullptr};
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

  explicit State(std::unique_ptr<CompletionPrimitive> primitive)
      : status(CompletionStatus::pending), primitive(std::move(primitive)) {
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

  void record_first_error_locked(std::exception_ptr error,
                                 CompletionStatus failure_status) {
    if (!first_error) {
      first_error = std::move(error);
      first_error_text = exception_message(first_error);
    }
    status.store(failure_status, std::memory_order_release);
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
  std::unique_ptr<CompletionPrimitive> primitive;
  std::shared_ptr<RuntimeCompletionResources> resources;
  std::exception_ptr first_error;
  std::string first_error_text;
};

RuntimeCompletion::RuntimeCompletion(Arch backend,
                                     std::uint64_t program_domain,
                                     std::uint64_t sequence,
                                     std::shared_ptr<State> state) noexcept
    : backend_(backend),
      program_domain_(program_domain),
      sequence_(sequence),
      state_(std::move(state)) {
}

RuntimeCompletion RuntimeCompletion::completed(
    Arch backend,
    std::uint64_t program_domain,
    std::uint64_t sequence) noexcept {
  static std::shared_ptr<State> completed_state = std::make_shared<State>();
  return RuntimeCompletion(backend, program_domain, sequence,
                           completed_state);
}

RuntimeCompletion RuntimeCompletion::from_stream_semaphore(
    Arch backend,
    std::uint64_t program_domain,
    std::uint64_t sequence,
    StreamSemaphore semaphore) {
  if (!semaphore) {
    return completed(backend, program_domain, sequence);
  }
  auto primitive =
      std::make_unique<StreamSemaphoreCompletion>(std::move(semaphore));
  return RuntimeCompletion(backend, program_domain, sequence,
                           std::make_shared<State>(std::move(primitive)));
}

RuntimeCompletion RuntimeCompletion::from_cuda_stream(
    std::uint64_t program_domain,
    std::uint64_t sequence,
    void *stream) {
#ifdef TI_WITH_CUDA
  auto primitive = std::make_unique<CudaEventCompletion>(stream);
  return RuntimeCompletion(Arch::cuda, program_domain, sequence,
                           std::make_shared<State>(std::move(primitive)));
#else
  (void)program_domain;
  (void)sequence;
  (void)stream;
  TI_ERROR("CUDA runtime completion requested without CUDA support");
#endif
}

bool RuntimeCompletion::valid() const noexcept {
  return state_ != nullptr && program_domain_ != 0;
}

bool RuntimeCompletion::done() const {
  TI_ERROR_IF(!valid(), "Invalid runtime completion");
  return state_->done();
}

void RuntimeCompletion::wait() const {
  TI_ERROR_IF(!valid(), "Invalid runtime completion");
  state_->wait();
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

}  // namespace taichi::lang
