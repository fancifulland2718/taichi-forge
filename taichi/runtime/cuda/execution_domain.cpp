#include "taichi/runtime/cuda/execution_domain.h"

#include <atomic>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <utility>

#include "taichi/program/runtime_fault.h"
#include "taichi/rhi/cuda/cuda_context.h"

namespace taichi::lang::cuda {
namespace {
std::atomic<std::uint64_t> next_domain_identity{1};

template <typename Function>
void check_result(Function &function,
                  std::uint32_t result,
                  const char *operation) {
  if (result != CUDA_SUCCESS)
    throw BackendRuntimeError(Arch::cuda, result, operation,
                              function.get_error_message(result));
}
}  // namespace

struct ExecutionDomain::State {
  struct Pending {
    std::uint64_t sequence{0};
    RuntimeCompletion completion;
    // Retained even if event allocation/recording fails after work was queued.
    std::shared_ptr<RuntimeCompletionResources> resources;
  };

  mutable std::mutex mutex;
  std::condition_variable changed;
  const std::shared_ptr<RuntimeFaultDomain> owner;
  const std::uint64_t owner_id;
  const std::uint64_t identity;
  const Options options;
  std::shared_ptr<RuntimeCompletionCudaEventPool> events;
  std::vector<Pending> pending;
  std::size_t head{0};
  std::size_t count{0};
  std::uint64_t last_sequence{0};
  void *stream{nullptr};
  void *context{nullptr};
  int effective_priority{0};
  int least_priority{0};
  int greatest_priority{0};
  bool closing{false};
  bool closed{false};
  std::exception_ptr first_error;

  State(std::shared_ptr<RuntimeFaultDomain> owner, Options options)
      : owner(std::move(owner)),
        owner_id(this->owner->snapshot().program_domain),
        identity(next_domain_identity.fetch_add(1, std::memory_order_relaxed)),
        options(options),
        events(
            std::make_shared<RuntimeCompletionCudaEventPool>(this->owner,
                                                             options.capacity)),
        pending(options.capacity) {
    if (owner_id == 0 || identity == 0)
      throw std::invalid_argument(
          "CUDA execution domain requires a live runtime identity");
  }

  void publish(Pending &slot) {
    slot.completion = RuntimeCompletion::from_cuda_stream(
        owner_id, slot.sequence, stream, owner, events);
    if (slot.resources) {
      slot.completion.attach_resources(slot.resources);
      slot.resources.reset();
    }
  }

  void report(std::exception_ptr error, std::uint64_t sequence) noexcept {
    if (!first_error)
      first_error = error;
    try {
      std::rethrow_exception(error);
    } catch (const BackendRuntimeError &backend) {
      owner->report_backend_error(backend, sequence);
    } catch (...) {
    }
  }
};

bool ExecutionDomain::available() {
  auto &driver = CUDADriver::get_instance_without_context();
  return driver.nvidia_extensions_available() &&
         driver.stream_create_with_priority.available() &&
         driver.stream_get_priority.available() &&
         driver.context_get_stream_priority_range.available();
}

ExecutionDomain::ExecutionDomain(std::shared_ptr<RuntimeFaultDomain> owner,
                                 Options options) {
  if (!owner || owner->backend() != Arch::cuda || options.capacity == 0)
    throw std::invalid_argument(
        "CUDA execution domain requires an owner and positive capacity");
  owner->throw_if_new_submission_disallowed("create CUDA execution domain");
  if (!available())
    throw std::runtime_error(
        "CUDA execution domains are unavailable on this driver");
  state_ = std::make_unique<State>(std::move(owner), options);
  auto &state = *state_;
  auto &cuda = CUDAContext::get_instance();
  auto guard = cuda.get_guard();
  auto &driver = CUDADriver::get_instance();
  state.context = cuda.get_context();
  try {
    check_result(driver.context_get_stream_priority_range,
                 driver.context_get_stream_priority_range.call(
                     &state.least_priority, &state.greatest_priority),
                 "context_get_stream_priority_range");
    check_result(driver.stream_create_with_priority,
                 driver.stream_create_with_priority.call(
                     &state.stream, CU_STREAM_NON_BLOCKING, options.priority),
                 "stream_create_with_priority");
    check_result(driver.stream_get_priority,
                 driver.stream_get_priority.call(state.stream,
                                                 &state.effective_priority),
                 "stream_get_priority");
    // Capacity is explicit and bounded. Prepare its untimed events at domain
    // creation so a later burst does not allocate driver handles on admission.
    // Newly created, unrecorded events have no pending work; query that fact
    // before returning them to the ordinary completed-event cache.
    std::vector<void *> prepared_events;
    prepared_events.reserve(options.capacity);
    try {
      for (std::size_t i = 0; i < options.capacity; ++i) {
        auto *event = state.events->acquire();
        prepared_events.push_back(event);
        check_result(driver.event_query, driver.event_query.call(event),
                     "domain event preallocation");
      }
    } catch (...) {
      state.report(std::current_exception(), 0);
      for (auto *event : prepared_events)
        state.events->release(event, false);
      throw;
    }
    for (auto *event : prepared_events)
      state.events->release(event, true);
  } catch (...) {
    state.report(std::current_exception(), 0);
    if (state.stream && state.owner->backend_calls_safe())
      driver.stream_destroy.call(state.stream);
    throw;
  }
}

ExecutionDomain::~ExecutionDomain() {
  try {
    close();
  } catch (...) {
    // No successful drain means no proof that device work stopped accessing
    // leases. Keep this finite domain allocation quarantined until process
    // exit, just as fatal CUDA handles cannot safely be returned to a pool.
    // The owning runtime is poisoned on fatal errors and cannot create more.
    (void)state_.release();
  }
}

RuntimeCompletion ExecutionDomain::submit(
    const std::vector<RuntimeCompletion> &dependencies,
    std::shared_ptr<RuntimeCompletionResources> resources,
    const std::function<void(void *)> &enqueue) {
  if (!enqueue)
    throw std::invalid_argument(
        "CUDA submission requires an enqueue operation");
  auto &state = *state_;
  // At most one FIFO prefix entry is polled per hot admission. Since every
  // entry here is on the same stream, a pending head cannot hide a ready tail.
  collect_ready(1);
  std::lock_guard<std::mutex> lock(state.mutex);
  state.owner->throw_if_new_submission_disallowed(
      "submit CUDA execution domain");
  if (state.first_error)
    std::rethrow_exception(state.first_error);
  if (state.closing || state.closed)
    throw std::runtime_error("CUDA execution domain is closing or closed");
  for (const auto &dependency : dependencies)
    dependency.validate_cuda_dependency(state.owner_id, state.owner);
  if (state.count == state.pending.size())
    throw ExecutionDomainBusy();

  auto context = CUDAContext::get_instance().get_guard();
  const auto sequence = state.owner->allocate_submission_sequence();
  auto &slot = state.pending[(state.head + state.count) % state.pending.size()];
  slot.sequence = sequence;
  slot.resources = std::move(resources);
  state.last_sequence = sequence;
  ++state.count;  // Ownership is published before any possible device effect.
  try {
    for (const auto &dependency : dependencies)
      dependency.enqueue_cuda_wait(state.stream, state.owner_id, state.owner);
    enqueue(state.stream);
    state.publish(slot);
    return slot.completion;
  } catch (...) {
    auto error = std::current_exception();
    state.report(error, sequence);
    // Seal the partial prefix without synchronizing it. Failure of this
    // recovery retains the preallocated slot and its leases until cold close.
    if (!slot.completion.valid() && state.owner->backend_calls_safe()) {
      try {
        state.publish(slot);
      } catch (...) {
        state.report(std::current_exception(), sequence);
      }
    }
    std::rethrow_exception(error);
  }
}

std::size_t ExecutionDomain::collect_ready(std::size_t budget) {
  auto &state = *state_;
  std::size_t collected = 0;
  while (collected < budget) {
    RuntimeCompletion completion;
    std::uint64_t sequence = 0;
    {
      std::lock_guard<std::mutex> lock(state.mutex);
      if (state.closing || state.closed || state.count == 0)
        break;
      const auto &slot = state.pending[state.head];
      completion = slot.completion;
      sequence = slot.sequence;
    }
    // The recording-failure case is owned until cold stream retirement.
    try {
      if (!completion.valid() || !completion.done())
        break;
    } catch (...) {
      std::lock_guard<std::mutex> lock(state.mutex);
      state.report(std::current_exception(), sequence);
      throw;
    }
    State::Pending retired;
    {
      std::lock_guard<std::mutex> lock(state.mutex);
      if (state.closing || state.closed || state.count == 0)
        break;
      auto &slot = state.pending[state.head];
      if (slot.sequence != sequence)
        continue;  // Another collector retired this exact prefix.
      retired = std::move(slot);
      slot = {};
      state.head = (state.head + 1) % state.pending.size();
      --state.count;
      ++collected;
    }
  }
  return collected;
}

ExecutionDomain::Snapshot ExecutionDomain::snapshot() const {
  const auto &state = *state_;
  std::lock_guard<std::mutex> lock(state.mutex);
  return {state.identity,
          state.owner_id,
          reinterpret_cast<std::uintptr_t>(state.context),
          reinterpret_cast<std::uintptr_t>(state.stream),
          state.options.priority,
          state.effective_priority,
          state.least_priority,
          state.greatest_priority,
          state.options.capacity,
          state.count,
          !state.closing && !state.closed && !state.first_error &&
              state.owner->submission_allowed(),
          state.closed};
}

void ExecutionDomain::close() {
  if (!state_)
    return;
  auto &state = *state_;
  {
    std::unique_lock<std::mutex> lock(state.mutex);
    state.changed.wait(lock, [&] { return !state.closing; });
    if (state.closed)
      return;
    state.closing = true;
  }
  std::exception_ptr error;
  try {
    if (!state.owner->backend_calls_safe())
      throw std::runtime_error(
          "Cannot drain CUDA domain after a fatal backend error");
    auto context = CUDAContext::get_instance().get_guard();
    auto &driver = CUDADriver::get_instance();
    const auto result = driver.stream_synchronize.call_concurrent(state.stream);
    if (result != CUDA_SUCCESS)
      throw BackendRuntimeError(
          Arch::cuda, result, "domain stream_synchronize",
          driver.stream_synchronize.get_error_message(result));
    check_result(driver.stream_destroy,
                 driver.stream_destroy.call(state.stream), "stream_destroy");
  } catch (...) {
    error = std::current_exception();
  }
  std::vector<State::Pending> retired;
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!error) {
      retired.swap(state.pending);
      state.count = 0;
      state.closed = true;
      state.stream = nullptr;
    } else {
      state.report(error, state.last_sequence);
    }
    state.closing = false;
    state.changed.notify_all();
  }
  if (error)
    std::rethrow_exception(error);
  for (auto &slot : retired)
    slot.completion.mark_completed();
  retired.clear();
  state.events->clear();
}

}  // namespace taichi::lang::cuda
