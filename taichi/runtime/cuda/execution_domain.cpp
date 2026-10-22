#include "taichi/runtime/cuda/execution_domain.h"

#include <atomic>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <thread>
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
  std::vector<StreamGpuTiming> timings;
  std::vector<std::shared_ptr<RuntimeClient>> instances;
  std::size_t preparing_instances{0};
  std::size_t head{0};
  std::size_t count{0};
  std::uint64_t last_sequence{0};
  void *stream{nullptr};
  void *context{nullptr};
  int effective_priority{0};
  int least_priority{0};
  int greatest_priority{0};
  bool accepting{true};
  bool closing{false};
  std::thread::id closing_thread;
  bool closed{false};
  std::exception_ptr first_error;
  std::exception_ptr instance_retirement_error;

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
    instances.reserve(options.instance_capacity);
  }

  void publish(Pending &slot, StreamGpuTiming timing = nullptr) {
    slot.completion = RuntimeCompletion::from_cuda_stream(
        owner_id, slot.sequence, stream, owner, events, std::move(timing));
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
  if (!owner || owner->backend() != Arch::cuda || options.capacity == 0 ||
      options.instance_capacity == 0)
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
    if (options.gpu_timing) {
      state.timings.reserve(options.capacity);
      for (std::size_t i = 0; i < options.capacity; ++i)
        state.timings.push_back(
            RuntimeCompletion::prepare_cuda_gpu_timing(state.owner));
    }
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
    const std::function<void(void *)> &enqueue,
    const std::function<void()> &admit,
    const std::function<void(void *)> &before_timing,
    const std::function<void(void *)> &after_timing,
    bool sample_gpu) {
  if (!enqueue)
    throw std::invalid_argument(
        "CUDA submission requires an enqueue operation");
  CUDAContext::check_runtime_health();
  auto &state = *state_;
  if (sample_gpu && !state.options.gpu_timing)
    throw std::invalid_argument(
        "GPU timing must be enabled when creating the execution domain");
  // At most one FIFO prefix entry is polled per hot admission. Since every
  // entry here is on the same stream, a pending head cannot hide a ready tail.
  collect_ready(1);
  std::lock_guard<std::mutex> lock(state.mutex);
  state.owner->throw_if_new_submission_disallowed(
      "submit CUDA execution domain");
  if (state.first_error)
    std::rethrow_exception(state.first_error);
  if (!state.accepting || state.closing || state.closed)
    throw std::runtime_error("CUDA execution domain is closing or closed");
  for (const auto &dependency : dependencies)
    dependency.validate_cuda_dependency(state.owner_id, state.owner);
  if (state.count == state.pending.size())
    throw ExecutionDomainBusy();

  // Reservation failures enqueue nothing, including predecessor waits. They
  // must not enter the partial-device-submission fault path below. Resource
  // parameters outlive the lock guard, so cancellation callbacks can reenter.
  if (admit)
    admit();
  auto context = CUDAContext::get_instance().get_guard();
  const auto sequence = state.owner->allocate_submission_sequence();
  const auto index = (state.head + state.count) % state.pending.size();
  auto &slot = state.pending[index];
  const auto timing = sample_gpu ? state.timings[index] : StreamGpuTiming{};
  slot.sequence = sequence;
  slot.resources = std::move(resources);
  state.last_sequence = sequence;
  ++state.count;  // Ownership is published before any possible device effect.
  bool finalized = false;
  try {
    for (const auto &dependency : dependencies)
      dependency.enqueue_cuda_wait(state.stream, state.owner_id, state.owner);
    if (before_timing)
      before_timing(state.stream);
    if (timing)
      RuntimeCompletion::restart_cuda_gpu_timing(timing, state.stream);
    enqueue(state.stream);
    if (timing)
      RuntimeCompletion::end_cuda_gpu_timing(timing, state.stream, state.owner);
    finalized = true;
    if (after_timing)
      after_timing(state.stream);
    state.publish(slot, timing);
    return slot.completion;
  } catch (...) {
    auto error = std::current_exception();
    // External release must precede sealing the failed prefix as well. This
    // keeps cleanup from enqueuing work after the completion's event.
    if (!finalized && after_timing) {
      try {
        after_timing(state.stream);
      } catch (...) {
        state.report(std::current_exception(), sequence);
      }
    }
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
          state.accepting && !state.closing && !state.closed &&
              !state.first_error && state.owner->submission_allowed(),
          state.closed && !state.instance_retirement_error};
}

RuntimeCompletion ExecutionDomain::latest_completion() const {
  const auto &state = *state_;
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.first_error)
    std::rethrow_exception(state.first_error);
  if (state.count)
    return state.pending[(state.head + state.count - 1) % state.pending.size()]
        .completion;
  return RuntimeCompletion::completed(Arch::cuda, state.owner_id,
                                      state.last_sequence, state.owner);
}

bool ExecutionDomain::belongs_to(
    const std::shared_ptr<RuntimeFaultDomain> &owner) const {
  return state_->owner == owner;
}

RuntimeCompletion ExecutionDomain::retirement_completion() const {
  const auto &state = *state_;
  std::lock_guard<std::mutex> lock(state.mutex);
  // A provider error can still have a successfully sealed device prefix.
  // Drain that prefix without changing the public sticky-error contract. If
  // event recording itself failed, close() must prove retirement via its
  // stream drain; an invalid token never authorizes storage release.
  return state.count ? state
                           .pending[(state.head + state.count - 1) %
                                    state.pending.size()]
                           .completion
                     : RuntimeCompletion{};
}

void ExecutionDomain::prepare_instance(
    const std::shared_ptr<RuntimeClient> &instance,
    const std::function<void()> &prepare) {
  if (!instance || !prepare)
    throw std::invalid_argument(
        "CUDA execution instance requires an owner and preparation");
  auto &state = *state_;
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    state.owner->throw_if_new_submission_disallowed(
        "prepare CUDA execution instance");
    if (!state.accepting || state.closing || state.closed || state.first_error)
      throw std::runtime_error(
          "CUDA execution domain is not accepting instances");
    if (state.instances.size() + state.preparing_instances ==
        state.options.instance_capacity)
      throw std::runtime_error("CUDA execution-instance capacity exhausted");
    ++state.preparing_instances;
  }
  std::exception_ptr error, retirement_error;
  try {
    prepare();
  } catch (...) {
    error = std::current_exception();
    try {
      instance->retire();
    } catch (...) {
      retirement_error = std::current_exception();
    }
  }
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    // Storage was reserved at domain creation. Failed cleanup retains the
    // instance until cold close can retry or quarantine its finite storage.
    if (!error || retirement_error)
      state.instances.push_back(instance);
    --state.preparing_instances;
    if (retirement_error)
      state.report(retirement_error, state.last_sequence);
    state.changed.notify_all();
  }
  if (error)
    std::rethrow_exception(error);
}

void ExecutionDomain::stop_accepting() {
  std::lock_guard<std::mutex> lock(state_->mutex);
  state_->accepting = false;
}

void ExecutionDomain::close() {
  if (!state_)
    return;
  auto &state = *state_;
  {
    std::unique_lock<std::mutex> lock(state.mutex);
    if (state.closing && state.closing_thread == std::this_thread::get_id())
      return;
    state.changed.wait(lock, [&] { return !state.closing; });
    if (state.closed) {
      if (state.instance_retirement_error)
        std::rethrow_exception(state.instance_retirement_error);
      return;
    }
    state.accepting = false;
    state.closing = true;
    state.closing_thread = std::this_thread::get_id();
    state.changed.wait(lock, [&] { return state.preparing_instances == 0; });
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
  std::vector<std::shared_ptr<RuntimeClient>> instances;
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!error) {
      retired.swap(state.pending);
      instances.swap(state.instances);
      state.count = 0;
      state.stream = nullptr;
    } else {
      state.report(error, state.last_sequence);
      state.closing = false;
      state.closing_thread = {};
      state.changed.notify_all();
    }
  }
  if (error)
    std::rethrow_exception(error);
  for (auto &slot : retired)
    slot.completion.mark_completed();
  retired.clear();
  state.timings.clear();
  state.events->clear();
  std::exception_ptr retirement_error;
  for (const auto &instance : instances) {
    try {
      instance->retire();
    } catch (...) {
      if (!retirement_error)
        retirement_error = std::current_exception();
    }
  }
  if (!retirement_error)
    instances.clear();
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    if (retirement_error) {
      state.instances.swap(instances);
      state.instance_retirement_error = retirement_error;
      state.report(retirement_error, state.last_sequence);
    }
    state.closed = true;
    state.closing = false;
    state.closing_thread = {};
    state.changed.notify_all();
  }
  if (retirement_error)
    std::rethrow_exception(retirement_error);
}

struct ExecutionDomainRegistry::State {
  std::shared_ptr<RuntimeFaultDomain> owner;
  std::vector<std::shared_ptr<ExecutionDomain>> domains;
  mutable std::mutex mutex;
  std::condition_variable changed;
  bool accepting{true};
  bool retiring{false};
  bool retired{false};
  std::thread::id retiring_thread;
  std::exception_ptr first_error;
};

ExecutionDomainRegistry::ExecutionDomainRegistry(
    std::shared_ptr<RuntimeFaultDomain> owner,
    std::size_t capacity)
    : state_(std::make_unique<State>()) {
  if (!owner || owner->backend() != Arch::cuda || capacity == 0)
    throw std::invalid_argument(
        "CUDA domain registry requires a CUDA owner and capacity");
  owner->throw_if_new_submission_disallowed("create CUDA domain registry");
  state_->owner = std::move(owner);
  state_->domains.resize(capacity);
}

ExecutionDomainRegistry::~ExecutionDomainRegistry() {
  try {
    retire();
  } catch (...) {
    // Individual domains quarantine any leases whose drain was not proven.
  }
}

std::shared_ptr<ExecutionDomain> ExecutionDomainRegistry::create(
    ExecutionDomain::Options options) {
  auto &state = *state_;
  std::lock_guard<std::mutex> lock(state.mutex);
  state.owner->throw_if_new_submission_disallowed(
      "create CUDA execution domain");
  if (!state.accepting)
    throw std::runtime_error("CUDA domain registry has retired");
  for (auto &slot : state.domains) {
    if (!slot || slot->snapshot().closed) {
      auto domain = std::make_shared<ExecutionDomain>(state.owner, options);
      slot = domain;
      return domain;
    }
  }
  throw std::runtime_error("CUDA runtime execution-domain capacity exhausted");
}

bool ExecutionDomainRegistry::belongs_to(
    const std::shared_ptr<RuntimeFaultDomain> &owner) const {
  return state_->owner == owner;
}

bool ExecutionDomainRegistry::accepting() const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  return state_->accepting && state_->owner->submission_allowed();
}

void ExecutionDomainRegistry::synchronize() {
  std::vector<std::shared_ptr<ExecutionDomain>> domains;
  {
    std::lock_guard<std::mutex> lock(state_->mutex);
    domains = state_->domains;
  }
  std::vector<RuntimeCompletion> prefixes;
  prefixes.reserve(domains.size());
  for (const auto &domain : domains)
    if (domain)
      prefixes.push_back(domain->latest_completion());
  for (const auto &prefix : prefixes)
    prefix.wait();
}

void ExecutionDomainRegistry::retire() {
  auto &state = *state_;
  {
    std::unique_lock<std::mutex> lock(state.mutex);
    if (state.retiring && state.retiring_thread == std::this_thread::get_id())
      return;  // A resource retirement callback may inspect its owner.
    state.changed.wait(lock, [&] { return !state.retiring; });
    if (state.retired) {
      if (state.first_error)
        std::rethrow_exception(state.first_error);
      return;
    }
    state.accepting = false;
    state.retiring = true;
    state.retiring_thread = std::this_thread::get_id();
  }
  // With creation closed, slots are immutable. No registry lock is held
  // across domain admission, backend waits, or arbitrary resource callbacks.
  for (const auto &domain : state.domains)
    if (domain)
      domain->stop_accepting();
  std::exception_ptr failure;
  // Finish all accepted work before destructive cleanup of any domain. CUDA
  // event destruction and vendor handle cleanup can synchronize the context;
  // they must not accidentally become the wait for a different domain's work.
  for (const auto &domain : state.domains) {
    if (!domain)
      continue;
    try {
      auto prefix = domain->retirement_completion();
      if (prefix.valid())
        prefix.wait();
    } catch (...) {
      if (!failure)
        failure = std::current_exception();
    }
  }
  for (const auto &domain : state.domains) {
    if (!domain)
      continue;
    try {
      domain->close();
    } catch (...) {
      if (!failure)
        failure = std::current_exception();
    }
  }
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    state.first_error = failure;
    state.retired = true;
    state.retiring = false;
    state.retiring_thread = {};
  }
  state.changed.notify_all();
  if (failure)
    std::rethrow_exception(failure);
}

}  // namespace taichi::lang::cuda
