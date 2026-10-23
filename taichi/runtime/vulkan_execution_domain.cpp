#include "taichi/runtime/vulkan_execution_domain.h"

#ifdef TI_WITH_VULKAN
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

#include "taichi/program/runtime_fault.h"

namespace taichi::lang::vulkan {
namespace {
std::atomic<std::uint64_t> next_domain_identity{1};

void report(const std::shared_ptr<RuntimeFaultDomain> &owner,
            std::exception_ptr failure,
            std::uint64_t sequence) noexcept {
  try {
    std::rethrow_exception(failure);
  } catch (const BackendRuntimeError &error) {
    owner->report_backend_error(error, sequence);
  } catch (...) {
  }
}
}  // namespace

// Mutable fields use the domain admission mutex, except the published proof.
class ExecutionInstance {
 public:
  const std::uint64_t domain;
  std::shared_ptr<RuntimeClient> client;
  RuntimeCompletion latest;
  std::exception_ptr error;
  bool accepting{false};
  bool retiring{false};
  bool unsealed_submission{false};
  std::atomic<bool> retired{false};

  ExecutionInstance(std::uint64_t domain, std::shared_ptr<RuntimeClient> client)
      : domain(domain), client(std::move(client)) {}
};

struct ExecutionDomain::State {
  const std::shared_ptr<RuntimeFaultDomain> owner;
  const std::uint64_t identity;
  const Options options;
  VulkanExecutionQueue queue;
  mutable std::mutex mutex;
  std::condition_variable changed;
  std::vector<RuntimeCompletion> pending;
  std::vector<std::shared_ptr<ExecutionInstance>> instances;
  RuntimeCompletion latest;
  std::size_t head{0}, count{0};
  std::size_t preparing_instances{0}, retiring_instances{0};
  bool accepting{true}, closing{false}, closed{false};
  std::thread::id closing_thread;
  std::exception_ptr first_error, retirement_error;

  State(VulkanDevice &device, std::shared_ptr<RuntimeFaultDomain> owner,
        Options options)
      : owner(std::move(owner)),
        identity(next_domain_identity.fetch_add(1, std::memory_order_relaxed)),
        options(options), queue(device, options.capacity),
        pending(options.capacity),
        latest(RuntimeCompletion::completed(Arch::vulkan,
                   this->owner->program_domain(), 0, this->owner)) {
    if (!identity)
      throw std::overflow_error("Vulkan domain identity exhausted");
    instances.reserve(options.instance_capacity);
  }
  void check_accepting(const char *operation) const {
    owner->throw_if_new_submission_disallowed(operation);
    if (first_error)
      std::rethrow_exception(first_error);
    if (!accepting || closing || closed)
      throw std::runtime_error("Vulkan execution domain is closing or closed");
  }
};

ExecutionDomain::ExecutionDomain(VulkanDevice &device,
                                 std::shared_ptr<RuntimeFaultDomain> owner,
                                 Options options) {
  if (!owner || owner->backend() != Arch::vulkan || !owner->program_domain() ||
      device.backend_fault_reporter() != owner || !options.capacity ||
      !options.instance_capacity)
    throw std::invalid_argument(
        "Vulkan domain requires its device owner and positive capacities");
  owner->throw_if_new_submission_disallowed("create Vulkan execution domain");
  state_ = std::make_unique<State>(device, std::move(owner), options);
}

ExecutionDomain::~ExecutionDomain() {
  try {
    close();
  } catch (...) {
    // Unproven work/client teardown keeps its finite resource closure alive.
    (void)state_.release();
  }
}

std::unique_ptr<CommandList> ExecutionDomain::new_command_list() {
  auto &state = *state_;
  std::lock_guard<std::mutex> lock(state.mutex);
  state.check_accepting("record Vulkan execution commands");
  return state.queue.new_command_list();
}

bool ExecutionDomain::gpu_timing_enabled() const {
  return state_->options.gpu_timing;
}

StreamGpuTiming ExecutionDomain::prepare_timing(CommandList &begin,
                                                CommandList &end) {
  auto &state = *state_;
  std::lock_guard<std::mutex> lock(state.mutex);
  state.check_accepting("prepare Vulkan domain timing");
  if (!state.options.gpu_timing)
    throw std::invalid_argument("GPU timing was not enabled for this Vulkan domain");
  return state.queue.prepare_timing(begin, end);
}

std::shared_ptr<const ExecutionDomain::Commands> ExecutionDomain::prepare_commands(
    const std::vector<CommandList *> &commands, std::shared_ptr<void> operands) {
  auto &state = *state_;
  std::lock_guard<std::mutex> lock(state.mutex);
  state.check_accepting("prepare Vulkan execution commands");
  return state.queue.prepare(commands, operands);
}

RuntimeCompletion ExecutionDomain::submit(
    std::shared_ptr<const Commands> commands,
    const std::vector<RuntimeCompletion> &dependencies,
    std::shared_ptr<RuntimeCompletionResources> resources,
    const Admission &admit) {
  return submit_impl({}, std::move(commands), dependencies, std::move(resources), admit);
}

RuntimeCompletion ExecutionDomain::submit_instance(
    const std::shared_ptr<ExecutionInstance> &instance,
    std::shared_ptr<const Commands> commands,
    const std::vector<RuntimeCompletion> &dependencies,
    std::shared_ptr<RuntimeCompletionResources> resources,
    const Admission &admit) {
  if (!instance)
    throw std::invalid_argument("Vulkan submission requires an instance");
  return submit_impl(instance, std::move(commands), dependencies,
                     std::move(resources), admit);
}

RuntimeCompletion ExecutionDomain::submit_impl(
    const std::shared_ptr<ExecutionInstance> &instance,
    std::shared_ptr<const Commands> commands,
    const std::vector<RuntimeCompletion> &dependencies,
    std::shared_ptr<RuntimeCompletionResources> resources,
    const Admission &admit,
    const CommandSelector &select,
    const TimingSelector &timing) {
  auto &state = *state_;
  // Both completion and command operand destruction run outside admission.
  collect_ready(1);
  std::vector<VulkanExecutionQueue::Point> points;
  points.reserve(dependencies.size());
  RuntimeCompletion ticket;
  std::lock_guard<std::mutex> lock(state.mutex);
  state.check_accepting("submit Vulkan execution domain");
  if (instance && (instance->domain != state.identity || !instance->accepting))
    throw std::invalid_argument("Vulkan instance is foreign or retired");
  for (const auto &dependency : dependencies) {
    auto origin = dependency.reusable_stream_dependency(
        Arch::vulkan, state.owner->program_domain(), state.owner);
    if (!origin)
      continue;
    auto point = std::dynamic_pointer_cast<VulkanTimelinePoint>(origin);
    if (!point)
      throw std::invalid_argument("Vulkan dependency has no timeline prefix");
    points.push_back(std::move(point));
  }
  if (state.count == state.pending.size())
    throw Busy();
  const auto index = (state.head + state.count) % state.pending.size();
  bool reserved = false;
  try {
    state.queue.submit_precollected(commands, points,
        [&](std::size_t slot, const auto &point) {
          ticket = RuntimeCompletion::from_reusable_stream_dependency(
              Arch::vulkan, state.owner->program_domain(),
              state.owner->allocate_submission_sequence(), point, state.owner,
              timing ? timing(slot) : nullptr);
          if (resources)
            ticket.attach_resources(resources);
          if (admit)
            admit(slot, ticket);
          // From here onward all host construction succeeded; retain the full
          // submission even if the native enqueue fails without a valid signal.
          state.pending[index] = ticket;
          state.latest = ticket;
          ++state.count;
          if (instance)
            instance->unsealed_submission = true;
          reserved = true;
        }, select ? CommandSelector([&](std::size_t slot) {
          // Keep selected operands alive outside the wider admission lock,
          // including a failure before the queue accepts the commands.
          commands = select(slot);
          return commands;
        }) : CommandSelector{});
  } catch (...) {
    if (reserved) {
      state.first_error = std::current_exception();
      state.accepting = false;
      ticket.invalidate("Vulkan domain submission failed before completion was sealed");
      report(state.owner, state.first_error, ticket.sequence());
    }
    throw;
  }
  if (instance) {
    instance->latest = ticket;
    instance->unsealed_submission = false;
  }
  return ticket;
}

RuntimeCompletion ExecutionDomain::submit_instance_selected(
    const std::shared_ptr<ExecutionInstance> &instance,
    const CommandSelector &select,
    const std::vector<RuntimeCompletion> &dependencies,
    std::shared_ptr<RuntimeCompletionResources> resources,
    const Admission &admit,
    const TimingSelector &timing) {
  if (!instance || !select)
    throw std::invalid_argument("Vulkan submission requires an instance and selector");
  return submit_impl(instance, {}, dependencies, std::move(resources), admit, select, timing);
}

std::size_t ExecutionDomain::collect_ready(std::size_t budget) {
  auto &state = *state_;
  std::size_t collected = 0;
  while (collected < budget) {
    RuntimeCompletion completion;
    {
      std::lock_guard<std::mutex> lock(state.mutex);
      if (state.closing || state.closed || !state.count || state.first_error)
        break;
      completion = state.pending[state.head];
    }
    if (!completion.done())
      break;
    RuntimeCompletion retired;
    {
      std::lock_guard<std::mutex> lock(state.mutex);
      if (state.closing || state.closed || !state.count)
        break;
      auto &head = state.pending[state.head];
      if (head.sequence() != completion.sequence())
        continue;
      retired = std::move(head);
      head = {};
      state.head = (state.head + 1) % state.pending.size();
      --state.count;
      ++collected;
    }
  }
  state.queue.collect(budget);
  return collected;
}

ExecutionDomain::Snapshot ExecutionDomain::snapshot() const {
  auto &state = *state_;
  std::lock_guard<std::mutex> lock(state.mutex);
  return {state.identity, state.owner->program_domain(), state.queue.queue_index(),
          state.options.capacity, state.count,
          state.accepting && !state.closing && !state.closed && !state.first_error &&
              state.owner->submission_allowed(),
          state.closed && !state.retirement_error};
}

RuntimeCompletion ExecutionDomain::latest_completion() const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  if (state_->first_error)
    std::rethrow_exception(state_->first_error);
  return state_->latest;
}

RuntimeCompletion ExecutionDomain::retirement_completion() const {
  std::lock_guard<std::mutex> lock(state_->mutex);
  return state_->latest;
}

bool ExecutionDomain::belongs_to(const std::shared_ptr<RuntimeFaultDomain> &owner) const {
  return state_->owner == owner;
}

void ExecutionDomain::stop_accepting() {
  std::lock_guard<std::mutex> lock(state_->mutex);
  state_->accepting = false;
}

std::shared_ptr<ExecutionInstance> ExecutionDomain::prepare_owned_instance(
    const std::shared_ptr<RuntimeClient> &client,
    const std::function<void(const std::shared_ptr<ExecutionInstance> &)> &prepare) {
  if (!client || !prepare)
    throw std::invalid_argument("Vulkan instance requires its owner and preparation");
  auto &state = *state_;
  auto instance = std::make_shared<ExecutionInstance>(state.identity, client);
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    state.check_accepting("prepare Vulkan execution instance");
    if (state.instances.size() + state.preparing_instances ==
        state.options.instance_capacity)
      throw std::runtime_error("Vulkan execution-instance capacity exhausted");
    ++state.preparing_instances;
  }
  std::exception_ptr failure, cleanup_failure;
  try {
    prepare(instance);
  } catch (...) {
    failure = std::current_exception();
    try {
      client->retire();
    } catch (...) {
      cleanup_failure = std::current_exception();
      report(state.owner, cleanup_failure, 0);
    }
  }
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!failure || cleanup_failure)
      state.instances.push_back(instance);  // Capacity reserved before prepare.
    instance->accepting = !failure;
    instance->error = cleanup_failure;
    instance->retired = failure && !cleanup_failure;
    --state.preparing_instances;
    state.changed.notify_all();
  }
  if (failure)
    std::rethrow_exception(failure);
  return instance;
}

void ExecutionDomain::retire_instance(const std::shared_ptr<ExecutionInstance> &instance) {
  auto &state = *state_;
  RuntimeCompletion prefix;
  std::shared_ptr<RuntimeClient> client;
  {
    std::unique_lock<std::mutex> lock(state.mutex);
    if (!instance || instance->domain != state.identity)
      throw std::invalid_argument("Vulkan instance belongs to another domain");
    state.changed.wait(lock, [&] { return !state.closing && !instance->retiring; });
    if (instance->error)
      std::rethrow_exception(instance->error);
    if (instance->retired)
      return;
    if (!instance->accepting || instance->unsealed_submission)
      throw std::runtime_error("Vulkan instance has no proven retirement prefix");
    instance->accepting = false;
    instance->retiring = true;
    ++state.retiring_instances;
    prefix = instance->latest;
    client = instance->client;
  }
  std::exception_ptr failure;
  try {
    if (prefix.valid())
      prefix.wait();
    // Waiting a timeline point does not release the queue slot's Commands.
    // Collect completed slots outside admission locks before retiring their
    // operand owners (including shared image read loans). This never waits
    // for a later prefix belonging to another instance.
    collect_ready(state.options.capacity);
    client->retire();
  } catch (...) {
    failure = std::current_exception();
    report(state.owner, failure, prefix.sequence());
  }
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    instance->error = failure;
    instance->retiring = false;
    --state.retiring_instances;
    if (!failure) {
      instance->client.reset();  // Local client releases outside the lock.
      instance->latest = {};
      const auto found = std::find(state.instances.begin(), state.instances.end(), instance);
      TI_ASSERT(found != state.instances.end());
      state.instances.erase(found);
      instance->retired.store(true, std::memory_order_release);
    }
    state.changed.notify_all();
  }
  if (failure)
    std::rethrow_exception(failure);
}

bool ExecutionDomain::instance_retired(const std::shared_ptr<ExecutionInstance> &instance) const {
  if (!instance || instance->domain != state_->identity)
    throw std::invalid_argument("Vulkan instance belongs to another domain");
  return instance->retired.load(std::memory_order_acquire);
}

void ExecutionDomain::close() {
  auto &state = *state_;
  {
    std::unique_lock<std::mutex> lock(state.mutex);
    if (state.closing && state.closing_thread == std::this_thread::get_id())
      return;
    state.changed.wait(lock, [&] { return !state.closing; });
    if (state.closed) {
      if (state.retirement_error)
        std::rethrow_exception(state.retirement_error);
      return;
    }
    state.accepting = false;
    state.closing = true;
    state.closing_thread = std::this_thread::get_id();
    state.changed.wait(lock, [&] {
      return !state.preparing_instances && !state.retiring_instances;
    });
  }
  try {
    state.queue.close();
  } catch (...) {
    const auto failure = std::current_exception();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.first_error)
      state.first_error = failure;
    state.closing = false;
    state.closing_thread = {};
    state.changed.notify_all();
    throw;
  }
  std::vector<RuntimeCompletion> pending;
  std::vector<std::shared_ptr<ExecutionInstance>> instances;
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    pending.swap(state.pending);
    instances.swap(state.instances);
    state.count = 0;
  }
  for (const auto &completion : pending)
    completion.mark_completed();
  pending.clear();
  std::exception_ptr failure;
  for (const auto &instance : instances) {
    try {
      if (instance->error)
        std::rethrow_exception(instance->error);
      instance->client->retire();
      instance->client.reset();
      instance->latest = {};
      instance->accepting = false;
      instance->retired.store(true, std::memory_order_release);
    } catch (...) {
      instance->error = std::current_exception();
      if (!failure)
        failure = instance->error;
      report(state.owner, instance->error, 0);
    }
  }
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    if (failure) {
      state.instances.swap(instances);
      state.retirement_error = failure;
    }
    state.closed = true;
    state.closing = false;
    state.closing_thread = {};
    state.changed.notify_all();
  }
  if (failure)
    std::rethrow_exception(failure);
}

struct ExecutionDomainRegistry::State {
  VulkanDevice &device;
  const std::shared_ptr<RuntimeFaultDomain> owner;
  std::vector<std::shared_ptr<ExecutionDomain>> domains;
  mutable std::mutex mutex;
  std::condition_variable changed;
  bool accepting{true}, retiring{false}, retired{false};
  std::thread::id retiring_thread;
  std::exception_ptr first_error;

  State(VulkanDevice &device, std::shared_ptr<RuntimeFaultDomain> owner)
      : device(device), owner(std::move(owner)) {}
};

ExecutionDomainRegistry::ExecutionDomainRegistry(
    VulkanDevice &device, std::shared_ptr<RuntimeFaultDomain> owner) {
  if (!owner || owner->backend() != Arch::vulkan ||
      device.backend_fault_reporter() != owner)
    throw std::invalid_argument("Vulkan domain registry requires its device owner");
  owner->throw_if_new_submission_disallowed("create Vulkan domain registry");
  state_ = std::make_unique<State>(device, std::move(owner));
}

ExecutionDomainRegistry::~ExecutionDomainRegistry() {
  try {
    retire();
  } catch (...) {
    // Individual domains quarantine unproven work/client retirement.
  }
}

std::shared_ptr<ExecutionDomain> ExecutionDomainRegistry::create(ExecutionDomain::Options options) {
  auto &state = *state_;
  std::lock_guard<std::mutex> lock(state.mutex);
  state.owner->throw_if_new_submission_disallowed("create Vulkan execution domain");
  if (!state.accepting)
    throw std::runtime_error("Vulkan domain registry has retired");
  for (auto &slot : state.domains) {
    if (!slot || slot->snapshot().closed) {
      auto domain = std::make_shared<ExecutionDomain>(state.device, state.owner, options);
      slot = domain;
      return domain;
    }
  }
  state.domains.emplace_back();  // Reserve host ownership before native creation.
  try {
    auto domain = std::make_shared<ExecutionDomain>(state.device, state.owner, options);
    state.domains.back() = domain;
    return domain;
  } catch (...) {
    state.domains.pop_back();
    throw;
  }
}

bool ExecutionDomainRegistry::belongs_to(const std::shared_ptr<RuntimeFaultDomain> &owner) const {
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
      return;
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
  // Registration is closed, so these slots cannot change. Drain every accepted
  // prefix before resource callbacks may tear down cross-domain providers.
  for (const auto &domain : state.domains)
    if (domain)
      domain->stop_accepting();
  std::exception_ptr failure;
  for (const auto &domain : state.domains) {
    if (!domain)
      continue;
    try {
      domain->retirement_completion().wait();
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

}  // namespace taichi::lang::vulkan
#endif
