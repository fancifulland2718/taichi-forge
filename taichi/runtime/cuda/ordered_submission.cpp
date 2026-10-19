#include "taichi/runtime/cuda/ordered_submission.h"

#include <algorithm>
#include <limits>
#include <utility>
#include "taichi/program/runtime_completion.h"
#include "taichi/program/runtime_fault.h"
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/runtime/native_operation.h"

namespace taichi::lang {
std::shared_ptr<void> CudaProviderCompletionResource::bind_execution_domain(
    std::uint64_t identity) {
  std::lock_guard<std::mutex> lock(completion_mutex_);
  auto binding = execution_domain_binding_.lock();
  if (binding) {
    if (*binding != identity)
      throw std::invalid_argument(
          "Mutable CUDA provider resource belongs to another execution domain");
    return binding;
  }
  auto result = std::make_shared<std::uint64_t>(identity);
  execution_domain_binding_ = result;
  return result;
}

void CudaProviderCompletionResource::track_ordered_completion(
    const std::shared_ptr<RuntimeCompletion> &completion) {
  std::lock_guard<std::mutex> lock(completion_mutex_);
  ordered_completions_.push_back(completion);
}

void CudaProviderCompletionResource::retire_completed_ordered_submissions() {
  std::vector<std::weak_ptr<RuntimeCompletion>> pending;
  {
    std::lock_guard<std::mutex> lock(completion_mutex_);
    pending.swap(ordered_completions_);
  }
  // done() retires provider tokens. Never hold the resource's tracking mutex
  // while invoking its callbacks or the backend.
  try {
    pending.erase(std::remove_if(pending.begin(), pending.end(),
                                 [](const auto &weak) {
                                   const auto completion = weak.lock();
                                   return !completion || completion->done();
                                 }),
                  pending.end());
  } catch (...) {
    std::lock_guard<std::mutex> lock(completion_mutex_);
    ordered_completions_.insert(ordered_completions_.end(), pending.begin(),
                                pending.end());
    throw;
  }
  std::lock_guard<std::mutex> lock(completion_mutex_);
  ordered_completions_.insert(ordered_completions_.end(), pending.begin(),
                              pending.end());
}

namespace cuda {
struct OrderedSubmissionResources::State {
  struct Entry {
    std::shared_ptr<NativeOperation> operation;
    std::shared_ptr<CudaProviderCompletionResource> owner;
    std::uint64_t token{0};
    bool reserved{false};
    bool submitted{false};
  };
  std::vector<Entry> entries;
  std::shared_ptr<RuntimeCompletionResources> operands;
  bool attempted{false};
};

OrderedSubmissionResources::OrderedSubmissionResources(
    std::size_t operation_count,
    std::shared_ptr<RuntimeCompletionResources> operands)
    : state_(std::make_unique<State>()) {
  state_->operands = std::move(operands);
  state_->entries.resize(operation_count);
}

OrderedSubmissionResources::~OrderedSubmissionResources() {
  for (auto it = state_->entries.rbegin(); it != state_->entries.rend(); ++it) {
    if (!it->reserved)
      continue;
    if (!it->submitted)
      it->operation->cancel_cuda_ordered();
    else if (it->owner)
      it->owner->on_submission_retired(it->token);
  }
}

void OrderedSubmissionResources::reserve(
    const std::vector<std::shared_ptr<NativeOperation>> &operations) {
  if (state_->attempted || operations.size() != state_->entries.size())
    throw std::logic_error("Ordered domain admission was already attempted");
  state_->attempted = true;
  for (std::size_t i = 0; i < operations.size(); ++i) {
    auto &entry = state_->entries[i];
    entry.operation = operations[i];
    entry.owner = entry.operation->cuda_ordered_resource();
    entry.operation->reserve_cuda_ordered();
    entry.reserved = true;
    if (entry.owner)
      entry.token = entry.owner->submission_retirement_token();
  }
}

void OrderedSubmissionResources::run(std::size_t index, void *stream) {
  auto &entry = state_->entries.at(index);
  if (!stream || !entry.reserved || entry.submitted)
    throw std::logic_error(
        "Ordered domain operation has no unused reservation");
  // A failing provider may already have enqueued work. Transfer from cancel
  // to completion retirement before entering it, including its error path.
  entry.submitted = true;
  entry.operation->run_cuda_ordered(stream);
}

std::size_t OrderedSubmissionResources::retained_resource_count(
    std::uint32_t kind) const noexcept {
  return state_->operands ? state_->operands->retained_resource_count(kind) : 0;
}

struct OrderedSubmissions::State {
  struct Resources final : RuntimeCompletionResources {
    std::vector<std::pair<std::shared_ptr<CudaProviderCompletionResource>,
                          std::uint64_t>>
        leases;
    ~Resources() override {
      for (const auto &[owner, token] : leases)
        if (owner)
          owner->on_submission_retired(token);
    }
    std::size_t retained_resource_count(std::uint32_t) const noexcept override {
      // Program's resource-kind counters do not classify this AOT-owned batch.
      return 0;
    }
  };
  struct Pending {
    std::uint64_t sequence{0};
    std::shared_ptr<RuntimeCompletion> completion;
    // Kept here until event publication succeeds, including allocation failure
    // after a provider has already enqueued work.
    std::shared_ptr<Resources> resources;
  };
  std::size_t operation_count;
  std::uint64_t domain{0}, next_sequence{1};
  std::shared_ptr<RuntimeFaultDomain> fault;
  std::shared_ptr<RuntimeCompletionCudaEventPool> pool;
  std::vector<Pending> pending;
  bool active{false};

  bool backend_safe() const {
    return !fault || (fault->backend_calls_safe() &&
                      fault->state() != RuntimeLifecycleState::kFinalized);
  }
};

OrderedSubmissions::OrderedSubmissions(
    std::size_t operations,
    std::shared_ptr<RuntimeFaultDomain> fault_domain)
    : state_(std::make_unique<State>()) {
  state_->operation_count = operations;
  // Completion identities are private to this live owner, never serialized.
  state_->domain = reinterpret_cast<std::uintptr_t>(state_.get());
  state_->fault = fault_domain
                      ? std::move(fault_domain)
                      : std::make_shared<RuntimeFaultDomain>(Arch::cuda, 0);
  state_->pool = std::make_shared<RuntimeCompletionCudaEventPool>(
      state_->fault, 0, RuntimeCompletionCudaEventPool::Retention::until_clear);
}

OrderedSubmissions::~OrderedSubmissions() {
  auto &state = *state_;
  try {
    if (state.backend_safe()) {
      auto &cuda = CUDAContext::get_instance();
      auto guard = cuda.get_guard();
      auto lock = cuda.get_submission_lock_guard();
      for (const auto &pending : state.pending) {
        if (pending.completion->valid())
          pending.completion->wait();
        else
          CUDADriver::get_instance().stream_synchronize(nullptr);
      }
    }
  } catch (const BackendRuntimeError &error) {
    if (state.fault)
      state.fault->report_backend_error(error, 0);
  } catch (...) {
    // Owner retirement is noexcept, including a previously faulted completion.
  }
  for (auto &pending : state.pending)
    pending.completion->invalidate_and_release("ordered execution retired");
  state.pending.clear();
  state.pool->clear();
}

void OrderedSubmissions::begin() {
  auto &state = *state_;
  if (state.active)
    throw std::logic_error("Ordered CUDA submission is already active");
  if (state.next_sequence == std::numeric_limits<std::uint64_t>::max())
    throw std::overflow_error("Ordered CUDA submission sequence is exhausted");
  state.pending.erase(std::remove_if(state.pending.begin(), state.pending.end(),
                                     [](const auto &pending) {
                                       return pending.completion->done();
                                     }),
                      state.pending.end());
  State::Pending pending;
  pending.sequence = state.next_sequence++;
  pending.completion = std::make_shared<RuntimeCompletion>();
  pending.resources = std::make_shared<State::Resources>();
  pending.resources->leases.reserve(state.operation_count);
  state.pending.push_back(std::move(pending));
  state.active = true;
}

void OrderedSubmissions::run(NativeOperation &operation) {
  auto &state = *state_;
  if (!state.active)
    throw std::logic_error("Ordered CUDA submission has not begun");
  auto owner = operation.cuda_ordered_resource();
  if (owner)
    owner->retire_completed_ordered_submissions();
  operation.reserve_cuda_ordered();
  // Capacity was reserved before any work. Acquire the transaction token
  // before the provider can enqueue or report a partial failure.
  if (owner)
    state.pending.back().resources->leases.emplace_back(
        owner, owner->submission_retirement_token());
  operation.run_cuda_ordered(nullptr);
}

void OrderedSubmissions::finish() {
  auto &state = *state_;
  if (!state.active)
    return;
  auto &pending = state.pending.back();
  if (!pending.completion->valid())
    *pending.completion = RuntimeCompletion::from_cuda_stream(
        state.domain, pending.sequence, nullptr, state.fault, state.pool);
  if (pending.resources) {
    // Keep the resource list alive while associating the weak completion with
    // shared owners; attach_resources itself can complete synchronously.
    auto resources = pending.resources;
    pending.completion->attach_resources(resources);
    pending.resources.reset();
    for (const auto &[owner, token] : resources->leases) {
      (void)token;
      owner->track_ordered_completion(pending.completion);
    }
  }
  state.active = false;
}

void OrderedSubmissions::finish_after_failure(
    const BackendRuntimeError *error) noexcept {
  auto &state = *state_;
  if (error)
    state.fault->report_backend_error(*error, 0);
  if (!state.active)
    return;
  try {
    if (state.backend_safe()) {
      finish();
      return;
    }
  } catch (...) {
    // Preserve the original provider error. The pending batch still owns its
    // leases and is retired at the execution's cold destruction boundary.
  }
  state.active = false;
}
}  // namespace cuda
}  // namespace taichi::lang
