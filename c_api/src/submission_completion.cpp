#include "taichi_core_impl.h"
#include "taichi/program/runtime_fault.h"
#include "taichi/runtime/submission_completion.h"
#include "taichi/runtime/completion_wait_set.h"
#include "taichi/runtime/aot_runtime_client.h"
#include "completion.h"
#ifdef TI_WITH_VULKAN
#include "taichi_gfx_impl.h"
#endif

namespace capi {
struct SubmissionCompletionState {
  const std::uint64_t domain{Runtime::new_handle_token()};
  taichi::lang::SubmissionCompletionQueue pending;
  std::shared_ptr<taichi::lang::RuntimeCompletionCudaEventPool> cuda_events;
};
}  // namespace capi

namespace {
taichi::lang::SubmissionCompletionHandles &tickets() {
  static taichi::lang::SubmissionCompletionHandles handles;
  return handles;
}

struct CompletionSets {
  std::mutex mutex;
  std::unordered_map<TiCompletionSet,
                     std::shared_ptr<taichi::lang::CompletionWaitSet>> entries;
};
CompletionSets &completion_sets() {
  static auto *sets = new CompletionSets;
  return *sets;
}
std::shared_ptr<taichi::lang::CompletionWaitSet> get_completion_set(
    TiCompletionSet handle, bool release = false) {
  auto &sets = completion_sets();
  std::lock_guard<std::mutex> lock(sets.mutex);
  const auto found = sets.entries.find(handle);
  if (found == sets.entries.end())
    throw MemoryBindingError("Completion set is closed or unknown");
  auto set = found->second;
  if (release)
    sets.entries.erase(found);
  return set;
}
}  // namespace

namespace taichi::lang {
std::uintptr_t publish_aot_completion(const RuntimeCompletion &completion) {
  const auto handle = Runtime::new_handle_token();
  tickets().publish(handle, completion);
  return handle;
}
}  // namespace taichi::lang

TiCompletionSet ti_create_completion_set_ext(
    uint64_t count, const TiCompletionSource *sources, int64_t poll_interval_ns) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL_RV(sources);
  std::vector<taichi::lang::RuntimeCompletion> completions;
  completions.reserve(count);
  for (uint64_t index = 0; index < count; ++index) {
    const auto &source = sources[index];
    if (source.kind == TI_COMPLETION_SOURCE_SUBMISSION) {
      completions.push_back(tickets().get(
          reinterpret_cast<std::uintptr_t>(source.handle)));
    }
    else if (source.kind == TI_COMPLETION_SOURCE_DOMAIN) {
      completions.push_back(
          capi::get_completion(reinterpret_cast<TiCompletion>(source.handle)));
    }
    else {
      throw MemoryBindingError("Unknown completion source kind");
    }
  }
  auto set = std::make_shared<taichi::lang::CompletionWaitSet>(
      std::move(completions), std::chrono::nanoseconds(poll_interval_ns));
  auto handle = reinterpret_cast<TiCompletionSet>(Runtime::new_handle_token());
  auto &sets = completion_sets();
  std::lock_guard<std::mutex> lock(sets.mutex);
  sets.entries.emplace(handle, set);
  return handle;
  TI_CAPI_TRY_CATCH_END();
  return TI_NULL_HANDLE;
}

void ti_wait_completion_set_ext(TiCompletionSet set, int64_t timeout_ns,
                                TiCompletionWaitResult *result) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(result);
  *result = {TI_COMPLETION_WAIT_FAILED, 0};
  const auto observed = get_completion_set(set)->wait(
      std::chrono::nanoseconds(timeout_ns));
  *result = {static_cast<TiCompletionWaitStatus>(observed.status),
             static_cast<uint64_t>(observed.index)};
  TI_CAPI_TRY_CATCH_END();
}

void ti_cancel_completion_set_ext(TiCompletionSet set) {
  TI_CAPI_TRY_CATCH_BEGIN();
  get_completion_set(set)->cancel();
  TI_CAPI_TRY_CATCH_END();
}

void ti_release_completion_set_ext(TiCompletionSet set) {
  TI_CAPI_TRY_CATCH_BEGIN();
  get_completion_set(set, true)->cancel();
  TI_CAPI_TRY_CATCH_END();
}

void Runtime::borrow_completion_recorder(
    std::function<taichi::lang::RuntimeCompletion()> recorder) {
  std::lock_guard<std::recursive_mutex> lock(submission_mutex());
  if (!borrowed_fault_domain_ || submission_completions_ ||
      borrowed_completion_recorder_ || !recorder)
    throw std::logic_error("Completion recorder must be bound during borrowing");
  borrowed_completion_recorder_ = std::move(recorder);
}

void Runtime::prepare_submission_completion() {
  using namespace taichi::lang;
  runtime_fault_domain_->throw_if_submission_disallowed("AOT ticket submission");
  if (!taichi::arch_is_cpu(arch) && arch != taichi::Arch::cuda &&
      arch != taichi::Arch::vulkan)
    throw std::invalid_argument("Submission tickets require CPU, CUDA or Vulkan");
  if (borrowed_completion_recorder_)
    return;
  if (!submission_completions_) {
    auto state = std::make_shared<capi::SubmissionCompletionState>();
    if (arch == taichi::Arch::cuda)
      state->cuda_events = std::make_shared<RuntimeCompletionCudaEventPool>(
          runtime_fault_domain_, 0,
          RuntimeCompletionCudaEventPool::Retention::until_clear);
    submission_completions_ = std::move(state);
  }
  submission_completions_->pending.collect_one();
}

TiSubmissionTicket Runtime::finish_submission_completion() {
  using namespace taichi::lang;
  const auto identity = new_handle_token();
  RuntimeCompletion completion;
  try {
    if (borrowed_completion_recorder_) {
      completion = borrowed_completion_recorder_();
    } else {
      auto &state = *submission_completions_;
      if (taichi::arch_is_cpu(arch)) {
        wait();
        completion = RuntimeCompletion::completed(
            arch, state.domain, identity, runtime_fault_domain_);
      } else if (arch == taichi::Arch::cuda) {
        completion = RuntimeCompletion::from_cuda_stream(
            state.domain, identity, nullptr, runtime_fault_domain_,
            state.cuda_events);
      }
#ifdef TI_WITH_VULKAN
      else if (arch == taichi::Arch::vulkan) {
        completion = RuntimeCompletion::from_stream_semaphore(
            arch, state.domain, identity,
            static_cast<GfxRuntime &>(*this).get_gfx_runtime()
                .record_completion_semaphore(),
            runtime_fault_domain_);
      }
#endif
      state.pending.retain(completion);
    }
    tickets().publish(identity, completion);
    return reinterpret_cast<TiSubmissionTicket>(identity);
  } catch (...) {
    // Record/publication can fail after kernels were submitted. Preserve the
    // original failure, drain work where possible, and never invent a ticket.
    const auto failure = std::current_exception();
    try {
      wait();
      completion.mark_completed();
      complete_submission_completions();
    } catch (...) {
      // Existing owner/fault retirement remains responsible for dependencies.
      // Pending records already retained by this owner or Program stay tracked.
    }
    std::rethrow_exception(failure);
  }
}

void Runtime::complete_submission_completions() {
  if (submission_completions_)
    submission_completions_->pending.complete_synchronized();
}

void Runtime::clear_submission_completions() {
  std::lock_guard<std::recursive_mutex> lock(submission_mutex());
  if (!submission_completions_)
    return;
  if (submission_completions_->pending.size())
    wait();
  complete_submission_completions();
  if (submission_completions_->cuda_events)
    submission_completions_->cuda_events->clear();
  submission_completions_.reset();
}

TiBool ti_query_submission_ext(TiSubmissionTicket ticket) {
  TI_CAPI_TRY_CATCH_BEGIN();
  if (!ticket) {
    ti_set_last_error(TI_ERROR_ARGUMENT_NULL, "ticket");
    return TI_FALSE;
  }
  return tickets().get(reinterpret_cast<std::uintptr_t>(ticket)).done()
             ? TI_TRUE : TI_FALSE;
  TI_CAPI_TRY_CATCH_END();
  return TI_FALSE;
}

void ti_wait_submission_ext(TiSubmissionTicket ticket) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(ticket);
  tickets().get(reinterpret_cast<std::uintptr_t>(ticket)).wait();
  TI_CAPI_TRY_CATCH_END();
}

void ti_destroy_submission_ext(TiSubmissionTicket ticket) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(ticket);
  tickets().release(reinterpret_cast<std::uintptr_t>(ticket));
  TI_CAPI_TRY_CATCH_END();
}
