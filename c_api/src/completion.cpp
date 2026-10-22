#include "completion.h"
#include "taichi_core_impl.h"

#include <mutex>
#include <unordered_map>

namespace capi {
namespace {
using taichi::lang::RuntimeCompletion;

struct Handles {
  std::mutex mutex;
  std::unordered_map<TiCompletion, RuntimeCompletion> completions;
};

Handles &handles() {
  // Explicit public releases own teardown. Do not destroy pending resources
  // from a process-global destructor after their runtime/device has retired.
  static auto *registry = new Handles;
  return *registry;
}

RuntimeCompletion take_completion(TiCompletion handle) {
  auto &registry = handles();
  std::lock_guard<std::mutex> lock(registry.mutex);
  const auto it = registry.completions.find(handle);
  if (it == registry.completions.end())
    throw MemoryBindingError("Completion handle is closed or unknown");
  auto result = std::move(it->second);
  registry.completions.erase(it);
  return result;
}
}  // namespace

RuntimeCompletion get_completion(TiCompletion handle) {
  auto &registry = handles();
  std::lock_guard<std::mutex> lock(registry.mutex);
  const auto it = registry.completions.find(handle);
  if (it == registry.completions.end() || !it->second.valid())
    throw MemoryBindingError("Completion handle is closed or unknown");
  return it->second;
}

CompletionPublication::CompletionPublication() {
  auto &registry = handles();
  auto handle = reinterpret_cast<TiCompletion>(Runtime::new_handle_token());
  std::lock_guard<std::mutex> lock(registry.mutex);
  registry.completions.emplace(handle, RuntimeCompletion{});
  handle_ = handle;
}

CompletionPublication::~CompletionPublication() {
  if (handle_)
    (void)take_completion(handle_);
}

TiCompletion CompletionPublication::commit(RuntimeCompletion completion) {
  if (!completion.valid())
    throw std::logic_error("Cannot publish an invalid completion");
  auto &registry = handles();
  std::lock_guard<std::mutex> lock(registry.mutex);
  registry.completions.at(handle_) = std::move(completion);
  const auto result = handle_;
  handle_ = nullptr;
  return result;
}
}  // namespace capi

TiBool ti_query_completion_ext(TiCompletion completion) {
  TI_CAPI_TRY_CATCH_BEGIN();
  return capi::get_completion(completion).done();
  TI_CAPI_TRY_CATCH_END();
  return TI_FALSE;
}

void ti_wait_completion_ext(TiCompletion completion) {
  TI_CAPI_TRY_CATCH_BEGIN();
  capi::get_completion(completion).wait();
  TI_CAPI_TRY_CATCH_END();
}

void ti_get_completion_info_ext(TiCompletion completion,
                                TiCompletionInfo *info) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(info);
  *info = {};
  const auto ticket = capi::get_completion(completion);
  *info = {ticket.program_domain(), ticket.sequence(),
           static_cast<TiBool>(ticket.has_backend_work())};
  TI_CAPI_TRY_CATCH_END();
}

void ti_release_completion_ext(TiCompletion completion) {
  TI_CAPI_TRY_CATCH_BEGIN();
  (void)capi::take_completion(completion);
  TI_CAPI_TRY_CATCH_END();
}

void ti_get_completion_gpu_timing_ext(TiCompletion completion,
                                      TiCompletionGpuTiming *info) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(info);
  *info = {};
  const auto ticket = capi::get_completion(completion);
  info->owner = ticket.program_domain();
  info->sequence = ticket.sequence();
  info->scope = 1;  // Complete compute envelope, see the public contract.
  try {
    ticket.done();  // Nonblocking poll; freezes events before their reuse.
  } catch (...) {
    info->status = TI_COMPLETION_GPU_TIMING_FAILED;
    return;
  }
  const auto timing = ticket.gpu_timing_snapshot();
  if (timing.available) {
    info->status = TI_COMPLETION_GPU_TIMING_READY;
    info->duration_ns = timing.duration_ns;
  } else if (timing.status == "pending" || timing.status == "not_ready") {
    info->status = TI_COMPLETION_GPU_TIMING_PENDING;
  } else if (timing.status == "unavailable") {
    info->status = TI_COMPLETION_GPU_TIMING_NOT_SAMPLED;
  } else if (timing.status == "unsupported") {
    info->status = TI_COMPLETION_GPU_TIMING_UNSUPPORTED;
  } else {
    info->status = TI_COMPLETION_GPU_TIMING_FAILED;
  }
  TI_CAPI_TRY_CATCH_END();
}
