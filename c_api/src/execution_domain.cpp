#include "execution_domain.h"

#ifdef TI_WITH_CUDA
#include <algorithm>
#include <limits>
#include <mutex>
#include <unordered_map>
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/rhi/interop/cuda_external_interop.h"
#include "taichi/program/runtime_fault.h"
#include "taichi/runtime/cuda/execution_domain.h"

namespace capi {
namespace {
using taichi::lang::RuntimeCompletion;
using taichi::lang::cuda::ExecutionDomain;

struct Handles {
  std::mutex mutex;
  std::unordered_map<TiCudaExecutionDomain, std::shared_ptr<ExecutionDomain>>
      domains;
  std::unordered_map<TiCompletion, RuntimeCompletion> completions;
};

Handles &handles() {
  // Public handles have explicit release semantics. Do not run CUDA teardown
  // from a process-global destructor after CUDAContext has already retired.
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

std::shared_ptr<ExecutionDomain> get_execution_domain(
    TiCudaExecutionDomain handle) {
  auto &registry = handles();
  std::lock_guard<std::mutex> lock(registry.mutex);
  const auto it = registry.domains.find(handle);
  if (it == registry.domains.end() || !it->second)
    throw MemoryBindingError(
        "CUDA execution-domain handle is closed or unknown");
  return it->second;
}

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

TiMemory ti_import_cuda_external_memory_ext(
    TiRuntime runtime,
    const TiCudaExternalMemoryImportInfo *import_info) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL_RV(runtime);
  TI_CAPI_ARGUMENT_NULL_RV(import_info);
  using namespace taichi::lang;
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  TI_CAPI_NOT_SUPPORTED_IF_RV(rt.arch != taichi::Arch::cuda);
  rt.runtime_fault_domain()->throw_if_submission_disallowed(
      "import external CUDA memory");
  const auto &info = *import_info;
  if ((info.handle_type != TI_CUDA_EXTERNAL_HANDLE_OPAQUE_WIN32 &&
       info.handle_type != TI_CUDA_EXTERNAL_HANDLE_OPAQUE_FD) ||
      info.dedicated > 1 || !info.allocation_size || !info.byte_size ||
      info.allocation_size > std::numeric_limits<std::size_t>::max() ||
      info.byte_offset > info.allocation_size ||
      info.byte_size > info.allocation_size - info.byte_offset ||
      info.memory_handle > std::numeric_limits<std::uintptr_t>::max() ||
      info.ready_for_cuda > std::numeric_limits<std::uintptr_t>::max() ||
      info.ready_for_external > std::numeric_limits<std::uintptr_t>::max())
    throw MemoryBindingError("Invalid external CUDA memory import descriptor");
  const auto type = info.handle_type == TI_CUDA_EXTERNAL_HANDLE_OPAQUE_WIN32
                        ? ExternalOpaqueHandleType::kOpaqueWin32
                        : ExternalOpaqueHandleType::kOpaqueFd;
  CudaExternalMemoryImport memory;
  memory.handle = {type, static_cast<std::uintptr_t>(info.memory_handle)};
  memory.allocation_size = static_cast<std::size_t>(info.allocation_size);
  memory.dedicated = info.dedicated != 0;
  std::copy(std::begin(info.device_uuid), std::end(info.device_uuid),
            memory.device_uuid.begin());
  const CudaExternalSemaphorePairImport semaphores{
      {type, static_cast<std::uintptr_t>(info.ready_for_cuda)},
      {type, static_cast<std::uintptr_t>(info.ready_for_external)}};
  auto submission = CUDAContext::get_instance().get_submission_lock_guard();
  auto allocation = CudaExternalAllocation::create_borrowed(
      &static_cast<cuda::CudaDevice &>(rt.get()), memory, semaphores);
  auto storage = std::make_shared<Runtime::BorrowedMemory>();
  storage->owner = allocation;
  storage->external_access = allocation;
  return rt.register_memory(allocation->cuda_allocation(), info.byte_size,
                            false, info.byte_offset, {}, info.byte_offset,
                            false, std::move(storage));
  TI_CAPI_TRY_CATCH_END();
  return TI_NULL_HANDLE;
}

TiCudaExecutionDomain ti_create_cuda_execution_domain_ext(
    TiRuntime runtime,
    const TiCudaExecutionDomainCreateInfo *create_info) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL_RV(runtime);
  TI_CAPI_ARGUMENT_NULL_RV(create_info);
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  TI_CAPI_NOT_SUPPORTED_IF_RV(rt.arch != taichi::Arch::cuda);
  if (!create_info->capacity || !create_info->instance_capacity)
    throw MemoryBindingError(
        "CUDA execution-domain capacities must be positive");
  auto handle =
      reinterpret_cast<TiCudaExecutionDomain>(Runtime::new_handle_token());
  auto &registry = capi::handles();
  {
    std::lock_guard<std::mutex> lock(registry.mutex);
    registry.domains.emplace(handle, nullptr);
  }
  try {
    auto domain = rt.cuda_execution_domains()->create(
        {create_info->capacity, create_info->priority,
         create_info->instance_capacity});
    std::lock_guard<std::mutex> lock(registry.mutex);
    registry.domains.at(handle) = std::move(domain);
  } catch (...) {
    std::lock_guard<std::mutex> lock(registry.mutex);
    registry.domains.erase(handle);
    throw;
  }
  return handle;
  TI_CAPI_TRY_CATCH_END();
  return TI_NULL_HANDLE;
}

void ti_get_cuda_execution_domain_info_ext(TiCudaExecutionDomain domain,
                                           TiCudaExecutionDomainInfo *info) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(info);
  *info = {};
  const auto snapshot = capi::get_execution_domain(domain)->snapshot();
  *info = {snapshot.identity,
           snapshot.owner,
           snapshot.context,
           snapshot.stream,
           snapshot.requested_priority,
           snapshot.effective_priority,
           snapshot.least_priority,
           snapshot.greatest_priority,
           snapshot.capacity,
           snapshot.pending,
           static_cast<TiBool>(snapshot.accepting),
           static_cast<TiBool>(snapshot.closed)};
  TI_CAPI_TRY_CATCH_END();
}

void ti_stop_cuda_execution_domain_ext(TiCudaExecutionDomain domain) {
  TI_CAPI_TRY_CATCH_BEGIN();
  capi::get_execution_domain(domain)->stop_accepting();
  TI_CAPI_TRY_CATCH_END();
}

void ti_close_cuda_execution_domain_ext(TiCudaExecutionDomain domain) {
  TI_CAPI_TRY_CATCH_BEGIN();
  capi::get_execution_domain(domain)->close();
  TI_CAPI_TRY_CATCH_END();
}

void ti_release_cuda_execution_domain_ext(TiCudaExecutionDomain domain) {
  TI_CAPI_TRY_CATCH_BEGIN();
  std::shared_ptr<taichi::lang::cuda::ExecutionDomain> released;
  auto &registry = capi::handles();
  {
    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto it = registry.domains.find(domain);
    if (it == registry.domains.end())
      throw MemoryBindingError(
          "CUDA execution-domain handle is closed or unknown");
    released = std::move(it->second);
    registry.domains.erase(it);
  }
  TI_CAPI_TRY_CATCH_END();
}

uint32_t ti_collect_cuda_execution_domain_ext(TiCudaExecutionDomain domain,
                                              uint32_t budget) {
  TI_CAPI_TRY_CATCH_BEGIN();
  return static_cast<uint32_t>(
      capi::get_execution_domain(domain)->collect_ready(budget));
  TI_CAPI_TRY_CATCH_END();
  return 0;
}

TiCompletion ti_get_cuda_execution_domain_completion_ext(
    TiCudaExecutionDomain domain) {
  TI_CAPI_TRY_CATCH_BEGIN();
  auto completion = capi::get_execution_domain(domain)->latest_completion();
  return capi::CompletionPublication{}.commit(std::move(completion));
  TI_CAPI_TRY_CATCH_END();
  return TI_NULL_HANDLE;
}

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
#endif
