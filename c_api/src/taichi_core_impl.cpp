#include <algorithm>
#include <atomic>
#include <limits>
#include <unordered_set>
#include "argument_binding.h"

#include "taichi_core_impl.h"
#include "taichi_opengl_impl.h"
#include "taichi_vulkan_impl.h"
#include "taichi_llvm_impl.h"
#include "taichi_metal_impl.h"
#include "taichi/program/ndarray.h"
#include "taichi/program/texture.h"
#include "taichi/program/matrix.h"
#include "taichi/program/launch_context_builder.h"
#include "taichi/program/runtime_fault.h"
#include "taichi/common/virtual_dir.h"
#include "taichi/common/utils.h"
#ifdef TI_WITH_CUDA
#include "taichi/runtime/cuda/execution_domain.h"
#endif

bool is_vulkan_available() {
#ifdef TI_WITH_VULKAN
  return taichi::lang::vulkan::is_vulkan_api_available();
#else
  return false;
#endif
}

bool is_opengl_available() {
#ifdef TI_WITH_OPENGL
  return taichi::lang::opengl::is_opengl_api_available();
#else
  return false;
#endif
}

bool is_cuda_available() {
#ifdef TI_WITH_CUDA
  return taichi::is_cuda_api_available();
#else
  return false;
#endif
}

bool is_x64_available() {
#if defined(TI_WITH_LLVM) &&                                           \
    (defined(__x86_64__) || defined(__x86_64) || defined(__amd64__) || \
     defined(__amd64) || defined(_M_X64))
  return true;
#else
  return false;
#endif
}

bool is_arm64_available() {
#if defined(TI_WITH_LLVM) && (defined(__arm64__) || defined(__aarch64__))
  return true;
#else
  return false;
#endif
}

bool is_metal_available() {
#if defined(__APPLE__) && defined(TI_WITH_METAL)
  return taichi::lang::metal::is_metal_api_available();
#else
  return false;
#endif  // __APPLE__
}

struct ErrorCache {
  TiError error{TI_ERROR_SUCCESS};
  std::string message{};
};

namespace {
// Error is recorded on a per-thread basis.
thread_local ErrorCache thread_error_cache;

const char *describe_error(TiError error) {
  switch (error) {
    case TI_ERROR_SUCCESS:
      return "success";
    case TI_ERROR_NOT_SUPPORTED:
      return "not supported";
    case TI_ERROR_CORRUPTED_DATA:
      return "corrupted data";
    case TI_ERROR_NAME_NOT_FOUND:
      return "name not found";
    case TI_ERROR_INVALID_ARGUMENT:
      return "invalid argument";
    case TI_ERROR_ARGUMENT_NULL:
      return "argument null";
    case TI_ERROR_ARGUMENT_OUT_OF_RANGE:
      return "argument out of range";
    case TI_ERROR_ARGUMENT_NOT_FOUND:
      return "argument not found";
    case TI_ERROR_INVALID_INTEROP:
      return "invalid interop";
    case TI_ERROR_INVALID_STATE:
      return "invalid state";
    case TI_ERROR_INCOMPATIBLE_MODULE:
      return "incompatible module";
    case TI_ERROR_OUT_OF_MEMORY:
      return "out of memory";
    default:
      return "unknown error";
  }
}
}  // namespace

Runtime::Runtime(taichi::Arch arch)
    : runtime_fault_domain_(std::make_shared<taichi::lang::RuntimeFaultDomain>(
          arch, new_handle_token())),
      arch(arch) {
}
Runtime::~Runtime() {
  if (!borrowed_fault_domain_)
    runtime_fault_domain_->mark_finalized();
}

void Runtime::borrow_runtime_fault_domain(
    std::shared_ptr<taichi::lang::RuntimeFaultDomain> domain,
    std::shared_ptr<taichi::lang::cuda::ExecutionDomainRegistry> domains) {
  std::lock_guard<std::recursive_mutex> lock(submission_mutex());
  if (!domain || domain->backend() != arch || prepared_ || native_resources_ ||
      !modules_.empty() || !memories_.empty() || borrowed_fault_domain_ ||
      cuda_execution_domains_ || execution_domains_retiring_ ||
      (arch == taichi::Arch::cuda && !domains))
    throw std::logic_error("Runtime fault domain must be bound before use");
#ifdef TI_WITH_CUDA
  if (domains && !domains->belongs_to(domain))
    throw std::invalid_argument(
        "CUDA execution domains belong to another runtime");
#endif
  runtime_fault_domain_ = std::move(domain);
  cuda_execution_domains_ = std::move(domains);
  borrowed_fault_domain_ = true;
}

std::shared_ptr<taichi::lang::cuda::ExecutionDomainRegistry>
Runtime::cuda_execution_domains() {
#ifdef TI_WITH_CUDA
  std::lock_guard<std::recursive_mutex> lock(submission_mutex());
  runtime_fault_domain_->throw_if_new_submission_disallowed(
      "access CUDA execution domains");
  if (arch != taichi::Arch::cuda || execution_domains_retiring_)
    throw std::runtime_error("CUDA execution-domain owner is unavailable");
  if (!cuda_execution_domains_)
    cuda_execution_domains_ =
        std::make_shared<taichi::lang::cuda::ExecutionDomainRegistry>(
            runtime_fault_domain_);
  return cuda_execution_domains_;
#else
  throw std::runtime_error("CUDA support is not compiled in");
#endif
}

void Runtime::synchronize_execution_domains() {
#ifdef TI_WITH_CUDA
  std::shared_ptr<taichi::lang::cuda::ExecutionDomainRegistry> domains;
  {
    std::lock_guard<std::recursive_mutex> lock(submission_mutex());
    domains = cuda_execution_domains_;
  }
  if (domains)
    domains->synchronize();
#endif
}

void Runtime::retire_execution_domains() {
#ifdef TI_WITH_CUDA
  std::shared_ptr<taichi::lang::cuda::ExecutionDomainRegistry> domains;
  {
    std::lock_guard<std::recursive_mutex> lock(submission_mutex());
    if (borrowed_fault_domain_)
      return;
    execution_domains_retiring_ = true;
    domains = cuda_execution_domains_;
  }
  if (domains)
    domains->retire();
#endif
}

VulkanRuntime *Runtime::as_vk() {
  TI_ASSERT(arch == taichi::Arch::vulkan);
#ifdef TI_WITH_VULKAN
  return static_cast<VulkanRuntime *>(this);
#else
  return nullptr;
#endif
}
capi::MetalRuntime *Runtime::as_mtl() {
  TI_ASSERT(arch == taichi::Arch::metal);
#ifdef TI_WITH_METAL
  return static_cast<capi::MetalRuntime *>(this);
#else
  return nullptr;
#endif
}

TiMemory Runtime::allocate_memory(
    const taichi::lang::Device::AllocParams &params) {
  taichi::lang::DeviceAllocation devalloc;
  auto allocation_params = params;
  if (arch == taichi::Arch::vulkan &&
      int(params.usage & taichi::lang::AllocUsage::Storage)) {
    // Like JIT ndarrays, AOT storage can carry GPU-produced dispatch packets.
    // This changes only buffer usage, without staging or extra allocation.
    allocation_params.usage =
        params.usage | taichi::lang::AllocUsage::Indirect;
  }
  taichi::lang::RhiResult res =
      this->get().allocate_memory(allocation_params, &devalloc);
  if (res != taichi::lang::RhiResult::success) {
    throw std::bad_alloc();
  }
  return devalloc2devmem(*this, devalloc, params.size);
}
void Runtime::free_memory(TiMemory devmem) {
  std::lock_guard<std::recursive_mutex> submission_lock(submission_mutex());
  retire_native_memory(devmem);
  retire_prepared_memory(devmem);
  std::shared_ptr<void> released;
  std::shared_ptr<taichi::lang::ExternalSynchronizationDomain> released_access;
  {
    std::lock_guard<std::mutex> lock(memory_mutex_);
    const auto found = memories_.find(devmem);
    if (found == memories_.end()) {
      throw MemoryBindingError(
          "Memory is not a live allocation of this runtime");
    }
    if (found->second.owns_allocation)
      this->get().dealloc_memory(found->second.allocation);
    const auto borrowed = found->second.borrowed;
    memories_.erase(found);
    if (borrowed && std::none_of(memories_.begin(), memories_.end(),
                                 [&](const auto &entry) {
                                   return entry.second.borrowed == borrowed;
                                 })) {
      borrowed->live = false;
      released_access = std::move(borrowed->external_access);
      released = std::move(borrowed->owner);
    }
  }
}

TiMemory Runtime::register_memory(taichi::lang::DeviceAllocation allocation,
                                  uint64_t size,
                                  bool release_on_failure,
                                  uint64_t base_offset,
                                  std::shared_ptr<void> storage_owner,
                                  uint64_t byte_offset,
                                  bool owns_allocation,
                                  std::shared_ptr<BorrowedMemory> borrowed) {
  TI_ASSERT(allocation.device == &get());
  try {
    if (byte_offset > std::numeric_limits<size_t>::max() ||
        size > std::numeric_limits<size_t>::max() - byte_offset)
      throw MemoryBindingError("Memory range overflows device address space");
    // Never reuse a token, including across runtime destruction. Refuse the
    // machine's exhausted handle space instead of wrapping into a live token.
    const auto handle = reinterpret_cast<TiMemory>(new_handle_token());
    std::lock_guard<std::mutex> lock(memory_mutex_);
    memories_.emplace(
        handle, MemoryAllocation{
                    allocation, size, base_offset, storage_owner, byte_offset,
                    owns_allocation && !storage_owner && !borrowed, borrowed});
    return handle;
  } catch (...) {
    // RHI import paths allocate their own wrapper too. Retire that wrapper;
    // the backend retains ownership rules for the external storage itself.
    if (release_on_failure && owns_allocation && !storage_owner && !borrowed)
      get().dealloc_memory(allocation);
    throw;
  }
}

Runtime::MemoryAllocation Runtime::get_memory(
    TiMemory memory,
    bool prepared_external_access) const {
  std::lock_guard<std::mutex> lock(memory_mutex_);
  const auto found = memories_.find(memory);
  if (found == memories_.end()) {
    throw MemoryBindingError("Memory is not a live allocation of this runtime");
  }
  if (found->second.borrowed && !found->second.borrowed->live)
    throw MemoryBindingError("Borrowed storage has retired with its owner");
  if (!prepared_external_access && found->second.borrowed &&
      found->second.borrowed->external_access)
    throw MemoryBindingError(
        "Synchronized external storage requires a prepared execution domain");
  return found->second;
}

void Runtime::retire_borrowed_memory(
    const std::shared_ptr<BorrowedMemory> &borrowed) {
  std::lock_guard<std::recursive_mutex> submission_lock(submission_mutex());
  if (!borrowed->live)
    return;
  std::vector<TiMemory> handles;
  {
    std::lock_guard<std::mutex> lock(memory_mutex_);
    for (const auto &[handle, memory] : memories_)
      if (memory.borrowed == borrowed)
        handles.push_back(handle);
  }
  // Do not revoke the generation or release backing ownership on a failed
  // drain. Retrying the original Program retirement remains safe.
  for (auto handle : handles) {
    retire_native_memory(handle);
    retire_prepared_memory(handle);
  }
  std::shared_ptr<void> released;
  std::shared_ptr<taichi::lang::ExternalSynchronizationDomain> released_access;
  {
    std::lock_guard<std::mutex> lock(memory_mutex_);
    borrowed->live = false;
    released_access = std::move(borrowed->external_access);
    released = std::move(borrowed->owner);
  }
  // Host handle shells survive invalidation, so late close is idempotent.
  // The owner's destructor may detach its Program callback outside this lock.
}

AotModule::AotModule(Runtime &runtime,
                     std::unique_ptr<taichi::lang::aot::Module> aot_module)
    : runtime_(&runtime), aot_module_(std::move(aot_module)) {
  runtime.register_module(this);
}

taichi::lang::aot::Kernel *AotModule::get_kernel(const std::string &name) {
  check_code_live();
  return aot_module_->get_kernel(name);
}
taichi::lang::aot::CompiledGraph *AotModule::get_cgraph(
    const std::string &name) {
  check_code_live();
  auto it = loaded_cgraphs_.find(name);
  if (it == loaded_cgraphs_.end()) {
    return loaded_cgraphs_
        .emplace(std::make_pair(name, aot_module_->get_graph(name)))
        .first->second.get();
  } else {
    return it->second.get();
  }
}
taichi::lang::aot::Module &AotModule::get() {
  check_code_live();
  return *aot_module_;
}
Runtime &AotModule::runtime() {
  return *runtime_;
}

void AotModule::check_code_live() const {
  if (!code_live_)
    throw MemoryBindingError(
        "Borrowed JIT code has retired with its SNode tree");
}

void AotModule::retire_borrowed_code() {
  if (!code_live_)
    return;
  runtime_->retire_prepared_module(this);
  runtime_->invalidate_module_entries(this);
  code_live_ = false;
}

// -----------------------------------------------------------------------------

uint32_t ti_get_version() {
  return TI_C_API_VERSION;
}

void ti_get_available_archs(uint32_t *arch_count, TiArch *archs) {
  if (arch_count == nullptr) {
    return;
  }

  thread_local std::vector<TiArch> AVAILABLE_ARCHS{};
  if (AVAILABLE_ARCHS.empty()) {
    if (is_vulkan_available()) {
      AVAILABLE_ARCHS.emplace_back(TI_ARCH_VULKAN);
    }
    if (is_metal_available()) {
      AVAILABLE_ARCHS.emplace_back(TI_ARCH_METAL);
    }
    if (is_cuda_available()) {
      AVAILABLE_ARCHS.emplace_back(TI_ARCH_CUDA);
    }
    if (is_x64_available()) {
      AVAILABLE_ARCHS.emplace_back(TI_ARCH_X64);
    }
    if (is_arm64_available()) {
      AVAILABLE_ARCHS.emplace_back(TI_ARCH_ARM64);
    }
    if (is_opengl_available()) {
      AVAILABLE_ARCHS.emplace_back(TI_ARCH_OPENGL);
    }
  }

  size_t n = std::min((size_t)*arch_count, AVAILABLE_ARCHS.size());
  *arch_count = (uint32_t)AVAILABLE_ARCHS.size();
  if (archs != nullptr) {
    for (size_t i = 0; i < n; ++i) {
      archs[i] = AVAILABLE_ARCHS.at(i);
    }
  }
}

TiError ti_get_last_error(uint64_t *message_size, char *message) {
  TiError out = TI_ERROR_INVALID_STATE;
  TI_CAPI_TRY_CATCH_BEGIN();
  out = thread_error_cache.error;

  // Emit message only if the output buffer is property provided.
  if (message_size == nullptr) {
    return out;
  }
  size_t buffer_size = *message_size;
  *message_size = thread_error_cache.message.size() + 1;

  if (buffer_size > 0 && message != nullptr) {
    // -1 for the byte of `\0`.
    size_t n = std::min(thread_error_cache.message.size(), buffer_size - 1);
    std::memcpy(message, thread_error_cache.message.data(), n);
    message[n] = '\0';
  }
  TI_CAPI_TRY_CATCH_END();
  return out;
}
// C-API errors MUST be set via this interface. No matter from internal or
// external procedures.
void ti_set_last_error(TiError error, const char *message) {
  TI_CAPI_TRY_CATCH_BEGIN();
  if (error < TI_ERROR_SUCCESS) {
    TI_WARN("C-API error: ({}) {}", describe_error(error), message);
    if (message != nullptr) {
      thread_error_cache.message = message;
    } else {
      thread_error_cache.message.clear();
    }
    thread_error_cache.error = error;
  } else {
    thread_error_cache.error = TI_ERROR_SUCCESS;
    thread_error_cache.message.clear();
  }
  TI_CAPI_TRY_CATCH_END();
}

TiRuntime ti_create_runtime(TiArch arch, uint32_t device_index) {
  TiRuntime out = TI_NULL_HANDLE;
  TI_CAPI_TRY_CATCH_BEGIN();
  // TI_INFO("Taichi Runtime C-API version is: {}", TI_C_API_VERSION);
  switch (arch) {
#ifdef TI_WITH_VULKAN
    case TI_ARCH_VULKAN: {
      taichi::lang::vulkan::set_vulkan_visible_device(
          std::to_string(device_index));
      VulkanRuntimeOwned *vulkan_runtime;
      if (is_ci()) {
        auto param = make_vulkan_runtime_creator_params();
        param.enable_validation_layer = true;
        vulkan_runtime = new VulkanRuntimeOwned(std::move(param));
      } else {
        vulkan_runtime = new VulkanRuntimeOwned;
      }
      out = (TiRuntime)(static_cast<Runtime *>(vulkan_runtime));
      break;
    }
#endif  // TI_WITH_VULKAN
#ifdef TI_WITH_OPENGL
    case TI_ARCH_OPENGL: {
      TI_CAPI_NOT_SUPPORTED_IF_RV(device_index != 0);
      out = (TiRuntime)(static_cast<Runtime *>(new OpenglRuntime));
      break;
    }
#endif  // TI_WITH_OPENGL
#ifdef TI_WITH_LLVM
    case TI_ARCH_X64: {
      TI_CAPI_NOT_SUPPORTED_IF_RV(device_index != 0);
      out = (TiRuntime)(static_cast<Runtime *>(
          new capi::LlvmRuntime(taichi::Arch::x64)));
      break;
    }
    case TI_ARCH_ARM64: {
      TI_CAPI_NOT_SUPPORTED_IF_RV(device_index != 0);
      out = (TiRuntime)(static_cast<Runtime *>(
          new capi::LlvmRuntime(taichi::Arch::arm64)));
      break;
    }
    case TI_ARCH_CUDA: {
      TI_CAPI_NOT_SUPPORTED_IF_RV(device_index != 0);
      out = (TiRuntime)(static_cast<Runtime *>(
          new capi::LlvmRuntime(taichi::Arch::cuda)));
      break;
    }
#endif  // TI_WITH_LLVM
#ifdef TI_WITH_METAL
    case TI_ARCH_METAL: {
      TI_CAPI_NOT_SUPPORTED_IF_RV(device_index != 0);
      out = (TiRuntime)(static_cast<Runtime *>(new capi::MetalRuntime()));
      break;
    }
#endif  // TI_WITH_METAL
    default: {
      TI_CAPI_NOT_SUPPORTED(arch);
      return TI_NULL_HANDLE;
    }
  }
  TI_CAPI_TRY_CATCH_END();
  return out;
}
void ti_destroy_runtime(TiRuntime runtime) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);
  ((Runtime *)runtime)->retire_execution_domains();
  ((Runtime *)runtime)->clear_prepared();
  ((Runtime *)runtime)->clear_native_resources();
  delete (Runtime *)runtime;
  TI_CAPI_TRY_CATCH_END();
}

void ti_set_runtime_capabilities_ext(
    TiRuntime runtime,
    uint32_t capability_count,
    const TiCapabilityLevelInfo *capabilities) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);

  Runtime *runtime2 = (Runtime *)runtime;
  taichi::lang::DeviceCapabilityConfig devcaps;
  if (capability_count && !capabilities) {
    ti_set_last_error(TI_ERROR_ARGUMENT_NULL, "capabilities");
    return;
  }
  for (uint32_t i = 0; i < capability_count; ++i) {
    const auto &cap_level_info = capabilities[i];
    switch (cap_level_info.capability) {
#define PER_CAPABILITY(public_name, internal_name)             \
  case TI_CAPABILITY_##public_name:                            \
    devcaps.set(taichi::lang::DeviceCapability::internal_name, \
                cap_level_info.level);                         \
    break;
#include "device_capability.inc.h"
#undef PER_CAPABILITY
      default:
        ti_set_last_error(TI_ERROR_ARGUMENT_OUT_OF_RANGE,
                          "Unknown public device capability");
        return;
    }
  }
  runtime2->get().set_caps(std::move(devcaps));

  TI_CAPI_TRY_CATCH_END();
}

void ti_get_runtime_capabilities(TiRuntime runtime,
                                 uint32_t *capability_count,
                                 TiCapabilityLevelInfo *capabilities) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);

  Runtime *runtime2 = (Runtime *)runtime;
  const taichi::lang::DeviceCapabilityConfig &devcaps =
      runtime2->get().get_caps();

  if (capability_count == nullptr) {
    return;
  }

  uint32_t count = 0;
  for (const auto &entry : devcaps.to_inner()) {
    TiCapability capability;
    switch (entry.first) {
#define PER_CAPABILITY(public_name, internal_name)    \
  case taichi::lang::DeviceCapability::internal_name: \
    capability = TI_CAPABILITY_##public_name;         \
    break;
#include "device_capability.inc.h"
#undef PER_CAPABILITY
      default:
        continue;  // Internal-only capabilities have no public ABI entry.
    }
    if (capabilities != nullptr && count < *capability_count)
      capabilities[count] = {capability, entry.second};
    ++count;
  }

  *capability_count = count;

  TI_CAPI_TRY_CATCH_END();
}

TiMemory ti_allocate_memory(TiRuntime runtime,
                            const TiMemoryAllocateInfo *create_info) {
  TiMemory out = TI_NULL_HANDLE;
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL_RV(runtime);
  TI_CAPI_ARGUMENT_NULL_RV(create_info);

  taichi::lang::AllocUsage usage{};
  if (create_info->usage & TI_MEMORY_USAGE_STORAGE_BIT) {
    usage = usage | taichi::lang::AllocUsage::Storage;
  }
  if (create_info->usage & TI_MEMORY_USAGE_UNIFORM_BIT) {
    usage = usage | taichi::lang::AllocUsage::Uniform;
  }
  if (create_info->usage & TI_MEMORY_USAGE_VERTEX_BIT) {
    usage = usage | taichi::lang::AllocUsage::Vertex;
  }
  if (create_info->usage & TI_MEMORY_USAGE_INDEX_BIT) {
    usage = usage | taichi::lang::AllocUsage::Index;
  }
  if (create_info->usage & TI_MEMORY_USAGE_INDIRECT_BIT) {
    usage = usage | taichi::lang::AllocUsage::Indirect;
  }

  taichi::lang::Device::AllocParams params{};
  params.size = create_info->size;
  params.host_write = create_info->host_write;
  params.host_read = create_info->host_read;
  params.export_sharing = create_info->export_sharing;
  params.usage = usage;

  try {
    out = ((Runtime *)runtime)->allocate_memory(params);
  } catch (const std::bad_alloc &e) {
    ti_set_last_error(TI_ERROR_OUT_OF_MEMORY, "allocate_memory");
    return TI_NULL_HANDLE;
  }
  TI_CAPI_TRY_CATCH_END();
  return out;
}

void ti_free_memory(TiRuntime runtime, TiMemory devmem) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(devmem);

  Runtime *runtime2 = (Runtime *)runtime;
  runtime2->free_memory(devmem);
  TI_CAPI_TRY_CATCH_END();
}

void *ti_map_memory(TiRuntime runtime, TiMemory devmem) {
  void *out = nullptr;
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL_RV(runtime);
  TI_CAPI_ARGUMENT_NULL_RV(devmem);

  Runtime *runtime2 = (Runtime *)runtime;
  const auto memory = runtime2->get_memory(devmem);
  TI_ASSERT(runtime2->get().map(memory.allocation, &out) ==
                taichi::lang::RhiResult::success &&
            "RHI map memory failed: "
            "Mapping Memory without VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT set");
  if (out)
    out = static_cast<char *>(out) + memory.byte_offset;
  TI_CAPI_TRY_CATCH_END();
  return out;
}
void ti_unmap_memory(TiRuntime runtime, TiMemory devmem) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(devmem);

  Runtime *runtime2 = (Runtime *)runtime;
  runtime2->get().unmap(devmem2devalloc(*runtime2, devmem));
  TI_CAPI_TRY_CATCH_END();
}

TiImage ti_allocate_image(TiRuntime runtime,
                          const TiImageAllocateInfo *allocate_info) {
  TiImage out = TI_NULL_HANDLE;
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL_RV(runtime);
  TI_CAPI_ARGUMENT_NULL_RV(allocate_info);

  TI_CAPI_NOT_SUPPORTED_IF_RV(allocate_info->mip_level_count > 1);
  TI_CAPI_NOT_SUPPORTED_IF_RV(allocate_info->extent.array_layer_count > 1);

  taichi::lang::ImageAllocUsage usage{};
  if (allocate_info->usage & TI_IMAGE_USAGE_STORAGE_BIT) {
    usage = usage | taichi::lang::ImageAllocUsage::Storage;
  }
  if (allocate_info->usage & TI_IMAGE_USAGE_SAMPLED_BIT) {
    usage = usage | taichi::lang::ImageAllocUsage::Sampled;
  }
  if (allocate_info->usage & TI_IMAGE_USAGE_ATTACHMENT_BIT) {
    usage = usage | taichi::lang::ImageAllocUsage::Attachment;
  }

  switch ((taichi::lang::ImageDimension)allocate_info->dimension) {
#define PER_IMAGE_DIMENSION(x) case taichi::lang::ImageDimension::x:
#include "taichi/inc/rhi_constants.inc.h"
#undef PER_IMAGE_DIMENSION
    break;
    default: {
      ti_set_last_error(TI_ERROR_ARGUMENT_OUT_OF_RANGE,
                        "allocate_info->dimension");
      return TI_NULL_HANDLE;
    }
  }

  switch ((taichi::lang::BufferFormat)allocate_info->format) {
#define PER_BUFFER_FORMAT(x) case taichi::lang::BufferFormat::x:
#include "taichi/inc/rhi_constants.inc.h"
#undef PER_BUFFER_FORMAT
    break;
    default: {
      ti_set_last_error(TI_ERROR_ARGUMENT_OUT_OF_RANGE,
                        "allocate_info->format");
      return TI_NULL_HANDLE;
    }
  }

  taichi::lang::ImageParams params{};
  params.x = allocate_info->extent.width;
  params.y = allocate_info->extent.height;
  params.z = allocate_info->extent.depth;
  params.dimension = (taichi::lang::ImageDimension)allocate_info->dimension;
  params.format = (taichi::lang::BufferFormat)allocate_info->format;
  params.export_sharing = false;
  params.usage = usage;

  out = ((Runtime *)runtime)->allocate_image(params);
  TI_CAPI_TRY_CATCH_END();
  return out;
}
void ti_free_image(TiRuntime runtime, TiImage image) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(image);

  ((Runtime *)runtime)->free_image(image);
  TI_CAPI_TRY_CATCH_END();
}

TiSampler ti_create_sampler(TiRuntime runtime,
                            const TiSamplerCreateInfo *create_info) {
  TiSampler out = TI_NULL_HANDLE;
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_NOT_SUPPORTED(ti_create_sampler);
  TI_CAPI_TRY_CATCH_END();
  return out;
}
void ti_destroy_sampler(TiRuntime runtime, TiSampler sampler) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_NOT_SUPPORTED(ti_destroy_sampler);
  TI_CAPI_TRY_CATCH_END();
}

namespace {
taichi::lang::DevicePtr resolve_memory_slice(Runtime &runtime,
                                             const TiMemorySlice &slice) {
  const auto allocation = runtime.get_memory(slice.memory);
  if (slice.offset > allocation.size ||
      slice.size > allocation.size - slice.offset ||
      slice.size > std::numeric_limits<size_t>::max()) {
    throw MemoryBindingError("Memory slice exceeds its allocation capacity");
  }
  return allocation.allocation.get_ptr(allocation.byte_offset + slice.offset);
}

void check_transfer_result(taichi::lang::RhiResult result) {
  using taichi::lang::RhiResult;
  switch (result) {
    case RhiResult::success:
      break;
    case RhiResult::out_of_memory:
      ti_set_last_error(TI_ERROR_OUT_OF_MEMORY,
                        "Host transfer allocation failed");
      break;
    case RhiResult::not_supported:
      ti_set_last_error(TI_ERROR_NOT_SUPPORTED,
                        "Host transfer is not supported");
      break;
    case RhiResult::invalid_usage:
      ti_set_last_error(TI_ERROR_INVALID_ARGUMENT, "Invalid host transfer");
      break;
    default:
      ti_set_last_error(TI_ERROR_INVALID_STATE, "Host transfer failed");
      break;
  }
}
}  // namespace

void ti_copy_memory_from_host_ext(TiRuntime runtime,
                                  const TiMemorySlice *dst_memory,
                                  const void *src) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(dst_memory);
  TI_CAPI_ARGUMENT_NULL(dst_memory->memory);
  auto &runtime2 = *reinterpret_cast<Runtime *>(runtime);
  auto dst = resolve_memory_slice(runtime2, *dst_memory);
  if (dst_memory->size == 0)
    return;
  TI_CAPI_ARGUMENT_NULL(src);
  size_t size = dst_memory->size;
  runtime2.wait();
  check_transfer_result(runtime2.get().upload_data(&dst, &src, &size));
  TI_CAPI_TRY_CATCH_END();
}

void ti_copy_memory_to_host_ext(TiRuntime runtime,
                                void *dst,
                                const TiMemorySlice *src_memory) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(src_memory);
  TI_CAPI_ARGUMENT_NULL(src_memory->memory);
  auto &runtime2 = *reinterpret_cast<Runtime *>(runtime);
  auto src = resolve_memory_slice(runtime2, *src_memory);
  if (src_memory->size == 0)
    return;
  TI_CAPI_ARGUMENT_NULL(dst);
  size_t size = src_memory->size;
  runtime2.wait();
  check_transfer_result(runtime2.get().readback_data(&src, &dst, &size));
  TI_CAPI_TRY_CATCH_END();
}

void ti_copy_memory_device_to_device(TiRuntime runtime,
                                     const TiMemorySlice *dst_memory,
                                     const TiMemorySlice *src_memory) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(dst_memory);
  TI_CAPI_ARGUMENT_NULL(dst_memory->memory);
  TI_CAPI_ARGUMENT_NULL(src_memory);
  TI_CAPI_ARGUMENT_NULL(src_memory->memory);
  if (dst_memory->size != src_memory->size) {
    ti_set_last_error(TI_ERROR_INVALID_ARGUMENT,
                      "The size of memory slices are not match");
    return;
  }

  Runtime *runtime2 = (Runtime *)runtime;
  auto dst = resolve_memory_slice(*runtime2, *dst_memory);
  auto src = resolve_memory_slice(*runtime2, *src_memory);
  if (dst_memory->size == 0)
    return;
  if (dst.alloc_id == src.alloc_id &&
      dst.offset < src.offset + src_memory->size &&
      src.offset < dst.offset + dst_memory->size) {
    throw MemoryBindingError("Device memory copy slices must not overlap");
  }
  runtime2->buffer_copy(dst, src, dst_memory->size);
  TI_CAPI_TRY_CATCH_END();
}

void ti_copy_image_device_to_device(TiRuntime runtime,
                                    const TiImageSlice *dst_image,
                                    const TiImageSlice *src_image) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(dst_image);
  TI_CAPI_ARGUMENT_NULL(dst_image->image);
  TI_CAPI_ARGUMENT_NULL(src_image);
  TI_CAPI_ARGUMENT_NULL(src_image->image);
  TI_CAPI_INVALID_ARGUMENT(src_image->extent.width != dst_image->extent.width);
  TI_CAPI_INVALID_ARGUMENT(src_image->extent.height !=
                           dst_image->extent.height);
  TI_CAPI_INVALID_ARGUMENT(src_image->extent.depth != dst_image->extent.depth);
  TI_CAPI_INVALID_ARGUMENT(src_image->extent.array_layer_count !=
                           dst_image->extent.array_layer_count);

  Runtime *runtime2 = (Runtime *)runtime;
  auto dst = devimg2devalloc(*runtime2, dst_image->image);
  auto src = devimg2devalloc(*runtime2, src_image->image);

  taichi::lang::ImageCopyParams params{};
  params.width = dst_image->extent.width;
  params.height = dst_image->extent.height;
  params.depth = dst_image->extent.depth;
  runtime2->copy_image(dst, src, params);
  TI_CAPI_TRY_CATCH_END();
}
void ti_track_image_ext(TiRuntime runtime,
                        TiImage image,
                        TiImageLayout layout) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(image);

  Runtime *runtime2 = (Runtime *)runtime;
  auto image2 = devimg2devalloc(*runtime2, image);
  auto layout2 = (taichi::lang::ImageLayout)layout;

  switch ((taichi::lang::ImageLayout)layout) {
#define PER_IMAGE_LAYOUT(x) case taichi::lang::ImageLayout::x:
#include "taichi/inc/rhi_constants.inc.h"
#undef PER_IMAGE_LAYOUT
    break;
    default: {
      ti_set_last_error(TI_ERROR_ARGUMENT_OUT_OF_RANGE, "layout");
      return;
    }
  }

  runtime2->track_image(image2, layout2);
  TI_CAPI_TRY_CATCH_END();
}
void ti_transition_image(TiRuntime runtime,
                         TiImage texture,
                         TiImageLayout layout) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(texture);

  Runtime *runtime2 = (Runtime *)runtime;
  auto image = devimg2devalloc(*runtime2, texture);
  auto layout2 = (taichi::lang::ImageLayout)layout;

  switch ((taichi::lang::ImageLayout)layout) {
#define PER_IMAGE_LAYOUT(x) case taichi::lang::ImageLayout::x:
#include "taichi/inc/rhi_constants.inc.h"
#undef PER_IMAGE_LAYOUT
    break;
    default: {
      ti_set_last_error(TI_ERROR_ARGUMENT_OUT_OF_RANGE, "layout");
      return;
    }
  }

  runtime2->transition_image(image, layout2);
  TI_CAPI_TRY_CATCH_END();
}

TiAotModule ti_load_aot_module(TiRuntime runtime, const char *module_path) {
  TiAotModule out = TI_NULL_HANDLE;
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL_RV(runtime);
  TI_CAPI_ARGUMENT_NULL_RV(module_path);

  // (penguinliong) Should call `create_aot_module` directly after all backends
  // adapted to it.
  TiAotModule aot_module = ((Runtime *)runtime)->load_aot_module(module_path);

  // The backend records the specific failure (for example an unmet device
  // capability). Do not overwrite it with a generic corrupted-data error.
  out = aot_module;
  TI_CAPI_TRY_CATCH_END();
  return out;
}
TiAotModule ti_create_aot_module(TiRuntime runtime,
                                 const void *tcm,
                                 uint64_t size) {
  TiAotModule out = TI_NULL_HANDLE;
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL_RV(runtime);
  TI_CAPI_ARGUMENT_NULL_RV(tcm);

  auto dir = taichi::io::VirtualDir::from_zip(tcm, size);
  if (dir == TI_NULL_HANDLE) {
    ti_set_last_error(TI_ERROR_CORRUPTED_DATA, "tcm");
    return TI_NULL_HANDLE;
  }

  Error err = ((Runtime *)runtime)->create_aot_module(std::move(dir), out);
  err.set_last_error();

  TI_CAPI_TRY_CATCH_END();
  return out;
}
void ti_destroy_aot_module(TiAotModule aot_module) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(aot_module);

  auto *module = reinterpret_cast<AotModule *>(aot_module);
  std::lock_guard<std::recursive_mutex> lock(
      module->runtime().submission_mutex());
  module->runtime().retire_prepared_module(module);
  delete module;
  TI_CAPI_TRY_CATCH_END();
}

TiKernel ti_get_aot_module_kernel(TiAotModule aot_module, const char *name) {
  TiKernel out = TI_NULL_HANDLE;
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL_RV(aot_module);
  TI_CAPI_ARGUMENT_NULL_RV(name);

  TiKernel kernel = TI_NULL_HANDLE;
  try {
    kernel = ((AotModule *)aot_module)->get_kernel_handle(name);
  } catch (const taichi::lang::aot::ArtifactLoadError &e) {
    ti_set_last_error(TI_ERROR_CORRUPTED_DATA, e.what());
    return TI_NULL_HANDLE;
  }

  if (kernel == nullptr) {
    ti_set_last_error(TI_ERROR_NAME_NOT_FOUND, name);
    return TI_NULL_HANDLE;
  }

  out = kernel;
  TI_CAPI_TRY_CATCH_END();
  return out;
}

TiComputeGraph ti_get_aot_module_compute_graph(TiAotModule aot_module,
                                               const char *name) {
  TiComputeGraph out = TI_NULL_HANDLE;
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL_RV(aot_module);
  TI_CAPI_ARGUMENT_NULL_RV(name);

  TiComputeGraph cgraph = TI_NULL_HANDLE;
  try {
    cgraph = ((AotModule *)aot_module)->get_graph_handle(name);
  } catch (const taichi::lang::aot::ArtifactLoadError &e) {
    ti_set_last_error(TI_ERROR_CORRUPTED_DATA, e.what());
    return TI_NULL_HANDLE;
  }

  if (cgraph == nullptr) {
    ti_set_last_error(TI_ERROR_NAME_NOT_FOUND, name);
    return TI_NULL_HANDLE;
  }

  out = cgraph;
  TI_CAPI_TRY_CATCH_END();
  return out;
}


namespace {
bool gradient_error(const std::string &message) {
  ti_set_last_error(TI_ERROR_INVALID_ARGUMENT, message.c_str());
  return false;
}
}  // namespace

void ti_launch_kernel(TiRuntime runtime,
                      TiKernel kernel,
                      uint32_t arg_count,
                      const TiArgument *args) {
  ti_launch_kernel_with_gradients_ext(runtime, kernel, arg_count, args, 0, nullptr);
}

void ti_launch_kernel_with_gradients_ext(
    TiRuntime runtime, TiKernel kernel, uint32_t arg_count,
    const TiArgument *args, uint32_t gradient_count,
    const TiNdArrayGradientBinding *gradients) {
  TI_CAPI_TRY_CATCH_BEGIN();
  if (gradient_count) TI_CAPI_ARGUMENT_NULL(gradients);
  if (gradient_count > arg_count) {
    gradient_error("Too many gradient bindings");
    return;
  }
  std::vector<TiNdArrayBinding> bindings;
  bindings.reserve(gradient_count);
  for (uint32_t i = 0; i < gradient_count; ++i) {
    bindings.push_back({gradients[i].argument_index, 0, TI_NULL_HANDLE,
                        &gradients[i].gradient});
  }
  ti_launch_kernel_with_bindings_ext(runtime, kernel, arg_count, args,
                                     gradient_count, bindings.data());
  TI_CAPI_TRY_CATCH_END();
}

void ti_launch_kernel_with_bindings_ext(
    TiRuntime runtime, TiKernel kernel, uint32_t arg_count,
    const TiArgument *args, uint32_t binding_count,
    const TiNdArrayBinding *bindings) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  auto bound = capi::bind_kernel_arguments(rt, kernel, arg_count, args,
                                           binding_count, bindings);
  if (bound)
    bound->kernel->launch(*bound->context);
  TI_CAPI_TRY_CATCH_END();
}

void ti_launch_compute_graph(TiRuntime runtime,
                             TiComputeGraph compute_graph,
                             uint32_t arg_count,
                             const TiNamedArgument *args) {
  ti_launch_compute_graph_with_gradients_ext(runtime, compute_graph, arg_count,
                                            args, 0, nullptr);
}

void ti_launch_compute_graph_with_gradients_ext(
    TiRuntime runtime, TiComputeGraph compute_graph, uint32_t arg_count,
    const TiNamedArgument *args, uint32_t gradient_count,
    const TiNamedNdArrayGradientBinding *gradients) {
  TI_CAPI_TRY_CATCH_BEGIN();
  if (gradient_count) TI_CAPI_ARGUMENT_NULL(gradients);
  if (gradient_count > arg_count) {
    gradient_error("Too many Graph gradient bindings");
    return;
  }
  std::vector<TiNamedNdArrayBinding> bindings;
  bindings.reserve(gradient_count);
  for (uint32_t i = 0; i < gradient_count; ++i) {
    bindings.push_back({gradients[i].name, 0, TI_NULL_HANDLE, &gradients[i].gradient});
  }
  ti_launch_compute_graph_with_bindings_ext(runtime, compute_graph, arg_count,
                                           args, gradient_count, bindings.data());
  TI_CAPI_TRY_CATCH_END();
}

void ti_launch_compute_graph_with_bindings_ext(
    TiRuntime runtime, TiComputeGraph compute_graph, uint32_t arg_count,
    const TiNamedArgument *args, uint32_t binding_count,
    const TiNamedNdArrayBinding *bindings) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  auto bound = capi::bind_graph_arguments(rt, compute_graph, arg_count, args,
                                          binding_count, bindings);
  if (bound)
    bound->graph->run(bound->arguments, bound->gradients);
  TI_CAPI_TRY_CATCH_END();
}

void ti_flush(TiRuntime runtime) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);

  ((Runtime *)runtime)->flush();
  TI_CAPI_TRY_CATCH_END();
}
void ti_wait(TiRuntime runtime) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);

  ((Runtime *)runtime)->wait();
  TI_CAPI_TRY_CATCH_END();
}
