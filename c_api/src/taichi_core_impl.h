#pragma once
#include <vector>
#include <memory>
#include <string>
#include <exception>
#include <functional>
#include <stdexcept>
#include <mutex>
#include <map>
#include <unordered_map>
#include <unordered_set>

// Taichi runtime is not necessarily using the same 3rd-party headers as the
// user codes. For C-API implementations we only use the internal headers.
#ifdef TI_WITH_VULKAN
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES 1
#endif  // VK_NO_PROTOTYPES
#include "vulkan/vulkan.h"
#define TI_NO_VULKAN_INCLUDES 1
#endif  // TI_WITH_VULKAN

#ifdef TI_WITH_OPENGL
#include "glad/gl.h"
#define TI_NO_OPENGL_INCLUDES 1
#endif  // TI_WITH_OPENGL

// Then Include all C-API symbols.
#include "taichi/taichi.h"

// Include for the base types.
#include "taichi/rhi/arch.h"
#define TI_RUNTIME_HOST 1
#include "taichi/program/context.h"
#undef TI_RUNTIME_HOST
#include "taichi/rhi/device.h"
#include "taichi/aot/graph_data.h"
#include "taichi/aot/module_loader.h"
#include "taichi/aot/argpack_layout.h"
#include "taichi/aot/struct_layout.h"
#include "taichi/common/virtual_dir.h"
#include "taichi/runtime/native_provider.h"

namespace taichi::lang {
class RuntimeFaultDomain;
class RuntimeCompletion;
class ExternalSynchronizationDomain;
namespace cuda {
class ExecutionDomainRegistry;
}
}

#define TI_CAPI_NOT_SUPPORTED(x) ti_set_last_error(TI_ERROR_NOT_SUPPORTED, #x);
#define TI_CAPI_NOT_SUPPORTED_IF(x)                \
  if (x) {                                         \
    ti_set_last_error(TI_ERROR_NOT_SUPPORTED, #x); \
  }
#define TI_CAPI_NOT_SUPPORTED_IF_RV(x)             \
  if (x) {                                         \
    ti_set_last_error(TI_ERROR_NOT_SUPPORTED, #x); \
    return TI_NULL_HANDLE;                         \
  }

#define TI_CAPI_ARGUMENT_NULL(x)                   \
  if (x == TI_NULL_HANDLE) {                       \
    ti_set_last_error(TI_ERROR_ARGUMENT_NULL, #x); \
    return;                                        \
  }
#define TI_CAPI_ARGUMENT_NULL_RV(x)                \
  if (x == TI_NULL_HANDLE) {                       \
    ti_set_last_error(TI_ERROR_ARGUMENT_NULL, #x); \
    return TI_NULL_HANDLE;                         \
  }

#define TI_CAPI_INVALID_ARGUMENT(pred)                   \
  if (pred) {                                            \
    ti_set_last_error(TI_ERROR_INVALID_ARGUMENT, #pred); \
    return;                                              \
  }
#define TI_CAPI_INVALID_ARGUMENT_RV(pred)                \
  if (pred) {                                            \
    ti_set_last_error(TI_ERROR_INVALID_ARGUMENT, #pred); \
    return TI_NULL_HANDLE;                               \
  }

#define TI_CAPI_INVALID_INTEROP_ARCH(x, arch)                    \
  if (x != taichi::Arch::arch) {                                 \
    ti_set_last_error(TI_ERROR_INVALID_INTEROP, "arch!=" #arch); \
    return;                                                      \
  }
#define TI_CAPI_INVALID_INTEROP_ARCH_RV(x, arch)                 \
  if (x != taichi::Arch::arch) {                                 \
    ti_set_last_error(TI_ERROR_INVALID_INTEROP, "arch!=" #arch); \
    return TI_NULL_HANDLE;                                       \
  }

class MemoryBindingError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

#define TI_CAPI_TRY_CATCH_BEGIN() try {
#define TI_CAPI_TRY_CATCH_END()                                 \
  }                                                             \
  catch (const MemoryBindingError &e) {                         \
    ti_set_last_error(TI_ERROR_INVALID_ARGUMENT, e.what());     \
  }                                                             \
  catch (const std::exception &e) {                             \
    ti_set_last_error(TI_ERROR_INVALID_STATE, e.what());        \
  }                                                             \
  catch (const std::string &e) {                                \
    ti_set_last_error(TI_ERROR_INVALID_STATE, e.c_str());       \
  }                                                             \
  catch (...) {                                                 \
    ti_set_last_error(TI_ERROR_INVALID_STATE, "c++ exception"); \
  }

struct Error {
  TiError error;
  std::string message;

  Error(TiError error, const std::string &message)
      : error(error), message(message) {
  }
  Error() : error(TI_ERROR_SUCCESS), message() {
  }
  Error(const Error &) = delete;
  Error(Error &&) = default;
  Error &operator=(const Error &) = delete;
  Error &operator=(Error &&) = default;

  // Set this error as the last error if it's not `TI_ERROR_SUCCESS`.
  inline void set_last_error() const {
    if (error != TI_ERROR_SUCCESS) {
      ti_set_last_error(error, message.c_str());
    }
  }
};

namespace capi {
class MetalRuntime;
class PreparedExecutionRegistry;
class NativeResourceRegistry;
struct SubmissionCompletionState;
struct BoundKernelArguments;
}  // namespace capi

class Runtime {
 public:
  struct KernelBinding {
    class AotModule *owner;
    taichi::lang::aot::Kernel *kernel;
    std::unordered_map<uint32_t, TiNdArrayLayout> layouts;
    mutable std::string return_layout;
    // A failed launch/wait may leave a host readback in flight. Keep the whole
    // binding until a successful wait, or the owner's synchronized retirement.
    mutable std::shared_ptr<capi::BoundKernelArguments> pending_return;
    struct ValueWindow {
      std::size_t offset;
      std::size_t size;
    };
    mutable std::string argument_layout;
    mutable std::map<uint32_t, ValueWindow> value_arguments;
    using ArgPackLayouts =
        std::map<uint32_t,
                 std::shared_ptr<const taichi::lang::aot::ArgPackLayout>>;
    // Most kernels have no packs. Do not allocate an empty map sentinel for
    // every loaded callable; construct it only after successful reflection.
    mutable std::shared_ptr<ArgPackLayouts> argpack_layouts;
  };
  struct GraphBinding {
    class AotModule *owner;
    taichi::lang::aot::CompiledGraph *graph;
    std::unordered_map<std::string, TiNdArrayLayout> layouts;
  };
  struct LayoutBinding {
    class AotModule *owner;
    taichi::lang::aot::StructLayout layout;
  };
  struct BorrowedMemory {
    // Shared by all finite views of this loan. Mutation is cold and serialized
    // by submission_mutex; get_memory observes live under memory_mutex.
    std::shared_ptr<void> owner;
    std::shared_ptr<taichi::lang::ExternalSynchronizationDomain>
        external_access;
    bool live{true};
  };
  struct MemoryAllocation {
    taichi::lang::DeviceAllocation allocation;
    uint64_t size;
    // External pointer/binding offset; owned RHI allocations are naturally
    // aligned. Captured once on import, without querying the driver at launch.
    uint64_t base_offset{0};
    // A same-device storage lease owned by another API. Drop this reference
    // after recordings retire; never deallocate its RHI allocation directly.
    std::shared_ptr<void> storage_owner;
    // Logical offset within the RHI allocation, independent of an imported
    // VkDeviceMemory binding offset. Used by internal execution-plan views.
    uint64_t byte_offset{0};
    // Internal aliases can instead be owned by their enclosing module. That
    // lifetime is ordered by prepared retirement, not by a placeholder lease.
    bool owns_allocation{true};
    std::shared_ptr<BorrowedMemory> borrowed;
  };

 private:
  mutable std::mutex memory_mutex_;
  std::unordered_map<TiMemory, MemoryAllocation> memories_;
  std::unordered_map<TiKernel, KernelBinding> kernels_;
  std::unordered_map<TiComputeGraph, GraphBinding> graphs_;
  std::unordered_map<TiNdArrayLayout, LayoutBinding> layouts_;
  std::unordered_set<class AotModule *> modules_;
  std::recursive_mutex submission_mutex_;
  std::shared_ptr<capi::PreparedExecutionRegistry> prepared_;
  std::shared_ptr<capi::NativeResourceRegistry> native_resources_;
  taichi::lang::NativeProviderLocations native_provider_locations_;
  taichi::lang::NativeLibraryLocations native_library_locations_;
  std::shared_ptr<taichi::lang::RuntimeFaultDomain> runtime_fault_domain_;
  std::shared_ptr<taichi::lang::cuda::ExecutionDomainRegistry>
      cuda_execution_domains_;
  bool execution_domains_retiring_{false};
  bool borrowed_fault_domain_{false};
  std::shared_ptr<capi::SubmissionCompletionState> submission_completions_;
  std::function<taichi::lang::RuntimeCompletion()> borrowed_completion_recorder_;

 protected:
  // 32 is a magic number in `taichi/inc/constants.h`.
  std::array<uint64_t, 32> host_result_buffer_;

  explicit Runtime(taichi::Arch arch);

 public:
  const taichi::Arch arch;

  virtual ~Runtime();

  virtual taichi::lang::Device &get() = 0;

  [[deprecated("create_aot_module")]] virtual TiAotModule load_aot_module(
      const char *module_path) {
    auto dir = taichi::io::VirtualDir::open(module_path);
    if (!dir) {
      ti_set_last_error(TI_ERROR_CORRUPTED_DATA, module_path);
      return TI_NULL_HANDLE;
    }
    TiAotModule aot_module = TI_NULL_HANDLE;
    Error err = create_aot_module(std::move(dir), aot_module);
    err.set_last_error();
    return aot_module;
  }

  virtual Error create_aot_module(std::shared_ptr<const taichi::io::VirtualDir> dir,
                                  TiAotModule &out) {
    TI_NOT_IMPLEMENTED
  }
  virtual TiMemory allocate_memory(
      const taichi::lang::Device::AllocParams &params);
  virtual void free_memory(TiMemory devmem);

  // Adopts an RHI allocation, including imported allocation wrappers. The
  // process-unique opaque handle is never reinterpreted as an RHI slot ID.
  TiMemory register_memory(taichi::lang::DeviceAllocation allocation,
                           uint64_t size,
                           bool release_on_failure = true,
                           uint64_t base_offset = 0,
                           std::shared_ptr<void> storage_owner = {},
                           uint64_t byte_offset = 0,
                           bool owns_allocation = true,
                           std::shared_ptr<BorrowedMemory> borrowed = {});
  // Only whole-recipe cold binding may opt into external synchronization.
  // Ordinary launch/copy/map/export paths cannot bypass the access epoch.
  MemoryAllocation get_memory(TiMemory memory,
                              bool prepared_external_access = false) const;
  void retire_borrowed_memory(const std::shared_ptr<BorrowedMemory> &borrowed);
  static std::uintptr_t new_handle_token();
  TiKernel register_kernel(KernelBinding binding);
  TiComputeGraph register_graph(GraphBinding binding);
  TiNdArrayLayout register_layout(class AotModule *owner,
                                  taichi::lang::DataType type);
  const KernelBinding &get_kernel(TiKernel kernel) const;
  const KernelBinding &get_kernel_value_arguments(TiKernel kernel) const;
  std::shared_ptr<const taichi::lang::aot::ArgPackLayout>
  get_kernel_argpack_layout(TiKernel kernel, uint32_t logical_argument);
  const GraphBinding &get_graph(TiComputeGraph graph) const;
  const taichi::lang::aot::StructLayout &get_layout(TiNdArrayLayout layout) const;
  void retire_module(class AotModule *owner);
  void wait_for_kernel_returns(class AotModule *owner);
  void invalidate_module_entries(class AotModule *owner);
  void register_module(class AotModule *owner);
  bool owns_module(class AotModule *owner) const {
    return modules_.count(owner) != 0;
  }
  // Explicit borrower retirement: recordings before code and buffers. The
  // Program finalizer has already synchronized and passes false.
  void retire_resources(bool synchronize);
  std::recursive_mutex &submission_mutex() {
    return submission_mutex_;
  }
  capi::PreparedExecutionRegistry &prepared_registry();
  capi::NativeResourceRegistry &native_resource_registry();
  void retire_native_memory(TiMemory memory);
  void clear_native_resources(bool synchronize = true);
  void retire_prepared_native_resource(TiNativeResource resource,
                                        bool synchronize = true);
  const taichi::lang::NativeProviderLocations &native_provider_locations()
      const {
    return native_provider_locations_;
  }
  void configure_native_provider(
      std::string provider,
      std::optional<taichi::lang::NativeProviderLocation> location);
  const taichi::lang::NativeLibraryLocations &native_library_locations() const {
    return native_library_locations_;
  }
  void configure_native_library(std::string name, const char *path);
  void retire_prepared_memory(TiMemory memory);
  void retire_prepared_module(class AotModule *owner);
  void clear_prepared(bool synchronize = true);
  const std::shared_ptr<taichi::lang::RuntimeFaultDomain> &runtime_fault_domain()
      const {
    return runtime_fault_domain_;
  }
  // Cold borrower construction only; ownership of the domain stays with the
  // Program. Closing this C API view must not finalize its owner's domain.
  void borrow_runtime_fault_domain(
      std::shared_ptr<taichi::lang::RuntimeFaultDomain> domain,
      std::shared_ptr<taichi::lang::cuda::ExecutionDomainRegistry> domains =
          {});
  std::shared_ptr<taichi::lang::cuda::ExecutionDomainRegistry>
  cuda_execution_domains();
  // A borrowed view never retires its Program's shared domains. Its resource
  // retirement already waits for outstanding execution before freeing views.
  void retire_execution_domains();
  void synchronize_execution_domains();
  void borrow_completion_recorder(
      std::function<taichi::lang::RuntimeCompletion()> recorder);
  void prepare_submission_completion();
  TiSubmissionTicket finish_submission_completion();
  void complete_submission_completions();
  void clear_submission_completions();

  virtual TiImage allocate_image(const taichi::lang::ImageParams &params) {
    TI_NOT_IMPLEMENTED
  }
  virtual void free_image(TiImage image) {
    TI_NOT_IMPLEMENTED
  }

  virtual void buffer_copy(const taichi::lang::DevicePtr &dst,
                           const taichi::lang::DevicePtr &src,
                           size_t size) = 0;
  virtual void copy_image(const taichi::lang::DeviceAllocation &dst,
                          const taichi::lang::DeviceAllocation &src,
                          const taichi::lang::ImageCopyParams &params) {
    TI_NOT_IMPLEMENTED
  }
  virtual void track_image(const taichi::lang::DeviceAllocation &image,
                           taichi::lang::ImageLayout layout) {
    TI_NOT_IMPLEMENTED
  }
  virtual void untrack_image(const taichi::lang::DeviceAllocation &image) {
    TI_NOT_IMPLEMENTED
  }
  virtual void transition_image(const taichi::lang::DeviceAllocation &image,
                                taichi::lang::ImageLayout layout) {
    TI_NOT_IMPLEMENTED
  }
  virtual void flush() = 0;
  virtual void wait() = 0;

  class VulkanRuntime *as_vk();
  class capi::MetalRuntime *as_mtl();
};

class AotModule {
  Runtime *runtime_;
  bool code_live_{true};
  std::unique_ptr<taichi::lang::aot::Module> aot_module_;
  std::unordered_map<std::string,
                     std::unique_ptr<taichi::lang::aot::CompiledGraph>>
      loaded_cgraphs_;
  std::unordered_map<std::string, TiKernel> kernel_handles_;
  std::unordered_map<std::string, TiComputeGraph> graph_handles_;
  std::unordered_map<const taichi::lang::Type *, TiNdArrayLayout> layout_handles_;

  TiNdArrayLayout layout_handle(taichi::lang::DataType type);
  void check_code_live() const;

 public:
  AotModule(Runtime &runtime,
            std::unique_ptr<taichi::lang::aot::Module> aot_module);
  ~AotModule();

  TiKernel get_kernel_handle(const std::string &name);
  TiComputeGraph get_graph_handle(const std::string &name);
  TiNdArrayLayout get_layout_handle(const std::string &name);

  taichi::lang::aot::Kernel *get_kernel(const std::string &name);
  taichi::lang::aot::CompiledGraph *get_cgraph(const std::string &name);
  taichi::lang::aot::Module &get();
  Runtime &runtime();
  // Cold live-JIT loan retirement. Keep the host shell owned by Runtime but
  // drain its independent users before invalidating entry handles/code access.
  void retire_borrowed_code();
};

namespace {

template <typename THandle>
struct devalloc_cast_t {
  static inline taichi::lang::DeviceAllocation handle2devalloc(Runtime &runtime,
                                                               THandle handle) {
    return taichi::lang::DeviceAllocation{
        &runtime.get(), (taichi::lang::DeviceAllocationId)((size_t)handle - 1)};
  }
  static inline THandle devalloc2handle(
      Runtime &runtime,
      taichi::lang::DeviceAllocation devalloc) {
    return (THandle)((size_t)devalloc.alloc_id + 1);
  }
};

[[maybe_unused]] taichi::lang::DeviceAllocation devmem2devalloc(
    Runtime &runtime,
    TiMemory devmem) {
  return runtime.get_memory(devmem).allocation;
}

[[maybe_unused]] TiMemory devalloc2devmem(
    Runtime &runtime,
    const taichi::lang::DeviceAllocation &devalloc,
    uint64_t size,
    bool release_on_failure = true,
    uint64_t base_offset = 0) {
  return runtime.register_memory(devalloc, size, release_on_failure, base_offset);
}

[[maybe_unused]] taichi::lang::DeviceAllocation devimg2devalloc(
    Runtime &runtime,
    TiImage devimg) {
  return devalloc_cast_t<TiImage>::handle2devalloc(runtime, devimg);
}

[[maybe_unused]] TiImage devalloc2devimg(
    Runtime &runtime,
    const taichi::lang::DeviceAllocation &devalloc) {
  return devalloc_cast_t<TiImage>::devalloc2handle(runtime, devalloc);
}

}  // namespace
