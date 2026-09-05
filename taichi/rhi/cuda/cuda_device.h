#pragma once
#include <memory>
#include <mutex>
#include <vector>
#include <set>

#include "taichi/common/core.h"
#include "taichi/rhi/common/allocation_registry.h"
#include "taichi/rhi/cuda/cuda_driver.h"
#include "taichi/rhi/llvm/allocator.h"
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/llvm/llvm_device.h"

namespace taichi::lang {
namespace cuda {

class CudaPipeline : public Pipeline {
 public:
  ~CudaPipeline() override {
  }
};

class CudaCommandList : public CommandList {
 public:
  ~CudaCommandList() override {
  }

  void bind_pipeline(Pipeline *p) noexcept override { TI_NOT_IMPLEMENTED };
  RhiResult bind_shader_resources(ShaderResourceSet *res,
                                  int set_index = 0) noexcept final {
    TI_NOT_IMPLEMENTED
  };
  RhiResult bind_raster_resources(RasterResources *res) noexcept final {
    TI_NOT_IMPLEMENTED
  };
  void buffer_barrier(DevicePtr ptr, size_t size) noexcept override {
    TI_NOT_IMPLEMENTED
  };
  void buffer_barrier(DeviceAllocation alloc) noexcept override {
    TI_NOT_IMPLEMENTED
  };
  void memory_barrier() noexcept override { TI_NOT_IMPLEMENTED };
  void buffer_copy(DevicePtr dst,
                   DevicePtr src,
                   size_t size) noexcept override {
    TI_NOT_IMPLEMENTED
  };
  void buffer_fill(DevicePtr ptr,
                   size_t size,
                   uint32_t data) noexcept override {
    TI_NOT_IMPLEMENTED
  };
  RhiResult dispatch(uint32_t x,
                     uint32_t y = 1,
                     uint32_t z = 1) noexcept override {
    TI_NOT_IMPLEMENTED
  };
};

class CudaStream : public Stream {
 public:
  ~CudaStream() override {};

  RhiResult new_command_list(CommandList **out_cmdlist) noexcept final {
    TI_NOT_IMPLEMENTED
  };
  StreamSemaphore submit(
      CommandList *cmdlist,
      const std::vector<StreamSemaphore> &wait_semaphores = {}) override {
    TI_NOT_IMPLEMENTED
  };
  StreamSemaphore submit_synced(
      CommandList *cmdlist,
      const std::vector<StreamSemaphore> &wait_semaphores = {}) override {
    TI_NOT_IMPLEMENTED
  };

  void command_sync() override { TI_NOT_IMPLEMENTED };
};

class CudaDevice : public LlvmDevice {
 public:
  class AllocationLease;

  // Cold lifecycle hook: invalidate published Graph frames before releasing
  // allocation registries. Weak registration never retains unused executors.
  class RetainedGraphResource {
   public:
    virtual ~RetainedGraphResource() = default;
    virtual void retire_graph_resource() noexcept = 0;
  };
  void register_graph_resource(
      const std::shared_ptr<RetainedGraphResource> &resource);

  // Owned pools retire only after all allocation owners and their queued
  // frees have finished. Collection is restricted to cold pool boundaries.
  void register_graph_memory_pool(std::shared_ptr<void> pool);
  void collect_graph_memory_pools();

  struct AllocInfo {
    void *ptr{nullptr};
    size_t size{0};
    bool is_imported{false};
    /* Note: Memory allocation in CUDA device.
     * CudaDevice can use either its own cuda malloc mechanism via
     * `allocate_memory` or the preallocated memory managed by Llvmprogramimpl
     * via `allocate_memory_runtime`. The `use_preallocated` is used to track
     * this option. For now, we keep both options and the preallocated method is
     * used by default for CUDA backend. The `use_cached` is to enable/disable
     * the caching behavior in `allocate_memory_runtime`. Later it should be
     * always enabled, for now we keep both options to allow a scenario when
     * using preallocated memory while disabling the caching behavior.
     * */
    bool use_preallocated{true};
    bool use_cached{false};
    bool use_memory_pool{false};
    void *mapped{nullptr};
    bool is_mapped{false};
  };

  AllocInfo get_alloc_info(const DeviceAllocation handle);
  // Pins one generation-qualified registry allocation while a CUDA graph
  // executable may still contain its device address. The returned lease must
  // be destroyed before this CudaDevice.
  std::unique_ptr<AllocationLease> acquire_allocation_lease(
      DeviceAllocation handle);

  CudaDevice();
  ~CudaDevice() override;

  RhiResult allocate_memory(const AllocParams &params,
                            DeviceAllocation *out_devalloc) override;
  DeviceAllocation allocate_memory_runtime(
      const LlvmRuntimeAllocParams &params) override;

  // Cold Graph materialization only. Handles keep the normal registry/lease
  // retirement contract; the caller owns the non-default CUDA pool.
  DeviceAllocation allocate_memory_from_pool(std::size_t bytes,
                                             void *pool,
                                             std::shared_ptr<void> pool_owner);
  void dealloc_memory(DeviceAllocation handle) override;

  uint64_t *allocate_llvm_runtime_memory_jit(
      const LlvmRuntimeAllocParams &params) override;

  RhiResult upload_data(DevicePtr *device_ptr,
                        const void **data,
                        size_t *size,
                        int num_alloc = 1) noexcept override;

  RhiResult readback_data(
      DevicePtr *device_ptr,
      void **data,
      size_t *size,
      int num_alloc = 1,
      const std::vector<StreamSemaphore> &wait_sema = {}) noexcept override;

  ShaderResourceSet *create_resource_set() final { TI_NOT_IMPLEMENTED };

  RhiResult create_pipeline(Pipeline **out_pipeline,
                            const PipelineSourceDesc &src,
                            std::string name,
                            PipelineCache *cache) noexcept final {
    TI_NOT_IMPLEMENTED;
  }

  RhiResult map_range(DevicePtr ptr, uint64_t size, void **mapped_ptr) final {
    TI_NOT_IMPLEMENTED;
  }
  RhiResult map(DeviceAllocation alloc, void **mapped_ptr) final;

  void unmap(DevicePtr ptr) final { TI_NOT_IMPLEMENTED };
  void unmap(DeviceAllocation alloc) final;

  void memcpy_internal(DevicePtr dst, DevicePtr src, uint64_t size) override;

  // Internal interop helpers. They retain the source/destination registry
  // lease through the CUDA copy; callers retain ownership of external_ptr.
  RhiResult copy_to_external(void *external_ptr,
                             DevicePtr src,
                             uint64_t size);
  RhiResult copy_from_external(DevicePtr dst,
                               void *external_ptr,
                               uint64_t size);

  DeviceAllocation import_memory(void *ptr, size_t size) override;

  // Runtime shutdown probes allocations that may already have been retired by
  // a Program-owned Ndarray. A raw-address query has no error return channel,
  // so a stale/wrong-device allocation is represented as nullptr rather than
  // turning normal reset cleanup into a fatal error.
  void *get_memory_addr(DeviceAllocation devalloc) override;

  std::size_t get_total_memory() override {
    return CUDAContext::get_instance().get_total_memory();
  }

  Stream *get_compute_stream() override { TI_NOT_IMPLEMENTED };

  void wait_idle() override { TI_NOT_IMPLEMENTED };

  void clear() override;

 private:
  struct MappingState {
    std::mutex mutex;
    std::unique_ptr<char[]> staging;
    bool mapped{false};
  };

  struct AllocationRecord {
    AllocationRecord(CudaDevice *owner,
                     void *ptr,
                     size_t size,
                     bool is_imported,
                     bool use_preallocated,
                     bool use_cached,
                     bool use_memory_pool,
                     CUstream stream,
                     std::unique_ptr<MappingState> mapping)
        : owner(owner),
          ptr(ptr),
          size(size),
          is_imported(is_imported),
          use_preallocated(use_preallocated),
          use_cached(use_cached),
          use_memory_pool(use_memory_pool),
          stream(stream),
          mapping(std::move(mapping)) {
    }
    ~AllocationRecord();
    AllocationRecord(const AllocationRecord &) = delete;
    AllocationRecord &operator=(const AllocationRecord &) = delete;
    AllocationRecord(AllocationRecord &&other) noexcept;
    AllocationRecord &operator=(AllocationRecord &&other) noexcept;

    AllocInfo info() const {
      AllocInfo result;
      result.ptr = ptr;
      result.size = size;
      result.is_imported = is_imported;
      result.use_preallocated = use_preallocated;
      result.use_cached = use_cached;
      result.use_memory_pool = use_memory_pool;
      return result;
    }

    void release();

    CudaDevice *owner{nullptr};
    void *ptr{nullptr};
    size_t size{0};
    bool is_imported{false};
    bool use_preallocated{true};
    bool use_cached{false};
    bool use_memory_pool{false};
    CUstream stream{nullptr};
    std::unique_ptr<MappingState> mapping;
    // Cold allocation lifetime only: destroy an owned pool after its final
    // allocation has submitted free, not when the factory closes.
    std::shared_ptr<void> pool_owner;
  };

  AllocationRegistry<AllocationRecord> allocations_;
  std::vector<std::weak_ptr<RetainedGraphResource>> graph_resources_;
  std::vector<std::shared_ptr<void>> graph_memory_pools_;
  // Serializes transitions between mapped and retiring allocations. The lock
  // is held only while map/unmap copies or allocation metadata changes; it is
  // not held while callers use the returned host pointer.
  std::mutex mapping_lifecycle_mutex_;
  size_t mapped_allocation_count_{0};
};

class CudaDevice::AllocationLease {
 public:
  ~AllocationLease();
  AllocationLease(const AllocationLease &) = delete;
  AllocationLease &operator=(const AllocationLease &) = delete;

 private:
  friend class CudaDevice;
  explicit AllocationLease(
      CudaDevice *device,
      AllocationRegistry<AllocationRecord>::Lease registry_lease)
      : device_(device), registry_lease_(std::move(registry_lease)) {
  }

  CudaDevice *device_{nullptr};
  AllocationRegistry<AllocationRecord>::Lease registry_lease_;
};

}  // namespace cuda

}  // namespace taichi::lang
