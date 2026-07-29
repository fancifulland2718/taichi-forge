#include <taichi/rhi/device.h>

#if TI_WITH_VULKAN
#include <taichi/rhi/vulkan/vulkan_device.h>
#include <taichi/rhi/interop/vulkan_cpu_interop.h>
#if TI_WITH_LLVM
#include <taichi/rhi/cpu/cpu_device.h>
#endif
#if TI_WITH_CUDA
#include <taichi/rhi/cuda/cuda_device.h>
#include <taichi/rhi/interop/vulkan_cuda_interop.h>
#endif  // TI_WITH_CUDA
#endif  // TI_WITH_VULKAN

namespace taichi::lang {
namespace {

// Program enforces one live instance process-wide. Keeping the reporter in a
// process slot avoids changing the ABI/layout of every Device implementation,
// including objects used across JIT and plugin boundaries. Allocate the slot
// permanently so static destruction cannot race backend teardown.
std::shared_ptr<BackendFaultReporter> &process_backend_fault_reporter() {
  static auto *slot = new std::shared_ptr<BackendFaultReporter>();
  return *slot;
}

}  // namespace

void Device::set_backend_fault_reporter(
    std::shared_ptr<BackendFaultReporter> reporter) noexcept {
  std::atomic_store_explicit(&process_backend_fault_reporter(),
                             std::move(reporter), std::memory_order_release);
}

void Device::clear_backend_fault_reporter(
    const std::shared_ptr<BackendFaultReporter> &reporter) noexcept {
  auto &slot = process_backend_fault_reporter();
  auto current =
      std::atomic_load_explicit(&slot, std::memory_order_acquire);
  if (current == reporter) {
    std::atomic_store_explicit(&slot,
                               std::shared_ptr<BackendFaultReporter>{},
                               std::memory_order_release);
  }
}

std::shared_ptr<BackendFaultReporter> Device::backend_fault_reporter() const
    noexcept {
  return std::atomic_load_explicit(&process_backend_fault_reporter(),
                                   std::memory_order_acquire);
}

bool Device::backend_calls_safe() const noexcept {
  auto reporter = backend_fault_reporter();
  return !reporter || reporter->backend_calls_safe();
}

void Device::throw_if_backend_submission_disallowed(
    const char *operation) const {
  auto reporter = backend_fault_reporter();
  if (reporter) {
    reporter->throw_if_submission_disallowed(operation);
  }
}

void Device::report_backend_error(
    const BackendRuntimeError &error,
    std::uint64_t submission_sequence) const noexcept {
  auto reporter = backend_fault_reporter();
  if (reporter) {
    reporter->report_backend_error(error, submission_sequence);
  }
}

[[noreturn]] void Device::raise_backend_error(std::int64_t backend_code,
                                              std::string operation,
                                              std::string message) const {
  BackendRuntimeError error(arch(), backend_code, std::move(operation),
                            std::move(message));
  report_backend_error(error);
  throw error;
}

const std::string rhi_result_to_string(RhiResult result) {
  switch (result) {
    case RhiResult::success:
      return "success";
    case RhiResult::error:
      return "error";
    case RhiResult::invalid_usage:
      return "invalid_usage";
    case RhiResult::not_supported:
      return "not_supported";
    case RhiResult::out_of_memory:
      return "out_of_memory";
    default:
      return "unknown";
  }
}

DeviceAllocationGuard::~DeviceAllocationGuard() {
  device->dealloc_memory(*this);
}

DeviceImageGuard::~DeviceImageGuard() {
  dynamic_cast<GraphicsDevice *>(device)->destroy_image(*this);
}

DevicePtr DeviceAllocation::get_ptr(uint64_t offset) const {
  return DevicePtr{{device, alloc_id}, offset};
}

Device::MemcpyCapability Device::check_memcpy_capability(DevicePtr dst,
                                                         DevicePtr src,
                                                         uint64_t size) {
  if (dst.device == src.device) {
    return Device::MemcpyCapability::Direct;
  }

#if TI_WITH_VULKAN
#if TI_WITH_LLVM
  if (dynamic_cast<vulkan::VulkanDevice *>(dst.device) &&
      dynamic_cast<cpu::CpuDevice *>(src.device)) {
    // TODO: support direct copy if dst itself supports host write.
    return Device::MemcpyCapability::RequiresStagingBuffer;
  } else if (dynamic_cast<cpu::CpuDevice *>(dst.device) &&
             dynamic_cast<vulkan::VulkanDevice *>(src.device)) {
    return Device::MemcpyCapability::RequiresStagingBuffer;
  }
#endif
#if TI_WITH_CUDA
  if (dynamic_cast<vulkan::VulkanDevice *>(dst.device) &&
      dynamic_cast<cuda::CudaDevice *>(src.device)) {
    // FIXME: direct copy isn't always possible.
    // The vulkan buffer needs export_sharing turned on.
    // Otherwise, needs staging buffer
    return Device::MemcpyCapability::Direct;
  } else if (dynamic_cast<cuda::CudaDevice *>(dst.device) &&
             dynamic_cast<vulkan::VulkanDevice *>(src.device)) {
    return Device::MemcpyCapability::Direct;
  }
#endif  // TI_WITH_CUDA
#endif  // TI_WITH_VULKAN
  return Device::MemcpyCapability::RequiresHost;
}

void Device::memcpy_direct(DevicePtr dst, DevicePtr src, uint64_t size) {
  // Intra-device copy
  if (dst.device == src.device) {
    dst.device->memcpy_internal(dst, src, size);
    return;
  }
#if TI_WITH_VULKAN && TI_WITH_LLVM
  // cross-device copy directly
  else if (dynamic_cast<vulkan::VulkanDevice *>(dst.device) &&
           dynamic_cast<cpu::CpuDevice *>(src.device)) {
    memcpy_cpu_to_vulkan(dst, src, size);
    return;
  }
#endif
#if TI_WITH_VULKAN && TI_WITH_CUDA
  if (dynamic_cast<vulkan::VulkanDevice *>(dst.device) &&
      dynamic_cast<cuda::CudaDevice *>(src.device)) {
    memcpy_cuda_to_vulkan(dst, src, size);
    return;
  } else if (dynamic_cast<cuda::CudaDevice *>(dst.device) &&
             dynamic_cast<vulkan::VulkanDevice *>(src.device)) {
    memcpy_vulkan_to_cuda(dst, src, size);
    return;
  }
#endif
  TI_NOT_IMPLEMENTED;
}

void Device::memcpy_via_staging(DevicePtr dst,
                                DevicePtr staging,
                                DevicePtr src,
                                uint64_t size) {
  // Inter-device copy
#if defined(TI_WITH_VULKAN) && defined(TI_WITH_LLVM)
  if (dynamic_cast<vulkan::VulkanDevice *>(dst.device) &&
      dynamic_cast<cpu::CpuDevice *>(src.device)) {
    memcpy_cpu_to_vulkan_via_staging(dst, staging, src, size);
    return;
  }
#endif

  TI_NOT_IMPLEMENTED;
}

void Device::memcpy_via_host(DevicePtr dst,
                             void *host_buffer,
                             DevicePtr src,
                             uint64_t size) {
  TI_NOT_IMPLEMENTED;
}

void GraphicsDevice::image_transition(DeviceAllocation img,
                                      ImageLayout old_layout,
                                      ImageLayout new_layout) {
  Stream *stream = get_graphics_stream();
  auto [cmd_list, res] = stream->new_command_list_unique();
  TI_ASSERT(res == RhiResult::success);
  cmd_list->image_transition(img, old_layout, new_layout);
  stream->submit_synced(cmd_list.get());
}
void GraphicsDevice::buffer_to_image(DeviceAllocation dst_img,
                                     DevicePtr src_buf,
                                     ImageLayout img_layout,
                                     const BufferImageCopyParams &params) {
  Stream *stream = get_graphics_stream();
  auto [cmd_list, res] = stream->new_command_list_unique();
  TI_ASSERT(res == RhiResult::success);
  cmd_list->buffer_to_image(dst_img, src_buf, img_layout, params);
  stream->submit_synced(cmd_list.get());
}
void GraphicsDevice::image_to_buffer(DevicePtr dst_buf,
                                     DeviceAllocation src_img,
                                     ImageLayout img_layout,
                                     const BufferImageCopyParams &params) {
  Stream *stream = get_graphics_stream();
  auto [cmd_list, res] = stream->new_command_list_unique();
  TI_ASSERT(res == RhiResult::success);
  cmd_list->image_to_buffer(dst_buf, src_img, img_layout, params);
  stream->submit_synced(cmd_list.get());
}

RhiResult Device::upload_data(DevicePtr *device_ptr,
                              const void **data,
                              size_t *size,
                              int num_alloc) noexcept {
  if (!device_ptr || !data || !size) {
    return RhiResult::invalid_usage;
  }

  std::vector<DeviceAllocationUnique> stagings;
  for (int i = 0; i < num_alloc; i++) {
    if (device_ptr[i].device != this || !data[i]) {
      return RhiResult::invalid_usage;
    }
    auto [staging, res] = this->allocate_memory_unique(
        {size[i], /*host_write=*/true, /*host_read=*/false,
         /*export_sharing=*/false, AllocUsage::Upload});
    if (res != RhiResult::success) {
      return res;
    }

    void *mapped{nullptr};
    res = this->map(*staging, &mapped);
    if (res != RhiResult::success) {
      return res;
    }
    memcpy(mapped, data[i], size[i]);
    this->unmap(*staging);

    stagings.push_back(std::move(staging));
  }

  Stream *s = this->get_compute_stream();
  auto [cmdlist, res] = s->new_command_list_unique();
  if (res != RhiResult::success) {
    return res;
  }
  for (int i = 0; i < num_alloc; i++) {
    cmdlist->buffer_copy(device_ptr[i], stagings[i]->get_ptr(0), size[i]);
  }
  s->submit_synced(cmdlist.get());

  return RhiResult::success;
}

RhiResult Device::readback_data(
    DevicePtr *device_ptr,
    void **data,
    size_t *size,
    int num_alloc,
    const std::vector<StreamSemaphore> &wait_sema) noexcept {
  if (!device_ptr || !data || !size) {
    return RhiResult::invalid_usage;
  }

  Stream *s = this->get_compute_stream();
  auto [cmdlist, res] = s->new_command_list_unique();
  if (res != RhiResult::success) {
    return res;
  }

  std::vector<DeviceAllocationUnique> stagings;
  for (int i = 0; i < num_alloc; i++) {
    if (device_ptr[i].device != this || !data[i]) {
      return RhiResult::invalid_usage;
    }
    auto [staging, res] = this->allocate_memory_unique(
        {size[i], /*host_write=*/false, /*host_read=*/true,
         /*export_sharing=*/false, AllocUsage::None});
    if (res != RhiResult::success) {
      return res;
    }

    cmdlist->buffer_copy(staging->get_ptr(0), device_ptr[i], size[i]);
    stagings.push_back(std::move(staging));
  }
  s->submit_synced(cmdlist.get(), wait_sema);

  for (int i = 0; i < num_alloc; i++) {
    void *mapped{nullptr};
    RhiResult res = this->map(*stagings[i], &mapped);
    if (res != RhiResult::success) {
      return res;
    }
    memcpy(data[i], mapped, size[i]);
    this->unmap(*stagings[i]);
  }

  return RhiResult::success;
}

RhiResult Device::readback_data_packed(
    DevicePtr *device_ptr,
    void **data,
    size_t *size,
    int num_alloc,
    DevicePtr staging,
    size_t staging_size,
    const std::vector<StreamSemaphore> &wait_sema) noexcept {
  if (!device_ptr || !data || !size || num_alloc < 0 ||
      staging.device != this || staging.offset != 0) {
    return RhiResult::invalid_usage;
  }
  if (num_alloc == 0) {
    return RhiResult::success;
  }

  constexpr size_t kCopyAlignment = 4;
  std::vector<size_t> offsets;
  offsets.reserve(num_alloc);
  size_t packed_size = 0;
  for (int i = 0; i < num_alloc; ++i) {
    if (device_ptr[i].device != this || !data[i]) {
      return RhiResult::invalid_usage;
    }
    if (device_ptr[i].offset % kCopyAlignment != 0 ||
        size[i] % kCopyAlignment != 0) {
      return RhiResult::not_supported;
    }
    if (packed_size >
        std::numeric_limits<size_t>::max() - (kCopyAlignment - 1)) {
      return RhiResult::invalid_usage;
    }
    packed_size =
        (packed_size + kCopyAlignment - 1) & ~(kCopyAlignment - 1);
    offsets.push_back(packed_size);
    if (size[i] > std::numeric_limits<size_t>::max() - packed_size) {
      return RhiResult::invalid_usage;
    }
    packed_size += size[i];
  }
  if (packed_size > staging_size) {
    return RhiResult::invalid_usage;
  }

  Stream *stream = this->get_compute_stream();
  auto [cmdlist, res] = stream->new_command_list_unique();
  if (res != RhiResult::success) {
    return res;
  }
  for (int i = 0; i < num_alloc; ++i) {
    if (size[i] == 0) {
      continue;
    }
    DevicePtr destination = staging;
    destination.offset += offsets[i];
    cmdlist->buffer_copy(destination, device_ptr[i], size[i]);
  }
  stream->submit_synced(cmdlist.get(), wait_sema);

  void *mapped = nullptr;
  res = this->map_range(staging, packed_size, &mapped);
  if (res != RhiResult::success) {
    return res;
  }
  for (int i = 0; i < num_alloc; ++i) {
    if (size[i] != 0) {
      std::memcpy(data[i], static_cast<uint8_t *>(mapped) + offsets[i],
                  size[i]);
    }
  }
  this->unmap(staging);
  return RhiResult::success;
}

}  // namespace taichi::lang
