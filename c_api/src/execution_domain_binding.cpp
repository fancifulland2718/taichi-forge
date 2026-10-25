#include "execution_domain_binding.h"

#include <stdexcept>
#include <cstring>
#include <limits>
#include <algorithm>
#include <map>
#include <unordered_set>
#include "taichi/runtime/native_array_contract.h"
#include "taichi/runtime/native_image.h"
#ifdef TI_WITH_CUDA
#include "taichi/runtime/cuda/execution_domain.h"
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_device.h"
#endif
#ifdef TI_WITH_VULKAN
#include "taichi/runtime/vulkan_execution_domain.h"
#include "taichi/runtime/gfx/runtime.h"
#endif

namespace capi {

taichi::lang::RuntimeCompletion
PreparedExecutionDomain::join_allocation_initialization(
    taichi::lang::Device &device,
    const std::vector<taichi::lang::DeviceAllocation> &allocations) {
  using namespace taichi::lang;
#ifdef TI_WITH_CUDA
  if (cuda) {
    if (allocations.empty())
      return {};
    struct Resources final : RuntimeCompletionResources {
      std::vector<
          std::unique_ptr<taichi::lang::cuda::CudaDevice::AllocationLease>>
          leases;
      std::size_t retained_resource_count(
          std::uint32_t) const noexcept override {
        return 0;
      }
    };
    auto resources = std::make_shared<Resources>();
    auto &cuda_device = dynamic_cast<taichi::lang::cuda::CudaDevice &>(device);
    std::unordered_set<DeviceAllocationId> seen;
    for (const auto &allocation : allocations) {
      if (allocation.device != &device)
        throw std::invalid_argument(
            "Cold dependency belongs to another device");
      if (!seen.insert(allocation.alloc_id).second)
        continue;
      auto lease = cuda_device.acquire_allocation_lease(allocation.get_ptr());
      if (!lease)
        throw std::invalid_argument("Cold dependency allocation is retired");
      resources->leases.push_back(std::move(lease));
    }
    return cuda->submit({}, resources, [resources](void *stream) {
      for (const auto &lease : resources->leases)
        lease->wait_for_initialization(stream);
    });
  }
#endif
  throw std::invalid_argument("Allocation initialization join requires CUDA");
}

bool PreparedExecutionDomain::complete_owned_initialization(
    taichi::lang::Device &device,
    taichi::lang::gfx::GfxRuntime *gfx_runtime,
    std::shared_ptr<taichi::lang::NativeOperation> operation,
    const std::vector<taichi::lang::NativeArrayBinding> &inputs) {
  using namespace taichi::lang;
  if (!operation || operation->arch() != backend() ||
      &operation->device() != &device)
    throw std::invalid_argument(
        "Cold initialization requires a local operation");
  for (const auto &input : inputs) {
    if (input.pointer.device != &device || input.strides)
      throw std::invalid_argument(
          "Cold initialization requires packed local inputs");
    resolve_native_storage_range(backend(), input.pointer, input.bytes);
  }
#ifdef TI_WITH_VULKAN
  if (vulkan) {
    const auto requirements = operation->vulkan_recording_requirements();
    auto &vk_device =
        dynamic_cast<taichi::lang::vulkan::VulkanDevice &>(device);
    if (!requirements.buffers || !requirements.image_uses ||
        !requirements.stateful_resources.empty() ||
        (requirements.graphics_queue &&
         vulkan->snapshot().queue_family_index !=
             vk_device.graphics_queue_family_index()))
      return false;
    if (!gfx_runtime || gfx_runtime->get_ti_device() != &device)
      throw std::invalid_argument(
          "Cold image initialization requires its GFX owner");
    for (const auto &buffer : *requirements.buffers)
      if (std::none_of(inputs.begin(), inputs.end(), [&](const auto &input) {
            return input.pointer.device == buffer.device &&
                   input.pointer.alloc_id == buffer.alloc_id;
          }))
        throw std::invalid_argument(
            "Cold initialization has an undescribed input");
    std::map<DeviceAllocationId, std::shared_ptr<NativeSampledImage>> images;
    for (const auto &use : *requirements.image_uses) {
      if (!use.image || &use.image->device() != &device ||
          use.image->gfx_managed_owner())
        throw std::invalid_argument(
            "Cold initialization requires newly owned images");
      images.emplace(use.image->gfx_sampled_allocation().alloc_id, use.image);
    }
    for (const auto &[allocation, layout] : requirements.images)
      if (allocation.device != &device || !images.count(allocation.alloc_id))
        throw std::invalid_argument(
            "Cold initialization image has no retained owner");
    auto commands = vulkan->new_command_list();
    for (const auto &[allocation, layout] : requirements.images)
      commands->image_transition(
          allocation,
          images.at(allocation.alloc_id)->parameters().initial_layout, layout);
    operation->record_vulkan_independent(&device, commands.get());
    for (const auto &[allocation, layout] : requirements.images) {
      const auto &params = images.at(allocation.alloc_id)->parameters();
      const auto boundary = bool(params.usage & ImageAllocUsage::Sampled)
                                ? ImageLayout::shader_read
                                : ImageLayout::shader_read_write;
      commands->image_transition(allocation, layout, boundary);
    }
    // Commands retain native buffers; their operand owner retains images even
    // if submission or waiting fails and the factory cannot publish a result.
    vulkan->submit(vulkan->prepare_commands({commands.get()}, operation))
        .wait();
    vulkan->collect_ready(1);
    for (const auto &[id, image] : images) {
      const auto boundary =
          bool(image->parameters().usage & ImageAllocUsage::Sampled)
              ? ImageLayout::shader_read
              : ImageLayout::shader_read_write;
      gfx_runtime->track_image(image->gfx_sampled_allocation(), boundary);
    }
    return true;
  }
#endif
#ifdef TI_WITH_CUDA
  if (cuda) {
    if (operation->execution_mode() != NativeOperationMode::cuda_capture)
      return false;
    operation->validate_cuda_execution_domain();
    struct Resources final : RuntimeCompletionResources {
      std::shared_ptr<NativeOperation> operation;
      std::vector<
          std::unique_ptr<taichi::lang::cuda::CudaDevice::AllocationLease>>
          inputs;
      std::size_t retained_resource_count(
          std::uint32_t) const noexcept override {
        return 0;
      }
    };
    auto resources = std::make_shared<Resources>();
    resources->operation = std::move(operation);
    auto &cuda_device = dynamic_cast<taichi::lang::cuda::CudaDevice &>(device);
    for (const auto &input : inputs) {
      auto lease = cuda_device.acquire_allocation_lease(input.pointer);
      if (!lease)
        throw std::invalid_argument("Cold initialization input is retired");
      resources->inputs.push_back(std::move(lease));
    }
    cuda->submit({}, resources,
                 [resources](void *stream) {
                   for (const auto &input : resources->inputs)
                     input->wait_for_initialization(stream);
                   resources->operation->record_cuda(stream);
                 })
        .wait();
    cuda->collect_ready(1);
    return true;
  }
#endif
  throw std::runtime_error("Cold initialization domain backend is unavailable");
}

void PreparedExecutionDomain::upload_owned_data(
    taichi::lang::Device &device,
    const std::vector<taichi::lang::NativeDataUpload> &uploads) {
  using namespace taichi::lang;
  std::vector<NativeStorageRange> ranges;
  std::size_t total = 0;
  // Validate the entire batch before allocation, command recording or enqueue.
  // In particular, Vulkan buffer_copy's clamping is not a bounds certificate.
  for (const auto &upload : uploads) {
    if (upload.destination.device != &device ||
        (upload.bytes && !upload.source) ||
        upload.bytes > std::numeric_limits<std::size_t>::max() - total)
      throw std::invalid_argument("Invalid owned data upload batch");
    ranges.push_back(resolve_native_storage_range(backend(), upload.destination,
                                                  upload.bytes));
    total += upload.bytes;
  }
  if (!total)
    return;
#ifdef TI_WITH_VULKAN
  if (vulkan) {
    auto commands = vulkan->new_command_list();
    for (const auto &upload : uploads) {
      if (!upload.bytes)
        continue;
      auto [staging, status] = device.allocate_memory_unique(
          {upload.bytes, true, false, false, AllocUsage::Upload});
      if (status != RhiResult::success)
        throw std::runtime_error("Unable to allocate cold upload storage");
      void *mapped = nullptr;
      if (device.map(*staging, &mapped) != RhiResult::success)
        throw std::runtime_error("Unable to map cold upload storage");
      std::memcpy(mapped, upload.source, upload.bytes);
      device.unmap(*staging);
      commands->buffer_copy(upload.destination, staging->get_ptr(),
                            upload.bytes);
      commands->buffer_barrier(upload.destination, upload.bytes);
      // Vulkan commands retain both native buffers and their allocation owners,
      // even after these temporary allocation-registry handles are released.
    }
    vulkan->submit(vulkan->prepare_commands({commands.get()})).wait();
    vulkan->collect_ready(1);
    return;
  }
#endif
#ifdef TI_WITH_CUDA
  if (cuda) {
    struct Resources final : RuntimeCompletionResources {
      std::shared_ptr<void> staging;
      std::vector<
          std::unique_ptr<taichi::lang::cuda::CudaDevice::AllocationLease>>
          destinations;
      std::size_t retained_resource_count(
          std::uint32_t) const noexcept override {
        return 0;  // AOT-owned construction data, not a Program resource kind.
      }
    };
    auto &cuda_device = dynamic_cast<taichi::lang::cuda::CudaDevice &>(device);
    auto resources = std::make_shared<Resources>();
    for (const auto &upload : uploads) {
      auto lease = cuda_device.acquire_allocation_lease(upload.destination);
      if (!lease)
        throw std::invalid_argument("Cold upload destination is retired");
      resources->destinations.push_back(std::move(lease));
    }
    // Reuse the domain's pinned cold storage pool; do not unpin on submission
    // completion, which can synchronize unrelated streams on some drivers.
    resources->staging = cuda->acquire_parameter_storage(total);
    auto *data = static_cast<char *>(resources->staging.get());
    std::size_t offset = 0;
    for (const auto &upload : uploads) {
      if (upload.bytes)
        std::memcpy(data + offset, upload.source, upload.bytes);
      offset += upload.bytes;
    }
    cuda->submit(
            {}, resources,
            [&](void *stream) {
              std::size_t offset = 0;
              for (std::size_t i = 0; i < uploads.size(); ++i) {
                if (uploads[i].bytes) {
                  resources->destinations[i]->wait_for_initialization(stream);
                  CUDADriver::get_instance().memcpy_host_to_device_async(
                      reinterpret_cast<void *>(ranges[i].memory_offset),
                      data + offset, uploads[i].bytes, stream);
                }
                offset += uploads[i].bytes;
              }
            })
        .wait();
    cuda->collect_ready(1);
    return;
  }
#endif
  throw std::runtime_error("Cold upload domain backend is unavailable");
}

PreparedExecutionDomain::PreparedExecutionDomain(
    std::shared_ptr<taichi::lang::cuda::ExecutionDomain> domain)
    : cuda(std::move(domain)) {
  if (!cuda)
    throw std::invalid_argument("Missing CUDA execution domain");
}
PreparedExecutionDomain::PreparedExecutionDomain(
    std::shared_ptr<taichi::lang::vulkan::ExecutionDomain> domain)
    : vulkan(std::move(domain)) {
  if (!vulkan)
    throw std::invalid_argument("Missing Vulkan execution domain");
}

taichi::Arch PreparedExecutionDomain::backend() const {
  return cuda ? taichi::Arch::cuda : taichi::Arch::vulkan;
}

PreparedExecutionDomain::Snapshot PreparedExecutionDomain::snapshot() const {
#ifdef TI_WITH_CUDA
  if (cuda) {
    const auto value = cuda->snapshot();
    return {value.owner, value.accepting, value.closed};
  }
#endif
#ifdef TI_WITH_VULKAN
  if (vulkan) {
    const auto value = vulkan->snapshot();
    return {value.owner, value.accepting, value.closed};
  }
#endif
  throw std::runtime_error("Execution domain backend is unavailable");
}

bool PreparedExecutionDomain::belongs_to(
    const std::shared_ptr<taichi::lang::RuntimeFaultDomain> &owner) const {
#ifdef TI_WITH_CUDA
  if (cuda)
    return cuda->belongs_to(owner);
#endif
#ifdef TI_WITH_VULKAN
  if (vulkan)
    return vulkan->belongs_to(owner);
#endif
  return false;
}

void PreparedExecutionDomain::close() {
#ifdef TI_WITH_CUDA
  if (cuda)
    return cuda->close();
#endif
#ifdef TI_WITH_VULKAN
  if (vulkan)
    return vulkan->close();
#endif
  throw std::runtime_error("Execution domain backend is unavailable");
}

bool PreparedExecutionDomain::instance_retired(
    const PreparedExecutionInstance &instance) const {
#ifdef TI_WITH_CUDA
  if (cuda)
    return cuda->instance_retired(instance.cuda);
#endif
#ifdef TI_WITH_VULKAN
  if (vulkan)
    return vulkan->instance_retired(instance.vulkan);
#endif
  return false;
}

void PreparedExecutionDomain::retire_instance(
    const PreparedExecutionInstance &instance) {
#ifdef TI_WITH_CUDA
  if (cuda)
    return cuda->retire_instance(instance.cuda);
#endif
#ifdef TI_WITH_VULKAN
  if (vulkan)
    return vulkan->retire_instance(instance.vulkan);
#endif
  throw std::runtime_error("Execution domain backend is unavailable");
}

PreparedExecutionInstance PreparedExecutionDomain::prepare_owned_instance(
    const std::shared_ptr<taichi::lang::RuntimeClient> &client,
    const std::function<void(const PreparedExecutionInstance &)> &prepare,
    std::function<std::shared_ptr<taichi::lang::RuntimeCompletionObserver>(
        void *,
        std::size_t)> collect_cuda_result) {
  PreparedExecutionInstance result;
#ifdef TI_WITH_CUDA
  if (cuda) {
    result.cuda = cuda->prepare_owned_instance(
        client, [&](const auto &instance) { prepare({instance, {}}); },
        std::move(collect_cuda_result));
    return result;
  }
#endif
#ifdef TI_WITH_VULKAN
  if (vulkan) {
    if (collect_cuda_result)
      throw std::invalid_argument(
          "CUDA result collector requires a CUDA domain");
    result.vulkan = vulkan->prepare_owned_instance(client, [&](const auto &instance) {
      prepare({{}, instance});
    });
    return result;
  }
#endif
  throw std::runtime_error("Execution domain backend is unavailable");
}

}  // namespace capi
