#include "taichi/runtime/native_array_contract.h"

#include <limits>
#include <type_traits>
#ifdef TI_WITH_LLVM
#include "taichi/rhi/cpu/cpu_device.h"
#endif
#ifdef TI_WITH_CUDA
#include "taichi/rhi/cuda/cuda_device.h"
#endif
#ifdef TI_WITH_VULKAN
#include "taichi/rhi/vulkan/vulkan_device.h"
#endif

namespace taichi::lang {
namespace {
bool disjoint(std::uint64_t a,
              std::uint64_t a_size,
              std::uint64_t b,
              std::uint64_t b_size) noexcept {
  return a <= b ? a_size <= b - a : b_size <= a - b;
}

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b) {
  if (b > std::numeric_limits<std::uint64_t>::max() - a)
    throw std::invalid_argument("Native storage range overflows address space");
  return a + b;
}

void extent(std::uint64_t offset, std::uint64_t bytes, std::uint64_t capacity) {
  if (offset > capacity || bytes > capacity - offset)
    throw std::invalid_argument("Native storage range exceeds allocation");
}

template <typename Handle>
std::uint64_t identity(Handle handle) {
  if constexpr (std::is_pointer_v<Handle>)
    return reinterpret_cast<std::uintptr_t>(handle);
  else
    return static_cast<std::uint64_t>(handle);
}
}  // namespace

bool NativeStorageRange::proven_disjoint(
    const NativeStorageRange &other) const noexcept {
  if (!device || device != other.device)
    return false;
  if (!bytes || !other.bytes)
    return true;
  if (buffer && buffer == other.buffer)
    return disjoint(buffer_offset, bytes, other.buffer_offset, other.bytes);
  if (!memory || !other.memory)
    return false;
  return memory != other.memory ||
         disjoint(memory_offset, bytes, other.memory_offset, other.bytes);
}

bool NativeStorageRange::proven_same_range(
    const NativeStorageRange &other) const noexcept {
  if (!device || device != other.device || bytes != other.bytes)
    return false;
  if (buffer && buffer == other.buffer)
    return buffer_offset == other.buffer_offset;
  return memory && memory == other.memory &&
         memory_offset == other.memory_offset;
}

NativeStorageRange resolve_native_storage_range(Arch arch,
                                                DevicePtr pointer,
                                                std::uint64_t bytes) {
  if (!pointer.device)
    throw std::invalid_argument("Native storage requires a live device");
  NativeStorageRange result;
  result.device = pointer.device;
  result.bytes = bytes;
#ifdef TI_WITH_LLVM
  if (arch_is_cpu(arch)) {
    auto *device = dynamic_cast<cpu::CpuDevice *>(pointer.device);
    void *address = nullptr;
    if (!device ||
        device->map_range_for_cpu_native(pointer, bytes, &address) !=
            RhiResult::success ||
        !address)
      throw std::invalid_argument("Native CPU storage range is invalid");
    result.memory = 1;  // One virtual address space within this owner.
    result.memory_offset = identity(address);
    checked_add(result.memory_offset, bytes);
    return result;
  }
#endif
#ifdef TI_WITH_CUDA
  if (arch == Arch::cuda) {
    auto *device = dynamic_cast<cuda::CudaDevice *>(pointer.device);
    if (!device)
      throw std::invalid_argument("Native CUDA storage requires a CUDA device");
    const auto allocation = device->get_alloc_info(pointer);
    if (!allocation.ptr)
      throw std::invalid_argument("Native CUDA storage is unavailable");
    extent(pointer.offset, bytes, allocation.size);
    result.memory = 1;
    result.memory_offset =
        checked_add(identity(allocation.ptr), pointer.offset);
    checked_add(result.memory_offset, bytes);
    return result;
  }
#endif
#ifdef TI_WITH_VULKAN
  if (arch == Arch::vulkan) {
    auto *device = dynamic_cast<vulkan::VulkanDevice *>(pointer.device);
    if (!device)
      throw std::invalid_argument(
          "Native Vulkan storage requires a Vulkan device");
    const auto [memory, offset, capacity] =
        device->get_vkmemory_offset_size(pointer);
    extent(pointer.offset, bytes, capacity);
    const auto buffer = device->get_vkbuffer(pointer);
    if (!buffer)
      throw std::invalid_argument("Native Vulkan buffer is unavailable");
    result.memory = identity(memory);
    result.memory_offset = checked_add(offset, pointer.offset);
    checked_add(result.memory_offset, bytes);
    result.buffer = identity(buffer->buffer);
    result.buffer_offset = pointer.offset;
    return result;
  }
#endif
  throw std::invalid_argument("Native storage requires CPU, CUDA or Vulkan");
}

void validate_native_array_layout(
    const NativeArrayBinding &array,
    const NativeArrayLayoutRequirement &required) {
  if (!required.record_bytes || !required.alignment ||
      array.shape.size() != required.minimum_shape.size() ||
      array.element_shape != required.element_shape)
    throw std::invalid_argument(
        "Array layout does not match its required shape");
  if (required.exact_shape && array.shape != required.minimum_shape)
    throw std::invalid_argument("Array does not have the required exact shape");
  for (std::size_t i = 0; i < array.shape.size(); ++i)
    if (array.shape[i] < required.minimum_shape[i])
      throw std::invalid_argument("Array shape is smaller than required");
  std::uint64_t record = data_type_size(array.scalar_type);
  if (!record)
    throw std::invalid_argument("Array requires a scalar storage type");
  for (auto axis : array.element_shape) {
    if (!axis || axis > std::numeric_limits<std::uint64_t>::max() / record)
      throw std::invalid_argument("Array record size overflows");
    record *= axis;
  }
  const auto complementary_offset =
      (required.alignment - array.base_offset % required.alignment) %
      required.alignment;
  if (record != required.record_bytes ||
      array.pointer.offset % required.alignment != complementary_offset)
    throw std::invalid_argument(
        "Array record size or alignment is incompatible");
}

}  // namespace taichi::lang
