#include "taichi/runtime/native_array_contract.h"

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <tuple>
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

void validate_native_disjoint_writes(
    const std::vector<NativeStorageRange> &reads,
    const std::vector<NativeStorageWrite> &writes) {
  Device *owner = nullptr;
  std::map<uint64_t, std::pair<uint64_t, uint64_t>> buffers;
  auto validate_owner = [&](const NativeStorageRange &range) {
    if (!range.device || (owner && owner != range.device))
      throw std::invalid_argument("Array access group requires one live owner");
    owner = range.device;
    checked_add(range.memory_offset, range.bytes);
    checked_add(range.buffer_offset, range.bytes);
    if (range.buffer && range.memory) {
      if (range.memory_offset < range.buffer_offset)
        throw std::invalid_argument("Array buffer memory offset is invalid");
      const auto identity = std::make_pair(
          range.memory, range.memory_offset - range.buffer_offset);
      const auto [entry, inserted] = buffers.emplace(range.buffer, identity);
      if (!inserted && entry->second != identity)
        throw std::invalid_argument(
            "Array buffer memory identity is inconsistent");
    }
  };
  for (const auto &read : reads)
    validate_owner(read);
  bool has_writes = false;
  std::optional<uint64_t> unknown_buffer;
  bool unknown_write = false;
  auto unknown = [&](const NativeStorageRange &range, bool write) {
    if (range.bytes && !range.memory) {
      if (!range.buffer || (unknown_buffer && *unknown_buffer != range.buffer))
        throw std::invalid_argument(
            "Cannot prove imported array group disjoint");
      unknown_buffer = range.buffer;
      unknown_write |= write;
    }
  };
  for (const auto &write : writes) {
    validate_owner(write.range);
    if (write.inplace_read && *write.inplace_read >= reads.size())
      throw std::invalid_argument(
          "In-place read index is outside the access group");
    has_writes |= write.range.bytes != 0;
  }
  if (!has_writes)
    return;
  for (const auto &read : reads)
    unknown(read, false);
  for (const auto &write : writes)
    unknown(write.range, true);
  if (unknown_buffer) {
    // Unknown memory can only be certified against the same buffer. If an
    // unknown port writes, every nonempty read must also use that buffer.
    for (const auto &write : writes)
      if (write.range.bytes && write.range.buffer != *unknown_buffer)
        throw std::invalid_argument("Cannot prove imported writes disjoint");
    if (unknown_write)
      for (const auto &read : reads)
        if (read.bytes && read.buffer != *unknown_buffer)
          throw std::invalid_argument("Cannot prove imported reads disjoint");
  }

  struct Event {
    uint64_t memory, position;
    bool begin, write;
    std::size_t index;
    auto key() const {
      // End events precede beginnings, so adjacent ranges do not overlap.
      return std::tie(memory, position, begin, write, index);
    }
  };
  auto sweep = [&](bool buffer_view) {
    std::vector<Event> events;
    auto append = [&](const NativeStorageRange &range, bool write,
                      std::size_t index) {
      if (!range.bytes ||
          (buffer_view ? range.buffer != *unknown_buffer : !range.memory))
        return;
      const auto memory = buffer_view ? 1 : range.memory;
      const auto begin =
          buffer_view ? range.buffer_offset : range.memory_offset;
      events.push_back({memory, begin, true, write, index});
      events.push_back(
          {memory, checked_add(begin, range.bytes), false, write, index});
    };
    for (std::size_t i = 0; i < reads.size(); ++i)
      append(reads[i], false, i);
    for (std::size_t i = 0; i < writes.size(); ++i)
      append(writes[i].range, true, i);
    std::sort(events.begin(), events.end(),
              [](const auto &a, const auto &b) { return a.key() < b.key(); });
    std::set<std::size_t> active_reads;
    std::optional<std::size_t> active_write;
    auto allowed = [&](std::size_t write, std::size_t read) {
      return writes[write].inplace_read == read &&
             writes[write].range.proven_same_range(reads[read]);
    };
    for (const auto &event : events) {
      if (!event.begin) {
        if (event.write)
          active_write.reset();
        else
          active_reads.erase(event.index);
      } else if (event.write) {
        if (active_write || active_reads.size() > 1 ||
            (!active_reads.empty() &&
             !allowed(event.index, *active_reads.begin())))
          throw std::invalid_argument(
              "Array group write overlaps another port");
        active_write = event.index;
      } else {
        if (active_write && !allowed(*active_write, event.index))
          throw std::invalid_argument(
              "Array group read overlaps another output");
        active_reads.insert(event.index);
      }
    }
  };
  if (!unknown_write)
    sweep(false);
  if (unknown_buffer)
    sweep(true);
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
    extent(pointer.offset, bytes, device->get_vkbuffer_size(pointer));
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
  if (!required.dtype.empty() &&
      array.scalar_type->to_string() != required.dtype)
    throw std::invalid_argument(
        "Array does not have the required scalar dtype");
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
