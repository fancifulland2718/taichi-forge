#include "taichi/runtime/native_array_contract.h"

#include <algorithm>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <tuple>
#include <type_traits>
#include "taichi/ir/type_utils.h"
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

namespace {
void validate_access_group(const std::vector<NativeStorageRange> &reads,
                           const std::vector<NativeStorageWrite> &writes,
                           bool exclusive_writes) {
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
  bool unknown_read = false;
  auto unknown = [&](const NativeStorageRange &range, bool write) {
    if (range.bytes && !range.memory) {
      if (!range.buffer || (unknown_buffer && *unknown_buffer != range.buffer))
        throw std::invalid_argument(
            "Cannot prove imported array group disjoint");
      unknown_buffer = range.buffer;
      unknown_write |= write;
      unknown_read |= !write;
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
  if (!exclusive_writes &&
      std::none_of(reads.begin(), reads.end(),
                   [](const auto &range) { return range.bytes != 0; }))
    return;
  for (const auto &read : reads)
    unknown(read, false);
  for (const auto &write : writes)
    unknown(write.range, true);
  if (unknown_buffer) {
    // Unknown memory can only be certified against the same buffer. If an
    // unknown port writes, every nonempty read must also use that buffer.
    if (exclusive_writes || unknown_read)
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
    if (!exclusive_writes) {
      std::size_t active_reads = 0, active_writes = 0;
      for (const auto &event : events) {
        auto &active = event.write ? active_writes : active_reads;
        if (!event.begin) {
          --active;
        } else {
          if (event.write ? active_reads != 0 : active_writes != 0)
            throw std::invalid_argument(
                "Array read group overlaps write group");
          ++active;
        }
      }
      return;
    }
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
  if (!exclusive_writes || !unknown_write)
    sweep(false);
  if (unknown_buffer)
    sweep(true);
}
}  // namespace

void validate_native_disjoint_writes(
    const std::vector<NativeStorageRange> &reads,
    const std::vector<NativeStorageWrite> &writes) {
  validate_access_group(reads, writes, true);
}

void validate_native_read_write_disjoint(
    const std::vector<NativeStorageRange> &reads,
    const std::vector<NativeStorageRange> &writes) {
  std::vector<NativeStorageWrite> outputs;
  outputs.reserve(writes.size());
  for (const auto &write : writes)
    outputs.push_back({write, std::nullopt});
  validate_access_group(reads, outputs, false);
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

void validate_native_allocation_groups_disjoint(
    Arch arch,
    const std::vector<DeviceAllocation> &first,
    const std::vector<DeviceAllocation> &second) {
  if (first.empty() || second.empty())
    return;
  const auto resolve = [&](const std::vector<DeviceAllocation> &allocations) {
    std::vector<NativeStorageRange> ranges;
    std::set<std::pair<Device *, DeviceAllocationId>> seen;
    for (const auto allocation : allocations) {
      if (!seen.emplace(allocation.device, allocation.alloc_id).second)
        continue;
      std::optional<std::uint64_t> bytes;
#ifdef TI_WITH_LLVM
      if (arch_is_cpu(arch))
        if (auto *device = dynamic_cast<cpu::CpuDevice *>(allocation.device))
          bytes = device->get_alloc_info(allocation).size;
#endif
#ifdef TI_WITH_CUDA
      if (arch == Arch::cuda)
        if (auto *device = dynamic_cast<cuda::CudaDevice *>(allocation.device))
          bytes = device->get_alloc_info(allocation).size;
#endif
#ifdef TI_WITH_VULKAN
      if (arch == Arch::vulkan)
        if (auto *device = dynamic_cast<vulkan::VulkanDevice *>(allocation.device))
          bytes = device->get_vkbuffer_size(allocation);
#endif
      if (!bytes)
        throw std::invalid_argument("Allocation group requires its backend device");
      ranges.push_back(
          resolve_native_storage_range(arch, allocation.get_ptr(), *bytes));
    }
    return ranges;
  };
  validate_native_read_write_disjoint(resolve(first), resolve(second));
}

void validate_native_shader_storage(Arch arch, DeviceAllocation allocation) {
#ifdef TI_WITH_VULKAN
  if (arch == Arch::vulkan) {
    auto *device = dynamic_cast<vulkan::VulkanDevice *>(allocation.device);
    if (!device ||
        (device->allocation_usage(allocation) & AllocUsage::Storage) ==
            AllocUsage::None)
      throw std::invalid_argument(
          "Vulkan kernel ndarray requires storage buffer usage");
  }
#endif
}

void validate_native_array_layout(
    const NativeArrayBinding &array,
    const NativeArrayLayoutRequirement &required) {
  if (array.strides)
    throw std::invalid_argument("Array requires explicit affine qualification");
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

NativeDenseArrayLayout validate_native_dense_array_layout(
    const NativeArrayBinding &array) {
  if (!array.scalar_type->is<PrimitiveType>() || is_quant(array.scalar_type) ||
      array.scalar_type->is_primitive(PrimitiveTypeID::unknown))
    throw std::invalid_argument("Dense array requires a scalar storage type");
  const auto width = data_type_size(array.scalar_type);
  if (width <= 0)
    throw std::invalid_argument("Dense array has an invalid scalar width");
  NativeDenseArrayLayout result;
  auto &layout = result.layout;
  layout.scalar_type = array.scalar_type;
  auto shape = [](const auto &source, bool element) {
    std::vector<std::int64_t> result;
    result.reserve(source.size());
    for (auto axis : source) {
      if (axis > std::uint64_t(std::numeric_limits<std::int64_t>::max()) ||
          (element && !axis))
        throw std::invalid_argument("Dense array has an invalid shape");
      result.push_back(static_cast<std::int64_t>(axis));
    }
    return result;
  };
  layout.index_shape = shape(array.shape, false);
  layout.element_shape = shape(array.element_shape, true);
  if (array.strides) {
    layout.index_strides_bytes = array.strides->index_bytes;
    layout.element_strides_bytes = array.strides->element_bytes;
    layout.byte_offset = array.strides->byte_offset;
  } else {
    std::int64_t stride = width;
    auto packed = [&](const auto &shape, auto &strides) {
      strides.resize(shape.size());
      for (std::size_t i = shape.size(); i > 0; --i) {
        strides[i - 1] = stride;
        if (shape[i - 1] &&
            stride > std::numeric_limits<std::int64_t>::max() / shape[i - 1])
          throw std::invalid_argument("Dense array packed strides overflow");
        stride *= shape[i - 1];
      }
    };
    packed(layout.element_shape, layout.element_strides_bytes);
    packed(layout.index_shape, layout.index_strides_bytes);
  }
  const auto analysis = storage::analyze_dense_storage_layout(layout);
  if (!analysis)
    throw std::invalid_argument("Dense array layout is invalid: " +
                                std::to_string(static_cast<int>(analysis.reason)));
  result.properties = analysis.properties;
  auto &p = result.properties;
  if (std::uint64_t(p.reachable_end) > array.bytes)
    throw std::invalid_argument("Dense array exceeds its accessible storage window");
  // Reduce before adding: even imported origins at UINT64_MAX must not wrap.
  const auto alignment = p.scalar_alignment;
  const auto origin_remainder =
      (array.base_offset % alignment + array.pointer.offset % alignment) %
      alignment;
  p.aligned =
      (origin_remainder + std::uint64_t(layout.byte_offset) % alignment) %
          alignment ==
      0;
  auto aligned_strides = [&](const auto &shape, const auto &strides) {
    for (std::size_t i = 0; i < shape.size(); ++i)
      if (shape[i] > 1 && strides[i] % std::int64_t(alignment))
        return false;
    return true;
  };
  p.aligned &= aligned_strides(layout.index_shape, layout.index_strides_bytes) &&
               aligned_strides(layout.element_shape,
                               layout.element_strides_bytes);
  if (!p.aligned)
    throw std::invalid_argument("Dense array storage is misaligned");
  return result;
}

namespace {
NativeStorageRange subrange(NativeStorageRange range,
                            std::uint64_t offset,
                            std::uint64_t bytes) {
  extent(offset, bytes, range.bytes);
  range.memory_offset = checked_add(range.memory_offset, offset);
  range.buffer_offset = checked_add(range.buffer_offset, offset);
  range.bytes = bytes;
  return range;
}

std::uint64_t stride_gcd(const storage::DenseStorageLayoutSpec &layout) {
  std::uint64_t result = 0;
  auto include = [&](const auto &shape, const auto &strides) {
    for (std::size_t i = 0; i < shape.size(); ++i) {
      if (shape[i] <= 1)
        continue;
      const auto stride = strides[i];
      const auto magnitude = stride < 0 ? std::uint64_t(-(stride + 1)) + 1
                                        : std::uint64_t(stride);
      result = std::gcd(result, magnitude);
    }
  };
  include(layout.index_shape, layout.index_strides_bytes);
  include(layout.element_shape, layout.element_strides_bytes);
  return result;
}
}  // namespace

NativeDenseArrayView::NativeDenseArrayView(NativeDenseArrayLayout layout,
                                         NativeStorageRange storage)
    : layout_(std::move(layout)) {
  const auto &p = layout_.properties;
  origin_ = subrange(storage, layout_.layout.byte_offset,
                     p.empty ? 0 : p.scalar_size);
  range_ = subrange(storage, p.reachable_begin,
                    p.reachable_end - p.reachable_begin);
  stride_gcd_ = stride_gcd(layout_.layout);
  const auto origin_offset =
      origin_.buffer ? origin_.buffer_offset : origin_.memory_offset;
  if (origin_offset % p.scalar_alignment)
    throw std::invalid_argument("Dense array physical origin is misaligned");
}

NativeDenseArrayView resolve_native_dense_array(
    Arch arch, const NativeArrayBinding &array) {
  auto layout = validate_native_dense_array_layout(array);
  auto storage = resolve_native_storage_range(arch, array.pointer, array.bytes);
  return NativeDenseArrayView(std::move(layout), storage);
}

bool NativeDenseArrayView::proven_disjoint(
    const NativeDenseArrayView &other) const noexcept {
  if (range_.proven_disjoint(other.range_))
    return true;
  if (!origin_.device || origin_.device != other.origin_.device)
    return false;
  std::uint64_t a, b;
  if (origin_.buffer && origin_.buffer == other.origin_.buffer) {
    a = origin_.buffer_offset;
    b = other.origin_.buffer_offset;
  } else if (origin_.memory && origin_.memory == other.origin_.memory) {
    a = origin_.memory_offset;
    b = other.origin_.memory_offset;
  } else {
    return false;
  }
  // All scalar starts are congruent to their origin modulo this GCD. Prove
  // separation only if the scalar byte intervals cannot meet on that lattice.
  const auto step = std::gcd(stride_gcd_, other.stride_gcd_);
  if (!step)
    return false;
  const auto residue = b >= a ? (b - a) % step
                             : (step - (a - b) % step) % step;
  return residue >= layout_.properties.scalar_size &&
         step - residue >= other.layout_.properties.scalar_size;
}

bool NativeDenseArrayView::proven_same_mapping(
    const NativeDenseArrayView &other) const noexcept {
  const auto &a = layout_.layout;
  const auto &b = other.layout_.layout;
  if (!origin_.proven_same_range(other.origin_) ||
      a.scalar_type != b.scalar_type || a.index_shape != b.index_shape ||
      a.element_shape != b.element_shape)
    return false;
  auto same_strides = [](const auto &shape, const auto &a, const auto &b) {
    for (std::size_t i = 0; i < shape.size(); ++i)
      if (shape[i] > 1 && a[i] != b[i])
        return false;
    return true;
  };
  return same_strides(a.index_shape, a.index_strides_bytes,
                      b.index_strides_bytes) &&
         same_strides(a.element_shape, a.element_strides_bytes,
                      b.element_strides_bytes);
}

}  // namespace taichi::lang
