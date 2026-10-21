#include "taichi/runtime/native_argpack_storage.h"
#include "taichi/runtime/native_array_contract.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace taichi::lang {
NativeArgPackStorage::NativeArgPackStorage(
    Arch arch, Device &device,
    std::shared_ptr<const aot::ArgPackLayout> layout, const Values &values)
    : arch_(arch), device_(device), layout_(std::move(layout)) {
  const bool host = arch_is_cpu(arch) || arch == Arch::cuda;
  if ((!host && arch != Arch::vulkan) || !layout_ ||
      host != (layout_->value_storage() == aot::ValueStorage::Host))
    throw std::invalid_argument("ArgPack storage requires a matching backend layout");
  const auto &packs = layout_->storages();
  std::size_t value_count = 0;
  // Validate every payload before allocating or uploading anything. User
  // padding is not copied, so it cannot populate a nested pointer or leak
  // uninitialized bytes into a device packet.
  for (const auto &pack : packs) {
    const auto bytes = std::max<std::size_t>(pack.bytes, 1);
    if (bytes > std::numeric_limits<std::size_t>::max() - bytes_)
      throw std::invalid_argument("ArgPack storage byte count overflows");
    bytes_ += bytes;
    for (const auto &value : pack.values) {
      const auto found = values.find(value.origin.logical);
      if (found == values.end() ||
          found->second.size() != value.layout.buffer_size)
        throw std::invalid_argument("ArgPack value does not match its compiled window");
      ++value_count;
    }
  }
  if (value_count != values.size())
    throw std::invalid_argument("Unexpected ArgPack value payload");
  std::vector<std::vector<std::uint8_t>> buffers;
  for (const auto &pack : packs) {
    // Empty resource-only packs still need a valid allocation descriptor.
    const auto bytes = std::max<std::size_t>(pack.bytes, 1);
    auto [allocation, status] = device.allocate_memory_unique(
        {bytes, false, false, false, AllocUsage::Storage | AllocUsage::Uniform});
    if (status != RhiResult::success)
      throw std::runtime_error("Unable to allocate ArgPack storage");
    const auto range = resolve_native_storage_range(arch, allocation->get_ptr(), bytes);
    addresses_.push_back(host ? range.memory_offset : 0);
    allocations_.push_back(std::move(allocation));
    buffers.emplace_back(bytes, 0);
  }
  for (std::size_t i = 0; i < packs.size(); ++i) {
    auto &buffer = buffers[i];
    for (const auto &value : packs[i].values) {
      const auto &input = values.at(value.origin.logical);
      for (const auto &leaf : value.layout.members) {
        if (leaf.component_offsets.empty()) {
          std::memcpy(buffer.data() + value.offset + leaf.offset,
                      input.data() + leaf.offset, leaf.size);
        } else {
          for (const auto offset : leaf.component_offsets)
            std::memcpy(buffer.data() + value.offset + offset,
                        input.data() + offset, leaf.scalar_size);
        }
      }
    }
    if (host)
      for (const auto &child : packs[i].children) {
        const auto address = addresses_.at(child.storage);
        std::memcpy(buffer.data() + child.pointer_offset, &address, sizeof(address));
      }
  }
  std::vector<DevicePtr> destinations;
  std::vector<const void *> sources;
  std::vector<std::size_t> sizes;
  for (std::size_t i = 0; i < buffers.size(); ++i) {
    destinations.push_back(allocations_[i]->get_ptr());
    sources.push_back(buffers[i].data());
    sizes.push_back(buffers[i].size());
  }
  // The device API has an int count; partition calls instead of imposing a
  // library member-count limit. Upload completion also makes failure cleanup
  // safe before the partially built owner is discarded.
  for (std::size_t begin = 0; begin < sizes.size();) {
    const auto count = std::min<std::size_t>(sizes.size() - begin,
                                           std::numeric_limits<int>::max());
    if (device.upload_data(destinations.data() + begin, sources.data() + begin,
                           sizes.data() + begin, static_cast<int>(count)) !=
        RhiResult::success)
      throw std::runtime_error("Unable to upload ArgPack storage");
    begin += count;
  }
}
}  // namespace taichi::lang
