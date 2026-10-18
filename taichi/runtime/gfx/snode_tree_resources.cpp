#include "taichi/runtime/gfx/snode_tree_resources.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "taichi/codegen/spirv/snode_struct_compiler.h"

namespace taichi::lang::gfx {
namespace {

std::size_t checked_product(std::size_t lhs, std::size_t rhs) {
  if (rhs != 0 && lhs > std::numeric_limits<std::size_t>::max() / rhs) {
    throw std::overflow_error("SNode list capacity overflows host size_t");
  }
  return lhs * rhs;
}

std::size_t checked_offset(std::size_t base, std::size_t offset) {
  if (offset > std::numeric_limits<std::size_t>::max() - base) {
    throw std::overflow_error("SNode resource offset overflows host size_t");
  }
  return base + offset;
}

bool needs_list(const SNode *node) {
  if (!node || node->type == SNodeType::root || node->is_place() ||
      node->is_path_all_dense) {
    return false;
  }
  return node->type == SNodeType::dense || node->need_activation();
}

}  // namespace

void validate_snode_tree_resources(const SNodeTreeResources &resources) {
  auto require = [](bool condition, const char *message) {
    if (!condition)
      throw std::invalid_argument(message);
  };
  const auto &ids = resources.snode_ids;
  require(std::is_sorted(ids.begin(), ids.end()) &&
              std::adjacent_find(ids.begin(), ids.end()) == ids.end() &&
              (ids.empty() || ids.front() >= 0),
          "SNode resource IDs must be sorted, unique and nonnegative");
  require(resources.listgen_capacity_entries <=
              std::numeric_limits<uint32_t>::max(),
          "SNode list capacity exceeds uint32 shader indexing");
  auto owns = [&](int id) {
    return std::binary_search(ids.begin(), ids.end(), id);
  };
  using Range = std::pair<std::size_t, std::size_t>;
  std::vector<Range> root_ranges;
  auto append = [&](std::vector<Range> &ranges, std::size_t offset,
                    std::size_t bytes, std::size_t extent) {
    require(offset % 4 == 0 && bytes % 4 == 0 && bytes != 0,
            "SNode allocator ranges must be nonempty and word aligned");
    require(offset <= extent && bytes <= extent - offset,
            "SNode allocator range exceeds its buffer");
    require(uint64_t(offset) + uint64_t(bytes) <= (uint64_t(1) << 32),
            "SNode allocator range exceeds uint32 shader addressing");
    ranges.emplace_back(offset, offset + bytes);
  };
  auto disjoint = [&](std::vector<Range> &ranges) {
    std::sort(ranges.begin(), ranges.end());
    for (std::size_t i = 1; i < ranges.size(); ++i)
      require(ranges[i - 1].second <= ranges[i].first,
              "SNode allocator storage ranges overlap");
  };
  int previous = -1;
  for (const auto &pool : resources.pointer_pools) {
    const auto &c = pool.contract;
    require(c.snode_id > previous && owns(c.snode_id),
            "Pointer pools must have ordered, unique, owned SNode IDs");
    previous = c.snode_id;
    const bool independent = c.pool_buffer_binding_id >= 0;
    require((independent && c.pool_buffer_binding_id == c.snode_id &&
             pool.independent_size > 0 && pool.independent_size % 4 == 0) ||
                (!independent && c.pool_buffer_binding_id == -1 &&
                 pool.independent_size == 0),
            "Pointer pool binding and storage ownership disagree");
    require(c.pool_capacity > 0 && c.cell_stride_bytes > 0 &&
                c.cell_stride_bytes % 4 == 0 &&
                uint64_t(c.watermark_offset) + 4 <= c.pool_data_offset,
            "Pointer pool capacity, stride or watermark is invalid");
    using Contract = spirv::SpirvAllocatorContract;
    require(c.alloc_protocol == Contract::AllocProtocol::Legacy ||
                c.alloc_protocol == Contract::AllocProtocol::CasMarker,
            "Unknown pointer allocation protocol");
    // This is diagnostic provenance, not an allocation input: capacity is
    // already baked. In particular, -1 is the compiler's unscaled sentinel.
    require(std::isfinite(c.pool_fraction),
            "Invalid pointer pool fraction");
    if (c.allocator_kind == Contract::AllocatorKind::Bump) {
      require(c.chunk_log2_capacity == 0 && c.max_chunks == 0 &&
                  c.chunk_descriptor_array_first_binding == -1,
              "Bump pool contains chunk descriptor state");
    } else {
      require(c.allocator_kind == Contract::AllocatorKind::Chunked &&
                  c.chunk_log2_capacity < 32 && c.max_chunks > 0 &&
                  c.chunk_descriptor_array_first_binding == -1 &&
                  !c.deterministic_slot,
              "Unsupported chunked pool contract");
      const uint64_t cells = uint64_t(1) << c.chunk_log2_capacity;
      require(cells * c.max_chunks >= c.pool_capacity &&
                  cells * c.cell_stride_bytes <=
                      std::numeric_limits<uint32_t>::max(),
              "Chunk geometry exceeds its capacity or shader addressing");
    }
    std::vector<Range> independent_ranges;
    auto &ranges = independent ? independent_ranges : root_ranges;
    const auto extent =
        independent ? pool.independent_size : resources.root_size;
    append(ranges, c.watermark_offset, 4, extent);
    append(ranges, c.pool_data_offset,
           checked_product(c.pool_capacity, c.cell_stride_bytes), extent);
    if (c.has_freelist) {
      append(ranges, c.freelist_head_offset, 4, extent);
      append(ranges, c.freelist_links_offset,
             checked_product(c.pool_capacity, 4), extent);
    }
    if (c.has_ambient_zone)
      append(ranges, c.ambient_offset, c.cell_stride_bytes, extent);
    if (independent)
      disjoint(ranges);
  }
  previous = -1;
  for (const auto &watch : resources.hash_watches) {
    require(watch.snode_id > previous && owns(watch.snode_id),
            "Hash watches must have ordered, unique, owned SNode IDs");
    previous = watch.snode_id;
    const auto pool = std::lower_bound(
        resources.pointer_pools.begin(), resources.pointer_pools.end(),
        watch.snode_id,
        [](const auto &entry, int id) { return entry.contract.snode_id < id; });
    require(pool == resources.pointer_pools.end() ||
                pool->contract.snode_id != watch.snode_id,
            "SNode cannot own both a pointer pool and hash watch");
    append(root_ranges, watch.overflow_offset, 4, resources.root_size);
    append(root_ranges, watch.active_count_offset, 4, resources.root_size);
    if (watch.tombstone_offset != static_cast<std::size_t>(-1))
      append(root_ranges, watch.tombstone_offset, 4, resources.root_size);
  }
  disjoint(root_ranges);
}

SNodeTreeResources describe_snode_tree_resources(
    const spirv::CompiledSNodeStructs &compiled) {
  SNodeTreeResources result;
  result.root_size = compiled.root_size;
  result.snode_ids.reserve(compiled.snode_descriptors.size());
  for (const auto &[id, desc] : compiled.snode_descriptors) {
    result.snode_ids.push_back(id);
    if (needs_list(desc.snode)) {
      std::size_t entries = desc.total_num_cells_from_root;
      std::size_t suffix = 1;
      for (auto *node = desc.snode; node && node->type != SNodeType::root;
           node = node->parent) {
        const auto it = compiled.snode_descriptors.find(node->id);
        if (it != compiled.snode_descriptors.end() &&
            node->type == SNodeType::hash &&
            it->second.hash_table_capacity > 0) {
          entries = checked_product(it->second.hash_table_capacity, suffix);
          break;
        }
        suffix = checked_product(suffix, node->num_cells_per_container);
      }
      result.listgen_capacity_entries =
          std::max(result.listgen_capacity_entries, entries);
    }
    // Preserve the existing root-level counter watches. Nested overflows use
    // the shared device diagnostic buffer instead of host topology walks.
    if (desc.snode && desc.snode->type == SNodeType::hash &&
        desc.hash_table_capacity > 0 && desc.snode->parent &&
        desc.snode->parent->type == SNodeType::root) {
      const auto base = desc.mem_offset_in_parent_cell;
      const auto no_offset = static_cast<std::size_t>(-1);
      result.hash_watches.push_back(
          {id, checked_offset(base, desc.hash_overflow_count_offset),
           checked_offset(base, desc.hash_active_count_offset),
           desc.hash_tombstone_count_offset == no_offset
               ? no_offset
               : checked_offset(base, desc.hash_tombstone_count_offset)});
    }
  }
  for (const auto &[id, contract] : compiled.pointer_contracts) {
    TI_ASSERT_INFO(contract.snode_id == id,
                   "Pointer resource identity does not match SNode {}", id);
    std::size_t size = 0;
    if (contract.pool_buffer_binding_id >= 0) {
      const auto it = compiled.pool_buffer_sizes.find(id);
      TI_ASSERT_INFO(it != compiled.pool_buffer_sizes.end() && it->second > 0,
                     "Pool buffer size missing or zero for pointer SNode {}",
                     id);
      size = it->second;
    }
    result.pointer_pools.push_back({contract, size});
  }
  std::sort(result.snode_ids.begin(), result.snode_ids.end());
  std::sort(result.pointer_pools.begin(), result.pointer_pools.end(),
            [](const auto &a, const auto &b) {
              return a.contract.snode_id < b.contract.snode_id;
            });
  std::sort(
      result.hash_watches.begin(), result.hash_watches.end(),
      [](const auto &a, const auto &b) { return a.snode_id < b.snode_id; });
  return result;
}

}  // namespace taichi::lang::gfx
