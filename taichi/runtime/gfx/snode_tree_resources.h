#pragma once

#include <cstddef>
#include <vector>

#include "taichi/codegen/spirv/spirv_allocator_contract.h"
#include "taichi/runtime/snode_resource_id.h"

namespace taichi::lang {
namespace spirv {
struct CompiledSNodeStructs;
}
namespace gfx {

struct SparseTaskResources {
  SNodeResourceId list;
  SNodeResourceId parent;
  SNodeResourceId mutation;
};

// Cold materialization facts, independent of live SNodes and device handles.
// Offsets remain relative to the physical root/pool selected by the compiler.
struct SNodeTreeResources {
  struct PointerPool {
    spirv::SpirvAllocatorContract contract;
    std::size_t independent_size{0};
  };
  struct HashWatch {
    int snode_id{-1};
    std::size_t overflow_offset{0};
    std::size_t active_count_offset{0};
    std::size_t tombstone_offset{static_cast<std::size_t>(-1)};
  };

  std::size_t root_size{0};
  std::size_t listgen_capacity_entries{0};
  // Stable order also permits cold module-local identity validation/remapping.
  std::vector<int> snode_ids;
  std::vector<PointerPool> pointer_pools;
  std::vector<HashWatch> hash_watches;
};

SNodeTreeResources describe_snode_tree_resources(
    const spirv::CompiledSNodeStructs &compiled);

// Validate cold layout facts before allocating any device storage. The same
// contract applies to compiler projections and independently loaded artifacts.
// Limits follow the shader's uint32 addressing and the actual buffer extents.
void validate_snode_tree_resources(const SNodeTreeResources &resources);

}  // namespace gfx
}  // namespace taichi::lang
