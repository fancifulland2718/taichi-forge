#include "gtest/gtest.h"

#ifdef TI_WITH_VULKAN
#include <limits>
#include <stdexcept>

#include "taichi/codegen/spirv/snode_struct_compiler.h"
#include "taichi/runtime/gfx/snode_tree_resources.h"

namespace taichi::lang {

TEST(GfxSNodeResources, DenseLayoutHasNoSparseResources) {
  SNode root{0, SNodeType::root};
  auto &dense = root.dense({Axis{0}}, 8);
  dense.insert_children(SNodeType::place).dt = PrimitiveType::i32;
  const auto compiled = spirv::compile_snode_structs(root);
  const auto resources = gfx::describe_snode_tree_resources(compiled);
  EXPECT_EQ(resources.root_size, compiled.root_size);
  EXPECT_EQ(resources.listgen_capacity_entries, 0);
  EXPECT_EQ(resources.snode_ids.size(), compiled.snode_descriptors.size());
  EXPECT_TRUE(resources.pointer_pools.empty());
  EXPECT_TRUE(resources.hash_watches.empty());
}

TEST(GfxSNodeResources, PointerPoliciesSurviveSourceTreeRetirement) {
  for (bool independent : {false, true}) {
    for (const auto *kind : {"bump", "chunked"}) {
      gfx::SNodeTreeResources resources;
      int pointer_id = -1;
      {
        SNode root{0, SNodeType::root};
        auto &pointer = root.pointer({Axis{0}}, 8);
        pointer_id = pointer.id;
        pointer.dense({Axis{0}}, 4).insert_children(SNodeType::place).dt =
            PrimitiveType::i32;
        spirv::PointerLayoutPolicy policy;
        policy.independent_pool = independent;
        policy.allocator_kind = kind;
        policy.max_chunks = 2;
        const auto compiled = spirv::compile_snode_structs(root, policy);
        resources = gfx::describe_snode_tree_resources(compiled);
        ASSERT_EQ(resources.pointer_pools.size(), 1);
        const auto &original = compiled.pointer_contracts.at(pointer_id);
        const auto &copy = resources.pointer_pools[0].contract;
        EXPECT_EQ(copy.pool_capacity, original.pool_capacity);
        EXPECT_EQ(copy.watermark_offset, original.watermark_offset);
        EXPECT_EQ(copy.pool_data_offset, original.pool_data_offset);
        EXPECT_EQ(copy.freelist_links_offset, original.freelist_links_offset);
        EXPECT_EQ(copy.ambient_offset, original.ambient_offset);
        EXPECT_EQ(copy.allocator_kind, original.allocator_kind);
        EXPECT_EQ(copy.max_chunks, original.max_chunks);
        EXPECT_EQ(copy.pool_buffer_binding_id, original.pool_buffer_binding_id);
        EXPECT_EQ(resources.pointer_pools[0].independent_size,
                  independent ? compiled.pool_buffer_sizes.at(pointer_id) : 0);
      }
      EXPECT_EQ(resources.pointer_pools[0].contract.snode_id, pointer_id);
      EXPECT_EQ(resources.listgen_capacity_entries, 32);
    }
  }
}

TEST(GfxSNodeResources, HashCapacityAndCounterOffsetsRemainPhysical) {
  SNode root{0, SNodeType::root};
  auto &hash = root.insert_children(SNodeType::hash);
  hash.num_cells_per_container = 1024;
  auto &dense = hash.dense({Axis{0}}, 4);
  auto &nested = dense.insert_children(SNodeType::hash);
  spirv::CompiledSNodeStructs compiled;
  compiled.root_size = 4096;
  auto &table = compiled.snode_descriptors[hash.id];
  table.snode = &hash;
  table.total_num_cells_from_root = 1024;
  table.hash_table_capacity = 16;
  table.mem_offset_in_parent_cell = 32;
  table.hash_active_count_offset = 128;
  table.hash_overflow_count_offset = 132;
  table.hash_tombstone_count_offset = static_cast<std::size_t>(-1);
  auto &tail = compiled.snode_descriptors[dense.id];
  tail.snode = &dense;
  tail.total_num_cells_from_root = 4096;
  auto &inner = compiled.snode_descriptors[nested.id];
  inner.snode = &nested;
  inner.hash_table_capacity = 8;
  auto resources = gfx::describe_snode_tree_resources(compiled);
  EXPECT_EQ(resources.listgen_capacity_entries, 64);
  ASSERT_EQ(resources.hash_watches.size(), 1);
  EXPECT_EQ(resources.hash_watches[0].snode_id, hash.id);
  EXPECT_EQ(resources.hash_watches[0].active_count_offset, 160);
  EXPECT_EQ(resources.hash_watches[0].overflow_offset, 164);
  EXPECT_EQ(resources.hash_watches[0].tombstone_offset,
            static_cast<std::size_t>(-1));
  table.hash_tombstone_count_offset = 136;
  resources = gfx::describe_snode_tree_resources(compiled);
  EXPECT_EQ(resources.hash_watches[0].tombstone_offset, 168);
}

TEST(GfxSNodeResources, GeometryOverflowIsRejectedWithoutCapacityThreshold) {
  SNode root{0, SNodeType::root};
  auto &hash = root.insert_children(SNodeType::hash);
  auto &dense = hash.dense({Axis{0}}, 4);
  spirv::CompiledSNodeStructs compiled;
  auto &table = compiled.snode_descriptors[hash.id];
  table.snode = &hash;
  table.hash_table_capacity = std::numeric_limits<std::size_t>::max();
  table.hash_tombstone_count_offset = static_cast<std::size_t>(-1);
  auto &tail = compiled.snode_descriptors[dense.id];
  tail.snode = &dense;
  EXPECT_THROW(gfx::describe_snode_tree_resources(compiled),
               std::overflow_error);
  table.hash_table_capacity = 16;
  table.mem_offset_in_parent_cell = std::numeric_limits<std::size_t>::max();
  table.hash_active_count_offset = 4;
  EXPECT_THROW(gfx::describe_snode_tree_resources(compiled),
               std::overflow_error);
}

}  // namespace taichi::lang
#endif
