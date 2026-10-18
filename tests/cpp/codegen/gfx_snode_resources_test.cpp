#include "gtest/gtest.h"

#ifdef TI_WITH_VULKAN
#include <limits>
#include <stdexcept>

#include "taichi/codegen/spirv/snode_struct_compiler.h"
#include "taichi/runtime/gfx/snode_tree_resources.h"
#include "taichi/runtime/gfx/runtime.h"
#include "taichi/codegen/spirv/spirv_ir_builder.h"
#include "taichi/program/launch_context_builder.h"
#include "taichi/rhi/vulkan/vulkan_device_creator.h"
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"

namespace taichi::lang {

TEST(GfxSNodeResources, RepeatedLocalIdsHaveIndependentListsAndRetirement) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  vulkan::VulkanDeviceCreator::Params device_params;
  vulkan::VulkanDeviceCreator creator(device_params);
  auto *device = creator.device();
  gfx::GfxRuntime::Params runtime_params;
  runtime_params.device = device;
  runtime_params.listgen_reuse = true;
  runtime_params.listgen_dynamic_size = true;
  gfx::GfxRuntime runtime(runtime_params);
  for (bool independent_pool : {false, true}) {
    SCOPED_TRACE(independent_pool);
    gfx::SNodeTreeResources resources;
    resources.root_size = 4;
    resources.snode_ids = {10, 11};
    if (independent_pool) {
      spirv::SpirvAllocatorContract contract;
      contract.snode_id = 11;
      contract.pool_buffer_binding_id = 11;
      contract.pool_capacity = 1;
      contract.cell_stride_bytes = 4;
      contract.pool_data_offset = 4;
      resources.pointer_pools.push_back({contract, 8});
    }
    for (int root : {0, 1})
      runtime.materialize_snode_tree_resources(root, resources);

    DeviceCapabilityConfig caps;
    caps.set(DeviceCapability::spirv_version, 0x10300);
    spirv::IRBuilder builder(Arch::vulkan, &caps);
    builder.init_header();
    const auto buffer =
        builder.buffer_argument(builder.u32_type(), 0, 0, "counter");
    const auto mirror = independent_pool ? builder.buffer_argument(
                                               builder.u32_type(), 0, 1, "root")
                                         : spirv::Value{};
    const auto function = builder.new_function();
    builder.start_function(function);
    const auto ptr = builder.struct_array_access(
        builder.u32_type(), buffer,
        independent_pool ? builder.const_i32_one_ : builder.const_i32_zero_);
    const auto old = builder.load_variable(ptr, builder.u32_type());
    builder.store_variable(
        ptr, builder.make_value(
                 spv::OpIAdd, builder.u32_type(), old,
                 builder.uint_immediate_number(builder.u32_type(), 1)));
    if (independent_pool) {
      builder.store_variable(
          builder.struct_array_access(builder.u32_type(), mirror,
                                      builder.const_i32_zero_),
          builder.load_variable(ptr, builder.u32_type()));
    }
    builder.make_inst(spv::OpReturn);
    builder.make_inst(spv::OpFunctionEnd);
    builder.commit_kernel_function(function, "main", {}, {1, 1, 1});
    const auto code = builder.finalize();
    auto make_kernel = [&](int root, bool mutation = false) {
      gfx::GfxRuntime::RegisterParams params;
      params.num_snode_trees = 1;
      params.root_buffer_bindings = {root};
      params.snode_tree_ids = {root};
      spirv::TaskAttributes task;
      task.name = "scoped_list_counter";
      task.task_type =
          mutation ? OffloadedTaskType::serial : OffloadedTaskType::listgen;
      task.advisory_total_num_threads = 1;
      task.advisory_num_threads_per_group = 1;
      using BufferType = spirv::TaskAttributes::BufferType;
      task.buffer_binds.push_back(
          {{independent_pool ? BufferType::NodeAllocatorPool : BufferType::Root,
            independent_pool ? 11 : 0},
           0});
      if (independent_pool)
        task.buffer_binds.push_back({{BufferType::Root, 0}, 1});
      if (mutation) {
        task.may_mutate_sparse_topology = true;
        task.sparse_mutation_snode_id = 10;
      } else {
        task.sparse_list_op = spirv::TaskAttributes::kSparseListOpListgen;
        task.sparse_list_snode_id = 11;
        task.sparse_list_parent_snode_id = 10;
      }
      params.kernel_attribs.tasks_attribs.push_back(task);
      params.task_spirv_source_codes = {code};
      return runtime.register_taichi_kernel(std::move(params));
    };
    CallableBase callable;
    LaunchContextBuilder context(&callable);
    auto first = make_kernel(0), second = make_kernel(1),
         mutation = make_kernel(1, true);
    runtime.debug_reset_sparse_listgen_statistics();
    runtime.launch_kernel(first, context);
    runtime.launch_kernel(first, context);
    runtime.launch_kernel(second, context);
    runtime.launch_kernel(second, context);
    runtime.launch_kernel(first, context);
    // Mutating tree 1 cannot dirty tree 0's currently resident list.
    runtime.launch_kernel(mutation, context);
    runtime.launch_kernel(first, context);
    auto read_counter = [&](int root) {
      runtime.synchronize();
      auto ptr = runtime.get_root_buffer(root)->get_ptr();
      uint32_t value = 0;
      void *host = &value;
      std::size_t size = sizeof(value);
      EXPECT_EQ(device->readback_data(&ptr, &host, &size, 1),
                RhiResult::success);
      return value;
    };
    EXPECT_EQ(read_counter(0), 2);
    EXPECT_EQ(read_counter(1), 2);
    auto stats = runtime.debug_sparse_listgen_statistics({11}, 0);
    ASSERT_EQ(stats.nodes.size(), 1);
    EXPECT_EQ(stats.nodes[0].requests, 4);
    EXPECT_EQ(stats.nodes[0].reuse_hits, 2);
    stats = runtime.debug_sparse_listgen_statistics({11}, 1);
    ASSERT_EQ(stats.nodes.size(), 1);
    EXPECT_EQ(stats.nodes[0].requests, 2);
    runtime.retire_snode_tree_kernels(0);
    runtime.remove_root_buffer(0);
    EXPECT_TRUE(runtime.debug_sparse_listgen_statistics({11}, 0).nodes.empty());
    EXPECT_EQ(runtime.debug_sparse_listgen_statistics({11}, 1).nodes.size(), 1);
    runtime.materialize_snode_tree_resources(0, resources);
    auto replacement = make_kernel(0);
    runtime.launch_kernel(replacement, context);
    EXPECT_EQ(read_counter(0), 1);
    runtime.launch_kernel(second, context);
    EXPECT_EQ(read_counter(1), 3);
    runtime.release_aot_resources({replacement, second, mutation}, {0, 1});
  }
}

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
  EXPECT_NO_THROW(gfx::validate_snode_tree_resources(resources));
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
      EXPECT_NO_THROW(gfx::validate_snode_tree_resources(resources));
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
  EXPECT_NO_THROW(gfx::validate_snode_tree_resources(resources));
  table.hash_tombstone_count_offset = 136;
  resources = gfx::describe_snode_tree_resources(compiled);
  EXPECT_EQ(resources.hash_watches[0].tombstone_offset, 168);
  EXPECT_NO_THROW(gfx::validate_snode_tree_resources(resources));
}

TEST(GfxSNodeResources, RejectsMalformedPoolOwnershipAndRanges) {
  gfx::SNodeTreeResources valid;
  valid.root_size = 64;
  valid.snode_ids = {3, 4, 5};
  spirv::SpirvAllocatorContract c;
  c.snode_id = 4;
  c.pool_capacity = 4;
  c.cell_stride_bytes = 4;
  c.has_freelist = true;
  c.freelist_head_offset = 4;
  c.freelist_links_offset = 8;
  c.pool_data_offset = 24;
  c.has_ambient_zone = true;
  c.ambient_offset = 40;
  valid.pointer_pools.push_back({c, 0});
  valid.hash_watches.push_back({5, 48, 52, 56});
  EXPECT_NO_THROW(gfx::validate_snode_tree_resources(valid));
  auto rejects = [&](auto mutate) {
    auto invalid = valid;
    mutate(invalid);
    EXPECT_THROW(gfx::validate_snode_tree_resources(invalid),
                 std::invalid_argument);
  };
  rejects([](auto &r) { r.snode_ids = {3, 4, 4}; });
  rejects([](auto &r) { r.snode_ids = {4, 3, 5}; });
  rejects([](auto &r) { r.snode_ids = {3, 5}; });
  rejects([](auto &r) { r.pointer_pools.push_back(r.pointer_pools[0]); });
  rejects([](auto &r) { r.pointer_pools[0].independent_size = 64; });
  rejects(
      [](auto &r) { r.pointer_pools[0].contract.pool_buffer_binding_id = 5; });
  rejects([](auto &r) { r.pointer_pools[0].contract.cell_stride_bytes = 3; });
  rejects([](auto &r) { r.pointer_pools[0].contract.pool_capacity = 0; });
  rejects([](auto &r) { r.pointer_pools[0].contract.pool_capacity = 16; });
  rejects([](auto &r) { r.pointer_pools[0].contract.pool_data_offset = 16; });
  rejects([](auto &r) { r.pointer_pools[0].contract.ambient_offset = 48; });
  rejects([](auto &r) { r.pointer_pools[0].contract.watermark_offset = 1; });
  rejects([](auto &r) { r.hash_watches[0].snode_id = 4; });
  rejects([](auto &r) { r.hash_watches[0].overflow_offset = 64; });
  rejects([](auto &r) { r.hash_watches[0].active_count_offset = 48; });
  rejects([](auto &r) {
    r.pointer_pools[0].contract.pool_fraction =
        std::numeric_limits<double>::quiet_NaN();
  });
  rejects([](auto &r) {
    r.pointer_pools[0].contract.alloc_protocol =
        static_cast<spirv::SpirvAllocatorContract::AllocProtocol>(2);
  });
  rejects([](auto &r) {
    r.pointer_pools[0].contract.allocator_kind =
        static_cast<spirv::SpirvAllocatorContract::AllocatorKind>(2);
  });
  valid.pointer_pools[0].contract.pool_buffer_binding_id = 4;
  valid.pointer_pools[0].independent_size = 44;
  valid.root_size = 60;
  EXPECT_NO_THROW(gfx::validate_snode_tree_resources(valid));
  rejects([](auto &r) { r.pointer_pools[0].independent_size = 40; });
}

TEST(GfxSNodeResources, ChecksShaderBoundsWithoutAllocatingLargeBuffers) {
  gfx::SNodeTreeResources resources;
  resources.snode_ids = {1};
  resources.listgen_capacity_entries = std::numeric_limits<uint32_t>::max();
  resources.root_size = uint64_t(1) << 32;
  spirv::SpirvAllocatorContract c;
  c.snode_id = 1;
  c.pool_capacity = std::numeric_limits<uint32_t>::max() / 4;
  c.cell_stride_bytes = 4;
  c.pool_data_offset = 4;
  resources.pointer_pools.push_back({c, 0});
  EXPECT_NO_THROW(gfx::validate_snode_tree_resources(resources));
  ++resources.listgen_capacity_entries;
  EXPECT_THROW(gfx::validate_snode_tree_resources(resources),
               std::invalid_argument);
  --resources.listgen_capacity_entries;
  ++resources.pointer_pools[0].contract.pool_capacity;
  resources.root_size += 4;
  EXPECT_THROW(gfx::validate_snode_tree_resources(resources),
               std::invalid_argument);
  auto &chunk = resources.pointer_pools[0].contract;
  chunk.allocator_kind = spirv::SpirvAllocatorContract::AllocatorKind::Chunked;
  chunk.pool_capacity = 8;
  chunk.chunk_log2_capacity = 2;
  chunk.max_chunks = 2;
  EXPECT_NO_THROW(gfx::validate_snode_tree_resources(resources));
  chunk.max_chunks = 1;
  EXPECT_THROW(gfx::validate_snode_tree_resources(resources),
               std::invalid_argument);
  chunk.chunk_log2_capacity = 32;
  EXPECT_THROW(gfx::validate_snode_tree_resources(resources),
               std::invalid_argument);
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
