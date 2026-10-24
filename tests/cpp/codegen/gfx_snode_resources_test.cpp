#include "gtest/gtest.h"

#ifdef TI_WITH_VULKAN
#include <limits>
#include <stdexcept>

#include "taichi/codegen/spirv/snode_struct_compiler.h"
#include "taichi/runtime/gfx/snode_tree_resources.h"
#include "taichi/runtime/gfx/aot_snode_resources.h"
#include "taichi/runtime/gfx/descriptor_binding.h"
#include "taichi/runtime/gfx/runtime.h"
#include "taichi/codegen/spirv/spirv_ir_builder.h"
#include "taichi/program/launch_context_builder.h"
#include "taichi/program/runtime_fault.h"
#include "taichi/rhi/vulkan/vulkan_device_creator.h"
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"
#include "taichi/runtime/fixed_execution_vulkan.h"
#include "taichi/runtime/gfx/aot_graph_data.h"
#include "taichi/runtime/completion_wait_set.h"
#include "tests/cpp/aot/vulkan/timeline_gate.h"

namespace taichi::lang {

TEST(GfxDescriptorBinding, AlignsBaseWithoutChangingLogicalAddress) {
  const Device::StorageBufferBindingLimits limits{16, 64};
  const auto slice =
      gfx::describe_descriptor_buffer_slice(limits, 212, 17, true);
  EXPECT_EQ(slice.offset, 208);
  EXPECT_EQ(slice.size, 21);
  EXPECT_EQ(slice.byte_bias, 4);
  EXPECT_EQ(slice.offset + slice.byte_bias, 212);
  EXPECT_EQ(slice.offset + slice.size, 229);
  EXPECT_EQ(
      gfx::describe_descriptor_buffer_slice(limits, 32, 64, false).byte_bias,
      0);
  EXPECT_EQ(
      gfx::describe_descriptor_buffer_slice(limits, UINT64_MAX, 0, false).size,
      0);
  EXPECT_THROW(gfx::describe_descriptor_buffer_slice(limits, 212, 17, false),
               std::invalid_argument);
  EXPECT_THROW(gfx::describe_descriptor_buffer_slice(limits, 212, 61, true),
               std::invalid_argument);
  EXPECT_THROW(
      gfx::describe_descriptor_buffer_slice(limits, UINT64_MAX, 1, true),
      std::invalid_argument);
  EXPECT_THROW(gfx::describe_descriptor_buffer_slice({0, 64}, 0, 1, true),
               std::invalid_argument);
  EXPECT_THROW(gfx::describe_descriptor_buffer_slice(
                   {uint64_t(1) << 33, UINT64_MAX}, uint64_t(1) << 32, 1, true),
               std::invalid_argument);
}

TEST(GfxSNodeResources, KernelBindingsRejectMissingRoots) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  vulkan::VulkanDeviceCreator::Params device_params;
  vulkan::VulkanDeviceCreator creator(device_params);
  gfx::GfxRuntime::Params runtime_params;
  runtime_params.device = creator.device();
  gfx::GfxRuntime runtime(runtime_params);
  gfx::SNodeTreeResources resources;
  resources.root_size = 16;
  runtime.materialize_snode_tree_resources(0, resources);

  gfx::GfxRuntime::RegisterParams params;
  params.num_snode_trees = 3;
  spirv::TaskAttributes task;
  task.buffer_binds.push_back({{spirv::TaskAttributes::BufferType::Root, 2}, 0});
  params.kernel_attribs.tasks_attribs.push_back(task);
  // These fail before pipeline construction: only root zero is live.
  EXPECT_ANY_THROW(runtime.register_taichi_kernel(params));
  params.root_buffer_bindings = {0, 0, 17};
  EXPECT_ANY_THROW(runtime.register_taichi_kernel(params));
  params.root_buffer_bindings = {0, 0};
  EXPECT_ANY_THROW(runtime.register_taichi_kernel(params));
  params.root_buffer_bindings = {0, 0, 0};
  params.kernel_attribs.tasks_attribs[0].buffer_binds[0].buffer.root_id = {3};
  EXPECT_ANY_THROW(runtime.register_taichi_kernel(params));
  runtime.remove_root_buffer(0);
  params.kernel_attribs.tasks_attribs[0].buffer_binds[0].buffer.root_id = {0};
  EXPECT_ANY_THROW(runtime.register_taichi_kernel(params));
}

TEST(GfxSNodeResources, ReconstructedModulesOwnDiagnosticIdentity) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  vulkan::VulkanDeviceCreator::Params device_params;
  vulkan::VulkanDeviceCreator creator(device_params);
  gfx::GfxRuntime::Params runtime_params;
  runtime_params.device = creator.device();
  gfx::GfxRuntime runtime(runtime_params);
  gfx::SNodeTreeResources resources;
  resources.root_size = 16;
  resources.snode_ids = {3, 4};
  EXPECT_ANY_THROW(
      runtime.allocate_aot_snode_trees({{17, &resources}, {17, &resources}}));
  const std::vector<gfx::SNodeTreeResourceSource> sources{{17, &resources},
                                                          {31, &resources}};
  auto first = runtime.allocate_aot_snode_trees(sources);
  auto second = runtime.allocate_aot_snode_trees(sources);
  EXPECT_EQ(first, (std::vector<int>{0, 1}));
  EXPECT_EQ(second, (std::vector<int>{2, 3}));
  runtime.release_aot_resources({}, first);
  auto replacement = runtime.allocate_aot_snode_trees(sources);
  EXPECT_EQ(replacement, first);

  DeviceCapabilityConfig caps;
  caps.set(DeviceCapability::spirv_version, 0x10300);
  spirv::IRBuilder builder(Arch::vulkan, &caps);
  builder.init_header();
  const auto buffer =
      builder.buffer_argument(builder.u32_type(), 0, 0, "diagnostic");
  const auto function = builder.new_function();
  builder.start_function(function);
  const uint32_t record[] = {
      1, 31, 4, 8, 0, uint32_t(spirv::SparseOverflowKind::Pointer)};
  for (uint32_t i = 0; i < 6; ++i) {
    const auto ptr = builder.struct_array_access(
        builder.u32_type(), buffer,
        builder.uint_immediate_number(builder.u32_type(), i));
    builder.store_variable(
        ptr, builder.uint_immediate_number(builder.u32_type(), record[i]));
  }
  builder.make_inst(spv::OpReturn);
  builder.make_inst(spv::OpFunctionEnd);
  builder.commit_kernel_function(function, "main", {}, {1, 1, 1});
  gfx::GfxRuntime::RegisterParams params;
  params.num_snode_trees = 2;
  params.root_buffer_bindings = second;
  params.snode_tree_ids = second;
  params.sparse_diagnostic_domain = second.front();
  spirv::TaskAttributes task;
  task.name = "module_diagnostic";
  task.task_type = OffloadedTaskType::serial;
  task.advisory_total_num_threads = 1;
  task.advisory_num_threads_per_group = 1;
  task.buffer_binds.push_back(
      {spirv::TaskAttributes::BufferType::HashOverflow, 0});
  params.kernel_attribs.tasks_attribs.push_back(task);
  params.task_spirv_source_codes = {builder.finalize()};
  auto kernel = runtime.register_taichi_kernel(std::move(params));
  CallableBase callable;
  LaunchContextBuilder context(&callable);
  runtime.launch_kernel(kernel, context);
  try {
    runtime.synchronize();
    FAIL() << "Expected module-owned sparse overflow";
  } catch (const std::string &message) {
    EXPECT_NE(message.find("root 3 SNode 4"), std::string::npos) << message;
    EXPECT_EQ(message.find("root 31 SNode 4"), std::string::npos) << message;
  }
  runtime.release_aot_resources({kernel}, second);
  runtime.release_aot_resources({}, replacement);
}

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

TEST(GfxSNodeResources, FixedListsArePrivateAndInvalidateOnlyAffectedTopology) {
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
  gfx::SNodeTreeResources resources;
  resources.root_size = 8;
  resources.snode_ids = {10, 11};
  resources.listgen_capacity_entries = 1;
  runtime.materialize_snode_tree_resources(0, resources);
  runtime.materialize_snode_tree_resources(1, resources);
  auto make_kernel = [&](int root, bool mutation, bool reader_only = false) {
    DeviceCapabilityConfig caps;
    caps.set(DeviceCapability::spirv_version, 0x10300);
    spirv::IRBuilder builder(Arch::vulkan, &caps);
    builder.init_header();
    const auto counter =
        builder.buffer_argument(builder.u32_type(), 0, 0, "counter");
    const auto list =
        mutation ? spirv::Value{}
                 : builder.buffer_argument(builder.u32_type(), 0, 1, "list");
    const auto function = builder.new_function();
    builder.start_function(function);
    const auto ptr = builder.struct_array_access(builder.u32_type(), counter,
                                                 builder.const_i32_zero_);
    builder.store_variable(
        ptr, builder.make_value(
                 spv::OpIAdd, builder.u32_type(),
                 builder.load_variable(ptr, builder.u32_type()),
                 builder.uint_immediate_number(builder.u32_type(), 1)));
    if (!mutation) {
      const auto count = builder.struct_array_access(builder.u32_type(), list,
                                                     builder.const_i32_zero_);
      builder.store_variable(
          builder.struct_array_access(builder.u32_type(), counter,
                                      builder.const_i32_one_),
          builder.load_variable(count, builder.u32_type()));
      if (!reader_only)
        builder.store_variable(
            count, builder.uint_immediate_number(builder.u32_type(), 17));
    }
    builder.make_inst(spv::OpReturn);
    builder.make_inst(spv::OpFunctionEnd);
    builder.commit_kernel_function(function, "main", {}, {1, 1, 1});
    gfx::GfxRuntime::RegisterParams params;
    params.num_snode_trees = 1;
    params.root_buffer_bindings = {root};
    params.snode_tree_ids = {root};
    spirv::TaskAttributes task;
    task.name = "fixed_list_version_counter";
    task.task_type = mutation || reader_only ? OffloadedTaskType::serial
                                             : OffloadedTaskType::listgen;
    task.advisory_total_num_threads = task.advisory_num_threads_per_group = 1;
    using BufferType = spirv::TaskAttributes::BufferType;
    task.buffer_binds.push_back({{BufferType::Root, 0}, 0});
    if (mutation) {
      task.may_mutate_sparse_topology = true;
      task.sparse_mutation_snode_id = 10;
    } else {
      task.buffer_binds.push_back({{BufferType::ListGen}, 1});
      if (!reader_only) {
        task.sparse_list_op = spirv::TaskAttributes::kSparseListOpListgen;
        task.sparse_list_snode_id = 11;
        task.sparse_list_parent_snode_id = 10;
      }
    }
    params.kernel_attribs.tasks_attribs.push_back(task);
    params.task_spirv_source_codes = {builder.finalize()};
    return runtime.register_taichi_kernel(std::move(params));
  };
  CallableBase callable;
  LaunchContextBuilder context(&callable);
  const auto list = make_kernel(0, false);
  const auto mutation = make_kernel(0, true);
  const auto other_mutation = make_kernel(1, true);
  const auto prepare = [&](auto handle, int root) {
    return runtime.prepare_fixed_graph({{{handle, &context}, {}}}, {}, {root});
  };
  auto fixed = prepare(list, 0);
  auto changed = prepare(mutation, 0);
  auto unrelated = prepare(other_mutation, 1);
  const auto check = [&](int root, uint32_t expected,
                         uint32_t input_count = 0) {
    runtime.synchronize();
    auto ptr = runtime.get_root_buffer(root)->get_ptr();
    uint32_t values[2] = {};
    void *host = values;
    std::size_t bytes = sizeof(values);
    ASSERT_EQ(device->readback_data(&ptr, &host, &bytes, 1),
              RhiResult::success);
    EXPECT_EQ(values[0], expected);
    EXPECT_EQ(values[1], input_count);
  };
  runtime.launch_kernel(list, context);
  runtime.launch_kernel(list, context);  // Reuse the ordinary list.
  unrelated->launch_prepared();
  runtime.launch_kernel(list,
                        context);  // An unrelated mutation cannot evict it.
  check(0, 1);
  check(1, 1);
  changed->launch_prepared();
  runtime.launch_kernel(list, context);
  check(0, 3);
  fixed->launch_prepared();
  runtime.launch_kernel(list, context);  // Private list writing preserves it.
  check(0, 4);
  EXPECT_EQ(fixed->snapshot_stats().known_persistent_control_bytes,
            2 * sizeof(uint32_t));
  EXPECT_EQ(fixed->snapshot_stats().known_persistent_argument_bytes,
            2 * sizeof(uint32_t));  // Legacy total includes the workspace.

  // Only a few KiB are allocated; the one-invocation kernels stay unchanged.
  resources.listgen_capacity_entries = 4096;
  runtime.materialize_snode_tree_resources(2, resources);
  runtime.launch_kernel(list, context);
  fixed->launch_prepared();  // Its own list is independent of runtime growth.
  runtime.launch_kernel(list,
                        context);  // Its write cannot evict the new buffer.
  check(0, 6);
  auto current = prepare(list, 0);
  current->launch_prepared();
  runtime.launch_kernel(list, context);
  check(0, 7);
  // Imported/partial traversals without a preceding listgen keep their
  // ordinary binding. Replacing it with a fresh private list would lose input.
  const auto reader = make_kernel(0, false, true);
  auto consuming = prepare(reader, 0);
  EXPECT_EQ(consuming->snapshot_stats().known_persistent_control_bytes, 0);
  consuming->launch_prepared();
  check(0, 8, 17);
  fixed.reset();
  changed->launch_prepared();
  runtime.release_aot_resources({list, mutation, reader}, {0});
  EXPECT_ANY_THROW(current->launch_prepared());
  EXPECT_ANY_THROW(consuming->launch_prepared());
  unrelated->launch_prepared();
  check(1, 2);
  runtime.release_aot_resources({other_mutation}, {1, 2});
}

TEST(GfxSNodeResources, SparseLoansPreflightWholeGraphsAndRetireGenerations) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  vulkan::VulkanDeviceCreator::Params device_params;
  vulkan::VulkanDeviceCreator creator(device_params);
  auto *device = creator.device();
  auto owner = std::make_shared<RuntimeFaultDomain>(Arch::vulkan, 551);
  device->set_backend_fault_reporter(owner);
  gfx::GfxRuntime::Params runtime_params;
  runtime_params.device = device;
  gfx::GfxRuntime runtime(runtime_params);
  gfx::SNodeTreeResources resources;
  resources.root_size = 4;
  resources.snode_ids = {10, 11};
  resources.listgen_capacity_entries = 1;
  spirv::SpirvAllocatorContract pool;
  pool.snode_id = pool.pool_buffer_binding_id = 11;
  pool.pool_capacity = 1;
  pool.cell_stride_bytes = pool.pool_data_offset = 4;
  resources.pointer_pools.push_back({pool, 8});
  runtime.materialize_snode_tree_resources(0, resources);
  runtime.materialize_snode_tree_resources(1, resources);

  DeviceCapabilityConfig caps;
  caps.set(DeviceCapability::spirv_version, 0x10300);
  spirv::IRBuilder builder(Arch::vulkan, &caps);
  builder.init_header();
  const auto counter =
      builder.buffer_argument(builder.u32_type(), 0, 0, "pool");
  const auto mirror = builder.buffer_argument(builder.u32_type(), 0, 1, "root");
  const auto function = builder.new_function();
  builder.start_function(function);
  const auto ptr = builder.struct_array_access(builder.u32_type(), counter,
                                               builder.const_i32_one_);
  const auto incremented =
      builder.make_value(spv::OpIAdd, builder.u32_type(),
                         builder.load_variable(ptr, builder.u32_type()),
                         builder.uint_immediate_number(builder.u32_type(), 1));
  builder.store_variable(ptr, incremented);
  builder.store_variable(builder.struct_array_access(builder.u32_type(), mirror,
                                                     builder.const_i32_zero_),
                         incremented);
  builder.make_inst(spv::OpReturn);
  builder.make_inst(spv::OpFunctionEnd);
  builder.commit_kernel_function(function, "main", {}, {1, 1, 1});
  const auto code = builder.finalize();
  auto make_kernel = [&](int root) {
    gfx::GfxRuntime::RegisterParams params;
    params.num_snode_trees = 1;
    params.root_buffer_bindings = {root};
    params.snode_tree_ids = {root};
    spirv::TaskAttributes task;
    task.name = "sparse_pool_payload_writer";
    task.task_type = OffloadedTaskType::serial;
    task.advisory_total_num_threads = task.advisory_num_threads_per_group = 1;
    task.buffer_binds = {
        {{spirv::TaskAttributes::BufferType::NodeAllocatorPool, 11}, 0},
        {{spirv::TaskAttributes::BufferType::Root, 0}, 1}};
    // No listgen and no topology mutation: ordinary payload writes still
    // conflict with a domain borrowing the same tree and allocator pool.
    params.kernel_attribs.tasks_attribs.push_back(task);
    params.task_spirv_source_codes = {code};
    return runtime.register_taichi_kernel(std::move(params));
  };
  CallableBase callable;
  LaunchContextBuilder context(&callable);
  const auto first = make_kernel(0), second = make_kernel(1);
  auto fixed = runtime.prepare_fixed_graph({{{first, &context}, {}}}, {}, {0});
  auto mixed = runtime.prepare_fixed_graph(
      {{{second, &context}, {}}, {{first, &context}, {}}}, {}, {0, 1});
  runtime.launch_kernel(first, context);
  auto workspace = runtime.create_fixed_graph_workspace();
  runtime.prepare_fixed_shared_sparse_resources(workspace, {first}, owner);
  const auto dependencies = workspace->shared_resource_dependencies();
  ASSERT_EQ(dependencies.size(), 1);
  dependencies.front().wait();
  const auto sequence = owner->next_submission_sequence();
  auto conflict = runtime.create_fixed_graph_workspace();
  EXPECT_ANY_THROW(runtime.prepare_fixed_shared_sparse_resources(
      conflict, {second, first}, owner));
  EXPECT_EQ(owner->next_submission_sequence(), sequence);
  EXPECT_TRUE(conflict->shared_resource_dependencies().empty());
  EXPECT_ANY_THROW(runtime.launch_kernel(first, context));
  EXPECT_ANY_THROW(fixed->launch_prepared());
  EXPECT_ANY_THROW(mixed->launch_prepared());
  auto replay = runtime.register_graph_replay(1);
  EXPECT_ANY_THROW(runtime.try_launch_graph(
      {{second, &context}, {first, &context}}, replay->replay_key(), nullptr));
  EXPECT_ANY_THROW(runtime.get_root_buffer(0));
  EXPECT_ANY_THROW(runtime.remove_root_buffer(0));
  EXPECT_ANY_THROW(runtime.retire_snode_tree_kernels(0));
  EXPECT_ANY_THROW(runtime.release_aot_resources({first, second}, {1, 0}));
  runtime.launch_kernel(second,
                        context);  // Failed multi-tree admission rolled back.
  auto other = runtime.create_fixed_graph_workspace();
  runtime.prepare_fixed_shared_sparse_resources(other, {second}, owner);
  auto other_dependencies = other->shared_resource_dependencies();
  ASSERT_EQ(other_dependencies.size(), 1);
  other_dependencies.front().wait();
  other->retire_shared_sparse_loans(other_dependencies.front());
  workspace->retire_shared_sparse_loans(dependencies.front());
  workspace->retire_shared_sparse_loans(dependencies.front());
  const auto check = [&](int root, uint32_t expected) {
    runtime.synchronize();
    auto address = runtime.get_root_buffer(root)->get_ptr();
    uint32_t value = 0;
    void *host = &value;
    size_t bytes = sizeof(value);
    ASSERT_EQ(device->readback_data(&address, &host, &bytes, 1),
              RhiResult::success);
    EXPECT_EQ(value, expected);
  };
  check(0, 1);
  check(1, 1);  // No earlier dispatch leaked from either rejected mixed graph.
  fixed->launch_prepared();
  check(0, 2);
  runtime.remove_root_buffer(0);
  runtime.materialize_snode_tree_resources(0, resources);
  const auto replacement = make_kernel(0);
  EXPECT_ANY_THROW(runtime.launch_kernel(first, context));
  EXPECT_ANY_THROW(fixed->launch_prepared());
  EXPECT_ANY_THROW(mixed->launch_prepared());
  runtime.launch_kernel(replacement, context);
  check(0, 1);
  runtime.release_aot_resources({first, second, replacement}, {0, 1});
}

namespace {
void check_detached_diagnostics(bool aggregate) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  vulkan::VulkanDeviceCreator creator({});
  auto *device = creator.device();
  if (!device->vk_caps().timeline_semaphore ||
      device->created_compute_queue_count() < 3)
    GTEST_SKIP() << "Requires two independent compute queues";
  auto owner = std::make_shared<RuntimeFaultDomain>(Arch::vulkan, 561);
  device->set_backend_fault_reporter(owner);
  gfx::GfxRuntime::Params config;
  config.device = device;
  gfx::GfxRuntime runtime(config);
  gfx::SNodeTreeResources layout;
  layout.root_size = 16;
  layout.snode_ids = {4};
  layout.listgen_capacity_entries = 1;
  layout.hash_watches = {{4, 4, 8, 12}};
  std::array<std::vector<int>, 2> roots{
      runtime.allocate_aot_snode_trees({{31, &layout}}),
      runtime.allocate_aot_snode_trees({{31, &layout}})};

  // Actual selected shader writes both tree payload and a domain-local error
  // record. The second execution reports overflow; the first remains clean.
  DeviceCapabilityConfig caps;
  caps.set(DeviceCapability::spirv_version, 0x10300);
  spirv::IRBuilder ir(Arch::vulkan, &caps);
  ir.init_header();
  auto root = ir.buffer_argument(ir.u32_type(), 0, 0, "root");
  std::optional<spirv::Value> diagnostic;
  if (aggregate)
    diagnostic = ir.buffer_argument(ir.u32_type(), 0, 1, "diagnostic");
  auto function = ir.new_function();
  ir.start_function(function);
  auto cell = ir.struct_array_access(ir.u32_type(), root, ir.const_i32_zero_);
  auto one = ir.uint_immediate_number(ir.u32_type(), 1);
  auto value = ir.make_value(spv::OpIAdd, ir.u32_type(),
                             ir.load_variable(cell, ir.u32_type()), one);
  ir.store_variable(cell, value);
  auto failed = ir.make_value(spv::OpUGreaterThan, ir.bool_type(), value, one);
  auto flag = ir.make_value(spv::OpSelect, ir.u32_type(), failed, one,
                            ir.uint_immediate_number(ir.u32_type(), 0));
  if (aggregate) {
    const uint32_t words[]{0, 31, 4,
                           0, 0,  uint32_t(spirv::SparseOverflowKind::Pointer)};
    for (int i = 0; i < 6; ++i) {
      auto data = i == 0   ? flag
                  : i == 3 ? value
                           : ir.uint_immediate_number(ir.u32_type(), words[i]);
      ir.store_variable(
          ir.struct_array_access(ir.u32_type(), *diagnostic,
                                 ir.int_immediate_number(ir.i32_type(), i)),
          data);
    }
  } else {
    // Top-level Hash and compact-pool overflow can exist only in the root
    // counters, with no HashOverflow descriptor in the compiled shader.
    ir.store_variable(
        ir.struct_array_access(ir.u32_type(), root, ir.const_i32_one_), flag);
    ir.store_variable(
        ir.struct_array_access(ir.u32_type(), root,
                               ir.int_immediate_number(ir.i32_type(), 2)),
        value);
  }
  ir.make_inst(spv::OpReturn);
  ir.make_inst(spv::OpFunctionEnd);
  ir.commit_kernel_function(function, "main", {}, {1, 1, 1});
  const auto code = ir.finalize();

  struct Parts final : RuntimeClient, RuntimeCompletionResources {
    std::vector<std::shared_ptr<RuntimeClient>> parts;
    std::shared_ptr<gfx::GfxRuntime::FixedGraphWorkspace> workspace;
    std::size_t retained_resource_count(std::uint32_t) const noexcept override {
      return workspace ? 1 : 0;
    }
    void retire() override {
      for (const auto &part : parts)
        part->retire();
      parts.clear();
    }
    void retire_after(const RuntimeCompletion &prefix) override {
      retire();
      workspace->retire_shared_sparse_loans(prefix);
      workspace.reset();
    }
  };
  vulkan::ExecutionDomainRegistry registry(*device, owner);
  std::array<std::shared_ptr<vulkan::ExecutionDomain>, 2> domains;
  std::array<std::shared_ptr<vulkan::ExecutionInstance>, 2> registrations;
  std::array<std::shared_ptr<Parts>, 2> clients;
  std::array<std::unique_ptr<gfx::KernelImpl>, 2> kernels;
  std::array<std::unique_ptr<LaunchContextBuilder>, 2> contexts;
  std::array<std::unique_ptr<FixedExecution>, 2> executions;
  for (int i = 0; i < 2; ++i) {
    gfx::GfxRuntime::RegisterParams params;
    params.num_snode_trees = 1;
    params.root_buffer_bindings = params.snode_tree_ids = roots[i];
    params.sparse_diagnostic_domain = roots[i].front();
    spirv::TaskAttributes task;
    task.name = "domain_sparse_diagnostic";
    task.task_type = OffloadedTaskType::serial;
    task.advisory_total_num_threads = task.advisory_num_threads_per_group = 1;
    task.may_mutate_sparse_topology = true;
    task.buffer_binds = {{{spirv::TaskAttributes::BufferType::Root, 0}, 0}};
    if (aggregate)
      task.buffer_binds.push_back(
          {spirv::TaskAttributes::BufferType::HashOverflow, 1});
    params.kernel_attribs.tasks_attribs = {task};
    params.task_spirv_source_codes = {code};
    kernels[i] = std::make_unique<gfx::KernelImpl>(&runtime, std::move(params));
    contexts[i] = std::make_unique<LaunchContextBuilder>(kernels[i].get());
    clients[i] = std::make_shared<Parts>();
    auto workspace = runtime.create_fixed_graph_workspace();
    clients[i]->workspace = workspace;
    runtime.prepare_fixed_shared_sparse_resources(
        workspace, {kernels[i]->handle()}, owner);
    domains[i] = registry.create({2, 1, true});
    FixedExecution::VulkanExecutionBinding binding{domains[i], clients[i]};
    binding.workspace = workspace;
    binding.retirement_parts = &clients[i]->parts;
    registrations[i] = domains[i]->prepare_owned_instance(
        clients[i], [&](const auto &instance) {
          binding.instance = instance;
          executions[i] = std::make_unique<FixedExecution>(
              Arch::vulkan, *device, &runtime,
              std::vector<FixedExecution::Operation>{FixedExecution::Dispatch{
                  kernels[i].get(), contexts[i].get()}},
              nullptr, owner, nullptr, nullptr, nullptr, nullptr, nullptr,
              &binding);
        });
    EXPECT_EQ(workspace->bytes(), aggregate ? 24u : 0u);
    EXPECT_EQ(executions[i]->workspace_bytes(), aggregate ? 96u : 48u);
    auto initialize = domains[i]->new_command_list();
    workspace->record_initialization(*initialize);
    domains[i]
        ->submit_instance(
            registrations[i],
            domains[i]->prepare_commands({initialize.get()}, workspace),
            workspace->shared_resource_dependencies())
        .wait();
  }
  const auto ready = [](const RuntimeCompletion &completion) {
    CompletionWaitSet waits({completion}, std::chrono::milliseconds(1));
    return waits.wait(std::chrono::seconds(3)).status ==
           CompletionWaitSet::Status::ready;
  };
  RuntimeCompletion left, right;
  {
    test_detail::TimelineGate blocked(*device,
                                      domains[0]->snapshot().queue_index);
    {
      test_detail::TimelineGate ordinary(*device, 0);
      left = executions[0]->submit();
      auto queued = executions[0]->submit();
      EXPECT_FALSE(queued.done());
      EXPECT_THROW(executions[0]->submit(), vulkan::ExecutionDomain::Busy);
      right = executions[1]->submit({}, {}, true);
      EXPECT_TRUE(ready(right));
      EXPECT_FALSE(left.done());
      const auto calls = device->queue_submission_snapshot().queue_submit_calls;
      EXPECT_TRUE(right.done());
      EXPECT_EQ(device->queue_submission_snapshot().queue_submit_calls, calls);
    }
    // No error on the first invocation. Synchronizing the ordinary stream
    // must not read the live hash counters in the still-pending borrowed tree.
    EXPECT_NO_THROW(runtime.synchronize());
    EXPECT_FALSE(left.done());
    auto overflow = executions[1]->submit({right});
    EXPECT_TRUE(ready(overflow));
    EXPECT_FALSE(left.done());
    auto prefix = overflow.reusable_stream_dependency(Arch::vulkan, 561, owner);
    EXPECT_NE(std::dynamic_pointer_cast<vulkan::VulkanTimelinePoint>(prefix),
              nullptr);
    for (int j = 0; j < 6; ++j)
      executions[1]->submit().wait();  // reuse both snapshot slots
    EXPECT_TRUE(right.done());  // old completion cannot reread a reused slot
    domains[1]->retire_instance(registrations[1]);
    EXPECT_TRUE(clients[1]->parts.empty());
    // The error remains after private snapshot buffers retire, preserving
    // the module-local source-to-runtime identity used by the selected code.
    try {
      runtime.synchronize();
      FAIL() << "Expected detached sparse overflow";
    } catch (const std::string &message) {
      EXPECT_NE(message.find("root 1 SNode 4"), std::string::npos) << message;
      EXPECT_NE(
          message.find(aggregate ? "capacity is 2 cells" : "active_count=2"),
          std::string::npos)
          << message;
    }
  }
  left.wait();
  // Unqueried completion must still observe its snapshot on domain close.
  executions[0]->submit();
  auto empty = domains[0]->new_command_list();
  auto commands = domains[0]->prepare_commands({empty.get()});
  auto observed_on_close = std::make_shared<int>(0);
  auto unobserved = domains[0]->submit_instance_selected(
      registrations[0], [commands](std::size_t) { return commands; }, {}, {},
      {}, {}, [observed_on_close](std::size_t) { ++*observed_on_close; });
  EXPECT_EQ(*observed_on_close, 0);
  domains[0]->close();
  EXPECT_EQ(*observed_on_close, 1);
  EXPECT_TRUE(unobserved.done());
  unobserved.wait();
  EXPECT_EQ(*observed_on_close, 1);
  EXPECT_TRUE(clients[0]->parts.empty());
  registry.retire();
  for (int i = 0; i < 2; ++i) {
    auto ptr = runtime.get_root_buffer(roots[i][0])->get_ptr();
    uint32_t value = 0;
    void *host = &value;
    size_t bytes = sizeof(value);
    ASSERT_EQ(device->readback_data(&ptr, &host, &bytes, 1),
              RhiResult::success);
    EXPECT_EQ(value, i == 0 ? 3u : 8u);
    runtime.release_aot_resources({kernels[i]->handle()}, roots[i]);
  }
}

}  // namespace

TEST(GfxSNodeResources,
     DetachedDiagnosticsCompleteBeforeSlotReuseAndRetirement) {
  check_detached_diagnostics(true);
}

TEST(GfxSNodeResources, DetachedRootCountersSurvivePendingPeersAndRetirement) {
  check_detached_diagnostics(false);
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

TEST(GfxSNodeResources, PersistedLayoutsRoundTripWithoutLiveSNodes) {
  for (bool independent : {false, true}) {
    for (const auto *kind : {"bump", "chunked"}) {
      gfx::AotSNodeTreeResources artifact;
      artifact.producer_tree_id = 17;
      artifact.storage_kind = gfx::AotSNodeTreeResources::StorageKind::Sparse;
      {
        SNode root{0, SNodeType::root};
        root.pointer({Axis{0}}, 8)
            .dense({Axis{0}}, 4)
            .insert_children(SNodeType::place)
            .dt = PrimitiveType::i32;
        spirv::PointerLayoutPolicy policy;
        policy.independent_pool = independent;
        policy.allocator_kind = kind;
        policy.max_chunks = 2;
        policy.pool_fraction = 0.5;
        artifact.resources = gfx::describe_snode_tree_resources(
            spirv::compile_snode_structs(root, policy));
      }
      const auto json = liong::json::print(liong::json::serialize(artifact));
      gfx::AotSNodeTreeResources decoded;
      liong::json::deserialize(liong::json::parse(json), decoded);
      EXPECT_EQ(liong::json::print(liong::json::serialize(decoded)), json);
      EXPECT_EQ(decoded.producer_tree_id, 17);
      EXPECT_EQ(decoded.resources.root_size, artifact.resources.root_size);
      ASSERT_EQ(decoded.resources.pointer_pools.size(), 1);
      const auto &pool = decoded.resources.pointer_pools.front();
      EXPECT_EQ(pool.contract.has_freelist, true);
      EXPECT_EQ(pool.contract.has_ambient_zone, true);
      EXPECT_EQ(pool.contract.pool_fraction, 0.5);
      EXPECT_EQ(pool.contract.pool_buffer_binding_id >= 0, independent);
      EXPECT_EQ(pool.independent_size,
                artifact.resources.pointer_pools.front().independent_size);
    }
  }
  gfx::AotSNodeTreeResources artifact;
  artifact.producer_tree_id = 0;
  artifact.resources.snode_ids = {0, 1};
  artifact.resources.root_size = 16;
  for (auto kind : {gfx::AotSNodeTreeResources::StorageKind::PlainDense,
                    gfx::AotSNodeTreeResources::StorageKind::PackedDense,
                    gfx::AotSNodeTreeResources::StorageKind::Sparse}) {
    artifact.storage_kind = kind;
    if (kind == gfx::AotSNodeTreeResources::StorageKind::Sparse)
      artifact.resources.hash_watches.push_back({1, 4, 8});
    gfx::AotSNodeTreeResources decoded;
    liong::json::deserialize(liong::json::serialize(artifact), decoded);
    EXPECT_EQ(decoded.storage_kind, kind);
    if (!decoded.resources.hash_watches.empty())
      EXPECT_EQ(decoded.resources.hash_watches.front().tombstone_offset,
                static_cast<std::size_t>(-1));
  }
}

TEST(GfxSNodeResources, PersistedResourcesRejectMissingAndTruncatedValues) {
  gfx::AotSNodeTreeResources artifact;
  artifact.producer_tree_id = 7;
  artifact.storage_kind = gfx::AotSNodeTreeResources::StorageKind::Sparse;
  artifact.resources.root_size = 24;
  artifact.resources.snode_ids = {0, 1, 2};
  spirv::SpirvAllocatorContract c;
  c.snode_id = 1;
  c.pool_capacity = 2;
  c.cell_stride_bytes = 4;
  c.pool_data_offset = 4;
  artifact.resources.pointer_pools.push_back({c, 0});
  artifact.resources.hash_watches.push_back({2, 12, 16});
  const auto original = liong::json::serialize(artifact);
  auto rejects = [&](auto mutate) {
    auto json = original;
    mutate(json);
    auto decoded = artifact;
    EXPECT_ANY_THROW(liong::json::deserialize(json, decoded, false));
    EXPECT_EQ(liong::json::print(liong::json::serialize(decoded)),
              liong::json::print(original));
  };
  using liong::json::JsonValue;
  rejects([](auto &j) { j.obj.inner.erase("storage_kind"); });
  rejects([](auto &j) { j.obj.inner.erase("pointer_pools"); });
  rejects([](auto &j) {
    j["pointer_pools"][size_t(0)]["contract"].obj.inner.erase("has_freelist");
  });
  rejects([](auto &j) {
    j["pointer_pools"][size_t(0)]["contract"]["pool_capacity"] =
        JsonValue((uint64_t(1) << 32) + 2);
  });
  rejects([](auto &j) {
    j["pointer_pools"][size_t(0)]["contract"]["alloc_protocol"] =
        JsonValue(257);
  });
  rejects([](auto &j) {
    j["pointer_pools"][size_t(0)]["contract"]["pool_capacity"] = JsonValue(2.5);
  });
  rejects([](auto &j) {
    j["pointer_pools"][size_t(0)]["contract"]["has_freelist"] = JsonValue(0);
  });
  rejects([](auto &j) { j["root_size"] = JsonValue(-1); });
  rejects(
      [](auto &j) { j["producer_tree_id"] = JsonValue(uint64_t(1) << 32); });
  rejects([](auto &j) { j["storage_kind"] = JsonValue(0); });
  rejects([](auto &j) { j["snode_ids"][size_t(1)] = JsonValue(-1); });
  rejects([](auto &j) {
    j["hash_watches"][size_t(0)]["tombstone_offset"] = JsonValue(-2);
  });
}

}  // namespace taichi::lang
#endif
