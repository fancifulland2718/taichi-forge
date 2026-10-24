#include "gtest/gtest.h"
#include "tests/cpp/aot/vulkan/timeline_gate.h"
#include "taichi/codegen/spirv/spirv_ir_builder.h"
#include "taichi/common/json_serde.h"
#include "taichi/ir/type_factory.h"
#include "taichi/program/launch_context_builder.h"
#include "taichi/program/ndarray.h"
#include "taichi/program/runtime_fault.h"
#include "taichi/rhi/vulkan/vulkan_device_creator.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"
#include "taichi/runtime/completion_wait_set.h"
#include "taichi/runtime/fixed_execution_vulkan.h"
#include "taichi/runtime/gfx/aot_graph_data.h"
#include "taichi/runtime/gfx/aot_module_loader_impl.h"

#include <array>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <vector>

namespace taichi::lang {
namespace {
using Domain = vulkan::ExecutionDomain;
using Queue = vulkan::VulkanExecutionQueue;
using Operation = FixedExecution::Operation;

struct ParallelOperands final : RuntimeCompletionResources {
  std::shared_ptr<DeviceAllocationGuard> readback;
  std::size_t retained_resource_count(std::uint32_t) const noexcept override {
    return readback ? 1 : 0;
  }
};

struct ParallelParts final : RuntimeClient {
  std::vector<std::shared_ptr<RuntimeClient>> parts;
  void retire() override {
    for (const auto &part : parts)
      part->retire();
    parts.clear();
  }
};

// Self-contained native shader: private GlobalTmps increments once per
// dispatch; root words contain the counter and both halves of one exact u64
// parameter. No Python-generated artifact, Program, or float conversion is
// involved.
gfx::GfxRuntime::RegisterParams counter_parameters(int root,
                                                   bool invalid = false) {
  DeviceCapabilityConfig caps;
  caps.set(DeviceCapability::spirv_version, 0x10300);
  spirv::IRBuilder ir(Arch::vulkan, &caps);
  ir.init_header();
  std::vector<std::tuple<spirv::SType, std::string, std::size_t>> members{
      {ir.u32_type(), "lo", 0}, {ir.u32_type(), "hi", 4}};
  const auto args =
      ir.uniform_struct_argument(ir.create_struct_type(members), 0, 0, "args");
  const auto temp = ir.buffer_argument(ir.u32_type(), 0, 1, "temporary");
  const auto output = ir.buffer_argument(ir.u32_type(), 0, 2, "output");
  const auto function = ir.new_function();
  ir.start_function(function);
  const auto pointer =
      ir.struct_array_access(ir.u32_type(), temp, ir.const_i32_zero_);
  const auto value = ir.add(ir.load_variable(pointer, ir.u32_type()),
                            ir.uint_immediate_number(ir.u32_type(), 1));
  ir.store_variable(pointer, value);
  ir.store_variable(
      ir.struct_array_access(ir.u32_type(), output, ir.const_i32_zero_), value);
  for (int word = 0; word < 2; ++word) {
    const auto uniform_pointer = ir.make_access_chain(
        ir.get_pointer_type(ir.u32_type(), spv::StorageClassUniform), args,
        {word});
    ir.store_variable(ir.struct_array_access(
                          ir.u32_type(), output,
                          ir.int_immediate_number(ir.i32_type(), word + 1)),
                      ir.load_variable(uniform_pointer, ir.u32_type()));
  }
  ir.make_inst(spv::OpReturn);
  ir.make_inst(spv::OpFunctionEnd);
  ir.commit_kernel_function(function, "main", {}, {1, 1, 1});

  gfx::GfxRuntime::RegisterParams result;
  result.num_snode_trees = 1;
  result.root_buffer_bindings = {root};
  result.snode_tree_ids = {root};
  spirv::TaskAttributes task;
  task.name = "parallel_private_counter_parameters";
  task.task_type = OffloadedTaskType::serial;
  task.advisory_total_num_threads = task.advisory_num_threads_per_group = 1;
  using Buffer = spirv::TaskAttributes::BufferType;
  task.buffer_binds = {{Buffer::Args, 0},
                       {invalid ? Buffer::HashOverflow : Buffer::GlobalTmps, 1},
                       {{Buffer::Root, 0}, 2}};
  result.kernel_attribs.tasks_attribs = {task};
  result.task_spirv_source_codes = {ir.finalize()};
  auto serialized = liong::json::serialize(result.kernel_attribs.ctx_attribs);
  std::vector<std::pair<std::vector<int>,
                        spirv::KernelContextAttributes::ArgAttributes>>
      attributes;
  for (int word = 0; word < 2; ++word) {
    spirv::KernelContextAttributes::ArgAttributes attribute;
    attribute.indices = {word};
    attribute.dtype = PrimitiveTypeID::u32;
    attribute.stride = 4;
    attribute.offset_in_mem = word * 4;
    attributes.push_back({{word}, attribute});
  }
  serialized["arg_attribs_vec_"] = liong::json::serialize(attributes);
  serialized["args_bytes_"] = liong::json::serialize(std::size_t{8});
  serialized["args_type_"] =
      liong::json::serialize(TypeFactory::get_instance().get_struct_type(
          {{PrimitiveType::u32, "lo", 0}, {PrimitiveType::u32, "hi", 4}},
          "1b"));
  liong::json::deserialize(serialized, result.kernel_attribs.ctx_attribs);
  return result;
}

class VulkanFixedParallel : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!vulkan::is_vulkan_api_available())
      GTEST_SKIP();
    creator = std::make_unique<vulkan::VulkanDeviceCreator>(
        vulkan::VulkanDeviceCreator::Params{});
    device = creator->device();
    if (!device->vk_caps().timeline_semaphore ||
        device->created_compute_queue_count() < 3)
      GTEST_SKIP() << "Requires two actual non-default compute queues";
    owner = std::make_shared<RuntimeFaultDomain>(Arch::vulkan, 83);
    device->set_backend_fault_reporter(owner);
    gfx::GfxRuntime::Params params;
    params.device = device;
    runtime = std::make_unique<gfx::GfxRuntime>(params);
    for (int i = 0; i < 3; ++i) {
      runtime->add_root_buffer(12);
      kernels[i] = std::make_unique<gfx::KernelImpl>(runtime.get(),
                                                     counter_parameters(i));
      contexts[i] = std::make_unique<LaunchContextBuilder>(kernels[i].get());
      contexts[i]->set_arg_uint({0}, 0);
      contexts[i]->set_arg_uint({1}, 0);
    }
    runtime->synchronize();
    registry =
        std::make_unique<vulkan::ExecutionDomainRegistry>(*device, owner);
    domain = registry->create({2, 2, true});
    operands = std::make_shared<ParallelOperands>();
    auto [buffer, status] = device->allocate_memory_unique(
        {36, false, true, false, AllocUsage::Storage});
    ASSERT_EQ(status, RhiResult::success);
    operands->readback = std::move(buffer);
  }
  void TearDown() override {
    if (registry)
      registry->retire();
  }
  FixedExecution::Dispatch dispatch(int index) {
    FixedExecution::Dispatch result{kernels[index].get(),
                                    contexts[index].get()};
    result.parameters.push_back({static_cast<std::size_t>(index), 0});
    return result;
  }
  std::shared_ptr<const FixedExecution::Parallel> parallel() {
    auto result = std::make_shared<FixedExecution::Parallel>();
    result->branches = {{dispatch(1), dispatch(1)}, {dispatch(2), dispatch(2)}};
    return result;
  }
  std::vector<Operation> operations() {
    std::vector<Operation> result{dispatch(0), parallel(), dispatch(0)};
    for (int i = 0; i < 3; ++i)
      result.push_back(FixedExecution::BufferCopy{
          operands->readback->get_ptr(i * 12),
          runtime->get_root_buffer(i)->get_ptr(), 12});
    return result;
  }
  FixedExecution::VulkanExecutionBinding binding() {
    FixedExecution::VulkanExecutionBinding result{domain, operands};
    result.parameter_count = 3;
    return result;
  }
  std::unique_ptr<FixedExecution> prepare(
      const std::vector<Operation> &ops,
      const FixedExecution::VulkanExecutionBinding &bound) {
    return std::make_unique<FixedExecution>(
        Arch::vulkan, *device, runtime.get(), ops, nullptr, owner, nullptr,
        nullptr, nullptr, nullptr, nullptr, &bound);
  }
  static std::vector<std::uint64_t> values(std::uint64_t seed) {
    return {(std::uint64_t{1} << 63) + seed,
            (std::uint64_t{1} << 53) + seed * 3,
            (std::uint64_t{1} << 61) + seed * 7};
  }
  void check(std::uint32_t counter,
             const std::vector<std::uint64_t> &expected) {
    check_buffer(*operands->readback, counter, expected);
  }
  void check_buffer(DeviceAllocation buffer,
                    std::uint32_t counter,
                    const std::vector<std::uint64_t> &expected) {
    void *mapped = nullptr;
    ASSERT_EQ(device->map(buffer, &mapped), RhiResult::success);
    std::array<std::uint32_t, 9> words;
    std::memcpy(words.data(), mapped, sizeof(words));
    device->unmap(buffer);
    for (int i = 0; i < 3; ++i) {
      EXPECT_EQ(words[3 * i], counter);
      EXPECT_EQ(words[3 * i + 1], static_cast<std::uint32_t>(expected[i]));
      EXPECT_EQ(words[3 * i + 2],
                static_cast<std::uint32_t>(expected[i] >> 32));
    }
  }
  bool ready(const RuntimeCompletion &completion) {
    CompletionWaitSet waits({completion}, std::chrono::milliseconds(1));
    return waits.wait(std::chrono::seconds(3)).status ==
           CompletionWaitSet::Status::ready;
  }
  std::unique_ptr<vulkan::VulkanDeviceCreator> creator;
  vulkan::VulkanDevice *device{nullptr};
  std::shared_ptr<RuntimeFaultDomain> owner;
  std::unique_ptr<gfx::GfxRuntime> runtime;
  std::array<std::unique_ptr<gfx::KernelImpl>, 3> kernels;
  std::array<std::unique_ptr<LaunchContextBuilder>, 3> contexts;
  std::unique_ptr<vulkan::ExecutionDomainRegistry> registry;
  std::shared_ptr<Domain> domain;
  std::shared_ptr<ParallelOperands> operands;
};

TEST_F(VulkanFixedParallel, PrivateWorkspacesParametersAndSampledJoin) {
  const auto before = device->queue_submission_snapshot().queue_submit_calls;
  auto call = prepare(operations(), binding());
  EXPECT_EQ(device->queue_submission_snapshot().queue_submit_calls, before);
  EXPECT_GT(call->workspace_bytes(), 0u);
  RecordProperty("workspace_bytes", std::to_string(call->workspace_bytes()));
  RecordProperty("created_compute_queues",
                 device->created_compute_queue_count());
  // Two parameter slots and four variants must still consume one auxiliary.
  std::vector<std::unique_ptr<Queue>> spare;
  for (std::uint32_t i = 3; i < device->created_compute_queue_count(); ++i)
    spare.push_back(std::make_unique<Queue>(*device, 1));
  EXPECT_THROW(Queue(*device, 1), std::runtime_error);
  spare.clear();
  RuntimeCompletion sampled;
  StreamGpuTimingSnapshot saved;
  {
    test_detail::TimelineGate ordinary(*device, 0);
    test_detail::TimelineGate gate(*device, domain->snapshot().queue_index);
    auto input = values(1);
    sampled = call->submit({}, input, true);
    input.assign(3, 0);  // accepted parameters own a snapshot
    EXPECT_FALSE(sampled.done());
    EXPECT_EQ(sampled.gpu_timing_snapshot().status, "pending");
    gate.release();
    const bool completed_without_default = ready(sampled);
    ordinary.release();  // rescue cannot make the previous bounded check pass
    ASSERT_TRUE(completed_without_default);
    sampled.wait();
    check(2, values(1));
    saved = sampled.gpu_timing_snapshot();
    EXPECT_TRUE(saved.available);
    {
      test_detail::TimelineGate pending(*device,
                                        domain->snapshot().queue_index);
      auto second = call->submit({}, values(2));
      auto third = call->submit({}, values(3));
      const auto count = device->queue_submission_snapshot().queue_submit_calls;
      EXPECT_THROW(call->submit({}, values(99)), Domain::Busy);
      EXPECT_EQ(device->queue_submission_snapshot().queue_submit_calls, count);
      EXPECT_EQ(domain->snapshot().pending, 2u);
      EXPECT_FALSE(second.done());
      EXPECT_FALSE(third.done());
      pending.release();
      third.wait();
      check(6, values(3));
    }
    for (std::uint32_t iteration = 4; iteration <= 8; ++iteration) {
      const auto expected = values(iteration);
      const auto native_before =
          device->queue_submission_snapshot().queue_submit_calls;
      call->submit({}, expected, iteration == 5).wait();
      EXPECT_EQ(device->queue_submission_snapshot().queue_submit_calls,
                native_before + 4);
      check(2 * iteration, expected);
    }
  }
  EXPECT_EQ(sampled.gpu_timing_snapshot().duration_ns, saved.duration_ns);
  domain->close();
  EXPECT_TRUE(sampled.done());
  EXPECT_EQ(sampled.gpu_timing_snapshot().duration_ns, saved.duration_ns);
  for (std::uint32_t i = 1; i < device->created_compute_queue_count(); ++i)
    spare.push_back(std::make_unique<Queue>(*device, 1));
  EXPECT_EQ(spare.size(), device->created_compute_queue_count() - 1);
}

TEST_F(VulkanFixedParallel, SharedPhaseWorkspaceAndInstanceRetirement) {
  auto bound = binding();
  auto workspace = runtime->create_fixed_graph_workspace();
  auto parts = std::make_shared<ParallelParts>();
  bound.workspace = workspace;
  bound.retirement_parts = &parts->parts;
  std::unique_ptr<FixedExecution> call;
  auto registration =
      domain->prepare_owned_instance(parts, [&](const auto &instance) {
        bound.instance = instance;
        call = prepare(operations(), bound);
      });
  auto initialize = domain->new_command_list();
  workspace->record_initialization(*initialize);  // common workspace only
  domain
      ->submit_instance(registration,
                        domain->prepare_commands({initialize.get()}, workspace))
      .wait();
  auto completed =
      call->submit({}, values(3), true);  // private branches initialize here
  completed.wait();
  check(2, values(3));
  call->submit({}, values(4)).wait();
  check(4, values(4));
  {
    test_detail::TimelineGate ordinary(*device, 0);
    domain->retire_instance(registration);
    EXPECT_TRUE(parts->parts.empty());
    EXPECT_TRUE(domain->instance_retired(registration));
    EXPECT_THROW(call->submit({}, values(5)), std::exception);
  }
  // Retiring the instance returns its auxiliary while its domain anchor lives.
  std::vector<std::unique_ptr<Queue>> spare;
  for (std::uint32_t i = 2; i < device->created_compute_queue_count(); ++i)
    spare.push_back(std::make_unique<Queue>(*device, 1));
  EXPECT_EQ(spare.size(), device->created_compute_queue_count() - 2);
}

TEST_F(VulkanFixedParallel, ColdRejectionLeavesNoSubmissionOrLeakedLease) {
  auto bound = binding();
  const auto initial = device->queue_submission_snapshot().queue_submit_calls;
  EXPECT_THROW(prepare({parallel(), parallel()}, bound), std::invalid_argument);
  auto nested = std::make_shared<FixedExecution::Parallel>();
  nested->branches = {{parallel()}, {dispatch(2)}};
  EXPECT_THROW(
      prepare({std::shared_ptr<const FixedExecution::Parallel>(nested)}, bound),
      std::invalid_argument);
  auto indirect = dispatch(1);
  indirect.indirect_dispatch = runtime->get_root_buffer(0)->get_ptr();
  EXPECT_THROW(prepare({indirect, parallel()}, bound), std::invalid_argument);
  auto adaptive = dispatch(1);
  adaptive.adaptive_block_dim = true;
  auto adaptive_region = std::make_shared<FixedExecution::Parallel>();
  adaptive_region->branches = {{adaptive}, {dispatch(2)}};
  EXPECT_THROW(
      prepare(
          {std::shared_ptr<const FixedExecution::Parallel>(adaptive_region)},
          bound),
      std::invalid_argument);
  EXPECT_THROW(
      FixedExecution(Arch::vulkan, *device, runtime.get(), {parallel()}),
      std::invalid_argument);
  // Failure after topology acquisition still unwinds the owned instance and
  // branch leases. An unsupported shared runtime buffer reaches detached
  // validation.
  gfx::KernelImpl invalid(runtime.get(), counter_parameters(1, true));
  LaunchContextBuilder context(&invalid);
  context.set_arg_uint({0}, 0);
  context.set_arg_uint({1}, 0);
  FixedExecution::Dispatch bad{&invalid, &context};
  auto rejected = std::make_shared<FixedExecution::Parallel>();
  rejected->branches = {{bad}, {dispatch(2)}};
  EXPECT_ANY_THROW(prepare(
      {std::shared_ptr<const FixedExecution::Parallel>(rejected)}, bound));
  EXPECT_EQ(device->queue_submission_snapshot().queue_submit_calls, initial);
  auto replacement = prepare(operations(), bound);
  replacement->submit({}, values(7)).wait();
  check(2, values(7));
  domain->close();
}

TEST_F(VulkanFixedParallel,
       IndependentInstancesKeepParametersAndRetirementSeparate) {
  if (device->created_compute_queue_count() < 5)
    GTEST_SKIP() << "Requires two fork/join topologies plus the default queue";
  auto peer_domain = registry->create({2, 2, true});
  auto peer_operands = std::make_shared<ParallelOperands>();
  auto [readback, status] = device->allocate_memory_unique(
      {36, false, true, false, AllocUsage::Storage});
  ASSERT_EQ(status, RhiResult::success);
  peer_operands->readback = std::move(readback);
  std::array<std::unique_ptr<gfx::KernelImpl>, 3> peer_kernels;
  std::array<std::unique_ptr<LaunchContextBuilder>, 3> peer_contexts;
  for (int i = 0; i < 3; ++i) {
    runtime->add_root_buffer(12);
    peer_kernels[i] = std::make_unique<gfx::KernelImpl>(
        runtime.get(), counter_parameters(3 + i));
    peer_contexts[i] =
        std::make_unique<LaunchContextBuilder>(peer_kernels[i].get());
    peer_contexts[i]->set_arg_uint({0}, 0);
    peer_contexts[i]->set_arg_uint({1}, 0);
  }
  runtime->synchronize();
  const auto peer_dispatch = [&](int i) {
    FixedExecution::Dispatch result{peer_kernels[i].get(),
                                    peer_contexts[i].get()};
    result.parameters.push_back({static_cast<std::size_t>(i), 0});
    return result;
  };
  auto peer_parallel = std::make_shared<FixedExecution::Parallel>();
  peer_parallel->branches = {{peer_dispatch(1), peer_dispatch(1)},
                             {peer_dispatch(2), peer_dispatch(2)}};
  std::vector<Operation> peer_operations{
      peer_dispatch(0),
      std::shared_ptr<const FixedExecution::Parallel>(peer_parallel),
      peer_dispatch(0)};
  for (int i = 0; i < 3; ++i)
    peer_operations.push_back(FixedExecution::BufferCopy{
        peer_operands->readback->get_ptr(i * 12),
        runtime->get_root_buffer(3 + i)->get_ptr(), 12});
  auto peer_binding = binding();
  peer_binding.domain = peer_domain;
  peer_binding.operands = peer_operands;
  auto left = prepare(operations(), binding());
  auto right = prepare(peer_operations, peer_binding);
  {
    test_detail::TimelineGate ordinary(*device, 0);
    test_detail::TimelineGate blocked(*device, domain->snapshot().queue_index);
    const auto pending = left->submit({}, values(11), true);
    const auto completed = right->submit({}, values(71), true);
    const bool independent = ready(completed);
    if (!independent) {
      // Cleanup may rescue the queues, but cannot turn this check into a pass.
      blocked.release();
      ordinary.release();
    }
    ASSERT_TRUE(independent);
    EXPECT_FALSE(pending.done());
    check_buffer(*peer_operands->readback, 2, values(71));
    right->submit({}, values(72)).wait();
    check_buffer(*peer_operands->readback, 4, values(72));
    blocked.release();
    pending.wait();
    check(2, values(11));
    domain->close();
    EXPECT_THROW(left->submit({}, values(12)), std::exception);
    right->submit({}, values(73)).wait();
    check_buffer(*peer_operands->readback, 6, values(73));
    peer_domain->close();
    EXPECT_TRUE(pending.done());
    EXPECT_TRUE(completed.done());
  }
}
TEST_F(VulkanFixedParallel, DynamicRangeNdarrayForwardAndAdjointReplay) {
  const auto *folder = std::getenv("TAICHI_VULKAN_PARALLEL_GRADIENT_AOT_DIR");
  if (!folder)
    GTEST_SKIP() << "Requires the saved ndarray_autodiff Vulkan fixture";
  gfx::AotModuleParams params;
  params.module_path = folder;
  params.runtime = runtime.get();
  auto module = aot::Module::load(Arch::vulkan, params);
  ASSERT_TRUE(module);
  auto *forward = module->get_kernel("forward_scalar");
  auto *backward = module->get_kernel("backward_scalar");
  ASSERT_TRUE(forward && backward);
  struct Arrays final : RuntimeCompletionResources {
    std::array<std::unique_ptr<DeviceAllocationGuard>, 8> buffers;
    std::size_t retained_resource_count(std::uint32_t) const noexcept override {
      std::size_t count = 0;
      for (const auto &buffer : buffers)
        count += bool(buffer);
      return count;
    }
  };
  auto arrays = std::make_shared<Arrays>();
  std::array<std::unique_ptr<Ndarray>, 8> views;
  std::array<std::unique_ptr<LaunchContextBuilder>, 4> contexts;
  const std::array<int, 2> lengths{5, 8};
  auto region = std::make_shared<FixedExecution::Parallel>();
  for (int branch = 0; branch < 2; ++branch) {
    const auto count = lengths[branch];
    for (int j = 0; j < 4; ++j) {
      auto [buffer, status] = device->allocate_memory_unique(
          {count * sizeof(float), true, true, false, AllocUsage::Storage});
      ASSERT_EQ(status, RhiResult::success);
      arrays->buffers[branch * 4 + j] = std::move(buffer);
      views[branch * 4 + j] = std::make_unique<Ndarray>(
          *arrays->buffers[branch * 4 + j], PrimitiveType::f32,
          std::vector<int>{count});
    }
    auto &primal = contexts[branch * 2];
    auto &adjoint = contexts[branch * 2 + 1];
    primal = std::make_unique<LaunchContextBuilder>(forward);
    adjoint = std::make_unique<LaunchContextBuilder>(backward);
    for (auto *context : {primal.get(), adjoint.get()}) {
      context->set_arg_ndarray_with_grad({0}, *views[branch * 4],
                                        *views[branch * 4 + 2]);
      context->set_arg_ndarray_with_grad({1}, *views[branch * 4 + 1],
                                        *views[branch * 4 + 3]);
    }
    region->branches.push_back(
        {FixedExecution::Dispatch{forward, primal.get()},
         FixedExecution::Dispatch{backward, adjoint.get()}});
  }
  auto bound = binding();
  bound.parameter_count = 0;
  bound.operands = arrays;
  auto call =
      prepare({std::shared_ptr<const FixedExecution::Parallel>(region)}, bound);
  runtime->synchronize();
  test_detail::TimelineGate ordinary(*device, 0);
  for (int replay = 0; replay < 3; ++replay) {
    for (int branch = 0; branch < 2; ++branch) {
      for (int j = 0; j < 4; ++j) {
        void *mapped = nullptr;
        ASSERT_EQ(device->map(*arrays->buffers[branch * 4 + j], &mapped),
                  RhiResult::success);
        for (int i = 0; i < lengths[branch]; ++i) {
          const float x = 0.5f * i + branch - replay;
          static_cast<float *>(mapped)[i] = j == 0   ? x
                                            : j == 1 ? -777.f
                                            : j == 2 ? 0.f
                                                     : 1.f + 0.25f * i;
        }
        device->unmap(*arrays->buffers[branch * 4 + j]);
      }
    }
    auto done = call->submit({}, {}, replay == 1);
    const bool independent = ready(done);
    if (!independent)
      ordinary.release();
    ASSERT_TRUE(independent);
    for (int branch = 0; branch < 2; ++branch) {
      for (int j : {1, 2}) {
        void *mapped = nullptr;
        ASSERT_EQ(device->map(*arrays->buffers[branch * 4 + j], &mapped),
                  RhiResult::success);
        for (int i = 0; i < lengths[branch]; ++i) {
          const float x = 0.5f * i + branch - replay;
          const float expected =
              j == 1 ? x * x + 2 * x : (2 * x + 2) * (1 + 0.25f * i);
          EXPECT_FLOAT_EQ(static_cast<float *>(mapped)[i], expected);
        }
        device->unmap(*arrays->buffers[branch * 4 + j]);
      }
    }
  }
  domain->close();
}
}  // namespace
}  // namespace taichi::lang
