#include "gtest/gtest.h"
#include "taichi/codegen/spirv/spirv_ir_builder.h"
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/rhi/vulkan/vulkan_device_creator.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"
#include "tests/cpp/aot/gfx_utils.h"

#if defined(TI_WITH_CUDA)
#include "taichi/program/program.h"
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/rhi/cuda/cuda_driver.h"
#include "taichi/rhi/interop/vulkan_cuda_interop.h"
#endif

#include <array>
#include <atomic>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <vector>

using namespace taichi;
using namespace lang;

namespace {

std::unordered_set<uint32_t> spirv_capabilities(
    const std::vector<uint32_t> &module) {
  std::unordered_set<uint32_t> capabilities;
  for (std::size_t offset = 5; offset < module.size();) {
    const uint32_t instruction = module[offset];
    const uint32_t word_count = instruction >> 16;
    const uint32_t opcode = instruction & 0xffffu;
    if (word_count == 0 || offset + word_count > module.size()) {
      break;
    }
    if (opcode == spv::OpCapability && word_count == 2) {
      capabilities.insert(module[offset + 1]);
    }
    offset += word_count;
  }
  return capabilities;
}

bool spirv_contains_string(const std::vector<uint32_t> &module,
                           std::string_view expected) {
  const std::string_view bytes(
      reinterpret_cast<const char *>(module.data()),
      module.size() * sizeof(uint32_t));
  return bytes.find(expected) != std::string_view::npos;
}

std::vector<uint32_t> make_spirv_header(
    const DeviceCapabilityConfig &caps) {
  spirv::IRBuilder builder(Arch::vulkan, &caps);
  builder.init_header();
  return builder.finalize();
}

std::vector<uint32_t> make_indirect_packet_writer(
    const DeviceCapabilityConfig &caps) {
  spirv::IRBuilder builder(Arch::vulkan, &caps);
  builder.init_header();
  const auto u32 = builder.u32_type();
  const auto source = builder.buffer_argument(u32, 0, 0, "source");
  const auto control = builder.buffer_argument(u32, 0, 1, "control");
  const auto function = builder.new_function();
  builder.start_function(function);

  const auto zero = builder.uint_immediate_number(u32, 0);
  const auto one = builder.uint_immediate_number(u32, 1);
  const auto two = builder.uint_immediate_number(u32, 2);
  const auto group_count =
      builder.load_variable(builder.struct_array_access(u32, source, zero), u32);
  builder.store_variable(builder.struct_array_access(u32, control, zero),
                         group_count);
  builder.store_variable(builder.struct_array_access(u32, control, one), one);
  builder.store_variable(builder.struct_array_access(u32, control, two), one);
  builder.make_inst(spv::OpReturn);
  builder.make_inst(spv::OpFunctionEnd);

  std::vector<spirv::Value> interfaces;
  if (caps.get(DeviceCapability::spirv_version) > 0x10300) {
    interfaces = {source, control};
  }
  builder.commit_kernel_function(function, "main", std::move(interfaces),
                                 {1, 1, 1});
  return builder.finalize();
}

std::vector<uint32_t> make_indirect_target(
    const DeviceCapabilityConfig &caps) {
  spirv::IRBuilder builder(Arch::vulkan, &caps);
  builder.init_header();
  const auto u32 = builder.u32_type();
  const auto output = builder.buffer_argument(u32, 0, 0, "output");
  const auto function = builder.new_function();
  builder.start_function(function);

  const auto global_index = builder.get_global_invocation_id(0);
  const auto value =
      builder.add(global_index, builder.uint_immediate_number(u32, 1));
  builder.store_variable(
      builder.struct_array_access(u32, output, global_index), value);
  builder.make_inst(spv::OpReturn);
  builder.make_inst(spv::OpFunctionEnd);

  std::vector<spirv::Value> interfaces;
  if (caps.get(DeviceCapability::spirv_version) > 0x10300) {
    interfaces = {output};
  }
  builder.commit_kernel_function(function, "main", std::move(interfaces),
                                 {1, 1, 1});
  return builder.finalize();
}

#if defined(TI_WITH_CUDA)
class CudaInteropStream {
 public:
  CudaInteropStream() {
    auto context_guard = CUDAContext::get_instance().get_guard();
    CUDADriver::get_instance().stream_create(&stream_, CU_STREAM_NON_BLOCKING);
  }

  ~CudaInteropStream() {
    if (stream_ == nullptr) {
      return;
    }
    try {
      auto context_guard = CUDAContext::get_instance().get_guard();
      CUDADriver::get_instance().stream_synchronize(stream_);
      CUDADriver::get_instance().stream_destroy(stream_);
    } catch (...) {
    }
  }

  CUstream get() const noexcept {
    return stream_;
  }

 private:
  CUstream stream_{nullptr};
};

class CudaInteropAddKernel {
 public:
  CudaInteropAddKernel() {
    static constexpr char kPtx[] = R"ptx(
.version 6.4
.target sm_75
.address_size 64

.visible .entry add_constant(
    .param .u64 data,
    .param .u32 count,
    .param .u32 increment)
{
    .reg .pred %p<2>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<4>;

    ld.param.u64 %rd1, [data];
    ld.param.u32 %r1, [count];
    ld.param.u32 %r2, [increment];
    mov.u32 %r3, %tid.x;
    setp.ge.u32 %p1, %r3, %r1;
    @%p1 bra DONE;
    mul.wide.u32 %rd2, %r3, 4;
    add.s64 %rd3, %rd1, %rd2;
    ld.global.u32 %r4, [%rd3];
    add.u32 %r4, %r4, %r2;
    st.global.u32 [%rd3], %r4;
DONE:
    ret;
}
)ptx";
    auto context_guard = CUDAContext::get_instance().get_guard();
    auto &driver = CUDADriver::get_instance();
    driver.module_load_data_ex(&module_, kPtx, 0, nullptr, nullptr);
    driver.module_get_function(&function_, module_, "add_constant");
  }

  ~CudaInteropAddKernel() {
    if (module_ != nullptr) {
      auto context_guard = CUDAContext::get_instance().get_guard();
      CUDADriver::get_instance().module_unload(module_);
    }
  }

  void launch(void *data, uint32_t count, uint32_t increment, CUstream stream) {
    void *data_arg = data;
    void *args[] = {&data_arg, &count, &increment};
    CUDADriver::get_instance().launch_kernel(function_, 1, 1, 1, count, 1, 1, 0,
                                             stream, args, nullptr);
  }

 private:
  void *module_{nullptr};
  void *function_{nullptr};
};
#endif

}  // namespace

TEST(VulkanDeviceCapabilityTest, AtomicFloat2FeaturesMapIndependently) {
  VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT features{};
  features.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_2_FEATURES_EXT;
  features.shaderBufferFloat16AtomicAdd = VK_TRUE;
  features.shaderBufferFloat64AtomicMinMax = VK_TRUE;

  DeviceCapabilityConfig caps;
  vulkan::detail::record_shader_atomic_float2_capabilities(caps, features);

  EXPECT_EQ(caps.get(DeviceCapability::spirv_has_atomic_float16_add), 1u);
  EXPECT_EQ(caps.get(DeviceCapability::spirv_has_atomic_float64_minmax), 1u);
  EXPECT_EQ(caps.get(DeviceCapability::spirv_has_atomic_float_add), 0u);
  EXPECT_EQ(caps.get(DeviceCapability::spirv_has_atomic_float16), 0u);
  EXPECT_EQ(caps.get(DeviceCapability::spirv_has_atomic_float16_minmax), 0u);
  EXPECT_EQ(caps.get(DeviceCapability::spirv_has_atomic_float_minmax), 0u);
}

TEST(VulkanDeviceCapabilityTest, AtomicFloatHeaderMatchesScalarWidth) {
  DeviceCapabilityConfig caps;
  caps.set(DeviceCapability::spirv_version, 0x10300);
  caps.set(DeviceCapability::spirv_has_float16, true);
  caps.set(DeviceCapability::spirv_has_float64, true);
  caps.set(DeviceCapability::spirv_has_atomic_float16_add, true);
  caps.set(DeviceCapability::spirv_has_atomic_float64_minmax, true);

  const auto module = make_spirv_header(caps);
  const auto capabilities = spirv_capabilities(module);
  EXPECT_TRUE(capabilities.count(spv::CapabilityAtomicFloat16AddEXT));
  EXPECT_TRUE(capabilities.count(spv::CapabilityAtomicFloat64MinMaxEXT));
  EXPECT_FALSE(capabilities.count(spv::CapabilityAtomicFloat32AddEXT));
  EXPECT_TRUE(
      spirv_contains_string(module, "SPV_EXT_shader_atomic_float16_add"));
  EXPECT_TRUE(
      spirv_contains_string(module, "SPV_EXT_shader_atomic_float_min_max"));
  EXPECT_FALSE(
      spirv_contains_string(module, "SPV_EXT_shader_atomic_float_add"));

  DeviceCapabilityConfig float64_add_caps;
  float64_add_caps.set(DeviceCapability::spirv_version, 0x10300);
  float64_add_caps.set(DeviceCapability::spirv_has_float64, true);
  float64_add_caps.set(DeviceCapability::spirv_has_atomic_float64_add, true);
  const auto float64_add_module = make_spirv_header(float64_add_caps);
  EXPECT_TRUE(spirv_contains_string(float64_add_module,
                                   "SPV_EXT_shader_atomic_float_add"));
}

TEST(VulkanDeviceTest, ConcurrentQueueSubmissions) {
  if (!vulkan::is_vulkan_api_available()) {
    return;
  }

  vulkan::VulkanDeviceCreator::Params params;
  params.api_version = std::nullopt;
  auto creator = std::make_unique<vulkan::VulkanDeviceCreator>(params);
  auto *device = static_cast<vulkan::VulkanDevice *>(creator->device());

  // Exited callers must retire their per-thread streams instead of retaining
  // one command pool and submission tracker per historical host thread.
  constexpr int kThreadCount = 32;
  constexpr int kSubmitCount = 16;
  std::atomic<int> ready{0};
  std::atomic<bool> start{false};
  std::atomic<bool> succeeded{true};
  std::vector<std::thread> threads;
  threads.reserve(kThreadCount);

  for (int thread_index = 0; thread_index < kThreadCount; ++thread_index) {
    threads.emplace_back([&, thread_index] {
      ready.fetch_add(1, std::memory_order_release);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }

      Stream *stream = thread_index % 2 == 0 ? device->get_compute_stream()
                                             : device->get_graphics_stream();
      for (int submit_index = 0; submit_index < kSubmitCount; ++submit_index) {
        auto [cmdlist, result] = stream->new_command_list_unique();
        if (result != RhiResult::success || !cmdlist) {
          succeeded.store(false, std::memory_order_relaxed);
          return;
        }
        if (!stream->submit(cmdlist.get())) {
          succeeded.store(false, std::memory_order_relaxed);
          return;
        }
      }
      stream->command_sync();
    });
  }

  while (ready.load(std::memory_order_acquire) != kThreadCount) {
    std::this_thread::yield();
  }
  start.store(true, std::memory_order_release);
  for (auto &thread : threads) {
    thread.join();
  }

  EXPECT_TRUE(succeeded.load(std::memory_order_relaxed));
  device->wait_idle();
  const auto [compute_streams, graphics_streams] =
      device->debug_stream_cache_counts();
  EXPECT_EQ(compute_streams, 0u);
  EXPECT_EQ(graphics_streams, 0u);
}

TEST(VulkanDeviceTest, BoundsInFlightCommandBuffers) {
  if (!vulkan::is_vulkan_api_available()) {
    return;
  }

  vulkan::VulkanDeviceCreator::Params params;
  params.api_version = std::nullopt;
  auto creator = std::make_unique<vulkan::VulkanDeviceCreator>(params);
  auto *device = static_cast<vulkan::VulkanDevice *>(creator->device());
  auto *stream =
      static_cast<vulkan::VulkanStream *>(device->get_compute_stream());

  constexpr int kSubmitCount = 256;
  for (int submit_index = 0; submit_index < kSubmitCount; ++submit_index) {
    auto [cmdlist, result] = stream->new_command_list_unique();
    ASSERT_EQ(result, RhiResult::success);
    ASSERT_NE(cmdlist, nullptr);
    ASSERT_TRUE(stream->submit(cmdlist.get()));
    EXPECT_LE(stream->debug_in_flight_command_buffer_count(), 64u);
  }

  stream->command_sync();
  EXPECT_EQ(stream->debug_in_flight_command_buffer_count(), 0u);
}

TEST(VulkanDeviceTest, SubmissionBatchPublishesOrderedCommandBuffersOnce) {
  if (!vulkan::is_vulkan_api_available()) {
    return;
  }

  vulkan::VulkanDeviceCreator::Params params;
  params.api_version = std::nullopt;
  auto creator = std::make_unique<vulkan::VulkanDeviceCreator>(params);
  auto *device = static_cast<vulkan::VulkanDevice *>(creator->device());
  auto *stream =
      static_cast<vulkan::VulkanStream *>(device->get_compute_stream());

  constexpr int kCommandCount = 96;
  std::vector<StreamSemaphore> completions;
  completions.reserve(kCommandCount);
  const auto before = device->queue_submission_snapshot();
  stream->begin_submission_batch();
  for (int command_index = 0; command_index < kCommandCount;
       ++command_index) {
    auto [cmdlist, result] = stream->new_command_list_unique();
    ASSERT_EQ(result, RhiResult::success);
    ASSERT_NE(cmdlist, nullptr);
    auto completion = stream->submit(cmdlist.get());
    ASSERT_TRUE(completion);
    EXPECT_FALSE(completion->is_ready());
    completions.push_back(std::move(completion));
  }
  auto batch_completion = stream->end_submission_batch();
  ASSERT_TRUE(batch_completion);
  ASSERT_TRUE(batch_completion->wait());
  for (const auto &completion : completions) {
    EXPECT_TRUE(completion->is_ready());
  }
  const auto after = device->queue_submission_snapshot();
  EXPECT_EQ(after.queue_submit_calls - before.queue_submit_calls, 1u);
  EXPECT_EQ(after.submitted_command_buffers -
                before.submitted_command_buffers,
            static_cast<std::uint64_t>(kCommandCount));
  EXPECT_EQ(after.batched_queue_submit_calls -
                before.batched_queue_submit_calls,
            1u);
  EXPECT_EQ(after.batched_command_buffers -
                before.batched_command_buffers,
            static_cast<std::uint64_t>(kCommandCount));
  EXPECT_EQ(stream->debug_in_flight_command_buffer_count(), 0u);
}

TEST(VulkanDeviceTest, IndirectDispatchRejectsInvalidAllocations) {
  if (!vulkan::is_vulkan_api_available()) {
    GTEST_SKIP();
  }

  vulkan::VulkanDeviceCreator::Params params;
  params.api_version = std::nullopt;
  auto creator = std::make_unique<vulkan::VulkanDeviceCreator>(params);
  auto *device = static_cast<vulkan::VulkanDevice *>(creator->device());
  auto *stream = device->get_compute_stream();
  auto [cmdlist, command_result] = stream->new_command_list_unique();
  ASSERT_EQ(command_result, RhiResult::success);
  ASSERT_NE(cmdlist, nullptr);

  Device::AllocParams storage_params;
  storage_params.size = 3 * sizeof(uint32_t);
  storage_params.usage = AllocUsage::Storage;
  auto [storage, storage_result] =
      device->allocate_memory_unique(storage_params);
  ASSERT_EQ(storage_result, RhiResult::success);
  ASSERT_NE(storage, nullptr);

  EXPECT_EQ(cmdlist->dispatch_indirect(storage->get_ptr()),
            RhiResult::invalid_usage);
  EXPECT_EQ(cmdlist->dispatch_indirect(kDeviceNullPtr),
            RhiResult::invalid_usage);

  Device::AllocParams indirect_params;
  indirect_params.size = 4 * sizeof(uint32_t);
  indirect_params.usage = AllocUsage::Storage | AllocUsage::Indirect;
  auto [indirect, indirect_result] =
      device->allocate_memory_unique(indirect_params);
  ASSERT_EQ(indirect_result, RhiResult::success);
  ASSERT_NE(indirect, nullptr);

  EXPECT_EQ(cmdlist->dispatch_indirect(indirect->get_ptr(2)),
            RhiResult::invalid_usage);
  EXPECT_EQ(cmdlist->dispatch_indirect(indirect->get_ptr(8)),
            RhiResult::invalid_usage);

  vulkan::VulkanDeviceCreator::Params other_params;
  other_params.api_version = std::nullopt;
  auto other_creator =
      std::make_unique<vulkan::VulkanDeviceCreator>(other_params);
  auto *other_device =
      static_cast<vulkan::VulkanDevice *>(other_creator->device());
  auto [foreign, foreign_result] =
      other_device->allocate_memory_unique(indirect_params);
  ASSERT_EQ(foreign_result, RhiResult::success);
  ASSERT_NE(foreign, nullptr);
  EXPECT_EQ(cmdlist->dispatch_indirect(foreign->get_ptr()),
            RhiResult::invalid_usage);
}

TEST(VulkanDeviceTest, ConditionalCommandsValidateUsageAndNesting) {
  if (!vulkan::is_vulkan_api_available()) {
    GTEST_SKIP();
  }

  vulkan::VulkanDeviceCreator::Params params;
  params.api_version = std::nullopt;
  auto creator = std::make_unique<vulkan::VulkanDeviceCreator>(params);
  auto *device = static_cast<vulkan::VulkanDevice *>(creator->device());
  if (!device->supports_conditional_commands()) {
    GTEST_SKIP();
  }
  auto *stream = device->get_compute_stream();
  auto [cmdlist, command_result] = stream->new_command_list_unique();
  ASSERT_EQ(command_result, RhiResult::success);
  ASSERT_NE(cmdlist, nullptr);

  Device::AllocParams storage_params;
  storage_params.size = sizeof(uint32_t);
  storage_params.usage = AllocUsage::Storage;
  auto [storage, storage_result] =
      device->allocate_memory_unique(storage_params);
  ASSERT_EQ(storage_result, RhiResult::success);
  ASSERT_NE(storage, nullptr);
  EXPECT_EQ(cmdlist->begin_conditional(storage->get_ptr()),
            RhiResult::invalid_usage);
  EXPECT_EQ(cmdlist->begin_conditional(kDeviceNullPtr),
            RhiResult::invalid_usage);

  Device::AllocParams predicate_params;
  predicate_params.size = 2 * sizeof(uint32_t);
  predicate_params.usage =
      AllocUsage::Storage | AllocUsage::Conditional;
  auto [predicate, predicate_result] =
      device->allocate_memory_unique(predicate_params);
  ASSERT_EQ(predicate_result, RhiResult::success);
  ASSERT_NE(predicate, nullptr);
  EXPECT_EQ(cmdlist->begin_conditional(predicate->get_ptr(2)),
            RhiResult::invalid_usage);
  EXPECT_EQ(cmdlist->begin_conditional(
                predicate->get_ptr(device->get_vkbuffer_size(*predicate))),
            RhiResult::invalid_usage);

  EXPECT_EQ(cmdlist->begin_conditional(predicate->get_ptr()),
            RhiResult::success);
  EXPECT_EQ(cmdlist->begin_conditional(predicate->get_ptr()),
            RhiResult::invalid_usage);
  EXPECT_EQ(cmdlist->end_conditional(), RhiResult::success);
  EXPECT_EQ(cmdlist->end_conditional(), RhiResult::invalid_usage);
}

TEST(VulkanDeviceTest, DeviceWrittenIndirectDispatchReplaysWithoutStalePacket) {
  if (!vulkan::is_vulkan_api_available()) {
    GTEST_SKIP();
  }

  vulkan::VulkanDeviceCreator::Params params;
  params.api_version = std::nullopt;
  auto creator = std::make_unique<vulkan::VulkanDeviceCreator>(params);
  auto *device = static_cast<vulkan::VulkanDevice *>(creator->device());
  auto *stream = device->get_compute_stream();

  const auto writer_spirv = make_indirect_packet_writer(device->get_caps());
  const auto target_spirv = make_indirect_target(device->get_caps());
  const PipelineSourceDesc writer_source{
      PipelineSourceType::spirv_binary, writer_spirv.data(),
      writer_spirv.size() * sizeof(uint32_t), PipelineStageType::compute};
  const PipelineSourceDesc target_source{
      PipelineSourceType::spirv_binary, target_spirv.data(),
      target_spirv.size() * sizeof(uint32_t), PipelineStageType::compute};
  auto [writer_pipeline, writer_result] =
      device->create_pipeline_unique(writer_source, "indirect_packet_writer");
  ASSERT_EQ(writer_result, RhiResult::success);
  ASSERT_NE(writer_pipeline, nullptr);
  auto [target_pipeline, target_result] =
      device->create_pipeline_unique(target_source, "indirect_target");
  ASSERT_EQ(target_result, RhiResult::success);
  ASSERT_NE(target_pipeline, nullptr);

  Device::AllocParams source_params;
  source_params.size = sizeof(uint32_t);
  source_params.usage = AllocUsage::Storage;
  auto [source, source_result] =
      device->allocate_memory_unique(source_params);
  ASSERT_EQ(source_result, RhiResult::success);
  ASSERT_NE(source, nullptr);

  Device::AllocParams control_params;
  control_params.size = 3 * sizeof(uint32_t);
  control_params.usage = AllocUsage::Storage | AllocUsage::Indirect;
  auto [control, control_result] =
      device->allocate_memory_unique(control_params);
  ASSERT_EQ(control_result, RhiResult::success);
  ASSERT_NE(control, nullptr);

  constexpr uint32_t kOutputCount = 4;
  Device::AllocParams output_params;
  output_params.size = kOutputCount * sizeof(uint32_t);
  output_params.usage = AllocUsage::Storage;
  auto [output, output_result] =
      device->allocate_memory_unique(output_params);
  ASSERT_EQ(output_result, RhiResult::success);
  ASSERT_NE(output, nullptr);

  auto writer_resources = device->create_resource_set_unique();
  writer_resources->rw_buffer(0, *source);
  writer_resources->rw_buffer(1, *control);
  auto target_resources = device->create_resource_set_unique();
  target_resources->rw_buffer(0, *output);

  auto [cmdlist, command_result] = stream->new_command_list_unique();
  ASSERT_EQ(command_result, RhiResult::success);
  ASSERT_NE(cmdlist, nullptr);

  cmdlist->buffer_fill(output->get_ptr(), output_params.size, 0);
  cmdlist->buffer_transition(
      output->get_ptr(), output_params.size,
      {BufferBarrierStage::Transfer, BufferBarrierAccess::TransferWrite,
       BufferBarrierStage::Compute, BufferBarrierAccess::ShaderWrite});
  cmdlist->buffer_transition(
      source->get_ptr(), source_params.size,
      {BufferBarrierStage::Transfer, BufferBarrierAccess::TransferWrite,
       BufferBarrierStage::Compute, BufferBarrierAccess::ShaderRead});
  cmdlist->bind_pipeline(writer_pipeline.get());
  ASSERT_EQ(cmdlist->bind_shader_resources(writer_resources.get()),
            RhiResult::success);
  ASSERT_EQ(cmdlist->dispatch(1), RhiResult::success);
  cmdlist->buffer_transition(
      control->get_ptr(), control_params.size,
      {BufferBarrierStage::Compute, BufferBarrierAccess::ShaderWrite,
       BufferBarrierStage::IndirectCommand,
       BufferBarrierAccess::IndirectCommandRead});
  cmdlist->bind_pipeline(target_pipeline.get());
  ASSERT_EQ(cmdlist->bind_shader_resources(target_resources.get()),
            RhiResult::success);
  ASSERT_EQ(cmdlist->dispatch_indirect(control->get_ptr()),
            RhiResult::success);
  cmdlist->buffer_transition(
      output->get_ptr(), output_params.size,
      {BufferBarrierStage::Compute, BufferBarrierAccess::ShaderWrite,
       BufferBarrierStage::Transfer, BufferBarrierAccess::TransferRead});

  const auto replay = [&](uint32_t group_count) {
    std::array<uint32_t, 4> values{};
    DevicePtr source_ptr = source->get_ptr();
    const void *source_data = &group_count;
    size_t source_size = sizeof(group_count);
    const auto upload_result =
        device->upload_data(&source_ptr, &source_data, &source_size);
    EXPECT_EQ(upload_result, RhiResult::success);
    if (upload_result != RhiResult::success) {
      return values;
    }
    auto completion = stream->submit_synced(cmdlist.get());
    EXPECT_NE(completion, nullptr);
    if (!completion) {
      return values;
    }

    DevicePtr output_ptr = output->get_ptr();
    void *output_data = values.data();
    size_t output_size = sizeof(values);
    EXPECT_EQ(device->readback_data(&output_ptr, &output_data, &output_size),
              RhiResult::success);
    return values;
  };

  EXPECT_EQ(replay(0), (std::array<uint32_t, kOutputCount>{0, 0, 0, 0}));
  EXPECT_EQ(replay(kOutputCount),
            (std::array<uint32_t, kOutputCount>{1, 2, 3, 4}));
}

TEST(VulkanDeviceTest, ConcurrentDescriptorAndRenderPassCreation) {
  if (!vulkan::is_vulkan_api_available()) {
    return;
  }

  vulkan::VulkanDeviceCreator::Params params;
  params.api_version = std::nullopt;
  auto creator = std::make_unique<vulkan::VulkanDeviceCreator>(params);
  auto *device = static_cast<vulkan::VulkanDevice *>(creator->device());

  Device::AllocParams alloc_params;
  alloc_params.size = sizeof(uint32_t);
  alloc_params.usage = AllocUsage::Storage;
  DeviceAllocation allocation;
  ASSERT_EQ(device->allocate_memory(alloc_params, &allocation),
            RhiResult::success);
  DeviceAllocationGuard allocation_guard(allocation);

  constexpr int kThreadCount = 4;
  constexpr int kDescriptorSetsPerThread = 32;
  device->set_descriptor_set_cache_enabled(false);

  vulkan::VulkanRenderPassDesc renderpass_desc;
  renderpass_desc.color_attachments = {{VK_FORMAT_R8G8B8A8_UNORM, true}};
  std::array<vkapi::IVkSampler, kThreadCount> samplers;
  std::array<vkapi::IVkRenderPass, kThreadCount> renderpasses;
  std::atomic<int> ready{0};
  std::atomic<bool> start{false};
  std::atomic<bool> succeeded{true};
  std::vector<std::thread> threads;
  threads.reserve(kThreadCount);

  for (int thread_index = 0; thread_index < kThreadCount; ++thread_index) {
    threads.emplace_back([&, thread_index] {
      ready.fetch_add(1, std::memory_order_release);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (int set_index = 0; set_index < kDescriptorSetsPerThread;
           ++set_index) {
        vulkan::VulkanResourceSet set(device);
        set.rw_buffer(0, allocation);
        auto [status, descriptor_set] = set.finalize();
        if (status != RhiResult::success || !descriptor_set) {
          succeeded.store(false, std::memory_order_relaxed);
          return;
        }
      }
      samplers[thread_index] = device->get_default_sampler();
      renderpasses[thread_index] = device->get_renderpass(renderpass_desc);
      if (!samplers[thread_index] || !renderpasses[thread_index]) {
        succeeded.store(false, std::memory_order_relaxed);
      }
    });
  }

  while (ready.load(std::memory_order_acquire) != kThreadCount) {
    std::this_thread::yield();
  }
  start.store(true, std::memory_order_release);
  for (auto &thread : threads) {
    thread.join();
  }

  EXPECT_TRUE(succeeded.load(std::memory_order_relaxed));
  for (int i = 1; i < kThreadCount; ++i) {
    EXPECT_EQ(samplers[i], samplers[0]);
    EXPECT_EQ(renderpasses[i], renderpasses[0]);
  }
  device->wait_idle();
}

TEST(VulkanDescriptorSetTest, DirtyResourceSetDoesNotMutateRecordedSet) {
  if (!vulkan::is_vulkan_api_available()) {
    return;
  }

  vulkan::VulkanDeviceCreator::Params params;
  params.api_version = std::nullopt;
  auto creator = std::make_unique<vulkan::VulkanDeviceCreator>(params);
  auto *device = static_cast<vulkan::VulkanDevice *>(creator->device());

  Device::AllocParams alloc_params;
  alloc_params.size = sizeof(uint32_t);
  alloc_params.usage = AllocUsage::Storage;
  DeviceAllocation first_allocation;
  DeviceAllocation second_allocation;
  ASSERT_EQ(device->allocate_memory(alloc_params, &first_allocation),
            RhiResult::success);
  ASSERT_EQ(device->allocate_memory(alloc_params, &second_allocation),
            RhiResult::success);
  DeviceAllocationGuard first_guard(first_allocation);
  DeviceAllocationGuard second_guard(second_allocation);

  vulkan::VulkanResourceSet resource_set(device);
  resource_set.rw_buffer(0, first_allocation);
  auto [first_result, first_set] = resource_set.finalize();
  ASSERT_EQ(first_result, RhiResult::success);
  ASSERT_NE(first_set, nullptr);

  // This is the same owner pin added when a command buffer binds the set. It
  // models a recorded buffer whose reference has not yet been retired.
  {
    std::lock_guard<std::mutex> lock(first_set->mutex);
    ++first_set->recording_use_count;
  }
  resource_set.rw_buffer(0, second_allocation);
  auto [second_result, second_set] = resource_set.finalize();
  ASSERT_EQ(second_result, RhiResult::success);
  ASSERT_NE(second_set, nullptr);
  EXPECT_NE(second_set, first_set);
  {
    std::lock_guard<std::mutex> lock(first_set->mutex);
    ASSERT_EQ(first_set->recording_use_count, 1u);
    --first_set->recording_use_count;
  }

  // An unrecorded set can still be updated in place, avoiding an allocation on
  // every resource-set mutation after the command buffer has released its pin.
  resource_set.rw_buffer(0, first_allocation);
  auto [third_result, third_set] = resource_set.finalize();
  ASSERT_EQ(third_result, RhiResult::success);
  EXPECT_EQ(third_set, second_set);
}

TEST(VulkanDescriptorSetTest, CachedSetIsReplacedWhenAnotherRecordingUsesIt) {
  if (!vulkan::is_vulkan_api_available()) {
    return;
  }

  vulkan::VulkanDeviceCreator::Params params;
  params.api_version = std::nullopt;
  auto creator = std::make_unique<vulkan::VulkanDeviceCreator>(params);
  auto *device = static_cast<vulkan::VulkanDevice *>(creator->device());
  device->set_descriptor_set_cache_enabled(true);

  Device::AllocParams alloc_params;
  alloc_params.size = sizeof(uint32_t);
  alloc_params.usage = AllocUsage::Storage;
  DeviceAllocation first_allocation;
  DeviceAllocation second_allocation;
  ASSERT_EQ(device->allocate_memory(alloc_params, &first_allocation),
            RhiResult::success);
  ASSERT_EQ(device->allocate_memory(alloc_params, &second_allocation),
            RhiResult::success);
  DeviceAllocationGuard first_guard(first_allocation);
  DeviceAllocationGuard second_guard(second_allocation);

  vulkan::VulkanResourceSet first_resource_set(device);
  first_resource_set.rw_buffer(0, first_allocation);
  auto [first_result, first_set] = first_resource_set.finalize();
  ASSERT_EQ(first_result, RhiResult::success);
  ASSERT_NE(first_set, nullptr);

  vulkan::VulkanResourceSet cached_resource_set(device);
  cached_resource_set.rw_buffer(0, first_allocation);
  auto [cached_result, cached_set] = cached_resource_set.finalize();
  ASSERT_EQ(cached_result, RhiResult::success);
  ASSERT_EQ(cached_set, first_set);

  {
    std::lock_guard<std::mutex> lock(first_set->mutex);
    ++first_set->recording_use_count;
  }
  first_resource_set.rw_buffer(0, second_allocation);
  auto [replacement_result, replacement_set] =
      first_resource_set.finalize();
  ASSERT_EQ(replacement_result, RhiResult::success);
  ASSERT_NE(replacement_set, nullptr);
  EXPECT_NE(replacement_set, first_set);
  {
    std::lock_guard<std::mutex> lock(first_set->mutex);
    ASSERT_EQ(first_set->recording_use_count, 1u);
    --first_set->recording_use_count;
  }
}

TEST(VulkanSurfaceResultTest, ClassifiesRecoverableAndFatalResults) {
  using vulkan::VulkanSurfaceResult;
  using vulkan::classify_vulkan_surface_result;

  EXPECT_EQ(classify_vulkan_surface_result(VK_SUCCESS),
            VulkanSurfaceResult::kSuccess);
  EXPECT_EQ(classify_vulkan_surface_result(VK_SUBOPTIMAL_KHR),
            VulkanSurfaceResult::kSuboptimal);
  EXPECT_EQ(classify_vulkan_surface_result(VK_ERROR_OUT_OF_DATE_KHR),
            VulkanSurfaceResult::kOutOfDate);
  EXPECT_EQ(classify_vulkan_surface_result(VK_ERROR_DEVICE_LOST),
            VulkanSurfaceResult::kDeviceLost);
  EXPECT_EQ(classify_vulkan_surface_result(VK_ERROR_SURFACE_LOST_KHR),
            VulkanSurfaceResult::kError);
}

TEST(DeviceTest, ViewDevAllocAsNdarray) {
  // Otherwise will segfault on macOS VM,
  // where Vulkan is installed but no devices are present
  if (!vulkan::is_vulkan_api_available()) {
    return;
  }

  // Create Taichi Device for computation
  lang::vulkan::VulkanDeviceCreator::Params evd_params;
  evd_params.api_version = std::nullopt;
  auto embedded_device =
      std::make_unique<taichi::lang::vulkan::VulkanDeviceCreator>(evd_params);
  taichi::lang::vulkan::VulkanDevice *device_ =
      static_cast<taichi::lang::vulkan::VulkanDevice *>(
          embedded_device->device());

  aot_test_utils::view_devalloc_as_ndarray(device_);
}

TEST(VulkanPipelineCacheTest, SnapshotRoundTrip) {
  if (!vulkan::is_vulkan_api_available()) {
    return;
  }

  vulkan::VulkanDeviceCreator::Params params;
  params.api_version = std::nullopt;
  auto creator = std::make_unique<vulkan::VulkanDeviceCreator>(params);
  auto *device = static_cast<vulkan::VulkanDevice *>(creator->device());

  auto [cache, result] = device->create_pipeline_cache_unique();
  ASSERT_EQ(result, RhiResult::success);
  ASSERT_NE(cache, nullptr);

  auto *cache_data = static_cast<uint8_t *>(cache->data());
  const size_t cache_size = cache->size();
  ASSERT_NE(cache_data, nullptr);
  ASSERT_GT(cache_size, 0);
  std::vector<uint8_t> blob(cache_data, cache_data + cache_size);

  auto [restored, restored_result] =
      device->create_pipeline_cache_unique(blob.size(), blob.data());
  ASSERT_EQ(restored_result, RhiResult::success);
  ASSERT_NE(restored, nullptr);
  EXPECT_NE(restored->data(), nullptr);
  EXPECT_GT(restored->size(), 0);
}

TEST(VulkanPipelineCacheTest, ConcurrentSnapshots) {
  if (!vulkan::is_vulkan_api_available()) {
    return;
  }

  vulkan::VulkanDeviceCreator::Params params;
  params.api_version = std::nullopt;
  auto creator = std::make_unique<vulkan::VulkanDeviceCreator>(params);
  auto *device = static_cast<vulkan::VulkanDevice *>(creator->device());
  auto [cache, result] = device->create_pipeline_cache_unique();
  ASSERT_EQ(result, RhiResult::success);
  ASSERT_NE(cache, nullptr);

  constexpr int kThreadCount = 4;
  constexpr int kSnapshotsPerThread = 32;
  std::atomic<int> ready{0};
  std::atomic<bool> start{false};
  std::atomic<bool> succeeded{true};
  std::vector<std::thread> threads;
  threads.reserve(kThreadCount);
  for (int thread_index = 0; thread_index < kThreadCount; ++thread_index) {
    threads.emplace_back([&] {
      ready.fetch_add(1, std::memory_order_release);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (int snapshot_index = 0; snapshot_index < kSnapshotsPerThread;
           ++snapshot_index) {
        auto *data = cache->data();
        if (data == nullptr || cache->size() == 0) {
          succeeded.store(false, std::memory_order_relaxed);
          return;
        }
      }
    });
  }

  while (ready.load(std::memory_order_acquire) != kThreadCount) {
    std::this_thread::yield();
  }
  start.store(true, std::memory_order_release);
  for (auto &thread : threads) {
    thread.join();
  }

  EXPECT_TRUE(succeeded.load(std::memory_order_relaxed));
}

TEST(VulkanStreamTest, AdditionalSignalSemaphoreChainsSubmission) {
  if (!vulkan::is_vulkan_api_available()) {
    GTEST_SKIP();
  }

  vulkan::VulkanDeviceCreator::Params params;
  params.api_version = std::nullopt;
  auto creator = std::make_unique<vulkan::VulkanDeviceCreator>(params);
  auto *device = static_cast<vulkan::VulkanDevice *>(creator->device());
  auto *stream = static_cast<vulkan::VulkanStream *>(
      device->get_compute_stream());
  auto [producer, producer_result] = stream->new_command_list_unique();
  auto [consumer, consumer_result] = stream->new_command_list_unique();
  ASSERT_EQ(producer_result, RhiResult::success);
  ASSERT_EQ(consumer_result, RhiResult::success);

  producer->memory_barrier();
  consumer->memory_barrier();
  auto additional_vk_semaphore =
      vkapi::create_semaphore(device->vk_device(), 0);
  auto additional_semaphore =
      std::make_shared<vulkan::VulkanStreamSemaphoreObject>(
          device->backend_fault_reporter(), additional_vk_semaphore);
  ASSERT_TRUE(stream->submit_with_semaphores(
      producer.get(), {}, {additional_semaphore}));
  auto completion = stream->submit(consumer.get(), {additional_semaphore});
  ASSERT_TRUE(completion);
  EXPECT_TRUE(completion->wait());
}

TEST(VulkanProfilerTest, CommandListScopesKeepTheirOwnQueryPools) {
  if (!vulkan::is_vulkan_api_available()) {
    return;
  }

  vulkan::VulkanDeviceCreator::Params params;
  params.api_version = std::nullopt;
  auto creator = std::make_unique<vulkan::VulkanDeviceCreator>(params);
  auto *device = static_cast<vulkan::VulkanDevice *>(creator->device());
  auto *stream = device->get_compute_stream();
  auto [first, first_result] = stream->new_command_list_unique();
  auto [second, second_result] = stream->new_command_list_unique();
  ASSERT_EQ(first_result, RhiResult::success);
  ASSERT_EQ(second_result, RhiResult::success);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);

  first->begin_profiler_scope("first_scope");
  second->begin_profiler_scope("second_scope");
  first->end_profiler_scope();
  second->end_profiler_scope();
  ASSERT_TRUE(stream->submit(first.get()));
  ASSERT_TRUE(stream->submit(second.get()));
  stream->command_sync();

  // command_sync() drains completed profiler samplers as part of synchronizing
  // the stream, so the sampled records are ready to consume here.
  auto records = device->profiler_flush_sampled_time();
  ASSERT_EQ(records.size(), 2u);
  for (const auto &[name, duration_ms] : records) {
    EXPECT_TRUE(name == "first_scope" || name == "second_scope");
    EXPECT_GE(duration_ms, 0.0);
    EXPECT_LT(duration_ms, 10000.0);
  }
}

TEST(VulkanProfilerTest, CommandSyncCollectsOnlyItsOwnSubmissionFences) {
  if (!vulkan::is_vulkan_api_available()) {
    return;
  }

  vulkan::VulkanDeviceCreator::Params params;
  params.api_version = std::nullopt;
  auto creator = std::make_unique<vulkan::VulkanDeviceCreator>(params);
  auto *device = static_cast<vulkan::VulkanDevice *>(creator->device());
  auto *compute_stream = device->get_compute_stream();
  auto *graphics_stream = device->get_graphics_stream();
  auto [compute_cmd, compute_result] =
      compute_stream->new_command_list_unique();
  auto [graphics_cmd, graphics_result] =
      graphics_stream->new_command_list_unique();
  ASSERT_EQ(compute_result, RhiResult::success);
  ASSERT_EQ(graphics_result, RhiResult::success);
  ASSERT_NE(compute_cmd, nullptr);
  ASSERT_NE(graphics_cmd, nullptr);

  compute_cmd->begin_profiler_scope("compute_scope");
  compute_cmd->end_profiler_scope();
  graphics_cmd->begin_profiler_scope("graphics_scope");
  graphics_cmd->end_profiler_scope();
  ASSERT_TRUE(compute_stream->submit(compute_cmd.get()));
  ASSERT_TRUE(graphics_stream->submit(graphics_cmd.get()));

  // The two streams can alias the same VkQueue. Synchronizing the first one
  // must not turn into a queue-wide idle or collect the second stream's
  // profiler scope just because it was queued behind the first submission.
  compute_stream->command_sync();
  auto compute_records = device->profiler_flush_sampled_time();
  ASSERT_EQ(compute_records.size(), 1u);
  EXPECT_EQ(compute_records[0].first, "compute_scope");
  EXPECT_GE(compute_records[0].second, 0.0);

  graphics_stream->command_sync();
  auto graphics_records = device->profiler_flush_sampled_time();
  ASSERT_EQ(graphics_records.size(), 1u);
  EXPECT_EQ(graphics_records[0].first, "graphics_scope");
  EXPECT_GE(graphics_records[0].second, 0.0);
}

#if defined(TI_WITH_CUDA)
TEST(VulkanCudaInteropTest, ExternalMemoryCacheReleasesWithAllocation) {
  if (!vulkan::is_vulkan_api_available() ||
      !CUDADriver::get_instance_without_context().detected()) {
    GTEST_SKIP();
  }

  cuda::CudaDevice cuda_device;
  vulkan::VulkanDeviceCreator::Params params;
  params.api_version = std::nullopt;
  auto creator = std::make_unique<vulkan::VulkanDeviceCreator>(params);
  auto *vulkan_device =
      static_cast<vulkan::VulkanDevice *>(creator->device());
  if (!vulkan_device->vk_caps().external_memory) {
    GTEST_SKIP();
  }

  constexpr size_t kBytes = sizeof(uint32_t) * 16;
  Device::AllocParams vulkan_params;
  vulkan_params.size = kBytes;
  vulkan_params.export_sharing = true;
  vulkan_params.usage = AllocUsage::Storage;
  DeviceAllocation vulkan_allocation;
  ASSERT_EQ(vulkan_device->allocate_memory(vulkan_params, &vulkan_allocation),
            RhiResult::success);

  Device::AllocParams cuda_params;
  cuda_params.size = kBytes;
  DeviceAllocation cuda_source;
  DeviceAllocation cuda_destination;
  ASSERT_EQ(cuda_device.allocate_memory(cuda_params, &cuda_source),
            RhiResult::success);
  ASSERT_EQ(cuda_device.allocate_memory(cuda_params, &cuda_destination),
            RhiResult::success);

  std::array<uint32_t, 16> input{};
  for (uint32_t i = 0; i < input.size(); ++i) {
    input[i] = i * 17 + 3;
  }
  const void *input_ptr = input.data();
  size_t copy_size = kBytes;
  DevicePtr source_ptr = cuda_source.get_ptr();
  ASSERT_EQ(cuda_device.upload_data(&source_ptr, &input_ptr, &copy_size),
            RhiResult::success);

  Device::memcpy_direct(vulkan_allocation.get_ptr(), cuda_source.get_ptr(),
                        kBytes);
  Device::memcpy_direct(cuda_destination.get_ptr(),
                        vulkan_allocation.get_ptr(),
                        kBytes);

  std::array<uint32_t, 16> output{};
  void *output_ptr = output.data();
  copy_size = kBytes;
  DevicePtr destination_ptr = cuda_destination.get_ptr();
  ASSERT_EQ(cuda_device.readback_data(&destination_ptr, &output_ptr,
                                      &copy_size),
            RhiResult::success);
  EXPECT_EQ(output, input);

  // This exercises the generation-keyed interop cache purge before Vulkan
  // returns the allocation slot to its pointer-stable object list.
  vulkan_device->dealloc_memory(vulkan_allocation);
  cuda_device.dealloc_memory(cuda_source);
  cuda_device.dealloc_memory(cuda_destination);
}

TEST(VulkanCudaInteropTest,
     ExternalAllocationSemaphoreRoundTripAndLiveOwnerReset) {
  if (!vulkan::is_vulkan_api_available() ||
      !CUDADriver::get_instance_without_context().detected()) {
    GTEST_SKIP();
  }

  vulkan::VulkanDeviceCreator::Params params;
  params.api_version = std::nullopt;
  auto creator = std::make_unique<vulkan::VulkanDeviceCreator>(params);
  auto *vulkan_device = static_cast<vulkan::VulkanDevice *>(creator->device());
  if (!vulkan_device->vk_caps().external_memory ||
      !vulkan_device->vk_caps().external_semaphore) {
    GTEST_SKIP();
  }

  Program program(Arch::cuda);
  auto *cuda_device =
      dynamic_cast<cuda::CudaDevice *>(program.get_compute_device());
  ASSERT_NE(cuda_device, nullptr);

  constexpr uint32_t kElementCount = 32;
  constexpr size_t kBytes = sizeof(uint32_t) * kElementCount;
  Device::AllocParams shared_params;
  shared_params.size = kBytes;
  shared_params.export_sharing = true;
  shared_params.usage = AllocUsage::Storage;
  DeviceAllocation shared_allocation;
  ASSERT_EQ(vulkan_device->allocate_memory(shared_params, &shared_allocation),
            RhiResult::success);

  Device::AllocParams destination_params;
  destination_params.size = kBytes;
  destination_params.usage = AllocUsage::Storage;
  DeviceAllocation destination;
  ASSERT_EQ(vulkan_device->allocate_memory(destination_params, &destination),
            RhiResult::success);

  auto adapter = VulkanCudaExternalAllocation::create(
      vulkan_device, cuda_device, shared_allocation);
  ASSERT_NE(adapter, nullptr);
  ASSERT_EQ(adapter->allocation_size(), kBytes);
  EXPECT_EQ(adapter->access_state(),
            VulkanCudaExternalAllocation::AccessState::kVulkanOwned);
  auto *vulkan_stream =
      static_cast<vulkan::VulkanStream *>(vulkan_device->get_compute_stream());

  auto [producer, producer_result] = vulkan_stream->new_command_list_unique();
  ASSERT_EQ(producer_result, RhiResult::success);
  producer->buffer_fill(shared_allocation.get_ptr(), kBytes, 7u);
  ASSERT_TRUE(adapter->release_vulkan_to_cuda(*vulkan_stream, producer.get()));

  CudaInteropStream cuda_stream;
  const auto stream_domain =
      ExternalStreamDomain::cuda(program.runtime_program_generation(),
                                 adapter->identity(), cuda_stream.get());
  adapter->acquire_for_consumer(stream_domain);
  EXPECT_EQ(adapter->access_state(),
            VulkanCudaExternalAllocation::AccessState::kCudaOwned);
  CudaInteropAddKernel add_kernel;
  void *shared_cuda_ptr =
      cuda_device->get_memory_addr(adapter->cuda_allocation());
  ASSERT_NE(shared_cuda_ptr, nullptr);
  add_kernel.launch(shared_cuda_ptr, kElementCount, 5u, cuda_stream.get());
  adapter->release_from_consumer(stream_domain);
  EXPECT_EQ(adapter->access_state(),
            VulkanCudaExternalAllocation::AccessState::kAwaitingVulkanAcquire);

  auto [consumer, consumer_result] = vulkan_stream->new_command_list_unique();
  ASSERT_EQ(consumer_result, RhiResult::success);
  consumer->buffer_copy(destination.get_ptr(), shared_allocation.get_ptr(),
                        kBytes);
  auto completion =
      adapter->cycle_vulkan_to_cuda(*vulkan_stream, consumer.get());
  ASSERT_TRUE(completion);
  EXPECT_EQ(adapter->access_state(),
            VulkanCudaExternalAllocation::AccessState::kAwaitingCudaAcquire);
  ASSERT_TRUE(completion->wait());

  std::array<uint32_t, kElementCount> output{};
  void *output_ptr = output.data();
  size_t copy_size = kBytes;
  DevicePtr destination_ptr = destination.get_ptr();
  ASSERT_EQ(
      vulkan_device->readback_data(&destination_ptr, &output_ptr, &copy_size),
      RhiResult::success);
  for (const uint32_t value : output) {
    EXPECT_EQ(value, 12u);
  }

  const auto owner = program.register_external_dense_storage(
      adapter->cuda_allocation(), adapter->allocation_size(),
      [adapter] { adapter->close(); });
  EXPECT_TRUE(program.validate_external_dense_storage_owner(owner));
  program.finalize();
  EXPECT_TRUE(adapter->closed());

  vulkan_device->dealloc_memory(destination);
  vulkan_device->dealloc_memory(shared_allocation);
}

TEST(VulkanCudaInteropTest, HostFallbackWithoutExternalMemory) {
  if (!vulkan::is_vulkan_api_available() ||
      !CUDADriver::get_instance_without_context().detected()) {
    GTEST_SKIP();
  }

  cuda::CudaDevice cuda_device;
  vulkan::VulkanDeviceCreator::Params params;
  params.api_version = std::nullopt;
  auto creator = std::make_unique<vulkan::VulkanDeviceCreator>(params);
  auto *vulkan_device =
      static_cast<vulkan::VulkanDevice *>(creator->device());
  // Exercise the capability-negotiated fallback even on a developer machine
  // whose Vulkan driver supports external-memory exports.
  vulkan_device->vk_caps().external_memory = false;

  constexpr size_t kBytes = sizeof(uint32_t) * 16;
  Device::AllocParams vulkan_params;
  vulkan_params.size = kBytes;
  vulkan_params.usage = AllocUsage::Storage;
  DeviceAllocation vulkan_allocation;
  ASSERT_EQ(vulkan_device->allocate_memory(vulkan_params, &vulkan_allocation),
            RhiResult::success);

  Device::AllocParams cuda_params;
  cuda_params.size = kBytes;
  DeviceAllocation cuda_source;
  DeviceAllocation cuda_destination;
  ASSERT_EQ(cuda_device.allocate_memory(cuda_params, &cuda_source),
            RhiResult::success);
  ASSERT_EQ(cuda_device.allocate_memory(cuda_params, &cuda_destination),
            RhiResult::success);

  std::array<uint32_t, 16> input{};
  for (uint32_t i = 0; i < input.size(); ++i) {
    input[i] = i * 19 + 5;
  }
  const void *input_ptr = input.data();
  size_t copy_size = kBytes;
  DevicePtr source_ptr = cuda_source.get_ptr();
  ASSERT_EQ(cuda_device.upload_data(&source_ptr, &input_ptr, &copy_size),
            RhiResult::success);

  Device::memcpy_direct(vulkan_allocation.get_ptr(), cuda_source.get_ptr(),
                        kBytes);
  Device::memcpy_direct(cuda_destination.get_ptr(), vulkan_allocation.get_ptr(),
                        kBytes);

  std::array<uint32_t, 16> output{};
  void *output_ptr = output.data();
  copy_size = kBytes;
  DevicePtr destination_ptr = cuda_destination.get_ptr();
  ASSERT_EQ(cuda_device.readback_data(&destination_ptr, &output_ptr,
                                      &copy_size),
            RhiResult::success);
  EXPECT_EQ(output, input);

  vulkan_device->dealloc_memory(vulkan_allocation);
  cuda_device.dealloc_memory(cuda_source);
  cuda_device.dealloc_memory(cuda_destination);
}
#endif
