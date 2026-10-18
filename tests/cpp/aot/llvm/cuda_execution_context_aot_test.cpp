#include "gtest/gtest.h"

#if defined(TI_WITH_CUDA) && defined(TI_WITH_VULKAN)
#include <array>
#include <chrono>
#include <cstdlib>
#include <thread>
#include "llvm/Transforms/Utils/Cloning.h"
#include "taichi/program/runtime_fault.h"
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/runtime/cuda/execution_domain.h"
#include "taichi/runtime/cuda/kernel_launcher.h"
#include "taichi/runtime/llvm/aot_graph_data.h"
#include "taichi/runtime/llvm/llvm_aot_module_loader.h"
#include "taichi/runtime/llvm/runtime_context_abi.h"
#include "taichi/system/profiler_annotation.h"
#include "tests/cpp/rhi/common/cuda_external_gate.h"

namespace taichi::lang {
namespace {

struct ContextPackets {
  std::array<cuda::KernelLauncher::GraphLaunchPacket, 2> packets;
  ~ContextPackets() {
    // Domains are declared after this owner and drain before argument release.
    auto guard = CUDAContext::get_instance().get_guard();
    for (auto &packet : packets)
      if (packet.device_arg_buffer)
        CUDADriver::get_instance().mem_free.call(packet.device_arg_buffer);
  }
};

TEST(CudaExecutionContextAot, IsolatesTemporariesInsideAnInterleavedKernel) {
  const auto *folder = std::getenv("TAICHI_F3_CONTEXT_AOT_FOLDER");
  if (!folder || !cuda::ExecutionDomain::available())
    GTEST_SKIP() << "Requires the saved execution-context AOT fixture";
  CompileConfig config;
  config.arch = Arch::cuda;
  LlvmRuntimeExecutor executor(config, nullptr);
  uint64 *result = nullptr;
  executor.materialize_runtime(nullptr, &result);
  LLVM::AotModuleParams params;
  params.module_path = folder;
  params.executor_ = &executor;
  params.kernel_launcher = std::make_unique<cuda::KernelLauncher>(
      cuda::KernelLauncher::Config{&executor});
  auto module = LLVM::make_aot_module(std::move(params));
  auto &device =
      static_cast<cuda::CudaDevice &>(*executor.get_compute_device());
  auto context = CUDAContext::get_instance().get_guard();
  auto &driver = CUDADriver::get_instance();
  constexpr int count = 64;
  std::vector<DeviceAllocationUnique> storage;
  for (int i = 0; i < 5; ++i) {
    const std::size_t bytes = i < 2   ? count * sizeof(float)
                              : i < 4 ? taichi_global_tmp_buffer_size
                                      : 4;
    auto allocation = device.allocate_memory_unique({bytes});
    ASSERT_EQ(allocation.second, RhiResult::success);
    storage.push_back(std::move(allocation.first));
    driver.memsetd32(device.get_memory_addr(*storage.back()), 0, bytes / 4);
  }
  auto address = [&](int index) {
    return device.get_memory_addr(*storage[index]);
  };
  const std::uint32_t open = 7;
  driver.memcpy_host_to_device(address(4), const_cast<std::uint32_t *>(&open),
                               sizeof(open));
  driver.stream_synchronize(nullptr);

  for (const auto *name : {"stage", "gated_stage"}) {
    SCOPED_TRACE(name);
    auto *kernel =
        dynamic_cast<llvm_aot::KernelImpl *>(module->get_kernel(name));
    ASSERT_NE(kernel, nullptr);
    auto &launcher = dynamic_cast<cuda::KernelLauncher &>(kernel->launcher());
    const bool gated = std::string(name) == "gated_stage";
    ContextPackets captured;
    for (int i = 0; i < 2; ++i) {
      LaunchContextBuilder args(kernel);
      args.set_arg_external_array_with_shape(
          {0}, reinterpret_cast<std::uintptr_t>(address(i)), count * 4,
          {count});
      args.set_arg_float({1}, i == 0 ? 2.0 : 7.0);
      args.set_arg_int({2}, count);
      auto &packet = captured.packets[i];
      ASSERT_TRUE(
          gated ? launcher.prepare_cuda_graph_gated_launch(
                      kernel->handle(), args, packet, address(4), open, nullptr)
                : launcher.prepare_cuda_graph_launch(kernel->handle(), args,
                                                     packet, nullptr));
      driver.stream_synchronize(nullptr);  // Cold argument publication only.
      EXPECT_ANY_THROW(launcher.bind_execution_temporaries(
          packet, nullptr, taichi_global_tmp_buffer_size));
      EXPECT_ANY_THROW(
          launcher.bind_execution_temporaries(packet, address(2 + i), 8));
      launcher.bind_execution_temporaries(packet, address(2 + i),
                                          taichi_global_tmp_buffer_size);
      ASSERT_GE(packet.offloaded_tasks.size(), 3);
      ASSERT_EQ(packet.offloaded_tasks.front().task_type,
                OffloadedTaskType::serial);
    }

    auto fault = std::make_shared<RuntimeFaultDomain>(Arch::cuda, 719);
    auto prelude_events =
        std::make_shared<RuntimeCompletionCudaEventPool>(fault, 1);
    cuda::ExecutionDomain first(fault, {1, 0});
    cuda::ExecutionDomain second(fault, {1, 0});
    testing::CudaExternalGate gate;
    // Deliberately split the real compiled task sequence. The first task
    // publishes scalar/vector-derived values into global temporaries; the
    // external gate holds its consumers while another instance completes.
    auto prelude = captured.packets[0];
    prelude.offloaded_tasks.resize(1);
    auto suffix = captured.packets[0];
    suffix.offloaded_tasks.erase(suffix.offloaded_tasks.begin());
    RuntimeCompletion prelude_done;
    const auto prelude_sequence = fault->allocate_submission_sequence();
    RuntimeCompletion a;
    {
      ScopedExternalProfilerAnnotation annotation(
          std::string("F3/context_first/") + name);
      a = first.submit({}, nullptr, [&](void *stream) {
        launcher.capture_cuda_graph_launch(prelude, stream);
        prelude_done = RuntimeCompletion::from_cuda_stream(
            719, prelude_sequence, stream, fault, prelude_events);
        gate.enqueue_wait(stream);
        launcher.capture_cuda_graph_launch(suffix, stream);
      });
    }
    const auto prelude_deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!prelude_done.done() &&
           std::chrono::steady_clock::now() < prelude_deadline)
      std::this_thread::yield();
    ASSERT_TRUE(prelude_done.done());
    ASSERT_FALSE(a.done());
    RuntimeCompletion b;
    {
      ScopedExternalProfilerAnnotation annotation(
          std::string("F3/context_second/") + name);
      b = second.submit({}, nullptr, [&](void *stream) {
        launcher.capture_cuda_graph_launch(captured.packets[1], stream);
      });
    }
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!b.done() && std::chrono::steady_clock::now() < deadline)
      std::this_thread::yield();
    EXPECT_TRUE(b.done());
    EXPECT_FALSE(a.done());
    EXPECT_FALSE(gate.timed_out());
    {
      ScopedExternalProfilerAnnotation annotation(
          std::string("F3/context_release/") + name);
      gate.release();
    }
    a.wait();
    b.wait();
    std::array<float, count> values{};
    for (int i = 0; i < 2; ++i) {
      driver.memcpy_device_to_host(values.data(), address(i), sizeof(values));
      const float seed = i == 0 ? 2.0f : 7.0f;
      const float carried = seed * 5.0f + 1.0f;
      for (int j = 0; j < count; ++j)
        EXPECT_FLOAT_EQ(values[j], carried * (count * (count + 1) / 2) + j);
    }
  }
}

TEST(CudaExecutionContextAot,
     ContextLayoutMetadataSurvivesCloneAndRejectsUnknown) {
  llvm::LLVMContext context;
  llvm::Module legacy("legacy", context);
  EXPECT_EQ(LLVM::runtime_context_abi(legacy), 1);
  LLVM::declare_runtime_context_abi(legacy);
  EXPECT_EQ(LLVM::runtime_context_abi(legacy), 2);
  EXPECT_EQ(LLVM::runtime_context_abi(*llvm::CloneModule(legacy)), 2);
  EXPECT_ANY_THROW(LLVM::declare_runtime_context_abi(legacy));
  llvm::Module unknown("unknown", context);
  unknown.addModuleFlag(llvm::Module::Error, LLVM::kRuntimeContextABIFlag, 99);
  EXPECT_ANY_THROW(LLVM::runtime_context_abi(unknown));
  EXPECT_EQ(cuda::launch_runtime_context_abi(cuda::kContextLaunchABI), 2);
  EXPECT_FALSE(cuda::is_graph_gate_launch_abi(cuda::kContextLaunchABI));
  EXPECT_TRUE(cuda::is_graph_gate_launch_abi(cuda::kGraphGateContextLaunchABI));
  EXPECT_EQ(cuda::launch_runtime_context_abi("ordinary"), 1);
}

}  // namespace
}  // namespace taichi::lang
#endif
