#include "gtest/gtest.h"

#ifdef TI_WITH_CUDA
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include "taichi/platform/cuda/detect_cuda.h"
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/runtime/cuda/kernel_launcher.h"
#include "taichi/runtime/llvm/aot_graph_data.h"
#include "taichi/runtime/llvm/llvm_aot_module_loader.h"

namespace taichi::lang {
namespace {

struct GatedPackets {
  std::array<cuda::KernelLauncher::GraphLaunchPacket, 2> packets;
  CUstream stream{nullptr};
  CUgraph graph{nullptr};
  CUgraphExec executable{nullptr};
  ~GatedPackets() {
    auto &driver = CUDADriver::get_instance();
    driver.stream_synchronize(nullptr);
    if (stream)
      driver.stream_synchronize(stream);
    if (executable)
      driver.graph_exec_destroy(executable);
    if (graph)
      driver.graph_destroy(graph);
    for (auto &packet : packets)
      if (packet.device_arg_buffer)
        driver.mem_free(packet.device_arg_buffer);
    if (stream)
      driver.stream_destroy(stream);
  }
};

TEST(CudaGatedAot, LoadsDeclaredCodeAndRejectsMismatchedFrames) {
  const char *folder = std::getenv("TAICHI_AOT_FOLDER_PATH");
  if (!folder || !is_cuda_api_available())
    GTEST_SKIP() << "Requires the saved CUDA gated-code fixture";
  CompileConfig config;
  config.arch = Arch::cuda;
  LlvmRuntimeExecutor executor(config, nullptr);
  uint64 *result_buffer = nullptr;
  executor.materialize_runtime(nullptr, &result_buffer);
  LLVM::AotModuleParams params;
  params.module_path = folder;
  params.executor_ = &executor;
  params.kernel_launcher = std::make_unique<cuda::KernelLauncher>(
      cuda::KernelLauncher::Config{&executor});
  auto module = LLVM::make_aot_module(std::move(params));
  auto *ordinary = dynamic_cast<llvm_aot::KernelImpl *>(
      module->get_kernel("forward"));
  auto *forward = dynamic_cast<llvm_aot::KernelImpl *>(
      module->get_kernel("gated_forward"));
  auto *reverse = dynamic_cast<llvm_aot::KernelImpl *>(
      module->get_kernel("gated_reverse"));
  ASSERT_NE(ordinary, nullptr);
  ASSERT_NE(forward, nullptr);
  ASSERT_NE(reverse, nullptr);
  auto &launcher = dynamic_cast<cuda::KernelLauncher &>(forward->launcher());
  EXPECT_EQ(launcher.debug_registered_kernel_count(), 3);
  EXPECT_EQ(module->get_kernel("gated_forward"), forward);
  auto &device = *executor.get_compute_device();
  std::vector<DeviceAllocationUnique> storage;
  for (int i = 0; i < 6; ++i) {
    auto allocation = device.allocate_memory_unique({16});
    ASSERT_EQ(allocation.second, RhiResult::success);
    storage.push_back(std::move(allocation.first));
  }
  auto address = [&](int slot) {
    return static_cast<cuda::CudaDevice &>(device)
        .get_alloc_info(storage[slot]->get_ptr()).ptr;
  };
  auto &driver = CUDADriver::get_instance();
  std::array<float, 4> x{1, 2, 3, 4}, next_x{2, 3, 4, 5};
  std::array<float, 4> sentinel{-99, -99, -99, -99};
  std::array<float, 4> zero{0, 0, 0, 0}, one{1, 1, 1, 1};
  driver.memcpy_host_to_device(address(0), x.data(), 16);
  driver.memcpy_host_to_device(address(1), sentinel.data(), 16);
  driver.memcpy_host_to_device(address(2), zero.data(), 16);
  driver.memcpy_host_to_device(address(3), one.data(), 16);
  driver.memcpy_host_to_device(address(4), next_x.data(), 16);
  auto bind = [&](LaunchContextBuilder &context, int input) {
    context.set_arg_external_array_with_shape(
        {0}, reinterpret_cast<std::uintptr_t>(address(input)), 16, {4},
        reinterpret_cast<std::uintptr_t>(address(2)));
    context.set_arg_external_array_with_shape(
        {1}, reinterpret_cast<std::uintptr_t>(address(1)), 16, {4},
        reinterpret_cast<std::uintptr_t>(address(3)));
  };
  LaunchContextBuilder ordinary_context(ordinary);
  bind(ordinary_context, 0);
  GatedPackets captured;
  std::array<llvm_aot::KernelImpl *, 2> kernels{forward, reverse};
  for (std::size_t i = 0; i < kernels.size(); ++i) {
    LaunchContextBuilder context(kernels[i]);
    bind(context, 0);
    EXPECT_ANY_THROW(kernels[i]->launch(context));
    EXPECT_FALSE(launcher.prepare_cuda_graph_launch_packet(
        kernels[i]->handle(), context, captured.packets[i]));
    EXPECT_FALSE(launcher.prepare_cuda_graph_bounded_packet(
        kernels[i]->handle(), context, captured.packets[i], address(5), 4));
    ASSERT_TRUE(launcher.prepare_cuda_graph_gated_launch(
        kernels[i]->handle(), context, captured.packets[i], address(5), 7,
        nullptr));
    driver.stream_synchronize(nullptr);  // Keep host arguments alive for upload.
  }
  cuda::KernelLauncher::GraphLaunchPacket invalid;
  EXPECT_FALSE(launcher.prepare_cuda_graph_gated_packet(
      ordinary->handle(), ordinary_context, invalid, address(5), 7));
  driver.stream_synchronize(nullptr);
  std::array<float, 4> actual;
  driver.memcpy_device_to_host(actual.data(), address(1), 16);
  EXPECT_EQ(actual, sentinel);  // Preparation cannot execute a payload.
  driver.stream_create(&captured.stream, CU_STREAM_NON_BLOCKING);
  driver.stream_begin_capture(captured.stream, CU_STREAM_CAPTURE_MODE_RELAXED);
  for (const auto &packet : captured.packets)
    launcher.capture_cuda_graph_launch(packet, captured.stream);
  driver.stream_end_capture(captured.stream, &captured.graph);
  driver.graph_instantiate_with_flags(&captured.executable, captured.graph, 0);
  auto replay = [&](std::uint32_t gate) {
    driver.memcpy_host_to_device(address(5), &gate, sizeof(gate));
    driver.graph_launch(captured.executable, captured.stream);
    driver.stream_synchronize(captured.stream);
  };
  replay(1);  // Any nonzero value is insufficient; expected is exactly seven.
  driver.memcpy_device_to_host(actual.data(), address(1), 16);
  EXPECT_EQ(actual, sentinel);
  replay(7);
  driver.memcpy_device_to_host(actual.data(), address(1), 16);
  EXPECT_EQ(actual, (std::array<float, 4>{1, 4, 9, 16}));
  driver.memcpy_device_to_host(actual.data(), address(2), 16);
  EXPECT_EQ(actual, (std::array<float, 4>{2, 4, 6, 8}));
  driver.memcpy_device_to_host(actual.data(), address(3), 16);
  EXPECT_EQ(actual, zero);  // Reverse consumes the output adjoint.
  driver.memcpy_host_to_device(address(3), one.data(), 16);
  replay(0);
  driver.memcpy_device_to_host(actual.data(), address(2), 16);
  EXPECT_EQ(actual, (std::array<float, 4>{2, 4, 6, 8}));
  driver.memcpy_device_to_host(actual.data(), address(3), 16);
  EXPECT_EQ(actual, one);  // A closed gate cannot consume the next seed.
  std::array<std::vector<std::uint8_t>, 2> updates;
  for (std::size_t i = 0; i < kernels.size(); ++i) {
    LaunchContextBuilder context(kernels[i]);
    bind(context, 4);
    ASSERT_TRUE(launcher.update_cuda_graph_launch(
        captured.packets[i], context, updates[i], nullptr));
  }
  driver.stream_synchronize(nullptr);
  driver.memcpy_host_to_device(address(2), zero.data(), 16);
  replay(7);
  driver.memcpy_device_to_host(actual.data(), address(1), 16);
  EXPECT_EQ(actual, (std::array<float, 4>{4, 9, 16, 25}));
  driver.memcpy_device_to_host(actual.data(), address(2), 16);
  EXPECT_EQ(actual, (std::array<float, 4>{4, 6, 8, 10}));
  ordinary->launch(ordinary_context);
  driver.stream_synchronize(nullptr);
  driver.memcpy_device_to_host(actual.data(), address(1), 16);
  EXPECT_EQ(actual, (std::array<float, 4>{1, 4, 9, 16}));
  EXPECT_EQ(launcher.debug_registered_kernel_count(), 3);
  // A bounded host submission-to-completion observation, not GPU-only time
  // or an application performance claim. Reset AD seeds outside each sample.
  std::array<double, 16> samples;
  for (auto &sample : samples) {
    driver.memcpy_host_to_device(address(2), zero.data(), 16);
    driver.memcpy_host_to_device(address(3), one.data(), 16);
    const auto begin = std::chrono::steady_clock::now();
    driver.graph_launch(captured.executable, captured.stream);
    driver.stream_synchronize(captured.stream);
    sample = std::chrono::duration<double, std::micro>(
                 std::chrono::steady_clock::now() - begin).count();
  }
  std::sort(samples.begin(), samples.end());
  fmt::print("GATE_COSTS {{\"warm_complete_median_us\":{},"
             "\"device_prefix_bytes\":{}}}\n",
             (samples[7] + samples[8]) / 2,
             captured.packets[0].arg_buffer_prefix_size +
                 captured.packets[1].arg_buffer_prefix_size);
}

}  // namespace
}  // namespace taichi::lang
#endif
