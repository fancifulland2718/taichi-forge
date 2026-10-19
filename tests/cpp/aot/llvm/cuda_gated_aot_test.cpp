#include "gtest/gtest.h"

#ifdef TI_WITH_CUDA
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <limits>
#include "taichi/platform/cuda/detect_cuda.h"
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/runtime/cuda/kernel_launcher.h"
#include "taichi/runtime/cuda/fixed_graph.h"
#include "taichi/runtime/fixed_execution.h"
#include "taichi/rhi/cuda/graph_region_builder.h"
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
        driver.mem_free_async(packet.device_arg_buffer, nullptr);
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

struct MaskedAotFrame {
  Device &device;
  llvm_aot::KernelImpl *ordinary, *forward, *reverse, *control, *gated_control;
  std::vector<DeviceAllocationUnique> storage;
  std::array<float, 4> values;

  MaskedAotFrame(Device &device, aot::Module &module, float offset)
      : device(device) {
    const auto entry = [&](const char *name) {
      return dynamic_cast<llvm_aot::KernelImpl *>(module.get_kernel(name));
    };
    ordinary = entry("forward");
    forward = entry("gated_forward");
    reverse = entry("gated_reverse");
    control = entry("control");
    gated_control = entry("gated_control");
    for (int slot = 0; slot < 5; ++slot) {
      auto allocation = device.allocate_memory_unique({slot == 4 ? 56u : 16u});
      if (allocation.second != RhiResult::success)
        throw std::runtime_error("Cannot allocate masked AOT fixture");
      storage.push_back(std::move(allocation.first));
    }
    for (std::size_t i = 0; i < values.size(); ++i)
      values[i] = offset + i;
    CUDADriver::get_instance().memcpy_host_to_device(address(0), values.data(), 16);
  }
  DevicePtr ptr(int slot, std::size_t word = 0) {
    auto result = storage[slot]->get_ptr();
    result.offset = word * 4;
    return result;
  }
  void *address(int slot, std::size_t word = 0) {
    return static_cast<char *>(static_cast<cuda::CudaDevice &>(device)
                                   .get_alloc_info(ptr(slot)).ptr) + word * 4;
  }
  struct Prepared {
    std::unique_ptr<FixedExecution> fixed;
    std::unique_ptr<cuda::FixedGraph> direct;
    void run() { fixed ? fixed->run() : direct->run(); }
  };
  FixedExecution::Dispatch dispatch(
      llvm_aot::KernelImpl *kernel, int operation,
      std::vector<std::unique_ptr<LaunchContextBuilder>> &contexts) {
    auto context = std::make_unique<LaunchContextBuilder>(kernel);
    if (operation >= 0) {
      context->set_arg_external_array_with_shape(
          {0}, reinterpret_cast<std::uintptr_t>(address(4)), 56, {14});
      context->set_arg_external_array_with_shape(
          {1}, reinterpret_cast<std::uintptr_t>(address(3)), 16, {4});
      context->set_arg_int({2}, operation);
    } else {
      context->set_arg_external_array_with_shape(
          {0}, reinterpret_cast<std::uintptr_t>(address(0)), 16, {4},
          reinterpret_cast<std::uintptr_t>(address(2)));
      context->set_arg_external_array_with_shape(
          {1}, reinterpret_cast<std::uintptr_t>(address(1)), 16, {4},
          reinterpret_cast<std::uintptr_t>(address(3)));
    }
    auto result = FixedExecution::Dispatch{kernel, context.get()};
    contexts.push_back(std::move(context));
    return result;
  }
  static cuda::FixedGraph::Dispatch native_dispatch(
      const FixedExecution::Dispatch &source) {
    auto &entry = static_cast<llvm_aot::KernelImpl &>(*source.kernel);
    return {&static_cast<cuda::KernelLauncher &>(entry.launcher()),
            entry.handle(), source.context, {}, {}};
  }
  Prepared prepare(std::uint64_t outer, std::uint64_t inner, bool direct,
                   bool conditional = false, int invalid = 0, int flat = 0) {
    auto &driver = CUDADriver::get_instance();
    driver.memsetd32(address(4), 99, 14);
    std::vector<std::unique_ptr<LaunchContextBuilder>> contexts;
    const auto dispatch = [&](llvm_aot::KernelImpl *kernel, int operation) {
      return this->dispatch(kernel, operation, contexts);
    };
    auto loop = std::make_shared<FixedExecution::While>();
    loop->predicate = ptr(4, 1);
    loop->max_iterations = outer;
    loop->cuda_control.emplace();
    loop->cuda_control->encoding = CudaControlEncoding::masked;
    loop->cuda_control->inners.push_back({ptr(4, 3), 1, 2, 5, inner});
    loop->cuda_control->inners.push_back({ptr(4, 11), 5, 6, 9, inner});
    loop->condition = {dispatch(invalid == 2 ? gated_control : control, 0)};
    loop->body = {dispatch(gated_control, 1), dispatch(gated_control, 2),
                  dispatch(invalid == 1 ? ordinary : forward, -1),
                  dispatch(reverse, -1), dispatch(gated_control, 6),
                  dispatch(gated_control, 7), dispatch(forward, -1),
                  dispatch(reverse, -1), dispatch(gated_control, 3)};
    if (flat) {
      loop->cuda_control->inners.clear();
      loop->continue_while_nonzero = flat == 1;
      loop->condition = {
          dispatch(invalid == 2 ? gated_control : control, flat == 1 ? 0 : 8)};
      loop->body = {dispatch(gated_control, 2),
                    dispatch(invalid == 1 ? ordinary : forward, -1),
                    dispatch(reverse, -1),
                    dispatch(gated_control, flat == 1 ? 3 : 9)};
    }
    if (invalid == 3)
      loop->body.front() = FixedExecution::BufferFill{ptr(4, 2), 4, 0};
    if (invalid == 4) {
      auto &bounded = std::get<FixedExecution::Dispatch>(loop->body.front());
      bounded.bounded_extent = ptr(4, 4);
      bounded.bounded_capacity = 1;
    }
    Prepared result;
    if (direct) {
      auto native = std::make_shared<cuda::FixedGraph::While>();
      native->predicate = address(4, 1);
      native->max_iterations = outer;
      native->continue_while_nonzero = loop->continue_while_nonzero;
      native->recipe.emplace();
      native->recipe->encoding = CudaControlEncoding::masked;
      if (!flat) {
        native->recipe->inners.push_back({address(4, 3), 1, 2, 5, inner});
        native->recipe->inners.push_back({address(4, 11), 5, 6, 9, inner});
      }
      for (const auto &item : loop->condition)
        native->condition.emplace_back(
            native_dispatch(std::get<FixedExecution::Dispatch>(item)));
      for (const auto &item : loop->body)
        native->body.emplace_back(
            native_dispatch(std::get<FixedExecution::Dispatch>(item)));
      // The identical descriptor really occurs twice at the CUDA layer, where
      // ordinary arguments can share but gate-prefixed packets cannot.
      std::vector<cuda::FixedGraph::Operation> operations{
          native_dispatch(dispatch(control, 4)), native, native,
          native_dispatch(dispatch(control, 5))};
      result.direct = std::make_unique<cuda::FixedGraph>(operations);
      EXPECT_EQ(result.direct->control_bytes(), flat ? 8 : 24);
    } else {
      std::vector<FixedExecution::Operation> operations{
          FixedExecution::BufferFill{ptr(4, 7), 4, 77}, loop, loop,
          FixedExecution::BufferCopy{ptr(4, 8), ptr(4, 6), 4}};
      if (conditional) {
        auto branch = std::make_shared<FixedExecution::Branch>();
        branch->selector = ptr(4, 4);
        branch->selection = {BranchSelection::Kind::nonzero, true};
        branch->bodies = {{FixedExecution::BufferFill{ptr(4, 7), 4, 11}},
                          {FixedExecution::BufferFill{ptr(4, 7), 4, 22}}};
        operations.emplace_back(std::move(branch));
      }
      result.fixed = std::make_unique<FixedExecution>(Arch::cuda, device, nullptr,
                                                     operations);
      EXPECT_EQ(result.fixed->execution_kind(), 1);
    }
    std::array<std::uint32_t, 14> before;
    driver.memcpy_device_to_host(before.data(), address(4), 56);
    for (auto value : before)
      EXPECT_EQ(value, 99);  // Preparation cannot execute the prefix/condition.
    return result;  // Every cold source descriptor/context has been discarded.
  }
  Prepared prepare_branch(BranchSelection selection, std::size_t cases,
                          bool direct, int empty_case = -1, int invalid = 0,
                          bool conditional_sibling = false) {
    auto &driver = CUDADriver::get_instance();
    driver.memsetd32(address(4), 99, 14);
    std::vector<std::unique_ptr<LaunchContextBuilder>> contexts;
    const auto dispatch = [&](llvm_aot::KernelImpl *kernel, int operation) {
      return this->dispatch(kernel, operation, contexts);
    };
    auto branch = std::make_shared<FixedExecution::Branch>();
    branch->selector = ptr(4, 1);
    branch->selection = selection;
    branch->mode = BranchExecutionMode::cuda_masked;
    branch->condition = {dispatch(invalid == 2 ? gated_control : control, 10)};
    for (std::size_t i = 0; i < cases + std::size_t(selection.has_default); ++i) {
      branch->bodies.emplace_back();
      // -2 requests an all-empty branch, including its optional fallback.
      if (empty_case == -2 || empty_case == static_cast<int>(i))
        continue;
      branch->bodies.back() = {
          dispatch(gated_control, 100 + static_cast<int>(i)),
          dispatch(invalid == 1 ? ordinary : forward, -1),
          dispatch(reverse, -1)};
    }
    if (invalid == 3)
      branch->bodies.front().front() = FixedExecution::BufferFill{ptr(4, 2), 4, 0};
    if (invalid == 4) {
      auto &bounded = std::get<FixedExecution::Dispatch>(branch->bodies.front().front());
      bounded.bounded_extent = ptr(4, 4);
      bounded.bounded_capacity = 1;
    }
    if (invalid == 5)
      branch->mode = static_cast<BranchExecutionMode>(99);
    Prepared result;
    if (direct) {
      auto native = std::make_shared<cuda::FixedGraph::Branch>();
      native->selector = address(4, 1);
      native->selection = selection;
      native->mode = branch->mode;
      for (const auto &item : branch->condition)
        native->condition.emplace_back(
            native_dispatch(std::get<FixedExecution::Dispatch>(item)));
      for (const auto &body : branch->bodies) {
        native->bodies.emplace_back();
        for (const auto &item : body)
          native->bodies.back().emplace_back(
              native_dispatch(std::get<FixedExecution::Dispatch>(item)));
      }
      result.direct = std::make_unique<cuda::FixedGraph>(
          std::vector<cuda::FixedGraph::Operation>{
              native_dispatch(dispatch(control, 4)), native, native,
              native_dispatch(dispatch(control, 5))});
      EXPECT_EQ(result.direct->control_bytes(), empty_case == -2 ? 0 : 8);
    } else {
      std::vector<FixedExecution::Operation> operations{
          FixedExecution::BufferFill{ptr(4, 7), 4, 77}, branch, branch,
          FixedExecution::BufferCopy{ptr(4, 8), ptr(4, 6), 4}};
      if (conditional_sibling) {
        auto sibling = std::make_shared<FixedExecution::Branch>();
        sibling->selector = ptr(4, 4);
        sibling->selection.has_default = true;
        sibling->bodies = {{FixedExecution::BufferFill{ptr(4, 7), 4, 11}},
                           {FixedExecution::BufferFill{ptr(4, 7), 4, 22}}};
        operations.emplace_back(std::move(sibling));
      }
      result.fixed = std::make_unique<FixedExecution>(Arch::cuda, device, nullptr,
                                                     operations);
      EXPECT_EQ(result.fixed->execution_kind(), 1);
    }
    std::array<std::uint32_t, 14> before;
    driver.memcpy_device_to_host(before.data(), address(4), 56);
    for (auto value : before)
      EXPECT_EQ(value, 99);
    return result;
  }
  void reset(std::uint32_t outer, std::uint32_t inner) {
    auto &driver = CUDADriver::get_instance();
    std::array<float, 4> sentinel{-99, -99, -99, -99}, one{1, 1, 1, 1};
    driver.memcpy_host_to_device(address(1), sentinel.data(), 16);
    driver.memsetd32(address(2), 0, 4);
    driver.memcpy_host_to_device(address(3), one.data(), 16);
    std::array<std::uint32_t, 14> state{};
    state[4] = outer;
    state[5] = inner;
    state[12] = inner / 2;
    driver.memcpy_host_to_device(address(4), state.data(), 56);
  }
  void check(std::uint32_t outer, std::uint32_t inner, std::uint64_t outer_budget,
             std::uint64_t inner_budget, bool conditional = false,
             bool flat = false) {
    auto &driver = CUDADriver::get_instance();
    const auto count = std::min<std::uint64_t>(outer, outer_budget);
    const auto steps = flat ? 2 * count
                           : 2 * count *
                                 (std::min<std::uint64_t>(inner, inner_budget) +
                                  std::min<std::uint64_t>(inner / 2, inner_budget));
    std::array<std::uint32_t, 14> state;
    driver.memcpy_device_to_host(state.data(), address(4), 56);
    EXPECT_EQ(state[0], count);
    EXPECT_EQ(state[6], steps);
    EXPECT_EQ(state[7], conditional ? (outer ? 11 : 22) : 77);
    EXPECT_EQ(state[8], steps);
    EXPECT_EQ(state[9], 2);
    std::array<float, 4> y, dx, dy;
    driver.memcpy_device_to_host(y.data(), address(1), 16);
    driver.memcpy_device_to_host(dx.data(), address(2), 16);
    driver.memcpy_device_to_host(dy.data(), address(3), 16);
    for (std::size_t i = 0; i < y.size(); ++i) {
      EXPECT_EQ(y[i], steps ? values[i] * values[i] : -99);
      EXPECT_EQ(dx[i], steps * 2 * values[i]);
      EXPECT_EQ(dy[i], steps ? 0 : 1);
    }
  }
  void check_branch(BranchSelection selection, std::size_t cases,
                    std::int32_t selector, int empty_case = -1,
                    bool conditional_sibling = false) {
    const auto body_count = cases + std::size_t(selection.has_default);
    const auto chosen = selection.select(selector, body_count);
    const bool accepted = chosen < body_count && empty_case != -2 &&
                          static_cast<int>(chosen) != empty_case;
    const auto hits = accepted ? 2u : 0u;
    const auto weight = accepted ? 2 * (chosen + 1) : 0;
    auto &driver = CUDADriver::get_instance();
    std::array<std::uint32_t, 14> state;
    driver.memcpy_device_to_host(state.data(), address(4), 56);
    EXPECT_EQ(state[6], weight);
    EXPECT_EQ(state[8], weight);
    EXPECT_EQ(state[9], 2);
    EXPECT_EQ(state[13], hits);
    EXPECT_EQ(state[7], conditional_sibling ? (selector ? 11 : 22) : 77);
    std::array<float, 4> y, dx, dy;
    driver.memcpy_device_to_host(y.data(), address(1), 16);
    driver.memcpy_device_to_host(dx.data(), address(2), 16);
    driver.memcpy_device_to_host(dy.data(), address(3), 16);
    for (std::size_t i = 0; i < y.size(); ++i) {
      EXPECT_EQ(y[i], hits ? values[i] * values[i] : -99);
      EXPECT_EQ(dx[i], hits * 2 * values[i]);
      EXPECT_EQ(dy[i], hits ? 0 : 1);
    }
  }
};

TEST(CudaGatedAot, FixedMaskedExecutionOwnsGatesAndPreservesActualGradients) {
  const auto *folder = std::getenv("TAICHI_AOT_FOLDER_PATH");
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
  auto &device = *executor.get_compute_device();
  const auto registration_begin = std::chrono::steady_clock::now();
  MaskedAotFrame first(device, *module, 1), second(device, *module, 2);
  const auto registration_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - registration_begin).count();
  auto &driver = CUDADriver::get_instance();
  const auto first_prepare_begin = std::chrono::steady_clock::now();
  auto sibling = second.prepare(2, 3, false);
  const auto first_prepare_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - first_prepare_begin).count();
  const auto registered = first.control->launcher().debug_registered_kernel_count();
  for (bool direct : {false, true}) {
    for (int flat : {1, 2}) {
      for (const auto budget : {0u, 3u}) {
        auto call = first.prepare(budget, 0, direct, false, 0, flat);
        for (const auto limit : {0u, 2u, 7u, 0u}) {
          first.reset(limit, 0);
          call.run();
          driver.stream_synchronize(nullptr);
          first.check(limit, 0, budget, 0, false, true);
        }
        first.reset(2, 0);
        call.run();
        call = {};
        first.check(2, 0, budget, 0, false, true);
      }
    }
    for (int invalid : {1, 2})
      EXPECT_ANY_THROW(first.prepare(2, 3, direct, false, invalid));
    for (int invalid : {1, 2})
      EXPECT_ANY_THROW(first.prepare(0, 0, direct, false, invalid, 1));
    if (!direct)
      for (int invalid : {3, 4})
        EXPECT_ANY_THROW(first.prepare(2, 3, false, false, invalid));
    for (const auto budget : {std::array<std::uint64_t, 2>{2, 3}, {0, 3}, {2, 0}}) {
      auto call = first.prepare(budget[0], budget[1], direct);
      for (const auto limits : {std::array<std::uint32_t, 2>{0, 0}, {2, 1},
                                {7, 7}, {1, 0}, {0, 2}}) {
        first.reset(limits[0], limits[1]);
        second.reset(limits[1], limits[0]);
        call.run();
        sibling.run();
        driver.stream_synchronize(nullptr);
        first.check(limits[0], limits[1], budget[0], budget[1]);
        second.check(limits[1], limits[0], 2, 3);
      }
      first.reset(2, 2);
      call.run();
      call = {};  // Retirement waits before freeing the captured gate prefix.
      first.check(2, 2, budget[0], budget[1]);
    }
  }
  const bool conditional_sibling = cuda::GraphRegionBuilder::available();
  if (conditional_sibling) {
    auto mixed = first.prepare(2, 3, false, true);
    for (const auto outer : {2u, 0u}) {
      first.reset(outer, 1);
      mixed.run();
      driver.stream_synchronize(nullptr);
      first.check(outer, 1, 2, 3, true);
    }
  }
  const auto begin = std::chrono::steady_clock::now();
  auto measured = first.prepare(2, 3, true);
  const auto prepare_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - begin).count();
  std::array<double, 16> samples;
  first.reset(2, 1);
  measured.run();
  driver.stream_synchronize(nullptr);
  for (auto &sample : samples) {
    first.reset(2, 1);
    const auto start = std::chrono::steady_clock::now();
    measured.run();
    driver.stream_synchronize(nullptr);
    sample = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - start).count();
  }
  first.check(2, 1, 2, 3);
  std::sort(samples.begin(), samples.end());
  fmt::print("FIXED_COSTS {{\"registration_and_storage_ms\":{},"
             "\"first_prepare_ms\":{},\"warm_prepare_ms\":{},"
             "\"warm_complete_median_us\":{},\"arguments_bytes\":{},"
             "\"control_bytes\":{},\"nodes\":{},"
             "\"conditional_sibling_checked\":{}}}\n",
             registration_ms, first_prepare_ms, prepare_ms,
             (samples[7] + samples[8]) / 2,
             measured.direct->argument_bytes(), measured.direct->control_bytes(),
             measured.direct->node_count(), conditional_sibling);

  using Kind = BranchSelection::Kind;
  struct BranchCase {
    BranchSelection selection;
    std::size_t cases;
    int empty_case;
  };
  const BranchSelection indexed{Kind::indexed, true};
  auto other_branch = second.prepare_branch(indexed, 3, false);
  for (const bool direct : {false, true}) {
    for (const auto &test : {
             BranchCase{{Kind::nonzero, false}, 1, -1},
             BranchCase{{Kind::nonzero, true}, 1, 1},
             BranchCase{indexed, 3, 1},
             BranchCase{{Kind::indexed, false}, 3, -1},
             BranchCase{indexed, 3, -2}}) {
      auto branch = first.prepare_branch(test.selection, test.cases, direct,
                                          test.empty_case);
      const auto changed = test.selection.kind == Kind::nonzero ? 0u : ~0u;
      for (const auto selector : {std::numeric_limits<std::int32_t>::min(), -1,
                                  0, 1, 2, 3,
                                  std::numeric_limits<std::int32_t>::max(), 0}) {
        first.reset(static_cast<std::uint32_t>(selector), changed);
        second.reset(2, ~0u);
        branch.run();
        other_branch.run();
        driver.stream_synchronize(nullptr);
        first.check_branch(test.selection, test.cases, selector, test.empty_case);
        second.check_branch(indexed, 3, 2);
      }
      first.reset(2, changed);
      branch.run();
      branch = {};
      first.check_branch(test.selection, test.cases, 2, test.empty_case);
    }
    for (const int invalid : {1, 2, 5})
      EXPECT_ANY_THROW(first.prepare_branch(indexed, 3, direct, -1, invalid));
    if (!direct)
      for (const int invalid : {3, 4})
        EXPECT_ANY_THROW(first.prepare_branch(indexed, 3, false, -1, invalid));
  }
  if (conditional_sibling) {
    auto mixed = first.prepare_branch(indexed, 3, false, -1, 0, true);
    for (const auto selector : {0, 2, -1}) {
      first.reset(static_cast<std::uint32_t>(selector), ~0u);
      mixed.run();
      driver.stream_synchronize(nullptr);
      first.check_branch(indexed, 3, selector, -1, true);
    }
  }
  {
    auto empty = std::make_shared<FixedExecution::Branch>();
    empty->selector = first.ptr(4, 1);
    empty->mode = BranchExecutionMode::cuda_masked;
    empty->bodies = {{}};
    FixedExecution no_work(Arch::cuda, device, nullptr, {empty});
    no_work.run();
    driver.stream_synchronize(nullptr);
    auto native = std::make_shared<cuda::FixedGraph::Branch>();
    native->selector = first.address(4, 1);
    native->mode = BranchExecutionMode::cuda_masked;
    native->bodies = {{}};
    cuda::FixedGraph native_no_work(
        std::vector<cuda::FixedGraph::Operation>{native});
    EXPECT_EQ(native_no_work.control_bytes(), 0);
    EXPECT_EQ(native_no_work.argument_bytes(), 0);
    native_no_work.run();
    driver.stream_synchronize(nullptr);
  }
  const auto branch_begin = std::chrono::steady_clock::now();
  auto branch = first.prepare_branch(indexed, 3, true);
  const auto branch_prepare_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - branch_begin).count();
  for (auto &sample : samples) {
    first.reset(2, ~0u);
    const auto start = std::chrono::steady_clock::now();
    branch.run();
    driver.stream_synchronize(nullptr);
    sample = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - start).count();
  }
  first.check_branch(indexed, 3, 2);
  std::sort(samples.begin(), samples.end());
  fmt::print("BRANCH_COSTS {{\"prepare_ms\":{},\"warm_complete_median_us\":{},"
             "\"arguments_bytes\":{},\"control_bytes\":{},\"nodes\":{}}}\n",
             branch_prepare_ms, (samples[7] + samples[8]) / 2,
             branch.direct->argument_bytes(), branch.direct->control_bytes(),
             branch.direct->node_count());
  EXPECT_EQ(first.control->launcher().debug_registered_kernel_count(), registered);
}

}  // namespace
}  // namespace taichi::lang
#endif
