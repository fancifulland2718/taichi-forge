#include "gtest/gtest.h"

#ifdef TI_WITH_LLVM
#include "taichi/jit/jit_module.h"
#include "taichi/runtime/llvm/llvm_runtime_executor.h"
#include "taichi/runtime/llvm/snode_runtime_state.h"
#include "taichi/runtime/llvm/sparse_runtime_view.h"
#include <array>
#include <chrono>
#include <cstring>
#include <map>
#include <set>
#include <thread>
#ifdef TI_WITH_CUDA
#include "taichi/platform/cuda/detect_cuda.h"
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/runtime/cuda/kernel_launcher.h"
#include "taichi/runtime/cuda/fixed_graph.h"
#include "taichi/runtime/cuda/sparse_runtime_view.h"
#include "taichi/runtime/cuda/execution_domain.h"
#include "taichi/runtime/cuda/binding_frame_replay.h"
#include "taichi/runtime/fixed_execution.h"
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/runtime/llvm/aot_graph_data.h"
#include "taichi/runtime/llvm/llvm_aot_module_loader.h"
#ifdef TI_WITH_VULKAN
#include "tests/cpp/rhi/common/cuda_external_gate.h"
#endif
#endif

namespace taichi::lang {

TEST(LlvmRuntimeInitialization, OrdinaryWorkspaceMatchesRuntimeAndRetires) {
  CompileConfig config;
  config.arch = host_arch();
  LlvmRuntimeExecutor executor(config, nullptr);
  EXPECT_THROW(executor.get_runtime_temporary_ptr(), std::invalid_argument);
  uint64 *result = nullptr;
  executor.materialize_runtime(nullptr, &result);
  auto *workspace = executor.get_runtime_temporary_ptr();
  ASSERT_NE(workspace, nullptr);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(workspace) % taichi_page_size, 0);
  using Getter = void *(*)(void *);
  auto getter = reinterpret_cast<Getter>(
      executor.get_runtime_jit_module()->lookup_function(
          "LLVMRuntime_get_temporaries"));
  ASSERT_NE(getter, nullptr);
  EXPECT_EQ(getter(executor.get_llvm_runtime()), workspace);
  executor.finalize();
  EXPECT_THROW(executor.get_runtime_temporary_ptr(), std::invalid_argument);
}

// Exercise the embedded runtime itself, including manager allocations and its
// private device layout. No C++ shadow of LLVMRuntime is used by the fixture.
class LlvmSparseRuntimeView : public ::testing::TestWithParam<Arch> {
 protected:
  CompileConfig config;
  std::unique_ptr<LlvmRuntimeExecutor> executor;
  JITModule *module{nullptr};
  LLVMRuntime *source{nullptr};
  uint64 *result{nullptr};
  std::vector<DeviceAllocation> allocations;
  void *directory{nullptr};
  std::array<void *, 2> states{};
  std::array<LlvmSparseRuntimeViewTree, 2> trees{{{1, 0, 71}, {3, 0, 83}}};
  LlvmSparseRuntimeViewStatus *status{nullptr};
  struct View {
    LLVMRuntime *runtime;
    void *directory;
    uint64 *result;
  };

  void SetUp() override {
    config.arch = GetParam();
#ifdef TI_WITH_CUDA
    if (config.arch == Arch::cuda && !is_cuda_api_available())
      GTEST_SKIP();
#endif
    executor = std::make_unique<LlvmRuntimeExecutor>(config, nullptr);
    executor->materialize_runtime(nullptr, &result);
    source = executor->get_llvm_runtime();
    module = executor->get_runtime_jit_module();
    status =
        static_cast<LlvmSparseRuntimeViewStatus *>(allocate(sizeof(*status)));
    directory = allocate(8 * sizeof(void *));
    module->call("runtime_set_snode_tree_directory", source, directory, 8);
    // Small explicit fixture pool also covers the legacy global cursor route.
    constexpr std::size_t pool_bytes = 4 * 1024 * 1024;
    auto *pool = allocate(pool_bytes);
    module->call("runtime_initialize_memory", source, pool_bytes, pool);
    std::size_t state_bytes = 0;
    ASSERT_TRUE(llvm_snode_tree_runtime_state_bytes(3, &state_bytes));
    for (int i = 0; i < 2; ++i) {
      states[i] = allocate(state_bytes);
      auto *root = allocate(128);
      module->call("runtime_initialize_snodes", source, std::size_t(128), 0, 3,
                   trees[i].tree_id, trees[i].generation, std::size_t(128),
                   root, states[i], state_bytes, std::size_t(2), false);
      module->call("runtime_NodeAllocator_initialize_ex", source,
                   trees[i].tree_id, 1, std::size_t(16), 4);
      module->call("runtime_initialize_snode_element_list", source,
                   trees[i].tree_id, 1, std::size_t(2));
      if (i == 1) {
        constexpr std::size_t dedicated_bytes = 64 * 1024;
        auto *dedicated = allocate(dedicated_bytes);
        module->call("runtime_NodeAllocator_set_dedicated_pool", source,
                     trees[i].tree_id, 1, dedicated, dedicated_bytes);
      }
    }
    executor->synchronize();
    executor->check_runtime_error(result);
  }

  void TearDown() override {
    if (!executor)
      return;
    executor->synchronize();
    // The test registers raw runtime states, not executor-owned SNodeTrees.
    // All views must have been returned before these fixture allocations.
    for (auto i = allocations.rbegin(); i != allocations.rend(); ++i)
      executor->deallocate_memory_on_device(*i);
    executor->finalize();
  }

  void *allocate(std::size_t bytes) {
    auto allocation = executor->allocate_memory_on_device(bytes, result);
    allocations.push_back(allocation);
    return executor->get_device_alloc_info_ptr(allocation);
  }
  void write(void *dst, const void *src, std::size_t bytes) {
#ifdef TI_WITH_CUDA
    if (config.arch == Arch::cuda) {
      auto guard = CUDAContext::get_instance().get_guard();
      CUDADriver::get_instance().memcpy_host_to_device(
          dst, const_cast<void *>(src), bytes);
      return;
    }
#endif
    std::memcpy(dst, src, bytes);
  }
  void read(void *dst, const void *src, std::size_t bytes) {
    executor->synchronize();
#ifdef TI_WITH_CUDA
    if (config.arch == Arch::cuda) {
      auto guard = CUDAContext::get_instance().get_guard();
      CUDADriver::get_instance().memcpy_device_to_host(
          dst, const_cast<void *>(src), bytes);
      return;
    }
#endif
    std::memcpy(dst, src, bytes);
  }
  View make_view() {
    auto *size_result = static_cast<uint64 *>(allocate(sizeof(uint64)));
    module->call("runtime_sparse_view_get_size", size_result);
    uint64 bytes = 0;
    read(&bytes, size_result, sizeof(bytes));
    EXPECT_GT(bytes, 0u);
    RecordProperty("runtime_view_bytes", std::to_string(bytes));
    View view{static_cast<LLVMRuntime *>(allocate(bytes)),
              allocate(4 * sizeof(void *)),
              static_cast<uint64 *>(
                  allocate(taichi_result_buffer_entries * sizeof(uint64)))};
    auto *temporary = allocate(taichi_global_tmp_buffer_size);
    module->call("runtime_sparse_view_initialize", source, view.runtime,
                 view.directory, 4, temporary, view.result, status);
    EXPECT_EQ(read_status().error, LlvmSparseRuntimeViewError::success);
    return view;
  }
  LlvmSparseRuntimeViewStatus read_status() {
    LlvmSparseRuntimeViewStatus value;
    read(&value, status, sizeof(value));
    return value;
  }
  LlvmSparseRuntimeViewStatus transfer(
      const View &view,
      bool release,
      const std::vector<LlvmSparseRuntimeViewTree> &requests) {
    auto *input =
        allocate(std::max(std::size_t(1), requests.size()) * sizeof(trees[0]));
    if (!requests.empty())
      write(input, requests.data(), requests.size() * sizeof(trees[0]));
    module->call(
        release ? "runtime_sparse_view_release" : "runtime_sparse_view_bind",
        view.runtime, input, int(requests.size()), status);
    return read_status();
  }
  std::array<void *, 4> entries(const View &view) {
    std::array<void *, 4> value{};
    read(value.data(), view.directory, sizeof(value));
    return value;
  }
  LlvmSparseRuntimeViewStatus transfer_mapped(
      const View &view,
      bool release,
      const std::vector<LlvmSparseRuntimeViewBinding> &requests) {
    auto *input = allocate(std::max(std::size_t(1), requests.size()) *
                           sizeof(LlvmSparseRuntimeViewBinding));
    if (!requests.empty())
      write(input, requests.data(),
            requests.size() * sizeof(LlvmSparseRuntimeViewBinding));
    module->call(release ? "runtime_sparse_view_release_mapped"
                         : "runtime_sparse_view_bind_mapped",
                 view.runtime, input, int(requests.size()), status);
    return read_status();
  }
};

TEST_P(LlvmSparseRuntimeView, MapsPrivateCodeSlotsWithoutChangingSourceDirectory) {
  auto first = make_view();
  auto second = make_view();
  const LlvmSparseRuntimeViewBinding a{0, 1, 71, 0};
  const LlvmSparseRuntimeViewBinding b{0, 3, 83, 0};
  auto stale = b;
  ++stale.generation;
  EXPECT_EQ(transfer_mapped(first, false, {stale}).error,
            LlvmSparseRuntimeViewError::stale_generation);
  auto duplicate = a;
  duplicate.tree_id = 2;
  EXPECT_EQ(transfer_mapped(first, false, {a, duplicate}).error,
            LlvmSparseRuntimeViewError::invalid_request);
  EXPECT_EQ(entries(first), (std::array<void *, 4>{}));
  EXPECT_EQ(transfer_mapped(first, false, {a}).error,
            LlvmSparseRuntimeViewError::success);
  EXPECT_EQ(transfer_mapped(second, false, {b}).error,
            LlvmSparseRuntimeViewError::success);
  EXPECT_EQ(entries(first)[0], states[0]);
  EXPECT_EQ(entries(second)[0], states[1]);
  std::array<void *, 8> original{};
  read(original.data(), directory, sizeof(original));
  EXPECT_EQ(original[1], states[0]);
  EXPECT_EQ(original[3], states[1]);
  EXPECT_EQ(original[0], nullptr);
  EXPECT_EQ(transfer_mapped(first, true, {b}).error,
            LlvmSparseRuntimeViewError::stale_generation);
  EXPECT_EQ(entries(first)[0], states[0]);
  EXPECT_EQ(transfer_mapped(first, true, {a}).error,
            LlvmSparseRuntimeViewError::success);
  EXPECT_EQ(entries(second)[0], states[1]);
  EXPECT_EQ(transfer_mapped(second, true, {b}).error,
            LlvmSparseRuntimeViewError::success);
}

TEST_P(LlvmSparseRuntimeView, PreflightsEveryTreeBeforeBindingOrReturning) {
  auto first = make_view();
  auto second = make_view();
  auto stale = trees[1];
  ++stale.generation;
  EXPECT_EQ(transfer(first, false, {trees[0], stale}).error,
            LlvmSparseRuntimeViewError::stale_generation);
  EXPECT_EQ(entries(first), (std::array<void *, 4>{}));
  EXPECT_EQ(transfer(second, false, {trees[0]}).error,
            LlvmSparseRuntimeViewError::success);
  EXPECT_EQ(transfer(first, false, {trees[0], trees[1]}).error,
            LlvmSparseRuntimeViewError::foreign_manager);
  EXPECT_EQ(transfer(first, false, {trees[1], trees[1]}).error,
            LlvmSparseRuntimeViewError::invalid_request);
  EXPECT_EQ(transfer(first, false, {trees[1]}).error,
            LlvmSparseRuntimeViewError::success);
  EXPECT_EQ(transfer(first, true, {stale}).error,
            LlvmSparseRuntimeViewError::stale_generation);
  EXPECT_EQ(entries(first)[3], states[1]);
  EXPECT_EQ(transfer(first, true, {trees[1]}).error,
            LlvmSparseRuntimeViewError::success);
  EXPECT_EQ(transfer(second, true, {trees[0]}).error,
            LlvmSparseRuntimeViewError::success);
  auto whole = make_view();
  EXPECT_EQ(transfer(whole, false, {trees[0], trees[1]}).error,
            LlvmSparseRuntimeViewError::success);
  EXPECT_EQ(transfer(whole, true, {trees[0], stale}).error,
            LlvmSparseRuntimeViewError::stale_generation);
  EXPECT_EQ(entries(whole)[1], states[0]);
  EXPECT_EQ(entries(whole)[3], states[1]);
  EXPECT_EQ(transfer(whole, true, {trees[0], trees[1]}).error,
            LlvmSparseRuntimeViewError::success);
  auto unpublished = make_view();
  EXPECT_EQ(transfer(unpublished, true, {}).error,
            LlvmSparseRuntimeViewError::success);
}

TEST_P(LlvmSparseRuntimeView,
       KeepsDirectoryAndErrorsPrivateAcrossSourceGrowth) {
  auto first = make_view();
  auto second = make_view();
  EXPECT_EQ(transfer(first, false, {trees[0]}).error,
            LlvmSparseRuntimeViewError::success);
  EXPECT_EQ(transfer(second, false, {trees[1]}).error,
            LlvmSparseRuntimeViewError::success);
  std::array<void *, 16> expanded{};
  expanded[1] = states[0];
  expanded[3] = states[1];
  auto *replacement = allocate(sizeof(expanded));
  write(replacement, expanded.data(), sizeof(expanded));
  module->call("runtime_set_snode_tree_directory", source, replacement, 16);
  EXPECT_EQ(entries(first)[1], states[0]);
  EXPECT_EQ(entries(first)[3], nullptr);
  EXPECT_EQ(entries(second)[3], states[1]);
  // An existing runtime assertion writes only this view's error buffer. CPU
  // asserts return normally, so restore the deliberately invalid capacity.
  module->call("runtime_set_snode_tree_directory", first.runtime,
               first.directory, -1);
  module->call("runtime_set_snode_tree_directory", first.runtime,
               first.directory, 4);
  module->call("runtime_retrieve_and_reset_error_code", first.runtime);
  uint64 code = 0;
  read(&code, first.result + taichi_result_buffer_error_id, sizeof(code));
  EXPECT_EQ(code, 1u);
  module->call("runtime_retrieve_and_reset_error_code", second.runtime);
  read(&code, second.result + taichi_result_buffer_error_id, sizeof(code));
  EXPECT_EQ(code, 0u);
  executor->check_runtime_error(result);
  // Managers created through the view use the original pool and are returned
  // with the same state, rather than losing changes in a copied tree header.
  module->call("runtime_LLVMRuntime_get_total_requested_memory", source,
               source);
  uint64 before = 0, after = 0;
  read(&before, result + taichi_result_buffer_runtime_query_id, sizeof(before));
  module->call("runtime_initialize_snode_element_list", first.runtime, 1, 2,
               std::size_t(2));
  module->call("runtime_LLVMRuntime_get_total_requested_memory", source,
               source);
  read(&after, result + taichi_result_buffer_runtime_query_id, sizeof(after));
  EXPECT_GT(after, before);
  EXPECT_EQ(transfer(first, true, {trees[0]}).error,
            LlvmSparseRuntimeViewError::success);
  EXPECT_EQ(transfer(second, true, {trees[1]}).error,
            LlvmSparseRuntimeViewError::success);
  auto again = make_view();
  EXPECT_EQ(transfer(again, false, {trees[0], trees[1]}).error,
            LlvmSparseRuntimeViewError::success);
  EXPECT_EQ(transfer(again, true, {trees[0], trees[1]}).error,
            LlvmSparseRuntimeViewError::success);
}

TEST_P(LlvmSparseRuntimeView,
       SharesTheAllocationCursorAndLockWithOrdinaryWork) {
  auto first = make_view();
  auto second = make_view();
  EXPECT_EQ(transfer(first, false, {trees[0]}).error,
            LlvmSparseRuntimeViewError::success);
  EXPECT_EQ(transfer(second, false, {trees[1]}).error,
            LlvmSparseRuntimeViewError::success);
  constexpr int count = 32;
  auto *outputs = static_cast<uint64 *>(allocate(3 * count * sizeof(uint64)));
  std::array<LLVMRuntime *, 3> runtimes{source, first.runtime, second.runtime};
  std::size_t bytes = 64, alignment = 64;
#ifdef TI_WITH_CUDA
  if (config.arch == Arch::cuda) {
    auto guard = CUDAContext::get_instance().get_guard();
    auto &driver = CUDADriver::get_instance();
    auto *function = module->lookup_function("runtime_memory_allocate_aligned");
    std::array<void *, 3> streams{};
    for (auto &stream : streams)
      driver.stream_create(&stream, CU_STREAM_NON_BLOCKING);
    for (int i = 0; i < count; ++i) {
      for (int j = 0; j < 3; ++j) {
        auto *output = outputs + j * count + i;
        void *args[]{&runtimes[j], &bytes, &alignment, &output};
        driver.launch_kernel(function, 1, 1, 1, 1, 1, 1, 0, streams[j], args,
                             nullptr);
      }
    }
    for (auto stream : streams) {
      driver.stream_synchronize(stream);
      driver.stream_destroy(stream);
    }
  } else
#endif
  {
    auto allocate_fn =
        module->get_function<LLVMRuntime *, std::size_t, std::size_t, uint64 *>(
            "runtime_memory_allocate_aligned");
    std::array<std::thread, 3> workers;
    for (int j = 0; j < 3; ++j)
      workers[j] = std::thread([&, j] {
        for (int i = 0; i < count; ++i)
          allocate_fn(runtimes[j], bytes, alignment, outputs + j * count + i);
      });
    for (auto &worker : workers)
      worker.join();
  }
  std::array<uint64, 3 * count> values{};
  read(values.data(), outputs, sizeof(values));
  std::set<uint64> unique(values.begin(), values.end());
  EXPECT_EQ(unique.size(), values.size());
  for (auto address : unique) {
    EXPECT_NE(address, 0u);
    EXPECT_EQ(address % alignment, 0u);
  }
  EXPECT_EQ(transfer(first, true, {trees[0]}).error,
            LlvmSparseRuntimeViewError::success);
  EXPECT_EQ(transfer(second, true, {trees[1]}).error,
            LlvmSparseRuntimeViewError::success);
  executor->check_runtime_error(result);
}

#ifdef TI_WITH_CUDA
TEST(CudaSparseAccess, SameLoadedCodeUsesIndependentClonedTreeStates) {
  const auto *folder = std::getenv("TAICHI_CUDA_SPARSE_ACCESS_FIXTURE");
  if (!folder || !cuda::FixedGraph::available())
    GTEST_SKIP() << "Requires the CUDA sparse access AOT fixture";
  CompileConfig config;
  config.arch = Arch::cuda;
  LlvmRuntimeExecutor executor(config, nullptr);
  uint64 *result = nullptr;
  executor.materialize_runtime(nullptr, &result);
  LLVM::AotModuleParams params;
  params.module_path = folder;
  params.executor_ = &executor;
  params.allocation_result_buffer = result;
  params.kernel_launcher = std::make_unique<cuda::KernelLauncher>(
      cuda::KernelLauncher::Config{&executor});
  auto module = LLVM::make_aot_module(std::move(params));
  auto kernel = [&](const char *name) {
    return dynamic_cast<llvm_aot::KernelImpl *>(module->get_kernel(name));
  };
  auto *initialize = kernel("initialize");
  LaunchContextBuilder initial(initialize);
  initialize->launch(initial);
  std::map<std::string, llvm_aot::KernelImpl *> code;
  code.emplace("initialize", initialize);
  for (const auto *name : {"seed_left", "bump_left", "read_left", "deactivate_left"})
    code.emplace(name, kernel(name));
  auto layout = [&](const char *name) {
    return dynamic_cast<llvm_aot::FieldImpl &>(
               *static_cast<LLVM::LlvmAotModule &>(*module).get_snode_tree(name))
        .get_snode_tree_cache();
  };
  const auto original = layout("0"), unrelated = layout("1");
  const LlvmSparseRuntimeViewTree prototype{
      original.tree_id, 0, original.tree_generation};
  auto ids = executor.reserve_aot_snode_tree_ids(2);
  auto a = executor.clone_snode_tree_runtime(prototype, ids[0], result);
  auto b = executor.clone_snode_tree_runtime(prototype, ids[1], result);
  auto binding = [&](const auto &state) {
    return LlvmSparseRuntimeViewBinding{original.tree_id, state.tree_id,
                                       state.tree_generation,
                                       original.tree_generation};
  };
  // Equal-looking independent trees are not proof of code/layout compatibility.
  EXPECT_THROW(executor.validate_snode_tree_instance_bindings({binding(unrelated)}),
               std::invalid_argument);
  auto stale = binding(a);
  ++stale.code_generation;
  EXPECT_THROW(executor.validate_snode_tree_instance_bindings({stale}),
               std::invalid_argument);
  auto owner = std::make_shared<RuntimeFaultDomain>(Arch::cuda, 7602);
  cuda::SparseRuntimeView left, right;
  left.initialize_mapped(executor, {binding(a)}, owner);
  right.initialize_mapped(executor, {binding(b)}, owner);
  EXPECT_THROW(executor.destroy_snode_tree_runtime(a), std::invalid_argument);
  auto &launcher = dynamic_cast<cuda::KernelLauncher &>(code.at("bump_left")->launcher());
  auto guard = CUDAContext::get_instance().get_guard();
  auto &driver = CUDADriver::get_instance();
  std::array<void *, 2> output{}, streams{};
  for (int i = 0; i < 2; ++i) {
    driver.malloc(&output[i], 16 * sizeof(int));
    driver.stream_create(&streams[i], CU_STREAM_NON_BLOCKING);
  }
  struct Captured {
    std::vector<cuda::KernelLauncher::GraphLaunchPacket> packets;
    void *graph{nullptr}, *executable{nullptr};
    ~Captured() {
      auto context = CUDAContext::get_instance().get_guard();
      auto &driver = CUDADriver::get_instance();
      if (executable) driver.graph_exec_destroy(executable);
      if (graph) driver.graph_destroy(graph);
      for (const auto &packet : packets)
        if (packet.device_arg_buffer) driver.mem_free(packet.device_arg_buffer);
    }
  };
  auto capture = [&](cuda::SparseRuntimeView &view, int index,
                     const std::vector<std::string> &names) {
    auto captured = std::make_unique<Captured>();
    captured->packets.resize(names.size());
    for (std::size_t i = 0; i < names.size(); ++i) {
      auto *entry = code.at(names[i]);
      LaunchContextBuilder args(entry);
      args.discard_return_values();
      if (names[i] == "seed_left") args.set_arg_int({0}, index ? 100 : 10);
      if (names[i] == "read_left")
        args.set_arg_external_array_with_shape(
            {0}, reinterpret_cast<std::uintptr_t>(output[index]),
            16 * sizeof(int), {16});
      EXPECT_TRUE(launcher.prepare_cuda_graph_launch_packet(
          entry->handle(), args, captured->packets[i], true));
      launcher.bind_sparse_runtime_view(captured->packets[i], view);
      auto &packet = captured->packets[i];
      if (names[i] != "initialize")
        EXPECT_EQ(packet.context.result_buffer,
                  executor.get_snode_tree_root_ptr(index ? b.tree_id : a.tree_id));
      // Match ordinary FixedGraph cold upload. A launch packet by itself does
      // not own the temporary LaunchContextBuilder's host argument buffer.
      if (packet.arg_buffer_size) {
        driver.malloc(&packet.device_arg_buffer, packet.arg_buffer_size);
        driver.memcpy_host_to_device(packet.device_arg_buffer,
                                     packet.context.arg_buffer,
                                     packet.arg_buffer_size);
        packet.context.arg_buffer = static_cast<char *>(packet.device_arg_buffer);
      }
    }
    {
      auto lock = CUDAContext::get_instance().get_graph_capture_lock_guard();
      driver.stream_begin_capture(streams[index], CU_STREAM_CAPTURE_MODE_THREAD_LOCAL);
      for (auto &packet : captured->packets)
        launcher.capture_cuda_graph_launch(packet, streams[index]);
      driver.stream_end_capture(streams[index], &captured->graph);
    }
    driver.graph_instantiate_with_flags(&captured->executable, captured->graph, 0);
    return captured;
  };
  auto read_a = capture(left, 0, {"read_left"});
  auto read_b = capture(right, 1, {"read_left"});
  auto seed_a = capture(left, 0, {"seed_left", "bump_left", "read_left"});
  auto seed_b = capture(right, 1, {"seed_left", "bump_left", "read_left"});
  auto step_a = capture(left, 0, {"bump_left", "read_left"});
  auto step_b = capture(right, 1, {"bump_left", "read_left"});
  auto clear_a = capture(left, 0, {"deactivate_left", "bump_left", "read_left"});
  auto submit = [&](const auto &captured, int index) {
    driver.graph_launch(captured->executable, streams[index]);
  };
  auto values = [&](int index) {
    EXPECT_EQ(driver.stream_synchronize.call_concurrent(streams[index]), CUDA_SUCCESS);
    std::array<int, 16> data{};
    driver.memcpy_device_to_host(data.data(), output[index], sizeof(data));
    return data;
  };
  submit(read_a, 0);
  submit(read_b, 1);
  EXPECT_EQ(values(0), (std::array<int, 16>{}));
  EXPECT_EQ(values(1), (std::array<int, 16>{}));
  submit(seed_a, 0);
  submit(seed_b, 1);
  submit(step_a, 0);
  auto av = values(0), bv = values(1);
  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(av[i], i < 5 ? 12 + i : i < 8 ? 2 : 0);
    EXPECT_EQ(bv[i], i < 5 ? 101 + i : i < 8 ? 1 : 0);
  }
  submit(clear_a, 0);
  submit(step_b, 1);
  av = values(0);
  bv = values(1);
  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(av[i], i < 4 ? 0 : i == 4 ? 17 : i < 8 ? 3 : 0);
    EXPECT_EQ(bv[i], i < 5 ? 102 + i : i < 8 ? 2 : 0);
  }
  read_a.reset(); seed_a.reset(); step_a.reset(); clear_a.reset();
  left.retire();
  executor.destroy_snode_tree_runtime(a);
  submit(step_b, 1);
  EXPECT_EQ(values(1)[0], 103);
  EXPECT_THROW(executor.validate_snode_tree_instance_bindings({binding(a)}),
               std::invalid_argument);
  read_b.reset(); seed_b.reset(); step_b.reset();
  right.retire();
  executor.destroy_snode_tree_runtime(b);
  auto reused = executor.clone_snode_tree_runtime(prototype, ids[0], result);
  EXPECT_NE(reused.tree_generation, a.tree_generation);
  EXPECT_THROW(executor.validate_snode_tree_instance_bindings({binding(a)}),
               std::invalid_argument);
  EXPECT_NO_THROW(executor.validate_snode_tree_instance_bindings({binding(reused)}));
  const LlvmSparseRuntimeViewTree other_prototype{
      unrelated.tree_id, 0, unrelated.tree_generation};
  auto other = executor.clone_snode_tree_runtime(other_prototype, ids[1], result);
  cuda::SparseRuntimeView pair;
  pair.initialize_mapped(executor,
      {binding(reused), {unrelated.tree_id, other.tree_id,
                         other.tree_generation, unrelated.tree_generation}}, owner);
  const std::vector<int> pair_ids{original.tree_id, unrelated.tree_id};
  auto *root_table = pair.root_binding(pair_ids);
  EXPECT_EQ(root_table, pair.root_binding(pair_ids));
  std::array<void *, 2> roots{};
  driver.memcpy_device_to_host(roots.data(), root_table, sizeof(roots));
  EXPECT_EQ(roots[0], executor.get_snode_tree_root_ptr(reused.tree_id));
  EXPECT_EQ(roots[1], executor.get_snode_tree_root_ptr(other.tree_id));
  auto paired = capture(pair, 0, {"initialize", "bump_left", "read_left"});
  submit(paired, 0);
  av = values(0);
  for (int i = 0; i < 16; ++i) EXPECT_EQ(av[i], i + 4);
  paired.reset();
  pair.retire();
  executor.destroy_snode_tree_runtime(other);
  executor.destroy_snode_tree_runtime(reused);
  executor.release_aot_snode_tree_ids(ids);
  for (int i = 0; i < 2; ++i) {
    driver.stream_destroy(streams[i]);
    driver.mem_free(output[i]);
  }
  auto *peek = kernel("peek_left");
  LaunchContextBuilder original_values(peek);
  peek->launch(original_values);
  EXPECT_EQ(original_values.get_ret<int>(0), 3);
  EXPECT_EQ(kernel("bump_left"), code.at("bump_left"));
  module.reset();
  executor.finalize();
}

TEST(CudaSparseAccess, RuntimeViewsCaptureRealKernelsAndReturnManagers) {
  const auto *folder = std::getenv("TAICHI_CUDA_SPARSE_ACCESS_FIXTURE");
  if (!folder || !cuda::FixedGraph::available())
    GTEST_SKIP() << "Requires the CUDA sparse access AOT fixture";
  CompileConfig config;
  config.arch = Arch::cuda;
  LlvmRuntimeExecutor executor(config, nullptr);
  uint64 *result = nullptr;
  executor.materialize_runtime(nullptr, &result);
  LLVM::AotModuleParams params;
  params.module_path = folder;
  params.executor_ = &executor;
  params.allocation_result_buffer = result;
  params.kernel_launcher = std::make_unique<cuda::KernelLauncher>(
      cuda::KernelLauncher::Config{&executor});
  auto module = LLVM::make_aot_module(std::move(params));
  auto invoke = [&](const char *name, bool returns = false) {
    auto *kernel = module->get_kernel(name);
    LaunchContextBuilder args(kernel);
    kernel->launch(args);
    return returns ? args.get_ret<int>(0) : 0;
  };
  invoke("initialize");
  auto tree = [&](const char *name) {
    const auto &layout =
        dynamic_cast<llvm_aot::FieldImpl &>(
            *static_cast<LLVM::LlvmAotModule &>(*module).get_snode_tree(name))
            .get_snode_tree_cache();
    return LlvmSparseRuntimeViewTree{layout.tree_id, 0, layout.tree_generation};
  };
  auto owner = std::make_shared<RuntimeFaultDomain>(Arch::cuda, 7601);
  cuda::SparseRuntimeView left, right;
  left.initialize(executor, {tree("0")}, owner);
  right.initialize(executor, {tree("1")}, owner);
  EXPECT_NE(left.temporaries(), right.temporaries());
  EXPECT_GT(left.bytes(), taichi_global_tmp_buffer_size);
  RecordProperty("view_storage_bytes", std::to_string(left.bytes()));
  EXPECT_THROW(invoke("bump_left"), std::invalid_argument);
  EXPECT_THROW(invoke("bump_right"), std::invalid_argument);
  cuda::SparseRuntimeView conflict;
  EXPECT_THROW(conflict.initialize(executor, {tree("0")}, owner),
               std::invalid_argument);
  conflict.retire();
  auto stale = tree("1");
  ++stale.generation;
  cuda::SparseRuntimeView rejected;
  EXPECT_THROW(rejected.initialize(executor, {stale}, owner),
               std::invalid_argument);
  rejected.retire();

  struct Captured {
    cuda::KernelLauncher::GraphLaunchPacket packet;
    void *graph{nullptr};
    void *executable{nullptr};
    ~Captured() {
      auto guard = CUDAContext::get_instance().get_guard();
      auto &driver = CUDADriver::get_instance();
      if (executable)
        driver.graph_exec_destroy(executable);
      if (graph)
        driver.graph_destroy(graph);
    }
  };
  auto capture = [&](const char *name, cuda::SparseRuntimeView &view) {
    auto captured = std::make_unique<Captured>();
    auto *kernel =
        dynamic_cast<llvm_aot::KernelImpl *>(module->get_kernel(name));
    auto &launcher = dynamic_cast<cuda::KernelLauncher &>(kernel->launcher());
    LaunchContextBuilder args(kernel);
    args.discard_return_values();
    EXPECT_TRUE(launcher.prepare_cuda_graph_launch_packet(
        kernel->handle(), args, captured->packet, true));
    EXPECT_EQ(captured->packet.arg_buffer_size, 0u);
    auto *source = captured->packet.context.runtime;
    launcher.bind_sparse_runtime_view(captured->packet, view);
    EXPECT_EQ(args.get_context().runtime, source);
    EXPECT_NE(captured->packet.context.runtime, source);
    auto guard = CUDAContext::get_instance().get_guard();
    auto &driver = CUDADriver::get_instance();
    void *stream = nullptr;
    driver.stream_create(&stream, CU_STREAM_NON_BLOCKING);
    {
      auto lock = CUDAContext::get_instance().get_graph_capture_lock_guard();
      driver.stream_begin_capture(stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL);
      launcher.capture_cuda_graph_launch(captured->packet, stream);
      driver.stream_end_capture(stream, &captured->graph);
    }
    driver.graph_instantiate_with_flags(&captured->executable, captured->graph,
                                        0);
    // Match FixedGraph's independent-instance cold upload policy.
    driver.graph_upload(captured->executable, stream);
    EXPECT_EQ(driver.stream_synchronize.call_concurrent(stream), CUDA_SUCCESS);
    driver.stream_destroy(stream);
    return captured;
  };
  auto a = capture("bump_left", left);
  auto b = capture("bump_right", right);
  auto failure = capture("fail_left", left);
  auto *kernel =
      dynamic_cast<llvm_aot::KernelImpl *>(module->get_kernel("bump_right"));
  auto &launcher = dynamic_cast<cuda::KernelLauncher &>(kernel->launcher());
  auto wrong = b->packet;
  wrong.context.runtime = executor.get_llvm_runtime();
  EXPECT_THROW(launcher.bind_sparse_runtime_view(wrong, left),
               std::invalid_argument);
  wrong = {};
  cuda::ExecutionDomain::Options options;
  options.capacity = 2;
  options.gpu_timing = true;
  auto da = std::make_shared<cuda::ExecutionDomain>(owner, options);
  auto db = std::make_shared<cuda::ExecutionDomain>(owner, options);
  auto good = left.prepare_diagnostic_snapshot();
  auto bad = left.prepare_diagnostic_snapshot();
  auto peer = right.prepare_diagnostic_snapshot();
  struct RecordedOwner final : RuntimeClient {
    std::function<void()> release;
    void retire() override {
      release();
    }
  };
  auto left_owner = std::make_shared<RecordedOwner>();
  left_owner->release = [&] {
    failure.reset();
    a.reset();
    left.retire();
  };
  auto right_owner = std::make_shared<RecordedOwner>();
  right_owner->release = [&] {
    b.reset();
    right.retire();
  };
  std::array<std::shared_ptr<cuda::SparseRuntimeView::DiagnosticSnapshot>, 2>
      left_snapshots{bad, good};
  std::size_t collections = 0;
  auto left_instance = da->prepare_owned_instance(
      left_owner, [](const auto &) {},
      [&](void *stream, std::size_t slot) {
        ++collections;
        left.enqueue_diagnostics(*left_snapshots.at(slot), stream);
        return left_snapshots.at(slot);
      });
  auto right_instance = db->prepare_owned_instance(
      right_owner, [](const auto &) {},
      [&](void *stream, std::size_t) {
        right.enqueue_diagnostics(*peer, stream);
        return peer;
      });
  void *peer_graph_done = nullptr;
  {
    auto guard = CUDAContext::get_instance().get_guard();
    CUDADriver::get_instance().event_create(&peer_graph_done, CU_EVENT_DISABLE_TIMING);
  }
  RecordProperty("view_with_two_diagnostics_bytes",
                 std::to_string(left.bytes()));
  auto expect_diagnostic = [&] {
    try {
      bad->observe();
      ADD_FAILURE() << "Expected the private runtime assertion";
    } catch (const TaichiAssertionError &error) {
      EXPECT_NE(std::string(error.what()).find("left sparse diagnostic"),
                std::string::npos);
    }
  };
  int submission_index = 0;
  auto submit = [&](auto &domain, const auto &instance, Captured &graph) {
    const auto index = std::to_string(submission_index++);
    return domain->submit_instance(
        instance, {}, {},
        [&](void *stream) {
          const auto begin = std::chrono::steady_clock::now();
          CUDADriver::get_instance().graph_launch(graph.executable, stream);
          if (index == "1")
            CUDADriver::get_instance().event_record(peer_graph_done, stream);
          const auto launched = std::chrono::steady_clock::now();
          RecordProperty(
              "graph_enqueue_us_" + index,
              std::to_string(
                  std::chrono::duration<double, std::micro>(launched - begin)
                      .count()));
        },
        {}, {}, {}, true);
  };
#ifdef TI_WITH_VULKAN
  testing::CudaExternalGate gate;
  auto occupied =
      da->submit({}, {}, [&](void *stream) { gate.enqueue_wait(stream); });
#endif
  auto ca = submit(da, left_instance, *a);
#ifdef TI_WITH_VULKAN
  const auto collected_before_busy = collections;
  EXPECT_THROW(submit(da, left_instance, *a), cuda::ExecutionDomainBusy);
  EXPECT_EQ(collections, collected_before_busy);
  --submission_index;  // The rejected submission did not record the peer event.
#endif
  auto cb = submit(db, right_instance, *b);
  const auto peer_begin = std::chrono::steady_clock::now();
  {
    auto guard = CUDAContext::get_instance().get_guard();
    auto &driver = CUDADriver::get_instance();
    EXPECT_EQ(driver.event_synchronize.call_concurrent(peer_graph_done), CUDA_SUCCESS);
    driver.event_destroy(peer_graph_done);
  }
  RecordProperty("peer_graph_wait_us", std::to_string(
      std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - peer_begin).count()));
#ifdef TI_WITH_VULKAN
  RecordProperty("gate_pending_after_peer_graph", gate.producer_pending() ? "true" : "false");
#endif
  cb.wait();
#ifdef TI_WITH_VULKAN
  EXPECT_FALSE(ca.done());
  EXPECT_TRUE(gate.producer_pending());
  EXPECT_FALSE(gate.timed_out());
  gate.release();
  occupied.wait();
#endif
  ca.wait();
  EXPECT_NO_THROW(good->observe());
  EXPECT_NO_THROW(peer->observe());
  auto failed = submit(da, left_instance, *failure);
  EXPECT_THROW(failed.wait(), TaichiAssertionError);
  EXPECT_TRUE(failed.result_observation_failed());
  EXPECT_TRUE(failed.gpu_timing_snapshot().available);
  expect_diagnostic();
  auto recovered = submit(da, left_instance, *a);
  recovered.wait();
  EXPECT_NO_THROW(good->observe());
  expect_diagnostic();
  // Reuse the failed ticket's actual mapped slot. Its original exception must
  // remain immutable even though the snapshot now contains a successful result.
  auto reused = submit(da, left_instance, *a);
  reused.wait();
  EXPECT_NO_THROW(bad->observe());
  EXPECT_THROW(failed.wait(), TaichiAssertionError);
  EXPECT_TRUE(da->snapshot().accepting);
  // The peer's diagnostics and the default runtime are untouched.
  EXPECT_NO_THROW(peer->observe());
  executor.check_runtime_error(result);
  // Retirement observes an unpolled failing result, but still releases the
  // proven-complete owner and its instance budget before returning that error.
  auto last_failure = submit(da, left_instance, *failure);
  EXPECT_THROW(da->retire_instance(left_instance), TaichiAssertionError);
  EXPECT_TRUE(da->instance_retired(left_instance));
  db->retire_instance(right_instance);
  EXPECT_TRUE(db->instance_retired(right_instance));
  EXPECT_THROW(left.bind(executor.get_llvm_runtime(), {tree("0").tree_id}),
               std::invalid_argument);
  EXPECT_THROW(left.initialize(executor, {tree("0")}, owner),
               std::invalid_argument);
  EXPECT_THROW(failed.wait(), TaichiAssertionError);
  EXPECT_THROW(last_failure.wait(), TaichiAssertionError);
  EXPECT_EQ(invoke("peek_left", true), 6);
  EXPECT_EQ(invoke("peek_right", true), 8);
  invoke("bump_left");
  EXPECT_EQ(invoke("peek_left", true), 7);
  da->close();
  db->close();
  module.reset();
  executor.finalize();
}

TEST(CudaSparseAccess, RealKernelClosureRejectsConflictsAndRetires) {
  const auto *folder = std::getenv("TAICHI_CUDA_SPARSE_ACCESS_FIXTURE");
  if (!folder || !is_cuda_api_available())
    GTEST_SKIP() << "Requires the CUDA sparse access AOT fixture";
  CompileConfig config;
  config.arch = Arch::cuda;
  LlvmRuntimeExecutor executor(config, nullptr);
  uint64 *result = nullptr;
  executor.materialize_runtime(nullptr, &result);
  LLVM::AotModuleParams params;
  params.module_path = folder;
  params.executor_ = &executor;
  params.allocation_result_buffer = result;
  params.kernel_launcher = std::make_unique<cuda::KernelLauncher>(
      cuda::KernelLauncher::Config{&executor});
  auto module = LLVM::make_aot_module(std::move(params));
  auto invoke = [&](const char *name, bool returns = false) {
    auto *kernel = module->get_kernel(name);
    LaunchContextBuilder args(kernel);
    kernel->launch(args);
    return returns ? args.get_ret<int>(0) : 0;
  };
  invoke("initialize");
  auto layout = [&](const char *name) {
    return dynamic_cast<llvm_aot::FieldImpl &>(
               *static_cast<LLVM::LlvmAotModule &>(*module).get_snode_tree(name))
        .get_snode_tree_cache();
  };
  const auto left = layout("0"), right = layout("1");
  const LlvmSparseRuntimeViewTree a{left.tree_id, 0, left.tree_generation};
  const LlvmSparseRuntimeViewTree b{right.tree_id, 0, right.tree_generation};
  ASSERT_LT(a.tree_id, b.tree_id);
  auto permit = executor.prepare_cuda_sparse_access({a.tree_id, a.tree_id});
  ASSERT_NE(permit, nullptr);
  auto stale = b;
  ++stale.generation;
  EXPECT_THROW(executor.reserve_cuda_sparse_trees({a, stale}),
               std::invalid_argument);
  invoke("bump_left");  // Failed batch must not reserve its valid prefix.
  EXPECT_EQ(invoke("peek_left", true), 4);
  auto first = executor.reserve_cuda_sparse_trees({a});
  EXPECT_THROW(invoke("bump_left"), std::invalid_argument);
  // A cold registration after reservation must be blocked as well.
  EXPECT_THROW(invoke("cold_bump_left"), std::invalid_argument);
  invoke("bump_right");
  EXPECT_EQ(invoke("peek_right", true), 8);
  auto second = executor.reserve_cuda_sparse_trees({b});
  EXPECT_THROW(executor.reserve_cuda_sparse_trees({a, b}),
               std::invalid_argument);
  EXPECT_THROW(executor.destroy_snode_tree_runtime(left),
               std::invalid_argument);
  EXPECT_THROW(executor.finalize(), std::invalid_argument);
  auto new_tree = left;
  new_tree.tree_id = b.tree_id + 1;
  EXPECT_THROW(executor.initialize_llvm_runtime_snodes(new_tree, result),
               std::invalid_argument);
  EXPECT_FALSE(executor.has_snode_tree_runtime_state(new_tree.tree_id));
  EXPECT_TRUE(executor.has_snode_tree_runtime_state(a.tree_id));
  EXPECT_TRUE(executor.has_snode_tree_runtime_state(b.tree_id));
  first.reset();
  invoke("bump_left");
  EXPECT_EQ(invoke("peek_left", true), 5);
  invoke("cold_bump_left");
  EXPECT_EQ(invoke("peek_left", true), 15);
  EXPECT_THROW(invoke("bump_right"), std::invalid_argument);
  second.reset();
  invoke("bump_right");
  EXPECT_EQ(invoke("peek_right", true), 9);
  executor.synchronize();
  executor.check_runtime_error(result);

  // Closing the module retires tree identities even when a cached permit
  // keeps the owning allocation leases alive. It may never submit again.
  module.reset();
  bool submitted = false;
  EXPECT_THROW(permit->submit([&] { submitted = true; }),
               std::invalid_argument);
  EXPECT_FALSE(submitted);
  EXPECT_FALSE(executor.has_snode_tree_runtime_state(a.tree_id));
  permit.reset();
  executor.finalize();
}

TEST(CudaSparseAccess, WholeGraphRejectsBeforePrefixAndRecovers) {
  const auto *folder = std::getenv("TAICHI_CUDA_SPARSE_ACCESS_FIXTURE");
  if (!folder || !cuda::FixedGraph::available())
    GTEST_SKIP() << "Requires the CUDA sparse access AOT fixture";
  CompileConfig config;
  config.arch = Arch::cuda;
  LlvmRuntimeExecutor executor(config, nullptr);
  uint64 *result = nullptr;
  executor.materialize_runtime(nullptr, &result);
  LLVM::AotModuleParams params;
  params.module_path = folder;
  params.executor_ = &executor;
  params.allocation_result_buffer = result;
  params.kernel_launcher = std::make_unique<cuda::KernelLauncher>(
      cuda::KernelLauncher::Config{&executor});
  std::shared_ptr<aot::Module> module =
      LLVM::make_aot_module(std::move(params));
  auto invoke = [&](const char *name, bool returns = false) {
    auto *kernel = module->get_kernel(name);
    LaunchContextBuilder args(kernel);
    kernel->launch(args);
    return returns ? args.get_ret<int>(0) : 0;
  };
  invoke("initialize");
  auto left =
      dynamic_cast<llvm_aot::FieldImpl &>(
          *static_cast<LLVM::LlvmAotModule &>(*module).get_snode_tree("0"))
          .get_snode_tree_cache();
  const LlvmSparseRuntimeViewTree tree{left.tree_id, 0, left.tree_generation};
  auto graph = module->get_graph("right_then_left");
  auto *a =
      dynamic_cast<llvm_aot::KernelImpl *>(module->get_kernel("bump_left"));
  auto *b =
      dynamic_cast<llvm_aot::KernelImpl *>(module->get_kernel("bump_right"));
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);
  auto &launcher = dynamic_cast<cuda::KernelLauncher &>(a->launcher());
  LaunchContextBuilder ca(a), cb(b);
  ca.discard_return_values();
  cb.discard_return_values();
  const cuda::FixedGraph::Dispatch da{&launcher, a->handle(), &ca};
  const cuda::FixedGraph::Dispatch db{&launcher, b->handle(), &cb};
  cuda::FixedGraph fixed(std::vector<cuda::FixedGraph::Dispatch>{db, da});
  auto &device =
      static_cast<cuda::CudaDevice &>(*executor.get_compute_device());
  auto allocation = device.allocate_memory_unique({sizeof(int)});
  ASSERT_EQ(allocation.second, RhiResult::success);
  auto *predicate = device.get_memory_addr(*allocation.first);
  int zero = 0;
  {
    auto context = CUDAContext::get_instance().get_guard();
    CUDADriver::get_instance().memcpy_host_to_device(predicate, &zero,
                                                   sizeof(zero));
  }
  auto loop = std::make_shared<cuda::FixedGraph::While>();
  loop->predicate = predicate;
  loop->max_iterations = 2;
  loop->condition = {db};
  loop->body = {da};
  cuda::FixedGraph control(std::vector<cuda::FixedGraph::Operation>{db, loop});
  auto host_loop = std::make_shared<FixedExecution::While>();
  host_loop->predicate = allocation.first->get_ptr(0);
  host_loop->max_iterations = 2;
  host_loop->host_control = true;
  host_loop->condition = {FixedExecution::Dispatch{b, &cb}};
  host_loop->body = {FixedExecution::Dispatch{a, &ca}};
  FixedExecution host(Arch::cuda, device, nullptr,
                      {FixedExecution::Dispatch{b, &cb}, host_loop});
  auto replay = std::make_shared<cuda::BindingFrameReplay>(72, nullptr);
  FixedExecution frame(
      Arch::cuda, device, nullptr,
      {FixedExecution::Dispatch{b, &cb}, FixedExecution::Dispatch{a, &ca}},
      nullptr, {}, replay, module);

  auto verify = [&](auto &&run, int right_delta, int left_delta) {
    const auto before_right = invoke("peek_right", true);
    const auto before_left = invoke("peek_left", true);
    auto loan = executor.reserve_cuda_sparse_trees({tree});
    EXPECT_THROW(run(), std::invalid_argument);
    EXPECT_EQ(invoke("peek_right", true), before_right);
    loan.reset();
    run();
    EXPECT_EQ(invoke("peek_right", true), before_right + right_delta);
    EXPECT_EQ(invoke("peek_left", true), before_left + left_delta);
  };
  // Cold and warm ordinary Graph, native replay, and a zero-trip body in a
  // complete host/native control sequence all reject before the right prefix.
  verify([&] { graph->run({}); }, 1, 1);
  verify([&] { graph->run({}); }, 1, 1);
  verify([&] { fixed.run(); }, 1, 1);
  verify([&] { control.run(); }, 2, 0);
  verify([&] { host.run(); }, 2, 0);
  verify([&] { frame.run(); }, 1, 1);
  auto loan = executor.reserve_cuda_sparse_trees({tree});
  EXPECT_THROW(
      cuda::FixedGraph(std::vector<cuda::FixedGraph::Dispatch>{db, da}),
      std::invalid_argument);
  loan.reset();
  executor.synchronize();
  executor.check_runtime_error(result);
}

INSTANTIATE_TEST_SUITE_P(Backends,
                         LlvmSparseRuntimeView,
                         ::testing::Values(host_arch(), Arch::cuda));
#else
INSTANTIATE_TEST_SUITE_P(Backends,
                         LlvmSparseRuntimeView,
                         ::testing::Values(host_arch()));
#endif

}  // namespace taichi::lang
#endif
