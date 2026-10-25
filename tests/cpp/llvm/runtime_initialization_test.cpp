#include "gtest/gtest.h"

#ifdef TI_WITH_LLVM
#include "taichi/jit/jit_module.h"
#include "taichi/runtime/llvm/llvm_runtime_executor.h"
#include "taichi/runtime/llvm/snode_runtime_state.h"
#include "taichi/runtime/llvm/sparse_runtime_view.h"
#include <array>
#include <cstring>
#include <set>
#include <thread>
#ifdef TI_WITH_CUDA
#include "taichi/platform/cuda/detect_cuda.h"
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/runtime/cuda/kernel_launcher.h"
#include "taichi/runtime/llvm/aot_graph_data.h"
#include "taichi/runtime/llvm/llvm_aot_module_loader.h"
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
};

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
