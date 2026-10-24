#include "gtest/gtest.h"

#if defined(TI_WITH_CUDA) && defined(TI_WITH_VULKAN)
#include <array>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <thread>
#include "taichi/program/runtime_fault.h"
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/runtime/cuda/execution_domain.h"
#include "taichi/runtime/cuda/fixed_graph.h"
#include "taichi/runtime/cuda/range_graph.h"
#include "taichi/runtime/llvm/aot_graph_data.h"
#include "taichi/runtime/llvm/llvm_aot_module_loader.h"
#include "tests/cpp/rhi/common/cuda_external_gate.h"

namespace taichi::lang {
namespace {
struct RangeOperands final : RuntimeCompletionResources {
  std::size_t retained_resource_count(std::uint32_t) const noexcept override {
    return 0;
  }
};

class CudaRangeGraph : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto *folder = std::getenv("TAICHI_CUDA_RANGE_AOT_FOLDER");
    if (!folder || !cuda::ExecutionDomain::available())
      GTEST_SKIP() << "Requires the compiled CUDA range fixture";
    CompileConfig config;
    config.arch = Arch::cuda;
    executor = std::make_unique<LlvmRuntimeExecutor>(config, nullptr);
    uint64 *result = nullptr;
    executor->materialize_runtime(nullptr, &result);
    LLVM::AotModuleParams params;
    params.module_path = folder;
    params.executor_ = executor.get();
    params.kernel_launcher = std::make_unique<cuda::KernelLauncher>(
        cuda::KernelLauncher::Config{executor.get()});
    module = LLVM::make_aot_module(std::move(params));
    owner = std::make_shared<RuntimeFaultDomain>(Arch::cuda, 2048);
    for (auto &domain : domains)
      domain = std::make_shared<cuda::ExecutionDomain>(
          owner, cuda::ExecutionDomain::Options{1, 0, 4});
  }
  void TearDown() override {
    for (auto &domain : domains)
      if (domain)
        domain->close();
  }
  void *allocate(std::size_t bytes) {
    auto &device = static_cast<cuda::CudaDevice &>(*executor->get_compute_device());
    auto allocation = device.allocate_memory_unique({bytes});
    TI_ASSERT(allocation.second == RhiResult::success);
    auto address = device.get_memory_addr(*allocation.first);
    storage.push_back(std::move(allocation.first));
    return address;
  }
  template <typename T, std::size_t N>
  void write(void *address, const std::array<T, N> &value) {
    CUDADriver::get_instance().memcpy_host_to_device(
        address, const_cast<T *>(value.data()), sizeof(value));
  }
  template <typename T, std::size_t N>
  std::array<T, N> read(void *address) {
    std::array<T, N> value{};
    CUDADriver::get_instance().memcpy_device_to_host(value.data(), address, sizeof(value));
    return value;
  }
  llvm_aot::KernelImpl *kernel(const char *name) {
    return dynamic_cast<llvm_aot::KernelImpl *>(module->get_kernel(name));
  }
  cuda::FixedGraph::Dispatch dispatch(llvm_aot::KernelImpl *kernel,
                                      LaunchContextBuilder &arguments) {
    auto &launcher = dynamic_cast<cuda::KernelLauncher &>(kernel->launcher());
    cuda::KernelLauncher::GraphLaunchPacket packet;
    TI_ASSERT(launcher.prepare_cuda_graph_launch_packet(kernel->handle(), arguments,
                                                        packet, true));
    cuda::FixedGraph::Dispatch result{&launcher, kernel->handle(), &arguments};
    for (std::size_t i = 0; i < packet.offloaded_tasks.size(); ++i)
      if (packet.offloaded_tasks[i].device_range_dispatch_version == 1)
        result.device_range_tasks.push_back(i);
    return result;
  }
  std::unique_ptr<LlvmRuntimeExecutor> executor;
  std::unique_ptr<aot::Module> module;
  std::vector<DeviceAllocationUnique> storage;
  std::shared_ptr<RuntimeFaultDomain> owner;
  std::array<std::shared_ptr<cuda::ExecutionDomain>, 2> domains;
};

TEST_F(CudaRangeGraph, PerTaskSnapshotsBusyAndIndependentRetirement) {
  auto context = CUDAContext::get_instance().get_guard();
  auto *entry = kernel("stage");
  ASSERT_NE(entry, nullptr);
  std::array<void *, 2> bounds{allocate(8), allocate(8)};
  std::array<void *, 2> output{allocate(66 * 4), allocate(66 * 4)};
  std::array<std::unique_ptr<cuda::FixedGraph>, 2> graphs;
  std::array<std::unique_ptr<LaunchContextBuilder>, 2> arguments;
  std::uint32_t preparation_error = 0;
  ASSERT_TRUE(cuda::driver_graph_prepare_bounded_update(&preparation_error));
  testing::CudaExternalGate gate;
  for (int i = 0; i < 2; ++i) {
    arguments[i] = std::make_unique<LaunchContextBuilder>(entry);
    arguments[i]->set_arg_external_array_with_shape(
        {0}, reinterpret_cast<std::uintptr_t>(bounds[i]), 8, {2});
    arguments[i]->set_arg_external_array_with_shape(
        {1}, reinterpret_cast<std::uintptr_t>(output[i]), 66 * 4, {66});
    auto selected = dispatch(entry, *arguments[i]);
    ASSERT_EQ(selected.device_range_tasks.size(), 2);
    std::vector<cuda::FixedGraph::Operation> operations;
    if (i == 0)
      operations.push_back(cuda::FixedGraph::Dispatch{
          nullptr, {}, nullptr, [&](void *stream) { gate.enqueue_wait(stream); }});
    operations.push_back(selected);
    graphs[i] = cuda::FixedGraph::prepare_execution_instance(
        operations, owner, domains[i], std::make_shared<RangeOperands>());
    EXPECT_EQ(graphs[i]->control_bytes(), taichi_global_tmp_buffer_size + 96);
  }
  EXPECT_NE(graphs[0]->instance_snapshot().temporary_storage,
            graphs[1]->instance_snapshot().temporary_storage);
  const auto prepare = [&](int i, int n) {
    write(bounds[i], std::array<int, 2>{n, -99});
    std::array<int, 66> initial;
    initial.fill(-1);
    write(output[i], initial);
  };
  const auto check = [&](int i, int n) {
    auto values = read<int, 66>(output[i]);
    for (int j = 0; j < 65; ++j)
      EXPECT_EQ(values[j], j < n ? j + 1 + (j < n / 2 ? 100 : 0) : -1);
    EXPECT_EQ(values[65], 1000 + n / 2);
    EXPECT_EQ((read<int, 2>(bounds[i])[1]), n / 2);
  };
  prepare(0, 65);
  prepare(1, 5);
  auto a = graphs[0]->submit();
  EXPECT_FALSE(a.done());
  auto b = graphs[1]->submit();
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
  while (!b.done() && std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  EXPECT_TRUE(b.done());
  EXPECT_FALSE(a.done());
  EXPECT_FALSE(gate.timed_out());
  const auto sequence = domains[0]->latest_completion().sequence();
  EXPECT_THROW(graphs[0]->submit(), cuda::ExecutionDomainBusy);
  EXPECT_EQ(domains[0]->latest_completion().sequence(), sequence);
  // Release before any assertion/readback that might unwind or use the default
  // queue. The test proves progress via the independent completion above.
  gate.release();
  a.wait();
  b.wait();
  check(0, 65);
  check(1, 5);
  graphs[0]->retire_execution_instance();
  for (int n : {0, 3, 65, 0}) {
    prepare(1, n);
    graphs[1]->submit().wait();
    check(1, n);
  }
  // A serial task cannot gain qualification from a caller selecting its index.
  auto bad = dispatch(entry, *arguments[1]);
  bad.device_range_tasks = {0};
  bool captured = false;
  std::vector<cuda::FixedGraph::Operation> invalid{
      cuda::FixedGraph::Dispatch{nullptr, {}, nullptr,
                                [&](void *) { captured = true; }}, bad};
  for (int attempt = 0; attempt < 5; ++attempt)
    EXPECT_THROW(cuda::FixedGraph::prepare_execution_instance(
                     invalid, owner, domains[0], std::make_shared<RangeOperands>()),
                 std::invalid_argument);
  EXPECT_FALSE(captured);
  auto recovered = cuda::FixedGraph::prepare_execution_instance(
      {dispatch(entry, *arguments[0])}, owner, domains[0],
      std::make_shared<RangeOperands>());
  prepare(0, 3);
  recovered->submit().wait();
  check(0, 3);
  recovered->retire_execution_instance();
  graphs[1]->submit().wait();
  check(1, 0);
}

TEST_F(CudaRangeGraph, PrimalAndExplicitAdjointKeepIndependentProofs) {
  auto context = CUDAContext::get_instance().get_guard();
  void *x = allocate(65 * 4), *y = allocate(65 * 4);
  void *dx = allocate(65 * 4), *dy = allocate(65 * 4), *bounds = allocate(4);
  std::array<std::unique_ptr<LaunchContextBuilder>, 2> arguments;
  std::vector<cuda::FixedGraph::Operation> operations;
  const char *names[] = {"square", "square_grad"};
  for (int i = 0; i < 2; ++i) {
    auto *entry = kernel(names[i]);
    ASSERT_NE(entry, nullptr);
    arguments[i] = std::make_unique<LaunchContextBuilder>(entry);
    arguments[i]->set_arg_external_array_with_shape(
        {0}, reinterpret_cast<std::uintptr_t>(x), 65 * 4, {65},
        reinterpret_cast<std::uintptr_t>(dx));
    arguments[i]->set_arg_external_array_with_shape(
        {1}, reinterpret_cast<std::uintptr_t>(y), 65 * 4, {65},
        reinterpret_cast<std::uintptr_t>(dy));
    arguments[i]->set_arg_external_array_with_shape(
        {2}, reinterpret_cast<std::uintptr_t>(bounds), 4, {1});
    auto selected = dispatch(entry, *arguments[i]);
    ASSERT_EQ(selected.device_range_tasks.size(), 1);
    operations.push_back(selected);
  }
  for (bool independent : {true, false}) {
    auto graph = independent
        ? cuda::FixedGraph::prepare_execution_instance(
              operations, owner, domains[0], std::make_shared<RangeOperands>())
        : std::make_unique<cuda::FixedGraph>(operations, owner);
    std::array<float, 65> input;
    for (int j = 0; j < 65; ++j)
      input[j] = j + 1;
    write(x, input);
    for (int n : {0, 5, 65, 0}) {
      write(bounds, std::array<int, 1>{n});
      std::array<float, 65> initial;
      initial.fill(-1);
      write(y, initial);
      initial.fill(0);
      write(dx, initial);
      initial.fill(1);
      write(dy, initial);
      if (independent)
        graph->submit().wait();
      else {
        graph->run();
        executor->synchronize();
      }
      const auto result = read<float, 65>(y);
      const auto gradient = read<float, 65>(dx);
      for (int j = 0; j < 65; ++j) {
        EXPECT_EQ(result[j], j < n ? input[j] * input[j] : -1);
        EXPECT_EQ(gradient[j], j < n ? 2 * input[j] : 0);
      }
    }
  }
}

TEST_F(CudaRangeGraph, OrdinaryGraphsBorrowTheActualRuntimeWorkspace) {
  auto guard = CUDAContext::get_instance().get_guard();
  auto *entry = kernel("stage");
  ASSERT_NE(entry, nullptr);
  auto *bounds = allocate(8);
  auto *output = allocate(66 * 4);
  LaunchContextBuilder arguments(entry);
  arguments.set_arg_external_array_with_shape(
      {0}, reinterpret_cast<std::uintptr_t>(bounds), 8, {2});
  arguments.set_arg_external_array_with_shape(
      {1}, reinterpret_cast<std::uintptr_t>(output), 66 * 4, {66});
  auto selected = dispatch(entry, arguments);
  ASSERT_EQ(selected.device_range_tasks.size(), 2);
  auto *runtime_workspace = executor->get_runtime_temporary_ptr();
  cuda::KernelLauncher::GraphLaunchPacket packet;
  ASSERT_TRUE(selected.launcher->prepare_cuda_graph_launch_packet(
      entry->handle(), arguments, packet, true));
  EXPECT_EQ(packet.context.global_tmp_buffer, nullptr);
  selected.launcher->bind_runtime_temporaries(packet);
  EXPECT_EQ(packet.context.global_tmp_buffer, runtime_workspace);
  // A second or foreign binding cannot silently replace an explicit owner.
  EXPECT_THROW(selected.launcher->bind_runtime_temporaries(packet),
               std::invalid_argument);
  packet.context.global_tmp_buffer = nullptr;
  packet.context.runtime = nullptr;
  EXPECT_THROW(selected.launcher->bind_runtime_temporaries(packet),
               std::invalid_argument);

  auto first = std::make_unique<cuda::FixedGraph>(
      std::vector<cuda::FixedGraph::Dispatch>{selected}, owner);
  auto second = std::make_unique<cuda::FixedGraph>(
      std::vector<cuda::FixedGraph::Dispatch>{selected}, owner);
  auto isolated = cuda::FixedGraph::prepare_execution_instance(
      {selected}, owner, domains[0], std::make_shared<RangeOperands>());
  EXPECT_EQ(first->control_bytes(), 96);
  EXPECT_EQ(second->control_bytes(), 96);
  EXPECT_NE(isolated->instance_snapshot().temporary_storage,
            reinterpret_cast<std::uintptr_t>(runtime_workspace));
  const auto run = [&](int n, int route) {
    write(bounds, std::array<int, 2>{n, -99});
    std::array<int, 66> initial;
    initial.fill(-1);
    write(output, initial);
    if (route == 0)
      first->run();
    else if (route == 1)
      second->run();
    else if (route == 2)
      entry->launch(arguments);  // The ordinary null-override fallback remains.
    else
      isolated->submit().wait();
    executor->synchronize();
    const auto result = read<int, 66>(output);
    for (int j = 0; j < 65; ++j)
      EXPECT_EQ(result[j], j < n ? j + 1 + (j < n / 2 ? 100 : 0) : -1);
    EXPECT_EQ(result[65], 1000 + n / 2);
    EXPECT_EQ(executor->get_runtime_temporary_ptr(), runtime_workspace);
  };
  for (int n : {0, 3, 65, 0})
    for (int route = 0; route < 4; ++route)
      run(n, route);
  first.reset();
  isolated->retire_execution_instance();
  run(5, 1);
  second.reset();
  run(3, 2);
}

TEST_F(CudaRangeGraph, RejectsInvalidBoundsAndUnqualifiedStrategiesBeforeCapture) {
  auto context = CUDAContext::get_instance().get_guard();
  auto *entry = kernel("stage");
  ASSERT_NE(entry, nullptr);
  LaunchContextBuilder arguments(entry);
  arguments.set_arg_external_array_with_shape(
      {0}, reinterpret_cast<std::uintptr_t>(allocate(8)), 8, {2});
  arguments.set_arg_external_array_with_shape(
      {1}, reinterpret_cast<std::uintptr_t>(allocate(66 * 4)), 66 * 4, {66});
  auto selected = dispatch(entry, arguments);
  cuda::KernelLauncher::GraphLaunchPacket packet;
  ASSERT_TRUE(selected.launcher->prepare_cuda_graph_launch_packet(
      entry->handle(), arguments, packet, true));
  selected.launcher->bind_execution_temporaries(
      packet, allocate(taichi_global_tmp_buffer_size), taichi_global_tmp_buffer_size);
  ASSERT_EQ(selected.device_range_tasks.size(), 2);
  for (int mode = 0; mode < 4; ++mode) {
    auto bad = packet;
    auto selection = selected.device_range_tasks;
    auto &task = bad.offloaded_tasks[selection[0]];
    if (mode == 0)
      task.range_bounds->end.value = taichi_global_tmp_buffer_size;
    else if (mode == 1)
      task.device_range_dispatch_version = 0;
    else if (mode == 2)
      selection.push_back(selection[0]);
    else
      bad.context.global_tmp_buffer = nullptr;
    cuda::RangeGraphControls controls;
    std::uint32_t error = 0;
    EXPECT_THROW(controls.initialize({bad}, {selection},
                                    taichi_global_tmp_buffer_size, &error),
                 std::invalid_argument);
    EXPECT_EQ(controls.device_bytes(), 0);
  }
  // Sharing an executable across binding frames is still not qualified.
  EXPECT_THROW(cuda::FixedGraph::capture_binding_frame(
                   {selected}, std::make_shared<int>(0), owner),
               std::invalid_argument);
}

TEST_F(CudaRangeGraph, ControllerHandlesSignedExtremesWithoutClampingBounds) {
  auto context = CUDAContext::get_instance().get_guard();
  auto &driver = CUDADriver::get_instance();
  std::uint32_t error = 0;
  ASSERT_TRUE(cuda::driver_graph_prepare_bounded_update(&error));
  auto *control = static_cast<cuda::CudaGraphRangeControl *>(
      allocate(sizeof(cuda::CudaGraphRangeControl)));
  auto *bounds = allocate(8);
  auto *visited = allocate(4);
  struct Capture {
    void *stream{nullptr};
    CUgraph graph{nullptr};
    CUgraphExec executable{nullptr};
    bool active{false};
    ~Capture() {
      auto &driver = CUDADriver::get_instance();
      if (active)
        driver.stream_end_capture.call(stream, &graph);
      if (stream)
        driver.stream_synchronize.call_with_warning(stream);
      if (executable)
        driver.graph_exec_destroy.call_with_warning(executable);
      if (graph)
        driver.graph_destroy.call_with_warning(graph);
      if (stream)
        driver.stream_destroy.call_with_warning(stream);
    }
  } capture;
  driver.stream_create(&capture.stream, CU_STREAM_NON_BLOCKING);
  driver.stream_begin_capture(capture.stream, CU_STREAM_CAPTURE_MODE_GLOBAL);
  capture.active = true;
  cuda::driver_graph_update_range(control, capture.stream);
  TaichiCudaLaunchAttribute attribute{};
  attribute.id = TAICHI_CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE;
  attribute.value.device_updatable_kernel_node.device_updatable = 1;
  TaichiCudaLaunchConfig config{};
  config.grid_dim_x = 3;
  config.grid_dim_y = config.grid_dim_z = 1;
  config.block_dim_x = 32;
  config.block_dim_y = config.block_dim_z = 1;
  config.stream = capture.stream;
  config.attributes = &attribute;
  config.num_attributes = 1;
  void *arguments[]{&visited};
  driver.launch_kernel_ex(&config, cuda::driver_graph_bounded_probe_payload_function(),
                          arguments, nullptr);
  driver.stream_end_capture(capture.stream, &capture.graph);
  capture.active = false;
  driver.graph_instantiate_with_flags(&capture.executable, capture.graph, 0);
  driver.graph_upload(capture.executable, capture.stream);
  driver.stream_synchronize(capture.stream);
  cuda::CudaGraphRangeControl host;
  host.device_node = reinterpret_cast<std::uintptr_t>(
      attribute.value.device_updatable_kernel_node.device_node);
  ASSERT_NE(host.device_node, 0);
  host.block_dim = 32;
  host.max_grid_dim = 3;
  const auto lo = std::numeric_limits<std::int32_t>::min();
  const auto hi = std::numeric_limits<std::int32_t>::max();
  const std::array<std::array<int, 2>, 8> intervals{
      std::array<int, 2>{lo, hi}, {hi, lo}, {-3, 2}, {-3, 30},
      {0, 0}, {lo, lo + 65}, {hi - 5, hi}, {0, 0}};
  for (int constants : {0, 1, 2}) {
    for (const auto &interval : intervals) {
      write(bounds, interval);
      write(visited, std::array<std::uint32_t, 1>{0});
      host.begin_pointer = constants == 1 ? 0 : reinterpret_cast<std::uintptr_t>(bounds);
      host.begin_constant = interval[0];
      host.end_pointer = constants == 2 ? 0 : reinterpret_cast<std::uintptr_t>(bounds) + 4;
      host.end_constant = interval[1];
      driver.memcpy_host_to_device(control, &host, sizeof(host));
      driver.graph_launch(capture.executable, capture.stream);
      driver.stream_synchronize(capture.stream);
      cuda::CudaGraphRangeControl result;
      driver.memcpy_device_to_host(&result, control, sizeof(result));
      const auto count = std::max<std::int64_t>(
          0, std::int64_t(interval[1]) - interval[0]);
      const auto grid = std::max<std::int64_t>(1, std::min<std::int64_t>(3, (count + 31) / 32));
      EXPECT_EQ(result.driver_status, 0);
      EXPECT_EQ(result.last_grid_dim, grid);
      EXPECT_EQ((read<std::uint32_t, 1>(visited)[0]), grid * 32);
      EXPECT_EQ((read<int, 2>(bounds)), interval);
    }
  }
}
// A separate Python producer exports ordinary and selected names for one
// physical payload. Both load orders must register exactly one executable.
class CudaRangeAotEntry : public CudaRangeGraph {};

TEST_F(CudaRangeAotEntry, OrdinaryFirst) {
  auto *ordinary = kernel(std::getenv("TAICHI_RANGE_ORDINARY_ENTRY"));
  auto *selected = kernel(std::getenv("TAICHI_RANGE_SELECTED_ENTRY"));
  ASSERT_NE(ordinary, nullptr);
  ASSERT_NE(selected, nullptr);
  EXPECT_NE(ordinary, selected);
  EXPECT_EQ(&ordinary->launcher(), &selected->launcher());
  EXPECT_EQ(ordinary->handle().get_launch_id(), selected->handle().get_launch_id());
  EXPECT_TRUE(ordinary->device_range_tasks().empty());
  EXPECT_EQ(selected->device_range_tasks().size(), 2);
  EXPECT_EQ(ordinary->launcher().debug_registered_kernel_count(), 1);
  LaunchContextBuilder arguments(selected);
  EXPECT_ANY_THROW(selected->launch(arguments));
}

TEST_F(CudaRangeAotEntry, SelectedFirst) {
  auto *selected = kernel(std::getenv("TAICHI_RANGE_SELECTED_ENTRY"));
  auto *ordinary = kernel(std::getenv("TAICHI_RANGE_ORDINARY_ENTRY"));
  ASSERT_NE(ordinary, nullptr);
  ASSERT_NE(selected, nullptr);
  EXPECT_EQ(ordinary->handle().get_launch_id(), selected->handle().get_launch_id());
  EXPECT_TRUE(ordinary->device_range_tasks().empty());
  EXPECT_EQ(selected->device_range_tasks().size(), 2);
  EXPECT_EQ(selected->launcher().debug_registered_kernel_count(), 1);
}
}  // namespace
}  // namespace taichi::lang
#endif
