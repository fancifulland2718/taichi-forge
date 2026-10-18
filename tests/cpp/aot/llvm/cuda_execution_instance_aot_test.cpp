#include <gtest/gtest.h>

#if defined(TI_WITH_CUDA) && defined(TI_WITH_VULKAN)
#include <array>
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <mutex>
#include <cstdlib>
#include <limits>
#include "tests/cpp/rhi/common/cuda_external_gate.h"
#include "taichi/program/runtime_fault.h"
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/runtime/cuda/execution_domain.h"
#include "taichi/runtime/cuda/fixed_graph.h"
#include "taichi/runtime/cuda/kernel_launcher.h"
#include "taichi/runtime/fixed_execution.h"
#include "taichi/runtime/native_operation_registry.h"
#include "taichi/runtime/native_scan.h"
#include "taichi/runtime/llvm/aot_graph_data.h"
#include "taichi/runtime/llvm/llvm_aot_module_loader.h"
#include "taichi/system/profiler_annotation.h"

namespace taichi::lang {
namespace {
constexpr int count = 64;
constexpr std::uint32_t operand_kind = 193;

struct OrderedTransaction final : CudaProviderCompletionResource {
  mutable std::mutex mutex;
  std::uint64_t generation{0}, active{0};
  bool started{false};
  std::atomic<int> cancelled{0}, retired{0};
  void reserve() {
    std::lock_guard<std::mutex> lock(mutex);
    if (active)
      throw std::runtime_error("provider transaction is already reserved");
    active = ++generation;
    started = false;
  }
  void start() {
    std::lock_guard<std::mutex> lock(mutex);
    if (!active || started)
      throw std::logic_error("provider transaction is not reserved");
    started = true;
  }
  void cancel() noexcept {
    std::lock_guard<std::mutex> lock(mutex);
    if (active && !started) {
      active = 0;
      ++cancelled;
    }
  }
  std::uint64_t submission_retirement_token() const override {
    std::lock_guard<std::mutex> lock(mutex);
    return active;
  }
  void on_submission_retired(std::uint64_t token) noexcept override {
    std::lock_guard<std::mutex> lock(mutex);
    if (token && token == active) {
      active = 0;
      ++retired;
    }
  }
};

class OrderedCopy final : public NativeOperation {
 public:
  OrderedCopy(Device &device, void *source, void *target, bool fail = false)
      : device_(device), source_(source), target_(target), fail_(fail) {
  }
  Arch arch() const override {
    return Arch::cuda;
  }
  Device &device() const override {
    return device_;
  }
  std::size_t workspace_bytes() const override {
    return 0;
  }
  NativeOperationMode execution_mode() const override {
    return NativeOperationMode::cuda_ordered;
  }
  std::shared_ptr<CudaProviderCompletionResource> cuda_ordered_resource()
      const override {
    return transaction;
  }
  void reserve_cuda_ordered() override {
    transaction->reserve();
  }
  void cancel_cuda_ordered() noexcept override {
    transaction->cancel();
  }
  void run_cuda_ordered(void *stream) override {
    if (!stream)
      throw std::logic_error("Test provider requires an independent stream");
    transaction->start();
    ++calls;
    CUDADriver::get_instance().memcpy_device_to_device_async(
        target_, source_, count * sizeof(float), stream);
    if (fail_)
      throw std::runtime_error("Provider failed after copy enqueue");
  }
  std::shared_ptr<OrderedTransaction> transaction =
      std::make_shared<OrderedTransaction>();
  int calls{0};

 private:
  Device &device_;
  void *source_, *target_;
  bool fail_;
};

struct InstanceOperands final : RuntimeCompletionResources {
  CompileConfig config;
  std::unique_ptr<LlvmRuntimeExecutor> executor;
  std::unique_ptr<aot::Module> module;
  std::vector<DeviceAllocationUnique> storage;
  std::array<float *, 2> readback{};

  explicit InstanceOperands(const char *folder,
                            bool with_copy_targets = false) {
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
    auto &loaded = static_cast<LLVM::LlvmAotModule &>(*module);
    for (int tree : loaded.get_snode_tree_ids())
      LLVM::allocate_aot_snode_tree_type(
          module.get(), module->get_snode_tree(std::to_string(tree)), result);
    auto guard = CUDAContext::get_instance().get_guard();
    auto &driver = CUDADriver::get_instance();
    for (int i = 0; i < (with_copy_targets ? 4 : 2); ++i) {
      auto allocation =
          device().allocate_memory_unique({count * sizeof(float)});
      if (allocation.second != RhiResult::success)
        throw std::runtime_error(
            "Execution instance fixture allocation failed");
      storage.push_back(std::move(allocation.first));
      driver.memsetd32(address(i), 0, count);
    }
    for (auto &buffer : readback)
      driver.mem_host_alloc(reinterpret_cast<void **>(&buffer),
                            count * sizeof(float), 0);
    driver.stream_synchronize(nullptr);
  }
  ~InstanceOperands() override {
    auto guard = CUDAContext::get_instance().get_guard();
    for (auto *buffer : readback)
      if (buffer)
        CUDADriver::get_instance().mem_free_host.call(buffer);
  }
  std::size_t retained_resource_count(
      std::uint32_t kind) const noexcept override {
    return kind == operand_kind ? storage.size() : 0;
  }
  cuda::CudaDevice &device() {
    return static_cast<cuda::CudaDevice &>(*executor->get_compute_device());
  }
  void *address(int index) {
    return device().get_memory_addr(*storage[index]);
  }
  llvm_aot::KernelImpl &kernel(const char *name = "stage") {
    auto *result =
        dynamic_cast<llvm_aot::KernelImpl *>(module->get_kernel(name));
    if (!result)
      throw std::runtime_error("Missing saved stage kernel");
    return *result;
  }
  cuda::FixedGraph::Dispatch dispatch(LaunchContextBuilder &args,
                                      int index,
                                      const char *name = "stage") {
    args.set_arg_external_array_with_shape(
        {0}, reinterpret_cast<std::uintptr_t>(address(index)),
        count * sizeof(float), {count});
    args.set_arg_float({1}, index == 0 ? 2.0 : 7.0);
    args.set_arg_int({2}, count);
    auto &entry = kernel(name);
    return {&dynamic_cast<cuda::KernelLauncher &>(entry.launcher()),
            entry.handle(), &args};
  }
  cuda::FixedGraph::Dispatch read(int index) {
    return {nullptr,
            {},
            nullptr,
            [host = readback[index], device = address(index)](void *stream) {
              CUDADriver::get_instance().memcpy_device_to_host_async(
                  host, device, count * sizeof(float), stream);
            }};
  }
  cuda::FixedGraph::Dispatch copy(int index) {
    return {
        nullptr,
        {},
        nullptr,
        [source = address(index), target = address(index + 2)](void *stream) {
          CUDADriver::get_instance().memcpy_device_to_device_async(
              target, source, count * sizeof(float), stream);
        }};
  }
  void check(int index) {
    const float carried = (index == 0 ? 2.0f : 7.0f) * 5.0f + 1.0f;
    for (int j = 0; j < count; ++j)
      EXPECT_FLOAT_EQ(readback[index][j],
                      carried * (count * (count + 1) / 2) + j);
  }
  void poison_results() {
    auto guard = CUDAContext::get_instance().get_guard();
    auto &driver = CUDADriver::get_instance();
    for (std::size_t i = 0; i < storage.size(); ++i)
      driver.memsetd32(address(static_cast<int>(i)), 0x7fc00000u, count);
    for (auto *buffer : readback)
      std::fill(buffer, buffer + count,
                std::numeric_limits<float>::quiet_NaN());
    driver.stream_synchronize(nullptr);  // Cold oracle reset before the gate.
  }
};

bool poll(const RuntimeCompletion &ticket) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!ticket.done() && std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  return ticket.done();
}

TEST(CudaExecutionContextAot,
     FixedInstancesRetainPrivateStorageUntilDomainClose) {
  const auto *folder = std::getenv("TAICHI_F3_CONTEXT_AOT_FOLDER");
  if (!folder || !cuda::ExecutionDomain::available())
    GTEST_SKIP() << "Requires the saved execution-context AOT fixture";
  auto operands = std::make_shared<InstanceOperands>(folder);
  auto owner = std::make_shared<RuntimeFaultDomain>(Arch::cuda, 811);
  cuda::ExecutionDomainRegistry registry(owner, 2);
  auto output = registry.create({2, 0});
  auto physics = registry.create({2, -1});
  LaunchContextBuilder first_args(&operands->kernel()),
      second_args(&operands->kernel());
  auto first = cuda::FixedGraph::prepare_execution_instance(
      {operands->dispatch(first_args, 0)}, owner, output, operands);
  auto second = cuda::FixedGraph::prepare_execution_instance(
      {operands->dispatch(second_args, 1), operands->read(1)}, owner, physics,
      operands);
  const auto a = first->instance_snapshot(), b = second->instance_snapshot();
  EXPECT_GE(first->node_count(), 4);
  EXPECT_GE(second->node_count(), 5);
  EXPECT_NE(a.identity, b.identity);
  EXPECT_NE(a.temporary_storage, b.temporary_storage);
  EXPECT_EQ(a.temporary_bytes, taichi_global_tmp_buffer_size);
  EXPECT_EQ(b.temporary_bytes, taichi_global_tmp_buffer_size);
  EXPECT_EQ(first->control_bytes(), a.temporary_bytes);
  first->submit().wait();
  output->submit({}, nullptr, operands->read(0).external).wait();
  second->submit().wait();
  operands->check(0);
  operands->check(1);
  operands->poison_results();
  testing::CudaExternalGate gate;
  auto blocked = output->submit(
      {}, nullptr, [&](void *stream) { gate.enqueue_wait(stream); });
  ASSERT_FALSE(blocked.done());
  auto pending = first->submit();
  EXPECT_THROW(first->submit(), cuda::ExecutionDomainBusy);
  EXPECT_EQ(pending.retained_resource_count(operand_kind), 2);
  RuntimeCompletion sentinel;
  {
    ScopedExternalProfilerAnnotation annotation("F3/fixed_instance_submit");
    sentinel = second->submit();
  }
  ASSERT_TRUE(poll(sentinel));
  EXPECT_FALSE(pending.done());
  operands->check(1);  // Pinned readback is part of the selected Graph itself.
  // Mutating the original host contexts cannot change captured arguments.
  first_args.set_arg_float({1}, -123.0);
  second_args.set_arg_float({1}, -456.0);
  auto reused = second->submit({sentinel});
  ASSERT_TRUE(poll(reused));
  operands->check(1);
  std::weak_ptr<InstanceOperands> retained = operands;
  {
    ScopedExternalProfilerAnnotation annotation(
        "F3/drop_pending_instance_wrappers");
    first.reset();
    second.reset();
  }
  EXPECT_FALSE(gate.timed_out());
  EXPECT_FALSE(pending.done());
  {
    ScopedExternalProfilerAnnotation annotation("F3/fixed_instance_release");
    gate.release();
  }
  pending.wait();
  output->submit({}, nullptr, operands->read(0).external).wait();
  operands->check(0);
  operands.reset();
  EXPECT_FALSE(retained.expired());  // No driver frees in ticket retirement.
  registry.retire();
  EXPECT_TRUE(retained.expired());
}

TEST(CudaExecutionContextAot, FixedParallelBranchesHaveIndependentTemporaries) {
  const auto *folder = std::getenv("TAICHI_F3_CONTEXT_AOT_FOLDER");
  if (!folder || !cuda::ExecutionDomain::available())
    GTEST_SKIP() << "Requires the saved execution-context AOT fixture";
  auto operands = std::make_shared<InstanceOperands>(folder);
  auto owner = std::make_shared<RuntimeFaultDomain>(Arch::cuda, 812);
  cuda::ExecutionDomainRegistry registry(owner, 1);
  auto domain = registry.create({2, 0});
  LaunchContextBuilder a(&operands->kernel()), b(&operands->kernel());
  auto parallel = std::make_shared<cuda::FixedGraph::Parallel>();
  parallel->branches = {{operands->dispatch(a, 0), operands->read(0)},
                        {operands->dispatch(b, 1), operands->read(1)}};
  auto graph = cuda::FixedGraph::prepare_execution_instance({parallel}, owner,
                                                            domain, operands);
  EXPECT_EQ(graph->instance_snapshot().temporary_bytes,
            2 * taichi_global_tmp_buffer_size);
  for (int i = 0; i < 32; ++i) {
    graph->submit().wait();
    operands->check(0);
    operands->check(1);
  }
  registry.retire();
  EXPECT_TRUE(graph->instance_snapshot().retired);
  EXPECT_ANY_THROW(graph->submit());
}

void check_fixed_execution_recipe(bool ordered) {
  const auto *folder = std::getenv("TAICHI_F3_CONTEXT_AOT_FOLDER");
  if (!folder || !cuda::ExecutionDomain::available())
    GTEST_SKIP();
  auto operands = std::make_shared<InstanceOperands>(folder, true);
  auto owner = std::make_shared<RuntimeFaultDomain>(Arch::cuda, 833);
  cuda::ExecutionDomainRegistry registry(owner, 2);
  auto output = registry.create({2, 0}), physics = registry.create({2, -1});
  LaunchContextBuilder ca(&operands->kernel()), cb(&operands->kernel());
  std::array<std::shared_ptr<NativeOperation>, 2> scans;
  std::array<std::shared_ptr<OrderedCopy>, 2> copies;
  if (ordered)
    for (int i = 0; i < 2; ++i)
      copies[i] = std::make_shared<OrderedCopy>(
          operands->device(), operands->address(i), operands->address(i + 2));
  for (int i = 0; i < 2; ++i)
    scans[i] = native_operation_registry().prepare(
        Arch::cuda, operands->device(),
        ScanOperation{count,
                      1,
                      {operands->storage[i + 2]->get_ptr(),
                       count * sizeof(float),
                       PrimitiveType::f32,
                       {count},
                       {}}});
  const auto prepare =
      [&](int index, LaunchContextBuilder &context,
          const std::shared_ptr<cuda::ExecutionDomain> &domain) {
        operands->dispatch(context, index);
        FixedExecution::CudaExecutionBinding binding{domain, operands};
        std::vector<FixedExecution::Operation> operations{
            FixedExecution::BufferFill{operands->storage[index + 2]->get_ptr(),
                                       count * sizeof(float), 0},
            FixedExecution::Dispatch{&operands->kernel(), &context},
            ordered
                ? FixedExecution::Operation(
                      std::static_pointer_cast<NativeOperation>(copies[index]))
                : FixedExecution::Operation(FixedExecution::BufferCopy{
                      operands->storage[index + 2]->get_ptr(),
                      operands->storage[index]->get_ptr(),
                      count * sizeof(float)}),
            scans[index]};
        return std::make_unique<FixedExecution>(
            Arch::cuda, operands->device(), nullptr, operations, nullptr, owner,
            nullptr, nullptr, nullptr, nullptr, &binding);
      };
  auto first = prepare(0, ca, output), second = prepare(1, cb, physics);
  EXPECT_EQ(first->execution_kind(), ordered ? 4 : 1);
  EXPECT_GE(first->workspace_bytes(), taichi_global_tmp_buffer_size);
  first->submit().wait();
  second->submit().wait();
  operands->poison_results();
  const auto read = [&](int index, cuda::ExecutionDomain &domain) {
    domain
        .submit({}, nullptr,
                [&](void *stream) {
                  CUDADriver::get_instance().memcpy_device_to_host_async(
                      operands->readback[index], operands->address(index + 2),
                      count * sizeof(float), stream);
                })
        .wait();
    const float total = (index == 0 ? 11.f : 36.f) * (count * (count + 1) / 2);
    for (int j = 0; j < count; ++j)
      EXPECT_FLOAT_EQ(operands->readback[index][j],
                      total * (j + 1) + (j * (j + 1) / 2));
  };
  testing::CudaExternalGate gate;
  auto blocked =
      output->submit({}, nullptr, [&](void *s) { gate.enqueue_wait(s); });
  auto pending = first->submit();
  RuntimeCompletion sentinel;
  {
    ScopedExternalProfilerAnnotation range(
        ordered ? "F3/fixed_ordered_submit"
                : "F3/fixed_execution_native_submit");
    sentinel = second->submit();
  }
  ASSERT_TRUE(poll(sentinel));
  EXPECT_FALSE(pending.done());
  read(1, *physics);
  ca.set_arg_float({1}, -100);
  cb.set_arg_float({1}, -200);
  auto dependent = second->submit({pending});
  EXPECT_FALSE(dependent.done());
  std::weak_ptr<NativeOperation> first_scan = scans[0], second_scan = scans[1];
  scans = {};
  first.reset();
  second.reset();
  EXPECT_FALSE(first_scan.expired());
  EXPECT_FALSE(second_scan.expired());
  EXPECT_FALSE(blocked.done());
  EXPECT_FALSE(gate.timed_out());
  {
    ScopedExternalProfilerAnnotation range(
        ordered ? "F3/fixed_ordered_release" : "F3/fixed_execution_release");
    gate.release();
  }
  pending.wait();
  dependent.wait();
  read(0, *output);
  read(1, *physics);
  std::weak_ptr<InstanceOperands> retained = operands;
  operands.reset();
  registry.retire();
  EXPECT_TRUE(retained.expired());
  EXPECT_TRUE(first_scan.expired());
  EXPECT_TRUE(second_scan.expired());
  if (ordered) {
    EXPECT_EQ(copies[0]->transaction->retired.load(), 2);
    EXPECT_EQ(copies[1]->transaction->retired.load(), 3);
    EXPECT_EQ(copies[0]->transaction->cancelled.load(), 0);
    EXPECT_EQ(copies[1]->transaction->cancelled.load(), 0);
  }
}

TEST(CudaExecutionContextAot,
     FixedExecutionKeepsKernelCopyAndNativeRecipeOnItsDomain) {
  check_fixed_execution_recipe(false);
}

TEST(CudaExecutionContextAot, OrderedRecipePreservesRootsAndExactRetirement) {
  check_fixed_execution_recipe(true);
}

TEST(CudaExecutionContextAot, OrderedReservationFailureCancelsOnlyItsOwnBatch) {
  const auto *folder = std::getenv("TAICHI_F3_CONTEXT_AOT_FOLDER");
  if (!folder || !cuda::ExecutionDomain::available())
    GTEST_SKIP();
  auto operands = std::make_shared<InstanceOperands>(folder, true);
  auto owner = std::make_shared<RuntimeFaultDomain>(Arch::cuda, 835);
  cuda::ExecutionDomainRegistry registry(owner, 3);
  auto blocked_domain = registry.create({1, 0});
  auto domain = registry.create(
      {1, 0, 1});  // One instance slot owns all roots/providers.
  auto provider_domain = registry.create({1, 0, 1});
  auto first = std::make_shared<OrderedCopy>(
      operands->device(), operands->address(0), operands->address(1));
  auto second = std::make_shared<OrderedCopy>(
      operands->device(), operands->address(1), operands->address(2));
  FixedExecution::CudaExecutionBinding binding{domain, operands};
  FixedExecution recipe(
      Arch::cuda, operands->device(), nullptr,
      {FixedExecution::BufferFill{operands->storage[0]->get_ptr(),
                                  count * sizeof(float), 0x3f800000u},
       std::static_pointer_cast<NativeOperation>(first),
       std::static_pointer_cast<NativeOperation>(second),
       FixedExecution::BufferFill{operands->storage[3]->get_ptr(),
                                  count * sizeof(float), 0x40000000u}},
      nullptr, owner, nullptr, nullptr, nullptr, nullptr, &binding);
  auto provider = std::make_shared<OrderedCopy>(
      operands->device(), operands->address(0), operands->address(2));
  std::weak_ptr<OrderedCopy> retained_provider = provider;
  FixedExecution::CudaExecutionBinding provider_binding{provider_domain,
                                                        operands};
  auto provider_only = std::make_unique<FixedExecution>(
      Arch::cuda, operands->device(), nullptr,
      std::vector<FixedExecution::Operation>{
          std::static_pointer_cast<NativeOperation>(provider)},
      nullptr, owner, nullptr, nullptr, nullptr, nullptr, &provider_binding);
  EXPECT_EQ(provider_only->execution_kind(), 4);
  EXPECT_EQ(provider_only->workspace_bytes(), 0);
  provider.reset();
  operands->poison_results();
  testing::CudaExternalGate gate;
  auto blocked = blocked_domain->submit({}, nullptr,
                                        [&](void *s) { gate.enqueue_wait(s); });
  second->transaction->reserve();
  const auto foreign_reservation =
      second->transaction->submission_retirement_token();
  const auto next = owner->next_submission_sequence();
  EXPECT_THROW(recipe.submit({blocked}), std::runtime_error);
  EXPECT_EQ(owner->next_submission_sequence(), next);
  EXPECT_TRUE(domain->snapshot().accepting);
  EXPECT_EQ(first->transaction->cancelled.load(), 1);
  EXPECT_EQ(first->transaction->retired.load(), 0);
  EXPECT_EQ(first->calls, 0);
  EXPECT_EQ(second->calls, 0);
  EXPECT_EQ(second->transaction->submission_retirement_token(),
            foreign_reservation);
  EXPECT_EQ(second->transaction->cancelled.load(), 0);
  second->transaction->cancel();
  auto sentinel = recipe.submit();
  ASSERT_TRUE(poll(sentinel));
  EXPECT_FALSE(blocked.done());
  EXPECT_EQ(first->transaction->retired.load(), 1);
  EXPECT_EQ(second->transaction->retired.load(), 1);
  ASSERT_TRUE(poll(provider_only->submit({sentinel})));
  provider_only.reset();
  EXPECT_FALSE(retained_provider.expired());
  EXPECT_FALSE(gate.timed_out());
  gate.release();
  registry.retire();
  EXPECT_TRUE(retained_provider.expired());
  auto context = CUDAContext::get_instance().get_guard();
  for (int i = 0; i < 4; ++i) {
    CUDADriver::get_instance().memcpy_device_to_host(
        operands->readback[0], operands->address(i), count * sizeof(float));
    for (int j = 0; j < count; ++j)
      EXPECT_FLOAT_EQ(operands->readback[0][j], i == 3 ? 2.f : 1.f);
  }
  EXPECT_THROW(recipe.submit(), std::runtime_error);
}

TEST(CudaExecutionContextAot,
     OrderedPartialFailureRetiresSubmittedAndCancelsSuffix) {
  const auto *folder = std::getenv("TAICHI_F3_CONTEXT_AOT_FOLDER");
  if (!folder || !cuda::ExecutionDomain::available())
    GTEST_SKIP();
  auto operands = std::make_shared<InstanceOperands>(folder, true);
  auto owner = std::make_shared<RuntimeFaultDomain>(Arch::cuda, 836);
  cuda::ExecutionDomainRegistry registry(owner, 2);
  auto domain = registry.create({2, 0, 1});
  auto independent = registry.create({1, 0});
  auto first = std::make_shared<OrderedCopy>(
      operands->device(), operands->address(0), operands->address(1), true);
  auto second = std::make_shared<OrderedCopy>(
      operands->device(), operands->address(1), operands->address(2));
  FixedExecution::CudaExecutionBinding binding{domain, operands};
  auto recipe = std::make_unique<FixedExecution>(
      Arch::cuda, operands->device(), nullptr,
      std::vector<FixedExecution::Operation>{
          FixedExecution::BufferFill{operands->storage[0]->get_ptr(),
                                     count * sizeof(float), 0x3f800000u},
          std::static_pointer_cast<NativeOperation>(first),
          std::static_pointer_cast<NativeOperation>(second),
          FixedExecution::BufferFill{operands->storage[3]->get_ptr(),
                                     count * sizeof(float), 0x40000000u}},
      nullptr, owner, nullptr, nullptr, nullptr, nullptr, &binding);
  operands->poison_results();
  testing::CudaExternalGate gate;
  auto blocked =
      domain->submit({}, nullptr, [&](void *s) { gate.enqueue_wait(s); });
  EXPECT_THROW(recipe->submit(), std::runtime_error);
  EXPECT_FALSE(domain->snapshot().accepting);
  recipe.reset();
  EXPECT_EQ(first->transaction->retired.load(), 0);
  EXPECT_EQ(first->transaction->cancelled.load(), 0);
  EXPECT_EQ(second->transaction->cancelled.load(), 0);
  EXPECT_EQ(first->calls, 1);
  EXPECT_EQ(second->calls, 0);
  auto sentinel = independent->submit({}, nullptr, [&](void *s) {
    CUDADriver::get_instance().memsetd32_async(operands->address(3),
                                               0x40400000u, count, s);
  });
  ASSERT_TRUE(poll(sentinel));
  EXPECT_FALSE(blocked.done());
  EXPECT_FALSE(gate.timed_out());
  gate.release();
  registry.retire();
  EXPECT_EQ(first->transaction->retired.load(), 1);
  EXPECT_EQ(first->transaction->cancelled.load(), 0);
  EXPECT_EQ(second->transaction->cancelled.load(), 1);
  EXPECT_EQ(second->transaction->retired.load(), 0);
  auto context = CUDAContext::get_instance().get_guard();
  for (int i = 0; i < 4; ++i) {
    CUDADriver::get_instance().memcpy_device_to_host(
        operands->readback[0], operands->address(i), count * sizeof(float));
    for (int j = 0; j < count; ++j) {
      if (i == 2)
        EXPECT_TRUE(std::isnan(operands->readback[0][j]));
      else
        EXPECT_FLOAT_EQ(operands->readback[0][j], i == 3 ? 3.f : 1.f);
    }
  }
}

TEST(CudaExecutionContextAot, DenseFieldInstancesPreserveIndependentProgress) {
  const auto *folder = std::getenv("TAICHI_F3_CONTEXT_AOT_FOLDER");
  if (!folder || !cuda::ExecutionDomain::available())
    GTEST_SKIP();
  auto operands = std::make_shared<InstanceOperands>(folder);
  auto owner = std::make_shared<RuntimeFaultDomain>(Arch::cuda, 834);
  cuda::ExecutionDomainRegistry registry(owner, 2);
  auto output = registry.create({2, 0}), physics = registry.create({2, -1});
  LaunchContextBuilder ca(&operands->kernel("field_stage_0")),
      cb(&operands->kernel("field_stage_1"));
  const auto prepare =
      [&](int index, LaunchContextBuilder &context,
          const std::shared_ptr<cuda::ExecutionDomain> &domain) {
        const auto name = "field_stage_" + std::to_string(index);
        operands->dispatch(context, index, name.c_str());
        FixedExecution::CudaExecutionBinding binding{domain, operands};
        return std::make_unique<FixedExecution>(
            Arch::cuda, operands->device(), nullptr,
            std::vector<FixedExecution::Operation>{FixedExecution::Dispatch{
                &operands->kernel(name.c_str()), &context}},
            nullptr, owner, nullptr, nullptr, nullptr, nullptr, &binding);
      };
  auto first = prepare(0, ca, output), second = prepare(1, cb, physics);
  first->submit().wait();
  second->submit().wait();
  operands->poison_results();  // Roots retain one increment, ndarray outputs do
                               // not.
  const auto check = [&](int index, int executions,
                         cuda::ExecutionDomain &domain) {
    domain.submit({}, nullptr, operands->read(index).external).wait();
    const float total = (index == 0 ? 11.f : 36.f) * (count * (count + 1) / 2);
    for (int j = 0; j < count; ++j)
      EXPECT_FLOAT_EQ(operands->readback[index][j], executions * (total + j));
  };
  testing::CudaExternalGate gate;
  auto blocked =
      output->submit({}, nullptr, [&](void *s) { gate.enqueue_wait(s); });
  auto pending = first->submit();
  RuntimeCompletion sentinel;
  {
    ScopedExternalProfilerAnnotation range("F3/dense_field_submit");
    sentinel = second->submit();
  }
  ASSERT_TRUE(poll(sentinel));
  EXPECT_FALSE(pending.done());
  check(1, 2, *physics);
  ca.set_arg_float({1}, -100);
  cb.set_arg_float({1}, -200);
  auto dependent = second->submit({pending});
  EXPECT_FALSE(dependent.done());
  first.reset();
  second.reset();
  EXPECT_FALSE(blocked.done());
  EXPECT_FALSE(gate.timed_out());
  {
    ScopedExternalProfilerAnnotation range("F3/dense_field_release");
    gate.release();
  }
  pending.wait();
  dependent.wait();
  check(0, 2, *output);
  check(1, 3, *physics);
  std::weak_ptr<InstanceOperands> retained = operands;
  operands.reset();
  registry.retire();
  EXPECT_TRUE(retained.expired());
}

// A policy measurement, not an independence guarantee for DMA engines. Some
// drivers serialize two same-direction copies even with independent streams,
// graph instances and allocations. Preserve that observation separately from
// the required compute/Graph isolation assertions above.
TEST(CudaExecutionPolicyAot, CopyEngineProgressBehindExternalGate) {
  const auto *folder = std::getenv("TAICHI_F3_CONTEXT_AOT_FOLDER");
  if (!folder || !cuda::ExecutionDomain::available())
    GTEST_SKIP();
  for (int mask = 0; mask < 16; ++mask) {
    SCOPED_TRACE(mask);
    auto operands = std::make_shared<InstanceOperands>(folder, true);
    auto owner = std::make_shared<RuntimeFaultDomain>(Arch::cuda, 813 + mask);
    cuda::ExecutionDomainRegistry registry(owner, 2);
    auto a = registry.create({3, 0}), b = registry.create({3, 0});
    LaunchContextBuilder ca(&operands->kernel()), cb(&operands->kernel());
    const auto copy = [&](int index) {
      return (mask & 8) ? operands->copy(index) : operands->read(index);
    };
    std::vector<cuda::FixedGraph::Operation> oa{operands->dispatch(ca, 0)},
        ob{operands->dispatch(cb, 1)};
    if ((mask & 1) && !(mask & 4))
      oa.push_back(copy(0));
    if ((mask & 2) && !(mask & 4))
      ob.push_back(copy(1));
    auto ga =
        cuda::FixedGraph::prepare_execution_instance(oa, owner, a, operands);
    auto gb =
        cuda::FixedGraph::prepare_execution_instance(ob, owner, b, operands);
    ga->submit().wait();
    gb->submit().wait();
    operands->poison_results();
    testing::CudaExternalGate gate;
    auto blocked =
        a->submit({}, nullptr, [&](void *s) { gate.enqueue_wait(s); });
    auto pending = ga->submit();
    if ((mask & 1) && (mask & 4))
      pending = a->submit({}, nullptr, copy(0).external);
    auto ready = gb->submit();
    if ((mask & 2) && (mask & 4))
      ready = b->submit({}, nullptr, copy(1).external);
    const bool progressed = poll(ready);
    RecordProperty(
        "copy_mask_" + std::to_string(mask) + "_progress_before_release",
        progressed ? 1 : 0);
    EXPECT_FALSE(pending.done());
    EXPECT_FALSE(gate.timed_out());
    gate.release();
    pending.wait();
    ready.wait();
    for (int index = 0; index < 2; ++index) {
      auto guard = CUDAContext::get_instance().get_guard();
      const bool copied = mask & (1 << index);
      if (!copied || (mask & 8))
        CUDADriver::get_instance().memcpy_device_to_host(
            operands->readback[index],
            operands->address(copied && (mask & 8) ? index + 2 : index),
            count * sizeof(float));
      operands->check(index);
    }
    registry.retire();
  }
}

TEST(CudaExecutionPolicyAot, WarmGraphAdmissionProgressesBeforeRelease) {
  const auto *folder = std::getenv("TAICHI_F3_CONTEXT_AOT_FOLDER");
  if (!folder || !cuda::ExecutionDomain::available())
    GTEST_SKIP();
  auto operands = std::make_shared<InstanceOperands>(folder);
  auto owner = std::make_shared<RuntimeFaultDomain>(Arch::cuda, 832);
  cuda::ExecutionDomainRegistry registry(owner, 2);
  auto output = registry.create({2, 0}), physics = registry.create({8, -1});
  LaunchContextBuilder ca(&operands->kernel()), cb(&operands->kernel());
  auto ga = cuda::FixedGraph::prepare_execution_instance(
      {operands->dispatch(ca, 0)}, owner, output, operands);
  auto gb = cuda::FixedGraph::prepare_execution_instance(
      {operands->dispatch(cb, 1)}, owner, physics, operands);
  for (int i = 0; i < 50; ++i)
    gb->submit().wait();
  operands->poison_results();
  testing::CudaExternalGate gate;
  auto blocked =
      output->submit({}, nullptr, [&](void *s) { gate.enqueue_wait(s); });
  auto pending = ga->submit();
  std::vector<double> durations;
  durations.reserve(1000);
  std::size_t busy = 0;
  RuntimeCompletion last;
  {
    ScopedExternalProfilerAnnotation range(
        "F3/warm_graph_progress_before_release");
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (durations.size() < 1000 &&
           std::chrono::steady_clock::now() < deadline) {
      const auto begin = std::chrono::steady_clock::now();
      try {
        last = gb->submit();
        durations.push_back(std::chrono::duration<double, std::micro>(
                                std::chrono::steady_clock::now() - begin)
                                .count());
      } catch (const cuda::ExecutionDomainBusy &) {
        ++busy;
        std::this_thread::sleep_for(std::chrono::microseconds(50));
      }
    }
    EXPECT_TRUE(last.valid() && poll(last));
    EXPECT_FALSE(blocked.done());
    EXPECT_FALSE(pending.done());
    EXPECT_FALSE(gate.timed_out());
  }
  {
    ScopedExternalProfilerAnnotation range("F3/warm_graph_release");
    gate.release();
  }
  pending.wait();
  if (last.valid())
    last.wait();
  output->submit({}, nullptr, operands->read(0).external).wait();
  physics->submit({}, nullptr, operands->read(1).external).wait();
  operands->check(0);
  operands->check(1);
  ASSERT_EQ(durations.size(), 1000);
  std::sort(durations.begin(), durations.end());
  RecordProperty("submit_p50_us", std::to_string(durations[499]));
  RecordProperty("submit_p95_us", std::to_string(durations[949]));
  RecordProperty("busy_retries", std::to_string(busy));
  RecordProperty("effective_priority", physics->snapshot().effective_priority);
  RecordProperty("graph_nodes", static_cast<int>(gb->node_count()));
  std::printf(
      "F3 warm graph admission: n=1000 p50=%.3f us p95=%.3f us busy=%zu\n",
      durations[499], durations[949], busy);
  registry.retire();
}
}  // namespace
}  // namespace taichi::lang
#endif
