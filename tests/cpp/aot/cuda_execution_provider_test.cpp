#include <gtest/gtest.h>

#if defined(TI_WITH_CUDA) && defined(TI_WITH_VULKAN)
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include "tests/cpp/rhi/common/cuda_external_gate.h"
#include "taichi/program/runtime_fault.h"
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/runtime/cuda/execution_domain.h"
#include "taichi/runtime/fixed_execution.h"
#include "taichi/runtime/native_operation_registry.h"
#include "taichi/system/profiler_annotation.h"

namespace taichi::lang {
namespace {
TEST(CudaExecutionProvider, AffinityLeaseRollsBackAndSharesOnlyOneDomain) {
  CudaProviderCompletionResource resource;
  auto legacy = resource.bind_execution_domain(0);
  EXPECT_THROW(resource.bind_execution_domain(1), std::invalid_argument);
  legacy.reset();
  auto first = resource.bind_execution_domain(1);
  auto second = resource.bind_execution_domain(1);
  EXPECT_EQ(first, second);
  first.reset();
  EXPECT_THROW(resource.bind_execution_domain(2), std::invalid_argument);
  EXPECT_THROW(resource.bind_execution_domain(0), std::invalid_argument);
  second.reset();
  EXPECT_NO_THROW(resource.bind_execution_domain(2));
  EXPECT_NO_THROW(resource.bind_execution_domain(0));
}

struct SolverOperands final : RuntimeCompletionResources {
  struct Output {
    float solution[6];
    int status[2];
  };
  cuda::CudaDevice device;
  std::vector<DeviceAllocationUnique> storage;
  std::array<Output *, 2> outputs{};
  std::size_t retained_resource_count(
      std::uint32_t kind) const noexcept override {
    return kind == 193 ? storage.size() : 0;
  }
  SolverOperands() {
    auto guard = CUDAContext::get_instance().get_guard();
    auto &driver = CUDADriver::get_instance();
    for (int plan = 0; plan < 2; ++plan) {
      // Input, factor, RHS, solution and status are all disjoint.
      for (int elements : {9, 9, 6, 6, 2}) {
        auto allocation = device.allocate_memory_unique({size_t(elements * 4)});
        if (allocation.second != RhiResult::success)
          throw std::runtime_error("Cannot allocate provider fixture");
        storage.push_back(std::move(allocation.first));
      }
      std::array<float, 9> a{4, 1, 0, 1, 3, 1, 0, 1, 2};
      std::array<float, 6> rhs{};
      for (int r = 0; r < 2; ++r)
        for (int i = 0; i < 3; ++i)
          for (int j = 0; j < 3; ++j)
            rhs[r * 3 + i] += a[i * 3 + j] * expected(plan, r, j);
      driver.memcpy_host_to_device(address(plan, 0), a.data(), sizeof(a));
      driver.memcpy_host_to_device(address(plan, 2), rhs.data(), sizeof(rhs));
      driver.mem_host_alloc(reinterpret_cast<void **>(&outputs[plan]),
                            sizeof(Output), 0);
    }
    driver.stream_synchronize(nullptr);
  }
  ~SolverOperands() override {
    auto guard = CUDAContext::get_instance().get_guard();
    for (auto output : outputs)
      if (output)
        CUDADriver::get_instance().mem_free_host(output);
  }
  static float expected(int plan, int rhs, int index) {
    return float((plan + 1) * (rhs + 1) * (index + 1));
  }
  void *address(int plan, int port) {
    return device.get_memory_addr(*storage[plan * 5 + port]);
  }
  NativeArrayBinding array(int plan, int port) {
    const auto shape = port < 2   ? std::vector<uint64_t>{3, 3}
                       : port < 4 ? std::vector<uint64_t>{2, 3}
                                  : std::vector<uint64_t>{2};
    const uint64_t elements = port < 2 ? 9 : port < 4 ? 6 : 2;
    return {storage[plan * 5 + port]->get_ptr(),
            elements * 4,
            port == 4 ? PrimitiveType::i32 : PrimitiveType::f32,
            shape,
            {}};
  }
  void read(int plan, cuda::ExecutionDomain &domain) {
    domain
        .submit({}, nullptr,
                [&](void *stream) {
                  auto &driver = CUDADriver::get_instance();
                  driver.memcpy_device_to_host_async(
                      outputs[plan]->solution, address(plan, 3), 24, stream);
                  driver.memcpy_device_to_host_async(
                      outputs[plan]->status, address(plan, 4), 8, stream);
                })
        .wait();
    EXPECT_EQ(outputs[plan]->status[0], 0);
    EXPECT_EQ(outputs[plan]->status[1], 0);
    for (int r = 0; r < 2; ++r)
      for (int i = 0; i < 3; ++i)
        EXPECT_NEAR(outputs[plan]->solution[r * 3 + i], expected(plan, r, i),
                    2e-5f);
  }
};

bool ready(RuntimeCompletion &completion) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!completion.done() && std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  return completion.done();
}

TEST(CudaExecutionProvider,
     CusolverRetainedRootsProgressAndPauseIndependently) {
  const auto *adapter = std::getenv("TI_CUSOLVERDN_TEST_ADAPTER");
  const auto *library = std::getenv("TI_FORGE_TEST_CUSOLVERDN_LIBRARY_PATH");
  if (!adapter || !library || !cuda::ExecutionDomain::available())
    GTEST_SKIP() << "Requires real cuSOLVERDn adapter and vendor library";
  auto operands = std::make_shared<SolverOperands>();
  auto owner = std::make_shared<RuntimeFaultDomain>(Arch::cuda, 857);
  cuda::ExecutionDomainRegistry domains(owner, 2);
  auto output = domains.create({2, 0}), physics = domains.create({2, -1});
  NativePreparationContext context(Arch::cuda, operands->device,
                                   {{"cusolverdn", {adapter, library}}}, owner);
  auto &registry = native_operation_registry();
  const auto plan_desc = registry.decode(
      {"cusolverdn",
       "cholesky_plan",
       1,
       liong::json::parse(
           R"({"rows":3,"rhs_count":2,"dtype":"f32","seeded":false})"),
       {{"factor", "factor"}, {"status", "status"}}},
      NativeObjectKind::resource);
  const auto op_desc =
      registry.decode({"cusolverdn",
                       "cholesky",
                       1,
                       liong::json::parse(R"({"phase":"factor_and_solve"})"),
                       {{"plan", "plan"},
                        {"a", "a"},
                        {"rhs", "rhs"},
                        {"solution", "solution"}}});
  std::array<std::shared_ptr<NativeResource>, 2> plans;
  std::array<std::shared_ptr<NativeOperation>, 2> operations;
  for (int i = 0; i < 2; ++i) {
    plans[i] = registry.prepare_resource(
        context, plan_desc,
        {{"factor", operands->array(i, 1)}, {"status", operands->array(i, 4)}});
    operations[i] = registry.prepare(context, op_desc,
                                     {{"plan", plans[i]},
                                      {"a", operands->array(i, 0)},
                                      {"rhs", operands->array(i, 2)},
                                      {"solution", operands->array(i, 3)}});
  }
  auto alias = registry.prepare(context, op_desc,
                                {{"plan", plans[0]},
                                 {"a", operands->array(0, 0)},
                                 {"rhs", operands->array(0, 2)},
                                 {"solution", operands->array(0, 3)}});
  const auto prepare = [&](int index,
                           std::shared_ptr<cuda::ExecutionDomain> domain,
                           std::shared_ptr<NativeOperation> operation = {}) {
    FixedExecution::CudaExecutionBinding binding{std::move(domain), operands};
    return std::make_unique<FixedExecution>(
        Arch::cuda, operands->device, nullptr,
        std::vector<FixedExecution::Operation>{operation ? operation
                                                         : operations[index]},
        nullptr, owner, nullptr, nullptr, nullptr, nullptr, &binding);
  };
  // A default execution claim blocks migration before any domain preparation.
  auto legacy = std::make_unique<FixedExecution>(
      Arch::cuda, operands->device, nullptr,
      std::vector<FixedExecution::Operation>{operations[0]}, nullptr, owner);
  EXPECT_THROW(prepare(0, output), std::invalid_argument);
  legacy.reset();
  auto first = prepare(0, output), second = prepare(1, physics);
  EXPECT_EQ(first->execution_kind(), 4);
  EXPECT_EQ(second->execution_kind(), 4);
  EXPECT_THROW(prepare(0, physics), std::invalid_argument);
  EXPECT_THROW(prepare(0, physics, alias), std::invalid_argument);
  auto shared_plan = prepare(0, output, alias);
  // Each retained executable belongs to one instance, even on the same domain.
  EXPECT_THROW(prepare(0, output), std::invalid_argument);
  first->submit().wait();
  shared_plan->submit().wait();
  second->submit().wait();
  operands->read(0, *output);
  operands->read(1, *physics);
  testing::CudaExternalGate gate;
  auto blocked =
      output->submit({}, nullptr, [&](void *s) { gate.enqueue_wait(s); });
  auto pending = first->submit();
  RuntimeCompletion sentinel;
  {
    ScopedExternalProfilerAnnotation range("F3/cusolver_domain_submit");
    sentinel = second->submit();
  }
  ASSERT_TRUE(ready(sentinel));
  EXPECT_FALSE(pending.done());
  operands->read(1, *physics);
  std::vector<double> durations;
  durations.reserve(1000);
  std::size_t busy = 0;
  RuntimeCompletion last;
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  {
    ScopedExternalProfilerAnnotation range("F3/cusolver_domain_warm");
    while (durations.size() < 1000 &&
           std::chrono::steady_clock::now() < deadline) {
      const auto start = std::chrono::steady_clock::now();
      try {
        last = second->submit();
        durations.push_back(std::chrono::duration<double, std::micro>(
                                std::chrono::steady_clock::now() - start)
                                .count());
      } catch (const cuda::ExecutionDomainBusy &) {
        ++busy;
        std::this_thread::yield();
      }
    }
  }
  if (last.valid())
    last.wait();
  ASSERT_EQ(durations.size(), 1000);
  std::sort(durations.begin(), durations.end());
  RecordProperty("submit_p50_us", std::to_string(durations[499]));
  RecordProperty("submit_p95_us", std::to_string(durations[949]));
  RecordProperty("busy_retries", std::to_string(busy));
  std::printf(
      "F3 cuSOLVER admission: n=1000 p50=%.3f us p95=%.3f us busy=%zu\n",
      durations[499], durations[949], busy);
  operands->read(1, *physics);
  EXPECT_FALSE(gate.timed_out());
  EXPECT_FALSE(pending.done());
  // Pausing a live domain retains bounded recordings. Destructive vendor
  // teardown may synchronize the context and belongs to cold owner retirement.
  second.reset();
  operations[1].reset();
  plans[1].reset();
  {
    ScopedExternalProfilerAnnotation range("F3/cusolver_domain_pause");
    physics->stop_accepting();
    physics->latest_completion().wait();
  }
  EXPECT_FALSE(gate.timed_out());
  EXPECT_FALSE(pending.done());
  EXPECT_THROW(physics->submit({}, nullptr, [](void *) {}), std::runtime_error);
  std::weak_ptr<NativeOperation> retained = operations[0];
  auto provider = operations[0]->cuda_ordered_resource();
  first.reset();
  shared_plan.reset();
  alias.reset();
  operations = {};
  plans = {};
  EXPECT_FALSE(retained.expired());
  EXPECT_THROW(provider->bind_execution_domain(physics->snapshot().identity),
               std::invalid_argument);
  {
    ScopedExternalProfilerAnnotation range("F3/cusolver_domain_release");
    gate.release();
  }
  pending.wait();
  operands->read(0, *output);
  domains.retire();
  EXPECT_TRUE(retained.expired());
  EXPECT_NO_THROW(
      provider->bind_execution_domain(physics->snapshot().identity));
}
}  // namespace
}  // namespace taichi::lang
#endif
