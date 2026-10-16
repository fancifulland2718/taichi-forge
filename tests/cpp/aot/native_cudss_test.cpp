#include "gtest/gtest.h"
#include "taichi/runtime/native_cudss.h"
#include "taichi/runtime/native_csr_contract.h"
#include "taichi/runtime/native_operation_registry.h"
#include "taichi/runtime/fixed_execution.h"

#ifdef TI_WITH_CUDA
#include <array>
#include <cstdlib>
#include "taichi/program/runtime_fault.h"
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/runtime/cuda/cudss_plan.h"
#endif

namespace taichi::lang {
namespace {
NativeOperationDescription cudss_description(const char *text) {
  return NativeOperationDescription::parse(liong::json::parse_complete(text));
}

TEST(NativeCudss, SchemasDecodeWithoutProviderOrDevice) {
  const auto &registry = native_operation_registry();
  auto plan = cudss_description(R"({"kind":"native","provider":"cudss",
    "operation":"plan_f32","version":1,"parameters":{"rows":4,"nonzeros":10,
    "matrix_type":"spd","phase":"refactor_solve","reordering":"nested_dissection"},
    "bindings":{"row_offsets":"r","column_indices":"c","values":"v"}})");
  EXPECT_EQ(registry.decode(plan, NativeObjectKind::resource).kind(),
            CudssPlanDescription::kind);
  EXPECT_THROW(registry.decode(plan), std::invalid_argument);
  for (const char *patch :
       {R"({"rows":0})", R"({"rows":true})", R"({"rows":-1})",
        R"({"nonzeros":0})", R"({"nonzeros":2147483648})",
        R"({"phase":"analysis"})", R"({"matrix_type":"hermitian"})",
        R"({"matrix_type":"general","matrix_view":"lower"})",
        R"({"matrix_view":"auto"})", R"({"reordering":"nd"})", R"({"solve":1})",
        R"({"stride":1})"}) {
    auto bad = plan;
    for (const auto &[name, value] :
         liong::json::parse_complete(patch).obj.inner)
      bad.parameters.obj.inner[name] = value;
    EXPECT_THROW(registry.decode(bad, NativeObjectKind::resource),
                 std::invalid_argument)
        << patch;
  }
  auto group = cudss_description(R"({"kind":"native","provider":"cudss",
    "operation":"solve_group_f32","version":1,"parameters":{"rhs_count":2},
    "bindings":{"plan":"p","values":"v","rhs_0":"b","rhs_1":"b",
                "output_0":"x","output_1":"y"}})");
  EXPECT_EQ(registry.decode(group).kind(), CudssSolveGroupOperation::kind);
  group.bindings.erase("values");
  EXPECT_NO_THROW(registry.decode(group));
  for (const auto *value : {"true", "false"}) {
    group.parameters.obj.inner["disjoint_bindings"] =
        liong::json::parse_complete(value);
    EXPECT_NO_THROW(registry.decode(group));
  }
  for (const auto *value : {"0", "1", "null", "\"true\""}) {
    group.parameters.obj.inner["disjoint_bindings"] =
        liong::json::parse_complete(value);
    EXPECT_THROW(registry.decode(group), std::invalid_argument);
  }
  group.parameters.obj.inner.erase("disjoint_bindings");
  group.bindings = {{"plan", "p"}};
  group.parameters = liong::json::parse_complete(R"({"rhs_count":0})");
  EXPECT_THROW(registry.decode(group), std::invalid_argument);
  for (const auto *kind :
       {CudssPlanDescription::kind, CudssSolveGroupOperation::kind}) {
    EXPECT_EQ(registry.find(kind, 1, host_arch()), nullptr);
    EXPECT_EQ(registry.find(kind, 1, Arch::vulkan), nullptr);
  }
}

TEST(NativeCudss, CsrContractAllowsUnsortedRowsAndRejectsDuplicateEntries) {
  auto validate = [](std::vector<int32_t> rows, std::vector<int32_t> columns) {
    validate_native_csr_indices(rows, columns, 4);
  };
  EXPECT_NO_THROW(validate({0, 3, 3}, {2, 0, 1}));
  EXPECT_THROW(validate({}, {}), std::invalid_argument);
  EXPECT_THROW(validate({1, 2}, {0, 1}), std::invalid_argument);
  EXPECT_THROW(validate({0, 3}, {0, 1}), std::invalid_argument);
  EXPECT_THROW(validate({0, 2, 1, 3}, {0, 1, 2}), std::invalid_argument);
  EXPECT_THROW(validate({0, 2}, {0, 0}), std::invalid_argument);
  EXPECT_THROW(validate({0, 3}, {2, 0, 2}), std::invalid_argument);
  EXPECT_THROW(validate({0, 2}, {0, 4}), std::invalid_argument);
  EXPECT_THROW(validate({0, 2}, {0, -1}), std::invalid_argument);
}

TEST(NativeCudss, TransactionSchemasAreDistinctFromCapturedGroups) {
  const auto &registry = native_operation_registry();
  auto plan = cudss_description(R"({"kind":"native","provider":"cudss",
    "operation":"transaction_plan_f32","version":1,"parameters":{"rows":4,"nonzeros":10},
    "bindings":{"row_offsets":"r","column_indices":"c","values":"v"}})");
  EXPECT_EQ(registry.decode(plan, NativeObjectKind::resource).kind(),
            CudssTransactionPlanDescription::kind);
  plan.parameters.obj.inner["phase"] = liong::json::JsonValue("refactor_solve");
  EXPECT_THROW(registry.decode(plan, NativeObjectKind::resource),
               std::invalid_argument);
  auto operation = cudss_description(R"({"kind":"native","provider":"cudss",
    "operation":"refactor_solve_f32","version":1,"parameters":{},
    "bindings":{"plan":"p","values":"v","rhs":"b","output":"x"}})");
  EXPECT_EQ(registry.decode(operation).kind(),
            CudssRefactorSolveOperation::kind);
  operation.bindings.erase("values");
  EXPECT_THROW(registry.decode(operation), std::invalid_argument);
  for (const auto *kind : {CudssTransactionPlanDescription::kind,
                           CudssRefactorSolveOperation::kind}) {
    EXPECT_EQ(registry.find(kind, 1, host_arch()), nullptr);
    EXPECT_EQ(registry.find(kind, 1, Arch::vulkan), nullptr);
  }
}

#ifdef TI_WITH_CUDA
TEST(NativeCudss, OrderedTransactionsFreezePatternAndRetireFailures) {
  const auto *library = std::getenv("TI_CUDSS_TEST_LIBRARY");
  const auto *adapter = std::getenv("TI_CUDSS_TEST_ADAPTER");
  if (!library || !adapter)
    GTEST_SKIP() << "Set cuDSS runtime and adapter paths";
  cuda::CudaDevice device;
  auto guard = CUDAContext::get_instance().get_guard();
  auto submission = CUDAContext::get_instance().get_submission_lock_guard();
  auto &driver = CUDADriver::get_instance();
  auto fault = std::make_shared<RuntimeFaultDomain>(Arch::cuda, 32);
  NativePreparationContext context(Arch::cuda, device,
                                   {{"cudss", {adapter, library}}}, fault);
  const auto &registry = native_operation_registry();
  std::vector<DeviceAllocationUnique> allocations;
  auto allocate = [&](DataType type, uint64_t count) {
    auto result = device.allocate_memory_unique({std::size_t(count * 4)});
    if (result.second != RhiResult::success)
      throw std::runtime_error("Test CUDA allocation failed");
    NativeArrayBinding array{
        result.first->get_ptr(), count * 4, type, {count}, {}};
    allocations.push_back(std::move(result.first));
    return array;
  };
  auto ptr = [&](const NativeArrayBinding &array) {
    return device.get_memory_addr(array.pointer);
  };
  auto r = allocate(PrimitiveType::i32, 5),
       c = allocate(PrimitiveType::i32, 10);
  auto v = allocate(PrimitiveType::f32, 10),
       seed = allocate(PrimitiveType::f32, 10);
  auto b = allocate(PrimitiveType::f32, 4), x = allocate(PrimitiveType::f32, 4);
  std::array<int32_t, 5> rows{0, 2, 5, 8, 10};
  std::array<int32_t, 10> columns{0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
  std::array<float, 10> values{4, -1, -1, 4, -1, -1, 4, -1, -1, 3};
  std::array<float, 4> rhs{2, 4, 6, 9};
  driver.memcpy_host_to_device(ptr(r), rows.data(), sizeof(rows));
  driver.memcpy_host_to_device(ptr(c), columns.data(), sizeof(columns));
  driver.memsetd32(ptr(seed), 0,
                   10);  // Analysis seed need not have valid factors.
  driver.memcpy_host_to_device(ptr(b), rhs.data(), sizeof(rhs));
  driver.memsetd32(ptr(x), 0, 4);
  auto description = cudss_description(R"({"kind":"native","provider":"cudss",
    "operation":"transaction_plan_f32","version":1,"parameters":{"rows":4,"nonzeros":10,"matrix_type":"spd"},
    "bindings":{"row_offsets":"r","column_indices":"c","values":"v"}})");
  auto resource = registry.prepare_resource(
      context, registry.decode(description, NativeObjectKind::resource),
      {{"row_offsets", r}, {"column_indices", c}, {"values", seed}});
  EXPECT_EQ(resource->workspace_bytes(), 100);
  auto operation = cudss_description(R"({"kind":"native","provider":"cudss",
    "operation":"refactor_solve_f32","version":1,"parameters":{},
    "bindings":{"plan":"p","values":"v","rhs":"b","output":"x"}})");
  NativeOperationBindings bindings{
      {"plan", resource}, {"values", v}, {"rhs", b}, {"output", x}};
  auto op = registry.prepare(context, registry.decode(operation), bindings);
  auto owner =
      std::dynamic_pointer_cast<CudaCudssPlan>(op->cuda_ordered_resource());
  ASSERT_NE(owner, nullptr);
  EXPECT_EQ(owner->statistics().at("factorized"), 0);
  EXPECT_EQ(op->execution_mode(), NativeOperationMode::cuda_ordered);
  auto alias = bindings;
  alias.at("output") = b;
  EXPECT_THROW(registry.prepare(context, registry.decode(operation), alias),
               std::invalid_argument);
  alias = bindings;
  auto bad = v;
  bad.shape = {2, 5};
  alias.at("values") = bad;
  EXPECT_THROW(registry.prepare(context, registry.decode(operation), alias),
               std::invalid_argument);
  auto make_execution = [&] {
    return std::make_unique<FixedExecution>(
        Arch::cuda, device, nullptr, std::vector<FixedExecution::Operation>{op},
        nullptr, fault);
  };
  EXPECT_THROW((FixedExecution(Arch::cuda, device, nullptr,
                               std::vector<FixedExecution::Operation>{op, op},
                               nullptr, fault)),
               std::invalid_argument);
  auto execution = make_execution();
  driver.memsetd32(ptr(r), 0, 5);
  driver.memsetd32(ptr(c), 0, 10);  // Input topology is no longer referenced.
  auto check = [&](float scale) {
    std::array<float, 4> result{};
    driver.stream_synchronize(nullptr);
    driver.memcpy_device_to_host(result.data(), ptr(x), sizeof(result));
    for (int i = 0; i < 4; ++i)
      EXPECT_NEAR(result[i], (i + 1) * scale, 3e-5f);
  };
  check(0);  // Preparation did not solve or write outputs.
  driver.memcpy_host_to_device(ptr(v), values.data(), sizeof(values));
  execution->run();
  check(1);
  for (auto &value : values)
    value *= 2;
  driver.memcpy_host_to_device(ptr(v), values.data(), sizeof(values));
  execution->run();  // Polls and retires the preceding completed transaction.
  check(0.5f);
  execution.reset();
  EXPECT_EQ(owner->statistics().at("refactor_solve_retirements"), 2);
  owner->debug_fail_next_refactor_solve();
  execution = make_execution();
  EXPECT_ANY_THROW(execution->run());
  EXPECT_THROW(execution->run(), std::runtime_error);
  EXPECT_EQ(owner->statistics().at("factorized"), 0);
  execution.reset();
  EXPECT_EQ(owner->statistics().at("refactor_solve_inflight"), 0);
  EXPECT_EQ(owner->statistics().at("refactor_solve_retirements"), 3);
  execution = make_execution();
  execution->run();  // Failed factors require full factorization again.
  check(0.5f);
  execution.reset();
  EXPECT_EQ(owner->statistics().at("refactor_solve_successes"), 3);
  EXPECT_EQ(owner->statistics().at("refactor_solve_retirements"), 4);
}

TEST(NativeCudss, RebuildsFrozenPlansAndMultiRhsGroupsThroughTheRegistry) {
  const auto *library = std::getenv("TI_CUDSS_TEST_LIBRARY");
  const auto *adapter = std::getenv("TI_CUDSS_TEST_ADAPTER");
  if (!library || !adapter)
    GTEST_SKIP() << "Set cuDSS runtime and adapter paths";
  ASSERT_TRUE(CUDADriver::get_instance_without_context().detected());
  cuda::CudaDevice device;
  auto guard = CUDAContext::get_instance().get_guard();
  auto submission = CUDAContext::get_instance().get_submission_lock_guard();
  auto &driver = CUDADriver::get_instance();
  const auto &registry = native_operation_registry();
  const NativeProviderLocations locations{
      {"cudss", {adapter, library}}, {"unused", {"absent-adapter.dll", ""}}};
  auto fault = std::make_shared<RuntimeFaultDomain>(Arch::cuda, 31);
  std::vector<DeviceAllocationUnique> allocations;
  auto allocate = [&](DataType type, uint64_t count) {
    auto result = device.allocate_memory_unique({std::size_t(count * 4)});
    if (result.second != RhiResult::success)
      throw std::runtime_error("Test CUDA allocation failed");
    NativeArrayBinding array{
        result.first->get_ptr(), count * 4, type, {count}, {}};
    allocations.push_back(std::move(result.first));
    return array;
  };
  auto ptr = [&](const NativeArrayBinding &array) {
    return static_cast<char *>(device.get_memory_addr(array.pointer)) +
           array.pointer.offset;
  };
  auto r = allocate(PrimitiveType::i32, 5),
       c = allocate(PrimitiveType::i32, 10);
  auto v = allocate(PrimitiveType::f32, 10),
       v2 = allocate(PrimitiveType::f32, 10);
  auto b = allocate(PrimitiveType::f32, 4),
       b2 = allocate(PrimitiveType::f32, 4);
  auto x = allocate(PrimitiveType::f32, 4),
       x2 = allocate(PrimitiveType::f32, 4);
  auto x3 = allocate(PrimitiveType::f32, 4);
  // Row 0 is deliberately unsorted; its values follow the same permutation.
  std::array<int32_t, 5> rows{0, 2, 5, 8, 10};
  std::array<int32_t, 10> columns{1, 0, 0, 1, 2, 1, 2, 3, 2, 3};
  std::array<float, 10> values{-1, 4, -1, 4, -1, -1, 4, -1, -1, 3};
  std::array<float, 4> rhs{2, 4, 6, 9}, rhs2{4, 8, 12, 18};
  driver.memcpy_host_to_device(ptr(r), rows.data(), sizeof(rows));
  driver.memcpy_host_to_device(ptr(c), columns.data(), sizeof(columns));
  driver.memcpy_host_to_device(ptr(b), rhs.data(), sizeof(rhs));
  driver.memcpy_host_to_device(ptr(b2), rhs2.data(), sizeof(rhs2));
  auto plan_description =
      cudss_description(R"({"kind":"native","provider":"cudss",
    "operation":"plan_f32","version":1,"parameters":{"rows":4,"nonzeros":10,"matrix_type":"spd"},
    "bindings":{"row_offsets":"r","column_indices":"c","values":"v"}})");
  auto group_description =
      cudss_description(R"({"kind":"native","provider":"cudss",
    "operation":"solve_group_f32","version":1,"parameters":{"rhs_count":2},
    "bindings":{"plan":"p","rhs_0":"b","rhs_1":"b2","output_0":"x","output_1":"x2"}})");
  const NativeOperationBindings seed{
      {"row_offsets", r}, {"column_indices", c}, {"values", v}};
  auto check = [&](const NativeArrayBinding &array, float scale) {
    std::array<float, 4> result{};
    driver.context_synchronize();
    driver.memcpy_device_to_host(result.data(), ptr(array), sizeof(result));
    for (int i = 0; i < 4; ++i)
      EXPECT_NEAR(result[i], (i + 1) * scale, 2e-5f);
  };
  NativeProviderSession missing({{"cudss", {"absent-adapter.dll", library}}});
  EXPECT_FALSE(registry
                   .probe(CudssPlanDescription::kind, 1, Arch::cuda, missing,
                          NativeObjectKind::resource)
                   .available);
  for (const auto *phase : {"solve", "factor_solve", "refactor_solve"}) {
    SCOPED_TRACE(phase);
    const bool update = std::string(phase) != "solve";
    auto context = std::make_unique<NativePreparationContext>(
        Arch::cuda, device, locations, fault);
    EXPECT_TRUE(registry
                    .probe(CudssPlanDescription::kind, 1, Arch::cuda,
                           context->providers, NativeObjectKind::resource)
                    .available);
    std::weak_ptr<NativeProviderRuntime> provider =
        context->providers.runtime(cuda_cudss_provider_definition());
    plan_description.parameters.obj.inner["phase"] =
        liong::json::JsonValue(phase);
    const auto decoded =
        registry.decode(plan_description, NativeObjectKind::resource);
    driver.memcpy_host_to_device(ptr(v), values.data(), sizeof(values));
    for (const auto &output : {x, x2, x3})
      driver.memsetd32(ptr(output), 0, 4);
    if (!update) {
      NativePreparationContext ownerless(Arch::cuda, device, locations);
      EXPECT_THROW(registry.prepare_resource(ownerless, decoded, seed),
                   std::invalid_argument);
      auto invalid_columns = columns;
      invalid_columns[0] = 0;
      driver.memcpy_host_to_device(ptr(c), invalid_columns.data(),
                                   sizeof(columns));
      EXPECT_THROW(registry.prepare_resource(*context, decoded, seed),
                   std::invalid_argument);
      driver.memcpy_host_to_device(ptr(c), columns.data(), sizeof(columns));
      auto bad_seed = seed;
      auto bad_rows = r;
      bad_rows.scalar_type = PrimitiveType::f32;
      bad_seed.at("row_offsets") = bad_rows;
      EXPECT_THROW(registry.prepare_resource(*context, decoded, bad_seed),
                   std::invalid_argument);
    }
    auto plan = registry.prepare_resource(*context, decoded, seed);
    EXPECT_GT(plan->workspace_bytes(), 0);
    std::weak_ptr<NativeResource> plan_lease = plan;
    NativeOperationBindings bindings{{"plan", plan},
                                     {"rhs_0", b},
                                     {"rhs_1", b2},
                                     {"output_0", x},
                                     {"output_1", x2}};
    if (update) {
      group_description.bindings["values"] = "v";
      bindings.emplace("values", v);
    } else {
      group_description.bindings.erase("values");
    }
    const auto group = registry.decode(group_description);
    auto first = registry.prepare(*context, group, bindings);
    NativeRecordingState state;
    EXPECT_NO_THROW(first->validate_recording(state));
    auto branch = state.control_branch();
    EXPECT_NO_THROW(first->validate_recording(branch));
    auto alias = bindings;
    alias.at("output_1") = x;
    EXPECT_THROW(registry.prepare(*context, group, alias),
                 std::invalid_argument);
    alias = bindings;
    alias.at("output_1") = b;
    EXPECT_THROW(registry.prepare(*context, group, alias),
                 std::invalid_argument);
    alias = bindings;
    alias.at("rhs_1") = b;  // Repeated read-only RHS storage is legal.
    EXPECT_NO_THROW(registry.prepare(*context, group, alias));
    auto strict_description = group_description;
    strict_description.parameters.obj.inner["disjoint_bindings"] =
        liong::json::parse_complete("true");
    const auto strict = registry.decode(strict_description);
    EXPECT_THROW(registry.prepare(*context, strict, alias),
                 std::invalid_argument);
    EXPECT_NO_THROW(registry.prepare(*context, strict, bindings));
    alias.clear();
    auto wrong_phase = group_description;
    auto wrong_bindings = bindings;
    if (update) {
      wrong_phase.bindings.erase("values");
      wrong_bindings.erase("values");
    } else {
      wrong_phase.bindings["values"] = "v";
      wrong_bindings.emplace("values", v);
    }
    EXPECT_THROW(registry.prepare(*context, registry.decode(wrong_phase),
                                  wrong_bindings),
                 std::invalid_argument);
    wrong_bindings.clear();
    auto second_description = group_description;
    second_description.parameters =
        liong::json::parse_complete(R"({"rhs_count":1})");
    second_description.bindings.erase("rhs_1");
    second_description.bindings.erase("output_1");
    bindings.erase("rhs_1");
    bindings.erase("output_1");
    bindings.at("output_0") = x3;
    if (update)
      bindings.at("values") = v2;
    auto second = registry.prepare(
        *context, registry.decode(second_description), bindings);
    auto graph1 = std::make_unique<FixedExecution>(
        Arch::cuda, device, nullptr,
        std::vector<FixedExecution::Operation>{first}, nullptr, fault);
    auto graph2 = std::make_unique<FixedExecution>(
        Arch::cuda, device, nullptr,
        std::vector<FixedExecution::Operation>{second}, nullptr, fault);
    EXPECT_GT(graph1->workspace_bytes(), 0);
    check(x, 0);
    check(x2, 0);
    check(x3, 0);  // Preparing must not run user outputs.
    auto twice = values, four_times = values;
    for (auto &value : twice)
      value *= 2;
    for (auto &value : four_times)
      value *= 4;
    driver.memcpy_host_to_device(ptr(v), twice.data(), sizeof(twice));
    driver.memcpy_host_to_device(ptr(v2), four_times.data(),
                                 sizeof(four_times));
    context.reset();
    bindings.clear();
    plan.reset();
    first.reset();
    second.reset();
    EXPECT_FALSE(provider.expired());
    for (int repeat = 0; repeat < 3; ++repeat) {
      graph1->run();
      check(x, update ? .5f : 1.f);
      check(x2, update ? 1.f : 2.f);
      graph2->run();
      check(x3, update ? .25f : 1.f);
    }
    graph1->run();
    graph1.reset();  // Retirement with work in flight.
    EXPECT_FALSE(plan_lease.expired());
    graph2.reset();
    EXPECT_TRUE(plan_lease.expired());
    EXPECT_TRUE(provider.expired());
    EXPECT_TRUE(fault->submission_allowed());
  }
}
#endif
}  // namespace
}  // namespace taichi::lang
