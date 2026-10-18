#include <gtest/gtest.h>

#if defined(TI_WITH_CUDA) && defined(TI_WITH_VULKAN) && \
    defined(TI_TEST_RUNTIME_C_API)
#include <future>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <thread>
#include "tests/cpp/rhi/common/cuda_external_gate.h"
#include "c_api/src/taichi_llvm_impl.h"
#include "c_api/src/execution_domain.h"
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/program/program.h"
#include "taichi/program/graph_builder.h"
#include "taichi/ir/ir_builder.h"
#include "taichi/runtime/aot_runtime_client.h"
#include "taichi/runtime/program_impls/llvm/llvm_program.h"
#include "taichi/runtime/cuda/execution_domain.h"
#include "taichi/system/profiler_annotation.h"

namespace taichi::lang {
namespace {

TEST(CudaPublicExecutionAot, ExactDependenciesBusyAndReleasedInstanceLifetime) {
  const auto *folder = std::getenv("TAICHI_F3_CONTEXT_AOT_FOLDER");
  if (!folder || !cuda::ExecutionDomain::available())
    GTEST_SKIP() << "Requires the saved execution-context AOT fixture";
  auto runtime = std::make_unique<capi::LlvmRuntime>(Arch::cuda);
  auto foreign_runtime = std::make_unique<capi::LlvmRuntime>(Arch::cuda);
  auto rt = reinterpret_cast<TiRuntime>(runtime.get());
  const TiCudaExecutionDomainCreateInfo options{2, 4, 0};
  auto output = ti_create_cuda_execution_domain_ext(rt, &options);
  auto physics = ti_create_cuda_execution_domain_ext(rt, &options);
  auto foreign = ti_create_cuda_execution_domain_ext(
      reinterpret_cast<TiRuntime>(foreign_runtime.get()), &options);
  ASSERT_NE(output, nullptr);
  ASSERT_NE(physics, nullptr);
  ASSERT_NE(foreign, nullptr);
  auto foreign_ticket = ti_get_cuda_execution_domain_completion_ext(foreign);
  auto module = ti_load_aot_module(rt, folder);
  ASSERT_NE(module, nullptr);
  constexpr int count = 64;
  std::array<TiMemory, 2> memory{};
  std::array<void *, 2> pointers{};
  std::array<TiPreparedExecution, 2> calls{};
  const std::string plan =
      R"({"version":1,"operations":[{"kind":"kernel","entry":"stage","arguments":["out","seed","n"]}]})";
  for (int i = 0; i < 2; ++i) {
    TiMemoryAllocateInfo allocation_info{};
    allocation_info.size = count * sizeof(float);
    allocation_info.usage = TI_MEMORY_USAGE_STORAGE_BIT;
    memory[i] = ti_allocate_memory(rt, &allocation_info);
    auto &device = static_cast<cuda::CudaDevice &>(runtime->get());
    pointers[i] =
        device.get_memory_addr(runtime->get_memory(memory[i]).allocation);
    TiNamedArgument arguments[3]{};
    arguments[0].name = "out";
    arguments[0].argument.type = TI_ARGUMENT_TYPE_NDARRAY;
    arguments[0].argument.value.ndarray = {
        memory[i], {1, {count}}, {0, {}}, TI_DATA_TYPE_F32};
    arguments[1].name = "seed";
    arguments[1].argument.type = TI_ARGUMENT_TYPE_F32;
    arguments[1].argument.value.f32 = i == 0 ? 2.0f : 7.0f;
    arguments[2].name = "n";
    arguments[2].argument.type = TI_ARGUMENT_TYPE_I32;
    arguments[2].argument.value.i32 = count;
    calls[i] = ti_prepare_execution_plan_on_domain_ext(
        rt, module, i == 0 ? output : physics, plan.size(), plan.c_str(), 3,
        arguments, 0, nullptr);
    ASSERT_NE(calls[i], nullptr);
    // Mutating the source descriptor cannot change the immutable instance.
    arguments[1].argument.value.f32 = -999.0f;
  }
  TiCompletion ticket = nullptr;
  ti_set_last_error(TI_ERROR_SUCCESS, nullptr);
  EXPECT_FALSE(ti_try_submit_prepared_ext(rt, calls[0], 0, nullptr, &ticket));
  EXPECT_EQ(ticket, nullptr);
  EXPECT_EQ(ti_get_last_error(nullptr, nullptr), TI_ERROR_INVALID_ARGUMENT);
  ti_set_last_error(TI_ERROR_SUCCESS, nullptr);
  ti_initialize_prepared_ext(rt, calls[0]);
  ASSERT_EQ(ti_get_last_error(nullptr, nullptr), TI_ERROR_SUCCESS);

  testing::CudaExternalGate gate;
  capi::get_execution_domain(output)->submit(
      {}, nullptr, [&](void *stream) { gate.enqueue_wait(stream); });
  auto blocked = ti_get_cuda_execution_domain_completion_ext(output);
  // Cold initialization waits only its published prefix, never the blocked
  // output domain. This also proves the no-explicit-initializer contract.
  {
    ScopedExternalProfilerAnnotation annotation("F3/public/initialize");
    ti_initialize_prepared_ext(rt, calls[1]);
  }
  ASSERT_EQ(ti_get_last_error(nullptr, nullptr), TI_ERROR_SUCCESS);
  EXPECT_FALSE(gate.timed_out());
  EXPECT_FALSE(ti_query_completion_ext(blocked));
  EXPECT_TRUE(ti_try_submit_prepared_ext(rt, calls[0], 1, &blocked, &ticket));
  ASSERT_NE(ticket, nullptr);
  TiCompletion busy = nullptr;
  EXPECT_FALSE(ti_try_submit_prepared_ext(rt, calls[0], 0, nullptr, &busy));
  EXPECT_EQ(busy, nullptr);
  EXPECT_EQ(ti_get_last_error(nullptr, nullptr), TI_ERROR_SUCCESS);
  const auto sequence =
      capi::get_execution_domain(physics)->latest_completion().sequence();
  TiCompletion rejected = nullptr;
  EXPECT_FALSE(
      ti_try_submit_prepared_ext(rt, calls[1], 1, &foreign_ticket, &rejected));
  EXPECT_EQ(rejected, nullptr);
  EXPECT_NE(ti_get_last_error(nullptr, nullptr), TI_ERROR_SUCCESS);
  EXPECT_EQ(capi::get_execution_domain(physics)->latest_completion().sequence(),
            sequence);
  ti_set_last_error(TI_ERROR_SUCCESS, nullptr);
  TiCompletion active = nullptr;
  {
    ScopedExternalProfilerAnnotation annotation("F3/public/submit");
    ASSERT_TRUE(ti_try_submit_prepared_ext(rt, calls[1], 0, nullptr, &active));
  }
  ti_wait_completion_ext(active);
  ASSERT_EQ(ti_get_last_error(nullptr, nullptr), TI_ERROR_SUCCESS);
  auto context = CUDAContext::get_instance().get_guard();
  std::array<float, count> values{};
  CUDADriver::get_instance().memcpy_device_to_host(values.data(), pointers[1],
                                                   sizeof(values));
  for (int i = 0; i < count; ++i)
    EXPECT_EQ(values[i], 36.0f * count * (count + 1) / 2 + i);
  std::vector<double> times;
  times.reserve(1000);
  std::size_t busy_count = 0;
  {
    ScopedExternalProfilerAnnotation annotation("F3/public/warm");
    while (times.size() < 1000) {
      TiCompletion completed = nullptr;
      const auto begin = std::chrono::steady_clock::now();
      const auto accepted =
          ti_try_submit_prepared_ext(rt, calls[1], 0, nullptr, &completed);
      const auto end = std::chrono::steady_clock::now();
      ASSERT_EQ(ti_get_last_error(nullptr, nullptr), TI_ERROR_SUCCESS);
      if (!accepted) {
        ++busy_count;
        std::this_thread::yield();
        continue;
      }
      times.push_back(
          std::chrono::duration<double, std::micro>(end - begin).count());
      ti_release_completion_ext(completed);
    }
  }
  auto warm_end = ti_get_cuda_execution_domain_completion_ext(physics);
  ti_wait_completion_ext(warm_end);
  ti_release_completion_ext(warm_end);
  std::sort(times.begin(), times.end());
  std::cout << "public AOT submit_us p50=" << times[500]
            << " p95=" << times[950] << " accepted=" << times.size()
            << " busy=" << busy_count << std::endl;
  EXPECT_FALSE(gate.timed_out());
  EXPECT_FALSE(ti_query_completion_ext(ticket));
  TiCompletion dependent = nullptr;
  ASSERT_TRUE(ti_try_submit_prepared_ext(rt, calls[1], 1, &ticket, &dependent));
  EXPECT_FALSE(ti_query_completion_ext(dependent));

  // Public wrapper retirement must not wait, erase code/storage edges or
  // release the still-pending independent execution instance.
  {
    ScopedExternalProfilerAnnotation annotation("F3/public/release_instance");
    ti_destroy_prepared_ext(rt, calls[0]);
  }
  ASSERT_EQ(ti_get_last_error(nullptr, nullptr), TI_ERROR_SUCCESS);
  EXPECT_FALSE(gate.timed_out());
  EXPECT_FALSE(ti_query_completion_ext(ticket));
  ti_destroy_prepared_ext(rt, calls[1]);
  auto close_module = std::async(std::launch::async, [&] {
    ti_set_last_error(TI_ERROR_SUCCESS, nullptr);
    ti_destroy_aot_module(module);
    return ti_get_last_error(nullptr, nullptr);
  });
  EXPECT_EQ(close_module.wait_for(std::chrono::milliseconds(20)),
            std::future_status::timeout);
  EXPECT_FALSE(gate.timed_out());
  {
    ScopedExternalProfilerAnnotation annotation("F3/public/release_gate");
    gate.release();
  }
  EXPECT_EQ(close_module.get(), TI_ERROR_SUCCESS);
  EXPECT_TRUE(ti_query_completion_ext(ticket));
  EXPECT_TRUE(ti_query_completion_ext(dependent));
  CUDADriver::get_instance().memcpy_device_to_host(values.data(), pointers[0],
                                                   sizeof(values));
  for (int i = 0; i < count; ++i)
    EXPECT_EQ(values[i], 11.0f * count * (count + 1) / 2 + i);
  for (auto completion : {blocked, ticket, active, dependent, foreign_ticket})
    ti_release_completion_ext(completion);
  for (auto domain : {output, physics, foreign}) {
    ti_close_cuda_execution_domain_ext(domain);
    ti_release_cuda_execution_domain_ext(domain);
  }
  for (auto allocation : memory)
    ti_free_memory(rt, allocation);
  EXPECT_EQ(ti_get_last_error(nullptr, nullptr), TI_ERROR_SUCCESS);
}

TEST(CudaExecutionOwner, PublicTicketWaitDoesNotLockQueryOrPeerSubmission) {
  if (!cuda::ExecutionDomain::available())
    GTEST_SKIP();
  auto runtime = std::make_unique<capi::LlvmRuntime>(Arch::cuda);
  const auto handle = reinterpret_cast<TiRuntime>(runtime.get());
  auto context = CUDAContext::get_instance().get_guard();
  const TiCudaExecutionDomainCreateInfo options{2, 4, -1};
  auto output = ti_create_cuda_execution_domain_ext(handle, &options);
  auto physics = ti_create_cuda_execution_domain_ext(handle, &options);
  ASSERT_NE(output, nullptr);
  ASSERT_NE(physics, nullptr);
  TiCudaExecutionDomainInfo left{}, right{};
  ti_get_cuda_execution_domain_info_ext(output, &left);
  ti_get_cuda_execution_domain_info_ext(physics, &right);
  EXPECT_NE(left.identity, right.identity);
  EXPECT_EQ(left.owner, right.owner);
  EXPECT_EQ(left.context, right.context);
  EXPECT_NE(left.stream, 0);
  EXPECT_NE(left.stream, right.stream);
  EXPECT_EQ(left.capacity, 2);
  auto empty = ti_get_cuda_execution_domain_completion_ext(output);
  ASSERT_NE(empty, nullptr);
  TiCompletionInfo empty_info{};
  ti_get_completion_info_ext(empty, &empty_info);
  EXPECT_EQ(empty_info.sequence, 0);

  auto &device = static_cast<cuda::CudaDevice &>(runtime->get());
  auto allocation = device.allocate_memory_unique({sizeof(std::uint32_t)});
  ASSERT_EQ(allocation.second, RhiResult::success);
  auto *address = device.get_memory_addr(*allocation.first);
  testing::CudaExternalGate gate;
  auto native_output = capi::get_execution_domain(output);
  auto native_physics = capi::get_execution_domain(physics);
  native_output->submit({}, nullptr,
                        [&](void *stream) { gate.enqueue_wait(stream); });
  auto blocked = ti_get_cuda_execution_domain_completion_ext(output);
  ASSERT_NE(blocked, nullptr);
  EXPECT_FALSE(ti_query_completion_ext(blocked));
  EXPECT_TRUE(ti_query_completion_ext(empty));
  auto waiter = std::async(std::launch::async, [&] {
    ti_set_last_error(TI_ERROR_SUCCESS, nullptr);
    ti_wait_completion_ext(blocked);
    return ti_get_last_error(nullptr, nullptr);
  });
  EXPECT_EQ(waiter.wait_for(std::chrono::milliseconds(20)),
            std::future_status::timeout);
  auto query = std::async(std::launch::async, [&] {
    ti_set_last_error(TI_ERROR_SUCCESS, nullptr);
    return std::make_pair(ti_query_completion_ext(blocked),
                          ti_get_last_error(nullptr, nullptr));
  });
  const auto query_status = query.wait_for(std::chrono::milliseconds(200));
  if (query_status != std::future_status::ready)
    gate.release();
  EXPECT_EQ(query_status, std::future_status::ready);
  const auto queried = query.get();
  EXPECT_FALSE(queried.first);
  EXPECT_EQ(queried.second, TI_ERROR_SUCCESS);
  auto sentinel = native_physics->submit({}, nullptr, [&](void *stream) {
    CUDADriver::get_instance().memsetd32_async(address, 0x3f123456, 1, stream);
  });
  sentinel.wait();
  std::uint32_t observed = 0;
  CUDADriver::get_instance().memcpy_device_to_host(&observed, address,
                                                   sizeof(observed));
  EXPECT_EQ(observed, 0x3f123456u);
  EXPECT_FALSE(ti_query_completion_ext(blocked));
  EXPECT_FALSE(gate.timed_out());
  gate.release();
  EXPECT_EQ(waiter.get(), TI_ERROR_SUCCESS);
  EXPECT_TRUE(ti_query_completion_ext(blocked));
  ti_release_completion_ext(blocked);
  ti_release_completion_ext(empty);
  ti_release_cuda_execution_domain_ext(output);
  // Releasing the public handle must not close its runtime-owned domain.
  EXPECT_TRUE(native_output->snapshot().accepting);
  ti_close_cuda_execution_domain_ext(physics);
  ti_release_cuda_execution_domain_ext(physics);
  ti_set_last_error(TI_ERROR_SUCCESS, nullptr);
  EXPECT_FALSE(ti_query_completion_ext(blocked));
  EXPECT_EQ(ti_get_last_error(nullptr, nullptr), TI_ERROR_INVALID_ARGUMENT);
  ti_set_last_error(TI_ERROR_SUCCESS, nullptr);
}

TEST(CudaExecutionOwner,
     PublicDomainHandlesRetireWithRuntimeAndRejectBadCapacity) {
  if (!cuda::ExecutionDomain::available())
    GTEST_SKIP();
  auto runtime = std::make_unique<capi::LlvmRuntime>(Arch::cuda);
  auto handle = reinterpret_cast<TiRuntime>(runtime.get());
  const TiCudaExecutionDomainCreateInfo invalid{0, 4, 0}, valid{1, 4, 0};
  ti_set_last_error(TI_ERROR_SUCCESS, nullptr);
  EXPECT_EQ(ti_create_cuda_execution_domain_ext(handle, &invalid), nullptr);
  EXPECT_EQ(ti_get_last_error(nullptr, nullptr), TI_ERROR_INVALID_ARGUMENT);
  ti_set_last_error(TI_ERROR_SUCCESS, nullptr);
  auto domain = ti_create_cuda_execution_domain_ext(handle, &valid);
  ASSERT_NE(domain, nullptr);
  auto completion = ti_get_cuda_execution_domain_completion_ext(domain);
  ti_destroy_runtime(reinterpret_cast<TiRuntime>(runtime.release()));
  TiCudaExecutionDomainInfo info{};
  ti_get_cuda_execution_domain_info_ext(domain, &info);
  EXPECT_TRUE(info.closed);
  EXPECT_FALSE(info.accepting);
  EXPECT_TRUE(ti_query_completion_ext(completion));
  ti_wait_completion_ext(completion);
  ti_release_completion_ext(completion);
  ti_release_cuda_execution_domain_ext(domain);
  EXPECT_EQ(ti_get_last_error(nullptr, nullptr), TI_ERROR_SUCCESS);
}

TEST(CudaExecutionOwner, ProgramPrefixesAndBorrowedAotShareOneOwner) {
  if (!cuda::ExecutionDomain::available())
    GTEST_SKIP();
  Program program(Arch::cuda);
  program.materialize_runtime();
  program.synchronize();
  auto registry = program.cuda_execution_domains();
  auto borrower = AotRuntimeClient::create(program);
  auto *runtime = reinterpret_cast<::Runtime *>(borrower->handle());
  ASSERT_EQ(runtime->cuda_execution_domains(), registry);
  ASSERT_EQ(runtime->runtime_fault_domain(), program.runtime_fault_domain());
  auto output = registry->create({2, 0});
  auto physics = registry->create({1, 0});
  testing::CudaExternalGate gate;
  auto blocked = output->submit(
      {}, nullptr, [&](void *stream) { gate.enqueue_wait(stream); });
  auto empty = program.record_runtime_completion();
  EXPECT_TRUE(empty.done());
  EXPECT_EQ(empty.sequence(), 0);
  // Only this explicit publication marks default-queue work. A later domain
  // sequence must never be substituted for the last default completion.
  program.mark_runtime_submission_pending();
  auto default_prefix = program.record_runtime_completion();
  auto other = physics->submit({}, nullptr, [](void *) {});
  other.wait();
  EXPECT_LT(blocked.sequence(), default_prefix.sequence());
  EXPECT_LT(default_prefix.sequence(), other.sequence());
  EXPECT_EQ(program.record_runtime_completion().sequence(),
            default_prefix.sequence());
  EXPECT_FALSE(blocked.done());
  // Explicit ti.sync() must include independent streams. Its wait must not
  // hold the registry admission mutex or the completion query mutex.
  auto sync = std::async(std::launch::async, [&] { program.synchronize(); });
  EXPECT_EQ(sync.wait_for(std::chrono::milliseconds(20)),
            std::future_status::timeout);
  EXPECT_TRUE(registry->accepting());
  EXPECT_FALSE(blocked.done());
  auto during_sync = physics->submit({}, nullptr, [](void *) {});
  during_sync.wait();
  EXPECT_FALSE(blocked.done());
  EXPECT_FALSE(gate.timed_out());
  gate.release();
  sync.get();
  borrower->close();
  EXPECT_TRUE(registry->accepting());
  EXPECT_TRUE(output->snapshot().accepting);
  // Program finalization retires externally retained handles too.
  program.finalize();
  EXPECT_TRUE(output->snapshot().closed);
  EXPECT_TRUE(physics->snapshot().closed);
  EXPECT_FALSE(registry->accepting());
  EXPECT_ANY_THROW(registry->create({1, 0}));
}

TEST(CudaExecutionOwner, SNodeRetirementDrainsDependentDomainBeforeRootFree) {
  if (!cuda::ExecutionDomain::available())
    GTEST_SKIP();
  Program program(Arch::cuda);
  program.materialize_runtime();
  auto root = std::make_unique<SNode>(0, SNodeType::root);
  root->dense(Axis(0), 1).insert_children(SNodeType::place).dt =
      PrimitiveType::i32;
  auto *tree = program.add_snode_tree(std::move(root), false);
  auto *executor = static_cast<LlvmProgramImpl *>(program.get_program_impl())
                       ->get_runtime_executor();
  auto *root_address = executor->get_snode_tree_root_ptr(tree->id());
  auto &device = static_cast<cuda::CudaDevice &>(*program.get_compute_device());
  auto allocation = device.allocate_memory_unique({2 * sizeof(std::uint32_t)});
  ASSERT_EQ(allocation.second, RhiResult::success);
  auto *address =
      static_cast<std::uint32_t *>(device.get_memory_addr(*allocation.first));
  auto registry = program.cuda_execution_domains();
  auto dependent = registry->create({1, 0});
  auto peer = registry->create({1, 0});
  std::promise<void> entered;
  class TreeClient final : public RuntimeClient {
   public:
    TreeClient(std::shared_ptr<cuda::ExecutionDomain> domain,
               std::promise<void> &entered)
        : domain_(std::move(domain)), entered_(entered) {
    }
    void retire() override {
      entered_.set_value();
      domain_->close();
    }

   private:
    std::shared_ptr<cuda::ExecutionDomain> domain_;
    std::promise<void> &entered_;
  };
  program.retain_runtime_client(
      std::make_shared<TreeClient>(dependent, entered),
      program.snapshot_snode_tree_dependencies({tree->id()}));
  program.synchronize();
  testing::CudaExternalGate gate;
  auto blocked = dependent->submit({}, nullptr, [&](void *stream) {
    gate.enqueue_wait(stream);
    auto &driver = CUDADriver::get_instance();
    driver.memsetd32_async(root_address, 0x3210ABCD, 1, stream);
    driver.memcpy_device_to_device_async(address, root_address,
                                         sizeof(std::uint32_t), stream);
  });
  auto destroy =
      std::async(std::launch::async, [&] { program.destroy_snode_tree(tree); });
  entered.get_future().wait();
  EXPECT_EQ(destroy.wait_for(std::chrono::milliseconds(20)),
            std::future_status::timeout);
  auto sentinel = peer->submit({}, nullptr, [&](void *stream) {
    CUDADriver::get_instance().memsetd32_async(address + 1, 0x1234FEDC, 1,
                                               stream);
  });
  sentinel.wait();
  auto context = CUDAContext::get_instance().get_guard();
  std::uint32_t observed = 0;
  CUDADriver::get_instance().memcpy_device_to_host(&observed, address + 1,
                                                   sizeof(observed));
  EXPECT_EQ(observed, 0x1234FEDCu);
  EXPECT_FALSE(blocked.done());
  EXPECT_FALSE(gate.timed_out());
  gate.release();
  destroy.get();
  EXPECT_TRUE(blocked.done());
  EXPECT_TRUE(dependent->snapshot().closed);
  EXPECT_TRUE(peer->snapshot().accepting);
  CUDADriver::get_instance().memcpy_device_to_host(&observed, address,
                                                   sizeof(observed));
  EXPECT_EQ(observed, 0x3210ABCDu);
  allocation.first.reset();
  program.finalize();
}

TEST(CudaExecutionOwner, LiveJitCodeLoanDrainsBeforeFieldAndCodeRetirement) {
  if (!cuda::ExecutionDomain::available())
    GTEST_SKIP();
  Program program(Arch::cuda);
  program.materialize_runtime();
  auto root = std::make_unique<SNode>(0, SNodeType::root);
  auto *place = &root->dense(Axis(0), 1).insert_children(SNodeType::place);
  place->dt = PrimitiveType::i32;
  auto *tree = program.add_snode_tree(std::move(root), false);
  IRBuilder ir;
  auto *array = ir.create_ndarray_arg_load({0}, PrimitiveType::i32, 1, 0);
  auto *zero = ir.get_int32(0);
  auto *value = ir.get_int32(314159);
  auto *field = ir.create_global_ptr(place, {zero});
  ir.create_global_store(field, value);
  ir.create_global_store(ir.create_external_ptr(array, {zero}),
                         ir.create_global_load(field));
  auto kernel = std::make_unique<Kernel>(program, ir.extract_ir(), "jit_loan");
  kernel->insert_ndarray_param(PrimitiveType::i32, 1);
  kernel->finalize_params();
  kernel->finalize_rets();
  Dispatch dispatch(kernel.get(), {aot::Arg(aot::ArgKind::kNdarray, "out",
                                            PrimitiveType::i32, 1)});
  aot::CompiledGraph graph({dispatch.compile_dispatch()});
  auto borrower = AotRuntimeClient::create(program);
  auto *runtime = reinterpret_cast<::Runtime *>(borrower->handle());
  auto rt = reinterpret_cast<TiRuntime>(runtime);
  auto module = reinterpret_cast<TiAotModule>(
      borrower->import_jit_kernels({{"stage", &graph, 0, false}}));
  ASSERT_NE(module, nullptr);
  const TiCudaExecutionDomainCreateInfo options{2, 4, 0};
  auto output = ti_create_cuda_execution_domain_ext(rt, &options);
  auto peer = ti_create_cuda_execution_domain_ext(rt, &options);
  ASSERT_NE(output, nullptr);
  ASSERT_NE(peer, nullptr);
  TiMemoryAllocateInfo allocation_info{};
  allocation_info.size = 2 * sizeof(std::uint32_t);
  allocation_info.usage = TI_MEMORY_USAGE_STORAGE_BIT;
  auto memory = ti_allocate_memory(rt, &allocation_info);
  TiNamedArgument argument{};
  argument.name = "out";
  argument.argument.type = TI_ARGUMENT_TYPE_NDARRAY;
  argument.argument.value.ndarray = {
      memory, {1, {2}}, {0, {}}, TI_DATA_TYPE_I32};
  const std::string plan =
      R"({"version":1,"operations":[{"kind":"kernel","entry":"stage","arguments":["out"]}]})";
  auto prepared = ti_prepare_execution_plan_on_domain_ext(
      rt, module, output, plan.size(), plan.c_str(), 1, &argument, 0, nullptr);
  ASSERT_NE(prepared, nullptr);
  ti_initialize_prepared_ext(rt, prepared);
  ASSERT_EQ(ti_get_last_error(nullptr, nullptr), TI_ERROR_SUCCESS);
  auto &device = static_cast<cuda::CudaDevice &>(runtime->get());
  auto *address = static_cast<std::uint32_t *>(
      device.get_memory_addr(runtime->get_memory(memory).allocation));
  auto native_output = capi::get_execution_domain(output);
  auto native_peer = capi::get_execution_domain(peer);
  testing::CudaExternalGate gate;
  auto blocked = native_output->submit(
      {}, nullptr, [&](void *stream) { gate.enqueue_wait(stream); });
  TiCompletion ticket = nullptr;
  ASSERT_TRUE(ti_try_submit_prepared_ext(rt, prepared, 0, nullptr, &ticket));
  auto destroy =
      std::async(std::launch::async, [&] { program.destroy_snode_tree(tree); });
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (native_output->snapshot().accepting &&
         std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  EXPECT_FALSE(native_output->snapshot().accepting);
  EXPECT_EQ(destroy.wait_for(std::chrono::milliseconds(0)),
            std::future_status::timeout);
  native_peer
      ->submit({}, nullptr,
               [&](void *stream) {
                 CUDADriver::get_instance().memsetd32_async(address + 1, 271828,
                                                            1, stream);
               })
      .wait();
  auto context = CUDAContext::get_instance().get_guard();
  std::uint32_t observed = 0;
  CUDADriver::get_instance().memcpy_device_to_host(&observed, address + 1,
                                                   sizeof(observed));
  EXPECT_EQ(observed, 271828u);
  EXPECT_FALSE(blocked.done());
  EXPECT_FALSE(ti_query_completion_ext(ticket));
  EXPECT_FALSE(gate.timed_out());
  gate.release();
  destroy.get();
  EXPECT_TRUE(ti_query_completion_ext(ticket));
  EXPECT_TRUE(native_peer->snapshot().accepting);
  CUDADriver::get_instance().memcpy_device_to_host(&observed, address,
                                                   sizeof(observed));
  EXPECT_EQ(observed, 314159u);
  ti_release_completion_ext(ticket);
  ti_destroy_prepared_ext(rt, prepared);
  ti_destroy_aot_module(module);
  ti_free_memory(rt, memory);
  ti_close_cuda_execution_domain_ext(output);
  ti_close_cuda_execution_domain_ext(peer);
  ti_release_cuda_execution_domain_ext(output);
  ti_release_cuda_execution_domain_ext(peer);
  borrower->close();
  program.finalize();
}

TEST(CudaExecutionOwner, StandaloneAotWaitAndDestroyRetireIndependentWork) {
  if (!cuda::ExecutionDomain::available())
    GTEST_SKIP();
  auto runtime = std::make_unique<capi::LlvmRuntime>(Arch::cuda);
  auto registry = runtime->cuda_execution_domains();
  auto output = registry->create({1, 0});
  testing::CudaExternalGate gate;
  auto blocked = output->submit(
      {}, nullptr, [&](void *stream) { gate.enqueue_wait(stream); });
  auto waiter = std::async(std::launch::async, [&] {
    static_cast<::Runtime *>(runtime.get())->wait();
  });
  EXPECT_EQ(waiter.wait_for(std::chrono::milliseconds(20)),
            std::future_status::timeout);
  EXPECT_FALSE(blocked.done());
  EXPECT_TRUE(registry->accepting());
  EXPECT_FALSE(gate.timed_out());
  gate.release();
  waiter.get();
  // Exercise the public C API destruction entry as well as the derived
  // destructor; it must close domains before clearing prepared executions.
  ti_destroy_runtime(reinterpret_cast<TiRuntime>(runtime.release()));
  EXPECT_TRUE(blocked.done());
  EXPECT_TRUE(output->snapshot().closed);
  EXPECT_FALSE(registry->accepting());
  EXPECT_ANY_THROW(output->submit({}, nullptr, [](void *) {}));
}

TEST(CudaExecutionOwner, FinalizeDrainsBeforeClosingStorageAndRejectsNewWork) {
  if (!cuda::ExecutionDomain::available())
    GTEST_SKIP();
  Program program(Arch::cuda);
  program.materialize_runtime();
  auto registry = program.cuda_execution_domains();
  auto output = registry->create({1, 0});
  testing::CudaExternalGate gate;
  auto blocked = output->submit(
      {}, nullptr, [&](void *stream) { gate.enqueue_wait(stream); });
  auto finalizer = std::async(std::launch::async, [&] { program.finalize(); });
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (registry->accepting() && std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  EXPECT_FALSE(registry->accepting());
  EXPECT_ANY_THROW(registry->create({1, 0}));
  EXPECT_ANY_THROW(output->submit({}, nullptr, [](void *) {}));
  EXPECT_FALSE(blocked.done());
  EXPECT_EQ(finalizer.wait_for(std::chrono::milliseconds(0)),
            std::future_status::timeout);
  EXPECT_EQ(program.runtime_fault_domain()->state(),
            RuntimeLifecycleState::kFinalizing);
  EXPECT_FALSE(gate.timed_out());
  gate.release();
  finalizer.get();
  EXPECT_TRUE(blocked.done());
  EXPECT_TRUE(output->snapshot().closed);
  EXPECT_EQ(program.runtime_fault_domain()->state(),
            RuntimeLifecycleState::kFinalized);
}

}  // namespace
}  // namespace taichi::lang
#endif
