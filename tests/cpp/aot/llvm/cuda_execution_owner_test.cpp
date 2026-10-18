#include <gtest/gtest.h>

#if defined(TI_WITH_CUDA) && defined(TI_WITH_VULKAN) && \
    defined(TI_TEST_RUNTIME_C_API)
#include <future>
#include "tests/cpp/rhi/common/cuda_external_gate.h"
#include "c_api/src/taichi_llvm_impl.h"
#include "c_api/src/execution_domain.h"
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/program/program.h"
#include "taichi/runtime/aot_runtime_client.h"
#include "taichi/runtime/cuda/execution_domain.h"

namespace taichi::lang {
namespace {

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
