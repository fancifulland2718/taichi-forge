#include <gtest/gtest.h>

#if defined(TI_WITH_CUDA) && defined(TI_WITH_VULKAN) && \
    defined(TI_TEST_RUNTIME_C_API)
#include <future>
#include "tests/cpp/rhi/common/cuda_external_gate.h"
#include "c_api/src/taichi_llvm_impl.h"
#include "taichi/program/program.h"
#include "taichi/runtime/aot_runtime_client.h"
#include "taichi/runtime/cuda/execution_domain.h"

namespace taichi::lang {
namespace {

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
