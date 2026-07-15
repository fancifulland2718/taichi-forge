#include "gtest/gtest.h"

#include <atomic>
#include <thread>
#include <vector>

#include "taichi/common/exceptions.h"
#include "taichi/program/runtime_fault.h"
#include "taichi/rhi/cuda/cuda_driver.h"

namespace taichi::lang {
namespace {

TEST(RuntimeFault, BackendErrorClassificationIsConservative) {
  EXPECT_EQ(classify_cuda_driver_error(CUDA_SUCCESS),
            BackendErrorClassification::kSuccess);
  EXPECT_EQ(classify_cuda_driver_error(CUDA_ERROR_NOT_READY),
            BackendErrorClassification::kRecoverable);
  EXPECT_EQ(classify_cuda_driver_error(CUDA_ERROR_ILLEGAL_ADDRESS),
            BackendErrorClassification::kFatal);
  EXPECT_EQ(classify_cuda_driver_error(CUDA_ERROR_ASSERT),
            BackendErrorClassification::kFatal);
  EXPECT_EQ(classify_cuda_driver_error(CUDA_ERROR_NOT_SUPPORTED),
            BackendErrorClassification::kOperation);

  EXPECT_EQ(classify_vulkan_result(0),
            BackendErrorClassification::kSuccess);
  EXPECT_EQ(classify_vulkan_result(1),
            BackendErrorClassification::kRecoverable);
  EXPECT_EQ(classify_vulkan_result(2),
            BackendErrorClassification::kRecoverable);
  EXPECT_EQ(classify_vulkan_result(1000001003),
            BackendErrorClassification::kRecoverable);
  EXPECT_EQ(classify_vulkan_result(-1000001004),
            BackendErrorClassification::kRecoverable);
  EXPECT_EQ(classify_vulkan_result(-4),
            BackendErrorClassification::kFatal);
  EXPECT_EQ(classify_vulkan_result(-1),
            BackendErrorClassification::kOperation);

  EXPECT_EQ(classify_rhi_result(RhiResult::success),
            BackendErrorClassification::kSuccess);
  EXPECT_EQ(classify_rhi_result(RhiResult::invalid_usage),
            BackendErrorClassification::kOperation);
  EXPECT_EQ(classify_rhi_result(RhiResult::out_of_memory),
            BackendErrorClassification::kOperation);
  EXPECT_EQ(classify_rhi_result(RhiResult::error),
            BackendErrorClassification::kOperation);
}

TEST(RuntimeFault, FirstFatalFaultIsImmutableAndRejectsSubmission) {
  RuntimeFaultDomain domain(Arch::cuda, 41);
  EXPECT_TRUE(domain.submission_allowed());
  EXPECT_TRUE(domain.backend_calls_safe());
  EXPECT_EQ(domain.state(), RuntimeLifecycleState::kHealthy);

  EXPECT_TRUE(domain.report_fatal({Arch::cuda, CUDA_ERROR_ILLEGAL_ADDRESS, 7,
                                   "completion.wait",
                                   "illegal memory access"}));
  EXPECT_FALSE(domain.report_fatal(
      {Arch::cuda, CUDA_ERROR_ASSERT, 8, "later", "must not replace"}));
  EXPECT_FALSE(domain.submission_allowed());
  EXPECT_FALSE(domain.backend_calls_safe());
  EXPECT_EQ(domain.state(), RuntimeLifecycleState::kFaulted);

  try {
    domain.throw_if_submission_disallowed("kernel launch");
    FAIL() << "faulted runtime accepted a submission";
  } catch (const TaichiRuntimeError &error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("Runtime is faulted"), std::string::npos);
    EXPECT_NE(message.find("kernel launch"), std::string::npos);
    EXPECT_NE(message.find("code=700"), std::string::npos);
    EXPECT_NE(message.find("completion.wait"), std::string::npos);
    EXPECT_NE(message.find("sequence=7"), std::string::npos);
    EXPECT_NE(message.find("may require restarting the process"),
              std::string::npos);
    EXPECT_EQ(message.find("must not replace"), std::string::npos);
  }

  const RuntimeFaultSnapshot snapshot = domain.snapshot();
  ASSERT_TRUE(snapshot.first_fault.has_value());
  EXPECT_EQ(snapshot.program_domain, 41u);
  EXPECT_EQ(snapshot.rejected_submissions, 1u);
  EXPECT_EQ(snapshot.first_fault->backend_code,
            static_cast<std::int64_t>(CUDA_ERROR_ILLEGAL_ADDRESS));
  EXPECT_EQ(snapshot.first_fault->submission_sequence, 7u);
}

TEST(RuntimeFault, ConcurrentReportsSelectExactlyOneFirstFault) {
  RuntimeFaultDomain domain(Arch::vulkan, 73);
  constexpr int kThreads = 32;
  std::atomic<int> winners{0};
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int i = 0; i < kThreads; ++i) {
    threads.emplace_back([&, i] {
      if (domain.report_fatal(
              {Arch::vulkan, -4, static_cast<std::uint64_t>(i + 1),
               "vkQueueSubmit", "device lost"})) {
        winners.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }
  for (auto &thread : threads) {
    thread.join();
  }

  EXPECT_EQ(winners.load(std::memory_order_relaxed), 1);
  const RuntimeFaultSnapshot snapshot = domain.snapshot();
  ASSERT_TRUE(snapshot.first_fault.has_value());
  EXPECT_GE(snapshot.first_fault->submission_sequence, 1u);
  EXPECT_LE(snapshot.first_fault->submission_sequence,
            static_cast<std::uint64_t>(kThreads));
  EXPECT_EQ(snapshot.state, RuntimeLifecycleState::kFaulted);
}

TEST(RuntimeFault, FinalizationPreservesButDoesNotCreatePostFinalizedFaults) {
  RuntimeFaultDomain domain(Arch::vulkan, 99);
  domain.begin_finalizing();
  EXPECT_EQ(domain.state(), RuntimeLifecycleState::kFinalizing);
  EXPECT_TRUE(domain.report_fatal(
      {Arch::vulkan, -4, 0, "teardown", "device lost during teardown"}));
  EXPECT_EQ(domain.state(), RuntimeLifecycleState::kFinalizing);
  domain.mark_finalized();
  EXPECT_EQ(domain.state(), RuntimeLifecycleState::kFinalized);
  EXPECT_FALSE(domain.report_fatal(
      {Arch::vulkan, -4, 1, "late destructor", "ignored"}));

  const RuntimeFaultSnapshot snapshot = domain.snapshot();
  ASSERT_TRUE(snapshot.first_fault.has_value());
  EXPECT_EQ(snapshot.first_fault->operation, "teardown");
  EXPECT_THROW(domain.throw_if_submission_disallowed("Graph submit"),
               TaichiRuntimeError);
}

TEST(RuntimeFault, HealthyFinalizationDoesNotInventAFault) {
  RuntimeFaultDomain domain(Arch::x64, 123);
  domain.begin_finalizing();
  EXPECT_NO_THROW(
      domain.throw_if_submission_disallowed("healthy backend drain"));
  std::atomic<bool> other_thread_rejected{false};
  std::thread submitter([&] {
    try {
      domain.throw_if_submission_disallowed("concurrent Graph submit");
    } catch (const TaichiRuntimeError &) {
      other_thread_rejected.store(true, std::memory_order_release);
    }
  });
  submitter.join();
  EXPECT_TRUE(other_thread_rejected.load(std::memory_order_acquire));
  domain.begin_finalizing();
  domain.mark_finalized();
  domain.mark_finalized();
  EXPECT_THROW(domain.throw_if_submission_disallowed("late backend drain"),
               TaichiRuntimeError);
  const RuntimeFaultSnapshot snapshot = domain.snapshot();
  EXPECT_EQ(snapshot.state, RuntimeLifecycleState::kFinalized);
  EXPECT_FALSE(snapshot.first_fault.has_value());
}

}  // namespace
}  // namespace taichi::lang
