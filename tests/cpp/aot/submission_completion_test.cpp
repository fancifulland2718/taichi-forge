#include "taichi/runtime/submission_completion.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include "gtest/gtest.h"

namespace taichi::lang {
namespace {
class Signal final : public StreamSemaphoreObject {
 public:
  bool is_ready() const override {
    ++polls;
    if (failed)
      throw std::runtime_error("first completion failure");
    return ready;
  }
  bool wait() const override {
    if (failed)
      throw std::runtime_error("first completion failure");
    ready = true;
    return true;
  }
  mutable std::atomic<bool> ready{false};
  std::atomic<bool> failed{false};
  mutable std::atomic<int> polls{0};
};

TEST(SubmissionCompletion, ReleasedHandleRetainsPendingBackendState) {
  SubmissionCompletionQueue owner;
  SubmissionCompletionHandles handles;
  auto signal = std::make_shared<Signal>();
  std::weak_ptr<Signal> weak = signal;
  auto completion = RuntimeCompletion::from_stream_semaphore(
      Arch::vulkan, 11, 1, signal);
  owner.retain(completion);
  handles.publish(17, completion);
  completion = {};
  signal.reset();
  handles.release(17);
  EXPECT_THROW(handles.get(17), std::invalid_argument);
  EXPECT_THROW(handles.release(17), std::invalid_argument);
  EXPECT_FALSE(weak.expired());
  owner.complete_synchronized();
  EXPECT_TRUE(weak.expired());
  EXPECT_EQ(owner.size(), 0);
}

TEST(SubmissionCompletion, CompletedHandlesSurviveOwnerRetirement) {
  SubmissionCompletionHandles handles;
  {
    SubmissionCompletionQueue owner;
    for (std::uintptr_t index = 1; index <= 65; ++index) {
      auto completion = RuntimeCompletion::from_stream_semaphore(
          Arch::vulkan, 11, index, std::make_shared<Signal>());
      owner.retain(completion);
      handles.publish(index, completion);
    }
    EXPECT_EQ(owner.size(), 65);
    owner.complete_synchronized();
  }
  for (std::uintptr_t index = 1; index <= 65; ++index) {
    EXPECT_TRUE(handles.get(index).done());
    EXPECT_FALSE(handles.get(index).has_backend_work());
    EXPECT_NO_THROW(handles.get(index).wait());
    handles.release(index);
  }
}

TEST(SubmissionCompletion, CollectionIsBoundedWithoutLimitingBacklog) {
  SubmissionCompletionQueue owner;
  auto signal = std::make_shared<Signal>();
  for (std::uint64_t index = 1; index <= 65; ++index)
    owner.retain(RuntimeCompletion::from_stream_semaphore(
        Arch::vulkan, 11, index, signal));
  owner.collect_one();
  EXPECT_EQ(signal->polls, 1);
  EXPECT_EQ(owner.size(), 65);
  signal->ready = true;
  owner.collect_one();
  EXPECT_EQ(signal->polls, 2);
  EXPECT_EQ(owner.size(), 64);
  owner.complete_synchronized();
}

TEST(SubmissionCompletion, FailureRetainsFirstErrorUntilSafeRetirement) {
  SubmissionCompletionQueue owner;
  SubmissionCompletionHandles handles;
  auto signal = std::make_shared<Signal>();
  std::weak_ptr<Signal> weak = signal;
  auto completion = RuntimeCompletion::from_stream_semaphore(
      Arch::vulkan, 11, 1, signal);
  owner.retain(completion);
  handles.publish(17, completion);
  signal->failed = true;
  EXPECT_THROW(owner.collect_one(), std::runtime_error);
  EXPECT_EQ(owner.size(), 1);
  signal.reset();
  EXPECT_FALSE(weak.expired());
  owner.fail_and_release("later owner teardown");
  EXPECT_TRUE(weak.expired());
  EXPECT_EQ(handles.get(17).first_error_message(), "first completion failure");
  EXPECT_THROW(handles.get(17).wait(), std::runtime_error);
  handles.release(17);
}

TEST(SubmissionCompletion, SynchronousTokensNeedNoBackendQueue) {
  SubmissionCompletionQueue owner;
  SubmissionCompletionHandles handles;
  auto completion = RuntimeCompletion::completed(Arch::x64, 11, 1);
  owner.retain(completion);
  EXPECT_EQ(owner.size(), 0);
  handles.publish(17, completion);
  EXPECT_THROW(handles.publish(17, completion), std::invalid_argument);
  EXPECT_THROW(handles.publish(0, completion), std::invalid_argument);
  EXPECT_THROW(handles.publish(18, {}), std::invalid_argument);
  EXPECT_THROW(owner.retain({}), std::invalid_argument);
  EXPECT_TRUE(handles.get(17).done());
  handles.release(17);
}

class GatedSignal final : public StreamSemaphoreObject {
 public:
  bool is_ready() const override { return false; }
  bool wait() const override {
    entered.set_value();
    release.get_future().wait();
    return true;
  }
  mutable std::promise<void> entered, release;
};

TEST(SubmissionCompletion, WaitingDoesNotHoldHandleOrSubmissionRegistryLock) {
  SubmissionCompletionQueue owner;
  SubmissionCompletionHandles handles;
  auto signal = std::make_shared<GatedSignal>();
  auto completion = RuntimeCompletion::from_stream_semaphore(
      Arch::vulkan, 11, 1, signal);
  owner.retain(completion);
  handles.publish(17, completion);
  auto waiting = std::async(std::launch::async, [&] { handles.get(17).wait(); });
  signal->entered.get_future().wait();
  auto progress = std::async(std::launch::async, [&] {
    handles.publish(18, RuntimeCompletion::completed(Arch::x64, 12, 1));
    handles.release(17);
    EXPECT_TRUE(handles.get(18).done());
  });
  const bool responsive = progress.wait_for(std::chrono::milliseconds(200)) ==
                          std::future_status::ready;
  signal->release.set_value();
  waiting.get();
  progress.get();
  EXPECT_TRUE(responsive);
  owner.complete_synchronized();
  handles.release(18);
}
}  // namespace
}  // namespace taichi::lang
