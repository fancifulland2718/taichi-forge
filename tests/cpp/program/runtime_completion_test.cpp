#include "gtest/gtest.h"

#include <atomic>
#include <memory>
#include <stdexcept>

#include "taichi/program/runtime_completion.h"
#include "taichi/program/runtime_fault.h"
#include "taichi/rhi/backend_error.h"

namespace taichi::lang {
namespace {

class FakeSemaphore final : public StreamSemaphoreObject {
 public:
  bool is_ready() const override {
    ++polls;
    if (fail) {
      throw std::runtime_error("fake completion failure");
    }
    return ready;
  }

  bool wait() const override {
    ++waits;
    if (fail) {
      throw std::runtime_error("fake completion failure");
    }
    ready = true;
    return true;
  }

  mutable bool ready{false};
  mutable bool fail{false};
  mutable int polls{0};
  mutable int waits{0};
};

class CountedResources final : public RuntimeCompletionResources {
 public:
  CountedResources(std::atomic<int> *released, std::size_t count)
      : released_(released), count_(count) {
  }

  ~CountedResources() override {
    released_->fetch_add(1, std::memory_order_relaxed);
  }

  std::size_t retained_resource_count(
      std::uint32_t kind) const noexcept override {
    return kind == 7 ? count_ : 0;
  }

 private:
  std::atomic<int> *released_;
  std::size_t count_;
};

class FatalBackendSemaphore final : public StreamSemaphoreObject {
 public:
  bool is_ready() const override {
    throw BackendRuntimeError(Arch::vulkan, -4, "vkGetFenceStatus",
                              "injected Vulkan device loss");
  }

  bool wait() const override {
    throw BackendRuntimeError(Arch::vulkan, -4, "vkWaitForFences",
                              "injected Vulkan device loss");
  }
};

TEST(RuntimeCompletion, CompletedTokenUsesNoBackendWork) {
  auto domain = std::make_shared<RuntimeFaultDomain>(Arch::x64, 11);
  auto completion =
      RuntimeCompletion::completed(Arch::x64, 11, 3, domain);
  EXPECT_TRUE(completion.valid());
  EXPECT_TRUE(completion.done());
  completion.wait();
  EXPECT_FALSE(completion.has_backend_work());
  EXPECT_EQ(completion.program_domain(), 11);
  EXPECT_EQ(completion.sequence(), 3);
  const auto statistics = domain->statistics().snapshot();
  EXPECT_EQ(statistics.synchronization.completion_polls, 1u);
  EXPECT_EQ(statistics.synchronization.completion_waits, 1u);
}

TEST(RuntimeCompletion, WaitReleasesResourcesExactlyOnce) {
  auto domain = std::make_shared<RuntimeFaultDomain>(Arch::vulkan, 17);
  auto semaphore = std::make_shared<FakeSemaphore>();
  auto completion = RuntimeCompletion::from_stream_semaphore(
      Arch::vulkan, 17, 9, semaphore, domain);
  std::atomic<int> released{0};
  completion.attach_resources(
      std::make_shared<CountedResources>(&released, 4));

  EXPECT_FALSE(completion.done());
  EXPECT_EQ(completion.retained_resource_count(7), 4);
  completion.wait();
  EXPECT_TRUE(completion.done());
  EXPECT_EQ(completion.retained_resource_count(7), 0);
  EXPECT_EQ(released.load(std::memory_order_relaxed), 1);
  completion.wait();
  EXPECT_EQ(released.load(std::memory_order_relaxed), 1);
  EXPECT_EQ(semaphore->waits, 1);
  const auto statistics = domain->statistics().snapshot();
  EXPECT_EQ(statistics.synchronization.completion_polls, 2u);
  EXPECT_EQ(statistics.synchronization.completion_waits, 2u);
}

TEST(RuntimeCompletion, FirstBackendErrorIsSticky) {
  auto semaphore = std::make_shared<FakeSemaphore>();
  semaphore->fail = true;
  auto completion = RuntimeCompletion::from_stream_semaphore(
      Arch::vulkan, 23, 5, semaphore);

  EXPECT_THROW(completion.done(), std::runtime_error);
  EXPECT_EQ(completion.first_error_message(), "fake completion failure");
  semaphore->fail = false;
  semaphore->ready = true;
  EXPECT_THROW(completion.done(), std::runtime_error);
  EXPECT_THROW(completion.wait(), std::runtime_error);
  EXPECT_EQ(completion.first_error_message(), "fake completion failure");
}

TEST(RuntimeCompletion, ProgramSyncRetiresFaultedResourcesWithoutHidingError) {
  auto semaphore = std::make_shared<FakeSemaphore>();
  semaphore->fail = true;
  auto completion = RuntimeCompletion::from_stream_semaphore(
      Arch::vulkan, 11, 7, semaphore);
  std::atomic<int> released{0};
  completion.attach_resources(
      std::make_shared<CountedResources>(&released, 1));
  EXPECT_THROW(completion.done(), std::runtime_error);
  EXPECT_EQ(released.load(std::memory_order_relaxed), 0);

  completion.mark_completed();
  EXPECT_EQ(released.load(std::memory_order_relaxed), 1);
  EXPECT_THROW(completion.wait(), std::runtime_error);
  EXPECT_NE(completion.first_error_message().find("fake completion failure"),
            std::string::npos);
}

TEST(RuntimeCompletion, FatalBackendErrorPoisonsOwningDomainWithSequence) {
  auto domain = std::make_shared<RuntimeFaultDomain>(Arch::vulkan, 81);
  auto semaphore = std::make_shared<FatalBackendSemaphore>();
  auto completion = RuntimeCompletion::from_stream_semaphore(
      Arch::vulkan, 81, 19, semaphore, domain);
  std::atomic<int> released{0};
  completion.attach_resources(
      std::make_shared<CountedResources>(&released, 2));

  EXPECT_THROW(completion.done(), BackendRuntimeError);
  const RuntimeFaultSnapshot snapshot = domain->snapshot();
  ASSERT_TRUE(snapshot.first_fault.has_value());
  EXPECT_EQ(snapshot.state, RuntimeLifecycleState::kFaulted);
  EXPECT_EQ(snapshot.first_fault->backend_code, -4);
  EXPECT_EQ(snapshot.first_fault->submission_sequence, 19u);
  EXPECT_EQ(snapshot.first_fault->operation, "vkGetFenceStatus");
  EXPECT_EQ(released.load(std::memory_order_relaxed), 0);

  completion.invalidate_and_release("runtime faulted");
  EXPECT_EQ(released.load(std::memory_order_relaxed), 1);
  EXPECT_THROW(completion.wait(), BackendRuntimeError);
  const auto statistics = domain->statistics().snapshot();
  EXPECT_EQ(statistics.synchronization.completion_polls, 1u);
  EXPECT_EQ(statistics.synchronization.completion_waits, 1u);
}

}  // namespace
}  // namespace taichi::lang
