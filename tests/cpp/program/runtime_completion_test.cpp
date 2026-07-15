#include "gtest/gtest.h"

#include <atomic>
#include <memory>
#include <stdexcept>

#include "taichi/program/runtime_completion.h"

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

TEST(RuntimeCompletion, CompletedTokenUsesNoBackendWork) {
  auto completion = RuntimeCompletion::completed(Arch::x64, 11, 3);
  EXPECT_TRUE(completion.valid());
  EXPECT_TRUE(completion.done());
  EXPECT_FALSE(completion.has_backend_work());
  EXPECT_EQ(completion.program_domain(), 11);
  EXPECT_EQ(completion.sequence(), 3);
}

TEST(RuntimeCompletion, WaitReleasesResourcesExactlyOnce) {
  auto semaphore = std::make_shared<FakeSemaphore>();
  auto completion = RuntimeCompletion::from_stream_semaphore(
      Arch::vulkan, 17, 9, semaphore);
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

}  // namespace
}  // namespace taichi::lang
