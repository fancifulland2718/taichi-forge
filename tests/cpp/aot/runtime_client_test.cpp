#include "taichi/runtime/runtime_client.h"

#include <functional>
#include <atomic>
#include <chrono>
#include <future>
#include <stdexcept>
#include "gtest/gtest.h"

namespace taichi::lang {
namespace {
class TestClient : public RuntimeClient {
 public:
  explicit TestClient(std::function<void()> retire)
      : retire_(std::move(retire)) {
  }
  void retire() override {
    retire_();
  }

 private:
  std::function<void()> retire_;
};
}  // namespace

TEST(RuntimeClientRegistry, WeakOwnershipAndClosedAdmission) {
  RuntimeClientRegistry registry;
  int retired = 0;
  auto first = std::make_shared<TestClient>([&] { ++retired; });
  auto second = std::make_shared<TestClient>([&] { ++retired; });
  registry.attach(first);
  registry.attach(second);
  EXPECT_THROW(registry.attach(first), std::invalid_argument);
  EXPECT_THROW(registry.attach(nullptr), std::invalid_argument);
  std::weak_ptr<TestClient> released = first;
  first.reset();
  EXPECT_TRUE(released.expired());
  EXPECT_FALSE(registry.closed());
  registry.retire();
  EXPECT_TRUE(registry.closed());
  EXPECT_EQ(retired, 1);
  registry.retire();
  EXPECT_EQ(retired, 1);
  EXPECT_THROW(registry.attach(second), std::runtime_error);
}

TEST(RuntimeClientRegistry, FailureDoesNotBlockRemainingClients) {
  RuntimeClientRegistry registry;
  int retired = 0;
  auto failing = std::make_shared<TestClient>([&] {
    ++retired;
    EXPECT_TRUE(registry.closed());
    throw std::runtime_error("retirement failed");
  });
  auto remaining = std::make_shared<TestClient>([&] {
    ++retired;
    EXPECT_TRUE(registry.closed());
  });
  registry.attach(failing);
  registry.attach(remaining);
  EXPECT_THROW(registry.retire(), std::runtime_error);
  EXPECT_EQ(retired, 2);
  EXPECT_NO_THROW(registry.retire());
}

TEST(RuntimeClientRegistry, CleanupRunsOutsideRegistryLock) {
  RuntimeClientRegistry registry;
  auto unexpected = std::make_shared<TestClient>([] {});
  auto client = std::make_shared<TestClient>([&] {
    EXPECT_TRUE(registry.closed());
    EXPECT_THROW(registry.attach(unexpected), std::runtime_error);
    registry.retire();
  });
  registry.attach(client);
  EXPECT_NO_THROW(registry.retire());
}

TEST(RuntimeClientRegistry, ConcurrentRetirementWaitsForClientCompletion) {
  RuntimeClientRegistry registry;
  std::promise<void> entered, release, second_started;
  auto release_future = release.get_future();
  std::atomic<bool> completed{false};
  auto client = std::make_shared<TestClient>([&] {
    entered.set_value();
    release_future.wait();
    completed = true;
  });
  registry.attach(client);
  auto first = std::async(std::launch::async, [&] { registry.retire(); });
  entered.get_future().wait();
  auto second = std::async(std::launch::async, [&] {
    second_started.set_value();
    registry.retire();
    EXPECT_TRUE(completed.load());
  });
  second_started.get_future().wait();
  EXPECT_EQ(second.wait_for(std::chrono::milliseconds(20)),
            std::future_status::timeout);
  release.set_value();
  first.get();
  second.get();
}
}  // namespace taichi::lang
