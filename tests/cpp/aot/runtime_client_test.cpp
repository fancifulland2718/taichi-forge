#include "taichi/runtime/runtime_client.h"

#include <functional>
#include <atomic>
#include <chrono>
#include <future>
#include <stdexcept>
#include "gtest/gtest.h"
#if defined(TI_WITH_LLVM) || defined(TI_WITH_VULKAN)
#include "taichi/program/program.h"
#endif
#ifdef TI_WITH_CUDA
#include "taichi/platform/cuda/detect_cuda.h"
#endif
#ifdef TI_WITH_VULKAN
#include "taichi/rhi/vulkan/vulkan_loader.h"
#endif

namespace taichi::lang {
namespace {
class TestClient : public RuntimeClient {
 public:
  explicit TestClient(std::function<void()> retire,
                      std::function<void()> destroy = {})
      : retire_(std::move(retire)), destroy_(std::move(destroy)) {
  }
  ~TestClient() override {
    if (destroy_)
      destroy_();
  }
  void retire() override {
    retire_();
  }

 private:
  std::function<void()> retire_;
  std::function<void()> destroy_;
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

TEST(RuntimeClientRegistry, RetainedBorrowerSurvivesWrapperRelease) {
  RuntimeClientRegistry registry;
  int retired = 0;
  auto client = std::make_shared<TestClient>([&] { ++retired; });
  std::weak_ptr<TestClient> weak = client;
  registry.retain(client);
  EXPECT_THROW(registry.attach(client), std::invalid_argument);
  client.reset();
  EXPECT_FALSE(weak.expired());
  registry.retire();
  EXPECT_EQ(retired, 1);
  EXPECT_TRUE(weak.expired());
}

TEST(RuntimeClientRegistry, ExplicitCloseDetachesRetainedBorrower) {
  RuntimeClientRegistry registry;
  int retired = 0;
  bool destroyed = false;
  auto client = std::make_shared<TestClient>([&] { ++retired; },
                                             [&] {
                                               EXPECT_FALSE(registry.closed());
                                               destroyed = true;
                                             });
  std::weak_ptr<TestClient> weak = client;
  registry.retain(client);
  client->retire();
  auto *identity = client.get();
  client.reset();
  registry.detach(identity);
  EXPECT_TRUE(destroyed);
  EXPECT_TRUE(weak.expired());
  registry.retire();
  EXPECT_EQ(retired, 1);
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
#if defined(TI_WITH_LLVM) || defined(TI_WITH_VULKAN)
namespace {
void check_program_retirement(Arch arch) {
  Program program(arch);
  program.materialize_runtime();
  auto *device = program.get_compute_device();
  Device::AllocParams params;
  params.size = 64;
  params.usage = AllocUsage::Storage;
  DeviceAllocation allocation;
  ASSERT_EQ(device->allocate_memory(params, &allocation), RhiResult::success);
  int retired = 0;
  auto client = std::make_shared<TestClient>([&] {
    EXPECT_EQ(program.runtime_fault_snapshot().state,
              RuntimeLifecycleState::kFinalizing);
    // The finalizer itself may drain backend work. Other threads must still
    // fail admission while its callbacks run.
    std::async(std::launch::async, [&] {
      EXPECT_THROW(program.ensure_runtime_submission_allowed("retirement test"),
                   std::exception);
    }).get();
    device->dealloc_memory(allocation);
    ++retired;
  });
  program.attach_runtime_client(client);
  int retained_retired = 0;
  auto retained = std::make_shared<TestClient>([&] { ++retained_retired; });
  std::weak_ptr<TestClient> retained_weak = retained;
  program.retain_runtime_client(retained);
  retained.reset();
  EXPECT_FALSE(retained_weak.expired());
  program.finalize();
  EXPECT_EQ(retired, 1);
  EXPECT_EQ(retained_retired, 1);
  EXPECT_TRUE(retained_weak.expired());
  program.finalize();
  EXPECT_EQ(retired, 1);
  EXPECT_THROW(program.attach_runtime_client(client), std::exception);
}
}  // namespace
#endif

#ifdef TI_WITH_LLVM
TEST(RuntimeClientRegistry, CpuProgramRetiresBeforeDeviceDestruction) {
  check_program_retirement(host_arch());
}

TEST(RuntimeClientRegistry, ConcurrentProgramFinalizationWaitsForClient) {
  Program program(host_arch());
  program.materialize_runtime();
  std::promise<void> entered, release, second_started;
  auto release_future = release.get_future();
  int retired = 0;
  auto client = std::make_shared<TestClient>([&] {
    program.finalize();  // Same-thread reentry must not drain twice.
    entered.set_value();
    release_future.wait();
    ++retired;
  });
  program.retain_runtime_client(client);
  auto first = std::async(std::launch::async, [&] { program.finalize(); });
  entered.get_future().wait();
  auto second = std::async(std::launch::async, [&] {
    second_started.set_value();
    program.finalize();
  });
  second_started.get_future().wait();
  EXPECT_EQ(second.wait_for(std::chrono::milliseconds(20)),
            std::future_status::timeout);
  release.set_value();
  first.get();
  second.get();
  EXPECT_EQ(retired, 1);
}
#endif
#ifdef TI_WITH_CUDA
TEST(RuntimeClientRegistry, CudaProgramRetiresBeforeDeviceDestruction) {
  if (!is_cuda_api_available())
    GTEST_SKIP() << "CUDA device unavailable";
  check_program_retirement(Arch::cuda);
}
#endif
#ifdef TI_WITH_VULKAN
TEST(RuntimeClientRegistry, VulkanProgramRetiresBeforeDeviceDestruction) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP() << "Vulkan device unavailable";
  check_program_retirement(Arch::vulkan);
}
#endif
}  // namespace taichi::lang
