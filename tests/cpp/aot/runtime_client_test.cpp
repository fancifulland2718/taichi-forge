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

TEST(RuntimeClientRegistry, SNodeRetirementIsSelectiveAndRetainsUntilDrain) {
  RuntimeClientRegistry registry;
  int selected = 0, peer = 0, global = 0;
  auto first = std::make_shared<TestClient>([&] {
    EXPECT_TRUE(registry.closed());
    ++selected;
  });
  auto second = std::make_shared<TestClient>([&] { ++peer; });
  auto third = std::make_shared<TestClient>([&] { ++global; });
  registry.retain(first, {7, 3, 7});
  registry.retain(second, {9});
  registry.retain(third);
  std::weak_ptr<TestClient> weak = first;
  first.reset();
  EXPECT_FALSE(weak.expired());
  registry.retire_snode_tree(3);
  EXPECT_TRUE(weak.expired());
  EXPECT_EQ(selected, 1);
  EXPECT_EQ(peer, 0);
  EXPECT_EQ(global, 0);
  EXPECT_FALSE(registry.closed());
  registry.retire_snode_tree(7);
  EXPECT_EQ(selected, 1);
  EXPECT_THROW(registry.retain(second, {-1}), std::invalid_argument);
  registry.retire();
  EXPECT_EQ(peer, 1);
  EXPECT_EQ(global, 1);
}

TEST(RuntimeClientRegistry, FailedSNodeDrainKeepsOwnerAndCanRetry) {
  RuntimeClientRegistry registry;
  int attempts = 0, other = 0;
  RuntimeClient *identity = nullptr;
  auto client = std::make_shared<TestClient>([&] {
    registry.detach(identity);
    if (++attempts == 1)
      throw std::runtime_error("pending device work");
  });
  identity = client.get();
  auto remaining = std::make_shared<TestClient>([&] { ++other; });
  registry.retain(client, {2});
  registry.retain(remaining, {2});
  std::weak_ptr<TestClient> weak = client;
  client.reset();
  EXPECT_THROW(registry.retire_snode_tree(2), std::runtime_error);
  EXPECT_EQ(other, 1);
  EXPECT_FALSE(weak.expired());
  EXPECT_FALSE(registry.closed());
  registry.retire_snode_tree(2);
  EXPECT_TRUE(weak.expired());
  EXPECT_EQ(attempts, 2);
  EXPECT_EQ(other, 1);
  registry.retire();
}

TEST(RuntimeClientRegistry, OwnerRetirementWaitsForSNodeDrain) {
  RuntimeClientRegistry registry;
  std::promise<void> entered, release, owner_started;
  auto release_future = release.get_future();
  std::atomic<bool> drained{false};
  int retired = 0;
  auto selected = std::make_shared<TestClient>([&] {
    EXPECT_THROW(registry.retire(), std::logic_error);
    EXPECT_THROW(registry.retire_snode_tree(4), std::logic_error);
    entered.set_value();
    release_future.wait();
    drained = true;
    ++retired;
  });
  auto peer = std::make_shared<TestClient>([&] {
    EXPECT_TRUE(drained.load());
    ++retired;
  });
  registry.retain(selected, {4});
  registry.retain(peer);
  auto tree =
      std::async(std::launch::async, [&] { registry.retire_snode_tree(4); });
  entered.get_future().wait();
  EXPECT_THROW(registry.attach(std::make_shared<TestClient>([] {})),
               std::runtime_error);
  auto owner = std::async(std::launch::async, [&] {
    owner_started.set_value();
    registry.retire();
  });
  owner_started.get_future().wait();
  EXPECT_EQ(owner.wait_for(std::chrono::milliseconds(20)),
            std::future_status::timeout);
  release.set_value();
  tree.get();
  owner.get();
  EXPECT_EQ(retired, 2);
  EXPECT_TRUE(registry.closed());
}
#if defined(TI_WITH_LLVM) || defined(TI_WITH_VULKAN)
namespace {
void check_program_retirement(Arch arch) {
  Program program(arch);
  const auto fault_domain = program.runtime_fault_domain();
  EXPECT_EQ(fault_domain, program.runtime_fault_domain());
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
    EXPECT_EQ(fault_domain->state(), RuntimeLifecycleState::kFinalizing);
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
  EXPECT_EQ(fault_domain->state(), RuntimeLifecycleState::kFinalized);
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

TEST(RuntimeClientRegistry, ProgramSNodeFailurePreservesTreeAndOtherClients) {
  Program program(host_arch());
  program.materialize_runtime();
  auto *first = program.add_snode_tree(
      std::make_unique<SNode>(0, SNodeType::root), false);
  auto *second = program.add_snode_tree(
      std::make_unique<SNode>(0, SNodeType::root), false);
  const int first_id = first->id(), second_id = second->id();
  const auto first_dependencies =
      program.snapshot_snode_tree_dependencies({first_id});
  int attempts = 0, peer = 0;
  auto selected = std::make_shared<TestClient>([&] {
    EXPECT_THROW(program.finalize(), std::logic_error);
    if (++attempts == 1)
      throw std::runtime_error("pending JIT code/root use");
  });
  auto other = std::make_shared<TestClient>([&] { ++peer; });
  program.retain_runtime_client(selected, first_dependencies);
  program.retain_runtime_client(
      other, program.snapshot_snode_tree_dependencies({second_id}));
  EXPECT_THROW(program.destroy_snode_tree(first), std::runtime_error);
  EXPECT_EQ(program.snapshot_snode_tree_dependencies({first_id}).size(), 1);
  EXPECT_EQ(peer, 0);
  program.destroy_snode_tree(first);
  EXPECT_EQ(attempts, 2);
  EXPECT_EQ(peer, 0);
  EXPECT_ANY_THROW(program.retain_runtime_client(selected, first_dependencies));
  auto *replacement = program.add_snode_tree(
      std::make_unique<SNode>(0, SNodeType::root), false);
  ASSERT_EQ(replacement->id(), first_id);
  EXPECT_NE(replacement->generation(), first_dependencies[0].generation);
  EXPECT_ANY_THROW(program.retain_runtime_client(selected, first_dependencies));
  program.destroy_snode_tree(replacement);
  EXPECT_EQ(attempts, 2);
  EXPECT_EQ(program.snapshot_snode_tree_dependencies({second_id}).size(), 1);
  program.destroy_snode_tree(second);
  EXPECT_EQ(peer, 1);
  program.finalize();
  EXPECT_EQ(attempts, 2);
  EXPECT_EQ(peer, 1);
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

TEST(RuntimeClientRegistry, ProgramFinalizationCannotOvertakeSNodeRetirement) {
  Program program(host_arch());
  program.materialize_runtime();
  auto *tree = program.add_snode_tree(
      std::make_unique<SNode>(0, SNodeType::root), false);
  std::promise<void> entered, release, finalizer_started;
  auto release_future = release.get_future();
  std::atomic<bool> tree_drained{false};
  auto selected = std::make_shared<TestClient>([&] {
    entered.set_value();
    release_future.wait();
    tree_drained = true;
  });
  auto owner =
      std::make_shared<TestClient>([&] { EXPECT_TRUE(tree_drained.load()); });
  program.retain_runtime_client(
      selected, program.snapshot_snode_tree_dependencies({tree->id()}));
  program.retain_runtime_client(owner);
  auto destroy =
      std::async(std::launch::async, [&] { program.destroy_snode_tree(tree); });
  entered.get_future().wait();
  auto finalize = std::async(std::launch::async, [&] {
    finalizer_started.set_value();
    program.finalize();
  });
  finalizer_started.get_future().wait();
  EXPECT_EQ(finalize.wait_for(std::chrono::milliseconds(20)),
            std::future_status::timeout);
  EXPECT_EQ(program.runtime_fault_domain()->state(),
            RuntimeLifecycleState::kHealthy);
  release.set_value();
  destroy.get();
  finalize.get();
  EXPECT_THROW(program.destroy_snode_tree(nullptr), std::runtime_error);
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
