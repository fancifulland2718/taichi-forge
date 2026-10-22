#include <gtest/gtest.h>

#if defined(TI_WITH_VULKAN) && defined(TI_TEST_RUNTIME_C_API)
#include "c_api/src/taichi_vulkan_impl.h"
#include "taichi/program/program.h"
#include "taichi/program/runtime_fault.h"
#include "taichi/runtime/aot_runtime_client.h"
#include "taichi/runtime/vulkan_execution_domain.h"

namespace taichi::lang {
namespace {
class DomainStorage final : public RuntimeClient {
 public:
  DeviceAllocationUnique storage;
  std::shared_ptr<const vulkan::ExecutionDomain::Commands> commands;
  std::function<void()> on_retire;
  void retire() override {
    if (on_retire)
      on_retire();
    commands.reset();
    storage.reset();
  }
};

std::shared_ptr<vulkan::ExecutionInstance> prepare_fill(
    vulkan::ExecutionDomain &domain, vulkan::VulkanDevice &device,
    const std::shared_ptr<DomainStorage> &client) {
  return domain.prepare_owned_instance(client, [&](const auto &) {
    auto allocation = device.allocate_memory_unique(
        {4, false, true, false, AllocUsage::Storage});
    if (allocation.second != RhiResult::success)
      throw std::runtime_error("test allocation failed");
    client->storage = std::move(allocation.first);
    auto commands = domain.new_command_list();
    commands->buffer_fill(client->storage->get_ptr(), 4, 83);
    commands->memory_barrier();
    client->commands = domain.prepare_commands({commands.get()});
  });
}

void expect_written(vulkan::VulkanDevice &device, const DomainStorage &client) {
  void *mapped = nullptr;
  ASSERT_EQ(device.map(*client.storage, &mapped), RhiResult::success);
  EXPECT_EQ(*static_cast<const std::uint32_t *>(mapped), 83u);
  device.unmap(*client.storage);
}
}  // namespace

TEST(VulkanExecutionOwner, BorrowedViewsShareProgramRegistryAndRetireBeforeDevice) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  Program program(Arch::vulkan);
  program.materialize_runtime();
  auto &device = static_cast<vulkan::VulkanDevice &>(*program.get_compute_device());
  if (!device.vk_caps().timeline_semaphore || device.created_compute_queue_count() < 2)
    GTEST_SKIP();
  auto registry = program.vulkan_execution_domains();
  EXPECT_EQ(program.vulkan_execution_domains(), registry);
  auto first = AotRuntimeClient::create(program);
  auto second = AotRuntimeClient::create(program);
  auto *first_runtime = reinterpret_cast<::Runtime *>(first->handle());
  auto *second_runtime = reinterpret_cast<::Runtime *>(second->handle());
  EXPECT_EQ(first_runtime->vulkan_execution_domains(), registry);
  EXPECT_EQ(second_runtime->vulkan_execution_domains(), registry);
  auto domain = first_runtime->vulkan_execution_domains()->create({1, 1});
  auto client = std::make_shared<DomainStorage>();
  int retired = 0;
  client->on_retire = [&] {
    EXPECT_EQ(program.runtime_fault_domain()->state(), RuntimeLifecycleState::kFinalizing);
    EXPECT_NE(program.get_compute_device(), nullptr);
    ++retired;
  };
  auto instance = prepare_fill(*domain, device, client);
  auto ticket = domain->submit_instance(instance, client->commands);
  first->close();
  EXPECT_TRUE(first->closed());
  EXPECT_TRUE(registry->accepting());
  EXPECT_TRUE(domain->snapshot().accepting);
  ticket = domain->submit_instance(instance, client->commands);
  program.synchronize();
  EXPECT_TRUE(ticket.done());
  expect_written(device, *client);
  const auto prefix = ticket.reusable_stream_dependency(
      Arch::vulkan, program.runtime_fault_domain()->program_domain(),
      program.runtime_fault_domain());
  program.finalize();
  EXPECT_EQ(retired, 1);
  EXPECT_FALSE(client->storage);
  EXPECT_TRUE(second->closed());
  EXPECT_TRUE(domain->snapshot().closed);
  EXPECT_TRUE(domain->instance_retired(instance));
  EXPECT_TRUE(ticket.done());
  EXPECT_TRUE(prefix->is_ready());
  EXPECT_TRUE(prefix->wait());
  EXPECT_ANY_THROW(registry->create({1, 1}));
}

TEST(VulkanExecutionOwner, StandaloneNativeDestructionDrainsBeforeDeviceExit) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  for (bool public_destroy : {false, true}) {
    auto runtime = std::make_unique<VulkanRuntimeOwned>(
        vulkan::VulkanDeviceCreator::Params{});
    auto &device = runtime->get_vk();
    if (!device.vk_caps().timeline_semaphore || device.created_compute_queue_count() < 2)
      GTEST_SKIP();
    auto owner = runtime->runtime_fault_domain();
    EXPECT_EQ(device.backend_fault_reporter(), owner);
    auto domain = runtime->vulkan_execution_domains()->create({1, 1});
    auto client = std::make_shared<DomainStorage>();
    int retired = 0;
    client->on_retire = [&] {
      expect_written(device, *client);  // Accepted GPU work was drained first.
      ++retired;
    };
    auto instance = prepare_fill(*domain, device, client);
    auto ticket = domain->submit_instance(instance, client->commands);
    const auto prefix = ticket.reusable_stream_dependency(
        Arch::vulkan, owner->program_domain(), owner);
    if (public_destroy) {
      auto handle = reinterpret_cast<TiRuntime>(static_cast<::Runtime *>(runtime.release()));
      ti_set_last_error(TI_ERROR_SUCCESS, "");
      ti_destroy_runtime(handle);
      EXPECT_EQ(ti_get_last_error(nullptr, nullptr), TI_ERROR_SUCCESS);
    } else {
      runtime.reset();
    }
    EXPECT_EQ(retired, 1);
    EXPECT_FALSE(client->storage);
    EXPECT_EQ(owner->state(), RuntimeLifecycleState::kFinalized);
    EXPECT_TRUE(domain->snapshot().closed);
    EXPECT_TRUE(domain->instance_retired(instance));
    EXPECT_TRUE(ticket.done());
    EXPECT_NO_THROW(ticket.wait());
    EXPECT_TRUE(prefix->is_ready());
    EXPECT_TRUE(prefix->wait());
  }
}

TEST(VulkanExecutionOwner, IndependentRuntimeHealthDoesNotLeakAcrossDevices) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  auto first = std::make_unique<VulkanRuntimeOwned>(
      vulkan::VulkanDeviceCreator::Params{});
  auto second = std::make_unique<VulkanRuntimeOwned>(
      vulkan::VulkanDeviceCreator::Params{});
  for (auto *runtime : {first.get(), second.get()}) {
    auto &device = runtime->get_vk();
    if (!device.vk_caps().timeline_semaphore || device.created_compute_queue_count() < 2)
      GTEST_SKIP();
  }
  auto first_owner = first->runtime_fault_domain();
  auto second_owner = second->runtime_fault_domain();
  EXPECT_NE(first_owner, second_owner);
  EXPECT_EQ(first->get_vk().backend_fault_reporter(), first_owner);
  EXPECT_EQ(second->get_vk().backend_fault_reporter(), second_owner);
  // A stale detach for another owner cannot erase this device's reporter.
  second->get_vk().clear_backend_fault_reporter(first_owner);
  EXPECT_EQ(second->get_vk().backend_fault_reporter(), second_owner);
  auto left = first->vulkan_execution_domains()->create({1, 1});
  auto right = second->vulkan_execution_domains()->create({1, 1});
  auto left_ticket = left->submit(left->prepare_commands({}));
  auto right_ticket = right->submit(right->prepare_commands({}));
  EXPECT_THROW(right->submit(right->prepare_commands({}), {left_ticket}),
                std::invalid_argument);
  first.reset();
  EXPECT_TRUE(left_ticket.done());
  EXPECT_EQ(first_owner->state(), RuntimeLifecycleState::kFinalized);
  EXPECT_EQ(second_owner->state(), RuntimeLifecycleState::kHealthy);
  EXPECT_TRUE(right->snapshot().accepting);
  right->submit(right->prepare_commands({}), {right_ticket}).wait();
  second.reset();
  EXPECT_TRUE(right_ticket.done());
}
}  // namespace taichi::lang
#endif
