#include <gtest/gtest.h>

#if defined(TI_WITH_VULKAN) && defined(TI_TEST_RUNTIME_C_API)
#include "c_api/src/taichi_vulkan_impl.h"
#include "taichi/program/program.h"
#include "taichi/program/runtime_fault.h"
#include "taichi/runtime/aot_runtime_client.h"
#include "taichi/runtime/vulkan_execution_domain.h"
#include "tests/cpp/aot/vulkan/timeline_gate.h"
#include <array>
#include <cstdlib>

namespace taichi::lang {
namespace {
void check_domain_error() {
  char message[1024]{};
  uint64_t size = sizeof(message);
  ASSERT_EQ(ti_get_last_error(&size, message), TI_ERROR_SUCCESS) << message;
}

struct PublicDomainSession {
  TiRuntime runtime{nullptr};
  TiAotModule module{nullptr};
  std::vector<TiVulkanExecutionDomain> domains;
  std::vector<TiMemory> memories;
  std::vector<TiPreparedExecution> executions;
  std::vector<TiCompletion> completions;
  ~PublicDomainSession() {
    for (auto domain : domains)
      ti_close_vulkan_execution_domain_ext(domain);
    for (auto call : executions)
      if (call)
        ti_destroy_prepared_ext(runtime, call);
    for (auto completion : completions)
      ti_release_completion_ext(completion);
    for (auto domain : domains)
      ti_release_vulkan_execution_domain_ext(domain);
    if (module)
      ti_destroy_aot_module(module);
    for (auto memory : memories)
      ti_free_memory(runtime, memory);
    if (runtime)
      ti_destroy_runtime(runtime);
  }
};

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

TEST(VulkanExecutionOwner, PublicFixedPlanParametersAndIndependentRetirement) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  const char *path = std::getenv("TAICHI_VULKAN_PARAMETERS_AOT_DIR");
  if (!path)
    GTEST_SKIP() << "Requires the compiled Vulkan parameter fixture";
  PublicDomainSession session;
  const auto *selected = std::getenv("TI_TEST_VULKAN_DEVICE_INDEX");
  const auto device_index = selected ? std::stoul(selected) : 0;
  session.runtime = ti_create_runtime(TI_ARCH_VULKAN, device_index);
  ASSERT_NE(session.runtime, nullptr);
  auto &runtime = *reinterpret_cast<VulkanRuntime *>(session.runtime);
  auto &device = runtime.get_vk();
  RecordProperty("device_index", device_index);
  RecordProperty("compute_queues", device.created_compute_queue_count());
  RecordProperty("timeline_enabled", device.vk_caps().timeline_semaphore);
  if (!device.vk_caps().timeline_semaphore || device.created_compute_queue_count() < 3)
    GTEST_SKIP();
  session.module = ti_load_aot_module(session.runtime, path);
  ASSERT_NE(session.module, nullptr);
  const std::string plan = R"({"version":26,
    "resources":[{"kind":"ndarray","name":"seed","dtype":"u64","shape":[3]}],
    "initialization":[{"kind":"kernel","entry":"initialize","arguments":["seed"]}],
    "operations":[{"kind":"kernel","entry":"publish_from_seed","arguments":["epoch","step","timestamp","seed","header"]}]})";
  const char *names[]{"epoch", "step", "timestamp"};
  const TiVulkanExecutionDomainCreateInfo options{2, 1, TI_TRUE};
  std::array<TiNamedArgument, 4> args{};
  for (int i = 0; i < 3; ++i) {
    args[i].name = names[i];
    args[i].argument.type = TI_ARGUMENT_TYPE_SCALAR;
    args[i].argument.value.scalar.type = TI_DATA_TYPE_U64;
  }
  args[3].name = "header";
  args[3].argument.type = TI_ARGUMENT_TYPE_NDARRAY;
  auto &array = args[3].argument.value.ndarray;
  array.shape.dim_count = 1;
  array.shape.dims[0] = 3;
  array.elem_type = TI_DATA_TYPE_U64;
  const auto prepare = [&](int index) {
    array.memory = session.memories[index];
    return ti_prepare_execution_plan_on_vulkan_domain_ext(
        session.runtime, session.module, session.domains[index], plan.size(), plan.data(),
        args.size(), args.data(), 0, nullptr, 3, names);
  };
  for (int i = 0; i < 2; ++i) {
    session.domains.push_back(ti_create_vulkan_execution_domain_ext(session.runtime, &options));
    ASSERT_NE(session.domains.back(), nullptr);
    const TiMemoryAllocateInfo info{24, TI_FALSE, TI_TRUE, TI_FALSE, TI_MEMORY_USAGE_STORAGE_BIT};
    session.memories.push_back(ti_allocate_memory(session.runtime, &info));
    ASSERT_NE(session.memories.back(), nullptr);
    session.executions.push_back(prepare(i));
    check_domain_error();
    ASSERT_NE(session.executions.back(), nullptr);
    ti_initialize_prepared_ext(session.runtime, session.executions.back());
    check_domain_error();
  }
  // Initializer and replay together consume one logical instance slot.
  EXPECT_EQ(prepare(0), nullptr);
  EXPECT_NE(ti_get_last_error(nullptr, nullptr), TI_ERROR_SUCCESS);
  ti_set_last_error(TI_ERROR_SUCCESS, nullptr);
  const auto values = [](uint64_t seed) {
    return std::array<uint64_t, 3>{(uint64_t{1} << 63) + seed,
        (uint64_t{1} << 53) + seed * 3, (uint64_t{1} << 61) + seed * 7};
  };
  const auto check = [&](int index, const std::array<uint64_t, 3> &expected,
                         uint64_t seed) {
    const auto memory = runtime.get_memory(session.memories[index]).allocation;
    void *mapped = nullptr;
    ASSERT_EQ(device.map(memory, &mapped), RhiResult::success);
    const auto *data = static_cast<const uint64_t *>(mapped);
    for (int i = 0; i < 3; ++i)
      EXPECT_EQ(data[i], expected[i] + seed);
    device.unmap(memory);
  };
  TiCompletion old = nullptr;
  {
    test_detail::TimelineGate default_gate(device, 0);
    for (uint64_t iteration = 0; iteration < 25; ++iteration) {
      std::array<TiCompletion, 2> pair{};
      for (int i = 0; i < 2; ++i) {
        auto input = values(iteration * 2 + i + 1);
        const auto before = device.queue_submission_snapshot().queue_submit_calls;
        ASSERT_TRUE(ti_try_submit_prepared_sampled_ext(session.runtime, session.executions[i],
            0, nullptr, &pair[i], TI_FALSE, iteration == 0, 3, input.data()));
        check_domain_error();
        session.completions.push_back(pair[i]);
        input.fill(0);
        EXPECT_EQ(device.queue_submission_snapshot().queue_submit_calls, before + 1);
      }
      for (auto completion : pair)
        ti_wait_completion_ext(completion);
      check_domain_error();
      check(0, values(iteration * 2 + 1), iteration + 7);
      check(1, values(iteration * 2 + 2), iteration + 7);
      if (iteration == 0)
        old = pair[0];
    }
    TiCompletionGpuTiming timing{};
    ti_get_completion_gpu_timing_ext(old, &timing);
    check_domain_error();
    EXPECT_EQ(timing.status, device.queue_timestamp_valid_bits(device.compute_queue_family_index())
                                 ? TI_COMPLETION_GPU_TIMING_READY
                                 : TI_COMPLETION_GPU_TIMING_UNSUPPORTED);
    EXPECT_EQ(timing.scope, 1u);
    auto retired = ti_get_prepared_vulkan_instance_ext(session.runtime, session.executions[0]);
    ASSERT_NE(retired, nullptr);
    // Neither retirement step may wait for the unrelated default queue.
    ti_retire_vulkan_execution_instance_ext(retired);
    ti_finish_prepared_instance_retirement_ext(session.runtime, session.executions[0]);
    check_domain_error();
    session.executions[0] = nullptr;
    ti_release_vulkan_execution_instance_ext(retired);
    TiCompletionGpuTiming after{};
    ti_get_completion_gpu_timing_ext(old, &after);
    EXPECT_EQ(after.status, timing.status);
    EXPECT_EQ(after.duration_ns, timing.duration_ns);
  }
  // Capacity was reclaimed; old completion handles remain independently valid.
  session.executions[0] = prepare(0);
  ASSERT_NE(session.executions[0], nullptr);
  ti_initialize_prepared_ext(session.runtime, session.executions[0]);
  EXPECT_EQ(ti_query_completion_ext(old), TI_TRUE);
  check_domain_error();
}

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
