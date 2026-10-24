#include <gtest/gtest.h>

#if defined(TI_WITH_VULKAN) && defined(TI_TEST_RUNTIME_C_API)
#include "c_api/src/taichi_vulkan_impl.h"
#include "taichi/program/program.h"
#include "taichi/program/runtime_fault.h"
#include "taichi/runtime/aot_runtime_client.h"
#include "taichi/runtime/vulkan_execution_domain.h"
#include "tests/cpp/aot/vulkan/timeline_gate.h"
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <iterator>

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

TEST(VulkanExecutionOwner, PrivateOffscreenIgnoresDefaultQueue) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  const char *path = std::getenv("TAICHI_VULKAN_OFFSCREEN_AOT_DIR");
  if (!path)
    GTEST_SKIP() << "Requires the compiled offscreen fixture";
  PublicDomainSession session;
  const char *selected = std::getenv("TI_TEST_VULKAN_DEVICE_INDEX");
  session.runtime = ti_create_runtime(TI_ARCH_VULKAN, selected ? std::stoul(selected) : 0);
  ASSERT_NE(session.runtime, nullptr);
  auto &runtime = *reinterpret_cast<VulkanRuntime *>(session.runtime);
  auto &device = runtime.get_vk();
  using Kind = vulkan::VulkanExecutionQueueKind;
  if (device.created_execution_queue_count(Kind::graphics) < 3 ||
      !device.vk_caps().timeline_semaphore)
    GTEST_SKIP() << "Requires two additional graphics+compute queues";
  session.module = ti_load_aot_module(session.runtime, path);
  ASSERT_NE(session.module, nullptr);
  std::ifstream input(std::string(path) + "/offscreen.json");
  ASSERT_TRUE(input.good());
  const std::string plan((std::istreambuf_iterator<char>(input)), {});
  const char *names[]{"output", "doutput", "loss", "dloss"};
  const char *parameter = "scale";
  const TiVulkanExecutionDomainCreateInfo options{
      1, 1, TI_TRUE, TI_VULKAN_EXECUTION_QUEUE_GRAPHICS};
  std::array<TiVulkanExecutionDomainInfo, 2> domain_info{};
  for (int lane = 0; lane < 2; ++lane) {
    auto domain = ti_create_vulkan_execution_domain_ext(session.runtime, &options);
    ASSERT_NE(domain, nullptr);
    session.domains.push_back(domain);
    ti_get_vulkan_execution_domain_info_ext(domain, &domain_info[lane]);
    EXPECT_EQ(domain_info[lane].queue_family_index, device.graphics_queue_family_index());
    std::array<TiNamedArgument, 5> args{};
    for (int i = 0; i < 4; ++i) {
      const uint32_t size = i < 2 ? 128 : 1;
      TiMemoryAllocateInfo allocation{size * sizeof(float), TI_FALSE, TI_TRUE, TI_FALSE,
                                     TI_MEMORY_USAGE_STORAGE_BIT};
      auto memory = ti_allocate_memory(session.runtime, &allocation);
      ASSERT_NE(memory, nullptr);
      session.memories.push_back(memory);
      args[i].name = names[i];
      args[i].argument.type = TI_ARGUMENT_TYPE_NDARRAY;
      auto &array = args[i].argument.value.ndarray;
      array.memory = memory;
      array.shape.dim_count = 1;
      array.shape.dims[0] = size;
      array.elem_type = TI_DATA_TYPE_F32;
    }
    args[4].name = parameter;
    args[4].argument.type = TI_ARGUMENT_TYPE_SCALAR;
    args[4].argument.value.scalar.type = TI_DATA_TYPE_U64;
    auto call = ti_prepare_execution_plan_on_vulkan_domain_ext(
        session.runtime, session.module, domain, plan.size(), plan.data(),
        args.size(), args.data(), 0, nullptr, 1, &parameter);
    check_domain_error();
    ASSERT_NE(call, nullptr);
    session.executions.push_back(call);
    ti_initialize_prepared_ext(session.runtime, call);
    check_domain_error();
  }
  const auto submit = [&](int lane, uint64_t value, bool sampled = false) {
    TiCompletion completion = nullptr;
    const bool accepted = ti_try_submit_prepared_sampled_ext(
        session.runtime, session.executions[lane], 0, nullptr, &completion,
        TI_FALSE, sampled, 1, &value);
    check_domain_error();
    if (accepted) {
      session.completions.push_back(completion);
    } else {
      EXPECT_EQ(completion, nullptr);
    }
    return completion;
  };
  const auto check = [&](int lane, float scale) {
    for (int derivative = 0; derivative < 2; ++derivative) {
      const auto memory = runtime.get_memory(session.memories[4 * lane + derivative]).allocation;
      void *mapped = nullptr;
      ASSERT_EQ(device.map(memory, &mapped), RhiResult::success);
      const auto *values = static_cast<const float *>(mapped);
      for (int i = 0; i < 128; ++i) {
        const float expected = i < 64 ? (i % 8 < 4 ? 0.5f * scale : 0)
                                       : (i % 8 < 4 ? 0 : scale);
        EXPECT_EQ(values[i], expected) << i;
      }
      device.unmap(memory);
    }
  };
  {
    test_detail::TimelineGate default_gate(device, 0);
    test_detail::TimelineGate left_gate(device, domain_info[0].queue_index, Kind::graphics);
    const auto before = device.queue_submission_snapshot().queue_submit_calls;
    auto left = submit(0, 2, true);
    ASSERT_NE(left, nullptr);
    EXPECT_EQ(device.queue_submission_snapshot().queue_submit_calls, before + 1);
    EXPECT_EQ(ti_query_completion_ext(left), TI_FALSE);
    EXPECT_EQ(submit(0, 99), nullptr);
    EXPECT_EQ(device.queue_submission_snapshot().queue_submit_calls, before + 1);
    auto right = submit(1, 3, true);
    ASSERT_NE(right, nullptr);
    ti_wait_completion_ext(right);
    check_domain_error();
    check(1, 3);
    EXPECT_EQ(ti_query_completion_ext(left), TI_FALSE);
    left_gate.release();
    ti_wait_completion_ext(left);
    check(0, 2);  // The rejected parameter update did not alter the accepted work.
    for (auto ticket : {left, right}) {
      TiCompletionGpuTiming timing{};
      ti_get_completion_gpu_timing_ext(ticket, &timing);
      check_domain_error();
      EXPECT_EQ(timing.status, TI_COMPLETION_GPU_TIMING_READY);
      EXPECT_EQ(timing.scope, 1u);
      EXPECT_GT(timing.duration_ns, 0u);
    }
    // Attachment/pipeline destruction waits only for this instance's prefix,
    // even while an unrelated default queue remains deliberately blocked.
    auto retired = ti_get_prepared_vulkan_instance_ext(session.runtime, session.executions[0]);
    ASSERT_NE(retired, nullptr);
    ti_retire_vulkan_execution_instance_ext(retired);
    ti_finish_prepared_instance_retirement_ext(session.runtime, session.executions[0]);
    check_domain_error();
    session.executions[0] = nullptr;
    ti_release_vulkan_execution_instance_ext(retired);
    auto tail = submit(1, 4);
    ASSERT_NE(tail, nullptr);
    ti_wait_completion_ext(tail);
    check(1, 4);
  }
}

TEST(VulkanExecutionOwner, PrivateRayInitializationAndIndependentRefit) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  const char *path = std::getenv("TAICHI_VULKAN_RAY_DOMAIN_AOT_DIR");
  if (!path)
    GTEST_SKIP() << "Requires the compiled private ray fixture";
  PublicDomainSession session;
  const char *selected = std::getenv("TI_TEST_VULKAN_DEVICE_INDEX");
  session.runtime = ti_create_runtime(TI_ARCH_VULKAN, selected ? std::stoul(selected) : 0);
  ASSERT_NE(session.runtime, nullptr);
  auto &runtime = *reinterpret_cast<VulkanRuntime *>(session.runtime);
  auto &device = runtime.get_vk();
  if (device.created_compute_queue_count() < 3 || !device.vk_caps().timeline_semaphore)
    GTEST_SKIP() << "Requires two independent compute queues";
  session.module = ti_load_aot_module(session.runtime, path);
  ASSERT_NE(session.module, nullptr);
  std::ifstream input(std::string(path) + "/instances.json");
  ASSERT_TRUE(input.good());
  const std::string plan((std::istreambuf_iterator<char>(input)), {});
  const char *names[]{"seed", "indices", "rays", "hits", "ids", "dhits", "loss", "initial_transforms"};
  const std::vector<std::vector<uint32_t>> shapes{
      {6, 3}, {2, 3}, {6, 8}, {6, 4}, {6, 4}, {6, 4}, {1}, {2, 3, 4}};
  const std::vector<float> vertices{-1, -1, 1, 1, -1, 1, 0, 1, 1,
                                   -1, -1, 2, 0, 1, 2, 1, -1, 2};
  const std::vector<int32_t> indices{0, 1, 2, 3, 4, 5};
  const std::vector<float> rays{0, 0, 0, .01f, 0, 0, 1, 10,
                               4, 0, 0, .01f, 0, 0, 1, 10,
                               0, 0, 0, .01f, 0, 0, 1, 1.5f,
                               0, 0, 1.5f, .01f, 0, 0, 1, 3,
                               8, 0, 0, .01f, 0, 0, 1, 10,
                               0, 0, 4, .01f, 0, 0, -1, 10};
  const std::vector<float> transforms{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
                                    1, 0, 0, 4, 0, 1, 0, 0, 0, 0, 1, 2};
  const char *parameter = "phase";
  const TiVulkanExecutionDomainCreateInfo options{1, 1, TI_TRUE, TI_VULKAN_EXECUTION_QUEUE_COMPUTE};
  std::array<TiVulkanExecutionDomainInfo, 2> domains{};
  std::array<std::array<TiNamedArgument, 9>, 2> arguments{};
  for (int lane = 0; lane < 2; ++lane) {
    auto domain = ti_create_vulkan_execution_domain_ext(session.runtime, &options);
    ASSERT_NE(domain, nullptr);
    session.domains.push_back(domain);
    ti_get_vulkan_execution_domain_info_ext(domain, &domains[lane]);
    for (size_t i = 0; i < shapes.size(); ++i) {
      uint32_t size = 1;
      for (auto extent : shapes[i])
        size *= extent;
      TiMemoryAllocateInfo allocation{size * sizeof(float), TI_TRUE, TI_TRUE, TI_FALSE,
                                     TI_MEMORY_USAGE_STORAGE_BIT};
      auto memory = ti_allocate_memory(session.runtime, &allocation);
      ASSERT_NE(memory, nullptr);
      session.memories.push_back(memory);
      auto &argument = arguments[lane][i];
      argument.name = names[i];
      argument.argument.type = TI_ARGUMENT_TYPE_NDARRAY;
      auto &array = argument.argument.value.ndarray;
      array.memory = memory;
      array.shape.dim_count = shapes[i].size();
      std::copy(shapes[i].begin(), shapes[i].end(), array.shape.dims);
      array.elem_type = (i == 1 || i == 4) ? TI_DATA_TYPE_I32 : TI_DATA_TYPE_F32;
      auto *mapped = ti_map_memory(session.runtime, memory);
      ASSERT_NE(mapped, nullptr);
      std::memset(mapped, 0, size * sizeof(float));
      if (i == 0) std::memcpy(mapped, vertices.data(), size * sizeof(float));
      if (i == 1) std::memcpy(mapped, indices.data(), size * sizeof(float));
      if (i == 2) std::memcpy(mapped, rays.data(), size * sizeof(float));
      if (i == 7) std::memcpy(mapped, transforms.data(), size * sizeof(float));
      ti_unmap_memory(session.runtime, memory);
    }
    arguments[lane][8].name = parameter;
    arguments[lane][8].argument.type = TI_ARGUMENT_TYPE_SCALAR;
    arguments[lane][8].argument.value.scalar.type = TI_DATA_TYPE_U64;
  }
  const auto prepare = [&](int lane) {
    return ti_prepare_execution_plan_on_vulkan_domain_ext(
        session.runtime, session.module, session.domains[lane], plan.size(), plan.data(),
        arguments[lane].size(), arguments[lane].data(), 0, nullptr, 1, &parameter);
  };
  session.executions.push_back(prepare(0));
  check_domain_error();
  ASSERT_NE(session.executions[0], nullptr);
  ti_initialize_prepared_ext(session.runtime, session.executions[0]);
  check_domain_error();
  {
    // Kernels are already registered. Reconstructing the peer's private AS
    // must not synchronously build on queue zero. Bootstrap can enqueue there;
    // initialize still waits for that cold prefix after the gate is released.
    test_detail::TimelineGate gate(device, 0);
    auto future = std::async(std::launch::async, [&] { return prepare(1); });
    const auto status = future.wait_for(std::chrono::seconds(5));
    gate.release();  // Always unblock before joining a failed implementation.
    session.executions.push_back(future.get());
    EXPECT_EQ(status, std::future_status::ready);
    ASSERT_NE(session.executions[1], nullptr);
  }
  ti_initialize_prepared_ext(session.runtime, session.executions[1]);
  check_domain_error();
  const auto submit = [&](int lane, uint64_t value) {
    TiCompletion ticket = nullptr;
    const bool accepted = ti_try_submit_prepared_sampled_ext(
        session.runtime, session.executions[lane], 0, nullptr, &ticket,
        TI_FALSE, TI_TRUE, 1, &value);
    check_domain_error();
    if (accepted)
      session.completions.push_back(ticket);
    else
      EXPECT_EQ(ticket, nullptr);
    return ticket;
  };
  const auto check = [&](int lane, float shift) {
    for (int column : {3, 5}) {
      void *mapped = nullptr;
      const auto allocation = runtime.get_memory(session.memories[lane * 8 + column]).allocation;
      ASSERT_EQ(device.map(allocation, &mapped), RhiResult::success);
      EXPECT_EQ(static_cast<const float *>(mapped)[0], 1 + shift);
      EXPECT_EQ(static_cast<const float *>(mapped)[4], 3 + shift);
      device.unmap(allocation);
    }
  };
  {
    test_detail::TimelineGate default_gate(device, 0);
    test_detail::TimelineGate left_gate(device, domains[0].queue_index);
    const auto before = device.queue_submission_snapshot().queue_submit_calls;
    auto left = submit(0, (uint64_t(1) << 48) | 1);
    ASSERT_NE(left, nullptr);
    EXPECT_EQ(device.queue_submission_snapshot().queue_submit_calls, before + 1);
    EXPECT_EQ(ti_query_completion_ext(left), TI_FALSE);
    EXPECT_EQ(submit(0, 7), nullptr);
    EXPECT_EQ(device.queue_submission_snapshot().queue_submit_calls, before + 1);
    auto right = submit(1, (uint64_t(1) << 49) | 2);
    ASSERT_NE(right, nullptr);
    ti_wait_completion_ext(right);
    check(1, 2);
    EXPECT_EQ(ti_query_completion_ext(left), TI_FALSE);
    left_gate.release();
    ti_wait_completion_ext(left);
    check(0, 1);
    auto retired = ti_get_prepared_vulkan_instance_ext(session.runtime, session.executions[0]);
    ASSERT_NE(retired, nullptr);
    ti_retire_vulkan_execution_instance_ext(retired);
    ti_finish_prepared_instance_retirement_ext(session.runtime, session.executions[0]);
    check_domain_error();
    session.executions[0] = nullptr;
    ti_release_vulkan_execution_instance_ext(retired);
    auto tail = submit(1, (uint64_t(1) << 49) | 3);
    ASSERT_NE(tail, nullptr);
    ti_wait_completion_ext(tail);
    check(1, 3);
    for (auto ticket : {left, right, tail}) {
      TiCompletionGpuTiming timing{};
      ti_get_completion_gpu_timing_ext(ticket, &timing);
      EXPECT_EQ(timing.status, TI_COMPLETION_GPU_TIMING_READY);
      EXPECT_GT(timing.duration_ns, 0u);
    }
    check_domain_error();
  }
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
