#include <gtest/gtest.h>

#if defined(TI_WITH_CUDA) && defined(TI_WITH_VULKAN) && \
    defined(TI_TEST_RUNTIME_C_API)
#include <future>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <thread>
#include "tests/cpp/rhi/common/cuda_external_gate.h"
#include "c_api/src/taichi_llvm_impl.h"
#include "c_api/src/execution_domain.h"
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/rhi/interop/vulkan_cuda_interop.h"
#include "taichi/program/program.h"
#include "taichi/program/storage_view.h"
#include "c_api/src/execution_resources.h"
#include "taichi/program/graph_builder.h"
#include "taichi/ir/ir_builder.h"
#include "taichi/runtime/aot_runtime_client.h"
#include "taichi/runtime/program_impls/llvm/llvm_program.h"
#include "taichi/runtime/cuda/execution_domain.h"
#include "taichi/system/profiler_annotation.h"

namespace taichi::lang {
namespace {

struct RecoverableInitializationProbe {
  CUDADriver &driver{CUDADriver::get_instance()};
  decltype(driver.stream_synchronize) saved_wait{driver.stream_synchronize};
  decltype(driver.module_unload) saved_unload{driver.module_unload};
  int waits{0};
  int unloads{0};
  static inline RecoverableInitializationProbe *active{nullptr};

  RecoverableInitializationProbe() {
    active = this;
    driver.stream_synchronize.set(reinterpret_cast<void *>(&wait));
    driver.module_unload.set(reinterpret_cast<void *>(&unload));
  }
  ~RecoverableInitializationProbe() {
    driver.stream_synchronize = std::move(saved_wait);
    driver.module_unload = std::move(saved_unload);
    active = nullptr;
  }
  static uint32 wait(void *stream) {
    // The second runtime result fetch happens after persistent runtime storage
    // was allocated. A transient operation error must release that owner.
    if (++active->waits == 2)
      return 2;  // CUDA_ERROR_OUT_OF_MEMORY, operation-local classification.
    return active->saved_wait.call_concurrent(stream);
  }
  static uint32 unload(void *module) {
    ++active->unloads;
    return active->saved_unload.call_concurrent(module);
  }
};

TEST(CudaExecutionOwner, RecoverableInitializationRetiresModulesAndAllowsRetry) {
  auto context = CUDAContext::get_instance().get_guard();
  for (int retry = 0; retry < 2; ++retry) {
    RecoverableInitializationProbe probe;
    try {
      capi::LlvmRuntime runtime(Arch::cuda);
      FAIL() << "The initialization fault was not injected";
    } catch (const BackendRuntimeError &error) {
      EXPECT_EQ(error.backend_code(), 2);
    }
    EXPECT_GE(probe.waits, 3);  // Explicit healthy retirement performed a drain.
    EXPECT_GT(probe.unloads, 0);  // Failed executor was destroyed, not leaked.
  }
  capi::LlvmRuntime recovered(Arch::cuda);
  EXPECT_EQ(recovered.get().backend_fault_reporter(),
            recovered.runtime_fault_domain());
  EXPECT_NO_THROW(static_cast<::Runtime &>(recovered).wait());
}

TEST(CudaExecutionOwner, ExternalStorageLoanPreservesCompleteAotEpoch) {
  const auto *folder = std::getenv("TAICHI_F3_CONTEXT_AOT_FOLDER");
  if (!folder || !cuda::ExecutionDomain::available())
    GTEST_SKIP();
  Program program(Arch::cuda);
  program.materialize_runtime();
  // Direct driver calls below have the same explicit context obligation as
  // an external native consumer; runtime entry points restore their caller.
  auto context_guard = CUDAContext::get_instance().get_guard();
  auto &device = static_cast<cuda::CudaDevice &>(*program.get_compute_device());
  vulkan::VulkanDeviceCreator creator({});
  auto *vk = static_cast<vulkan::VulkanDevice *>(creator.device());
  Device::AllocParams allocation{};
  allocation.size = 66 * sizeof(float);
  allocation.export_sharing = true;
  allocation.usage = AllocUsage::Storage;
  auto [shared, status] = vk->allocate_memory_unique(allocation);
  ASSERT_EQ(status, RhiResult::success);
  allocation.export_sharing = false;
  auto [observed, output_status] = vk->allocate_memory_unique(allocation);
  ASSERT_EQ(output_status, RhiResult::success);
  auto adapter = VulkanCudaExternalAllocation::create(vk, &device, *shared);
  auto owner = program.register_external_dense_storage(
      adapter->cuda_allocation(), allocation.size,
      [adapter] { adapter->close(); }, adapter);
  storage::DenseStorageLayoutSpec layout;
  layout.scalar_type = PrimitiveType::f32;
  layout.index_shape = {66};
  layout.index_strides_bytes = {sizeof(float)};
  auto description = storage::build_dense_storage_descriptor(
      owner, storage::StorageSourceKind::kExternalDense, layout);
  ASSERT_TRUE(description);
  auto borrower = AotRuntimeClient::create(program);
  auto *runtime = reinterpret_cast<::Runtime *>(borrower->handle());
  auto rt = reinterpret_cast<TiRuntime>(runtime);
  auto module = ti_load_aot_module(rt, folder);
  ASSERT_NE(module, nullptr);
  auto memory = reinterpret_cast<TiMemory>(
      std::get<0>(borrower->import_dense_view(*description.descriptor)));
  EXPECT_THROW(runtime->get_memory(memory), MemoryBindingError);
  EXPECT_EQ(runtime->get_memory(memory, true).borrowed->external_access,
            adapter);
  std::array<TiNamedArgument, 3> arguments{};
  arguments[0].name = "input";
  arguments[0].argument.type = TI_ARGUMENT_TYPE_NDARRAY;
  arguments[0].argument.value.ndarray = {
      memory, {1, {66}}, {0, {}}, TI_DATA_TYPE_F32};
  arguments[1].name = "seed";
  arguments[1].argument.type = TI_ARGUMENT_TYPE_F32;
  arguments[1].argument.value.f32 = 7.0f;
  arguments[2].name = "n";
  arguments[2].argument.type = TI_ARGUMENT_TYPE_I32;
  arguments[2].argument.value.i32 = 64;
  const std::string plan = R"({"version":3,"resources":[
    {"kind":"ndarray","name":"temporary","dtype":"f32","shape":[64]},
    {"kind":"ndarray_view","name":"view","array":"input","offset":1,"shape":[64]}],
    "initialization":[{"kind":"fill32","destination":"temporary","value":0}],
    "operations":[
      {"kind":"kernel","entry":"stage","arguments":["temporary","seed","n"]},
      {"kind":"copy","source":"temporary","destination":"view"},
      {"kind":"kernel","entry":"stage","arguments":["view","seed","n"]}]})";
  // Ordinary paths may not strip the imported epoch, including aliases.
  EXPECT_EQ(ti_prepare_execution_plan_ext(rt, module, plan.size(), plan.c_str(),
                                          arguments.size(), arguments.data(), 0,
                                          nullptr),
            nullptr);
  EXPECT_EQ(ti_get_last_error(nullptr, nullptr), TI_ERROR_INVALID_ARGUMENT);
  ti_set_last_error(TI_ERROR_SUCCESS, nullptr);
  std::array<TiArgument, 3> kernel_arguments{
      arguments[0].argument, arguments[1].argument, arguments[2].argument};
  ti_launch_kernel(rt, ti_get_aot_module_kernel(module, "stage"),
                   kernel_arguments.size(), kernel_arguments.data());
  EXPECT_EQ(ti_get_last_error(nullptr, nullptr), TI_ERROR_INVALID_ARGUMENT);
  ti_set_last_error(TI_ERROR_SUCCESS, nullptr);
  const TiCudaExecutionDomainCreateInfo options{1, 4, 0};
  auto selected = ti_create_cuda_execution_domain_ext(rt, &options);
  ASSERT_NE(selected, nullptr);
  auto prepared = ti_prepare_execution_plan_on_domain_ext(
      rt, module, selected, plan.size(), plan.c_str(), arguments.size(),
      arguments.data(), 0, nullptr);
  ASSERT_NE(prepared, nullptr);
  auto *vk_stream =
      static_cast<vulkan::VulkanStream *>(vk->get_compute_stream());
  auto [producer, producer_status] = vk_stream->new_command_list_unique();
  ASSERT_EQ(producer_status, RhiResult::success);
  producer->buffer_fill(shared->get_ptr(), allocation.size, 0);
  ASSERT_TRUE(adapter->release_vulkan_to_cuda(*vk_stream, producer.get()));
  ti_initialize_prepared_ext(rt, prepared);
  ASSERT_EQ(ti_get_last_error(nullptr, nullptr), TI_ERROR_SUCCESS);
  // Initializers touch only private storage and must not consume frame-ready.
  EXPECT_EQ(adapter->access_state(),
            VulkanCudaExternalAllocation::AccessState::kAwaitingCudaAcquire);
  auto domain = capi::get_execution_domain(selected);
  testing::CudaExternalGate capacity_gate;
  auto occupied = domain->submit(
      {}, nullptr, [&](void *stream) { capacity_gate.enqueue_wait(stream); });
  TiCompletion ticket = nullptr;
  EXPECT_FALSE(ti_try_submit_prepared_ext(rt, prepared, 0, nullptr, &ticket));
  EXPECT_EQ(ti_get_last_error(nullptr, nullptr), TI_ERROR_SUCCESS);
  EXPECT_EQ(adapter->access_state(),
            VulkanCudaExternalAllocation::AccessState::kAwaitingCudaAcquire);
  capacity_gate.release();
  occupied.wait();
  auto [consumer, consumer_status] = vk_stream->new_command_list_unique();
  ASSERT_EQ(consumer_status, RhiResult::success);
  consumer->buffer_copy(observed->get_ptr(), shared->get_ptr(),
                        allocation.size);
  auto initialized_plan = plan;
  const auto initializer =
      initialized_plan.find("\"destination\":\"temporary\"");
  ASSERT_NE(initializer, std::string::npos);
  initialized_plan.replace(initializer,
                           std::strlen("\"destination\":\"temporary\""),
                           "\"destination\":\"input\"");
  auto external_initializer = ti_prepare_execution_plan_on_domain_ext(
      rt, module, selected, initialized_plan.size(), initialized_plan.c_str(),
      arguments.size(), arguments.data(), 0, nullptr);
  ASSERT_NE(external_initializer, nullptr);
  ti_initialize_prepared_ext(rt, external_initializer);
  ASSERT_EQ(ti_get_last_error(nullptr, nullptr), TI_ERROR_SUCCESS);
  EXPECT_EQ(adapter->access_state(),
            VulkanCudaExternalAllocation::AccessState::kAwaitingVulkanAcquire);
  auto initialized = adapter->cycle_vulkan_to_cuda(*vk_stream, consumer.get());
  ASSERT_TRUE(initialized);
  ASSERT_TRUE(initialized->wait());
  ti_destroy_prepared_ext(rt, external_initializer);
  program.synchronize();
  testing::CudaExternalGate default_gate;
  auto &driver = CUDADriver::get_instance();
  void *default_pending = nullptr;
  driver.event_create(&default_pending, CU_EVENT_DISABLE_TIMING);
  default_gate.enqueue_wait(nullptr);
  driver.event_record(default_pending, nullptr);
  ASSERT_EQ(driver.event_query.call(default_pending), CUDA_ERROR_NOT_READY);
  const auto binding_stats = program.debug_dense_storage_binding_stats();
  const auto *requested_replays = std::getenv("TAICHI_F3_EXTERNAL_REPLAYS");
  const int replay_count = requested_replays ? std::atoi(requested_replays) : 5;
  ASSERT_GE(replay_count, 5);
  ASSERT_LE(replay_count, 1000);
  std::vector<double> submit_times;
  submit_times.reserve(replay_count);
  for (int i = 0; i < replay_count; ++i) {
    {
      ScopedExternalProfilerAnnotation submit("F3/public_external/submit");
      const auto begin = std::chrono::steady_clock::now();
      ASSERT_TRUE(
          ti_try_submit_prepared_ext(rt, prepared, 0, nullptr, &ticket));
      submit_times.push_back(std::chrono::duration<double, std::micro>(
                                 std::chrono::steady_clock::now() - begin)
                                 .count());
    }
    ASSERT_EQ(ti_get_last_error(nullptr, nullptr), TI_ERROR_SUCCESS);
    auto consumed = adapter->cycle_vulkan_to_cuda(*vk_stream, consumer.get());
    ASSERT_TRUE(consumed);
    ASSERT_TRUE(consumed->wait());
    ti_wait_completion_ext(ticket);
    ASSERT_EQ(ti_get_last_error(nullptr, nullptr), TI_ERROR_SUCCESS);
    ti_release_completion_ext(ticket);
    EXPECT_EQ(driver.event_query.call(default_pending), CUDA_ERROR_NOT_READY)
        << "Vulkan producer still pending=" << default_gate.producer_pending();
    EXPECT_FALSE(default_gate.timed_out());
  }
  std::sort(submit_times.begin(), submit_times.end());
  std::cout << "public external submit_us p50="
            << submit_times[replay_count / 2]
            << " p95=" << submit_times[replay_count * 95 / 100]
            << " accepted=" << replay_count << std::endl;
  EXPECT_EQ(program.debug_dense_storage_binding_stats(), binding_stats);
  std::array<float, 66> values{};
  auto pointer = observed->get_ptr();
  void *destination = values.data();
  auto bytes = sizeof(values);
  ASSERT_EQ(vk->readback_data(&pointer, &destination, &bytes),
            RhiResult::success);
  EXPECT_EQ(values.front(), 0.0f);
  EXPECT_EQ(values.back(), 0.0f);
  for (int i = 0; i < 64; ++i)
    EXPECT_EQ(values[i + 1], 36.0f * 64 * 65 / 2 + i);
  EXPECT_FALSE(default_gate.timed_out());
  // Draining the participating work must not inherit the default tail.
  // Destructive close is a cold boundary: the driver's actual memory frees
  // may synchronize more broadly. Never use it as a frame/cancel operation.
  domain->stop_accepting();
  domain->latest_completion().wait();
  EXPECT_EQ(driver.event_query.call(default_pending), CUDA_ERROR_NOT_READY);
  EXPECT_FALSE(default_gate.timed_out());
  {
    ScopedExternalProfilerAnnotation release(
        "F3/public_external/default_release");
    default_gate.release();
  }
  driver.event_synchronize(default_pending);
  driver.event_destroy(default_pending);
  program.retire_external_dense_storage(owner);
  EXPECT_TRUE(domain->snapshot().closed);
  EXPECT_TRUE(adapter->closed());
  EXPECT_THROW(runtime->get_memory(memory, true), MemoryBindingError);
  ti_destroy_prepared_ext(rt, prepared);
  ti_free_memory(rt, memory);
  ti_destroy_aot_module(module);
  ti_release_cuda_execution_domain_ext(selected);
  borrower->close();
  shared.reset();
  observed.reset();
  program.finalize();
}

TEST(CudaPublicInstanceRetirement, PrivateStorageAndOwnerIndependentLease) {
  const auto *folder = std::getenv("TAICHI_F3_CONTEXT_AOT_FOLDER");
  if (!folder || !cuda::ExecutionDomain::available())
    GTEST_SKIP() << "Requires the saved execution-context AOT fixture";
  if (!CUDAContext::get_instance().supports_mem_pool())
    GTEST_SKIP() << "Requires stream-ordered allocation release";
  auto runtime = std::make_unique<capi::LlvmRuntime>(Arch::cuda);
  auto rt = reinterpret_cast<TiRuntime>(runtime.get());
  auto module = ti_load_aot_module(rt, folder);
  ASSERT_NE(module, nullptr);
  std::ifstream file(std::string(folder) + "/executions/public_stage.json");
  ASSERT_TRUE(file.good());
  std::ostringstream contents;
  contents << file.rdbuf();
  const auto plan = contents.str();
  TiMemoryAllocateInfo allocation{};
  allocation.size = 64 * sizeof(float);
  allocation.usage = TI_MEMORY_USAGE_STORAGE_BIT;
  auto memory = ti_allocate_memory(rt, &allocation);
  ASSERT_NE(memory, nullptr);
  TiNamedArgument arguments[2]{};
  arguments[0].name = "out";
  arguments[0].argument.type = TI_ARGUMENT_TYPE_NDARRAY;
  arguments[0].argument.value.ndarray = {
      memory, {1, {64}}, {0, {}}, TI_DATA_TYPE_F32};
  arguments[1].name = "seed";
  arguments[1].argument.type = TI_ARGUMENT_TYPE_F32;
  arguments[1].argument.value.f32 = 2.0f;
  const TiCudaExecutionDomainCreateInfo options{2, 1, 0};
  auto domain = ti_create_cuda_execution_domain_ext(rt, &options);
  auto peer = ti_create_cuda_execution_domain_ext(rt, &options);
  const auto prepare = [&] {
    auto result = ti_prepare_execution_plan_on_domain_ext(
        rt, module, domain, plan.size(), plan.c_str(), 2, arguments, 0, nullptr);
    if (!result)
      throw std::runtime_error("Unable to prepare retirement fixture");
    ti_initialize_prepared_ext(rt, result);
    return result;
  };
  auto call = prepare();
  TiCompletion ticket = nullptr;
  ASSERT_TRUE(ti_try_submit_prepared_ext(rt, call, 0, nullptr, &ticket));
  ti_wait_completion_ext(ticket);
  ASSERT_EQ(ti_get_last_error(nullptr, nullptr), TI_ERROR_SUCCESS);
  auto lease = ti_get_prepared_cuda_instance_ext(rt, call);
  ASSERT_NE(lease, nullptr);
  testing::CudaExternalGate gate;
  auto pending = capi::get_execution_domain(peer)->submit(
      {}, nullptr, [&](void *stream) { gate.enqueue_wait(stream); });
  auto retirement = std::async(std::launch::async, [&] {
    ti_retire_cuda_execution_instance_ext(lease);
    auto error = ti_get_last_error(nullptr, nullptr);
    if (error == TI_ERROR_SUCCESS) {
      ti_finish_prepared_cuda_instance_retirement_ext(rt, call);
      error = ti_get_last_error(nullptr, nullptr);
    }
    return error;
  });
  const bool independent =
      retirement.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
  EXPECT_TRUE(independent);
  if (independent)
    EXPECT_FALSE(pending.done());
  EXPECT_FALSE(gate.timed_out());
  gate.release();
  EXPECT_EQ(retirement.get(), TI_ERROR_SUCCESS);
  pending.wait();
  EXPECT_TRUE(ti_query_completion_ext(ticket));
  ti_release_completion_ext(ticket);
  ti_release_cuda_execution_instance_ext(lease);
  for (int i = 0; i < 3; ++i) {
    call = prepare();
    lease = ti_get_prepared_cuda_instance_ext(rt, call);
    ASSERT_NE(lease, nullptr);
    ti_retire_cuda_execution_instance_ext(lease);
    ti_finish_prepared_cuda_instance_retirement_ext(rt, call);
    ti_release_cuda_execution_instance_ext(lease);
    ASSERT_EQ(ti_get_last_error(nullptr, nullptr), TI_ERROR_SUCCESS);
  }
  call = prepare();
  lease = ti_get_prepared_cuda_instance_ext(rt, call);
  ASSERT_NE(lease, nullptr);
  ti_destroy_runtime(reinterpret_cast<TiRuntime>(runtime.release()));
  ASSERT_EQ(ti_get_last_error(nullptr, nullptr), TI_ERROR_SUCCESS);
  // An outstanding host retirement lease must never touch the dead Runtime.
  ti_retire_cuda_execution_instance_ext(lease);
  ti_release_cuda_execution_instance_ext(lease);
  ti_release_cuda_execution_domain_ext(domain);
  ti_release_cuda_execution_domain_ext(peer);
  EXPECT_EQ(ti_get_last_error(nullptr, nullptr), TI_ERROR_SUCCESS);
}

TEST(CudaPublicExecutionAot, ExactDependenciesBusyAndReleasedInstanceLifetime) {
  const auto *folder = std::getenv("TAICHI_F3_CONTEXT_AOT_FOLDER");
  if (!folder || !cuda::ExecutionDomain::available())
    GTEST_SKIP() << "Requires the saved execution-context AOT fixture";
  auto runtime = std::make_unique<capi::LlvmRuntime>(Arch::cuda);
  auto foreign_runtime = std::make_unique<capi::LlvmRuntime>(Arch::cuda);
  auto rt = reinterpret_cast<TiRuntime>(runtime.get());
  const TiCudaExecutionDomainCreateInfo options{2, 4, 0};
  auto output = ti_create_cuda_execution_domain_ext(rt, &options);
  auto physics = ti_create_cuda_execution_domain_ext(rt, &options);
  auto foreign = ti_create_cuda_execution_domain_ext(
      reinterpret_cast<TiRuntime>(foreign_runtime.get()), &options);
  ASSERT_NE(output, nullptr);
  ASSERT_NE(physics, nullptr);
  ASSERT_NE(foreign, nullptr);
  auto foreign_ticket = ti_get_cuda_execution_domain_completion_ext(foreign);
  auto module = ti_load_aot_module(rt, folder);
  ASSERT_NE(module, nullptr);
  constexpr int count = 64;
  std::array<TiMemory, 2> memory{};
  std::array<void *, 2> pointers{};
  std::array<TiPreparedExecution, 2> calls{};
  const std::string plan =
      R"({"version":1,"operations":[{"kind":"kernel","entry":"stage","arguments":["out","seed","n"]}]})";
  for (int i = 0; i < 2; ++i) {
    TiMemoryAllocateInfo allocation_info{};
    allocation_info.size = count * sizeof(float);
    allocation_info.usage = TI_MEMORY_USAGE_STORAGE_BIT;
    memory[i] = ti_allocate_memory(rt, &allocation_info);
    auto &device = static_cast<cuda::CudaDevice &>(runtime->get());
    pointers[i] =
        device.get_memory_addr(runtime->get_memory(memory[i]).allocation);
    TiNamedArgument arguments[3]{};
    arguments[0].name = "out";
    arguments[0].argument.type = TI_ARGUMENT_TYPE_NDARRAY;
    arguments[0].argument.value.ndarray = {
        memory[i], {1, {count}}, {0, {}}, TI_DATA_TYPE_F32};
    arguments[1].name = "seed";
    arguments[1].argument.type = TI_ARGUMENT_TYPE_F32;
    arguments[1].argument.value.f32 = i == 0 ? 2.0f : 7.0f;
    arguments[2].name = "n";
    arguments[2].argument.type = TI_ARGUMENT_TYPE_I32;
    arguments[2].argument.value.i32 = count;
    calls[i] = ti_prepare_execution_plan_on_domain_ext(
        rt, module, i == 0 ? output : physics, plan.size(), plan.c_str(), 3,
        arguments, 0, nullptr);
    ASSERT_NE(calls[i], nullptr);
    // Mutating the source descriptor cannot change the immutable instance.
    arguments[1].argument.value.f32 = -999.0f;
  }
  TiCompletion ticket = nullptr;
  ti_set_last_error(TI_ERROR_SUCCESS, nullptr);
  EXPECT_FALSE(ti_try_submit_prepared_ext(rt, calls[0], 0, nullptr, &ticket));
  EXPECT_EQ(ticket, nullptr);
  EXPECT_EQ(ti_get_last_error(nullptr, nullptr), TI_ERROR_INVALID_ARGUMENT);
  ti_set_last_error(TI_ERROR_SUCCESS, nullptr);
  ti_initialize_prepared_ext(rt, calls[0]);
  ASSERT_EQ(ti_get_last_error(nullptr, nullptr), TI_ERROR_SUCCESS);

  testing::CudaExternalGate gate;
  capi::get_execution_domain(output)->submit(
      {}, nullptr, [&](void *stream) { gate.enqueue_wait(stream); });
  auto blocked = ti_get_cuda_execution_domain_completion_ext(output);
  // Cold initialization waits only its published prefix, never the blocked
  // output domain. This also proves the no-explicit-initializer contract.
  {
    ScopedExternalProfilerAnnotation annotation("F3/public/initialize");
    ti_initialize_prepared_ext(rt, calls[1]);
  }
  ASSERT_EQ(ti_get_last_error(nullptr, nullptr), TI_ERROR_SUCCESS);
  EXPECT_FALSE(gate.timed_out());
  EXPECT_FALSE(ti_query_completion_ext(blocked));
  EXPECT_TRUE(ti_try_submit_prepared_ext(rt, calls[0], 1, &blocked, &ticket));
  ASSERT_NE(ticket, nullptr);
  TiCompletion busy = nullptr;
  EXPECT_FALSE(ti_try_submit_prepared_ext(rt, calls[0], 0, nullptr, &busy));
  EXPECT_EQ(busy, nullptr);
  EXPECT_EQ(ti_get_last_error(nullptr, nullptr), TI_ERROR_SUCCESS);
  const auto sequence =
      capi::get_execution_domain(physics)->latest_completion().sequence();
  TiCompletion rejected = nullptr;
  EXPECT_FALSE(
      ti_try_submit_prepared_ready_ext(rt, calls[1], 1, &blocked, &rejected));
  EXPECT_EQ(rejected, nullptr);
  EXPECT_EQ(ti_get_last_error(nullptr, nullptr), TI_ERROR_SUCCESS);
  EXPECT_EQ(capi::get_execution_domain(physics)->latest_completion().sequence(),
            sequence);
  const TiCompletion mixed_dependencies[] = {blocked, foreign_ticket};
  EXPECT_FALSE(ti_try_submit_prepared_ready_ext(rt, calls[1], 2,
                                                mixed_dependencies, &rejected));
  EXPECT_EQ(rejected, nullptr);
  EXPECT_NE(ti_get_last_error(nullptr, nullptr), TI_ERROR_SUCCESS);
  ti_set_last_error(TI_ERROR_SUCCESS, nullptr);
  EXPECT_FALSE(
      ti_try_submit_prepared_ext(rt, calls[1], 1, &foreign_ticket, &rejected));
  EXPECT_EQ(rejected, nullptr);
  EXPECT_NE(ti_get_last_error(nullptr, nullptr), TI_ERROR_SUCCESS);
  EXPECT_EQ(capi::get_execution_domain(physics)->latest_completion().sequence(),
            sequence);
  ti_set_last_error(TI_ERROR_SUCCESS, nullptr);
  TiCompletion active = nullptr;
  {
    ScopedExternalProfilerAnnotation annotation("F3/public/submit");
    ASSERT_TRUE(ti_try_submit_prepared_ext(rt, calls[1], 0, nullptr, &active));
  }
  ti_wait_completion_ext(active);
  ASSERT_EQ(ti_get_last_error(nullptr, nullptr), TI_ERROR_SUCCESS);
  TiCompletion ready_dependent = nullptr;
  ASSERT_TRUE(ti_try_submit_prepared_ready_ext(rt, calls[1], 1, &active,
                                               &ready_dependent));
  ti_wait_completion_ext(ready_dependent);
  ti_release_completion_ext(ready_dependent);
  auto context = CUDAContext::get_instance().get_guard();
  std::array<float, count> values{};
  CUDADriver::get_instance().memcpy_device_to_host(values.data(), pointers[1],
                                                   sizeof(values));
  for (int i = 0; i < count; ++i)
    EXPECT_EQ(values[i], 36.0f * count * (count + 1) / 2 + i);
  std::vector<double> times;
  times.reserve(1000);
  std::size_t busy_count = 0;
  {
    ScopedExternalProfilerAnnotation annotation("F3/public/warm");
    while (times.size() < 1000) {
      TiCompletion completed = nullptr;
      const auto begin = std::chrono::steady_clock::now();
      const auto accepted =
          ti_try_submit_prepared_ext(rt, calls[1], 0, nullptr, &completed);
      const auto end = std::chrono::steady_clock::now();
      ASSERT_EQ(ti_get_last_error(nullptr, nullptr), TI_ERROR_SUCCESS);
      if (!accepted) {
        ++busy_count;
        std::this_thread::yield();
        continue;
      }
      times.push_back(
          std::chrono::duration<double, std::micro>(end - begin).count());
      ti_release_completion_ext(completed);
    }
  }
  auto warm_end = ti_get_cuda_execution_domain_completion_ext(physics);
  ti_wait_completion_ext(warm_end);
  ti_release_completion_ext(warm_end);
  std::sort(times.begin(), times.end());
  std::cout << "public AOT submit_us p50=" << times[500]
            << " p95=" << times[950] << " accepted=" << times.size()
            << " busy=" << busy_count << std::endl;
  EXPECT_FALSE(gate.timed_out());
  EXPECT_FALSE(ti_query_completion_ext(ticket))
      << "Vulkan producer still pending=" << gate.producer_pending();
  TiCompletion dependent = nullptr;
  ASSERT_TRUE(ti_try_submit_prepared_ext(rt, calls[1], 1, &ticket, &dependent));
  EXPECT_FALSE(ti_query_completion_ext(dependent));

  // Public wrapper retirement must not wait, erase code/storage edges or
  // release the still-pending independent execution instance.
  {
    ScopedExternalProfilerAnnotation annotation("F3/public/release_instance");
    ti_destroy_prepared_ext(rt, calls[0]);
  }
  ASSERT_EQ(ti_get_last_error(nullptr, nullptr), TI_ERROR_SUCCESS);
  EXPECT_FALSE(gate.timed_out());
  EXPECT_FALSE(ti_query_completion_ext(ticket));
  ti_destroy_prepared_ext(rt, calls[1]);
  auto close_module = std::async(std::launch::async, [&] {
    ti_set_last_error(TI_ERROR_SUCCESS, nullptr);
    ti_destroy_aot_module(module);
    return ti_get_last_error(nullptr, nullptr);
  });
  EXPECT_EQ(close_module.wait_for(std::chrono::milliseconds(20)),
            std::future_status::timeout);
  EXPECT_FALSE(gate.timed_out());
  {
    ScopedExternalProfilerAnnotation annotation("F3/public/release_gate");
    gate.release();
  }
  EXPECT_EQ(close_module.get(), TI_ERROR_SUCCESS);
  EXPECT_TRUE(ti_query_completion_ext(ticket));
  EXPECT_TRUE(ti_query_completion_ext(dependent));
  CUDADriver::get_instance().memcpy_device_to_host(values.data(), pointers[0],
                                                   sizeof(values));
  for (int i = 0; i < count; ++i)
    EXPECT_EQ(values[i], 11.0f * count * (count + 1) / 2 + i);
  for (auto completion : {blocked, ticket, active, dependent, foreign_ticket})
    ti_release_completion_ext(completion);
  for (auto domain : {output, physics, foreign}) {
    ti_close_cuda_execution_domain_ext(domain);
    ti_release_cuda_execution_domain_ext(domain);
  }
  for (auto allocation : memory)
    ti_free_memory(rt, allocation);
  EXPECT_EQ(ti_get_last_error(nullptr, nullptr), TI_ERROR_SUCCESS);
}

TEST(CudaExecutionOwner, SubmissionParametersStayOnTheirSelectedDomains) {
  const auto *folder = std::getenv("TAICHI_A12_AOT_FOLDER");
  if (!folder || !cuda::ExecutionDomain::available())
    GTEST_SKIP() << "Requires the submission-parameter AOT fixture";
  auto runtime = std::make_unique<capi::LlvmRuntime>(Arch::cuda);
  auto rt = reinterpret_cast<TiRuntime>(runtime.get());
  auto module = ti_load_aot_module(rt, folder);
  ASSERT_NE(module, nullptr);
  const TiCudaExecutionDomainCreateInfo options{3, 4, 0};
  std::array<TiCudaExecutionDomain, 2> domains{};
  std::array<TiMemory, 2> memory{};
  std::array<TiPreparedExecution, 2> calls{};
  const char *names[] = {"epoch", "simulation_step", "timestamp_ns"};
  const std::string plan =
      R"({"version":1,"operations":[{"kind":"kernel","entry":"publish","arguments":["out","epoch","simulation_step","timestamp_ns"]}]})";
  for (int lane = 0; lane < 2; ++lane) {
    domains[lane] = ti_create_cuda_execution_domain_ext(rt, &options);
    ASSERT_NE(domains[lane], nullptr);
    TiMemoryAllocateInfo allocation{};
    allocation.size = 1000 * 3 * sizeof(uint64_t);
    allocation.usage = TI_MEMORY_USAGE_STORAGE_BIT;
    memory[lane] = ti_allocate_memory(rt, &allocation);
    TiNamedArgument arguments[4]{};
    arguments[0].name = "out";
    arguments[0].argument.type = TI_ARGUMENT_TYPE_NDARRAY;
    arguments[0].argument.value.ndarray = {
        memory[lane], {2, {1000, 3}}, {0, {}}, TI_DATA_TYPE_U64};
    for (int i = 0; i < 3; ++i) {
      arguments[i + 1].name = names[i];
      arguments[i + 1].argument.type = TI_ARGUMENT_TYPE_SCALAR;
      arguments[i + 1].argument.value.scalar.type = TI_DATA_TYPE_U64;
      arguments[i + 1].argument.value.scalar.value.x64 = 0;
    }
    calls[lane] = ti_prepare_execution_plan_parameters_on_domain_ext(
        rt, module, domains[lane], plan.size(), plan.c_str(), 4, arguments, 0,
        nullptr, 3, names);
    ASSERT_NE(calls[lane], nullptr);
    ti_initialize_prepared_ext(rt, calls[lane]);
    ASSERT_EQ(ti_get_last_error(nullptr, nullptr), TI_ERROR_SUCCESS);
  }
  const std::array<uint64_t, 3> expected{
      uint64_t(1) << 63, (uint64_t(1) << 62) + 17, ~uint64_t(0)};
  for (int lane = 0; lane < 2; ++lane) {
    testing::CudaExternalGate gate;
    auto domain = capi::get_execution_domain(domains[lane]);
    auto prefix = domain->submit(
        {}, nullptr, [&](void *stream) { gate.enqueue_wait(stream); });
    auto values = expected;
    TiCompletion accepted = nullptr, rejected = nullptr, peer = nullptr;
    ASSERT_TRUE(ti_try_submit_prepared_parameters_ext(
        rt, calls[lane], 0, nullptr, &accepted, TI_FALSE, 3, values.data()));
    values.fill(41);  // Accepted work must no longer read this host array.
    const auto sequence = domain->latest_completion().sequence();
    EXPECT_FALSE(ti_try_submit_prepared_parameters_ext(
        rt, calls[lane], 0, nullptr, &rejected, TI_FALSE, 3, values.data()));
    EXPECT_EQ(rejected, nullptr);
    EXPECT_EQ(ti_get_last_error(nullptr, nullptr), TI_ERROR_SUCCESS);
    EXPECT_EQ(domain->latest_completion().sequence(), sequence);
    ASSERT_TRUE(ti_try_submit_prepared_parameters_ext(
        rt, calls[1 - lane], 0, nullptr, &peer, TI_FALSE, 3, expected.data()));
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!ti_query_completion_ext(peer) &&
           std::chrono::steady_clock::now() < deadline)
      std::this_thread::yield();
    EXPECT_TRUE(ti_query_completion_ext(peer));
    EXPECT_FALSE(ti_query_completion_ext(accepted));
    EXPECT_FALSE(gate.timed_out());
    EXPECT_TRUE(gate.producer_pending());
    EXPECT_FALSE(ti_try_submit_prepared_parameters_ext(
        rt, calls[1 - lane], 1, &accepted, &rejected, TI_TRUE, 3,
        values.data()));
    EXPECT_EQ(rejected, nullptr);
    EXPECT_EQ(ti_get_last_error(nullptr, nullptr), TI_ERROR_SUCCESS);
    gate.release();
    ti_wait_completion_ext(accepted);
    for (int output = 0; output < 2; ++output) {
      auto &device = static_cast<cuda::CudaDevice &>(runtime->get());
      auto pointer = device.get_memory_addr(
          runtime->get_memory(memory[output]).allocation);
      std::array<uint64_t, 3> actual{};
      auto current = CUDAContext::get_instance().get_guard();
      CUDADriver::get_instance().memcpy_device_to_host(actual.data(), pointer,
                                                       sizeof(actual));
      EXPECT_EQ(actual, expected);
    }
    ti_release_completion_ext(accepted);
    ti_release_completion_ext(peer);
  }
  for (int lane = 0; lane < 2; ++lane) {
    // Parameter staging is pinned host memory and has its own release path.
    // Retiring it must not turn an unrelated pending prefix into a host wait.
    testing::CudaExternalGate gate;
    auto pending = capi::get_execution_domain(domains[1 - lane])->submit(
        {}, nullptr, [&](void *stream) { gate.enqueue_wait(stream); });
    auto lease = ti_get_prepared_cuda_instance_ext(rt, calls[lane]);
    ASSERT_NE(lease, nullptr);
    auto retirement = std::async(std::launch::async, [&] {
      ti_retire_cuda_execution_instance_ext(lease);
      auto error = ti_get_last_error(nullptr, nullptr);
      if (error == TI_ERROR_SUCCESS) {
        ti_finish_prepared_cuda_instance_retirement_ext(rt, calls[lane]);
        error = ti_get_last_error(nullptr, nullptr);
      }
      return error;
    });
    const bool independent =
        retirement.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
    EXPECT_TRUE(independent);
    if (independent)
      EXPECT_FALSE(pending.done());
    EXPECT_FALSE(gate.timed_out());
    gate.release();
    EXPECT_EQ(retirement.get(), TI_ERROR_SUCCESS);
    pending.wait();
    ti_release_cuda_execution_instance_ext(lease);
  }
  for (int lane = 0; lane < 2; ++lane) {
    ti_close_cuda_execution_domain_ext(domains[lane]);
    ti_release_cuda_execution_domain_ext(domains[lane]);
    ti_free_memory(rt, memory[lane]);
  }
  ti_destroy_aot_module(module);
  EXPECT_EQ(ti_get_last_error(nullptr, nullptr), TI_ERROR_SUCCESS);
}

TEST(CudaExecutionOwner, PublicTicketWaitDoesNotLockQueryOrPeerSubmission) {
  if (!cuda::ExecutionDomain::available())
    GTEST_SKIP();
  auto runtime = std::make_unique<capi::LlvmRuntime>(Arch::cuda);
  const auto handle = reinterpret_cast<TiRuntime>(runtime.get());
  auto context = CUDAContext::get_instance().get_guard();
  const TiCudaExecutionDomainCreateInfo options{2, 4, -1};
  auto output = ti_create_cuda_execution_domain_ext(handle, &options);
  auto physics = ti_create_cuda_execution_domain_ext(handle, &options);
  ASSERT_NE(output, nullptr);
  ASSERT_NE(physics, nullptr);
  TiCudaExecutionDomainInfo left{}, right{};
  ti_get_cuda_execution_domain_info_ext(output, &left);
  ti_get_cuda_execution_domain_info_ext(physics, &right);
  EXPECT_NE(left.identity, right.identity);
  EXPECT_EQ(left.owner, right.owner);
  EXPECT_EQ(left.context, right.context);
  EXPECT_NE(left.stream, 0);
  EXPECT_NE(left.stream, right.stream);
  EXPECT_EQ(left.capacity, 2);
  auto empty = ti_get_cuda_execution_domain_completion_ext(output);
  ASSERT_NE(empty, nullptr);
  TiCompletionInfo empty_info{};
  ti_get_completion_info_ext(empty, &empty_info);
  EXPECT_EQ(empty_info.sequence, 0);

  auto &device = static_cast<cuda::CudaDevice &>(runtime->get());
  auto allocation = device.allocate_memory_unique({sizeof(std::uint32_t)});
  ASSERT_EQ(allocation.second, RhiResult::success);
  auto *address = device.get_memory_addr(*allocation.first);
  testing::CudaExternalGate gate;
  auto native_output = capi::get_execution_domain(output);
  auto native_physics = capi::get_execution_domain(physics);
  native_output->submit({}, nullptr,
                        [&](void *stream) { gate.enqueue_wait(stream); });
  auto blocked = ti_get_cuda_execution_domain_completion_ext(output);
  ASSERT_NE(blocked, nullptr);
  EXPECT_FALSE(ti_query_completion_ext(blocked));
  EXPECT_TRUE(ti_query_completion_ext(empty));
  auto waiter = std::async(std::launch::async, [&] {
    ti_set_last_error(TI_ERROR_SUCCESS, nullptr);
    ti_wait_completion_ext(blocked);
    return ti_get_last_error(nullptr, nullptr);
  });
  EXPECT_EQ(waiter.wait_for(std::chrono::milliseconds(20)),
            std::future_status::timeout);
  auto query = std::async(std::launch::async, [&] {
    ti_set_last_error(TI_ERROR_SUCCESS, nullptr);
    return std::make_pair(ti_query_completion_ext(blocked),
                          ti_get_last_error(nullptr, nullptr));
  });
  const auto query_status = query.wait_for(std::chrono::milliseconds(200));
  if (query_status != std::future_status::ready)
    gate.release();
  EXPECT_EQ(query_status, std::future_status::ready);
  const auto queried = query.get();
  EXPECT_FALSE(queried.first);
  EXPECT_EQ(queried.second, TI_ERROR_SUCCESS);
  auto sentinel = native_physics->submit({}, nullptr, [&](void *stream) {
    CUDADriver::get_instance().memsetd32_async(address, 0x3f123456, 1, stream);
  });
  sentinel.wait();
  std::uint32_t observed = 0;
  CUDADriver::get_instance().memcpy_device_to_host(&observed, address,
                                                   sizeof(observed));
  EXPECT_EQ(observed, 0x3f123456u);
  EXPECT_FALSE(ti_query_completion_ext(blocked));
  EXPECT_FALSE(gate.timed_out());
  gate.release();
  EXPECT_EQ(waiter.get(), TI_ERROR_SUCCESS);
  EXPECT_TRUE(ti_query_completion_ext(blocked));
  ti_release_completion_ext(blocked);
  ti_release_completion_ext(empty);
  ti_release_cuda_execution_domain_ext(output);
  // Releasing the public handle must not close its runtime-owned domain.
  EXPECT_TRUE(native_output->snapshot().accepting);
  ti_close_cuda_execution_domain_ext(physics);
  ti_release_cuda_execution_domain_ext(physics);
  ti_set_last_error(TI_ERROR_SUCCESS, nullptr);
  EXPECT_FALSE(ti_query_completion_ext(blocked));
  EXPECT_EQ(ti_get_last_error(nullptr, nullptr), TI_ERROR_INVALID_ARGUMENT);
  ti_set_last_error(TI_ERROR_SUCCESS, nullptr);
}

TEST(CudaCompletionSet, ExactPrefixesTimeoutCancellationAndOwnerRetirement) {
  if (!cuda::ExecutionDomain::available())
    GTEST_SKIP();
  auto runtime = std::make_unique<capi::LlvmRuntime>(Arch::cuda);
  auto rt = reinterpret_cast<TiRuntime>(runtime.get());
  const TiCudaExecutionDomainCreateInfo options{2, 2, 0};
  auto left = ti_create_cuda_execution_domain_ext(rt, &options);
  auto right = ti_create_cuda_execution_domain_ext(rt, &options);
  testing::CudaExternalGate gate;
  auto pending = capi::get_execution_domain(left)->submit(
      {}, nullptr, [&](void *stream) { gate.enqueue_wait(stream); });
  capi::get_execution_domain(right)->submit({}, nullptr, [](void *) {});
  auto first = ti_get_cuda_execution_domain_completion_ext(left);
  auto second = ti_get_cuda_execution_domain_completion_ext(right);
  TiCompletionSource sources[] = {{TI_COMPLETION_SOURCE_CUDA_DOMAIN, first},
                                   {TI_COMPLETION_SOURCE_CUDA_DOMAIN, second}};
  auto both = ti_create_completion_set_ext(2, sources, 1000000);
  auto blocked = ti_create_completion_set_ext(1, sources, 1000000);
  ASSERT_NE(both, nullptr);
  ASSERT_NE(blocked, nullptr);
  ti_release_completion_ext(first);
  ti_release_completion_ext(second);
  TiCompletionWaitResult result{};
  ti_wait_completion_set_ext(both, 1000000000, &result);
  ASSERT_EQ(ti_get_last_error(nullptr, nullptr), TI_ERROR_SUCCESS);
  EXPECT_EQ(result.status, TI_COMPLETION_WAIT_READY);
  EXPECT_EQ(result.index, 1);
  EXPECT_FALSE(pending.done());
  ti_wait_completion_set_ext(blocked, 2000000, &result);
  EXPECT_EQ(result.status, TI_COMPLETION_WAIT_TIMEOUT);
  auto waiting = std::async(std::launch::async, [&] {
    TiCompletionWaitResult observed{};
    ti_wait_completion_set_ext(blocked, -1, &observed);
    return observed.status;
  });
  ti_cancel_completion_set_ext(blocked);
  EXPECT_EQ(waiting.wait_for(std::chrono::seconds(1)), std::future_status::ready);
  EXPECT_EQ(waiting.get(), TI_COMPLETION_WAIT_CANCELLED);
  ti_release_completion_set_ext(blocked);
  EXPECT_FALSE(gate.timed_out());
  EXPECT_FALSE(pending.done());
  gate.release();
  pending.wait();
  ti_destroy_runtime(reinterpret_cast<TiRuntime>(runtime.release()));
  ti_release_cuda_execution_domain_ext(left);
  ti_release_cuda_execution_domain_ext(right);
  ti_wait_completion_set_ext(both, 0, &result);
  EXPECT_EQ(result.status, TI_COMPLETION_WAIT_READY);
  EXPECT_EQ(result.index, 0);
  ti_release_completion_set_ext(both);
  EXPECT_EQ(ti_get_last_error(nullptr, nullptr), TI_ERROR_SUCCESS);
}

TEST(CudaExecutionOwner,
     PublicDomainHandlesRetireWithRuntimeAndRejectBadCapacity) {
  if (!cuda::ExecutionDomain::available())
    GTEST_SKIP();
  auto runtime = std::make_unique<capi::LlvmRuntime>(Arch::cuda);
  auto handle = reinterpret_cast<TiRuntime>(runtime.get());
  const TiCudaExecutionDomainCreateInfo invalid{0, 4, 0}, valid{1, 4, 0};
  ti_set_last_error(TI_ERROR_SUCCESS, nullptr);
  EXPECT_EQ(ti_create_cuda_execution_domain_ext(handle, &invalid), nullptr);
  EXPECT_EQ(ti_get_last_error(nullptr, nullptr), TI_ERROR_INVALID_ARGUMENT);
  ti_set_last_error(TI_ERROR_SUCCESS, nullptr);
  auto domain = ti_create_cuda_execution_domain_ext(handle, &valid);
  ASSERT_NE(domain, nullptr);
  auto completion = ti_get_cuda_execution_domain_completion_ext(domain);
  ti_destroy_runtime(reinterpret_cast<TiRuntime>(runtime.release()));
  TiCudaExecutionDomainInfo info{};
  ti_get_cuda_execution_domain_info_ext(domain, &info);
  EXPECT_TRUE(info.closed);
  EXPECT_FALSE(info.accepting);
  EXPECT_TRUE(ti_query_completion_ext(completion));
  ti_wait_completion_ext(completion);
  ti_release_completion_ext(completion);
  ti_release_cuda_execution_domain_ext(domain);
  EXPECT_EQ(ti_get_last_error(nullptr, nullptr), TI_ERROR_SUCCESS);
}

TEST(CudaExecutionOwner, ProgramPrefixesAndBorrowedAotShareOneOwner) {
  if (!cuda::ExecutionDomain::available())
    GTEST_SKIP();
  Program program(Arch::cuda);
  program.materialize_runtime();
  program.synchronize();
  auto registry = program.cuda_execution_domains();
  auto borrower = AotRuntimeClient::create(program);
  auto *runtime = reinterpret_cast<::Runtime *>(borrower->handle());
  ASSERT_EQ(runtime->cuda_execution_domains(), registry);
  ASSERT_EQ(runtime->runtime_fault_domain(), program.runtime_fault_domain());
  auto output = registry->create({2, 0});
  auto physics = registry->create({1, 0});
  testing::CudaExternalGate gate;
  auto blocked = output->submit(
      {}, nullptr, [&](void *stream) { gate.enqueue_wait(stream); });
  auto empty = program.record_runtime_completion();
  EXPECT_TRUE(empty.done());
  EXPECT_EQ(empty.sequence(), 0);
  // Only this explicit publication marks default-queue work. A later domain
  // sequence must never be substituted for the last default completion.
  program.mark_runtime_submission_pending();
  auto default_prefix = program.record_runtime_completion();
  auto other = physics->submit({}, nullptr, [](void *) {});
  other.wait();
  EXPECT_LT(blocked.sequence(), default_prefix.sequence());
  EXPECT_LT(default_prefix.sequence(), other.sequence());
  EXPECT_EQ(program.record_runtime_completion().sequence(),
            default_prefix.sequence());
  EXPECT_FALSE(blocked.done());
  // Explicit ti.sync() must include independent streams. Its wait must not
  // hold the registry admission mutex or the completion query mutex.
  auto sync = std::async(std::launch::async, [&] { program.synchronize(); });
  EXPECT_EQ(sync.wait_for(std::chrono::milliseconds(20)),
            std::future_status::timeout);
  EXPECT_TRUE(registry->accepting());
  EXPECT_FALSE(blocked.done());
  auto during_sync = physics->submit({}, nullptr, [](void *) {});
  during_sync.wait();
  EXPECT_FALSE(blocked.done());
  EXPECT_FALSE(gate.timed_out());
  gate.release();
  sync.get();
  borrower->close();
  EXPECT_TRUE(registry->accepting());
  EXPECT_TRUE(output->snapshot().accepting);
  // Program finalization retires externally retained handles too.
  program.finalize();
  EXPECT_TRUE(output->snapshot().closed);
  EXPECT_TRUE(physics->snapshot().closed);
  EXPECT_FALSE(registry->accepting());
  EXPECT_ANY_THROW(registry->create({1, 0}));
}

TEST(CudaExecutionOwner, SNodeRetirementDrainsDependentDomainBeforeRootFree) {
  if (!cuda::ExecutionDomain::available())
    GTEST_SKIP();
  Program program(Arch::cuda);
  program.materialize_runtime();
  auto root = std::make_unique<SNode>(0, SNodeType::root);
  root->dense(Axis(0), 1).insert_children(SNodeType::place).dt =
      PrimitiveType::i32;
  auto *tree = program.add_snode_tree(std::move(root), false);
  auto *executor = static_cast<LlvmProgramImpl *>(program.get_program_impl())
                       ->get_runtime_executor();
  auto *root_address = executor->get_snode_tree_root_ptr(tree->id());
  auto &device = static_cast<cuda::CudaDevice &>(*program.get_compute_device());
  auto allocation = device.allocate_memory_unique({2 * sizeof(std::uint32_t)});
  ASSERT_EQ(allocation.second, RhiResult::success);
  auto *address =
      static_cast<std::uint32_t *>(device.get_memory_addr(*allocation.first));
  auto registry = program.cuda_execution_domains();
  auto dependent = registry->create({1, 0});
  auto peer = registry->create({1, 0});
  std::promise<void> entered;
  class TreeClient final : public RuntimeClient {
   public:
    TreeClient(std::shared_ptr<cuda::ExecutionDomain> domain,
               std::promise<void> &entered)
        : domain_(std::move(domain)), entered_(entered) {
    }
    void retire() override {
      entered_.set_value();
      domain_->close();
    }

   private:
    std::shared_ptr<cuda::ExecutionDomain> domain_;
    std::promise<void> &entered_;
  };
  program.retain_runtime_client(
      std::make_shared<TreeClient>(dependent, entered),
      program.snapshot_snode_tree_dependencies({tree->id()}));
  program.synchronize();
  testing::CudaExternalGate gate;
  auto blocked = dependent->submit({}, nullptr, [&](void *stream) {
    gate.enqueue_wait(stream);
    auto &driver = CUDADriver::get_instance();
    driver.memsetd32_async(root_address, 0x3210ABCD, 1, stream);
    driver.memcpy_device_to_device_async(address, root_address,
                                         sizeof(std::uint32_t), stream);
  });
  auto destroy =
      std::async(std::launch::async, [&] { program.destroy_snode_tree(tree); });
  entered.get_future().wait();
  EXPECT_EQ(destroy.wait_for(std::chrono::milliseconds(20)),
            std::future_status::timeout);
  auto sentinel = peer->submit({}, nullptr, [&](void *stream) {
    CUDADriver::get_instance().memsetd32_async(address + 1, 0x1234FEDC, 1,
                                               stream);
  });
  sentinel.wait();
  auto context = CUDAContext::get_instance().get_guard();
  std::uint32_t observed = 0;
  CUDADriver::get_instance().memcpy_device_to_host(&observed, address + 1,
                                                   sizeof(observed));
  EXPECT_EQ(observed, 0x1234FEDCu);
  EXPECT_FALSE(blocked.done());
  EXPECT_FALSE(gate.timed_out());
  gate.release();
  destroy.get();
  EXPECT_TRUE(blocked.done());
  EXPECT_TRUE(dependent->snapshot().closed);
  EXPECT_TRUE(peer->snapshot().accepting);
  CUDADriver::get_instance().memcpy_device_to_host(&observed, address,
                                                   sizeof(observed));
  EXPECT_EQ(observed, 0x3210ABCDu);
  allocation.first.reset();
  program.finalize();
}

TEST(CudaExecutionOwner,
     ExternalStorageRetirementDrainsOnlyItsDependentDomain) {
  if (!cuda::ExecutionDomain::available())
    GTEST_SKIP();
  Program program(Arch::cuda);
  program.materialize_runtime();
  auto &device = static_cast<cuda::CudaDevice &>(*program.get_compute_device());
  auto allocation = device.allocate_memory_unique({sizeof(std::uint32_t)});
  auto peer_allocation = device.allocate_memory_unique({sizeof(std::uint32_t)});
  ASSERT_EQ(allocation.second, RhiResult::success);
  ASSERT_EQ(peer_allocation.second, RhiResult::success);
  auto *address = device.get_memory_addr(*allocation.first);
  auto *peer_address = device.get_memory_addr(*peer_allocation.first);
  auto registry = program.cuda_execution_domains();
  auto dependent = registry->create({1, 0});
  auto peer = registry->create({1, 0});
  std::atomic<int> frees{0};
  const auto owner = program.register_external_dense_storage(
      *allocation.first, sizeof(std::uint32_t), [&] {
        EXPECT_TRUE(dependent->snapshot().closed);
        auto context = CUDAContext::get_instance().get_guard();
        std::uint32_t observed = 0;
        CUDADriver::get_instance().memcpy_device_to_host(&observed, address,
                                                         sizeof(observed));
        EXPECT_EQ(observed, 0x1234ABCDu);
        allocation.first.reset();
        ++frees;
      });
  std::promise<void> entered;
  class StorageClient final : public RuntimeClient {
   public:
    StorageClient(std::shared_ptr<cuda::ExecutionDomain> domain,
                  std::promise<void> &entered)
        : domain_(std::move(domain)), entered_(entered) {
    }
    void retire() override {
      entered_.set_value();
      domain_->close();
    }

   private:
    std::shared_ptr<cuda::ExecutionDomain> domain_;
    std::promise<void> &entered_;
  };
  program.retain_external_storage_client(
      std::make_shared<StorageClient>(dependent, entered), owner);
  program.synchronize();
  testing::CudaExternalGate gate;
  auto blocked = dependent->submit({}, nullptr, [&](void *stream) {
    gate.enqueue_wait(stream);
    CUDADriver::get_instance().memsetd32_async(address, 0x1234ABCD, 1, stream);
  });
  auto retirement = std::async(std::launch::async, [&] {
    program.retire_external_dense_storage(owner);
  });
  entered.get_future().wait();
  EXPECT_EQ(retirement.wait_for(std::chrono::milliseconds(20)),
            std::future_status::timeout);
  EXPECT_EQ(frees.load(), 0);
  peer->submit({}, nullptr,
               [&](void *stream) {
                 CUDADriver::get_instance().memsetd32_async(
                     peer_address, 0x9876FEDC, 1, stream);
               })
      .wait();
  auto context = CUDAContext::get_instance().get_guard();
  std::uint32_t observed = 0;
  CUDADriver::get_instance().memcpy_device_to_host(&observed, peer_address,
                                                   sizeof(observed));
  EXPECT_EQ(observed, 0x9876FEDCu);
  EXPECT_FALSE(blocked.done());
  EXPECT_FALSE(gate.timed_out());
  gate.release();
  retirement.get();
  EXPECT_EQ(frees.load(), 1);
  EXPECT_TRUE(blocked.done());
  EXPECT_TRUE(peer->snapshot().accepting);
  peer_allocation.first.reset();
  program.finalize();
}

TEST(CudaExecutionOwner, LiveJitCodeLoanDrainsBeforeFieldAndCodeRetirement) {
  if (!cuda::ExecutionDomain::available())
    GTEST_SKIP();
  Program program(Arch::cuda);
  program.materialize_runtime();
  auto root = std::make_unique<SNode>(0, SNodeType::root);
  auto *place = &root->dense(Axis(0), 1).insert_children(SNodeType::place);
  place->dt = PrimitiveType::i32;
  auto *tree = program.add_snode_tree(std::move(root), false);
  IRBuilder ir;
  auto *array = ir.create_ndarray_arg_load({0}, PrimitiveType::i32, 1, 0);
  auto *zero = ir.get_int32(0);
  auto *value = ir.get_int32(314159);
  auto *field = ir.create_global_ptr(place, {zero});
  ir.create_global_store(field, value);
  ir.create_global_store(ir.create_external_ptr(array, {zero}),
                         ir.create_global_load(field));
  auto kernel = std::make_unique<Kernel>(program, ir.extract_ir(), "jit_loan");
  kernel->insert_ndarray_param(PrimitiveType::i32, 1);
  kernel->finalize_params();
  kernel->finalize_rets();
  Dispatch dispatch(kernel.get(), {aot::Arg(aot::ArgKind::kNdarray, "out",
                                            PrimitiveType::i32, 1)});
  aot::CompiledGraph graph({dispatch.compile_dispatch()});
  auto borrower = AotRuntimeClient::create(program);
  auto *runtime = reinterpret_cast<::Runtime *>(borrower->handle());
  auto rt = reinterpret_cast<TiRuntime>(runtime);
  auto module = reinterpret_cast<TiAotModule>(
      borrower->import_jit_kernels({{"stage", &graph, 0, false}}));
  ASSERT_NE(module, nullptr);
  const TiCudaExecutionDomainCreateInfo options{2, 4, 0};
  auto output = ti_create_cuda_execution_domain_ext(rt, &options);
  auto peer = ti_create_cuda_execution_domain_ext(rt, &options);
  ASSERT_NE(output, nullptr);
  ASSERT_NE(peer, nullptr);
  TiMemoryAllocateInfo allocation_info{};
  allocation_info.size = 2 * sizeof(std::uint32_t);
  allocation_info.usage = TI_MEMORY_USAGE_STORAGE_BIT;
  auto memory = ti_allocate_memory(rt, &allocation_info);
  TiNamedArgument argument{};
  argument.name = "out";
  argument.argument.type = TI_ARGUMENT_TYPE_NDARRAY;
  argument.argument.value.ndarray = {
      memory, {1, {2}}, {0, {}}, TI_DATA_TYPE_I32};
  const std::string plan =
      R"({"version":1,"operations":[{"kind":"kernel","entry":"stage","arguments":["out"]}]})";
  auto prepared = ti_prepare_execution_plan_on_domain_ext(
      rt, module, output, plan.size(), plan.c_str(), 1, &argument, 0, nullptr);
  ASSERT_NE(prepared, nullptr);
  ti_initialize_prepared_ext(rt, prepared);
  ASSERT_EQ(ti_get_last_error(nullptr, nullptr), TI_ERROR_SUCCESS);
  auto &device = static_cast<cuda::CudaDevice &>(runtime->get());
  auto *address = static_cast<std::uint32_t *>(
      device.get_memory_addr(runtime->get_memory(memory).allocation));
  auto native_output = capi::get_execution_domain(output);
  auto native_peer = capi::get_execution_domain(peer);
  testing::CudaExternalGate gate;
  auto blocked = native_output->submit(
      {}, nullptr, [&](void *stream) { gate.enqueue_wait(stream); });
  TiCompletion ticket = nullptr;
  ASSERT_TRUE(ti_try_submit_prepared_ext(rt, prepared, 0, nullptr, &ticket));
  auto destroy =
      std::async(std::launch::async, [&] { program.destroy_snode_tree(tree); });
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (native_output->snapshot().accepting &&
         std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  EXPECT_FALSE(native_output->snapshot().accepting);
  EXPECT_EQ(destroy.wait_for(std::chrono::milliseconds(0)),
            std::future_status::timeout);
  native_peer
      ->submit({}, nullptr,
               [&](void *stream) {
                 CUDADriver::get_instance().memsetd32_async(address + 1, 271828,
                                                            1, stream);
               })
      .wait();
  auto context = CUDAContext::get_instance().get_guard();
  std::uint32_t observed = 0;
  CUDADriver::get_instance().memcpy_device_to_host(&observed, address + 1,
                                                   sizeof(observed));
  EXPECT_EQ(observed, 271828u);
  EXPECT_FALSE(blocked.done());
  EXPECT_FALSE(ti_query_completion_ext(ticket));
  EXPECT_FALSE(gate.timed_out());
  gate.release();
  destroy.get();
  EXPECT_TRUE(ti_query_completion_ext(ticket));
  EXPECT_TRUE(native_peer->snapshot().accepting);
  CUDADriver::get_instance().memcpy_device_to_host(&observed, address,
                                                   sizeof(observed));
  EXPECT_EQ(observed, 314159u);
  ti_release_completion_ext(ticket);
  ti_destroy_prepared_ext(rt, prepared);
  ti_destroy_aot_module(module);
  ti_free_memory(rt, memory);
  ti_close_cuda_execution_domain_ext(output);
  ti_close_cuda_execution_domain_ext(peer);
  ti_release_cuda_execution_domain_ext(output);
  ti_release_cuda_execution_domain_ext(peer);
  borrower->close();
  program.finalize();
}

TEST(CudaExecutionOwner,
     DenseFieldLoanRetiresAliasesWithoutKernelFieldClosure) {
  if (!cuda::ExecutionDomain::available())
    GTEST_SKIP();
  Program program(Arch::cuda);
  program.materialize_runtime();
  auto root = std::make_unique<SNode>(0, SNodeType::root);
  auto *place = &root->dense(Axis(0), 4).insert_children(SNodeType::place);
  place->dt = PrimitiveType::i32;
  auto *tree = program.add_snode_tree(std::move(root), false);
  auto description = storage::describe_dense_field_storage(
      program, place, {place}, PrimitiveType::i32, {4}, {});
  ASSERT_TRUE(description);
  IRBuilder ir;
  auto *input = ir.create_ndarray_arg_load({0}, PrimitiveType::i32, 1, 0);
  auto *out = ir.create_ndarray_arg_load({1}, PrimitiveType::i32, 1, 0);
  auto *zero = ir.get_int32(0);
  auto *address = ir.create_external_ptr(input, {zero});
  ir.create_global_store(address, ir.get_int32(1234567));
  ir.create_global_store(ir.create_external_ptr(out, {zero}),
                         ir.create_global_load(address));
  auto kernel = std::make_unique<Kernel>(program, ir.extract_ir(), "view_loan");
  kernel->insert_ndarray_param(PrimitiveType::i32, 1);
  kernel->insert_ndarray_param(PrimitiveType::i32, 1);
  kernel->finalize_params();
  kernel->finalize_rets();
  Dispatch dispatch(
      kernel.get(),
      {aot::Arg(aot::ArgKind::kNdarray, "x", PrimitiveType::i32, 1),
       aot::Arg(aot::ArgKind::kNdarray, "out", PrimitiveType::i32, 1)});
  aot::CompiledGraph graph({dispatch.compile_dispatch()});
  ASSERT_TRUE(graph.dispatches.front().snode_tree_dependencies.empty());
  auto borrower = AotRuntimeClient::create(program);
  auto *runtime = reinterpret_cast<::Runtime *>(borrower->handle());
  auto rt = reinterpret_cast<TiRuntime>(runtime);
  auto module = reinterpret_cast<TiAotModule>(
      borrower->import_jit_kernels({{"stage", &graph, 0, false}}));
  auto memory = reinterpret_cast<TiMemory>(
      std::get<0>(borrower->import_dense_view(*description.descriptor)));
  // Retain an alias independent of the execution's own derived view, too.
  capi::ExecutionResourceStorage aliases(*runtime);
  auto alias =
      aliases.view(memory, sizeof(std::int32_t), 2 * sizeof(std::int32_t));
  TiMemoryAllocateInfo allocation{};
  allocation.size = 2 * sizeof(std::int32_t);
  allocation.usage = TI_MEMORY_USAGE_STORAGE_BIT;
  auto output_memory = ti_allocate_memory(rt, &allocation);
  const TiCudaExecutionDomainCreateInfo options{2, 4, 0};
  auto selected = ti_create_cuda_execution_domain_ext(rt, &options);
  auto peer = runtime->cuda_execution_domains()->create({1, 0});
  std::array<TiNamedArgument, 2> arguments{};
  arguments[0].name = "input";
  arguments[0].argument.type = TI_ARGUMENT_TYPE_NDARRAY;
  arguments[0].argument.value.ndarray = {
      memory, {1, {4}}, {0, {}}, TI_DATA_TYPE_I32};
  arguments[1].name = "out";
  arguments[1].argument.type = TI_ARGUMENT_TYPE_NDARRAY;
  arguments[1].argument.value.ndarray = {
      output_memory, {1, {2}}, {0, {}}, TI_DATA_TYPE_I32};
  const std::string plan =
      R"({"version":3,"resources":[{"kind":"ndarray_view","name":"view","array":"input","offset":1,"shape":[2]}],"operations":[{"kind":"kernel","entry":"stage","arguments":["view","out"]}]})";
  auto prepared = ti_prepare_execution_plan_on_domain_ext(
      rt, module, selected, plan.size(), plan.c_str(), arguments.size(),
      arguments.data(), 0, nullptr);
  ASSERT_NE(prepared, nullptr);
  ti_initialize_prepared_ext(rt, prepared);
  ASSERT_EQ(ti_get_last_error(nullptr, nullptr), TI_ERROR_SUCCESS);
  auto domain = capi::get_execution_domain(selected);
  auto &device = static_cast<cuda::CudaDevice &>(runtime->get());
  auto *output_address = static_cast<std::uint32_t *>(
      device.get_memory_addr(runtime->get_memory(output_memory).allocation));
  testing::CudaExternalGate gate;
  auto blocked = domain->submit(
      {}, nullptr, [&](void *stream) { gate.enqueue_wait(stream); });
  TiCompletion ticket = nullptr;
  ASSERT_TRUE(ti_try_submit_prepared_ext(rt, prepared, 0, nullptr, &ticket));
  auto destroy =
      std::async(std::launch::async, [&] { program.destroy_snode_tree(tree); });
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (domain->snapshot().accepting &&
         std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  EXPECT_FALSE(domain->snapshot().accepting);
  EXPECT_EQ(destroy.wait_for(std::chrono::milliseconds(0)),
            std::future_status::timeout);
  peer->submit({}, nullptr,
               [&](void *stream) {
                 CUDADriver::get_instance().memsetd32_async(output_address + 1,
                                                            7654321, 1, stream);
               })
      .wait();
  auto context = CUDAContext::get_instance().get_guard();
  std::uint32_t observed = 0;
  CUDADriver::get_instance().memcpy_device_to_host(
      &observed, output_address + 1, sizeof(observed));
  EXPECT_EQ(observed, 7654321u);
  EXPECT_FALSE(blocked.done());
  EXPECT_FALSE(gate.timed_out());
  gate.release();
  destroy.get();
  EXPECT_TRUE(ti_query_completion_ext(ticket));
  EXPECT_TRUE(peer->snapshot().accepting);
  CUDADriver::get_instance().memcpy_device_to_host(&observed, output_address,
                                                   sizeof(observed));
  EXPECT_EQ(observed, 1234567u);
  EXPECT_THROW(runtime->get_memory(memory), MemoryBindingError);
  EXPECT_THROW(runtime->get_memory(alias), MemoryBindingError);
  EXPECT_ANY_THROW(borrower->import_dense_view(*description.descriptor));
  // Destroy the alias shell while Runtime still exists.
  aliases.retain({});
  ti_release_completion_ext(ticket);
  ti_destroy_prepared_ext(rt, prepared);
  ti_destroy_aot_module(module);
  ti_free_memory(rt, memory);
  ti_free_memory(rt, output_memory);
  ti_release_cuda_execution_domain_ext(selected);
  borrower->close();
  program.finalize();
}

TEST(CudaExecutionOwner,
     DenseBorrowSnapshotDoesNotKeepCallbackArmedAfterReset) {
  if (!cuda::ExecutionDomain::available())
    GTEST_SKIP();
  Program program(Arch::cuda);
  program.materialize_runtime();
  auto root = std::make_unique<SNode>(0, SNodeType::root);
  auto *place = &root->dense(Axis(0), 4).insert_children(SNodeType::place);
  place->dt = PrimitiveType::i32;
  program.add_snode_tree(std::move(root), false);
  const auto description = storage::describe_dense_field_storage(
      program, place, {place}, PrimitiveType::i32, {4}, {});
  ASSERT_TRUE(description);
  auto borrower = AotRuntimeClient::create(program);
  auto *runtime = reinterpret_cast<::Runtime *>(borrower->handle());
  auto memory = reinterpret_cast<TiMemory>(
      std::get<0>(borrower->import_dense_view(*description.descriptor)));
  const auto snapshot = runtime->get_memory(memory);
  ASSERT_TRUE(snapshot.borrowed && snapshot.borrowed->owner);
  const auto second = reinterpret_cast<TiMemory>(
      std::get<0>(borrower->import_dense_view(*description.descriptor)));
  const auto released_snapshot = runtime->get_memory(second);
  runtime->free_memory(second);
  EXPECT_FALSE(released_snapshot.borrowed->live);
  EXPECT_FALSE(released_snapshot.borrowed->owner);
  program.finalize();
  EXPECT_TRUE(borrower->closed());
  EXPECT_FALSE(snapshot.borrowed->live);
  EXPECT_FALSE(snapshot.borrowed->owner);
  borrower->close();
}

TEST(CudaExecutionOwner, StandaloneAotWaitAndDestroyRetireIndependentWork) {
  if (!cuda::ExecutionDomain::available())
    GTEST_SKIP();
  auto runtime = std::make_unique<capi::LlvmRuntime>(Arch::cuda);
  auto registry = runtime->cuda_execution_domains();
  auto output = registry->create({1, 0});
  testing::CudaExternalGate gate;
  auto blocked = output->submit(
      {}, nullptr, [&](void *stream) { gate.enqueue_wait(stream); });
  auto waiter = std::async(std::launch::async, [&] {
    static_cast<::Runtime *>(runtime.get())->wait();
  });
  EXPECT_EQ(waiter.wait_for(std::chrono::milliseconds(20)),
            std::future_status::timeout);
  EXPECT_FALSE(blocked.done());
  EXPECT_TRUE(registry->accepting());
  EXPECT_FALSE(gate.timed_out());
  gate.release();
  waiter.get();
  // Exercise the public C API destruction entry as well as the derived
  // destructor; it must close domains before clearing prepared executions.
  ti_destroy_runtime(reinterpret_cast<TiRuntime>(runtime.release()));
  EXPECT_TRUE(blocked.done());
  EXPECT_TRUE(output->snapshot().closed);
  EXPECT_FALSE(registry->accepting());
  EXPECT_ANY_THROW(output->submit({}, nullptr, [](void *) {}));
}

TEST(CudaExecutionOwner, FinalizeDrainsBeforeClosingStorageAndRejectsNewWork) {
  if (!cuda::ExecutionDomain::available())
    GTEST_SKIP();
  Program program(Arch::cuda);
  program.materialize_runtime();
  auto registry = program.cuda_execution_domains();
  auto output = registry->create({1, 0});
  testing::CudaExternalGate gate;
  auto blocked = output->submit(
      {}, nullptr, [&](void *stream) { gate.enqueue_wait(stream); });
  auto finalizer = std::async(std::launch::async, [&] { program.finalize(); });
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (registry->accepting() && std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  EXPECT_FALSE(registry->accepting());
  EXPECT_ANY_THROW(registry->create({1, 0}));
  EXPECT_ANY_THROW(output->submit({}, nullptr, [](void *) {}));
  EXPECT_FALSE(blocked.done());
  EXPECT_EQ(finalizer.wait_for(std::chrono::milliseconds(0)),
            std::future_status::timeout);
  EXPECT_EQ(program.runtime_fault_domain()->state(),
            RuntimeLifecycleState::kFinalizing);
  EXPECT_FALSE(gate.timed_out());
  gate.release();
  finalizer.get();
  EXPECT_TRUE(blocked.done());
  EXPECT_TRUE(output->snapshot().closed);
  EXPECT_EQ(program.runtime_fault_domain()->state(),
            RuntimeLifecycleState::kFinalized);
}

}  // namespace
}  // namespace taichi::lang
#endif
