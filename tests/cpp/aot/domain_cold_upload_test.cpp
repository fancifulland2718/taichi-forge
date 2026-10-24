#include "gtest/gtest.h"

#if defined(TI_TEST_RUNTIME_C_API)
#include <array>
#include <chrono>
#include <cstring>
#include <future>
#include "c_api/src/execution_domain_binding.h"
#include "taichi/program/program.h"
#include "taichi/runtime/native_data_upload.h"
#include "taichi/runtime/native_operation_registry.h"
#ifdef TI_WITH_VULKAN
#include "taichi/runtime/vulkan_execution_domain.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"
#include "tests/cpp/aot/vulkan/pending_domain_action.h"
#endif
#if defined(TI_WITH_CUDA) && defined(TI_WITH_VULKAN)
#include "taichi/runtime/cuda/execution_domain.h"
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/runtime/program_impls/llvm/llvm_program.h"
#include "taichi/runtime/llvm/aot_graph_data.h"
#include "taichi/runtime/cuda/kernel_launcher.h"
#include "c_api/src/taichi_core_impl.h"
#include "tests/cpp/rhi/common/cuda_external_gate.h"
#endif

namespace taichi::lang {
namespace {
using namespace std::chrono_literals;

struct UploadArrays {
  Device &device;
  std::array<DeviceAllocationUnique, 2> arrays;
  // Exercise all 64 bits and a second differently sized allocation.
  std::array<uint64_t, 2> words{0x12345678deadbeefULL, 0xfedcba9876543210ULL};
  std::array<uint8_t, 5> bytes{1, 3, 9, 27, 81};
  explicit UploadArrays(Device &device) : device(device) {
    for (int i = 0; i < 2; ++i) {
      auto [allocation, status] = device.allocate_memory_unique(
          {i ? sizeof(bytes) : sizeof(words), false, false, false,
           AllocUsage::Storage});
      if (status != RhiResult::success)
        throw std::runtime_error("Cold upload fixture allocation failed");
      arrays[i] = std::move(allocation);
    }
  }
  std::vector<NativeDataUpload> uploads() const {
    return {{arrays[0]->get_ptr(), words.data(), sizeof(words)},
            {arrays[1]->get_ptr(), bytes.data(), sizeof(bytes)}};
  }
  void check() {
    std::array<uint64_t, 2> result{};
    std::array<uint8_t, 5> observed_bytes{};
    DevicePtr destinations[]{arrays[0]->get_ptr(), arrays[1]->get_ptr()};
    void *outputs[]{result.data(), observed_bytes.data()};
    size_t sizes[]{sizeof(result), sizeof(observed_bytes)};
    ASSERT_EQ(device.readback_data(destinations, outputs, sizes, 2),
              RhiResult::success);
    EXPECT_EQ(result, words);
    EXPECT_EQ(observed_bytes, bytes);
  }
};

#ifdef TI_WITH_VULKAN
TEST(DomainColdUpload, VulkanBatchIgnoresDefaultAndPeerButHonorsCapacity) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  Program program(Arch::vulkan);
  program.materialize_runtime();
  auto &device =
      static_cast<vulkan::VulkanDevice &>(*program.get_compute_device());
  if (device.created_compute_queue_count() < 3 ||
      !device.vk_caps().timeline_semaphore)
    GTEST_SKIP();
  auto domain = std::make_shared<vulkan::ExecutionDomain>(
      device, program.runtime_fault_domain(),
      vulkan::ExecutionDomain::Options{1, 1});
  auto peer = std::make_shared<vulkan::ExecutionDomain>(
      device, program.runtime_fault_domain(),
      vulkan::ExecutionDomain::Options{1, 1});
  capi::PreparedExecutionDomain selected(domain);
  UploadArrays data(device);
  auto invalid = data.uploads();
  invalid[1].bytes += 1;
  const auto before = device.queue_submission_snapshot().queue_submit_calls;
  EXPECT_THROW(selected.upload_owned_data(device, invalid),
               std::invalid_argument);
  EXPECT_EQ(device.queue_submission_snapshot().queue_submit_calls, before);
  program.synchronize();
  auto gates = std::make_shared<test_detail::GateOwner>(3);
  gates->gates[0] = std::make_unique<test_detail::TimelineGate>(device, 0);
  gates->gates[1] = std::make_unique<test_detail::TimelineGate>(
      device, peer->snapshot().queue_index);
  auto pending_peer = peer->submit(peer->prepare_commands({}));
  test_detail::PendingAction upload(
      gates, [&] { selected.upload_owned_data(device, data.uploads()); });
  const auto done = upload.status(2s);
  EXPECT_EQ(done, std::future_status::ready);
  if (done != std::future_status::ready)
    gates->release_all();
  EXPECT_NO_THROW(upload.finish());
  EXPECT_FALSE(pending_peer.done());
  gates->gates[2] = std::make_unique<test_detail::TimelineGate>(
      device, domain->snapshot().queue_index);
  auto full = domain->submit(domain->prepare_commands({}));
  const auto admitted = device.queue_submission_snapshot().queue_submit_calls;
  EXPECT_THROW(selected.upload_owned_data(device, data.uploads()),
               vulkan::ExecutionDomain::Busy);
  EXPECT_EQ(device.queue_submission_snapshot().queue_submit_calls, admitted);
  gates->release(2);
  full.wait();
  domain->collect_ready(1);
  selected.upload_owned_data(device, data.uploads());
  gates->release_all();
  pending_peer.wait();
  data.check();
  peer->close();
  domain->close();
}
#endif

#if defined(TI_WITH_CUDA) && defined(TI_WITH_VULKAN)
TEST(DomainPreparation, CudaUsesActualKernelFieldDependencies) {
  const auto *path = std::getenv("TI_TEST_PREPARATION_AOT_DIR");
  if (!path || !cuda::ExecutionDomain::available())
    GTEST_SKIP() << "Requires the compiled CUDA Field dependency fixture";
  struct Session {
    TiRuntime runtime{nullptr};
    TiAotModule module{nullptr};
    ~Session() {
      if (module)
        ti_destroy_aot_module(module);
      if (runtime)
        ti_destroy_runtime(runtime);
    }
  } session;
  session.runtime = ti_create_runtime(TI_ARCH_CUDA, 0);
  ASSERT_NE(session.runtime, nullptr);
  session.module = ti_load_aot_module(session.runtime, path);
  ASSERT_NE(session.module, nullptr);
  auto &runtime = *reinterpret_cast<::Runtime *>(session.runtime);
  for (const auto &entry :
       {std::make_pair("rootless", false), std::make_pair("field", true)}) {
    const auto handle = ti_get_aot_module_kernel(session.module, entry.first);
    ASSERT_NE(handle, nullptr);
    auto *kernel =
        dynamic_cast<llvm_aot::KernelImpl *>(runtime.get_kernel(handle).kernel);
    ASSERT_NE(kernel, nullptr);
    auto *launcher = dynamic_cast<cuda::KernelLauncher *>(&kernel->launcher());
    ASSERT_NE(launcher, nullptr);
    EXPECT_EQ(launcher->has_field_dependencies(kernel->handle()), entry.second);
  }
}

TEST(DomainColdUpload, CudaInitializationJoinIgnoresLaterDefaultWork) {
  if (!cuda::ExecutionDomain::available())
    GTEST_SKIP();
  Program program(Arch::cuda);
  program.materialize_runtime();
  auto &device = *program.get_compute_device();
  auto domain = std::make_shared<cuda::ExecutionDomain>(
      program.runtime_fault_domain(), cuda::ExecutionDomain::Options{1, 0, 1});
  capi::PreparedExecutionDomain selected(domain);
  UploadArrays data(device);
  program.synchronize();
  testing::CudaExternalGate gate;
  gate.enqueue_wait(nullptr);
  auto joined = selected.join_allocation_initialization(
      device, {*data.arrays[0], *data.arrays[1], *data.arrays[0]});
  auto wait = std::async(std::launch::async, [&] { joined.wait(); });
  const auto status = wait.wait_for(2s);
  EXPECT_EQ(status, std::future_status::ready);
  if (status != std::future_status::ready)
    gate.release();
  wait.get();
  EXPECT_FALSE(gate.timed_out());
  gate.release();
  domain->close();
}

TEST(DomainColdUpload, CudaUploadWaitsForAllocationInitialization) {
  if (!cuda::ExecutionDomain::available())
    GTEST_SKIP();
  Program program(Arch::cuda);
  program.materialize_runtime();
  auto &device = static_cast<cuda::CudaDevice &>(*program.get_compute_device());
  auto *executor = static_cast<LlvmProgramImpl *>(program.get_program_impl())
                       ->get_runtime_executor();
  auto domain = std::make_shared<cuda::ExecutionDomain>(
      program.runtime_fault_domain(), cuda::ExecutionDomain::Options{1, 0, 1});
  capi::PreparedExecutionDomain selected(domain);
  LlvmDevice::LlvmRuntimeAllocParams params;
  params.size = 8192;
  params.runtime_jit = executor->get_runtime_jit_module();
  params.runtime = executor->get_llvm_runtime();
  params.result_buffer = program.result_buffer;
  params.use_memory_pool = executor->use_device_memory_pool();
  // Exercise the runtime's actual allocation policy and warm its pool. Its
  // zero-fill still precedes the selected-domain upload on the default stream.
  auto warm = device.allocate_memory_runtime(params);
  program.synchronize();
  std::array<uint64_t, 2> expected{0x12345678deadbeefULL,
                                   0xfedcba9876543210ULL};
  selected.upload_owned_data(
      device, {{warm.get_ptr(), expected.data(), sizeof(expected)}});
  program.synchronize();
  device.dealloc_memory(warm);
  program.synchronize();
  testing::CudaExternalGate gate;
  gate.enqueue_wait(nullptr);
  auto allocation = std::async(std::launch::async, [&] {
    return device.allocate_memory_runtime(params);
  });
  const auto allocation_status = allocation.wait_for(500ms);
  if (allocation_status != std::future_status::ready)
    gate.release();
  auto target = allocation.get();
  EXPECT_EQ(allocation_status, std::future_status::ready);
  auto upload = std::async(std::launch::async, [&] {
    selected.upload_owned_data(
        device, {{target.get_ptr(), expected.data(), sizeof(expected)}});
  });
  EXPECT_EQ(upload.wait_for(200ms), std::future_status::timeout);
  gate.release();
  EXPECT_NO_THROW(upload.get());
  program.synchronize();
  std::array<uint64_t, 2> observed{};
  auto ptr = target.get_ptr();
  void *out = observed.data();
  size_t bytes = sizeof(observed);
  EXPECT_EQ(device.readback_data(&ptr, &out, &bytes), RhiResult::success);
  EXPECT_EQ(observed, expected);
  EXPECT_FALSE(gate.timed_out());
  device.dealloc_memory(target);
  domain->close();
}

TEST(DomainColdUpload, CudaImageConstructionCopyIgnoresDefaultAndPeer) {
  if (!cuda::ExecutionDomain::available())
    GTEST_SKIP();
  Program program(Arch::cuda);
  program.materialize_runtime();
  auto context = CUDAContext::get_instance().get_guard();
  auto &device = *program.get_compute_device();
  auto domain = std::make_shared<cuda::ExecutionDomain>(
      program.runtime_fault_domain(), cuda::ExecutionDomain::Options{1, 0, 1});
  auto peer = std::make_shared<cuda::ExecutionDomain>(
      program.runtime_fault_domain(), cuda::ExecutionDomain::Options{1, 0, 1});
  capi::PreparedExecutionDomain selected(domain);
  UploadArrays data(device);
  selected.upload_owned_data(device, data.uploads());
  NativeArrayBinding pixels{
      data.arrays[0]->get_ptr(), sizeof(data.words), PrimitiveType::f32, {4}};
  auto &registry = native_operation_registry();
  const auto request = registry.decode(
      NativeOperationDescription::parse(liong::json::parse_complete(
          R"({"kind":"native","provider":"image","operation":"sampled","version":1,
        "parameters":{"shape":[4],"format":"r32f"},"bindings":{"pixels":"pixels"}})")),
      NativeObjectKind::resource);
  NativePreparationContext preparation(Arch::cuda, device, {},
                                       program.runtime_fault_domain());
  testing::CudaExternalGate default_gate, peer_gate, capacity_gate;
  RuntimeCompletion pending_peer;
  bool called = false;
  preparation.complete_owned_initialization = [&](auto operation,
                                                  const auto &inputs) {
    called = true;
    // Allocation precedes these gates: this checks the construction copy and
    // retirement boundary, without promising asynchronous driver allocation.
    default_gate.enqueue_wait(nullptr);
    pending_peer = peer->submit(
        {}, nullptr, [&](void *stream) { peer_gate.enqueue_wait(stream); });
    auto full = domain->submit(
        {}, nullptr, [&](void *stream) { capacity_gate.enqueue_wait(stream); });
    const auto admitted = domain->latest_completion().sequence();
    EXPECT_THROW(selected.complete_owned_initialization(device, nullptr,
                                                        operation, inputs),
                 cuda::ExecutionDomainBusy);
    EXPECT_EQ(domain->latest_completion().sequence(), admitted);
    capacity_gate.release();
    full.wait();
    domain->collect_ready(1);
    return selected.complete_owned_initialization(device, nullptr, operation,
                                                  inputs);
  };
  program.synchronize();
  auto construction = std::async(std::launch::async, [&] {
    return registry.prepare_resource(preparation, request,
                                     {{"pixels", pixels}});
  });
  const auto done = construction.wait_for(2s);
  EXPECT_EQ(done, std::future_status::ready);
  if (done != std::future_status::ready) {
    default_gate.release();
    peer_gate.release();
    capacity_gate.release();
  }
  auto image = construction.get();
  EXPECT_TRUE(called);
  EXPECT_NE(image, nullptr);
  EXPECT_FALSE(pending_peer.done());
  EXPECT_FALSE(default_gate.timed_out());
  default_gate.release();
  peer_gate.release();
  pending_peer.wait();
  image.reset();
  peer->close();
  domain->close();
}

TEST(DomainColdUpload, CudaBatchIgnoresDefaultAndPeerButHonorsCapacity) {
  if (!cuda::ExecutionDomain::available())
    GTEST_SKIP();
  Program program(Arch::cuda);
  program.materialize_runtime();
  auto context = CUDAContext::get_instance().get_guard();
  auto &device = *program.get_compute_device();
  auto domain = std::make_shared<cuda::ExecutionDomain>(
      program.runtime_fault_domain(), cuda::ExecutionDomain::Options{1, 0, 1});
  auto peer = std::make_shared<cuda::ExecutionDomain>(
      program.runtime_fault_domain(), cuda::ExecutionDomain::Options{1, 0, 1});
  capi::PreparedExecutionDomain selected(domain);
  UploadArrays data(device);
  auto invalid = data.uploads();
  invalid[1].bytes += 1;
  const auto before = domain->latest_completion().sequence();
  EXPECT_THROW(selected.upload_owned_data(device, invalid),
               std::invalid_argument);
  EXPECT_EQ(domain->latest_completion().sequence(), before);
  // Warm the domain-owned pinned pool before gating unrelated streams. First
  // pinning/allocation costs are cold and are not a queue-isolation promise.
  selected.upload_owned_data(device, data.uploads());
  program.synchronize();
  testing::CudaExternalGate default_gate, peer_gate, capacity_gate;
  default_gate.enqueue_wait(nullptr);
  auto pending_peer = peer->submit(
      {}, nullptr, [&](void *stream) { peer_gate.enqueue_wait(stream); });
  auto upload = std::async(std::launch::async, [&] {
    selected.upload_owned_data(device, data.uploads());
  });
  const auto done = upload.wait_for(2s);
  EXPECT_EQ(done, std::future_status::ready);
  if (done != std::future_status::ready) {
    default_gate.release();
    peer_gate.release();
  }
  EXPECT_NO_THROW(upload.get());
  EXPECT_FALSE(pending_peer.done());
  EXPECT_FALSE(default_gate.timed_out());
  auto full = domain->submit(
      {}, nullptr, [&](void *stream) { capacity_gate.enqueue_wait(stream); });
  const auto admitted = domain->latest_completion().sequence();
  EXPECT_THROW(selected.upload_owned_data(device, data.uploads()),
               cuda::ExecutionDomainBusy);
  EXPECT_EQ(domain->latest_completion().sequence(), admitted);
  capacity_gate.release();
  full.wait();
  domain->collect_ready(1);
  selected.upload_owned_data(device, data.uploads());
  default_gate.release();
  peer_gate.release();
  pending_peer.wait();
  data.check();
  peer->close();
  domain->close();
}
#endif
}  // namespace
}  // namespace taichi::lang
#endif
