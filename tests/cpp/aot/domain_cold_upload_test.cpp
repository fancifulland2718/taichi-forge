#include "gtest/gtest.h"

#if defined(TI_TEST_RUNTIME_C_API)
#include <array>
#include <chrono>
#include <cstring>
#include <future>
#include "c_api/src/execution_domain_binding.h"
#include "taichi/program/program.h"
#include "taichi/runtime/native_data_upload.h"
#ifdef TI_WITH_VULKAN
#include "taichi/runtime/vulkan_execution_domain.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"
#include "tests/cpp/aot/vulkan/pending_domain_action.h"
#endif
#if defined(TI_WITH_CUDA) && defined(TI_WITH_VULKAN)
#include "taichi/runtime/cuda/execution_domain.h"
#include "taichi/rhi/cuda/cuda_context.h"
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
