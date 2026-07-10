#include "gtest/gtest.h"
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/rhi/vulkan/vulkan_device_creator.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"
#include "tests/cpp/aot/gfx_utils.h"

#include <array>
#include <atomic>
#include <thread>
#include <vector>

using namespace taichi;
using namespace lang;

TEST(VulkanDeviceTest, ConcurrentQueueSubmissions) {
  if (!vulkan::is_vulkan_api_available()) {
    return;
  }

  vulkan::VulkanDeviceCreator::Params params;
  params.api_version = std::nullopt;
  auto creator = std::make_unique<vulkan::VulkanDeviceCreator>(params);
  auto *device = static_cast<vulkan::VulkanDevice *>(creator->device());

  constexpr int kThreadCount = 4;
  constexpr int kSubmitCount = 128;
  std::atomic<int> ready{0};
  std::atomic<bool> start{false};
  std::atomic<bool> succeeded{true};
  std::vector<std::thread> threads;
  threads.reserve(kThreadCount);

  for (int thread_index = 0; thread_index < kThreadCount; ++thread_index) {
    threads.emplace_back([&, thread_index] {
      ready.fetch_add(1, std::memory_order_release);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }

      Stream *stream = thread_index % 2 == 0 ? device->get_compute_stream()
                                             : device->get_graphics_stream();
      for (int submit_index = 0; submit_index < kSubmitCount; ++submit_index) {
        auto [cmdlist, result] = stream->new_command_list_unique();
        if (result != RhiResult::success || !cmdlist) {
          succeeded.store(false, std::memory_order_relaxed);
          return;
        }
        if (!stream->submit(cmdlist.get())) {
          succeeded.store(false, std::memory_order_relaxed);
          return;
        }
      }
      stream->command_sync();
    });
  }

  while (ready.load(std::memory_order_acquire) != kThreadCount) {
    std::this_thread::yield();
  }
  start.store(true, std::memory_order_release);
  for (auto &thread : threads) {
    thread.join();
  }

  EXPECT_TRUE(succeeded.load(std::memory_order_relaxed));
  device->wait_idle();
}

TEST(VulkanDeviceTest, ConcurrentDescriptorAndRenderPassCreation) {
  if (!vulkan::is_vulkan_api_available()) {
    return;
  }

  vulkan::VulkanDeviceCreator::Params params;
  params.api_version = std::nullopt;
  auto creator = std::make_unique<vulkan::VulkanDeviceCreator>(params);
  auto *device = static_cast<vulkan::VulkanDevice *>(creator->device());

  Device::AllocParams alloc_params;
  alloc_params.size = sizeof(uint32_t);
  alloc_params.usage = AllocUsage::Storage;
  DeviceAllocation allocation;
  ASSERT_EQ(device->allocate_memory(alloc_params, &allocation),
            RhiResult::success);
  DeviceAllocationGuard allocation_guard(allocation);

  constexpr int kThreadCount = 4;
  constexpr int kDescriptorSetsPerThread = 32;
  device->set_descriptor_set_cache_enabled(false);

  vulkan::VulkanRenderPassDesc renderpass_desc;
  renderpass_desc.color_attachments = {{VK_FORMAT_R8G8B8A8_UNORM, true}};
  std::array<vkapi::IVkSampler, kThreadCount> samplers;
  std::array<vkapi::IVkRenderPass, kThreadCount> renderpasses;
  std::atomic<int> ready{0};
  std::atomic<bool> start{false};
  std::atomic<bool> succeeded{true};
  std::vector<std::thread> threads;
  threads.reserve(kThreadCount);

  for (int thread_index = 0; thread_index < kThreadCount; ++thread_index) {
    threads.emplace_back([&, thread_index] {
      ready.fetch_add(1, std::memory_order_release);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (int set_index = 0; set_index < kDescriptorSetsPerThread;
           ++set_index) {
        vulkan::VulkanResourceSet set(device);
        set.rw_buffer(0, allocation);
        auto [status, descriptor_set] = set.finalize();
        if (status != RhiResult::success || !descriptor_set) {
          succeeded.store(false, std::memory_order_relaxed);
          return;
        }
      }
      samplers[thread_index] = device->get_default_sampler();
      renderpasses[thread_index] = device->get_renderpass(renderpass_desc);
      if (!samplers[thread_index] || !renderpasses[thread_index]) {
        succeeded.store(false, std::memory_order_relaxed);
      }
    });
  }

  while (ready.load(std::memory_order_acquire) != kThreadCount) {
    std::this_thread::yield();
  }
  start.store(true, std::memory_order_release);
  for (auto &thread : threads) {
    thread.join();
  }

  EXPECT_TRUE(succeeded.load(std::memory_order_relaxed));
  for (int i = 1; i < kThreadCount; ++i) {
    EXPECT_EQ(samplers[i], samplers[0]);
    EXPECT_EQ(renderpasses[i], renderpasses[0]);
  }
  device->wait_idle();
}

TEST(VulkanSurfaceResultTest, ClassifiesRecoverableAndFatalResults) {
  using vulkan::VulkanSurfaceResult;
  using vulkan::classify_vulkan_surface_result;

  EXPECT_EQ(classify_vulkan_surface_result(VK_SUCCESS),
            VulkanSurfaceResult::kSuccess);
  EXPECT_EQ(classify_vulkan_surface_result(VK_SUBOPTIMAL_KHR),
            VulkanSurfaceResult::kSuboptimal);
  EXPECT_EQ(classify_vulkan_surface_result(VK_ERROR_OUT_OF_DATE_KHR),
            VulkanSurfaceResult::kOutOfDate);
  EXPECT_EQ(classify_vulkan_surface_result(VK_ERROR_DEVICE_LOST),
            VulkanSurfaceResult::kDeviceLost);
  EXPECT_EQ(classify_vulkan_surface_result(VK_ERROR_SURFACE_LOST_KHR),
            VulkanSurfaceResult::kError);
}

TEST(DeviceTest, ViewDevAllocAsNdarray) {
  // Otherwise will segfault on macOS VM,
  // where Vulkan is installed but no devices are present
  if (!vulkan::is_vulkan_api_available()) {
    return;
  }

  // Create Taichi Device for computation
  lang::vulkan::VulkanDeviceCreator::Params evd_params;
  evd_params.api_version = std::nullopt;
  auto embedded_device =
      std::make_unique<taichi::lang::vulkan::VulkanDeviceCreator>(evd_params);
  taichi::lang::vulkan::VulkanDevice *device_ =
      static_cast<taichi::lang::vulkan::VulkanDevice *>(
          embedded_device->device());

  aot_test_utils::view_devalloc_as_ndarray(device_);
}

TEST(VulkanPipelineCacheTest, SnapshotRoundTrip) {
  if (!vulkan::is_vulkan_api_available()) {
    return;
  }

  vulkan::VulkanDeviceCreator::Params params;
  params.api_version = std::nullopt;
  auto creator = std::make_unique<vulkan::VulkanDeviceCreator>(params);
  auto *device = static_cast<vulkan::VulkanDevice *>(creator->device());

  auto [cache, result] = device->create_pipeline_cache_unique();
  ASSERT_EQ(result, RhiResult::success);
  ASSERT_NE(cache, nullptr);

  auto *cache_data = static_cast<uint8_t *>(cache->data());
  const size_t cache_size = cache->size();
  ASSERT_NE(cache_data, nullptr);
  ASSERT_GT(cache_size, 0);
  std::vector<uint8_t> blob(cache_data, cache_data + cache_size);

  auto [restored, restored_result] =
      device->create_pipeline_cache_unique(blob.size(), blob.data());
  ASSERT_EQ(restored_result, RhiResult::success);
  ASSERT_NE(restored, nullptr);
  EXPECT_NE(restored->data(), nullptr);
  EXPECT_GT(restored->size(), 0);
}

TEST(VulkanPipelineCacheTest, ConcurrentSnapshots) {
  if (!vulkan::is_vulkan_api_available()) {
    return;
  }

  vulkan::VulkanDeviceCreator::Params params;
  params.api_version = std::nullopt;
  auto creator = std::make_unique<vulkan::VulkanDeviceCreator>(params);
  auto *device = static_cast<vulkan::VulkanDevice *>(creator->device());
  auto [cache, result] = device->create_pipeline_cache_unique();
  ASSERT_EQ(result, RhiResult::success);
  ASSERT_NE(cache, nullptr);

  constexpr int kThreadCount = 4;
  constexpr int kSnapshotsPerThread = 32;
  std::atomic<int> ready{0};
  std::atomic<bool> start{false};
  std::atomic<bool> succeeded{true};
  std::vector<std::thread> threads;
  threads.reserve(kThreadCount);
  for (int thread_index = 0; thread_index < kThreadCount; ++thread_index) {
    threads.emplace_back([&] {
      ready.fetch_add(1, std::memory_order_release);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (int snapshot_index = 0; snapshot_index < kSnapshotsPerThread;
           ++snapshot_index) {
        auto *data = cache->data();
        if (data == nullptr || cache->size() == 0) {
          succeeded.store(false, std::memory_order_relaxed);
          return;
        }
      }
    });
  }

  while (ready.load(std::memory_order_acquire) != kThreadCount) {
    std::this_thread::yield();
  }
  start.store(true, std::memory_order_release);
  for (auto &thread : threads) {
    thread.join();
  }

  EXPECT_TRUE(succeeded.load(std::memory_order_relaxed));
}

TEST(VulkanProfilerTest, CommandListScopesKeepTheirOwnQueryPools) {
  if (!vulkan::is_vulkan_api_available()) {
    return;
  }

  vulkan::VulkanDeviceCreator::Params params;
  params.api_version = std::nullopt;
  auto creator = std::make_unique<vulkan::VulkanDeviceCreator>(params);
  auto *device = static_cast<vulkan::VulkanDevice *>(creator->device());
  auto *stream = device->get_compute_stream();
  auto [first, first_result] = stream->new_command_list_unique();
  auto [second, second_result] = stream->new_command_list_unique();
  ASSERT_EQ(first_result, RhiResult::success);
  ASSERT_EQ(second_result, RhiResult::success);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);

  first->begin_profiler_scope("first_scope");
  second->begin_profiler_scope("second_scope");
  first->end_profiler_scope();
  second->end_profiler_scope();
  ASSERT_TRUE(stream->submit(first.get()));
  ASSERT_TRUE(stream->submit(second.get()));
  stream->command_sync();

  // command_sync() drains completed profiler samplers as part of synchronizing
  // the stream, so the sampled records are ready to consume here.
  auto records = device->profiler_flush_sampled_time();
  ASSERT_EQ(records.size(), 2u);
  for (const auto &[name, duration_ms] : records) {
    EXPECT_TRUE(name == "first_scope" || name == "second_scope");
    EXPECT_GE(duration_ms, 0.0);
    EXPECT_LT(duration_ms, 10000.0);
  }
}
