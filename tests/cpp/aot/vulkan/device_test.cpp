#include "gtest/gtest.h"
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/rhi/vulkan/vulkan_device_creator.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"
#include "tests/cpp/aot/gfx_utils.h"

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
