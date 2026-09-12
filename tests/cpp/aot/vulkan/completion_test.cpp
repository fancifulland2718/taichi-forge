#include "gtest/gtest.h"
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/rhi/vulkan/vulkan_device_creator.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"

#include <thread>

namespace taichi::lang::vulkan {
namespace {

StreamSemaphore submit_empty(Stream *stream) {
  auto [commands, result] = stream->new_command_list_unique();
  if (result != RhiResult::success || !commands) {
    return nullptr;
  }
  return stream->submit(commands.get());
}

TEST(VulkanCompletionTest, QueueTailIncludesOtherThreadsAndExternalProviders) {
  if (!is_vulkan_api_available()) {
    GTEST_SKIP() << "Vulkan is unavailable";
  }
  VulkanDeviceCreator::Params params;
  VulkanDeviceCreator creator(params);
  auto *device = creator.device();
  auto *stream = device->get_compute_stream();
  auto first = submit_empty(stream);
  ASSERT_TRUE(first);
  EXPECT_TRUE(stream->is_last_submission(first));

  StreamSemaphore second;
  std::thread producer([&] {
    second = submit_empty(device->get_compute_stream());
  });
  producer.join();
  ASSERT_TRUE(second);
  EXPECT_FALSE(stream->is_last_submission(first));
  EXPECT_TRUE(stream->is_last_submission(second));
  EXPECT_TRUE(second->wait());

  {
    auto provider_lock = device->acquire_external_compute_queue_lock();
  }
  EXPECT_FALSE(stream->is_last_submission(second));
  auto third = submit_empty(stream);
  ASSERT_TRUE(third);
  EXPECT_TRUE(stream->is_last_submission(third));
  EXPECT_TRUE(third->wait());
}

TEST(VulkanCompletionTest, BatchTokensBecomeCurrentOnlyAfterPublication) {
  if (!is_vulkan_api_available()) {
    GTEST_SKIP() << "Vulkan is unavailable";
  }
  VulkanDeviceCreator::Params params;
  VulkanDeviceCreator creator(params);
  auto *stream = creator.device()->get_compute_stream();
  stream->begin_submission_batch();
  auto first = submit_empty(stream);
  auto second = submit_empty(stream);
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  EXPECT_FALSE(stream->is_last_submission(first));
  EXPECT_FALSE(stream->is_last_submission(second));
  auto batch = stream->end_submission_batch();
  ASSERT_TRUE(batch);
  // Different binary semaphore tokens share the fence for the whole batch.
  EXPECT_TRUE(stream->is_last_submission(first));
  EXPECT_TRUE(stream->is_last_submission(second));
  EXPECT_TRUE(stream->is_last_submission(batch));
  EXPECT_TRUE(batch->wait());
}

TEST(VulkanCompletionTest, GraphicsAndComputeUseTheirActualQueueIdentity) {
  if (!is_vulkan_api_available()) {
    GTEST_SKIP() << "Vulkan is unavailable";
  }
  VulkanDeviceCreator::Params params;
  VulkanDeviceCreator creator(params);
  auto *device = creator.device();
  auto *compute = device->get_compute_stream();
  auto *graphics = device->get_graphics_stream();
  auto before = submit_empty(compute);
  auto draw = submit_empty(graphics);
  ASSERT_TRUE(before);
  ASSERT_TRUE(draw);
  const bool aliased = device->compute_queue() == device->graphics_queue();
  EXPECT_EQ(compute->is_last_submission(before), !aliased);
  EXPECT_EQ(compute->is_last_submission(draw), aliased);
  EXPECT_TRUE(graphics->is_last_submission(draw));
  EXPECT_TRUE(before->wait());
  EXPECT_TRUE(draw->wait());
}

}  // namespace
}  // namespace taichi::lang::vulkan
