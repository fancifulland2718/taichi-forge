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

TEST(VulkanCompletionTest, ReusedDependencyCommandsRetainPerSubmitCompletion) {
  if (!is_vulkan_api_available()) {
    GTEST_SKIP() << "Vulkan is unavailable";
  }
  VulkanDeviceCreator::Params params;
  VulkanDeviceCreator creator(params);
  auto *device = creator.device();
  auto *stream = device->get_compute_stream();
  auto producer = submit_empty(device->get_graphics_stream());
  ASSERT_TRUE(producer);
  auto before = device->queue_submission_snapshot();
  auto dependency = stream->submit_dependency({producer});
  ASSERT_TRUE(dependency);
  producer.reset();  // The submission, not the caller, owns the wait semaphore.
  auto after = device->queue_submission_snapshot();
  EXPECT_EQ(after.queue_submit_calls - before.queue_submit_calls, 1);
  EXPECT_EQ(after.submitted_command_buffers - before.submitted_command_buffers, 1);
  EXPECT_TRUE(stream->is_last_submission(dependency));

  stream->begin_submission_batch();
  auto pending = stream->submit_dependency({dependency});
  ASSERT_TRUE(pending);
  EXPECT_FALSE(stream->is_last_submission(pending));
  dependency.reset();
  before = device->queue_submission_snapshot();
  auto batch = stream->end_submission_batch();
  ASSERT_TRUE(batch);
  after = device->queue_submission_snapshot();
  EXPECT_EQ(after.queue_submit_calls - before.queue_submit_calls, 1);
  EXPECT_EQ(after.submitted_command_buffers - before.submitted_command_buffers, 1);
  EXPECT_EQ(after.batched_command_buffers - before.batched_command_buffers, 1);
  EXPECT_TRUE(stream->is_last_submission(pending));
  EXPECT_TRUE(batch->wait());

  stream->begin_submission_batch();
  auto next_dependency = stream->submit_dependency({batch});
  auto command = submit_empty(stream);
  ASSERT_TRUE(command);
  before = device->queue_submission_snapshot();
  auto mixed = stream->end_submission_batch();
  ASSERT_TRUE(mixed);
  after = device->queue_submission_snapshot();
  EXPECT_EQ(after.queue_submit_calls - before.queue_submit_calls, 1);
  EXPECT_EQ(after.submitted_command_buffers - before.submitted_command_buffers, 2);
  EXPECT_EQ(after.batched_command_buffers - before.batched_command_buffers, 2);
  EXPECT_TRUE(stream->is_last_submission(next_dependency));
  EXPECT_TRUE(mixed->wait());

  // Signal identity is per submission, even when the dependency command
  // buffer is shared by several pending submissions in the same batch.
  stream->begin_submission_batch();
  auto prefix = stream->submit_dependency({mixed});
  auto repeated = stream->submit_dependency({prefix});
  auto [commands, result] = stream->new_command_list_unique();
  ASSERT_EQ(result, RhiResult::success);
  auto consumer = stream->submit(commands.get(), {repeated});
  ASSERT_TRUE(consumer);
  auto ordered = stream->end_submission_batch();
  ASSERT_TRUE(ordered);
  EXPECT_TRUE(ordered->wait());
  stream->command_sync();
}

}  // namespace
}  // namespace taichi::lang::vulkan
