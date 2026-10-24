// Dedicated Windows executable: link the real Vulkan backend objects with
// the matching CRT. Never add this allocator override to a production library
// or the general test executable.
#include <cstdlib>
#include <malloc.h>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>
#include "gtest/gtest.h"
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/rhi/vulkan/vulkan_device_creator.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"

namespace {
thread_local bool deny_allocations = false;
thread_local long allocations_until_failure = -1;
void allocation_checkpoint() {
  if (deny_allocations || allocations_until_failure == 0)
    throw std::bad_alloc();
  if (allocations_until_failure > 0)
    --allocations_until_failure;
}
}  // namespace

void *operator new(std::size_t size) {
  allocation_checkpoint();
  if (void *p = std::malloc(size ? size : 1))
    return p;
  throw std::bad_alloc();
}
void *operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void *p) noexcept { std::free(p); }
void operator delete[](void *p) noexcept { std::free(p); }
void operator delete(void *p, std::size_t) noexcept { std::free(p); }
void operator delete[](void *p, std::size_t) noexcept { std::free(p); }
void *operator new(std::size_t size, std::align_val_t alignment) {
  allocation_checkpoint();
  if (void *p = _aligned_malloc(size ? size : 1, static_cast<std::size_t>(alignment)))
    return p;
  throw std::bad_alloc();
}
void *operator new[](std::size_t size, std::align_val_t alignment) {
  return ::operator new(size, alignment);
}
void operator delete(void *p, std::align_val_t) noexcept { _aligned_free(p); }
void operator delete[](void *p, std::align_val_t) noexcept { _aligned_free(p); }
void operator delete(void *p, std::size_t, std::align_val_t) noexcept { _aligned_free(p); }
void operator delete[](void *p, std::size_t, std::align_val_t) noexcept { _aligned_free(p); }

namespace taichi::lang::vulkan {
namespace {
PFN_vkQueueSubmit real_submit;
PFN_vkGetFenceStatus real_query;
PFN_vkDestroySemaphore real_destroy_semaphore;
bool reject_native = false, arm_after_success = false;
int query_mode = 0, queries = 0, submits = 0, destroyed_semaphores = 0;
VKAPI_ATTR VkResult VKAPI_CALL submit_hook(
    VkQueue queue, uint32_t count, const VkSubmitInfo *info, VkFence fence) {
  ++submits;
  if (reject_native)
    return VK_ERROR_OUT_OF_HOST_MEMORY;
  const auto result = real_submit(queue, count, info, fence);
  if (result == VK_SUCCESS && arm_after_success)
    deny_allocations = true;
  return result;
}
VKAPI_ATTR VkResult VKAPI_CALL query_hook(VkDevice device, VkFence fence) {
  if (query_mode == 1)
    return VK_NOT_READY;
  if (query_mode == 2)
    return ++queries == 2 ? VK_ERROR_OUT_OF_HOST_MEMORY : VK_NOT_READY;
  return real_query(device, fence);
}
VKAPI_ATTR void VKAPI_CALL destroy_semaphore_hook(
    VkDevice device, VkSemaphore semaphore, const VkAllocationCallbacks *callbacks) {
  ++destroyed_semaphores;
  real_destroy_semaphore(device, semaphore, callbacks);
}

class VulkanSubmitClosureProbe : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!is_vulkan_api_available())
      GTEST_SKIP();
    creator = std::make_unique<VulkanDeviceCreator>(VulkanDeviceCreator::Params{});
    device = creator->device();
    stream = static_cast<VulkanStream *>(device->get_compute_stream());
    api = &const_cast<VolkDeviceTable &>(device->vk_device().functions());
    real_submit = api->vkQueueSubmit;
    real_query = api->vkGetFenceStatus;
    real_destroy_semaphore = api->vkDestroySemaphore;
    api->vkQueueSubmit = submit_hook;
    api->vkGetFenceStatus = query_hook;
    api->vkDestroySemaphore = destroy_semaphore_hook;
    reject_native = arm_after_success = false;
    query_mode = queries = submits = destroyed_semaphores = 0;
  }
  void TearDown() override {
    deny_allocations = false;
    allocations_until_failure = -1;
    if (!api)
      return;
    api->vkQueueSubmit = real_submit;
    api->vkGetFenceStatus = real_query;
    api->vkDestroySemaphore = real_destroy_semaphore;
    stream->command_sync();
  }
  std::unique_ptr<CommandList> profiled_commands() {
    auto [commands, result] = stream->new_command_list_unique();
    if (result != RhiResult::success || !commands)
      throw std::runtime_error("probe command allocation failed");
    commands->begin_profiler_scope("submit_closure_probe_has_a_nonempty_sampler");
    commands->memory_barrier();
    commands->end_profiler_scope();
    return std::move(commands);
  }
  std::unique_ptr<VulkanDeviceCreator> creator;
  VulkanDevice *device = nullptr;
  VulkanStream *stream = nullptr;
  VolkDeviceTable *api = nullptr;
};

TEST_F(VulkanSubmitClosureProbe, NoCppAllocationAfterSuccessfulImmediateOrBatchEnqueue) {
  for (bool batch : {false, true}) {
    auto commands = profiled_commands();
    if (batch) {
      stream->begin_submission_batch();
      ASSERT_TRUE(stream->submit(commands.get()));
    }
    StreamSemaphore completion;
    bool host_oom = false;
    arm_after_success = true;
    try {
      completion = batch ? stream->end_submission_batch() : stream->submit(commands.get());
    } catch (const std::bad_alloc &) {
      host_oom = true;
    } catch (...) {
      deny_allocations = false;
      arm_after_success = false;
      throw;
    }
    deny_allocations = false;  // Disable before gtest itself can allocate.
    arm_after_success = false;
    ASSERT_FALSE(host_oom);
    ASSERT_TRUE(completion);
    EXPECT_TRUE(stream->is_last_submission(completion));
    EXPECT_TRUE(completion->wait());
    stream->command_sync();
    EXPECT_EQ(device->profiler_get_sampler_count(), 0u);
    device->profiler_flush_sampled_time();
  }
}

TEST_F(VulkanSubmitClosureProbe, PreEnqueueOomDiscardsImmediateButRetainsPendingBatchBudget) {
  auto immediate = profiled_commands();
  allocations_until_failure = 0;
  bool failed = false;
  try { stream->submit(immediate.get()); }
  catch (const std::bad_alloc &) { failed = true; }
  catch (...) { allocations_until_failure = -1; throw; }
  allocations_until_failure = -1;
  ASSERT_TRUE(failed);
  EXPECT_EQ(submits, 0);
  EXPECT_EQ(device->profiler_get_sampler_count(), 0u);
  auto batched = profiled_commands();
  stream->begin_submission_batch();
  auto pending = stream->submit(batched.get());
  allocations_until_failure = 0;
  failed = false;
  try { stream->end_submission_batch(); }
  catch (const std::bad_alloc &) { failed = true; }
  catch (...) { allocations_until_failure = -1; throw; }
  allocations_until_failure = -1;
  ASSERT_TRUE(failed);
  EXPECT_EQ(submits, 0);
  EXPECT_EQ(device->profiler_get_sampler_count(), 1u);
  EXPECT_EQ(stream->flush_submission_batch(), pending);
  EXPECT_TRUE(pending->wait());
  stream->command_sync();
  EXPECT_EQ(device->profiler_get_sampler_count(), 0u);
}

TEST_F(VulkanSubmitClosureProbe, NativeRejectionIsTypedAndReturnsProfilerReservations) {
  for (bool batch : {false, true}) {
    auto commands = profiled_commands();
    if (batch) {
      stream->begin_submission_batch();
      stream->submit(commands.get());
    }
    reject_native = true;
    try {
      if (batch) stream->end_submission_batch();
      else stream->submit(commands.get());
      FAIL() << "native rejection was not reported";
    } catch (const BackendRuntimeError &error) {
      EXPECT_EQ(error.operation(), "vkQueueSubmit");
      EXPECT_EQ(error.backend_code(), VK_ERROR_OUT_OF_HOST_MEMORY);
    }
    reject_native = false;
    EXPECT_EQ(device->profiler_get_sampler_count(), 0u);
    EXPECT_EQ(stream->debug_in_flight_command_buffer_count(), 0u);
  }
}

TEST_F(VulkanSubmitClosureProbe, LaterFenceQueryFailureDoesNotDropEarlierPendingOwner) {
  query_mode = 1;
  for (int i = 0; i < 2; ++i) {
    auto commands = profiled_commands();
    ASSERT_TRUE(stream->submit(commands.get()));
  }
  const int before = destroyed_semaphores;
  allocations_until_failure = 0;
  bool host_oom = false;
  try { stream->debug_in_flight_command_buffer_count(); }
  catch (const std::bad_alloc &) { host_oom = true; }
  catch (...) { allocations_until_failure = -1; throw; }
  allocations_until_failure = -1;
  ASSERT_TRUE(host_oom);
  EXPECT_EQ(destroyed_semaphores, before);
  EXPECT_EQ(stream->debug_in_flight_command_buffer_count(), 2u);
  query_mode = 2;
  EXPECT_THROW(stream->debug_in_flight_command_buffer_count(), BackendRuntimeError);
  EXPECT_EQ(destroyed_semaphores, before);
  query_mode = 1;
  EXPECT_EQ(stream->debug_in_flight_command_buffer_count(), 2u);
  query_mode = 0;
  stream->command_sync();
  EXPECT_EQ(device->profiler_get_sampler_count(), 0u);
}
}  // namespace
}  // namespace taichi::lang::vulkan
