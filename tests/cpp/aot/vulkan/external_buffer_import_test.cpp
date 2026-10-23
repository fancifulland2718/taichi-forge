// Windows RHI qualification, not an execution-domain or Flora capability gate.
#if defined(TI_WITH_VULKAN) && defined(_WIN64)
#include "gtest/gtest.h"
#include "taichi/rhi/vulkan/vulkan_device_creator.h"
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/rhi/vulkan/vulkan_external_buffer.h"
#include "taichi/rhi/vulkan/vulkan_execution_queue.h"
#include "taichi/rhi/vulkan/vulkan_external_access.h"
#include "taichi/runtime/vulkan_execution_domain.h"
#include "taichi/runtime/fixed_execution_vulkan.h"
#include "taichi/runtime/gfx/aot_module_loader_impl.h"
#include "taichi/runtime/completion_wait_set.h"
#include "taichi/program/launch_context_builder.h"
#include "taichi/program/ndarray.h"
#include "taichi/program/runtime_fault.h"
#include "tests/cpp/aot/vulkan/timeline_gate.h"
#include <array>
#include <chrono>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <cstdlib>
#include <future>
#include <thread>

namespace {
using taichi::lang::vulkan::VulkanCommandList;
using taichi::lang::vulkan::VulkanDevice;
using taichi::lang::vulkan::VulkanDeviceCreator;
using taichi::lang::vulkan::VulkanExecutionQueue;
using taichi::lang::vulkan::VulkanExternalBuffer;
using taichi::lang::vulkan::VulkanExternalBufferDescription;
using taichi::lang::vulkan::VulkanExternalBufferHandles;
using taichi::lang::vulkan::VulkanExternalBufferOwner;
using taichi::lang::vulkan::VulkanExternalAccess;
using taichi::lang::vulkan::VulkanStream;
using taichi::lang::vulkan::VulkanStreamSemaphoreObject;
constexpr VkDeviceSize bytes = 4 * sizeof(uint32_t);
constexpr auto memory_handle_type =
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
constexpr auto semaphore_handle_type =
    VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;

VulkanDeviceCreator::Params parameters() {
  VulkanDeviceCreator::Params result;
  result.enable_validation_layer = true;
  if (const auto index = std::getenv("TI_TEST_VULKAN_DEVICE_INDEX"))
    result.device_index = std::stoul(index);
  return result;
}

void checked(VkResult result) {
  if (result != VK_SUCCESS)
    throw std::runtime_error("Vulkan probe result " + std::to_string(result));
}

struct NativeHandle {
  HANDLE handle{};
  ~NativeHandle() {
    if (handle)
      CloseHandle(handle);
  }
};

struct HostGate {
  explicit HostGate(vkapi::DeviceHandle device) : device(device) {
    VkEventCreateInfo info{VK_STRUCTURE_TYPE_EVENT_CREATE_INFO};
    checked(device.functions().vkCreateEvent(device, &info, nullptr, &event));
  }
  ~HostGate() {
    device.functions().vkSetEvent(device, event);
    if (pending_fence)
      device.functions().vkWaitForFences(device, 1, &pending_fence, VK_TRUE,
                                         UINT64_MAX);
    device.functions().vkDestroyEvent(device, event, nullptr);
  }
  void release() {
    checked(device.functions().vkSetEvent(device, event));
  }
  vkapi::DeviceHandle device;
  VkEvent event{};
  VkFence pending_fence{};
};

struct Buffer {
  vkapi::DeviceHandle device;
  VkBuffer buffer{};
  VkDeviceMemory memory{};
  VkDeviceSize allocation_size{};
  uint32_t memory_type{};
  VkBufferUsageFlags usage{};
  VkMemoryAllocateFlags allocation_flags{};
  ~Buffer() {
    if (buffer)
      device.functions().vkDestroyBuffer(device, buffer, nullptr);
    if (memory)
      device.functions().vkFreeMemory(device, memory, nullptr);
  }
};

std::unique_ptr<Buffer> make_buffer(VulkanDevice &device,
                                    bool external,
                                    const Buffer *producer = nullptr,
                                    HANDLE imported = nullptr,
                                    bool padded = false,
                                    bool address = false) {
  auto result = std::make_unique<Buffer>();
  result->device = device.vk_device();
  const auto &api = result->device.functions();
  VkExternalMemoryBufferCreateInfo sharing{
      VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
  sharing.handleTypes = memory_handle_type;
  VkBufferCreateInfo create{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  create.pNext = external ? &sharing : nullptr;
  create.size = bytes;
  create.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  if (address)
    create.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  result->usage = create.usage;
  result->allocation_flags =
      address ? VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT : 0;
  create.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  checked(
      api.vkCreateBuffer(result->device, &create, nullptr, &result->buffer));
  VkMemoryRequirements requirements{};
  api.vkGetBufferMemoryRequirements(result->device, result->buffer,
                                    &requirements);
  VkPhysicalDeviceMemoryProperties properties{};
  result->device.instance_functions().vkGetPhysicalDeviceMemoryProperties(
      device.vk_physical_device(), &properties);
  result->allocation_size = requirements.size;
  if (padded)
    result->allocation_size += requirements.alignment;
  if (producer) {
    result->memory_type = producer->memory_type;
    result->allocation_size = producer->allocation_size;
    if (!(requirements.memoryTypeBits & (1u << result->memory_type)) ||
        requirements.size > result->allocation_size)
      throw std::runtime_error("Import requirements differ from the producer");
  } else {
    const auto required = external ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                                   : VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    bool found = false;
    for (uint32_t i = 0; i < properties.memoryTypeCount; ++i)
      if ((requirements.memoryTypeBits & (1u << i)) &&
          (properties.memoryTypes[i].propertyFlags & required) == required) {
        result->memory_type = i;
        found = true;
        break;
      }
    if (!found)
      throw std::runtime_error("No required memory type");
  }
  VkMemoryDedicatedAllocateInfo dedicated{
      VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
  dedicated.buffer = result->buffer;
  VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
  flags.flags = result->allocation_flags;
  dedicated.pNext = &flags;
  VkExportMemoryAllocateInfo export_info{
      VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
  export_info.handleTypes = memory_handle_type;
  export_info.pNext = &dedicated;
  VkImportMemoryWin32HandleInfoKHR import_info{
      VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR};
  import_info.handleType = memory_handle_type;
  import_info.handle = imported;
  import_info.pNext = &dedicated;
  VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  allocation.allocationSize = result->allocation_size;
  allocation.memoryTypeIndex = result->memory_type;
  allocation.pNext = imported   ? static_cast<void *>(&import_info)
                     : external ? static_cast<void *>(&export_info)
                                : nullptr;
  checked(api.vkAllocateMemory(result->device, &allocation, nullptr,
                               &result->memory));
  checked(api.vkBindBufferMemory(result->device, result->buffer, result->memory,
                                 0));
  return result;
}

struct Endpoint {
  explicit Endpoint(VulkanDevice &source)
      : device(source), native(source.vk_device()) {
    auto &api = native.functions();
    api.vkGetDeviceQueue(native, source.compute_queue_family_index(), 0,
                         &queue);
    VkCommandPoolCreateInfo pool_info{
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = source.compute_queue_family_index();
    checked(api.vkCreateCommandPool(native, &pool_info, nullptr, &pool));
    VkCommandBufferAllocateInfo allocate{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocate.commandPool = pool;
    allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate.commandBufferCount = 1;
    checked(api.vkAllocateCommandBuffers(native, &allocate, &commands));
    VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    checked(api.vkCreateFence(native, &fence_info, nullptr, &fence));
    readback = make_buffer(source, false);
  }
  ~Endpoint() {
    if (fence)
      native.functions().vkDestroyFence(native, fence, nullptr);
    if (pool)
      native.functions().vkDestroyCommandPool(native, pool, nullptr);
  }
  void begin() {
    checked(native.functions().vkResetFences(native, 1, &fence));
    checked(native.functions().vkResetCommandBuffer(commands, 0));
    VkCommandBufferBeginInfo begin_info{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    checked(native.functions().vkBeginCommandBuffer(commands, &begin_info));
  }
  void barrier(Buffer &buffer,
               VkAccessFlags src,
               VkAccessFlags dst,
               uint32_t from = VK_QUEUE_FAMILY_IGNORED,
               uint32_t to = VK_QUEUE_FAMILY_IGNORED) {
    VkBufferMemoryBarrier b{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    b.srcAccessMask = src;
    b.dstAccessMask = dst;
    b.srcQueueFamilyIndex = from;
    b.dstQueueFamilyIndex = to;
    b.buffer = buffer.buffer;
    b.size = bytes;
    const auto dst_stage = dst & VK_ACCESS_HOST_READ_BIT
                               ? VK_PIPELINE_STAGE_HOST_BIT
                               : VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    native.functions().vkCmdPipelineBarrier(
        commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, dst_stage, 0, 0, nullptr,
        1, &b, 0, nullptr);
  }
  void acquire(Buffer &buffer) {
    barrier(buffer, 0,
            VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_QUEUE_FAMILY_EXTERNAL, device.compute_queue_family_index());
  }
  void release(Buffer &buffer) {
    barrier(buffer, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
            0, device.compute_queue_family_index(), VK_QUEUE_FAMILY_EXTERNAL);
  }
  void copy(Buffer &buffer) {
    VkBufferCopy copy{0, 0, bytes};
    native.functions().vkCmdCopyBuffer(commands, buffer.buffer,
                                       readback->buffer, 1, &copy);
    barrier(*readback, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
  }
  void update(Buffer &buffer, const std::array<uint32_t, 4> &values) {
    native.functions().vkCmdUpdateBuffer(commands, buffer.buffer, 0, bytes,
                                         values.data());
  }
  void submit(VkSemaphore wait = {}, VkSemaphore signal = {}) {
    checked(native.functions().vkEndCommandBuffer(commands));
    VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo submit_info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &commands;
    submit_info.waitSemaphoreCount = wait ? 1 : 0;
    submit_info.pWaitSemaphores = &wait;
    submit_info.pWaitDstStageMask = &stage;
    submit_info.signalSemaphoreCount = signal ? 1 : 0;
    submit_info.pSignalSemaphores = &signal;
    checked(native.functions().vkQueueSubmit(queue, 1, &submit_info, fence));
  }
  void wait() {
    checked(native.functions().vkWaitForFences(native, 1, &fence, VK_TRUE,
                                               2'000'000'000));
  }
  void expect(const std::array<uint32_t, 4> &values) {
    void *pointer = nullptr;
    checked(native.functions().vkMapMemory(native, readback->memory, 0, bytes,
                                           0, &pointer));
    std::array<uint32_t, 4> observed{};
    std::memcpy(observed.data(), pointer, bytes);
    native.functions().vkUnmapMemory(native, readback->memory);
    EXPECT_EQ(observed, values);
  }
  VulkanDevice &device;
  vkapi::DeviceHandle native;
  VkQueue queue{};
  VkCommandPool pool{};
  VkCommandBuffer commands{};
  VkFence fence{};
  std::unique_ptr<Buffer> readback;
};

vkapi::IVkSemaphore semaphore(const vkapi::DeviceHandle &device,
                              bool exported,
                              HANDLE imported = nullptr) {
  VkExportSemaphoreCreateInfo export_info{
      VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO};
  export_info.handleTypes = semaphore_handle_type;
  auto result =
      vkapi::create_semaphore(device, 0, exported ? &export_info : nullptr);
  if (!result)
    throw std::runtime_error("Cannot create semaphore");
  if (imported) {
    VkImportSemaphoreWin32HandleInfoKHR info{
        VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR};
    info.semaphore = result->semaphore;
    info.handleType = semaphore_handle_type;
    info.handle = imported;
    checked(device.functions().vkImportSemaphoreWin32HandleKHR(device, &info));
  }
  return result;
}

TEST(VulkanWin32ExternalProbe, FourElementsAcrossTwoDevicesAndEightEpochs) {
  auto params = parameters();
  VulkanDeviceCreator producer_creator(params), consumer_creator(params);
  auto &p_device = *producer_creator.device();
  auto &c_device = *consumer_creator.device();
  ASSERT_NE(static_cast<VkDevice>(p_device.vk_device()),
            static_cast<VkDevice>(c_device.vk_device()));
  ASSERT_TRUE(p_device.vk_caps().external_memory &&
              p_device.vk_caps().external_semaphore);
  ASSERT_TRUE(c_device.vk_caps().external_memory &&
              c_device.vk_caps().external_semaphore);
  auto identity = [](VulkanDevice &device) {
    VkPhysicalDeviceIDProperties ids{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceProperties2 properties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    properties.pNext = &ids;
    device.vk_device().instance_functions().vkGetPhysicalDeviceProperties2(
        device.vk_physical_device(), &properties);
    return ids;
  };
  const auto p_id = identity(p_device), c_id = identity(c_device);
  ASSERT_EQ(std::memcmp(p_id.deviceUUID, c_id.deviceUUID, VK_UUID_SIZE), 0);
  ASSERT_EQ(std::memcmp(p_id.driverUUID, c_id.driverUUID, VK_UUID_SIZE), 0);
  auto source = make_buffer(p_device, true);
  std::unique_ptr<Buffer> imported;
  auto ready = semaphore(p_device.vk_device(), true);
  auto consumed = semaphore(p_device.vk_device(), true);
  vkapi::IVkSemaphore imported_ready, imported_consumed;
  {
    NativeHandle memory_handle, ready_handle, consumed_handle;
    VkMemoryGetWin32HandleInfoKHR memory_info{
        VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR};
    memory_info.memory = source->memory;
    memory_info.handleType = memory_handle_type;
    checked(p_device.vk_device().functions().vkGetMemoryWin32HandleKHR(
        p_device.vk_device(), &memory_info, &memory_handle.handle));
    VkSemaphoreGetWin32HandleInfoKHR sem_info{
        VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR};
    sem_info.handleType = semaphore_handle_type;
    sem_info.semaphore = ready->semaphore;
    checked(p_device.vk_device().functions().vkGetSemaphoreWin32HandleKHR(
        p_device.vk_device(), &sem_info, &ready_handle.handle));
    sem_info.semaphore = consumed->semaphore;
    checked(p_device.vk_device().functions().vkGetSemaphoreWin32HandleKHR(
        p_device.vk_device(), &sem_info, &consumed_handle.handle));
    imported = make_buffer(c_device, true, source.get(), memory_handle.handle);
    imported_ready =
        semaphore(c_device.vk_device(), false, ready_handle.handle);
    imported_consumed =
        semaphore(c_device.vk_device(), false, consumed_handle.handle);
  }  // Close all exported NT handles before either device uses the payload.
  Endpoint producer(p_device), consumer(c_device);
  for (uint32_t epoch = 0; epoch < 8; ++epoch) {
    const std::array<uint32_t, 4> before{
        {epoch, 0x80000001u + epoch, 17u, 0xffffff00u + epoch}};
    const std::array<uint32_t, 4> after{
        {epoch + 100u, 0x81234567u, 29u, 0xfffffe00u + epoch}};
    producer.begin();
    if (epoch)
      producer.barrier(*source, VK_ACCESS_TRANSFER_READ_BIT,
                       VK_ACCESS_TRANSFER_WRITE_BIT);
    producer.update(*source, before);
    producer.release(*source);
    producer.submit({}, ready->semaphore);
    consumer.begin();
    consumer.acquire(*imported);
    consumer.copy(*imported);
    consumer.barrier(*imported, VK_ACCESS_TRANSFER_READ_BIT,
                     VK_ACCESS_TRANSFER_WRITE_BIT);
    consumer.update(*imported, after);
    consumer.release(*imported);
    consumer.submit(imported_ready->semaphore, imported_consumed->semaphore);
    consumer.wait();
    consumer.expect(before);
    producer.wait();
    producer.begin();
    producer.acquire(*source);
    producer.copy(*source);
    producer.submit(consumed->semaphore);
    producer.wait();
    producer.expect(after);
  }
  std::cout
      << "opaque_win32_roundtrip_epochs=8 elements=4 matching_memory_type="
      << source->memory_type << " allocation_bytes=" << source->allocation_size
      << '\n';
}

struct Exported {
  explicit Exported(VulkanDevice &device,
                    bool padded = false,
                    bool address = false) {
    source = make_buffer(device, true, nullptr, nullptr, padded, address);
    ready = semaphore(device.vk_device(), true);
    consumed = semaphore(device.vk_device(), true);
    VkMemoryGetWin32HandleInfoKHR memory_info{
        VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR};
    memory_info.memory = source->memory;
    memory_info.handleType = memory_handle_type;
    checked(device.vk_device().functions().vkGetMemoryWin32HandleKHR(
        device.vk_device(), &memory_info, &memory_handle.handle));
    VkSemaphoreGetWin32HandleInfoKHR info{
        VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR};
    info.handleType = semaphore_handle_type;
    info.semaphore = ready->semaphore;
    checked(device.vk_device().functions().vkGetSemaphoreWin32HandleKHR(
        device.vk_device(), &info, &ready_handle.handle));
    info.semaphore = consumed->semaphore;
    checked(device.vk_device().functions().vkGetSemaphoreWin32HandleKHR(
        device.vk_device(), &info, &consumed_handle.handle));
    description.size = bytes;
    description.allocation_size = source->allocation_size;
    description.memory_type_index = source->memory_type;
    description.usage = source->usage;
    description.allocation_flags = source->allocation_flags;
    VkPhysicalDeviceIDProperties ids{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceProperties2 props{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props.pNext = &ids;
    device.vk_device().instance_functions().vkGetPhysicalDeviceProperties2(
        device.vk_physical_device(), &props);
    std::copy_n(ids.deviceUUID, VK_UUID_SIZE, description.device_uuid.begin());
    std::copy_n(ids.driverUUID, VK_UUID_SIZE, description.driver_uuid.begin());
  }
  VulkanExternalBufferHandles handles() const {
    return {reinterpret_cast<uintptr_t>(memory_handle.handle),
            reinterpret_cast<uintptr_t>(ready_handle.handle),
            reinterpret_cast<uintptr_t>(consumed_handle.handle)};
  }
  std::unique_ptr<Buffer> source;
  vkapi::IVkSemaphore ready, consumed;
  NativeHandle memory_handle, ready_handle, consumed_handle;
  VulkanExternalBufferDescription description;
};

void qualify_submission(bool independent, bool compose = false,
                        bool managed = false) {
  VulkanDeviceCreator producer_creator(parameters()),
      consumer_creator(parameters());
  auto &pd = *producer_creator.device();
  auto &cd = *consumer_creator.device();
  if (independent && cd.created_compute_queue_count() < 2)
    GTEST_SKIP() << "Selected device has no independent compute queue";
  // Preserve producer flags, including optional device address.
  Exported exported(pd, false, cd.vk_caps().buffer_device_address);
  bool epoch_ready = false;
  bool query_throws = false;
  std::uint64_t expected_epoch = 1;
  int queries = 0;
  auto producer_lifetime = std::make_shared<int>(1);
  std::weak_ptr<void> producer_weak = producer_lifetime;
  auto callback_lifetime = std::make_shared<int>(2);
  std::weak_ptr<void> callback_weak = callback_lifetime;
  std::shared_ptr<VulkanExternalBufferOwner> owner;
  std::shared_ptr<VulkanExternalAccess> access;
  std::shared_ptr<VulkanExternalBuffer> imported;
  if (managed) {
    owner = VulkanExternalBufferOwner::import_borrowed(
        cd, exported.description, exported.handles(),
        [&, callback_marker = callback_lifetime](auto epoch) {
          (void)callback_marker;
          ++queries;
          EXPECT_EQ(epoch, expected_epoch);
          if (query_throws)
            throw std::runtime_error("ready query failed");
          return epoch_ready;
        }, producer_lifetime);
    access = std::make_shared<VulkanExternalAccess>(
        std::vector<std::shared_ptr<VulkanExternalBufferOwner>>{owner, owner});
    EXPECT_EQ(access->size(), 1u);
    imported = owner->buffer();
  } else {
    imported = VulkanExternalBuffer::import_borrowed(
        cd, exported.description, exported.handles());
  }
  producer_lifetime.reset();
  callback_lifetime.reset();
  std::weak_ptr<VulkanExternalBuffer> weak = imported;
  // Caller originals are still valid, then closed before any GPU use.
  for (auto handle : {&exported.memory_handle, &exported.ready_handle,
                      &exported.consumed_handle}) {
    DWORD flags{};
    EXPECT_TRUE(GetHandleInformation(handle->handle, &flags));
    CloseHandle(handle->handle);
    handle->handle = nullptr;
  }
  auto ready = managed ? nullptr : std::make_shared<VulkanStreamSemaphoreObject>(
      cd.backend_fault_reporter(), imported->ready());
  auto consumed = managed ? nullptr : std::make_shared<VulkanStreamSemaphoreObject>(
      cd.backend_fault_reporter(), imported->consumed());
  auto &stream = *static_cast<VulkanStream *>(cd.get_compute_stream());
  std::unique_ptr<VulkanExecutionQueue> queue;
  if (independent) {
    queue = std::make_unique<VulkanExecutionQueue>(cd, 1);
    EXPECT_NE(queue->queue_index(), 0);
    EXPECT_THROW(queue->prepare({}, {}, {imported->ready(), imported->ready()}),
                 std::invalid_argument);
    EXPECT_THROW(
        queue->prepare({}, {}, {imported->ready()}, {imported->ready()}),
        std::invalid_argument);
    EXPECT_THROW(queue->prepare({}, {}, {exported.ready}),
                 std::invalid_argument);
  }
  VulkanExecutionQueue::Point previous;
  Endpoint producer(pd), readback(cd);
  for (uint32_t epoch = 0; epoch < 8; ++epoch) {
    const std::array<uint32_t, 4> before{
        {epoch, 0x80000001u + epoch, 17u, 0xffffff00u + epoch}};
    const std::array<uint32_t, 4> after{
        {epoch + 100u, 0x81234567u, 29u, 0xfffffe00u + epoch}};
    std::unique_ptr<taichi::lang::CommandList> commands;
    if (independent) {
      commands = queue->new_command_list();
    } else {
      auto created = stream.new_command_list_unique();
      ASSERT_EQ(created.second, taichi::lang::RhiResult::success);
      commands = std::move(created.first);
    }
    auto &vk = *static_cast<VulkanCommandList *>(commands.get());
    if (!compose)
      imported->record_acquire(vk);
    const auto native_commands = vk.vk_command_buffer()->buffer;
    VkBufferCopy copy{0, 0, bytes};
    cd.vk_device().functions().vkCmdCopyBuffer(
        native_commands, imported->buffer, readback.readback->buffer, 1, &copy);
    VkBufferMemoryBarrier barriers[2]{};
    for (auto &b : barriers) {
      b.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
      b.size = bytes;
      b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    }
    barriers[0].buffer = readback.readback->buffer;
    barriers[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barriers[0].dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    barriers[1].buffer = imported->buffer;
    barriers[1].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barriers[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    cd.vk_device().functions().vkCmdPipelineBarrier(
        native_commands, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
        nullptr, 2, barriers, 0, nullptr);
    cd.vk_device().functions().vkCmdUpdateBuffer(
        native_commands, imported->buffer, 0, bytes, after.data());
    if (!compose)
      imported->record_release(vk);
    // Binary waits require the signal to have been submitted already. The
    // producer's host event holds that enqueued signal pending without a spin.
    HostGate gate(pd.vk_device());
    producer.begin();
    pd.vk_device().functions().vkCmdWaitEvents(
        producer.commands, 1, &gate.event, VK_PIPELINE_STAGE_HOST_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, nullptr, 0, nullptr, 0, nullptr);
    if (epoch)
      producer.barrier(*exported.source, VK_ACCESS_TRANSFER_READ_BIT,
                       VK_ACCESS_TRANSFER_WRITE_BIT);
    producer.update(*exported.source, before);
    producer.release(*exported.source);
    producer.submit({}, exported.ready->semaphore);
    gate.pending_fence = producer.fence;
    taichi::lang::StreamSemaphore completion;
    std::shared_ptr<const VulkanExecutionQueue::Commands> fixed;
    try {
      if (independent) {
        std::weak_ptr<void> body_owner, boundary_owner;
        if (managed) {
          auto body = queue->prepare({commands.get()});
          fixed = queue->prepare_external_access(body, access);
          EXPECT_THROW(queue->prepare_external_access(fixed, access),
                       std::invalid_argument);
        } else if (compose) {
          auto body_lifetime = std::make_shared<int>(1);
          auto boundary_lifetime = std::make_shared<int>(2);
          body_owner = body_lifetime;
          boundary_owner = boundary_lifetime;
          auto body = queue->prepare({commands.get()}, body_lifetime);
          auto acquire = queue->new_command_list();
          auto release = queue->new_command_list();
          imported->record_acquire(*static_cast<VulkanCommandList *>(acquire.get()));
          imported->record_release(*static_cast<VulkanCommandList *>(release.get()));
          EXPECT_THROW(queue->prepare_external_access(
                           nullptr, *acquire, *release, {imported->ready()},
                           {imported->consumed()}),
                       std::invalid_argument);
          EXPECT_THROW(queue->prepare_external_access(
                           body, *acquire, *release,
                           {imported->ready(), imported->ready()},
                           {imported->consumed()}),
                       std::invalid_argument);
          EXPECT_THROW(queue->prepare_external_access(
                           body, *acquire, *release, {exported.ready},
                           {imported->consumed()}),
                       std::invalid_argument);
          fixed = queue->prepare_external_access(
              body, *acquire, *release, {imported->ready()},
              {imported->consumed()}, boundary_lifetime);
          EXPECT_THROW(queue->prepare_external_access(
                           fixed, *acquire, *release, {imported->ready()},
                           {imported->consumed()}),
                       std::invalid_argument);
        } else {
          fixed = queue->prepare({commands.get()}, {}, {imported->ready()},
                                 {imported->consumed()});
        }
        std::vector<VulkanExecutionQueue::Point> dependencies;
        if (previous)
          dependencies.push_back(previous);
        const auto before_submit =
            cd.queue_submission_snapshot().queue_submit_calls;
        // Rejected host admission must not consume the pending ready signal,
        // publish a completion, or occupy the only slot.
        const auto old_tail = queue->latest();
        if (managed) {
          bool admitted = false;
          epoch_ready = false;
          EXPECT_THROW(queue->submit(fixed, dependencies, [&](auto, const auto &) {
            admitted = true;
          }), VulkanExecutionQueue::Busy);
          EXPECT_FALSE(admitted);
          query_throws = true;
          EXPECT_THROW(queue->submit(fixed, dependencies), std::runtime_error);
          query_throws = false;
          EXPECT_EQ(owner->snapshot().next_epoch, expected_epoch);
          epoch_ready = true;  // Matching producer signal is already submitted.
        }
        EXPECT_THROW(
            queue->submit(fixed, dependencies,
                          [](auto, const auto &) {
                            throw std::runtime_error("rejected external epoch");
                          }),
            std::runtime_error);
        EXPECT_EQ(queue->latest(), old_tail);
        EXPECT_EQ(queue->pending(), 0);
        EXPECT_EQ(cd.queue_submission_snapshot().queue_submit_calls,
                  before_submit);
        if (managed)
          EXPECT_EQ(owner->snapshot().next_epoch, expected_epoch);
        const auto before_buffers =
            cd.queue_submission_snapshot().submitted_command_buffers;
        previous = queue->submit(fixed, dependencies);
        if (managed) {
          ++expected_epoch;
          EXPECT_EQ(owner->snapshot().next_epoch, expected_epoch);
        }
        EXPECT_EQ(cd.queue_submission_snapshot().queue_submit_calls,
                  before_submit + 1);
        EXPECT_EQ(cd.queue_submission_snapshot().submitted_command_buffers,
                  before_buffers + (compose ? 3 : 1));
        if (compose && !managed) {
          EXPECT_FALSE(body_owner.expired());
          EXPECT_FALSE(boundary_owner.expired());
        }
        completion = previous;
        const auto before_busy = queue->latest();
        bool admitted = false;
        const auto queries_before_busy = queries;
        EXPECT_THROW(
            queue->submit(fixed, {},
                          [&](auto, const auto &) { admitted = true; }),
            VulkanExecutionQueue::Busy);
        EXPECT_FALSE(admitted);
        EXPECT_EQ(queries, queries_before_busy);
        EXPECT_EQ(queue->latest(), before_busy);
        EXPECT_EQ(queue->pending(), 1);
        EXPECT_EQ(cd.queue_submission_snapshot().queue_submit_calls,
                  before_submit + 1);
      } else {
        completion =
            stream.submit_with_semaphores(commands.get(), {ready}, {consumed});
      }
    } catch (const taichi::lang::BackendRuntimeError &error) {
      std::cerr << "RHI submit epoch=" << epoch
                << " backend_code=" << error.backend_code() << '\n';
      throw;
    }
    commands.reset();
    if (epoch == 7)
      imported.reset();
    EXPECT_FALSE(weak.expired());
    EXPECT_FALSE(completion->is_ready());
    std::future<void> closing;
    if (managed && epoch == 7) {
      closing = std::async(std::launch::async, [&] { owner->close(); });
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
      while (owner->snapshot().accepting && std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
      EXPECT_FALSE(owner->snapshot().accepting);
      EXPECT_EQ(closing.wait_for(std::chrono::milliseconds(0)), std::future_status::timeout);
      EXPECT_FALSE(producer_weak.expired());
    }
    gate.release();
    producer.wait();
    EXPECT_TRUE(completion->wait());
    if (closing.valid()) {
      closing.get();
      EXPECT_TRUE(owner->snapshot().closed);
      // The pending slot still retains its recorded imports and producer.
      EXPECT_FALSE(producer_weak.expired());
      EXPECT_FALSE(callback_weak.expired());
    }
    readback.expect(before);
    producer.begin();
    producer.acquire(*exported.source);
    producer.copy(*exported.source);
    producer.submit(exported.consumed->semaphore);
    producer.wait();
    producer.expect(after);
  }
  if (independent) {
    EXPECT_EQ(queue->collect(1), 1);
    queue->close();
  } else {
    stream.command_sync();
  }
  EXPECT_TRUE(weak.expired());
  if (managed) {
    EXPECT_TRUE(producer_weak.expired());
    EXPECT_TRUE(callback_weak.expired());
    EXPECT_EQ(queries, 32);  // Deduplicated owner, four queries per epoch.
    EXPECT_THROW(queue->prepare_external_access(queue->prepare({}), access),
                 std::runtime_error);
  }
}

TEST(VulkanExternalBufferImport,
     RhiSubmissionPendingReadyAndResourceRetention) {
  qualify_submission(false);
}

TEST(VulkanExternalBufferImport,
     IndependentQueueAdmissionAndMixedDependencies) {
  qualify_submission(true);
}

TEST(VulkanExternalBufferImport,
     FrozenBodyCompositionPreservesAdmissionAndOwnership) {
  qualify_submission(true, true);
}

TEST(VulkanExternalBufferImport, ManagedEpochRollbackAndExactRetirement) {
  qualify_submission(true, true, true);
}

TEST(VulkanExternalBufferImport,
     ManagedFixedShaderParametersTimingAndRetirement) {
  using taichi::Arch;
  using namespace taichi::lang;
  const auto *folder =
      std::getenv("TAICHI_VULKAN_EXTERNAL_PARAMETERS_AOT_DIR");
  if (!folder)
    GTEST_SKIP() << "Generate vulkan_external_parameters.py first";
  VulkanDeviceCreator producer_creator(parameters()),
      consumer_creator(parameters());
  auto &pd = *producer_creator.device();
  auto &cd = *consumer_creator.device();
  const bool physical_addresses = cd.get_caps().get(
      DeviceCapability::spirv_has_physical_storage_buffer);
  if (cd.created_compute_queue_count() < 3 ||
      !cd.vk_caps().timeline_semaphore ||
      (physical_addresses && (!cd.vk_caps().buffer_device_address ||
                              !pd.vk_caps().buffer_device_address)))
    GTEST_SKIP() << "Requires two independent queues and the ndarray address mode";
  RecordProperty("ndarray_address_mode",
                 physical_addresses ? "physical" : "descriptor");
  auto fault = std::make_shared<RuntimeFaultDomain>(Arch::vulkan, 926);
  cd.set_backend_fault_reporter(fault);
  gfx::GfxRuntime::Params params;
  params.device = &cd;
  gfx::GfxRuntime runtime(params);
  gfx::AotModuleParams module_params;
  module_params.runtime = &runtime;
  module_params.module_path = folder;
  auto module = aot::Module::load(Arch::vulkan, module_params);
  ASSERT_NE(module, nullptr);
  auto *kernel = module->get_kernel("publish_external");
  ASSERT_NE(kernel, nullptr);

  vulkan::ExecutionDomainRegistry registry(cd, fault);
  std::array<std::shared_ptr<vulkan::ExecutionDomain>, 2> domains{
      registry.create({1, 1, true}), registry.create({1, 1, true})};
  std::array<std::shared_ptr<Exported>, 2> exported;
  std::array<std::weak_ptr<Exported>, 2> producer_weak;
  std::array<std::shared_ptr<VulkanExternalBufferOwner>, 2> owners;
  std::array<std::weak_ptr<VulkanExternalBuffer>, 2> import_weak;
  std::array<std::unique_ptr<Endpoint>, 2> producer;
  std::array<std::unique_ptr<FixedExecution>, 2> calls;
  struct Operands final : RuntimeCompletionResources {
    std::shared_ptr<VulkanExternalBufferOwner> owner;
    std::size_t retained_resource_count(std::uint32_t) const noexcept override {
      return owner ? 1 : 0;
    }
  };
  std::array<std::weak_ptr<Operands>, 2> operand_weak;
  std::array<bool, 2> ready{false, false};
  std::array<std::uint64_t, 2> expected_epoch{1, 1};
  std::array<int, 2> queries{};
  for (std::size_t i = 0; i < calls.size(); ++i) {
    exported[i] = std::make_shared<Exported>(pd, false, physical_addresses);
    producer_weak[i] = exported[i];
    producer[i] = std::make_unique<Endpoint>(pd);
    owners[i] = VulkanExternalBufferOwner::import_borrowed(
        cd, exported[i]->description, exported[i]->handles(),
        [&, i](auto epoch) {
          EXPECT_EQ(epoch, expected_epoch[i]);
          ++queries[i];
          return ready[i];
        }, exported[i]);
    import_weak[i] = owners[i]->buffer();
    auto allocation = owners[i]->ndarray_allocation();
    EXPECT_EQ(allocation, owners[i]->ndarray_allocation());
    Ndarray array(allocation, PrimitiveType::u32, {4});
    LaunchContextBuilder context(kernel);
    context.set_arg_uint({0}, 0);
    context.set_arg_ndarray({1}, array);
    FixedExecution::Dispatch dispatch{kernel, &context};
    dispatch.parameters.push_back({0, kernel->args_type->get_element_offset({0})});
    FixedExecution::VulkanExecutionBinding binding;
    binding.domain = domains[i];
    auto operands = std::make_shared<Operands>();
    operands->owner = owners[i];
    operand_weak[i] = operands;
    binding.operands = std::move(operands);
    binding.parameter_count = 1;
    const auto before = cd.queue_submission_snapshot().queue_submit_calls;
    calls[i] = std::make_unique<FixedExecution>(
        Arch::vulkan, cd, &runtime,
        std::vector<FixedExecution::Operation>{dispatch}, nullptr, fault,
        nullptr, nullptr, nullptr, nullptr, nullptr, &binding);
    calls[i]->bind_external_access({owners[i], owners[i]});
    EXPECT_EQ(cd.queue_submission_snapshot().queue_submit_calls, before);
  }
  // All ordinary cold setup finishes before queue zero is held pending.
  runtime.synchronize();
  RuntimeCompletion first_sampled;
  std::uint64_t first_duration = 0;
  bool first_timing_available = false;
  {
    test_detail::TimelineGate default_gate(cd, 0);
    auto *ordinary = static_cast<VulkanStream *>(cd.get_compute_stream());
    const auto ordinary_pending = ordinary->submit(nullptr);
    for (std::uint32_t round = 0; round < 8; ++round) {
      SCOPED_TRACE(round);
      // Futures precede gates so exceptional unwind releases the device waits
      // before joining an asynchronous owner close.
      std::array<std::future<void>, 2> closing;
      test_detail::TimelineGate left_gate(cd, domains[0]->snapshot().queue_index);
      test_detail::TimelineGate right_gate(cd, domains[1]->snapshot().queue_index);
      std::array<RuntimeCompletion, 2> tickets;
      std::array<std::array<std::uint32_t, 4>, 2> expected;
      for (std::size_t i = 0; i < calls.size(); ++i) {
        SCOPED_TRACE(i);
        domains[i]->collect_ready(1);
        const std::array<std::uint32_t, 4> before{
            0x11111111u + round, 0x22222222u + static_cast<std::uint32_t>(i),
            0xfffffffeu - round, 0x83c00000u + round * 17u + static_cast<std::uint32_t>(i)};
        const std::uint64_t value =
            (0xfedcba9800000000ull | (0x81234560ull + round * 13u + i)) ^
            (static_cast<std::uint64_t>(round) << 36) ^
            (static_cast<std::uint64_t>(i) << 44);
        expected[i] = {static_cast<std::uint32_t>(value),
                       static_cast<std::uint32_t>(value >> 32),
                       before[2] + 3u, before[3] ^ 0xa5c39e71u};
        auto &endpoint = *producer[i];
        auto &source = *exported[i]->source;
        endpoint.begin();
        if (round)
          endpoint.barrier(source, VK_ACCESS_TRANSFER_READ_BIT,
                           VK_ACCESS_TRANSFER_WRITE_BIT);
        endpoint.update(source, before);
        endpoint.release(source);
        endpoint.submit({}, exported[i]->ready->semaphore);
        endpoint.wait();  // The matching binary signal is actually enqueued.

        const bool sample = (round + i) % 2 == 0;
        const auto submits = cd.queue_submission_snapshot().queue_submit_calls;
        const auto prior = domains[i]->latest_completion().sequence();
        ready[i] = false;
        EXPECT_THROW(calls[i]->submit({}, {value ^ UINT64_MAX}, sample),
                     vulkan::ExecutionDomain::Busy);
        EXPECT_EQ(owners[i]->snapshot().next_epoch, expected_epoch[i]);
        EXPECT_EQ(domains[i]->latest_completion().sequence(), prior);
        EXPECT_EQ(domains[i]->snapshot().pending, 0u);
        EXPECT_EQ(cd.queue_submission_snapshot().queue_submit_calls, submits);
        ready[i] = true;
        std::vector<SubmissionParameterValue> values{value};
        tickets[i] = calls[i]->submit({}, values, sample);
        values[0] = 0;  // Accepted parameters must no longer reference this vector.
        ++expected_epoch[i];
        EXPECT_EQ(owners[i]->snapshot().next_epoch, expected_epoch[i]);
        EXPECT_EQ(cd.queue_submission_snapshot().queue_submit_calls, submits + 1);
        EXPECT_FALSE(tickets[i].done());
        const auto busy_queries = queries[i];
        EXPECT_THROW(calls[i]->submit({}, {value ^ UINT64_MAX}, !sample),
                     vulkan::ExecutionDomain::Busy);
        EXPECT_EQ(queries[i], busy_queries);
        EXPECT_EQ(owners[i]->snapshot().next_epoch, expected_epoch[i]);
        EXPECT_EQ(cd.queue_submission_snapshot().queue_submit_calls, submits + 1);
      }
      if (round == 7) {
        // Public recording destruction cannot release a resident instance.
        calls[0].reset();
        for (std::size_t i = 0; i < owners.size(); ++i)
          closing[i] = std::async(std::launch::async,
              [external = owners[i]] { external->close(); });
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        for (std::size_t i = 0; i < owners.size(); ++i) {
          while (owners[i]->snapshot().accepting &&
                 std::chrono::steady_clock::now() < deadline)
            std::this_thread::yield();
          EXPECT_FALSE(owners[i]->snapshot().accepting);
          EXPECT_EQ(closing[i].wait_for(std::chrono::milliseconds(0)),
                    std::future_status::timeout);
          EXPECT_FALSE(import_weak[i].expired());
          EXPECT_FALSE(operand_weak[i].expired());
        }
      }
      left_gate.release();
      right_gate.release();
      for (std::size_t i = 0; i < tickets.size(); ++i) {
        CompletionWaitSet wait({tickets[i]}, std::chrono::milliseconds(1));
        const auto status = wait.wait(std::chrono::seconds(3)).status;
        // A regression onto queue zero must fail without stranding close futures.
        if (status != CompletionWaitSet::Status::ready)
          default_gate.release();
        ASSERT_EQ(status, CompletionWaitSet::Status::ready);
        const auto timing = tickets[i].gpu_timing_snapshot();
        if ((round + i) % 2) {
          EXPECT_EQ(timing.status, "unavailable");
        } else if (cd.queue_timestamp_valid_bits(cd.compute_queue_family_index())) {
          EXPECT_TRUE(timing.available);
        } else {
          EXPECT_EQ(timing.status, "unsupported");
        }
        if (round == 0 && i == 0) {
          first_sampled = tickets[i];
          first_duration = timing.duration_ns;
          first_timing_available = timing.available;
        }
        if (closing[i].valid()) {
          closing[i].get();
          EXPECT_TRUE(owners[i]->snapshot().closed);
          EXPECT_THROW(owners[i]->ndarray_allocation(), std::exception);
          EXPECT_FALSE(import_weak[i].expired());
        }
        auto &endpoint = *producer[i];
        endpoint.begin();
        endpoint.acquire(*exported[i]->source);
        endpoint.copy(*exported[i]->source);
        endpoint.submit(exported[i]->consumed->semaphore);
        endpoint.wait();
        endpoint.expect(expected[i]);
      }
      EXPECT_FALSE(ordinary_pending->is_ready());
    }
    for (auto &source : exported)
      source.reset();
    for (std::size_t i = 0; i < domains.size(); ++i) {
      EXPECT_FALSE(producer_weak[i].expired());
      domains[i]->close();
      EXPECT_TRUE(import_weak[i].expired());
      EXPECT_TRUE(producer_weak[i].expired());
      EXPECT_TRUE(operand_weak[i].expired());
    }
    EXPECT_THROW(calls[1]->submit({}, {1}), std::exception);
    EXPECT_FALSE(ordinary_pending->is_ready());
  }
  registry.retire();
  EXPECT_TRUE(first_sampled.done());
  EXPECT_EQ(first_sampled.gpu_timing_snapshot().available, first_timing_available);
  EXPECT_EQ(first_sampled.gpu_timing_snapshot().duration_ns, first_duration);
}

int ndarray_address_queries = 0;
VKAPI_ATTR VkDeviceAddress VKAPI_CALL count_ndarray_address_query(
    VkDevice, const VkBufferDeviceAddressInfo *) {
  ++ndarray_address_queries;
  return 0;
}

TEST(VulkanExternalBufferImport, ManagedNdarrayHonorsAddressModeBeforeQuery) {
  VulkanDeviceCreator producer(parameters()), consumer(parameters());
  auto &pd = *producer.device();
  auto &cd = *consumer.device();
  const bool physical_addresses = cd.get_caps().get(
      taichi::lang::DeviceCapability::spirv_has_physical_storage_buffer);
  // Native BDA support is not the ndarray ABI selection. The normal creator
  // currently keeps physical storage disabled even on BDA-capable devices.
  RecordProperty("ndarray_address_mode",
                 physical_addresses ? "physical" : "descriptor");
  Exported exported(pd);  // Valid raw transfer/storage import, without BDA usage.
  auto owner = VulkanExternalBufferOwner::import_borrowed(
      cd, exported.description, exported.handles(), [](auto) { return false; });
  auto &api = const_cast<VolkDeviceTable &>(cd.vk_device().functions());
  const auto original = api.vkGetBufferDeviceAddress;
  auto restore = std::shared_ptr<void>(reinterpret_cast<void *>(1),
      [&](void *) { api.vkGetBufferDeviceAddress = original; });
  ndarray_address_queries = 0;
  api.vkGetBufferDeviceAddress = count_ndarray_address_query;
  if (physical_addresses) {
    EXPECT_THROW(owner->ndarray_allocation(), std::invalid_argument);
  } else {
    auto allocation = owner->ndarray_allocation();
    EXPECT_NE(allocation, taichi::lang::kDeviceNullAllocation);
    EXPECT_EQ(allocation, owner->ndarray_allocation());
  }
  EXPECT_EQ(ndarray_address_queries, 0);
  EXPECT_EQ(owner->snapshot().next_epoch, 1u);
  owner->close();
}

TEST(VulkanExternalBufferImport, ManagedNotReadyPreservesRuntimeSequence) {
  VulkanDeviceCreator producer(parameters()), consumer(parameters());
  auto &pd = *producer.device();
  auto &cd = *consumer.device();
  if (cd.created_compute_queue_count() < 2)
    GTEST_SKIP();
  auto fault = std::make_shared<taichi::lang::RuntimeFaultDomain>(
      taichi::Arch::vulkan, 925);
  cd.set_backend_fault_reporter(fault);
  taichi::lang::vulkan::ExecutionDomain domain(cd, fault, {1, 1});
  bool cold_released = false;
  auto reentrant = std::shared_ptr<void>(reinterpret_cast<void *>(1), [&](void *) {
    (void)domain.snapshot();
    cold_released = true;
  });
  auto rejected_body = domain.prepare_commands({}, reentrant);
  reentrant.reset();
  EXPECT_THROW(domain.prepare_external_access(std::move(rejected_body), nullptr),
               std::invalid_argument);
  EXPECT_TRUE(cold_released);
  Exported exported(pd);
  int queries = 0;
  auto owner = VulkanExternalBufferOwner::import_borrowed(
      cd, exported.description, exported.handles(), [&](auto epoch) {
        EXPECT_EQ(epoch, 1u);
        ++queries;
        return false;
      });
  auto access = std::make_shared<VulkanExternalAccess>(
      std::vector<std::shared_ptr<VulkanExternalBufferOwner>>{owner});
  auto body = domain.prepare_commands({});
  auto commands = domain.prepare_external_access(body, access);
  auto first = domain.submit(body);
  first.wait();
  bool admitted = false;
  const auto submits = cd.queue_submission_snapshot().queue_submit_calls;
  EXPECT_THROW(domain.submit(commands, {}, {}, [&](auto, const auto &) {
    admitted = true;
  }), taichi::lang::vulkan::ExecutionDomain::Busy);
  EXPECT_FALSE(admitted);
  EXPECT_EQ(queries, 1);
  EXPECT_EQ(domain.latest_completion().sequence(), first.sequence());
  EXPECT_EQ(domain.snapshot().pending, 0u);
  EXPECT_EQ(cd.queue_submission_snapshot().queue_submit_calls, submits);
  auto next = domain.submit(body);
  next.wait();
  EXPECT_EQ(next.sequence(), first.sequence() + 1);
  EXPECT_EQ(owner->snapshot().next_epoch, 1u);
  domain.close();
  owner->close();
}

TEST(VulkanExternalBufferImport, ManagedMultipleOwnersRollbackInIdentityOrder) {
  VulkanDeviceCreator producer(parameters()), consumer(parameters());
  auto &pd = *producer.device();
  auto &cd = *consumer.device();
  if (cd.created_compute_queue_count() < 3)
    GTEST_SKIP();
  VulkanExecutionQueue first(cd, 1), second(cd, 1);
  Exported a(pd), b(pd);
  std::vector<int> queries;
  auto left = VulkanExternalBufferOwner::import_borrowed(
      cd, a.description, a.handles(), [&](auto epoch) {
        EXPECT_EQ(epoch, 1u);
        queries.push_back(1);
        return true;
      });
  auto right = VulkanExternalBufferOwner::import_borrowed(
      cd, b.description, b.handles(), [&](auto epoch) {
        EXPECT_EQ(epoch, 1u);
        queries.push_back(2);
        return false;
      });
  auto forward = std::make_shared<VulkanExternalAccess>(
      std::vector<std::shared_ptr<VulkanExternalBufferOwner>>{left, right, left});
  auto reverse = std::make_shared<VulkanExternalAccess>(
      std::vector<std::shared_ptr<VulkanExternalBufferOwner>>{right, left});
  auto fc = first.prepare_external_access(first.prepare({}), forward);
  auto sc = second.prepare_external_access(second.prepare({}), reverse);
  auto reject = [](auto &queue, auto &commands) {
    for (int i = 0; i < 8; ++i)
      EXPECT_THROW(queue.submit(commands), VulkanExecutionQueue::Busy);
  };
  auto concurrent = std::async(std::launch::async, [&] { reject(second, sc); });
  reject(first, fc);
  concurrent.get();
  ASSERT_EQ(queries.size(), 32u);
  for (std::size_t i = 0; i < queries.size(); i += 2) {
    EXPECT_EQ(queries[i], 1);
    EXPECT_EQ(queries[i + 1], 2);
  }
  EXPECT_EQ(left->snapshot().next_epoch, 1u);
  EXPECT_EQ(right->snapshot().next_epoch, 1u);
  EXPECT_FALSE(first.latest());
  EXPECT_FALSE(second.latest());
  first.close();
  second.close();
  left->close();
  right->close();
}

VKAPI_ATTR VkResult VKAPI_CALL reject_queue_submit(
    VkQueue, uint32_t, const VkSubmitInfo *, VkFence) {
  return VK_ERROR_OUT_OF_HOST_MEMORY;
}

TEST(VulkanExternalBufferImport, ManagedEnqueueFailurePoisonsUntilQueueDrain) {
  VulkanDeviceCreator producer(parameters()), consumer(parameters());
  auto &pd = *producer.device();
  auto &cd = *consumer.device();
  if (cd.created_compute_queue_count() < 2)
    GTEST_SKIP();
  VulkanExecutionQueue queue(cd, 1);
  Exported exported(pd);
  auto retention = std::make_shared<int>(1);
  std::weak_ptr<void> weak = retention;
  auto owner = VulkanExternalBufferOwner::import_borrowed(
      cd, exported.description, exported.handles(), [](auto) { return true; },
      retention);
  retention.reset();
  auto access = std::make_shared<VulkanExternalAccess>(
      std::vector<std::shared_ptr<VulkanExternalBufferOwner>>{owner});
  auto commands = queue.prepare_external_access(queue.prepare({}), access);
  // No native work is sent by this injection. Keep a real ready signal anyway,
  // so the fixture obeys the producer contract before reaching driver failure.
  Endpoint endpoint(pd);
  endpoint.begin();
  endpoint.release(*exported.source);
  endpoint.submit({}, exported.ready->semaphore);
  endpoint.wait();
  auto &api = const_cast<VolkDeviceTable &>(cd.vk_device().functions());
  const auto submit = api.vkQueueSubmit;
  api.vkQueueSubmit = reject_queue_submit;
  EXPECT_THROW(queue.submit(commands), taichi::lang::BackendRuntimeError);
  api.vkQueueSubmit = submit;
  EXPECT_EQ(owner->snapshot().next_epoch, 1u);
  EXPECT_TRUE(owner->snapshot().poisoned);
  EXPECT_FALSE(owner->snapshot().accepting);
  EXPECT_EQ(queue.pending(), 1u);
  EXPECT_THROW(owner->close(), std::runtime_error);
  EXPECT_FALSE(weak.expired());
  EXPECT_THROW(queue.submit(commands), taichi::lang::BackendRuntimeError);
  queue.close();  // Successful native queue drain proves safe retirement.
  owner->close();
  EXPECT_TRUE(owner->snapshot().closed);
  EXPECT_TRUE(owner->snapshot().poisoned);  // Never made usable again.
  EXPECT_FALSE(weak.expired());  // Cold recorded imports still own the peer.
  commands.reset();
  EXPECT_TRUE(weak.expired());
}

// Inject errors into this device's immutable-in-production dispatch table.
// This lets failed imports prove exact cleanup without feeding invalid handles
// or mismatched allocation metadata to a real driver.
class ImportCalls {
 public:
  explicit ImportCalls(VulkanDevice &device)
      : api(const_cast<VolkDeviceTable &>(device.vk_device().functions())),
        saved(api) {
    EXPECT_EQ(active, nullptr);
    active = this;
    api.vkCreateBuffer = create_buffer;
    api.vkDestroyBuffer = destroy_buffer;
    api.vkAllocateMemory = allocate;
    api.vkFreeMemory = free_memory;
    api.vkBindBufferMemory = bind;
    api.vkCreateSemaphore = create_semaphore;
    api.vkDestroySemaphore = destroy_semaphore;
    api.vkImportSemaphoreWin32HandleKHR = import_semaphore;
  }
  ~ImportCalls() {
    api = saved;
    active = nullptr;
  }
  int fail_stage{0}, buffers{0}, memories{0}, semaphores{0}, creates{0},
      imports{0};
  VkDeviceSize requested_bytes{0};

 private:
  static VKAPI_ATTR VkResult VKAPI_CALL
  create_buffer(VkDevice d,
                const VkBufferCreateInfo *i,
                const VkAllocationCallbacks *a,
                VkBuffer *b) {
    ++active->creates;
    auto r = active->saved.vkCreateBuffer(d, i, a, b);
    if (r == VK_SUCCESS)
      ++active->buffers;
    return r;
  }
  static VKAPI_ATTR void VKAPI_CALL
  destroy_buffer(VkDevice d, VkBuffer b, const VkAllocationCallbacks *a) {
    --active->buffers;
    active->saved.vkDestroyBuffer(d, b, a);
  }
  static VKAPI_ATTR VkResult VKAPI_CALL allocate(VkDevice d,
                                                 const VkMemoryAllocateInfo *i,
                                                 const VkAllocationCallbacks *a,
                                                 VkDeviceMemory *m) {
    active->requested_bytes = i->allocationSize;
    if (active->fail_stage == 1)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    auto r = active->saved.vkAllocateMemory(d, i, a, m);
    if (r == VK_SUCCESS)
      ++active->memories;
    return r;
  }
  static VKAPI_ATTR void VKAPI_CALL
  free_memory(VkDevice d, VkDeviceMemory m, const VkAllocationCallbacks *a) {
    --active->memories;
    active->saved.vkFreeMemory(d, m, a);
  }
  static VKAPI_ATTR VkResult VKAPI_CALL bind(VkDevice d,
                                             VkBuffer b,
                                             VkDeviceMemory m,
                                             VkDeviceSize o) {
    return active->fail_stage == 2
               ? VK_ERROR_OUT_OF_DEVICE_MEMORY
               : active->saved.vkBindBufferMemory(d, b, m, o);
  }
  static VKAPI_ATTR VkResult VKAPI_CALL
  create_semaphore(VkDevice d,
                   const VkSemaphoreCreateInfo *i,
                   const VkAllocationCallbacks *a,
                   VkSemaphore *s) {
    auto r = active->saved.vkCreateSemaphore(d, i, a, s);
    if (r == VK_SUCCESS)
      ++active->semaphores;
    return r;
  }
  static VKAPI_ATTR void VKAPI_CALL
  destroy_semaphore(VkDevice d, VkSemaphore s, const VkAllocationCallbacks *a) {
    --active->semaphores;
    active->saved.vkDestroySemaphore(d, s, a);
  }
  static VKAPI_ATTR VkResult VKAPI_CALL
  import_semaphore(VkDevice d, const VkImportSemaphoreWin32HandleInfoKHR *i) {
    ++active->imports;
    if ((active->fail_stage == 3 && active->imports == 1) ||
        (active->fail_stage == 4 && active->imports == 2))
      return VK_ERROR_INVALID_EXTERNAL_HANDLE;
    return active->saved.vkImportSemaphoreWin32HandleKHR(d, i);
  }
  VolkDeviceTable &api;
  VolkDeviceTable saved;
  static ImportCalls *active;
};
ImportCalls *ImportCalls::active = nullptr;

TEST(VulkanExternalBufferImport, ColdValidationAndFailureRollback) {
  VulkanDeviceCreator producer(parameters()), consumer(parameters());
  Exported exported(*producer.device());
  auto &device = *consumer.device();
  ImportCalls calls(device);
  for (int variant = 0; variant < 10; ++variant) {
    auto d = exported.description;
    auto h = exported.handles();
    switch (variant) {
      case 0:
        d.memory_type_index = UINT32_MAX;
        break;
      case 1:
        d.driver_uuid.fill(0);
        break;
      case 2:
        d.device_uuid[0] ^= 1;
        break;
      case 3:
        d.bind_offset = 4;
        break;
      case 4:
        d.size = d.allocation_size + 1;
        break;
      case 5:
        d.flags = VK_BUFFER_CREATE_SPARSE_BINDING_BIT;
        break;
      case 6:
        d.sharing_mode = VK_SHARING_MODE_CONCURRENT;
        break;
      case 7:
        h.consumed = h.ready;
        break;
      case 8:
        d.dedicated = false;
        break;
      case 9:
        d.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        d.allocation_flags = 0;
        break;
    }
    EXPECT_THROW(VulkanExternalBuffer::import_borrowed(device, d, h),
                 std::invalid_argument);
  }
  EXPECT_EQ(calls.creates, 0);
  {
    NativeHandle alias;
    ASSERT_TRUE(DuplicateHandle(
        GetCurrentProcess(), exported.ready_handle.handle, GetCurrentProcess(),
        &alias.handle, 0, FALSE, DUPLICATE_SAME_ACCESS));
    auto handles = exported.handles();
    handles.consumed = reinterpret_cast<uintptr_t>(alias.handle);
    EXPECT_THROW(VulkanExternalBuffer::import_borrowed(
                     device, exported.description, handles),
                 std::invalid_argument);
    EXPECT_EQ(calls.creates, 0);
  }
  DWORD before{}, after{};
  ASSERT_TRUE(GetProcessHandleCount(GetCurrentProcess(), &before));
  for (int stage = 1; stage <= 4; ++stage) {
    calls.fail_stage = stage;
    calls.imports = 0;
    EXPECT_THROW(VulkanExternalBuffer::import_borrowed(
                     device, exported.description, exported.handles()),
                 std::runtime_error);
    EXPECT_EQ(calls.buffers, 0);
    EXPECT_EQ(calls.memories, 0);
    EXPECT_EQ(calls.semaphores, 0);
    for (auto h : {exported.handles().memory, exported.handles().ready,
                   exported.handles().consumed}) {
      DWORD flags{};
      EXPECT_TRUE(GetHandleInformation(reinterpret_cast<HANDLE>(h), &flags));
    }
  }
  ASSERT_TRUE(GetProcessHandleCount(GetCurrentProcess(), &after));
  EXPECT_EQ(before, after);
  // Current Vulkan allows dedicated allocation beyond minimum requirements.
  // Verify producer bytes reach allocate without requiring an older installed
  // validation layer to implement the revised dedicated-allocation VUID.
  calls.fail_stage = 1;
  auto padded = exported.description;
  padded.allocation_size += 256;
  EXPECT_THROW(
      VulkanExternalBuffer::import_borrowed(device, padded, exported.handles()),
      std::runtime_error);
  EXPECT_EQ(calls.requested_bytes, padded.allocation_size);
  EXPECT_EQ(calls.buffers, 0);
  calls.fail_stage = 0;
  auto imported = VulkanExternalBuffer::import_borrowed(
      device, exported.description, exported.handles());
  EXPECT_EQ(calls.buffers, 1);
  EXPECT_EQ(calls.memories, 1);
  EXPECT_EQ(calls.semaphores, 2);
  imported.reset();
  EXPECT_EQ(calls.buffers, 0);
  EXPECT_EQ(calls.memories, 0);
  EXPECT_EQ(calls.semaphores, 0);
}
}  // namespace
#endif
