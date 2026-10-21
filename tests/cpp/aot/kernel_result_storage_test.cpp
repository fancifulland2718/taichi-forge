#include "gtest/gtest.h"
#include "taichi/program/callable.h"
#include "taichi/runtime/native_kernel_result.h"

#include <array>
#include <cstring>
#include <stdexcept>
#ifdef TI_WITH_LLVM
#include "taichi/rhi/cpu/cpu_device.h"
#endif
#ifdef TI_WITH_CUDA
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_device.h"
#endif
#ifdef TI_WITH_VULKAN
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/rhi/vulkan/vulkan_device_creator.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"
#endif

namespace taichi::lang {
namespace {
CallableBase result_kernel() {
  CallableBase kernel;
  kernel.ret_type = TypeFactory::get_instance().get_struct_type(
      {{PrimitiveType::f32, "0", 0}, {PrimitiveType::i64, "1", 8}},
      "result-test")->as<StructType>();
  kernel.ret_size = 16;
  return kernel;
}

// These tests qualify storage ownership and transfers, not execution. Public
// prepared-kernel tests separately qualify compiled producers and completion.
void check_storage(Arch arch, Device &device) {
  const auto kernel = result_kernel();
  NativeKernelResultStorage first(arch, device, kernel);
  NativeKernelResultStorage second(arch, device, kernel);
  EXPECT_TRUE(first.matches(arch, device, kernel));
  EXPECT_EQ(first.device_payload_bytes(), 16u);
  EXPECT_EQ(first.workspace_bytes(), 0u);
  EXPECT_NE(first.allocation(), second.allocation());
  if (arch_is_cpu(arch) || arch == Arch::cuda) {
    EXPECT_NE(first.address(), 0u);
    EXPECT_NE(first.address(), second.address());
  }
  std::array<std::uint8_t, 16> result;
  result.fill(0xff);
  first.read_completed(result.data(), result.size());
  EXPECT_EQ(result, (std::array<std::uint8_t, 16>{}));
  // Write only leaves, preserving cold-zeroed padding.
  const float number = 3.5f;
  const std::int64_t integer = 0x123456789abcdef;
  DevicePtr destinations[]{first.allocation().get_ptr(),
                           first.allocation().get_ptr(8)};
  const void *sources[]{&number, &integer};
  std::size_t sizes[]{sizeof(number), sizeof(integer)};
  ASSERT_EQ(device.upload_data(destinations, sources, sizes, 2), RhiResult::success);
  first.read_completed(result.data(), result.size());
  float actual_number;
  std::int64_t actual_integer;
  std::memcpy(&actual_number, result.data(), sizeof(actual_number));
  std::memcpy(&actual_integer, result.data() + 8, sizeof(actual_integer));
  EXPECT_EQ(actual_number, number);
  EXPECT_EQ(actual_integer, integer);
  for (int i = 4; i < 8; ++i)
    EXPECT_EQ(result[i], 0u);
  second.read_completed(result.data(), result.size());
  EXPECT_EQ(result, (std::array<std::uint8_t, 16>{}));
  EXPECT_THROW(first.read_completed(nullptr, result.size()), std::invalid_argument);
  EXPECT_THROW(first.read_completed(result.data(), result.size() - 1), std::invalid_argument);
  auto different = kernel;
  different.ret_size = 15;
  EXPECT_FALSE(first.matches(arch, device, different));
  different = kernel;
  different.ret_type = TypeFactory::get_instance().get_struct_type(
      {{PrimitiveType::i32, "0", 0}, {PrimitiveType::i64, "1", 8}},
      "result-test")->as<StructType>();
  EXPECT_FALSE(first.matches(arch, device, different));
}

void check_discarded_pool(Arch arch, Device &device) {
  const auto kernel = result_kernel();
  std::shared_ptr<NativeKernelResultStorage> retained;
  {
    NativeKernelResultPool sequential(arch, device), concurrent(arch, device);
    retained = sequential.acquire(kernel);
    EXPECT_FALSE(retained->observable());
    // Logical dispatch count is not allocation count, and no scale cap is
    // needed to share an identical compiled result layout in this region.
    for (int i = 0; i < 65; ++i)
      EXPECT_EQ(sequential.acquire(kernel), retained);
    EXPECT_EQ(sequential.resources().size(), 1u);
    EXPECT_NE(concurrent.acquire(kernel)->allocation(), retained->allocation());
    auto changed = kernel;
    changed.ret_size = 24;
    EXPECT_NE(sequential.acquire(changed)->allocation(), retained->allocation());
    changed = kernel;
    changed.ret_type = TypeFactory::get_instance().get_struct_type(
        {{PrimitiveType::i32, "0", 0}, {PrimitiveType::i64, "1", 8}},
        "result-test")->as<StructType>();
    EXPECT_NE(sequential.acquire(changed)->allocation(), retained->allocation());
    EXPECT_EQ(sequential.resources().size(), 3u);
    CallableBase empty;
    EXPECT_EQ(sequential.acquire(empty), nullptr);
    changed.ret_size = 1;
    EXPECT_THROW(sequential.acquire(changed), std::invalid_argument);
    EXPECT_EQ(sequential.resources().size(), 3u);
  }
  // The prepared owner can keep the allocation after cold analysis retires.
  EXPECT_TRUE(retained->matches(arch, device, kernel));
  std::array<std::uint8_t, 16> output{};
  EXPECT_THROW(retained->read_completed(output.data(), output.size()), std::invalid_argument);
}

#ifdef TI_WITH_LLVM
class FailingResultDevice : public cpu::CpuDevice {
 public:
  int allocations{0}, live{0};
  bool fail_allocate{false}, fail_upload{false}, fail_read{false};
  RhiResult allocate_memory(const AllocParams &params, DeviceAllocation *out) override {
    ++allocations;
    if (fail_allocate)
      return RhiResult::out_of_memory;
    const auto status = cpu::CpuDevice::allocate_memory(params, out);
    if (status == RhiResult::success)
      ++live;
    return status;
  }
  void dealloc_memory(DeviceAllocation allocation) override {
    --live;
    cpu::CpuDevice::dealloc_memory(allocation);
  }
  RhiResult upload_data(DevicePtr *ptr, const void **data, size_t *sizes,
                       int count = 1) noexcept override {
    const auto status = cpu::CpuDevice::upload_data(ptr, data, sizes, count);
    return fail_upload ? RhiResult::error : status;
  }
  RhiResult readback_data(DevicePtr *ptr, void **data, size_t *sizes,
                         int count = 1,
                         const std::vector<StreamSemaphore> &waits = {}) noexcept override {
    if (fail_read)
      return RhiResult::error;
    return cpu::CpuDevice::readback_data(ptr, data, sizes, count, waits);
  }
};
TEST(KernelResultStorage, CpuIndependentFramesPreservePadding) {
  cpu::CpuDevice device;
  check_storage(host_arch(), device);
}
TEST(KernelResultStorage, CpuDiscardedScopeSharesOnlyMatchingLayouts) {
  FailingResultDevice device;
  device.fail_upload = true;  // Discarded scratch must not initialize/upload.
  check_discarded_pool(host_arch(), device);
  EXPECT_EQ(device.live, 0);
}
TEST(KernelResultStorage, InvalidLayoutsAndVoidDoNotAllocate) {
  FailingResultDevice device;
  CallableBase empty;
  NativeKernelResultStorage result(host_arch(), device, empty);
  EXPECT_EQ(result.device_payload_bytes(), 0u);
  EXPECT_EQ(result.allocation(), kDeviceNullAllocation);
  result.read_completed(nullptr, 0);
  empty.ret_size = 1;
  EXPECT_THROW(NativeKernelResultStorage(host_arch(), device, empty), std::invalid_argument);
  auto invalid = result_kernel();
  invalid.ret_size = 12;
  EXPECT_THROW(NativeKernelResultStorage(host_arch(), device, invalid), std::invalid_argument);
  invalid.ret_type = TypeFactory::get_instance().get_struct_type(
      {{TypeFactory::get_instance().get_pointer_type(PrimitiveType::f32), "pointer"}},
      "result-test")->as<StructType>();
  EXPECT_THROW(NativeKernelResultStorage(host_arch(), device, invalid), std::invalid_argument);
  EXPECT_THROW(NativeKernelResultStorage(Arch::opengl, device, result_kernel()), std::invalid_argument);
  EXPECT_EQ(device.allocations, 0);
}
TEST(KernelResultStorage, FailedPreparationAndReadsDoNotLoseOwners) {
  FailingResultDevice device;
  const auto kernel = result_kernel();
  device.fail_allocate = true;
  EXPECT_THROW(NativeKernelResultStorage(host_arch(), device, kernel), std::runtime_error);
  EXPECT_EQ(device.live, 0);
  device.fail_allocate = false;
  device.fail_upload = true;
  EXPECT_THROW(NativeKernelResultStorage(host_arch(), device, kernel), std::runtime_error);
  EXPECT_EQ(device.live, 0);
  device.fail_upload = false;
  {
    auto owner = std::make_shared<NativeKernelResultStorage>(host_arch(), device, kernel);
    cpu::CpuDevice foreign;
    EXPECT_FALSE(owner->matches(host_arch(), foreign, kernel));
    std::weak_ptr<NativeKernelResultStorage> lifetime = owner;
    std::shared_ptr<NativeResource> retained = owner;
    owner.reset();
    EXPECT_FALSE(lifetime.expired());
    device.fail_read = true;
    std::array<std::uint8_t, 16> data{};
    EXPECT_THROW(lifetime.lock()->read_completed(data.data(), data.size()), std::runtime_error);
    EXPECT_EQ(device.live, 1);
    device.fail_read = false;
    lifetime.lock()->read_completed(data.data(), data.size());
    NativeKernelResultStorage retry(host_arch(), device, kernel);
    EXPECT_EQ(device.live, 2);
    retained.reset();
    EXPECT_TRUE(lifetime.expired());
  }
  EXPECT_EQ(device.live, 0);
}
#endif
#ifdef TI_WITH_CUDA
TEST(KernelResultStorage, CudaIndependentFramesPreservePadding) {
  if (!CUDADriver::get_instance_without_context().detected())
    GTEST_SKIP();
  cuda::CudaDevice device;
  auto guard = CUDAContext::get_instance().get_guard();
  auto lock = CUDAContext::get_instance().get_submission_lock_guard();
  check_storage(Arch::cuda, device);
}
TEST(KernelResultStorage, CudaDiscardedScopeSharesOnlyMatchingLayouts) {
  if (!CUDADriver::get_instance_without_context().detected())
    GTEST_SKIP();
  cuda::CudaDevice device;
  auto guard = CUDAContext::get_instance().get_guard();
  auto lock = CUDAContext::get_instance().get_submission_lock_guard();
  check_discarded_pool(Arch::cuda, device);
}
#endif
#ifdef TI_WITH_VULKAN
TEST(KernelResultStorage, VulkanIndependentFramesPreservePadding) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  vulkan::VulkanDeviceCreator creator({});
  check_storage(Arch::vulkan, *creator.device());
}
TEST(KernelResultStorage, VulkanDiscardedScopeSharesOnlyMatchingLayouts) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  vulkan::VulkanDeviceCreator creator({});
  check_discarded_pool(Arch::vulkan, *creator.device());
}
#endif
}  // namespace
}  // namespace taichi::lang
