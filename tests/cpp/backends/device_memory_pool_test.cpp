#include "gtest/gtest.h"

#ifdef TI_WITH_LLVM
#include "taichi/rhi/llvm/device_memory_pool.h"

namespace taichi::lang {

TEST(DeviceMemoryPool, BackendOwnershipWithoutDeviceInitialization) {
  auto &cuda = DeviceMemoryPool::get_instance(Arch::cuda);
  auto &amdgpu = DeviceMemoryPool::get_instance(Arch::amdgpu, false);
  EXPECT_NE(&cuda, &amdgpu);
  EXPECT_EQ(&cuda, &DeviceMemoryPool::get_instance());
  EXPECT_EQ(&amdgpu, &DeviceMemoryPool::get_instance(Arch::amdgpu));
}

}  // namespace taichi::lang
#endif
