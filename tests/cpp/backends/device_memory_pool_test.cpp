#include "gtest/gtest.h"

#ifdef TI_WITH_LLVM
#include "taichi/rhi/llvm/device_memory_pool.h"
#include "taichi/rhi/cpu/cpu_device.h"
#include <array>

namespace taichi::lang {

namespace {
class TestArenaDevice : public cpu::CpuDevice {
 public:
  std::array<uint64_t, 1024> arena{};
  int allocations{0};
  uint64_t *allocate_llvm_runtime_memory_jit(
      const LlvmRuntimeAllocParams &params) override {
    EXPECT_EQ(params.size, 4096);
    EXPECT_LT(allocations, 2);
    return arena.data() + 512 * allocations++;
  }
};
}  // namespace

TEST(DeviceMemoryPool, CudaFallbackCacheIsOwnedByItsDevice) {
  // Exercise the fallback allocator without a GPU or a large runtime arena.
  TestArenaDevice a, b;
  DeviceMemoryPool pool(Arch::cuda, true);
  LlvmDevice::LlvmRuntimeAllocParams params{};
  params.size = 4096;
  void *first = pool.allocate_with_cache(&a, params);
  pool.release_cached(&a, params.size, first);
  void *second = pool.allocate_with_cache(&b, params);
  EXPECT_EQ(second, b.arena.data());
  EXPECT_NE(first, second);
  EXPECT_EQ(pool.allocate_with_cache(&a, params), first);
  EXPECT_EQ(a.allocations, 1);
  pool.release_cached(&a, params.size, first);
  pool.release_cached(&b, params.size, second);
  pool.release_device_cache(&a);
  EXPECT_EQ(pool.allocate_with_cache(&b, params), second);
  EXPECT_EQ(b.allocations, 1);
  // A late release from a retired allocation must not revive its old cache.
  pool.release_cached(&a, params.size, first);
  EXPECT_NE(pool.allocate_with_cache(&a, params), first);
  EXPECT_EQ(a.allocations, 2);
}

TEST(DeviceMemoryPool, BackendOwnershipWithoutDeviceInitialization) {
  auto &cuda = DeviceMemoryPool::get_instance(Arch::cuda);
  auto &amdgpu = DeviceMemoryPool::get_instance(Arch::amdgpu, false);
  EXPECT_NE(&cuda, &amdgpu);
  EXPECT_EQ(&cuda, &DeviceMemoryPool::get_instance());
  EXPECT_EQ(&amdgpu, &DeviceMemoryPool::get_instance(Arch::amdgpu));
}

}  // namespace taichi::lang
#endif
