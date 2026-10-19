#include "gtest/gtest.h"
#include "taichi/runtime/native_sort.h"
#ifdef TI_WITH_LLVM
#include "taichi/rhi/cpu/cpu_device.h"
#include <algorithm>
#include <array>
#include <limits>
#endif

namespace taichi::lang {
#ifdef TI_WITH_LLVM
TEST(NativeSortWorkspace, CpuSharedStorageRetainsStableOrderAndLifetime) {
  cpu::CpuDevice device;
  constexpr std::size_t n = 7;
  auto [keys, kr] = device.allocate_memory_unique({n * 4});
  auto [values, vr] = device.allocate_memory_unique({n * 4});
  ASSERT_EQ(kr, RhiResult::success);
  ASSERT_EQ(vr, RhiResult::success);
  void *kp = nullptr, *vp = nullptr;
  ASSERT_EQ(device.map_range_for_cpu_native(keys->get_ptr(), n * 4, &kp),
            RhiResult::success);
  ASSERT_EQ(device.map_range_for_cpu_native(values->get_ptr(), n * 4, &vp),
            RhiResult::success);
  auto *k = static_cast<std::int32_t *>(kp);
  auto *v = static_cast<std::uint32_t *>(vp);
  const std::array<std::int32_t, n> input{3, -1, 3, 2, -1, 2, 3};
  const std::array<std::uint32_t, n> expected{1, 4, 3, 5, 0, 2, 6};
  SortOperation d{
      keys->get_ptr(), values->get_ptr(), n, NativeSortKeyType::i32, 4, 4};
  auto scratch = prepare_sort_workspace(host_arch(), device,
                                        SortWorkspaceDescription::from(d));
  ASSERT_EQ(scratch->workspace_bytes(), n * 8);
  std::weak_ptr<NativeSortWorkspace> lifetime = scratch;
  d.workspace = scratch;
  auto first = prepare_sort_operation(host_arch(), device, d);
  auto second = prepare_sort_operation(host_arch(), device, d);
  EXPECT_EQ(first->workspace_bytes(),
            0);  // Resource counted by enclosing owner.
  EXPECT_EQ(second->workspace_bytes(), 0);
  scratch.reset();
  d.workspace.reset();
  for (int replay = 0; replay < 3; ++replay) {
    std::copy(input.begin(), input.end(), k);
    for (std::size_t i = 0; i < n; ++i)
      v[i] = static_cast<std::uint32_t>(i);
    (replay == 0 ? first : second)->run_cpu();
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), v));
    EXPECT_TRUE(std::is_sorted(k, k + n));
    if (replay == 0)
      first.reset();
    EXPECT_FALSE(lifetime.expired());
  }
  second.reset();
  EXPECT_TRUE(lifetime.expired());
  auto own = prepare_sort_operation(host_arch(), device, d);
  EXPECT_EQ(own->workspace_bytes(), n * 8);
}

TEST(NativeSortWorkspace, RejectsMismatchedShapeTypeDeviceBeforeExecution) {
  cpu::CpuDevice device, other;
  auto [keys, result] = device.allocate_memory_unique({64});
  ASSERT_EQ(result, RhiResult::success);
  SortOperation d{keys->get_ptr(), {}, 7, NativeSortKeyType::i32};
  d.workspace = prepare_sort_workspace(host_arch(), device,
                                       SortWorkspaceDescription::from(d));
  auto invalid = d;
  invalid.size = 6;
  EXPECT_THROW(prepare_sort_operation(host_arch(), device, invalid),
               std::invalid_argument);
  invalid = d;
  invalid.key_type = NativeSortKeyType::u32;
  EXPECT_THROW(prepare_sort_operation(host_arch(), device, invalid),
               std::invalid_argument);
  invalid = d;
  invalid.workspace = prepare_sort_workspace(host_arch(), other,
                                             SortWorkspaceDescription::from(d));
  EXPECT_THROW(prepare_sort_operation(host_arch(), device, invalid),
               std::invalid_argument);
  auto description = SortWorkspaceDescription::from(d);
  description.size = std::size_t(std::numeric_limits<int>::max()) + 1;
  EXPECT_THROW(prepare_sort_workspace(host_arch(), device, description),
               std::invalid_argument);
  description = SortWorkspaceDescription::from(d);
  description.value_bytes = 12;
  EXPECT_THROW(prepare_sort_workspace(host_arch(), device, description),
               std::invalid_argument);
}

TEST(NativeSortWorkspace, RawPayloadAndEmptyStoragePreserveTheirContracts) {
  cpu::CpuDevice device;
  constexpr std::size_t n = 5;
  auto [allocation, status] = device.allocate_memory_unique({n * 16});
  ASSERT_EQ(status, RhiResult::success);
  void *mapped = nullptr;
  ASSERT_EQ(
      device.map_range_for_cpu_native(allocation->get_ptr(), n * 16, &mapped),
      RhiResult::success);
  auto *data = static_cast<std::uint32_t *>(mapped);
  SortOperation d{allocation->get_ptr(),
                  allocation->get_ptr(n * 4),
                  n,
                  NativeSortKeyType::u32,
                  12,
                  4};
  d.workspace = prepare_sort_workspace(host_arch(), device,
                                       SortWorkspaceDescription::from(d));
  auto operation = prepare_sort_operation(host_arch(), device, d);
  EXPECT_EQ(d.workspace->workspace_bytes(), n * 20);
  for (int replay = 0; replay < 2; ++replay) {
    for (std::uint32_t i = 0; i < n; ++i) {
      data[i] = n - i;
      for (std::uint32_t lane = 0; lane < 3; ++lane)
        data[n + 3 * i + lane] = (n - i) * 10 + lane;
    }
    operation->run_cpu();
    for (std::uint32_t i = 0; i < n; ++i) {
      EXPECT_EQ(data[i], i + 1);
      for (std::uint32_t lane = 0; lane < 3; ++lane)
        EXPECT_EQ(data[n + 3 * i + lane], (i + 1) * 10 + lane);
    }
  }
  for (std::size_t count : {0, 1}) {
    auto empty = SortWorkspaceDescription{count, NativeSortKeyType::u32, 0, 0};
    EXPECT_EQ(
        prepare_sort_workspace(host_arch(), device, empty)->workspace_bytes(),
        0);
  }
}
#endif
}  // namespace taichi::lang
