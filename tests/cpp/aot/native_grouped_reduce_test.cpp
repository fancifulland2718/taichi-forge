#include <gtest/gtest.h>
#include <array>
#include "taichi/runtime/native_grouped_reduce.h"
#include "taichi/runtime/native_operation_registry.h"
#ifdef TI_WITH_LLVM
#include "taichi/rhi/cpu/cpu_device.h"
#include "taichi/system/threading.h"
#endif

namespace taichi::lang {
TEST(NativeGroupedReduce, SchemaPreservesEmptyInputAndBackendSpecificCounts) {
  const auto &registry = native_operation_registry();
  NativeOperationDescription d;
  d.provider = "builtin";
  d.operation = "grouped_reduce";
  d.bindings = {{"keys", "k"}, {"values", "v"}, {"output", "out"}};
  for (
      const auto *text :
      {R"({"size":0,"num_groups":7,"dtype":"i32"})",
       R"({"size":4294967296,"num_groups":4294967296,"dtype":"f64","cpu_max_threads":4})"}) {
    d.parameters = liong::json::parse_complete(text);
    EXPECT_EQ(registry.decode(d).kind(), "grouped_reduce");
  }
  for (
      const auto *text :
      {R"({"size":1,"num_groups":0,"dtype":"f32"})",
       R"({"size":true,"num_groups":7,"dtype":"i32"})",
       R"({"size":1,"num_groups":7,"dtype":"i16"})",
       R"({"size":1,"num_groups":7,"dtype":"i32","cpu_max_threads":0})",
       R"({"size":1,"num_groups":7,"dtype":"i32","cpu_max_threads":2147483648})",
       R"({"size":1,"num_groups":7,"dtype":"i32","strategy":"bucket"})"}) {
    d.parameters = liong::json::parse_complete(text);
    EXPECT_THROW(registry.decode(d), std::invalid_argument);
  }
  d.operation = "scatter_add";
  d.bindings = {{"indices", "k"}, {"src", "v"}, {"dst", "out"}};
  d.parameters =
      liong::json::parse_complete(R"({"size":0,"dst_items":7,"dtype":"i32"})");
  EXPECT_THROW(registry.decode(d), std::invalid_argument);
  d.parameters =
      liong::json::parse_complete(R"({"size":7,"dst_items":7,"dtype":"i32"})");
  EXPECT_EQ(registry.decode(d).kind(), "scatter_add");
}

#ifdef TI_WITH_LLVM
TEST(NativeGroupedReduce, CpuOwnerReusesPartialAndOverwritesInvalidGroups) {
  cpu::CpuDevice device;
  ThreadPool pool(2);
  NativePreparationContext context(host_arch(), device);
  context.cpu_thread_pool = &pool;
  context.cpu_max_threads = 2;
  constexpr size_t n = 262147, groups = 7;
  auto [keys, kr] = device.allocate_memory_unique({n * 4});
  auto [values, vr] = device.allocate_memory_unique({n * 4});
  auto [output, ore] = device.allocate_memory_unique({(groups + 2) * 4});
  ASSERT_EQ(kr, RhiResult::success);
  ASSERT_EQ(vr, RhiResult::success);
  ASSERT_EQ(ore, RhiResult::success);
  void *key_ptr = nullptr, *value_ptr = nullptr, *output_ptr = nullptr;
  ASSERT_EQ(device.map_range_for_cpu_native(keys->get_ptr(), n * 4, &key_ptr),
            RhiResult::success);
  ASSERT_EQ(
      device.map_range_for_cpu_native(values->get_ptr(), n * 4, &value_ptr),
      RhiResult::success);
  ASSERT_EQ(device.map_range_for_cpu_native(output->get_ptr(), (groups + 2) * 4,
                                            &output_ptr),
            RhiResult::success);
  auto *k = static_cast<int32_t *>(key_ptr);
  auto *v = static_cast<float *>(value_ptr);
  auto *out = static_cast<float *>(output_ptr);
  std::fill_n(out, groups + 2, 37.0f);
  GroupedReduceOperation description;
  description.size = n;
  description.num_groups = groups;
  description.value_type = 1;
  description.cpu_max_threads =
      4;  // Frozen producer partitions on a smaller pool.
  description.keys = {keys->get_ptr(), n * 4, PrimitiveType::i32, {n}, {}};
  description.values = {values->get_ptr(), n * 4, PrimitiveType::f32, {n}, {}};
  description.output = {
      output->get_ptr(4), groups * 4, PrimitiveType::f32, {groups}, {}};
  const auto &registry = native_operation_registry();
  auto operation = registry.prepare(context, description);
  EXPECT_EQ(operation->workspace_bytes(), 4 * groups * 4);
  for (size_t i = 0; i < groups + 2; ++i)
    EXPECT_EQ(out[i], 37);
  for (int replay = 0; replay < 3; ++replay) {
    std::array<float, groups> expected{};
    for (size_t i = 0; i < n; ++i) {
      k[i] = replay == 1 ? -1 : static_cast<int32_t>(i % (groups + 2)) - 1;
      v[i] = float(i % 5 + 1);
      if (k[i] >= 0 && k[i] < groups)
        expected[k[i]] += v[i];
    }
    operation->run_cpu();
    for (size_t i = 0; i < groups; ++i)
      EXPECT_EQ(out[i + 1], expected[i]);
    EXPECT_EQ(out[0], 37);
    EXPECT_EQ(out[groups + 1], 37);
    EXPECT_EQ(operation->workspace_bytes(), 4 * groups * 4);
    for (size_t i = 0; i < n; ++i)
      ASSERT_EQ(v[i], float(i % 5 + 1));
  }
  auto invalid = description;
  invalid.output = {
      values->get_ptr(), groups * 4, PrimitiveType::f32, {groups}, {}};
  EXPECT_THROW(registry.prepare(context, invalid), std::invalid_argument);
  NativePreparationContext no_pool(host_arch(), device);
  EXPECT_THROW(registry.prepare(no_pool, description), std::invalid_argument);
  description.size = 0;
  auto empty = registry.prepare(context, description);
  EXPECT_EQ(empty->workspace_bytes(), 0);
  empty->run_cpu();
  for (size_t i = 0; i < groups; ++i)
    EXPECT_EQ(out[i + 1], 0);
  EXPECT_EQ(out[0], 37);
  EXPECT_EQ(out[groups + 1], 37);
}
#endif
}  // namespace taichi::lang
