#include <gtest/gtest.h>
#include <array>
#include "taichi/runtime/native_diagnostic.h"
#include "taichi/runtime/native_operation_registry.h"
#ifdef TI_WITH_LLVM
#include "taichi/rhi/cpu/cpu_device.h"
#include "taichi/system/threading.h"
#endif

namespace taichi::lang {
TEST(NativeDiagnostic, SchemaPreservesSignedBoundsAndBackendSpecificCounts) {
  const auto &registry = native_operation_registry();
  NativeOperationDescription d;
  d.provider = "builtin";
  d.operation = "check_count";
  d.bindings = {{"values", "v"}, {"output", "out"}};
  for (
      const auto *text :
      {R"({"size":0,"dtype":"i32","op":"nonzero"})",
       R"({"size":4294967296,"dtype":"u64","op":"index_oob","lower":-2147483648,"upper":2147483647,"cpu_max_threads":4,"vulkan_items_per_group":8589934592})"}) {
    d.parameters = liong::json::parse_complete(text);
    EXPECT_EQ(registry.decode(d).kind(), "check_count");
  }
  for (
      const auto *text :
      {R"({"size":1,"dtype":"f32","op":"index_oob"})",
       R"({"size":true,"dtype":"i32","op":"nonzero"})",
       R"({"size":1,"dtype":"i32","op":"unknown"})",
       R"({"size":1,"dtype":"i32","op":"index_oob","lower":true})",
       R"({"size":1,"dtype":"i32","op":"index_oob","upper":2147483648})",
       R"({"size":1,"dtype":"i32","op":"index_oob","lower":-2147483649})",
       R"({"size":1,"dtype":"i32","op":"nonzero","cpu_max_threads":0})",
       R"({"size":1,"dtype":"i32","op":"nonzero","vulkan_items_per_group":-1})"}) {
    d.parameters = liong::json::parse_complete(text);
    EXPECT_THROW(registry.decode(d), std::invalid_argument);
  }
  d.operation = "metric_reduce";
  d.parameters =
      liong::json::parse_complete(R"({"size":0,"dtype":"f64","op":"max_abs"})");
  EXPECT_EQ(registry.decode(d).kind(), "metric_reduce");
  for (const auto *text :
       {R"({"size":1,"dtype":"i32","op":"max_abs"})",
        R"({"size":1,"dtype":"f32","op":"sum"})",
        R"({"size":1,"dtype":"f32","op":"max_abs","lower":0})"}) {
    d.parameters = liong::json::parse_complete(text);
    EXPECT_THROW(registry.decode(d), std::invalid_argument);
  }
}

#ifdef TI_WITH_LLVM
TEST(NativeDiagnostic, CpuOwnerReusesPartialAndPreservesOutputNeighbors) {
  cpu::CpuDevice device;
  ThreadPool pool(4);
  NativePreparationContext context(host_arch(), device);
  context.cpu_thread_pool = &pool;
  context.cpu_max_threads = 4;
  const size_t n = 262147;
  auto [values, values_result] = device.allocate_memory_unique({n * 4});
  auto [other, other_result] = device.allocate_memory_unique({n * 4});
  auto [output, output_result] = device.allocate_memory_unique({5 * 4});
  ASSERT_EQ(values_result, RhiResult::success);
  ASSERT_EQ(other_result, RhiResult::success);
  ASSERT_EQ(output_result, RhiResult::success);
  void *source = nullptr, *rhs = nullptr, *result = nullptr;
  ASSERT_EQ(device.map_range_for_cpu_native(values->get_ptr(), n * 4, &source),
            RhiResult::success);
  ASSERT_EQ(device.map_range_for_cpu_native(other->get_ptr(), n * 4, &rhs),
            RhiResult::success);
  ASSERT_EQ(device.map_range_for_cpu_native(output->get_ptr(), 5 * 4, &result),
            RhiResult::success);
  auto *a = static_cast<float *>(source);
  auto *b = static_cast<float *>(rhs);
  auto *words = static_cast<int32_t *>(result);
  std::fill_n(words, 5, 37);
  CheckCountOperation check;
  check.size = n;
  check.value_type = 1;
  check.cpu_max_threads = 4;
  check.values = {values->get_ptr(), n * 4, PrimitiveType::f32, {n}, {}};
  check.output = {output->get_ptr(4), 4, PrimitiveType::i32, {1}, {}};
  MetricReduceOperation metric;
  metric.size = n;
  metric.value_type = 1;
  metric.op = 1;
  metric.values = check.values;
  metric.other =
      NativeArrayBinding{other->get_ptr(), n * 4, PrimitiveType::f32, {n}, {}};
  metric.output = {output->get_ptr(12), 4, PrimitiveType::f32, {}, {}};
  const auto &registry = native_operation_registry();
  auto first = registry.prepare(context, check);
  auto second = registry.prepare(context, metric);
  EXPECT_EQ(first->workspace_bytes(), 4 * sizeof(uint64_t));
  EXPECT_EQ(second->workspace_bytes(), 4 * sizeof(float));
  EXPECT_EQ(words[1], 37);
  EXPECT_EQ(words[3], 37);
  for (int replay = 0; replay < 3; ++replay) {
    std::fill_n(a, n, replay == 1 ? 0.0f : 2.0f);
    std::fill_n(b, n, 0.0f);
    first->run_cpu();
    second->run_cpu();
    EXPECT_EQ(words[1], replay == 1 ? 0 : n);
    EXPECT_EQ(static_cast<float *>(result)[3], replay == 1 ? 0.0f : 2.0f);
    for (auto index : {0, 2, 4})
      EXPECT_EQ(words[index], 37);
    EXPECT_TRUE(std::all_of(
        a, a + n, [&](float v) { return v == (replay == 1 ? 0.0f : 2.0f); }));
  }
  auto invalid = check;
  invalid.output = {values->get_ptr(), 4, PrimitiveType::i32, {1}, {}};
  EXPECT_THROW(registry.prepare(context, invalid), std::invalid_argument);
  NativePreparationContext no_pool(host_arch(), device);
  EXPECT_THROW(registry.prepare(no_pool, check), std::invalid_argument);
  auto missing_rhs = metric;
  missing_rhs.other.reset();
  EXPECT_THROW(registry.prepare(context, missing_rhs), std::invalid_argument);
  auto unused_rhs = metric;
  unused_rhs.op = 0;
  EXPECT_THROW(registry.prepare(context, unused_rhs), std::invalid_argument);
  check.size = metric.size = 0;
  registry.prepare(context, check)->run_cpu();
  registry.prepare(context, metric)->run_cpu();
  EXPECT_EQ(words[1], 0);
  EXPECT_EQ(static_cast<float *>(result)[3], 0.0f);
}
#endif
}  // namespace taichi::lang
