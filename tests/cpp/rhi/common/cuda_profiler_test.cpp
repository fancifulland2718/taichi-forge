#include "gtest/gtest.h"

#include "taichi/rhi/cuda/cuda_profiler.h"

namespace taichi::lang {
namespace {

class TestableCUDAProfiler : public KernelProfilerCUDA {
 public:
  TestableCUDAProfiler() : KernelProfilerCUDA(false) {
  }

  void append_unaggregated_record(const std::string &name,
                                  float duration_ms) {
    ensure_record_capacity();
    KernelProfileTracedRecord record;
    record.name = name;
    record.kernel_elapsed_time_in_ms = duration_ms;
    traced_records_.push_back(std::move(record));
  }

  const std::vector<KernelProfileStatisticalResult> &statistics() const {
    return statistical_results_;
  }

  double total_time_ms() const {
    return total_time_ms_;
  }
};

TEST(CUDAProfilerTest, AggregatesEachTracedRecordExactlyOnce) {
  TestableCUDAProfiler profiler;
  profiler.append_unaggregated_record("first", 1.25f);
  ASSERT_TRUE(profiler.statistics_on_traced_records());
  ASSERT_TRUE(profiler.statistics_on_traced_records());

  profiler.append_unaggregated_record("first", 2.75f);
  profiler.append_unaggregated_record("second", 4.0f);
  ASSERT_TRUE(profiler.statistics_on_traced_records());
  ASSERT_EQ(profiler.statistics().size(), 2u);
  EXPECT_EQ(profiler.statistics()[0].name, "first");
  EXPECT_EQ(profiler.statistics()[0].counter, 2);
  EXPECT_DOUBLE_EQ(profiler.statistics()[0].total, 4.0);
  EXPECT_EQ(profiler.statistics()[1].name, "second");
  EXPECT_EQ(profiler.statistics()[1].counter, 1);
  EXPECT_DOUBLE_EQ(profiler.statistics()[1].total, 4.0);
  EXPECT_DOUBLE_EQ(profiler.total_time_ms(), 8.0);
}

TEST(CUDAProfilerTest, ClearResetsTraceAndAggregateCursors) {
  TestableCUDAProfiler profiler;
  profiler.append_unaggregated_record("before-clear", 3.0f);
  ASSERT_TRUE(profiler.statistics_on_traced_records());
  profiler.clear();

  EXPECT_EQ(profiler.record_count(), 0u);
  EXPECT_TRUE(profiler.statistics().empty());
  EXPECT_DOUBLE_EQ(profiler.total_time_ms(), 0.0);

  profiler.append_unaggregated_record("after-clear", 5.0f);
  ASSERT_TRUE(profiler.statistics_on_traced_records());
  ASSERT_EQ(profiler.statistics().size(), 1u);
  EXPECT_EQ(profiler.statistics()[0].counter, 1);
  EXPECT_DOUBLE_EQ(profiler.total_time_ms(), 5.0);
}

TEST(CUDAProfilerTest, RecordBudgetDoesNotCorruptAggregates) {
  TestableCUDAProfiler profiler;
  profiler.set_record_capacity_for_testing(2);
  profiler.append_unaggregated_record("bounded", 1.0f);
  profiler.append_unaggregated_record("bounded", 2.0f);
  ASSERT_TRUE(profiler.statistics_on_traced_records());

  EXPECT_ANY_THROW(profiler.append_unaggregated_record("overflow", 3.0f));
  ASSERT_TRUE(profiler.statistics_on_traced_records());
  ASSERT_EQ(profiler.statistics().size(), 1u);
  EXPECT_EQ(profiler.statistics()[0].counter, 2);
  EXPECT_DOUBLE_EQ(profiler.total_time_ms(), 3.0);
}

}  // namespace
}  // namespace taichi::lang
