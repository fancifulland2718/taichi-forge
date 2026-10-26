#include "gtest/gtest.h"

#include <limits>

#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/program/compile_config.h"

namespace taichi::lang {
namespace {

struct CpuRangePartitionTest : ::testing::Test {
  Block root;
  CompileConfig config;
  OffloadedStmt *task;

  void SetUp() override {
    config.arch = Arch::x64;
    config.cpu_max_num_threads = 32;
    task = root.push_back<OffloadedStmt>(OffloadedTaskType::range_for,
                                        Arch::x64, nullptr)
               ->as<OffloadedStmt>();
    task->const_begin = task->const_end = true;
    task->begin_value = -17;
    task->end_value = 1007;
    task->num_cpu_threads = 4;
    task->body->push_back<LoopIndexStmt>(task, 0);
  }

  void run() {
    irpass::make_cpu_multithreaded_range_for(&root, config);
  }
};

TEST_F(CpuRangePartitionTest, TrimsEmptyChunksWithoutChangingWorkerRequest) {
  run();
  EXPECT_EQ(task->begin_value, 0);
  EXPECT_EQ(task->end_value, 2);
  EXPECT_EQ(task->block_dim, 1);
  EXPECT_EQ(task->num_cpu_threads, 4);
  auto *inner = task->body->back()->as<RangeForStmt>();
  ASSERT_NE(inner, nullptr);
  EXPECT_TRUE(inner->strictly_serialized);
  EXPECT_EQ(inner->body->statements.front()->as<LoopIndexStmt>()->loop, inner);
}

TEST_F(CpuRangePartitionTest, KeepsPartialAndSaturatedChunks) {
  task->end_value = task->begin_value + 1025;
  run();
  EXPECT_EQ(task->end_value, 3);
}

TEST_F(CpuRangePartitionTest, LargeRangeKeepsEveryProductiveChunk) {
  task->end_value = task->begin_value + 100000;
  run();
  EXPECT_EQ(task->end_value, 32);
}

TEST_F(CpuRangePartitionTest, EmptyRangeHasNoChunks) {
  task->end_value = task->begin_value;
  run();
  EXPECT_EQ(task->end_value, 0);
}

TEST_F(CpuRangePartitionTest, DynamicBoundsAreNotSpecialized) {
  task->const_end = false;
  task->end_offset = 0;
  run();
  EXPECT_EQ(task->end_value, 32);
}

TEST_F(CpuRangePartitionTest, DeviceBoundedRangeKeepsLogicalBounds) {
  task->one_to_one = true;
  run();
  EXPECT_EQ(task->begin_value, -17);
  EXPECT_EQ(task->end_value, 1007);
}

TEST_F(CpuRangePartitionTest, AuxiliaryEffectsArePreserved) {
  task->tls_epilogue = std::make_unique<Block>();
  task->tls_epilogue->push_back<ConstStmt>(TypedConstant(7));
  run();
  EXPECT_EQ(task->end_value, 32);
}

TEST_F(CpuRangePartitionTest, UnprovenI32PartitionIsUnchanged) {
  task->begin_value = std::numeric_limits<int32_t>::max() - 10;
  task->end_value = std::numeric_limits<int32_t>::max();
  run();
  EXPECT_EQ(task->end_value, 32);
}

}  // namespace
}  // namespace taichi::lang
