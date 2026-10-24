#include "gtest/gtest.h"

#include "taichi/analysis/range_for_bounds.h"
#include "taichi/codegen/spirv/compiled_kernel_data.h"
#ifdef TI_WITH_LLVM
#include "taichi/codegen/llvm/compiled_kernel_data.h"
#endif

namespace taichi::lang {
namespace {

TEST(RangeForBounds, SignedExtentsAndWorkspaceValidation) {
  RangeForBounds negative{RangeBound::constant(-17), RangeBound::constant(-2)};
  ASSERT_TRUE(negative.constant_size());
  EXPECT_EQ(*negative.constant_size(), 15);
  RangeForBounds reversed{RangeBound::constant(9), RangeBound::constant(-2)};
  EXPECT_EQ(*reversed.constant_size(), 0);
  RangeForBounds wide{RangeBound::constant(INT32_MIN),
                      RangeBound::constant(INT32_MAX)};
  EXPECT_EQ(*wide.constant_size(), INT64_C(4294967295));
  auto offset = RangeBound::global_tmp(12);
  EXPECT_TRUE(offset.fits_global_tmps(16));
  EXPECT_FALSE(offset.fits_global_tmps(15));
  EXPECT_FALSE(offset.fits_global_tmps(0));
  auto wide_offset = RangeBound::global_tmp(INT64_C(1) << 33);
  EXPECT_TRUE(wide_offset.valid());
  EXPECT_TRUE(wide_offset.fits_global_tmps((UINT64_C(1) << 33) + 4));
  EXPECT_FALSE(RangeBound::global_tmp(-4).valid());
  EXPECT_FALSE(RangeBound::global_tmp(2).valid());
  EXPECT_FALSE((RangeBound{RangeBoundKind::unknown, 4}).valid());
  EXPECT_FALSE((RangeBound{static_cast<RangeBoundKind>(99), 0}).valid());
  EXPECT_FALSE((RangeBound{RangeBoundKind::constant_i32, INT64_C(2147483648)})
                   .valid());
  EXPECT_FALSE(RangeBound::constant(0).fits_global_tmps(128));
  EXPECT_FALSE(RangeForBounds{}.constant_size());
}

TEST(RangeForBounds, FinalTaskSourcesDoNotInventTemporaryOffsets) {
  for (auto arch : {Arch::x64, Arch::cuda, Arch::vulkan}) {
    OffloadedStmt task(OffloadedTaskType::range_for, arch, nullptr);
    task.const_begin = true;
    task.begin_value = -3;
    task.const_end = false;
    task.end_offset = 0;
    auto bounds = describe_range_for_bounds(task);
    ASSERT_TRUE(bounds);
    EXPECT_EQ(bounds->begin.kind, RangeBoundKind::constant_i32);
    EXPECT_EQ(bounds->begin.value, -3);
    EXPECT_EQ(bounds->end.kind, RangeBoundKind::global_tmp_i32);
    EXPECT_EQ(bounds->end.value, 0);
    auto expression = Stmt::make_typed<ConstStmt>(TypedConstant(7));
    task.end_stmt = expression.get();
    auto direct_expression = describe_range_for_bounds(task);
    EXPECT_EQ(direct_expression->end.kind, RangeBoundKind::unknown);
    EXPECT_FALSE(direct_expression->constant_size());
    // The prior task's frozen description is unaffected by offset reuse.
    EXPECT_EQ(bounds->end.kind, RangeBoundKind::global_tmp_i32);
    task.end_stmt = nullptr;
    task.const_begin = false;
    task.begin_offset = 16;
    task.const_end = true;
    task.end_value = 11;
    auto mixed = describe_range_for_bounds(task);
    EXPECT_EQ(mixed->begin.kind, RangeBoundKind::global_tmp_i32);
    EXPECT_EQ(mixed->begin.value, 16);
    EXPECT_EQ(mixed->end.kind, RangeBoundKind::constant_i32);
    OffloadedStmt serial(OffloadedTaskType::serial, arch, nullptr);
    EXPECT_FALSE(describe_range_for_bounds(serial));
  }
}

TEST(RangeForBounds, SharedCodecPreservesSignedAndUnknownSources) {
  const RangeForBounds sources[] = {
      {RangeBound::constant(-9), RangeBound::constant(4)},
      {RangeBound::global_tmp(0), RangeBound::global_tmp(20)},
      {RangeBound::constant(0), RangeBound{}},
  };
  for (const auto &original : sources) {
    BinaryOutputSerializer output;
    output.initialize();
    output(original);
    output.finalize();
    RangeForBounds decoded;
    BinaryInputSerializer input;
    input.initialize(output.data.data());
    input(decoded);
    EXPECT_TRUE(decoded.valid());
    EXPECT_EQ(decoded.begin.kind, original.begin.kind);
    EXPECT_EQ(decoded.begin.value, original.begin.value);
    EXPECT_EQ(decoded.end.kind, original.end.kind);
    EXPECT_EQ(decoded.end.value, original.end.value);
    auto json = liong::json::serialize(original);
    RangeForBounds from_json;
    liong::json::deserialize(json, from_json, true);
    EXPECT_EQ(from_json.end.kind, original.end.kind);
    EXPECT_EQ(from_json.begin.value, original.begin.value);
  }
}

TEST(RangeForBounds, BackendManifestsProjectTheSameCompiledFacts) {
  const RangeForBounds bounds{RangeBound::constant(-9), RangeBound::constant(4)};
  spirv::CompiledKernelData::InternalData data;
  spirv::TaskAttributes task;
  task.task_type = OffloadedTaskType::range_for;
  task.range_for_attribs = bounds;
  data.metadata.kernel_attribs.tasks_attribs.push_back(task);
  spirv::CompiledKernelData vulkan(Arch::vulkan, std::move(data));
  auto manifest = vulkan.task_manifest();
  ASSERT_EQ(manifest.size(), 1);
  EXPECT_EQ(*manifest[0].constant_range_size, 13);
  EXPECT_EQ(manifest[0].range_bounds->begin.value, -9);
  std::ostringstream encoded;
  ASSERT_EQ(vulkan.dump(encoded), CompiledKernelData::Err::kNoError);
  spirv::CompiledKernelData decoded;
  std::istringstream input(encoded.str());
  ASSERT_EQ(decoded.load(input), CompiledKernelData::Err::kNoError);
  EXPECT_EQ(decoded.task_manifest()[0].range_bounds->begin.value, -9);
#ifdef TI_WITH_LLVM
  for (auto arch : {Arch::x64, Arch::cuda}) {
    LLVM::CompiledKernelData::InternalData llvm_data;
    OffloadedTask llvm_task;
    llvm_task.task_type = OffloadedTaskType::range_for;
    llvm_task.range_bounds = bounds;
    EXPECT_EQ(llvm_task.constant_range_size(), 13);
    BinaryOutputSerializer task_output;
    task_output.initialize();
    task_output(llvm_task);
    task_output.finalize();
    BinaryInputSerializer task_input;
    task_input.initialize(task_output.data.data());
    OffloadedTask decoded_task;
    task_input(decoded_task);
    EXPECT_EQ(decoded_task.range_bounds->begin.value, -9);
    EXPECT_EQ(decoded_task.constant_range_size(), 13);
    llvm_data.compiled_data.tasks.push_back(llvm_task);
    LLVM::CompiledKernelData compiled(arch, std::move(llvm_data));
    auto llvm_manifest = compiled.task_manifest();
    EXPECT_EQ(llvm_manifest[0].constant_range_size, manifest[0].constant_range_size);
    EXPECT_EQ(llvm_manifest[0].range_bounds->begin.value, -9);
  }
#endif
}

TEST(RangeForBounds, CompiledSpirvReaderRejectsInvalidSource) {
  spirv::CompiledKernelData::InternalData data;
  spirv::TaskAttributes task;
  task.task_type = OffloadedTaskType::range_for;
  task.range_for_attribs = RangeForBounds{RangeBound::constant(0),
                                        RangeBound::global_tmp(3)};
  data.metadata.kernel_attribs.tasks_attribs.push_back(task);
  spirv::CompiledKernelData malformed(Arch::vulkan, std::move(data));
  std::ostringstream output;
  ASSERT_EQ(malformed.dump(output), CompiledKernelData::Err::kNoError);
  std::istringstream input(output.str());
  spirv::CompiledKernelData loaded;
  EXPECT_EQ(loaded.load(input), CompiledKernelData::Err::kCompiledKernelDataBroken);
}

#ifdef TI_WITH_LLVM
TEST(RangeForBounds, LlvmRangeProofSurvivesCodecAndChecksPhysicalContract) {
  OffloadedTask task;
  task.task_type = OffloadedTaskType::range_for;
  task.range_bounds = RangeForBounds{RangeBound::constant(-3), RangeBound::global_tmp(4)};
  task.block_dim = 32;
  task.grid_dim = 64;
  task.device_range_dispatch_version = 1;
  ASSERT_TRUE(task.valid_device_range_dispatch_contract());
  BinaryOutputSerializer output;
  output.initialize();
  output(task);
  output.finalize();
  BinaryInputSerializer input;
  input.initialize(output.data.data());
  OffloadedTask decoded;
  input(decoded);
  EXPECT_EQ(decoded.device_range_dispatch_version, 1);
  EXPECT_TRUE(decoded.valid_device_range_dispatch_contract());
  decoded.one_to_one = true;
  EXPECT_FALSE(decoded.valid_device_range_dispatch_contract());
  decoded.one_to_one = false;
  decoded.device_range_dispatch_version = 2;
  EXPECT_FALSE(decoded.valid_device_range_dispatch_contract());
  decoded.device_range_dispatch_version = 1;
  decoded.range_bounds->end = RangeBound::constant(4);
  EXPECT_FALSE(decoded.valid_device_range_dispatch_contract());
}
#endif

}  // namespace
}  // namespace taichi::lang
