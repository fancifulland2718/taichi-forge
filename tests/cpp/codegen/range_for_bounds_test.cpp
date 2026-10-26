#include "gtest/gtest.h"

#include "taichi/analysis/range_for_bounds.h"
#include "taichi/analysis/host_range_bounds.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/transforms.h"
#include "taichi/codegen/spirv/compiled_kernel_data.h"
#ifdef TI_WITH_LLVM
#include "taichi/codegen/llvm/compiled_kernel_data.h"
#endif

namespace taichi::lang {
namespace {

TEST(HostRangeBounds, ScalarAndShapeWorkloadDoNotImposeACap) {
  std::int32_t arguments[] = {-3, 126, 3, 43};
  HostRangeBounds scalar{{1, {0}, false}, {1, {4}, false}};
  EXPECT_EQ(scalar.grid(8160, 128, arguments, sizeof(arguments)), 2);
  arguments[1] = -8;
  EXPECT_EQ(scalar.grid(8160, 128, arguments, sizeof(arguments)), 1);
  arguments[1] = INT32_MAX;
  EXPECT_EQ(scalar.grid(8160, 128, arguments, sizeof(arguments)), 8160);
  HostRangeBounds shape{{0, {}, false}, {1, {8, 12}, true}};
  EXPECT_EQ(shape.grid(8160, 128, arguments, sizeof(arguments)), 2);
  arguments[2] = 0;
  EXPECT_EQ(shape.grid(8160, 128, arguments, sizeof(arguments)), 1);
  arguments[2] = INT32_MAX;
  EXPECT_EQ(shape.grid(8160, 128, arguments, sizeof(arguments)), 8160);
  arguments[2] = -1;
  EXPECT_EQ(shape.grid(8160, 128, arguments, sizeof(arguments)), 8160);
}

TEST(HostRangeBounds, InvalidArgumentImageFallsBackWithoutReadingIt) {
  HostRangeBounds bounds{{0, {}, false}, {1, {4}, true}};
  EXPECT_FALSE(bounds.valid(7));
  EXPECT_EQ(bounds.grid(37, 128, nullptr, 8), 37);
  EXPECT_EQ(bounds.grid(37, 128, nullptr, 0), 37);
  bounds.end.offsets = {UINT64_MAX};
  EXPECT_FALSE(bounds.valid(UINT64_MAX));
  bounds.end.offsets = {1};
  EXPECT_FALSE(bounds.valid(8));
  bounds.end = {2, {0}, false};
  EXPECT_FALSE(bounds.valid(8));
}

TEST(HostRangeBounds, AffineCompositionPreservesIntermediateOverflow) {
  HostRangeBound bound{1, {0}, false};
  ASSERT_TRUE(compose_host_range_affine(bound, 3, 0));
  ASSERT_TRUE(compose_host_range_affine(bound, 1, -1));
  std::int32_t value = 64;
  EXPECT_EQ(bound.evaluate(&value, 4), 191);
  value = -4;
  EXPECT_EQ(bound.evaluate(&value, 4), -13);
  value = INT32_MAX;
  EXPECT_FALSE(bound.evaluate(&value, 4));
  EXPECT_TRUE(bound.valid(4));
  bound.input_max = INT32_MAX;
  EXPECT_FALSE(bound.valid(4));

  bound = {1, {0}, false};
  ASSERT_TRUE(compose_host_range_affine(bound, 1, 1));
  ASSERT_TRUE(compose_host_range_affine(bound, 1, -1));
  EXPECT_EQ(bound.scale, 1);
  EXPECT_EQ(bound.bias, 0);
  EXPECT_FALSE(bound.evaluate(&value, 4));  // (INT_MAX + 1) - 1
  value = INT32_MAX - 1;
  EXPECT_EQ(bound.evaluate(&value, 4), value);

  bound = {1, {0}, false};
  ASSERT_TRUE(compose_host_range_affine(bound, -1, 7));
  value = -3;
  EXPECT_EQ(bound.evaluate(&value, 4), 10);
  value = INT32_MIN;
  EXPECT_FALSE(bound.evaluate(&value, 4));
  ASSERT_TRUE(compose_host_range_affine(bound, 0, 0));
  EXPECT_FALSE(bound.evaluate(&value, 4));  // Cancellation keeps prior guards.
}

TEST(HostRangeBounds, AffineGuardMatchesCheckedSourceOperations) {
  for (int scale : {-7, -1, 0, 1, 3, INT32_MAX}) {
    for (int bias : {INT32_MIN, -1, 0, 9, INT32_MAX}) {
      HostRangeBound bound{1, {0}, false};
      ASSERT_TRUE(compose_host_range_affine(bound, scale, 0));
      ASSERT_TRUE(compose_host_range_affine(bound, 1, bias));
      EXPECT_TRUE(bound.valid(4));
      for (int value : {INT32_MIN, INT32_MIN + 1, -8, -1, 0, 1, 8,
                        INT32_MAX - 1, INT32_MAX}) {
        const auto product = std::int64_t(value) * scale;
        const auto sum = product + bias;
        const bool safe = product >= INT32_MIN && product <= INT32_MAX &&
                          sum >= INT32_MIN && sum <= INT32_MAX;
        const auto result = bound.evaluate(&value, 4);
        EXPECT_EQ(result.has_value(), safe);
        if (safe) EXPECT_EQ(*result, sum);
      }
    }
  }
}

TEST(HostRangeBounds, MutableBoundsRetainEnvelopeButOtherParametersDoNot) {
  const std::int32_t arguments[] = {7, 64};
  RangeForBounds bounds{RangeBound::constant(0), RangeBound::global_tmp(0),
                        HostRangeBounds{{0, {}, false}, {1, {4}, true}},
                        EmptyRangeGroupProof::independent_iterations};
  std::vector<std::pair<std::uint64_t, std::uint64_t>> patches{{0, 4}};
  EXPECT_EQ(bounds.launch_grid(8160, 128, arguments, sizeof(arguments), &patches), 1);
  patches = {{4, 4}};
  EXPECT_EQ(bounds.launch_grid(8160, 128, arguments, sizeof(arguments), &patches), 8160);
  patches = {{0, 8}};
  EXPECT_EQ(bounds.launch_grid(8160, 128, arguments, sizeof(arguments), &patches), 8160);
  patches = {{8, 4}};
  EXPECT_EQ(bounds.launch_grid(8160, 128, arguments, sizeof(arguments), &patches), 1);
}

TEST(HostRangeBounds, CompositeProgramsCheckArithmeticAndMutableInputs) {
  HostRangeBound bound;
  bound.program = {{HostRangeOp::argument_i32},
                   {HostRangeOp::floor_divide, 3, 0},
                   {HostRangeOp::constant_i32, -4},
                   {HostRangeOp::maximum, 0, 1, 2}};
  std::int32_t value = -7;
  EXPECT_EQ(bound.evaluate(&value, 4), -3);
  bound.program[1].op = HostRangeOp::divide;
  EXPECT_EQ(bound.evaluate(&value, 4), -2);
  EXPECT_TRUE(bound.reads(0, 1));
  EXPECT_TRUE(bound.reads(3, 1));
  EXPECT_FALSE(bound.reads(4, 1));
  EXPECT_FALSE(bound.reads(UINT64_MAX, 4));
  EXPECT_FALSE(bound.evaluate(nullptr, 4));
  EXPECT_FALSE(bound.evaluate(&value, 3));
  bound.program = {{HostRangeOp::argument_i32},
                   {HostRangeOp::constant_i32, 1},
                   {HostRangeOp::add, 0, 0, 1},
                   {HostRangeOp::constant_i32, 65},
                   {HostRangeOp::minimum, 0, 2, 3}};
  value = INT32_MAX;
  EXPECT_FALSE(bound.evaluate(&value, 4));  // Clamp cannot conceal overflow.
  value = 66;
  EXPECT_EQ(bound.evaluate(&value, 4), 65);
  RangeForBounds range{RangeBound::constant(0), RangeBound{},
                       HostRangeBounds{{0, {}, false}, bound},
                       EmptyRangeGroupProof::independent_iterations};
  std::vector<std::pair<std::uint64_t, std::uint64_t>> patches{{0, 4}};
  EXPECT_EQ(range.launch_grid(37, 128, &value, 4), 1);
  EXPECT_EQ(range.launch_grid(37, 128, &value, 4, &patches), 37);
  RangeForBounds decoded;
  liong::json::deserialize(liong::json::serialize(range), decoded, true);
  EXPECT_EQ(range.host_launch, decoded.host_launch);
  BinaryOutputSerializer output;
  output.initialize();
  output(range);
  output.finalize();
  BinaryInputSerializer input;
  input.initialize(output.data.data());
  input(decoded);
  EXPECT_EQ(range.host_launch, decoded.host_launch);
}

TEST(HostRangeBounds, CompositeProgramsRejectMalformedNodesWithoutACap) {
  HostRangeBound bound;
  bound.program = {{HostRangeOp::argument_i32}};
  for (int i = 0; i < 128; ++i)
    bound.program.push_back({HostRangeOp::floor_divide, 1, std::uint64_t(i)});
  std::int32_t value = -123;
  EXPECT_EQ(bound.evaluate(&value, 4), value);
  for (int invalid = 0; invalid < 5; ++invalid) {
    auto bad = bound;
    if (invalid == 0) bad.program.back().lhs = bad.program.size();
    if (invalid == 1) bad.program.back().immediate = 0;
    if (invalid == 2) bad.program.back().op = static_cast<HostRangeOp>(999);
    if (invalid == 3) bad.program[0].offset = UINT64_MAX;
    if (invalid == 4) bad.bias = 1;
    EXPECT_FALSE(bad.valid(4));
    EXPECT_FALSE(bad.evaluate(&value, 4));
  }
  bound.program = {{HostRangeOp::shape_i32}};
  EXPECT_FALSE(bound.evaluate(&value, 4));
}

TEST(HostRangeBounds, DescriptionSharesCompositeDagNodes) {
  const auto *arguments = TypeFactory::get_instance()
      .get_struct_type({{PrimitiveType::i32, "n"}})->as<StructType>();
  Block body;
  auto *n = body.push_back<ArgLoadStmt>(
      std::vector<int>{0}, PrimitiveType::i32, false, true, 0);
  auto *limit = body.push_back<ConstStmt>(TypedConstant(65));
  Stmt *expression = body.push_back<BinaryOpStmt>(BinaryOpType::min, n, limit);
  for (int i = 0; i < 24; ++i)
    expression = body.push_back<BinaryOpStmt>(BinaryOpType::max, expression, expression);
  irpass::type_check(&body, CompileConfig{});
  const auto bound = describe_host_range_bound(expression, arguments);
  ASSERT_TRUE(bound);
  EXPECT_EQ(bound->program.size(), 27);  // Shared nodes are not expanded twice.
  std::int32_t value = 75;
  EXPECT_EQ(bound->evaluate(&value, 4), 65);
  const auto affine = describe_host_range_bound(n, arguments);
  ASSERT_TRUE(affine);
  EXPECT_TRUE(affine->program.empty());
  auto *divisor = body.push_back<ConstStmt>(TypedConstant(0));
  auto *division = body.push_back<BinaryOpStmt>(BinaryOpType::floordiv, expression, divisor);
  division->ret_type = PrimitiveType::i32;
  EXPECT_FALSE(describe_host_range_bound(division, arguments));
}

TEST(HostRangeBounds, ClampedProductsKeepIntermediateOverflowGuards) {
  const auto *arguments = TypeFactory::get_instance()
      .get_struct_type({{PrimitiveType::i32, "width"},
                        {PrimitiveType::i32, "height"}})->as<StructType>();
  Block body;
  auto *width = body.push_back<ArgLoadStmt>(
      std::vector<int>{0}, PrimitiveType::i32, false, true, 0);
  auto *height = body.push_back<ArgLoadStmt>(
      std::vector<int>{1}, PrimitiveType::i32, false, true, 0);
  auto *zero = body.push_back<ConstStmt>(TypedConstant(0));
  auto *w = body.push_back<BinaryOpStmt>(BinaryOpType::max, width, zero);
  auto *h = body.push_back<BinaryOpStmt>(BinaryOpType::max, height, zero);
  auto *product = body.push_back<BinaryOpStmt>(BinaryOpType::mul, w, h);
  auto *limit = body.push_back<ConstStmt>(TypedConstant(65));
  auto *clamped = body.push_back<BinaryOpStmt>(BinaryOpType::min, product, limit);
  auto *hidden = body.push_back<BinaryOpStmt>(BinaryOpType::mul, product, zero);
  irpass::type_check(&body, CompileConfig{});
  const auto bound = describe_host_range_bound(product, arguments);
  const auto clipped = describe_host_range_bound(clamped, arguments);
  const auto cancelled = describe_host_range_bound(hidden, arguments);
  ASSERT_TRUE(bound);
  ASSERT_TRUE(clipped);
  ASSERT_TRUE(cancelled);
  std::int32_t values[]{320, 240};
  EXPECT_EQ(bound->evaluate(values, sizeof(values)), 76800);
  values[0] = -3;
  EXPECT_EQ(bound->evaluate(values, sizeof(values)), 0);
  values[0] = values[1] = 50000;
  EXPECT_FALSE(bound->evaluate(values, sizeof(values)));
  EXPECT_FALSE(clipped->evaluate(values, sizeof(values)));
  EXPECT_FALSE(cancelled->evaluate(values, sizeof(values)));
  values[0] = INT32_MAX;
  values[1] = 1;
  EXPECT_EQ(bound->evaluate(values, sizeof(values)), INT32_MAX);
}

TEST(HostRangeBounds, CodecAndCloneRetainTheFrozenProof) {
  HostRangeBounds host{{0, {}, false}, {1, {8, 12}, true}};
  ASSERT_TRUE(compose_host_range_affine(host.end, 3, -1));
  RangeForBounds bounds{RangeBound::constant(0), RangeBound::global_tmp(4), host,
                        EmptyRangeGroupProof::independent_iterations};
  BinaryOutputSerializer output;
  output.initialize();
  output(bounds);
  output.finalize();
  BinaryInputSerializer input;
  input.initialize(output.data.data());
  RangeForBounds decoded;
  input(decoded);
  ASSERT_TRUE(decoded.host_launch);
  EXPECT_TRUE(*decoded.host_launch == host);
  EXPECT_FALSE(decoded.valid(15));
  EXPECT_TRUE(decoded.valid(16));
  RangeForBounds json;
  liong::json::deserialize(liong::json::serialize(bounds), json, true);
  EXPECT_EQ(json.host_launch, decoded.host_launch);
  EXPECT_EQ(json.empty_group_elision, bounds.empty_group_elision);
  EXPECT_EQ(decoded.empty_group_elision, bounds.empty_group_elision);
  OffloadedStmt task(OffloadedTaskType::range_for, Arch::cuda, nullptr);
  task.host_launch_range = host;
  auto clone = task.clone();
  EXPECT_EQ(clone->as<OffloadedStmt>()->host_launch_range, task.host_launch_range);
}

TEST(HostRangeBounds, KnownWorkloadDoesNotAuthorizeAnUnprovedTransformation) {
  const std::int32_t count = 65;
  RangeForBounds bounds{RangeBound::constant(0), RangeBound::global_tmp(0),
                        HostRangeBounds{{0, {}, false}, {1, {0}, false}}};
  for (auto proof : {EmptyRangeGroupProof::unknown,
                     EmptyRangeGroupProof::auxiliary_region,
                     EmptyRangeGroupProof::physical_thread}) {
    bounds.empty_group_elision = proof;
    EXPECT_TRUE(bounds.valid(sizeof(count)));
    EXPECT_EQ(bounds.host_launch->end.evaluate(&count, sizeof(count)), count);
    EXPECT_EQ(bounds.launch_grid(8160, 128, &count, sizeof(count)), 8160);
    EXPECT_FALSE(bounds.supports_device_dispatch());
    RangeForBounds decoded;
    liong::json::deserialize(liong::json::serialize(bounds), decoded, true);
    EXPECT_EQ(decoded.empty_group_elision, proof);
    EXPECT_EQ(decoded.host_launch, bounds.host_launch);
  }
  bounds.empty_group_elision = EmptyRangeGroupProof::independent_iterations;
  EXPECT_TRUE(bounds.supports_device_dispatch());
  EXPECT_EQ(bounds.launch_grid(8160, 128, &count, sizeof(count)), 1);
  bounds.empty_group_elision = static_cast<EmptyRangeGroupProof>(999);
  EXPECT_FALSE(bounds.valid());
  EXPECT_EQ(bounds.launch_grid(8160, 128, &count, sizeof(count)), 8160);
}

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
    task.begin_stmt = expression.get();
    EXPECT_EQ(describe_range_for_bounds(task)->begin.kind, RangeBoundKind::unknown);
    OffloadedStmt serial(OffloadedTaskType::serial, arch, nullptr);
    EXPECT_FALSE(describe_range_for_bounds(serial));
  }
}

TEST(RangeForBounds, TaskClonePreservesItsOwnedDirectBounds) {
  for (auto arch : {Arch::cuda, Arch::vulkan}) {
    OffloadedStmt task(OffloadedTaskType::range_for, arch, nullptr);
    task.begin_stmt = task.body->push_back<ArgLoadStmt>(
        std::vector<int>{1}, PrimitiveType::i32, false, true, 0);
    task.end_stmt = task.body->push_back<ArgLoadStmt>(
        std::vector<int>{0}, PrimitiveType::i32, false, true, 0);
    auto copy = task.clone();
    auto *cloned = copy->as<OffloadedStmt>();
    ASSERT_NE(cloned->begin_stmt, nullptr);
    EXPECT_NE(cloned->begin_stmt, task.begin_stmt);
    EXPECT_EQ(cloned->begin_stmt->parent, cloned->body.get());
    EXPECT_EQ(describe_range_for_bounds(*cloned)->begin.kind, RangeBoundKind::unknown);
    ASSERT_NE(cloned->end_stmt, nullptr);
    EXPECT_NE(cloned->end_stmt, task.end_stmt);
    EXPECT_EQ(cloned->end_stmt->parent, cloned->body.get());
    EXPECT_EQ(describe_range_for_bounds(*cloned)->end.kind, RangeBoundKind::unknown);
    task.body->statements.clear();
    EXPECT_TRUE(cloned->begin_stmt->is<ArgLoadStmt>());
    EXPECT_TRUE(cloned->end_stmt->is<ArgLoadStmt>());
  }
}

TEST(RangeForBounds, DirectReferencesSurviveSimplificationAndScopedReplacement) {
  for (int pass = 0; pass < 5; ++pass) {
    SCOPED_TRACE(pass);
    auto root = std::make_unique<Block>();
    auto *task = root->push_back<OffloadedStmt>(OffloadedTaskType::range_for,
                                                Arch::cuda, nullptr)->as<OffloadedStmt>();
    auto *input = task->body->push_back<ArgLoadStmt>(
        std::vector<int>{0}, PrimitiveType::i32, false, true, 0);
    auto *zero = task->body->push_back<ConstStmt>(TypedConstant(pass >= 3 ? 2 : 0));
    auto *one = task->body->push_back<ConstStmt>(TypedConstant(1));
    task->begin_stmt = task->body->push_back<BinaryOpStmt>(
        BinaryOpType::add, pass == 1 ? static_cast<Stmt *>(one) : input, zero);
    task->end_stmt = task->begin_stmt;
    CompileConfig config;
    irpass::type_check(root.get(), config);
    if (pass == 0) {
      EXPECT_TRUE(irpass::alg_simp(root.get(), config));
      EXPECT_EQ(task->begin_stmt, input);
    } else if (pass == 1) {
      EXPECT_TRUE(irpass::constant_fold(root.get()));
      EXPECT_TRUE(task->begin_stmt->is<ConstStmt>());
    } else if (pass == 2) {
      // A subtree replacement cannot change a task bound defined elsewhere.
      auto separate = std::make_unique<Block>();
      irpass::replace_all_usages_with(separate.get(), task->begin_stmt, input);
      EXPECT_NE(task->begin_stmt, input);
      task->begin_stmt->replace_usages_with(input);
      EXPECT_EQ(task->begin_stmt, input);
    } else {
      task->end_stmt = task->body->push_back<BinaryOpStmt>(BinaryOpType::add, input, zero);
      irpass::type_check(root.get(), config);
      config.advanced_optimization = pass == 4;
      irpass::full_simplify(root.get(), config, {true, false});
    }
    EXPECT_EQ(task->begin_stmt, task->end_stmt);
    EXPECT_FALSE(task->begin_stmt->erased);
    EXPECT_EQ(task->begin_stmt->parent, task->body.get());
    EXPECT_NO_THROW(irpass::analysis::verify(root.get()));
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
  task.source_block_dim_explicit = true;
  data.metadata.kernel_attribs.tasks_attribs.push_back(task);
  spirv::CompiledKernelData vulkan(Arch::vulkan, std::move(data));
  auto manifest = vulkan.task_manifest();
  ASSERT_EQ(manifest.size(), 1);
  EXPECT_EQ(*manifest[0].constant_range_size, 13);
  EXPECT_EQ(manifest[0].range_bounds->begin.value, -9);
  EXPECT_TRUE(manifest[0].source_block_size_explicit);
  std::ostringstream encoded;
  ASSERT_EQ(vulkan.dump(encoded), CompiledKernelData::Err::kNoError);
  spirv::CompiledKernelData decoded;
  std::istringstream input(encoded.str());
  ASSERT_EQ(decoded.load(input), CompiledKernelData::Err::kNoError);
  EXPECT_EQ(decoded.task_manifest()[0].range_bounds->begin.value, -9);
  EXPECT_TRUE(decoded.task_manifest()[0].source_block_size_explicit);
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
  task.range_bounds->empty_group_elision = EmptyRangeGroupProof::independent_iterations;
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
  decoded.thread_local_bytes = 4;
  EXPECT_FALSE(decoded.valid_device_range_dispatch_contract());
  decoded.range_bounds->empty_group_elision = EmptyRangeGroupProof::compiler_tls_reduction;
  EXPECT_TRUE(decoded.valid_device_range_dispatch_contract());
  decoded.thread_local_bytes = 0;
  EXPECT_FALSE(decoded.valid_device_range_dispatch_contract());
  decoded.range_bounds->empty_group_elision = EmptyRangeGroupProof::independent_iterations;
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
