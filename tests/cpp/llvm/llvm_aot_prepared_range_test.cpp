#include "gtest/gtest.h"

#ifdef TI_WITH_LLVM
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SourceMgr.h"
#include "taichi/runtime/llvm/aot_prepared_range.h"
#include "taichi/codegen/range_for_bounds.h"

namespace taichi::lang {
namespace {

std::unique_ptr<llvm::Module> range_module(llvm::LLVMContext &context,
                                         const std::string &body,
                                         const std::string &extra = "",
                                         const std::string &before = "",
                                         const std::string &prologue = "null") {
  llvm::SMDiagnostic error;
  auto module = llvm::parseAssemblyString(
      "declare void @gpu_parallel_range_for_with_tls(ptr, i32, i32, ptr, ptr, ptr, ptr)\n"
      "define void @entry(ptr %ctx) {\n" + before +
      " call void @gpu_parallel_range_for_with_tls(ptr %ctx, i32 0, i32 1, ptr " +
      prologue + ", ptr @body, ptr null, ptr null)\n ret void\n}\n"
      "define internal void @body(ptr %ctx, ptr %tls, i32 %i) {\n" + body +
      "\n ret void\n}\n" + extra, error, context);
  if (!module)
    error.print("prepared range", llvm::errs());
  EXPECT_NE(module, nullptr);
  if (module)
    EXPECT_FALSE(llvm::verifyModule(*module, &llvm::errs()));
  return module;
}

TEST(AotPreparedRange, LinkedHelperRefinesGridWithoutFreezingLiveBounds) {
  llvm::LLVMContext context;
  auto module = range_module(context, "call void @helper(ptr %ctx, i32 %i)", R"(
    define internal void @helper(ptr %x, i32 %i) {
      %p = getelementptr i32, ptr %x, i32 %i
      %old = atomicrmw add ptr %p, i32 1 monotonic
      ret void
    })");
  ASSERT_TRUE(module);
  AotPreparedRangeAnalysis analysis;
  const auto proof = analysis.inspect(module->getFunction("entry"));
  EXPECT_TRUE(proof.empty_has_no_effects);
  ASSERT_TRUE(proof.independent_iterations);
  RangeForBounds range;
  range.host_launch.emplace();
  range.host_launch->begin.factor = 0;
  range.host_launch->end.offsets = {0};
  range.empty_group_elision = EmptyRangeGroupProof::independent_iterations;
  int count = 1;
  EXPECT_EQ(range.launch_grid(8160, 128, &count, sizeof(count)), 1);
  count = 257;
  EXPECT_EQ(range.launch_grid(8160, 128, &count, sizeof(count)), 3);
  std::vector<std::pair<uint64_t, uint64_t>> mutable_args{{0, sizeof(count)}};
  EXPECT_EQ(range.launch_grid(8160, 128, &count, sizeof(count), &mutable_args), 8160);
}

TEST(AotPreparedRange, UnknownAndPhysicalCallsKeepNonemptyLaunch) {
  for (const auto *name : {"external", "llvm.nvvm.read.ptx.sreg.nctaid.x",
                           "llvm.nvvm.read.ptx.sreg.tid.x"}) {
    llvm::LLVMContext context;
    auto module = range_module(context, "call void @helper()",
        "declare i32 @" + std::string(name) + "()\n"
        "define internal void @helper() {\n %x = call i32 @" + name +
        "()\n ret void\n}\n");
    ASSERT_TRUE(module);
    AotPreparedRangeAnalysis analysis;
    const auto proof = analysis.inspect(module->getFunction("entry"));
    EXPECT_TRUE(proof.empty_has_no_effects);
    EXPECT_FALSE(proof.independent_iterations) << name;
    EXPECT_FALSE(proof.cuda_block_collective) << name;
  }
}

TEST(AotPreparedRange, CollectiveTlsIndirectAndAssemblyStayUnproved) {
  for (const auto &body : {
           "call void @llvm.nvvm.barrier0()",
           "store i32 %i, ptr %tls",
           "%fn = load ptr, ptr %ctx\n call void %fn()",
           "call void asm sideeffect \"\", \"\"()",
           "store i32 %i, ptr addrspacecast (ptr addrspace(3) @shared to ptr)"}) {
    llvm::LLVMContext context;
    auto module = range_module(context, body,
        "declare void @llvm.nvvm.barrier0()\n@shared = addrspace(3) global i32 0\n");
    ASSERT_TRUE(module);
    AotPreparedRangeAnalysis analysis;
    const auto proof = analysis.inspect(module->getFunction("entry"));
    EXPECT_TRUE(proof.empty_has_no_effects);
    EXPECT_FALSE(proof.independent_iterations) << body;
  }
}

TEST(AotPreparedRange, LinkedBlockBarrierUsesOnlyAlignedWholeGroups) {
  llvm::LLVMContext context;
  auto module = range_module(context, "call void @helper(i32 %i)", R"(
    @shared = internal addrspace(3) global [32 x i32] undef
    declare void @llvm.nvvm.barrier0()
    define internal void @helper(i32 %i) {
      %lane = and i32 %i, 31
      %p = getelementptr [32 x i32], ptr addrspace(3) @shared, i32 0, i32 %lane
      store i32 %i, ptr addrspace(3) %p
      call void @llvm.nvvm.barrier0()
      ret void
    })");
  ASSERT_TRUE(module);
  AotPreparedRangeAnalysis analysis;
  const auto proof = analysis.inspect(module->getFunction("entry"));
  EXPECT_TRUE(proof.empty_has_no_effects);
  EXPECT_FALSE(proof.independent_iterations);
  ASSERT_TRUE(proof.cuda_block_collective);
  RangeForBounds range;
  range.host_launch.emplace();
  range.host_launch->begin.factor = 0;
  range.host_launch->end.offsets = {0};
  range.empty_group_elision = EmptyRangeGroupProof::cuda_block_collective;
  int count = 64 * 32;
  EXPECT_EQ(range.launch_grid(8160, 32, &count, sizeof(count)), 64);
  EXPECT_FALSE(range.supports_device_dispatch());
  count = 33;
  EXPECT_EQ(range.launch_grid(8160, 32, &count, sizeof(count)), 8160);
  count = 64 * 32;
  std::vector<std::pair<uint64_t, uint64_t>> mutable_args{{0, sizeof(count)}};
  EXPECT_EQ(range.launch_grid(8160, 32, &count, sizeof(count), &mutable_args), 8160);
}

TEST(AotPreparedRange, LocalBarrierDoesNotAdmitUnknownOrGridCalls) {
  for (const auto *name : {"external", "llvm.nvvm.read.ptx.sreg.nctaid.x"}) {
    llvm::LLVMContext context;
    auto module = range_module(context, "call void @helper()",
        "declare void @llvm.nvvm.barrier0()\n"
        "declare i32 @" + std::string(name) + "()\n"
        "define internal void @helper() {\n call void @llvm.nvvm.barrier0()\n"
        " %x = call i32 @" + name + "()\n ret void\n}\n");
    ASSERT_TRUE(module);
    AotPreparedRangeAnalysis analysis;
    const auto proof = analysis.inspect(module->getFunction("entry"));
    EXPECT_TRUE(proof.empty_has_no_effects);
    EXPECT_FALSE(proof.independent_iterations);
    EXPECT_FALSE(proof.cuda_block_collective);
  }
}

TEST(AotPreparedRange, EffectsOutsideRangeAndAuxiliaryCallbacksCannotBePruned) {
  for (bool auxiliary : {false, true}) {
    llvm::LLVMContext context;
    auto module = range_module(context, "", "declare void @init(ptr, ptr)\n",
                               auxiliary ? "" : "store i32 1, ptr %ctx\n",
                               auxiliary ? "@init" : "null");
    ASSERT_TRUE(module);
    AotPreparedRangeAnalysis analysis;
    const auto proof = analysis.inspect(module->getFunction("entry"));
    EXPECT_FALSE(proof.empty_has_no_effects);
    EXPECT_FALSE(proof.independent_iterations);
  }
}

TEST(AotPreparedRange, ScalarMathDoesNotRequireInlining) {
  llvm::LLVMContext context;
  auto module = range_module(context, "%x = call float @helper(float 2.0)", R"(
    declare float @llvm.nvvm.sqrt.f(float)
    define internal float @helper(float %x) {
      %y = call float @llvm.nvvm.sqrt.f(float %x)
      ret float %y
    })");
  ASSERT_TRUE(module);
  AotPreparedRangeAnalysis analysis;
  EXPECT_TRUE(analysis.inspect(module->getFunction("entry")).independent_iterations);
}

TEST(AotPreparedRange, PreparedBlockBoundKeepsOtherBackendPolicies) {
  llvm::LLVMContext context;
  auto module = range_module(context, "", R"(
    !nvvm.annotations = !{!0, !1, !2, !3}
    !0 = !{ptr @entry, !"maxntidx", i32 1024}
    !1 = !{ptr @entry, !"minctasm", i32 2}
    !2 = !{ptr @entry, !"maxnreg", i32 192}
    !3 = !{ptr @body, !"maxntidx", i32 256}
  )");
  ASSERT_TRUE(module);
  auto &entry = *module->getFunction("entry");
  EXPECT_FALSE(specialize_prepared_cuda_block_bound(entry, 1024, 0));
  EXPECT_FALSE(specialize_prepared_cuda_block_bound(entry, 1024, 2048));
  ASSERT_TRUE(specialize_prepared_cuda_block_bound(entry, 1024, 32));
  auto *annotations = module->getNamedMetadata("nvvm.annotations");
  const int expected[] = {32, 2, 192, 256};
  for (int i = 0; i < 4; ++i)
    EXPECT_EQ(llvm::mdconst::extract<llvm::ConstantInt>(
                  annotations->getOperand(i)->getOperand(2))->getSExtValue(), expected[i]);
  EXPECT_FALSE(llvm::verifyModule(*module, &llvm::errs()));
}

}  // namespace
}  // namespace taichi::lang
#endif
