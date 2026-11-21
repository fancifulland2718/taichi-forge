#include "gtest/gtest.h"

#ifdef TI_WITH_LLVM
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SourceMgr.h"
#include "taichi/runtime/llvm/aot_cpu_private_frames.h"
#include "taichi/runtime/llvm/aot_prepared_range.h"
#include "taichi/runtime/llvm/aot_prepared_local_tensor.h"
#include "taichi/runtime/llvm/aot_prepared_schedule.h"
#include "taichi/codegen/range_for_bounds.h"

namespace taichi::lang {
namespace {

TEST(AotPreparedCpu, PromotesMixedPrivateWordsAndRetainsEscapedStorage) {
  llvm::LLVMContext context;
  llvm::SMDiagnostic error;
  auto module = llvm::parseAssemblyString(R"(
    declare void @observe(ptr)
    define <2 x i32> @local(<2 x i32> %values, i32 %last) {
      %frame = alloca [3 x i32], align 8
      %middle = getelementptr i32, ptr %frame, i64 1
      %end = getelementptr i32, ptr %frame, i64 2
      store <2 x i32> %values, ptr %frame, align 8
      store i32 %last, ptr %end, align 4
      %result = load <2 x i32>, ptr %middle, align 4
      ret <2 x i32> %result
    }
    define i32 @escaped(ptr %out, <2 x i32> %values) {
      %frame = alloca [2 x i32], align 8
      store <2 x i32> %values, ptr %frame, align 8
      call void @observe(ptr %frame)
      %result = load i32, ptr %frame, align 4
      store volatile <2 x i32> %values, ptr %out, align 4
      ret i32 %result
    }
  )", error, context);
  ASSERT_NE(module, nullptr);
  auto &escaped = module->getFunction("escaped")->getEntryBlock();
  const auto original = escaped.size();
  LLVM::promote_cpu_private_frames(*module);
  ASSERT_FALSE(llvm::verifyModule(*module));
  const auto &local = module->getFunction("local")->getEntryBlock();
  EXPECT_FALSE(llvm::any_of(local, [](const auto &inst) {
    return inst.mayReadOrWriteMemory() || llvm::isa<llvm::AllocaInst>(inst);
  }));
  EXPECT_EQ(escaped.size(), original);
  EXPECT_TRUE(llvm::isa<llvm::AllocaInst>(escaped.front()));
}

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

TEST(AotPreparedRange, LocalTensorRequiresAnExclusiveBoundedRow) {
  enum Scenario { local, public_array, outside_row, escape, another_task,
                  short_outer_extent, oversized_frame, unknown_stride,
                  wrapped_offsets, wrong_step, live_index, atomic_access };
  for (const auto width : {1, 4, 7}) {
    for (const auto scenario : {local, public_array, outside_row, escape,
                                another_task, short_outer_extent,
                                oversized_frame, unknown_stride, wrapped_offsets,
                                wrong_step, live_index, atomic_access}) {
      SCOPED_TRACE(::testing::Message() << "width " << width << " scenario " << scenario);
      llvm::LLVMContext context;
      const int bytes = scenario == oversized_frame ? 2048 : width * 4;
      std::string body = fmt::format(R"(
        entry:
          %args = load ptr, ptr %ctx
          %slot = getelementptr i8, ptr %args, i64 8
          %array = load ptr, ptr %slot
          %iteration = sext i32 %i to i64
          %row = mul i64 %iteration, {}
          br label %test
        test:
          %index = phi i32 [0, %entry], [%next, %loop]
          %more = icmp slt i32 %index, {}
          br i1 %more, label %loop, label %done
        loop:
          %j = sext i32 %index to i64
          %inner = mul i64 %j, 4
          %offset = add i64 %row, %inner
          %p = getelementptr i8, ptr %array, i64 %offset
          {}
          %next = add i32 %index, {}
          br label %test
        done:
      )", scenario == unknown_stride ? bytes + 4 : bytes,
                       scenario == outside_row ? width + 1 : width,
                       scenario == wrapped_offsets ? R"(
                          %past = getelementptr i8, ptr %p, i64 -4
                          %again = getelementptr i8, ptr %past, i64 4
                          store i32 %index, ptr %again, align 4)"
                       : scenario == live_index ? R"(
                          %live = getelementptr i8, ptr %p, i64 %iteration
                          store i32 %index, ptr %live, align 1)"
                       : scenario == atomic_access
                           ? "store atomic i32 %index, ptr %p monotonic, align 4"
                           : "store i32 %index, ptr %p, align 4",
                       scenario == wrong_step ? 2 : 1);
      if (scenario == escape)
        body += "call void @observe(ptr %array)\n";
      auto module = range_module(context, body, R"(
        declare void @observe(ptr)
        define void @read_other_task(ptr %ctx) {
          %args = load ptr, ptr %ctx
          %slot = getelementptr i8, ptr %args, i64 8
          %array = load ptr, ptr %slot
          %seen = load i32, ptr %array
          ret void
        })");
      ASSERT_TRUE(module);
      LLVM::prepared_local_tensor::Workspaces spaces;
      if (scenario != public_array)
        spaces.emplace(8, LLVM::prepared_local_tensor::Workspace{
                              static_cast<unsigned>(bytes),
                              scenario == short_outer_extent ? 0u : 1u});
      std::vector<OffloadedTask> tasks(1);
      tasks.front().name = "entry";
      if (scenario == another_task)
        tasks.emplace_back().name = "read_other_task";
      LLVM::specialize_prepared_local_tensors(*module, tasks, spaces);
      bool has_private = false;
      for (auto &inst : module->getFunction("body")->getEntryBlock())
        if (auto *storage = llvm::dyn_cast<llvm::AllocaInst>(&inst)) {
          has_private = true;
          EXPECT_EQ(storage->getAllocatedType()->getArrayNumElements(), bytes);
        }
      EXPECT_EQ(has_private, scenario == local);
      EXPECT_FALSE(llvm::verifyModule(*module, &llvm::errs()));
    }
  }
}

TEST(AotPreparedRange, ForwardsPrivateBoundsAndRetainsObservedWrites) {
  enum Scenario {
    empty,
    observed,
    missing_inventory,
    volatile_write,
    live_value,
    intervening_write,
    conflicting,
    nonempty
  };
  for (auto scenario : {empty, observed, missing_inventory, volatile_write,
                        live_value, intervening_write, conflicting, nonempty}) {
    llvm::LLVMContext context;
    auto module =
        range_module(context, "store i32 %i, ptr %ctx", R"(
      declare ptr @RuntimeContext_get_temporary_pointer(ptr, i64)
      define void @writer(ptr %ctx) !taichi.scratch.slots !1 {
        %live = load i32, ptr %ctx
        %tmp = call ptr @RuntimeContext_get_temporary_pointer(ptr %ctx, i64 4)
        store i32 0, ptr %tmp, !taichi.scratch.store !1
        %again = call ptr @RuntimeContext_get_temporary_pointer(ptr %ctx, i64 4)
        store i32 0, ptr %again, !taichi.scratch.store !1
        ret void
      }
      define void @observe(ptr %ctx) {
        store i32 7, ptr %ctx
        ret void
      }
      define void @mutate(ptr %ctx) !taichi.scratch.slots !1 {
        %live = load i32, ptr %ctx
        %tmp = call ptr @RuntimeContext_get_temporary_pointer(ptr %ctx, i64 4)
        store i32 %live, ptr %tmp, !taichi.scratch.store !1
        ret void
      }
      !0 = !{}
      !1 = !{i32 4}
    )",
                     "%tmp = call ptr "
                     "@RuntimeContext_get_temporary_pointer(ptr %ctx, i64 4)\n"
                     "%end = load i32, ptr %tmp\n");
    ASSERT_TRUE(module);
    auto *writer = module->getFunction("writer");
    auto *entry = module->getFunction("entry");
    auto *observe = module->getFunction("observe");
    auto *slot = writer->getMetadata("taichi.scratch.slots");
    auto *none = llvm::MDNode::get(context, {});
    if (scenario != missing_inventory)
      entry->setMetadata("taichi.scratch.slots", none);
    observe->setMetadata("taichi.scratch.slots",
                         scenario == observed ? slot : none);
    auto *dispatch = LLVM::prepared_range_dispatch(*entry);
    ASSERT_TRUE(dispatch);
    dispatch->setArgOperand(2, dispatch->getPrevNode());
    auto *last = llvm::cast<llvm::StoreInst>(
        writer->front().getTerminator()->getPrevNode());
    if (scenario == volatile_write)
      last->setVolatile(true);
    if (scenario == live_value)
      last->setOperand(0, &writer->front().front());
    if (scenario == conflicting || scenario == nonempty)
      last->setOperand(
          0, llvm::ConstantInt::get(last->getValueOperand()->getType(), 128));
    if (scenario == nonempty)
      for (auto &inst : writer->front())
        if (auto *store = llvm::dyn_cast<llvm::StoreInst>(&inst))
          store->setOperand(0, last->getValueOperand());
    std::vector<OffloadedTask> tasks(3);
    tasks[0].name = "writer";
    tasks[1].name = "entry";
    tasks[2].name = "observe";
    tasks[1].task_type = OffloadedTaskType::range_for;
    auto &range = tasks[1].range_bounds.emplace();
    range.begin = {RangeBoundKind::constant_i32, 0};
    range.end = {RangeBoundKind::global_tmp_i32, 4};
    if (scenario == intervening_write) {
      OffloadedTask mutation;
      mutation.name = "mutate";
      tasks.insert(tasks.begin() + 1, mutation);
    }
    LLVM::specialize_prepared_task_schedule(*module, tasks);
    if (scenario == intervening_write) {
      ASSERT_GE(tasks.size(), 3u);
      EXPECT_EQ(tasks[tasks.size() - 3].name, "mutate");
      const auto &active = tasks[tasks.size() - 2];
      ASSERT_EQ(active.name, "entry");
      EXPECT_EQ(active.range_bounds->end.kind, RangeBoundKind::global_tmp_i32);
      EXPECT_FALSE(llvm::isa<llvm::ConstantInt>(dispatch->getArgOperand(2)));
      EXPECT_FALSE(llvm::verifyModule(*module, &llvm::errs()));
      continue;
    }
    EXPECT_EQ(tasks.size(), scenario == empty ? 1u
                            : scenario == observed || scenario == nonempty
                                ? 2u
                                : 3u);
    EXPECT_EQ(tasks.back().name, "observe");
    if (scenario == nonempty) {
      ASSERT_EQ(tasks.front().name, "entry");
      EXPECT_EQ(tasks.front().range_bounds->constant_size(), 128);
      EXPECT_EQ(llvm::cast<llvm::ConstantInt>(dispatch->getArgOperand(2))
                    ->getSExtValue(),
                128);
    }
    EXPECT_FALSE(llvm::verifyModule(*module, &llvm::errs()));
  }
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

TEST(AotPreparedRange, LinkedWarpShuffleAndBarrierUseOnlyAlignedWholeGroups) {
  llvm::LLVMContext context;
  auto module = range_module(context, "call void @helper(i32 %i)", R"(
    @shared = internal addrspace(3) global [32 x i32] undef
    declare void @llvm.nvvm.barrier0()
    declare i32 @llvm.nvvm.shfl.sync.down.i32(i32, i32, i32, i32)
    define internal void @helper(i32 %i) {
      %lane = and i32 %i, 31
      %p = getelementptr [32 x i32], ptr addrspace(3) @shared, i32 0, i32 %lane
      %v = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %i, i32 1, i32 31)
      store i32 %v, ptr addrspace(3) %p
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
