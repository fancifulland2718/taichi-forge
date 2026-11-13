#include "gtest/gtest.h"

#ifdef TI_WITH_LLVM
#include <sstream>
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SourceMgr.h"
#include "taichi/codegen/llvm/compiled_kernel_data.h"
#include "taichi/runtime/llvm/llvm_module_options.h"
#include "taichi/runtime/llvm/aot_ir_finalizer.h"

namespace taichi::lang {

TEST(LLVMModuleOptions, O0PromotesPrivateSlotsAfterInlining) {
  llvm::LLVMContext context;
  llvm::SMDiagnostic error;
  auto module = llvm::parseAssemblyString(R"(
    declare void @escape(ptr)
    define internal float @helper(float %x) alwaysinline {
      %slot = alloca float
      store float %x, ptr %slot
      %value = load float, ptr %slot
      ret float %value
    }
    define float @entry(float %x) {
      %dead = alloca i32
      %escaped = alloca i32
      %observable = alloca i32
      store i32 42, ptr %dead
      store i32 3, ptr %escaped
      call void @escape(ptr %escaped)
      store volatile i32 7, ptr %observable
      %v = load volatile i32, ptr %observable
      %h = call float @helper(float %x)
      %sum = fadd float %h, 0.0
      ret float %sum
    }
  )", error, context);
  ASSERT_NE(module, nullptr);
  LLVMOptPipelineOptions options;
  options.opt_level = llvm::OptimizationLevel::O0;
  options.run_post_gep_passes = false;
  run_module_opt_pipeline(*module, nullptr, options);
  EXPECT_FALSE(llvm::verifyModule(*module));
  int allocas = 0, volatile_loads = 0, volatile_stores = 0, adds = 0;
  for (auto &block : *module->getFunction("entry")) {
    for (auto &inst : block) {
      allocas += llvm::isa<llvm::AllocaInst>(inst);
      if (auto *load = llvm::dyn_cast<llvm::LoadInst>(&inst))
        volatile_loads += load->isVolatile();
      if (auto *store = llvm::dyn_cast<llvm::StoreInst>(&inst))
        volatile_stores += store->isVolatile();
      if (inst.getOpcode() == llvm::Instruction::FAdd) {
        ++adds;
        EXPECT_FALSE(inst.getFastMathFlags().any());
      }
      if (auto *call = llvm::dyn_cast<llvm::CallInst>(&inst))
        EXPECT_NE(call->getCalledFunction(), module->getFunction("helper"));
    }
  }
  EXPECT_EQ(allocas, 2);
  EXPECT_EQ(volatile_loads, 1);
  EXPECT_EQ(volatile_stores, 1);
  EXPECT_EQ(adds, 1);
}

TEST(LLVMModuleOptions, KernelOptionsSurviveCloneAndOfflineCache) {
  for (auto arch : {Arch::x64, Arch::arm64, Arch::cuda, Arch::amdgpu}) {
    for (const std::string tier : {"fast", "full"}) {
      SCOPED_TRACE(tier);
      CompileConfig kernel;
      kernel.arch = arch;
      kernel.compile_tier = tier;
      kernel.fast_math = false;
      const auto expected = LLVMModuleOptions::from_config(kernel);
      llvm::LLVMContext context;
      LLVM::CompiledKernelData::InternalData data;
      data.compiled_data.module =
          std::make_unique<llvm::Module>("kernel", context);
      expected.write(*data.compiled_data.module);
      LLVM::CompiledKernelData original(arch, std::move(data));
      auto clone = original.clone();
      std::stringstream stream;
      ASSERT_EQ(clone->dump(stream), CompiledKernelData::Err::kNoError);
      LLVM::CompiledKernelData loaded;
      ASSERT_EQ(loaded.load(stream), CompiledKernelData::Err::kNoError);
      CompileConfig changed_program;
      changed_program.arch = arch;
      changed_program.compile_tier = tier == "fast" ? "full" : "fast";
      changed_program.fast_math = true;
      const auto actual = LLVMModuleOptions::read(
          *loaded.get_internal_data().compiled_data.module, changed_program);
      EXPECT_EQ(actual.opt_level, tier == "fast" ? 1 : 3);
      EXPECT_FALSE(actual.fast_math);
    }
  }
}

TEST(LLVMModuleOptions, RuntimeAndLegacyModulesUseProgramOptions) {
  llvm::LLVMContext context;
  llvm::Module module("runtime", context);
  CompileConfig config;
  config.arch = Arch::cuda;
  config.compile_tier = "fast";
  EXPECT_EQ(LLVMModuleOptions::read(module, config).opt_level, 1);
  config.compile_tier = "balanced";
  config.llvm_opt_level = 2;
  EXPECT_EQ(LLVMModuleOptions::read(module, config).opt_level, 2);
}

TEST(LLVMModuleOptions, CpuFastTierEliminatesPrivateAggregateTraffic) {
  llvm::LLVMContext context;
  llvm::SMDiagnostic error;
  auto module = llvm::parseAssemblyString(R"(
    define float @entry(float %x, float %y, i1 %choose) {
      %slot = alloca [2 x float]
      %first = getelementptr [2 x float], ptr %slot, i64 0, i64 0
      %second = getelementptr [2 x float], ptr %slot, i64 0, i64 1
      store float %x, ptr %first
      store float %y, ptr %second
      br i1 %choose, label %a, label %b
    a:
      %va = load float, ptr %first
      ret float %va
    b:
      %vb = load float, ptr %second
      ret float %vb
    }
  )", error, context);
  ASSERT_NE(module, nullptr);
  CompileConfig config;
  config.arch = Arch::x64;
  config.compile_tier = "fast";
  LLVMOptPipelineOptions options;
  options.opt_level = llvm_opt_level_from_int(
      LLVMModuleOptions::from_config(config).opt_level);
  run_module_opt_pipeline(*module, nullptr, options);
  EXPECT_FALSE(llvm::verifyModule(*module));
  for (auto &block : *module->getFunction("entry")) {
    for (auto &inst : block) {
      EXPECT_FALSE(llvm::isa<llvm::AllocaInst>(inst));
      EXPECT_FALSE(inst.mayReadOrWriteMemory());
    }
  }
  config.compile_tier = "balanced";
  config.llvm_opt_level = 0;
  EXPECT_EQ(LLVMModuleOptions::from_config(config).opt_level, 0);
}

TEST(LLVMAotFinalization, SharesUnitsAndPrunesDeadTransitiveDependencies) {
  llvm::LLVMContext context;
  llvm::SMDiagnostic error;
  auto original = llvm::parseAssemblyString(R"(
    target datalayout = "e-p:64:64-i64:64-n16:32:64"
    target triple = "nvptx64-nvidia-cuda"
    declare void @observable(i32)
    define internal i32 @leaf(i32 %x) "taichi.aot.real_function" {
      %s = alloca i32
      store i32 %x, ptr %s
      %v = load i32, ptr %s
      %r = add i32 %v, 1
      ret i32 %r
    }
    define internal void @parent(i32 %x) "taichi.aot.real_function" {
      br i1 false, label %dead, label %live
    dead:
      %r = call i32 @leaf(i32 %x)
      call void @observable(i32 %r)
      br label %live
    live:
      call void @observable(i32 %x)
      ret void
    }
    define void @entry(i32 %x) {
      call void @parent(i32 %x)
      ret void
    }
  )",
                                            error, context);
  ASSERT_NE(original, nullptr);
  LLVM::AotIRFinalizer finalizer;
  auto first = llvm::CloneModule(*original);
  finalizer.export_units(*first, true, false);  // streaming seal
  auto second = llvm::CloneModule(*original);
  finalizer.export_units(*second, true, true);  // ordinary dump
  EXPECT_EQ(finalizer.statistics().at("aot_cleaned_units"), 2);
  EXPECT_EQ(finalizer.statistics().at("aot_unit_finalization_hits"), 2);
  finalizer.finalize_root(first, true);
  finalizer.finalize_root(second, true);
  EXPECT_EQ(finalizer.statistics().at("aot_cleaned_roots"), 1);
  EXPECT_EQ(finalizer.statistics().at("aot_root_finalization_hits"), 1);
  ASSERT_NE(first->getNamedMetadata(LLVM::kCodeUnits), nullptr);
  EXPECT_EQ(first->getNamedMetadata(LLVM::kCodeUnits)->getNumOperands(), 1);
  int emitted = 0;
  finalizer.emit_dependencies(*first, [&](const auto &name, const auto &bytes) {
    ++emitted;
    EXPECT_EQ(name, "code_units/" + LLVM::aot_artifact_sha256(bytes) + ".bc");
    auto decoded = decode_llvm_module(llvm::MemoryBufferRef(bytes, name),
                                      context, LLVMModuleEncoding::bitcode);
    ASSERT_TRUE(bool(decoded));
    EXPECT_FALSE(llvm::verifyModule(**decoded));
    EXPECT_EQ((*decoded)->getDataLayout(), first->getDataLayout());
    EXPECT_NE((*decoded)->getNamedMetadata("taichi.aot.cleanup.v1"), nullptr);
    for (const auto &f : **decoded)
      if (f.isDeclaration() && !f.use_empty())
        EXPECT_EQ(f.getName(), "observable");
  });
  EXPECT_EQ(emitted, 1);
  EXPECT_FALSE(llvm::verifyModule(*first));
  EXPECT_FALSE(original->getFunction("leaf")->isDeclaration());
  EXPECT_TRUE(llvm::isa<llvm::AllocaInst>(
      original->getFunction("leaf")->front().front()));
  EXPECT_EQ(original->getNamedMetadata("taichi.aot.cleanup.v1"), nullptr);
}

TEST(LLVMAotFinalization, PreservesEffectsNumericFlagsAndDisabledOptimization) {
  llvm::LLVMContext context;
  llvm::SMDiagnostic error;
  auto source = llvm::parseAssemblyString(R"(
    target datalayout = "e-p:64:64-i64:64-n16:32:64"
    declare void @escape(ptr)
    define float @entry(float %x, ptr %device, i32 %n) {
      %slot = alloca float
      %index = alloca i32
      %escaped = alloca i32
      store float %x, ptr %slot
      store i32 0, ptr %index
      call void @escape(ptr %escaped)
      br label %loop
    loop:
      %i = load i32, ptr %index
      %more = icmp slt i32 %i, %n
      br i1 %more, label %body, label %end
    body:
      %next = add i32 %i, 1
      store i32 %next, ptr %index
      br label %loop
    end:
      %v = load float, ptr %slot
      %iterations = sitofp i32 %i to float
      %sum = fadd float %v, %iterations
      store volatile float %sum, ptr %device
      %old = atomicrmw add ptr %escaped, i32 1 monotonic
      ret float %sum
    }
    !forge.contract = !{!0}
    !0 = !{!"preserve"}
  )",
                                          error, context);
  ASSERT_NE(source, nullptr);
  LLVM::AotIRFinalizer finalizer;
  auto disabled = llvm::CloneModule(*source);
  finalizer.finalize_root(disabled, false);
  EXPECT_TRUE(llvm::isa<llvm::AllocaInst>(
      disabled->getFunction("entry")->front().front()));
  EXPECT_EQ(disabled->getNamedMetadata("taichi.aot.cleanup.v1"), nullptr);
  auto cleaned = llvm::CloneModule(*source);
  finalizer.finalize_root(cleaned, true);
  EXPECT_FALSE(llvm::verifyModule(*cleaned));
  EXPECT_NE(cleaned->getNamedMetadata("forge.contract"), nullptr);
  int allocas = 0, atomics = 0, volatile_stores = 0, additions = 0, calls = 0,
      phis = 0;
  for (auto &b : *cleaned->getFunction("entry"))
    for (auto &i : b) {
      allocas += llvm::isa<llvm::AllocaInst>(i);
      atomics += llvm::isa<llvm::AtomicRMWInst>(i);
      calls += llvm::isa<llvm::CallInst>(i);
      phis += llvm::isa<llvm::PHINode>(i);
      if (auto *store = llvm::dyn_cast<llvm::StoreInst>(&i))
        volatile_stores += store->isVolatile();
      if (i.getOpcode() == llvm::Instruction::FAdd) {
        ++additions;
        EXPECT_FALSE(i.getFastMathFlags().any());
      }
    }
  EXPECT_EQ(allocas, 1);
  EXPECT_EQ(atomics, 1);
  EXPECT_EQ(volatile_stores, 1);
  EXPECT_EQ(additions, 1);
  EXPECT_EQ(calls, 1);
  EXPECT_GT(phis, 0);
  EXPECT_EQ(finalizer.statistics().at("aot_cleaned_roots"), 1);
}

}  // namespace taichi::lang
#endif
