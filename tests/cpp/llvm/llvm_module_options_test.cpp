#include "gtest/gtest.h"

#ifdef TI_WITH_LLVM
#include <sstream>
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SourceMgr.h"
#include "taichi/codegen/llvm/compiled_kernel_data.h"
#include "taichi/runtime/llvm/llvm_module_options.h"

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
  for (auto arch : {Arch::cuda, Arch::amdgpu}) {
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

}  // namespace taichi::lang
#endif
