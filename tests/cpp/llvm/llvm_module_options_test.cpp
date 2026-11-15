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
#include "taichi/runtime/llvm/function_code_cache.h"

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
  EXPECT_EQ(finalizer.statistics().at("aot_unit_snapshot_hits"), 2);
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

TEST(LLVMAotFinalization, SnapshotReuseChecksFullInputAndCanonicalFallback) {
  LLVM::AotIRFinalizer finalizer;
  const auto build = [&](const std::string &type_name, int bias, int abi,
                         bool optimize) {
    llvm::LLVMContext context;
    llvm::SMDiagnostic error;
    auto module = llvm::parseAssemblyString(
        "target datalayout = \"e-p:64:64-i64:64-n16:32:64\"\n%" +
            type_name + " = type { i32 }\n"
            "define internal i32 @leaf(ptr %data) \"taichi.aot.real_function\" {\n"
            "  %p = getelementptr %" + type_name + ", ptr %data, i32 0, i32 0\n"
            "  %v = load volatile i32, ptr %p\n"
            "  %r = add i32 %v, " + std::to_string(bias) + "\n"
            "  ret i32 %r\n}\n"
            "define i32 @entry(ptr %data) {\n"
            "  %r = call i32 @leaf(ptr %data)\n  ret i32 %r\n}\n",
        error, context);
    EXPECT_NE(module, nullptr);
    if (!module)
      return std::string{};
    module->addModuleFlag(llvm::Module::Error, "forge.test.abi", abi);
    finalizer.export_units(*module, optimize, false);
    finalizer.finalize_root(module, optimize);
    EXPECT_FALSE(llvm::verifyModule(*module));
    std::string result;
    finalizer.emit_dependencies(*module, [&](const auto &, const auto &bytes) {
      result += bytes;
    });
    EXPECT_FALSE(result.empty());
    return result;
  };
  const auto first = build("Capture", 3, 1, true);
  EXPECT_EQ(first, build("Capture", 3, 1, true));
  EXPECT_EQ(finalizer.statistics().at("aot_unit_snapshot_hits"), 1);
  // Different context-local type names miss the exact snapshot, but still
  // converge through the original canonical normalization and policy key.
  EXPECT_EQ(first, build("Capture.123", 3, 1, true));
  EXPECT_EQ(finalizer.statistics().at("aot_unit_snapshot_hits"), 1);
  EXPECT_EQ(finalizer.statistics().at("aot_unit_finalization_hits"), 2);
  EXPECT_EQ(first, build("Capture.123", 3, 1, true));
  EXPECT_EQ(finalizer.statistics().at("aot_unit_snapshot_hits"), 2);
  EXPECT_NE(first, build("Capture", 7, 1, true));
  EXPECT_NE(first, build("Capture", 3, 2, true));
  EXPECT_NE(first, build("Capture", 3, 1, false));
  EXPECT_EQ(finalizer.statistics().at("aot_unit_snapshot_hits"), 2);
  EXPECT_EQ(finalizer.statistics().at("aot_cleaned_units"), 3);
}

TEST(LLVMAotFinalization, ConsumingExtractionMatchesImmutableRecording) {
  llvm::LLVMContext context;
  llvm::SMDiagnostic error;
  auto original = llvm::parseAssemblyString(R"(
    target datalayout = "e-p:64:64-i64:64-n16:32:64"
    target triple = "nvptx64-nvidia-cuda"
    @bias = internal constant i32 7
    declare i32 @personality(...)
    declare void @observable(i32)
    define internal i32 @shared(i32 %x) {
      %b = load i32, ptr @bias
      %r = add i32 %x, %b
      ret i32 %r
    }
    define internal fastcc i32 @leaf(ptr nonnull align 4 %out, i32 %n)
        "taichi.aot.real_function" "target-features"="+ptx75"
        prefix i32 3 prologue i32 11 personality ptr @personality
        !forge.contract !0 {
    start:
      br label %loop
    loop:
      %i = phi i32 [ 0, %start ], [ %next, %body ]
      %more = icmp slt i32 %i, %n
      br i1 %more, label %body, label %end
    body:
      %v = call i32 @shared(i32 %i)
      store volatile i32 %v, ptr %out, align 4, !forge.effect !0
      %old = atomicrmw add ptr %out, i32 1 monotonic
      %next = add i32 %i, 1
      br label %loop
    end:
      call void @observable(i32 %i)
      ret i32 %i
    }
    define i32 @entry(ptr %out, i32 %n) {
      %r = call fastcc i32 @leaf(ptr nonnull align 4 %out, i32 %n)
      %s = call i32 @shared(i32 %r)
      ret i32 %s
    }
    !llvm.module.flags = !{!1}
    !0 = !{!"effect", ptr @shared}
    !1 = !{i32 1, !"forge.test.abi", i32 3}
  )", error, context);
  ASSERT_NE(original, nullptr);
  ASSERT_FALSE(llvm::verifyModule(*original));
  const auto serialize = [](const llvm::Module &module) {
    std::string bytes;
    llvm::raw_string_ostream stream(bytes);
    encode_llvm_module(module, stream, LLVMModuleEncoding::bitcode);
    return bytes;
  };
  const auto before = serialize(*original);
  std::map<std::string, std::string> recorded, consumed;
  std::string recorded_symbol, consumed_symbol;
  LLVM::export_code_units(
      *original,
      [&](const auto &name, const auto &bytes) { recorded[name] = bytes; },
      true, original->getFunction("leaf"), false, &recorded_symbol);
  EXPECT_EQ(before, serialize(*original));
  auto copy = llvm::CloneModule(*original);
  LLVM::export_code_units(
      *copy,
      [&](const auto &name, const auto &bytes) { consumed[name] = bytes; },
      true, copy->getFunction("leaf"), true, &consumed_symbol);
  ASSERT_EQ(recorded.size(), 1);
  EXPECT_EQ(recorded, consumed);
  EXPECT_EQ(recorded_symbol, consumed_symbol);
  EXPECT_FALSE(llvm::verifyModule(*copy));
  EXPECT_TRUE(copy->getFunction(consumed_symbol)->isDeclaration());
  // This helper is still used by the caller, so extracting the leaf must
  // neither consume nor redirect its original body/constant dependencies.
  EXPECT_FALSE(copy->getFunction("shared")->isDeclaration());
  EXPECT_TRUE(copy->getNamedGlobal("bias")->hasInitializer());
  for (const auto &[name, bytes] : consumed) {
    llvm::LLVMContext unit_context;
    auto decoded = decode_llvm_module(llvm::MemoryBufferRef(bytes, name),
                                      unit_context, LLVMModuleEncoding::bitcode);
    ASSERT_TRUE(bool(decoded));
    EXPECT_FALSE(llvm::verifyModule(**decoded));
    auto *leaf = (*decoded)->getFunction(consumed_symbol);
    ASSERT_NE(leaf, nullptr);
    EXPECT_EQ(leaf->getCallingConv(), llvm::CallingConv::Fast);
    EXPECT_TRUE(leaf->hasPersonalityFn());
    EXPECT_TRUE(leaf->hasPrefixData());
    EXPECT_TRUE(leaf->hasPrologueData());
    EXPECT_NE(leaf->getMetadata("forge.contract"), nullptr);
  }
}

TEST(LLVMAotFinalization, RejectedEarlyRelocationsLeaveBodiesIntact) {
  llvm::LLVMContext context;
  llvm::SMDiagnostic error;
  auto module = llvm::parseAssemblyString(R"(
    @__ti_aot_tree_id_1 = external constant i32
    define internal i32 @leaf() "taichi.aot.real_function" {
      %r = load i32, ptr @__ti_aot_tree_id_1
      ret i32 %r
    }
    define i32 @entry() {
      %r = call i32 @leaf()
      ret i32 %r
    }
  )", error, context);
  ASSERT_NE(module, nullptr);
  int emitted = 0;
  LLVM::export_code_units(*module, [&](const auto &, const auto &) { ++emitted; },
                         false);
  EXPECT_EQ(emitted, 0);
  EXPECT_FALSE(module->getFunction("leaf")->isDeclaration());
  EXPECT_FALSE(llvm::verifyModule(*module));
  LLVM::export_code_units(*module, [&](const auto &, const auto &) { ++emitted; });
  EXPECT_EQ(emitted, 1);
  EXPECT_EQ(module->getFunction("leaf"), nullptr);
  EXPECT_FALSE(llvm::verifyModule(*module));
}

TEST(LLVMFunctionCodeCache, SharesIndependentDependenciesWithoutChangingProducer) {
  llvm::LLVMContext context;
  llvm::SMDiagnostic error;
  auto source = llvm::parseAssemblyString(R"(
    target datalayout = "e-p:64:64-i64:64-n16:32:64"
    define internal i32 @leaf(i32 %x) "taichi.aot.real_function" {
      %r = add i32 %x, 1
      ret i32 %r
    }
    define internal i32 @a(i32 %x) "taichi.aot.real_function" {
      %r = call i32 @leaf(i32 %x)
      %v = add i32 %r, 2
      ret i32 %v
    }
    define internal i32 @b(i32 %x) "taichi.aot.real_function" {
      %r = call i32 @leaf(i32 %x)
      %v = add i32 %r, 3
      ret i32 %v
    }
  )", error, context);
  ASSERT_NE(source, nullptr);
  const auto serialize = [](const llvm::Module &module) {
    std::string bytes;
    llvm::raw_string_ostream stream(bytes);
    encode_llvm_module(module, stream, LLVMModuleEncoding::bitcode);
    return bytes;
  };
  const auto original = serialize(*source);
  LLVM::FunctionCodeCache cache;
  auto a = cache.record("a", *source, source->getFunction("a"));
  auto b = cache.record("b", *source, source->getFunction("b"));
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);
  ASSERT_EQ(a->units.size(), 2);
  ASSERT_EQ(b->units.size(), 2);
  EXPECT_EQ(a->units.front(), b->units.front());
  EXPECT_EQ(original, serialize(*source));
  auto leaf_record = cache.record("leaf", *source, source->getFunction("leaf"));
  LLVM::register_function_code(*source, *leaf_record, source->getFunction("leaf"));
  const auto registered = serialize(*source);
  cache.record("a_registered", *source, source->getFunction("a"));
  cache.record("b_registered", *source, source->getFunction("b"));
  EXPECT_EQ(registered, serialize(*source));
  EXPECT_EQ(cache.statistics().at("llvm_function_unit_record_reuses"), 2);
  EXPECT_EQ(cache.statistics().at("cached_llvm_function_units"), 3);
  for (bool optimize : {false, true}) {
    auto consumer = std::make_unique<llvm::Module>("consumer", context);
    consumer->setDataLayout(source->getDataLayout());
    auto *first = LLVM::import_function_code(*consumer, *a);
    auto *second = LLVM::import_function_code(*consumer, *b);
    EXPECT_NE(first, second);
    EXPECT_EQ(consumer->size(), 3);
    EXPECT_EQ(second, LLVM::import_function_code(*consumer, *b));
    EXPECT_EQ(consumer->size(), 3);
    EXPECT_FALSE(llvm::verifyModule(*consumer));
    LLVM::AotIRFinalizer finalizer;
    finalizer.export_units(*consumer, optimize, false);
    EXPECT_EQ(finalizer.statistics().at("shared_code_units"), 3);
    EXPECT_FALSE(llvm::verifyModule(*consumer));
  }
  // A just-emitted leaf can be registered without mutating its body. Importing
  // a cached parent then uses that same definition rather than a second leaf.
  auto local = llvm::CloneModule(*source);
  local->getFunction("a")->eraseFromParent();
  local->getFunction("b")->eraseFromParent();
  auto leaf = cache.record("leaf", *local, local->getFunction("leaf"));
  LLVM::register_function_code(*local, *leaf, local->getFunction("leaf"));
  LLVM::import_function_code(*local, *b);
  EXPECT_EQ(local->size(), 2);
  EXPECT_FALSE(llvm::verifyModule(*local));
  local->setDataLayout("e-p:32:32");
  EXPECT_ANY_THROW(LLVM::import_function_code(*local, *b));
  auto conflicting = llvm::CloneModule(*source);
  conflicting->getFunction("a")->setName(a->units.front()->symbol);
  EXPECT_ANY_THROW(LLVM::import_function_code(*conflicting, *a));
}

TEST(LLVMFunctionCodeCache, SeparatesChangedBodiesAndRejectsUnsupportedRoots) {
  llvm::LLVMContext context;
  llvm::SMDiagnostic error;
  auto source = llvm::parseAssemblyString(R"(
    target datalayout = "e-p:64:64-i64:64-n16:32:64"
    @state = internal global i32 1
    define internal i32 @leaf(i32 %x) "taichi.aot.real_function" {
      %r = add i32 %x, 1
      ret i32 %r
    }
    define internal i32 @parent(i32 %x) "taichi.aot.real_function" {
      %r = call i32 @leaf(i32 %x)
      ret i32 %r
    }
    define internal i32 @unsupported(i32 %x) "taichi.aot.real_function" {
      %r = call i32 @leaf(i32 %x)
      %s = load i32, ptr @state
      %v = add i32 %r, %s
      ret i32 %v
    }
  )", error, context);
  ASSERT_NE(source, nullptr);
  LLVM::FunctionCodeCache cache;
  EXPECT_EQ(cache.record("unsupported", *source,
                         source->getFunction("unsupported")), nullptr);
  EXPECT_EQ(cache.find("unsupported"), nullptr);
  const auto first = cache.record("first", *source, source->getFunction("parent"));
  auto changed = llvm::CloneModule(*source);
  changed->getFunction("leaf")->front().front().setOperand(
      1, llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), 9));
  const auto second = cache.record("second", *changed,
                                   changed->getFunction("parent"));
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  EXPECT_NE(first->units.front()->digest, second->units.front()->digest);
  EXPECT_NE(first->symbol, second->symbol);
  llvm::Module consumer("consumer", context);
  consumer.setDataLayout(source->getDataLayout());
  LLVM::import_function_code(consumer, *first);
  LLVM::import_function_code(consumer, *second);
  EXPECT_EQ(consumer.size(), 4);
  EXPECT_FALSE(llvm::verifyModule(consumer));

  // A proof for the existing leaf does not make an incompatible declaration
  // (or calls already using that declaration's calling convention) compatible.
  const auto leaf = cache.record("leaf", *source, source->getFunction("leaf"));
  LLVM::register_function_code(*source, *leaf, source->getFunction("leaf"));
  auto *wrong = llvm::Function::Create(source->getFunction("leaf")->getFunctionType(),
                                       llvm::GlobalValue::ExternalLinkage,
                                       leaf->symbol, *source);
  wrong->setCallingConv(llvm::CallingConv::Fast);
  EXPECT_ANY_THROW(LLVM::import_function_code(*source, *first));
  wrong->eraseFromParent();
  auto *bad_declaration = llvm::Function::Create(
      llvm::FunctionType::get(llvm::Type::getVoidTy(context), false),
      llvm::GlobalValue::ExternalLinkage, first->symbol, *source);
  EXPECT_ANY_THROW(LLVM::import_function_code(*source, *first));
  bad_declaration->eraseFromParent();
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
