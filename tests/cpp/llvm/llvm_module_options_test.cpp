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

TEST(LLVMAotFinalization, SpecializesHelpersBeforeArchivingDynamicEntries) {
  llvm::LLVMContext context;
  llvm::SMDiagnostic error;
  auto source = llvm::parseAssemblyString(R"(
    target datalayout = "e-p:64:64-i64:64-n16:32:64"
    target triple = "nvptx64-nvidia-cuda"
    declare void @unselected_effect()
    define internal i32 @helper(ptr %frame) alwaysinline "taichi.aot.real_function" {
      %mode = load i32, ptr %frame
      %data = getelementptr i32, ptr %frame, i32 1
      %value = load i32, ptr %data
      %condition = icmp eq i32 %mode, 0
      br i1 %condition, label %chosen, label %other
    chosen:
      %result = add i32 %value, 7
      ret i32 %result
    other:
      call void @unselected_effect()
      ret i32 99
    }
    define i32 @entry(ptr %scene_value, ptr %output) {
      %dynamic = load volatile i32, ptr %scene_value
      %frame = alloca [2 x i32]
      %value_slot = getelementptr i32, ptr %frame, i32 1
      store i32 0, ptr %frame
      store i32 %dynamic, ptr %value_slot
      %result = call i32 @helper(ptr %frame)
      store volatile i32 %result, ptr %output
      ret i32 %result
    }
  )", error, context);
  ASSERT_NE(source, nullptr);
  LLVM::AotIRFinalizer finalizer;
  auto enabled = llvm::CloneModule(*source);
  const auto snapshot = [](const llvm::Module &module) {
    std::string bytes;
    llvm::raw_string_ostream output(bytes);
    encode_llvm_module(module, output, LLVMModuleEncoding::bitcode);
    return bytes;
  };
  finalizer.export_units(*enabled, true, false);
  const auto sealed = snapshot(*enabled);
  finalizer.export_units(*enabled, true, true);
  EXPECT_EQ(sealed, snapshot(*enabled));
  finalizer.finalize_root(enabled, true);
  EXPECT_FALSE(llvm::verifyModule(*enabled));
  EXPECT_NE(enabled->getNamedMetadata("taichi.aot.helper_specialization.v1"), nullptr);
  int reads = 0, writes = 0, calls = 0;
  for (auto &block : *enabled->getFunction("entry")) {
    for (auto &instruction : block) {
      if (auto *load = llvm::dyn_cast<llvm::LoadInst>(&instruction))
        reads += load->isVolatile();
      if (auto *store = llvm::dyn_cast<llvm::StoreInst>(&instruction))
        writes += store->isVolatile();
      calls += llvm::isa<llvm::CallBase>(instruction);
      EXPECT_FALSE(llvm::isa<llvm::AllocaInst>(instruction));
    }
  }
  EXPECT_EQ(reads, 1);
  EXPECT_EQ(writes, 1);
  EXPECT_EQ(calls, 0);  // Known helper mode folded; scene data is still live.
  EXPECT_FALSE(source->getFunction("helper")->isDeclaration());
  auto disabled = llvm::CloneModule(*source);
  finalizer.export_units(*disabled, false, false);
  finalizer.finalize_root(disabled, false);
  EXPECT_EQ(disabled->getNamedMetadata("taichi.aot.helper_specialization.v1"), nullptr);
  ASSERT_NE(disabled->getNamedMetadata(LLVM::kCodeUnits), nullptr);
  EXPECT_EQ(disabled->getNamedMetadata(LLVM::kCodeUnits)->getNumOperands(), 1);
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

#ifdef TI_WITH_LLVM
#include "llvm/Transforms/IPO/Internalize.h"

namespace taichi::lang {
namespace {

class LLVMTaskFunctionUnits : public testing::Test {
 protected:
  using UnitRef = std::shared_ptr<const LLVM::FunctionCodeUnit>;
  using UnitMap = std::map<std::string, UnitRef>;

  static std::unique_ptr<llvm::Module> parse(llvm::LLVMContext &context,
                                           const std::string &body) {
    llvm::SMDiagnostic error;
    // LLVM 20 upgrades x86 bitcode layouts with i128 alignment. Specify it
    // before recording units so a context transfer preserves the same layout.
    auto module = llvm::parseAssemblyString(
        "target triple = \"x86_64-pc-windows-msvc\"\n"
        "target datalayout = \"e-p:64:64-i64:64-i128:128-n16:32:64\"\n" + body,
        error, context);
    EXPECT_NE(module, nullptr);
    return module;
  }

  static std::unique_ptr<llvm::Module> task(llvm::LLVMContext &context,
                                          const std::string &entry,
                                          int bias,
                                          int leaf_bias = 1) {
    return parse(context,
        "define internal i32 @leaf(i32 %x) \"taichi.aot.real_function\" {\n"
        "  %r = add i32 %x, " + std::to_string(leaf_bias) + "\n"
        "  ret i32 %r\n}\n"
        "define internal i32 @parent(i32 %x) \"taichi.aot.real_function\" {\n"
        "  %r = call i32 @leaf(i32 %x)\n"
        "  %v = add i32 %r, " + std::to_string(bias) + "\n"
        "  ret i32 %v\n}\n"
        "define i32 @" + entry + "(i32 %x) {\n"
        "  %r = call i32 @parent(i32 %x)\n"
        "  ret i32 %r\n}\n");
  }

  static bool record(llvm::Module &module,
                     const std::shared_ptr<LLVM::FunctionCodeCache> &cache,
                     const std::string &key,
                     const std::string &name) {
    auto *function = module.getFunction(name);
    EXPECT_NE(function, nullptr);
    if (!function)
      return false;
    const auto entry = cache->record(key, module, function);
    EXPECT_NE(entry, nullptr);
    if (!entry)
      return false;
    LLVM::register_function_code(module, *entry, function);
    return true;
  }

  static void prune(llvm::Module &module,
                    const std::unordered_set<std::string> &entries) {
    llvm::ModuleAnalysisManager analysis;
    llvm::PassBuilder passes;
    passes.registerModuleAnalyses(analysis);
    llvm::ModulePassManager manager;
    manager.addPass(llvm::InternalizePass([&](const llvm::GlobalValue &value) {
      return entries.count(value.getName().str()) != 0;
    }));
    manager.addPass(llvm::GlobalDCEPass());
    manager.run(module, analysis);
  }

  static std::vector<UnitRef> detach(
      llvm::Module &module,
      const std::shared_ptr<LLVM::FunctionCodeCache> &cache,
      const std::unordered_set<std::string> &entries) {
    const auto captured = LLVM::capture_task_function_units(module, cache);
    EXPECT_EQ(module.getNamedMetadata(LLVM::kImportedFunctionUnits), nullptr);
    prune(module, entries);
    return LLVM::detach_task_function_units(module, captured);
  }

  static std::unique_ptr<llvm::Module> transfer(const llvm::Module &module,
                                               llvm::LLVMContext &context) {
    std::string bytes;
    {
      llvm::raw_string_ostream stream(bytes);
      encode_llvm_module(module, stream, LLVMModuleEncoding::bitcode);
    }
    auto decoded = decode_llvm_module(llvm::MemoryBufferRef(bytes, "thin_task"),
                                      context, LLVMModuleEncoding::bitcode);
    if (!decoded) {
      ADD_FAILURE() << llvm::toString(decoded.takeError());
      return nullptr;
    }
    return std::move(*decoded);
  }

  static UnitMap collect(const std::vector<UnitRef> &units) {
    UnitMap result;
    for (const auto &unit : units)
      result.emplace(unit->digest, unit);
    return result;
  }

  static int definitions(const llvm::Module &module) {
    int result = 0;
    for (const auto &function : module)
      result += !function.isDeclaration();
    return result;
  }
};

TEST_F(LLVMTaskFunctionUnits, SharesLeafAcrossContextsBeforeFinalOptimization) {
  llvm::LLVMContext first_context, second_context, linked_context;
  auto cache = std::make_shared<LLVM::FunctionCodeCache>();
  auto first = task(first_context, "entry_a", 2);
  auto second = task(second_context, "entry_b", 3);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  ASSERT_TRUE(record(*first, cache, "a.leaf", "leaf"));
  ASSERT_TRUE(record(*first, cache, "a.parent", "parent"));
  ASSERT_TRUE(record(*second, cache, "b.leaf", "leaf"));
  ASSERT_TRUE(record(*second, cache, "b.parent", "parent"));
  const auto leaf = cache->find("a.leaf")->units.back();
  EXPECT_EQ(leaf, cache->find("b.leaf")->units.back());
  auto units = collect(detach(*first, cache, {"entry_a"}));
  const auto second_units = collect(detach(*second, cache, {"entry_b"}));
  EXPECT_EQ(units.size(), 2);
  EXPECT_EQ(second_units.size(), 2);
  units.insert(second_units.begin(), second_units.end());
  ASSERT_EQ(units.size(), 3);
  EXPECT_EQ(definitions(*first), 1);
  EXPECT_EQ(definitions(*second), 1);
  EXPECT_FALSE(llvm::verifyModule(*first));
  EXPECT_FALSE(llvm::verifyModule(*second));

  auto linked = transfer(*first, linked_context);
  auto other = transfer(*second, linked_context);
  ASSERT_NE(linked, nullptr);
  ASSERT_NE(other, nullptr);
  ASSERT_FALSE(llvm::Linker(*linked).linkInModule(std::move(other)));
  LLVM::link_task_function_units(*linked, units);
  prune(*linked, {"entry_a", "entry_b"});
  EXPECT_EQ(definitions(*linked), 5);  // Two entries, two parents, one leaf.
  auto *shared_leaf = linked->getFunction(leaf->symbol);
  ASSERT_NE(shared_leaf, nullptr);
  EXPECT_FALSE(shared_leaf->isDeclaration());
  EXPECT_TRUE(shared_leaf->hasInternalLinkage());
  EXPECT_EQ(shared_leaf->getNumUses(), 2);
  EXPECT_EQ(linked->getNamedMetadata(LLVM::kImportedFunctionUnits), nullptr);
  EXPECT_FALSE(llvm::verifyModule(*linked));
}

TEST_F(LLVMTaskFunctionUnits, DropsDceDeathsAndRetainsOwnershipThroughTaskClone) {
  llvm::LLVMContext context;
  LLVMCompiledTask cloned;
  std::weak_ptr<const LLVM::FunctionCodeUnit> retained;
  {
    auto cache = std::make_shared<LLVM::FunctionCodeCache>();
    auto module = task(context, "entry", 2);
    ASSERT_NE(module, nullptr);
    auto *dead = llvm::Function::Create(
        module->getFunction("leaf")->getFunctionType(),
        llvm::GlobalValue::InternalLinkage, "dead", *module);
    dead->addFnAttr(LLVM::kRealFunctionUnit);
    auto *block = llvm::BasicBlock::Create(context, "", dead);
    llvm::ReturnInst::Create(context, dead->getArg(0), block);
    ASSERT_TRUE(record(*module, cache, "leaf", "leaf"));
    ASSERT_TRUE(record(*module, cache, "parent", "parent"));
    ASSERT_TRUE(record(*module, cache, "dead", "dead"));
    auto captured = LLVM::capture_task_function_units(*module, cache);
    ASSERT_EQ(captured.size(), 3);
    prune(*module, {"entry"});
    EXPECT_EQ(module->getFunction("dead"), nullptr);
    int deaths = 0;
    for (const auto &item : captured)
      deaths += item.function == nullptr;
    EXPECT_EQ(deaths, 1);
    auto units = LLVM::detach_task_function_units(*module, captured);
    ASSERT_EQ(units.size(), 2);
    retained = units.front();
    LLVMCompiledTask original({}, std::move(module), {}, {}, std::move(units));
    cloned = original.clone();
    EXPECT_EQ(cloned.function_units, original.function_units);
  }
  ASSERT_FALSE(retained.expired());
  EXPECT_EQ(definitions(*cloned.module), 1);
  auto units = collect(cloned.function_units);
  LLVM::link_task_function_units(*cloned.module, units);
  prune(*cloned.module, {"entry"});
  EXPECT_EQ(definitions(*cloned.module), 3);
  EXPECT_EQ(cloned.module->getNamedMetadata(LLVM::kImportedFunctionUnits), nullptr);
  EXPECT_FALSE(llvm::verifyModule(*cloned.module));
  units.clear();
  cloned.function_units.clear();
  EXPECT_TRUE(retained.expired());
}

TEST_F(LLVMTaskFunctionUnits, RejectsObservableAddressesTransitively) {
  const std::vector<std::string> bodies = {
      R"(
        define internal i32 @leaf(i32 %x) "taichi.aot.real_function" {
          ret i32 %x
        }
        define internal i32 @parent(i32 %x) "taichi.aot.real_function" {
          %r = call i32 @leaf(i32 %x)
          ret i32 %r
        }
        define i32 @entry(i32 %x, ptr %out) {
          store ptr @leaf, ptr %out
          %r = call i32 @parent(i32 %x)
          ret i32 %r
        }
      )",
      R"(
        @private_address = internal constant i32 7
        define internal ptr @leaf() "taichi.aot.real_function" {
          ret ptr @private_address
        }
        define internal ptr @parent() "taichi.aot.real_function" {
          %r = call ptr @leaf()
          ret ptr %r
        }
        define ptr @entry() {
          %r = call ptr @parent()
          ret ptr %r
        }
      )"};
  for (size_t i = 0; i < bodies.size(); ++i) {
    SCOPED_TRACE(i);
    llvm::LLVMContext context;
    auto module = parse(context, bodies[i]);
    auto cache = std::make_shared<LLVM::FunctionCodeCache>();
    ASSERT_NE(module, nullptr);
    ASSERT_TRUE(record(*module, cache, "leaf", "leaf"));
    ASSERT_TRUE(record(*module, cache, "parent", "parent"));
    const auto units = detach(*module, cache, {"entry"});
    EXPECT_TRUE(units.empty());
    EXPECT_FALSE(module->getFunction("leaf")->isDeclaration());
    EXPECT_FALSE(module->getFunction("parent")->isDeclaration());
    EXPECT_FALSE(llvm::verifyModule(*module));
  }
}

TEST_F(LLVMTaskFunctionUnits, RejectsLateExternalDefinitionsAndTypedABI) {
  llvm::LLVMContext context;
  auto cache = std::make_shared<LLVM::FunctionCodeCache>();
  auto module = parse(context, R"(
    declare i32 @external(i32)
    define internal i32 @parent(i32 %x) "taichi.aot.real_function" {
      %r = call i32 @external(i32 %x)
      ret i32 %r
    }
    define i32 @entry(i32 %x) {
      %r = call i32 @parent(i32 %x)
      ret i32 %r
    }
  )");
  ASSERT_NE(module, nullptr);
  ASSERT_TRUE(record(*module, cache, "parent", "parent"));
  // Codegen can finish this dependency after the parent was recorded, but
  // before the capture/Internalize/GlobalDCE contract begins.
  auto *external = module->getFunction("external");
  auto *block = llvm::BasicBlock::Create(context, "", external);
  llvm::ReturnInst::Create(context, external->getArg(0), block);
  EXPECT_TRUE(detach(*module, cache, {"entry"}).empty());
  EXPECT_FALSE(module->getFunction("parent")->isDeclaration());
  EXPECT_TRUE(external->hasInternalLinkage());
  EXPECT_FALSE(llvm::verifyModule(*module));

  for (bool typed_external : {false, true}) {
    SCOPED_TRACE(typed_external);
    llvm::LLVMContext typed_context;
    auto typed = parse(typed_context, typed_external ? R"(
      declare i32 @external(ptr byval(i32))
      define internal i32 @parent(ptr %p) "taichi.aot.real_function" {
        %r = call i32 @external(ptr byval(i32) %p)
        ret i32 %r
      }
      define i32 @entry(ptr %p) {
        %r = call i32 @parent(ptr %p)
        ret i32 %r
      }
    )" : R"(
      define internal i32 @parent(ptr byval(i32) %p) "taichi.aot.real_function" {
        %r = load i32, ptr %p
        ret i32 %r
      }
      define i32 @entry(ptr %p) {
        %r = call i32 @parent(ptr byval(i32) %p)
        ret i32 %r
      }
    )");
    ASSERT_NE(typed, nullptr);
    auto typed_cache = std::make_shared<LLVM::FunctionCodeCache>();
    ASSERT_TRUE(record(*typed, typed_cache, "typed", "parent"));
    EXPECT_TRUE(detach(*typed, typed_cache, {"entry"}).empty());
    EXPECT_FALSE(typed->getFunction("parent")->isDeclaration());
    EXPECT_FALSE(llvm::verifyModule(*typed));
  }
}

TEST_F(LLVMTaskFunctionUnits, RejectsTargetABIAndGlobalSymbolConflicts) {
  llvm::LLVMContext context;
  auto cache = std::make_shared<LLVM::FunctionCodeCache>();
  auto module = task(context, "entry", 2);
  ASSERT_NE(module, nullptr);
  ASSERT_TRUE(record(*module, cache, "leaf", "leaf"));
  ASSERT_TRUE(record(*module, cache, "parent", "parent"));
  const auto units = collect(detach(*module, cache, {"entry"}));
  ASSERT_EQ(units.size(), 2);
  const auto symbol = units.begin()->second->symbol;
  auto wrong_target = llvm::CloneModule(*module);
  wrong_target->setTargetTriple("aarch64-unknown-linux-gnu");
  EXPECT_ANY_THROW(LLVM::link_task_function_units(*wrong_target, units));
  auto wrong_layout = llvm::CloneModule(*module);
  wrong_layout->setDataLayout("e-p:32:32");
  EXPECT_ANY_THROW(LLVM::link_task_function_units(*wrong_layout, units));
  auto wrong_cc = llvm::CloneModule(*module);
  wrong_cc->getFunction(symbol)->setCallingConv(llvm::CallingConv::Fast);
  EXPECT_ANY_THROW(LLVM::link_task_function_units(*wrong_cc, units));
  auto wrong_attrs = llvm::CloneModule(*module);
  wrong_attrs->getFunction(symbol)->addFnAttr(llvm::Attribute::NoUnwind);
  EXPECT_ANY_THROW(LLVM::link_task_function_units(*wrong_attrs, units));
  auto wrong_type = parse(context, "declare void @" + symbol + "()\n");
  ASSERT_NE(wrong_type, nullptr);
  EXPECT_ANY_THROW(LLVM::link_task_function_units(*wrong_type, units));
  auto global_collision = parse(context, "@" + symbol + " = global i32 0\n");
  ASSERT_NE(global_collision, nullptr);
  EXPECT_ANY_THROW(LLVM::link_task_function_units(*global_collision, units));
  EXPECT_FALSE(llvm::verifyModule(*module));
}

TEST_F(LLVMTaskFunctionUnits, RejectsExternalDependencyABIConflicts) {
  llvm::LLVMContext context;
  auto cache = std::make_shared<LLVM::FunctionCodeCache>();
  auto module = parse(context, R"(
    declare i32 @external(i32)
    define internal i32 @parent(i32 %x) "taichi.aot.real_function" {
      %r = call i32 @external(i32 %x)
      ret i32 %r
    }
    define i32 @entry(i32 %x) {
      %r = call i32 @parent(i32 %x)
      ret i32 %r
    }
  )");
  ASSERT_NE(module, nullptr);
  ASSERT_TRUE(record(*module, cache, "parent", "parent"));
  const auto units = collect(detach(*module, cache, {"entry"}));
  ASSERT_EQ(units.size(), 1);
  auto compatible = llvm::CloneModule(*module);
  LLVM::link_task_function_units(*compatible, units);
  EXPECT_FALSE(llvm::verifyModule(*compatible));
  auto incompatible = llvm::CloneModule(*module);
  incompatible->getFunction("external")->setCallingConv(llvm::CallingConv::Fast);
  EXPECT_ANY_THROW(LLVM::link_task_function_units(*incompatible, units));
  auto private_definition = llvm::CloneModule(*module);
  auto *external = private_definition->getFunction("external");
  external->setLinkage(llvm::GlobalValue::InternalLinkage);
  auto *block = llvm::BasicBlock::Create(context, "", external);
  llvm::ReturnInst::Create(context, external->getArg(0), block);
  EXPECT_ANY_THROW(LLVM::link_task_function_units(*private_definition, units));
}

TEST_F(LLVMTaskFunctionUnits, KeepsDifferentLeafBodiesAndParentsSeparate) {
  llvm::LLVMContext first_context, second_context, linked_context;
  auto cache = std::make_shared<LLVM::FunctionCodeCache>();
  auto first = task(first_context, "entry_a", 2, 1);
  auto second = task(second_context, "entry_b", 2, 9);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  ASSERT_TRUE(record(*first, cache, "a.leaf", "leaf"));
  ASSERT_TRUE(record(*first, cache, "a.parent", "parent"));
  ASSERT_TRUE(record(*second, cache, "b.leaf", "leaf"));
  ASSERT_TRUE(record(*second, cache, "b.parent", "parent"));
  const auto first_leaf = cache->find("a.leaf")->units.back();
  const auto second_leaf = cache->find("b.leaf")->units.back();
  EXPECT_NE(first_leaf->digest, second_leaf->digest);
  auto units = collect(detach(*first, cache, {"entry_a"}));
  const auto other_units = collect(detach(*second, cache, {"entry_b"}));
  units.insert(other_units.begin(), other_units.end());
  ASSERT_EQ(units.size(), 4);
  auto linked = transfer(*first, linked_context);
  auto other = transfer(*second, linked_context);
  ASSERT_NE(linked, nullptr);
  ASSERT_NE(other, nullptr);
  ASSERT_FALSE(llvm::Linker(*linked).linkInModule(std::move(other)));
  LLVM::link_task_function_units(*linked, units);
  prune(*linked, {"entry_a", "entry_b"});
  EXPECT_EQ(definitions(*linked), 6);
  for (const auto &[unit, bias] :
       std::vector<std::pair<UnitRef, int>>{{first_leaf, 1}, {second_leaf, 9}}) {
    auto *function = linked->getFunction(unit->symbol);
    ASSERT_NE(function, nullptr);
    ASSERT_FALSE(function->isDeclaration());
    auto *constant = llvm::dyn_cast<llvm::ConstantInt>(
        function->front().front().getOperand(1));
    ASSERT_NE(constant, nullptr);
    EXPECT_EQ(constant->getSExtValue(), bias);
  }
  EXPECT_FALSE(llvm::verifyModule(*linked));
}

}  // namespace
}  // namespace taichi::lang
#endif
