#include "gtest/gtest.h"

#if defined(TI_WITH_CUDA)
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/Support/TargetSelect.h"
#include "taichi/runtime/cuda/jit_cuda.h"
#include "taichi/rhi/cuda/cuda_capability.h"
#include "taichi/platform/cuda/detect_cuda.h"
#include "taichi/runtime/llvm/aot_snode_relocation.h"
#include "taichi/runtime/llvm/cuda_lowering_target.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Transforms/Utils/Cloning.h"

namespace taichi::lang {
namespace {

std::unique_ptr<llvm::Module> make_cuda_artifact_fixture(llvm::LLVMContext &ctx) {
  LLVMInitializeNVPTXTargetInfo();
  LLVMInitializeNVPTXTarget();
  LLVMInitializeNVPTXTargetMC();
  LLVMInitializeNVPTXAsmPrinter();
  auto module = std::make_unique<llvm::Module>("target_fixture", ctx);
  module->setTargetTriple("nvptx64-nvidia-cuda");
  auto type = llvm::FunctionType::get(llvm::Type::getVoidTy(ctx), {}, false);
  auto function = llvm::Function::Create(type, llvm::GlobalValue::ExternalLinkage,
                                         "target_entry", *module);
  // Ensure an explicit target also replaces stale function attributes.
  function->addFnAttr("target-cpu", "sm_90");
  function->addFnAttr("target-features", "+ptx80");
  llvm::IRBuilder<> builder(llvm::BasicBlock::Create(ctx, "entry", function));
  builder.CreateRetVoid();
  module->getOrInsertNamedMetadata("nvvm.annotations")->addOperand(
      llvm::MDNode::get(ctx, {llvm::ValueAsMetadata::get(function),
                            llvm::MDString::get(ctx, "kernel"),
                            llvm::ConstantAsMetadata::get(builder.getInt32(1))}));
  return module;
}

}  // namespace

TEST(CUDAArtifactTarget, ExplicitTargetsDoNotConsultTheCurrentDevice) {
  // This fixture creates LLVM IR directly; it never initializes CUDAContext or
  // loads a driver module. Whole-Python headless export is a separate contract.
  llvm::LLVMContext context;
  CompileConfig config;
  config.llvm_opt_level = 0;
  for (const auto &target : {CUDACodegenTarget{"sm_70", "+ptx60"},
                             CUDACodegenTarget{"sm_80", "+ptx70"}}) {
    auto module = make_cuda_artifact_fixture(context);
    auto artifact = compile_cuda_kernel_artifact(module, config, target, 0,
                                                 JITModuleRole::user_kernel);
    const std::string ptx(artifact.payload.data(), artifact.code_size());
    EXPECT_EQ(artifact.target_identity, target.identity());
    EXPECT_NE(ptx.find(target.mcpu), std::string::npos);
    EXPECT_EQ(artifact.entry_names, std::vector<std::string>{"target_entry"});
    EXPECT_EQ(artifact.kind, CUDAArtifactKind::ptx);
    EXPECT_EQ(artifact.payload.back(), '\0');
  }
}

TEST(CUDAArtifactTarget, KernelLoweringTargetSurvivesBitcodeAndSpecializesOnlyItsCopy) {
  llvm::LLVMContext context;
  auto producer = make_cuda_artifact_fixture(context);
  auto *type = llvm::FunctionType::get(llvm::Type::getInt32Ty(context), {}, false);
  auto *helper = llvm::Function::Create(type, llvm::GlobalValue::InternalLinkage,
                                        "cuda_compute_capability", *producer);
  llvm::IRBuilder<> builder(llvm::BasicBlock::Create(context, "entry", helper));
  builder.CreateRet(builder.getInt32(120));
  auto module = llvm::CloneModule(*producer);
  CUDALoweringTarget{60, 60}.write(*module);
  auto value = [](llvm::Module &source) {
    auto *ret = llvm::cast<llvm::ReturnInst>(source.getFunction("cuda_compute_capability")->front().getTerminator());
    return llvm::cast<llvm::ConstantInt>(ret->getReturnValue())->getSExtValue();
  };
  EXPECT_EQ(value(*producer), 120);
  EXPECT_EQ(value(*module), 60);
  EXPECT_FALSE(CUDALoweringTarget::read(*producer));
  llvm::SmallVector<char, 0> bytes;
  llvm::raw_svector_ostream stream(bytes);
  llvm::WriteBitcodeToFile(*module, stream);
  llvm::LLVMContext decoded_context;
  auto decoded = llvm::parseBitcodeFile(llvm::MemoryBufferRef(llvm::StringRef(bytes.data(), bytes.size()), "target"), decoded_context);
  ASSERT_TRUE(bool(decoded));
  const auto target = CUDALoweringTarget::read(**decoded);
  ASSERT_TRUE(target);
  EXPECT_EQ(target->compute_capability, 60);
  EXPECT_EQ(value(**decoded), 60);
  EXPECT_ANY_THROW(target->require_finalization_target("sm_50", "+ptx60"));
  EXPECT_ANY_THROW(target->require_finalization_target("sm_60", "+ptx50"));
  CompileConfig config;
  auto artifact = compile_cuda_kernel_artifact(*decoded, config, {"sm_80", "+ptx70"}, 0, JITModuleRole::user_kernel);
  EXPECT_EQ(artifact.target_identity, "sm_80|+ptx70");
}

TEST(CUDAArtifactTarget, RejectsUnknownTargetsAndFeatures) {
  llvm::LLVMContext context;
  CompileConfig config;
  for (const auto &target : {CUDACodegenTarget{"sm_not_real", "+ptx60"},
                             CUDACodegenTarget{"sm_70", "+ptx999"},
                             CUDACodegenTarget{"sm_70", "ptx60"}}) {
    auto module = make_cuda_artifact_fixture(context);
    EXPECT_ANY_THROW(compile_cuda_kernel_artifact(
        module, config, target, 0, JITModuleRole::user_kernel));
  }
}

TEST(CUDAArtifactTarget, PascalTargetCanEmitSynchronizedWarpShuffle) {
  llvm::LLVMContext context;
  auto module = make_cuda_artifact_fixture(context);
  auto *entry = module->getFunction("target_entry");
  llvm::IRBuilder<> builder(entry->getEntryBlock().getTerminator());
  auto *shuffle = llvm::Intrinsic::getDeclaration(
      module.get(), llvm::Intrinsic::nvvm_shfl_sync_idx_i32);
  auto *value = builder.CreateCall(shuffle,
      {builder.getInt32(-1), builder.getInt32(42), builder.getInt32(0),
       builder.getInt32(31)});
  auto *output = new llvm::GlobalVariable(
      *module, builder.getInt32Ty(), false, llvm::GlobalValue::ExternalLinkage,
      nullptr, "output", nullptr, llvm::GlobalValue::NotThreadLocal, 1);
  builder.CreateStore(value, output);
  const auto resolved = cuda::detail::resolve_compute_capability_target(60);
  EXPECT_EQ(resolved.ptx_version, 60);
  const CUDACodegenTarget target{"sm_60", "+ptx60"};
  CompileConfig config;
  auto artifact = compile_cuda_kernel_artifact(module, config, target, 0,
                                               JITModuleRole::user_kernel);
  const std::string ptx(artifact.payload.data(), artifact.code_size());
  EXPECT_NE(ptx.find(".target sm_60"), std::string::npos);
  EXPECT_NE(ptx.find("shfl.sync"), std::string::npos);
}

TEST(CUDAArtifactTarget, RelocatableConstantSurvivesOptimizationAndBinds) {
  if (!is_cuda_api_available() ||
      !CUDADriver::get_instance().nvidia_extensions_available() ||
      CUDAContext::get_instance().get_compute_capability() < 80)
    GTEST_SKIP();
  llvm::LLVMContext context;
  auto module = make_cuda_artifact_fixture(context);
  module->getNamedMetadata("nvvm.annotations")->clearOperands();
  module->getFunction("target_entry")->eraseFromParent();
  auto *type =
      llvm::FunctionType::get(llvm::Type::getVoidTy(context),
                              {llvm::PointerType::get(context, 1)}, false);
  auto *entry = llvm::Function::Create(type, llvm::GlobalValue::ExternalLinkage,
                                       "target_entry", *module);
  llvm::IRBuilder<> builder(llvm::BasicBlock::Create(context, "entry", entry));
  auto *id = LLVM::emit_aot_tree_id(*module, builder, 7, true);
  LLVM::emit_aot_tree_id(*module, builder, 9,
                         true);  // Dead after optimization.
  builder.CreateStore(id, entry->getArg(0));
  builder.CreateRetVoid();
  module->getNamedMetadata("nvvm.annotations")
      ->addOperand(llvm::MDNode::get(
          context, {llvm::ValueAsMetadata::get(entry),
                    llvm::MDString::get(context, "kernel"),
                    llvm::ConstantAsMetadata::get(builder.getInt32(1))}));
  EXPECT_EQ(LLVM::prepare_ptx_tree_relocations(*module),
            (std::vector<int>{7, 9}));
  CompileConfig config;
  config.arch = Arch::cuda;
  config.llvm_opt_level = 3;
  auto artifact = compile_cuda_kernel_artifact(
      module, config, {"sm_80", "+ptx70"}, 0, JITModuleRole::user_kernel);
  JITSessionCUDA session(nullptr, config, llvm::DataLayout(""));
  auto *loaded = static_cast<JITModuleCUDA *>(session.add_artifact(artifact));
  EXPECT_EQ(artifact.aot_tree_relocations, std::vector<int>{7});
  EXPECT_EQ(std::string(artifact.payload.data()).find("__ti_aot_tree_id_9"),
            std::string::npos);
  loaded->bind_aot_tree_ids({{7, 29}});
  auto &driver = CUDADriver::get_instance();
  void *device_data = nullptr;
  driver.malloc(&device_data, sizeof(int32_t));
  auto release = [&](void *ptr) { driver.mem_free(ptr); };
  std::unique_ptr<void, decltype(release)> allocation(device_data, release);
  loaded->launch("target_entry", 1, 1, 0, {&device_data},
                 {static_cast<int>(sizeof(device_data))});
  int32_t actual = -1;
  driver.memcpy_device_to_host(&actual, device_data, sizeof(actual));
  EXPECT_EQ(actual, 29);  // Not the producer's placeholder value 7.
  EXPECT_TRUE(session.remove_module(loaded));
}

TEST(CUDAArtifactTarget, DriverLoaderHonorsRegisterLimit) {
  if (!is_cuda_api_available() ||
      !CUDADriver::get_instance().nvidia_extensions_available() ||
      CUDAContext::get_instance().get_compute_capability() < 80) {
    GTEST_SKIP();
  }
  // Keep enough values live to make the explicit register limit observable.
  // This exercises the shared JIT/AOT loader, not LLVM register allocation.
  std::string ptx =
      ".version 7.0\n.target sm_80\n.address_size 64\n"
      ".visible .entry register_probe(.param .u64 address) {\n"
      ".reg .b64 p;\n.reg .f32 f<65>;\nld.param.u64 p, [address];\n";
  for (int i = 0; i < 64; ++i) {
    ptx += fmt::format("ld.volatile.global.f32 f{}, [p+{}];\n", i, i * 4);
  }
  ptx += "mov.f32 f64, 0f00000000;\n";
  for (int i = 0; i < 64; ++i) {
    ptx += fmt::format("add.rn.f32 f64, f64, f{};\n", i);
  }
  ptx += "st.global.f32 [p], f64;\nret;\n}\n";
  CUDAKernelArtifact artifact;
  artifact.payload.assign(ptx.begin(), ptx.end());
  artifact.payload.push_back('\0');
  artifact.entry_names = {"register_probe"};
  artifact.target_identity = "sm_80|+ptx70";
  artifact.provider_identity = "register_limit_test";
  artifact.max_registers = 32;
  CompileConfig config;
  config.arch = Arch::cuda;
  // add_artifact needs no LLVM context or runtime bootstrap.
  JITSessionCUDA session(nullptr, config, llvm::DataLayout(""));
  auto *module = session.add_artifact(artifact);
  auto *function = module->lookup_function("register_probe");
  int registers = 0;
  auto &driver = CUDADriver::get_instance();
  driver.kernel_get_attribute(&registers, CU_FUNC_ATTRIBUTE_NUM_REGS, function);
  EXPECT_GT(registers, 0);
  EXPECT_LE(registers, 32);

  void *device_data = nullptr;
  driver.malloc(&device_data, 64 * sizeof(float));
  auto release = [&](void *ptr) { driver.mem_free(ptr); };
  std::unique_ptr<void, decltype(release)> allocation(device_data, release);
  std::vector<float> values(64, 1.0f);
  driver.memcpy_host_to_device(device_data, values.data(), 64 * sizeof(float));
  module->launch("register_probe", 1, 1, 0, {&device_data},
                  {static_cast<int>(sizeof(device_data))});
  driver.memcpy_device_to_host(values.data(), device_data, sizeof(float));
  EXPECT_FLOAT_EQ(values[0], 64.0f);
  EXPECT_TRUE(session.remove_module(module));
}

}  // namespace taichi::lang
#endif
