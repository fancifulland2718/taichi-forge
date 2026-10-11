#include "gtest/gtest.h"

#if defined(TI_WITH_CUDA)
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/TargetSelect.h"
#include "taichi/runtime/cuda/jit_cuda.h"

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

}  // namespace taichi::lang
#endif
