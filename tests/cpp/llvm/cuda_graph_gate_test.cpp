#include "gtest/gtest.h"

#ifdef TI_WITH_LLVM
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include "taichi/codegen/llvm/cuda_graph_gate.h"
#include "taichi/codegen/llvm/llvm_compiled_data.h"

namespace taichi::lang {
namespace {

LLVMCompiledKernel source_kernel() {
  auto context = std::make_shared<llvm::LLVMContext>();
  auto module = std::make_unique<llvm::Module>("gate-source", *context);
  auto *void_type = llvm::Type::getVoidTy(*context);
  auto *helper = llvm::Function::Create(
      llvm::FunctionType::get(void_type, false),
      llvm::Function::InternalLinkage, "helper", *module);
  llvm::IRBuilder<>(llvm::BasicBlock::Create(*context, "entry", helper))
      .CreateRetVoid();
  std::vector<OffloadedTask> tasks;
  for (const auto *name : {"serial", "range"}) {
    auto *entry = llvm::Function::Create(
        llvm::FunctionType::get(
            void_type, {llvm::PointerType::get(*context, 0)}, false),
        llvm::Function::ExternalLinkage, name, *module);
    entry->addFnAttr("target-cpu", "sm_80");
    llvm::IRBuilder<> builder(
        llvm::BasicBlock::Create(*context, "original", entry));
    builder.CreateCall(helper);
    builder.CreateRetVoid();
    tasks.emplace_back(name, 64, 128, 256);
    tasks.back().task_id = std::string("source/") + name;
  }
  return LLVMCompiledKernel(std::move(tasks), std::move(module),
                            std::move(context), 48);
}

std::string module_text(const llvm::Module &module) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  module.print(stream, nullptr);
  return result;
}

TEST(CudaGraphGate, ClonesAllTaskEntriesAndPreservesOrdinaryCodeAndOwnership) {
  LLVMCompiledKernel variant;
  std::weak_ptr<llvm::LLVMContext> lifetime;
  {
    auto source = source_kernel();
    lifetime = source.module_context_owner;
    const auto ordinary = module_text(*source.module);
    variant = LLVM::make_cuda_graph_gated_kernel(source);
    EXPECT_EQ(module_text(*source.module), ordinary);
    EXPECT_NE(source.module.get(), variant.module.get());
    EXPECT_EQ(variant.module_context_owner, source.module_context_owner);
    EXPECT_EQ(variant.cuda_max_registers, 48);
    ASSERT_EQ(variant.tasks.size(), 2);
    for (std::size_t i = 0; i < source.tasks.size(); ++i) {
      const auto &task = variant.tasks[i];
      EXPECT_EQ(task.name, source.tasks[i].name);
      EXPECT_EQ(task.task_id, source.tasks[i].task_id);
      EXPECT_EQ(task.block_dim, 64);
      EXPECT_EQ(task.grid_dim, 128);
      EXPECT_EQ(task.dynamic_shared_array_bytes, 256);
      auto *function = variant.module->getFunction(task.name);
      ASSERT_NE(function, nullptr);
      EXPECT_EQ(function->getFnAttribute("target-cpu").getValueAsString(),
                "sm_80");
      auto *branch = llvm::dyn_cast<llvm::BranchInst>(
          function->getEntryBlock().getTerminator());
      ASSERT_NE(branch, nullptr);
      ASSERT_TRUE(branch->isConditional());
      EXPECT_EQ(branch->getSuccessor(0)->getName(), "original");
      EXPECT_TRUE(llvm::isa<llvm::CallInst>(branch->getSuccessor(0)->front()));
      // The disabled path returns before any original payload or helper call.
      EXPECT_EQ(branch->getSuccessor(1)->size(), 1);
      EXPECT_TRUE(llvm::isa<llvm::ReturnInst>(branch->getSuccessor(1)->front()));
    }
    EXPECT_EQ(variant.module->getFunction("helper")->size(), 1);
    EXPECT_EQ(variant.module->size(), source.module->size());
    EXPECT_FALSE(llvm::verifyModule(*variant.module, &llvm::errs()));
  }
  EXPECT_FALSE(lifetime.expired());
  variant = LLVMCompiledKernel{};
  EXPECT_TRUE(lifetime.expired());
}

TEST(CudaGraphGate, RejectsIncompleteOrIncompatibleEntryAbiWithoutMutation) {
  LLVMCompiledKernel empty;
  EXPECT_ANY_THROW(LLVM::make_cuda_graph_gated_kernel(empty));
  auto source = source_kernel();
  const auto ordinary = module_text(*source.module);
  source.tasks.emplace_back("missing");
  EXPECT_ANY_THROW(LLVM::make_cuda_graph_gated_kernel(source));
  EXPECT_EQ(module_text(*source.module), ordinary);
  source.tasks.back().name = "helper";  // Valid definition, wrong entry ABI.
  EXPECT_ANY_THROW(LLVM::make_cuda_graph_gated_kernel(source));
  EXPECT_EQ(module_text(*source.module), ordinary);
}

}  // namespace
}  // namespace taichi::lang
#endif
