#include "gtest/gtest.h"

#ifdef TI_WITH_LLVM
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include "taichi/codegen/llvm/cuda_graph_gate.h"
#include "taichi/codegen/llvm/llvm_compiled_data.h"
#include "taichi/runtime/llvm/aot_kernel_contract.h"

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
    EXPECT_FALSE(LLVM::has_cuda_graph_gate(*source.module));
    EXPECT_TRUE(LLVM::has_cuda_graph_gate(*variant.module));
    EXPECT_ANY_THROW(LLVM::make_cuda_graph_gated_kernel(variant));
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

TEST(CudaGraphGate, HiddenLaunchContractComesFromTransformedCode) {
  for (bool context_v2 : {false, true}) {
    auto source = source_kernel();
    if (context_v2)
      LLVM::declare_runtime_context_abi(*source.module);
    const auto text = module_text(*source.module);
    auto variant = LLVM::make_cuda_graph_gated_kernel(source);
    EXPECT_STREQ(LLVM::compiled_kernel_launch_abi(*source.module),
                 context_v2 ? "runtime_context_v2" : "ordinary");
    EXPECT_STREQ(LLVM::compiled_kernel_launch_abi(*variant.module),
                 context_v2 ? "cuda_graph_gate_context_v2"
                            : "cuda_graph_gate_v1");
    EXPECT_EQ(LLVM::runtime_context_abi(*variant.module), context_v2 ? 2 : 1);
    // Public entry types are unchanged; only the hidden argument prefix differs.
    for (const auto &task : source.tasks) {
      EXPECT_EQ(source.module->getFunction(task.name)->getFunctionType(),
                variant.module->getFunction(task.name)->getFunctionType());
    }
    EXPECT_EQ(module_text(*source.module), text);
  }
}

TEST(LLVMAotContract, RelocationProjectsCodeAndDependenciesTogether) {
  auto source = source_kernel();
  auto &context = source.module->getContext();
  auto *entry = llvm::Function::Create(
      llvm::FunctionType::get(llvm::Type::getInt32Ty(context), false),
      llvm::Function::ExternalLinkage, "tree_ids", *source.module);
  llvm::IRBuilder<> builder(llvm::BasicBlock::Create(context, "entry", entry));
  auto *first = LLVM::emit_aot_tree_id(*source.module, builder, 2, false);
  auto *second = LLVM::emit_aot_tree_id(*source.module, builder, 9, false);
  builder.CreateRet(builder.CreateAdd(first, second));
  LLVM::KernelCallContract contract;
  contract.used_snode_tree_ids = {2, 9};
  const auto text = module_text(*source.module);
  auto artifact = source.clone();
  auto projected = contract;
  LLVM::remap_aot_kernel_tree_ids(*artifact.module, projected, {{2, 0}, {9, 1}});
  EXPECT_EQ(projected.used_snode_tree_ids, (std::vector<int>{0, 1}));
  std::vector<int> code_ids;
  for (const auto &[id, global] : LLVM::aot_tree_id_globals(*artifact.module)) {
    code_ids.push_back(id);
    EXPECT_FALSE(global->use_empty());
  }
  EXPECT_EQ(code_ids, projected.used_snode_tree_ids);
  EXPECT_FALSE(llvm::verifyModule(*artifact.module, &llvm::errs()));
  EXPECT_EQ(module_text(*source.module), text);
  EXPECT_EQ(contract.used_snode_tree_ids, (std::vector<int>{2, 9}));

  // Invalid closure cannot publish half-remapped code or metadata.
  auto failed = source.clone();
  EXPECT_ANY_THROW(
      LLVM::remap_aot_kernel_tree_ids(*failed.module, contract, {{2, 0}}));
  EXPECT_EQ(module_text(*failed.module), text);
  EXPECT_EQ(contract.used_snode_tree_ids, (std::vector<int>{2, 9}));
  LLVM::KernelCallContract missing;
  missing.used_snode_tree_ids = {2};
  EXPECT_ANY_THROW(LLVM::remap_aot_kernel_tree_ids(
      *failed.module, missing, {{2, 0}, {9, 1}}));
  EXPECT_EQ(module_text(*failed.module), text);
}

}  // namespace
}  // namespace taichi::lang
#endif
