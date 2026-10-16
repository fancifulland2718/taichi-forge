#include "taichi/codegen/llvm/cuda_graph_gate.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Verifier.h"
#include "taichi/codegen/llvm/llvm_compiled_data.h"
#include "taichi/program/context.h"

namespace taichi::lang::LLVM {

LLVMCompiledKernel make_cuda_graph_gated_kernel(
    const LLVMCompiledKernel &source) {
  TI_ERROR_IF(source.module == nullptr,
              "CUDA Graph gate requires lowered LLVM code");
  for (const auto &task : source.tasks) {
    const auto *function = source.module->getFunction(task.name);
    TI_ERROR_IF(function == nullptr || function->empty() ||
                    !function->getReturnType()->isVoidTy() ||
                    function->arg_size() != 1 || function->isVarArg() ||
                    !function->getArg(0)->getType()->isPointerTy(),
                "CUDA Graph gate requires a void task with one context "
                "pointer: {}", task.name);
  }
  auto compiled = source.clone();
  auto &module = *compiled.module;
  auto &llvm_context = module.getContext();
  auto *pointer_type = llvm::PointerType::get(llvm_context, 0);
  auto *i8_type = llvm::Type::getInt8Ty(llvm_context);
  auto *i32_type = llvm::Type::getInt32Ty(llvm_context);
  auto *i64_type = llvm::Type::getInt64Ty(llvm_context);

  for (const auto &task : compiled.tasks) {
    auto *function = module.getFunction(task.name);
    auto *original_entry = &function->getEntryBlock();
    auto *gate_entry = llvm::BasicBlock::Create(
        llvm_context, "graph_execution_gate", function, original_entry);
    auto *masked_exit = llvm::BasicBlock::Create(
        llvm_context, "graph_execution_masked_exit", function, original_entry);
    llvm::IRBuilder<> builder(gate_entry);
    // RuntimeContext::arg_buffer is its first member. The private gated launch
    // allocation stores GraphExecutionGateBinding immediately before that
    // pointer, without adding a runtime symbol or changing RuntimeContext.
    auto *arg_buffer = builder.CreateAlignedLoad(
        pointer_type, function->getArg(0), llvm::Align(8), "graph_args");
    auto *binding = builder.CreateInBoundsGEP(
        i8_type, arg_buffer,
        llvm::ConstantInt::getSigned(
            i64_type,
            -static_cast<std::int64_t>(sizeof(GraphExecutionGateBinding))),
        "graph_gate_binding");
    auto *gate_bits = builder.CreateAlignedLoad(
        i64_type, binding, llvm::Align(8), "graph_gate_address");
    auto *expected_address = builder.CreateInBoundsGEP(
        i8_type, binding,
        llvm::ConstantInt::get(i64_type,
                               offsetof(GraphExecutionGateBinding, expected)));
    auto *expected = builder.CreateAlignedLoad(
        i32_type, expected_address, llvm::Align(4), "graph_gate_expected");
    auto *gate = builder.CreateIntToPtr(gate_bits, pointer_type);
    auto *value = builder.CreateAlignedLoad(i32_type, gate, llvm::Align(4),
                                            "graph_gate_value");
    auto *execute = builder.CreateICmpEQ(value, expected);
    builder.CreateCondBr(execute, original_entry, masked_exit);
    builder.SetInsertPoint(masked_exit);
    builder.CreateRetVoid();
  }
  TI_ASSERT(!llvm::verifyModule(module, &llvm::errs()));
  return compiled;
}

}  // namespace taichi::lang::LLVM
