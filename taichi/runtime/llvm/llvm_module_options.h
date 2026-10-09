#pragma once

#include "llvm/IR/Constants.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "taichi/program/compile_config.h"
#include "taichi/runtime/llvm/llvm_opt_pipeline.h"

namespace taichi::lang {

// GPU LLVM optimization happens at launcher registration, possibly after the
// kernel's effective CompileConfig has gone out of scope. Store the options in
// the module: cloning, bitcode/offline-cache and AOT round trips retain them.
struct LLVMModuleOptions {
  int opt_level;
  bool fast_math;

  static LLVMModuleOptions from_config(const CompileConfig &config) {
    const int floor =
        config.arch == Arch::cuda || config.arch == Arch::amdgpu ? 1 : 0;
    return {effective_llvm_opt_level(config.llvm_opt_level, config.compile_tier,
                                     floor),
            config.fast_math};
  }

  void write(llvm::Module &module) const {
    auto &context = module.getContext();
    auto integer = [&](int value) {
      return llvm::ConstantAsMetadata::get(
          llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), value));
    };
    auto *metadata = module.getOrInsertNamedMetadata("taichi.jit.options.v1");
    metadata->clearOperands();
    metadata->addOperand(llvm::MDNode::get(
        context, {integer(opt_level), integer(fast_math)}));
  }

  static LLVMModuleOptions read(const llvm::Module &module,
                                const CompileConfig &fallback) {
    auto *metadata = module.getNamedMetadata("taichi.jit.options.v1");
    // Runtime modules and legacy AOT modules have no kernel specialization.
    if (!metadata)
      return from_config(fallback);
    TI_ASSERT(metadata->getNumOperands() == 1);
    auto *node = metadata->getOperand(0);
    TI_ASSERT(node->getNumOperands() == 2);
    auto integer = [&](unsigned index) {
      auto *value = llvm::mdconst::dyn_extract<llvm::ConstantInt>(
          node->getOperand(index));
      TI_ASSERT(value != nullptr);
      return static_cast<int>(value->getSExtValue());
    };
    const int level = integer(0);
    const int fast = integer(1);
    TI_ASSERT(fast == 0 || fast == 1);
    return {level, fast != 0};
  }
};

}  // namespace taichi::lang
