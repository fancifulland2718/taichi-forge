#pragma once

#include <algorithm>
#include <vector>

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "taichi/common/core.h"

namespace taichi::lang::LLVM {

inline constexpr char kCudaRootBindingPrefix[] = "__ti_cuda_root_binding_";

// Keep compact indices symbolic until all task lowering has contributed its
// roots. A late mesh table may turn a direct root into a table or shift a slot.
inline llvm::Value *emit_cuda_root_binding(llvm::Module &module,
                                           llvm::IRBuilder<> &builder,
                                           llvm::Value *binding,
                                           int tree_id) {
  auto *pointer = builder.getPtrTy();
  auto function = module.getOrInsertFunction(
      fmt::format("{}{}", kCudaRootBindingPrefix, tree_id),
      llvm::FunctionType::get(pointer, {pointer}, false));
  return builder.CreateCall(function, {binding});
}

inline void resolve_cuda_root_bindings(llvm::Module &module,
                                       const std::vector<int> &tree_ids) {
  for (auto iter = module.begin(); iter != module.end();) {
    auto &function = *iter++;
    auto name = function.getName();
    if (!name.consume_front(kCudaRootBindingPrefix)) {
      continue;
    }
    int tree_id = -1;
    TI_ERROR_IF(name.getAsInteger(10, tree_id) || tree_id < 0 ||
                    !function.isDeclaration(),
                "Invalid CUDA root binding symbol {}",
                function.getName().str());
    auto slot = std::lower_bound(tree_ids.begin(), tree_ids.end(), tree_id);
    TI_ERROR_IF(slot == tree_ids.end() || *slot != tree_id,
                "CUDA root binding {} has no declared resource", tree_id);
    while (!function.use_empty()) {
      auto *call = llvm::dyn_cast<llvm::CallInst>(*function.user_begin());
      TI_ASSERT(call && call->getCalledFunction() == &function &&
                call->arg_size() == 1 && call->getType()->isPointerTy());
      llvm::IRBuilder<> builder(call);
      llvm::Value *root = call->getArgOperand(0);
      if (tree_ids.size() > 1) {
        auto *pointer = builder.getPtrTy();
        auto *address = builder.CreateInBoundsGEP(
            pointer, root, builder.getInt64(slot - tree_ids.begin()));
        auto *load = builder.CreateAlignedLoad(pointer, address, llvm::Align(8),
                                               "cuda_snode_root");
        load->setMetadata(llvm::LLVMContext::MD_invariant_load,
                          llvm::MDNode::get(module.getContext(), {}));
        root = load;
      }
      call->replaceAllUsesWith(root);
      call->eraseFromParent();
    }
    function.eraseFromParent();
  }
}

}  // namespace taichi::lang::LLVM
