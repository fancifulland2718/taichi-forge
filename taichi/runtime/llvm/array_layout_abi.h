#pragma once

#include <algorithm>
#include <vector>
#include "llvm/IR/Constants.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "taichi/common/core.h"

namespace taichi::lang::LLVM {
inline constexpr char kRuntimeAffineArrays[] = "taichi.ndarray.runtime_affine.v1";

inline void declare_runtime_affine_arrays(
    llvm::Module &module, const std::vector<std::vector<int>> &arguments) {
  auto *metadata = module.getOrInsertNamedMetadata(kRuntimeAffineArrays);
  for (const auto &argument : arguments) {
    std::vector<llvm::Metadata *> path;
    for (int index : argument)
      path.push_back(llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
          llvm::Type::getInt32Ty(module.getContext()), index)));
    metadata->addOperand(llvm::MDNode::get(module.getContext(), path));
  }
}

inline std::vector<std::vector<int>> runtime_affine_arrays(const llvm::Module &module) {
  std::vector<std::vector<int>> result;
  if (auto *metadata = module.getNamedMetadata(kRuntimeAffineArrays)) {
    for (const auto *node : metadata->operands()) {
      std::vector<int> argument;
      for (const auto &operand : node->operands()) {
        const auto *value = llvm::mdconst::dyn_extract<llvm::ConstantInt>(operand);
        TI_ERROR_IF(!value || value->getBitWidth() != 32 || value->getSExtValue() < 0,
                    "Invalid compiled ndarray layout qualification");
        argument.push_back(static_cast<int>(value->getSExtValue()));
      }
      TI_ERROR_IF(argument.empty(), "Missing compiled ndarray parameter path");
      result.push_back(std::move(argument));
    }
  }
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}
}  // namespace taichi::lang::LLVM
