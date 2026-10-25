#pragma once

#include "llvm/IR/Constants.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "taichi/common/logging.h"
#include "taichi/codegen/runtime_state.h"

namespace taichi::lang::LLVM {

// This describes the actual compiled context layout, not a validator/cache
// version. Version 2 appends exactly one execution-instance temporary pointer.
// Module flags survive cloning and BC caches; PTX/cubin exports carry the same
// contract through the existing AOT kernel launch ABI declaration.
inline constexpr unsigned kRuntimeContextABI = 2;
inline constexpr const char *kRuntimeContextABIFlag =
    "taichi.runtime.context_abi";
inline constexpr const char *kRuntimeStateFlag = "taichi.runtime.state_use";

// Code qualification, independent of RuntimeContext layout and state uses.
// Version 1 resolves sparse allocation through the source runtime while list
// managers and diagnostics can belong to an execution-owned runtime view.
inline constexpr unsigned kSparseRuntimeViewABI = 1;
inline constexpr const char *kSparseRuntimeViewABIFlag =
    "taichi.runtime.sparse_view_abi";

inline unsigned sparse_runtime_view_abi(const llvm::Module &module) {
  const auto *flag = module.getModuleFlag(kSparseRuntimeViewABIFlag);
  if (!flag)
    return 0;
  const auto *value = llvm::mdconst::dyn_extract<llvm::ConstantInt>(flag);
  TI_ERROR_IF(!value || !value->equalsInt(kSparseRuntimeViewABI),
              "Unsupported compiled sparse runtime view contract");
  return kSparseRuntimeViewABI;
}

inline std::uint32_t runtime_state_use(const llvm::Module &module) {
  const auto *flag = module.getModuleFlag(kRuntimeStateFlag);
  if (!flag)
    return 0;
  const auto *value = llvm::mdconst::dyn_extract<llvm::ConstantInt>(flag);
  TI_ERROR_IF(!value || value->getValue().getActiveBits() > 32,
              "Invalid compiled runtime-state inventory");
  return static_cast<std::uint32_t>(value->getZExtValue());
}

inline void declare_runtime_state_use(llvm::Module &module,
                                      std::uint32_t state) {
  TI_ERROR_IF(module.getModuleFlag(kRuntimeStateFlag),
              "Runtime-state inventory was already declared");
  module.addModuleFlag(llvm::Module::Error, kRuntimeStateFlag, state);
}

inline unsigned runtime_context_abi(const llvm::Module &module) {
  const auto *flag = module.getModuleFlag(kRuntimeContextABIFlag);
  if (!flag)
    return 1;
  const auto *value = llvm::mdconst::dyn_extract<llvm::ConstantInt>(flag);
  TI_ERROR_IF(!value || !value->equalsInt(kRuntimeContextABI),
              "Unsupported compiled RuntimeContext layout");
  return kRuntimeContextABI;
}

inline void declare_runtime_context_abi(llvm::Module &module) {
  TI_ERROR_IF(module.getModuleFlag(kRuntimeContextABIFlag),
              "RuntimeContext layout was already declared");
  module.addModuleFlag(llvm::Module::Error, kRuntimeContextABIFlag,
                       kRuntimeContextABI);
}

}  // namespace taichi::lang::LLVM
