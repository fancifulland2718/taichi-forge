#pragma once

#include <algorithm>
#include <map>
#include <vector>

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "taichi/common/core.h"

namespace taichi::lang::LLVM {

inline constexpr char kAotTreeIdPrefix[] = "__ti_aot_tree_id_";

inline std::string aot_tree_id_symbol(int id) {
  TI_ASSERT(id >= 0);
  return fmt::format("{}{}", kAotTreeIdPrefix, id);
}

inline llvm::Value *emit_aot_tree_id(llvm::Module &module,
                                     llvm::IRBuilder<> &builder,
                                     int id,
                                     bool cuda) {
  auto *type = builder.getInt32Ty();
  const auto name = aot_tree_id_symbol(id);
  auto *global = module.getNamedGlobal(name);
  if (!global) {
    global = new llvm::GlobalVariable(
        module, type, true, llvm::GlobalValue::ExternalLinkage, nullptr, name,
        nullptr, llvm::GlobalValue::NotThreadLocal, cuda ? 4 : 0);
  }
  return builder.CreateLoad(type, global, "aot_tree_id");
}

inline std::vector<std::pair<int, llvm::GlobalVariable *>> aot_tree_id_globals(
    llvm::Module &module) {
  std::vector<std::pair<int, llvm::GlobalVariable *>> result;
  for (auto &global : module.globals()) {
    auto name = global.getName();
    if (!name.consume_front(kAotTreeIdPrefix))
      continue;
    int id = -1;
    TI_ERROR_IF(
        name.getAsInteger(10, id) || id < 0 ||
            !global.getValueType()->isIntegerTy(32) || !global.isConstant(),
        "Invalid LLVM AOT tree relocation symbol {}", global.getName().str());
    for (auto *user : global.users()) {
      auto *load = llvm::dyn_cast<llvm::LoadInst>(user);
      TI_ERROR_IF(!load || load->getPointerOperand() != &global ||
                      !load->getType()->isIntegerTy(32),
                  "LLVM AOT tree relocation {} has a non-load use", id);
    }
    result.emplace_back(id, &global);
  }
  return result;
}

inline void remap_aot_tree_id_symbols(llvm::Module &module,
                                      const std::map<int, int> &mapping) {
  auto globals = aot_tree_id_globals(module);
  for (const auto &[id, global] : globals) {
    TI_ERROR_IF(mapping.find(id) == mapping.end(),
                "LLVM AOT tree relocation {} has no declared resource", id);
  }
  // Avoid collisions when a compacted ID equals another producer ID.
  for (const auto &[id, global] : globals)
    global->setName("");
  for (const auto &[id, global] : globals) {
    global->setName(aot_tree_id_symbol(mapping.at(id)));
  }
}

inline void resolve_aot_tree_ids(llvm::Module &module,
                                 const std::map<int, int> &mapping) {
  auto globals = aot_tree_id_globals(module);
  for (const auto &[id, global] : globals) {
    TI_ERROR_IF(mapping.find(id) == mapping.end() || mapping.at(id) < 0,
                "LLVM AOT tree relocation {} has no runtime binding", id);
  }
  for (const auto &[id, global] : globals) {
    auto *value =
        llvm::ConstantInt::get(global->getValueType(), mapping.at(id));
    while (!global->use_empty()) {
      auto *load = llvm::cast<llvm::LoadInst>(*global->user_begin());
      load->replaceAllUsesWith(value);
      load->eraseFromParent();
    }
    global->eraseFromParent();
  }
}

inline std::vector<int> prepare_ptx_tree_relocations(llvm::Module &module) {
  std::vector<int> result;
  for (const auto &[id, global] : aot_tree_id_globals(module)) {
    if (global->use_empty()) {
      global->eraseFromParent();
      continue;
    }
    TI_ERROR_IF(global->getAddressSpace() != 4,
                "CUDA AOT tree relocation must use constant address space");
    // The loader overwrites this constant once, before exposing the kernel.
    // externally_initialized prevents LLVM from folding the placeholder value.
    global->setInitializer(llvm::ConstantInt::get(global->getValueType(), id));
    global->setExternallyInitialized(true);
    result.push_back(id);
  }
  std::sort(result.begin(), result.end());
  return result;
}

}  // namespace taichi::lang::LLVM
