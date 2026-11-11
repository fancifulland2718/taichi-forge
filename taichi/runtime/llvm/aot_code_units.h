#pragma once

#include <fstream>
#include <unordered_set>

#include "taichi/common/virtual_dir.h"
#include "llvm/IR/Constants.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "taichi/runtime/llvm/aot_artifact_integrity.h"
#include "taichi/runtime/llvm/llvm_module_codec.h"

namespace taichi::lang::LLVM {

inline constexpr char kCodeUnits[] = "taichi.aot.code_units.v1";
inline constexpr char kRealFunctionUnit[] = "taichi.aot.real_function";

// Prototype library representation: a typed real-function closure is stored
// once, independently of its callers. Only local symbols are canonicalized;
// external symbols, target, layout, attributes and module ABI flags remain.
inline void export_code_units(llvm::Module &module,
                              const std::string &directory) {
  std::vector<llvm::Function *> roots;
  for (auto &function : module)
    if (!function.isDeclaration() && function.hasFnAttribute(kRealFunctionUnit))
      roots.push_back(&function);
  for (auto *root : roots) {
    std::unordered_set<const llvm::Value *> seen;
    std::unordered_set<const llvm::GlobalValue *> definitions;
    std::vector<const llvm::GlobalValue *> order;
    bool supported = true;
    std::function<void(const llvm::Value *)> visit =
        [&](const llvm::Value *value) {
          if (!seen.insert(value).second)
            return;
          if (llvm::isa<llvm::BlockAddress>(value)) {
            supported = false;
            return;
          }
          if (auto *global = llvm::dyn_cast<llvm::GlobalValue>(value)) {
            if (global->isDeclaration())
              return;
            if (global != root && !global->hasLocalLinkage()) {
              supported = false;
              return;
            }
            if (auto *variable = llvm::dyn_cast<llvm::GlobalVariable>(global))
              if (!variable->isConstant()) {
                supported = false;
                return;
              }
            definitions.insert(global);
            order.push_back(global);
          }
          if (auto *function = llvm::dyn_cast<llvm::Function>(value)) {
            for (const auto &block : *function)
              for (const auto &instruction : block)
                for (const auto &operand : instruction.operands())
                  visit(operand.get());
          } else if (auto *user = llvm::dyn_cast<llvm::User>(value)) {
            for (const auto &operand : user->operands())
              visit(operand.get());
          }
        };
    visit(root);
    if (!supported)
      continue;
    llvm::ValueToValueMapTy values;
    auto private_context = std::make_unique<llvm::LLVMContext>();
    auto unit =
        llvm::CloneModule(module, values, [&](const llvm::GlobalValue *value) {
          return definitions.count(value) != 0;
        });
    unit->setModuleIdentifier("aot_code_unit");
    unit->setSourceFileName("");
    // Entry annotations refer to the caller, not the function library.
    std::vector<llvm::NamedMDNode *> remove;
    for (auto &metadata : unit->named_metadata())
      if (metadata.getName() != "llvm.module.flags")
        remove.push_back(&metadata);
    for (auto *metadata : remove)
      unit->eraseNamedMetadata(metadata);
    for (size_t i = 0; i < order.size(); ++i) {
      auto *value = llvm::cast<llvm::GlobalValue>(values[order[i]]);
      value->setName("__ti_unit_" + std::to_string(i));
      value->setLinkage(i == 0 ? llvm::GlobalValue::ExternalLinkage
                               : llvm::GlobalValue::InternalLinkage);
    }
    llvm::ModuleAnalysisManager analysis;
    llvm::PassBuilder passes;
    passes.registerModuleAnalyses(analysis);
    llvm::GlobalDCEPass().run(*unit, analysis);
    // Unused declarations otherwise retain names of unrelated entry points.
    for (auto iterator = unit->begin(); iterator != unit->end();) {
      auto &function = *iterator++;
      if (function.isDeclaration() && function.use_empty())
        function.eraseFromParent();
    }
    // LLVM assigns context-local suffixes to identified structs during task
    // cloning. Normalize only private copies, retaining the complete layouts.
    // Renaming a type in the producer context would mutate other modules.
    std::string snapshot;
    llvm::raw_string_ostream snapshot_stream(snapshot);
    encode_llvm_module(*unit, snapshot_stream, LLVMModuleEncoding::bitcode);
    auto private_unit =
        decode_llvm_module(llvm::MemoryBufferRef(snapshot, "unit"),
                           *private_context, LLVMModuleEncoding::bitcode);
    TI_ERROR_IF(!private_unit, "Cannot normalize AOT code unit");
    unit = std::move(*private_unit);
    size_t type_index = 0;
    for (auto *type : unit->getIdentifiedStructTypes())
      if (!type->isOpaque())
        type->setName("__ti_unit_type_" + std::to_string(type_index++));
    std::string canonical;
    llvm::raw_string_ostream canonical_stream(canonical);
    encode_llvm_module(*unit, canonical_stream, LLVMModuleEncoding::bitcode);
    const auto identity = aot_artifact_sha256(canonical);
    auto *exported = unit->getFunction("__ti_unit_0");
    const auto symbol = "__ti_unit_" + identity;
    exported->setName(symbol);
    std::string bytes;
    llvm::raw_string_ostream stream(bytes);
    encode_llvm_module(*unit, stream, LLVMModuleEncoding::bitcode);
    const auto digest = aot_artifact_sha256(bytes);
    const auto filename = "code_units/" + digest + ".bc";
    taichi::create_directories(taichi::join_path(directory, "code_units"));
    std::ofstream output(taichi::join_path(directory, filename),
                         std::ios::binary);
    output.write(bytes.data(), bytes.size());
    TI_ERROR_IF(!output, "Cannot write AOT code unit {}", filename);
    root->deleteBody();
    root->setLinkage(llvm::GlobalValue::ExternalLinkage);
    if (auto *existing = module.getFunction(symbol)) {
      TI_ERROR_IF(existing->getFunctionType() != root->getFunctionType(),
                  "AOT code unit function type mismatch");
      root->replaceAllUsesWith(existing);
      root->eraseFromParent();
    } else {
      root->setName(symbol);
    }
    module.getOrInsertNamedMetadata(kCodeUnits)
        ->addOperand(llvm::MDNode::get(
            module.getContext(),
            {llvm::MDString::get(module.getContext(), filename),
             llvm::MDString::get(module.getContext(), digest)}));
    unit.reset();  // The private LLVMContext must outlive its module.
  }
  llvm::ModuleAnalysisManager analysis;
  llvm::PassBuilder passes;
  passes.registerModuleAnalyses(analysis);
  llvm::GlobalDCEPass().run(module, analysis);
}

inline void link_code_units(llvm::Module &module,
                            const io::VirtualDir &directory) {
  auto *metadata = module.getNamedMetadata(kCodeUnits);
  if (!metadata)
    return;
  std::vector<std::pair<std::string, std::string>> dependencies;
  for (const auto *item : metadata->operands()) {
    TI_ERROR_IF(item->getNumOperands() != 2,
                "Malformed AOT code unit reference");
    auto *name = llvm::dyn_cast<llvm::MDString>(item->getOperand(0));
    auto *hash = llvm::dyn_cast<llvm::MDString>(item->getOperand(1));
    TI_ERROR_IF(!name || !hash, "Malformed AOT code unit identity");
    const auto digest = hash->getString().str();
    TI_ERROR_IF(
        digest.size() != 64 ||
            digest.find_first_not_of("0123456789abcdef") != std::string::npos ||
            name->getString() != "code_units/" + digest + ".bc",
        "Invalid AOT code unit path");
    dependencies.emplace_back(name->getString().str(), digest);
  }
  module.eraseNamedMetadata(metadata);
  llvm::Linker linker(module);
  std::unordered_set<std::string> linked;
  for (const auto &[name, digest] : dependencies) {
    if (!linked.insert(digest).second)
      continue;
    size_t size = 0;
    TI_ERROR_IF(!directory.get_file_size(name, size),
                "Missing AOT code unit {}", name);
    std::string bytes(size, '\0');
    TI_ERROR_IF(directory.load_file(name, bytes.data(), size) != size,
                "Cannot read AOT code unit {}", name);
    TI_ERROR_IF(aot_artifact_sha256(bytes) != digest,
                "AOT code unit checksum mismatch: {}", name);
    auto unit =
        decode_llvm_module(llvm::MemoryBufferRef(bytes, name),
                           module.getContext(), LLVMModuleEncoding::bitcode);
    TI_ERROR_IF(!unit, "Invalid AOT code unit {}: {}", name,
                unit ? "" : llvm::toString(unit.takeError()));
    TI_ERROR_IF((*unit)->getTargetTriple() != module.getTargetTriple() ||
                    (*unit)->getDataLayout() != module.getDataLayout(),
                "AOT code unit target/layout mismatch");
    TI_ERROR_IF(linker.linkInModule(std::move(*unit)),
                "Cannot link AOT code unit {}", name);
  }
  // External linkage is needed only to resolve library references. Restore
  // the producer's local-function semantics before specialization/optimization.
  for (auto &function : module)
    if (!function.isDeclaration() &&
        function.hasFnAttribute(kRealFunctionUnit) &&
        function.getName().starts_with("__ti_unit_"))
      function.setLinkage(llvm::GlobalValue::InternalLinkage);
}

}  // namespace taichi::lang::LLVM
