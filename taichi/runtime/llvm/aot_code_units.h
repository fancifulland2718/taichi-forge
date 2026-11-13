#pragma once

#include <fstream>
#include <functional>
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

struct CodeUnitArtifact {
  std::string symbol;
  std::string digest;
  std::string filename;
  std::string bytes;
};

// The module is already a normalized private copy. AOT finalization may run
// here, before either the exported symbol or the content digest is published.
inline CodeUnitArtifact encode_code_unit(llvm::Module &unit) {
  std::string canonical;
  llvm::raw_string_ostream canonical_stream(canonical);
  encode_llvm_module(unit, canonical_stream, LLVMModuleEncoding::bitcode);
  CodeUnitArtifact result;
  result.symbol = "__ti_unit_" + aot_artifact_sha256(canonical);
  unit.getFunction("__ti_unit_0")->setName(result.symbol);
  llvm::raw_string_ostream stream(result.bytes);
  encode_llvm_module(unit, stream, LLVMModuleEncoding::bitcode);
  result.digest = aot_artifact_sha256(result.bytes);
  result.filename = "code_units/" + result.digest + ".bc";
  return result;
}

using CodeUnitFinalizer = std::function<CodeUnitArtifact(llvm::Module &)>;

// Prototype library representation: a typed real-function closure is stored
// once, independently of its callers. Only local symbols are canonicalized;
// external symbols, target, layout, attributes and module ABI flags remain.
inline void export_code_units(
    llvm::Module &module,
    const std::function<void(const std::string &, const std::string &)> &emit,
    bool allow_resource_relocations = true,
    llvm::Function *selected_root = nullptr,
    bool rewrite_caller = true,
    std::string *selected_symbol = nullptr,
    const CodeUnitFinalizer &finalize = {}) {
  std::vector<llvm::Function *> roots;
  for (auto &function : module)
    if (!function.isDeclaration() && function.hasFnAttribute(kRealFunctionUnit) &&
        (!selected_root || &function == selected_root))
      roots.push_back(&function);
  // Extract callees before callers. Otherwise each parent's closure embeds
  // its children's bodies, even when those same children get their own units
  // later. The entry's flat dependency list already retains every exported
  // unit, so parent units need only the canonical callee declarations.
  std::unordered_set<llvm::Function *> visited;
  std::vector<llvm::Function *> ordered;
  std::function<void(llvm::Function *)> order = [&](llvm::Function *function) {
    if (!visited.insert(function).second)
      return;
    for (auto &block : *function)
      for (auto &instruction : block)
        if (auto *call = llvm::dyn_cast<llvm::CallBase>(&instruction))
          if (auto *callee = llvm::dyn_cast<llvm::Function>(
                  call->getCalledOperand()->stripPointerCasts()))
            if (!callee->isDeclaration() && callee->hasFnAttribute(kRealFunctionUnit))
              order(callee);
    ordered.push_back(function);
  };
  if (!selected_root) {
    for (auto *root : roots)
      order(root);
    roots = std::move(ordered);
  }
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
            // The module's final resource numbering is assigned at archive
            // time. Do not persist a unit containing producer tree IDs before
            // that remap. Argument-only functions can be shared immediately.
            if (!allow_resource_relocations &&
                global->getName().starts_with("__ti_aot_tree_id_")) {
              supported = false;
              return;
            }
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
    // Imported units already contain canonical local names. Clear the whole
    // closure before assigning names, otherwise setName() silently adds a
    // suffix when another member still owns the desired name. Drop unused
    // declarations first, as they may also retain names from imported units.
    for (const auto *original : order)
      if (llvm::Value *value = values[original])
        value->setName("");
    for (size_t i = 0; i < order.size(); ++i)
      if (llvm::Value *value = values[order[i]])
        value->setName("__ti_unit_" + std::to_string(i));
    // LLVM linking can change definition/declaration insertion order without
    // changing a function. Canonical names alone do not canonicalize bitcode's
    // value numbering or its identified-type traversal order.
    unit->getFunctionList().sort([](const llvm::Function &a,
                                   const llvm::Function &b) {
      return a.getName() < b.getName();
    });
    std::vector<llvm::GlobalVariable *> globals;
    for (auto &global : unit->globals())
      globals.push_back(&global);
    std::sort(globals.begin(), globals.end(), [](const auto *a, const auto *b) {
      return a->getName() < b->getName();
    });
    for (auto *global : globals) {
      unit->removeGlobalVariable(global);
      unit->insertGlobalVariable(global);
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
    const auto artifact = finalize ? finalize(*unit) : encode_code_unit(*unit);
    const auto &symbol = artifact.symbol;
    if (selected_symbol)
      *selected_symbol = symbol;
    const auto &filename = artifact.filename;
    const auto &digest = artifact.digest;
    emit(filename, artifact.bytes);
    if (!rewrite_caller) {
      unit.reset();
      continue;
    }
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
  if (rewrite_caller) {
    llvm::ModuleAnalysisManager analysis;
    llvm::PassBuilder passes;
    passes.registerModuleAnalyses(analysis);
    llvm::GlobalDCEPass().run(module, analysis);
  }
}

inline void export_code_units(llvm::Module &module,
                              const std::string &directory) {
  export_code_units(module, [&](const std::string &filename,
                                const std::string &bytes) {
    taichi::create_directories(taichi::join_path(directory, "code_units"));
    std::ofstream output(taichi::join_path(directory, filename), std::ios::binary);
    output.write(bytes.data(), bytes.size());
    TI_ERROR_IF(!output, "Cannot write AOT code unit {}", filename);
  });
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
