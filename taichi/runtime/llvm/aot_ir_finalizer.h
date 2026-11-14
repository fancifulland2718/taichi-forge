#pragma once

#include <cstdint>
#include <map>
#include <set>

#include "llvm/Config/llvm-config.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Utils/Mem2Reg.h"
#include "taichi/runtime/llvm/aot_code_units.h"

namespace taichi::lang::LLVM {

// Build-owned artifact finalization. Neither the Program's FunctionCodeCache nor
// its original modules enter this object. Keys cover the complete normalized
// input and policy; output hashes continue to describe the actual final bytes.
class AotIRFinalizer {
 public:
  void export_units(llvm::Module &module,
                    bool optimize,
                    bool allow_resource_relocations) {
    export_code_units(
        module, [](const auto &, const auto &) {}, allow_resource_relocations,
        nullptr, true, nullptr,
        [&](llvm::Module &unit) {
          const auto key = input_key(unit, optimize);
          const auto found = units_.find(key);
          if (found != units_.end()) {
            ++unit_cache_hits_;
            return found->second.artifact;
          }
          if (optimize) {
            clean(unit);
            ++cleaned_units_;
          }
          Unit entry;
          entry.references = referenced_symbols(unit);
          entry.artifact = encode_code_unit(unit);
          symbols_.emplace(entry.artifact.symbol, key);
          files_.emplace(entry.artifact.filename, key);
          auto [item, inserted] = units_.emplace(key, std::move(entry));
          return item->second.artifact;
        });
  }

  void finalize_root(std::unique_ptr<llvm::Module> &module, bool optimize) {
    const auto key = input_key(*module, optimize);
    const auto found = roots_.find(key);
    if (found != roots_.end()) {
      auto decoded = decode_llvm_module(
          llvm::MemoryBufferRef(found->second, "finalized_aot_root"),
          module->getContext(), LLVMModuleEncoding::bitcode);
      TI_ERROR_IF(!decoded, "Cannot restore finalized AOT root");
      module = std::move(*decoded);
      ++root_cache_hits_;
      return;
    }
    if (optimize) {
      clean(*module);
      ++cleaned_roots_;
    }
    // The format uses a flat closure. Recompute it from live references after
    // cleanup, including references inside shared units, before serialization.
    std::set<std::string> dependencies;
    std::function<void(const std::string &)> visit = [&](const auto &symbol) {
      const auto found = symbols_.find(symbol);
      if (found == symbols_.end())
        return;  // Runtime/external symbols are not library units.
      const auto &unit = units_.at(found->second);
      if (!dependencies.insert(unit.artifact.filename).second)
        return;
      for (const auto &reference : unit.references)
        visit(reference);
    };
    for (const auto &symbol : referenced_symbols(*module))
      visit(symbol);
    if (auto *old = module->getNamedMetadata(kCodeUnits)) {
      for (const auto *item : old->operands()) {
        const auto file =
            llvm::cast<llvm::MDString>(item->getOperand(0))->getString().str();
        TI_ERROR_IF(files_.count(file) == 0,
                    "Unknown AOT dependency during finalization: {}", file);
      }
      module->eraseNamedMetadata(old);
    }
    for (const auto &file : dependencies) {
      const auto digest = file.substr(11, 64);  // code_units/<sha256>.bc
      module->getOrInsertNamedMetadata(kCodeUnits)
          ->addOperand(llvm::MDNode::get(
              module->getContext(),
              {llvm::MDString::get(module->getContext(), file),
               llvm::MDString::get(module->getContext(), digest)}));
    }
    roots_.emplace(key, serialize(*module));
  }

  // Emit only units reachable from this finalized root. A discarded call must
  // not leave a required dependency or an unreferenced payload in the archive.
  void emit_dependencies(
      const llvm::Module &module,
      const std::function<void(const std::string &, const std::string &)> &emit)
      const {
    const auto *metadata = module.getNamedMetadata(kCodeUnits);
    if (!metadata)
      return;
    for (const auto *item : metadata->operands()) {
      const auto file =
          llvm::cast<llvm::MDString>(item->getOperand(0))->getString().str();
      const auto found = files_.find(file);
      TI_ERROR_IF(found == files_.end(), "Missing finalized AOT code unit {}",
                  file);
      const auto &artifact = units_.at(found->second).artifact;
      emit(file, artifact.bytes);
    }
  }

  std::map<std::string, std::uint64_t> statistics() const {
    std::uint64_t bytes = 0;
    for (const auto &[_, key] : files_)
      bytes += units_.at(key).artifact.bytes.size();
    return {{"shared_code_units", files_.size()},
            {"serialized_code_unit_bytes", bytes},
            {"aot_cleaned_units", cleaned_units_},
            {"aot_cleaned_roots", cleaned_roots_},
            {"aot_unit_finalization_hits", unit_cache_hits_},
            {"aot_root_finalization_hits", root_cache_hits_}};
  }

 private:
  struct Unit {
    CodeUnitArtifact artifact;
    std::set<std::string> references;
  };
  static std::string serialize(const llvm::Module &module) {
    std::string bytes;
    llvm::raw_string_ostream stream(bytes);
    encode_llvm_module(module, stream, LLVMModuleEncoding::bitcode);
    return bytes;
  }
  static std::string input_key(const llvm::Module &module, bool optimize) {
    return aot_artifact_sha256(
        std::string(optimize ? "aot-local-v1:" : "aot-unchanged:") +
        LLVM_VERSION_STRING + ":" + serialize(module));
  }
  static std::set<std::string> referenced_symbols(const llvm::Module &module) {
    std::set<std::string> result;
    for (const auto &function : module)
      if (function.isDeclaration() && !function.use_empty())
        result.insert(function.getName().str());
    return result;
  }
  static void clean(llvm::Module &module) {
    TI_ERROR_IF(module.getDataLayoutStr().empty(),
                "AOT cleanup requires an effective target DataLayout");
    llvm::ModuleAnalysisManager mam;
    llvm::CGSCCAnalysisManager cgam;
    llvm::FunctionAnalysisManager fam;
    llvm::LoopAnalysisManager lam;
    llvm::PassBuilder passes;
    passes.registerModuleAnalyses(mam);
    passes.registerCGSCCAnalyses(cgam);
    passes.registerFunctionAnalyses(fam);
    passes.registerLoopAnalyses(lam);
    passes.crossRegisterProxies(lam, fam, cgam, mam);
    llvm::FunctionPassManager pipeline;
    pipeline.addPass(llvm::PromotePass());
    pipeline.addPass(llvm::EarlyCSEPass());
    pipeline.addPass(llvm::InstCombinePass());
    pipeline.addPass(llvm::SimplifyCFGPass());
    pipeline.addPass(llvm::DCEPass());
    for (auto &function : module)
      if (!function.isDeclaration() && !function.hasOptNone())
        pipeline.run(function, fam);
    llvm::GlobalDCEPass().run(module, mam);
    auto *policy = module.getOrInsertNamedMetadata("taichi.aot.cleanup.v1");
    policy->clearOperands();
    policy->addOperand(llvm::MDNode::get(
        module.getContext(),
        {llvm::MDString::get(module.getContext(),
                             "mem2reg-earlycse-instcombine-simplifycfg-dce-v1"),
         llvm::MDString::get(module.getContext(), LLVM_VERSION_STRING)}));
  }
  std::map<std::string, Unit> units_;
  std::map<std::string, std::string> symbols_, files_, roots_;
  std::uint64_t cleaned_units_{0}, cleaned_roots_{0};
  std::uint64_t unit_cache_hits_{0}, root_cache_hits_{0};
};

}  // namespace taichi::lang::LLVM
