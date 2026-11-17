#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <set>

#include "llvm/Config/llvm-config.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Scalar/SROA.h"
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
    if (optimize)
      specialize_helpers(module);
    export_code_units(
        module, [](const auto &, const auto &) {}, allow_resource_relocations,
        nullptr, true, nullptr,
        [&](const std::string &snapshot) {
          const auto snapshot_key = input_key(snapshot, optimize);
          const auto previous = unit_snapshots_.find(snapshot_key);
          if (previous != unit_snapshots_.end()) {
            ++unit_snapshot_hits_;
            ++unit_cache_hits_;
            return units_.at(previous->second).artifact;
          }
          return finalize_code_unit_snapshot(snapshot, [&](llvm::Module &unit) {
            const auto key = input_key(unit, optimize);
            const auto found = units_.find(key);
            if (found != units_.end()) {
              unit_snapshots_.emplace(snapshot_key, key);
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
            unit_snapshots_.emplace(snapshot_key, key);
            return item->second.artifact;
          });
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
            {"aot_unit_snapshot_hits", unit_snapshot_hits_},
            {"aot_root_finalization_hits", root_cache_hits_}};
  }

 private:
  // Source ti.func bodies are compiled and cached as generic functions first.
  // Apply their existing backend-inline contract only to this build-owned
  // artifact, before extraction, hashes and publication. This exposes facts
  // already present at the call site; it never binds an entry's scene inputs.
  // Streaming seal and ordinary dump share this stage. The marker prevents a
  // sealed root from paying for it again at archive time.
  static void specialize_helpers(llvm::Module &module) {
    constexpr auto marker = "taichi.aot.helper_specialization.v3";
    if (module.getNamedMetadata(marker))
      return;
    const bool needed =
        std::any_of(module.begin(), module.end(), [](const auto &function) {
          return !function.isDeclaration() && !function.use_empty() &&
                 function.hasFnAttribute(kRealFunctionUnit) &&
                 function.hasFnAttribute(llvm::Attribute::AlwaysInline) &&
                 !function.hasFnAttribute(llvm::Attribute::NoInline) &&
                 !function.hasOptNone();
        });
    if (needed) {
      TI_COMPILE_PROFILER("cpp.compile.llvm.aot_helper_specialization");
      // Proxies must be destroyed before the analyses they refer to.
      llvm::LoopAnalysisManager lam;
      llvm::FunctionAnalysisManager fam;
      llvm::CGSCCAnalysisManager cgam;
      llvm::ModuleAnalysisManager mam;
      llvm::PassBuilder passes;
      passes.registerModuleAnalyses(mam);
      passes.registerCGSCCAnalyses(cgam);
      passes.registerFunctionAnalyses(fam);
      passes.registerLoopAnalyses(lam);
      passes.crossRegisterProxies(lam, fam, cgam, mam);
      // This manually assembled pipeline has no optnone instrumentation.
      // Honor per-function optimization requests explicitly, as clean() does.
      const auto run_functions = [&](llvm::FunctionPassManager &pipeline) {
        for (auto &function : module)
          if (!function.isDeclaration() && !function.hasOptNone())
            pipeline.run(function, fam);
      };
      // Promote each generic body's scalar temporaries before the inliner
      // replicates them into every call site. Promoting only after inlining
      // makes SSA construction operate on the much larger combined CFG.
      // Escaping/private argument buffers remain for post-inline SROA.
      llvm::FunctionPassManager preparation;
      preparation.addPass(llvm::PromotePass());
      run_functions(preparation);
      // Range dispatch is also marked alwaysinline, but its callback ABI is
      // consumed by prepare's argument-context and empty-group proofs. Keep
      // that boundary until those proofs have run. Numerical helper inlining
      // and scalar cleanup still happen here; restore backend attributes on
      // the private artifact so final device compilation can inline dispatch.
      std::vector<std::pair<llvm::Function *, llvm::AttributeList>> dispatchers;
      for (auto &function : module) {
        const auto name = function.getName();
        if (name.starts_with("gpu_parallel_range_for") ||
            name.starts_with("cpu_parallel_range_for")) {
          dispatchers.emplace_back(&function, function.getAttributes());
          function.removeFnAttr(llvm::Attribute::AlwaysInline);
          function.addFnAttr(llvm::Attribute::NoInline);
        }
      }
      llvm::ModulePassManager inline_pipeline;
      inline_pipeline.addPass(llvm::AlwaysInlinerPass());
      inline_pipeline.run(module, mam);
      for (auto &[function, attributes] : dispatchers)
        function->setAttributes(attributes);
      // Inlining can expose new promotable slots; clean those before SROA.
      llvm::FunctionPassManager cleanup;
      cleanup.addPass(llvm::PromotePass());
      // The generic helper ABI uses a private argument buffer. After inlining,
      // scalarize that local storage so the ordinary final cleanup can fold
      // fixed helper arguments while leaving dynamic entry loads intact.
      cleanup.addPass(llvm::SROAPass(llvm::SROAOptions::ModifyCFG));
      run_functions(cleanup);
      llvm::GlobalDCEPass().run(module, mam);
    }
    module.getOrInsertNamedMetadata(marker)->addOperand(llvm::MDNode::get(
        module.getContext(),
        {llvm::MDString::get(module.getContext(), "promote-inline-preserve-dispatch-v3"),
         llvm::MDString::get(module.getContext(), LLVM_VERSION_STRING)}));
  }

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
    return input_key(serialize(module), optimize);
  }
  static std::string input_key(const std::string &bytes, bool optimize) {
    return aot_artifact_sha256(
        std::string(optimize ? "aot-local-v2:" : "aot-unchanged:") +
        LLVM_VERSION_STRING + ":" + bytes);
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
    llvm::LoopAnalysisManager lam;
    llvm::FunctionAnalysisManager fam;
    llvm::CGSCCAnalysisManager cgam;
    llvm::ModuleAnalysisManager mam;
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
  // Exact input aliases only: retain hashes, not another copy of every
  // unoptimized bitcode body or its LLVMContext. Type-suffix differences still
  // take the canonical path and converge in units_.
  std::map<std::string, std::string> unit_snapshots_;
  std::map<std::string, std::string> symbols_, files_, roots_;
  std::uint64_t cleaned_units_{0}, cleaned_roots_{0};
  std::uint64_t unit_cache_hits_{0}, root_cache_hits_{0};
  std::uint64_t unit_snapshot_hits_{0};
};

}  // namespace taichi::lang::LLVM
