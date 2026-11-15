#pragma once

#include <mutex>
#include "taichi/common/serialization.h"
#include "taichi/program/function.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/visitors.h"
#include "taichi/runtime/llvm/aot_code_units.h"

namespace taichi::lang {
class Kernel;
}

namespace taichi::lang::LLVM {

class FunctionTaskIndependence : public BasicStmtVisitor {
 public:
  bool independent{true};
  std::unordered_set<Function *> visited;
  std::unordered_set<Function *> active;
  using BasicStmtVisitor::visit;
  FunctionTaskIndependence() {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }
  void visit(InternalFuncStmt *) override { independent = false; }
  void visit(ExternalFuncCallStmt *) override { independent = false; }
  void visit(ThreadLocalPtrStmt *) override { independent = false; }
  void visit(BlockLocalPtrStmt *) override { independent = false; }
  void visit(GlobalTemporaryStmt *) override { independent = false; }
  void visit(FuncCallStmt *stmt) override {
    if (active.count(stmt->func)) {
      independent = false;
    } else if (visited.insert(stmt->func).second) {
      active.insert(stmt->func);
      stmt->func->ir->accept(this);
      active.erase(stmt->func);
    }
  }
};

// A producer transaction or JIT compiler owns these immutable, typed units.
// The persistent kernel cache keeps self-contained modules. Reusing one
// function skips LLVM body generation; it does not bypass resource analysis or
// treat a caller's configuration as the identity of every function it calls.
inline constexpr char kImportedFunctionUnits[] = "taichi.function.units.v1";

class FunctionCodeCache {
 public:
  struct Unit {
    std::string symbol;
    std::string digest;
    std::string bitcode;
  };
  struct Entry {
    std::string symbol;
    std::string target;
    std::string layout;
    std::vector<std::shared_ptr<const Unit>> units;
  };

  std::unique_lock<std::mutex> lock_function(const std::string &key) {
    std::shared_ptr<std::mutex> function_mutex;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto &slot = function_mutexes_[key];
      if (!slot)
        slot = std::make_shared<std::mutex>();
      function_mutex = slot;
    }
    // Keep unrelated units parallel. Recursive call graphs are excluded by
    // the proof above, so nested compilation follows an acyclic dependency.
    return std::unique_lock<std::mutex>(*function_mutex);
  }

  std::shared_ptr<const Entry> find(const std::string &key) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = entries_.find(key);
    if (found == entries_.end())
      return nullptr;
    ++hits_;
    return found->second;
  }

  void record_body_emission() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++emissions_;
  }

  std::shared_ptr<const Entry> record(const std::string &key,
                                      llvm::Module &module,
                                      llvm::Function *function) {
    auto entry = std::make_shared<Entry>();
    entry->target = module.getTargetTriple();
    entry->layout = module.getDataLayoutStr();
    bool recorded_root = false;
    std::map<const llvm::Function *, std::string> known;
    if (const auto *metadata = module.getNamedMetadata(kImportedFunctionUnits))
      for (const auto *item : metadata->operands()) {
        TI_ERROR_IF(item->getNumOperands() != 2,
                    "Malformed producer function unit reference");
        auto *value = llvm::dyn_cast<llvm::ValueAsMetadata>(item->getOperand(0));
        auto *digest = llvm::dyn_cast<llvm::MDString>(item->getOperand(1));
        if (value && digest)
          if (auto *source = llvm::dyn_cast<llvm::Function>(value->getValue()))
            known.emplace(source, digest->getString().str());
      }
    export_code_units(module, [](const auto &, const auto &) {},
        false, function, false, nullptr, {}, true,
        [&](const llvm::Function *source, const CodeUnitArtifact &artifact) {
          std::lock_guard<std::mutex> lock(mutex_);
          auto &unit = units_[artifact.digest];
          if (!unit) {
            unit = std::make_shared<Unit>(Unit{
                artifact.symbol, artifact.digest, artifact.bytes});
            bytes_ += artifact.bytes.size();
          }
          entry->units.push_back(unit);
          if (source == function) {
            recorded_root = true;
            entry->symbol = artifact.symbol;
          }
        },
        [&](const llvm::Function *source) -> std::optional<CodeUnitArtifact> {
          const auto found = known.find(source);
          if (found == known.end())
            return std::nullopt;
          std::lock_guard<std::mutex> lock(mutex_);
          const auto unit = units_.find(found->second);
          if (unit == units_.end())
            return std::nullopt;
          ++record_reuses_;
          const auto &stored = *unit->second;
          return CodeUnitArtifact{stored.symbol, stored.digest,
                                  "code_units/" + stored.digest + ".bc",
                                  stored.bitcode};
        });
    std::lock_guard<std::mutex> lock(mutex_);
    // A supported dependency cannot stand in for an unsupported selected root.
    if (!recorded_root)
      return nullptr;
    return entries_.emplace(key, std::move(entry)).first->second;
  }

  std::map<std::string, std::uint64_t> statistics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {{"llvm_function_body_emissions", emissions_},
            {"llvm_function_body_reuses", hits_},
            {"cached_llvm_functions", entries_.size()},
            {"cached_llvm_function_units", units_.size()},
            {"llvm_function_unit_record_reuses", record_reuses_},
            {"cached_llvm_function_bytes", bytes_}};
  }

 private:
  mutable std::mutex mutex_;
  std::map<std::string, std::shared_ptr<const Entry>> entries_;
  std::map<std::string, std::shared_ptr<const Unit>> units_;
  std::map<std::string, std::shared_ptr<std::mutex>> function_mutexes_;
  std::uint64_t emissions_{0}, hits_{0}, bytes_{0}, record_reuses_{0};
};

struct FunctionCodeContext {
  std::shared_ptr<FunctionCodeCache> cache;
  std::string target;

  std::string identity(const Function &function,
                       const std::vector<int> &root_bindings) const {
    const auto &body = function.try_get_ast_serialization_data();
    // Direct SNode relocations need their own slot mapping. Captured arrays
    // already have a typed function ABI and do not depend on instance storage.
    if (!body || !function.source_snode_tree_dependencies().empty())
      return {};
    FunctionTaskIndependence proof;
    function.ir->accept(&proof);
    if (!proof.independent)
      return {};
    BinaryOutputSerializer serializer;
    serializer.initialize();
    serializer(target);
    serializer(function.parameter_list);
    serializer(function.rets);
    serializer(*body);
    serializer(root_bindings);
    serializer.finalize();
    return aot_artifact_sha256(std::string(
        reinterpret_cast<const char *>(serializer.data.data()),
        serializer.data.size()));
  }
};

// Codegen tasks may run on worker threads. Bind the context to the compiling
// kernel instead of using thread-local or process-global active build state.
inline auto &function_code_contexts() {
  static std::pair<std::mutex,
      std::map<const Kernel *, FunctionCodeContext>> contexts;
  return contexts;
}

inline FunctionCodeContext function_code_context(const Kernel *kernel) {
  auto &[mutex, contexts] = function_code_contexts();
  std::lock_guard<std::mutex> lock(mutex);
  const auto found = contexts.find(kernel);
  return found == contexts.end() ? FunctionCodeContext{} : found->second;
}

class FunctionCodeScope {
 public:
  FunctionCodeScope(const Kernel *kernel,
                    std::shared_ptr<FunctionCodeCache> cache,
                    std::string target) : kernel_(cache ? kernel : nullptr) {
    if (!kernel_)
      return;
    auto &[mutex, contexts] = function_code_contexts();
    std::lock_guard<std::mutex> lock(mutex);
    TI_ERROR_IF(!contexts.emplace(kernel_, FunctionCodeContext{
                    std::move(cache), std::move(target)}).second,
                "Concurrent function-code scopes for the same kernel");
  }

  ~FunctionCodeScope() {
    if (!kernel_)
      return;
    auto &[mutex, contexts] = function_code_contexts();
    std::lock_guard<std::mutex> lock(mutex);
    contexts.erase(kernel_);
  }

  FunctionCodeScope(const FunctionCodeScope &) = delete;
  FunctionCodeScope &operator=(const FunctionCodeScope &) = delete;

 private:
  const Kernel *kernel_;
};

// These references are created only while assembling an unoptimized codegen
// module. They prove which immutable unit supplied a definition; a coincident
// local symbol name is never sufficient. Extraction drops this named metadata.
inline void register_function_code(llvm::Module &module,
                                   const FunctionCodeCache::Entry &entry,
                                   llvm::Function *function) {
  if (entry.units.empty())
    return;
  const auto &unit = *entry.units.back();
  // Registering a just-emitted root does not change its body or producer name.
  module.getOrInsertNamedMetadata(kImportedFunctionUnits)->addOperand(
      llvm::MDNode::get(module.getContext(),
          {llvm::ValueAsMetadata::get(function),
           llvm::MDString::get(module.getContext(), unit.digest)}));
}

inline llvm::Function *import_function_code(
    llvm::Module &module, const FunctionCodeCache::Entry &entry) {
  TI_ERROR_IF(entry.target != module.getTargetTriple() ||
                  entry.layout != module.getDataLayoutStr(),
              "Producer function unit target/layout mismatch");
  std::map<std::string, llvm::Function *> available;
  if (auto *metadata = module.getNamedMetadata(kImportedFunctionUnits))
    for (const auto *item : metadata->operands()) {
      TI_ERROR_IF(item->getNumOperands() != 2,
                  "Malformed producer function unit reference");
      auto *value = llvm::dyn_cast<llvm::ValueAsMetadata>(item->getOperand(0));
      auto *digest = llvm::dyn_cast<llvm::MDString>(item->getOperand(1));
      if (value && digest)
        if (auto *function = llvm::dyn_cast<llvm::Function>(value->getValue()))
          if (!function->isDeclaration())
            available.emplace(digest->getString().str(), function);
    }
  llvm::Linker linker(module);
  std::vector<llvm::Function *> imported;
  llvm::Function *result = nullptr;
  for (const auto &unit : entry.units) {
    const auto found = available.find(unit->digest);
    llvm::Function *function = nullptr;
    if (found != available.end()) {
      function = found->second;
      // Existing source definitions can have producer names. The canonical
      // symbol is needed only to resolve subsequent independent units.
      if (auto *collision = module.getFunction(unit->symbol);
          collision && collision != function) {
        TI_ERROR_IF(!collision->isDeclaration() ||
                        collision->getFunctionType() != function->getFunctionType() ||
                        collision->getCallingConv() != function->getCallingConv(),
                    "Conflicting producer function unit symbol");
        collision->replaceAllUsesWith(function);
        collision->eraseFromParent();
      }
      function->setName(unit->symbol);
      function->setLinkage(llvm::GlobalValue::ExternalLinkage);
    } else {
      TI_ERROR_IF(module.getFunction(unit->symbol) &&
                      !module.getFunction(unit->symbol)->isDeclaration(),
                  "Unverified producer function unit symbol collision");
      auto decoded = decode_llvm_module(
          llvm::MemoryBufferRef(unit->bitcode, unit->symbol), module.getContext(),
          LLVMModuleEncoding::bitcode);
      TI_ERROR_IF(!decoded, "Cannot decode producer function unit");
      if (auto *declaration = module.getFunction(unit->symbol)) {
        const auto *definition = (*decoded)->getFunction(unit->symbol);
        TI_ERROR_IF(!definition ||
                        declaration->getFunctionType() != definition->getFunctionType() ||
                        declaration->getCallingConv() != definition->getCallingConv(),
                    "Producer function unit declaration ABI mismatch");
      }
      TI_ERROR_IF(linker.linkInModule(std::move(*decoded)),
                  "Cannot import producer function unit");
      function = module.getFunction(unit->symbol);
      TI_ASSERT(function && !function->isDeclaration());
      module.getOrInsertNamedMetadata(kImportedFunctionUnits)->addOperand(
          llvm::MDNode::get(module.getContext(),
              {llvm::ValueAsMetadata::get(function),
               llvm::MDString::get(module.getContext(), unit->digest)}));
      available.emplace(unit->digest, function);
    }
    imported.push_back(function);
    result = function;
  }
  // Keep all dependencies externally resolvable until the complete closure has
  // linked, then restore the original JIT/local-function semantics.
  for (auto *function : imported)
    function->setLinkage(llvm::GlobalValue::InternalLinkage);
  TI_ASSERT(result);
  return result;
}

}  // namespace taichi::lang::LLVM
