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
class FunctionCodeCache {
 public:
  struct Entry {
    std::string symbol;
    std::string bitcode;
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

  void record(const std::string &key, llvm::Module &module,
              llvm::Function *function) {
    auto entry = std::make_shared<Entry>();
    export_code_units(module,
        [&](const std::string &, const std::string &bytes) {
          entry->bitcode = bytes;
        }, false, function, false, &entry->symbol);
    std::lock_guard<std::mutex> lock(mutex_);
    // Unsupported closures retain the normal lowering and archive path.
    if (!entry->bitcode.empty()) {
      const auto [found, inserted] = entries_.emplace(key, entry);
      if (inserted)
        bytes_ += entry->bitcode.size();
    }
  }

  std::map<std::string, std::uint64_t> statistics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {{"llvm_function_body_emissions", emissions_},
            {"llvm_function_body_reuses", hits_},
            {"cached_llvm_functions", entries_.size()},
            {"cached_llvm_function_bytes", bytes_}};
  }

 private:
  mutable std::mutex mutex_;
  std::map<std::string, std::shared_ptr<const Entry>> entries_;
  std::map<std::string, std::shared_ptr<std::mutex>> function_mutexes_;
  std::uint64_t emissions_{0}, hits_{0}, bytes_{0};
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

inline llvm::Function *import_function_code(
    llvm::Module &module, const FunctionCodeCache::Entry &entry) {
  if (auto *existing = module.getFunction(entry.symbol);
      existing && !existing->isDeclaration())
    return existing;
  auto decoded = decode_llvm_module(
      llvm::MemoryBufferRef(entry.bitcode, entry.symbol), module.getContext(),
      LLVMModuleEncoding::bitcode);
  TI_ERROR_IF(!decoded, "Cannot decode producer function unit");
  TI_ERROR_IF((*decoded)->getTargetTriple() != module.getTargetTriple() ||
                  (*decoded)->getDataLayout() != module.getDataLayout(),
              "Producer function unit target/layout mismatch");
  llvm::Linker linker(module);
  TI_ERROR_IF(linker.linkInModule(std::move(*decoded)),
              "Cannot import producer function unit");
  auto *function = module.getFunction(entry.symbol);
  TI_ASSERT(function && !function->isDeclaration());
  function->setLinkage(llvm::GlobalValue::InternalLinkage);
  return function;
}

}  // namespace taichi::lang::LLVM
