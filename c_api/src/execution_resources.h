#pragma once

#include <unordered_set>

#include "argument_binding.h"
#include "taichi/aot/execution_resources.h"

namespace capi {
// Private allocations retire after the owning execution's backend recordings.
class ExecutionResourceStorage {
 public:
  explicit ExecutionResourceStorage(Runtime &runtime) : runtime_(runtime) {
  }
  ~ExecutionResourceStorage();
  ExecutionResourceStorage(const ExecutionResourceStorage &) = delete;
  ExecutionResourceStorage &operator=(const ExecutionResourceStorage &) =
      delete;
  TiMemory allocate(uint64_t bytes);
  uint64_t bytes() const {
    return bytes_;
  }
  void remove_owned_dependencies(std::vector<TiMemory> &dependencies) const;
  void retain(const std::unordered_set<TiMemory> &memories);

 private:
  Runtime &runtime_;
  std::vector<TiMemory> memories_;
  std::unordered_map<TiMemory, uint64_t> sizes_;
  uint64_t bytes_{0};
};

// These host descriptors disappear after FixedExecution copies bindings.
// Only storage moves into the prepared owner.
struct ExecutionResourceBindings {
  struct ZeroInitialization {
    TiMemory memory;
    uint64_t bytes;
  };
  std::unique_ptr<ExecutionResourceStorage> storage;
  std::vector<TiArgument> arguments;
  std::vector<TiNamedNdArrayBinding> gradients;
  std::vector<ZeroInitialization> zero_initializations;
};
ExecutionResourceBindings bind_execution_resources(
    Runtime &runtime,
    taichi::lang::aot::Module &module,
    const taichi::lang::aot::ExecutionResources &resources,
    std::unordered_map<std::string, const TiArgument *> &arguments,
    std::unordered_map<std::string, const TiNamedNdArrayBinding *> &layouts);
}  // namespace capi
