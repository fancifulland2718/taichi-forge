#pragma once

#include <functional>
#include <unordered_set>

#include "argument_binding.h"
#include "taichi/aot/execution_resources.h"
#include "taichi/runtime/native_memory_allocator.h"

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
  TiMemory allocate(uint64_t bytes,
                    taichi::lang::AllocUsage usage,
                    taichi::lang::NativeMemoryAllocator *allocator = nullptr);
  // Non-owning aliases; the execution resource closure retains their parents.
  TiMemory view(TiMemory parent, uint64_t offset, uint64_t bytes);
  // The enclosing prepared execution belongs to this same module and retires
  // before module roots. Only the alias handle is owned by this storage set.
  TiMemory module_view(taichi::lang::DevicePtr pointer, uint64_t bytes);
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

// Explicit binding families share owned arrays, never arbitrary public views.
// Each frame still owns its aliases and resolves public binding metadata.
struct SharedExecutionResources {
  std::shared_ptr<ExecutionResourceStorage> storage;
  std::unordered_map<std::string, TiNdArray> arrays;
  // Views feeding shared native resources have frozen public storage parents.
  // The caller validates those public parents before reusing these aliases.
  std::unordered_set<std::string> stable_names;
  bool sealed{false};
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
  // Affine members are native-only views, never disguised packed TiArguments.
  std::map<std::string, taichi::lang::NativeArrayBinding> native_arrays;
  std::vector<TiMemory> native_memories;
  std::unordered_set<std::string> layout_sources;
  std::vector<ZeroInitialization> zero_initializations;
};
ExecutionResourceBindings bind_execution_resources(
    Runtime &runtime,
    AotModule &module,
    const taichi::lang::aot::ExecutionResources &resources,
    std::unordered_map<std::string, const TiArgument *> &arguments,
    std::unordered_map<std::string, const TiNamedNdArrayBinding *> &layouts,
    SharedExecutionResources *shared = nullptr,
    const std::unordered_set<std::string> &frozen_metadata = {},
    const std::function<std::shared_ptr<taichi::lang::NativeMemoryAllocator>(
        const std::string &)> &allocator = {},
    bool prepared_external_access = false,
    // Called after the metadata closure, before allocators, uploads or aliases.
    const std::function<void()> &validate_metadata = {});
}  // namespace capi
