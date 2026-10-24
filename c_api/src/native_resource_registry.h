#pragma once

#include "taichi_core_impl.h"
#include "taichi/runtime/native_operation_registry.h"

namespace capi {

std::unique_ptr<taichi::lang::NativePreparationContext>
make_native_preparation_context(Runtime &runtime);

// Cold owner/dependency registry. Entries are created only from live parents,
// so process-monotonic handle order also gives a resource destruction order.
// Retired entries remain tombstones until their public handle is destroyed.
class NativeResourceRegistry {
 public:
  using ParentLinks = std::vector<std::pair<std::string, TiNativeResource>>;
  explicit NativeResourceRegistry(Runtime &runtime) : runtime_(runtime) {
  }
  TiNativeResource publish(std::shared_ptr<taichi::lang::NativeResource> object,
                           std::vector<TiMemory> memories,
                           std::vector<TiNativeResource> parents,
                           ParentLinks parent_links = {});
  // Cold import transaction. Inputs are original objects, not reconstruction
  // recipes. Private input handles preserve named edges and retire with root.
  TiNativeResource publish_borrowed(
      std::shared_ptr<taichi::lang::NativeResource> object,
      const std::vector<std::pair<std::string,
          std::shared_ptr<taichi::lang::NativeResource>>> &inputs);
  std::shared_ptr<taichi::lang::NativeResource> get(
      TiNativeResource handle) const;
  TiNativeResource resolve(TiNativeResource handle,
                            const std::vector<std::string> &path) const;
  void destroy(TiNativeResource handle);
  void retire_memory(TiMemory memory);
  void clear(bool synchronize);

 private:
  struct Entry {
    std::shared_ptr<taichi::lang::NativeResource> object;
    std::vector<TiMemory> memories;
    std::vector<TiNativeResource> parents;
    ParentLinks parent_links;
    std::vector<TiNativeResource> owned_parents;
  };
  void retire(std::vector<TiNativeResource> roots);
  void remove_edges(TiNativeResource handle, const Entry &entry);
  Runtime &runtime_;
  std::unordered_map<TiNativeResource, Entry> entries_;
  std::unordered_map<TiMemory, std::unordered_set<TiNativeResource>> memories_;
  std::unordered_map<TiNativeResource, std::unordered_set<TiNativeResource>>
      children_;
};
}  // namespace capi
