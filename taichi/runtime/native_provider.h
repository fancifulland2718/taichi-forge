#pragma once

#include <atomic>
#include <memory>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>

#include "taichi/external_runtime/forge_runtime_provider.h"

namespace taichi {
class DynamicLoader;
}

namespace taichi::lang {

// The caller chooses the adapter and vendor locations. Neither AOT artifact
// decoding nor process-wide provider selection belongs in this ownership layer.
struct NativeProviderDefinition {
  std::string provider_id;
  std::string query_symbol;
  std::string supported_version_family;
};

struct NativeProviderInfo {
  std::string provider_id;
  std::string provider_name;
  std::string supported_version_family;
  std::string build_identity;
  uint64_t features{0};
  uint32_t required_symbol_count{0};
};

struct NativeProviderRuntimeInfo {
  uint32_t version_major{0};
  uint32_t version_minor{0};
  uint32_t version_patch{0};
  uint32_t cuda_runtime_version{0};
  std::string library_path;
  std::string build_version;
};

class NativeProviderError : public std::runtime_error {
 public:
  NativeProviderError(TiForgeRuntimeProviderResult result,
                      const std::string &message);
  TiForgeRuntimeProviderResult result() const {
    return result_;
  }

 private:
  TiForgeRuntimeProviderResult result_;
};

class NativeProviderRuntime;

// Loading queries adapter metadata only; vendor probing/creation is explicit.
// A runtime leases its module, and each domain plan must lease its runtime.
class NativeProviderModule
    : public std::enable_shared_from_this<NativeProviderModule> {
 public:
  static std::shared_ptr<NativeProviderModule> load(
      const NativeProviderDefinition &definition,
      const std::string &adapter_path);
  ~NativeProviderModule();
  NativeProviderModule(const NativeProviderModule &) = delete;
  NativeProviderModule &operator=(const NativeProviderModule &) = delete;

  const NativeProviderInfo &info() const {
    return info_;
  }
  NativeProviderRuntimeInfo probe(const std::string &library_path = {}) const;
  std::shared_ptr<NativeProviderRuntime> create_runtime(
      const std::string &library_path = {});
  // Domain-specific plan calls share the adapter's result/error contract.
  void check(TiForgeRuntimeProviderResult result, const char *operation) const;

 private:
  friend class NativeProviderRuntime;
  NativeProviderModule(const NativeProviderDefinition &definition,
                       const std::string &adapter_path);
  void destroy_runtime(TiForgeRuntimeProviderRuntime runtime) noexcept;

  std::unique_ptr<DynamicLoader> library_;
  TiForgeRuntimeProviderApi api_{};
  NativeProviderInfo info_;
  // On foreign cleanup failure the live runtime still needs its adapter code.
  // Retain the DLL until process exit instead of leaving dangling function
  // ptrs.
  std::atomic<bool> unload_safe_{true};
};

class NativeProviderRuntime {
 public:
  ~NativeProviderRuntime();
  NativeProviderRuntime(const NativeProviderRuntime &) = delete;
  NativeProviderRuntime &operator=(const NativeProviderRuntime &) = delete;

  const NativeProviderRuntimeInfo &info() const {
    return info_;
  }
  const NativeProviderInfo &provider_info() const {
    return module_->info();
  }
  TiForgeRuntimeProviderRuntime handle() const {
    return runtime_;
  }
  void check(TiForgeRuntimeProviderResult result, const char *operation) const {
    module_->check(result, operation);
  }

  // The returned table is valid only while this runtime is retained. Domain
  // factories also validate their required entries before creating any plans.
  template <typename Api>
  Api execution_api(uint32_t version =
                        TI_FORGE_RUNTIME_PROVIDER_EXECUTION_ABI_VERSION) const {
    static_assert(std::is_standard_layout_v<Api>);
    static_assert(offsetof(Api, struct_size) == 0);
    static_assert(offsetof(Api, execution_abi_version) == sizeof(uint32_t));
    Api api{};
    check(
        module_->api_.query_execution_api(runtime_, version, sizeof(api), &api),
        "query execution API");
    if (api.struct_size < sizeof(api) || api.execution_abi_version != version) {
      throw NativeProviderError(TI_FORGE_RUNTIME_PROVIDER_ERROR_ABI_MISMATCH,
                                provider_info().provider_id +
                                    ": invalid execution API size or version");
    }
    return api;
  }

 private:
  friend class NativeProviderModule;
  explicit NativeProviderRuntime(std::shared_ptr<NativeProviderModule> module);

  std::shared_ptr<NativeProviderModule> module_;
  TiForgeRuntimeProviderRuntime runtime_{nullptr};
  NativeProviderRuntimeInfo info_;
};

struct NativeProviderLocation {
  std::string adapter_path;
  // Empty requests the adapter's platform-default vendor search.
  std::string library_path;
};
using NativeProviderLocations = std::map<std::string, NativeProviderLocation>;

void validate_native_provider_location(const std::string &provider,
                                       const NativeProviderLocation &location);

// One cold preparation owns a configuration snapshot and a lazy dependency
// cache. Unused entries never load code. The session is not a process-wide
// registry; prepared plans retain runtime/module leases after it is destroyed.
class NativeProviderSession {
 public:
  explicit NativeProviderSession(NativeProviderLocations locations = {});
  const NativeProviderRuntimeInfo &probe(
      const NativeProviderDefinition &definition);
  // The C ABI runtime is parameterized only by its library location. Share one
  // owner per preparation; individual vendor plans remain independent.
  std::shared_ptr<NativeProviderRuntime> runtime(
      const NativeProviderDefinition &definition);

 private:
  struct Entry {
    std::shared_ptr<NativeProviderModule> module;
    std::string library_path;
    std::optional<NativeProviderRuntimeInfo> facts;
    std::shared_ptr<NativeProviderRuntime> runtime;
  };
  Entry &resolve(const NativeProviderDefinition &definition);
  NativeProviderLocations locations_;
  // Include the expected contract so conflicting definitions never reuse a
  // module whose identity was checked against a different query or family.
  using Key = std::tuple<std::string, std::string, std::string>;
  std::map<Key, Entry> entries_;
};

}  // namespace taichi::lang
