#include "taichi/runtime/native_provider.h"

#include <vector>

#include "taichi/common/dynamic_loader.h"

namespace taichi::lang {
namespace {

std::string copy_string(const char *text) {
  return text ? text : "";
}

NativeProviderRuntimeInfo copy_runtime_info(const TiForgeRuntimeInfo &info) {
  if (info.struct_size < sizeof(info)) {
    throw NativeProviderError(TI_FORGE_RUNTIME_PROVIDER_ERROR_ABI_MISMATCH,
                              "provider returned truncated runtime info");
  }
  return {info.version_major,
          info.version_minor,
          info.version_patch,
          info.cuda_runtime_version,
          copy_string(info.library_path),
          copy_string(info.build_version)};
}

}  // namespace

NativeProviderError::NativeProviderError(TiForgeRuntimeProviderResult result,
                                         const std::string &message)
    : std::runtime_error(message), result_(result) {
}

std::shared_ptr<NativeProviderModule> NativeProviderModule::load(
    const NativeProviderDefinition &definition,
    const std::string &adapter_path) {
  return std::shared_ptr<NativeProviderModule>(
      new NativeProviderModule(definition, adapter_path));
}

NativeProviderModule::NativeProviderModule(
    const NativeProviderDefinition &definition,
    const std::string &adapter_path) {
  if (definition.provider_id.empty() || definition.query_symbol.empty() ||
      definition.supported_version_family.empty() || adapter_path.empty()) {
    throw std::invalid_argument(
        "provider definition and adapter path must be set");
  }
  library_ = std::make_unique<DynamicLoader>(adapter_path);
  if (!library_->loaded()) {
    throw NativeProviderError(
        TI_FORGE_RUNTIME_PROVIDER_ERROR_RUNTIME_UNAVAILABLE,
        "cannot load provider adapter: " + adapter_path);
  }
  const auto query = reinterpret_cast<TiForgeRuntimeProviderQueryFn>(
      library_->load_function_optional(definition.query_symbol));
  if (!query) {
    throw NativeProviderError(
        TI_FORGE_RUNTIME_PROVIDER_ERROR_ABI_MISMATCH,
        "provider query symbol missing: " + definition.query_symbol);
  }
  const auto result =
      query(TI_FORGE_RUNTIME_PROVIDER_ABI_VERSION, sizeof(api_), &api_);
  // Do not invoke a returned function pointer until its containing API passed
  // validation, including when querying the provider itself failed.
  if (result != TI_FORGE_RUNTIME_PROVIDER_SUCCESS) {
    throw NativeProviderError(result, definition.provider_id +
                                          ": provider query failed (result " +
                                          std::to_string(result) + ")");
  }
  const auto &info = api_.info;
  constexpr uint64_t required_features =
      TI_FORGE_RUNTIME_PROVIDER_FEATURE_VERSION_QUERY |
      TI_FORGE_RUNTIME_PROVIDER_FEATURE_REQUIRED_SYMBOL_AUDIT |
      TI_FORGE_RUNTIME_PROVIDER_FEATURE_TRANSIENT_PROBE |
      TI_FORGE_RUNTIME_PROVIDER_FEATURE_EXECUTION_API;
  if (api_.struct_size < sizeof(api_) ||
      api_.provider_abi_version != TI_FORGE_RUNTIME_PROVIDER_ABI_VERSION ||
      info.struct_size < sizeof(info) ||
      info.provider_abi_version != TI_FORGE_RUNTIME_PROVIDER_ABI_VERSION ||
      copy_string(info.provider_id) != definition.provider_id ||
      copy_string(info.supported_version_family) !=
          definition.supported_version_family ||
      (info.features & required_features) != required_features ||
      (info.required_symbol_count == 0 &&
       !(info.features & TI_FORGE_RUNTIME_PROVIDER_FEATURE_EMBEDDED_RUNTIME)) ||
      !api_.probe_runtime ||
      !api_.create_runtime || !api_.destroy_runtime ||
      !api_.query_execution_api || !api_.get_last_error) {
    throw NativeProviderError(
        TI_FORGE_RUNTIME_PROVIDER_ERROR_ABI_MISMATCH,
        definition.provider_id + ": invalid provider ABI, identity or entries");
  }
  info_ = {copy_string(info.provider_id),
           copy_string(info.provider_name),
           copy_string(info.supported_version_family),
           copy_string(info.build_identity),
           info.features,
           info.required_symbol_count};
}

NativeProviderModule::~NativeProviderModule() {
  if (!unload_safe_.load()) {
    // No allocation or foreign call on this failure path. Deliberately keep the
    // loader as well as its OS reference alive after unrecoverable cleanup.
    (void)library_.release();
  }
}

void NativeProviderModule::check(TiForgeRuntimeProviderResult result,
                                 const char *operation) const {
  if (result == TI_FORGE_RUNTIME_PROVIDER_SUCCESS) {
    return;
  }
  const auto size = api_.get_last_error(nullptr, 0);
  std::vector<char> message(size ? size : 1, '\0');
  api_.get_last_error(message.data(), message.size());
  message.back() = '\0';
  throw NativeProviderError(
      result, info_.provider_id + ": " + operation + " failed (result " +
                  std::to_string(result) + "): " + message.data());
}

NativeProviderRuntimeInfo NativeProviderModule::probe(
    const std::string &library_path) const {
  TiForgeRuntimeInfo info{};
  info.struct_size = sizeof(info);
  check(api_.probe_runtime(
            library_path.empty() ? nullptr : library_path.c_str(), &info),
        "probe runtime");
  // Probe strings can be overwritten by the next probe on this thread.
  return copy_runtime_info(info);
}

std::shared_ptr<NativeProviderRuntime> NativeProviderModule::create_runtime(
    const std::string &library_path) {
  // Establish RAII before calling the provider: malformed outputs, failure with
  // a non-null handle and allocation failures all follow the same rollback.
  auto runtime = std::shared_ptr<NativeProviderRuntime>(
      new NativeProviderRuntime(shared_from_this()));
  TiForgeRuntimeInfo info{};
  info.struct_size = sizeof(info);
  check(
      api_.create_runtime(library_path.empty() ? nullptr : library_path.c_str(),
                          &runtime->runtime_, &info),
      "create runtime");
  if (!runtime->runtime_) {
    throw NativeProviderError(
        TI_FORGE_RUNTIME_PROVIDER_ERROR_INTERNAL,
        info_.provider_id + ": provider returned no runtime");
  }
  runtime->info_ = copy_runtime_info(info);
  return runtime;
}

void NativeProviderModule::destroy_runtime(
    TiForgeRuntimeProviderRuntime runtime) noexcept {
  // Successful domain-plan destruction must precede release of its runtime
  // lease. If a foreign provider nevertheless refuses cleanup, pin its code.
  try {
    const auto result = api_.destroy_runtime(runtime);
    if (result == TI_FORGE_RUNTIME_PROVIDER_SUCCESS) {
      return;
    }
    unload_safe_.store(false);
    TI_WARN(
        "{}: runtime cleanup failed (result {}); retaining provider library",
        info_.provider_id, static_cast<int>(result));
  } catch (...) {
    unload_safe_.store(false);
  }
}

NativeProviderRuntime::NativeProviderRuntime(
    std::shared_ptr<NativeProviderModule> module)
    : module_(std::move(module)) {
}

NativeProviderRuntime::~NativeProviderRuntime() {
  if (runtime_) {
    module_->destroy_runtime(runtime_);
  }
}

void validate_native_provider_location(const std::string &provider,
                                       const NativeProviderLocation &location) {
  if (provider.empty() || provider.find('\0') != std::string::npos ||
      location.adapter_path.empty() ||
      location.adapter_path.find('\0') != std::string::npos ||
      location.library_path.find('\0') != std::string::npos) {
    throw std::invalid_argument(
        "Provider name and adapter path must be nonempty; paths cannot contain "
        "NUL");
  }
}

void validate_native_library_location(const std::string &name,
                                      const std::string &path) {
  if (name.empty() || path.empty() || name.find('\0') != std::string::npos ||
      path.find('\0') != std::string::npos)
    throw std::invalid_argument(
        "Native library name/path must be nonempty without NUL");
}

NativeLibrary::NativeLibrary(std::string name, const std::string &path)
    : name_(std::move(name)) {
  validate_native_library_location(name_, path);
  library_ = std::make_unique<DynamicLoader>(path);
  if (!library_->loaded())
    throw NativeProviderError(
        TI_FORGE_RUNTIME_PROVIDER_ERROR_RUNTIME_UNAVAILABLE,
        "Cannot load native library " + name_ + ": " + path);
}

NativeLibrary::~NativeLibrary() = default;

void *NativeLibrary::symbol(const std::string &name) const {
  auto *result = library_->load_function_optional(name);
  if (!result)
    throw NativeProviderError(TI_FORGE_RUNTIME_PROVIDER_ERROR_ABI_MISMATCH,
                              name_ + ": missing native symbol " + name);
  return result;
}

NativeProviderSession::NativeProviderSession(NativeProviderLocations locations,
                                             NativeLibraryLocations libraries)
    : locations_(std::move(locations)),
      library_locations_(std::move(libraries)) {
  for (const auto &[provider, location] : locations_)
    validate_native_provider_location(provider, location);
  for (const auto &[name, path] : library_locations_)
    validate_native_library_location(name, path);
}

std::shared_ptr<NativeLibrary> NativeProviderSession::library(
    const std::string &name) {
  if (const auto found = libraries_.find(name); found != libraries_.end())
    return found->second;
  const auto location = library_locations_.find(name);
  if (location == library_locations_.end())
    throw NativeProviderError(
        TI_FORGE_RUNTIME_PROVIDER_ERROR_RUNTIME_UNAVAILABLE,
        "Native library location is not configured: " + name);
  auto result = std::make_shared<NativeLibrary>(name, location->second);
  libraries_.emplace(name, result);
  return result;
}

NativeProviderSession::Entry &NativeProviderSession::resolve(
    const NativeProviderDefinition &definition) {
  const Key key{definition.provider_id, definition.query_symbol,
                definition.supported_version_family};
  if (const auto found = entries_.find(key); found != entries_.end())
    return found->second;
  const auto configured = locations_.find(definition.provider_id);
  if (configured == locations_.end())
    throw NativeProviderError(
        TI_FORGE_RUNTIME_PROVIDER_ERROR_RUNTIME_UNAVAILABLE,
        "Provider location is not configured: " + definition.provider_id);
  const auto &location = configured->second;
  auto module = NativeProviderModule::load(definition, location.adapter_path);
  return entries_
      .emplace(key, Entry{std::move(module), location.library_path, {}})
      .first->second;
}

const NativeProviderRuntimeInfo &NativeProviderSession::probe(
    const NativeProviderDefinition &definition) {
  auto &entry = resolve(definition);
  if (!entry.facts)
    entry.facts = entry.module->probe(entry.library_path);
  return *entry.facts;
}

std::shared_ptr<NativeProviderRuntime> NativeProviderSession::runtime(
    const NativeProviderDefinition &definition) {
  auto &entry = resolve(definition);
  if (!entry.runtime)
    entry.runtime = entry.module->create_runtime(entry.library_path);
  return entry.runtime;
}

}  // namespace taichi::lang
