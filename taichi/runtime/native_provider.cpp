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
      info.required_symbol_count == 0 || !api_.probe_runtime ||
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

}  // namespace taichi::lang
