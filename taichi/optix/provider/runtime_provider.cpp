#include "taichi/optix/forge_optix_runtime_provider.h"

#include <exception>
#include <memory>
#include <new>
#include <string>

#include "taichi/optix/provider/loader_internal.h"

namespace taichi::optix_provider {

namespace {

TiForgeRuntimeProviderResult translate(TiForgeOptixResult result) {
  switch (result) {
    case TI_FORGE_OPTIX_SUCCESS:
      return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
    case TI_FORGE_OPTIX_ERROR_INVALID_ARGUMENT:
      return TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT;
    case TI_FORGE_OPTIX_ERROR_ABI_MISMATCH:
      return TI_FORGE_RUNTIME_PROVIDER_ERROR_ABI_MISMATCH;
    case TI_FORGE_OPTIX_ERROR_CUDA_CONTEXT:
      return TI_FORGE_RUNTIME_PROVIDER_ERROR_RUNTIME_INCOMPATIBLE;
    case TI_FORGE_OPTIX_ERROR_OPTIX_UNAVAILABLE:
      return TI_FORGE_RUNTIME_PROVIDER_ERROR_RUNTIME_UNAVAILABLE;
    case TI_FORGE_OPTIX_ERROR_OPTIX_CALL:
    case TI_FORGE_OPTIX_ERROR_CUDA_CALL:
      return TI_FORGE_RUNTIME_PROVIDER_ERROR_VENDOR_CALL;
    case TI_FORGE_OPTIX_ERROR_OUT_OF_MEMORY:
      return TI_FORGE_RUNTIME_PROVIDER_ERROR_OUT_OF_MEMORY;
    case TI_FORGE_OPTIX_ERROR_LIFETIME:
      return TI_FORGE_RUNTIME_PROVIDER_ERROR_LIFETIME;
    default:
      return TI_FORGE_RUNTIME_PROVIDER_ERROR_INTERNAL;
  }
}

// No exception, including a diagnostic string allocation, may cross the C ABI.
template <typename Fn>
TiForgeRuntimeProviderResult guarded(Fn &&fn) noexcept {
  try {
    clear_error_state();
    return fn();
  } catch (const std::bad_alloc &) {
    last_error.clear();
    return TI_FORGE_RUNTIME_PROVIDER_ERROR_OUT_OF_MEMORY;
  } catch (const std::exception &error) {
    try {
      last_error = error.what();
    } catch (...) {
      last_error.clear();
    }
    return TI_FORGE_RUNTIME_PROVIDER_ERROR_INTERNAL;
  } catch (...) {
    last_error.clear();
    return TI_FORGE_RUNTIME_PROVIDER_ERROR_INTERNAL;
  }
}

TiForgeRuntimeProviderResult invalid(const char *message) {
  return translate(fail(TI_FORGE_OPTIX_ERROR_INVALID_ARGUMENT, message));
}

struct Runtime {
  explicit Runtime(const char *path) : library_path(path ? path : "") {
  }
  ~Runtime() {
    if (retained) {
      release_optix_loader();
    }
  }
  std::string library_path;
  bool retained{false};
};

TiForgeRuntimeInfo runtime_info(const std::string &path) {
  TiForgeRuntimeInfo info{};
  info.struct_size = sizeof(info);
  info.library_path = path.c_str();
  info.build_version = "";
  return info;
}

TiForgeRuntimeProviderResult probe(const char *path,
                                   TiForgeRuntimeInfo *out_info) {
  return guarded([&] {
    if (!out_info || out_info->struct_size < sizeof(*out_info)) {
      return invalid("OptiX runtime info output is too small");
    }
    // Probe strings remain valid until the next probe on this thread. Common
    // NativeProviderModule copies them before returning to its caller.
    static thread_local std::string probe_path;
    probe_path = path ? path : "";
    const auto result = probe_runtime(probe_path.c_str());
    if (result != TI_FORGE_OPTIX_SUCCESS) {
      return translate(result);
    }
    *out_info = runtime_info(probe_path);
    return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
  });
}

TiForgeRuntimeProviderResult create(const char *path,
                                    TiForgeRuntimeProviderRuntime *out_runtime,
                                    TiForgeRuntimeInfo *out_info) {
  if (out_runtime) {
    *out_runtime = nullptr;
  }
  return guarded([&] {
    if (!out_runtime || !out_info ||
        out_info->struct_size < sizeof(*out_info)) {
      return invalid("OptiX runtime creation requires full output storage");
    }
    auto runtime = std::make_unique<Runtime>(path);
    const auto result = retain_optix_loader(runtime->library_path.c_str());
    if (result != TI_FORGE_OPTIX_SUCCESS) {
      return translate(result);
    }
    runtime->retained = true;
    *out_info = runtime_info(runtime->library_path);
    *out_runtime = runtime.release();
    return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
  });
}

TiForgeRuntimeProviderResult destroy(TiForgeRuntimeProviderRuntime runtime) {
  return guarded([&] {
    if (!runtime) {
      return invalid("OptiX runtime is null");
    }
    delete static_cast<Runtime *>(runtime);
    return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
  });
}

TiForgeRuntimeProviderResult execution(TiForgeRuntimeProviderRuntime runtime,
                                       uint32_t version,
                                       size_t api_size,
                                       void *out_api) {
  return guarded([&] {
    if (version != TI_FORGE_RUNTIME_PROVIDER_EXECUTION_ABI_VERSION) {
      return translate(fail(TI_FORGE_OPTIX_ERROR_ABI_MISMATCH,
                            "unsupported OptiX execution ABI"));
    }
    if (!runtime || !out_api || api_size < sizeof(TiForgeOptixExecutionApi)) {
      return invalid("OptiX execution API requires a runtime and full table");
    }
    TiForgeOptixExecutionApi api{};
    api.struct_size = sizeof(api);
    api.execution_abi_version = version;
    const auto result = taichi_forge_optix_provider_query(
        TI_FORGE_OPTIX_PROVIDER_ABI_VERSION, sizeof(api.provider), &api.provider);
    if (result != TI_FORGE_OPTIX_SUCCESS) {
      return translate(result);
    }
    *static_cast<TiForgeOptixExecutionApi *>(out_api) = api;
    return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
  });
}

}  // namespace
}  // namespace taichi::optix_provider

extern "C" TI_FORGE_OPTIX_EXPORT TiForgeRuntimeProviderResult
taichi_forge_optix_runtime_provider_query(uint32_t requested_abi_version,
                                        size_t api_size,
                                        TiForgeRuntimeProviderApi *out_api) {
  using namespace taichi::optix_provider;
  return guarded([&] {
    if (requested_abi_version != TI_FORGE_RUNTIME_PROVIDER_ABI_VERSION) {
      return translate(fail(TI_FORGE_OPTIX_ERROR_ABI_MISMATCH,
                            "unsupported common OptiX provider ABI"));
    }
    if (!out_api || api_size < sizeof(*out_api)) {
      return invalid("common OptiX provider API table is too small");
    }
    TiForgeOptixProviderApi legacy{};
    const auto result = taichi_forge_optix_provider_query(
        TI_FORGE_OPTIX_PROVIDER_ABI_VERSION, sizeof(legacy), &legacy);
    if (result != TI_FORGE_OPTIX_SUCCESS) {
      return translate(result);
    }
    TiForgeRuntimeProviderApi api{};
    api.struct_size = sizeof(api);
    api.provider_abi_version = TI_FORGE_RUNTIME_PROVIDER_ABI_VERSION;
    api.info.struct_size = sizeof(api.info);
    api.info.provider_abi_version = api.provider_abi_version;
    api.info.features = TI_FORGE_RUNTIME_PROVIDER_FEATURE_VERSION_QUERY |
                        TI_FORGE_RUNTIME_PROVIDER_FEATURE_REQUIRED_SYMBOL_AUDIT |
                        TI_FORGE_RUNTIME_PROVIDER_FEATURE_TRANSIENT_PROBE |
                        TI_FORGE_RUNTIME_PROVIDER_FEATURE_EXECUTION_API;
    api.info.required_symbol_count = 1;  // optixQueryFunctionTable
    api.info.provider_id = "optix";
    api.info.provider_name = legacy.info.provider_name;
    api.info.supported_version_family = TI_FORGE_OPTIX_RUNTIME_VERSION_FAMILY;
    api.info.build_identity = legacy.info.build_identity;
    api.probe_runtime = probe;
    api.create_runtime = create;
    api.destroy_runtime = destroy;
    api.query_execution_api = execution;
    api.get_last_error = get_last_error;
    *out_api = api;
    return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
  });
}
