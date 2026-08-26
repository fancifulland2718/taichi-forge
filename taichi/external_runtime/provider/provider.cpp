#include "taichi/external_runtime/forge_runtime_provider.h"

#if defined(_WIN32) && !defined(NOMINMAX)
#define NOMINMAX
#endif

#include <algorithm>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#ifndef TI_FORGE_RUNTIME_PROVIDER_KIND
#error "TI_FORGE_RUNTIME_PROVIDER_KIND must select one vendor adapter"
#endif

namespace {

thread_local std::string last_error;
thread_local std::string probed_library_path;
thread_local std::string probed_build_version;

constexpr uint64_t kFeatures =
    TI_FORGE_RUNTIME_PROVIDER_FEATURE_VERSION_QUERY |
    TI_FORGE_RUNTIME_PROVIDER_FEATURE_REQUIRED_SYMBOL_AUDIT |
    TI_FORGE_RUNTIME_PROVIDER_FEATURE_TRANSIENT_PROBE;

#if TI_FORGE_RUNTIME_PROVIDER_KIND == 1
constexpr char kProviderId[] = "cusparselt";
constexpr char kProviderName[] = "NVIDIA cuSPARSELt";
constexpr char kSupportedVersionFamily[] = "0.4.x-0.9.x";
constexpr char kBuildIdentity[] =
    "forge-runtime-provider-abi1-cusparselt-api-0.4-0.9";
constexpr const char *kWindowsCandidates[] = {"cusparseLt64_0.dll",
                                              "cusparseLt.dll"};
constexpr const char *kLinuxCandidates[] = {"libcusparseLt.so.0",
                                            "libcusparseLt.so"};
constexpr const char *kRequiredSymbols[] = {
    "cusparseLtGetProperty",
    "cusparseLtInit",
    "cusparseLtDestroy",
    "cusparseLtDenseDescriptorInit",
    "cusparseLtStructuredDescriptorInit",
    "cusparseLtMatDescriptorDestroy",
    "cusparseLtMatmulDescriptorInit",
    "cusparseLtMatmulAlgSelectionInit",
    "cusparseLtMatmulPlanInit",
    "cusparseLtMatmulPlanDestroy",
    "cusparseLtMatmulGetWorkspace",
    "cusparseLtMatmul",
    "cusparseLtMatmulSearch",
};
#define TI_FORGE_RUNTIME_PROVIDER_QUERY_FUNCTION \
  taichi_forge_cusparselt_provider_query
#elif TI_FORGE_RUNTIME_PROVIDER_KIND == 2
constexpr char kProviderId[] = "cutensor";
constexpr char kProviderName[] = "NVIDIA cuTENSOR";
constexpr char kSupportedVersionFamily[] = "2.0.x-2.7.x";
constexpr char kBuildIdentity[] =
    "forge-runtime-provider-abi1-cutensor-api-2.0-2.7";
constexpr const char *kWindowsCandidates[] = {"cutensor64_2.dll",
                                              "cutensor.dll"};
constexpr const char *kLinuxCandidates[] = {"libcutensor.so.2",
                                            "libcutensor.so"};
constexpr const char *kRequiredSymbols[] = {
    "cutensorGetVersion",
    "cutensorGetCudartVersion",
    "cutensorCreate",
    "cutensorDestroy",
    "cutensorCreateTensorDescriptor",
    "cutensorDestroyTensorDescriptor",
    "cutensorCreateContraction",
    "cutensorCreateReduction",
    "cutensorCreatePlanPreference",
    "cutensorDestroyPlanPreference",
    "cutensorEstimateWorkspaceSize",
    "cutensorCreatePlan",
    "cutensorDestroyPlan",
    "cutensorContract",
    "cutensorReduce",
    "cutensorPermute",
};
#define TI_FORGE_RUNTIME_PROVIDER_QUERY_FUNCTION \
  taichi_forge_cutensor_provider_query
#elif TI_FORGE_RUNTIME_PROVIDER_KIND == 3
constexpr char kProviderId[] = "amgx";
constexpr char kProviderName[] = "NVIDIA AmgX";
constexpr char kSupportedVersionFamily[] = "stable C API";
constexpr char kBuildIdentity[] =
    "forge-runtime-provider-abi1-amgx-stable-c-api";
constexpr const char *kWindowsCandidates[] = {"amgxsh.dll"};
constexpr const char *kLinuxCandidates[] = {"libamgxsh.so"};
constexpr const char *kRequiredSymbols[] = {
    "AMGX_get_api_version",
    "AMGX_get_build_info_strings",
    "AMGX_get_error_string",
    "AMGX_initialize",
    "AMGX_finalize",
    "AMGX_config_create",
    "AMGX_config_destroy",
    "AMGX_resources_create_simple",
    "AMGX_resources_destroy",
    "AMGX_matrix_create",
    "AMGX_matrix_destroy",
    "AMGX_matrix_upload_all",
    "AMGX_matrix_replace_coefficients",
    "AMGX_vector_create",
    "AMGX_vector_destroy",
    "AMGX_vector_upload",
    "AMGX_vector_download",
    "AMGX_solver_create",
    "AMGX_solver_destroy",
    "AMGX_solver_setup",
    "AMGX_solver_solve",
    "AMGX_solver_get_status",
};
#define TI_FORGE_RUNTIME_PROVIDER_QUERY_FUNCTION \
  taichi_forge_amgx_provider_query
#else
#error "Unsupported TI_FORGE_RUNTIME_PROVIDER_KIND"
#endif

constexpr uint32_t kRequiredSymbolCount = static_cast<uint32_t>(
    sizeof(kRequiredSymbols) / sizeof(kRequiredSymbols[0]));

TiForgeRuntimeProviderResult fail(TiForgeRuntimeProviderResult result,
                                  std::string message) {
  last_error = std::move(message);
  return result;
}

#if defined(_WIN32)
using LibraryHandle = HMODULE;

LibraryHandle open_library(const std::string &path) {
  const int wide_size = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                                            path.c_str(), -1, nullptr, 0);
  if (wide_size <= 0) {
    return nullptr;
  }
  std::wstring wide_path(static_cast<std::size_t>(wide_size), L'\0');
  if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path.c_str(), -1,
                          wide_path.data(), wide_size) <= 0) {
    return nullptr;
  }
  return LoadLibraryW(wide_path.c_str());
}

void close_library(LibraryHandle handle) {
  if (handle != nullptr) {
    FreeLibrary(handle);
  }
}

void *load_symbol(LibraryHandle handle, const char *name) {
  return reinterpret_cast<void *>(GetProcAddress(handle, name));
}
#else
using LibraryHandle = void *;

LibraryHandle open_library(const std::string &path) {
  return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
}

void close_library(LibraryHandle handle) {
  if (handle != nullptr) {
    dlclose(handle);
  }
}

void *load_symbol(LibraryHandle handle, const char *name) {
  return dlsym(handle, name);
}
#endif

struct Runtime {
  LibraryHandle library{nullptr};
  std::string library_path;
  std::string build_version;
  uint32_t version_major{0};
  uint32_t version_minor{0};
  uint32_t version_patch{0};
  uint32_t cuda_runtime_version{0};

  ~Runtime() {
    close_library(library);
  }
};

bool audit_required_symbols(Runtime &runtime) {
  for (const char *symbol : kRequiredSymbols) {
    if (load_symbol(runtime.library, symbol) == nullptr) {
      last_error = std::string(kProviderName) +
                   " runtime is missing required symbol " + symbol;
      return false;
    }
  }
  return true;
}

bool query_version(Runtime &runtime) {
#if TI_FORGE_RUNTIME_PROVIDER_KIND == 1
  using GetProperty = int (*)(int, int *);
  auto get_property = reinterpret_cast<GetProperty>(
      load_symbol(runtime.library, "cusparseLtGetProperty"));
  int major = -1;
  int minor = -1;
  int patch = -1;
  if (get_property == nullptr || get_property(0, &major) != 0 ||
      get_property(1, &minor) != 0 || get_property(2, &patch) != 0) {
    last_error = "cuSPARSELt runtime version query failed";
    return false;
  }
  if (major != 0 || minor < 4 || minor > 9) {
    last_error = "cuSPARSELt adapter supports runtime versions 0.4.x-0.9.x";
    return false;
  }
  runtime.version_major = static_cast<uint32_t>(major);
  runtime.version_minor = static_cast<uint32_t>(minor);
  runtime.version_patch = static_cast<uint32_t>(patch);
#elif TI_FORGE_RUNTIME_PROVIDER_KIND == 2
  using GetVersion = size_t (*)();
  auto get_version = reinterpret_cast<GetVersion>(
      load_symbol(runtime.library, "cutensorGetVersion"));
  auto get_cudart_version = reinterpret_cast<GetVersion>(
      load_symbol(runtime.library, "cutensorGetCudartVersion"));
  if (get_version == nullptr || get_cudart_version == nullptr) {
    last_error = "cuTENSOR runtime version query symbols are missing";
    return false;
  }
  const size_t version = get_version();
  const uint32_t major = static_cast<uint32_t>(version / 10000);
  const uint32_t minor = static_cast<uint32_t>((version / 100) % 100);
  const uint32_t patch = static_cast<uint32_t>(version % 100);
  if (major != 2 || minor > 7) {
    last_error = "cuTENSOR adapter supports runtime versions 2.0.x-2.7.x";
    return false;
  }
  runtime.version_major = major;
  runtime.version_minor = minor;
  runtime.version_patch = patch;
  runtime.cuda_runtime_version = static_cast<uint32_t>(get_cudart_version());
#else
  using GetApiVersion = int (*)(int *, int *);
  auto get_api_version = reinterpret_cast<GetApiVersion>(
      load_symbol(runtime.library, "AMGX_get_api_version"));
  int major = 0;
  int minor = 0;
  if (get_api_version == nullptr || get_api_version(&major, &minor) != 0 ||
      major <= 0 || minor < 0) {
    last_error = "AmgX stable C API version query failed";
    return false;
  }
  runtime.version_major = static_cast<uint32_t>(major);
  runtime.version_minor = static_cast<uint32_t>(minor);
  runtime.version_patch = 0;
  runtime.build_version =
      "api-" + std::to_string(major) + "." + std::to_string(minor);
#endif
  return true;
}

void fill_runtime_info(const Runtime &runtime, TiForgeRuntimeInfo *info) {
  if (info == nullptr) {
    return;
  }
  info->struct_size = sizeof(*info);
  info->version_major = runtime.version_major;
  info->version_minor = runtime.version_minor;
  info->version_patch = runtime.version_patch;
  info->cuda_runtime_version = runtime.cuda_runtime_version;
  info->reserved = 0;
  info->library_path = runtime.library_path.c_str();
  info->build_version =
      runtime.build_version.empty() ? nullptr : runtime.build_version.c_str();
}

TiForgeRuntimeProviderResult make_runtime(const char *library_path,
                                          Runtime **out_runtime) {
  if (out_runtime == nullptr) {
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                std::string(kProviderName) + " runtime output is null");
  }
  *out_runtime = nullptr;
  last_error.clear();
  std::vector<std::string> candidates;
  if (library_path != nullptr && library_path[0] != '\0') {
    candidates.emplace_back(library_path);
  } else {
#if defined(_WIN32)
    for (const char *candidate : kWindowsCandidates) {
      candidates.emplace_back(candidate);
    }
#else
    for (const char *candidate : kLinuxCandidates) {
      candidates.emplace_back(candidate);
    }
#endif
  }
  std::string attempted;
  for (const auto &candidate : candidates) {
    auto runtime = std::unique_ptr<Runtime>(new (std::nothrow) Runtime());
    if (!runtime) {
      return fail(
          TI_FORGE_RUNTIME_PROVIDER_ERROR_OUT_OF_MEMORY,
          std::string(kProviderName) + " adapter runtime allocation failed");
    }
    runtime->library_path = candidate;
    runtime->library = open_library(candidate);
    if (runtime->library == nullptr) {
      if (!attempted.empty()) {
        attempted += ", ";
      }
      attempted += candidate;
      continue;
    }
    if (!audit_required_symbols(*runtime) || !query_version(*runtime)) {
      return TI_FORGE_RUNTIME_PROVIDER_ERROR_RUNTIME_INCOMPATIBLE;
    }
    *out_runtime = runtime.release();
    return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
  }
  return fail(
      TI_FORGE_RUNTIME_PROVIDER_ERROR_RUNTIME_UNAVAILABLE,
      std::string(kProviderName) +
          " vendor runtime could not be loaded; attempted: " + attempted);
}

TiForgeRuntimeProviderResult probe_runtime(const char *library_path,
                                           TiForgeRuntimeInfo *out_info) {
  if (out_info != nullptr && out_info->struct_size < sizeof(*out_info)) {
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                "runtime info output is truncated");
  }
  Runtime *runtime = nullptr;
  const auto result = make_runtime(library_path, &runtime);
  if (result != TI_FORGE_RUNTIME_PROVIDER_SUCCESS) {
    return result;
  }
  probed_library_path = runtime->library_path;
  probed_build_version = runtime->build_version;
  fill_runtime_info(*runtime, out_info);
  if (out_info != nullptr) {
    out_info->library_path = probed_library_path.c_str();
    out_info->build_version =
        probed_build_version.empty() ? nullptr : probed_build_version.c_str();
  }
  delete runtime;
  return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
}

TiForgeRuntimeProviderResult create_runtime(
    const char *library_path,
    TiForgeRuntimeProviderRuntime *out_runtime,
    TiForgeRuntimeInfo *out_info) {
  if (out_runtime == nullptr ||
      (out_info != nullptr && out_info->struct_size < sizeof(*out_info))) {
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                "runtime output is null or truncated");
  }
  *out_runtime = nullptr;
  Runtime *runtime = nullptr;
  const auto result = make_runtime(library_path, &runtime);
  if (result != TI_FORGE_RUNTIME_PROVIDER_SUCCESS) {
    return result;
  }
  fill_runtime_info(*runtime, out_info);
  *out_runtime = runtime;
  return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
}

TiForgeRuntimeProviderResult destroy_runtime(
    TiForgeRuntimeProviderRuntime runtime) {
  if (runtime == nullptr) {
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                "adapter runtime is null");
  }
  delete static_cast<Runtime *>(runtime);
  return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
}

size_t get_last_error(char *destination, size_t destination_size) {
  const std::size_t required = last_error.size() + 1;
  if (destination != nullptr && destination_size > 0) {
    const std::size_t count = std::min(last_error.size(), destination_size - 1);
    std::memcpy(destination, last_error.data(), count);
    destination[count] = '\0';
  }
  return required;
}

}  // namespace

extern "C" TI_FORGE_RUNTIME_PROVIDER_EXPORT TiForgeRuntimeProviderResult
TI_FORGE_RUNTIME_PROVIDER_QUERY_FUNCTION(uint32_t requested_abi_version,
                                         size_t api_size,
                                         TiForgeRuntimeProviderApi *out_api) {
  if (out_api == nullptr || api_size < sizeof(TiForgeRuntimeProviderApi)) {
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                "provider API output is null or truncated");
  }
  if (requested_abi_version != TI_FORGE_RUNTIME_PROVIDER_ABI_VERSION) {
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_ABI_MISMATCH,
                "unsupported Forge runtime-provider ABI");
  }
  std::memset(out_api, 0, sizeof(*out_api));
  out_api->struct_size = sizeof(*out_api);
  out_api->provider_abi_version = TI_FORGE_RUNTIME_PROVIDER_ABI_VERSION;
  out_api->info = {sizeof(TiForgeRuntimeProviderInfo),
                   TI_FORGE_RUNTIME_PROVIDER_ABI_VERSION,
                   kFeatures,
                   kRequiredSymbolCount,
                   0,
                   kProviderId,
                   kProviderName,
                   kSupportedVersionFamily,
                   kBuildIdentity};
  out_api->probe_runtime = probe_runtime;
  out_api->create_runtime = create_runtime;
  out_api->destroy_runtime = destroy_runtime;
  out_api->get_last_error = get_last_error;
  last_error.clear();
  return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
}
