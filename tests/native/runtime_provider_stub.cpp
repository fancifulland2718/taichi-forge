#include "tests/native/runtime_provider_stub.h"

#include <algorithm>
#include <cstring>
#include <string>

namespace {
using provider_stub::Mode;
Mode mode = Mode::good;
provider_stub::Stats stats{};
thread_local std::string probe_path;

struct Runtime {
  std::string path;
};

void fill_info(TiForgeRuntimeInfo *info, const std::string &path) {
  *info = {sizeof(*info), 2, 7, 1, 13000, 0, path.c_str(), "stub-build"};
  if (mode == Mode::short_runtime_info) {
    info->struct_size = 0;
  }
}

TiForgeRuntimeProviderResult probe(const char *path, TiForgeRuntimeInfo *info) {
  ++stats.probes;
  if (mode == Mode::probe_failure) {
    return TI_FORGE_RUNTIME_PROVIDER_ERROR_RUNTIME_UNAVAILABLE;
  }
  probe_path = path ? path : "system-default";
  fill_info(info, probe_path);
  return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
}

TiForgeRuntimeProviderResult create(const char *path,
                                    TiForgeRuntimeProviderRuntime *out,
                                    TiForgeRuntimeInfo *info) {
  ++stats.creates;
  *out = nullptr;
  if (mode == Mode::create_failure) {
    return TI_FORGE_RUNTIME_PROVIDER_ERROR_VENDOR_CALL;
  }
  if (mode == Mode::create_null_handle) {
    return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
  }
  auto *runtime = new Runtime{path ? path : "system-default"};
  *out = runtime;
  ++stats.live;
  fill_info(info, runtime->path);
  return mode == Mode::create_failure_with_handle
             ? TI_FORGE_RUNTIME_PROVIDER_ERROR_VENDOR_CALL
             : TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
}

TiForgeRuntimeProviderResult destroy(TiForgeRuntimeProviderRuntime runtime) {
  ++stats.destroys;
  if (mode == Mode::destroy_failure) {
    return TI_FORGE_RUNTIME_PROVIDER_ERROR_LIFETIME;
  }
  delete static_cast<Runtime *>(runtime);
  --stats.live;
  return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
}

int value(TiForgeRuntimeProviderRuntime runtime) {
  return static_cast<Runtime *>(runtime)->path == "chosen-vendor" ? 42 : 7;
}

TiForgeRuntimeProviderResult execution(TiForgeRuntimeProviderRuntime runtime,
                                       uint32_t version,
                                       size_t size,
                                       void *out) {
  if (!runtime || size != sizeof(provider_stub::ExecutionApi) ||
      version != TI_FORGE_RUNTIME_PROVIDER_EXECUTION_ABI_VERSION) {
    return TI_FORGE_RUNTIME_PROVIDER_ERROR_ABI_MISMATCH;
  }
  if (mode == Mode::execution_failure) {
    return TI_FORGE_RUNTIME_PROVIDER_ERROR_EXECUTION_UNSUPPORTED;
  }
  auto *api = static_cast<provider_stub::ExecutionApi *>(out);
  *api = {sizeof(*api), version, value};
  if (mode == Mode::short_execution) {
    api->struct_size = 0;
  }
  if (mode == Mode::execution_version) {
    ++api->execution_abi_version;
  }
  return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
}

size_t error(char *out, size_t size) {
  constexpr char text[] = "stub failure detail";
  if (out && size) {
    const auto count = std::min(size - 1, sizeof(text) - 1);
    std::memcpy(out, text, count);
    out[count] = '\0';
  }
  return sizeof(text);
}
}  // namespace

extern "C" TI_FORGE_RUNTIME_PROVIDER_EXPORT void stub_set_mode(Mode value) {
  mode = value;
}

extern "C" TI_FORGE_RUNTIME_PROVIDER_EXPORT provider_stub::Stats stub_stats() {
  return stats;
}

extern "C" TI_FORGE_RUNTIME_PROVIDER_EXPORT TiForgeRuntimeProviderResult
stub_query(uint32_t version, size_t size, TiForgeRuntimeProviderApi *out) {
  if (mode == Mode::query_failure) {
    return TI_FORGE_RUNTIME_PROVIDER_ERROR_INTERNAL;
  }
  if (size != sizeof(*out) ||
      version != TI_FORGE_RUNTIME_PROVIDER_ABI_VERSION) {
    return TI_FORGE_RUNTIME_PROVIDER_ERROR_ABI_MISMATCH;
  }
  *out = {};
  out->struct_size = sizeof(*out);
  out->provider_abi_version = version;
  out->info = {sizeof(out->info), version, 15,         1, 0, "stub",
               "Test provider",   "2.x",   "test-only"};
  out->probe_runtime = probe;
  out->create_runtime = create;
  out->destroy_runtime = destroy;
  out->query_execution_api = execution;
  out->get_last_error = error;
  switch (mode) {
    case Mode::short_api:
      out->struct_size = 0;
      break;
    case Mode::api_version:
      ++out->provider_abi_version;
      break;
    case Mode::short_info:
      out->info.struct_size = 0;
      break;
    case Mode::info_version:
      ++out->info.provider_abi_version;
      break;
    case Mode::wrong_id:
      out->info.provider_id = "another-provider";
      break;
    case Mode::wrong_family:
      out->info.supported_version_family = "1.x";
      break;
    case Mode::missing_feature:
      out->info.features &= ~TI_FORGE_RUNTIME_PROVIDER_FEATURE_EXECUTION_API;
      break;
    case Mode::empty_audit:
      out->info.required_symbol_count = 0;
      break;
    case Mode::missing_probe:
      out->probe_runtime = nullptr;
      break;
    case Mode::missing_create:
      out->create_runtime = nullptr;
      break;
    case Mode::missing_destroy:
      out->destroy_runtime = nullptr;
      break;
    case Mode::missing_execution:
      out->query_execution_api = nullptr;
      break;
    case Mode::missing_error:
      out->get_last_error = nullptr;
      break;
    default:
      break;
  }
  return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
}
