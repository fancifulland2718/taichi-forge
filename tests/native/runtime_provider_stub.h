#pragma once

#include "taichi/external_runtime/forge_runtime_provider.h"

namespace provider_stub {
enum class Mode {
  good,
  query_failure,
  short_api,
  api_version,
  short_info,
  info_version,
  wrong_id,
  wrong_family,
  missing_feature,
  empty_audit,
  embedded_runtime,
  missing_probe,
  missing_create,
  missing_destroy,
  missing_execution,
  missing_error,
  probe_failure,
  short_runtime_info,
  create_failure,
  create_failure_with_handle,
  create_null_handle,
  execution_failure,
  short_execution,
  execution_version,
  destroy_failure,
};

struct Stats {
  int probes;
  int creates;
  int destroys;
  int live;
};

struct ExecutionApi {
  uint32_t struct_size;
  uint32_t execution_abi_version;
  int (*value)(TiForgeRuntimeProviderRuntime);
};
}  // namespace provider_stub
