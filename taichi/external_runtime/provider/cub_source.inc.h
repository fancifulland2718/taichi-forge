// Bridge to existing addon C ABI 1. No CUDA Toolkit headers/link dependencies.
struct CubSourceInfo {
  uint32_t struct_size, provider_abi_version, cuda_runtime_version, cub_version;
  uint64_t features;
};
struct CubSourceInvocation {
  uint32_t struct_size, operation, flags, reserved;
  uint64_t num_items;
  const void *input0, *input1;
  void *output0, *output1, *workspace;
  size_t workspace_bytes;
  void *stream;
};
using CubQuery = uint32_t (*)(uint32_t, size_t, CubSourceInfo *);
using CubWorkspace = uint32_t (*)(const CubSourceInvocation *, size_t *);
using CubExecute = uint32_t (*)(const CubSourceInvocation *);
using CubError = size_t (*)(char *, size_t);

TiForgeRuntimeProviderResult cub_status(uint32_t status, CubError error) {
  if (!status)
    return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
  const auto count = error(nullptr, 0);
  std::vector<char> message(count ? count : 1, '\0');
  error(message.data(), message.size());
  message.back() = '\0';
  return fail(
      TI_FORGE_RUNTIME_PROVIDER_ERROR_VENDOR_CALL,
      "CUB addon status " + std::to_string(status) + ": " + message.data());
}

bool query_cub_source_version(Runtime &runtime) {
  auto query = reinterpret_cast<CubQuery>(
      load_symbol(runtime.library, "ti_forge_cub_source_provider_query"));
  CubSourceInfo info{};
  if (query(1, sizeof(info), &info) || info.struct_size != sizeof(info) ||
      info.provider_abi_version != 1 || (info.features & 0xf) != 0xf ||
      !info.cub_version || !info.cuda_runtime_version) {
    last_error = "CUB source addon ABI identity is incomplete";
    return false;
  }
  runtime.version_major = info.cub_version / 100000;
  runtime.version_minor = (info.cub_version / 100) % 1000;
  runtime.version_patch = info.cub_version % 100;
  runtime.cuda_runtime_version = info.cuda_runtime_version;
  runtime.source_features = info.features;
  return identify_loaded_source_addon(runtime);
}

struct CubSourcePlan {
  Runtime *runtime;
  CubSourceInvocation invocation{};
  CubExecute execute;
  CubError error;
};

TiForgeRuntimeProviderResult cub_create_plan(
    TiForgeRuntimeProviderRuntime value,
    const TiForgeCubSourcePlanDesc *desc,
    TiForgeCubSourcePlan *out_plan,
    TiForgeCubSourcePlanInfo *out_info) {
  if (out_plan)
    *out_plan = nullptr;
  if (!value || !desc || desc->struct_size < sizeof(*desc) || !out_plan ||
      !out_info || out_info->struct_size < sizeof(*out_info) ||
      desc->num_items > uint64_t(std::numeric_limits<int>::max()) ||
      desc->operation < 1 || desc->operation > 6)
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                "CUB plan descriptor is invalid or exceeds addon int ABI");
  auto &runtime = *static_cast<Runtime *>(value);
  if (!(runtime.source_features & (uint64_t(1) << (desc->operation - 1))))
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_EXECUTION_UNSUPPORTED,
                "CUB addon does not provide the selected operation");
  try {
    auto plan = std::make_unique<CubSourcePlan>();
    plan->runtime = &runtime;
    plan->execute = reinterpret_cast<CubExecute>(
        load_symbol(runtime.library, "ti_forge_cub_source_provider_execute"));
    plan->error = reinterpret_cast<CubError>(load_symbol(
        runtime.library, "ti_forge_cub_source_provider_get_last_error"));
    auto workspace = reinterpret_cast<CubWorkspace>(load_symbol(
        runtime.library, "ti_forge_cub_source_provider_workspace_bytes"));
    plan->invocation.struct_size = sizeof(CubSourceInvocation);
    plan->invocation.operation = desc->operation;
    plan->invocation.num_items = desc->num_items;
    size_t bytes = 0;
    auto result = cub_status(workspace(&plan->invocation, &bytes), plan->error);
    if (result != TI_FORGE_RUNTIME_PROVIDER_SUCCESS)
      return result;
    plan->invocation.workspace_bytes = bytes;
    *out_info = {sizeof(*out_info), 0, bytes};
    runtime.live_execution_resources.fetch_add(1);
    *out_plan = plan.release();
    return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
  } catch (const std::bad_alloc &) {
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_OUT_OF_MEMORY,
                "CUB plan allocation failed");
  } catch (const std::exception &error) {
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INTERNAL, error.what());
  }
}

TiForgeRuntimeProviderResult cub_execute(TiForgeCubSourcePlan value,
                                         const TiForgeCubSourceExecDesc *desc) {
  if (!value || !desc || desc->struct_size < sizeof(*desc) || desc->reserved)
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                "CUB execution descriptor is invalid");
  auto &plan = *static_cast<CubSourcePlan *>(value);
  if (desc->workspace_bytes < plan.invocation.workspace_bytes ||
      desc->workspace_bytes > std::numeric_limits<size_t>::max())
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                "CUB execution workspace is too small or overflows");
  for (auto address : {desc->input0, desc->input1, desc->output0, desc->output1,
                       desc->workspace, desc->cuda_stream})
    if (address > std::numeric_limits<uintptr_t>::max())
      return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                  "CUB execution pointer exceeds platform ABI");
  auto pointer = [](uint64_t address) {
    return reinterpret_cast<void *>(static_cast<uintptr_t>(address));
  };
  auto invocation = plan.invocation;
  invocation.input0 = pointer(desc->input0);
  invocation.input1 = pointer(desc->input1);
  invocation.output0 = pointer(desc->output0);
  invocation.output1 = pointer(desc->output1);
  invocation.workspace = pointer(desc->workspace);
  invocation.workspace_bytes = static_cast<size_t>(desc->workspace_bytes);
  invocation.stream = pointer(desc->cuda_stream);
  try {
    return cub_status(plan.execute(&invocation), plan.error);
  } catch (const std::exception &error) {
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INTERNAL, error.what());
  }
}

TiForgeRuntimeProviderResult cub_destroy_plan(TiForgeCubSourcePlan value) {
  if (!value)
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                "CUB plan is null");
  auto *plan = static_cast<CubSourcePlan *>(value);
  plan->runtime->live_execution_resources.fetch_sub(1);
  delete plan;
  return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
}

TiForgeRuntimeProviderResult initialize_execution_runtime(Runtime &) {
  return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
}
TiForgeRuntimeProviderResult shutdown_execution_runtime(Runtime &) {
  return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
}
TiForgeRuntimeProviderResult query_execution_api(
    TiForgeRuntimeProviderRuntime runtime,
    uint32_t version,
    size_t api_size,
    void *out_api) {
  if (!runtime || !out_api || api_size < sizeof(TiForgeCubSourceExecutionApi))
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                "CUB execution API output is truncated");
  if (version != TI_FORGE_RUNTIME_PROVIDER_EXECUTION_ABI_VERSION)
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_ABI_MISMATCH,
                "Unsupported CUB execution ABI");
  auto *api = static_cast<TiForgeCubSourceExecutionApi *>(out_api);
  *api = {sizeof(*api), TI_FORGE_RUNTIME_PROVIDER_EXECUTION_ABI_VERSION,
          cub_create_plan, cub_execute, cub_destroy_plan};
  return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
}
