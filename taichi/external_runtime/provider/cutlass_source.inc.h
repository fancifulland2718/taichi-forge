// Bridge to the existing CUTLASS matmul C ABI 1; no Toolkit/header dependency.
struct CutlassInvocation {
  uint32_t struct_size, strategy, transpose_a, transpose_b;
  int32_t m, n, k, activation;
  float alpha, beta;
  const float *a, *b;
  float *output;
  void *workspace;
  size_t workspace_bytes;
  void *stream;
};
using CutlassIdentity = uint32_t (*)();
using CutlassQuery = uint32_t (*)(const CutlassInvocation *, size_t *);
using CutlassExecute = uint32_t (*)(const CutlassInvocation *);
using CutlassError = size_t (*)(char *, size_t);

TiForgeRuntimeProviderResult cutlass_status(uint32_t status,
                                            CutlassError error) {
  if (!status)
    return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
  const auto count = error(nullptr, 0);
  std::vector<char> message(count ? count : 1, '\0');
  error(message.data(), message.size());
  message.back() = '\0';
  return fail(
      TI_FORGE_RUNTIME_PROVIDER_ERROR_VENDOR_CALL,
      "CUTLASS addon status " + std::to_string(status) + ": " + message.data());
}

bool query_cutlass_source_version(Runtime &runtime) {
  auto abi = reinterpret_cast<CutlassIdentity>(
      load_symbol(runtime.library, "ti_forge_cutlass_abi"));
  auto version = reinterpret_cast<CutlassIdentity>(
      load_symbol(runtime.library, "ti_forge_cutlass_version"));
  auto value = version();
  if (abi() != 1 || !value) {
    last_error = "CUTLASS source addon ABI identity is incomplete";
    return false;
  }
  runtime.version_major = value / 10000;
  runtime.version_minor = (value / 100) % 100;
  runtime.version_patch = value % 100;
  return identify_loaded_source_addon(runtime);
}

struct CutlassSourcePlan {
  Runtime *runtime;
  CutlassInvocation invocation{};
  CutlassExecute execute;
  CutlassError error;
};

TiForgeRuntimeProviderResult cutlass_create_plan(
    TiForgeRuntimeProviderRuntime value,
    const TiForgeCutlassSourcePlanDesc *desc,
    TiForgeCutlassSourcePlan *out_plan,
    TiForgeCutlassSourcePlanInfo *out_info) {
  if (out_plan)
    *out_plan = nullptr;
  if (!value || !desc || desc->struct_size < sizeof(*desc) || !out_plan ||
      !out_info || out_info->struct_size < sizeof(*out_info) ||
      desc->strategy > 2 || desc->transpose_a > 1 || desc->transpose_b > 1 ||
      desc->m <= 0 || desc->n <= 0 || desc->k <= 0 || desc->activation < 0 ||
      desc->activation > 1 || !std::isfinite(desc->alpha) ||
      !std::isfinite(desc->beta))
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                "CUTLASS plan descriptor is invalid");
  // Split-K query multiplies the entire output by its fixed partition count.
  // Check representability before handing dimensions to the old size_t ABI.
  const uint64_t partitions = desc->strategy == 2 ? 128 : 16;
  const uint64_t elements = uint64_t(desc->m) * uint64_t(desc->n);
  if (elements > std::numeric_limits<size_t>::max() / sizeof(float) ||
      (desc->strategy && elements > std::numeric_limits<size_t>::max() /
                                        sizeof(float) / partitions))
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                "CUTLASS workspace exceeds platform ABI");
  auto &runtime = *static_cast<Runtime *>(value);
  try {
    auto plan = std::make_unique<CutlassSourcePlan>();
    plan->runtime = &runtime;
    plan->execute = reinterpret_cast<CutlassExecute>(
        load_symbol(runtime.library, "ti_forge_cutlass_capture"));
    plan->error = reinterpret_cast<CutlassError>(
        load_symbol(runtime.library, "ti_forge_cutlass_error"));
    auto query = reinterpret_cast<CutlassQuery>(
        load_symbol(runtime.library, "ti_forge_cutlass_query"));
    plan->invocation = {sizeof(CutlassInvocation),
                        desc->strategy,
                        desc->transpose_a,
                        desc->transpose_b,
                        desc->m,
                        desc->n,
                        desc->k,
                        desc->activation,
                        desc->alpha,
                        desc->beta};
    size_t bytes = 0;
    auto result = cutlass_status(query(&plan->invocation, &bytes), plan->error);
    if (result != TI_FORGE_RUNTIME_PROVIDER_SUCCESS)
      return result;
    plan->invocation.workspace_bytes = bytes;
    *out_info = {sizeof(*out_info), 0, bytes};
    runtime.live_execution_resources.fetch_add(1);
    *out_plan = plan.release();
    return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
  } catch (const std::bad_alloc &) {
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_OUT_OF_MEMORY,
                "CUTLASS plan allocation failed");
  } catch (const std::exception &error) {
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INTERNAL, error.what());
  }
}

TiForgeRuntimeProviderResult cutlass_execute(
    TiForgeCutlassSourcePlan value,
    const TiForgeCutlassSourceExecDesc *desc) {
  if (!value || !desc || desc->struct_size < sizeof(*desc) || desc->reserved)
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                "CUTLASS execution descriptor is invalid");
  auto &plan = *static_cast<CutlassSourcePlan *>(value);
  if (desc->workspace_bytes != plan.invocation.workspace_bytes)
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                "CUTLASS execution workspace differs from preparation");
  for (auto address :
       {desc->a, desc->b, desc->output, desc->workspace, desc->cuda_stream})
    if (address > std::numeric_limits<uintptr_t>::max())
      return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                  "CUTLASS execution pointer exceeds platform ABI");
  auto pointer = [](uint64_t address) {
    return reinterpret_cast<void *>(static_cast<uintptr_t>(address));
  };
  auto invocation = plan.invocation;
  invocation.a = static_cast<const float *>(pointer(desc->a));
  invocation.b = static_cast<const float *>(pointer(desc->b));
  invocation.output = static_cast<float *>(pointer(desc->output));
  invocation.workspace = pointer(desc->workspace);
  invocation.stream = pointer(desc->cuda_stream);
  try {
    return cutlass_status(plan.execute(&invocation), plan.error);
  } catch (const std::exception &error) {
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INTERNAL, error.what());
  }
}

TiForgeRuntimeProviderResult cutlass_destroy_plan(
    TiForgeCutlassSourcePlan value) {
  if (!value)
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                "CUTLASS plan is null");
  auto *plan = static_cast<CutlassSourcePlan *>(value);
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
  if (!runtime || !out_api ||
      api_size < sizeof(TiForgeCutlassSourceExecutionApi))
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                "CUTLASS execution API output is truncated");
  if (version != TI_FORGE_RUNTIME_PROVIDER_EXECUTION_ABI_VERSION)
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_ABI_MISMATCH,
                "Unsupported CUTLASS execution ABI");
  auto *api = static_cast<TiForgeCutlassSourceExecutionApi *>(out_api);
  *api = {sizeof(*api), TI_FORGE_RUNTIME_PROVIDER_EXECUTION_ABI_VERSION,
          cutlass_create_plan, cutlass_execute, cutlass_destroy_plan};
  return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
}
