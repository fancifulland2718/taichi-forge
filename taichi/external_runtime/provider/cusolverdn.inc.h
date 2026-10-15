// Included in the common adapter's anonymous namespace for provider kind 4.
// No CUDA SDK headers or link-time dependency on the vendor runtime are needed.
using SolverCreateFn = int (*)(void **);
using SolverDestroyFn = int (*)(void *);
using SolverSetStreamFn = int (*)(void *, void *);
using SolverBufferSizeFn = int (*)(void *,
                                   void *,
                                   int,
                                   int64_t,
                                   int,
                                   const void *,
                                   int64_t,
                                   int,
                                   size_t *,
                                   size_t *);
using SolverFactorFn = int (*)(void *,
                               void *,
                               int,
                               int64_t,
                               int,
                               void *,
                               int64_t,
                               int,
                               void *,
                               size_t,
                               void *,
                               size_t,
                               int *);
using SolverSolveFn = int (*)(void *,
                              void *,
                              int,
                              int64_t,
                              int64_t,
                              int,
                              const void *,
                              int64_t,
                              int,
                              void *,
                              int64_t,
                              int *);

struct SolverPlan {
  Runtime *runtime;
  TiForgeCusolverDnPlanDesc desc;
  void *handle{nullptr};
  void *params{nullptr};
  SolverDestroyFn destroy_handle{nullptr}, destroy_params{nullptr};
  SolverSetStreamFn set_stream{nullptr};
  SolverFactorFn factor{nullptr};
  SolverSolveFn solve{nullptr};
  size_t device_bytes{0};
  std::vector<unsigned char> host_workspace;
};

void *solver_pointer(uint64_t value) {
  return reinterpret_cast<void *>(static_cast<uintptr_t>(value));
}

TiForgeRuntimeProviderResult solver_result(int result, const char *operation) {
  return result ? fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_VENDOR_CALL,
                       std::string(operation) + " failed with status " +
                           std::to_string(result))
                : TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
}

TiForgeRuntimeProviderResult solver_release(SolverPlan &plan) {
  if (plan.params) {
    const auto result = solver_result(plan.destroy_params(plan.params),
                                      "cusolverDnDestroyParams");
    if (result)
      return result;
    plan.params = nullptr;
  }
  if (plan.handle) {
    const auto result =
        solver_result(plan.destroy_handle(plan.handle), "cusolverDnDestroy");
    if (result)
      return result;
    plan.handle = nullptr;
  }
  return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
}

TiForgeRuntimeProviderResult solver_create_plan(
    TiForgeRuntimeProviderRuntime runtime_value,
    const TiForgeCusolverDnPlanDesc *desc,
    TiForgeCusolverDnPlan *out_plan,
    TiForgeCusolverDnPlanInfo *out_info) {
  if (out_plan)
    *out_plan = nullptr;
  if (!runtime_value || !desc || !out_plan || !out_info ||
      desc->struct_size < sizeof(*desc) ||
      out_info->struct_size < sizeof(*out_info) || desc->scalar > 1 ||
      desc->rows <= 0 || desc->rhs_count <= 0 || !desc->factor || !desc->status)
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                "cuSOLVERDn requires positive dimensions, f32/f64 and storage");
  const size_t scalar_bytes = desc->scalar ? 8 : 4;
  const auto limit = std::numeric_limits<size_t>::max() / scalar_bytes;
  if (uint64_t(desc->rows) > limit / uint64_t(desc->rows) ||
      uint64_t(desc->rhs_count) > limit / uint64_t(desc->rows) ||
      desc->factor % scalar_bytes || desc->status % 4)
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                "cuSOLVERDn storage size overflows or is misaligned");
  auto *runtime = static_cast<Runtime *>(runtime_value);
  auto plan = std::unique_ptr<SolverPlan>(new (std::nothrow) SolverPlan());
  if (!plan)
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_OUT_OF_MEMORY,
                "cuSOLVERDn plan allocation failed");
  plan->runtime = runtime;
  plan->desc = *desc;
  auto symbol = [&](const char *name) {
    return load_symbol(runtime->library, name);
  };
  const auto create =
      reinterpret_cast<SolverCreateFn>(symbol("cusolverDnCreate"));
  const auto create_params =
      reinterpret_cast<SolverCreateFn>(symbol("cusolverDnCreateParams"));
  const auto buffer_size = reinterpret_cast<SolverBufferSizeFn>(
      symbol("cusolverDnXpotrf_bufferSize"));
  plan->destroy_handle =
      reinterpret_cast<SolverDestroyFn>(symbol("cusolverDnDestroy"));
  plan->destroy_params =
      reinterpret_cast<SolverDestroyFn>(symbol("cusolverDnDestroyParams"));
  plan->set_stream =
      reinterpret_cast<SolverSetStreamFn>(symbol("cusolverDnSetStream"));
  plan->factor = reinterpret_cast<SolverFactorFn>(symbol("cusolverDnXpotrf"));
  plan->solve = reinterpret_cast<SolverSolveFn>(symbol("cusolverDnXpotrs"));
  auto result = solver_result(create(&plan->handle), "cusolverDnCreate");
  if (!result)
    result =
        solver_result(create_params(&plan->params), "cusolverDnCreateParams");
  if (!result)
    result = solver_result(plan->set_stream(plan->handle, nullptr),
                           "cusolverDnSetStream");
  size_t host_bytes = 0;
  if (!result)
    result = solver_result(
        buffer_size(plan->handle, plan->params, 1, desc->rows, desc->scalar,
                    solver_pointer(desc->factor), desc->rows, desc->scalar,
                    &plan->device_bytes, &host_bytes),
        "cusolverDnXpotrf_bufferSize");
  if (!result) {
    try {
      plan->host_workspace.resize(host_bytes);
    } catch (const std::exception &) {
      result = fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_OUT_OF_MEMORY,
                    "cuSOLVERDn host workspace allocation failed");
    }
  }
  if (result) {
    // A failed foreign cleanup must not unload code or free host workspace
    // still referenced by that handle. Keep the owner visible to runtime close.
    if (solver_release(*plan)) {
      runtime->live_execution_resources.fetch_add(1);
      plan.release();
    }
    return result;
  }
  *out_info = {sizeof(*out_info), 0, plan->device_bytes, host_bytes};
  runtime->live_execution_resources.fetch_add(1);
  *out_plan = plan.release();
  return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
}

TiForgeRuntimeProviderResult solver_execute(
    TiForgeCusolverDnPlan plan_value,
    const TiForgeCusolverDnExecDesc *desc) {
  if (!plan_value || !desc || desc->struct_size < sizeof(*desc) ||
      desc->phase > 1)
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                "Invalid cuSOLVERDn execution descriptor");
  auto &plan = *static_cast<SolverPlan *>(plan_value);
  if (!plan.handle || !plan.params ||
      (!desc->phase && (desc->workspace_bytes < plan.device_bytes ||
                        (plan.device_bytes && !desc->workspace))) ||
      (desc->phase &&
       (!desc->solution || desc->solution % (plan.desc.scalar ? 8 : 4))))
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                "cuSOLVERDn execution needs live handles and adequate storage");
  auto result =
      desc->cuda_stream
          ? solver_result(
                plan.set_stream(plan.handle, solver_pointer(desc->cuda_stream)),
                "cusolverDnSetStream")
          : TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
  if (result)
    return result;
  const auto &p = plan.desc;
  if (!desc->phase)
    result = solver_result(
        plan.factor(
            plan.handle, plan.params, 1, p.rows, p.scalar,
            solver_pointer(p.factor), p.rows, p.scalar,
            solver_pointer(desc->workspace), plan.device_bytes,
            plan.host_workspace.empty() ? nullptr : plan.host_workspace.data(),
            plan.host_workspace.size(),
            static_cast<int *>(solver_pointer(p.status))),
        "cusolverDnXpotrf");
  else
    result = solver_result(
        plan.solve(plan.handle, plan.params, 1, p.rows, p.rhs_count, p.scalar,
                   solver_pointer(p.factor), p.rows, p.scalar,
                   solver_pointer(desc->solution), p.rows,
                   static_cast<int *>(solver_pointer(p.status)) + 1),
        "cusolverDnXpotrs");
  if (desc->cuda_stream) {
    // Capture streams are temporary. Restore the fixed default-stream state
    // before the caller destroys them; ordinary submissions need no setter.
    const auto restored = solver_result(plan.set_stream(plan.handle, nullptr),
                                        "restore cusolverDn stream");
    if (!result)
      result = restored;
  }
  return result;
}

TiForgeRuntimeProviderResult solver_destroy_plan(TiForgeCusolverDnPlan value) {
  if (!value)
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                "cuSOLVERDn plan is null");
  auto *plan = static_cast<SolverPlan *>(value);
  const auto result = solver_release(*plan);
  if (result)
    return result;
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
  if (!runtime || !out_api || api_size < sizeof(TiForgeCusolverDnExecutionApi))
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                "cuSOLVERDn execution API output is null or truncated");
  if (version != TI_FORGE_RUNTIME_PROVIDER_EXECUTION_ABI_VERSION)
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_ABI_MISMATCH,
                "Unsupported cuSOLVERDn execution ABI");
  auto *api = static_cast<TiForgeCusolverDnExecutionApi *>(out_api);
  *api = {sizeof(*api), TI_FORGE_RUNTIME_PROVIDER_EXECUTION_ABI_VERSION,
          solver_create_plan, solver_execute, solver_destroy_plan};
  return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
}
