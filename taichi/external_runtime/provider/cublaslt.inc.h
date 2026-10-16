// Included in the common adapter's namespace for provider kind 5. Only the
// documented dynamic C ABI is used; no CUDA SDK headers or vendor linkage.
struct LtAlgorithm {
  uint64_t data[8];
};
struct LtResult {
  LtAlgorithm algorithm;
  size_t workspace;
  int state;
  float waves;
  int reserved[4];
};
using LtCreate = int (*)(void **);
using LtDestroy = int (*)(void *);
using LtSetAttribute = int (*)(void *, int, const void *, size_t);
using LtMatmul = int (*)(void *,
                         void *,
                         const void *,
                         const void *,
                         void *,
                         const void *,
                         void *,
                         const void *,
                         const void *,
                         void *,
                         void *,
                         void *,
                         const LtAlgorithm *,
                         void *,
                         size_t,
                         void *);
using LtGetConfig = int (*)(const LtAlgorithm *, int, void *, size_t, size_t *);

struct LtFailure {
  TiForgeRuntimeProviderResult result;
  std::string message;
};
void lt_require(int status, const char *operation) {
  if (status)
    throw LtFailure{TI_FORGE_RUNTIME_PROVIDER_ERROR_VENDOR_CALL,
                    std::string(operation) + " failed with status " +
                        std::to_string(status)};
}
void lt_invalid(const char *message) {
  throw LtFailure{TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT, message};
}
struct LtPlan {
  Runtime *runtime{nullptr};
  TiForgeCublasLtPlanDesc desc{};
  TiForgeCublasLtPlanInfo info{};
  LtAlgorithm algorithm{};
  void *handle{nullptr}, *descriptor{nullptr}, *preference{nullptr};
  std::array<void *, 4> layouts{};
  std::array<uint64_t, 3> bytes{};
  LtDestroy destroy_handle{nullptr}, destroy_descriptor{nullptr},
      destroy_layout{nullptr}, destroy_preference{nullptr};
  LtMatmul matmul{nullptr};
  template <typename F>
  F symbol(const char *name) const {
    return reinterpret_cast<F>(load_symbol(runtime->library, name));
  }
};

int lt_release(LtPlan &plan) {
  int error = 0;
  const auto release = [&](void *&handle, LtDestroy destroy) {
    if (handle) {
      const auto status = destroy(handle);
      if (!status)
        handle = nullptr;
      else if (!error)
        error = status;
    }
  };
  release(plan.preference, plan.destroy_preference);
  for (auto i = plan.layouts.rbegin(); i != plan.layouts.rend(); ++i)
    release(*i, plan.destroy_layout);
  release(plan.descriptor, plan.destroy_descriptor);
  // Keep the handle if a descriptor could not retire; runtime close stays
  // pinned.
  if (!error)
    release(plan.handle, plan.destroy_handle);
  return error;
}

void lt_validate(const TiForgeCublasLtPlanDesc &d,
                 const TiForgeCublasLtAlgorithm *choice,
                 std::array<uint64_t, 3> &bytes) {
  if (d.layout > 1 || d.epilogue > 1 || d.transpose_a > 1 ||
      d.transpose_b > 1 || d.m <= 0 || d.n <= 0 || d.k <= 0 || !d.batch_count ||
      d.batch_count > uint32_t(std::numeric_limits<int>::max()) ||
      !std::isfinite(d.alpha) || !std::isfinite(d.beta) ||
      d.workspace_limit_bytes > std::numeric_limits<size_t>::max())
    lt_invalid(
        "Invalid cuBLASLt dimensions, layout, scalars or workspace limit");
  const auto limit = std::numeric_limits<size_t>::max() / sizeof(float);
  const std::array<std::pair<uint64_t, uint64_t>, 3> shapes{
      {{uint64_t(d.m), uint64_t(d.k)},
       {uint64_t(d.k), uint64_t(d.n)},
       {uint64_t(d.m), uint64_t(d.n)}}};
  for (size_t i = 0; i < shapes.size(); ++i) {
    const auto [rows, columns] = shapes[i];
    if (rows > limit / columns || rows * columns > limit / d.batch_count)
      lt_invalid("cuBLASLt compact array size overflows address space");
    bytes[i] = rows * columns * d.batch_count * sizeof(float);
  }
  if (!choice)
    return;
  if (choice->struct_size < sizeof(*choice) ||
      (choice->present_mask & 127u) != 127u || (choice->present_mask & ~511u) ||
      choice->workspace_bytes > d.workspace_limit_bytes)
    lt_invalid("Invalid cuBLASLt frozen configuration or workspace");
  for (int i = 0; i < 9; ++i) {
    const auto value = choice->values[i];
    if (!(choice->present_mask & (1u << i))) {
      if (value)
        lt_invalid("Absent cuBLASLt configuration attributes must be zero");
    } else if ((i == 0 || i == 2)
                   ? (value < std::numeric_limits<int32_t>::min() ||
                      value > std::numeric_limits<int32_t>::max())
                   : (value < 0 ||
                      uint64_t(value) >
                          (i >= 7
                               ? uint64_t(std::numeric_limits<uint16_t>::max())
                               : uint64_t(
                                     std::numeric_limits<uint32_t>::max())))) {
      lt_invalid("cuBLASLt configuration exceeds its vendor attribute ABI");
    }
  }
}

TiForgeCublasLtAlgorithm lt_describe(const LtPlan &plan, size_t workspace) {
  TiForgeCublasLtAlgorithm result{};
  result.struct_size = sizeof(result);
  result.workspace_bytes = workspace;
  const auto get =
      plan.symbol<LtGetConfig>("cublasLtMatmulAlgoConfigGetAttribute");
  for (int i = 0; i < 9; ++i) {
    uint32_t value = 0;
    const size_t bytes = i >= 7 ? sizeof(uint16_t) : sizeof(uint32_t);
    size_t written = 0;
    const auto status = get(&plan.algorithm, i, &value, bytes, &written);
    if (i >= 7 && (status == 7 || status == 15))
      continue;  // Optional shape attributes may be absent in older vendor
                 // APIs.
    lt_require(status, "cuBLASLt algorithm configuration query");
    if (written != bytes)
      lt_invalid("cuBLASLt configuration attribute ABI size differs");
    result.present_mask |= 1u << i;
    result.values[i] =
        i == 0 || i == 2 ? int64_t(int32_t(value)) : int64_t(value);
  }
  return result;
}

void lt_descriptors(LtPlan &plan) {
  const auto &d = plan.desc;
  const auto create = plan.symbol<LtCreate>("cublasLtCreate");
  using CreateDesc = int (*)(void **, int, int);
  const auto desc = plan.symbol<CreateDesc>("cublasLtMatmulDescCreate");
  const auto attribute =
      plan.symbol<LtSetAttribute>("cublasLtMatmulDescSetAttribute");
  lt_require(create(&plan.handle), "cuBLASLt handle creation");
  lt_require(desc(&plan.descriptor, 68, 0), "cuBLASLt f32 descriptor creation");
  const int trans_a = d.layout ? d.transpose_b : d.transpose_a;
  const int trans_b = d.layout ? d.transpose_a : d.transpose_b;
  lt_require(attribute(plan.descriptor, 3, &trans_a, sizeof(trans_a)),
             "cuBLASLt transpose A");
  lt_require(attribute(plan.descriptor, 4, &trans_b, sizeof(trans_b)),
             "cuBLASLt transpose B");
  if (d.epilogue) {
    const uint32_t relu = 2;
    lt_require(attribute(plan.descriptor, 7, &relu, sizeof(relu)),
               "cuBLASLt ReLU epilogue");
  }
  std::array<std::pair<uint64_t, uint64_t>, 4> shapes{
      {{uint64_t(d.transpose_a ? d.k : d.m),
        uint64_t(d.transpose_a ? d.m : d.k)},
       {uint64_t(d.transpose_b ? d.n : d.k),
        uint64_t(d.transpose_b ? d.k : d.n)},
       {uint64_t(d.m), uint64_t(d.n)},
       {uint64_t(d.m), uint64_t(d.n)}}};
  if (d.layout) {
    std::swap(shapes[0], shapes[1]);
    for (auto &shape : shapes)
      std::swap(shape.first, shape.second);
  }
  using CreateLayout = int (*)(void **, int, uint64_t, uint64_t, int64_t);
  const auto layout = plan.symbol<CreateLayout>("cublasLtMatrixLayoutCreate");
  const auto set =
      plan.symbol<LtSetAttribute>("cublasLtMatrixLayoutSetAttribute");
  for (size_t i = 0; i < shapes.size(); ++i) {
    const auto [rows, columns] = shapes[i];
    lt_require(
        layout(&plan.layouts[i], 0, rows, columns, d.layout ? rows : columns),
        "cuBLASLt layout creation");
    const int order = d.layout ? 0 : 1;
    lt_require(set(plan.layouts[i], 1, &order, sizeof(order)),
               "cuBLASLt layout order");
    if (d.batch_count > 1) {
      const int batch = d.batch_count;
      const int64_t stride = rows * columns;
      lt_require(set(plan.layouts[i], 5, &batch, sizeof(batch)),
                 "cuBLASLt batch count");
      lt_require(set(plan.layouts[i], 6, &stride, sizeof(stride)),
                 "cuBLASLt batch stride");
    }
  }
}

void lt_select(LtPlan &plan, const TiForgeCublasLtAlgorithm *choice) {
  LtResult result{};
  if (choice) {
    using Init =
        int (*)(void *, int, int, int, int, int, int, int, LtAlgorithm *);
    using Set = int (*)(LtAlgorithm *, int, const void *, size_t);
    using Check = int (*)(void *, void *, void *, void *, void *, void *,
                          const LtAlgorithm *, LtResult *);
    ++plan.info.restoration_calls;
    lt_require(plan.symbol<Init>("cublasLtMatmulAlgoInit")(
                   plan.handle, 68, 0, 0, 0, 0, 0, int(choice->values[0]),
                   &plan.algorithm),
               "cuBLASLt algorithm initialization");
    const auto set = plan.symbol<Set>("cublasLtMatmulAlgoConfigSetAttribute");
    for (int i = 1; i < 9; ++i) {
      if (!(choice->present_mask & (1u << i)))
        continue;
      if (i >= 7) {
        const uint16_t value = static_cast<uint16_t>(choice->values[i]);
        lt_require(set(&plan.algorithm, i, &value, sizeof(value)),
                   "cuBLASLt shape configuration");
      } else {
        const uint32_t value = static_cast<uint32_t>(choice->values[i]);
        lt_require(set(&plan.algorithm, i, &value, sizeof(value)),
                   "cuBLASLt algorithm configuration");
      }
    }
    lt_require(
        plan.symbol<Check>("cublasLtMatmulAlgoCheck")(
            plan.handle, plan.descriptor, plan.layouts[0], plan.layouts[1],
            plan.layouts[2], plan.layouts[3], &plan.algorithm, &result),
        "cuBLASLt restored algorithm check");
  } else {
    using Heuristic = int (*)(void *, void *, void *, void *, void *, void *,
                              void *, int, LtResult *, int *);
    lt_require(plan.symbol<LtCreate>("cublasLtMatmulPreferenceCreate")(
                   &plan.preference),
               "cuBLASLt preference creation");
    const size_t limit = plan.desc.workspace_limit_bytes;
    lt_require(
        plan.symbol<LtSetAttribute>("cublasLtMatmulPreferenceSetAttribute")(
            plan.preference, 1, &limit, sizeof(limit)),
        "cuBLASLt workspace preference");
    int count = 0;
    ++plan.info.heuristic_calls;
    lt_require(plan.symbol<Heuristic>("cublasLtMatmulAlgoGetHeuristic")(
                   plan.handle, plan.descriptor, plan.layouts[0],
                   plan.layouts[1], plan.layouts[2], plan.layouts[3],
                   plan.preference, 1, &result, &count),
               "cuBLASLt algorithm heuristic");
    if (count != 1)
      lt_invalid("cuBLASLt found no compatible f32 matmul algorithm");
    plan.algorithm = result.algorithm;
    lt_require(plan.destroy_preference(plan.preference),
               "cuBLASLt preference destruction");
    plan.preference = nullptr;
  }
  lt_require(result.state, "cuBLASLt selected algorithm state");
  if (result.workspace > plan.desc.workspace_limit_bytes)
    lt_invalid("cuBLASLt selected workspace exceeds the configured limit");
  plan.info.algorithm = lt_describe(plan, result.workspace);
  if (choice &&
      (choice->present_mask != plan.info.algorithm.present_mask ||
       choice->workspace_bytes != plan.info.algorithm.workspace_bytes ||
       !std::equal(std::begin(choice->values), std::end(choice->values),
                   std::begin(plan.info.algorithm.values))))
    lt_invalid(
        "cuBLASLt restored algorithm configuration or workspace drifted");
  const auto get =
      plan.symbol<LtGetConfig>("cublasLtMatmulAlgoCapGetAttribute");
  // Public AlgoCap attributes 16..19 describe physical A/B/C/D alignment.
  for (int i = 0; i < 4; ++i) {
    uint32_t alignment = 0;
    size_t written = 0;
    lt_require(
        get(&plan.algorithm, 16 + i, &alignment, sizeof(alignment), &written),
        "cuBLASLt algorithm alignment query");
    if (written != sizeof(alignment))
      lt_invalid("cuBLASLt alignment attribute ABI size differs");
    plan.info.minimum_alignment[i] =
        std::max(alignment, uint32_t(sizeof(float)));
  }
}

TiForgeRuntimeProviderResult lt_create_plan(
    TiForgeRuntimeProviderRuntime runtime_value,
    const TiForgeCublasLtPlanDesc *desc,
    const TiForgeCublasLtAlgorithm *choice,
    TiForgeCublasLtPlan *out_plan,
    TiForgeCublasLtPlanInfo *out_info) {
  if (out_plan)
    *out_plan = nullptr;
  if (!runtime_value || !desc || desc->struct_size < sizeof(*desc) ||
      !out_plan || !out_info || out_info->struct_size < sizeof(*out_info))
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                "cuBLASLt plan descriptor/output is null or truncated");
  auto *runtime = static_cast<Runtime *>(runtime_value);
  std::unique_ptr<LtPlan> plan;
  TiForgeRuntimeProviderResult error;
  std::string message;
  try {
    plan = std::make_unique<LtPlan>();
    plan->runtime = runtime;
    plan->desc = *desc;
    plan->info.struct_size = sizeof(plan->info);
    lt_validate(*desc, choice, plan->bytes);
    plan->destroy_handle = plan->symbol<LtDestroy>("cublasLtDestroy");
    plan->destroy_descriptor =
        plan->symbol<LtDestroy>("cublasLtMatmulDescDestroy");
    plan->destroy_layout =
        plan->symbol<LtDestroy>("cublasLtMatrixLayoutDestroy");
    plan->destroy_preference =
        plan->symbol<LtDestroy>("cublasLtMatmulPreferenceDestroy");
    plan->matmul = plan->symbol<LtMatmul>("cublasLtMatmul");
    lt_descriptors(*plan);
    lt_select(*plan, choice);
    *out_info = plan->info;
    runtime->live_execution_resources.fetch_add(1);
    *out_plan = plan.release();
    return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
  } catch (const LtFailure &failure) {
    error = failure.result;
    message = failure.message;
  } catch (const std::exception &) {
    error = TI_FORGE_RUNTIME_PROVIDER_ERROR_OUT_OF_MEMORY;
    message = "cuBLASLt plan host allocation failed";
  }
  // A failed vendor cleanup keeps code loaded and cannot become a dangling
  // library reference. No partially initialized plan is published to callers.
  if (plan && lt_release(*plan)) {
    runtime->live_execution_resources.fetch_add(1);
    plan.release();
  }
  return fail(error, std::move(message));
}

TiForgeRuntimeProviderResult lt_execute(TiForgeCublasLtPlan value,
                                        const TiForgeCublasLtExecDesc *desc) {
  if (!value || !desc || desc->struct_size < sizeof(*desc) || desc->reserved)
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                "Invalid cuBLASLt execution descriptor");
  auto &plan = *static_cast<LtPlan *>(value);
  const auto &info = plan.info;
  const auto &p = plan.desc;
  const std::array<uint64_t, 4> pointers{p.layout ? desc->b : desc->a,
                                         p.layout ? desc->a : desc->b,
                                         desc->output, desc->output};
  const std::array<uint64_t, 4> sizes{plan.bytes[0], plan.bytes[1],
                                      plan.bytes[2],
                                      info.algorithm.workspace_bytes};
  const std::array<uint64_t, 4> storage{desc->a, desc->b, desc->output,
                                        desc->workspace};
  for (int i = 0; i < 4; ++i)
    if (!pointers[i] || pointers[i] % info.minimum_alignment[i])
      return fail(
          TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
          "cuBLASLt operand does not satisfy selected algorithm alignment");
  if (desc->workspace_bytes < info.algorithm.workspace_bytes ||
      (info.algorithm.workspace_bytes &&
       (!desc->workspace || desc->workspace % 256)))
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                "cuBLASLt requires adequate 256-byte aligned workspace");
  for (int i = 0; i < 4; ++i) {
    if (storage[i] > std::numeric_limits<uintptr_t>::max() ||
        sizes[i] > std::numeric_limits<uintptr_t>::max() - storage[i])
      return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                  "cuBLASLt pointer range overflows");
    for (int j = 0; j < i; ++j) {
      if (i < 2 || !sizes[i] || !sizes[j])
        continue;
      if (storage[i] < storage[j] + sizes[j] &&
          storage[j] < storage[i] + sizes[i])
        return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                    "cuBLASLt output/workspace must not overlap operands");
    }
  }
  const auto pointer = [](uint64_t address) {
    return reinterpret_cast<void *>(static_cast<uintptr_t>(address));
  };
  const auto status = plan.matmul(
      plan.handle, plan.descriptor, &p.alpha, pointer(pointers[0]),
      plan.layouts[0], pointer(pointers[1]), plan.layouts[1], &p.beta,
      pointer(desc->output), plan.layouts[2], pointer(desc->output),
      plan.layouts[3], &plan.algorithm, pointer(desc->workspace),
      static_cast<size_t>(info.algorithm.workspace_bytes),
      pointer(desc->cuda_stream));
  return status ? fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_VENDOR_CALL,
                       "cuBLASLt matmul failed with status " +
                           std::to_string(status))
                : TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
}

TiForgeRuntimeProviderResult lt_destroy_plan(TiForgeCublasLtPlan value) {
  if (!value)
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                "cuBLASLt plan is null");
  auto *plan = static_cast<LtPlan *>(value);
  const auto error = lt_release(*plan);
  if (error)
    return fail(
        TI_FORGE_RUNTIME_PROVIDER_ERROR_VENDOR_CALL,
        "cuBLASLt plan retirement failed with status " + std::to_string(error));
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
  if (!runtime || !out_api || api_size < sizeof(TiForgeCublasLtExecutionApi))
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT,
                "cuBLASLt execution API output is truncated");
  if (version != TI_FORGE_RUNTIME_PROVIDER_EXECUTION_ABI_VERSION)
    return fail(TI_FORGE_RUNTIME_PROVIDER_ERROR_ABI_MISMATCH,
                "Unsupported cuBLASLt execution ABI");
  auto *api = static_cast<TiForgeCublasLtExecutionApi *>(out_api);
  *api = {sizeof(*api), TI_FORGE_RUNTIME_PROVIDER_EXECUTION_ABI_VERSION,
          lt_create_plan, lt_execute, lt_destroy_plan};
  return TI_FORGE_RUNTIME_PROVIDER_SUCCESS;
}
