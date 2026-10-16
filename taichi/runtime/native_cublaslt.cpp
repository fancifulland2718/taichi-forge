#include "taichi/runtime/native_operation_registry.h"

#include <array>
#include <limits>
#include "taichi/common/operation_json.h"
#include "taichi/external_runtime/forge_cublaslt_provider.h"
#include "taichi/runtime/native_array_contract.h"
#ifdef TI_WITH_CUDA
#include "taichi/program/runtime_fault.h"
#include "taichi/rhi/cuda/cuda_device.h"
#endif

namespace taichi::lang {
namespace {
struct MatmulDescription {
  static constexpr const char *kind = "cublaslt_matmul_plan_f32";
  static constexpr uint32_t version = 1;
  TiForgeCublasLtPlanDesc plan{sizeof(TiForgeCublasLtPlanDesc)};
  TiForgeCublasLtAlgorithm algorithm{sizeof(TiForgeCublasLtAlgorithm)};
  uint32_t provider_version{0};
};

class MatmulResource : public NativeResource {
 public:
  explicit MatmulResource(MatmulDescription description)
      : description(std::move(description)) {
  }
  const MatmulDescription description;
};

struct MatmulOperation {
  static constexpr const char *kind = "cublaslt_matmul_f32";
  static constexpr uint32_t version = 1;
  std::shared_ptr<MatmulResource> plan;
  std::array<NativeArrayBinding, 3> arrays;
};

void validate_description(const MatmulDescription &d) {
  const auto &p = d.plan;
  const uint64_t limit = std::numeric_limits<size_t>::max() / sizeof(float);
  if (!d.provider_version || p.m <= 0 || p.n <= 0 || p.k <= 0 ||
      !p.batch_count ||
      p.batch_count > uint32_t(std::numeric_limits<int32_t>::max()) ||
      p.layout > 1 || p.epilogue > 1 || p.transpose_a > 1 ||
      p.transpose_b > 1 || !std::isfinite(p.alpha) || !std::isfinite(p.beta) ||
      p.workspace_limit_bytes > std::numeric_limits<size_t>::max() ||
      d.algorithm.workspace_bytes > p.workspace_limit_bytes)
    throw std::invalid_argument("Invalid frozen cuBLASLt plan description");
  for (const auto &[rows, columns] :
       {std::pair<uint64_t, uint64_t>{p.m, p.k}, {p.k, p.n}, {p.m, p.n}}) {
    if (rows > limit / columns || rows * columns > limit / p.batch_count)
      throw std::invalid_argument("cuBLASLt compact array size overflows");
  }
}

std::array<std::vector<uint64_t>, 3> shapes(const MatmulDescription &d) {
  const auto &p = d.plan;
  std::array<std::vector<uint64_t>, 3> result{
      {{uint64_t(p.transpose_a ? p.k : p.m),
        uint64_t(p.transpose_a ? p.m : p.k)},
       {uint64_t(p.transpose_b ? p.n : p.k),
        uint64_t(p.transpose_b ? p.k : p.n)},
       {uint64_t(p.m), uint64_t(p.n)}}};
  if (p.batch_count > 1)
    for (auto &shape : result)
      shape.insert(shape.begin(), p.batch_count);
  return result;
}

void validate_operation(Device &device, const MatmulOperation &op) {
  if (!op.plan || &op.plan->device() != &device ||
      op.plan->arch() != Arch::cuda)
    throw std::invalid_argument("cuBLASLt requires a local CUDA plan");
  const auto expected = shapes(op.plan->description);
  std::array<NativeStorageRange, 3> ranges;
  for (size_t i = 0; i < ranges.size(); ++i) {
    const auto &array = op.arrays[i];
    uint64_t bytes = sizeof(float);
    for (const auto extent : expected[i])
      bytes *= extent;  // validate_description checked the complete product.
    if (array.pointer.device != &device ||
        array.scalar_type != PrimitiveType::f32 || array.bytes != bytes)
      throw std::invalid_argument(
          "cuBLASLt requires compact scalar f32 arrays");
    validate_native_array_layout(array, {expected[i], {}, 4, 4, true});
    ranges[i] =
        resolve_native_storage_range(Arch::cuda, array.pointer, array.bytes);
  }
  if (!ranges[2].proven_disjoint(ranges[0]) ||
      !ranges[2].proven_disjoint(ranges[1]))
    throw std::invalid_argument("cuBLASLt output must not alias either input");
}

MatmulDescription decode_plan(const liong::json::JsonValue &p) {
  using namespace operation_json;
  fields(p, {"m", "n", "k", "batch_count", "transpose_a", "transpose_b",
             "alpha", "beta", "layout", "epilogue", "workspace_limit_bytes",
             "algorithm", "provider_version"});
  MatmulDescription d;
  auto positive = [&](const char *key, uint64_t maximum) {
    const auto value = integer(p[key]);
    if (!value || value > maximum)
      throw std::invalid_argument(std::string("cuBLASLt ") + key +
                                  " exceeds its ABI");
    return value;
  };
  d.plan.m = positive("m", std::numeric_limits<int64_t>::max());
  d.plan.n = positive("n", std::numeric_limits<int64_t>::max());
  d.plan.k = positive("k", std::numeric_limits<int64_t>::max());
  d.plan.batch_count =
      positive("batch_count", std::numeric_limits<int32_t>::max());
  d.provider_version =
      positive("provider_version", std::numeric_limits<uint32_t>::max());
  d.plan.transpose_a = optional_bool(p, "transpose_a");
  d.plan.transpose_b = optional_bool(p, "transpose_b");
  d.plan.alpha = optional_f32(p, "alpha", 1);
  d.plan.beta = optional_f32(p, "beta", 0);
  const auto layout = name(p["layout"]), epilogue = name(p["epilogue"]);
  if ((layout != "row_major" && layout != "transposed_column_major") ||
      (epilogue != "identity" && epilogue != "relu"))
    throw std::invalid_argument("Invalid cuBLASLt physical layout/epilogue");
  d.plan.layout = layout == "transposed_column_major";
  d.plan.epilogue = epilogue == "relu";
  d.plan.workspace_limit_bytes = integer(p["workspace_limit_bytes"]);
  const auto &algorithm = p["algorithm"];
  fields(algorithm, {"configuration", "workspace_bytes"});
  d.algorithm.workspace_bytes = integer(algorithm["workspace_bytes"]);
  const auto &config = algorithm["configuration"];
  fields(config, {"id", "tile", "split_k", "reduction", "swizzle", "custom",
                  "stages", "inner_shape", "cluster_shape"});
  const std::array<const char *, 9> keys{
      "id",     "tile",   "split_k",     "reduction",    "swizzle",
      "custom", "stages", "inner_shape", "cluster_shape"};
  for (size_t i = 0; i < keys.size(); ++i) {
    const auto found = config.obj.inner.find(keys[i]);
    if (i >= 7 && (found == config.obj.inner.end() ||
                   found->second.ty == liong::json::L_JSON_NULL))
      continue;
    if (found == config.obj.inner.end())
      throw std::invalid_argument("Missing cuBLASLt configuration attribute");
    const auto &value = found->second;
    if (value.ty != liong::json::L_JSON_INT)
      throw std::invalid_argument(
          "cuBLASLt configuration values must be integers");
    const auto number = value.num_int;
    const bool signed_value = i == 0 || i == 2;
    const int64_t minimum =
        signed_value ? std::numeric_limits<int32_t>::min() : 0;
    const int64_t maximum =
        signed_value ? int64_t(std::numeric_limits<int32_t>::max())
        : i >= 7     ? int64_t(std::numeric_limits<uint16_t>::max())
                     : int64_t(std::numeric_limits<uint32_t>::max());
    if (number < minimum || number > maximum)
      throw std::invalid_argument(
          "cuBLASLt configuration exceeds vendor attribute ABI");
    d.algorithm.present_mask |= 1u << i;
    d.algorithm.values[i] = number;
  }
  validate_description(d);
  return d;
}

#ifdef TI_WITH_CUDA
const NativeProviderDefinition definition{
    "cublaslt", TI_FORGE_CUBLASLT_PROVIDER_QUERY_SYMBOL,
    "documented f32 matmul configuration C API"};

TiForgeCublasLtExecutionApi execution_api(
    const NativeProviderRuntime &runtime) {
  auto api = runtime.execution_api<TiForgeCublasLtExecutionApi>();
  if (!api.create_plan || !api.execute || !api.destroy_plan)
    throw NativeProviderError(TI_FORGE_RUNTIME_PROVIDER_ERROR_ABI_MISMATCH,
                              "cuBLASLt execution API is incomplete");
  return api;
}

uint32_t version(const NativeProviderRuntime &runtime) {
  const auto &info = runtime.info();
  const uint64_t result = uint64_t(info.version_major) * 10000 +
                          uint64_t(info.version_minor) * 100 +
                          info.version_patch;
  if (!result || result > std::numeric_limits<uint32_t>::max())
    throw std::invalid_argument(
        "cuBLASLt runtime version exceeds the plan ABI");
  return uint32_t(result);
}

NativeOperationAvailability probe(NativeProviderSession &providers) {
  auto &driver = CUDADriver::get_instance_without_context();
  if (!driver.stream_begin_capture.available() ||
      !driver.stream_end_capture.available() ||
      !driver.graph_instantiate_with_flags.available() ||
      !driver.graph_launch.available())
    return {};
  try {
    auto runtime = providers.runtime(definition);
    execution_api(*runtime);
    return {true, version(*runtime)};
  } catch (const NativeProviderError &error) {
    if (error.result() == TI_FORGE_RUNTIME_PROVIDER_ERROR_RUNTIME_UNAVAILABLE ||
        error.result() == TI_FORGE_RUNTIME_PROVIDER_ERROR_RUNTIME_INCOMPATIBLE)
      return {};
    throw;
  }
}

struct PlanLease {
  explicit PlanLease(std::shared_ptr<NativeProviderRuntime> runtime)
      : runtime(std::move(runtime)), api(execution_api(*this->runtime)) {
  }
  ~PlanLease() {
    if (handle) {
      const auto result = api.destroy_plan(handle);
      TI_WARN_IF(result != TI_FORGE_RUNTIME_PROVIDER_SUCCESS,
                 "cuBLASLt plan cleanup failed (result {}).", int(result));
    }
  }
  std::shared_ptr<NativeProviderRuntime> runtime;
  TiForgeCublasLtExecutionApi api;
  TiForgeCublasLtPlan handle{nullptr};
};

class CudaMatmulResource final : public MatmulResource {
 public:
  CudaMatmulResource(NativePreparationContext &context, MatmulDescription d)
      : MatmulResource(std::move(d)),
        device_(context.device),
        fault_(context.fault_domain),
        lease_(std::make_unique<PlanLease>(
            context.providers.runtime(definition))) {
    if (version(*lease_->runtime) != description.provider_version)
      throw std::invalid_argument(
          "cuBLASLt provider version differs from the frozen plan");
    TiForgeCublasLtPlanInfo info{sizeof(info)};
    lease_->runtime->check(
        lease_->api.create_plan(lease_->runtime->handle(), &description.plan,
                                &description.algorithm, &lease_->handle, &info),
        "restore matmul plan");
    if (!lease_->handle || info.struct_size < sizeof(info) ||
        info.heuristic_calls || info.restoration_calls != 1 ||
        info.algorithm.workspace_bytes != description.algorithm.workspace_bytes)
      throw std::runtime_error(
          "cuBLASLt returned an invalid restored plan/workspace");
    bytes_ = size_t(info.algorithm.workspace_bytes);
    if (bytes_) {
      auto allocation = device_.allocate_memory_unique({bytes_});
      if (allocation.second != RhiResult::success)
        throw std::runtime_error("cuBLASLt workspace allocation failed");
      workspace_ = std::move(allocation.first);
      workspace_address_ = reinterpret_cast<uint64_t>(
          static_cast<cuda::CudaDevice &>(device_).get_memory_addr(
              *workspace_));
    }
  }
  ~CudaMatmulResource() override {
    auto abandon = [&] {
      lease_->handle =
          nullptr;  // Live adapter count prevents unloading its code.
      workspace_.release();
    };
    if (fault_ && (!fault_->backend_calls_safe() ||
                   fault_->state() == RuntimeLifecycleState::kFinalized)) {
      abandon();
      return;
    }
    try {
      auto &cuda = CUDAContext::get_instance();
      auto lock = cuda.get_submission_lock_guard();
      auto guard = cuda.get_guard();
      lease_.reset();
      workspace_.reset();
    } catch (const BackendRuntimeError &error) {
      if (fault_)
        fault_->report_backend_error(error, 0);
      if (lease_)
        abandon();
    } catch (...) {
      if (lease_)
        abandon();
    }
  }
  Arch arch() const override {
    return Arch::cuda;
  }
  Device &device() const override {
    return device_;
  }
  size_t workspace_bytes() const override {
    return bytes_;
  }
  void record(const std::array<uint64_t, 3> &pointers, void *stream) {
    TiForgeCublasLtExecDesc exec{
        sizeof(exec), 0,
        pointers[0],  pointers[1],
        pointers[2],  workspace_address_,
        bytes_,       reinterpret_cast<uint64_t>(stream)};
    lease_->runtime->check(lease_->api.execute(lease_->handle, &exec),
                           "record matmul");
  }

 private:
  Device &device_;
  std::shared_ptr<RuntimeFaultDomain> fault_;
  std::unique_ptr<PlanLease> lease_;
  DeviceAllocationUnique workspace_;
  uint64_t workspace_address_{0};
  size_t bytes_{0};
};

class CudaMatmulOperation final : public NativeOperation {
 public:
  explicit CudaMatmulOperation(const MatmulOperation &op)
      : plan_(std::dynamic_pointer_cast<CudaMatmulResource>(op.plan)) {
    if (!plan_)
      throw std::invalid_argument("cuBLASLt requires a native matmul plan");
    for (size_t i = 0; i < pointers_.size(); ++i)
      pointers_[i] = resolve_native_storage_range(
                         Arch::cuda, op.arrays[i].pointer, op.arrays[i].bytes)
                         .memory_offset;
  }
  Arch arch() const override {
    return Arch::cuda;
  }
  Device &device() const override {
    return plan_->device();
  }
  size_t workspace_bytes() const override {
    return 0;
  }
  void record_cuda(void *stream) override {
    plan_->record(pointers_, stream);
  }

 private:
  std::shared_ptr<CudaMatmulResource> plan_;
  std::array<uint64_t, 3> pointers_;
};
#endif
}  // namespace

void register_cublaslt_operations(NativeOperationRegistry &registry) {
  registry.define<MatmulDescription>(
      [](Device &, const MatmulDescription &d) { validate_description(d); },
      NativeObjectKind::resource);
  registry.register_schema<MatmulDescription>(
      "cublaslt", "matmul_plan_f32", {},
      [](const liong::json::JsonValue &p)
          -> NativeOperationRegistry::Binder<MatmulDescription> {
        auto d = decode_plan(p);
        return [d](const NativeOperationBindings &) { return d; };
      });
  registry.define<MatmulOperation>(validate_operation);
  registry.register_schema<MatmulOperation>(
      "cublaslt", "matmul_f32", {"plan", "a", "b", "output"},
      [](const liong::json::JsonValue &p)
          -> NativeOperationRegistry::Binder<MatmulOperation> {
        operation_json::fields(p, {});
        return [](const NativeOperationBindings &b) {
          return MatmulOperation{
              b.at("plan").resource<MatmulResource>(),
              {b.at("a").array(), b.at("b").array(), b.at("output").array()}};
        };
      });
#ifdef TI_WITH_CUDA
  registry.register_context_factory<MatmulDescription>(
      Arch::cuda, NativeOperationMode::cuda_capture,
      [](NativePreparationContext &context, const MatmulDescription &d) {
        if (!dynamic_cast<cuda::CudaDevice *>(&context.device))
          throw std::invalid_argument("cuBLASLt requires a CUDA device");
        auto &cuda = CUDAContext::get_instance();
        auto lock = cuda.get_submission_lock_guard();
        auto guard = cuda.get_guard();
        return std::make_shared<CudaMatmulResource>(context, d);
      },
      probe);
  registry.register_context_factory<MatmulOperation>(
      Arch::cuda, NativeOperationMode::cuda_capture,
      [](NativePreparationContext &, const MatmulOperation &op) {
        return std::make_shared<CudaMatmulOperation>(op);
      },
      probe);
#endif
}
}  // namespace taichi::lang
