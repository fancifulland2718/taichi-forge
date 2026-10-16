#include "taichi/runtime/native_operation_registry.h"

#include <limits>
#include "taichi/common/operation_json.h"
#include "taichi/external_runtime/forge_cutlass_source_provider.h"
#include "taichi/runtime/native_array_contract.h"
#include "taichi/runtime/native_source_addon.h"
#ifdef TI_WITH_CUDA
#include "taichi/program/runtime_fault.h"
#include "taichi/rhi/cuda/cuda_device.h"
#endif

namespace taichi::lang {
namespace {
struct CutlassDescription {
  static constexpr const char *kind = "cutlass_source_plan";
  static constexpr uint32_t version = 1;
  TiForgeCutlassSourcePlanDesc plan{sizeof(TiForgeCutlassSourcePlanDesc)};
  uint64_t workspace_bytes{0};
  uint32_t cutlass_version{0};
  NativeSourceAddonIdentity identity;
};
class CutlassResource : public NativeResource {
 public:
  explicit CutlassResource(CutlassDescription d) : description(std::move(d)) {
  }
  const CutlassDescription description;
};
struct CutlassOperation {
  static constexpr const char *kind = "cutlass_source_execute";
  static constexpr uint32_t version = 1;
  std::shared_ptr<CutlassResource> plan;
  std::array<NativeArrayBinding, 3> arrays;
};

CutlassDescription decode(const liong::json::JsonValue &p) {
  using namespace operation_json;
  fields(p, {"m", "n", "k", "transpose_a", "transpose_b", "alpha", "beta",
             "activation", "strategy", "workspace_bytes", "cutlass_version",
             "binary_sha256", "target_code", "minimum_driver_api_version",
             "ptx_driver_api_version"});
  CutlassDescription d;
  auto positive = [&](const char *key) {
    const auto value = integer(p[key]);
    if (!value || value > uint64_t(std::numeric_limits<int32_t>::max()))
      throw std::invalid_argument("CUTLASS dimensions exceed addon int ABI");
    return int32_t(value);
  };
  d.plan.m = positive("m");
  d.plan.n = positive("n");
  d.plan.k = positive("k");
  d.plan.transpose_a = optional_bool(p, "transpose_a");
  d.plan.transpose_b = optional_bool(p, "transpose_b");
  d.plan.alpha = optional_f32(p, "alpha", 1);
  d.plan.beta = optional_f32(p, "beta", 0);
  const auto activation = name(p["activation"]);
  if (activation != "identity" && activation != "relu")
    throw std::invalid_argument("Invalid CUTLASS activation");
  d.plan.activation = activation == "relu";
  const std::array<std::string, 3> strategies{
      "direct_fused", "split_k_reduce_fused", "split_k_wide_reduce_fused"};
  auto it =
      std::find(strategies.begin(), strategies.end(), name(p["strategy"]));
  if (it == strategies.end())
    throw std::invalid_argument("Invalid CUTLASS strategy");
  d.plan.strategy = static_cast<uint32_t>(it - strategies.begin());
  d.workspace_bytes = integer(p["workspace_bytes"]);
  const uint64_t limit = std::numeric_limits<size_t>::max() / sizeof(float);
  for (auto [rows, cols] : {std::pair<uint64_t, uint64_t>{d.plan.m, d.plan.k},
                            {d.plan.k, d.plan.n},
                            {d.plan.m, d.plan.n}})
    if (rows > limit / cols)
      throw std::invalid_argument("CUTLASS array size exceeds platform ABI");
  const uint64_t partitions = d.plan.strategy == 2 ? 128 : 16;
  if (d.workspace_bytes > std::numeric_limits<size_t>::max() ||
      (d.plan.strategy && uint64_t(d.plan.m) * d.plan.n > limit / partitions))
    throw std::invalid_argument("CUTLASS workspace exceeds platform ABI");
  d.cutlass_version = source_addon_u32(p, "cutlass_version");
  if (!d.cutlass_version)
    throw std::invalid_argument("CUTLASS frozen version is invalid");
  d.identity = decode_source_addon_identity(p);
  return d;
}

void validate(Device &device, const CutlassOperation &op) {
  if (!op.plan || &op.plan->device() != &device ||
      op.plan->arch() != Arch::cuda)
    throw std::invalid_argument("CUTLASS requires a local CUDA plan");
  const auto &p = op.plan->description.plan;
  const std::array<std::vector<uint64_t>, 3> shapes{
      {{uint64_t(p.transpose_a ? p.k : p.m),
        uint64_t(p.transpose_a ? p.m : p.k)},
       {uint64_t(p.transpose_b ? p.n : p.k),
        uint64_t(p.transpose_b ? p.k : p.n)},
       {uint64_t(p.m), uint64_t(p.n)}}};
  std::array<NativeStorageRange, 3> ranges;
  for (size_t i = 0; i < ranges.size(); ++i) {
    const auto &array = op.arrays[i];
    if (array.pointer.device != &device ||
        array.scalar_type != PrimitiveType::f32 ||
        array.bytes != shapes[i][0] * shapes[i][1] * sizeof(float))
      throw std::invalid_argument("CUTLASS requires compact scalar f32 arrays");
    validate_native_array_layout(array, {shapes[i], {}, 4, 4, true});
    ranges[i] =
        resolve_native_storage_range(Arch::cuda, array.pointer, array.bytes);
  }
  if (!ranges[2].proven_disjoint(ranges[0]) ||
      !ranges[2].proven_disjoint(ranges[1]))
    throw std::invalid_argument("CUTLASS output must not overlap either input");
}

#ifdef TI_WITH_CUDA
const NativeProviderDefinition definition{
    "cutlass_source", TI_FORGE_CUTLASS_SOURCE_PROVIDER_QUERY_SYMBOL,
    "source addon C ABI 1"};
TiForgeCutlassSourceExecutionApi execution_api(
    const NativeProviderRuntime &runtime) {
  auto api = runtime.execution_api<TiForgeCutlassSourceExecutionApi>();
  if (!api.create_plan || !api.execute || !api.destroy_plan)
    throw NativeProviderError(TI_FORGE_RUNTIME_PROVIDER_ERROR_ABI_MISMATCH,
                              "CUTLASS execution API is incomplete");
  return api;
}
uint32_t version(const NativeProviderRuntime &runtime) {
  const auto &i = runtime.info();
  const auto value = uint64_t(i.version_major) * 10000 +
                     uint64_t(i.version_minor) * 100 + i.version_patch;
  if (!value || value > std::numeric_limits<uint32_t>::max())
    throw std::invalid_argument("CUTLASS version exceeds its ABI");
  return uint32_t(value);
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
struct CutlassLease {
  explicit CutlassLease(std::shared_ptr<NativeProviderRuntime> r)
      : runtime(std::move(r)), api(execution_api(*runtime)) {
  }
  ~CutlassLease() {
    if (handle) {
      auto result = api.destroy_plan(handle);
      TI_WARN_IF(result != TI_FORGE_RUNTIME_PROVIDER_SUCCESS,
                 "CUTLASS plan cleanup failed (result {}).", int(result));
    }
  }
  std::shared_ptr<NativeProviderRuntime> runtime;
  TiForgeCutlassSourceExecutionApi api;
  TiForgeCutlassSourcePlan handle{nullptr};
};
class CudaCutlassResource final : public CutlassResource {
 public:
  CudaCutlassResource(NativePreparationContext &context, CutlassDescription d)
      : CutlassResource(std::move(d)),
        device_(context.device),
        fault_(context.fault_domain),
        lease_(std::make_unique<CutlassLease>(
            context.providers.runtime(definition))) {
    auto &runtime = *lease_->runtime;
    if (runtime.info().build_version !=
            "sha256:" + description.identity.binary_sha256 ||
        version(runtime) != description.cutlass_version)
      throw std::invalid_argument(
          "CUTLASS binary/version differs from frozen identity");
    auto &driver = CUDADriver::get_instance_without_context();
    validate_source_addon_target(
        description.identity,
        uint32_t(CUDAContext::get_instance().get_compute_capability()),
        uint32_t(driver.get_version_major() * 1000 +
                 driver.get_version_minor() * 10));
    TiForgeCutlassSourcePlanInfo info{sizeof(info)};
    runtime.check(lease_->api.create_plan(runtime.handle(), &description.plan,
                                          &lease_->handle, &info),
                  "restore CUTLASS plan");
    if (!lease_->handle || info.struct_size < sizeof(info) || info.reserved ||
        info.workspace_bytes != description.workspace_bytes)
      throw std::invalid_argument("CUTLASS workspace differs from frozen plan");
    if (info.workspace_bytes) {
      auto allocation =
          device_.allocate_memory_unique({size_t(info.workspace_bytes)});
      if (allocation.second != RhiResult::success)
        throw std::runtime_error("CUTLASS workspace allocation failed");
      workspace_ = std::move(allocation.first);
      workspace_address_ = reinterpret_cast<uint64_t>(
          static_cast<cuda::CudaDevice &>(device_).get_memory_addr(
              *workspace_));
    }
  }
  ~CudaCutlassResource() override {
    auto abandon = [&] {
      lease_->handle =
          nullptr;  // Pin code and allocations after unsafe device loss.
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
    return size_t(description.workspace_bytes);
  }
  void record(const std::array<uint64_t, 3> &pointers, void *stream) {
    TiForgeCutlassSourceExecDesc invocation{sizeof(invocation),
                                            0,
                                            pointers[0],
                                            pointers[1],
                                            pointers[2],
                                            workspace_address_,
                                            description.workspace_bytes,
                                            reinterpret_cast<uint64_t>(stream)};
    lease_->runtime->check(lease_->api.execute(lease_->handle, &invocation),
                           "record CUTLASS operation");
  }

 private:
  Device &device_;
  std::shared_ptr<RuntimeFaultDomain> fault_;
  std::unique_ptr<CutlassLease> lease_;
  DeviceAllocationUnique workspace_;
  uint64_t workspace_address_{0};
};
class CudaCutlassOperation final : public NativeOperation {
 public:
  explicit CudaCutlassOperation(const CutlassOperation &op)
      : plan_(std::dynamic_pointer_cast<CudaCutlassResource>(op.plan)) {
    if (!plan_)
      throw std::invalid_argument("CUTLASS requires its native plan");
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
  std::shared_ptr<CudaCutlassResource> plan_;
  std::array<uint64_t, 3> pointers_{};
};
#endif
}  // namespace

void register_cutlass_source_operations(NativeOperationRegistry &registry) {
  registry.define<CutlassDescription>(
      [](Device &, const CutlassDescription &) {}, NativeObjectKind::resource);
  registry.register_schema<CutlassDescription>(
      "cutlass_source", "plan", {},
      [](const liong::json::JsonValue &p)
          -> NativeOperationRegistry::Binder<CutlassDescription> {
        auto d = decode(p);
        return [d](const NativeOperationBindings &) { return d; };
      });
  registry.define<CutlassOperation>(validate);
  registry.register_schema<CutlassOperation>(
      "cutlass_source", "execute", {"plan", "a", "b", "output"},
      [](const liong::json::JsonValue &p)
          -> NativeOperationRegistry::Binder<CutlassOperation> {
        operation_json::fields(p, {});
        return [](const NativeOperationBindings &b) {
          return CutlassOperation{
              b.at("plan").resource<CutlassResource>(),
              {b.at("a").array(), b.at("b").array(), b.at("output").array()}};
        };
      });
#ifdef TI_WITH_CUDA
  registry.register_context_factory<CutlassDescription>(
      Arch::cuda, NativeOperationMode::cuda_capture,
      [](NativePreparationContext &context, const CutlassDescription &d) {
        if (!dynamic_cast<cuda::CudaDevice *>(&context.device))
          throw std::invalid_argument("CUTLASS requires a CUDA device");
        auto &cuda = CUDAContext::get_instance();
        auto lock = cuda.get_submission_lock_guard();
        auto guard = cuda.get_guard();
        return std::make_shared<CudaCutlassResource>(context, d);
      },
      probe);
  registry.register_context_factory<CutlassOperation>(
      Arch::cuda, NativeOperationMode::cuda_capture,
      [](NativePreparationContext &, const CutlassOperation &op) {
        return std::make_shared<CudaCutlassOperation>(op);
      },
      probe);
#endif
}
}  // namespace taichi::lang
