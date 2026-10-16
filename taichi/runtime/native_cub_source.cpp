#include "taichi/runtime/native_operation_registry.h"

#include <limits>
#include "taichi/common/operation_json.h"
#include "taichi/external_runtime/forge_cub_source_provider.h"
#include "taichi/runtime/native_array_contract.h"
#include "taichi/runtime/native_source_addon.h"
#ifdef TI_WITH_CUDA
#include "taichi/program/runtime_fault.h"
#include "taichi/rhi/cuda/cuda_device.h"
#endif

namespace taichi::lang {
namespace {
struct CubDescription {
  static constexpr const char *kind = "cub_source_plan";
  static constexpr uint32_t version = 1;
  TiForgeCubSourcePlanDesc plan{sizeof(TiForgeCubSourcePlanDesc)};
  uint64_t workspace_bytes{0};
  uint32_t cub_version{0}, cuda_runtime_version{0};
  NativeSourceAddonIdentity identity;
  bool signed_values{false};
};
class CubResource : public NativeResource {
 public:
  explicit CubResource(CubDescription d) : description(std::move(d)) {
  }
  const CubDescription description;
};
struct CubOperation {
  static constexpr const char *kind = "cub_source_execute";
  static constexpr uint32_t version = 1;
  std::shared_ptr<CubResource> plan;
  std::array<std::optional<NativeArrayBinding>, 4> arrays;
};

CubDescription decode(const liong::json::JsonValue &p) {
  using namespace operation_json;
  fields(p,
         {"operation", "num_items", "dtype", "workspace_bytes", "cub_version",
          "cuda_runtime_version", "binary_sha256", "target_code",
          "minimum_driver_api_version", "ptx_driver_api_version"});
  CubDescription d;
  const std::array<std::string, 6> names{
      "radix_sort_pairs_u32",         "radix_sort_pairs_u64",
      "exclusive_scan_u32",           "select_flagged_u32",
      "segmented_inclusive_scan_u32", "segmented_exclusive_scan_u32"};
  auto it = std::find(names.begin(), names.end(), name(p["operation"]));
  if (it == names.end())
    throw std::invalid_argument("Unknown CUB source operation");
  d.plan.operation = static_cast<uint32_t>(it - names.begin() + 1);
  d.plan.num_items = integer(p["num_items"]);
  d.workspace_bytes = integer(p["workspace_bytes"]);
  if (d.plan.num_items > uint64_t(std::numeric_limits<int>::max()) ||
      d.workspace_bytes > std::numeric_limits<size_t>::max())
    throw std::invalid_argument("CUB source description exceeds addon ABI");
  const auto dtype = name(p["dtype"]);
  d.signed_values = dtype == "i32";
  if (dtype != (d.plan.operation == 2 ? "u64" : "u32") &&
      !(d.plan.operation >= 5 && d.signed_values))
    throw std::invalid_argument("CUB source operation dtype is incompatible");
  d.cub_version = source_addon_u32(p, "cub_version");
  d.cuda_runtime_version = source_addon_u32(p, "cuda_runtime_version");
  if (!d.cub_version || !d.cuda_runtime_version)
    throw std::invalid_argument("CUB source frozen identity is invalid");
  d.identity = decode_source_addon_identity(p);
  return d;
}

void validate(Device &device, const CubOperation &op) {
  if (!op.plan || &op.plan->device() != &device ||
      op.plan->arch() != Arch::cuda)
    throw std::invalid_argument("CUB source requires a local CUDA plan");
  const auto &d = op.plan->description;
  std::array<bool, 4> present{true, d.plan.operation != 3, true,
                              d.plan.operation <= 2 || d.plan.operation == 4};
  std::array<NativeStorageRange, 4> ranges;
  for (size_t i = 0; i < 4; ++i) {
    if (bool(op.arrays[i]) != present[i])
      throw std::invalid_argument(
          "CUB source ports differ from selected operation");
    if (!present[i])
      continue;
    const auto &array = *op.arrays[i];
    const bool keys = d.plan.operation == 2 && (i == 0 || i == 2);
    const bool signed_values = d.signed_values && (i == 0 || i == 2);
    const auto dtype = keys            ? PrimitiveType::u64
                       : signed_values ? PrimitiveType::i32
                                       : PrimitiveType::u32;
    const uint32_t item_bytes = keys ? 8 : 4;
    const bool heads = d.plan.operation >= 5 && i == 1;
    const bool count = d.plan.operation == 4 && i == 3;
    const auto required =
        heads   ? std::max<uint64_t>(1, (d.plan.num_items + 31) / 32)
        : count ? 1
                : d.plan.num_items;
    if (array.pointer.device != &device || array.scalar_type != dtype ||
        array.shape.size() != 1 || array.shape[0] < required ||
        ((heads || count) && array.shape[0] != required) ||
        array.shape[0] > std::numeric_limits<size_t>::max() / item_bytes ||
        array.bytes != array.shape[0] * item_bytes)
      throw std::invalid_argument(
          "CUB source array dtype, shape or capacity mismatch");
    validate_native_array_layout(
        array, {array.shape, {}, item_bytes, item_bytes, true});
    ranges[i] =
        resolve_native_storage_range(Arch::cuda, array.pointer, array.bytes);
    for (size_t j = 0; j < i; ++j)
      if (i >= 2 && present[j] && !ranges[i].proven_disjoint(ranges[j]))
        throw std::invalid_argument(
            "CUB source outputs must not overlap other operands");
  }
}

#ifdef TI_WITH_CUDA
const NativeProviderDefinition definition{
    "cub_source", TI_FORGE_CUB_SOURCE_PROVIDER_QUERY_SYMBOL,
    "source addon C ABI 1"};
TiForgeCubSourceExecutionApi execution_api(
    const NativeProviderRuntime &runtime) {
  auto api = runtime.execution_api<TiForgeCubSourceExecutionApi>();
  if (!api.create_plan || !api.execute || !api.destroy_plan)
    throw NativeProviderError(TI_FORGE_RUNTIME_PROVIDER_ERROR_ABI_MISMATCH,
                              "CUB source execution API is incomplete");
  return api;
}
uint32_t version(const NativeProviderRuntime &runtime) {
  const auto &i = runtime.info();
  const auto value = uint64_t(i.version_major) * 100000 +
                     uint64_t(i.version_minor) * 100 + i.version_patch;
  if (!value || value > std::numeric_limits<uint32_t>::max())
    throw std::invalid_argument("CUB source version exceeds its ABI");
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
struct CubLease {
  explicit CubLease(std::shared_ptr<NativeProviderRuntime> r)
      : runtime(std::move(r)), api(execution_api(*runtime)) {
  }
  ~CubLease() {
    if (handle) {
      auto result = api.destroy_plan(handle);
      TI_WARN_IF(result != TI_FORGE_RUNTIME_PROVIDER_SUCCESS,
                 "CUB plan cleanup failed (result {}).", int(result));
    }
  }
  std::shared_ptr<NativeProviderRuntime> runtime;
  TiForgeCubSourceExecutionApi api;
  TiForgeCubSourcePlan handle{nullptr};
};

class CudaCubResource final : public CubResource {
 public:
  CudaCubResource(NativePreparationContext &context, CubDescription d)
      : CubResource(std::move(d)),
        device_(context.device),
        fault_(context.fault_domain),
        lease_(
            std::make_unique<CubLease>(context.providers.runtime(definition))) {
    auto &runtime = *lease_->runtime;
    if (runtime.info().build_version != "sha256:" + description.identity.binary_sha256 ||
        version(runtime) != description.cub_version ||
        runtime.info().cuda_runtime_version != description.cuda_runtime_version)
      throw std::invalid_argument(
          "CUB source binary/version differs from frozen identity");
    const auto capability =
        uint32_t(CUDAContext::get_instance().get_compute_capability());
    auto &driver = CUDADriver::get_instance_without_context();
    const auto driver_version = uint32_t(driver.get_version_major() * 1000 +
                                         driver.get_version_minor() * 10);
    validate_source_addon_target(description.identity, capability, driver_version);
    TiForgeCubSourcePlanInfo info{sizeof(info)};
    runtime.check(lease_->api.create_plan(runtime.handle(), &description.plan,
                                          &lease_->handle, &info),
                  "restore CUB source plan");
    if (!lease_->handle || info.struct_size < sizeof(info) || info.reserved ||
        info.workspace_bytes != description.workspace_bytes)
      throw std::invalid_argument(
          "CUB source workspace differs from frozen plan");
    if (info.workspace_bytes) {
      auto allocation =
          device_.allocate_memory_unique({size_t(info.workspace_bytes)});
      if (allocation.second != RhiResult::success)
        throw std::runtime_error("CUB source workspace allocation failed");
      workspace_ = std::move(allocation.first);
      workspace_address_ = reinterpret_cast<uint64_t>(
          static_cast<cuda::CudaDevice &>(device_).get_memory_addr(
              *workspace_));
    }
  }
  ~CudaCubResource() override {
    auto abandon = [&] {
      lease_->handle = nullptr;  // Live adapter count pins code after a fault.
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
  void record(const std::array<uint64_t, 4> &pointers, void *stream) {
    TiForgeCubSourceExecDesc invocation{sizeof(invocation),
                                        0,
                                        pointers[0],
                                        pointers[1],
                                        pointers[2],
                                        pointers[3],
                                        workspace_address_,
                                        description.workspace_bytes,
                                        reinterpret_cast<uint64_t>(stream)};
    lease_->runtime->check(lease_->api.execute(lease_->handle, &invocation),
                           "record CUB source operation");
  }

 private:
  Device &device_;
  std::shared_ptr<RuntimeFaultDomain> fault_;
  std::unique_ptr<CubLease> lease_;
  DeviceAllocationUnique workspace_;
  uint64_t workspace_address_{0};
};
class CudaCubOperation final : public NativeOperation {
 public:
  explicit CudaCubOperation(const CubOperation &op)
      : plan_(std::dynamic_pointer_cast<CudaCubResource>(op.plan)) {
    if (!plan_)
      throw std::invalid_argument("CUB source requires its native plan");
    for (size_t i = 0; i < pointers_.size(); ++i)
      if (op.arrays[i])
        pointers_[i] =
            resolve_native_storage_range(Arch::cuda, op.arrays[i]->pointer,
                                         op.arrays[i]->bytes)
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
  std::shared_ptr<CudaCubResource> plan_;
  std::array<uint64_t, 4> pointers_{};
};
#endif
}  // namespace

void register_cub_source_operations(NativeOperationRegistry &registry) {
  registry.define<CubDescription>([](Device &, const CubDescription &) {},
                                  NativeObjectKind::resource);
  registry.register_schema<CubDescription>(
      "cub_source", "plan", {},
      [](const liong::json::JsonValue &p)
          -> NativeOperationRegistry::Binder<CubDescription> {
        auto d = decode(p);
        return [d](const NativeOperationBindings &) { return d; };
      });
  registry.define<CubOperation>(validate);
  registry.register_schema<CubOperation>(
      "cub_source", "execute", {"plan", "input0", "output0"},
      [](const liong::json::JsonValue &p)
          -> NativeOperationRegistry::Binder<CubOperation> {
        operation_json::fields(p, {});
        return [](const NativeOperationBindings &b) {
          CubOperation op{b.at("plan").resource<CubResource>(), {}};
          const std::array<std::string, 4> ports{"input0", "input1", "output0",
                                                 "output1"};
          for (size_t i = 0; i < ports.size(); ++i)
            if (b.count(ports[i]))
              op.arrays[i] = b.at(ports[i]).array();
          return op;
        };
      },
      {}, {"input1", "output1"});
#ifdef TI_WITH_CUDA
  registry.register_context_factory<CubDescription>(
      Arch::cuda, NativeOperationMode::cuda_capture,
      [](NativePreparationContext &context, const CubDescription &d) {
        if (!dynamic_cast<cuda::CudaDevice *>(&context.device))
          throw std::invalid_argument("CUB source requires a CUDA device");
        auto &cuda = CUDAContext::get_instance();
        auto lock = cuda.get_submission_lock_guard();
        auto guard = cuda.get_guard();
        return std::make_shared<CudaCubResource>(context, d);
      },
      probe);
  registry.register_context_factory<CubOperation>(
      Arch::cuda, NativeOperationMode::cuda_capture,
      [](NativePreparationContext &, const CubOperation &op) {
        return std::make_shared<CudaCubOperation>(op);
      },
      probe);
#endif
}
}  // namespace taichi::lang
