#include "taichi/runtime/native_cutensor.h"

#include <limits>
#include <set>
#include "taichi/common/operation_json.h"
#include "taichi/runtime/native_operation_registry.h"
#ifdef TI_WITH_CUDA
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_device.h"
#endif

namespace taichi::lang {
namespace {

std::size_t tensor_bytes(const CutensorTensorDescription &tensor) {
  if (tensor.shape.empty() || tensor.shape.size() != tensor.modes.size() ||
      tensor.shape.size() > std::numeric_limits<uint32_t>::max())
    throw std::invalid_argument(
        "cuTENSOR requires nonempty shapes and matching modes");
  std::size_t bytes = sizeof(float);
  std::set<int32_t> modes;
  for (std::size_t i = 0; i < tensor.shape.size(); ++i) {
    const auto extent = tensor.shape[i];
    if (extent <= 0 || static_cast<uint64_t>(extent) >
                           std::numeric_limits<std::size_t>::max() / bytes)
      throw std::invalid_argument(
          "cuTENSOR extent is zero or overflows storage size");
    bytes *= static_cast<std::size_t>(extent);
    if (!modes.insert(tensor.modes[i]).second)
      throw std::invalid_argument("cuTENSOR tensor modes must be unique");
  }
  return bytes;
}

void validate_plan(const CutensorPlanDescription &description) {
  if (!description.alignment_bytes ||
      (description.alignment_bytes & (description.alignment_bytes - 1)) ||
      description.compute_mode > TI_FORGE_CUTENSOR_COMPUTE_TF32 ||
      description.workspace_preference < TI_FORGE_CUTENSOR_WORKSPACE_MIN ||
      description.workspace_preference > TI_FORGE_CUTENSOR_WORKSPACE_MAX)
    throw std::invalid_argument(
        "Invalid cuTENSOR alignment, compute mode or workspace policy");
  std::map<int32_t, int64_t> extents;
  for (const auto &tensor : description.tensors) {
    tensor_bytes(tensor);
    for (std::size_t i = 0; i < tensor.shape.size(); ++i) {
      auto entry = extents.emplace(tensor.modes[i], tensor.shape[i]);
      if (!entry.second && entry.first->second != tensor.shape[i])
        throw std::invalid_argument(
            "cuTENSOR mode extents disagree between tensors");
    }
  }
}

void validate_contract(Device &device,
                       const CutensorContractOperation &operation) {
  if (!operation.plan || &operation.plan->device() != &device ||
      !std::isfinite(operation.alpha) || !std::isfinite(operation.beta))
    throw std::invalid_argument(
        "cuTENSOR plan owner or execution scalars are invalid");
  for (std::size_t i = 0; i < operation.arrays.size(); ++i) {
    const auto &array = operation.arrays[i];
    const auto &tensor = operation.plan->description().tensors[i];
    const std::vector<uint64_t> shape(tensor.shape.begin(), tensor.shape.end());
    if (array.pointer.device != &device ||
        array.scalar_type != PrimitiveType::f32 ||
        !array.element_shape.empty() || array.shape != shape ||
        array.bytes != tensor_bytes(tensor))
      throw std::invalid_argument(
          "cuTENSOR requires compact scalar f32 arrays matching the plan");
  }
}

#ifdef TI_WITH_CUDA
const NativeProviderDefinition cutensor_definition{
    "cutensor", TI_FORGE_CUTENSOR_PROVIDER_QUERY_SYMBOL, "2.0.x-2.7.x"};

NativeOperationAvailability probe_cutensor(NativeProviderSession &providers) {
  if (!CUDADriver::get_instance_without_context().nvidia_extensions_available())
    return {};
  try {
    const auto &info = providers.probe(cutensor_definition);
    const uint64_t version = uint64_t(info.version_major) * 10000 +
                             uint64_t(info.version_minor) * 100 +
                             info.version_patch;
    return {true, version <= std::numeric_limits<uint32_t>::max()
                      ? static_cast<uint32_t>(version)
                      : 0};
  } catch (const NativeProviderError &error) {
    if (error.result() == TI_FORGE_RUNTIME_PROVIDER_ERROR_RUNTIME_UNAVAILABLE ||
        error.result() == TI_FORGE_RUNTIME_PROVIDER_ERROR_RUNTIME_INCOMPATIBLE)
      return {};
    throw;
  }
}

struct CutensorPlanLease {
  std::shared_ptr<NativeProviderRuntime> runtime;
  TiForgeCutensorExecutionApi api;
  TiForgeCutensorContractionPlan handle{nullptr};
  explicit CutensorPlanLease(std::shared_ptr<NativeProviderRuntime> runtime)
      : runtime(std::move(runtime)),
        api(this->runtime->execution_api<TiForgeCutensorExecutionApi>()) {
    if (!api.create_contraction_plan || !api.execute_contraction ||
        !api.destroy_contraction_plan)
      throw NativeProviderError(
          TI_FORGE_RUNTIME_PROVIDER_ERROR_ABI_MISMATCH,
          "cuTENSOR execution API is missing required entries");
  }
  ~CutensorPlanLease() {
    if (handle) {
      const auto result = api.destroy_contraction_plan(handle);
      TI_WARN_IF(result != TI_FORGE_RUNTIME_PROVIDER_SUCCESS,
                 "cuTENSOR plan cleanup failed (result {}).",
                 static_cast<int>(result));
    }
  }
};

class CudaCutensorPlan final : public CutensorPlanResource {
 public:
  CudaCutensorPlan(Device &device,
                   CutensorPlanDescription description,
                   std::shared_ptr<NativeProviderRuntime> runtime)
      : CutensorPlanResource(std::move(description)),
        device_(device),
        plan_(std::make_unique<CutensorPlanLease>(std::move(runtime))) {
    std::array<std::vector<int64_t>, 4> strides;
    std::array<TiForgeCutensorTensorDesc, 4> tensors;
    for (std::size_t i = 0; i < tensors.size(); ++i) {
      const auto &tensor = description_.tensors[i];
      strides[i].resize(tensor.shape.size());
      int64_t stride = 1;
      for (auto j = tensor.shape.size(); j-- > 0;) {
        strides[i][j] = stride;
        // tensor_bytes already checked a tighter size_t / sizeof(float) bound.
        stride *= tensor.shape[j];
      }
      tensors[i] = {sizeof(TiForgeCutensorTensorDesc),
                    static_cast<uint32_t>(tensor.shape.size()),
                    tensor.shape.data(), strides[i].data(),
                    tensor.modes.data()};
    }
    TiForgeCutensorContractionPlanDesc desc{sizeof(desc),
                                            description_.compute_mode,
                                            description_.alignment_bytes,
                                            description_.workspace_preference,
                                            description_.workspace_limit_bytes,
                                            tensors[0],
                                            tensors[1],
                                            tensors[2],
                                            tensors[3]};
    TiForgeCutensorContractionPlanInfo info{};
    info.struct_size = sizeof(info);
    plan_->runtime->check(
        plan_->api.create_contraction_plan(plan_->runtime->handle(), &desc,
                                           &plan_->handle, &info),
        "create contraction plan");
    if (!plan_->handle || info.struct_size < sizeof(info) ||
        info.workspace_required_bytes > std::numeric_limits<std::size_t>::max())
      throw std::runtime_error(
          "cuTENSOR returned an invalid plan or workspace size");
    workspace_bytes_ = static_cast<std::size_t>(info.workspace_required_bytes);
    if (description_.expected_workspace_bytes &&
        workspace_bytes_ != *description_.expected_workspace_bytes)
      throw std::runtime_error("cuTENSOR restored plan workspace drifted");
    if (workspace_bytes_) {
      auto allocation = device_.allocate_memory_unique({workspace_bytes_});
      if (allocation.second != RhiResult::success)
        throw std::runtime_error("cuTENSOR workspace allocation failed");
      workspace_ = std::move(allocation.first);
      workspace_address_ = reinterpret_cast<uint64_t>(
          static_cast<cuda::CudaDevice &>(device_).get_memory_addr(
              *workspace_));
    }
  }
  ~CudaCutensorPlan() override {
    auto &cuda = CUDAContext::get_instance();
    auto lock = cuda.get_submission_lock_guard();
    auto guard = cuda.get_guard();
    plan_.reset();
    workspace_.reset();
  }
  Arch arch() const override {
    return Arch::cuda;
  }
  Device &device() const override {
    return device_;
  }
  std::size_t workspace_bytes() const override {
    return workspace_bytes_;
  }
  void record(const std::array<uint64_t, 4> &addresses,
              float alpha,
              float beta,
              void *stream) {
    TiForgeCutensorContractionExecDesc desc{sizeof(desc),
                                            0,
                                            alpha,
                                            beta,
                                            addresses[0],
                                            addresses[1],
                                            addresses[2],
                                            addresses[3],
                                            workspace_address_,
                                            workspace_bytes_,
                                            reinterpret_cast<uint64_t>(stream)};
    plan_->runtime->check(plan_->api.execute_contraction(plan_->handle, &desc),
                          "record contraction");
  }

 private:
  Device &device_;
  std::unique_ptr<CutensorPlanLease> plan_;
  DeviceAllocationUnique workspace_;
  uint64_t workspace_address_{0};
  std::size_t workspace_bytes_{0};
};

class CudaCutensorOperation final : public NativeOperation {
 public:
  CudaCutensorOperation(std::shared_ptr<CudaCutensorPlan> plan,
                        std::array<uint64_t, 4> addresses,
                        float alpha,
                        float beta)
      : plan_(std::move(plan)),
        addresses_(addresses),
        alpha_(alpha),
        beta_(beta) {
  }
  Arch arch() const override {
    return Arch::cuda;
  }
  Device &device() const override {
    return plan_->device();
  }
  std::size_t workspace_bytes() const override {
    return 0;
  }
  void record_cuda(void *stream) override {
    plan_->record(addresses_, alpha_, beta_, stream);
  }

 private:
  std::shared_ptr<CudaCutensorPlan> plan_;
  std::array<uint64_t, 4> addresses_;
  float alpha_, beta_;
};

std::shared_ptr<NativeResource> prepare_cutensor_plan(
    NativePreparationContext &context,
    const CutensorPlanDescription &description) {
  if (!dynamic_cast<cuda::CudaDevice *>(&context.device))
    throw std::invalid_argument("cuTENSOR requires a CUDA device");
  auto &cuda = CUDAContext::get_instance();
  auto lock = cuda.get_submission_lock_guard();
  auto guard = cuda.get_guard();
  auto runtime = context.providers.runtime(cutensor_definition);
  return std::make_shared<CudaCutensorPlan>(context.device, description,
                                            std::move(runtime));
}

std::shared_ptr<NativeResource> prepare_cutensor_contract(
    NativePreparationContext &context,
    const CutensorContractOperation &operation) {
  auto plan = std::dynamic_pointer_cast<CudaCutensorPlan>(operation.plan);
  auto *device = dynamic_cast<cuda::CudaDevice *>(&context.device);
  if (!plan || !device)
    throw std::invalid_argument(
        "cuTENSOR requires a native CUDA contraction plan");
  std::array<uint64_t, 4> addresses;
  for (std::size_t i = 0; i < addresses.size(); ++i) {
    const auto ptr = operation.arrays[i].pointer;
    const auto base = reinterpret_cast<uintptr_t>(device->get_memory_addr(ptr));
    if (!base || ptr.offset > std::numeric_limits<uintptr_t>::max() - base)
      throw std::invalid_argument("Invalid cuTENSOR device address");
    addresses[i] = base + ptr.offset;
    if (addresses[i] % plan->description().alignment_bytes)
      throw std::invalid_argument(
          "cuTENSOR binding does not meet the plan alignment");
  }
  const auto destination = addresses[3];
  for (std::size_t i = 0; i < 3; ++i) {
    const auto source = addresses[i];
    const bool overlap = source <= destination
                             ? destination - source < operation.arrays[i].bytes
                             : source - destination < operation.arrays[3].bytes;
    if (!overlap)
      continue;
    const auto &tensors = plan->description().tensors;
    if (i != 2 || source != destination ||
        tensors[2].shape != tensors[3].shape ||
        tensors[2].modes != tensors[3].modes)
      throw std::invalid_argument(
          "cuTENSOR output may only alias an identical C tensor");
  }
  return std::make_shared<CudaCutensorOperation>(
      std::move(plan), addresses, operation.alpha, operation.beta);
}
#endif

}  // namespace

void register_cutensor_operations(NativeOperationRegistry &registry) {
  registry.define<CutensorPlanDescription>(
      [](Device &, const CutensorPlanDescription &description) {
        validate_plan(description);
      },
      NativeObjectKind::resource);
  registry.register_schema<CutensorPlanDescription>(
      "cutensor", "contraction_plan_f32", {},
      [](const liong::json::JsonValue &parameters)
          -> NativeOperationRegistry::Binder<CutensorPlanDescription> {
        using namespace operation_json;
        fields(parameters,
               {"a", "b", "c", "d", "compute_mode", "alignment_bytes",
                "workspace_preference", "workspace_limit_bytes",
                "expected_workspace_bytes"});
        CutensorPlanDescription result;
        const std::array<const char *, 4> names{"a", "b", "c", "d"};
        for (std::size_t i = 0; i < names.size(); ++i) {
          const auto &value = parameters[names[i]];
          fields(value, {"shape", "modes"});
          if (!value["shape"].is_arr() || !value["modes"].is_arr())
            throw std::invalid_argument(
                "cuTENSOR shapes and modes must be arrays");
          for (const auto &extent : value["shape"].arr.inner)
            result.tensors[i].shape.push_back(
                static_cast<int64_t>(integer(extent)));
          for (const auto &mode : value["modes"].arr.inner) {
            if (mode.ty != liong::json::L_JSON_INT ||
                mode.num_int < std::numeric_limits<int32_t>::min() ||
                mode.num_int > std::numeric_limits<int32_t>::max())
              throw std::invalid_argument(
                  "cuTENSOR modes must fit signed int32");
            result.tensors[i].modes.push_back(
                static_cast<int32_t>(mode.num_int));
          }
        }
        const auto alignment =
            optional_integer(parameters, "alignment_bytes").value_or(256);
        if (alignment > std::numeric_limits<uint32_t>::max())
          throw std::invalid_argument("cuTENSOR alignment must fit uint32");
        result.alignment_bytes = static_cast<uint32_t>(alignment);
        result.workspace_limit_bytes =
            optional_integer(parameters, "workspace_limit_bytes").value_or(0);
        result.expected_workspace_bytes =
            optional_integer(parameters, "expected_workspace_bytes");
        auto option = [&](const char *key, const char *fallback) {
          const auto found = parameters.obj.inner.find(key);
          return found == parameters.obj.inner.end() ? std::string(fallback)
                                                     : name(found->second);
        };
        const auto compute = option("compute_mode", "f32");
        if (compute != "f32" && compute != "tf32")
          throw std::invalid_argument(
              "cuTENSOR compute_mode must be f32 or tf32");
        result.compute_mode = compute == "f32" ? TI_FORGE_CUTENSOR_COMPUTE_F32
                                               : TI_FORGE_CUTENSOR_COMPUTE_TF32;
        const auto workspace = option("workspace_preference", "default");
        if (workspace != "min" && workspace != "default" && workspace != "max")
          throw std::invalid_argument("Invalid cuTENSOR workspace preference");
        result.workspace_preference =
            workspace == "min"   ? TI_FORGE_CUTENSOR_WORKSPACE_MIN
            : workspace == "max" ? TI_FORGE_CUTENSOR_WORKSPACE_MAX
                                 : TI_FORGE_CUTENSOR_WORKSPACE_DEFAULT;
        validate_plan(result);
        return [result](const NativeOperationBindings &) { return result; };
      });
  registry.define<CutensorContractOperation>(validate_contract);
  registry.register_schema<CutensorContractOperation>(
      "cutensor", "contract_f32", {"plan", "a", "b", "c", "d"},
      [](const liong::json::JsonValue &parameters)
          -> NativeOperationRegistry::Binder<CutensorContractOperation> {
        using namespace operation_json;
        fields(parameters, {"alpha", "beta"});
        const auto alpha = optional_f32(parameters, "alpha", 1);
        const auto beta = optional_f32(parameters, "beta", 0);
        return [alpha, beta](const NativeOperationBindings &bindings) {
          return CutensorContractOperation{
              bindings.at("plan").resource<CutensorPlanResource>(),
              {bindings.at("a").array(), bindings.at("b").array(),
               bindings.at("c").array(), bindings.at("d").array()},
              alpha,
              beta};
        };
      });
#ifdef TI_WITH_CUDA
  registry.register_context_factory<CutensorPlanDescription>(
      Arch::cuda, NativeOperationMode::cuda_capture, prepare_cutensor_plan,
      probe_cutensor);
  registry.register_context_factory<CutensorContractOperation>(
      Arch::cuda, NativeOperationMode::cuda_capture, prepare_cutensor_contract,
      probe_cutensor);
#endif
}

}  // namespace taichi::lang
