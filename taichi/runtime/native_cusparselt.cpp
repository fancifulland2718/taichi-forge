#include "taichi/runtime/native_cusparselt.h"

#include <limits>
#include "taichi/common/operation_json.h"
#include "taichi/runtime/native_operation_registry.h"
#ifdef TI_WITH_CUDA
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_device.h"
#endif

namespace taichi::lang {
namespace {
std::size_t matrix_bytes(int64_t rows, int64_t columns) {
  if (rows <= 0 || columns <= 0 ||
      uint64_t(rows) >
          std::numeric_limits<std::size_t>::max() / 2 / uint64_t(columns))
    throw std::invalid_argument("cuSPARSELt matrix storage size overflows");
  return std::size_t(rows) * std::size_t(columns) * 2;
}

void validate_plan(const CusparseLtPlanDescription &description) {
  matrix_bytes(description.m, description.k);
  matrix_bytes(description.n, description.k);
  matrix_bytes(description.m, description.n);
  if (description.m % 16 || description.n % 16 || description.k % 16 ||
      !description.alignment_bytes ||
      (description.alignment_bytes & (description.alignment_bytes - 1)) ||
      description.configuration.algorithm_id < -1 ||
      description.configuration.relu > 1)
    throw std::invalid_argument(
        "cuSPARSELt requires FP16 dimensions divisible by 16, power-of-two "
        "alignment and a valid configuration");
}

void validate_array(Device &device,
                    const NativeArrayBinding &array,
                    int64_t rows,
                    int64_t columns) {
  const std::vector<uint64_t> shape{uint64_t(rows), uint64_t(columns)};
  if (array.pointer.device != &device ||
      array.scalar_type != PrimitiveType::f16 || !array.element_shape.empty() ||
      array.shape != shape || array.bytes != matrix_bytes(rows, columns))
    throw std::invalid_argument(
        "cuSPARSELt requires matching compact scalar f16 matrices");
}

void validate_owner(Device &device,
                    const std::shared_ptr<CusparseLtPlanResource> &plan) {
  if (!plan || &plan->device() != &device)
    throw std::invalid_argument("cuSPARSELt plan belongs to another owner");
}

#ifdef TI_WITH_CUDA
const NativeProviderDefinition definition{
    "cusparselt", TI_FORGE_CUSPARSELT_PROVIDER_QUERY_SYMBOL, "0.8.x-0.9.x"};

NativeOperationAvailability probe(NativeProviderSession &providers) {
  if (!CUDADriver::get_instance_without_context().nvidia_extensions_available())
    return {};
  try {
    const auto &info = providers.probe(definition);
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

struct PlanLease {
  std::shared_ptr<NativeProviderRuntime> runtime;
  TiForgeCusparseLtExecutionApi api;
  TiForgeCusparseLtMatmulPlan handle{nullptr};
  explicit PlanLease(std::shared_ptr<NativeProviderRuntime> runtime)
      : runtime(std::move(runtime)),
        api(this->runtime->execution_api<TiForgeCusparseLtExecutionApi>()) {
    if (!api.create_matmul_plan || !api.compress_sparse_a ||
        !api.execute_matmul || !api.destroy_matmul_plan)
      throw NativeProviderError(TI_FORGE_RUNTIME_PROVIDER_ERROR_ABI_MISMATCH,
                                "cuSPARSELt execution API is missing entries");
  }
  ~PlanLease() {
    if (handle) {
      const auto result = api.destroy_matmul_plan(handle);
      TI_WARN_IF(result != TI_FORGE_RUNTIME_PROVIDER_SUCCESS,
                 "cuSPARSELt plan cleanup failed (result {}).", int(result));
    }
  }
};

class CudaPlan final : public CusparseLtPlanResource {
 public:
  CudaPlan(Device &device,
           CusparseLtPlanDescription description,
           std::shared_ptr<NativeProviderRuntime> runtime);
  ~CudaPlan() override {
    auto &cuda = CUDAContext::get_instance();
    auto lock = cuda.get_submission_lock_guard();
    auto guard = cuda.get_guard();
    plan_.reset();
    for (auto &buffer : buffers_)
      buffer.reset();
  }
  Arch arch() const override {
    return Arch::cuda;
  }
  Device &device() const override {
    return device_;
  }
  std::size_t workspace_bytes() const override {
    return bytes_;
  }
  void compress(uint64_t a, void *stream) {
    TiForgeCusparseLtCompressDesc desc{sizeof(desc),
                                       0,
                                       a,
                                       addresses_[0],
                                       addresses_[1],
                                       sizes_[1],
                                       reinterpret_cast<uint64_t>(stream)};
    plan_->runtime->check(plan_->api.compress_sparse_a(plan_->handle, &desc),
                          "record compression");
  }
  void multiply(const std::array<uint64_t, 3> &arrays,
                float alpha,
                float beta,
                void *stream) {
    TiForgeCusparseLtMatmulExecDesc desc{sizeof(desc),
                                         0,
                                         alpha,
                                         beta,
                                         addresses_[0],
                                         arrays[0],
                                         arrays[1],
                                         arrays[2],
                                         addresses_[2],
                                         sizes_[2],
                                         reinterpret_cast<uint64_t>(stream)};
    plan_->runtime->check(plan_->api.execute_matmul(plan_->handle, &desc),
                          "record matmul");
  }

 private:
  Device &device_;
  std::unique_ptr<PlanLease> plan_;
  std::array<DeviceAllocationUnique, 3> buffers_;
  std::array<uint64_t, 3> sizes_{}, addresses_{};
  std::size_t bytes_{0};
};

CudaPlan::CudaPlan(Device &device,
                   CusparseLtPlanDescription description,
                   std::shared_ptr<NativeProviderRuntime> runtime)
    : CusparseLtPlanResource(std::move(description)),
      device_(device),
      plan_(std::make_unique<PlanLease>(std::move(runtime))) {
  const auto &config = description_.configuration;
  TiForgeCusparseLtMatmulPlanDesc desc{
      sizeof(desc), description_.alignment_bytes, description_.m,
      description_.n, description_.k};
  TiForgeCusparseLtMatmulPlanInfo info{};
  info.struct_size = sizeof(info);
  TiForgeRuntimeProviderResult result;
  if (config.algorithm_id == -1 && !config.relu &&
      config.split_k == TI_FORGE_CUSPARSELT_ATTRIBUTE_DEFAULT &&
      config.split_k_mode == TI_FORGE_CUSPARSELT_ATTRIBUTE_DEFAULT &&
      config.split_k_buffers == TI_FORGE_CUSPARSELT_ATTRIBUTE_DEFAULT) {
    result = plan_->api.create_matmul_plan(plan_->runtime->handle(), &desc,
                                           &plan_->handle, &info);
  } else {
    const auto configured =
        plan_->runtime->execution_api<TiForgeCusparseLtConfiguredExecutionApi>(
            TI_FORGE_CUSPARSELT_CONFIG_EXECUTION_ABI_VERSION);
    if (!configured.create_matmul_plan)
      throw NativeProviderError(TI_FORGE_RUNTIME_PROVIDER_ERROR_ABI_MISMATCH,
                                "cuSPARSELt configured creation is missing");
    result = configured.create_matmul_plan(plan_->runtime->handle(), &desc,
                                           &config, &plan_->handle, &info);
  }
  plan_->runtime->check(result, "create matmul plan");
  if (!plan_->handle || info.struct_size < sizeof(info) ||
      !info.compressed_bytes)
    throw std::runtime_error(
        "cuSPARSELt returned an invalid plan or storage size");
  sizes_ = {info.compressed_bytes, info.compression_buffer_bytes,
            info.workspace_bytes};
  for (std::size_t i = 0; i < sizes_.size(); ++i) {
    if (sizes_[i] > std::numeric_limits<std::size_t>::max() - bytes_)
      throw std::runtime_error("cuSPARSELt storage size overflows");
    bytes_ += static_cast<std::size_t>(sizes_[i]);
    if (!sizes_[i])
      continue;
    auto allocation =
        device_.allocate_memory_unique({static_cast<std::size_t>(sizes_[i])});
    if (allocation.second != RhiResult::success)
      throw std::runtime_error("cuSPARSELt storage allocation failed");
    buffers_[i] = std::move(allocation.first);
    addresses_[i] = reinterpret_cast<uint64_t>(
        static_cast<cuda::CudaDevice &>(device_).get_memory_addr(*buffers_[i]));
  }
}

class CudaCompression final : public NativeOperation {
 public:
  CudaCompression(std::shared_ptr<CudaPlan> plan, uint64_t a)
      : plan_(std::move(plan)), a_(a) {
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
  void validate_recording(NativeRecordingState &state) const override {
    state.initialized.insert(plan_.get());
  }
  void record_cuda(void *stream) override {
    plan_->compress(a_, stream);
  }

 private:
  std::shared_ptr<CudaPlan> plan_;
  uint64_t a_;
};

class CudaMatmul final : public NativeOperation {
 public:
  CudaMatmul(std::shared_ptr<CudaPlan> plan,
             std::array<uint64_t, 3> arrays,
             float alpha,
             float beta)
      : plan_(std::move(plan)), arrays_(arrays), alpha_(alpha), beta_(beta) {
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
  void validate_recording(NativeRecordingState &state) const override {
    if (!state.initialized.count(plan_.get()))
      throw std::invalid_argument(
          "cuSPARSELt matmul requires a preceding compression in this "
          "sequence");
  }
  void record_cuda(void *stream) override {
    plan_->multiply(arrays_, alpha_, beta_, stream);
  }

 private:
  std::shared_ptr<CudaPlan> plan_;
  std::array<uint64_t, 3> arrays_;
  float alpha_, beta_;
};

uint64_t address(CudaPlan &plan, const NativeArrayBinding &array) {
  auto &device = static_cast<cuda::CudaDevice &>(plan.device());
  const auto base =
      reinterpret_cast<uintptr_t>(device.get_memory_addr(array.pointer));
  if (!base ||
      array.pointer.offset > std::numeric_limits<uintptr_t>::max() - base)
    throw std::invalid_argument("Invalid cuSPARSELt device address");
  const auto result = base + array.pointer.offset;
  if (result % plan.description().alignment_bytes)
    throw std::invalid_argument(
        "cuSPARSELt binding does not meet plan alignment");
  return result;
}

std::shared_ptr<NativeResource> prepare_plan(
    NativePreparationContext &context,
    const CusparseLtPlanDescription &description) {
  if (!dynamic_cast<cuda::CudaDevice *>(&context.device))
    throw std::invalid_argument("cuSPARSELt requires a CUDA device");
  auto &cuda = CUDAContext::get_instance();
  auto lock = cuda.get_submission_lock_guard();
  auto guard = cuda.get_guard();
  return std::make_shared<CudaPlan>(context.device, description,
                                    context.providers.runtime(definition));
}

std::shared_ptr<NativeResource> prepare_compression(
    NativePreparationContext &,
    const CusparseLtCompressOperation &operation) {
  auto plan = std::dynamic_pointer_cast<CudaPlan>(operation.plan);
  if (!plan)
    throw std::invalid_argument("cuSPARSELt requires a native CUDA plan");
  const auto a = address(*plan, operation.a);
  return std::make_shared<CudaCompression>(std::move(plan), a);
}

std::shared_ptr<NativeResource> prepare_matmul(
    NativePreparationContext &,
    const CusparseLtMatmulOperation &operation) {
  auto plan = std::dynamic_pointer_cast<CudaPlan>(operation.plan);
  if (!plan)
    throw std::invalid_argument("cuSPARSELt requires a native CUDA plan");
  std::array<uint64_t, 3> arrays;
  for (std::size_t i = 0; i < arrays.size(); ++i)
    arrays[i] = address(*plan, operation.arrays[i]);
  for (std::size_t i = 0; i < 2; ++i) {
    const bool overlap =
        arrays[i] <= arrays[2]
            ? arrays[2] - arrays[i] < operation.arrays[i].bytes
            : arrays[i] - arrays[2] < operation.arrays[2].bytes;
    if (overlap && (i != 1 || arrays[i] != arrays[2]))
      throw std::invalid_argument(
          "cuSPARSELt output may only alias an identical C matrix");
  }
  return std::make_shared<CudaMatmul>(std::move(plan), arrays, operation.alpha,
                                      operation.beta);
}
#endif
}  // namespace
void register_cusparselt_operations(NativeOperationRegistry &registry) {
  registry.define<CusparseLtPlanDescription>(
      [](Device &, const CusparseLtPlanDescription &description) {
        validate_plan(description);
      },
      NativeObjectKind::resource);
  registry.register_schema<CusparseLtPlanDescription>(
      "cusparselt", "matmul_plan_f16", {},
      [](const liong::json::JsonValue &parameters)
          -> NativeOperationRegistry::Binder<CusparseLtPlanDescription> {
        using namespace operation_json;
        fields(parameters,
               {"m", "n", "k", "alignment_bytes", "algorithm_id", "split_k",
                "split_k_mode", "split_k_buffers", "relu"});
        CusparseLtPlanDescription result;
        result.m = static_cast<int64_t>(integer(parameters["m"]));
        result.n = static_cast<int64_t>(integer(parameters["n"]));
        result.k = static_cast<int64_t>(integer(parameters["k"]));
        const auto alignment =
            optional_integer(parameters, "alignment_bytes").value_or(16);
        if (alignment > std::numeric_limits<uint32_t>::max())
          throw std::invalid_argument("cuSPARSELt alignment must fit uint32");
        result.alignment_bytes = static_cast<uint32_t>(alignment);
        auto option = [&](const char *key, int32_t fallback) {
          const auto found = parameters.obj.inner.find(key);
          if (found == parameters.obj.inner.end())
            return fallback;
          const auto &value = found->second;
          if (value.ty != liong::json::L_JSON_INT ||
              value.num_int < std::numeric_limits<int32_t>::min() ||
              value.num_int > std::numeric_limits<int32_t>::max())
            throw std::invalid_argument(
                "cuSPARSELt configuration must fit signed int32");
          return static_cast<int32_t>(value.num_int);
        };
        auto &config = result.configuration;
        config.algorithm_id = option("algorithm_id", config.algorithm_id);
        config.split_k = option("split_k", config.split_k);
        config.split_k_mode = option("split_k_mode", config.split_k_mode);
        config.split_k_buffers =
            option("split_k_buffers", config.split_k_buffers);
        config.relu = optional_bool(parameters, "relu");
        validate_plan(result);
        return [result](const NativeOperationBindings &) { return result; };
      });
  registry.define<CusparseLtCompressOperation>(
      [](Device &device, const CusparseLtCompressOperation &operation) {
        validate_owner(device, operation.plan);
        const auto &plan = operation.plan->description();
        validate_array(device, operation.a, plan.m, plan.k);
      });
  registry.register_schema<CusparseLtCompressOperation>(
      "cusparselt", "compress_f16", {"plan", "a"},
      [](const liong::json::JsonValue &parameters)
          -> NativeOperationRegistry::Binder<CusparseLtCompressOperation> {
        operation_json::fields(parameters, {});
        return [](const NativeOperationBindings &bindings) {
          return CusparseLtCompressOperation{
              bindings.at("plan").resource<CusparseLtPlanResource>(),
              bindings.at("a").array()};
        };
      });
  registry.define<CusparseLtMatmulOperation>(
      [](Device &device, const CusparseLtMatmulOperation &operation) {
        validate_owner(device, operation.plan);
        if (!std::isfinite(operation.alpha) || !std::isfinite(operation.beta))
          throw std::invalid_argument("cuSPARSELt scalars must be finite f32");
        const auto &plan = operation.plan->description();
        validate_array(device, operation.arrays[0], plan.n, plan.k);
        validate_array(device, operation.arrays[1], plan.m, plan.n);
        validate_array(device, operation.arrays[2], plan.m, plan.n);
      });
  registry.register_schema<CusparseLtMatmulOperation>(
      "cusparselt", "matmul_f16", {"plan", "b", "c", "d"},
      [](const liong::json::JsonValue &parameters)
          -> NativeOperationRegistry::Binder<CusparseLtMatmulOperation> {
        using namespace operation_json;
        fields(parameters, {"alpha", "beta"});
        const auto alpha = optional_f32(parameters, "alpha", 1);
        const auto beta = optional_f32(parameters, "beta", 0);
        return [alpha, beta](const NativeOperationBindings &bindings) {
          return CusparseLtMatmulOperation{
              bindings.at("plan").resource<CusparseLtPlanResource>(),
              {bindings.at("b").array(), bindings.at("c").array(),
               bindings.at("d").array()},
              alpha,
              beta};
        };
      });
#ifdef TI_WITH_CUDA
  registry.register_context_factory<CusparseLtPlanDescription>(
      Arch::cuda, NativeOperationMode::cuda_capture, prepare_plan, probe);
  registry.register_context_factory<CusparseLtCompressOperation>(
      Arch::cuda, NativeOperationMode::cuda_capture, prepare_compression,
      probe);
  registry.register_context_factory<CusparseLtMatmulOperation>(
      Arch::cuda, NativeOperationMode::cuda_capture, prepare_matmul, probe);
#endif
}
}  // namespace taichi::lang
