#include "taichi/runtime/native_operation_registry.h"

#include <array>
#include <limits>
#include "taichi/common/operation_json.h"
#include "taichi/runtime/native_array_contract.h"
#ifdef TI_WITH_CUDA
#include "taichi/external_runtime/forge_cusolverdn_provider.h"
#include "taichi/program/runtime_fault.h"
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/runtime/cuda/fixed_graph.h"
#endif

namespace taichi::lang {
namespace {
struct SolverDescription {
  static constexpr const char *kind = "cusolverdn_cholesky_plan";
  static constexpr uint32_t version = 1;
  uint64_t rows{0}, rhs_count{0};
  DataType dtype;
  bool seeded{false};
  NativeArrayBinding factor, status;
};

class SolverResource : public NativeResource {
 public:
  explicit SolverResource(SolverDescription description)
      : description(std::move(description)) {
  }
  const SolverDescription description;
};

struct SolverOperation {
  static constexpr const char *kind = "cusolverdn_cholesky";
  static constexpr uint32_t version = 1;
  std::shared_ptr<SolverResource> plan;
  NativeArrayBinding a, rhs, solution;
  bool refactor{false};
};

uint64_t element_bytes(const SolverDescription &d) {
  if (d.dtype != PrimitiveType::f32 && d.dtype != PrimitiveType::f64)
    throw std::invalid_argument("cuSOLVERDn requires f32 or f64");
  return d.dtype == PrimitiveType::f64 ? 8 : 4;
}

void validate_dimensions(const SolverDescription &d) {
  const auto bytes = element_bytes(d);
  const auto limit = std::numeric_limits<std::size_t>::max() / bytes;
  if (!d.rows || !d.rhs_count ||
      d.rows > uint64_t(std::numeric_limits<int64_t>::max()) ||
      d.rhs_count > uint64_t(std::numeric_limits<int64_t>::max()) ||
      d.rows > limit / d.rows || d.rhs_count > limit / d.rows)
    throw std::invalid_argument("cuSOLVERDn dimensions overflow storage");
}

NativeStorageRange layout(Device &device,
                          const NativeArrayBinding &array,
                          DataType type,
                          std::vector<uint64_t> shape,
                          uint64_t count,
                          uint64_t bytes) {
  if (array.pointer.device != &device || array.scalar_type != type ||
      count > std::numeric_limits<std::size_t>::max() / bytes ||
      array.bytes != count * bytes)
    throw std::invalid_argument("cuSOLVERDn requires matching compact arrays");
  validate_native_array_layout(array, {shape, {}, bytes, bytes, true});
  const auto range =
      resolve_native_storage_range(Arch::cuda, array.pointer, array.bytes);
  if (range.memory_offset % bytes)
    throw std::invalid_argument("cuSOLVERDn requires scalar-aligned storage");
  return range;
}

void validate_plan(Device &device, const SolverDescription &d) {
  validate_dimensions(d);
  const auto factor = layout(device, d.factor, d.dtype, {d.rows, d.rows},
                             d.rows * d.rows, element_bytes(d));
  const auto status = layout(device, d.status, PrimitiveType::i32, {2}, 2, 4);
  if (!factor.proven_disjoint(status))
    throw std::invalid_argument("cuSOLVERDn factor and status must not alias");
}

void validate_operation(Device &device, const SolverOperation &op) {
  if (!op.plan || &op.plan->device() != &device ||
      op.plan->arch() != Arch::cuda)
    throw std::invalid_argument("cuSOLVERDn requires a local CUDA plan");
  const auto &d = op.plan->description;
  const auto shape = d.rhs_count == 1
                         ? std::vector<uint64_t>{d.rows}
                         : std::vector<uint64_t>{d.rhs_count, d.rows};
  const auto bytes = element_bytes(d);
  const std::array<NativeStorageRange, 5> ranges{
      layout(device, op.a, d.dtype, {d.rows, d.rows}, d.rows * d.rows, bytes),
      layout(device, op.rhs, d.dtype, shape, d.rows * d.rhs_count, bytes),
      layout(device, op.solution, d.dtype, shape, d.rows * d.rhs_count, bytes),
      layout(device, d.factor, d.dtype, {d.rows, d.rows}, d.rows * d.rows,
             bytes),
      layout(device, d.status, PrimitiveType::i32, {2}, 2, 4)};
  for (size_t i = 0; i < ranges.size(); ++i)
    for (size_t j = i + 1; j < ranges.size(); ++j)
      if (!ranges[i].proven_disjoint(ranges[j]) &&
          !(i == 1 && j == 2 && ranges[i].proven_same_range(ranges[j])))
        throw std::invalid_argument(
            "cuSOLVERDn only permits identical RHS/solution aliasing");
}

#ifdef TI_WITH_CUDA
const NativeProviderDefinition definition{
    "cusolverdn", TI_FORGE_CUSOLVERDN_PROVIDER_QUERY_SYMBOL,
    "generic Cholesky C API"};

TiForgeCusolverDnExecutionApi execution_api(
    const NativeProviderRuntime &runtime) {
  auto api = runtime.execution_api<TiForgeCusolverDnExecutionApi>();
  if (!api.create_plan || !api.execute || !api.destroy_plan)
    throw NativeProviderError(TI_FORGE_RUNTIME_PROVIDER_ERROR_ABI_MISMATCH,
                              "cuSOLVERDn execution API is incomplete");
  return api;
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
    const auto &info = runtime->info();
    return {true, info.version_major * 10000 + info.version_minor * 100 +
                      info.version_patch};
  } catch (const NativeProviderError &error) {
    if (error.result() == TI_FORGE_RUNTIME_PROVIDER_ERROR_RUNTIME_UNAVAILABLE ||
        error.result() == TI_FORGE_RUNTIME_PROVIDER_ERROR_RUNTIME_INCOMPATIBLE)
      return {};
    throw;
  }
}

uint64_t address(const NativeArrayBinding &array) {
  return resolve_native_storage_range(Arch::cuda, array.pointer, array.bytes)
      .memory_offset;
}

struct PlanLease {
  explicit PlanLease(std::shared_ptr<NativeProviderRuntime> runtime)
      : runtime(std::move(runtime)), api(execution_api(*this->runtime)) {
  }
  ~PlanLease() {
    if (handle) {
      const auto result = api.destroy_plan(handle);
      TI_WARN_IF(result != TI_FORGE_RUNTIME_PROVIDER_SUCCESS,
                 "cuSOLVERDn plan cleanup failed (result {}).", int(result));
    }
  }
  std::shared_ptr<NativeProviderRuntime> runtime;
  TiForgeCusolverDnExecutionApi api;
  TiForgeCusolverDnPlan handle{nullptr};
};

class CudaSolverResource final : public SolverResource,
                                 public CudaProviderCompletionResource {
 public:
  CudaSolverResource(NativePreparationContext &context, SolverDescription d)
      : SolverResource(std::move(d)),
        device_(context.device),
        fault_(context.fault_domain),
        lease_(std::make_unique<PlanLease>(
            context.providers.runtime(definition))) {
    const auto &p = description;
    TiForgeCusolverDnPlanDesc desc{
        sizeof(desc),      p.dtype == PrimitiveType::f64 ? 1u : 0u,
        int64_t(p.rows),   int64_t(p.rhs_count),
        address(p.factor), address(p.status)};
    TiForgeCusolverDnPlanInfo info{sizeof(info)};
    lease_->runtime->check(
        lease_->api.create_plan(lease_->runtime->handle(), &desc,
                                &lease_->handle, &info),
        "create Cholesky plan");
    if (!lease_->handle || info.struct_size < sizeof(info) ||
        info.device_workspace_bytes > std::numeric_limits<size_t>::max())
      throw std::runtime_error("cuSOLVERDn returned an invalid plan/workspace");
    bytes_ = size_t(info.device_workspace_bytes);
    if (bytes_) {
      auto allocation = device_.allocate_memory_unique({bytes_});
      if (allocation.second != RhiResult::success)
        throw std::runtime_error("cuSOLVERDn workspace allocation failed");
      workspace_ = std::move(allocation.first);
      workspace_address_ = reinterpret_cast<uint64_t>(
          static_cast<cuda::CudaDevice &>(device_).get_memory_addr(
              *workspace_));
    }
  }
  ~CudaSolverResource() override {
    auto abandon = [&] {
      // The adapter still counts this live plan and refuses runtime unload.
      // A faulted/finalized device must not be called just to reclaim storage.
      lease_->handle = nullptr;
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
  void record(bool refactor,
              uint64_t a,
              uint64_t rhs,
              uint64_t output,
              void *stream) {
    auto &driver = CUDADriver::get_instance();
    const auto &d = description;
    auto ptr = [](uint64_t value) {
      return reinterpret_cast<void *>(uintptr_t(value));
    };
    TiForgeCusolverDnExecDesc exec{
        sizeof(exec), 0,      workspace_address_,
        bytes_,       output, reinterpret_cast<uint64_t>(stream)};
    if (refactor) {
      driver.memsetd32_async(ptr(address(d.status) + 4), 0xffffffffu, 1,
                             stream);
      driver.memcpy_device_to_device_async(ptr(address(d.factor)), ptr(a),
                                           d.factor.bytes, stream);
      lease_->runtime->check(lease_->api.execute(lease_->handle, &exec),
                             "record Cholesky factor");
    }
    if (rhs != output)
      driver.memcpy_device_to_device_async(
          ptr(output), ptr(rhs), d.rows * d.rhs_count * element_bytes(d),
          stream);
    exec.phase = 1;
    lease_->runtime->check(lease_->api.execute(lease_->handle, &exec),
                           "record Cholesky solve");
  }

 private:
  Device &device_;
  std::shared_ptr<RuntimeFaultDomain> fault_;
  std::unique_ptr<PlanLease> lease_;
  DeviceAllocationUnique workspace_;
  uint64_t workspace_address_{0};
  size_t bytes_{0};
};

// Preserve the producer's retained root-ordered graph. This is deliberately
// not advertised as an enclosing capture or a conditional-body operation.
class CudaSolverOperation final : public NativeOperation {
 public:
  CudaSolverOperation(NativePreparationContext &context,
                      const SolverOperation &op)
      : plan_(std::dynamic_pointer_cast<CudaSolverResource>(op.plan)),
        refactor_(op.refactor) {
    if (!plan_)
      throw std::invalid_argument("cuSOLVERDn requires a native Cholesky plan");
    const auto a = address(op.a), rhs = address(op.rhs),
               output = address(op.solution);
    cuda::FixedGraph::Dispatch dispatch{};
    dispatch.external = [this, a, rhs, output](void *stream) {
      plan_->record(refactor_, a, rhs, output, stream);
    };
    graph_ = std::make_unique<cuda::FixedGraph>(
        std::vector<cuda::FixedGraph::Dispatch>{dispatch},
        context.fault_domain);
  }
  Arch arch() const override {
    return Arch::cuda;
  }
  Device &device() const override {
    return plan_->device();
  }
  size_t workspace_bytes() const override {
    return graph_->capture_bytes();
  }
  NativeOperationMode execution_mode() const override {
    return NativeOperationMode::cuda_ordered;
  }
  void validate_recording(NativeRecordingState &state) const override {
    if (refactor_)
      state.initialized.insert(plan_.get());
    else if (!plan_->description.seeded &&
             !state.initialized.count(plan_.get()))
      throw std::invalid_argument(
          "cuSOLVERDn solve requires a factor seed or a preceding factor "
          "operation");
  }
  std::shared_ptr<CudaProviderCompletionResource> cuda_ordered_resource()
      const override {
    return plan_;
  }
  void validate_cuda_execution_domain() const override {
    graph_->validate_retained_provider();
  }
  void prepare_cuda_execution_domain(
      const std::shared_ptr<RuntimeFaultDomain> &owner,
      std::vector<std::shared_ptr<RuntimeClient>> &parts) override {
    graph_->adopt_retained_provider(owner, parts);
  }
  void run_cuda_ordered(void *stream) override {
    if (stream)
      graph_->enqueue_retained_provider(stream);
    else
      graph_->run();
  }

 private:
  std::shared_ptr<CudaSolverResource> plan_;
  bool refactor_;
  std::unique_ptr<cuda::FixedGraph> graph_;
};
#endif
}  // namespace

void register_cusolverdn_operations(NativeOperationRegistry &registry) {
  registry.define<SolverDescription>(validate_plan, NativeObjectKind::resource);
  registry.register_schema<SolverDescription>(
      "cusolverdn", "cholesky_plan", {"factor", "status"},
      [](const liong::json::JsonValue &p)
          -> NativeOperationRegistry::Binder<SolverDescription> {
        using namespace operation_json;
        fields(p, {"rows", "rhs_count", "dtype", "seeded"});
        SolverDescription d;
        d.rows = integer(p["rows"]);
        d.rhs_count = integer(p["rhs_count"]);
        const auto dtype = name(p["dtype"]);
        if (dtype != "f32" && dtype != "f64")
          throw std::invalid_argument("cuSOLVERDn requires f32 or f64");
        d.dtype = dtype == "f64" ? PrimitiveType::f64 : PrimitiveType::f32;
        d.seeded = optional_bool(p, "seeded");
        validate_dimensions(d);
        return [d](const NativeOperationBindings &b) {
          auto bound = d;
          bound.factor = b.at("factor").array();
          bound.status = b.at("status").array();
          return bound;
        };
      });
  registry.define<SolverOperation>(validate_operation);
  registry.register_schema<SolverOperation>(
      "cusolverdn", "cholesky", {"plan", "a", "rhs", "solution"},
      [](const liong::json::JsonValue &p)
          -> NativeOperationRegistry::Binder<SolverOperation> {
        using namespace operation_json;
        fields(p, {"phase"});
        const auto phase = name(p["phase"]);
        if (phase != "solve" && phase != "factor_and_solve")
          throw std::invalid_argument("Invalid cuSOLVERDn numerical phase");
        return [refactor = phase == "factor_and_solve"](
                   const NativeOperationBindings &b) {
          return SolverOperation{b.at("plan").resource<SolverResource>(),
                                 b.at("a").array(), b.at("rhs").array(),
                                 b.at("solution").array(), refactor};
        };
      });
#ifdef TI_WITH_CUDA
  registry.register_context_factory<SolverDescription>(
      Arch::cuda, NativeOperationMode::cuda_ordered,
      [](NativePreparationContext &context, const SolverDescription &d) {
        if (!dynamic_cast<cuda::CudaDevice *>(&context.device))
          throw std::invalid_argument("cuSOLVERDn requires a CUDA device");
        auto &cuda = CUDAContext::get_instance();
        auto lock = cuda.get_submission_lock_guard();
        auto guard = cuda.get_guard();
        return std::make_shared<CudaSolverResource>(context, d);
      },
      probe);
  registry.register_context_factory<SolverOperation>(
      Arch::cuda, NativeOperationMode::cuda_ordered,
      [](NativePreparationContext &context, const SolverOperation &op) {
        return std::make_shared<CudaSolverOperation>(context, op);
      },
      probe);
#endif
}
}  // namespace taichi::lang
