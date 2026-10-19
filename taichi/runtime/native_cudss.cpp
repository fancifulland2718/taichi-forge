#include "taichi/runtime/native_cudss.h"

#include <algorithm>
#include <limits>
#include <utility>
#include "taichi/common/operation_json.h"
#include "taichi/runtime/native_array_contract.h"
#include "taichi/runtime/native_csr_contract.h"
#include "taichi/runtime/native_operation_registry.h"
#ifdef TI_WITH_CUDA
#include "taichi/cudss/forge_cudss_runtime_provider.h"
#include "taichi/program/runtime_fault.h"
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/runtime/cuda/cudss_plan.h"
#endif

namespace taichi::lang {
namespace {
void validate_policy(const CudssPlanDescription &d) {
  if (!d.rows || d.rows > uint64_t(std::numeric_limits<int64_t>::max()) ||
      !d.nonzeros || d.nonzeros > uint64_t(std::numeric_limits<int32_t>::max()))
    throw std::invalid_argument(
        "cuDSS requires positive dimensions and i32 CSR offsets");
  if ((d.matrix_type != 0 && d.matrix_type != 1 && d.matrix_type != 3) ||
      d.matrix_view < 0 || d.matrix_view > 2 ||
      (d.matrix_type == 0 && d.matrix_view != 0) ||
      (d.phase != 0 && d.phase != 4 && d.phase != 8) || d.reordering < 0 ||
      d.reordering > 3 || d.solve_algorithm < 0 || d.solve_algorithm > 1)
    throw std::invalid_argument("Invalid cuDSS preparation policy");
}

NativeStorageRange vector_layout(Device &device,
                                 const NativeArrayBinding &array,
                                 DataType type,
                                 uint64_t count) {
  if (array.pointer.device != &device || array.scalar_type != type ||
      count > std::numeric_limits<std::size_t>::max() / 4 ||
      array.bytes != count * 4)
    throw std::invalid_argument(
        "cuDSS requires compact matching vectors on its device");
  validate_native_array_layout(array, {{count}, {}, 4, 4, true});
  // Includes allocation capacity and effective address overflow/alignment.
  const auto range =
      resolve_native_storage_range(Arch::cuda, array.pointer, array.bytes);
  if (range.memory_offset % 4)
    throw std::invalid_argument("cuDSS binding address must be scalar-aligned");
  return range;
}

void validate_plan(Device &device, const CudssPlanDescription &d) {
  validate_policy(d);
  vector_layout(device, d.arrays[0], PrimitiveType::i32, d.rows + 1);
  vector_layout(device, d.arrays[1], PrimitiveType::i32, d.nonzeros);
  vector_layout(device, d.arrays[2], PrimitiveType::f32, d.nonzeros);
}

void validate_group(Device &device, const CudssSolveGroupOperation &op) {
  if (!op.plan || &op.plan->device() != &device ||
      op.plan->arch() != Arch::cuda || op.plan->transactional() ||
      op.rhs_outputs.empty())
    throw std::invalid_argument(
        "cuDSS requires a local plan and nonempty RHS group");
  const auto &d = op.plan->description();
  if (bool(d.phase) != bool(op.values))
    throw std::invalid_argument(
        "cuDSS values binding must match the plan numerical phase");
  struct Interval {
    uint64_t begin, end;
    bool write;
  };
  std::vector<Interval> intervals;
  auto add = [&](const NativeArrayBinding &array, uint64_t count, bool write) {
    const auto range = vector_layout(device, array, PrimitiveType::f32, count);
    intervals.push_back(
        {range.memory_offset, range.memory_offset + range.bytes,
         write || op.disjoint_bindings});
  };
  if (op.values)
    add(*op.values, d.nonzeros, false);
  for (const auto &[rhs, output] : op.rhs_outputs) {
    add(rhs, d.rows, false);
    add(output, d.rows, true);
  }
  std::sort(intervals.begin(), intervals.end(),
            [](const auto &a, const auto &b) { return a.begin < b.begin; });
  uint64_t read_end = 0, write_end = 0;
  for (const auto &range : intervals) {
    if (range.begin < write_end || (range.write && range.begin < read_end))
      throw std::invalid_argument(
          op.disjoint_bindings
              ? "cuDSS capture bindings must not overlap"
              : "cuDSS outputs must not alias group inputs or other outputs");
    auto &end = range.write ? write_end : read_end;
    end = std::max(end, range.end);
  }
}

void validate_transaction(Device &device,
                          const CudssRefactorSolveOperation &op) {
  if (!op.plan || &op.plan->device() != &device ||
      op.plan->arch() != Arch::cuda || !op.plan->transactional())
    throw std::invalid_argument(
        "cuDSS refactor requires a local transaction plan");
  const auto &d = op.plan->description();
  const std::array<NativeStorageRange, 3> ranges{
      vector_layout(device, op.values, PrimitiveType::f32, d.nonzeros),
      vector_layout(device, op.rhs, PrimitiveType::f32, d.rows),
      vector_layout(device, op.output, PrimitiveType::f32, d.rows)};
  for (std::size_t i = 0; i < ranges.size(); ++i)
    for (std::size_t j = i + 1; j < ranges.size(); ++j)
      if (!ranges[i].proven_disjoint(ranges[j]))
        throw std::invalid_argument(
            "cuDSS transaction bindings must not alias");
}

int option(const liong::json::JsonValue &parameters,
           const char *key,
           std::initializer_list<std::pair<const char *, int>> choices,
           int fallback = 0) {
  const auto found = parameters.obj.inner.find(key);
  if (found == parameters.obj.inner.end())
    return fallback;
  const auto value = operation_json::name(found->second);
  for (const auto &[name, number] : choices)
    if (value == name)
      return number;
  throw std::invalid_argument(std::string("Unsupported cuDSS ") + key + ": " +
                              value);
}

CudssPlanDescription decode_plan(const liong::json::JsonValue &p,
                                 bool transactional) {
  using namespace operation_json;
  if (transactional)
    fields(p, {"rows", "nonzeros", "matrix_type", "matrix_view", "reordering",
               "solve"});
  else
    fields(p, {"rows", "nonzeros", "matrix_type", "matrix_view", "phase",
               "reordering", "solve"});
  CudssPlanDescription d;
  d.rows = integer(p["rows"]);
  d.nonzeros = integer(p["nonzeros"]);
  d.matrix_type =
      option(p, "matrix_type", {{"general", 0}, {"symmetric", 1}, {"spd", 3}});
  d.matrix_view =
      option(p, "matrix_view", {{"full", 0}, {"lower", 1}, {"upper", 2}});
  if (!transactional)
    d.phase = option(
        p, "phase", {{"solve", 0}, {"factor_solve", 4}, {"refactor_solve", 8}});
  d.reordering = option(
      p, "reordering",
      {{"default", 0}, {"amd", 1}, {"nested_dissection", 2}, {"natural", 3}});
  d.solve_algorithm = option(p, "solve", {{"default", 0}, {"general", 1}});
  validate_policy(d);
  return d;
}

#ifdef TI_WITH_CUDA
NativeOperationAvailability probe_cudss_mode(NativeProviderSession &providers,
                                             bool capture) {
  auto &driver = CUDADriver::get_instance_without_context();
  if (!driver.nvidia_extensions_available() || driver.get_version_major() < 12)
    return {};
  if (capture && (!driver.malloc_async_impl.available() ||
                  !driver.mem_free_async_impl.available() ||
                  !driver.stream_get_capture_info_v2.available() ||
                  !driver.graph_node_get_type.available() ||
                  !driver.graph_memcpy_node_get_params.available() ||
                  !driver.graph_memcpy_node_set_params.available()))
    return {};
  try {
    // One retained DLL runtime serves the cold probe and subsequent factories.
    // No analysis, factorization, solver handle or device allocation here.
    auto runtime = providers.runtime(cuda_cudss_provider_definition());
    const auto api = runtime->execution_api<TiForgeCudssExecutionApi>();
    if (!api.configuration.struct_size ||
        (capture && !api.allocator.struct_size))
      return {};
    if (api.configuration.struct_size < sizeof(api.configuration) ||
        api.configuration.abi_version !=
            TI_FORGE_CUDSS_CONFIGURATION_ABI_VERSION ||
        (capture && (api.allocator.struct_size < sizeof(api.allocator) ||
                     api.allocator.abi_version != 1)))
      throw NativeProviderError(TI_FORGE_RUNTIME_PROVIDER_ERROR_ABI_MISMATCH,
                                "cuDSS capture extension ABI is incompatible");
    if (!api.configuration.configure ||
        !api.configuration.analysis_memory_estimates ||
        (capture && !api.allocator.set_allocator))
      return {};
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

NativeOperationAvailability probe_cudss(NativeProviderSession &providers) {
  return probe_cudss_mode(providers, true);
}

NativeOperationAvailability probe_transaction(
    NativeProviderSession &providers) {
  return probe_cudss_mode(providers, false);
}

void *address(const NativeArrayBinding &array) {
  return reinterpret_cast<void *>(static_cast<uintptr_t>(
      resolve_native_storage_range(Arch::cuda, array.pointer, array.bytes)
          .memory_offset));
}

// Ordinary plans borrow their CSR pointers. A reconstructed resource owns an
// immutable private copy until the shared solver and all submissions retire.
struct TransactionSeed {
  std::shared_ptr<RuntimeFaultDomain> fault;
  std::array<void *, 3> buffers{};
  std::size_t bytes{0};
  explicit TransactionSeed(std::shared_ptr<RuntimeFaultDomain> fault)
      : fault(std::move(fault)) {
  }
  void copy(const CudssPlanDescription &d) {
    auto &driver = CUDADriver::get_instance();
    try {
      for (std::size_t i = 0; i < buffers.size(); ++i) {
        driver.malloc(&buffers[i], d.arrays[i].bytes);
        bytes += d.arrays[i].bytes;
        driver.memcpy_device_to_device(buffers[i], address(d.arrays[i]),
                                       d.arrays[i].bytes);
      }
    } catch (const BackendRuntimeError &error) {
      fault->report_backend_error(error, 0);
      throw;
    }
  }
  ~TransactionSeed() {
    if (!fault->backend_calls_safe() ||
        fault->state() == RuntimeLifecycleState::kFinalized)
      return;
    try {
      auto &cuda = CUDAContext::get_instance();
      auto lock = cuda.get_submission_lock_guard();
      auto guard = cuda.get_guard();
      for (auto *buffer : buffers)
        if (buffer)
          CUDADriver::get_instance().mem_free(buffer);
    } catch (const BackendRuntimeError &error) {
      fault->report_backend_error(error, 0);
    } catch (...) {
    }
  }
};

class CudaCudssResource final : public CudssPlanResource {
  // Declaration order keeps the input storage alive through solver destruction.
  std::shared_ptr<TransactionSeed> seed_;

 public:
  CudaCudssResource(NativePreparationContext &context,
                    CudssPlanDescription description,
                    bool transactional = false)
      : CudssPlanResource(std::move(description), transactional),
        device_(context.device) {
    const auto &d = description_;
    auto &driver = CUDADriver::get_instance();
    // Sample topology once at preparation, after any producer on other streams.
    driver.context_synchronize();
    std::vector<int32_t> rows(d.rows + 1), columns(d.nonzeros);
    driver.memcpy_device_to_host(rows.data(), address(d.arrays[0]),
                                 rows.size() * 4);
    driver.memcpy_device_to_host(columns.data(), address(d.arrays[1]),
                                 columns.size() * 4);
    validate_native_csr_indices(rows, columns, d.rows);
    CudaCudssMatrixView matrix{std::size_t(d.rows),     std::size_t(d.rows),
                               std::size_t(d.nonzeros), address(d.arrays[0]),
                               address(d.arrays[1]),    address(d.arrays[2])};
    if (transactional) {
      seed_ = std::make_shared<TransactionSeed>(context.fault_domain);
      seed_->copy(d);
      matrix.row_offsets = seed_->buffers[0];
      matrix.column_indices = seed_->buffers[1];
      matrix.values = seed_->buffers[2];
      // Ordinary shared plans require a live matrix identity. This immutable
      // seed's identity is private to its owner, never persisted in an artifact.
      matrix.matrix_id = reinterpret_cast<std::uintptr_t>(seed_.get());
      matrix.pattern_version = 1;
    }
    plan = make_cuda_cudss_plan(
        matrix, d.matrix_type, d.matrix_view,
        context.providers.runtime(cuda_cudss_provider_definition()),
        context.fault_domain, {d.reordering, d.solve_algorithm},
        !transactional);
    if (transactional) {
      try {
        plan->analyze(matrix);
        driver.stream_synchronize(nullptr);
      } catch (const BackendRuntimeError &error) {
        context.fault_domain->report_backend_error(error, 0);
        throw;
      }
      workspace_bytes_ = seed_->bytes;  // Vendor allocations remain opaque.
      return;
    }
    plan->claim_graph_phase(d.phase);
    // Includes the private seed snapshot. Capture parameter storage is
    // accounted separately by FixedGraph; the vendor's physical pool residency
    // is opaque.
    const auto live =
        plan->configuration().at("allocator_live_requested_bytes");
    if (live < 0 || uint64_t(live) > std::numeric_limits<std::size_t>::max())
      throw std::invalid_argument(
          "cuDSS requested storage accounting overflows");
    workspace_bytes_ = std::size_t(live);
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
  std::shared_ptr<CudaCudssPlan> plan;

 private:
  Device &device_;
  std::size_t workspace_bytes_{0};
};

class CudaCudssGroup final : public NativeOperation {
 public:
  CudaCudssGroup(std::shared_ptr<CudaCudssResource> plan,
                 const CudssSolveGroupOperation &op)
      : plan_(std::move(plan)) {
    if (!plan_)
      throw std::invalid_argument("cuDSS requires a native CUDA plan");
    values_ = op.values ? address(*op.values) : nullptr;
    for (const auto &[rhs, output] : op.rhs_outputs)
      pairs_.emplace_back(address(rhs), address(output));
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
  // The resource's cold constructor establishes valid seed factors and freezes
  // one numerical policy for its lifetime. A solve-only group preserves them;
  // an update group performs the entire update before every RHS solve. Neither
  // needs a fact produced by a previous branch or iteration. Zero-trip bodies
  // leave the ready resource intact, so conditional composition needs no opaque
  // provider-state merge (unlike a separately recorded analyze/factor/solve).
  void record_cuda(void *stream) override {
    plan_->plan->record_capture(plan_->description().phase, values_, pairs_,
                                stream, resources_);
  }
  std::shared_ptr<CudaGraphCaptureResources> take_cuda_capture_resources()
      override {
    return std::exchange(resources_, {});
  }

 private:
  std::shared_ptr<CudaCudssResource> plan_;
  void *values_{nullptr};
  std::vector<std::pair<void *, void *>> pairs_;
  std::shared_ptr<CudaGraphCaptureResources> resources_;
};

class CudaCudssTransaction final : public NativeOperation {
 public:
  CudaCudssTransaction(std::shared_ptr<CudaCudssResource> resource,
                       const CudssRefactorSolveOperation &op)
      : resource_(std::move(resource)),
        values_(address(op.values)),
        rhs_(address(op.rhs)),
        output_(address(op.output)) {
    if (!resource_ || !resource_->transactional())
      throw std::invalid_argument("cuDSS requires a native transaction owner");
  }
  Arch arch() const override {
    return Arch::cuda;
  }
  Device &device() const override {
    return resource_->device();
  }
  std::size_t workspace_bytes() const override {
    return 0;
  }
  NativeOperationMode execution_mode() const override {
    return NativeOperationMode::cuda_ordered;
  }
  void validate_recording(NativeRecordingState &state) const override {
    auto &used = state.provider_state<bool>(resource_.get());
    if (used)
      throw std::invalid_argument(
          "cuDSS transaction owner permits one in-flight invocation; use "
          "separate owners or a captured RHS group");
    used = true;
  }
  std::shared_ptr<CudaProviderCompletionResource> cuda_ordered_resource()
      const override {
    return resource_->plan;
  }
  void reserve_cuda_ordered() override {
    resource_->plan->reserve_refactor_solve();
  }
  void cancel_cuda_ordered() noexcept override {
    resource_->plan->cancel_unsubmitted_refactor_solve();
  }
  void validate_cuda_execution_domain() const override {
    throw std::invalid_argument(
        "Ordinary cuDSS transactions require a domain-bound provider plan");
  }
  void run_cuda_ordered(void *stream) override {
    if (stream)
      throw std::invalid_argument(
          "Ordinary cuDSS transactions use the runtime stream");
    resource_->plan->execute_reserved_refactor_solve(values_, rhs_, output_);
  }

 private:
  std::shared_ptr<CudaCudssResource> resource_;
  void *values_, *rhs_, *output_;
};

std::shared_ptr<NativeResource> prepare_plan(NativePreparationContext &context,
                                             const CudssPlanDescription &d,
                                             bool transactional) {
  if (!context.fault_domain ||
      !dynamic_cast<cuda::CudaDevice *>(&context.device))
    throw std::invalid_argument(
        "cuDSS preparation requires its CUDA owner's fault domain");
  auto &cuda = CUDAContext::get_instance();
  auto lock = cuda.get_submission_lock_guard();
  auto guard = cuda.get_guard();
  try {
    return std::make_shared<CudaCudssResource>(context, d, transactional);
  } catch (const BackendRuntimeError &error) {
    context.fault_domain->report_backend_error(error, 0);
    throw;
  }
}
#endif
}  // namespace

void register_cudss_operations(NativeOperationRegistry &registry) {
  registry.define<CudssPlanDescription>(validate_plan,
                                        NativeObjectKind::resource);
  registry.register_schema<CudssPlanDescription>(
      "cudss", "plan_f32", {"row_offsets", "column_indices", "values"},
      [](const liong::json::JsonValue &p)
          -> NativeOperationRegistry::Binder<CudssPlanDescription> {
        const auto d = decode_plan(p, false);
        return [d](const NativeOperationBindings &b) {
          auto bound = d;
          bound.arrays = {b.at("row_offsets").array(),
                          b.at("column_indices").array(),
                          b.at("values").array()};
          return bound;
        };
      });
  registry.define<CudssTransactionPlanDescription>(validate_plan,
                                                   NativeObjectKind::resource);
  registry.register_schema<CudssTransactionPlanDescription>(
      "cudss", "transaction_plan_f32",
      {"row_offsets", "column_indices", "values"},
      [](const liong::json::JsonValue &p)
          -> NativeOperationRegistry::Binder<CudssTransactionPlanDescription> {
        const auto d = decode_plan(p, true);
        return [d](const NativeOperationBindings &b) {
          CudssTransactionPlanDescription bound;
          static_cast<CudssPlanDescription &>(bound) = d;
          bound.arrays = {b.at("row_offsets").array(),
                          b.at("column_indices").array(),
                          b.at("values").array()};
          return bound;
        };
      });
  registry.define<CudssRefactorSolveOperation>(validate_transaction);
  registry.register_schema<CudssRefactorSolveOperation>(
      "cudss", "refactor_solve_f32", {"plan", "values", "rhs", "output"},
      [](const liong::json::JsonValue &p)
          -> NativeOperationRegistry::Binder<CudssRefactorSolveOperation> {
        operation_json::fields(p, {});
        return [](const NativeOperationBindings &b) {
          return CudssRefactorSolveOperation{
              b.at("plan").resource<CudssPlanResource>(),
              b.at("values").array(), b.at("rhs").array(),
              b.at("output").array()};
        };
      });
  registry.define<CudssSolveGroupOperation>(validate_group);
  registry.register_schema<CudssSolveGroupOperation>(
      "cudss", "solve_group_f32", {"plan"},
      [](const liong::json::JsonValue &p)
          -> NativeOperationRegistry::Binder<CudssSolveGroupOperation> {
        operation_json::fields(p, {"rhs_count", "disjoint_bindings"});
        const auto count = operation_json::integer(p["rhs_count"]);
        const bool disjoint =
            operation_json::optional_bool(p, "disjoint_bindings");
        if (!count)
          throw std::invalid_argument("cuDSS requires a nonempty RHS group");
        return [count, disjoint](const NativeOperationBindings &b) {
          CudssSolveGroupOperation op;
          op.disjoint_bindings = disjoint;
          op.plan = b.at("plan").resource<CudssPlanResource>();
          if (const auto values = b.find("values"); values != b.end())
            op.values = values->second.array();
          for (uint64_t i = 0; i < count; ++i)
            op.rhs_outputs.emplace_back(
                b.at("rhs_" + std::to_string(i)).array(),
                b.at("output_" + std::to_string(i)).array());
          return op;
        };
      },
      {}, {"values"},
      {NativeOperationRegistry::BindingGroup{"rhs_count", {"rhs", "output"}}});
#ifdef TI_WITH_CUDA
  registry.register_context_factory<CudssPlanDescription>(
      Arch::cuda, NativeOperationMode::cuda_capture,
      [](NativePreparationContext &context, const CudssPlanDescription &d) {
        return prepare_plan(context, d, false);
      },
      probe_cudss);
  registry.register_context_factory<CudssTransactionPlanDescription>(
      Arch::cuda, NativeOperationMode::cuda_ordered,
      [](NativePreparationContext &context,
         const CudssTransactionPlanDescription &d) {
        return prepare_plan(context, d, true);
      },
      probe_transaction);
  registry.register_context_factory<CudssRefactorSolveOperation>(
      Arch::cuda, NativeOperationMode::cuda_ordered,
      [](NativePreparationContext &, const CudssRefactorSolveOperation &op) {
        return std::make_shared<CudaCudssTransaction>(
            std::dynamic_pointer_cast<CudaCudssResource>(op.plan), op);
      },
      probe_transaction);
  registry.register_context_factory<CudssSolveGroupOperation>(
      Arch::cuda, NativeOperationMode::cuda_capture,
      [](NativePreparationContext &, const CudssSolveGroupOperation &op) {
        return std::make_shared<CudaCudssGroup>(
            std::dynamic_pointer_cast<CudaCudssResource>(op.plan), op);
      },
      probe_cudss);
#endif
}
}  // namespace taichi::lang
