#include "taichi/runtime/native_amgx.h"

#include <limits>
#include <tuple>
#include <unordered_map>
#include "taichi/common/operation_json.h"
#include "taichi/runtime/native_operation_registry.h"
#ifdef TI_WITH_CUDA
#include "taichi/rhi/cuda/cuda_device.h"
#endif

namespace taichi::lang {
namespace {
void validate_description(const AmgxSolverDescription &desc) {
  if (desc.rows <= 0 || desc.nonzeros <= 0 ||
      (desc.dtype != PrimitiveType::f32 && desc.dtype != PrimitiveType::f64) ||
      desc.config.empty() || desc.config.find('\0') != std::string::npos)
    throw std::invalid_argument(
        "AmgX requires positive int32 CSR dimensions, f32/f64 values and a "
        "config string");
}
void validate_owner(Device &device,
                    const std::shared_ptr<AmgxSolverResource> &solver) {
  if (!solver || &solver->device() != &device)
    throw std::invalid_argument("AmgX solver belongs to another owner");
}
void validate_array(Device &device,
                    const NativeArrayBinding &array,
                    DataType dtype,
                    uint64_t count) {
  const auto width = dtype == PrimitiveType::f64 ? 8u : 4u;
  if (array.pointer.device != &device || array.scalar_type != dtype ||
      array.shape != std::vector<uint64_t>{count} ||
      !array.element_shape.empty() ||
      count > std::numeric_limits<std::size_t>::max() / width ||
      array.bytes != count * width)
    throw std::invalid_argument("AmgX requires matching compact scalar arrays");
}
// Ordinary device bindings initialize their guess independently, even when
// several bindings share one solver. Version 1 retains solver-wide semantics.
struct BoundAmgxSolveOperation : AmgxSolveOperation {
  static constexpr uint32_t version = 2;
  std::string binding_id;
};

template <typename Operation>
void register_solve_schema(NativeOperationRegistry &registry) {
  registry.define<Operation>([](Device &device, const Operation &op) {
    validate_owner(device, op.solver);
    const auto &desc = op.solver->description();
    validate_array(device, op.rhs, desc.dtype, desc.rows);
    validate_array(device, op.solution, desc.dtype, desc.rows);
    if (op.initial_guess > 2)
      throw std::invalid_argument("Unknown AmgX initial guess policy");
    if (op.status)
      validate_array(device, *op.status, PrimitiveType::i32, 1);
    if (op.iterations)
      validate_array(device, *op.iterations, PrimitiveType::i32, 1);
    if (op.residual)
      validate_array(device, *op.residual, PrimitiveType::f64, 1);
  });
  registry.register_schema<Operation>(
      "amgx", "solve", {"solver", "rhs", "solution"},
      [](const liong::json::JsonValue &parameters)
          -> NativeOperationRegistry::Binder<Operation> {
        using namespace operation_json;
        Operation result;
        if constexpr (Operation::version == 2) {
          fields(parameters,
                 {"initial_guess", "retain_initial_guess", "binding_id"});
          result.binding_id = name(parameters.obj.inner.at("binding_id"));
        } else {
          fields(parameters, {"initial_guess", "retain_initial_guess"});
        }
        if (const auto found = parameters.obj.inner.find("initial_guess");
            found != parameters.obj.inner.end()) {
          const auto value = name(found->second);
          if (value != "upload" && value != "zero" && value != "retained")
            throw std::invalid_argument("Unknown AmgX initial guess policy");
          result.initial_guess = value == "upload" ? 0 : value == "zero" ? 1 : 2;
        }
        result.retain_initial_guess =
            optional_bool(parameters, "retain_initial_guess");
        return [description = result](const NativeOperationBindings &bindings) {
          auto result = description;
          result.solver = bindings.at("solver").resource<AmgxSolverResource>();
          result.rhs = bindings.at("rhs").array();
          result.solution = bindings.at("solution").array();
          auto optional =
              [&](const char *port) -> std::optional<NativeArrayBinding> {
            const auto found = bindings.find(port);
            return found == bindings.end()
                       ? std::nullopt
                       : std::optional(found->second.array());
          };
          result.status = optional("status");
          result.iterations = optional("iterations");
          result.residual = optional("residual");
          return result;
        };
      },
      {}, {"status", "iterations", "residual"});
}

#ifdef TI_WITH_CUDA
const NativeProviderDefinition definition{
    "amgx", TI_FORGE_AMGX_PROVIDER_QUERY_SYMBOL, "stable C API"};

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

void *address(Device &owner, const NativeArrayBinding &array) {
  const auto base = reinterpret_cast<uintptr_t>(
      static_cast<cuda::CudaDevice &>(owner).get_memory_addr(array.pointer));
  const auto width = array.scalar_type == PrimitiveType::f64 ? 8u : 4u;
  if (!base ||
      array.pointer.offset > std::numeric_limits<uintptr_t>::max() - base ||
      (base + array.pointer.offset) % width)
    throw std::invalid_argument("Invalid or misaligned AmgX array address");
  return reinterpret_cast<void *>(base + array.pointer.offset);
}

// Deferred solver: numerical setup is an explicit operation, never preparation.
class CudaSolver final : public AmgxSolverResource {
 public:
  CudaSolver(Device &device,
             AmgxSolverDescription desc,
             std::shared_ptr<NativeProviderRuntime> runtime)
      : AmgxSolverResource(std::move(desc)),
        device_(device),
        solution_fact_(device),
        runtime_(std::move(runtime)),
        api_(runtime_->execution_api<TiForgeAmgxExecutionApi>()) {
    if (!api_.create_solver || !api_.replace_coefficients || !api_.solve ||
        !api_.destroy_solver)
      throw std::invalid_argument("AmgX adapter is missing execution entries");
    if (!description_.compute_residual &&
        !(runtime_->provider_info().features &
          TI_FORGE_RUNTIME_PROVIDER_FEATURE_AMGX_OPTIONAL_RESIDUAL))
      throw std::invalid_argument("AmgX adapter cannot omit the residual pass");
  }
  ~CudaSolver() override {
    auto &cuda = CUDAContext::get_instance();
    auto lock = cuda.get_submission_lock_guard();
    auto guard = cuda.get_guard();
    destroy(handle_);
  }
  Arch arch() const override {
    return Arch::cuda;
  }
  Device &device() const override {
    return device_;
  }
  std::size_t workspace_bytes() const override {
    return 0;
  }  // Vendor-owned storage is opaque.
  bool supports_retained_guess() const {
    return runtime_->provider_info().features &
           TI_FORGE_RUNTIME_PROVIDER_FEATURE_AMGX_RETAINED_GUESS;
  }
  const NativeResource *solution_fact() const {
    return &solution_fact_;
  }
  std::shared_ptr<bool> binding_state(const std::string &name,
                                    const AmgxSolveOperation &op) {
    const BindingIdentity identity{
        address(device_, op.rhs), address(device_, op.solution), op.initial_guess,
        op.retain_initial_guess};
    auto [found, inserted] = binding_states_.try_emplace(name);
    if (inserted) {
      found->second = {identity, std::make_shared<bool>(false)};
    } else if (found->second.identity != identity) {
      throw std::invalid_argument(
          "AmgX binding identity has inconsistent ports or guess policy");
    }
    return found->second.initialized;
  }
  void setup(const int32_t *rows, const int32_t *columns, const void *values) {
    TiForgeAmgxSolverDesc desc{};
    desc.struct_size = sizeof(desc);
    desc.value_type = description_.dtype == PrimitiveType::f64
                          ? TI_FORGE_AMGX_VALUE_F64
                          : TI_FORGE_AMGX_VALUE_F32;
    desc.config_source = TI_FORGE_AMGX_CONFIG_STRING;
    desc.reserved = description_.compute_residual
                        ? 0
                        : TI_FORGE_AMGX_SOLVER_SKIP_RESIDUAL_NORM;
    desc.rows = description_.rows;
    desc.nonzeros = description_.nonzeros;
    desc.row_offsets = rows;
    desc.column_indices = columns;
    desc.values = values;
    desc.config = description_.config.c_str();
    TiForgeAmgxSolver replacement = nullptr;
    const auto result =
        api_.create_solver(runtime_->handle(), &desc, &replacement);
    if (result != TI_FORGE_RUNTIME_PROVIDER_SUCCESS) {
      destroy(replacement);
      runtime_->check(result, "create and set up CSR solver");
    }
    if (!replacement)
      throw std::runtime_error("AmgX returned a null solver");
    destroy(handle_);
    handle_ = replacement;
    has_solution_ = false;
    for (const auto &[name, state] : binding_states_)
      *state.initialized = false;
  }
  void replace(const void *values) {
    require_setup();
    runtime_->check(
        api_.replace_coefficients(handle_, values, description_.nonzeros),
        "replace CSR coefficients");
  }
  TiForgeAmgxSolveInfo solve(const void *rhs,
                             void *solution,
                             uint32_t guess,
                             bool retain) {
    require_setup();
    if (retain && has_solution_)
      guess = 2;
    if (guess == 2 && !has_solution_)
      throw std::runtime_error(
          "AmgX retained guess requires a preceding successful solve");
    TiForgeAmgxSolveDesc desc{sizeof(desc), guess, rhs, solution};
    TiForgeAmgxSolveInfo info{};
    info.struct_size = sizeof(info);
    has_solution_ = false;
    runtime_->check(api_.solve(handle_, &desc, &info), "solve CSR system");
    has_solution_ = true;
    return info;
  }

 private:
  void require_setup() const {
    if (!handle_)
      throw std::runtime_error("AmgX solver has not been initialized");
  }
  void destroy(TiForgeAmgxSolver handle) noexcept {
    if (!handle)
      return;
    const auto result = api_.destroy_solver(handle);
    TI_WARN_IF(result != TI_FORGE_RUNTIME_PROVIDER_SUCCESS,
               "AmgX solver cleanup failed (result {}).", int(result));
  }
  Device &device_;
  // Cold validity of the vendor solution is a separate initialized resource
  // fact. The common control-flow join intersects it across branches; setup
  // invalidates it, and a skipped solve cannot establish it for a later user.
  class SolutionFact final : public NativeResource {
   public:
    explicit SolutionFact(Device &device) : device_(device) {}
    Arch arch() const override { return Arch::cuda; }
    Device &device() const override { return device_; }
    std::size_t workspace_bytes() const override { return 0; }

   private:
    Device &device_;
  } solution_fact_;
  std::shared_ptr<NativeProviderRuntime> runtime_;
  TiForgeAmgxExecutionApi api_;
  TiForgeAmgxSolver handle_{nullptr};
  bool has_solution_{false};
  using BindingIdentity = std::tuple<void *, void *, uint32_t, bool>;
  struct BindingState {
    BindingIdentity identity;
    std::shared_ptr<bool> initialized;
  };
  std::unordered_map<std::string, BindingState> binding_states_;
};

class CudaAction : public NativeOperation {
 public:
  explicit CudaAction(std::shared_ptr<CudaSolver> solver)
      : solver_(std::move(solver)) {
  }
  Arch arch() const override {
    return Arch::cuda;
  }
  Device &device() const override {
    return solver_->device();
  }
  std::size_t workspace_bytes() const override {
    return 0;
  }
  NativeOperationMode execution_mode() const override {
    return NativeOperationMode::cuda_synchronous;
  }
  void validate_recording(NativeRecordingState &state) const override {
    if (!state.initialized.count(solver_.get()))
      throw std::invalid_argument(
          "AmgX operation requires a preceding setup in this sequence");
  }

 protected:
  std::shared_ptr<CudaSolver> solver_;
};

class CudaSetup final : public CudaAction {
 public:
  CudaSetup(std::shared_ptr<CudaSolver> solver, const AmgxSetupOperation &op)
      : CudaAction(std::move(solver)),
        rows_(std::size_t(solver_->description().rows) + 1),
        columns_(solver_->description().nonzeros),
        rows_source_(address(device(), op.row_offsets)),
        columns_source_(address(device(), op.column_indices)),
        values_(address(device(), op.values)) {
  }
  std::size_t workspace_bytes() const override {
    return (rows_.size() + columns_.size()) * sizeof(int32_t);
  }
  void validate_recording(NativeRecordingState &state) const override {
    state.initialized.insert(solver_.get());
    state.initialized.erase(solver_->solution_fact());
  }
  void run_cuda_synchronous() override {
    auto &driver = CUDADriver::get_instance();
    driver.memcpy_device_to_host(rows_.data(), rows_source_,
                                 rows_.size() * sizeof(int32_t));
    driver.memcpy_device_to_host(columns_.data(), columns_source_,
                                 columns_.size() * sizeof(int32_t));
    solver_->setup(rows_.data(), columns_.data(), values_);
  }

 private:
  std::vector<int32_t> rows_, columns_;
  void *rows_source_, *columns_source_, *values_;
};

class CudaReplace final : public CudaAction {
 public:
  CudaReplace(std::shared_ptr<CudaSolver> solver,
              const AmgxReplaceOperation &op)
      : CudaAction(std::move(solver)), values_(address(device(), op.values)) {
  }
  void run_cuda_synchronous() override {
    solver_->replace(values_);
  }

 private:
  void *values_;
};

class CudaSolve final : public CudaAction {
 public:
  CudaSolve(std::shared_ptr<CudaSolver> solver,
            const AmgxSolveOperation &op,
            std::shared_ptr<bool> initialized = {})
      : CudaAction(std::move(solver)),
        rhs_(address(device(), op.rhs)),
        solution_(address(device(), op.solution)),
        guess_(op.initial_guess),
        retain_(op.retain_initial_guess),
        initialized_(std::move(initialized)) {
    if ((guess_ == 2 || retain_) && !solver_->supports_retained_guess())
      throw std::invalid_argument(
          "AmgX adapter cannot retain an initial guess");
    const std::array<const std::optional<NativeArrayBinding> *, 3> info{
        &op.status, &op.iterations, &op.residual};
    std::vector<std::pair<uintptr_t, uint64_t>> bound{
        {reinterpret_cast<uintptr_t>(rhs_), op.rhs.bytes},
        {reinterpret_cast<uintptr_t>(solution_), op.solution.bytes}};
    for (std::size_t i = 0; i < info.size(); ++i) {
      if (!*info[i])
        continue;
      output_[i] = address(device(), **info[i]);
      const auto pointer = reinterpret_cast<uintptr_t>(output_[i]);
      const auto bytes = (*info[i])->bytes;
      for (const auto &[other, size] : bound)
        if (pointer <= other ? other - pointer < bytes : pointer - other < size)
          throw std::invalid_argument(
              "AmgX diagnostic outputs must not overlap other arrays");
      bound.emplace_back(pointer, bytes);
    }
  }
  void validate_recording(NativeRecordingState &state) const override {
    CudaAction::validate_recording(state);
    if (guess_ == 2 && !state.initialized.count(solver_->solution_fact()))
      throw std::invalid_argument(
          "AmgX retained guess requires a preceding solve in this sequence");
    state.initialized.insert(solver_->solution_fact());
  }
  void run_cuda_synchronous() override {
    const auto guess = initialized_ && retain_ && *initialized_ ? 2 : guess_;
    auto info = solver_->solve(rhs_, solution_, guess, !initialized_ && retain_);
    if (initialized_)
      *initialized_ = true;
    auto &driver = CUDADriver::get_instance();
    if (output_[0])
      driver.memcpy_host_to_device(output_[0], &info.solve_status,
                                   sizeof(int32_t));
    if (output_[1])
      driver.memcpy_host_to_device(output_[1], &info.iterations,
                                   sizeof(int32_t));
    if (output_[2])
      driver.memcpy_host_to_device(output_[2], &info.residual_norm,
                                   sizeof(double));
  }

 private:
  void *rhs_, *solution_;
  uint32_t guess_;
  bool retain_;
  std::shared_ptr<bool> initialized_;
  std::array<void *, 3> output_{};
};
#endif
}  // namespace

void register_amgx_operations(NativeOperationRegistry &registry) {
  registry.define<AmgxSolverDescription>(
      [](Device &, const AmgxSolverDescription &description) {
        validate_description(description);
      },
      NativeObjectKind::resource);
  registry.register_schema<AmgxSolverDescription>(
      "amgx", "solver_csr", {},
      [](const liong::json::JsonValue &parameters)
          -> NativeOperationRegistry::Binder<AmgxSolverDescription> {
        using namespace operation_json;
        fields(parameters,
               {"rows", "nonzeros", "dtype", "config", "compute_residual"});
        AmgxSolverDescription result;
        const auto rows = integer(parameters["rows"]),
                   nonzeros = integer(parameters["nonzeros"]);
        if (rows > std::numeric_limits<int32_t>::max() ||
            nonzeros > std::numeric_limits<int32_t>::max())
          throw std::invalid_argument(
              "AmgX scalar CSR ABI requires int32 dimensions");
        result.rows = static_cast<int32_t>(rows);
        result.nonzeros = static_cast<int32_t>(nonzeros);
        const auto dtype = name(parameters["dtype"]);
        if (dtype != "f32" && dtype != "f64")
          throw std::invalid_argument("AmgX values require f32 or f64");
        result.dtype = dtype == "f32" ? PrimitiveType::f32 : PrimitiveType::f64;
        result.config = name(parameters["config"]);
        result.compute_residual =
            optional_bool(parameters, "compute_residual", true);
        validate_description(result);
        return [result](const NativeOperationBindings &) { return result; };
      });
  registry.define<AmgxSetupOperation>(
      [](Device &device, const AmgxSetupOperation &op) {
        validate_owner(device, op.solver);
        const auto &desc = op.solver->description();
        validate_array(device, op.row_offsets, PrimitiveType::i32,
                       uint64_t(desc.rows) + 1);
        validate_array(device, op.column_indices, PrimitiveType::i32,
                       desc.nonzeros);
        validate_array(device, op.values, desc.dtype, desc.nonzeros);
      });
  registry.register_schema<AmgxSetupOperation>(
      "amgx", "setup_csr",
      {"solver", "row_offsets", "column_indices", "values"},
      [](const liong::json::JsonValue &parameters)
          -> NativeOperationRegistry::Binder<AmgxSetupOperation> {
        operation_json::fields(parameters, {});
        return [](const NativeOperationBindings &bindings) {
          return AmgxSetupOperation{
              bindings.at("solver").resource<AmgxSolverResource>(),
              bindings.at("row_offsets").array(),
              bindings.at("column_indices").array(),
              bindings.at("values").array()};
        };
      });
  registry.define<AmgxReplaceOperation>(
      [](Device &device, const AmgxReplaceOperation &op) {
        validate_owner(device, op.solver);
        const auto &desc = op.solver->description();
        validate_array(device, op.values, desc.dtype, desc.nonzeros);
      });
  registry.register_schema<AmgxReplaceOperation>(
      "amgx", "replace_coefficients", {"solver", "values"},
      [](const liong::json::JsonValue &parameters)
          -> NativeOperationRegistry::Binder<AmgxReplaceOperation> {
        operation_json::fields(parameters, {});
        return [](const NativeOperationBindings &bindings) {
          return AmgxReplaceOperation{
              bindings.at("solver").resource<AmgxSolverResource>(),
              bindings.at("values").array()};
        };
      });
  register_solve_schema<AmgxSolveOperation>(registry);
  register_solve_schema<BoundAmgxSolveOperation>(registry);
#ifdef TI_WITH_CUDA
  registry.register_context_factory<AmgxSolverDescription>(
      Arch::cuda, NativeOperationMode::cuda_synchronous,
      [](NativePreparationContext &context, const AmgxSolverDescription &desc) {
        if (!dynamic_cast<cuda::CudaDevice *>(&context.device))
          throw std::invalid_argument("AmgX requires a CUDA device");
        auto &cuda = CUDAContext::get_instance();
        auto lock = cuda.get_submission_lock_guard();
        auto guard = cuda.get_guard();
        return std::make_shared<CudaSolver>(
            context.device, desc, context.providers.runtime(definition));
      },
      probe);
  registry.register_context_factory<AmgxSetupOperation>(
      Arch::cuda, NativeOperationMode::cuda_synchronous,
      [](NativePreparationContext &, const AmgxSetupOperation &op) {
        auto solver = std::dynamic_pointer_cast<CudaSolver>(op.solver);
        if (!solver)
          throw std::invalid_argument("AmgX requires a native CUDA solver");
        return std::make_shared<CudaSetup>(std::move(solver), op);
      },
      probe);
  registry.register_context_factory<AmgxReplaceOperation>(
      Arch::cuda, NativeOperationMode::cuda_synchronous,
      [](NativePreparationContext &, const AmgxReplaceOperation &op) {
        auto solver = std::dynamic_pointer_cast<CudaSolver>(op.solver);
        if (!solver)
          throw std::invalid_argument("AmgX requires a native CUDA solver");
        return std::make_shared<CudaReplace>(std::move(solver), op);
      },
      probe);
  registry.register_context_factory<AmgxSolveOperation>(
      Arch::cuda, NativeOperationMode::cuda_synchronous,
      [](NativePreparationContext &, const AmgxSolveOperation &op) {
        auto solver = std::dynamic_pointer_cast<CudaSolver>(op.solver);
        if (!solver)
          throw std::invalid_argument("AmgX requires a native CUDA solver");
        return std::make_shared<CudaSolve>(std::move(solver), op);
      },
      probe);
  registry.register_context_factory<BoundAmgxSolveOperation>(
      Arch::cuda, NativeOperationMode::cuda_synchronous,
      [](NativePreparationContext &, const BoundAmgxSolveOperation &op) {
        auto solver = std::dynamic_pointer_cast<CudaSolver>(op.solver);
        if (!solver)
          throw std::invalid_argument("AmgX requires a native CUDA solver");
        auto initialized = solver->binding_state(op.binding_id, op);
        return std::make_shared<CudaSolve>(std::move(solver), op,
                                           std::move(initialized));
      },
      probe);
#endif
}
}  // namespace taichi::lang
