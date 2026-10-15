#include "taichi/runtime/native_cusparse.h"

#include <cmath>
#include <limits>
#include <unordered_set>
#include "taichi/common/operation_json.h"
#include "taichi/runtime/native_array_contract.h"
#include "taichi/runtime/native_operation_registry.h"
#ifdef TI_WITH_CUDA
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/rhi/cuda/cusparse_plan.h"
#endif

namespace taichi::lang {
namespace {
std::size_t index_bytes(DataType dtype) {
  if (dtype == PrimitiveType::i32)
    return 4;
  if (dtype == PrimitiveType::i64)
    return 8;
  throw std::invalid_argument("CSR indices require i32 or i64 storage");
}

void vector_layout(Device &device,
                   const NativeArrayBinding &array,
                   DataType dtype,
                   std::uint64_t count,
                   std::size_t bytes) {
  if (array.pointer.device != &device || array.scalar_type != dtype)
    throw std::invalid_argument("CSR array dtype or owner mismatch");
  if (count > std::numeric_limits<std::size_t>::max() / bytes ||
      array.bytes != count * bytes)
    throw std::invalid_argument(
        "CSR requires a compact vector with matching byte size");
  validate_native_array_layout(array, {{count}, {}, bytes, bytes, true});
}

std::uint64_t nonzeros(const CsrF32Description &description) {
  const auto &values = description.arrays[2];
  if (values.shape.size() != 1 ||
      values.shape[0] > std::uint64_t(std::numeric_limits<std::int64_t>::max()))
    throw std::invalid_argument(
        "CSR values require a vector with int64 extent");
  return values.shape[0];
}

void disjoint(const NativeArrayBinding &left, const NativeArrayBinding &right) {
  if (!resolve_native_storage_range(Arch::cuda, left.pointer, left.bytes)
           .proven_disjoint(resolve_native_storage_range(
               Arch::cuda, right.pointer, right.bytes)))
    throw std::invalid_argument(
        "CSR writable storage overlaps a read-only input");
}

void validate_matrix(Device &device, const CsrF32Description &description) {
  if (description.rows < 0 || description.columns < 0)
    throw std::invalid_argument("CSR dimensions must be nonnegative");
  const auto count = nonzeros(description);
  const auto &arrays = description.arrays;
  if (arrays[0].scalar_type != arrays[1].scalar_type)
    throw std::invalid_argument(
        "CSR row and column indices require matching widths");
  vector_layout(device, arrays[0], arrays[0].scalar_type,
                std::uint64_t(description.rows) + 1,
                index_bytes(arrays[0].scalar_type));
  vector_layout(device, arrays[1], arrays[1].scalar_type, count,
                index_bytes(arrays[1].scalar_type));
  vector_layout(device, arrays[2], PrimitiveType::f32, count, 4);
  // Values may change; index storage must remain immutable even when imported
  // handles refer to partially overlapping physical ranges.
  disjoint(arrays[2], arrays[0]);
  disjoint(arrays[2], arrays[1]);
}

void validate_spmv(Device &device, const CsrSpmvF32Operation &op) {
  if (!op.matrix || &op.matrix->device() != &device ||
      op.matrix->arch() != Arch::cuda || !std::isfinite(op.alpha) ||
      !std::isfinite(op.beta) || (op.algorithm != 1 && op.algorithm != 2))
    throw std::invalid_argument("Invalid CSR SpMV owner or parameters");
  const auto &description = op.matrix->description();
  vector_layout(device, op.input, PrimitiveType::f32,
                op.transpose ? description.rows : description.columns, 4);
  vector_layout(device, op.output, PrimitiveType::f32,
                op.transpose ? description.columns : description.rows, 4);
  disjoint(op.output, op.input);
  for (const auto &array : description.arrays)
    disjoint(op.output, array);
}

void validate_spmm(Device &device, const CsrSpmmF32Operation &op) {
  if (!op.matrix || &op.matrix->device() != &device ||
      op.matrix->arch() != Arch::cuda || op.rhs_count <= 0 ||
      !std::isfinite(op.alpha) || !std::isfinite(op.beta) ||
      (op.algorithm != 2 && op.algorithm != 3) ||
      (op.preprocess != CsrSpmmF32Operation::Preprocess::none &&
       op.preprocess != CsrSpmmF32Operation::Preprocess::optional &&
       op.preprocess != CsrSpmmF32Operation::Preprocess::required))
    throw std::invalid_argument("Invalid CSR SpMM owner or parameters");
  const auto &d = op.matrix->description();
  const auto rhs = std::uint64_t(op.rhs_count);
  auto dense = [&](const NativeArrayBinding &array, std::uint64_t rows) {
    if (array.pointer.device != &device ||
        array.scalar_type != PrimitiveType::f32 ||
        rows > std::numeric_limits<std::size_t>::max() / 4 / rhs ||
        array.bytes != rows * rhs * 4)
      throw std::invalid_argument(
          "CSR SpMM requires compact f32 matrices on its device");
    validate_native_array_layout(array, {{rows, rhs}, {}, 4, 4, true});
  };
  dense(op.input, d.columns);
  dense(op.output, d.rows);
  disjoint(op.output, op.input);
  for (const auto &array : d.arrays)
    disjoint(op.output, array);
}

void validate_triangular(Device &device,
                         const CsrTriangularF32Description &op) {
  if (!op.matrix || &op.matrix->device() != &device ||
      op.matrix->arch() != Arch::cuda || op.rhs_count <= 0)
    throw std::invalid_argument("Invalid CSR triangular owner or rhs_count");
  const auto &d = op.matrix->description();
  if (d.rows != d.columns)
    throw std::invalid_argument("CSR triangular matrix must be square");
  const auto rows = std::uint64_t(d.rows), rhs = std::uint64_t(op.rhs_count);
  if (rows > std::numeric_limits<std::size_t>::max() / 4 / rhs)
    throw std::invalid_argument("CSR triangular dense extent overflows");
  for (const auto *array : {&op.input, &op.output}) {
    if (array->pointer.device != &device ||
        array->scalar_type != PrimitiveType::f32 ||
        array->bytes != rows * rhs * 4)
      throw std::invalid_argument(
          "CSR triangular requires compact f32 ports on its device");
    const std::vector<std::uint64_t> shape =
        rhs == 1 ? std::vector<std::uint64_t>{rows}
                 : std::vector<std::uint64_t>{rows, rhs};
    validate_native_array_layout(*array, {shape, {}, 4, 4, true});
  }
  disjoint(op.output, op.input);
  for (const auto &array : d.arrays)
    disjoint(op.output, array);
}

template <bool Update>
void validate_triangular_action(Device &device,
                                const CsrTriangularF32Operation<Update> &op) {
  if (!op.plan || &op.plan->device() != &device ||
      op.plan->arch() != Arch::cuda)
    throw std::invalid_argument("Invalid CSR triangular plan owner");
  if (Update && !op.plan->description().mutable_values)
    throw std::invalid_argument(
        "CSR triangular update requires mutable_values");
}

#ifdef TI_WITH_CUDA
NativeOperationAvailability probe_cusparse() {
  if (!CUDADriver::get_instance_without_context().nvidia_extensions_available())
    return {};
  auto &driver = CUSPARSEDriver::get_instance();
  if (!driver.is_loaded() && !driver.load_cusparse())
    return {};
  const auto capabilities = driver.capabilities();
  const auto version = capabilities.library_version_major < 0
                           ? 0
                           : capabilities.library_version_major * 10000 +
                                 capabilities.library_version_minor * 100 +
                                 capabilities.library_version_patch;
  return {
      capabilities.scalar_spmv_available && driver.cpSetStream.available() &&
          driver.cpGetStream.available() && driver.cpSetPointerMode.available(),
      static_cast<std::uint32_t>(version)};
}

NativeOperationAvailability probe_spmm() {
  auto result = probe_cusparse();
  result.available =
      result.available &&
      CUSPARSEDriver::get_instance().capabilities().spmm_f32_available;
  return result;
}

void *address(Device &device, const NativeArrayBinding &array) {
  auto *cuda = dynamic_cast<cuda::CudaDevice *>(&device);
  if (!cuda)
    throw std::invalid_argument("CSR requires a CUDA device");
  auto *base = static_cast<char *>(cuda->get_memory_addr(array.pointer));
  return base ? base + array.pointer.offset : nullptr;
}

template <typename Index>
std::vector<Index> read_indices(Device &device,
                                const NativeArrayBinding &array) {
  std::vector<Index> values(array.shape[0]);
  if (!values.empty())
    CUDADriver::get_instance().memcpy_device_to_host(
        values.data(), address(device, array), values.size() * sizeof(Index));
  return values;
}

template <typename Offset, typename Index>
void validate_indices(const std::vector<Offset> &offsets,
                      const std::vector<Index> &columns,
                      std::int64_t column_count) {
  if (offsets.front() != 0 || offsets.back() < 0 ||
      std::uint64_t(offsets.back()) != columns.size())
    throw std::invalid_argument(
        "CSR offsets must start at zero and end at nnz");
  std::unordered_set<Index> seen;
  for (std::size_t row = 0; row + 1 < offsets.size(); ++row) {
    const auto begin = offsets[row], end = offsets[row + 1];
    if (begin < 0 || end < begin || std::uint64_t(end) > columns.size())
      throw std::invalid_argument("CSR offsets must be monotonic within nnz");
    bool unordered = false;
    for (auto i = begin; i < end; ++i) {
      const auto column = columns[i];
      if (column < 0 || column >= column_count)
        throw std::invalid_argument("CSR column index is out of bounds");
      if (i > begin) {
        if (columns[i - 1] == column)
          throw std::invalid_argument("CSR duplicate column within a row");
        unordered |= columns[i - 1] > column;
      }
    }
    // Sorted rows need no candidate set. cuSPARSE permits unsorted rows, but
    // does not guarantee correctness for duplicate entries within one row.
    if (unordered) {
      seen.clear();
      for (auto i = begin; i < end; ++i)
        if (!seen.insert(columns[i]).second)
          throw std::invalid_argument("CSR duplicate column within a row");
    }
  }
}

class CudaCsr final : public CsrF32Resource {
 public:
  CudaCsr(Device &device, CsrF32Description description)
      : CsrF32Resource(std::move(description)), device_(device) {
    // The cold resource samples topology once, regardless of how many SpMV
    // operations use it. Wait for prior producers, including nondefault
    // streams.
    CUDADriver::get_instance().context_synchronize();
    const auto &arrays = description_.arrays;
    if (arrays[0].scalar_type == PrimitiveType::i32)
      validate_indices(read_indices<std::int32_t>(device, arrays[0]),
                       read_indices<std::int32_t>(device, arrays[1]),
                       description_.columns);
    else
      validate_indices(read_indices<std::int64_t>(device, arrays[0]),
                       read_indices<std::int64_t>(device, arrays[1]),
                       description_.columns);
  }
  Arch arch() const override {
    return Arch::cuda;
  }
  Device &device() const override {
    return device_;
  }
  std::size_t workspace_bytes() const override {
    return 0;
  }

 private:
  Device &device_;
};

NativeOperationAvailability probe_triangular(bool update) {
  auto result = probe_cusparse();
  const auto caps = CUSPARSEDriver::get_instance().capabilities();
  result.available &= (caps.spsv_f32_available &&
                       (!update || caps.spsv_value_update_available)) ||
                      (caps.spsm_f32_available &&
                       (!update || caps.spsm_value_update_available));
  return result;
}

class CudaTriangularPlan final : public CsrTriangularF32Resource {
 public:
  CudaTriangularPlan(Device &device, CsrTriangularF32Description description)
      : CsrTriangularF32Resource(std::move(description)), device_(device) {
    const auto &op = description_;
    const auto caps = CUSPARSEDriver::get_instance().capabilities();
    if (op.mutable_values &&
        (op.rhs_count == 1 ? !caps.spsv_value_update_available
                           : !caps.spsm_value_update_available))
      throw std::invalid_argument(
          "The cuSPARSE provider lacks the selected triangular value-update "
          "capability");
    const auto &matrix = op.matrix->description();
    CudaCsrTriangularDescription native{
        matrix.rows,
        static_cast<std::int64_t>(nonzeros(matrix)),
        op.rhs_count,
        address(device, matrix.arrays[0]),
        address(device, matrix.arrays[1]),
        address(device, matrix.arrays[2])};
    native.index_type = matrix.arrays[0].scalar_type == PrimitiveType::i32
                            ? CUSPARSE_INDEX_32I
                            : CUSPARSE_INDEX_64I;
    native.upper = op.upper;
    native.unit_diagonal = op.unit_diagonal;
    native.transpose = op.transpose;
    plan_ = std::make_unique<CudaCsrTriangularPlan>(native);
    plan_->prepare(address(device, op.input), address(device, op.output),
                   nullptr);
    CUDADriver::get_instance().stream_synchronize(nullptr);
  }
  ~CudaTriangularPlan() override {
    auto &cuda = CUDAContext::get_instance();
    auto lock = cuda.get_submission_lock_guard();
    auto guard = cuda.get_guard();
    plan_.reset();
  }
  Arch arch() const override {
    return Arch::cuda;
  }
  Device &device() const override {
    return device_;
  }
  std::size_t workspace_bytes() const override {
    return plan_->workspace_bytes();
  }
  void record(void *stream, bool update) {
    if (update)
      plan_->update_values(static_cast<CUstream>(stream));
    else
      plan_->record(static_cast<CUstream>(stream));
  }

 private:
  Device &device_;
  std::unique_ptr<CudaCsrTriangularPlan> plan_;
};

template <bool Update>
class CudaTriangularAction final : public NativeOperation {
 public:
  explicit CudaTriangularAction(std::shared_ptr<CudaTriangularPlan> plan)
      : plan_(std::move(plan)) {
    if (!plan_)
      throw std::invalid_argument(
          "CSR triangular resource has no CUDA implementation");
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
    plan_->record(stream, Update);
  }
  void validate_recording(NativeRecordingState &state) const override {
    if constexpr (Update) {
      state.initialized.insert(plan_.get());
    } else if (plan_->description().mutable_values &&
               !state.initialized.count(plan_.get())) {
      throw std::invalid_argument(
          "Mutable CSR triangular solve requires a preceding update in this "
          "sequence");
    }
  }

 private:
  std::shared_ptr<CudaTriangularPlan> plan_;
};

class CudaSpmv final : public NativeOperation {
 public:
  CudaSpmv(Device &device, const CsrSpmvF32Operation &op)
      : device_(device), matrix_(op.matrix) {
    const auto &d = matrix_->description();
    CudaCsrSpmvDescription description{d.rows,
                                       d.columns,
                                       static_cast<std::int64_t>(nonzeros(d)),
                                       address(device, d.arrays[0]),
                                       address(device, d.arrays[1]),
                                       address(device, d.arrays[2])};
    description.row_type = d.arrays[0].scalar_type == PrimitiveType::i32
                               ? CUSPARSE_INDEX_32I
                               : CUSPARSE_INDEX_64I;
    description.column_type = d.arrays[1].scalar_type == PrimitiveType::i32
                                  ? CUSPARSE_INDEX_32I
                                  : CUSPARSE_INDEX_64I;
    description.transpose = op.transpose;
    description.algorithm =
        op.algorithm == 1 ? CUSPARSE_SPMV_CSR_ALG1 : CUSPARSE_SPMV_CSR_ALG2;
    description.alpha = op.alpha;
    description.beta = op.beta;
    plan_ = std::make_unique<CudaCsrSpmvPlan>(description);
    plan_->prepare(address(device, op.input), address(device, op.output),
                   nullptr, op.preprocess);
    // Workspace analysis must complete before recording onto another stream.
    CUDADriver::get_instance().stream_synchronize(nullptr);
  }
  ~CudaSpmv() override {
    auto &cuda = CUDAContext::get_instance();
    auto lock = cuda.get_submission_lock_guard();
    auto guard = cuda.get_guard();
    plan_.reset();
  }
  Arch arch() const override {
    return Arch::cuda;
  }
  Device &device() const override {
    return device_;
  }
  std::size_t workspace_bytes() const override {
    return plan_->workspace_bytes();
  }
  void record_cuda(void *stream) override {
    plan_->record(static_cast<CUstream>(stream));
  }

 private:
  Device &device_;
  std::shared_ptr<CsrF32Resource> matrix_;
  std::unique_ptr<CudaCsrSpmvPlan> plan_;
};

class CudaSpmm final : public NativeOperation {
 public:
  CudaSpmm(Device &device, const CsrSpmmF32Operation &op)
      : device_(device), matrix_(op.matrix) {
    const auto &d = matrix_->description();
    CudaCsrSpmmDescription description{d.rows,
                                       d.columns,
                                       static_cast<std::int64_t>(nonzeros(d)),
                                       address(device, d.arrays[0]),
                                       address(device, d.arrays[1]),
                                       address(device, d.arrays[2]),
                                       op.rhs_count};
    description.row_type = d.arrays[0].scalar_type == PrimitiveType::i32
                               ? CUSPARSE_INDEX_32I
                               : CUSPARSE_INDEX_64I;
    description.column_type = d.arrays[1].scalar_type == PrimitiveType::i32
                                  ? CUSPARSE_INDEX_32I
                                  : CUSPARSE_INDEX_64I;
    description.algorithm =
        op.algorithm == 2 ? CUSPARSE_SPMM_CSR_ALG2 : CUSPARSE_SPMM_CSR_ALG3;
    description.preprocess =
        op.preprocess == CsrSpmmF32Operation::Preprocess::none
            ? CudaSpmmPreprocess::none
        : op.preprocess == CsrSpmmF32Operation::Preprocess::required
            ? CudaSpmmPreprocess::required
            : CudaSpmmPreprocess::optional;
    description.alpha = op.alpha;
    description.beta = op.beta;
    plan_ = std::make_unique<CudaCsrSpmmPlan>(description);
    plan_->prepare(address(device, op.input), address(device, op.output),
                   nullptr);
    CUDADriver::get_instance().stream_synchronize(nullptr);
    if (op.expected_workspace_bytes &&
        *op.expected_workspace_bytes != plan_->workspace_bytes())
      throw std::invalid_argument(
          "Selected CSR SpMM workspace changed during reconstruction");
  }
  ~CudaSpmm() override {
    auto &cuda = CUDAContext::get_instance();
    auto lock = cuda.get_submission_lock_guard();
    auto guard = cuda.get_guard();
    plan_.reset();
  }
  Arch arch() const override {
    return Arch::cuda;
  }
  Device &device() const override {
    return device_;
  }
  std::size_t workspace_bytes() const override {
    return plan_->workspace_bytes();
  }
  void record_cuda(void *stream) override {
    plan_->record(static_cast<CUstream>(stream));
  }

 private:
  Device &device_;
  std::shared_ptr<CsrF32Resource> matrix_;
  std::unique_ptr<CudaCsrSpmmPlan> plan_;
};
#endif
template <bool Update>
void register_triangular_action(NativeOperationRegistry &registry) {
  using Op = CsrTriangularF32Operation<Update>;
  registry.define<Op>(validate_triangular_action<Update>);
  registry.register_schema<Op>(
      "cusparse", Update ? "triangular_update_f32" : "triangular_solve_f32",
      {"plan"},
      [](const liong::json::JsonValue &parameters)
          -> NativeOperationRegistry::Binder<Op> {
        operation_json::fields(parameters, {});
        return [](const NativeOperationBindings &bindings) {
          return Op{bindings.at("plan").resource<CsrTriangularF32Resource>()};
        };
      });
#ifdef TI_WITH_CUDA
  registry.register_factory<Op>(
      Arch::cuda, NativeOperationMode::cuda_capture,
      [](Arch, Device &, const Op &description) {
        return std::make_shared<CudaTriangularAction<Update>>(
            std::dynamic_pointer_cast<CudaTriangularPlan>(description.plan));
      },
      [] { return probe_triangular(Update); });
#endif
}
}  // namespace

void register_cusparse_operations(NativeOperationRegistry &registry) {
  registry.define<CsrTriangularF32Description>(validate_triangular,
                                               NativeObjectKind::resource);
  registry.register_schema<CsrTriangularF32Description>(
      "cusparse", "triangular_plan_f32", {"matrix", "input", "output"},
      [](const liong::json::JsonValue &parameters)
          -> NativeOperationRegistry::Binder<CsrTriangularF32Description> {
        operation_json::fields(
            parameters, {"rhs_count", "fill_mode", "unit_diagonal", "transpose",
                         "mutable_values"});
        CsrTriangularF32Description op;
        op.rhs_count = static_cast<std::int64_t>(
            operation_json::integer(parameters["rhs_count"]));
        if (op.rhs_count <= 0)
          throw std::invalid_argument(
              "CSR triangular requires positive rhs_count");
        if (const auto found = parameters.obj.inner.find("fill_mode");
            found != parameters.obj.inner.end()) {
          const auto name = operation_json::name(found->second);
          if (name != "lower" && name != "upper")
            throw std::invalid_argument(
                "CSR triangular fill_mode must be lower or upper");
          op.upper = name == "upper";
        }
        op.unit_diagonal =
            operation_json::optional_bool(parameters, "unit_diagonal");
        op.transpose = operation_json::optional_bool(parameters, "transpose");
        op.mutable_values =
            operation_json::optional_bool(parameters, "mutable_values");
        return [op](const NativeOperationBindings &bindings) {
          auto result = op;
          result.matrix = bindings.at("matrix").resource<CsrF32Resource>();
          result.input = bindings.at("input").array();
          result.output = bindings.at("output").array();
          return result;
        };
      });
  register_triangular_action<false>(registry);
  register_triangular_action<true>(registry);
#ifdef TI_WITH_CUDA
  registry.register_factory<CsrTriangularF32Description>(
      Arch::cuda, NativeOperationMode::cuda_capture,
      [](Arch, Device &device, const CsrTriangularF32Description &description) {
        auto &cuda = CUDAContext::get_instance();
        auto lock = cuda.get_submission_lock_guard();
        auto guard = cuda.get_guard();
        return std::make_shared<CudaTriangularPlan>(device, description);
      },
      [] { return probe_triangular(false); });
#endif
  registry.define<CsrF32Description>(validate_matrix,
                                     NativeObjectKind::resource);
  registry.register_schema<CsrF32Description>(
      "cusparse", "csr_f32", {"row_offsets", "column_indices", "values"},
      [](const liong::json::JsonValue &parameters)
          -> NativeOperationRegistry::Binder<CsrF32Description> {
        operation_json::fields(parameters, {"rows", "columns"});
        const auto rows = operation_json::integer(parameters["rows"]);
        const auto columns = operation_json::integer(parameters["columns"]);
        return [rows, columns](const NativeOperationBindings &bindings) {
          return CsrF32Description{static_cast<std::int64_t>(rows),
                                   static_cast<std::int64_t>(columns),
                                   {bindings.at("row_offsets").array(),
                                    bindings.at("column_indices").array(),
                                    bindings.at("values").array()}};
        };
      });
  registry.define<CsrSpmvF32Operation>(validate_spmv);
  registry.register_schema<CsrSpmvF32Operation>(
      "cusparse", "spmv_f32", {"matrix", "input", "output"},
      [](const liong::json::JsonValue &parameters)
          -> NativeOperationRegistry::Binder<CsrSpmvF32Operation> {
        operation_json::fields(parameters, {"transpose", "preprocess",
                                            "algorithm", "alpha", "beta"});
        CsrSpmvF32Operation op;
        op.transpose = operation_json::optional_bool(parameters, "transpose");
        op.preprocess =
            operation_json::optional_bool(parameters, "preprocess", true);
        op.alpha = operation_json::optional_f32(parameters, "alpha", 1);
        op.beta = operation_json::optional_f32(parameters, "beta", 0);
        if (const auto found = parameters.obj.inner.find("algorithm");
            found != parameters.obj.inner.end()) {
          const auto algorithm = operation_json::name(found->second);
          if (algorithm != "csr_alg1" && algorithm != "csr_alg2")
            throw std::invalid_argument(
                "CSR SpMV algorithm must be csr_alg1 or csr_alg2");
          op.algorithm = algorithm == "csr_alg1" ? 1 : 2;
        }
        return [op](const NativeOperationBindings &bindings) {
          auto result = op;
          result.matrix = bindings.at("matrix").resource<CsrF32Resource>();
          result.input = bindings.at("input").array();
          result.output = bindings.at("output").array();
          return result;
        };
      });
  registry.define<CsrSpmmF32Operation>(validate_spmm);
  registry.register_schema<CsrSpmmF32Operation>(
      "cusparse", "spmm_f32", {"matrix", "input", "output"},
      [](const liong::json::JsonValue &parameters)
          -> NativeOperationRegistry::Binder<CsrSpmmF32Operation> {
        operation_json::fields(parameters,
                               {"rhs_count", "algorithm", "preprocess", "alpha",
                                "beta", "expected_workspace_bytes"});
        CsrSpmmF32Operation op;
        op.expected_workspace_bytes = operation_json::optional_integer(
            parameters, "expected_workspace_bytes");
        op.rhs_count = static_cast<std::int64_t>(
            operation_json::integer(parameters["rhs_count"]));
        if (!op.rhs_count)
          throw std::invalid_argument("CSR SpMM requires positive rhs_count");
        op.alpha = operation_json::optional_f32(parameters, "alpha", 1);
        op.beta = operation_json::optional_f32(parameters, "beta", 0);
        if (const auto found = parameters.obj.inner.find("algorithm");
            found != parameters.obj.inner.end()) {
          const auto algorithm = operation_json::name(found->second);
          if (algorithm != "csr_alg2" && algorithm != "csr_alg3")
            throw std::invalid_argument(
                "CSR SpMM algorithm must be csr_alg2 or csr_alg3");
          op.algorithm = algorithm == "csr_alg2" ? 2 : 3;
        }
        if (const auto found = parameters.obj.inner.find("preprocess");
            found != parameters.obj.inner.end()) {
          const auto policy = operation_json::name(found->second);
          if (policy != "none" && policy != "optional" && policy != "required")
            throw std::invalid_argument(
                "CSR SpMM preprocess must be none, optional or required");
          op.preprocess = policy == "none"
                              ? CsrSpmmF32Operation::Preprocess::none
                          : policy == "required"
                              ? CsrSpmmF32Operation::Preprocess::required
                              : CsrSpmmF32Operation::Preprocess::optional;
        }
        return [op](const NativeOperationBindings &bindings) {
          auto result = op;
          result.matrix = bindings.at("matrix").resource<CsrF32Resource>();
          result.input = bindings.at("input").array();
          result.output = bindings.at("output").array();
          return result;
        };
      });
#ifdef TI_WITH_CUDA
  registry.register_factory<CsrF32Description>(
      Arch::cuda, NativeOperationMode::cuda_capture,
      [](Arch, Device &device, const CsrF32Description &description) {
        auto &cuda = CUDAContext::get_instance();
        auto lock = cuda.get_submission_lock_guard();
        auto guard = cuda.get_guard();
        return std::make_shared<CudaCsr>(device, description);
      },
      probe_cusparse);
  registry.register_factory<CsrSpmvF32Operation>(
      Arch::cuda, NativeOperationMode::cuda_capture,
      [](Arch, Device &device, const CsrSpmvF32Operation &description) {
        auto &cuda = CUDAContext::get_instance();
        auto lock = cuda.get_submission_lock_guard();
        auto guard = cuda.get_guard();
        return std::make_shared<CudaSpmv>(device, description);
      },
      probe_cusparse);
  registry.register_factory<CsrSpmmF32Operation>(
      Arch::cuda, NativeOperationMode::cuda_capture,
      [](Arch, Device &device, const CsrSpmmF32Operation &description) {
        auto &cuda = CUDAContext::get_instance();
        auto lock = cuda.get_submission_lock_guard();
        auto guard = cuda.get_guard();
        return std::make_shared<CudaSpmm>(device, description);
      },
      probe_spmm);
#endif
}
}  // namespace taichi::lang
