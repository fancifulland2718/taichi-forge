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
#endif
}  // namespace

void register_cusparse_operations(NativeOperationRegistry &registry) {
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
#endif
}
}  // namespace taichi::lang
