#include "taichi/runtime/native_cusparse_bsr.h"

#include <cmath>
#include <limits>
#include "taichi/common/operation_json.h"
#include "taichi/runtime/native_array_contract.h"
#include "taichi/runtime/native_csr_contract.h"
#include "taichi/runtime/native_operation_registry.h"
#ifdef TI_WITH_CUDA
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/rhi/cuda/cusparse_plan.h"
#endif

namespace taichi::lang {
namespace {
std::uint64_t product(std::uint64_t left, std::uint64_t right) {
  const auto limit = std::uint64_t(std::numeric_limits<std::int64_t>::max());
  if (right && left > limit / right)
    throw std::invalid_argument("BSR extent exceeds the vendor int64 contract");
  return left * right;
}

void geometry(const BsrF32Description &d) {
  if (d.block_rows < 0 || d.block_columns < 0 || d.block_size <= 0)
    throw std::invalid_argument(
        "BSR requires nonnegative block counts and a positive block size");
  product(d.block_rows, d.block_size);
  product(d.block_columns, d.block_size);
  product(d.block_size, d.block_size);
}

std::uint64_t blocks(const BsrF32Description &d) {
  if (d.arrays[1].shape.size() != 1 ||
      d.arrays[1].shape[0] >
          std::uint64_t(std::numeric_limits<std::int64_t>::max()))
    throw std::invalid_argument(
        "BSR block indices require a vector with int64 extent");
  return d.arrays[1].shape[0];
}

std::size_t index_bytes(DataType dtype) {
  if (dtype == PrimitiveType::i32)
    return 4;
  if (dtype == PrimitiveType::i64)
    return 8;
  throw std::invalid_argument("BSR indices require i32 or i64 storage");
}

void vector(Device &device,
            const NativeArrayBinding &array,
            DataType dtype,
            std::uint64_t count,
            std::size_t bytes) {
  if (array.pointer.device != &device || array.scalar_type != dtype ||
      count > std::numeric_limits<std::size_t>::max() / bytes ||
      array.bytes != count * bytes)
    throw std::invalid_argument(
        "BSR array requires its device, scalar type and exact byte extent");
  validate_native_array_layout(array, {{count}, {}, bytes, bytes, true});
}

void disjoint(const NativeArrayBinding &left, const NativeArrayBinding &right) {
  if (!resolve_native_storage_range(Arch::cuda, left.pointer, left.bytes)
           .proven_disjoint(resolve_native_storage_range(
               Arch::cuda, right.pointer, right.bytes)))
    throw std::invalid_argument(
        "BSR writable storage overlaps a read-only input");
}

void validate_matrix(Device &device, const BsrF32Description &d) {
  geometry(d);
  const auto count = blocks(d);
  const auto &a = d.arrays;
  if (a[0].scalar_type != a[1].scalar_type)
    throw std::invalid_argument("BSR row and column index widths must match");
  vector(device, a[0], a[0].scalar_type, std::uint64_t(d.block_rows) + 1,
         index_bytes(a[0].scalar_type));
  vector(device, a[1], a[1].scalar_type, count, index_bytes(a[1].scalar_type));
  vector(device, a[2], PrimitiveType::f32,
         product(count, product(d.block_size, d.block_size)), 4);
  disjoint(a[2], a[0]);
  disjoint(a[2], a[1]);
}

void validate_spmv(Device &device, const BsrSpmvF32Operation &op) {
  if (!op.matrix || &op.matrix->device() != &device ||
      op.matrix->arch() != Arch::cuda || !std::isfinite(op.alpha) ||
      !std::isfinite(op.beta))
    throw std::invalid_argument("Invalid BSR SpMV owner or parameters");
  const auto &d = op.matrix->description();
  vector(device, op.input, PrimitiveType::f32,
         product(d.block_columns, d.block_size), 4);
  vector(device, op.output, PrimitiveType::f32,
         product(d.block_rows, d.block_size), 4);
  disjoint(op.output, op.input);
  for (const auto &array : d.arrays)
    disjoint(op.output, array);
}

#ifdef TI_WITH_CUDA
NativeOperationAvailability probe_bsr() {
  if (!CUDADriver::get_instance_without_context().nvidia_extensions_available())
    return {};
  auto &driver = CUSPARSEDriver::get_instance();
  if (!driver.is_loaded() && !driver.load_cusparse())
    return {};
  const auto caps = driver.capabilities();
  const auto version = caps.library_version_major < 0
                           ? 0
                           : caps.library_version_major * 10000 +
                                 caps.library_version_minor * 100 +
                                 caps.library_version_patch;
  return {caps.generic_bsr_spmv_available && driver.cpSetStream.available() &&
              driver.cpGetStream.available() &&
              driver.cpSetPointerMode.available(),
          static_cast<std::uint32_t>(version)};
}

void *address(Device &device, const NativeArrayBinding &array) {
  auto *cuda = dynamic_cast<cuda::CudaDevice *>(&device);
  if (!cuda)
    throw std::invalid_argument("BSR requires a CUDA device");
  auto *base = static_cast<char *>(cuda->get_memory_addr(array.pointer));
  return base ? base + array.pointer.offset : nullptr;
}

template <typename Index>
std::vector<Index> read_indices(Device &device,
                                const NativeArrayBinding &array) {
  std::vector<Index> host(array.shape[0]);
  if (!host.empty())
    CUDADriver::get_instance().memcpy_device_to_host(
        host.data(), address(device, array), host.size() * sizeof(Index));
  return host;
}

class CudaBsr final : public BsrF32Resource {
 public:
  CudaBsr(Device &device, BsrF32Description d)
      : BsrF32Resource(std::move(d)), device_(device) {
    // Freeze/validate topology once per resource, including nondefault
    // producers.
    CUDADriver::get_instance().context_synchronize();
    const auto &a = description_.arrays;
    if (a[0].scalar_type == PrimitiveType::i32)
      validate_native_csr_indices(read_indices<std::int32_t>(device, a[0]),
                                  read_indices<std::int32_t>(device, a[1]),
                                  description_.block_columns);
    else
      validate_native_csr_indices(read_indices<std::int64_t>(device, a[0]),
                                  read_indices<std::int64_t>(device, a[1]),
                                  description_.block_columns);
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

class CudaBsrSpmv final : public NativeOperation {
 public:
  CudaBsrSpmv(Device &device, const BsrSpmvF32Operation &op)
      : device_(device), matrix_(op.matrix) {
    const auto &d = matrix_->description();
    CudaSparseSpmvDescription description{
        static_cast<std::int64_t>(product(d.block_rows, d.block_size)),
        static_cast<std::int64_t>(product(d.block_columns, d.block_size)),
        static_cast<std::int64_t>(blocks(d)),
        address(device, d.arrays[0]),
        address(device, d.arrays[1]),
        address(device, d.arrays[2])};
    description.row_type = description.column_type =
        d.arrays[0].scalar_type == PrimitiveType::i32 ? CUSPARSE_INDEX_32I
                                                      : CUSPARSE_INDEX_64I;
    description.block_size = d.block_size;
    description.algorithm = CUSPARSE_SPMV_BSR_ALG1;
    description.alpha = op.alpha;
    description.beta = op.beta;
    plan_ = std::make_unique<CudaSparseSpmvPlan>(description);
    plan_->prepare(address(device, op.input), address(device, op.output),
                   nullptr, false);
    CUDADriver::get_instance().stream_synchronize(nullptr);
  }
  ~CudaBsrSpmv() override {
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
  std::shared_ptr<BsrF32Resource> matrix_;
  std::unique_ptr<CudaSparseSpmvPlan> plan_;
};
#endif
}  // namespace

void register_cusparse_bsr_operations(NativeOperationRegistry &registry) {
  registry.define<BsrF32Description>(validate_matrix,
                                     NativeObjectKind::resource);
  registry.register_schema<BsrF32Description>(
      "cusparse", "bsr_f32", {"row_offsets", "column_indices", "values"},
      [](const liong::json::JsonValue &parameters)
          -> NativeOperationRegistry::Binder<BsrF32Description> {
        operation_json::fields(parameters,
                               {"block_rows", "block_columns", "block_size"});
        BsrF32Description d;
        d.block_rows = operation_json::integer(parameters["block_rows"]);
        d.block_columns = operation_json::integer(parameters["block_columns"]);
        d.block_size = operation_json::integer(parameters["block_size"]);
        geometry(d);
        return [d](const NativeOperationBindings &bindings) {
          auto result = d;
          result.arrays = {bindings.at("row_offsets").array(),
                           bindings.at("column_indices").array(),
                           bindings.at("values").array()};
          return result;
        };
      });
  registry.define<BsrSpmvF32Operation>(validate_spmv);
  registry.register_schema<BsrSpmvF32Operation>(
      "cusparse", "bsr_spmv_f32", {"matrix", "input", "output"},
      [](const liong::json::JsonValue &parameters)
          -> NativeOperationRegistry::Binder<BsrSpmvF32Operation> {
        operation_json::fields(parameters, {"alpha", "beta"});
        BsrSpmvF32Operation op;
        op.alpha = operation_json::optional_f32(parameters, "alpha", 1);
        op.beta = operation_json::optional_f32(parameters, "beta", 0);
        return [op](const NativeOperationBindings &bindings) {
          auto result = op;
          result.matrix = bindings.at("matrix").resource<BsrF32Resource>();
          result.input = bindings.at("input").array();
          result.output = bindings.at("output").array();
          return result;
        };
      });
#ifdef TI_WITH_CUDA
  registry.register_factory<BsrF32Description>(
      Arch::cuda, NativeOperationMode::cuda_capture,
      [](Arch, Device &device, const BsrF32Description &description) {
        auto &cuda = CUDAContext::get_instance();
        auto lock = cuda.get_submission_lock_guard();
        auto guard = cuda.get_guard();
        return std::make_shared<CudaBsr>(device, description);
      },
      probe_bsr);
  registry.register_factory<BsrSpmvF32Operation>(
      Arch::cuda, NativeOperationMode::cuda_capture,
      [](Arch, Device &device, const BsrSpmvF32Operation &description) {
        auto &cuda = CUDAContext::get_instance();
        auto lock = cuda.get_submission_lock_guard();
        auto guard = cuda.get_guard();
        return std::make_shared<CudaBsrSpmv>(device, description);
      },
      probe_bsr);
#endif
}
}  // namespace taichi::lang
