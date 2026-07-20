#include "taichi/program/sparse_matrix.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Eigen/Dense"
#include "Eigen/SparseLU"
#include "taichi/ir/type_factory.h"
#include "taichi/program/kernel.h"

#define BUILD(TYPE)                                                         \
  {                                                                         \
    using T = Eigen::Triplet<float##TYPE>;                                  \
    std::vector<T> *triplets = static_cast<std::vector<T> *>(triplets_adr); \
    matrix_.setFromTriplets(triplets->begin(), triplets->end());            \
  }

#define MAKE_MATRIX(TYPE, STORAGE)                                        \
  {Pair("f" #TYPE, #STORAGE),                                             \
   [](int rows, int cols, DataType dt) -> std::unique_ptr<SparseMatrix> { \
     using FC = Eigen::SparseMatrix<float##TYPE, Eigen::STORAGE>;         \
     return std::make_unique<EigenSparseMatrix<FC>>(rows, cols, dt);      \
   }}

#define INSTANTIATE_SPMV(type, storage)                               \
  template void                                                       \
  EigenSparseMatrix<Eigen::SparseMatrix<type, Eigen::storage>>::spmv( \
      Program *prog, const Ndarray &x, const Ndarray &y);

namespace {
using Pair = std::pair<std::string, std::string>;
struct key_hash {
  std::size_t operator()(const Pair &k) const {
    auto h1 = std::hash<std::string>{}(k.first);
    auto h2 = std::hash<std::string>{}(k.second);
    return h1 ^ h2;
  }
};

template <typename T, typename T1, typename T2>
void print_triplets_from_csr(int64_t n_rows,
                             int n_cols,
                             T *row,
                             T1 *col,
                             T2 *value,
                             std::ostringstream &ostr) {
  using Triplets = Eigen::Triplet<T2>;
  std::vector<Triplets> trips;
  for (int64_t i = 1; i <= n_rows; ++i) {
    auto n_i = row[i] - row[i - 1];
    for (auto j = 0; j < n_i; ++j) {
      trips.push_back({static_cast<int>(i - 1),
                       static_cast<int>(col[row[i - 1] + j]),
                       static_cast<float>(value[row[i - 1] + j])});
    }
  }
  Eigen::SparseMatrix<float> m(n_rows, n_cols);
  m.setFromTriplets(trips.begin(), trips.end());
  Eigen::IOFormat clean_fmt(4, 0, ", ", "\n", "[", "]");
  ostr << Eigen::MatrixXf(m.cast<float>()).format(clean_fmt);
}

template <typename T, typename T1, typename T2>
T2 get_element_from_csr(int row,
                        int col,
                        T *row_data,
                        T1 *col_data,
                        T2 *value) {
  for (T i = row_data[row]; i < row_data[row + 1]; ++i) {
    if (col == col_data[i])
      return value[i];
  }
  // zero entry
  return 0;
}

}  // namespace

namespace taichi::lang {

namespace {
std::atomic<std::uint64_t> next_sparse_matrix_id{1};
}  // namespace

std::uint64_t allocate_sparse_matrix_id() {
  return next_sparse_matrix_id.fetch_add(1, std::memory_order_relaxed);
}

SparseMatrixRuntimeStatistics SparseMatrix::make_runtime_statistics(
    const std::string &backend_family,
    const std::string &storage_format) const {
  SparseMatrixRuntimeStatistics result;
  result.backend_family = backend_family;
  result.storage_format = storage_format;
  result.dtype = data_type_name(dtype_);
  result.rows = rows_;
  result.cols = cols_;
  result.pattern_version = pattern_version_.load(std::memory_order_relaxed);
  result.numeric_version = numeric_version_.load(std::memory_order_relaxed);
  result.pattern_builds = pattern_builds_.load(std::memory_order_relaxed);
  result.numeric_updates = numeric_updates_.load(std::memory_order_relaxed);
  result.numeric_update_bytes =
      numeric_update_bytes_.load(std::memory_order_relaxed);
  result.spmv_calls = spmv_calls_.load(std::memory_order_relaxed);
  result.spmv_plan_builds =
      spmv_plan_builds_.load(std::memory_order_relaxed);
  result.spmv_plan_reuses =
      spmv_plan_reuses_.load(std::memory_order_relaxed);
  result.spmv_handle_creations =
      spmv_handle_creations_.load(std::memory_order_relaxed);
  result.dense_vector_descriptor_creations =
      dense_vector_descriptor_creations_.load(std::memory_order_relaxed);
  result.dense_vector_descriptor_rebinds =
      dense_vector_descriptor_rebinds_.load(std::memory_order_relaxed);
  result.spmv_workspace_allocations =
      spmv_workspace_allocations_.load(std::memory_order_relaxed);
  result.host_to_device_bytes =
      host_to_device_bytes_.load(std::memory_order_relaxed);
  result.device_to_host_bytes =
      device_to_host_bytes_.load(std::memory_order_relaxed);
  result.device_to_device_bytes =
      device_to_device_bytes_.load(std::memory_order_relaxed);
  return result;
}

SparseMatrixRuntimeStatistics SparseMatrix::debug_runtime_statistics() const {
  return make_runtime_statistics("unknown", "unknown");
}

void SparseMatrix::record_transfer_bytes(std::uint64_t host_to_device,
                                         std::uint64_t device_to_host,
                                         std::uint64_t device_to_device) {
  host_to_device_bytes_.fetch_add(host_to_device, std::memory_order_relaxed);
  device_to_host_bytes_.fetch_add(device_to_host, std::memory_order_relaxed);
  device_to_device_bytes_.fetch_add(device_to_device,
                                    std::memory_order_relaxed);
}

void SparseMatrix::record_pattern_build() {
  pattern_builds_.fetch_add(1, std::memory_order_relaxed);
  pattern_version_.fetch_add(1, std::memory_order_relaxed);
  numeric_version_.fetch_add(1, std::memory_order_relaxed);
}

void SparseMatrix::record_pattern_reference() {
  pattern_version_.fetch_add(1, std::memory_order_relaxed);
  numeric_version_.fetch_add(1, std::memory_order_relaxed);
}

void SparseMatrix::record_numeric_update(std::uint64_t bytes) {
  numeric_updates_.fetch_add(1, std::memory_order_relaxed);
  numeric_update_bytes_.fetch_add(bytes, std::memory_order_relaxed);
  numeric_version_.fetch_add(1, std::memory_order_relaxed);
}

void SparseMatrix::record_spmv_call() {
  spmv_calls_.fetch_add(1, std::memory_order_relaxed);
}

void SparseMatrix::record_spmv_plan_build() {
  spmv_plan_builds_.fetch_add(1, std::memory_order_relaxed);
}

void SparseMatrix::record_spmv_plan_reuse() {
  spmv_plan_reuses_.fetch_add(1, std::memory_order_relaxed);
}

void SparseMatrix::record_spmv_handle_creation() {
  spmv_handle_creations_.fetch_add(1, std::memory_order_relaxed);
}

void SparseMatrix::record_dense_vector_descriptor_creation(bool rebind) {
  dense_vector_descriptor_creations_.fetch_add(1, std::memory_order_relaxed);
  if (rebind) {
    dense_vector_descriptor_rebinds_.fetch_add(1,
                                               std::memory_order_relaxed);
  }
}

void SparseMatrix::record_spmv_workspace_allocation() {
  spmv_workspace_allocations_.fetch_add(1, std::memory_order_relaxed);
}

namespace {

bool is_scalar_kernel_parameter(const CallableBase::Parameter &parameter,
                                DataType dtype) {
  return parameter.ptype == ParameterType::kScalar &&
         !parameter.is_array && parameter.get_dtype() == dtype;
}

bool is_scalar_ndarray_kernel_parameter(
    const CallableBase::Parameter &parameter,
    DataType dtype,
    std::size_t dimensions) {
  const DataType expected_type(
      TypeFactory::get_instance().get_ndarray_struct_type(
          dtype, static_cast<int>(dimensions), false));
  return parameter.ptype == ParameterType::kNdarray && parameter.is_array &&
         !parameter.needs_grad && parameter.get_dtype() == expected_type &&
         parameter.get_element_shape().empty() &&
         parameter.total_dim == dimensions;
}

std::string compiled_operator_backend_family(Arch arch) {
  if (arch_is_cpu(arch)) {
    return "cpu";
  }
  if (arch_is_cuda(arch)) {
    return "cuda";
  }
  if (arch == Arch::vulkan) {
    return "vulkan";
  }
  return arch_name(arch);
}

void validate_compiled_operator_data(Program *program,
                                     const Ndarray &data,
                                     const char *role) {
  TI_ERROR_IF(data.owning_program() != program ||
                  !data.get_element_shape().empty() || data.shape.empty() ||
                  data.get_nelement() == 0,
              "Compiled-kernel linear operator {} must be a non-empty "
              "scalar ndarray owned by the same Program.",
              role);
}

}  // namespace

CompiledKernelLinearOperator::CompiledKernelLinearOperator(
    Program *program,
    Kernel &kernel,
    int size,
    std::uint64_t topology_version,
    std::uint64_t numeric_version,
    const Ndarray &operator_data)
    : CompiledKernelLinearOperator(program, kernel, size, topology_version,
                                   numeric_version, operator_data, nullptr) {
}

CompiledKernelLinearOperator::CompiledKernelLinearOperator(
    Program *program,
    Kernel &kernel,
    int size,
    std::uint64_t topology_version,
    std::uint64_t numeric_version,
    const Ndarray &topology_data,
    const Ndarray &numeric_data)
    : CompiledKernelLinearOperator(program, kernel, size, topology_version,
                                   numeric_version, topology_data,
                                   &numeric_data) {
}

CompiledKernelLinearOperator::CompiledKernelLinearOperator(
    Program *program,
    Kernel &kernel,
    int size,
    std::uint64_t topology_version,
    std::uint64_t numeric_version,
    const Ndarray &topology_data,
    const Ndarray *numeric_data)
    : SparseMatrix(size, size, PrimitiveType::f32) {
  TI_ERROR_IF(!program || size <= 0 || topology_version == 0 ||
                  numeric_version == 0,
              "Compiled-kernel linear operators require an owning Program, "
              "positive size, and positive topology/numeric versions.");
  const Arch arch = program->compile_config().arch;
  TI_ERROR_IF(!arch_is_cpu(arch) && !arch_is_cuda(arch) &&
                  arch != Arch::vulkan,
              "Compiled-kernel linear operators support CPU, CUDA, and "
              "Vulkan only; got {}. No fallback was performed.",
              arch_name(arch));
  TI_ERROR_IF(kernel.program != program || kernel.arch != arch,
              "Compiled-kernel linear operator kernels must belong to the "
              "same Program and backend.");
  validate_compiled_operator_data(program, topology_data, "topology data");
  if (numeric_data) {
    validate_compiled_operator_data(program, *numeric_data, "numeric data");
  }

  const auto &parameters = kernel.parameter_list;
  const DataType topology_dtype = topology_data.get_element_data_type();
  const bool has_numeric_data = numeric_data != nullptr;
  const std::size_t input_arg_index = has_numeric_data ? 3 : 2;
  const std::size_t output_arg_index = input_arg_index + 1;
  bool valid_abi =
      parameters.size() == output_arg_index + 1 && kernel.rets.empty() &&
      is_scalar_kernel_parameter(parameters[0], PrimitiveType::i32) &&
      is_scalar_ndarray_kernel_parameter(
          parameters[1], topology_dtype, topology_data.shape.size());
  if (has_numeric_data) {
    valid_abi =
        valid_abi &&
        is_scalar_ndarray_kernel_parameter(
            parameters[2], numeric_data->get_element_data_type(),
            numeric_data->shape.size());
  }
  valid_abi =
      valid_abi && is_scalar_ndarray_kernel_parameter(
                       parameters[input_arg_index], PrimitiveType::f32, 1) &&
      is_scalar_ndarray_kernel_parameter(parameters[output_arg_index],
                                         PrimitiveType::f32, 1);
  if (has_numeric_data) {
    TI_ERROR_IF(!valid_abi,
                "Compiled-kernel linear operator ABI must be exactly "
                "(i32 active_size, scalar topology_data ndarray, scalar "
                "numeric_data ndarray, f32[1D] input, f32[1D] output) with "
                "no return values.");
  } else {
    TI_ERROR_IF(!valid_abi,
                "Compiled-kernel linear operator ABI must be exactly "
                "(i32 active_size, scalar operator_data ndarray, f32[1D] "
                "input, f32[1D] output) with no return values.");
  }

  const auto &compiled = program->compile_kernel(
      program->compile_config(), program->get_device_caps(), kernel);
  TI_ERROR_IF(!compiled.snode_tree_ids().empty(),
              "Compiled-kernel linear operators must not depend on any "
              "SNodeTree; use explicit topology/numeric ndarray arguments.");

  Ndarray *owned_topology_data = nullptr;
  Ndarray *owned_numeric_data = nullptr;
  std::unique_ptr<LaunchContextBuilder> launch_context;
  try {
    owned_topology_data = program->create_ndarray(
        topology_dtype, topology_data.shape, topology_data.layout, false);
    program->copy_ndarray_fast(
        owned_topology_data, const_cast<Ndarray *>(&topology_data));
    if (has_numeric_data) {
      owned_numeric_data = program->create_ndarray(
          numeric_data->get_element_data_type(), numeric_data->shape,
          numeric_data->layout, false);
      program->copy_ndarray_fast(
          owned_numeric_data, const_cast<Ndarray *>(numeric_data));
    }
    launch_context = std::make_unique<LaunchContextBuilder>(&kernel);
    launch_context->set_arg_int({0}, size);
    launch_context->set_arg_ndarray({1}, *owned_topology_data);
    if (has_numeric_data) {
      launch_context->set_arg_ndarray({2}, *owned_numeric_data);
    }
  } catch (...) {
    if (owned_numeric_data) {
      program->delete_ndarray(owned_numeric_data);
    }
    if (owned_topology_data) {
      program->delete_ndarray(owned_topology_data);
    }
    throw;
  }

  program_ = program;
  kernel_ = &kernel;
  compiled_kernel_ = &compiled;
  topology_data_ = owned_topology_data;
  numeric_data_ = owned_numeric_data;
  launch_context_ = std::move(launch_context);
  topology_data_bytes_ = static_cast<std::uint64_t>(
      topology_data.get_nelement() * topology_data.get_element_size());
  numeric_data_bytes_ =
      numeric_data
          ? static_cast<std::uint64_t>(numeric_data->get_nelement() *
                                       numeric_data->get_element_size())
          : 0;
  input_arg_index_ = input_arg_index;
  output_arg_index_ = output_arg_index;
  topology_version_ = topology_version;
  numeric_version_ = numeric_version;
  record_pattern_build();
  record_spmv_plan_build();
  record_transfer_bytes(0, 0,
                        topology_data_bytes_ + numeric_data_bytes_);
}

CompiledKernelLinearOperator::~CompiledKernelLinearOperator() {
  if (program_ && numeric_data_) {
    program_->delete_ndarray(numeric_data_);
  }
  if (program_ && topology_data_) {
    program_->delete_ndarray(topology_data_);
  }
}

void CompiledKernelLinearOperator::nd_spmv(Program *program,
                                           const Ndarray &input,
                                           const Ndarray &output) {
  TI_ERROR_IF(program != program_,
              "Compiled-kernel linear operator apply requires its owning "
              "Program; no fallback was performed.");
  auto validate_vector = [&](const char *role, const Ndarray &array) {
    TI_ERROR_IF(array.owning_program() != program_ ||
                    array.get_element_data_type() != PrimitiveType::f32 ||
                    !array.get_element_shape().empty() ||
                    array.shape.size() != 1 ||
                    array.get_nelement() != static_cast<std::size_t>(rows_),
                "Compiled-kernel linear operator {} must contain exactly {} "
                "scalar f32 entries owned by the same Program.",
                role, rows_);
  };
  validate_vector("input", input);
  validate_vector("output", output);
  TI_ERROR_IF(input.get_device_allocation_ptr_as_int() ==
                  output.get_device_allocation_ptr_as_int(),
              "Compiled-kernel linear operator input and output must not "
              "alias.");

  auto numeric_guard = acquire_numeric_access_guard();
  std::lock_guard<std::mutex> lock(spmv_mutex_);
  // CPU launchers lower kNdarray placeholders to raw pointers in place. Rebind
  // every ndarray slot before reuse so the persistent context is backend-
  // neutral and generation validation observes only the current resources.
  launch_context_->set_arg_ndarray({1}, *topology_data_);
  if (numeric_data_) {
    launch_context_->set_arg_ndarray({2}, *numeric_data_);
  }
  launch_context_->set_arg_ndarray(
      {static_cast<int>(input_arg_index_)}, input);
  launch_context_->set_arg_ndarray(
      {static_cast<int>(output_arg_index_)}, output);
  record_spmv_call();
  record_spmv_plan_reuse();
  program_->launch_kernel(*compiled_kernel_, *launch_context_);
}

void CompiledKernelLinearOperator::update_numeric_data(
    Program *program,
    const Ndarray &numeric_data,
    std::uint64_t expected_topology_version,
    std::uint64_t expected_numeric_version) {
  TI_ERROR_IF(program != program_ || !numeric_data_,
              "Compiled-kernel numeric update requires the owning Program "
              "and a dual-resource operator.");
  validate_compiled_operator_data(program, numeric_data, "numeric update");

  auto numeric_guard = acquire_numeric_access_guard();
  std::lock_guard<std::mutex> lock(spmv_mutex_);
  TI_ERROR_IF(expected_topology_version != topology_version_ ||
                  expected_numeric_version != numeric_version_,
              "Compiled-kernel numeric update version mismatch: expected "
              "topology/numeric ({}, {}), current ({}, {}).",
              expected_topology_version, expected_numeric_version,
              topology_version_, numeric_version_);
  TI_ERROR_IF(numeric_version_ == std::numeric_limits<std::uint64_t>::max(),
              "Compiled-kernel numeric version overflow.");
  TI_ERROR_IF(numeric_data.get_element_data_type() !=
                      numeric_data_->get_element_data_type() ||
                  numeric_data.shape != numeric_data_->shape ||
                  numeric_data.layout != numeric_data_->layout,
              "Compiled-kernel numeric update must preserve dtype, shape, "
              "and layout.");

  program_->copy_ndarray_fast(numeric_data_,
                              const_cast<Ndarray *>(&numeric_data));
  numeric_version_++;
  record_numeric_update(numeric_data_bytes_);
  record_transfer_bytes(0, 0, numeric_data_bytes_);
}

SparseMatrixRuntimeStatistics
CompiledKernelLinearOperator::debug_runtime_statistics() const {
  std::lock_guard<std::mutex> lock(spmv_mutex_);
  auto result = make_runtime_statistics(
      compiled_operator_backend_family(program_->compile_config().arch),
      "matrix_free_kernel");
  result.provider_name = "forge_compiled_taichi_kernel";
  result.pattern_version = topology_version_;
  result.numeric_version = numeric_version_;
  result.nnz = 0;
  result.pattern_reserved_bytes = topology_data_bytes_;
  result.values_reserved_bytes = numeric_data_bytes_;
  result.operator_owned_reserved_bytes =
      topology_data_bytes_ + numeric_data_bytes_;
  result.operator_exclusive_reserved_bytes =
      topology_data_bytes_ + numeric_data_bytes_;
  return result;
}

SparseMatrixBuilder::SparseMatrixBuilder(int rows,
                                         int cols,
                                         int max_num_triplets,
                                         DataType dtype,
                                         const std::string &storage_format)
    : rows_(rows),
      cols_(cols),
      max_num_triplets_(max_num_triplets),
      dtype_(dtype),
      storage_format_(storage_format) {
  auto element_size = data_type_size(dtype);
  TI_ERROR_IF(rows <= 0 || cols <= 0,
              "SparseMatrixBuilder rows and columns must be positive.");
  TI_ERROR_IF(max_num_triplets < 0 ||
                  max_num_triplets >
                      (std::numeric_limits<int>::max() - 1) / 3,
              "SparseMatrixBuilder max_num_triplets must be in [0, {}].",
              (std::numeric_limits<int>::max() - 1) / 3);
  TI_ERROR_IF(element_size != 4 && element_size != 8,
              "SparseMatrixBuilder supports only 32-bit and 64-bit floating "
              "point values.");
}

SparseMatrixBuilder::~SparseMatrixBuilder() = default;

void SparseMatrixBuilder::create_ndarray(Program *prog) {
  TI_ERROR_IF(!prog, "SparseMatrixBuilder requires an active Program.");
  TI_ERROR_IF(ndarray_data_base_ptr_ || program_,
              "SparseMatrixBuilder storage has already been created.");
  const bool descriptor_storage = prog->compile_config().arch == Arch::vulkan;
  TI_ERROR_IF(descriptor_storage && dtype_ != PrimitiveType::f32,
              "Vulkan sparse matrix builders support f32 values only.");
  const DataType storage_dtype =
      descriptor_storage ? PrimitiveType::i32 : dtype_;
  const auto element_size = data_type_size(storage_dtype);
  auto *storage = prog->create_ndarray(
      storage_dtype, std::vector<int>{3 * (int)max_num_triplets_ + 2});
  ndarray_data_base_ptr_ = storage;
  program_ = prog;
  ndarray_data_ptr_ = prog->get_ndarray_data_ptr_as_int(storage);
  storage->write_int(std::vector<int>{0}, 0);
  TypedConstant raw_capacity(storage_dtype);
  if (element_size == sizeof(int32_t)) {
    raw_capacity.val_i32 = static_cast<int32_t>(max_num_triplets_);
  } else {
    raw_capacity.val_i64 = static_cast<int64_t>(max_num_triplets_);
  }
  storage->write(std::vector<int>{1}, raw_capacity);
}

void SparseMatrixBuilder::delete_ndarray(Program *prog) {
  if (!ndarray_data_base_ptr_) {
    return;
  }
  TI_ERROR_IF(!prog || prog != program_,
              "SparseMatrixBuilder storage must be deleted by its owning "
              "Program.");
  cuda_assembly_plan_.reset();
  vulkan_assembly_plan_.reset();
  prog->delete_ndarray(ndarray_data_base_ptr_);
  ndarray_data_base_ptr_ = nullptr;
  ndarray_data_ptr_ = 0;
  program_ = nullptr;
  num_triplets_ = 0;
  built_ = false;
}

template <typename T, typename G>
void SparseMatrixBuilder::print_triplets_template() {
  TI_ERROR_IF(!ndarray_data_base_ptr_,
              "SparseMatrixBuilder storage is not available.");
  auto ptr = get_ndarray_data_ptr();
  G *data = reinterpret_cast<G *>(ptr);
  const G raw_count = data[0];
  TI_ERROR_IF(raw_count < 0 ||
                  static_cast<uint64>(raw_count) > max_num_triplets_,
              "SparseMatrixBuilder triplet count {} exceeds capacity {}.",
              raw_count, max_num_triplets_);
  num_triplets_ = static_cast<uint64>(raw_count);
  fmt::print("n={}, m={}, num_triplets={} (max={})\n", rows_, cols_,
             num_triplets_, max_num_triplets_);
  data += 2;
  for (int i = 0; i < num_triplets_; i++) {
    fmt::print("[{}, {}] = {}\n", data[i * 3], data[i * 3 + 1],
               taichi_union_cast<T>(data[i * 3 + 2]));
  }
}

void SparseMatrixBuilder::print_triplets_eigen() {
  auto element_size = data_type_size(dtype_);
  switch (element_size) {
    case 4:
      print_triplets_template<float32, int32>();
      break;
    case 8:
      print_triplets_template<float64, int64>();
      break;
    default:
      TI_ERROR("Unsupported sparse matrix data type!");
      break;
  }
}

void SparseMatrixBuilder::print_triplets_cuda() {
#ifdef TI_WITH_CUDA
  TI_ERROR_IF(!ndarray_data_base_ptr_,
              "SparseMatrixBuilder storage is not available.");
  int32_t raw_count = 0;
  CUDADriver::get_instance().memcpy_device_to_host(
      &raw_count, (void *)get_ndarray_data_ptr(), sizeof(raw_count));
  TI_ERROR_IF(raw_count < 0 ||
                  static_cast<uint64>(raw_count) > max_num_triplets_,
              "SparseMatrixBuilder triplet count {} exceeds capacity {}.",
              raw_count, max_num_triplets_);
  num_triplets_ = static_cast<uint64>(raw_count);
  fmt::print("n={}, m={}, num_triplets={} (max={})\n", rows_, cols_,
             num_triplets_, max_num_triplets_);
  auto len = 3 * num_triplets_ + 2;
  std::vector<float32> trips(len);
  CUDADriver::get_instance().memcpy_device_to_host(
      (void *)trips.data(), (void *)get_ndarray_data_ptr(),
      len * sizeof(float32));
  for (auto i = 0; i < num_triplets_; i++) {
    int row = taichi_union_cast<int>(trips[3 * i + 2]);
    int col = taichi_union_cast<int>(trips[3 * i + 3]);
    auto val = trips[i * 3 + 4];
    fmt::print("[{}, {}] = {}\n", row, col, val);
  }
#endif
}

intptr_t SparseMatrixBuilder::get_ndarray_data_ptr() const {
  return ndarray_data_ptr_;
}

Ndarray *SparseMatrixBuilder::get_ndarray() const {
  TI_ERROR_IF(!ndarray_data_base_ptr_,
              "SparseMatrixBuilder storage is not available.");
  return ndarray_data_base_ptr_;
}

template <typename T>
struct BuilderEntry {
  int row{0};
  int column{0};
  T value{0};
  uint64 ordinal{0};
};

template <typename T, typename G>
std::unique_ptr<SparseMatrix> SparseMatrixBuilder::build_template() {
  TI_ERROR_IF(!ndarray_data_base_ptr_ || !program_,
              "SparseMatrixBuilder storage is not available.");
  auto ptr = get_ndarray_data_ptr();
  G *data = reinterpret_cast<G *>(ptr);
  const G raw_count = data[0];
  TI_ERROR_IF(raw_count < 0 ||
                  static_cast<uint64>(raw_count) > max_num_triplets_,
              "SparseMatrixBuilder triplet count {} exceeds capacity {}.",
              raw_count, max_num_triplets_);
  num_triplets_ = static_cast<uint64>(raw_count);
  data += 2;
  std::vector<BuilderEntry<T>> entries;
  entries.reserve(static_cast<std::size_t>(num_triplets_));
  for (uint64 i = 0; i < num_triplets_; ++i) {
    const G row = data[i * 3];
    const G column = data[i * 3 + 1];
    const T value = taichi_union_cast<T>(data[i * 3 + 2]);
    TI_ERROR_IF(row < 0 || row >= static_cast<G>(rows_) || column < 0 ||
                    column >= static_cast<G>(cols_),
                "SparseMatrixBuilder triplet {} index [{}, {}] is outside "
                "matrix dimensions [{}, {}].",
                i, row, column, rows_, cols_);
    TI_ERROR_IF(!std::isfinite(value),
                "SparseMatrixBuilder triplet {} contains a non-finite value.",
                i);
    entries.push_back(
        {static_cast<int>(row), static_cast<int>(column), value, i});
  }
  std::sort(entries.begin(), entries.end(),
            [](const BuilderEntry<T> &left, const BuilderEntry<T> &right) {
              if (left.row != right.row)
                return left.row < right.row;
              if (left.column != right.column)
                return left.column < right.column;
              return left.ordinal < right.ordinal;
            });

  using Triplet = Eigen::Triplet<T>;
  std::vector<Triplet> triplets;
  triplets.reserve(entries.size());
  for (std::size_t begin = 0; begin < entries.size();) {
    std::size_t end = begin + 1;
    T sum = entries[begin].value;
    while (end < entries.size() && entries[end].row == entries[begin].row &&
           entries[end].column == entries[begin].column) {
      sum += entries[end].value;
      TI_ERROR_IF(!std::isfinite(sum),
                  "SparseMatrixBuilder duplicate sum at [{}, {}] is "
                  "non-finite.",
                  entries[begin].row, entries[begin].column);
      ++end;
    }
    triplets.emplace_back(entries[begin].row, entries[begin].column, sum);
    begin = end;
  }
  auto matrix = make_sparse_matrix(rows_, cols_, dtype_, storage_format_);
  matrix->build_triplets(static_cast<void *>(&triplets));
  return matrix;
}

std::unique_ptr<SparseMatrix> SparseMatrixBuilder::build() {
  TI_ERROR_IF(built_, "SparseMatrixBuilder build is already in progress.");
  built_ = true;
  try {
    std::unique_ptr<SparseMatrix> matrix;
    const auto element_size = data_type_size(dtype_);
    switch (element_size) {
      case 4:
        matrix = build_template<float32, int32>();
        break;
      case 8:
        matrix = build_template<float64, int64>();
        break;
      default:
        TI_ERROR("Unsupported sparse matrix data type!");
    }
    clear();
    return matrix;
  } catch (...) {
    clear();
    throw;
  }
}

std::unique_ptr<SparseMatrix> SparseMatrixBuilder::build_cuda() {
  TI_ERROR_IF(built_, "SparseMatrixBuilder build is already in progress.");
  built_ = true;
#ifdef TI_WITH_CUDA
  try {
    TI_ERROR_IF(!ndarray_data_base_ptr_ || !program_,
                "SparseMatrixBuilder storage is not available.");
    TI_ERROR_IF(dtype_ != PrimitiveType::f32,
                "CUDA sparse assembly supports f32 values only.");
    TI_ERROR_IF(max_num_triplets_ == 0,
                "CUDA sparse assembly requires a positive triplet "
                "capacity.");
    if (!cuda_assembly_plan_) {
      cuda_assembly_plan_ = std::make_unique<CudaSparseAssemblyPlan>(
          program_, rows_, cols_, static_cast<int>(max_num_triplets_));
    }
    auto matrix =
        cuda_assembly_plan_->build_packed(program_, *ndarray_data_base_ptr_);
    clear();
    return matrix;
  } catch (...) {
    clear();
    throw;
  }
#else
  clear();
  TI_NOT_IMPLEMENTED;
#endif
}

std::unique_ptr<SparseMatrix> SparseMatrixBuilder::build_vulkan() {
  TI_ERROR_IF(built_, "SparseMatrixBuilder build is already in progress.");
  built_ = true;
#ifdef TI_WITH_VULKAN
  try {
    TI_ERROR_IF(!ndarray_data_base_ptr_ || !program_,
                "SparseMatrixBuilder storage is not available.");
    TI_ERROR_IF(dtype_ != PrimitiveType::f32,
                "Vulkan sparse assembly supports f32 values only.");
    TI_ERROR_IF(max_num_triplets_ == 0,
                "Vulkan sparse assembly requires a positive triplet "
                "capacity.");
    if (!vulkan_assembly_plan_) {
      vulkan_assembly_plan_ = std::make_unique<VulkanSparseAssemblyPlan>(
          program_, rows_, cols_, static_cast<int>(max_num_triplets_));
    }
    auto matrix = vulkan_assembly_plan_->build_packed(
        program_, *ndarray_data_base_ptr_);
    clear();
    return matrix;
  } catch (...) {
    clear();
    throw;
  }
#else
  clear();
  TI_NOT_IMPLEMENTED;
#endif
}

void SparseMatrixBuilder::clear() {
  built_ = false;
  if (ndarray_data_base_ptr_) {
    ndarray_data_base_ptr_->write_int(std::vector<int>{0}, 0);
  }
  num_triplets_ = 0;
}

template <class EigenMatrix>
const std::string EigenSparseMatrix<EigenMatrix>::to_string() const {
  Eigen::IOFormat clean_fmt(4, 0, ", ", "\n", "[", "]");
  // Note that the code below first converts the sparse matrix into a dense one.
  // https://stackoverflow.com/questions/38553335/how-can-i-print-in-console-a-formatted-sparse-matrix-with-eigen
  std::ostringstream ostr;
  ostr << Eigen::MatrixXf(matrix_.template cast<float>()).format(clean_fmt);
  return ostr.str();
}

template <class EigenMatrix>
void EigenSparseMatrix<EigenMatrix>::mmwrite(const std::string &filename) {
  std::ofstream file(filename);
  file << "%%MatrixMarket matrix coordinate real general\n%" << std::endl;
  file << matrix_.rows() << " " << matrix_.cols() << " " << matrix_.nonZeros()
       << std::endl;
  for (int k = 0; k < matrix_.outerSize(); ++k) {
    for (typename EigenMatrix::InnerIterator it(matrix_, k); it; ++it) {
      file << it.row() + 1 << " " << it.col() + 1 << " " << it.value()
           << std::endl;
    }
  }
  file.close();
}

template <class EigenMatrix>
void EigenSparseMatrix<EigenMatrix>::build_triplets(void *triplets_adr) {
  std::string sdtype = taichi::lang::data_type_name(dtype_);
  if (sdtype == "f32") {
    BUILD(32)
  } else if (sdtype == "f64") {
    BUILD(64)
  } else {
    TI_ERROR("Unsupported sparse matrix data type {}!", sdtype);
  }
  record_pattern_build();
}

template <class EigenMatrix>
void EigenSparseMatrix<EigenMatrix>::spmv(Program *prog,
                                          const Ndarray &x,
                                          const Ndarray &y) {
  size_t dX = prog->get_ndarray_data_ptr_as_int(&x);
  size_t dY = prog->get_ndarray_data_ptr_as_int(&y);
  record_spmv_call();
  std::string sdtype = taichi::lang::data_type_name(dtype_);
  if (sdtype == "f32") {
    Eigen::Map<Eigen::VectorXf>((float *)dY, rows_) =
        matrix_.template cast<float>() *
        Eigen::Map<Eigen::VectorXf>((float *)dX, cols_);
  } else if (sdtype == "f64") {
    Eigen::Map<Eigen::VectorXd>((double *)dY, rows_) =
        matrix_.template cast<double>() *
        Eigen::Map<Eigen::VectorXd>((double *)dX, cols_);
  } else {
    TI_ERROR("Unsupported sparse matrix data type {}!", sdtype);
  }
}

template <class EigenMatrix>
void EigenSparseMatrix<EigenMatrix>::update_values(
    Program *prog,
    const Ndarray &values) {
  const auto nnz = static_cast<std::size_t>(matrix_.nonZeros());
  const auto value_bytes = data_type_size(dtype_);
  TI_ERROR_IF(values.get_element_data_type() != dtype_ ||
                  !values.get_element_shape().empty() ||
                  values.get_nelement() != nnz ||
                  values.get_element_size() != value_bytes,
              "SparseMatrix value-only update expects exactly {} scalar {} "
              "values in storage order, got {} element(s) of {} byte(s).",
              nnz, data_type_name(dtype_), values.get_nelement(),
              values.get_element_size());
  matrix_.makeCompressed();
  record_numeric_update(nnz * value_bytes);
  if (nnz == 0) {
    return;
  }
  auto src = prog->get_ndarray_data_ptr_as_int(&values);
  std::memcpy(matrix_.valuePtr(), reinterpret_cast<const void *>(src),
              nnz * value_bytes);
}

INSTANTIATE_SPMV(float32, ColMajor)
INSTANTIATE_SPMV(float32, RowMajor)
INSTANTIATE_SPMV(float64, ColMajor)
INSTANTIATE_SPMV(float64, RowMajor)

std::unique_ptr<SparseMatrix> make_sparse_matrix(
    int rows,
    int cols,
    DataType dt,
    const std::string &storage_format = "col_major") {
  using func_type = std::unique_ptr<SparseMatrix> (*)(int, int, DataType);
  static const std::unordered_map<Pair, func_type, key_hash> map = {
      MAKE_MATRIX(32, ColMajor), MAKE_MATRIX(32, RowMajor),
      MAKE_MATRIX(64, ColMajor), MAKE_MATRIX(64, RowMajor)};
  std::unordered_map<std::string, std::string> format_map = {
      {"col_major", "ColMajor"}, {"row_major", "RowMajor"}};
  std::string tdt = taichi::lang::data_type_name(dt);
  Pair key = std::make_pair(tdt, format_map.at(storage_format));
  auto it = map.find(key);
  if (it != map.end()) {
    auto func = map.at(key);
    return func(rows, cols, dt);
  } else
    TI_ERROR("Unsupported sparse matrix data type: {}, storage format: {}", tdt,
             storage_format);
}

namespace {

std::atomic<std::uint64_t> next_sparse_pattern_id{1};

void validate_compressed_host_pattern(
    const char *storage_format,
    int compressed_rows,
    int columns,
    std::size_t nnz,
    const std::vector<int32_t> &row_offsets,
    const std::vector<int32_t> &column_indices) {
  TI_ERROR_IF(row_offsets.size() !=
                      static_cast<std::size_t>(compressed_rows) + 1 ||
                  column_indices.size() != nnz,
              "{} host pattern validation received inconsistent storage "
              "sizes.",
              storage_format);
  TI_ERROR_IF(row_offsets.front() != 0 ||
                  row_offsets.back() != static_cast<int32_t>(nnz),
              "{} row offsets must start at 0 and end at the stored count "
              "{}.",
              storage_format, nnz);
  for (int row = 0; row < compressed_rows; ++row) {
    const int32_t begin = row_offsets[row];
    const int32_t end = row_offsets[row + 1];
    TI_ERROR_IF(begin < 0 || end < begin ||
                    end > static_cast<int32_t>(nnz),
                "{} row offsets are not monotone at row {}.",
                storage_format, row);
    int32_t previous_column = -1;
    for (int32_t offset = begin; offset < end; ++offset) {
      const int32_t column = column_indices[offset];
      TI_ERROR_IF(column < 0 || column >= columns,
                  "{} column {} at offset {} is outside [0, {}).",
                  storage_format, column, offset, columns);
      TI_ERROR_IF(column <= previous_column,
                  "{} columns must be strictly increasing and unique "
                  "within row {}, got {} after {}.",
                  storage_format, row, column, previous_column);
      previous_column = column;
    }
  }
}

}  // namespace

SparseCsrPattern::SparseCsrPattern(Program *program,
                                   int rows,
                                   int cols,
                                   const Ndarray &row_offsets,
                                   const Ndarray &column_indices) {
  TI_ERROR_IF(!program,
              "Internal CSR patterns require an active Program.");
  const Arch arch = program->compile_config().arch;
  TI_ERROR_IF(!arch_is_cpu(arch) && !arch_is_cuda(arch) &&
                  arch != Arch::vulkan,
              "Internal CSR patterns support CPU, CUDA, and Vulkan backends, "
              "got {}.",
              arch_name(arch));
  TI_ERROR_IF(rows <= 0 || cols <= 0,
              "Internal CSR patterns require positive dimensions, got {} x "
              "{}.",
              rows, cols);
  TI_ERROR_IF(row_offsets.get_element_data_type() != PrimitiveType::i32 ||
                  !row_offsets.get_element_shape().empty() ||
                  row_offsets.get_nelement() !=
                      static_cast<std::size_t>(rows) + 1 ||
                  row_offsets.get_element_size() != sizeof(int32_t),
              "Internal CSR pattern row offsets must contain exactly {} "
              "scalar i32 entries.",
              rows + 1);
  TI_ERROR_IF(column_indices.get_element_data_type() != PrimitiveType::i32 ||
                  !column_indices.get_element_shape().empty() ||
                  column_indices.get_element_size() != sizeof(int32_t),
              "Internal CSR pattern column indices must be a scalar i32 "
              "ndarray.");

  const std::size_t nnz_size = column_indices.get_nelement();
  TI_ERROR_IF(nnz_size == 0,
              "Internal CSR patterns currently require at least one stored "
              "entry.");
  TI_ERROR_IF(nnz_size >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "Internal CSR pattern nnz exceeds the i32 limit.");

  const auto row_bytes =
      (static_cast<std::size_t>(rows) + 1) * sizeof(int32_t);
  const auto column_bytes = nnz_size * sizeof(int32_t);
  if (arch_is_cpu(arch)) {
    cpu_row_offsets_.resize(static_cast<std::size_t>(rows) + 1);
    cpu_column_indices_.resize(nnz_size);
    const auto source_row_offsets = reinterpret_cast<const void *>(
        program->get_ndarray_data_ptr_as_int(&row_offsets));
    const auto source_column_indices = reinterpret_cast<const void *>(
        program->get_ndarray_data_ptr_as_int(&column_indices));
    std::memcpy(cpu_row_offsets_.data(), source_row_offsets, row_bytes);
    std::memcpy(cpu_column_indices_.data(), source_column_indices,
                column_bytes);
    validate_compressed_host_pattern(
        "CSR", rows, cols, nnz_size, cpu_row_offsets_, cpu_column_indices_);
  } else if (arch_is_cuda(arch)) {
#if defined(TI_WITH_CUDA)
    std::vector<int32_t> host_row_offsets(
        static_cast<std::size_t>(rows) + 1);
    std::vector<int32_t> host_column_indices(nnz_size);
    auto source_row_offsets = reinterpret_cast<void *>(
        program->get_ndarray_data_ptr_as_int(&row_offsets));
    auto source_column_indices = reinterpret_cast<void *>(
        program->get_ndarray_data_ptr_as_int(&column_indices));
    CUDADriver::get_instance().memcpy_device_to_host(
        host_row_offsets.data(), source_row_offsets, row_bytes);
    CUDADriver::get_instance().memcpy_device_to_host(
        host_column_indices.data(), source_column_indices, column_bytes);
    validate_compressed_host_pattern(
        "CSR", rows, cols, nnz_size, host_row_offsets, host_column_indices);

    void *owned_row_offsets = nullptr;
    void *owned_column_indices = nullptr;
    try {
      CUDADriver::get_instance().malloc(&owned_row_offsets, row_bytes);
      CUDADriver::get_instance().malloc(&owned_column_indices, column_bytes);
      CUDADriver::get_instance().memcpy_device_to_device(
          owned_row_offsets, source_row_offsets, row_bytes);
      CUDADriver::get_instance().memcpy_device_to_device(
          owned_column_indices, source_column_indices, column_bytes);
    } catch (...) {
      if (owned_column_indices) {
        CUDADriver::get_instance().mem_free.call_with_warning(
            owned_column_indices);
      }
      if (owned_row_offsets) {
        CUDADriver::get_instance().mem_free.call_with_warning(
            owned_row_offsets);
      }
      throw;
    }
    cuda_row_offsets_ = owned_row_offsets;
    cuda_column_indices_ = owned_column_indices;
    device_to_host_bytes_ = row_bytes + column_bytes;
    device_to_device_bytes_ = row_bytes + column_bytes;
#else
    TI_NOT_IMPLEMENTED;
#endif
  } else {
#if defined(TI_WITH_VULKAN)
    std::vector<int32_t> host_row_offsets(
        static_cast<std::size_t>(rows) + 1);
    std::vector<int32_t> host_column_indices(nnz_size);
    program->copy_ndarray_to_host(const_cast<Ndarray *>(&row_offsets),
                                  host_row_offsets.data(), row_bytes);
    program->copy_ndarray_to_host(const_cast<Ndarray *>(&column_indices),
                                  host_column_indices.data(), column_bytes);
    validate_compressed_host_pattern(
        "CSR", rows, cols, nnz_size, host_row_offsets, host_column_indices);

    Ndarray *owned_row_offsets = nullptr;
    Ndarray *owned_column_indices = nullptr;
    try {
      owned_row_offsets = program->create_ndarray(
          PrimitiveType::i32, {rows + 1}, ExternalArrayLayout::kNull, false);
      owned_column_indices = program->create_ndarray(
          PrimitiveType::i32, {static_cast<int>(nnz_size)},
          ExternalArrayLayout::kNull, false);
      auto submission_guard =
          program->acquire_runtime_resource_submission_guard();
      const Ndarray *copy_resources[] = {
          owned_row_offsets, &row_offsets, owned_column_indices,
          &column_indices};
      program->retain_ndarrays_for_external_submission(
          copy_resources, std::size(copy_resources));
      program->copy_ndarray_fast(owned_row_offsets,
                                 const_cast<Ndarray *>(&row_offsets));
      program->copy_ndarray_fast(owned_column_indices,
                                 const_cast<Ndarray *>(&column_indices));
    } catch (...) {
      if (owned_column_indices) {
        program->delete_ndarray(owned_column_indices);
      }
      if (owned_row_offsets) {
        program->delete_ndarray(owned_row_offsets);
      }
      throw;
    }
    vulkan_row_offsets_ = owned_row_offsets;
    vulkan_column_indices_ = owned_column_indices;
    device_to_host_bytes_ = row_bytes + column_bytes;
    device_to_device_bytes_ = row_bytes + column_bytes;
#else
    TI_NOT_IMPLEMENTED;
#endif
  }

  program_ = program;
  arch_ = arch;
  rows_ = rows;
  cols_ = cols;
  nnz_ = static_cast<int>(nnz_size);
  pattern_id_ =
      next_sparse_pattern_id.fetch_add(1, std::memory_order_relaxed);
}

SparseCsrPattern::~SparseCsrPattern() {
#if defined(TI_WITH_CUDA)
  if (arch_is_cuda(arch_)) {
    if (cuda_column_indices_) {
      CUDADriver::get_instance().mem_free.call_with_warning(
          cuda_column_indices_);
    }
    if (cuda_row_offsets_) {
      CUDADriver::get_instance().mem_free.call_with_warning(cuda_row_offsets_);
    }
  }
#endif
#if defined(TI_WITH_VULKAN)
  if (arch_ == Arch::vulkan && program_) {
    if (vulkan_column_indices_) {
      program_->delete_ndarray(vulkan_column_indices_);
    }
    if (vulkan_row_offsets_) {
      program_->delete_ndarray(vulkan_row_offsets_);
    }
  }
#endif
}

std::uint64_t SparseCsrPattern::pattern_reserved_bytes() const {
  if (arch_is_cpu(arch_)) {
    return (static_cast<std::uint64_t>(cpu_row_offsets_.capacity()) +
            static_cast<std::uint64_t>(cpu_column_indices_.capacity())) *
           sizeof(int32_t);
  }
  return (static_cast<std::uint64_t>(rows_) + 1 +
          static_cast<std::uint64_t>(nnz_)) *
         sizeof(int32_t);
}

const std::vector<int32_t> &SparseCsrPattern::cpu_row_offsets() const {
  TI_ERROR_IF(!arch_is_cpu(arch_),
              "CPU CSR row offsets require a CPU-owned pattern.");
  return cpu_row_offsets_;
}

const std::vector<int32_t> &SparseCsrPattern::cpu_column_indices() const {
  TI_ERROR_IF(!arch_is_cpu(arch_),
              "CPU CSR column indices require a CPU-owned pattern.");
  return cpu_column_indices_;
}

void *SparseCsrPattern::cuda_row_offsets() const {
  TI_ERROR_IF(!arch_is_cuda(arch_),
              "CUDA CSR row offsets require a CUDA-owned pattern.");
  return cuda_row_offsets_;
}

void *SparseCsrPattern::cuda_column_indices() const {
  TI_ERROR_IF(!arch_is_cuda(arch_),
              "CUDA CSR column indices require a CUDA-owned pattern.");
  return cuda_column_indices_;
}

const Ndarray *SparseCsrPattern::vulkan_row_offsets() const {
  TI_ERROR_IF(arch_ != Arch::vulkan,
              "Vulkan CSR row offsets require a Vulkan-owned pattern.");
  return vulkan_row_offsets_;
}

const Ndarray *SparseCsrPattern::vulkan_column_indices() const {
  TI_ERROR_IF(arch_ != Arch::vulkan,
              "Vulkan CSR column indices require a Vulkan-owned pattern.");
  return vulkan_column_indices_;
}

void SparseCsrPattern::retain_operator_reference() {
  operator_references_.fetch_add(1, std::memory_order_relaxed);
}

void SparseCsrPattern::release_operator_reference() {
  const auto previous =
      operator_references_.fetch_sub(1, std::memory_order_relaxed);
  TI_ASSERT(previous > 0);
}

SparsePatternRuntimeStatistics
SparseCsrPattern::debug_runtime_statistics() const {
  SparsePatternRuntimeStatistics result;
  result.backend_family =
      arch_is_cpu(arch_) ? "cpu" : (arch_is_cuda(arch_) ? "cuda" : "vulkan");
  result.storage_format = "csr";
  result.index_dtype = "i32";
  result.value_order = "row_major_compressed";
  result.rows = rows_;
  result.cols = cols_;
  result.nnz = nnz_;
  result.pattern_id = pattern_id_;
  result.pattern_version = 1;
  result.pattern_builds = 1;
  result.operator_references = operator_references();
  result.immutable = true;
  result.pattern_reserved_bytes = pattern_reserved_bytes();
  result.device_to_host_bytes = device_to_host_bytes_;
  result.device_to_device_bytes = device_to_device_bytes_;
  return result;
}

CpuSparseCsrMatrix::CpuSparseCsrMatrix(
    std::shared_ptr<SparseCsrPattern> pattern,
    const Ndarray &values,
    bool pattern_built_for_operator) {
  TI_ERROR_IF(!pattern || !arch_is_cpu(pattern->arch()) ||
                  !pattern->program(),
              "Internal CPU CSR matrices require a CPU-owned pattern.");
  Program *prog = pattern->program();
  const DataType value_dtype = values.get_element_data_type();
  TI_ERROR_IF((value_dtype != PrimitiveType::f32 &&
               value_dtype != PrimitiveType::f64) ||
                  !values.get_element_shape().empty() ||
                  values.get_element_size() != data_type_size(value_dtype),
              "Internal CPU CSR values must be a scalar f32 or f64 ndarray.");
  const std::size_t value_count = static_cast<std::size_t>(pattern->nnz());
  TI_ERROR_IF(values.get_nelement() != value_count,
              "Internal CPU CSR values must contain exactly {} scalar {} "
              "entries, got {}.",
              value_count, data_type_name(value_dtype),
              values.get_nelement());

  const auto source_values = reinterpret_cast<const void *>(
      prog->get_ndarray_data_ptr_as_int(&values));
  if (value_dtype == PrimitiveType::f32) {
    values_f32_.resize(value_count);
    std::memcpy(values_f32_.data(), source_values,
                value_count * sizeof(float32));
  } else {
    values_f64_.resize(value_count);
    std::memcpy(values_f64_.data(), source_values,
                value_count * sizeof(float64));
  }
  program_ = prog;
  rows_ = pattern->num_rows();
  cols_ = pattern->num_cols();
  dtype_ = value_dtype;
  nnz_ = pattern->nnz();
  pattern_ = std::move(pattern);
  if (pattern_built_for_operator) {
    record_pattern_build();
  } else {
    record_pattern_reference();
  }
  pattern_->retain_operator_reference();
}

CpuSparseCsrMatrix::~CpuSparseCsrMatrix() {
  if (pattern_) {
    pattern_->release_operator_reference();
  }
}

namespace {
template <typename T>
void cpu_csr_spmv(const std::vector<int32_t> &row_offsets,
                  const std::vector<int32_t> &column_indices,
                  const T *values,
                  const T *x,
                  T *y,
                  int rows) {
  for (int row = 0; row < rows; ++row) {
    T sum = static_cast<T>(0);
    for (int32_t offset = row_offsets[row];
         offset < row_offsets[row + 1]; ++offset) {
      sum += values[offset] * x[column_indices[offset]];
    }
    y[row] = sum;
  }
}
}  // namespace

void CpuSparseCsrMatrix::nd_spmv(Program *prog,
                                 const Ndarray &x,
                                 const Ndarray &y) {
  TI_ERROR_IF(prog != program_ || !arch_is_cpu(prog->compile_config().arch),
              "Internal CPU CSR SpMV requires its owning CPU Program.");
  auto validate_vector = [&](const char *role, const Ndarray &array,
                             int elements) {
    TI_ERROR_IF(array.get_element_data_type() != dtype_ ||
                    !array.get_element_shape().empty() ||
                    array.shape.size() != 1 ||
                    array.get_nelement() !=
                        static_cast<std::size_t>(elements) ||
                    array.get_element_size() != data_type_size(dtype_),
                "Internal CPU CSR SpMV {} must contain exactly {} scalar {} "
                "entries.",
                role, elements, data_type_name(dtype_));
  };
  validate_vector("input", x, cols_);
  validate_vector("output", y, rows_);
  const auto input = prog->get_ndarray_data_ptr_as_int(&x);
  const auto output = prog->get_ndarray_data_ptr_as_int(&y);
  TI_ERROR_IF(input == output,
              "Internal CPU CSR SpMV input and output must not alias.");
  spmv_cpu_raw(prog, input, output);
}

void CpuSparseCsrMatrix::spmv_cpu_raw(Program *prog,
                                      std::uintptr_t input,
                                      std::uintptr_t output) {
  TI_ERROR_IF(prog != program_ || !arch_is_cpu(prog->compile_config().arch) ||
                  input == 0 || output == 0 || input == output,
              "Internal CPU CSR raw SpMV requires its owning CPU Program "
              "and distinct non-null input/output pointers.");
  std::lock_guard<std::mutex> lock(spmv_mutex_);
  record_spmv_call();
  if (spmv_plan_initialized_) {
    record_spmv_plan_reuse();
  } else {
    record_spmv_plan_build();
    spmv_plan_initialized_ = true;
  }
  if (dtype_ == PrimitiveType::f32) {
    cpu_csr_spmv(pattern_->cpu_row_offsets(),
                 pattern_->cpu_column_indices(), values_f32_.data(),
                 reinterpret_cast<const float32 *>(input),
                 reinterpret_cast<float32 *>(output), rows_);
  } else {
    cpu_csr_spmv(pattern_->cpu_row_offsets(),
                 pattern_->cpu_column_indices(), values_f64_.data(),
                 reinterpret_cast<const float64 *>(input),
                 reinterpret_cast<float64 *>(output), rows_);
  }
}

void CpuSparseCsrMatrix::update_values(Program *prog,
                                       const Ndarray &values) {
  TI_ERROR_IF(prog != program_ || !arch_is_cpu(prog->compile_config().arch),
              "Internal CPU CSR value updates require the owning CPU "
              "Program.");
  const std::size_t value_bytes = data_type_size(dtype_);
  TI_ERROR_IF(values.get_element_data_type() != dtype_ ||
                  !values.get_element_shape().empty() ||
                  values.get_nelement() !=
                      static_cast<std::size_t>(nnz_) ||
                  values.get_element_size() != value_bytes,
              "Internal CPU CSR value updates require exactly {} scalar {} "
              "entries.",
              nnz_, data_type_name(dtype_));
  const auto source = reinterpret_cast<const void *>(
      prog->get_ndarray_data_ptr_as_int(&values));
  std::lock_guard<std::mutex> lock(spmv_mutex_);
  if (dtype_ == PrimitiveType::f32) {
    std::memcpy(values_f32_.data(), source,
                static_cast<std::size_t>(nnz_) * sizeof(float32));
  } else {
    std::memcpy(values_f64_.data(), source,
                static_cast<std::size_t>(nnz_) * sizeof(float64));
  }
  record_numeric_update(static_cast<std::uint64_t>(nnz_) * value_bytes);
}

SparseMatrixRuntimeStatistics
CpuSparseCsrMatrix::debug_runtime_statistics() const {
  std::lock_guard<std::mutex> lock(spmv_mutex_);
  auto result = make_runtime_statistics("cpu", "csr");
  result.provider_name = "forge_cpu_native";
  result.nnz = nnz_;
  result.pattern_reserved_bytes = pattern_->pattern_reserved_bytes();
  const std::uint64_t value_capacity =
      dtype_ == PrimitiveType::f32
          ? static_cast<std::uint64_t>(values_f32_.capacity())
          : static_cast<std::uint64_t>(values_f64_.capacity());
  result.values_reserved_bytes = value_capacity * data_type_size(dtype_);
  result.operator_owned_reserved_bytes =
      result.pattern_reserved_bytes + result.values_reserved_bytes;
  result.operator_exclusive_reserved_bytes = result.values_reserved_bytes;
  result.shared_pattern_id = pattern_->pattern_id();
  result.shared_pattern_operator_references =
      pattern_->operator_references();
  result.pattern_storage_shared = true;
  return result;
}

SparseBsrPattern::SparseBsrPattern(Program *program,
                                   int block_rows,
                                   int block_cols,
                                   int block_size,
                                   const Ndarray &row_offsets,
                                   const Ndarray &column_indices) {
  TI_ERROR_IF(!program,
              "Internal BSR patterns require an active Program.");
  const Arch arch = program->compile_config().arch;
  TI_ERROR_IF(!arch_is_cpu(arch) && !arch_is_cuda(arch) &&
                  arch != Arch::vulkan,
              "Internal BSR patterns support CPU, CUDA, and Vulkan backends, "
              "got {}.",
              arch_name(arch));
  TI_ERROR_IF(block_rows <= 0 || block_cols <= 0,
              "Internal BSR patterns require positive block dimensions, got "
              "{} x {}.",
              block_rows, block_cols);
  TI_ERROR_IF(block_size != 2 && block_size != 3 && block_size != 6 &&
                  block_size != 12,
              "Internal BSR patterns support block sizes 2, 3, 6, and 12, "
              "got {}.",
              block_size);
  TI_ERROR_IF(row_offsets.get_element_data_type() != PrimitiveType::i32 ||
                  !row_offsets.get_element_shape().empty() ||
                  row_offsets.get_nelement() !=
                      static_cast<std::size_t>(block_rows) + 1 ||
                  row_offsets.get_element_size() != sizeof(int32_t),
              "Internal BSR pattern row offsets must contain exactly {} "
              "scalar i32 entries.",
              block_rows + 1);
  TI_ERROR_IF(column_indices.get_element_data_type() != PrimitiveType::i32 ||
                  !column_indices.get_element_shape().empty() ||
                  column_indices.get_element_size() != sizeof(int32_t),
              "Internal BSR pattern column indices must be a scalar i32 "
              "ndarray.");

  const std::size_t block_nnz_size = column_indices.get_nelement();
  TI_ERROR_IF(block_nnz_size == 0,
              "Internal BSR patterns currently require at least one block.");
  TI_ERROR_IF(block_nnz_size >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "Internal BSR pattern block count exceeds the i32 limit.");
  const std::size_t block_width = static_cast<std::size_t>(block_size);
  TI_ERROR_IF(block_nnz_size >
                  std::numeric_limits<std::size_t>::max() / block_width /
                      block_width,
              "Internal BSR pattern value count overflows size_t.");
  const std::size_t value_count =
      block_nnz_size * block_width * block_width;
  TI_ERROR_IF(
      static_cast<std::uint64_t>(block_rows) * block_size >
              static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
          static_cast<std::uint64_t>(block_cols) * block_size >
              static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
          value_count >
              static_cast<std::size_t>(std::numeric_limits<int>::max()),
      "Internal BSR pattern scalar dimensions exceed the i32 SparseMatrix "
      "limit.");

  const auto row_bytes =
      (static_cast<std::size_t>(block_rows) + 1) * sizeof(int32_t);
  const auto column_bytes = block_nnz_size * sizeof(int32_t);
  if (arch_is_cpu(arch)) {
    cpu_row_offsets_.resize(static_cast<std::size_t>(block_rows) + 1);
    cpu_column_indices_.resize(block_nnz_size);
    const auto source_row_offsets = reinterpret_cast<const void *>(
        program->get_ndarray_data_ptr_as_int(&row_offsets));
    const auto source_column_indices = reinterpret_cast<const void *>(
        program->get_ndarray_data_ptr_as_int(&column_indices));
    std::memcpy(cpu_row_offsets_.data(), source_row_offsets, row_bytes);
    std::memcpy(cpu_column_indices_.data(), source_column_indices,
                column_bytes);
    validate_compressed_host_pattern(
        "BSR", block_rows, block_cols, block_nnz_size, cpu_row_offsets_,
        cpu_column_indices_);
  } else if (arch_is_cuda(arch)) {
#if defined(TI_WITH_CUDA)
    std::vector<int32_t> host_row_offsets(
        static_cast<std::size_t>(block_rows) + 1);
    std::vector<int32_t> host_column_indices(block_nnz_size);
    auto source_row_offsets = reinterpret_cast<void *>(
        program->get_ndarray_data_ptr_as_int(&row_offsets));
    auto source_column_indices = reinterpret_cast<void *>(
        program->get_ndarray_data_ptr_as_int(&column_indices));
    CUDADriver::get_instance().memcpy_device_to_host(
        host_row_offsets.data(), source_row_offsets, row_bytes);
    CUDADriver::get_instance().memcpy_device_to_host(
        host_column_indices.data(), source_column_indices, column_bytes);
    validate_compressed_host_pattern(
        "BSR", block_rows, block_cols, block_nnz_size, host_row_offsets,
        host_column_indices);

    void *owned_row_offsets = nullptr;
    void *owned_column_indices = nullptr;
    try {
      CUDADriver::get_instance().malloc(&owned_row_offsets, row_bytes);
      CUDADriver::get_instance().malloc(&owned_column_indices, column_bytes);
      CUDADriver::get_instance().memcpy_device_to_device(
          owned_row_offsets, source_row_offsets, row_bytes);
      CUDADriver::get_instance().memcpy_device_to_device(
          owned_column_indices, source_column_indices, column_bytes);
    } catch (...) {
      if (owned_column_indices) {
        CUDADriver::get_instance().mem_free.call_with_warning(
            owned_column_indices);
      }
      if (owned_row_offsets) {
        CUDADriver::get_instance().mem_free.call_with_warning(
            owned_row_offsets);
      }
      throw;
    }
    cuda_row_offsets_ = owned_row_offsets;
    cuda_column_indices_ = owned_column_indices;
    device_to_host_bytes_ = row_bytes + column_bytes;
    device_to_device_bytes_ = row_bytes + column_bytes;
#else
    TI_NOT_IMPLEMENTED;
#endif
  } else {
#if defined(TI_WITH_VULKAN)
    std::vector<int32_t> host_row_offsets(
        static_cast<std::size_t>(block_rows) + 1);
    std::vector<int32_t> host_column_indices(block_nnz_size);
    program->copy_ndarray_to_host(const_cast<Ndarray *>(&row_offsets),
                                  host_row_offsets.data(), row_bytes);
    program->copy_ndarray_to_host(const_cast<Ndarray *>(&column_indices),
                                  host_column_indices.data(), column_bytes);
    validate_compressed_host_pattern(
        "BSR", block_rows, block_cols, block_nnz_size, host_row_offsets,
        host_column_indices);

    Ndarray *owned_row_offsets = nullptr;
    Ndarray *owned_column_indices = nullptr;
    try {
      owned_row_offsets = program->create_ndarray(
          PrimitiveType::i32, {block_rows + 1}, ExternalArrayLayout::kNull,
          false);
      owned_column_indices = program->create_ndarray(
          PrimitiveType::i32, {static_cast<int>(block_nnz_size)},
          ExternalArrayLayout::kNull, false);
      auto submission_guard =
          program->acquire_runtime_resource_submission_guard();
      const Ndarray *copy_resources[] = {
          owned_row_offsets, &row_offsets, owned_column_indices,
          &column_indices};
      program->retain_ndarrays_for_external_submission(
          copy_resources, std::size(copy_resources));
      program->copy_ndarray_fast(owned_row_offsets,
                                 const_cast<Ndarray *>(&row_offsets));
      program->copy_ndarray_fast(owned_column_indices,
                                 const_cast<Ndarray *>(&column_indices));
    } catch (...) {
      if (owned_column_indices) {
        program->delete_ndarray(owned_column_indices);
      }
      if (owned_row_offsets) {
        program->delete_ndarray(owned_row_offsets);
      }
      throw;
    }
    vulkan_row_offsets_ = owned_row_offsets;
    vulkan_column_indices_ = owned_column_indices;
    device_to_host_bytes_ = row_bytes + column_bytes;
    device_to_device_bytes_ = row_bytes + column_bytes;
#else
    TI_NOT_IMPLEMENTED;
#endif
  }

  program_ = program;
  arch_ = arch;
  rows_ = block_rows * block_size;
  cols_ = block_cols * block_size;
  block_rows_ = block_rows;
  block_cols_ = block_cols;
  block_size_ = block_size;
  block_nnz_ = static_cast<int>(block_nnz_size);
  scalar_nnz_ = static_cast<int>(value_count);
  value_count_ = value_count;
  pattern_id_ =
      next_sparse_pattern_id.fetch_add(1, std::memory_order_relaxed);
}

SparseBsrPattern::~SparseBsrPattern() {
#if defined(TI_WITH_CUDA)
  if (arch_is_cuda(arch_)) {
    if (cuda_column_indices_) {
      CUDADriver::get_instance().mem_free.call_with_warning(
          cuda_column_indices_);
    }
    if (cuda_row_offsets_) {
      CUDADriver::get_instance().mem_free.call_with_warning(cuda_row_offsets_);
    }
  }
#endif
#if defined(TI_WITH_VULKAN)
  if (arch_ == Arch::vulkan && program_) {
    if (vulkan_column_indices_) {
      program_->delete_ndarray(vulkan_column_indices_);
    }
    if (vulkan_row_offsets_) {
      program_->delete_ndarray(vulkan_row_offsets_);
    }
  }
#endif
}

std::uint64_t SparseBsrPattern::pattern_reserved_bytes() const {
  if (arch_is_cpu(arch_)) {
    return (static_cast<std::uint64_t>(cpu_row_offsets_.capacity()) +
            static_cast<std::uint64_t>(cpu_column_indices_.capacity())) *
           sizeof(int32_t);
  }
  return (static_cast<std::uint64_t>(block_rows_) + 1 +
          static_cast<std::uint64_t>(block_nnz_)) *
         sizeof(int32_t);
}

const std::vector<int32_t> &SparseBsrPattern::cpu_row_offsets() const {
  TI_ERROR_IF(!arch_is_cpu(arch_),
              "CPU BSR row offsets require a CPU-owned pattern.");
  return cpu_row_offsets_;
}

const std::vector<int32_t> &SparseBsrPattern::cpu_column_indices() const {
  TI_ERROR_IF(!arch_is_cpu(arch_),
              "CPU BSR column indices require a CPU-owned pattern.");
  return cpu_column_indices_;
}

void *SparseBsrPattern::cuda_row_offsets() const {
  TI_ERROR_IF(!arch_is_cuda(arch_),
              "CUDA BSR row offsets require a CUDA-owned pattern.");
  return cuda_row_offsets_;
}

void *SparseBsrPattern::cuda_column_indices() const {
  TI_ERROR_IF(!arch_is_cuda(arch_),
              "CUDA BSR column indices require a CUDA-owned pattern.");
  return cuda_column_indices_;
}

const Ndarray *SparseBsrPattern::vulkan_row_offsets() const {
  TI_ERROR_IF(arch_ != Arch::vulkan,
              "Vulkan BSR row offsets require a Vulkan-owned pattern.");
  return vulkan_row_offsets_;
}

const Ndarray *SparseBsrPattern::vulkan_column_indices() const {
  TI_ERROR_IF(arch_ != Arch::vulkan,
              "Vulkan BSR column indices require a Vulkan-owned pattern.");
  return vulkan_column_indices_;
}

void SparseBsrPattern::retain_operator_reference() {
  operator_references_.fetch_add(1, std::memory_order_relaxed);
}

void SparseBsrPattern::release_operator_reference() {
  const auto previous =
      operator_references_.fetch_sub(1, std::memory_order_relaxed);
  TI_ASSERT(previous > 0);
}

SparsePatternRuntimeStatistics
SparseBsrPattern::debug_runtime_statistics() const {
  SparsePatternRuntimeStatistics result;
  result.backend_family =
      arch_is_cpu(arch_) ? "cpu" : (arch_is_cuda(arch_) ? "cuda" : "vulkan");
  result.storage_format = "bsr";
  result.index_dtype = "i32";
  result.value_order = "block_row_major_dense_row_major";
  result.rows = rows_;
  result.cols = cols_;
  result.nnz = scalar_nnz_;
  result.block_rows = block_rows_;
  result.block_cols = block_cols_;
  result.block_size = block_size_;
  result.block_nnz = block_nnz_;
  result.pattern_id = pattern_id_;
  result.pattern_version = 1;
  result.pattern_builds = 1;
  result.operator_references = operator_references();
  result.immutable = true;
  result.pattern_reserved_bytes = pattern_reserved_bytes();
  result.device_to_host_bytes = device_to_host_bytes_;
  result.device_to_device_bytes = device_to_device_bytes_;
  return result;
}

CpuSparseBsrMatrix::CpuSparseBsrMatrix(
    Program *prog,
    int block_rows,
    int block_cols,
    int block_size,
    const Ndarray &row_offsets,
    const Ndarray &column_indices,
    const Ndarray &values)
    : CpuSparseBsrMatrix(
          std::make_shared<SparseBsrPattern>(
              prog, block_rows, block_cols, block_size, row_offsets,
              column_indices),
          values,
          true) {
}

CpuSparseBsrMatrix::CpuSparseBsrMatrix(
    std::shared_ptr<SparseBsrPattern> pattern,
    const Ndarray &values,
    bool pattern_built_for_operator) {
  TI_ERROR_IF(!pattern || !arch_is_cpu(pattern->arch()) ||
                  !pattern->program(),
              "Internal CPU BSR matrices require a CPU-owned pattern.");
  Program *prog = pattern->program();
  const DataType value_dtype = values.get_element_data_type();
  TI_ERROR_IF((value_dtype != PrimitiveType::f32 &&
               value_dtype != PrimitiveType::f64) ||
                  !values.get_element_shape().empty() ||
                  values.get_element_size() != data_type_size(value_dtype),
              "Internal CPU BSR values must be a scalar f32 or f64 "
              "ndarray.");

  const std::size_t value_count = pattern->value_count();
  TI_ERROR_IF(values.get_nelement() != value_count,
              "Internal CPU BSR values must contain exactly {} scalar {} "
              "entries for {} dense {} x {} blocks, got {}.",
              value_count, data_type_name(value_dtype), pattern->block_nnz(),
              pattern->block_size(), pattern->block_size(),
              values.get_nelement());

  const auto source_values = reinterpret_cast<const void *>(
      prog->get_ndarray_data_ptr_as_int(&values));
  if (value_dtype == PrimitiveType::f32) {
    values_f32_.resize(value_count);
    std::memcpy(values_f32_.data(), source_values,
                value_count * sizeof(float32));
  } else {
    values_f64_.resize(value_count);
    std::memcpy(values_f64_.data(), source_values,
                value_count * sizeof(float64));
  }
  program_ = prog;
  rows_ = pattern->num_rows();
  cols_ = pattern->num_cols();
  dtype_ = value_dtype;
  block_rows_ = pattern->block_rows();
  block_cols_ = pattern->block_cols();
  block_size_ = pattern->block_size();
  block_nnz_ = pattern->block_nnz();
  scalar_nnz_ = pattern->scalar_nnz();
  value_count_ = value_count;
  pattern_ = std::move(pattern);
  if (pattern_built_for_operator) {
    record_pattern_build();
  } else {
    record_pattern_reference();
  }
  pattern_->retain_operator_reference();
}

CpuSparseBsrMatrix::~CpuSparseBsrMatrix() {
  if (pattern_) {
    pattern_->release_operator_reference();
  }
}

namespace {
template <typename T>
void cpu_bsr_spmv(const std::vector<int32_t> &row_offsets,
                  const std::vector<int32_t> &column_indices,
                  const T *values,
                  const T *x,
                  T *y,
                  int block_rows,
                  int block_size) {
  const std::size_t block_width =
      static_cast<std::size_t>(block_size * block_size);
  for (int block_row = 0; block_row < block_rows; ++block_row) {
    for (int local_row = 0; local_row < block_size; ++local_row) {
      T sum = static_cast<T>(0);
      for (int32_t offset = row_offsets[block_row];
           offset < row_offsets[block_row + 1]; ++offset) {
        const int32_t block_column = column_indices[offset];
        const T *block =
            values + static_cast<std::size_t>(offset) * block_width;
        const T *input =
            x + static_cast<std::size_t>(block_column) * block_size;
        for (int local_column = 0; local_column < block_size;
             ++local_column) {
          sum += block[local_row * block_size + local_column] *
                 input[local_column];
        }
      }
      y[block_row * block_size + local_row] = sum;
    }
  }
}
}  // namespace

void CpuSparseBsrMatrix::nd_spmv(Program *prog,
                                 const Ndarray &x,
                                 const Ndarray &y) {
  TI_ERROR_IF(prog != program_ || !arch_is_cpu(prog->compile_config().arch),
              "Internal CPU BSR SpMV requires its owning CPU Program.");
  auto validate_vector = [&](const char *role, const Ndarray &array,
                             int elements) {
    TI_ERROR_IF(array.get_element_data_type() != dtype_ ||
                    !array.get_element_shape().empty() ||
                    array.shape.size() != 1 ||
                    array.get_nelement() !=
                        static_cast<std::size_t>(elements) ||
                    array.get_element_size() != data_type_size(dtype_),
                "Internal CPU BSR SpMV {} must contain exactly {} scalar {} "
                "entries.",
                role, elements, data_type_name(dtype_));
  };
  validate_vector("input", x, cols_);
  validate_vector("output", y, rows_);
  const auto input = prog->get_ndarray_data_ptr_as_int(&x);
  const auto output = prog->get_ndarray_data_ptr_as_int(&y);
  TI_ERROR_IF(input == output,
              "Internal CPU BSR SpMV input and output must not alias.");
  spmv_cpu_raw(prog, input, output);
}

void CpuSparseBsrMatrix::spmv_cpu_raw(Program *prog,
                                      std::uintptr_t input,
                                      std::uintptr_t output) {
  TI_ERROR_IF(prog != program_ || !arch_is_cpu(prog->compile_config().arch) ||
                  input == 0 || output == 0 || input == output,
              "Internal CPU BSR raw SpMV requires its owning CPU Program "
              "and distinct non-null input/output pointers.");
  std::lock_guard<std::mutex> lock(spmv_mutex_);
  record_spmv_call();
  if (spmv_plan_initialized_) {
    record_spmv_plan_reuse();
  } else {
    record_spmv_plan_build();
    spmv_plan_initialized_ = true;
  }
  if (dtype_ == PrimitiveType::f32) {
    cpu_bsr_spmv(pattern_->cpu_row_offsets(),
                 pattern_->cpu_column_indices(), values_f32_.data(),
                 reinterpret_cast<const float32 *>(input),
                 reinterpret_cast<float32 *>(output), block_rows_,
                 block_size_);
  } else {
    cpu_bsr_spmv(pattern_->cpu_row_offsets(),
                 pattern_->cpu_column_indices(), values_f64_.data(),
                 reinterpret_cast<const float64 *>(input),
                 reinterpret_cast<float64 *>(output), block_rows_,
                 block_size_);
  }
}

void CpuSparseBsrMatrix::update_values(Program *prog,
                                       const Ndarray &values) {
  TI_ERROR_IF(prog != program_ || !arch_is_cpu(prog->compile_config().arch),
              "Internal CPU BSR value updates require the owning CPU "
              "Program.");
  const std::size_t value_bytes = data_type_size(dtype_);
  TI_ERROR_IF(values.get_element_data_type() != dtype_ ||
                  !values.get_element_shape().empty() ||
                  values.get_nelement() != value_count_ ||
                  values.get_element_size() != value_bytes,
              "Internal CPU BSR value updates require exactly {} scalar {} "
              "entries.",
              value_count_, data_type_name(dtype_));
  const auto source = reinterpret_cast<const void *>(
      prog->get_ndarray_data_ptr_as_int(&values));
  std::lock_guard<std::mutex> lock(spmv_mutex_);
  if (dtype_ == PrimitiveType::f32) {
    std::memcpy(values_f32_.data(), source,
                value_count_ * sizeof(float32));
  } else {
    std::memcpy(values_f64_.data(), source,
                value_count_ * sizeof(float64));
  }
  record_numeric_update(value_count_ * value_bytes);
}

SparseMatrixRuntimeStatistics
CpuSparseBsrMatrix::debug_runtime_statistics() const {
  std::lock_guard<std::mutex> lock(spmv_mutex_);
  auto result = make_runtime_statistics("cpu", "bsr");
  result.provider_name = "forge_cpu_native";
  result.nnz = scalar_nnz_;
  result.block_rows = block_rows_;
  result.block_cols = block_cols_;
  result.block_size = block_size_;
  result.block_nnz = block_nnz_;
  result.pattern_reserved_bytes = pattern_->pattern_reserved_bytes();
  const std::uint64_t value_capacity =
      dtype_ == PrimitiveType::f32
          ? static_cast<std::uint64_t>(values_f32_.capacity())
          : static_cast<std::uint64_t>(values_f64_.capacity());
  result.values_reserved_bytes = value_capacity * data_type_size(dtype_);
  result.operator_owned_reserved_bytes =
      result.pattern_reserved_bytes + result.values_reserved_bytes;
  result.operator_exclusive_reserved_bytes = result.values_reserved_bytes;
  result.shared_pattern_id = pattern_->pattern_id();
  result.shared_pattern_operator_references =
      pattern_->operator_references();
  result.pattern_storage_shared = true;
  return result;
}

CuSparseMatrix::CuSparseMatrix(
    std::shared_ptr<SparseCsrPattern> pattern,
    const Ndarray &values,
    bool pattern_built_for_operator) {
#if defined(TI_WITH_CUDA)
  TI_ERROR_IF(!pattern || !arch_is_cuda(pattern->arch()) ||
                  !pattern->program(),
              "Internal CUDA CSR matrices require a CUDA-owned pattern.");
  Program *prog = pattern->program();
  auto &cusparse = CUSPARSEDriver::get_instance();
  if (!cusparse.is_loaded() && !cusparse.load_cusparse()) {
    TI_ERROR("Failed to load cusparse library!");
  }
  TI_ERROR_IF(values.get_element_data_type() != PrimitiveType::f32 ||
                  !values.get_element_shape().empty() ||
                  values.get_element_size() != sizeof(float32),
              "Internal CUDA CSR values must be a scalar f32 ndarray.");
  const std::size_t value_count = static_cast<std::size_t>(pattern->nnz());
  TI_ERROR_IF(values.get_nelement() != value_count,
              "Internal CUDA CSR values must contain exactly {} scalar f32 "
              "entries, got {}.",
              value_count, values.get_nelement());

  const auto value_bytes = value_count * sizeof(float32);
  auto source_values =
      reinterpret_cast<void *>(prog->get_ndarray_data_ptr_as_int(&values));
  void *owned_values = nullptr;
  cusparseSpMatDescr_t matrix = nullptr;
  try {
    CUDADriver::get_instance().malloc(&owned_values, value_bytes);
    CUDADriver::get_instance().memcpy_device_to_device(
        owned_values, source_values, value_bytes);
    cusparse.cpCreateCsr(
        &matrix, pattern->num_rows(), pattern->num_cols(), pattern->nnz(),
        pattern->cuda_row_offsets(), pattern->cuda_column_indices(),
        owned_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  } catch (...) {
    if (matrix) {
      cusparse.cpDestroySpMat.call_with_warning(matrix);
    }
    if (owned_values) {
      CUDADriver::get_instance().mem_free.call_with_warning(owned_values);
    }
    throw;
  }

  rows_ = pattern->num_rows();
  cols_ = pattern->num_cols();
  dtype_ = PrimitiveType::f32;
  nnz_ = pattern->nnz();
  csr_row_ptr_ = pattern->cuda_row_offsets();
  csr_col_ind_ = pattern->cuda_column_indices();
  csr_val_ = owned_values;
  pattern_ = std::move(pattern);
  matrix_ = matrix;
  if (pattern_built_for_operator) {
    record_transfer_bytes(
        0, pattern_->device_to_host_bytes(),
        pattern_->device_to_device_bytes() + value_bytes);
    record_pattern_build();
  } else {
    record_transfer_bytes(0, 0, value_bytes);
    record_pattern_reference();
  }
  pattern_->retain_operator_reference();
#else
  TI_NOT_IMPLEMENTED;
#endif
}

std::unique_ptr<SparseMatrix> make_cu_sparse_matrix(int rows,
                                                    int cols,
                                                    DataType dt) {
  return std::unique_ptr<SparseMatrix>(
      std::make_unique<CuSparseMatrix>(rows, cols, dt));
}

std::unique_ptr<SparseMatrix> make_cu_sparse_matrix(cusparseSpMatDescr_t mat,
                                                    int rows,
                                                    int cols,
                                                    DataType dt,
                                                    void *csr_row_ptr,
                                                    void *csr_col_ind,
                                                    void *csr_val_,
                                                    int nnz,
                                                    std::uint64_t
                                                        device_to_device_bytes) {
  return std::unique_ptr<SparseMatrix>(std::make_unique<CuSparseMatrix>(
      mat, rows, cols, dt, csr_row_ptr, csr_col_ind, csr_val_, nnz,
      device_to_device_bytes));
}

template <typename T>
void build_ndarray_template(SparseMatrix &sm,
                            intptr_t data_ptr,
                            size_t num_triplets) {
  using V = Eigen::Triplet<T>;
  std::vector<V> triplets;
  T *data = reinterpret_cast<T *>(data_ptr);
  for (int i = 0; i < num_triplets; i++) {
    triplets.push_back(
        V(data[i * 3], data[i * 3 + 1], taichi_union_cast<T>(data[i * 3 + 2])));
  }
  sm.build_triplets(static_cast<void *>(&triplets));
}

void make_sparse_matrix_from_ndarray(Program *prog,
                                     SparseMatrix &sm,
                                     const Ndarray &ndarray) {
  std::string sdtype = taichi::lang::data_type_name(sm.get_data_type());
  auto data_ptr = prog->get_ndarray_data_ptr_as_int(&ndarray);
  auto num_triplets = ndarray.get_nelement() * ndarray.get_element_size() / 3;
  if (sdtype == "f32") {
    build_ndarray_template<float32>(sm, data_ptr, num_triplets);
  } else if (sdtype == "f64") {
    build_ndarray_template<float64>(sm, data_ptr, num_triplets);
  } else {
    TI_ERROR("Unsupported sparse matrix data type {}!", sdtype);
  }
}

void CuSparseMatrix::build_csr_from_coo(void *coo_row_ptr,
                                        void *coo_col_ptr,
                                        void *coo_values_ptr,
                                        int nnz) {
#if defined(TI_WITH_CUDA)
  // Step 1: Sort coo first
  cusparseHandle_t cusparse_handle = nullptr;
  CUSPARSEDriver::get_instance().cpCreate(&cusparse_handle);
  cusparseSpVecDescr_t vec_permutation;
  cusparseDnVecDescr_t vec_values;
  void *d_permutation = nullptr, *d_values_sorted = nullptr;
  CUDADriver::get_instance().malloc(&d_permutation, nnz * sizeof(int));
  CUDADriver::get_instance().malloc(&d_values_sorted, nnz * sizeof(float));
  CUSPARSEDriver::get_instance().cpCreateSpVec(
      &vec_permutation, nnz, nnz, d_permutation, d_values_sorted,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  CUSPARSEDriver::get_instance().cpCreateDnVec(&vec_values, nnz, coo_values_ptr,
                                               CUDA_R_32F);
  size_t bufferSize = 0;
  CUSPARSEDriver::get_instance().cpXcoosort_bufferSizeExt(
      cusparse_handle, rows_, cols_, nnz, coo_row_ptr, coo_col_ptr,
      &bufferSize);
  void *dbuffer = nullptr;
  if (bufferSize > 0)
    CUDADriver::get_instance().malloc(&dbuffer, bufferSize);
  // Setup permutation vector to identity
  CUSPARSEDriver::get_instance().cpCreateIdentityPermutation(
      cusparse_handle, nnz, d_permutation);
  CUSPARSEDriver::get_instance().cpXcoosortByRow(cusparse_handle, rows_, cols_,
                                                 nnz, coo_row_ptr, coo_col_ptr,
                                                 d_permutation, dbuffer);
  CUSPARSEDriver::get_instance().cpGather(cusparse_handle, vec_values,
                                          vec_permutation);
  CUDADriver::get_instance().memcpy_device_to_device(
      coo_values_ptr, d_values_sorted, nnz * sizeof(float));
  // Step 2: coo to csr
  void *csr_row_offset_ptr = nullptr;
  CUDADriver::get_instance().malloc(&csr_row_offset_ptr,
                                    sizeof(int) * (rows_ + 1));
  CUSPARSEDriver::get_instance().cpCoo2Csr(
      cusparse_handle, (void *)coo_row_ptr, nnz, rows_,
      (void *)csr_row_offset_ptr, CUSPARSE_INDEX_BASE_ZERO);

  CUSPARSEDriver::get_instance().cpCreateCsr(
      &matrix_, rows_, cols_, nnz, csr_row_offset_ptr, coo_col_ptr,
      coo_values_ptr, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  if (vec_permutation)
    CUSPARSEDriver::get_instance().cpDestroySpVec(vec_permutation);
  if (vec_values)
    CUSPARSEDriver::get_instance().cpDestroyDnVec(vec_values);
  if (cusparse_handle)
    CUSPARSEDriver::get_instance().cpDestroy(cusparse_handle);
  if (coo_row_ptr)
    CUDADriver::get_instance().mem_free(coo_row_ptr);
  if (d_values_sorted)
    CUDADriver::get_instance().mem_free(d_values_sorted);
  if (d_permutation)
    CUDADriver::get_instance().mem_free(d_permutation);
  if (dbuffer)
    CUDADriver::get_instance().mem_free(dbuffer);
  csr_row_ptr_ = csr_row_offset_ptr;
  csr_col_ind_ = coo_col_ptr;
  csr_val_ = coo_values_ptr;
  nnz_ = nnz;
  record_transfer_bytes(0, 0, nnz * sizeof(float));
  record_pattern_build();
#endif
}

void CuSparseMatrix::reset_spmv_resources() {
#if defined(TI_WITH_CUDA)
  if (spmv_vec_x_)
    CUSPARSEDriver::get_instance().cpDestroyDnVec(spmv_vec_x_);
  if (spmv_vec_y_)
    CUSPARSEDriver::get_instance().cpDestroyDnVec(spmv_vec_y_);
  if (spmv_handle_)
    CUSPARSEDriver::get_instance().cpDestroy(spmv_handle_);
  if (spmv_buffer_)
    CUDADriver::get_instance().mem_free(spmv_buffer_);
  spmv_vec_x_ = nullptr;
  spmv_vec_y_ = nullptr;
  spmv_handle_ = nullptr;
  spmv_buffer_ = nullptr;
  spmv_x_ptr_ = 0;
  spmv_y_ptr_ = 0;
  spmv_buffer_size_ = 0;
  spmv_buffer_initialized_ = false;
#endif
}

void CuSparseMatrix::update_values(Program *prog, const Ndarray &values) {
#if defined(TI_WITH_CUDA)
  TI_ERROR_IF(pattern_ && prog != pattern_->program(),
              "Internal shared CUDA CSR value updates require the owning "
              "Program.");
  TI_ERROR_IF(dtype_ != PrimitiveType::f32,
              "CUDA SparseMatrix value-only update supports f32 only.");
  TI_ERROR_IF(values.get_element_data_type() != dtype_ ||
                  !values.get_element_shape().empty() ||
                  values.get_nelement() != static_cast<std::size_t>(nnz_) ||
                  values.get_element_size() != sizeof(float32),
              "CUDA SparseMatrix value-only update expects exactly {} scalar "
               "f32 values in CSR order, got {} element(s) of {} byte(s).",
               nnz_, values.get_nelement(), values.get_element_size());
  record_numeric_update(
      static_cast<std::uint64_t>(nnz_) * sizeof(float32));
  if (nnz_ == 0) {
    return;
  }
  auto src = prog->get_ndarray_data_ptr_as_int(&values);
  std::lock_guard<std::mutex> lock(spmv_mutex_);
  CUDADriver::get_instance().memcpy_device_to_device(
      csr_val_, reinterpret_cast<void *>(src), nnz_ * sizeof(float32));
  record_transfer_bytes(0, 0, nnz_ * sizeof(float32));
#else
  TI_NOT_IMPLEMENTED;
#endif
}

CuSparseMatrix::~CuSparseMatrix() {
#if defined(TI_WITH_CUDA)
  reset_spmv_resources();
  if (matrix_)
    CUSPARSEDriver::get_instance().cpDestroySpMat(matrix_);
  if (csr_row_ptr_ && !pattern_)
    CUDADriver::get_instance().mem_free(csr_row_ptr_);
  if (csr_col_ind_ && !pattern_)
    CUDADriver::get_instance().mem_free(csr_col_ind_);
  if (csr_val_)
    CUDADriver::get_instance().mem_free(csr_val_);
  if (pattern_)
    pattern_->release_operator_reference();
#endif
}

// Reference::https://docs.nvidia.com/cuda/cusparse/index.html#csrgeam2
std::unique_ptr<SparseMatrix> CuSparseMatrix::addition(
    const CuSparseMatrix &other,
    const float alpha,
    const float beta) const {
#if defined(TI_WITH_CUDA)
  // Get information of this matrix: A
  size_t nrows_A = 0, ncols_A = 0, nnz_A = 0;
  void *drow_offsets_A = nullptr, *dcol_indices_A = nullptr,
       *dvalues_A = nullptr;
  cusparseIndexType_t csrRowOffsetsType_A, csrColIndType_A;
  cusparseIndexBase_t idxBase_A;
  cudaDataType valueType_A;
  TI_ASSERT(matrix_ != nullptr);

  CUSPARSEDriver::get_instance().cpCsrGet(
      matrix_, &nrows_A, &ncols_A, &nnz_A, &drow_offsets_A, &dcol_indices_A,
      &dvalues_A, &csrRowOffsetsType_A, &csrColIndType_A, &idxBase_A,
      &valueType_A);
  // Get information of other matrix: B
  size_t nrows_B = 0, ncols_B = 0, nnz_B = 0;
  void *drow_offsets_B = nullptr, *dcol_indices_B = nullptr,
       *dvalues_B = nullptr;
  cusparseIndexType_t csrRowOffsetsType_B, csrColIndType_B;
  cusparseIndexBase_t idxBase_B;
  cudaDataType valueType_B;
  CUSPARSEDriver::get_instance().cpCsrGet(
      other.matrix_, &nrows_B, &ncols_B, &nnz_B, &drow_offsets_B,
      &dcol_indices_B, &dvalues_B, &csrRowOffsetsType_B, &csrColIndType_B,
      &idxBase_B, &valueType_B);

  // Create sparse matrix: C
  int *drow_offsets_C = nullptr;
  int *dcol_indices_C = nullptr;
  float *dvalues_C = nullptr;
  cusparseMatDescr_t descrA = nullptr, descrB = nullptr, descrC = nullptr;
  CUSPARSEDriver::get_instance().cpCreateMatDescr(&descrA);
  CUSPARSEDriver::get_instance().cpCreateMatDescr(&descrB);
  CUSPARSEDriver::get_instance().cpCreateMatDescr(&descrC);
  CUSPARSEDriver::get_instance().cpSetMatType(descrA,
                                              CUSPARSE_MATRIX_TYPE_GENERAL);
  CUSPARSEDriver::get_instance().cpSetMatType(descrB,
                                              CUSPARSE_MATRIX_TYPE_GENERAL);
  CUSPARSEDriver::get_instance().cpSetMatType(descrC,
                                              CUSPARSE_MATRIX_TYPE_GENERAL);
  CUSPARSEDriver::get_instance().cpSetMatIndexBase(descrC,
                                                   CUSPARSE_INDEX_BASE_ZERO);
  CUSPARSEDriver::get_instance().cpSetMatIndexBase(descrA,
                                                   CUSPARSE_INDEX_BASE_ZERO);
  CUSPARSEDriver::get_instance().cpSetMatIndexBase(descrB,
                                                   CUSPARSE_INDEX_BASE_ZERO);

  // Start to do addition
  cusparseHandle_t cusparse_handle;
  CUSPARSEDriver::get_instance().cpCreate(&cusparse_handle);
  // alpha, nnzTotalDevHostPtr points to host memory
  size_t BufferSizeInBytes;
  char *buffer = nullptr;
  int nnzC;
  int *nnzTotalDevHostPtr = &nnzC;
  CUSPARSEDriver::get_instance().cpSetPointerMode(cusparse_handle,
                                                  CUSPARSE_POINTER_MODE_HOST);
  CUDADriver::get_instance().malloc((void **)(&drow_offsets_C),
                                    sizeof(int) * (nrows_A + 1));
  // Prepare buffer
  CUSPARSEDriver::get_instance().cpScsrgeam2_bufferSizeExt(
      cusparse_handle, nrows_A, ncols_A, (void *)(&alpha), descrA, nnz_A,
      dvalues_A, drow_offsets_A, dcol_indices_A, (void *)&beta, descrB, nnz_B,
      dvalues_B, drow_offsets_B, dcol_indices_B, descrC, dvalues_C,
      drow_offsets_C, dcol_indices_C, &BufferSizeInBytes);

  if (BufferSizeInBytes > 0)
    CUDADriver::get_instance().malloc((void **)(&buffer), BufferSizeInBytes);

  // Determine drow_offsets_C and the total number of nonzero elements.
  CUSPARSEDriver::get_instance().cpXcsrgeam2Nnz(
      cusparse_handle, nrows_A, ncols_A, descrA, nnz_A, drow_offsets_A,
      dcol_indices_A, descrB, nnz_B, drow_offsets_B, dcol_indices_B, descrC,
      drow_offsets_C, nnzTotalDevHostPtr, buffer);

  int baseC;
  if (nullptr != nnzTotalDevHostPtr) {
    nnzC = *nnzTotalDevHostPtr;
  } else {
    CUDADriver::get_instance().memcpy_device_to_host(
        (void *)(&nnzC), (void *)(drow_offsets_C + nrows_A), sizeof(int));
    CUDADriver::get_instance().memcpy_device_to_host(
        (void *)(&baseC), (void *)(drow_offsets_C), sizeof(int));
    nnzC -= baseC;
  }

  CUDADriver::get_instance().malloc((void **)&dcol_indices_C,
                                    sizeof(int) * nnzC);
  CUDADriver::get_instance().malloc((void **)&dvalues_C, sizeof(float) * nnzC);

  CUSPARSEDriver::get_instance().cpScsrgeam2(
      cusparse_handle, nrows_A, ncols_A, (void *)(&alpha), descrA, nnz_A,
      dvalues_A, drow_offsets_A, dcol_indices_A, (void *)(&beta), descrB, nnz_B,
      dvalues_B, drow_offsets_B, dcol_indices_B, descrC, dvalues_C,
      drow_offsets_C, dcol_indices_C, buffer);

  cusparseSpMatDescr_t matrix_C;
  CUSPARSEDriver::get_instance().cpCreateCsr(
      &matrix_C, rows_, cols_, nnzC, drow_offsets_C, dcol_indices_C, dvalues_C,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
      CUDA_R_32F);

  CUSPARSEDriver::get_instance().cpDestroy(cusparse_handle);
  CUSPARSEDriver::get_instance().cpDestroyMatDescr(descrA);
  CUSPARSEDriver::get_instance().cpDestroyMatDescr(descrB);
  CUSPARSEDriver::get_instance().cpDestroyMatDescr(descrC);
  CUDADriver::get_instance().mem_free(buffer);
  return make_cu_sparse_matrix(matrix_C, rows_, cols_, PrimitiveType::f32,
                               drow_offsets_C, dcol_indices_C, dvalues_C, nnzC);
  ;
#else
  TI_NOT_IMPLEMENTED;
  return std::unique_ptr<SparseMatrix>();
#endif
}

std::unique_ptr<SparseMatrix> CuSparseMatrix::matmul(
    const CuSparseMatrix &other) const {
#if defined(TI_WITH_CUDA)
  return gemm(other, 1.0f, 0.0f);
#else
  TI_NOT_IMPLEMENTED;
  return std::unique_ptr<SparseMatrix>();
#endif
}

// Reference:
// https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuSPARSE/spgemm
std::unique_ptr<SparseMatrix> CuSparseMatrix::gemm(const CuSparseMatrix &other,
                                                   const float alpha,
                                                   const float beta) const {
#if defined(TI_WITH_CUDA)
  cusparseHandle_t handle = nullptr;
  CUSPARSEDriver::get_instance().cpCreate(&handle);
  cusparseOperation_t op_A = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseOperation_t op_B = CUSPARSE_OPERATION_NON_TRANSPOSE;

  size_t nrows_A = rows_;
  size_t ncols_B = other.cols_;
  auto mat_A = matrix_;
  auto mat_B = other.matrix_;

  // 1. create resulting matrix `C`
  cusparseSpMatDescr_t mat_C;
  CUSPARSEDriver::get_instance().cpCreateCsr(
      &mat_C, nrows_A, ncols_B, 0, nullptr, nullptr, nullptr,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
      CUDA_R_32F);

  // 2. create gemm descr
  cusparseSpGEMMDescr_t spgemm_desc;
  CUSPARSEDriver::get_instance().cpCreateSpGEMM(&spgemm_desc);

  // 3. ask buffer_size1 bytes for external memory
  void *d_buffer1 = nullptr;
  size_t buffer_size1 = 0;
  CUSPARSEDriver::get_instance().cpSpGEMM_workEstimation(
      handle, op_A, op_B, &alpha, this->matrix_, other.matrix_, &beta, mat_C,
      CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemm_desc, &buffer_size1, nullptr);
  CUDADriver::get_instance().malloc((void **)&d_buffer1, buffer_size1);
  // 4. inspect the matrices A and B to understand the memory requirement for
  // the next step
  CUSPARSEDriver::get_instance().cpSpGEMM_workEstimation(
      handle, op_A, op_B, &alpha, this->matrix_, other.matrix_, &beta, mat_C,
      CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemm_desc, &buffer_size1,
      d_buffer1);

  // 5. ask buffer_size2 bytes for external memory
  size_t buffer_size2 = 0;
  CUSPARSEDriver::get_instance().cpSpGEMM_compute(
      handle, op_A, op_B, &alpha, mat_A, mat_B, &beta, mat_C, CUDA_R_32F,
      CUSPARSE_SPGEMM_DEFAULT, spgemm_desc, &buffer_size2, nullptr);
  void *d_buffer2 = nullptr;
  CUDADriver::get_instance().malloc((void **)&d_buffer2, buffer_size2);

  // 6. compute the intermediate product of A * B
  CUSPARSEDriver::get_instance().cpSpGEMM_compute(
      handle, op_A, op_B, &alpha, mat_A, mat_B, &beta, mat_C, CUDA_R_32F,
      CUSPARSE_SPGEMM_DEFAULT, spgemm_desc, &buffer_size2, d_buffer2);

  // 7. get info of matrix C
  size_t nrows_C, cols_C, nnz_C;
  CUSPARSEDriver::get_instance().cpGetSize(mat_C, &nrows_C, &cols_C, &nnz_C);

  // 8. allocate matric C
  int *d_csr_row_ptr_C, *d_csr_col_ind_C;
  float *d_values_C;
  CUDADriver::get_instance().malloc((void **)&d_csr_row_ptr_C,
                                    (nrows_A + 1) * sizeof(int));
  CUDADriver::get_instance().malloc((void **)&d_csr_col_ind_C,
                                    nnz_C * sizeof(int));
  CUDADriver::get_instance().malloc((void **)&d_values_C,
                                    nnz_C * sizeof(float));

  // 9. update matrix C with new pointers
  CUSPARSEDriver::get_instance().cpCsrSetPointers(mat_C, d_csr_row_ptr_C,
                                                  d_csr_col_ind_C, d_values_C);

  // 10. copy the final products of C.
  CUSPARSEDriver::get_instance().cpSpGEMM_copy(
      handle, op_A, op_B, &alpha, mat_A, mat_B, &beta, mat_C, CUDA_R_32F,
      CUSPARSE_SPGEMM_DEFAULT, spgemm_desc);

  CUDADriver::get_instance().mem_free(d_buffer1);
  CUDADriver::get_instance().mem_free(d_buffer2);
  CUSPARSEDriver::get_instance().cpDestroy(handle);
  CUSPARSEDriver::get_instance().cpDestroySpGEMM(spgemm_desc);

  return make_cu_sparse_matrix(mat_C, nrows_A, ncols_B, PrimitiveType::f32,
                               d_csr_row_ptr_C, d_csr_col_ind_C, d_values_C,
                               nnz_C);
#else
  TI_NOT_IMPLEMENTED;
  return std::unique_ptr<SparseMatrix>();
#endif
}

// Convert CSR to CSC format using routine `Csr2cscEx2`
// to implement transpose.
// Reference
// https://stackoverflow.com/questions/57368010/how-to-transpose-a-sparse-matrix-in-cusparse
std::unique_ptr<SparseMatrix> CuSparseMatrix::transpose() const {
#if defined(TI_WITH_CUDA)
  cusparseHandle_t handle;
  CUSPARSEDriver::get_instance().cpCreate(&handle);
  size_t nrows_A, ncols_A, nnz;
  void *d_csr_val = nullptr, *d_csr_val_AT = nullptr;
  int *d_csr_row_ptr = nullptr, *d_csr_col_ind = nullptr;
  int *d_csr_row_ptr_AT = nullptr, *d_csr_col_ptr_AT = nullptr;
  cusparseIndexType_t csr_row_otr_type, csr_col_otr_type;
  cusparseIndexBase_t idx_base_type;
  cudaDataType value_type;
  size_t buffer_size;

  // 1. get pointers of A
  CUSPARSEDriver::get_instance().cpCsrGet(
      matrix_, &nrows_A, &ncols_A, &nnz, (void **)&d_csr_row_ptr,
      (void **)&d_csr_col_ind, (void **)&d_csr_val, &csr_row_otr_type,
      &csr_col_otr_type, &idx_base_type, &value_type);

  // 2. ask bufer size for Csr2cscEx2
  CUSPARSEDriver::get_instance().cpCsr2cscEx2_bufferSize(
      handle, nrows_A, ncols_A, nnz, (void *)&d_csr_val, (int *)&d_csr_row_ptr,
      (int *)&d_csr_col_ind, (void *)&d_csr_val_AT, (int *)&d_csr_row_ptr_AT,
      (int *)&d_csr_col_ptr_AT, CUDA_R_32F, CUSPARSE_ACTION_NUMERIC,
      CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, &buffer_size);
  void *buffer = nullptr;
  CUDADriver::get_instance().malloc((void **)&buffer, buffer_size);

  CUDADriver::get_instance().malloc((void **)&d_csr_val_AT,
                                    nnz * sizeof(float));
  CUDADriver::get_instance().malloc((void **)&d_csr_row_ptr_AT,
                                    (ncols_A + 1) * sizeof(int));
  CUDADriver::get_instance().malloc((void **)&d_csr_col_ptr_AT,
                                    nnz * sizeof(int));

  // 3. execute Csr2cscEx2
  CUSPARSEDriver::get_instance().cpCsr2cscEx2(
      handle, nrows_A, ncols_A, nnz, d_csr_val, d_csr_row_ptr, d_csr_col_ind,
      d_csr_val_AT, d_csr_row_ptr_AT, d_csr_col_ptr_AT, CUDA_R_32F,
      CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1,
      buffer);

  // 4. create AT.
  cusparseSpMatDescr_t mat_AT;
  CUSPARSEDriver::get_instance().cpCreateCsr(
      &mat_AT, ncols_A, nrows_A, nnz, (void *)d_csr_row_ptr_AT,
      (void *)d_csr_col_ptr_AT, (void *)d_csr_val_AT, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  CUDADriver::get_instance().mem_free(buffer);
  CUSPARSEDriver::get_instance().cpDestroy(handle);
  return make_cu_sparse_matrix(mat_AT, ncols_A, nrows_A, PrimitiveType::f32,
                               d_csr_row_ptr_AT, d_csr_col_ptr_AT, d_csr_val_AT,
                               nnz);
#else
  TI_NOT_IMPLEMENTED;
  return std::unique_ptr<SparseMatrix>();
#endif
}

void CuSparseMatrix::spmv(size_t dX, size_t dY) {
#if defined(TI_WITH_CUDA)
  std::lock_guard<std::mutex> lock(spmv_mutex_);
  record_spmv_call();
  if (!spmv_handle_) {
    CUSPARSEDriver::get_instance().cpCreate(&spmv_handle_);
    record_spmv_handle_creation();
  }
  if (!spmv_vec_x_ || spmv_x_ptr_ != dX) {
    const bool rebind = spmv_vec_x_ != nullptr;
    if (spmv_vec_x_)
      CUSPARSEDriver::get_instance().cpDestroyDnVec(spmv_vec_x_);
    CUSPARSEDriver::get_instance().cpCreateDnVec(&spmv_vec_x_, cols_,
                                                 (void *)dX, CUDA_R_32F);
    spmv_x_ptr_ = dX;
    record_dense_vector_descriptor_creation(rebind);
  }
  if (!spmv_vec_y_ || spmv_y_ptr_ != dY) {
    const bool rebind = spmv_vec_y_ != nullptr;
    if (spmv_vec_y_)
      CUSPARSEDriver::get_instance().cpDestroyDnVec(spmv_vec_y_);
    CUSPARSEDriver::get_instance().cpCreateDnVec(&spmv_vec_y_, rows_,
                                                 (void *)dY, CUDA_R_32F);
    spmv_y_ptr_ = dY;
    record_dense_vector_descriptor_creation(rebind);
  }

  float alpha = 1.0f, beta = 0.0f;
  if (!spmv_buffer_initialized_) {
    record_spmv_plan_build();
    CUSPARSEDriver::get_instance().cpSpMV_bufferSize(
        spmv_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matrix_,
        spmv_vec_x_, &beta, spmv_vec_y_, CUDA_R_32F,
        CUSPARSE_SPMV_CSR_ALG1, &spmv_buffer_size_);
    if (spmv_buffer_size_ > 0) {
      CUDADriver::get_instance().malloc(&spmv_buffer_, spmv_buffer_size_);
      record_spmv_workspace_allocation();
    }
    spmv_buffer_initialized_ = true;
  } else {
    record_spmv_plan_reuse();
  }
  CUSPARSEDriver::get_instance().cpSpMV(
      spmv_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matrix_,
      spmv_vec_x_, &beta, spmv_vec_y_, CUDA_R_32F,
      CUSPARSE_SPMV_CSR_ALG1, spmv_buffer_);
#endif
}

SparseMatrixRuntimeStatistics CuSparseMatrix::debug_runtime_statistics() const {
#if defined(TI_WITH_CUDA)
  std::lock_guard<std::mutex> lock(spmv_mutex_);
  auto result = make_runtime_statistics("cuda", "csr");
  const auto provider = CUSPARSEDriver::get_instance().capabilities();
  result.provider_name = "cusparse";
  result.provider_version_major = provider.library_version_major;
  result.provider_version_minor = provider.library_version_minor;
  result.provider_version_patch = provider.library_version_patch;
  result.provider_bsr_descriptor_available =
      provider.bsr_descriptor_available;
  result.provider_generic_bsr_spmv_available =
      provider.generic_bsr_spmv_available;
  result.nnz = nnz_;
  result.pattern_reserved_bytes =
      (static_cast<std::uint64_t>(rows_) + 1 +
       static_cast<std::uint64_t>(nnz_)) *
      sizeof(int);
  result.values_reserved_bytes =
      static_cast<std::uint64_t>(nnz_) * sizeof(float32);
  result.spmv_workspace_reserved_bytes =
      spmv_buffer_initialized_ ? spmv_buffer_size_ : 0;
  result.operator_owned_reserved_bytes = result.pattern_reserved_bytes +
                                         result.values_reserved_bytes +
                                         result.spmv_workspace_reserved_bytes;
  if (pattern_) {
    result.operator_exclusive_reserved_bytes =
        result.values_reserved_bytes + result.spmv_workspace_reserved_bytes;
    result.shared_pattern_id = pattern_->pattern_id();
    result.shared_pattern_operator_references =
        pattern_->operator_references();
    result.pattern_storage_shared = true;
  }
  result.matrix_descriptor_count = matrix_ != nullptr ? 1 : 0;
  result.dense_vector_descriptor_count =
      (spmv_vec_x_ != nullptr ? 1 : 0) + (spmv_vec_y_ != nullptr ? 1 : 0);
  result.spmv_handle_count = spmv_handle_ != nullptr ? 1 : 0;
  return result;
#else
  return make_runtime_statistics("cuda", "csr");
#endif
}

CuSparseBsrMatrix::CuSparseBsrMatrix(Program *prog,
                                     int block_rows,
                                     int block_cols,
                                     int block_size,
                                     const Ndarray &row_offsets,
                                     const Ndarray &column_indices,
                                     const Ndarray &values)
    : CuSparseBsrMatrix(
          std::make_shared<SparseBsrPattern>(
              prog, block_rows, block_cols, block_size, row_offsets,
              column_indices),
          values,
          true) {
}

CuSparseBsrMatrix::CuSparseBsrMatrix(
    std::shared_ptr<SparseBsrPattern> pattern,
    const Ndarray &values,
    bool pattern_built_for_operator) {
#if defined(TI_WITH_CUDA)
  TI_ERROR_IF(!pattern || !arch_is_cuda(pattern->arch()) ||
                  !pattern->program(),
              "Internal BSR matrices require a CUDA-owned pattern.");
  Program *prog = pattern->program();
  auto &cusparse = CUSPARSEDriver::get_instance();
  if (!cusparse.is_loaded() && !cusparse.load_cusparse()) {
    TI_ERROR("Failed to load cusparse library!");
  }
  const auto provider = cusparse.capabilities();
  TI_ERROR_IF(!provider.generic_bsr_spmv_available,
              "The loaded cuSPARSE provider does not support generic BSR "
              "SpMV (requires cusparseCreateBsr and cuSPARSE >= 12.6.3).");
  TI_ERROR_IF(values.get_element_data_type() != PrimitiveType::f32 ||
                  !values.get_element_shape().empty() ||
                  values.get_element_size() != sizeof(float32),
              "Internal BSR values must be a scalar f32 ndarray.");

  const auto value_count = pattern->value_count();
  TI_ERROR_IF(values.get_nelement() != value_count,
              "Internal BSR values must contain exactly {} scalar f32 "
              "entries for {} dense {} x {} blocks, got {}.",
              value_count, pattern->block_nnz(), pattern->block_size(),
              pattern->block_size(), values.get_nelement());

  const auto value_bytes = value_count * sizeof(float32);
  auto source_values =
      reinterpret_cast<void *>(prog->get_ndarray_data_ptr_as_int(&values));

  void *owned_values = nullptr;
  cusparseSpMatDescr_t matrix = nullptr;
  try {
    CUDADriver::get_instance().malloc(&owned_values, value_bytes);
    CUDADriver::get_instance().memcpy_device_to_device(
        owned_values, source_values, value_bytes);
    cusparse.cpCreateBsr(
        &matrix, pattern->block_rows(), pattern->block_cols(),
        pattern->block_nnz(), pattern->block_size(), pattern->block_size(),
        pattern->cuda_row_offsets(), pattern->cuda_column_indices(),
        owned_values,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_32F, CUSPARSE_ORDER_ROW);
  } catch (...) {
    if (matrix)
      cusparse.cpDestroySpMat.call_with_warning(matrix);
    if (owned_values)
      CUDADriver::get_instance().mem_free.call_with_warning(owned_values);
    throw;
  }

  rows_ = pattern->num_rows();
  cols_ = pattern->num_cols();
  dtype_ = PrimitiveType::f32;
  block_rows_ = pattern->block_rows();
  block_cols_ = pattern->block_cols();
  block_size_ = pattern->block_size();
  block_nnz_ = pattern->block_nnz();
  scalar_nnz_ = pattern->scalar_nnz();
  value_count_ = value_count;
  pattern_ = std::move(pattern);
  matrix_ = matrix;
  values_ = owned_values;
  if (pattern_built_for_operator) {
    record_transfer_bytes(
        0, pattern_->device_to_host_bytes(),
        pattern_->device_to_device_bytes() + value_bytes);
    record_pattern_build();
  } else {
    record_transfer_bytes(0, 0, value_bytes);
    record_pattern_reference();
  }
  pattern_->retain_operator_reference();
#else
  TI_NOT_IMPLEMENTED;
#endif
}

void CuSparseBsrMatrix::reset_spmv_resources() {
#if defined(TI_WITH_CUDA)
  if (spmv_vec_x_)
    CUSPARSEDriver::get_instance().cpDestroyDnVec(spmv_vec_x_);
  if (spmv_vec_y_)
    CUSPARSEDriver::get_instance().cpDestroyDnVec(spmv_vec_y_);
  if (spmv_handle_)
    CUSPARSEDriver::get_instance().cpDestroy(spmv_handle_);
  if (spmv_buffer_)
    CUDADriver::get_instance().mem_free(spmv_buffer_);
  spmv_vec_x_ = nullptr;
  spmv_vec_y_ = nullptr;
  spmv_handle_ = nullptr;
  spmv_buffer_ = nullptr;
  spmv_x_ptr_ = 0;
  spmv_y_ptr_ = 0;
  spmv_buffer_size_ = 0;
  spmv_buffer_initialized_ = false;
#endif
}

CuSparseBsrMatrix::~CuSparseBsrMatrix() {
#if defined(TI_WITH_CUDA)
  reset_spmv_resources();
  if (matrix_)
    CUSPARSEDriver::get_instance().cpDestroySpMat(matrix_);
  if (values_)
    CUDADriver::get_instance().mem_free(values_);
  if (pattern_)
    pattern_->release_operator_reference();
#endif
}

void CuSparseBsrMatrix::spmv(size_t dX, size_t dY) {
#if defined(TI_WITH_CUDA)
  std::lock_guard<std::mutex> lock(spmv_mutex_);
  record_spmv_call();
  if (!spmv_handle_) {
    CUSPARSEDriver::get_instance().cpCreate(&spmv_handle_);
    record_spmv_handle_creation();
  }
  if (!spmv_vec_x_ || spmv_x_ptr_ != dX) {
    const bool rebind = spmv_vec_x_ != nullptr;
    if (spmv_vec_x_)
      CUSPARSEDriver::get_instance().cpDestroyDnVec(spmv_vec_x_);
    CUSPARSEDriver::get_instance().cpCreateDnVec(&spmv_vec_x_, cols_,
                                                 reinterpret_cast<void *>(dX),
                                                 CUDA_R_32F);
    spmv_x_ptr_ = dX;
    record_dense_vector_descriptor_creation(rebind);
  }
  if (!spmv_vec_y_ || spmv_y_ptr_ != dY) {
    const bool rebind = spmv_vec_y_ != nullptr;
    if (spmv_vec_y_)
      CUSPARSEDriver::get_instance().cpDestroyDnVec(spmv_vec_y_);
    CUSPARSEDriver::get_instance().cpCreateDnVec(&spmv_vec_y_, rows_,
                                                 reinterpret_cast<void *>(dY),
                                                 CUDA_R_32F);
    spmv_y_ptr_ = dY;
    record_dense_vector_descriptor_creation(rebind);
  }

  float alpha = 1.0f;
  float beta = 0.0f;
  if (!spmv_buffer_initialized_) {
    record_spmv_plan_build();
    CUSPARSEDriver::get_instance().cpSpMV_bufferSize(
        spmv_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matrix_,
        spmv_vec_x_, &beta, spmv_vec_y_, CUDA_R_32F,
        CUSPARSE_SPMV_BSR_ALG1, &spmv_buffer_size_);
    if (spmv_buffer_size_ > 0) {
      CUDADriver::get_instance().malloc(&spmv_buffer_, spmv_buffer_size_);
      record_spmv_workspace_allocation();
    }
    spmv_buffer_initialized_ = true;
  } else {
    record_spmv_plan_reuse();
  }
  CUSPARSEDriver::get_instance().cpSpMV(
      spmv_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matrix_,
      spmv_vec_x_, &beta, spmv_vec_y_, CUDA_R_32F,
      CUSPARSE_SPMV_BSR_ALG1, spmv_buffer_);
#else
  TI_NOT_IMPLEMENTED;
#endif
}

void CuSparseBsrMatrix::nd_spmv(Program *prog,
                                const Ndarray &x,
                                const Ndarray &y) {
#if defined(TI_WITH_CUDA)
  TI_ERROR_IF(!prog || prog != pattern_->program() ||
                  !arch_is_cuda(prog->compile_config().arch),
              "Internal BSR SpMV requires its owning CUDA Program.");
  TI_ERROR_IF(x.get_element_data_type() != PrimitiveType::f32 ||
                  !x.get_element_shape().empty() ||
                  x.get_nelement() != static_cast<std::size_t>(cols_) ||
                  y.get_element_data_type() != PrimitiveType::f32 ||
                  !y.get_element_shape().empty() ||
                  y.get_nelement() != static_cast<std::size_t>(rows_),
              "Internal BSR SpMV expects scalar f32 vectors with shapes ({},) "
              "and ({},).",
              cols_, rows_);
  spmv(prog->get_ndarray_data_ptr_as_int(&x),
       prog->get_ndarray_data_ptr_as_int(&y));
#else
  TI_NOT_IMPLEMENTED;
#endif
}

void CuSparseBsrMatrix::update_values(Program *prog, const Ndarray &values) {
#if defined(TI_WITH_CUDA)
  TI_ERROR_IF(!prog || prog != pattern_->program() ||
                  !arch_is_cuda(prog->compile_config().arch),
              "Internal BSR value updates require the owning CUDA Program.");
  TI_ERROR_IF(values.get_element_data_type() != PrimitiveType::f32 ||
                  !values.get_element_shape().empty() ||
                  values.get_nelement() != value_count_ ||
                  values.get_element_size() != sizeof(float32),
              "Internal BSR value update expects exactly {} scalar f32 "
              "entries in block-row-major order.",
              value_count_);
  const auto bytes = value_count_ * sizeof(float32);
  auto source =
      reinterpret_cast<void *>(prog->get_ndarray_data_ptr_as_int(&values));
  std::lock_guard<std::mutex> lock(spmv_mutex_);
  CUDADriver::get_instance().memcpy_device_to_device(values_, source, bytes);
  record_numeric_update(bytes);
  record_transfer_bytes(0, 0, bytes);
#else
  TI_NOT_IMPLEMENTED;
#endif
}

SparseMatrixRuntimeStatistics
CuSparseBsrMatrix::debug_runtime_statistics() const {
#if defined(TI_WITH_CUDA)
  std::lock_guard<std::mutex> lock(spmv_mutex_);
  auto result = make_runtime_statistics("cuda", "bsr");
  const auto provider = CUSPARSEDriver::get_instance().capabilities();
  result.provider_name = "cusparse";
  result.provider_version_major = provider.library_version_major;
  result.provider_version_minor = provider.library_version_minor;
  result.provider_version_patch = provider.library_version_patch;
  result.provider_bsr_descriptor_available =
      provider.bsr_descriptor_available;
  result.provider_generic_bsr_spmv_available =
      provider.generic_bsr_spmv_available;
  result.nnz = scalar_nnz_;
  result.block_rows = block_rows_;
  result.block_cols = block_cols_;
  result.block_size = block_size_;
  result.block_nnz = block_nnz_;
  result.pattern_reserved_bytes = pattern_->pattern_reserved_bytes();
  result.values_reserved_bytes =
      static_cast<std::uint64_t>(value_count_) * sizeof(float32);
  result.spmv_workspace_reserved_bytes =
      spmv_buffer_initialized_ ? spmv_buffer_size_ : 0;
  result.operator_owned_reserved_bytes = result.pattern_reserved_bytes +
                                         result.values_reserved_bytes +
                                         result.spmv_workspace_reserved_bytes;
  result.operator_exclusive_reserved_bytes =
      result.values_reserved_bytes + result.spmv_workspace_reserved_bytes;
  result.shared_pattern_id = pattern_->pattern_id();
  result.shared_pattern_operator_references =
      pattern_->operator_references();
  result.pattern_storage_shared = true;
  result.matrix_descriptor_count = matrix_ != nullptr ? 1 : 0;
  result.dense_vector_descriptor_count =
      (spmv_vec_x_ != nullptr ? 1 : 0) + (spmv_vec_y_ != nullptr ? 1 : 0);
  result.spmv_handle_count = spmv_handle_ != nullptr ? 1 : 0;
  return result;
#else
  return make_runtime_statistics("cuda", "bsr");
#endif
}

VulkanSparseMatrix::VulkanSparseMatrix(
    std::shared_ptr<SparseCsrPattern> pattern,
    const Ndarray &values,
    bool pattern_built_for_operator) {
#if defined(TI_WITH_VULKAN)
  TI_ERROR_IF(!pattern || pattern->arch() != Arch::vulkan ||
                  !pattern->program(),
              "Internal Vulkan CSR matrices require a Vulkan-owned pattern.");
  Program *prog = pattern->program();
  TI_ERROR_IF(!prog->vulkan_sparse_algebra_available(),
              "Vulkan fixed-pattern sparse algebra is unavailable.");
  TI_ERROR_IF(values.get_element_data_type() != PrimitiveType::f32 ||
                  !values.get_element_shape().empty() ||
                  values.get_element_size() != sizeof(float32),
              "Internal Vulkan CSR values must be a scalar f32 ndarray.");
  const std::size_t value_count = static_cast<std::size_t>(pattern->nnz());
  TI_ERROR_IF(values.get_nelement() != value_count,
              "Internal Vulkan CSR values must contain exactly {} scalar "
              "f32 entries, got {}.",
              value_count, values.get_nelement());

  const auto value_bytes = value_count * sizeof(float32);
  Ndarray *owned_values = nullptr;
  try {
    owned_values = prog->create_ndarray(
        PrimitiveType::f32, {static_cast<int>(value_count)},
        ExternalArrayLayout::kNull, false);
    auto submission_guard =
        prog->acquire_runtime_resource_submission_guard();
    const Ndarray *copy_resources[] = {owned_values, &values};
    prog->retain_ndarrays_for_external_submission(
        copy_resources, std::size(copy_resources));
    prog->copy_ndarray_fast(owned_values,
                            const_cast<Ndarray *>(&values));
  } catch (...) {
    if (owned_values) {
      prog->delete_ndarray(owned_values);
    }
    throw;
  }

  rows_ = pattern->num_rows();
  cols_ = pattern->num_cols();
  dtype_ = PrimitiveType::f32;
  program_ = prog;
  nnz_ = pattern->nnz();
  pattern_ = std::move(pattern);
  values_ = owned_values;
  if (pattern_built_for_operator) {
    record_transfer_bytes(
        0, pattern_->device_to_host_bytes(),
        pattern_->device_to_device_bytes() + value_bytes);
    record_pattern_build();
  } else {
    record_transfer_bytes(0, 0, value_bytes);
    record_pattern_reference();
  }
  pattern_->retain_operator_reference();
#else
  TI_NOT_IMPLEMENTED;
#endif
}

VulkanSparseMatrix::VulkanSparseMatrix(Program *prog,
                                       int rows,
                                       int cols,
                                       const Ndarray &row_offsets,
                                       const Ndarray &column_indices,
                                       const Ndarray &values) {
#if defined(TI_WITH_VULKAN)
  TI_ERROR_IF(!prog || prog->compile_config().arch != Arch::vulkan,
              "Internal Vulkan CSR matrices require an active Vulkan "
              "Program.");
  TI_ERROR_IF(!prog->vulkan_sparse_algebra_available(),
              "Vulkan fixed-pattern sparse algebra is unavailable.");
  TI_ERROR_IF(rows <= 0 || cols <= 0,
              "Internal Vulkan CSR matrices require positive dimensions, got "
              "{} x {}.",
              rows, cols);
  TI_ERROR_IF(row_offsets.get_element_data_type() != PrimitiveType::i32 ||
                  !row_offsets.get_element_shape().empty() ||
                  row_offsets.get_nelement() !=
                      static_cast<std::size_t>(rows) + 1 ||
                  row_offsets.get_element_size() != sizeof(int32_t),
              "Internal Vulkan CSR row offsets must contain exactly {} scalar "
              "i32 entries.",
              rows + 1);
  TI_ERROR_IF(column_indices.get_element_data_type() != PrimitiveType::i32 ||
                  !column_indices.get_element_shape().empty() ||
                  column_indices.get_element_size() != sizeof(int32_t),
              "Internal Vulkan CSR column indices must be a scalar i32 "
              "ndarray.");
  TI_ERROR_IF(values.get_element_data_type() != PrimitiveType::f32 ||
                  !values.get_element_shape().empty() ||
                  values.get_element_size() != sizeof(float32),
              "Internal Vulkan CSR values must be a scalar f32 ndarray.");
  const auto nnz_size = column_indices.get_nelement();
  TI_ERROR_IF(nnz_size == 0,
              "Internal Vulkan CSR matrices currently require at least one "
              "stored value.");
  TI_ERROR_IF(nnz_size >
                  static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "Internal Vulkan CSR nnz exceeds the i32 implementation limit.");
  TI_ERROR_IF(values.get_nelement() != nnz_size,
              "Internal Vulkan CSR values must contain exactly {} scalar f32 "
              "entries, got {}.",
              nnz_size, values.get_nelement());

  const auto row_bytes =
      (static_cast<std::size_t>(rows) + 1) * sizeof(int32_t);
  const auto column_bytes = nnz_size * sizeof(int32_t);
  const auto value_bytes = nnz_size * sizeof(float32);
  std::vector<int32_t> host_row_offsets(
      static_cast<std::size_t>(rows) + 1);
  std::vector<int32_t> host_column_indices(nnz_size);
  prog->copy_ndarray_to_host(const_cast<Ndarray *>(&row_offsets),
                             host_row_offsets.data(), row_bytes);
  prog->copy_ndarray_to_host(const_cast<Ndarray *>(&column_indices),
                             host_column_indices.data(), column_bytes);
  TI_ERROR_IF(host_row_offsets.front() != 0 ||
                  host_row_offsets.back() != static_cast<int32_t>(nnz_size),
              "Internal Vulkan CSR row offsets must start at 0 and end at nnz "
              "{}.",
              nnz_size);
  for (int row = 0; row < rows; ++row) {
    const int32_t begin = host_row_offsets[row];
    const int32_t end = host_row_offsets[row + 1];
    TI_ERROR_IF(begin < 0 || end < begin ||
                    end > static_cast<int32_t>(nnz_size),
                "Internal Vulkan CSR row offsets are not monotone at row {}.",
                row);
    int32_t previous_column = -1;
    for (int32_t offset = begin; offset < end; ++offset) {
      const int32_t column = host_column_indices[offset];
      TI_ERROR_IF(column < 0 || column >= cols,
                  "Internal Vulkan CSR column {} at offset {} is outside [0, "
                  "{}).",
                  column, offset, cols);
      TI_ERROR_IF(column <= previous_column,
                  "Internal Vulkan CSR columns must be strictly increasing "
                  "and unique within row {}, got {} after {}.",
                  row, column, previous_column);
      previous_column = column;
    }
  }

  Ndarray *owned_row_offsets = nullptr;
  Ndarray *owned_column_indices = nullptr;
  Ndarray *owned_values = nullptr;
  try {
    owned_row_offsets = prog->create_ndarray(
        PrimitiveType::i32, {rows + 1}, ExternalArrayLayout::kNull, false);
    owned_column_indices = prog->create_ndarray(
        PrimitiveType::i32, {static_cast<int>(nnz_size)},
        ExternalArrayLayout::kNull, false);
    owned_values = prog->create_ndarray(
        PrimitiveType::f32, {static_cast<int>(nnz_size)},
        ExternalArrayLayout::kNull, false);
    auto submission_guard = prog->acquire_runtime_resource_submission_guard();
    const Ndarray *copy_resources[] = {
        owned_row_offsets, &row_offsets, owned_column_indices,
        &column_indices,   owned_values, &values};
    prog->retain_ndarrays_for_external_submission(
        copy_resources, std::size(copy_resources));
    prog->copy_ndarray_fast(owned_row_offsets,
                            const_cast<Ndarray *>(&row_offsets));
    prog->copy_ndarray_fast(owned_column_indices,
                            const_cast<Ndarray *>(&column_indices));
    prog->copy_ndarray_fast(owned_values, const_cast<Ndarray *>(&values));
  } catch (...) {
    if (owned_values)
      prog->delete_ndarray(owned_values);
    if (owned_column_indices)
      prog->delete_ndarray(owned_column_indices);
    if (owned_row_offsets)
      prog->delete_ndarray(owned_row_offsets);
    throw;
  }

  rows_ = rows;
  cols_ = cols;
  dtype_ = PrimitiveType::f32;
  program_ = prog;
  nnz_ = static_cast<int>(nnz_size);
  row_offsets_ = owned_row_offsets;
  column_indices_ = owned_column_indices;
  values_ = owned_values;
  record_transfer_bytes(0, row_bytes + column_bytes,
                        row_bytes + column_bytes + value_bytes);
  record_pattern_build();
#else
  TI_NOT_IMPLEMENTED;
#endif
}

VulkanSparseMatrix::VulkanSparseMatrix(
    Program *prog,
    int rows,
    int cols,
    int nnz,
    Ndarray *owned_row_offsets,
    Ndarray *owned_column_indices,
    Ndarray *owned_values,
    std::uint64_t device_to_device_bytes) {
#if defined(TI_WITH_VULKAN)
  TI_ERROR_IF(!prog || prog->compile_config().arch != Arch::vulkan,
              "Device-assembled Vulkan CSR matrices require an active Vulkan "
              "Program.");
  TI_ERROR_IF(!prog->vulkan_sparse_assembly_available(),
              "Device-assembled Vulkan CSR matrices require Vulkan "
              "shaderInt64 support.");
  TI_ERROR_IF(rows <= 0 || cols <= 0 || nnz < 0,
              "Device-assembled Vulkan CSR matrices require positive "
              "dimensions and nonnegative nnz.");
  TI_ERROR_IF(!owned_row_offsets ||
                  (nnz > 0 && (!owned_column_indices || !owned_values)) ||
                  (nnz == 0 && (owned_column_indices || owned_values)),
              "Device-assembled Vulkan CSR matrices received inconsistent "
              "owned storage for nnz {}.",
              nnz);
  TI_ERROR_IF(
      owned_row_offsets->get_element_data_type() != PrimitiveType::i32 ||
          !owned_row_offsets->get_element_shape().empty() ||
          owned_row_offsets->get_nelement() !=
              static_cast<std::size_t>(rows) + 1 ||
          owned_row_offsets->get_element_size() != sizeof(int32_t),
      "Device-assembled Vulkan CSR row offsets have an invalid shape or "
      "dtype.");
  TI_ERROR_IF(
      nnz > 0 &&
          (owned_column_indices->get_element_data_type() !=
               PrimitiveType::i32 ||
           !owned_column_indices->get_element_shape().empty() ||
           owned_column_indices->get_nelement() !=
               static_cast<std::size_t>(nnz) ||
           owned_column_indices->get_element_size() != sizeof(int32_t)),
      "Device-assembled Vulkan CSR columns have an invalid shape or dtype.");
  TI_ERROR_IF(
      nnz > 0 &&
          (owned_values->get_element_data_type() != PrimitiveType::f32 ||
           !owned_values->get_element_shape().empty() ||
           owned_values->get_nelement() != static_cast<std::size_t>(nnz) ||
           owned_values->get_element_size() != sizeof(float32)),
      "Device-assembled Vulkan CSR values have an invalid shape or dtype.");

  rows_ = rows;
  cols_ = cols;
  dtype_ = PrimitiveType::f32;
  program_ = prog;
  nnz_ = nnz;
  row_offsets_ = owned_row_offsets;
  column_indices_ = owned_column_indices;
  values_ = owned_values;
  record_transfer_bytes(0, 0, device_to_device_bytes);
  record_pattern_build();
#else
  TI_NOT_IMPLEMENTED;
#endif
}

VulkanSparseAssemblyPlan::VulkanSparseAssemblyPlan(Program *program,
                                                   int rows,
                                                   int cols,
                                                   int capacity)
    : program_(program), rows_(rows), cols_(cols), capacity_(capacity) {
#if defined(TI_WITH_VULKAN)
  TI_ERROR_IF(!program || program->compile_config().arch != Arch::vulkan,
              "Vulkan sparse assembly plans require an active Vulkan "
              "Program.");
  TI_ERROR_IF(!program->vulkan_sparse_assembly_available(),
              "Vulkan sparse assembly plans require Vulkan shaderInt64 "
              "support.");
  TI_ERROR_IF(rows <= 0 || cols <= 0 || capacity <= 0 ||
                  rows >= std::numeric_limits<int>::max() ||
                  capacity >= std::numeric_limits<int>::max(),
              "Vulkan sparse assembly rows, columns, and capacity must be "
              "positive, with rows/capacity below INT_MAX.");
  auto create = [&](DataType dtype, int count) {
    return program->create_ndarray(dtype, {count}, ExternalArrayLayout::kNull,
                                   false);
  };
  try {
    sorted_keys_ = create(PrimitiveType::u64, capacity);
    sorted_values_ = create(PrimitiveType::f32, capacity);
    segment_ids_ = create(PrimitiveType::i32, capacity);
    unique_keys_ = create(PrimitiveType::u64, capacity);
    segment_offsets_ = create(PrimitiveType::i32, capacity + 1);
    unique_values_ = create(PrimitiveType::f32, capacity);
    row_offsets_ = create(PrimitiveType::i32, rows + 1);
    column_indices_ = create(PrimitiveType::i32, capacity);
    active_count_ = create(PrimitiveType::i32, 1);
    control_ = create(PrimitiveType::i32, 2);
  } catch (...) {
    delete_workspace();
    throw;
  }
  statistics_.rows = rows;
  statistics_.cols = cols;
  statistics_.capacity = capacity;
  statistics_.persistent_workspace_reserved_bytes =
      static_cast<std::uint64_t>(capacity) * 36 +
      static_cast<std::uint64_t>(rows) * sizeof(int32_t) + 20;
#else
  TI_NOT_IMPLEMENTED;
#endif
}

VulkanSparseAssemblyPlan::~VulkanSparseAssemblyPlan() {
  delete_workspace();
}

void VulkanSparseAssemblyPlan::delete_workspace() noexcept {
#if defined(TI_WITH_VULKAN)
  if (!program_) {
    return;
  }
  auto destroy = [&](Ndarray *&array) {
    if (array) {
      program_->delete_ndarray(array);
      array = nullptr;
    }
  };
  destroy(control_);
  destroy(active_count_);
  destroy(column_indices_);
  destroy(row_offsets_);
  destroy(unique_values_);
  destroy(segment_offsets_);
  destroy(unique_keys_);
  destroy(segment_ids_);
  destroy(sorted_values_);
  destroy(sorted_keys_);
#endif
}

std::unique_ptr<VulkanSparseMatrix> VulkanSparseAssemblyPlan::build(
    Program *program,
    const Ndarray &triplet_rows,
    const Ndarray &triplet_columns,
    const Ndarray &triplet_values) {
  return build_internal(program, nullptr, &triplet_rows, &triplet_columns,
                        &triplet_values);
}

std::unique_ptr<VulkanSparseMatrix> VulkanSparseAssemblyPlan::build_packed(
    Program *program,
    const Ndarray &packed_triplets) {
  return build_internal(program, &packed_triplets, nullptr, nullptr, nullptr);
}

std::unique_ptr<VulkanSparseMatrix> VulkanSparseAssemblyPlan::build_internal(
    Program *program,
    const Ndarray *packed_triplets,
    const Ndarray *triplet_rows,
    const Ndarray *triplet_columns,
    const Ndarray *triplet_values) {
#if defined(TI_WITH_VULKAN)
  std::lock_guard<std::mutex> lock(mutex_);
  TI_ERROR_IF(program != program_,
              "Vulkan sparse assembly requires its owning Program.");
  auto check_input = [&](const char *name, const Ndarray *array,
                         DataType dtype, std::size_t count,
                         std::size_t item_bytes) {
    TI_ERROR_IF(!array || array->shape.size() != 1 ||
                    !array->get_element_shape().empty() ||
                    array->get_element_data_type() != dtype ||
                    array->get_nelement() != count ||
                    array->get_element_size() != item_bytes,
                "Vulkan sparse assembly {} must contain exactly {} scalar "
                "entries with the required dtype.",
                name, count);
  };
  if (packed_triplets) {
    TI_ERROR_IF(triplet_rows || triplet_columns || triplet_values,
                "Packed Vulkan sparse assembly cannot also receive separate "
                "triplet arrays.");
    check_input("packed builder storage", packed_triplets,
                PrimitiveType::i32,
                static_cast<std::size_t>(capacity_) * 3 + 2,
                sizeof(int32_t));
  } else {
    check_input("rows", triplet_rows, PrimitiveType::i32, capacity_,
                sizeof(int32_t));
    check_input("columns", triplet_columns, PrimitiveType::i32, capacity_,
                sizeof(int32_t));
    check_input("values", triplet_values, PrimitiveType::f32, capacity_,
                sizeof(float32));
  }

  statistics_.build_calls++;
  if (statistics_.workspace_builds == 0) {
    statistics_.workspace_builds = 1;
  } else {
    statistics_.workspace_reuses++;
  }
  const auto dispatch = program->vulkan_sparse_assemble_csr(
      const_cast<Ndarray *>(packed_triplets),
      const_cast<Ndarray *>(triplet_rows),
      const_cast<Ndarray *>(triplet_columns),
      const_cast<Ndarray *>(triplet_values), sorted_keys_, sorted_values_,
      segment_ids_, unique_keys_, segment_offsets_, unique_values_,
      row_offsets_, column_indices_, active_count_, control_,
      static_cast<std::size_t>(capacity_), static_cast<std::size_t>(rows_),
      static_cast<std::size_t>(cols_));
  statistics_.shared_radix_sort_workspace_reserved_bytes =
      dispatch.radix_sort_workspace_bytes;
  statistics_.shared_scan_workspace_reserved_bytes =
      dispatch.scan_workspace_bytes;
  if (dispatch.workspace_growth_synchronized) {
    statistics_.workspace_growth_synchronizations++;
  }

  program->synchronize();
  statistics_.host_synchronizations++;
  std::array<int32_t, 2> control_host{0, 0};
  program->copy_ndarray_to_host(control_, control_host.data(),
                                sizeof(control_host));
  statistics_.host_control_readbacks++;
  statistics_.host_scalar_readbacks += 2;
  statistics_.device_to_host_bytes += sizeof(control_host);
  const int encoded_status = control_host[0];
  int status = encoded_status < 0 ? -encoded_status : 0;
  const int input_triplets = encoded_status >= 0 ? encoded_status : 0;
  const int unique_nnz = control_host[1];
  if (status == 0 &&
      (input_triplets < 0 || input_triplets > capacity_ || unique_nnz < 0 ||
       unique_nnz > input_triplets)) {
    status = 4;
  }
  statistics_.last_status = status;
  statistics_.last_input_triplets = status == 0 ? input_triplets : 0;
  statistics_.last_unique_nnz = status == 0 ? unique_nnz : 0;
  statistics_.last_duplicate_triplets =
      status == 0 ? input_triplets - unique_nnz : 0;
  if (status != 0) {
    statistics_.failed_builds++;
    if (status == 5) {
      TI_ERROR("SparseMatrixBuilder triplet count {} exceeds capacity {}.",
               control_host[1], capacity_);
    }
    const char *reason =
        status == 1   ? "triplet index outside matrix dimensions"
        : status == 2 ? "non-finite input value"
        : status == 3 ? "non-finite duplicate sum"
        : status == 5 ? "active triplet count exceeds plan capacity"
                      : "invalid device segment/count state";
    TI_ERROR("Vulkan sparse assembly failed before publish (status {}): {}.",
             status, reason);
  }

  Ndarray *owned_row_offsets = nullptr;
  Ndarray *owned_column_indices = nullptr;
  Ndarray *owned_values = nullptr;
  try {
    owned_row_offsets = program->create_ndarray(
        PrimitiveType::i32, {rows_ + 1}, ExternalArrayLayout::kNull, false);
    if (unique_nnz > 0) {
      owned_column_indices = program->create_ndarray(
          PrimitiveType::i32, {unique_nnz}, ExternalArrayLayout::kNull, false);
      owned_values = program->create_ndarray(
          PrimitiveType::f32, {unique_nnz}, ExternalArrayLayout::kNull, false);
    }
    const std::size_t row_bytes =
        (static_cast<std::size_t>(rows_) + 1) * sizeof(int32_t);
    const std::size_t column_bytes =
        static_cast<std::size_t>(unique_nnz) * sizeof(int32_t);
    const std::size_t value_bytes =
        static_cast<std::size_t>(unique_nnz) * sizeof(float32);
    program->vulkan_copy_ndarray_prefix(owned_row_offsets, row_offsets_,
                                        row_bytes);
    if (column_bytes > 0) {
      program->vulkan_copy_ndarray_prefix(
          owned_column_indices, column_indices_, column_bytes);
    }
    if (value_bytes > 0) {
      program->vulkan_copy_ndarray_prefix(owned_values, unique_values_,
                                          value_bytes);
    }
    const std::uint64_t output_copy_bytes =
        static_cast<std::uint64_t>(row_bytes + column_bytes + value_bytes);
    auto matrix = std::make_unique<VulkanSparseMatrix>(
        program, rows_, cols_, unique_nnz, owned_row_offsets,
        owned_column_indices, owned_values, output_copy_bytes);
    owned_row_offsets = nullptr;
    owned_column_indices = nullptr;
    owned_values = nullptr;
    statistics_.successful_builds++;
    statistics_.device_to_device_bytes += output_copy_bytes;
    statistics_.last_output_pattern_bytes = row_bytes + column_bytes;
    statistics_.last_output_value_bytes = value_bytes;
    return matrix;
  } catch (...) {
    statistics_.failed_builds++;
    statistics_.last_status = 6;
    statistics_.last_input_triplets = 0;
    statistics_.last_unique_nnz = 0;
    statistics_.last_duplicate_triplets = 0;
    if (owned_values)
      program->delete_ndarray(owned_values);
    if (owned_column_indices)
      program->delete_ndarray(owned_column_indices);
    if (owned_row_offsets)
      program->delete_ndarray(owned_row_offsets);
    throw;
  }
#else
  TI_NOT_IMPLEMENTED;
#endif
}

SparseAssemblyRuntimeStatistics
VulkanSparseAssemblyPlan::debug_runtime_statistics() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return statistics_;
}

CudaSparseAssemblyPlan::CudaSparseAssemblyPlan(Program *program,
                                               int rows,
                                               int cols,
                                               int capacity)
    : program_(program), rows_(rows), cols_(cols), capacity_(capacity) {
#if defined(TI_WITH_CUDA)
  TI_ERROR_IF(!program || program->compile_config().arch != Arch::cuda,
              "CUDA sparse assembly plans require an active CUDA Program.");
  TI_ERROR_IF(!program->cuda_sparse_assembly_available(),
              "CUDA sparse assembly plans require the Driver hierarchical "
              "primitive provider.");
  TI_ERROR_IF(!CUSPARSEDriver::get_instance().load_cusparse(),
              "CUDA sparse assembly plans require a loadable cuSPARSE "
              "provider.");
  TI_ERROR_IF(rows <= 0 || cols <= 0 || capacity <= 0 ||
                  rows >= std::numeric_limits<int>::max() ||
                  capacity >= std::numeric_limits<int>::max(),
              "CUDA sparse assembly rows, columns, and capacity must be "
              "positive, with rows/capacity below INT_MAX.");
  auto create = [&](DataType dtype, int count) {
    return program->create_ndarray(dtype, {count}, ExternalArrayLayout::kNull,
                                   false);
  };
  try {
    sorted_keys_ = create(PrimitiveType::u64, capacity);
    sorted_values_ = create(PrimitiveType::f32, capacity);
    segment_ids_ = create(PrimitiveType::i32, capacity);
    unique_keys_ = create(PrimitiveType::u64, capacity);
    segment_offsets_ = create(PrimitiveType::i32, capacity + 1);
    unique_values_ = create(PrimitiveType::f32, capacity);
    row_offsets_ = create(PrimitiveType::i32, rows + 1);
    column_indices_ = create(PrimitiveType::i32, capacity);
    active_count_ = create(PrimitiveType::i32, 1);
    control_ = create(PrimitiveType::i32, 2);
  } catch (...) {
    delete_workspace();
    throw;
  }
  statistics_.rows = rows;
  statistics_.cols = cols;
  statistics_.capacity = capacity;
  statistics_.persistent_workspace_reserved_bytes =
      static_cast<std::uint64_t>(capacity) * 36 +
      static_cast<std::uint64_t>(rows) * sizeof(int32_t) + 20;
#else
  TI_NOT_IMPLEMENTED;
#endif
}

CudaSparseAssemblyPlan::~CudaSparseAssemblyPlan() {
  delete_workspace();
}

void CudaSparseAssemblyPlan::delete_workspace() noexcept {
#if defined(TI_WITH_CUDA)
  if (!program_) {
    return;
  }
  auto destroy = [&](Ndarray *&array) {
    if (array) {
      program_->delete_ndarray(array);
      array = nullptr;
    }
  };
  destroy(control_);
  destroy(active_count_);
  destroy(column_indices_);
  destroy(row_offsets_);
  destroy(unique_values_);
  destroy(segment_offsets_);
  destroy(unique_keys_);
  destroy(segment_ids_);
  destroy(sorted_values_);
  destroy(sorted_keys_);
#endif
}

std::unique_ptr<CuSparseMatrix> CudaSparseAssemblyPlan::build(
    Program *program,
    const Ndarray &triplet_rows,
    const Ndarray &triplet_columns,
    const Ndarray &triplet_values) {
  return build_internal(program, nullptr, &triplet_rows, &triplet_columns,
                        &triplet_values);
}

std::unique_ptr<CuSparseMatrix> CudaSparseAssemblyPlan::build_packed(
    Program *program,
    const Ndarray &packed_triplets) {
  return build_internal(program, &packed_triplets, nullptr, nullptr, nullptr);
}

std::unique_ptr<CuSparseMatrix> CudaSparseAssemblyPlan::build_internal(
    Program *program,
    const Ndarray *packed_triplets,
    const Ndarray *triplet_rows,
    const Ndarray *triplet_columns,
    const Ndarray *triplet_values) {
#if defined(TI_WITH_CUDA)
  std::lock_guard<std::mutex> lock(mutex_);
  TI_ERROR_IF(program != program_,
              "CUDA sparse assembly requires its owning Program.");
  auto check_input = [&](const char *name, const Ndarray *array,
                         DataType dtype, std::size_t count,
                         std::size_t item_bytes) {
    TI_ERROR_IF(!array || array->shape.size() != 1 ||
                    !array->get_element_shape().empty() ||
                    array->get_element_data_type() != dtype ||
                    array->get_nelement() != count ||
                    array->get_element_size() != item_bytes,
                "CUDA sparse assembly {} must contain exactly {} scalar "
                "entries with the required dtype.",
                name, count);
  };
  if (packed_triplets) {
    TI_ERROR_IF(triplet_rows || triplet_columns || triplet_values,
                "Packed CUDA sparse assembly cannot also receive separate "
                "triplet arrays.");
    check_input("packed builder storage", packed_triplets,
                PrimitiveType::f32,
                static_cast<std::size_t>(capacity_) * 3 + 2,
                sizeof(float32));
  } else {
    check_input("rows", triplet_rows, PrimitiveType::i32, capacity_,
                sizeof(int32_t));
    check_input("columns", triplet_columns, PrimitiveType::i32, capacity_,
                sizeof(int32_t));
    check_input("values", triplet_values, PrimitiveType::f32, capacity_,
                sizeof(float32));
  }

  statistics_.build_calls++;
  if (statistics_.workspace_builds == 0) {
    statistics_.workspace_builds = 1;
  } else {
    statistics_.workspace_reuses++;
  }
  const auto dispatch = program->cuda_sparse_assemble_csr(
      const_cast<Ndarray *>(packed_triplets),
      const_cast<Ndarray *>(triplet_rows),
      const_cast<Ndarray *>(triplet_columns),
      const_cast<Ndarray *>(triplet_values), sorted_keys_, sorted_values_,
      segment_ids_, unique_keys_, segment_offsets_, unique_values_,
      row_offsets_, column_indices_, active_count_, control_,
      static_cast<std::size_t>(capacity_), static_cast<std::size_t>(rows_),
      static_cast<std::size_t>(cols_));
  statistics_.shared_radix_sort_workspace_reserved_bytes =
      dispatch.radix_sort_workspace_bytes;
  statistics_.shared_scan_workspace_reserved_bytes =
      dispatch.scan_workspace_bytes;
  if (dispatch.workspace_growth_synchronized) {
    statistics_.workspace_growth_synchronizations++;
  }

  program->synchronize();
  statistics_.host_synchronizations++;
  std::array<int32_t, 2> control_host{0, 0};
  program->copy_ndarray_to_host(control_, control_host.data(),
                                sizeof(control_host));
  statistics_.host_control_readbacks++;
  statistics_.host_scalar_readbacks += 2;
  statistics_.device_to_host_bytes += sizeof(control_host);
  const int encoded_status = control_host[0];
  int status = encoded_status < 0 ? -encoded_status : 0;
  const int input_triplets = encoded_status >= 0 ? encoded_status : 0;
  const int unique_nnz = control_host[1];
  if (status == 0 &&
      (input_triplets < 0 || input_triplets > capacity_ || unique_nnz < 0 ||
       unique_nnz > input_triplets)) {
    status = 4;
  }
  statistics_.last_status = status;
  statistics_.last_input_triplets = status == 0 ? input_triplets : 0;
  statistics_.last_unique_nnz = status == 0 ? unique_nnz : 0;
  statistics_.last_duplicate_triplets =
      status == 0 ? input_triplets - unique_nnz : 0;
  if (status != 0) {
    statistics_.failed_builds++;
    if (status == 5) {
      TI_ERROR("SparseMatrixBuilder triplet count {} exceeds capacity {}.",
               control_host[1], capacity_);
    }
    const char *reason =
        status == 1   ? "triplet index outside matrix dimensions"
        : status == 2 ? "non-finite input value"
        : status == 3 ? "non-finite duplicate sum"
        : status == 5 ? "active triplet count exceeds plan capacity"
                      : "invalid device segment/count state";
    TI_ERROR("CUDA sparse assembly failed before publish (status {}): {}.",
             status, reason);
  }

  void *owned_row_offsets = nullptr;
  void *owned_column_indices = nullptr;
  void *owned_values = nullptr;
  cusparseSpMatDescr_t matrix = nullptr;
  try {
    const std::size_t row_bytes =
        (static_cast<std::size_t>(rows_) + 1) * sizeof(int32_t);
    const std::size_t column_bytes =
        static_cast<std::size_t>(unique_nnz) * sizeof(int32_t);
    const std::size_t value_bytes =
        static_cast<std::size_t>(unique_nnz) * sizeof(float32);
    CUDADriver::get_instance().malloc(&owned_row_offsets, row_bytes);
    if (column_bytes > 0) {
      CUDADriver::get_instance().malloc(&owned_column_indices, column_bytes);
    }
    if (value_bytes > 0) {
      CUDADriver::get_instance().malloc(&owned_values, value_bytes);
    }
    CUDADriver::get_instance().memcpy_device_to_device(
        owned_row_offsets,
        reinterpret_cast<void *>(
            program->get_ndarray_data_ptr_as_int(row_offsets_)),
        row_bytes);
    if (column_bytes > 0) {
      CUDADriver::get_instance().memcpy_device_to_device(
          owned_column_indices,
          reinterpret_cast<void *>(
              program->get_ndarray_data_ptr_as_int(column_indices_)),
          column_bytes);
    }
    if (value_bytes > 0) {
      CUDADriver::get_instance().memcpy_device_to_device(
          owned_values,
          reinterpret_cast<void *>(
              program->get_ndarray_data_ptr_as_int(unique_values_)),
          value_bytes);
    }
    CUSPARSEDriver::get_instance().cpCreateCsr(
        &matrix, rows_, cols_, unique_nnz, owned_row_offsets,
        owned_column_indices, owned_values, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    const std::uint64_t output_copy_bytes =
        static_cast<std::uint64_t>(row_bytes + column_bytes + value_bytes);
    auto result = std::make_unique<CuSparseMatrix>(
        matrix, rows_, cols_, PrimitiveType::f32, owned_row_offsets,
        owned_column_indices, owned_values, unique_nnz, output_copy_bytes);
    matrix = nullptr;
    owned_row_offsets = nullptr;
    owned_column_indices = nullptr;
    owned_values = nullptr;
    statistics_.successful_builds++;
    statistics_.device_to_device_bytes += output_copy_bytes;
    statistics_.last_output_pattern_bytes = row_bytes + column_bytes;
    statistics_.last_output_value_bytes = value_bytes;
    return result;
  } catch (...) {
    statistics_.failed_builds++;
    statistics_.last_status = 6;
    statistics_.last_input_triplets = 0;
    statistics_.last_unique_nnz = 0;
    statistics_.last_duplicate_triplets = 0;
    if (matrix)
      CUSPARSEDriver::get_instance().cpDestroySpMat(matrix);
    if (owned_values)
      CUDADriver::get_instance().mem_free(owned_values);
    if (owned_column_indices)
      CUDADriver::get_instance().mem_free(owned_column_indices);
    if (owned_row_offsets)
      CUDADriver::get_instance().mem_free(owned_row_offsets);
    throw;
  }
#else
  TI_NOT_IMPLEMENTED;
#endif
}

SparseAssemblyRuntimeStatistics
CudaSparseAssemblyPlan::debug_runtime_statistics() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return statistics_;
}

VulkanSparseMatrix::~VulkanSparseMatrix() {
#if defined(TI_WITH_VULKAN)
  if (!program_) {
    return;
  }
  if (values_)
    program_->delete_ndarray(values_);
  if (column_indices_ && !pattern_)
    program_->delete_ndarray(column_indices_);
  if (row_offsets_ && !pattern_)
    program_->delete_ndarray(row_offsets_);
  if (pattern_)
    pattern_->release_operator_reference();
#endif
}

void VulkanSparseMatrix::nd_spmv(Program *prog,
                                 const Ndarray &x,
                                 const Ndarray &y) {
#if defined(TI_WITH_VULKAN)
  TI_ERROR_IF(prog != program_,
              "Internal Vulkan CSR SpMV requires its owning Program.");
  std::lock_guard<std::mutex> lock(spmv_mutex_);
  prog->vulkan_csr_spmv(
      const_cast<Ndarray *>(get_row_offsets()),
      const_cast<Ndarray *>(get_column_indices()), values_,
      const_cast<Ndarray *>(&x), const_cast<Ndarray *>(&y), rows_, cols_,
      nnz_);
  record_spmv_call();
  if (spmv_plan_initialized_) {
    record_spmv_plan_reuse();
  } else {
    record_spmv_plan_build();
    spmv_plan_initialized_ = true;
  }
#else
  TI_NOT_IMPLEMENTED;
#endif
}

void VulkanSparseMatrix::update_values(Program *prog,
                                       const Ndarray &values) {
#if defined(TI_WITH_VULKAN)
  TI_ERROR_IF(prog != program_,
              "Internal Vulkan CSR value updates require the owning Program.");
  TI_ERROR_IF(values.get_element_data_type() != PrimitiveType::f32 ||
                  !values.get_element_shape().empty() ||
                  values.get_nelement() != static_cast<std::size_t>(nnz_) ||
                  values.get_element_size() != sizeof(float32),
              "Internal Vulkan CSR value update expects exactly {} scalar "
              "f32 entries in row-major compressed order.",
              nnz_);
  const auto bytes = static_cast<std::size_t>(nnz_) * sizeof(float32);
  std::lock_guard<std::mutex> lock(spmv_mutex_);
  if (nnz_ == 0) {
    record_numeric_update(0);
    return;
  }
  auto submission_guard = prog->acquire_runtime_resource_submission_guard();
  const Ndarray *copy_resources[] = {values_, &values};
  prog->retain_ndarrays_for_external_submission(
      copy_resources, std::size(copy_resources));
  prog->copy_ndarray_fast(values_, const_cast<Ndarray *>(&values));
  record_numeric_update(bytes);
  record_transfer_bytes(0, 0, bytes);
#else
  TI_NOT_IMPLEMENTED;
#endif
}

SparseMatrixRuntimeStatistics
VulkanSparseMatrix::debug_runtime_statistics() const {
  std::lock_guard<std::mutex> lock(spmv_mutex_);
  auto result = make_runtime_statistics("vulkan", "csr");
  result.provider_name = "forge_vulkan_native";
  result.nnz = nnz_;
  result.pattern_reserved_bytes =
      (static_cast<std::uint64_t>(rows_) + 1 +
       static_cast<std::uint64_t>(nnz_)) *
      sizeof(int32_t);
  result.values_reserved_bytes =
      static_cast<std::uint64_t>(nnz_) * sizeof(float32);
  result.operator_owned_reserved_bytes =
      result.pattern_reserved_bytes + result.values_reserved_bytes;
  if (pattern_) {
    result.operator_exclusive_reserved_bytes = result.values_reserved_bytes;
    result.shared_pattern_id = pattern_->pattern_id();
    result.shared_pattern_operator_references =
        pattern_->operator_references();
    result.pattern_storage_shared = true;
  }
  return result;
}

VulkanSparseBsrMatrix::VulkanSparseBsrMatrix(
    Program *prog,
    int block_rows,
    int block_cols,
    int block_size,
    const Ndarray &row_offsets,
    const Ndarray &column_indices,
    const Ndarray &values)
    : VulkanSparseBsrMatrix(
          std::make_shared<SparseBsrPattern>(
              prog, block_rows, block_cols, block_size, row_offsets,
              column_indices),
          values,
          true) {
}

VulkanSparseBsrMatrix::VulkanSparseBsrMatrix(
    std::shared_ptr<SparseBsrPattern> pattern,
    const Ndarray &values,
    bool pattern_built_for_operator) {
#if defined(TI_WITH_VULKAN)
  TI_ERROR_IF(!pattern || pattern->arch() != Arch::vulkan ||
                  !pattern->program(),
              "Internal Vulkan BSR matrices require a Vulkan-owned pattern.");
  Program *prog = pattern->program();
  TI_ERROR_IF(!prog->vulkan_sparse_algebra_available(),
              "Vulkan fixed-pattern sparse algebra is unavailable.");
  TI_ERROR_IF(values.get_element_data_type() != PrimitiveType::f32 ||
                  !values.get_element_shape().empty() ||
                  values.get_element_size() != sizeof(float32),
              "Internal Vulkan BSR values must be a scalar f32 ndarray.");

  const auto value_count = pattern->value_count();
  TI_ERROR_IF(values.get_nelement() != value_count,
              "Internal Vulkan BSR values must contain exactly {} scalar "
              "f32 entries for {} dense {} x {} blocks, got {}.",
              value_count, pattern->block_nnz(), pattern->block_size(),
              pattern->block_size(), values.get_nelement());

  const auto value_bytes = value_count * sizeof(float32);
  Ndarray *owned_values = nullptr;
  try {
    owned_values = prog->create_ndarray(
        PrimitiveType::f32, {static_cast<int>(value_count)},
        ExternalArrayLayout::kNull, false);
    auto submission_guard =
        prog->acquire_runtime_resource_submission_guard();
    const Ndarray *copy_resources[] = {owned_values, &values};
    prog->retain_ndarrays_for_external_submission(
        copy_resources, std::size(copy_resources));
    prog->copy_ndarray_fast(owned_values,
                            const_cast<Ndarray *>(&values));
  } catch (...) {
    if (owned_values)
      prog->delete_ndarray(owned_values);
    throw;
  }

  rows_ = pattern->num_rows();
  cols_ = pattern->num_cols();
  dtype_ = PrimitiveType::f32;
  program_ = prog;
  block_rows_ = pattern->block_rows();
  block_cols_ = pattern->block_cols();
  block_size_ = pattern->block_size();
  block_nnz_ = pattern->block_nnz();
  scalar_nnz_ = pattern->scalar_nnz();
  value_count_ = value_count;
  pattern_ = std::move(pattern);
  values_ = owned_values;
  if (pattern_built_for_operator) {
    record_transfer_bytes(
        0, pattern_->device_to_host_bytes(),
        pattern_->device_to_device_bytes() + value_bytes);
    record_pattern_build();
  } else {
    record_transfer_bytes(0, 0, value_bytes);
    record_pattern_reference();
  }
  pattern_->retain_operator_reference();
#else
  TI_NOT_IMPLEMENTED;
#endif
}

VulkanSparseBsrMatrix::~VulkanSparseBsrMatrix() {
#if defined(TI_WITH_VULKAN)
  if (!program_) {
    return;
  }
  if (values_)
    program_->delete_ndarray(values_);
  if (pattern_)
    pattern_->release_operator_reference();
#endif
}

void VulkanSparseBsrMatrix::nd_spmv(Program *prog,
                                    const Ndarray &x,
                                    const Ndarray &y) {
#if defined(TI_WITH_VULKAN)
  TI_ERROR_IF(prog != program_,
              "Internal Vulkan BSR SpMV requires its owning Program.");
  std::lock_guard<std::mutex> lock(spmv_mutex_);
  prog->vulkan_bsr_spmv(
      const_cast<Ndarray *>(pattern_->vulkan_row_offsets()),
      const_cast<Ndarray *>(pattern_->vulkan_column_indices()), values_,
      const_cast<Ndarray *>(&x), const_cast<Ndarray *>(&y), block_rows_,
      block_cols_, block_nnz_, block_size_);
  record_spmv_call();
  if (spmv_plan_initialized_) {
    record_spmv_plan_reuse();
  } else {
    record_spmv_plan_build();
    spmv_plan_initialized_ = true;
  }
#else
  TI_NOT_IMPLEMENTED;
#endif
}

void VulkanSparseBsrMatrix::update_values(Program *prog,
                                          const Ndarray &values) {
#if defined(TI_WITH_VULKAN)
  TI_ERROR_IF(prog != program_,
              "Internal Vulkan BSR value updates require the owning "
              "Program.");
  TI_ERROR_IF(values.get_element_data_type() != PrimitiveType::f32 ||
                  !values.get_element_shape().empty() ||
                  values.get_nelement() != value_count_ ||
                  values.get_element_size() != sizeof(float32),
              "Internal Vulkan BSR value update expects exactly {} scalar "
              "f32 entries in block-row-major order.",
              value_count_);
  const auto bytes = value_count_ * sizeof(float32);
  std::lock_guard<std::mutex> lock(spmv_mutex_);
  auto submission_guard =
      prog->acquire_runtime_resource_submission_guard();
  const Ndarray *copy_resources[] = {values_, &values};
  prog->retain_ndarrays_for_external_submission(
      copy_resources, std::size(copy_resources));
  prog->copy_ndarray_fast(values_, const_cast<Ndarray *>(&values));
  record_numeric_update(bytes);
  record_transfer_bytes(0, 0, bytes);
#else
  TI_NOT_IMPLEMENTED;
#endif
}

SparseMatrixRuntimeStatistics
VulkanSparseBsrMatrix::debug_runtime_statistics() const {
  std::lock_guard<std::mutex> lock(spmv_mutex_);
  auto result = make_runtime_statistics("vulkan", "bsr");
  result.provider_name = "forge_vulkan_native";
  result.nnz = scalar_nnz_;
  result.block_rows = block_rows_;
  result.block_cols = block_cols_;
  result.block_size = block_size_;
  result.block_nnz = block_nnz_;
  result.pattern_reserved_bytes = pattern_->pattern_reserved_bytes();
  result.values_reserved_bytes =
      static_cast<std::uint64_t>(value_count_) * sizeof(float32);
  result.operator_owned_reserved_bytes =
      result.pattern_reserved_bytes + result.values_reserved_bytes;
  result.operator_exclusive_reserved_bytes = result.values_reserved_bytes;
  result.shared_pattern_id = pattern_->pattern_id();
  result.shared_pattern_operator_references =
      pattern_->operator_references();
  result.pattern_storage_shared = true;
  return result;
}

void CuSparseMatrix::nd_spmv(Program *prog,
                             const Ndarray &x,
                             const Ndarray &y) {
#if defined(TI_WITH_CUDA)
  size_t dX = prog->get_ndarray_data_ptr_as_int(&x);
  size_t dY = prog->get_ndarray_data_ptr_as_int(&y);
  spmv(dX, dY);
#endif
}

const std::string CuSparseMatrix::to_string() const {
  std::ostringstream ostr;
#ifdef TI_WITH_CUDA
  size_t rows, cols, nnz;
  float *dR;
  int *dC, *dV;
  cusparseIndexType_t row_type, column_type;
  cusparseIndexBase_t idx_base;
  cudaDataType value_type;
  CUSPARSEDriver::get_instance().cpCsrGet(
      matrix_, &rows, &cols, &nnz, (void **)&dR, (void **)&dC, (void **)&dV,
      &row_type, &column_type, &idx_base, &value_type);

  auto *hR = new int[rows + 1];
  auto *hC = new int[nnz];
  auto *hV = new float[nnz];

  CUDADriver::get_instance().memcpy_device_to_host((void *)hR, (void *)dR,
                                                   (rows + 1) * sizeof(int));
  CUDADriver::get_instance().memcpy_device_to_host((void *)hC, (void *)dC,
                                                   (nnz) * sizeof(int));
  CUDADriver::get_instance().memcpy_device_to_host((void *)hV, (void *)dV,
                                                   (nnz) * sizeof(float));

  print_triplets_from_csr<int, int, float>(rows, cols, hR, hC, hV, ostr);
  delete[] hR;
  delete[] hC;
  delete[] hV;
#endif
  return ostr.str();
}

float CuSparseMatrix::get_element(int row, int col) const {
  float res = 0.0f;
#ifdef TI_WITH_CUDA
  size_t rows, cols, nnz;
  float *dR;
  int *dC, *dV;
  cusparseIndexType_t row_type, column_type;
  cusparseIndexBase_t idx_base;
  cudaDataType value_type;
  CUSPARSEDriver::get_instance().cpCsrGet(
      matrix_, &rows, &cols, &nnz, (void **)&dR, (void **)&dC, (void **)&dV,
      &row_type, &column_type, &idx_base, &value_type);

  TI_ASSERT(row < rows);
  TI_ASSERT(col < cols);

  auto *hR = new int[rows + 1];
  auto *hC = new int[nnz];
  auto *hV = new float[nnz];

  CUDADriver::get_instance().memcpy_device_to_host((void *)hR, (void *)dR,
                                                   (rows + 1) * sizeof(int));
  CUDADriver::get_instance().memcpy_device_to_host((void *)hC, (void *)dC,
                                                   (nnz) * sizeof(int));
  CUDADriver::get_instance().memcpy_device_to_host((void *)hV, (void *)dV,
                                                   (nnz) * sizeof(float));

  res = get_element_from_csr<int, int, float>(row, col, hR, hC, hV);

  delete[] hR;
  delete[] hC;
  delete[] hV;
#endif  // TI_WITH_CUDA
  return res;
}

void CuSparseMatrix::mmwrite(const std::string &filename) {
#ifdef TI_WITH_CUDA
  size_t rows, cols, nnz;
  float *dR;
  int *dC, *dV;
  cusparseIndexType_t row_type, column_type;
  cusparseIndexBase_t idx_base;
  cudaDataType value_type;
  CUSPARSEDriver::get_instance().cpCsrGet(
      matrix_, &rows, &cols, &nnz, (void **)&dR, (void **)&dC, (void **)&dV,
      &row_type, &column_type, &idx_base, &value_type);

  auto *hR = new int[rows + 1];
  auto *hC = new int[nnz];
  auto *hV = new float[nnz];

  CUDADriver::get_instance().memcpy_device_to_host((void *)hR, (void *)dR,
                                                   (rows + 1) * sizeof(int));
  CUDADriver::get_instance().memcpy_device_to_host((void *)hC, (void *)dC,
                                                   (nnz) * sizeof(int));
  CUDADriver::get_instance().memcpy_device_to_host((void *)hV, (void *)dV,
                                                   (nnz) * sizeof(float));

  std::ofstream file(filename);
  file << "%%MatrixMarket matrix coordinate real general\n%" << std::endl;
  file << rows << " " << cols << " " << nnz << std::endl;
  for (int r = 0; r < rows; r++) {
    for (int c = hR[r]; c < hR[r + 1]; c++) {
      file << r + 1 << " " << hC[c] + 1 << " " << hV[c] << std::endl;
    }
  }
  file.close();
  delete[] hR;
  delete[] hC;
  delete[] hV;
#endif
}

}  // namespace taichi::lang
