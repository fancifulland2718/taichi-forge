#pragma once

#include "taichi/common/core.h"
#include "taichi/inc/constants.h"
#include "taichi/ir/type_utils.h"
#include "taichi/program/ndarray.h"
#include "taichi/program/program.h"
#include "taichi/rhi/cuda/cuda_driver.h"

#include "Eigen/Sparse"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace taichi::lang {

class SparseMatrix;
std::uint64_t allocate_sparse_matrix_id();

class CudaSparseAssemblyPlan;
class VulkanSparseAssemblyPlan;
class Kernel;
class CompiledKernelData;
class LaunchContextBuilder;
class OperatorBinding;
class OperatorPinnedAction;
class OperatorResourceGenerationPublisher;
struct OperatorResourceStamp;
namespace aot {
struct CompiledGraph;
struct CompiledGraphJITCache;
}  // namespace aot

// Private diagnostic inventory for the algebraic sparse operator boundary.
// Persistent matrix/SpMV resources are operator-owned; input/output ndarrays
// and solver Krylov vectors are deliberately excluded.
struct SparseMatrixRuntimeStatistics {
  std::string backend_family{"unknown"};
  std::string storage_format{"unknown"};
  std::string dtype{"unknown"};
  std::string provider_name{"unknown"};
  int provider_version_major{-1};
  int provider_version_minor{-1};
  int provider_version_patch{-1};
  bool provider_bsr_descriptor_available{false};
  bool provider_generic_bsr_spmv_available{false};
  int rows{0};
  int cols{0};
  int nnz{0};
  int block_rows{0};
  int block_cols{0};
  int block_size{0};
  int block_nnz{0};

  std::uint64_t pattern_version{0};
  std::uint64_t numeric_version{0};
  std::uint64_t pattern_builds{0};
  std::uint64_t numeric_updates{0};
  std::uint64_t numeric_update_bytes{0};
  std::uint64_t spmv_calls{0};
  std::uint64_t spmv_plan_builds{0};
  std::uint64_t spmv_plan_reuses{0};
  std::uint64_t spmv_handle_creations{0};
  std::uint64_t dense_vector_descriptor_creations{0};
  std::uint64_t dense_vector_descriptor_rebinds{0};
  std::uint64_t spmv_workspace_allocations{0};
  std::uint64_t resource_generations_published{0};
  std::uint64_t resource_generations_retired{0};
  std::uint64_t resource_generations_released{0};
  std::uint64_t resource_generation_active_leases{0};
  bool resource_generation_current{false};

  std::uint64_t pattern_reserved_bytes{0};
  std::uint64_t values_reserved_bytes{0};
  std::uint64_t spmv_workspace_reserved_bytes{0};
  std::uint64_t operator_owned_reserved_bytes{0};
  std::uint64_t operator_exclusive_reserved_bytes{0};
  std::uint64_t numeric_update_peak_temporary_bytes{0};
  std::uint64_t shared_pattern_id{0};
  std::uint64_t shared_pattern_operator_references{0};
  bool pattern_storage_shared{false};
  std::uint64_t matrix_descriptor_count{0};
  std::uint64_t dense_vector_descriptor_count{0};
  std::uint64_t spmv_handle_count{0};

  std::uint64_t host_to_device_bytes{0};
  std::uint64_t device_to_host_bytes{0};
  std::uint64_t device_to_device_bytes{0};
};

struct SparsePatternRuntimeStatistics {
  std::string backend_family{"unknown"};
  std::string storage_format{"unknown"};
  std::string index_dtype{"unknown"};
  std::string value_order{"unknown"};
  int rows{0};
  int cols{0};
  int nnz{0};
  int block_rows{0};
  int block_cols{0};
  int block_size{0};
  int block_nnz{0};
  std::uint64_t pattern_id{0};
  std::uint64_t pattern_version{0};
  std::uint64_t pattern_builds{0};
  std::uint64_t operator_references{0};
  bool immutable{false};
  std::uint64_t pattern_reserved_bytes{0};
  std::uint64_t host_to_device_bytes{0};
  std::uint64_t device_to_host_bytes{0};
  std::uint64_t device_to_device_bytes{0};
};

struct SparseAssemblyRuntimeStatistics {
  int rows{0};
  int cols{0};
  int capacity{0};
  int last_status{0};
  int last_input_triplets{0};
  int last_unique_nnz{0};
  int last_duplicate_triplets{0};
  std::uint64_t build_calls{0};
  std::uint64_t successful_builds{0};
  std::uint64_t failed_builds{0};
  std::uint64_t workspace_builds{0};
  std::uint64_t workspace_reuses{0};
  std::uint64_t workspace_growth_synchronizations{0};
  std::uint64_t host_synchronizations{0};
  std::uint64_t host_control_readbacks{0};
  std::uint64_t host_scalar_readbacks{0};
  std::uint64_t device_to_host_bytes{0};
  std::uint64_t device_to_device_bytes{0};
  std::uint64_t persistent_workspace_reserved_bytes{0};
  std::uint64_t shared_radix_sort_workspace_reserved_bytes{0};
  std::uint64_t shared_scan_workspace_reserved_bytes{0};
  std::uint64_t last_output_pattern_bytes{0};
  std::uint64_t last_output_value_bytes{0};
};

class SparseMatrixBuilder {
 public:
  SparseMatrixBuilder(int rows,
                      int cols,
                      int max_num_triplets,
                      DataType dtype,
                      const std::string &storage_format);

  ~SparseMatrixBuilder();
  void print_triplets_eigen();
  void print_triplets_cuda();

  void create_ndarray(Program *prog);

  void delete_ndarray(Program *prog);

  intptr_t get_ndarray_data_ptr() const;

  Ndarray *get_ndarray() const;

  std::unique_ptr<SparseMatrix> build();

  std::unique_ptr<SparseMatrix> build_cuda();

  std::unique_ptr<SparseMatrix> build_vulkan();

  void clear();

 private:
  template <typename T, typename G>
  std::unique_ptr<SparseMatrix> build_template();

  template <typename T, typename G>
  void print_triplets_template();

 private:
  uint64 num_triplets_{0};
  Ndarray *ndarray_data_base_ptr_{nullptr};
  intptr_t ndarray_data_ptr_{0};
  int rows_{0};
  int cols_{0};
  uint64 max_num_triplets_{0};
  bool built_{false};
  DataType dtype_{PrimitiveType::f32};
  std::string storage_format_{"col_major"};
  Program *program_{nullptr};
  std::unique_ptr<CudaSparseAssemblyPlan> cuda_assembly_plan_;
  std::unique_ptr<VulkanSparseAssemblyPlan> vulkan_assembly_plan_;
};

class SparseMatrix {
 public:
  SparseMatrix() : rows_(0), cols_(0), dtype_(PrimitiveType::f32) {};
  SparseMatrix(int rows, int cols, DataType dt = PrimitiveType::f32)
      : rows_{rows}, cols_(cols), dtype_(dt) {};
  SparseMatrix(SparseMatrix &sm)
      : rows_(sm.rows_), cols_(sm.cols_), dtype_(sm.dtype_) {
  }
  SparseMatrix(SparseMatrix &&sm)
      : rows_(sm.rows_), cols_(sm.cols_), dtype_(sm.dtype_) {
  }
  virtual ~SparseMatrix() = default;

  virtual void build_triplets(void *triplets_adr) {
    TI_NOT_IMPLEMENTED;
  };

  virtual void build_csr_from_coo(void *coo_row_ptr,
                                  void *coo_col_ptr,
                                  void *coo_values_ptr,
                                  int nnz) {
    TI_NOT_IMPLEMENTED;
  }
  virtual int num_nonzero() const {
    TI_NOT_IMPLEMENTED;
    return 0;
  }
  // Internal ndarray operator hook shared by fixed and matrix-free providers.
  // Unsupported matrix types retain the default hard failure.
  virtual void nd_spmv(Program *prog,
                       const Ndarray &x,
                       const Ndarray &y) {
    TI_NOT_IMPLEMENTED;
  }
  virtual void update_values(Program *prog, const Ndarray &values) {
    TI_NOT_IMPLEMENTED;
  }
  virtual SparseMatrixRuntimeStatistics debug_runtime_statistics() const;

  // Fixed-pattern applies, value refreshes, preconditioner refreshes and
  // Krylov solves share one recursive transaction gate. A solve may retain
  // one numeric snapshot while nested operator applies reacquire the gate.
  using NumericAccessGuard = std::unique_lock<std::recursive_mutex>;
  NumericAccessGuard acquire_numeric_access_guard() const {
    return NumericAccessGuard(numeric_access_mutex_);
  }

  // Assembly helpers use direct backend copies which do not flow through a
  // Program launch. Attribute those bytes to the resulting operator here.
  void record_transfer_bytes(std::uint64_t host_to_device,
                             std::uint64_t device_to_host,
                             std::uint64_t device_to_device);
  inline const int num_rows() const {
    return rows_;
  }

  inline const int num_cols() const {
    return cols_;
  }

  virtual const std::string to_string() const {
    return "";
  }

  virtual const void *get_matrix() const {
    return nullptr;
  }

  inline const DataType get_data_type() const {
    return dtype_;
  }

  inline std::uint64_t matrix_id() const {
    return matrix_id_;
  }

  inline std::uint64_t pattern_version() const {
    return pattern_version_.load(std::memory_order_relaxed);
  }

  inline std::uint64_t numeric_version() const {
    return numeric_version_.load(std::memory_order_relaxed);
  }

  template <class T>
  T get_element(int row, int col) {
    TI_NOT_IMPLEMENTED;
  }

  template <class T>
  void set_element(int row, int col, T value) {
    TI_NOT_IMPLEMENTED;
  }

  virtual void mmwrite(const std::string &filename) {
    TI_NOT_IMPLEMENTED;
  }

 protected:
  SparseMatrixRuntimeStatistics make_runtime_statistics(
      const std::string &backend_family,
      const std::string &storage_format) const;
  void record_pattern_build();
  void record_pattern_reference();
  void record_numeric_update(std::uint64_t bytes);
  void record_spmv_call();
  void record_spmv_plan_build();
  void record_spmv_plan_reuse();
  void record_spmv_handle_creation();
  void record_dense_vector_descriptor_creation(bool rebind);
  void record_spmv_workspace_allocation();

  int rows_{0};
  int cols_{0};
  DataType dtype_{PrimitiveType::f32};

 private:
  mutable std::recursive_mutex numeric_access_mutex_;
  const std::uint64_t matrix_id_{allocate_sparse_matrix_id()};
  std::atomic<std::uint64_t> pattern_version_{0};
  std::atomic<std::uint64_t> numeric_version_{0};
  std::atomic<std::uint64_t> pattern_builds_{0};
  std::atomic<std::uint64_t> numeric_updates_{0};
  std::atomic<std::uint64_t> numeric_update_bytes_{0};
  std::atomic<std::uint64_t> spmv_calls_{0};
  std::atomic<std::uint64_t> spmv_plan_builds_{0};
  std::atomic<std::uint64_t> spmv_plan_reuses_{0};
  std::atomic<std::uint64_t> spmv_handle_creations_{0};
  std::atomic<std::uint64_t> dense_vector_descriptor_creations_{0};
  std::atomic<std::uint64_t> dense_vector_descriptor_rebinds_{0};
  std::atomic<std::uint64_t> spmv_workspace_allocations_{0};
  std::atomic<std::uint64_t> host_to_device_bytes_{0};
  std::atomic<std::uint64_t> device_to_host_bytes_{0};
  std::atomic<std::uint64_t> device_to_device_bytes_{0};
};

// Internal shell-style ndarray operator backed by one Program-owned compiled
// Taichi kernel. The operator snapshots immutable topology data and may own a
// separately versioned numeric snapshot for value-only updates. Public sparse
// capabilities stay disabled until a solver explicitly accepts this provider
// contract.
class CompiledKernelLinearOperator final : public SparseMatrix {
 public:
  CompiledKernelLinearOperator(Program *program,
                               Kernel &kernel,
                               int size,
                               std::uint64_t topology_version,
                               std::uint64_t numeric_version,
                               const Ndarray &operator_data);
  CompiledKernelLinearOperator(Program *program,
                               Kernel &kernel,
                               int size,
                               std::uint64_t topology_version,
                               std::uint64_t numeric_version,
                               const Ndarray &topology_data,
                               const Ndarray &numeric_data);
  ~CompiledKernelLinearOperator() override;

  void nd_spmv(Program *program,
               const Ndarray &input,
               const Ndarray &output) override;
  void update_numeric_data(Program *program,
                           const Ndarray &numeric_data,
                           std::uint64_t expected_topology_version,
                           std::uint64_t expected_numeric_version);
  SparseMatrixRuntimeStatistics debug_runtime_statistics() const override;
  OperatorBinding make_operator_binding();
  OperatorPinnedAction pin_operator_generation() const;
  OperatorResourceStamp current_operator_resource_stamp() const;

  Program *owning_program() const {
    return program_;
  }

  int num_nonzero() const override {
    return 0;
  }

 private:
  struct TopologyState;
  struct ResourceGeneration;

  CompiledKernelLinearOperator(Program *program,
                               Kernel &kernel,
                               int size,
                               std::uint64_t topology_version,
                               std::uint64_t numeric_version,
                               const Ndarray &topology_data,
                               const Ndarray *numeric_data);
  void publish_resource_generation(Ndarray *owned_numeric_data,
                                   std::uint64_t numeric_version,
                                   std::uint64_t binding_revision);

  Program *program_{nullptr};
  Kernel *kernel_{nullptr};
  const CompiledKernelData *compiled_kernel_{nullptr};
  std::shared_ptr<TopologyState> topology_state_;
  std::unique_ptr<OperatorResourceGenerationPublisher>
      resource_generations_;
  DataType numeric_data_type_{PrimitiveType::unknown};
  std::vector<int> numeric_data_shape_;
  ExternalArrayLayout numeric_data_layout_{ExternalArrayLayout::kNull};
  bool has_numeric_data_{false};
  std::uint64_t topology_data_bytes_{0};
  std::uint64_t numeric_data_bytes_{0};
  std::size_t input_arg_index_{2};
  std::size_t output_arg_index_{3};
  std::uint64_t topology_version_{0};
  std::uint64_t numeric_version_{0};
  std::uint64_t binding_revision_{1};
  std::shared_ptr<std::atomic<std::uint64_t>>
      generation_apply_calls_;
  mutable std::mutex spmv_mutex_;
};

// Internal square f32 operator backed by a fixed multi-kernel CGraph. The
// provider accepts only fixed i32 scalars, owned scalar ndarray snapshots, and
// reserved dynamic input/output vectors. Ndarray roles remain explicit so
// topology, numeric data, and mutable workspace are not conflated in memory
// telemetry. Public sparse capabilities stay disabled.
class CompiledGraphLinearOperator final : public SparseMatrix {
 public:
  using FixedI32Arguments = std::unordered_map<std::string, std::int32_t>;
  using NdarrayArguments =
      std::unordered_map<std::string, const Ndarray *>;

  CompiledGraphLinearOperator(
      Program *program,
      const aot::CompiledGraph &graph,
      int size,
      std::uint64_t topology_version,
      std::uint64_t numeric_version,
      FixedI32Arguments fixed_i32_arguments,
      NdarrayArguments topology_arguments,
      NdarrayArguments numeric_arguments,
      NdarrayArguments workspace_arguments);
  ~CompiledGraphLinearOperator() override;

  void nd_spmv(Program *program,
               const Ndarray &input,
               const Ndarray &output) override;
  void update_numeric_arguments(
      Program *program,
      NdarrayArguments numeric_arguments,
      std::uint64_t expected_topology_version,
      std::uint64_t expected_numeric_version);
  SparseMatrixRuntimeStatistics debug_runtime_statistics() const override;

  Program *owning_program() const {
    return program_;
  }

  int num_nonzero() const override {
    return 0;
  }

 private:
  enum class NdarrayRole {
    topology,
    numeric,
    workspace,
  };

  struct OwnedNdarrayArgument {
    std::string name;
    Ndarray *value{nullptr};
    NdarrayRole role{NdarrayRole::topology};
  };

  Program *program_{nullptr};
  std::unique_ptr<aot::CompiledGraph> graph_;
  std::unique_ptr<aot::CompiledGraphJITCache> cache_;
  FixedI32Arguments fixed_i32_arguments_;
  std::vector<OwnedNdarrayArgument> owned_ndarray_arguments_;
  std::uint64_t topology_reserved_bytes_{0};
  std::uint64_t numeric_reserved_bytes_{0};
  std::uint64_t workspace_reserved_bytes_{0};
  std::uint64_t numeric_update_peak_temporary_bytes_{0};
  std::uint64_t topology_version_{0};
  std::uint64_t numeric_version_{0};
  mutable std::mutex spmv_mutex_;
};

// Backend-neutral immutable sparse topology. Concrete providers may keep
// different physical storage while exposing one ownership/version contract.
class SparsePattern {
 public:
  virtual ~SparsePattern() = default;
  virtual Program *program() const = 0;
  virtual Arch arch() const = 0;
  virtual int num_rows() const = 0;
  virtual int num_cols() const = 0;
  virtual SparsePatternRuntimeStatistics debug_runtime_statistics() const = 0;
};

// Backend-neutral immutable scalar CSR topology shared by one or more
// numeric operators. CPU owns canonical host vectors, CUDA owns raw device
// index buffers, and Vulkan owns Program-managed index ndarrays.
class SparseCsrPattern final : public SparsePattern {
 public:
  SparseCsrPattern(Program *program,
                   int rows,
                   int cols,
                   const Ndarray &row_offsets,
                   const Ndarray &column_indices);
  ~SparseCsrPattern() override;

  Program *program() const override {
    return program_;
  }

  Arch arch() const override {
    return arch_;
  }

  int num_rows() const override {
    return rows_;
  }

  int num_cols() const override {
    return cols_;
  }

  int nnz() const {
    return nnz_;
  }

  std::uint64_t pattern_id() const {
    return pattern_id_;
  }

  std::uint64_t pattern_reserved_bytes() const;
  std::uint64_t device_to_host_bytes() const {
    return device_to_host_bytes_;
  }
  std::uint64_t device_to_device_bytes() const {
    return device_to_device_bytes_;
  }
  std::uint64_t operator_references() const {
    return operator_references_.load(std::memory_order_relaxed);
  }

  const std::vector<int32_t> &cpu_row_offsets() const;
  const std::vector<int32_t> &cpu_column_indices() const;
  void *cuda_row_offsets() const;
  void *cuda_column_indices() const;
  const Ndarray *vulkan_row_offsets() const;
  const Ndarray *vulkan_column_indices() const;

  void retain_operator_reference();
  void release_operator_reference();
  SparsePatternRuntimeStatistics debug_runtime_statistics() const override;

 private:
  Program *program_{nullptr};
  Arch arch_;
  int rows_{0};
  int cols_{0};
  int nnz_{0};
  std::uint64_t pattern_id_{0};
  std::vector<int32_t> cpu_row_offsets_;
  std::vector<int32_t> cpu_column_indices_;
  void *cuda_row_offsets_{nullptr};
  void *cuda_column_indices_{nullptr};
  Ndarray *vulkan_row_offsets_{nullptr};
  Ndarray *vulkan_column_indices_{nullptr};
  std::atomic<std::uint64_t> operator_references_{0};
  std::uint64_t device_to_host_bytes_{0};
  std::uint64_t device_to_device_bytes_{0};
};

template <class EigenMatrix>
class EigenSparseMatrix : public SparseMatrix {
 public:
  explicit EigenSparseMatrix(int rows, int cols, DataType dt)
      : SparseMatrix(rows, cols, dt), matrix_(rows, cols) {
  }
  EigenSparseMatrix(EigenSparseMatrix &sm)
      : SparseMatrix(sm.num_rows(), sm.num_cols(), sm.dtype_),
        matrix_(sm.matrix_) {
    record_pattern_build();
  }
  EigenSparseMatrix(EigenSparseMatrix &&sm)
      : SparseMatrix(sm.num_rows(), sm.num_cols(), sm.dtype_),
        matrix_(sm.matrix_) {
    record_pattern_build();
  }
  explicit EigenSparseMatrix(const EigenMatrix &em)
      : SparseMatrix(
            em.rows(), em.cols(),
            std::is_same_v<typename EigenMatrix::Scalar, float64>
                ? DataType(PrimitiveType::f64)
                : DataType(PrimitiveType::f32)),
        matrix_(em) {
    record_pattern_build();
  }

  ~EigenSparseMatrix() override = default;

  void build_triplets(void *triplets_adr) override;
  const std::string to_string() const override;

  // Write the sparse matrix to a Matrix Market file
  void mmwrite(const std::string &filename) override;

  const void *get_matrix() const override {
    return &matrix_;
  };

  void *get_matrix() {
    return &matrix_;
  };

  virtual EigenSparseMatrix &operator+=(const EigenSparseMatrix &other) {
    this->matrix_ += other.matrix_;
    record_pattern_build();
    return *this;
  };

  friend EigenSparseMatrix operator+(const EigenSparseMatrix &lhs,
                                     const EigenSparseMatrix &rhs) {
    return EigenSparseMatrix(lhs.matrix_ + rhs.matrix_);
  };

  virtual EigenSparseMatrix &operator-=(const EigenSparseMatrix &other) {
    this->matrix_ -= other.matrix_;
    record_pattern_build();
    return *this;
  }

  friend EigenSparseMatrix operator-(const EigenSparseMatrix &lhs,
                                     const EigenSparseMatrix &rhs) {
    return EigenSparseMatrix(lhs.matrix_ - rhs.matrix_);
  };

  virtual EigenSparseMatrix &operator*=(float scale) {
    this->matrix_ *= scale;
    record_numeric_update(
        static_cast<std::uint64_t>(matrix_.nonZeros()) *
        data_type_size(dtype_));
    return *this;
  }

  friend EigenSparseMatrix operator*(const EigenSparseMatrix &sm, float scale) {
    return EigenSparseMatrix(sm.matrix_ * scale);
  }

  friend EigenSparseMatrix operator*(float scale, const EigenSparseMatrix &sm) {
    return EigenSparseMatrix(sm.matrix_ * scale);
  }

  friend EigenSparseMatrix operator*(const EigenSparseMatrix &lhs,
                                     const EigenSparseMatrix &rhs) {
    return EigenSparseMatrix(lhs.matrix_.cwiseProduct(rhs.matrix_));
  }

  EigenSparseMatrix transpose() {
    return EigenSparseMatrix(matrix_.transpose());
  }

  EigenSparseMatrix matmul(const EigenSparseMatrix &sm) {
    return EigenSparseMatrix(matrix_ * sm.matrix_);
  }

  template <typename T>
  T get_element(int row, int col) {
    return matrix_.coeff(row, col);
  }

  template <typename T>
  void set_element(int row, int col, T value) {
    matrix_.coeffRef(row, col) = value;
    // coeffRef may insert a new entry. Treat it as a pattern mutation rather
    // than trying to infer insertion from Eigen's compressed state.
    record_pattern_build();
  }

  template <class VT>
  VT mat_vec_mul(const Eigen::Ref<const VT> &b) {
    record_spmv_call();
    return matrix_ * b;
  }

  void spmv(Program *prog, const Ndarray &x, const Ndarray &y);

  int num_nonzero() const override {
    return static_cast<int>(matrix_.nonZeros());
  }

  void update_values(Program *prog, const Ndarray &values) override;

  SparseMatrixRuntimeStatistics debug_runtime_statistics() const override {
    auto result = make_runtime_statistics(
        "cpu", EigenMatrix::IsRowMajor ? "csr" : "csc");
    result.provider_name = "eigen";
    using StorageIndex = typename EigenMatrix::StorageIndex;
    using Scalar = typename EigenMatrix::Scalar;
    const auto allocated =
        static_cast<std::uint64_t>(matrix_.data().allocatedSize());
    const auto outer = static_cast<std::uint64_t>(matrix_.outerSize());
    result.nnz = static_cast<int>(matrix_.nonZeros());
    result.pattern_reserved_bytes =
        (outer + 1 + allocated + (matrix_.isCompressed() ? 0 : outer)) *
        sizeof(StorageIndex);
    result.values_reserved_bytes = allocated * sizeof(Scalar);
    result.operator_owned_reserved_bytes = result.pattern_reserved_bytes +
                                           result.values_reserved_bytes;
    return result;
  }

 private:
  EigenMatrix matrix_;
};

// Fixed-pattern scalar CSR operator for CPU. It keeps the existing mutable
// Eigen SparseMatrix/Builder contract separate while sharing canonical
// compressed indices and owning only numeric values.
class CpuSparseCsrMatrix final : public SparseMatrix {
 public:
  CpuSparseCsrMatrix(std::shared_ptr<SparseCsrPattern> pattern,
                     const Ndarray &values,
                     bool pattern_built_for_operator = false);
  ~CpuSparseCsrMatrix() override;

  void nd_spmv(Program *prog, const Ndarray &x, const Ndarray &y);
  void spmv_cpu_raw(Program *prog,
                    std::uintptr_t input,
                    std::uintptr_t output);
  void update_values(Program *prog, const Ndarray &values) override;
  SparseMatrixRuntimeStatistics debug_runtime_statistics() const override;

  int num_nonzero() const override {
    return nnz_;
  }

  const std::vector<int32_t> &get_row_offsets() const {
    return pattern_->cpu_row_offsets();
  }

  const std::vector<int32_t> &get_column_indices() const {
    return pattern_->cpu_column_indices();
  }

  const void *get_values() const {
    return dtype_ == PrimitiveType::f32
               ? static_cast<const void *>(values_f32_.data())
               : static_cast<const void *>(values_f64_.data());
  }

 private:
  Program *program_{nullptr};
  int nnz_{0};
  std::shared_ptr<SparseCsrPattern> pattern_;
  std::vector<float32> values_f32_;
  std::vector<float64> values_f64_;
  mutable std::mutex spmv_mutex_;
  bool spmv_plan_initialized_{false};
};

// Internal immutable BSR topology shared by one or more numeric operators.
// CPU owns canonical host vectors, CUDA owns raw device index buffers, and
// Vulkan owns Program-managed index ndarrays.
class SparseBsrPattern final : public SparsePattern {
 public:
  SparseBsrPattern(Program *program,
                   int block_rows,
                   int block_cols,
                   int block_size,
                   const Ndarray &row_offsets,
                   const Ndarray &column_indices);
  ~SparseBsrPattern() override;

  Program *program() const override {
    return program_;
  }

  Arch arch() const override {
    return arch_;
  }

  int num_rows() const override {
    return rows_;
  }

  int num_cols() const override {
    return cols_;
  }

  int block_rows() const {
    return block_rows_;
  }

  int block_cols() const {
    return block_cols_;
  }

  int block_size() const {
    return block_size_;
  }

  int block_nnz() const {
    return block_nnz_;
  }

  int scalar_nnz() const {
    return scalar_nnz_;
  }

  std::size_t value_count() const {
    return value_count_;
  }

  std::uint64_t pattern_id() const {
    return pattern_id_;
  }

  std::uint64_t pattern_reserved_bytes() const;
  std::uint64_t device_to_host_bytes() const {
    return device_to_host_bytes_;
  }
  std::uint64_t device_to_device_bytes() const {
    return device_to_device_bytes_;
  }
  std::uint64_t operator_references() const {
    return operator_references_.load(std::memory_order_relaxed);
  }

  const std::vector<int32_t> &cpu_row_offsets() const;
  const std::vector<int32_t> &cpu_column_indices() const;
  void *cuda_row_offsets() const;
  void *cuda_column_indices() const;
  const Ndarray *vulkan_row_offsets() const;
  const Ndarray *vulkan_column_indices() const;

  void retain_operator_reference();
  void release_operator_reference();
  SparsePatternRuntimeStatistics debug_runtime_statistics() const override;

 private:
  Program *program_{nullptr};
  Arch arch_;
  int rows_{0};
  int cols_{0};
  int block_rows_{0};
  int block_cols_{0};
  int block_size_{0};
  int block_nnz_{0};
  int scalar_nnz_{0};
  std::size_t value_count_{0};
  std::uint64_t pattern_id_{0};
  std::vector<int32_t> cpu_row_offsets_;
  std::vector<int32_t> cpu_column_indices_;
  void *cuda_row_offsets_{nullptr};
  void *cuda_column_indices_{nullptr};
  Ndarray *vulkan_row_offsets_{nullptr};
  Ndarray *vulkan_column_indices_{nullptr};
  std::atomic<std::uint64_t> operator_references_{0};
  std::uint64_t device_to_host_bytes_{0};
  std::uint64_t device_to_device_bytes_{0};
};

// Internal CPU-only fixed-pattern BSR baseline. Blocks are stored row-major
// and remain distinct from the public scalar CSR/CSC Builder contract.
class CpuSparseBsrMatrix final : public SparseMatrix {
 public:
  CpuSparseBsrMatrix(Program *prog,
                     int block_rows,
                     int block_cols,
                     int block_size,
                     const Ndarray &row_offsets,
                     const Ndarray &column_indices,
                     const Ndarray &values);
  CpuSparseBsrMatrix(std::shared_ptr<SparseBsrPattern> pattern,
                     const Ndarray &values,
                     bool pattern_built_for_operator = false);
  ~CpuSparseBsrMatrix() override;

  void nd_spmv(Program *prog, const Ndarray &x, const Ndarray &y);
  void spmv_cpu_raw(Program *prog,
                    std::uintptr_t input,
                    std::uintptr_t output);
  void update_values(Program *prog, const Ndarray &values) override;
  SparseMatrixRuntimeStatistics debug_runtime_statistics() const override;

  int num_nonzero() const override {
    return scalar_nnz_;
  }

  int get_block_rows() const {
    return block_rows_;
  }

  int get_block_cols() const {
    return block_cols_;
  }

  int get_block_size() const {
    return block_size_;
  }

  int get_block_nnz() const {
    return block_nnz_;
  }

  const std::vector<int32_t> &get_block_row_offsets() const {
    return pattern_->cpu_row_offsets();
  }

  const std::vector<int32_t> &get_block_column_indices() const {
    return pattern_->cpu_column_indices();
  }

  const void *get_block_values() const {
    return dtype_ == PrimitiveType::f32
               ? static_cast<const void *>(values_f32_.data())
               : static_cast<const void *>(values_f64_.data());
  }

 private:
  Program *program_{nullptr};
  int block_rows_{0};
  int block_cols_{0};
  int block_size_{0};
  int block_nnz_{0};
  int scalar_nnz_{0};
  std::size_t value_count_{0};
  std::shared_ptr<SparseBsrPattern> pattern_;
  std::vector<float32> values_f32_;
  std::vector<float64> values_f64_;
  mutable std::mutex spmv_mutex_;
  bool spmv_plan_initialized_{false};
};

class CuSparseMatrix : public SparseMatrix {
 public:
  explicit CuSparseMatrix(int rows, int cols, DataType dt)
      : SparseMatrix(rows, cols, dt) {
#if defined(TI_WITH_CUDA)
    if (!CUSPARSEDriver::get_instance().is_loaded()) {
      bool load_success = CUSPARSEDriver::get_instance().load_cusparse();
      if (!load_success) {
        TI_ERROR("Failed to load cusparse library!");
      }
    }
#endif
  }
  explicit CuSparseMatrix(cusparseSpMatDescr_t A,
                          int rows,
                          int cols,
                          DataType dt,
                          void *csr_row_ptr,
                          void *csr_col_ind,
                          void *csr_val,
                          int nnz,
                          std::uint64_t device_to_device_bytes = 0)
      : SparseMatrix(rows, cols, dt),
        matrix_(A),
        csr_row_ptr_(csr_row_ptr),
        csr_col_ind_(csr_col_ind),
        csr_val_(csr_val),
        nnz_(nnz) {
    record_pattern_build();
    record_transfer_bytes(0, 0, device_to_device_bytes);
  }
  CuSparseMatrix(std::shared_ptr<SparseCsrPattern> pattern,
                 const Ndarray &values,
                 bool pattern_built_for_operator = false);
  CuSparseMatrix(const CuSparseMatrix &sm)
      : SparseMatrix(sm.rows_, sm.cols_, sm.dtype_), matrix_(sm.matrix_) {
  }

  ~CuSparseMatrix() override;

  // TODO: Overload +=, -= and *=
  friend std::unique_ptr<SparseMatrix> operator+(const CuSparseMatrix &lhs,
                                                 const CuSparseMatrix &rhs) {
    auto m = lhs.addition(rhs, 1.0, 1.0);
    return m;
  };

  friend std::unique_ptr<SparseMatrix> operator-(const CuSparseMatrix &lhs,
                                                 const CuSparseMatrix &rhs) {
    return lhs.addition(rhs, 1.0, -1.0);
  };

  friend std::unique_ptr<SparseMatrix> operator*(const CuSparseMatrix &sm,
                                                 float scale) {
    return sm.addition(sm, scale, 0.0);
  }

  friend std::unique_ptr<SparseMatrix> operator*(float scale,
                                                 const CuSparseMatrix &sm) {
    return sm.addition(sm, scale, 0.0);
  }

  std::unique_ptr<SparseMatrix> addition(const CuSparseMatrix &other,
                                         const float alpha,
                                         const float beta) const;

  std::unique_ptr<SparseMatrix> matmul(const CuSparseMatrix &other) const;

  std::unique_ptr<SparseMatrix> gemm(const CuSparseMatrix &other,
                                     const float alpha,
                                     const float beta) const;

  std::unique_ptr<SparseMatrix> transpose() const;

  void build_csr_from_coo(void *coo_row_ptr,
                          void *coo_col_ptr,
                          void *coo_values_ptr,
                          int nnz) override;

  void nd_spmv(Program *prog, const Ndarray &x, const Ndarray &y);

  void spmv(size_t x, size_t y);

  const void *get_matrix() const override {
    return &matrix_;
  };

  float get_element(int row, int col) const;

  const std::string to_string() const override;

  void *get_row_ptr() const {
    return csr_row_ptr_;
  }
  void *get_col_ind() const {
    return csr_col_ind_;
  }
  void *get_val_ptr() const {
    return csr_val_;
  }
  int get_nnz() const {
    return nnz_;
  }

  int num_nonzero() const override {
    return nnz_;
  }

  void update_values(Program *prog, const Ndarray &values) override;

  SparseMatrixRuntimeStatistics debug_runtime_statistics() const override;

  void mmwrite(const std::string &filename) override;

 private:
  void reset_spmv_resources();

  cusparseSpMatDescr_t matrix_{nullptr};
  void *csr_row_ptr_{nullptr};
  void *csr_col_ind_{nullptr};
  void *csr_val_{nullptr};
  int nnz_{0};
  std::shared_ptr<SparseCsrPattern> pattern_;
  mutable std::mutex spmv_mutex_;
  cusparseHandle_t spmv_handle_{nullptr};
  cusparseDnVecDescr_t spmv_vec_x_{nullptr};
  cusparseDnVecDescr_t spmv_vec_y_{nullptr};
  size_t spmv_x_ptr_{0};
  size_t spmv_y_ptr_{0};
  void *spmv_buffer_{nullptr};
  size_t spmv_buffer_size_{0};
  bool spmv_buffer_initialized_{false};
};

// Internal CUDA-only prototype for already-compressed, square dense blocks.
// Public SparseMatrixBuilder format selection intentionally remains CSR-only
// until assembly, solver, and cross-provider contracts are complete.
class CuSparseBsrMatrix final : public SparseMatrix {
 public:
  CuSparseBsrMatrix(Program *prog,
                    int block_rows,
                    int block_cols,
                    int block_size,
                    const Ndarray &row_offsets,
                    const Ndarray &column_indices,
                    const Ndarray &values);
  CuSparseBsrMatrix(std::shared_ptr<SparseBsrPattern> pattern,
                    const Ndarray &values,
                    bool pattern_built_for_operator = false);
  ~CuSparseBsrMatrix() override;

  void nd_spmv(Program *prog, const Ndarray &x, const Ndarray &y);
  void spmv(size_t x, size_t y);
  void update_values(Program *prog, const Ndarray &values) override;
  SparseMatrixRuntimeStatistics debug_runtime_statistics() const override;

  int num_nonzero() const override {
    return scalar_nnz_;
  }

  const void *get_matrix() const override {
    return &matrix_;
  }

  int get_block_rows() const {
    return block_rows_;
  }

  int get_block_cols() const {
    return block_cols_;
  }

  int get_block_size() const {
    return block_size_;
  }

  int get_block_nnz() const {
    return block_nnz_;
  }

  void *get_block_row_offsets() const {
    return pattern_->cuda_row_offsets();
  }

  void *get_block_column_indices() const {
    return pattern_->cuda_column_indices();
  }

  void *get_block_values() const {
    return values_;
  }

 private:
  void reset_spmv_resources();

  int block_rows_{0};
  int block_cols_{0};
  int block_size_{0};
  int block_nnz_{0};
  int scalar_nnz_{0};
  std::size_t value_count_{0};
  std::shared_ptr<SparseBsrPattern> pattern_;
  cusparseSpMatDescr_t matrix_{nullptr};
  void *values_{nullptr};
  mutable std::mutex spmv_mutex_;
  cusparseHandle_t spmv_handle_{nullptr};
  cusparseDnVecDescr_t spmv_vec_x_{nullptr};
  cusparseDnVecDescr_t spmv_vec_y_{nullptr};
  size_t spmv_x_ptr_{0};
  size_t spmv_y_ptr_{0};
  void *spmv_buffer_{nullptr};
  size_t spmv_buffer_size_{0};
  bool spmv_buffer_initialized_{false};
};

// Internal Vulkan-only fixed-pattern CSR baseline. Pattern/value storage is
// owned by Program-managed ndarrays so asynchronous command replay retains
// generation-qualified resources through completion.
class VulkanSparseMatrix final : public SparseMatrix {
 public:
  VulkanSparseMatrix(Program *prog,
                     int rows,
                     int cols,
                     const Ndarray &row_offsets,
                     const Ndarray &column_indices,
                     const Ndarray &values);
  VulkanSparseMatrix(std::shared_ptr<SparseCsrPattern> pattern,
                     const Ndarray &values,
                     bool pattern_built_for_operator = false);
  VulkanSparseMatrix(Program *prog,
                     int rows,
                     int cols,
                     int nnz,
                     Ndarray *owned_row_offsets,
                     Ndarray *owned_column_indices,
                     Ndarray *owned_values,
                     std::uint64_t device_to_device_bytes);
  ~VulkanSparseMatrix() override;

  void nd_spmv(Program *prog, const Ndarray &x, const Ndarray &y);
  void update_values(Program *prog, const Ndarray &values) override;
  SparseMatrixRuntimeStatistics debug_runtime_statistics() const override;

  int num_nonzero() const override {
    return nnz_;
  }

  const Ndarray *get_row_offsets() const {
    return pattern_ ? pattern_->vulkan_row_offsets() : row_offsets_;
  }

  const Ndarray *get_column_indices() const {
    return pattern_ ? pattern_->vulkan_column_indices() : column_indices_;
  }

  const Ndarray *get_values() const {
    return values_;
  }

 private:
  Program *program_{nullptr};
  int nnz_{0};
  std::shared_ptr<SparseCsrPattern> pattern_;
  Ndarray *row_offsets_{nullptr};
  Ndarray *column_indices_{nullptr};
  Ndarray *values_{nullptr};
  mutable std::mutex spmv_mutex_;
  bool spmv_plan_initialized_{false};
};

// Internal Vulkan-only fixed-pattern BSR baseline. It mirrors the CUDA
// 2/3/6/12 block contract while keeping Program-owned ndarray storage and
// generation-qualified command replay.
class VulkanSparseBsrMatrix final : public SparseMatrix {
 public:
  VulkanSparseBsrMatrix(Program *prog,
                        int block_rows,
                        int block_cols,
                        int block_size,
                        const Ndarray &row_offsets,
                        const Ndarray &column_indices,
                        const Ndarray &values);
  VulkanSparseBsrMatrix(std::shared_ptr<SparseBsrPattern> pattern,
                        const Ndarray &values,
                        bool pattern_built_for_operator = false);
  ~VulkanSparseBsrMatrix() override;

  void nd_spmv(Program *prog, const Ndarray &x, const Ndarray &y);
  void update_values(Program *prog, const Ndarray &values) override;
  SparseMatrixRuntimeStatistics debug_runtime_statistics() const override;

  int num_nonzero() const override {
    return scalar_nnz_;
  }

  int get_block_rows() const {
    return block_rows_;
  }

  int get_block_cols() const {
    return block_cols_;
  }

  int get_block_size() const {
    return block_size_;
  }

  int get_block_nnz() const {
    return block_nnz_;
  }

  const Ndarray *get_block_row_offsets() const {
    return pattern_->vulkan_row_offsets();
  }

  const Ndarray *get_block_column_indices() const {
    return pattern_->vulkan_column_indices();
  }

  const Ndarray *get_block_values() const {
    return values_;
  }

 private:
  Program *program_{nullptr};
  int block_rows_{0};
  int block_cols_{0};
  int block_size_{0};
  int block_nnz_{0};
  int scalar_nnz_{0};
  std::size_t value_count_{0};
  std::shared_ptr<SparseBsrPattern> pattern_;
  Ndarray *values_{nullptr};
  mutable std::mutex spmv_mutex_;
  bool spmv_plan_initialized_{false};
};

// Private Vulkan-only bounded triplet assembly plan. Separate triplet arrays
// and packed public-builder storage stay device-resident; only an 8-byte
// active-or-status/unique-count control record is read at transactional
// finalize. Successful builds publish exact-sized CSR buffers, while failed
// builds leave every previously returned matrix intact.
class VulkanSparseAssemblyPlan final {
 public:
  VulkanSparseAssemblyPlan(Program *program,
                           int rows,
                           int cols,
                           int capacity);
  ~VulkanSparseAssemblyPlan();

  std::unique_ptr<VulkanSparseMatrix> build(Program *program,
                                             const Ndarray &triplet_rows,
                                             const Ndarray &triplet_columns,
                                             const Ndarray &triplet_values);
  std::unique_ptr<VulkanSparseMatrix> build_packed(
      Program *program,
      const Ndarray &packed_triplets);
  SparseAssemblyRuntimeStatistics debug_runtime_statistics() const;

 private:
  void delete_workspace() noexcept;
  std::unique_ptr<VulkanSparseMatrix> build_internal(
      Program *program,
      const Ndarray *packed_triplets,
      const Ndarray *triplet_rows,
      const Ndarray *triplet_columns,
      const Ndarray *triplet_values);

  Program *program_{nullptr};
  int rows_{0};
  int cols_{0};
  int capacity_{0};
  Ndarray *sorted_keys_{nullptr};
  Ndarray *sorted_values_{nullptr};
  Ndarray *segment_ids_{nullptr};
  Ndarray *unique_keys_{nullptr};
  Ndarray *segment_offsets_{nullptr};
  Ndarray *unique_values_{nullptr};
  Ndarray *row_offsets_{nullptr};
  Ndarray *column_indices_{nullptr};
  Ndarray *active_count_{nullptr};
  Ndarray *control_{nullptr};
  mutable std::mutex mutex_;
  SparseAssemblyRuntimeStatistics statistics_;
};

// Private CUDA-only bounded triplet assembly plan. It mirrors the Vulkan
// transaction contract while using Toolkit-free Driver PTX for validation,
// sorting support, segment reduction, and CSR emission. Only the final 8-byte
// status/count record reaches the host before exact-sized cuSPARSE ownership
// is published.
class CudaSparseAssemblyPlan final {
 public:
  CudaSparseAssemblyPlan(Program *program,
                         int rows,
                         int cols,
                         int capacity);
  ~CudaSparseAssemblyPlan();

  std::unique_ptr<CuSparseMatrix> build(Program *program,
                                       const Ndarray &triplet_rows,
                                       const Ndarray &triplet_columns,
                                       const Ndarray &triplet_values);
  std::unique_ptr<CuSparseMatrix> build_packed(
      Program *program,
      const Ndarray &packed_triplets);
  SparseAssemblyRuntimeStatistics debug_runtime_statistics() const;

 private:
  void delete_workspace() noexcept;
  std::unique_ptr<CuSparseMatrix> build_internal(
      Program *program,
      const Ndarray *packed_triplets,
      const Ndarray *triplet_rows,
      const Ndarray *triplet_columns,
      const Ndarray *triplet_values);

  Program *program_{nullptr};
  int rows_{0};
  int cols_{0};
  int capacity_{0};
  Ndarray *sorted_keys_{nullptr};
  Ndarray *sorted_values_{nullptr};
  Ndarray *segment_ids_{nullptr};
  Ndarray *unique_keys_{nullptr};
  Ndarray *segment_offsets_{nullptr};
  Ndarray *unique_values_{nullptr};
  Ndarray *row_offsets_{nullptr};
  Ndarray *column_indices_{nullptr};
  Ndarray *active_count_{nullptr};
  Ndarray *control_{nullptr};
  mutable std::mutex mutex_;
  SparseAssemblyRuntimeStatistics statistics_;
};

std::unique_ptr<SparseMatrix> make_sparse_matrix(
    int rows,
    int cols,
    DataType dt,
    const std::string &storage_format);
std::unique_ptr<SparseMatrix> make_cu_sparse_matrix(int rows,
                                                    int cols,
                                                    DataType dt);
std::unique_ptr<SparseMatrix> make_cu_sparse_matrix(cusparseSpMatDescr_t mat,
                                                    int rows,
                                                    int cols,
                                                    DataType dt,
                                                    void *csr_row_ptr,
                                                    void *csr_col_ind,
                                                    void *csr_val_,
                                                    int nnz,
                                                    std::uint64_t
                                                        device_to_device_bytes =
                                                            0);

void make_sparse_matrix_from_ndarray(Program *prog,
                                     SparseMatrix &sm,
                                     const Ndarray &ndarray);
}  // namespace taichi::lang
