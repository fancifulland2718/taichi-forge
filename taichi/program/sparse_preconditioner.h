#pragma once

#include "taichi/program/sparse_matrix.h"

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace taichi::lang {

struct SparsePreconditionerPlanRuntimeStatistics {
  std::string backend_family{"unknown"};
  std::string method{"jacobi"};
  std::string dtype{"unknown"};
  int rows{0};
  int block_rows{0};
  int block_size{0};
  std::uint64_t operator_pattern_version_at_build{0};
  std::uint64_t operator_numeric_version_at_build{0};
  std::uint64_t operator_pattern_version_current{0};
  std::uint64_t operator_numeric_version_current{0};
  bool operator_stale{false};
  std::uint64_t preconditioner_pattern_version_at_build{0};
  std::uint64_t preconditioner_numeric_version_at_build{0};
  std::uint64_t preconditioner_pattern_version_current{0};
  std::uint64_t preconditioner_numeric_version_current{0};
  bool preconditioner_stale{false};
  std::uint64_t apply_calls{0};
  std::uint64_t persistent_inverse_count{0};
  std::uint64_t persistent_inverse_reserved_bytes{0};
  std::uint64_t construction_device_to_host_bytes{0};
  std::uint64_t construction_host_to_device_bytes{0};
  std::uint64_t construction_host_synchronizations{0};
  std::uint64_t numeric_refresh_calls{0};
  std::uint64_t numeric_refresh_successes{0};
  std::uint64_t numeric_refresh_noops{0};
  std::uint64_t numeric_refresh_failures{0};
  std::uint64_t refresh_device_to_host_bytes{0};
  std::uint64_t refresh_host_to_device_bytes{0};
  std::uint64_t refresh_host_synchronizations{0};
  std::uint64_t refresh_peak_temporary_host_bytes{0};
  std::uint64_t refresh_peak_temporary_device_bytes{0};
  bool numeric_refresh_supported{false};
  bool in_place_apply_supported{true};
};

// Internal fixed-CSR CPU f32/f64 and GPU f32 Jacobi baseline. Construction
// validates and freezes
// the operator diagonal. A numeric or pattern update invalidates the plan
// instead of silently applying a stale inverse.
class SparseJacobiPreconditionerPlan final {
 public:
  SparseJacobiPreconditionerPlan(Program *program, SparseMatrix &matrix);
  ~SparseJacobiPreconditionerPlan();

  void validate_compatible(Program *program,
                           const SparseMatrix &matrix) const;
  void refresh_numeric(Program *program);
  void apply_cpu_raw(Program *program,
                     std::uintptr_t input,
                     std::uintptr_t output);
  void apply_cuda_raw(Program *program,
                      std::uintptr_t input,
                      std::uintptr_t output);
  void apply(Program *program, const Ndarray &input, const Ndarray &output);
  SparsePreconditionerPlanRuntimeStatistics debug_runtime_statistics() const;

 private:
  void release_resources();

  Program *program_{nullptr};
  SparseMatrix *matrix_{nullptr};
  std::string backend_family_{"unknown"};
  int rows_{0};
  DataType dtype_{PrimitiveType::f32};
  std::uint64_t pattern_version_at_build_{0};
  std::uint64_t numeric_version_at_build_{0};
  std::vector<float32> host_inverse_f32_;
  std::vector<float64> host_inverse_f64_;
  std::vector<int32_t> diagonal_offsets_;
  Ndarray *device_inverse_{nullptr};
  mutable std::mutex apply_mutex_;
  std::uint64_t apply_calls_{0};
  std::uint64_t construction_device_to_host_bytes_{0};
  std::uint64_t construction_host_to_device_bytes_{0};
  std::uint64_t construction_host_synchronizations_{0};
  std::uint64_t numeric_refresh_calls_{0};
  std::uint64_t numeric_refresh_successes_{0};
  std::uint64_t numeric_refresh_noops_{0};
  std::uint64_t numeric_refresh_failures_{0};
  std::uint64_t refresh_device_to_host_bytes_{0};
  std::uint64_t refresh_host_to_device_bytes_{0};
  std::uint64_t refresh_host_synchronizations_{0};
  std::uint64_t refresh_peak_temporary_host_bytes_{0};
  std::uint64_t refresh_peak_temporary_device_bytes_{0};
};

std::unique_ptr<SparseJacobiPreconditionerPlan>
make_sparse_jacobi_preconditioner_plan(Program *program,
                                       SparseMatrix &matrix);

// Internal CPU/CUDA/Vulkan BSR baseline for 2/3/6/12-DOF dense diagonal
// blocks.
// It keeps one row-major inverse block per block row and shares the same
// stale-version and transactional numeric-refresh contract as scalar Jacobi.
class SparseBlockJacobiPreconditionerPlan final {
 public:
  SparseBlockJacobiPreconditionerPlan(Program *program,
                                      SparseMatrix &matrix);
  ~SparseBlockJacobiPreconditionerPlan();

  void validate_compatible(Program *program,
                           const SparseMatrix &matrix) const;
  void refresh_numeric(Program *program);
  void apply_cpu_raw(Program *program,
                     std::uintptr_t input,
                     std::uintptr_t output);
  void apply_cuda_raw(Program *program,
                      std::uintptr_t input,
                      std::uintptr_t output);
  void apply(Program *program, const Ndarray &input, const Ndarray &output);
  SparsePreconditionerPlanRuntimeStatistics debug_runtime_statistics() const;

 private:
  void release_resources();

 Program *program_{nullptr};
  SparseMatrix *matrix_{nullptr};
  std::string backend_family_{"unknown"};
  int rows_{0};
  int block_rows_{0};
  int block_size_{0};
  int block_nnz_{0};
  DataType dtype_{PrimitiveType::f32};
  std::uint64_t pattern_version_at_build_{0};
  std::uint64_t numeric_version_at_build_{0};
  std::vector<int32_t> diagonal_block_offsets_;
  std::vector<float32> host_inverse_blocks_f32_;
  std::vector<float64> host_inverse_blocks_f64_;
  Ndarray *device_inverse_blocks_{nullptr};
  mutable std::mutex apply_mutex_;
  std::uint64_t apply_calls_{0};
  std::uint64_t construction_device_to_host_bytes_{0};
  std::uint64_t construction_host_to_device_bytes_{0};
  std::uint64_t construction_host_synchronizations_{0};
  std::uint64_t numeric_refresh_calls_{0};
  std::uint64_t numeric_refresh_successes_{0};
  std::uint64_t numeric_refresh_noops_{0};
  std::uint64_t numeric_refresh_failures_{0};
  std::uint64_t refresh_device_to_host_bytes_{0};
  std::uint64_t refresh_host_to_device_bytes_{0};
  std::uint64_t refresh_host_synchronizations_{0};
  std::uint64_t refresh_peak_temporary_host_bytes_{0};
  std::uint64_t refresh_peak_temporary_device_bytes_{0};
};

std::unique_ptr<SparseBlockJacobiPreconditionerPlan>
make_sparse_block_jacobi_preconditioner_plan(Program *program,
                                             SparseMatrix &matrix);

// Internal matrix-free preconditioner contract. The compiled-kernel target
// operator and an independently compiled kernel or Graph inverse-apply
// provider keep their own storage and version streams; this plan only binds
// them transactionally. No diagonal, block structure, symmetry, or
// positive-definiteness is inferred from opaque operator data.
class CompiledKernelPreconditionerPlan final {
 public:
  CompiledKernelPreconditionerPlan(
      Program *program,
      CompiledKernelLinearOperator &target_operator,
      SparseMatrix &inverse_apply_operator,
      bool assume_symmetric_positive_definite);

  void validate_compatible(
      Program *program,
      const CompiledKernelLinearOperator &target_operator) const;
  void apply(Program *program,
             const CompiledKernelLinearOperator &target_operator,
             const Ndarray &input,
             const Ndarray &output);
  SparseMatrix::NumericAccessGuard acquire_numeric_access_guard() const;
  SparsePreconditionerPlanRuntimeStatistics debug_runtime_statistics() const;

 private:
  void validate_compatible_locked(
      Program *program,
      const CompiledKernelLinearOperator &target_operator) const;

  Program *program_{nullptr};
  CompiledKernelLinearOperator *target_operator_{nullptr};
  SparseMatrix *inverse_apply_operator_{nullptr};
  std::uint64_t target_pattern_version_at_build_{0};
  std::uint64_t target_numeric_version_at_build_{0};
  std::uint64_t inverse_pattern_version_at_build_{0};
  std::uint64_t inverse_numeric_version_at_build_{0};
  mutable std::mutex apply_mutex_;
  std::uint64_t apply_calls_{0};
};

std::unique_ptr<CompiledKernelPreconditionerPlan>
make_compiled_kernel_preconditioner_plan(
    Program *program,
    CompiledKernelLinearOperator &target_operator,
    SparseMatrix &inverse_apply_operator,
    bool assume_symmetric_positive_definite);

}  // namespace taichi::lang
