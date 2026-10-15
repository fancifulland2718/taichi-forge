#pragma once

#include <cstddef>
#include <cstdint>
#include "taichi/rhi/cuda/cuda_types.h"

namespace taichi::lang {

// The caller retains these buffers, their CUDA context and serialized access.
// A plan owns execution descriptors/workspace, never sparse or vector data.
struct CudaSparseSpmvDescription {
  std::int64_t rows{0}, columns{0}, nonzeros{0};
  void *row_offsets{nullptr};
  void *column_indices{nullptr};
  void *values{nullptr};
  cusparseIndexType_t row_type{CUSPARSE_INDEX_32I};
  cusparseIndexType_t column_type{CUSPARSE_INDEX_32I};
  cusparseSpMVAlg_t algorithm{CUSPARSE_SPMV_CSR_ALG1};
  bool transpose{false};
  float alpha{1}, beta{0};
  // Zero selects CSR; positive selects square row-major BSR blocks. rows and
  // columns remain scalar extents, while nonzeros counts stored blocks in BSR.
  std::int64_t block_size{0};
};

// Shared by live JIT matrices and reconstructed native operations. Independent
// plans may reference the same sparse data without sharing execution state.
class CudaSparseSpmvPlan final {
 public:
  struct Preparation {
    bool built{false};
    bool workspace_allocated{false};
    unsigned descriptor_creations{0};
    unsigned descriptor_rebinds{0};
  };
  struct Statistics {
    bool preprocessed{false};
    std::uint32_t preprocess_error{0};
    std::uint64_t preprocess_builds{0};
    std::uint64_t preprocess_reuses{0};
    std::uint64_t preprocess_fallbacks{0};
  };

  // An explicitly borrowed BSR descriptor outlives this plan; this avoids
  // duplicating the live matrix descriptor. It is never preprocessed here.
  explicit CudaSparseSpmvPlan(
      CudaSparseSpmvDescription description,
      cusparseSpMatDescr_t borrowed_bsr_matrix = nullptr);
  ~CudaSparseSpmvPlan();
  CudaSparseSpmvPlan(const CudaSparseSpmvPlan &) = delete;
  CudaSparseSpmvPlan &operator=(const CudaSparseSpmvPlan &) = delete;

  // Cold preparation may allocate and preprocess; it does not execute SpMV.
  Preparation prepare(void *input,
                      void *output,
                      CUstream stream,
                      bool enable_preprocess);
  // Capture/direct submission only: no allocation, preprocessing or rebinding.
  void record(CUstream stream);
  const Statistics &statistics() const {
    return statistics_;
  }
  std::size_t workspace_bytes() const {
    return workspace_bytes_;
  }
  unsigned dense_descriptor_count() const {
    return unsigned(input_ != nullptr) + unsigned(output_ != nullptr);
  }

 private:
  void bind_stream(CUstream stream);
  void clear();
  CudaSparseSpmvDescription description_;
  Statistics statistics_;
  cusparseHandle_t handle_{nullptr};
  cusparseSpMatDescr_t matrix_{nullptr};
  cusparseDnVecDescr_t input_{nullptr}, output_{nullptr};
  void *input_address_{nullptr}, *output_address_{nullptr};
  CUstream stream_{nullptr};
  void *workspace_{nullptr};
  std::size_t workspace_bytes_{0};
  bool initialized_{false};
  bool preprocess_disabled_{false};
  bool owns_matrix_{true};
};

enum class CudaSpmmPreprocess { none, optional, required };

struct CudaCsrSpmmDescription {
  std::int64_t rows{0}, columns{0}, nonzeros{0};
  void *row_offsets{nullptr};
  void *column_indices{nullptr};
  void *values{nullptr};
  std::int64_t rhs_count{0};
  cusparseIndexType_t row_type{CUSPARSE_INDEX_32I};
  cusparseIndexType_t column_type{CUSPARSE_INDEX_32I};
  cusparseSpMMAlg_t algorithm{CUSPARSE_SPMM_CSR_ALG2};
  CudaSpmmPreprocess preprocess{CudaSpmmPreprocess::none};
  float alpha{1}, beta{0};
};

// Row-major dense ports over one fixed CSR pattern. Each plan owns a separate
// sparse descriptor, because cuSPARSE associates its active preprocess buffer
// with that descriptor, even when other plans share all CSR storage.
class CudaCsrSpmmPlan final {
 public:
  struct Preparation {
    bool built{false};
    bool workspace_allocated{false};
    unsigned descriptor_rebinds{0};
  };
  struct Statistics {
    bool preprocess_attempted{false};
    bool preprocessed{false};
    std::uint32_t preprocess_error{0};
  };
  explicit CudaCsrSpmmPlan(CudaCsrSpmmDescription description);
  ~CudaCsrSpmmPlan();
  CudaCsrSpmmPlan(const CudaCsrSpmmPlan &) = delete;
  CudaCsrSpmmPlan &operator=(const CudaCsrSpmmPlan &) = delete;

  // Only cold preparation allocates, preprocesses or rebinds dense addresses.
  // It never executes SpMM. Numeric CSR and dense contents may change in place.
  Preparation prepare(void *input, void *output, CUstream stream);
  void record(CUstream stream);
  const Statistics &statistics() const {
    return statistics_;
  }
  std::size_t workspace_bytes() const {
    return workspace_bytes_;
  }

 private:
  void bind_stream(CUstream stream);
  void clear();
  CudaCsrSpmmDescription description_;
  Statistics statistics_;
  cusparseHandle_t handle_{nullptr};
  cusparseSpMatDescr_t matrix_{nullptr};
  cusparseDnMatDescr_t input_{nullptr}, output_{nullptr};
  void *input_address_{nullptr}, *output_address_{nullptr};
  CUstream stream_{nullptr};
  void *workspace_{nullptr};
  std::size_t workspace_bytes_{0};
  bool initialized_{false};
  bool stream_bound_{false};
};

struct CudaCsrTriangularDescription {
  std::int64_t rows{0}, nonzeros{0}, rhs_count{1};
  void *row_offsets{nullptr};
  void *column_indices{nullptr};
  void *values{nullptr};
  cusparseIndexType_t index_type{CUSPARSE_INDEX_32I};
  bool upper{false}, unit_diagonal{false}, transpose{false};
};

// One fixed CSR pattern and one analysis/workspace per plan. The caller keeps
// storage alive and serializes access, including explicit value updates after
// changing the numeric buffer. SpSV/SpSM retain numeric data in their analysis;
// mutating the CSR values alone is not a valid update protocol.
class CudaCsrTriangularPlan final {
 public:
  struct Preparation {
    bool built{false};
    unsigned descriptor_rebinds{0};
  };
  explicit CudaCsrTriangularPlan(CudaCsrTriangularDescription description);
  ~CudaCsrTriangularPlan();
  CudaCsrTriangularPlan(const CudaCsrTriangularPlan &) = delete;
  CudaCsrTriangularPlan &operator=(const CudaCsrTriangularPlan &) = delete;

  // Cold analysis never solves. Warm record neither rebinds nor allocates.
  Preparation prepare(void *input, void *output, CUstream stream);
  void record(CUstream stream);
  // Explicit refresh of retained numeric analysis at the same CSR address.
  // The no-argument form preserves the live JIT plan's current stream.
  void update_values(CUstream stream);
  std::size_t workspace_bytes() const {
    return workspace_bytes_;
  }

 private:
  void bind_stream(CUstream stream);
  void clear();
  cusparseOperation_t operation() const;
  CudaCsrTriangularDescription description_;
  cusparseHandle_t handle_{nullptr};
  cusparseSpMatDescr_t matrix_{nullptr};
  cusparseDnVecDescr_t input_vector_{nullptr}, output_vector_{nullptr};
  cusparseDnMatDescr_t input_matrix_{nullptr}, output_matrix_{nullptr};
  cusparseSpSVDescr_t vector_solve_{nullptr};
  cusparseSpSMDescr_t matrix_solve_{nullptr};
  void *input_address_{nullptr}, *output_address_{nullptr};
  void *workspace_{nullptr};
  std::size_t workspace_bytes_{0};
  CUstream stream_{nullptr};
  bool initialized_{false}, stream_bound_{false};
};

}  // namespace taichi::lang
