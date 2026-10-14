#pragma once

#include <cstddef>
#include <cstdint>
#include "taichi/rhi/cuda/cuda_types.h"

namespace taichi::lang {

// The caller retains these buffers, their CUDA context and serialized access.
// A plan owns only vendor descriptors/workspace, never the CSR or vector data.
struct CudaCsrSpmvDescription {
  std::int64_t rows{0}, columns{0}, nonzeros{0};
  void *row_offsets{nullptr};
  void *column_indices{nullptr};
  void *values{nullptr};
  cusparseIndexType_t row_type{CUSPARSE_INDEX_32I};
  cusparseIndexType_t column_type{CUSPARSE_INDEX_32I};
  cusparseSpMVAlg_t algorithm{CUSPARSE_SPMV_CSR_ALG1};
  bool transpose{false};
  float alpha{1}, beta{0};
};

// Shared by live JIT matrices and reconstructed native operations. Independent
// plans may reference the same CSR data without sharing mutable preprocess
// state.
class CudaCsrSpmvPlan final {
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

  explicit CudaCsrSpmvPlan(CudaCsrSpmvDescription description);
  ~CudaCsrSpmvPlan();
  CudaCsrSpmvPlan(const CudaCsrSpmvPlan &) = delete;
  CudaCsrSpmvPlan &operator=(const CudaCsrSpmvPlan &) = delete;

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
  CudaCsrSpmvDescription description_;
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
};

}  // namespace taichi::lang
